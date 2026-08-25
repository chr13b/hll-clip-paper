#!/usr/bin/env bash
# Phase 2: user-space PostgreSQL 16 -- no root, no Docker, no system changes.
#
# Why this exists: the phase-2 study (exp16) needs a reachable PostgreSQL, and a
# user-space install works on any Linux box without root or Docker. The zonky.io
# embedded-postgres binaries are plain
# tarballs of stock PostgreSQL published on Maven Central for exactly this use.
# Everything lands under $PG_HOME; remove it with `rm -rf` to uninstall.
#
# Usage:  bash pg_local_setup.sh          # install + init + start, prints the DSN
#         bash pg_local_setup.sh stop     # stop the server
set -euo pipefail

PG_VERSION="${PG_VERSION:-16.14.0}"
PG_HOME="${PG_HOME:-$HOME/hll_phase2_pg}"
PG_PORT="${PG_PORT:-5433}"
DIST="$PG_HOME/dist"
DATA="$PG_HOME/data"
SOCK="$PG_HOME/sock"
if (( ${#SOCK} > 90 )); then
    echo "!! PG_HOME is too deep: the Unix socket path '$SOCK/.s.PGSQL.<port>' would exceed PostgreSQL's 107-byte limit. Use a shorter PG_HOME (e.g. PG_HOME=\$HOME/hll_pg)."; exit 1
fi
JAR_URL="https://repo1.maven.org/maven2/io/zonky/test/postgres/embedded-postgres-binaries-linux-amd64/${PG_VERSION}/embedded-postgres-binaries-linux-amd64-${PG_VERSION}.jar"

if [ "${1:-}" = "stop" ]; then
    "$DIST/bin/pg_ctl" -D "$DATA" stop -m fast
    exit 0
fi

mkdir -p "$PG_HOME" "$SOCK"

if [ ! -x "$DIST/bin/postgres" ]; then
    echo "== downloading PostgreSQL $PG_VERSION (zonky embedded binaries)"
    curl -fsSL -o "$PG_HOME/pg.jar" "$JAR_URL"
    JAR_SHA256="${JAR_SHA256:-3278331b124b46fb9f8bea30c7ff3bd6227a3934dad38a6ddb625eb7dcfa8ca8}"   # Maven Central sidecar, 16.14.0
    echo "$JAR_SHA256  $PG_HOME/pg.jar" | sha256sum -c --quiet - || { echo "!! pg.jar checksum mismatch -- refusing to extract"; exit 1; }
    # the jar is a zip holding postgres-linux-x86_64.txz; extract both layers
    python3 - "$PG_HOME" <<'EOF'
import sys, zipfile, tarfile, os
home = sys.argv[1]
with zipfile.ZipFile(os.path.join(home, "pg.jar")) as z:
    member = [n for n in z.namelist() if n.endswith(".txz")][0]
    z.extract(member, home)
os.makedirs(os.path.join(home, "dist"), exist_ok=True)
with tarfile.open(os.path.join(home, member), "r:xz") as t:
    t.extractall(os.path.join(home, "dist"))
EOF
    rm -f "$PG_HOME/pg.jar" "$PG_HOME"/*.txz
fi
"$DIST/bin/postgres" --version

if [ ! -f "$DATA/PG_VERSION" ]; then
    echo "== initdb"
    "$DIST/bin/initdb" -D "$DATA" -U postgres --auth=trust --no-locale -E UTF8
    # modest resources, leaving room for the analysis processes on a small machine
    cat >> "$DATA/postgresql.conf" <<CONF
listen_addresses = 'localhost'
port = $PG_PORT
unix_socket_directories = '$SOCK'
shared_buffers = 512MB
work_mem = 128MB
maintenance_work_mem = 256MB
max_parallel_workers_per_gather = 2
CONF
fi

if ! "$DIST/bin/pg_ctl" -D "$DATA" status >/dev/null 2>&1; then
    echo "== starting"
    "$DIST/bin/pg_ctl" -D "$DATA" -l "$PG_HOME/server.log" start
fi

# zonky ships server binaries only (initdb/pg_ctl/postgres, no psql) -- use psycopg2
python3 - "$PG_PORT" <<'EOF'
import sys, psycopg2
port = sys.argv[1]
con = psycopg2.connect(f"host=localhost port={port} dbname=postgres user=postgres")
con.autocommit = True
cur = con.cursor()
cur.execute("SELECT version();")
print(cur.fetchone()[0])
cur.execute("SELECT 1 FROM pg_database WHERE datname='tpch'")
if not cur.fetchone():
    cur.execute("CREATE DATABASE tpch")
    print("created database tpch")
con.close()
EOF
echo "DSN: host=localhost port=$PG_PORT dbname=tpch user=postgres"
