#!/usr/bin/env python3
"""Phase 2 infrastructure: TPC-H into PostgreSQL, no dbgen build required.

DuckDB's tpch extension generates the data (CALL dbgen), we COPY it out as CSV on the
Linux filesystem and COPY it into PostgreSQL with the spec's PRIMARY KEY / FOREIGN KEY
constraints declared. Row counts are verified against the TPC-H specification before
the script reports success. The 22 benchmark queries are extracted from the same
extension (tpch_queries()) and written to phase2/queries_pg/qNN.sql.

Usage:
    python3 tpch_load.py --sf 1
    HLL_PG_DSN="host=localhost port=5433 dbname=tpch user=postgres" python3 tpch_load.py --sf 1
"""
import argparse
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("PG_HOME") or os.path.expanduser("~/hll_phase2_pg")
QUERY_DIR = os.path.join(HERE, "queries_pg")

TABLES = ["region", "nation", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]

# TPC-H specification v3 table 4-2 (row counts scale linearly except lineitem)
EXPECTED = {
    1:  {"region": 5, "nation": 25, "part": 200000, "supplier": 10000, "partsupp": 800000,
         "customer": 150000, "orders": 1500000, "lineitem": 6001215},
    10: {"region": 5, "nation": 25, "part": 2000000, "supplier": 100000, "partsupp": 8000000,
         "customer": 1500000, "orders": 15000000, "lineitem": 59986052},
}

DDL = """
DROP TABLE IF EXISTS lineitem, orders, partsupp, customer, supplier, part, nation, region CASCADE;

CREATE TABLE region (
    r_regionkey  INTEGER       NOT NULL,
    r_name       CHAR(25)      NOT NULL,
    r_comment    VARCHAR(152),
    PRIMARY KEY (r_regionkey));

CREATE TABLE nation (
    n_nationkey  INTEGER       NOT NULL,
    n_name       CHAR(25)      NOT NULL,
    n_regionkey  INTEGER       NOT NULL,
    n_comment    VARCHAR(152),
    PRIMARY KEY (n_nationkey));

CREATE TABLE part (
    p_partkey     INTEGER        NOT NULL,
    p_name        VARCHAR(55)    NOT NULL,
    p_mfgr        CHAR(25)       NOT NULL,
    p_brand       CHAR(10)       NOT NULL,
    p_type        VARCHAR(25)    NOT NULL,
    p_size        INTEGER        NOT NULL,
    p_container   CHAR(10)       NOT NULL,
    p_retailprice DECIMAL(15,2)  NOT NULL,
    p_comment     VARCHAR(23)    NOT NULL,
    PRIMARY KEY (p_partkey));

CREATE TABLE supplier (
    s_suppkey   INTEGER        NOT NULL,
    s_name      CHAR(25)       NOT NULL,
    s_address   VARCHAR(40)    NOT NULL,
    s_nationkey INTEGER        NOT NULL,
    s_phone     CHAR(15)       NOT NULL,
    s_acctbal   DECIMAL(15,2)  NOT NULL,
    s_comment   VARCHAR(101)   NOT NULL,
    PRIMARY KEY (s_suppkey));

CREATE TABLE partsupp (
    ps_partkey    INTEGER        NOT NULL,
    ps_suppkey    INTEGER        NOT NULL,
    ps_availqty   INTEGER        NOT NULL,
    ps_supplycost DECIMAL(15,2)  NOT NULL,
    ps_comment    VARCHAR(199)   NOT NULL,
    PRIMARY KEY (ps_partkey, ps_suppkey));

CREATE TABLE customer (
    c_custkey    INTEGER        NOT NULL,
    c_name       VARCHAR(25)    NOT NULL,
    c_address    VARCHAR(40)    NOT NULL,
    c_nationkey  INTEGER        NOT NULL,
    c_phone      CHAR(15)       NOT NULL,
    c_acctbal    DECIMAL(15,2)  NOT NULL,
    c_mktsegment CHAR(10)       NOT NULL,
    c_comment    VARCHAR(117)   NOT NULL,
    PRIMARY KEY (c_custkey));

CREATE TABLE orders (
    o_orderkey      INTEGER        NOT NULL,
    o_custkey       INTEGER        NOT NULL,
    o_orderstatus   CHAR(1)        NOT NULL,
    o_totalprice    DECIMAL(15,2)  NOT NULL,
    o_orderdate     DATE           NOT NULL,
    o_orderpriority CHAR(15)       NOT NULL,
    o_clerk         CHAR(15)       NOT NULL,
    o_shippriority  INTEGER        NOT NULL,
    o_comment       VARCHAR(79)    NOT NULL,
    PRIMARY KEY (o_orderkey));

CREATE TABLE lineitem (
    l_orderkey      INTEGER        NOT NULL,
    l_partkey       INTEGER        NOT NULL,
    l_suppkey       INTEGER        NOT NULL,
    l_linenumber    INTEGER        NOT NULL,
    l_quantity      DECIMAL(15,2)  NOT NULL,
    l_extendedprice DECIMAL(15,2)  NOT NULL,
    l_discount      DECIMAL(15,2)  NOT NULL,
    l_tax           DECIMAL(15,2)  NOT NULL,
    l_returnflag    CHAR(1)        NOT NULL,
    l_linestatus    CHAR(1)        NOT NULL,
    l_shipdate      DATE           NOT NULL,
    l_commitdate    DATE           NOT NULL,
    l_receiptdate   DATE           NOT NULL,
    l_shipinstruct  CHAR(25)       NOT NULL,
    l_shipmode      CHAR(10)       NOT NULL,
    l_comment       VARCHAR(44)    NOT NULL,
    PRIMARY KEY (l_orderkey, l_linenumber));
"""

# Spec 1.4.2 relationships. The two single-column lineitem FKs (part, supplier) follow
# from the composite partsupp FK plus partsupp's own FKs; declaring them gives the
# catalogue the tightest referential wall per column, matching exp14's construction.
FOREIGN_KEYS = [
    "ALTER TABLE nation   ADD FOREIGN KEY (n_regionkey) REFERENCES region  (r_regionkey)",
    "ALTER TABLE supplier ADD FOREIGN KEY (s_nationkey) REFERENCES nation  (n_nationkey)",
    "ALTER TABLE customer ADD FOREIGN KEY (c_nationkey) REFERENCES nation  (n_nationkey)",
    "ALTER TABLE partsupp ADD FOREIGN KEY (ps_partkey)  REFERENCES part    (p_partkey)",
    "ALTER TABLE partsupp ADD FOREIGN KEY (ps_suppkey)  REFERENCES supplier(s_suppkey)",
    "ALTER TABLE orders   ADD FOREIGN KEY (o_custkey)   REFERENCES customer(c_custkey)",
    "ALTER TABLE lineitem ADD FOREIGN KEY (l_orderkey)  REFERENCES orders  (o_orderkey)",
    "ALTER TABLE lineitem ADD FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp (ps_partkey, ps_suppkey)",
    "ALTER TABLE lineitem ADD FOREIGN KEY (l_partkey)   REFERENCES part    (p_partkey)",
    "ALTER TABLE lineitem ADD FOREIGN KEY (l_suppkey)   REFERENCES supplier(s_suppkey)",
]

# PostgreSQL does not index FK columns automatically; without these the planner has no
# index-nestloop alternative on any join and the study could only ever observe hash-join
# monoculture. Standard practice in the cardinality-estimation literature.
FK_INDEXES = [
    "CREATE INDEX ON nation   (n_regionkey)",
    "CREATE INDEX ON supplier (s_nationkey)",
    "CREATE INDEX ON customer (c_nationkey)",
    "CREATE INDEX ON partsupp (ps_suppkey)",
    "CREATE INDEX ON orders   (o_custkey)",
    "CREATE INDEX ON lineitem (l_partkey)",
    "CREATE INDEX ON lineitem (l_suppkey)",
]


def duckdb_generate(sf, csv_dir):
    import duckdb
    os.makedirs(csv_dir, exist_ok=True)
    done = os.path.join(csv_dir, "_DONE")
    if os.path.exists(done):
        print(f"CSVs already generated in {csv_dir}")
        return
    db = os.path.join(WORK, f"tpch_sf{sf:g}.duckdb")
    con = duckdb.connect(db)
    con.execute("PRAGMA memory_limit='1400MB'")
    con.execute("PRAGMA threads=8")
    con.execute(f"PRAGMA temp_directory='{WORK}/duck_tmp'")
    con.execute("INSTALL tpch; LOAD tpch;")
    have = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if "lineitem" not in have:
        t0 = time.time()
        if sf >= 5:
            for step in range(10):  # chunked generation keeps peak memory bounded
                con.execute(f"CALL dbgen(sf={sf}, children=10, step={step})")
                print(f"  dbgen step {step + 1}/10 ({time.time() - t0:.0f}s)", flush=True)
        else:
            con.execute(f"CALL dbgen(sf={sf})")
        print(f"dbgen sf={sf}: {time.time() - t0:.1f}s")
    for t in TABLES:
        path = os.path.join(csv_dir, f"{t}.csv")
        con.execute(f"COPY {t} TO '{path}' (FORMAT csv, HEADER)")
        print(f"  wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)
    con.close()
    open(done, "w").close()


def extract_queries():
    import duckdb
    os.makedirs(QUERY_DIR, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    rows = con.execute("SELECT query_nr, query FROM tpch_queries() ORDER BY query_nr").fetchall()
    con.close()
    assert len(rows) == 22, f"expected 22 queries, got {len(rows)}"
    for nr, q in rows:
        with open(os.path.join(QUERY_DIR, f"q{int(nr):02d}.sql"), "w") as f:
            f.write(q.strip() + "\n")
    print(f"extracted 22 queries to {QUERY_DIR}")


def pg_load(dsn, sf, csv_dir):
    import psycopg2
    con = psycopg2.connect(dsn)
    con.autocommit = True
    cur = con.cursor()
    cur.execute(DDL)
    print("schema created (8 tables, PKs declared)")
    for t in TABLES:
        t0 = time.time()
        with open(os.path.join(csv_dir, f"{t}.csv")) as f:
            cur.copy_expert(f"COPY {t} FROM STDIN (FORMAT csv, HEADER)", f)
        print(f"  loaded {t:9s} {time.time() - t0:6.1f}s", flush=True)
    for stmt in FOREIGN_KEYS:
        cur.execute(stmt)
    print("foreign keys declared (10 constraints)")
    for stmt in FK_INDEXES:
        cur.execute(stmt)
    print("FK indexes built (7)")

    exp = EXPECTED.get(int(sf)) if float(sf).is_integer() else None
    bad = []
    for t in TABLES:
        cur.execute(f"SELECT count(*) FROM {t}")
        n = cur.fetchone()[0]
        want = exp[t] if exp else None
        mark = "OK" if (want is None or n == want) else f"MISMATCH (spec: {want})"
        if want is not None and n != want:
            bad.append(t)
        print(f"  {t:9s} {n:>10,} rows  {mark}")
    con.close()
    if bad:
        sys.exit(f"row-count verification FAILED for: {bad}")
    print("row counts verified against the TPC-H spec" if exp else "row counts printed (no spec entry for this sf)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("HLL_PG_DSN", "host=localhost port=5433 dbname=tpch user=postgres"))
    ap.add_argument("--sf", type=float, default=1.0)
    a = ap.parse_args()
    csv_dir = os.path.join(WORK, f"csv_sf{a.sf:g}")
    duckdb_generate(a.sf, csv_dir)
    extract_queries()
    pg_load(a.dsn, a.sf, csv_dir)


if __name__ == "__main__":
    main()
