#!/usr/bin/env bash
# Idempotent setup for the UltraLogLog (hash4j) harness used by exp6.
# JDK: the portable Temurin under ull/jdk/ if present, else the system JDK (javac on PATH),
# else a PINNED portable Temurin 17.0.20.1+1 (linux/x86_64, ~190 MB, no root) verified by
# SHA-256. The hash4j jar from Maven Central is pinned by version and SHA-256. The java
# binary that compiled the driver is recorded in ull/.java_path so exp6 runs the same JVM.
# Exits 0 on success; non-zero with a message otherwise, so repro.sh can skip exp6 gracefully.
set -euo pipefail
cd "$(dirname "$0")"

HASH4J_VER=0.30.0
JAR="hash4j-${HASH4J_VER}.jar"
JAR_SHA256=d3224b113ea835ce3452097747e5209ec760cf443dbed770d6ba68a0d480178f
JDK_URL="https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.20.1%2B1/OpenJDK17U-jdk_x64_linux_hotspot_17.0.20.1_1.tar.gz"
JDK_SHA256=3808d1d15e3ec6bd5b84057fb5d84c33d8a1536a258146bcea2e603fc726e08e

# 1) JDK with javac
if [[ -x jdk/bin/javac ]]; then
  JAVAC="$PWD/jdk/bin/javac"; JAVA="$PWD/jdk/bin/java"
elif command -v javac >/dev/null 2>&1; then
  JAVAC="$(command -v javac)"; JAVA="$(command -v java)"
else
  if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "  no javac on PATH, and the pinned portable JDK is linux/x86_64 only -- install a JDK 17+ and re-run"; exit 1
  fi
  echo "  downloading portable Temurin JDK 17.0.20.1+1 (pinned, ~190 MB) ..."
  curl -fsSL -o jdk17.tar.gz "$JDK_URL"
  echo "$JDK_SHA256  jdk17.tar.gz" | sha256sum -c --quiet - \
    || { echo "!! JDK checksum mismatch -- refusing to extract"; rm -f jdk17.tar.gz; exit 1; }
  mkdir -p jdk && tar xzf jdk17.tar.gz -C jdk --strip-components=1 && rm -f jdk17.tar.gz
  JAVAC="$PWD/jdk/bin/javac"; JAVA="$PWD/jdk/bin/java"
fi

# 2) hash4j jar (pinned version, verified)
if [[ ! -f "$JAR" ]]; then
  echo "  downloading $JAR ..."
  curl -fsSL -o "$JAR" "https://repo1.maven.org/maven2/com/dynatrace/hash4j/hash4j/${HASH4J_VER}/${JAR}"
fi
echo "$JAR_SHA256  $JAR" | sha256sum -c --quiet - || { echo "!! $JAR checksum mismatch -- refusing to use it"; exit 1; }

# 3) compile the driver and record the JVM for exp6
"$JAVAC" -cp "$JAR" UllDriver.java
echo "$JAVA" > .java_path
echo "  ULL harness ready (javac: $("$JAVAC" -version 2>&1); java: $JAVA)"
