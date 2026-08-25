#!/usr/bin/env python3
"""Phase 2: inject raw vs clipped NDV into a real optimizer and measure what changes.

Design, in short: PostgreSQL exposes
`ALTER TABLE t ALTER COLUMN c SET (n_distinct = v)`, which overrides the catalogue's
distinct-count estimate and survives ANALYZE. That is exactly the injection point this
study needs, so no database has to be patched.

Run `--check-only` FIRST. It answers the five questions that decide whether the design
works at all, in about ten seconds:

  1. is a server reachable?
  2. is an n_distinct override accepted and visible in pg_stats?
  3. does it survive ANALYZE?
  4. is a value EXCEEDING the row count accepted, or silently clamped?

(4) is the important one. The paper's sharpest database finding is that a raw sketch
emits NDV > row count in 50-56% of key-column ANALYZE runs. If PostgreSQL accepts such a
value, we can measure what that impossible statistic does to a plan. If it clamps, that is
itself a one-sentence finding -- the DBMS defends itself where the sketch did not -- and
the study narrows to Q-error.

Pipeline (after tpch_load.py has loaded TPC-H and extracted the queries):

    python3 exp16_catalog_injection.py --check-only
    python3 exp16_catalog_injection.py --ndv       # exact + 20 HLL draws per column
    python3 exp16_catalog_injection.py --capture   # inject each catalogue, EXPLAIN 22 queries
    python3 exp16_catalog_injection.py --analyze   # plan diffs + Q-error -> exp16_results.json
    python3 exp16_catalog_injection.py --runtime   # 5-rep timings, changed plans only

Conditions per hash draw: `exact` (true NDV), `raw` (DataSketches HLL, lg_k=14, the
paper's deployment precision), `clipped` = min(raw, R), `clip_ri` = min(raw, R, rows of
the table referenced by a declared FK). `pg_default` (no override, PostgreSQL's own
sampling estimator) is captured once as context; it is not one of the four
conditions. The capture loop starts with the exact-vs-exact sanity gate: two full
inject+ANALYZE+EXPLAIN cycles with identical exact statistics must produce identical
canonical plans for all 22 queries, otherwise plan-change counts would be meaningless.
"""
import argparse
import glob
import gzip
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE
WORK = os.environ.get("PG_HOME") or os.path.expanduser("~/hll_phase2_pg")
QUERY_DIR = os.path.join(HERE, "queries_pg")

CONDITIONS = ["exact", "raw", "clipped", "clip_ri"]
TABLES = ["region", "nation", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]
LG_K = 14                     # m = 16384: the paper's deployment precision
DRAWS = 20                    # independent hash draws = whole-catalogue replicates

# Tightest referential wall per FK column: the row count of the referenced table under
# any declared FK containing the column (single-column FKs beat the composite one).
RI_REF = {
    ("nation", "n_regionkey"): "region",
    ("supplier", "s_nationkey"): "nation",
    ("customer", "c_nationkey"): "nation",
    ("partsupp", "ps_partkey"): "part",
    ("partsupp", "ps_suppkey"): "supplier",
    ("orders", "o_custkey"): "customer",
    ("lineitem", "l_orderkey"): "orders",
    ("lineitem", "l_partkey"): "part",
    ("lineitem", "l_suppkey"): "supplier",
}


def connect(dsn):
    try:
        import psycopg2                                   # noqa: F401
    except ImportError:
        sys.exit("psycopg2 not installed:  pip install psycopg2-binary")
    import psycopg2
    con = psycopg2.connect(dsn)
    con.autocommit = True
    return con


def check_only(dsn):
    """The five feasibility probes. Prints a go/no-go and exits nonzero on no-go."""
    con = connect(dsn)
    cur = con.cursor()
    ok = True

    cur.execute("SELECT version();")
    print(f"[1/5] server reachable: {cur.fetchone()[0].split(',')[0]}")

    cur.execute("DROP TABLE IF EXISTS _hllprobe;")
    cur.execute("CREATE TABLE _hllprobe (c int);")
    cur.execute("INSERT INTO _hllprobe SELECT g FROM generate_series(1, 1000) g;")
    cur.execute("ANALYZE _hllprobe;")
    rows = 1000

    cur.execute("ALTER TABLE _hllprobe ALTER COLUMN c SET (n_distinct = 700);")
    cur.execute("ANALYZE _hllprobe;")
    cur.execute("SELECT n_distinct FROM pg_stats WHERE tablename='_hllprobe';")
    got = cur.fetchone()[0]
    print(f"[2/5] override accepted and visible: n_distinct = {got} (set 700)")
    if got != 700:
        ok = False
        print("      ^ NOT honoured -- the whole injection design needs rethinking")

    cur.execute("ANALYZE _hllprobe;")
    cur.execute("SELECT n_distinct FROM pg_stats WHERE tablename='_hllprobe';")
    got2 = cur.fetchone()[0]
    print(f"[3/5] survives a second ANALYZE: n_distinct = {got2}")
    if got2 != 700:
        ok = False
        print("      ^ overwritten by ANALYZE -- inject AFTER analyzing, or re-inject")

    # The one that matters: an impossible statistic, NDV > row count.
    impossible = int(rows * 1.05)
    cur.execute(f"ALTER TABLE _hllprobe ALTER COLUMN c SET (n_distinct = {impossible});")
    cur.execute("ANALYZE _hllprobe;")
    cur.execute("SELECT n_distinct FROM pg_stats WHERE tablename='_hllprobe';")
    got3 = cur.fetchone()[0]
    print(f"[4/5] impossible value ({impossible} > {rows} rows): stored as {got3}")
    if got3 == impossible:
        print("      -> ACCEPTED. We can measure what an impossible NDV does to a plan.")
        print("         This is the paper's invariant finding, made operational.")
    else:
        print("      -> CLAMPED/ALTERED. Finding in its own right; narrow to Q-error.")

    # Probe 5: does the impossible statistic reach a COST on a column that has no unique
    # index? The GROUP-BY path clamps distinct counts at the row count; the join path
    # (eqjoinsel) divides by them unclamped. Inject 1.5x the row count and read both.
    cur.execute(f"ALTER TABLE _hllprobe ALTER COLUMN c SET (n_distinct = {int(rows * 1.5)});")
    cur.execute("ANALYZE _hllprobe;")
    cur.execute("EXPLAIN (FORMAT JSON) SELECT c, count(*) FROM _hllprobe GROUP BY c;")
    grp = cur.fetchone()[0][0]["Plan"]["Plan Rows"]
    cur.execute("EXPLAIN (FORMAT JSON) SELECT * FROM _hllprobe a JOIN _hllprobe b ON a.c = b.c;")
    jn = cur.fetchone()[0][0]["Plan"]["Plan Rows"]
    print(f"[5/5] n_distinct={int(rows*1.5)} on {rows} rows, no unique index: "
          f"GROUP BY estimate {grp} ({'clamped at the row count' if grp <= rows else 'NOT clamped'}); "
          f"self-join estimate {jn} rows (true {rows})")
    if jn < rows:
        print("      -> the join path consumed the impossible statistic: the impossible NDV reaches")
        print("         a cost wherever no unique index overrides it.")

    cur.execute("DROP TABLE _hllprobe;")
    con.close()
    print()
    print("GO" if ok else "NO-GO -- read the notes above before writing any more code")
    return 0 if ok else 1


def canonical_plan(node):
    """Reduce an EXPLAIN (FORMAT JSON) node to the shape we care about.

    Costs and row estimates differ under every condition, so diffing raw JSON would report
    a change for all 22 queries and mean nothing. What counts as a real plan change is the
    join order, the physical operator at each join, and the scan type per relation.
    """
    t = node.get("Node Type", "")
    parts = [t]
    if "Relation Name" in node:
        parts.append(node["Relation Name"])
    if "Join Type" in node:
        parts.append(node["Join Type"])
    kids = [canonical_plan(k) for k in node.get("Plans", [])]
    return f"({' '.join(parts)}{''.join(' ' + k for k in kids)})"


def join_structure(node):
    """The plan-change definition: join order, physical join operator,
    scan type per relation. Aggregation strategy, sorts, gathers etc. are ignored here;
    canonical_plan keeps them and serves as the stricter full-shape readout."""
    t = node.get("Node Type", "")
    kids = [join_structure(k) for k in node.get("Plans", [])]
    kids = [k for k in kids if k]
    is_join = t in ("Nested Loop", "Hash Join", "Merge Join")
    is_scan = "Relation Name" in node
    if is_join or is_scan:
        parts = [t]
        if is_scan:
            parts.append(node["Relation Name"])
        if "Join Type" in node:
            parts.append(node["Join Type"])
        return f"({' '.join(parts)}{''.join(' ' + k for k in kids)})"
    if not kids:
        return ""
    if len(kids) == 1:
        return kids[0]
    return f"[{' '.join(kids)}]"


def qerror(node, acc):
    """Collect max(est/act, act/est) per node from EXPLAIN (ANALYZE) output."""
    est = node.get("Plan Rows")
    act = node.get("Actual Rows")
    if est is not None and act is not None:
        e, a = max(float(est), 1.0), max(float(act), 1.0)
        acc.append(max(e / a, a / e))
    for k in node.get("Plans", []):
        qerror(k, acc)
    return acc


def est_rows(node, acc):
    """Per-node estimated rows in a fixed (pre-order) traversal, for same-shape diffs."""
    acc.append(float(node.get("Plan Rows", 0)))
    for k in node.get("Plans", []):
        est_rows(k, acc)
    return acc


# ---------------------------------------------------------------------------- queries

def load_queries():
    files = sorted(glob.glob(os.path.join(QUERY_DIR, "q*.sql")))
    if len(files) != 22:
        sys.exit(f"expected 22 query files in {QUERY_DIR}, found {len(files)} -- run tpch_load.py first")
    out = []
    for f in files:
        nr = int(os.path.basename(f)[1:3])
        text = open(f).read().strip()
        pre = post = None
        main = text
        if "create view" in text.lower():
            stmts = [s.strip() for s in text.split(";") if s.strip()]
            pre = [s for s in stmts if s.lower().startswith("create view")]
            post = [s for s in stmts if s.lower().startswith("drop view")]
            body = [s for s in stmts if not s.lower().startswith(("create view", "drop view"))]
            assert len(body) == 1, f"q{nr}: unexpected statement split"
            main = body[0]
        out.append({"nr": nr, "sql": main, "pre": pre or [], "post": post or []})
    return out


def explain(cur, q, analyze=False):
    for s in q["pre"]:
        cur.execute(s)
    try:
        mode = "ANALYZE, TIMING OFF, " if analyze else ""
        cur.execute(f"EXPLAIN ({mode}FORMAT JSON) {q['sql']}")
        row = cur.fetchone()[0]
        doc = json.loads(row) if isinstance(row, str) else row
        return doc[0]
    finally:
        for s in q["post"]:
            cur.execute(s)


# ---------------------------------------------------------------------------- NDV

def table_columns(cur):
    cur.execute("""SELECT table_name, column_name FROM information_schema.columns
                   WHERE table_schema='public' ORDER BY table_name, ordinal_position""")
    cols = {}
    for t, c in cur.fetchall():
        cols.setdefault(t, []).append(c)
    return {t: cols[t] for t in TABLES}


def _duck_connect(duck_path):
    import duckdb
    con = duckdb.connect(duck_path, read_only=True)
    # cap this worker's DuckDB memory: the default budget (~80% of RAM) times N workers
    # OOM-killed workers at sf=10, and a SIGKILLed Pool worker hangs the whole pool
    tmpdir = os.path.join(WORK, "duck_worker_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    con.execute("PRAGMA memory_limit='500MB'")
    con.execute("PRAGMA threads=1")
    con.execute(f"PRAGMA temp_directory='{tmpdir}'")
    return con


def _ndv_worker(job):
    """Exact NDV + `draws` salted HLL estimates for one column, streamed in batches."""
    duck_path, table, column, draws, dsn = job
    import datasketches as ds
    con = _duck_connect(duck_path)
    # Exact NDV in DuckDB when it fits the cap. Its 0.10 hash aggregate cannot spill on
    # very-high-NDV string columns (verified: temp_directory set and honored, zero spill
    # files, clean OOM on orders.o_comment at sf=10), so those columns fall back to
    # PostgreSQL's count(DISTINCT), whose external sort is bounded by work_mem.
    exact_source = "duckdb"
    try:
        exact = con.execute(
            f'SELECT count(*) FROM (SELECT "{column}" FROM {table} GROUP BY "{column}")').fetchone()[0]
    except Exception as e:
        if "Out of Memory" not in str(e):
            raise
        con.close()
        con = _duck_connect(duck_path)         # fresh connection after the aborted query
        import psycopg2
        pg = psycopg2.connect(dsn)
        pg.autocommit = True
        pcur = pg.cursor()
        pcur.execute("SET work_mem = '128MB'")
        pcur.execute(f'SELECT count(DISTINCT "{column}") FROM {table}')
        exact = pcur.fetchone()[0]
        pg.close()
        exact_source = "postgres"
    sketches = [ds.hll_sketch(LG_K) for _ in range(draws)]
    salts = [f"|{j}" for j in range(draws)]
    res = con.execute(f'SELECT "{column}" FROM {table}')
    while True:
        rows = res.fetchmany(200_000)
        if not rows:
            break
        for (v,) in rows:
            s = str(v)
            for sk, salt in zip(sketches, salts):
                sk.update(s + salt)
    con.close()
    return table, column, exact, [sk.get_estimate() for sk in sketches], exact_source


def ndv_compute(dsn, sf, draws, jobs):
    from multiprocessing import Pool
    duck_path = os.path.join(WORK, f"tpch_sf{sf:g}.duckdb")
    if not os.path.exists(duck_path):
        sys.exit(f"{duck_path} missing -- run tpch_load.py first")
    con = connect(dsn)
    cur = con.cursor()
    cols = table_columns(cur)
    rows = {}
    for t in TABLES:
        cur.execute(f"SELECT count(*) FROM {t}")
        rows[t] = cur.fetchone()[0]

    # partial-result cache: completed columns survive a crash/OOM and are not recomputed
    partial_path = os.path.join(OUT, f"exp16_ndv_sf{sf:g}.partial.jsonl")
    out = []
    done = set()
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("draws") == draws:
                    cell = {k: rec[k] for k in ("table", "column", "rows", "exact", "ri_wall", "raw")}
                    cell["exact_source"] = rec.get("exact_source", "duckdb")
                    out.append(cell)
                    done.add((rec["table"], rec["column"]))
        print(f"resuming: {len(done)} columns loaded from {partial_path}", flush=True)

    work = [(duck_path, t, c, draws, dsn) for t in TABLES for c in cols[t] if (t, c) not in done]
    print(f"computing NDV for {len(work)} columns, {draws} draws, lg_k={LG_K}, {jobs} workers", flush=True)
    t0 = time.time()
    with Pool(jobs) as pool:
        for table, column, exact, raw, exact_source in pool.imap_unordered(_ndv_worker, work):
            ref = RI_REF.get((table, column))
            cell = {"table": table, "column": column, "rows": rows[table],
                    "exact": exact, "ri_wall": rows[ref] if ref else None,
                    "raw": raw, "exact_source": exact_source}
            out.append(cell)
            with open(partial_path, "a") as f:
                f.write(json.dumps(dict(cell, draws=draws)) + "\n")
            imp = sum(1 for r in raw if r > rows[table]) / len(raw)
            print(f"  {table:9s}.{column:15s} exact={exact:>9,}  raw_mean={sum(raw)/len(raw):>12,.1f}  "
                  f"raw>R in {imp*100:3.0f}% of draws   ({time.time()-t0:5.0f}s)", flush=True)

    # cross-check the generator-side exact NDV against the loaded PostgreSQL data:
    # any CSV/load corruption would surface here. Columns whose exact already came from
    # PostgreSQL (DuckDB OOM fallback) carry no independent second value and are skipped.
    cur.execute("SET work_mem = '128MB'")
    mismatch, independent, fallback = [], 0, 0
    for cell in out:
        if cell.get("exact_source") == "postgres":
            fallback += 1
            continue
        cur.execute(f'SELECT count(DISTINCT "{cell["column"]}") FROM {cell["table"]}')
        pg_exact = cur.fetchone()[0]
        independent += 1
        if pg_exact != cell["exact"]:
            mismatch.append((cell["table"], cell["column"], cell["exact"], pg_exact))
    if mismatch:
        sys.exit(f"exact-NDV cross-check FAILED (duckdb vs postgres): {mismatch}")
    print(f"exact-NDV cross-check vs PostgreSQL: {independent} columns match"
          + (f"; {fallback} exact values sourced from PostgreSQL itself (DuckDB OOM fallback), "
             "no independent check possible" if fallback else ""))
    con.close()

    order = {(t, c): i for i, (t, cs) in enumerate([(t, cols[t]) for t in TABLES])
             for c in cs}
    out.sort(key=lambda cell: (TABLES.index(cell["table"]), cols[cell["table"]].index(cell["column"])))
    doc = {"params": {"sf": sf, "lg_k": LG_K, "draws": draws, "salt": "value + '|' + draw_index",
                      "sketch": "datasketches.hll_sketch (HLL_4 default), same as exp13"},
           "table_rows": rows, "columns": out}
    path = os.path.join(OUT, f"exp16_ndv_sf{sf:g}.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"wrote {path} ({time.time()-t0:.0f}s total)")


# ---------------------------------------------------------------------------- capture

def attnum_map(cur):
    cur.execute("""SELECT c.relname, a.attname, a.attnum
                   FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
                   JOIN pg_namespace n ON n.oid = c.relnamespace
                   WHERE n.nspname='public' AND c.relkind='r' AND a.attnum > 0
                     AND NOT a.attisdropped""")
    return {(t, c): n for t, c, n in cur.fetchall()}


def condition_values(ndv, condition, draw):
    """(table, column, value) for every column under one condition/draw."""
    vals = []
    for cell in ndv["columns"]:
        r = cell["rows"]
        if condition == "exact":
            v = float(cell["exact"])
        else:
            v = float(cell["raw"][draw])
            if condition in ("clipped", "clip_ri"):
                v = min(v, float(r))
            if condition == "clip_ri" and cell["ri_wall"] is not None:
                v = min(v, float(cell["ri_wall"]))
        vals.append((cell["table"], cell["column"], max(v, 1.0)))
    return vals


def inject(cur, values, attnums):
    """Write stadistinct directly into pg_statistic -- the same field the documented
    `ALTER ... SET (n_distinct)` route populates via ANALYZE (verified equivalent at the
    planner: identical estimates for identical values, including impossible ones).

    Why not the documented route per catalogue: every ANALYZE resamples the MCV lists and
    histograms, and that sampling noise alone flips near-tie plans (observed on q04,
    exact-vs-exact: the o_orderdate range estimate moved 23,832 -> 22,690 and the partial
    aggregation strategy flipped). With one base ANALYZE and surgical stadistinct
    updates, the conditions differ in NOTHING but the statistic under study.
    """
    for t, c, v in values:
        cur.execute("UPDATE pg_statistic SET stadistinct = %s "
                    "WHERE starelid = %s::regclass AND staattnum = %s AND NOT stainherit",
                    (v, t, attnums[(t, c)]))
        if cur.rowcount != 1:
            sys.exit(f"stadistinct update touched {cur.rowcount} rows for {t}.{c} -- "
                     "base ANALYZE missing?")


def reset_overrides(cur, ndv):
    for cell in ndv["columns"]:
        cur.execute(f'ALTER TABLE {cell["table"]} ALTER COLUMN "{cell["column"]}" RESET (n_distinct)')
    for t in TABLES:
        cur.execute(f"ANALYZE {t}")


def stored_ndistinct(cur):
    cur.execute("""SELECT tablename, attname, n_distinct FROM pg_stats
                   WHERE schemaname='public' ORDER BY tablename, attname""")
    return {f"{t}.{c}": float(v) for t, c, v in cur.fetchall()}


def verify_injection(cur, values):
    """Every injected value must be what pg_stats now reports, else the instrument lies."""
    got = stored_ndistinct(cur)
    bad = []
    for t, c, v in values:
        g = got.get(f"{t}.{c}")
        if g is None or abs(g - v) > max(1e-6 * abs(v), 1e-6):
            bad.append((t, c, v, g))
    return bad


def capture_one(cur, queries, label, do_analyze, timeout_ms=600_000):
    cur.execute(f"SET statement_timeout = {timeout_ms}")
    doc = {"label": label, "queries": {}}
    for q in queries:
        entry = {}
        plan = explain(cur, q, analyze=False)
        entry["plan"] = plan["Plan"]
        entry["canonical"] = canonical_plan(plan["Plan"])
        if do_analyze:
            try:
                aplan = explain(cur, q, analyze=True)
                entry["analyze_plan"] = aplan["Plan"]
                entry["exec_time_ms"] = aplan.get("Execution Time")
            except Exception as e:                        # timeout is a finding, not a crash
                cur.connection.rollback() if not cur.connection.autocommit else None
                entry["analyze_error"] = f"{type(e).__name__}: {e}".strip()
        doc["queries"][f"q{q['nr']:02d}"] = entry
    cur.execute("SET statement_timeout = 0")
    return doc


def cap_path(sf, label):
    d = os.path.join(OUT, f"capture_sf{sf:g}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{label}.json.gz")


def save_capture(sf, doc):
    with gzip.open(cap_path(sf, doc["label"]), "wt") as f:
        json.dump(doc, f)


def load_capture(sf, label):
    p = cap_path(sf, label)
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt") as f:
        return json.load(f)


def capture(dsn, sf, draws, analyze_draws):
    ndv = json.load(open(os.path.join(OUT, f"exp16_ndv_sf{sf:g}.json")))
    assert ndv["params"]["draws"] >= draws
    queries = load_queries()
    con = connect(dsn)
    cur = con.cursor()
    attnums = attnum_map(cur)

    plan_labels = []

    def run(label, values, do_analyze):
        plan_labels.append(label)
        if load_capture(sf, label) is not None:
            print(f"  [{label}] already captured, skipping", flush=True)
            return
        t0 = time.time()
        if values is None:
            # the base: clear any leftover reloptions, one plain ANALYZE; its MCV lists
            # and histograms are the shared substrate every later condition reuses
            reset_overrides(cur, ndv)
        else:
            inject(cur, values, attnums)
            bad = verify_injection(cur, values)
            if bad:
                sys.exit(f"[{label}] injection verification FAILED: {bad[:5]} ...")
        doc = capture_one(cur, queries, label, do_analyze)
        doc["stored_n_distinct"] = stored_ndistinct(cur)
        save_capture(sf, doc)
        errs = sum(1 for e in doc["queries"].values() if "analyze_error" in e)
        print(f"  [{label}] captured in {time.time()-t0:.0f}s"
              + (f"  ({errs} analyze errors)" if errs else ""), flush=True)

    # context condition: PostgreSQL's own sampling estimator, no overrides
    run("pg_default", None, do_analyze=True)

    # the sanity gate: exact injected twice -> canonical plans must be identical
    exact_vals = condition_values(ndv, "exact", 0)
    run("exact", exact_vals, do_analyze=True)
    run("exact_replicate", exact_vals, do_analyze=True)
    a = load_capture(sf, "exact")
    b = load_capture(sf, "exact_replicate")
    diffs = [k for k in a["queries"] if a["queries"][k]["canonical"] != b["queries"][k]["canonical"]]
    if diffs:
        sys.exit(f"SANITY GATE FAILED: exact-vs-exact canonical plans differ for {diffs}. "
                 "Fix the canonicalisation (or ANALYZE-sampling instability) before reporting anything.")
    print(f"SANITY GATE PASSED: exact vs exact -> 0 of {len(a['queries'])} canonical diffs", flush=True)

    for j in range(draws):
        for cond in ("raw", "clipped", "clip_ri"):
            run(f"{cond}_d{j:02d}", condition_values(ndv, cond, j), do_analyze=(j < analyze_draws))
    con.close()
    print("capture complete:", len(plan_labels), "catalogues")


# ---------------------------------------------------------------------------- analyze

def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def analyze(sf, draws):
    ndv = json.load(open(os.path.join(OUT, f"exp16_ndv_sf{sf:g}.json")))
    exact = load_capture(sf, "exact")
    exact_rep = load_capture(sf, "exact_replicate")
    pg_default = load_capture(sf, "pg_default")
    if not exact or not exact_rep:
        sys.exit("missing exact captures -- run --capture first")
    qnames = sorted(exact["queries"])

    # gate, recomputed from stored artefacts
    gate_diffs = [q for q in qnames
                  if exact["queries"][q]["canonical"] != exact_rep["queries"][q]["canonical"]]
    gate_js = [q for q in qnames
               if join_structure(exact["queries"][q]["plan"]) != join_structure(exact_rep["queries"][q]["plan"])]

    results = {"params": {"sf": sf, "draws": draws, "lg_k": LG_K,
                          "conditions": CONDITIONS + ["pg_default (context)"],
                          "injection": "pg_statistic stadistinct surgery over one shared base ANALYZE",
                          "canonicalisation": "full shape (canonical_plan) + join structure (join_structure)"},
               "gate": {"exact_vs_exact_plan_diffs": len(gate_diffs), "queries": gate_diffs,
                        "exact_vs_exact_join_structure_diffs": len(gate_js)}}

    # per-column NDV summary: impossible rate, wall effects
    ndv_summary = []
    for cell in ndv["columns"]:
        r, raws = cell["rows"], cell["raw"][:draws]
        imp = sum(1 for v in raws if v > r) / len(raws)
        ri = cell["ri_wall"]
        ri_binds = sum(1 for v in raws if ri is not None and min(v, r) > ri) / len(raws)
        rel = [abs(v - cell["exact"]) / cell["exact"] for v in raws]
        ndv_summary.append({"table": cell["table"], "column": cell["column"], "rows": r,
                            "exact": cell["exact"], "ri_wall": ri,
                            "raw_mean": sum(raws) / len(raws),
                            "raw_rel_err_mean": sum(rel) / len(rel),
                            "raw_rel_err_max": max(rel),
                            "impossible_rate": imp, "ri_wall_binds_rate": ri_binds})
    results["ndv"] = ndv_summary
    results["sketch_rel_err_max_over_all_columns"] = max(c["raw_rel_err_max"] for c in ndv_summary)

    # exact-vs-exact estimate noise floor (ANALYZE resampling of MCVs/histograms)
    noise = []
    for q in qnames:
        e1 = est_rows(exact["queries"][q]["plan"], [])
        e2 = est_rows(exact_rep["queries"][q]["plan"], [])
        if len(e1) == len(e2):
            noise += [abs(a - b) / max(a, b, 1.0) for a, b in zip(e1, e2)]
    results["gate"]["est_rows_noise_max_reldiff"] = max(noise) if noise else None

    # plan changes per condition (vs the exact catalogue), per draw and per query,
    # under both readouts: full shape (canonical_plan) and join structure (the join-structure
    # literal definition)
    plan_changes = {}
    for cond in ("raw", "clipped", "clip_ri"):
        per_query = {q: [] for q in qnames}        # full shape
        per_query_js = {q: [] for q in qnames}     # join structure only
        for j in range(draws):
            cap = load_capture(sf, f"{cond}_d{j:02d}")
            for q in qnames:
                per_query[q].append(cap["queries"][q]["canonical"] != exact["queries"][q]["canonical"])
                per_query_js[q].append(
                    join_structure(cap["queries"][q]["plan"]) != join_structure(exact["queries"][q]["plan"]))
        plan_changes[cond] = {
            "queries_changed_any_draw": sorted(q for q in qnames if any(per_query[q])),
            "n_queries_changed_any_draw": sum(1 for q in qnames if any(per_query[q])),
            "mean_changed_per_draw": sum(sum(v) for v in per_query.values()) / draws,
            "per_query_change_rate": {q: sum(per_query[q]) / draws for q in qnames if any(per_query[q])},
            "join_structure": {
                "queries_changed_any_draw": sorted(q for q in qnames if any(per_query_js[q])),
                "n_queries_changed_any_draw": sum(1 for q in qnames if any(per_query_js[q])),
                "mean_changed_per_draw": sum(sum(v) for v in per_query_js.values()) / draws,
            },
        }

    # pairwise within the same draw (raw/clipped/clip_ri), plus each condition vs exact:
    # same-shape per-node estimate differences show whether an injected difference even
    # reaches the cost model when no plan flips
    pair_stats = {}
    pairs = [("raw", "clipped"), ("clipped", "clip_ri"), ("raw", "clip_ri"),
             ("exact", "raw"), ("exact", "clipped"), ("exact", "clip_ri")]
    for a_c, b_c in pairs:
        n_diff, est_reldiffs = [], []
        for j in range(draws):
            ca = exact if a_c == "exact" else load_capture(sf, f"{a_c}_d{j:02d}")
            cb = load_capture(sf, f"{b_c}_d{j:02d}")
            d = 0
            for q in qnames:
                if ca["queries"][q]["canonical"] != cb["queries"][q]["canonical"]:
                    d += 1
                else:
                    ea = est_rows(ca["queries"][q]["plan"], [])
                    eb = est_rows(cb["queries"][q]["plan"], [])
                    est_reldiffs += [abs(x - y) / max(x, y, 1.0) for x, y in zip(ea, eb)]
            n_diff.append(d)
        pair_stats[f"{a_c}_vs_{b_c}"] = {
            "plan_diffs_per_draw_mean": sum(n_diff) / len(n_diff),
            "plan_diffs_per_draw_max": max(n_diff),
            "same_shape_est_rows_max_reldiff": max(est_reldiffs) if est_reldiffs else 0.0,
            "same_shape_est_rows_frac_nodes_differing": (
                sum(1 for v in est_reldiffs if v > 0) / len(est_reldiffs) if est_reldiffs else 0.0),
        }
    results["pairwise"] = pair_stats
    results["plan_changes_vs_exact"] = plan_changes

    # context / positive control: pg_default vs exact differ ONLY in stadistinct, i.e.
    # in PostgreSQL's own sampled NDV versus truth -- large errors, unlike the sketch's
    if pg_default:
        pd_full = [q for q in qnames
                   if pg_default["queries"][q]["canonical"] != exact["queries"][q]["canonical"]]
        pd_js = [q for q in qnames
                 if join_structure(pg_default["queries"][q]["plan"]) != join_structure(exact["queries"][q]["plan"])]
        errs = {q: e["analyze_error"] for q, e in pg_default["queries"].items() if "analyze_error" in e}
        ratios = []
        for cell in ndv["columns"]:
            key = f'{cell["table"]}.{cell["column"]}'
            s = pg_default["stored_n_distinct"].get(key)
            if s is None:
                continue
            eff = -s * ndv["table_rows"][cell["table"]] if s < 0 else s
            r = max(eff / cell["exact"], cell["exact"] / max(eff, 1e-9))
            ratios.append((r, key, eff, cell["exact"]))
        ratios.sort(reverse=True)
        results["pg_default_vs_exact"] = {
            "note": "identical MCVs/histograms; only stadistinct differs (sampler vs truth)",
            "full_shape_diffs": pd_full, "join_structure_diffs": pd_js,
            "analyze_errors": errs,
            "sampler_ndv_error_ratio_median": pct([r for r, *_ in ratios], 0.5),
            "sampler_ndv_error_ratio_max": ratios[0][0] if ratios else None,
            "sampler_worst5": [{"column": k, "pg_effective": e, "exact": x, "ratio": r}
                               for r, k, e, x in ratios[:5]],
        }
        conf_path = os.path.join(OUT, "q20_confirmatory.json")
        if sf == 10 and os.path.exists(conf_path):
            results["pg_default_vs_exact"]["q20_confirmatory"] = json.load(open(conf_path))

    # Q-error per condition, pooled over nodes of all 22 queries and the analyzed draws
    def suite_qerrors(cap):
        errs, missing = [], 0
        for q in qnames:
            e = cap["queries"][q]
            if "analyze_plan" in e:
                qerror(e["analyze_plan"], errs)
            else:
                missing += 1
        return errs, missing

    qerr = {}
    for cond in ("pg_default", "exact"):
        cap = pg_default if cond == "pg_default" else exact
        if cap:
            errs, missing = suite_qerrors(cap)
            qerr[cond] = {"n_nodes": len(errs), "median": pct(errs, 0.5), "p90": pct(errs, 0.9),
                          "p99": pct(errs, 0.99), "max": max(errs), "queries_missing": missing,
                          "draws_analyzed": 1}
    for cond in ("raw", "clipped", "clip_ri"):
        pooled, med_per_draw, missing_total, nd = [], [], 0, 0
        for j in range(draws):
            cap = load_capture(sf, f"{cond}_d{j:02d}")
            if not any("analyze_plan" in e for e in cap["queries"].values()):
                continue
            errs, missing = suite_qerrors(cap)
            pooled += errs
            missing_total += missing
            med_per_draw.append(pct(errs, 0.5))
            nd += 1
        qerr[cond] = {"n_nodes": len(pooled),
                      "median": pct(pooled, 0.5) if pooled else None,
                      "p90": pct(pooled, 0.9) if pooled else None,
                      "p99": pct(pooled, 0.99) if pooled else None,
                      "max": max(pooled) if pooled else None,
                      "median_per_draw_min": min(med_per_draw) if med_per_draw else None,
                      "median_per_draw_max": max(med_per_draw) if med_per_draw else None,
                      "queries_missing": missing_total, "draws_analyzed": nd}
    results["qerror"] = qerr

    path = os.path.join(OUT, "exp16_results.json")
    doc = {}
    if os.path.exists(path):
        doc = json.load(open(path))
    # a re-run of --analyze rebuilds the block; carry the --runtime measurements forward
    prev = doc.get(f"sf{sf:g}", {})
    if "runtime_changed_plans" in prev and "runtime_changed_plans" not in results:
        results["runtime_changed_plans"] = prev["runtime_changed_plans"]
    doc[f"sf{sf:g}"] = results
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)

    print(json.dumps({"gate": results["gate"]["exact_vs_exact_plan_diffs"],
                      "plan_changes": {c: plan_changes[c]["n_queries_changed_any_draw"] for c in plan_changes},
                      "pairwise_plan_diffs": {k: v["plan_diffs_per_draw_max"] for k, v in pair_stats.items()},
                      "pg_default_vs_exact_join_structure_diffs":
                          results.get("pg_default_vs_exact", {}).get("join_structure_diffs"),
                      "qerror_median": {c: (round(v["median"], 4) if v["median"] is not None else None)
                                        for c, v in qerr.items()}}, indent=1))
    print(f"wrote {path}")


# ---------------------------------------------------------------------------- runtime

def runtime(dsn, sf, reps=5):
    """5-rep timings for queries whose canonical plan changed vs exact, else nothing."""
    ndv = json.load(open(os.path.join(OUT, f"exp16_ndv_sf{sf:g}.json")))
    res_path = os.path.join(OUT, "exp16_results.json")
    results = json.load(open(res_path))[f"sf{sf:g}"]
    queries = {f"q{q['nr']:02d}": q for q in load_queries()}
    targets = []                          # (condition, draw, query)
    for cond, info in results["plan_changes_vs_exact"].items():
        for q in info["queries_changed_any_draw"]:
            for j in range(results["params"]["draws"]):
                cap = load_capture(sf, f"{cond}_d{j:02d}")
                exact = load_capture(sf, "exact")
                if cap["queries"][q]["canonical"] != exact["queries"][q]["canonical"]:
                    targets.append((cond, j, q))
                    break                 # first draw whose plan differs
    if not targets:
        print("no plan changes -> no runtime measurements (per protocol)")
        return

    con = connect(dsn)
    cur = con.cursor()
    cur.execute("SET statement_timeout = 900000")
    attnums = attnum_map(cur)
    out = []

    def timed_suite(q):
        times = []
        for i in range(reps):
            plan = explain(cur, queries[q], analyze=True)
            times.append(plan["Execution Time"])
        return times[1:]                  # discard the first (warm-up), per protocol

    for cond, j, q in targets:
        inject(cur, condition_values(ndv, "exact", 0), attnums)
        t_exact = timed_suite(q)
        inject(cur, condition_values(ndv, cond, j), attnums)
        t_cond = timed_suite(q)
        rec = {"query": q, "condition": cond, "draw": j,
               "exact_ms_median": statistics.median(t_exact), "exact_ms_all": t_exact,
               f"{cond}_ms_median": statistics.median(t_cond), f"{cond}_ms_all": t_cond}
        out.append(rec)
        print(json.dumps(rec))

    doc = json.load(open(res_path))
    doc[f"sf{sf:g}"]["runtime_changed_plans"] = out
    with open(res_path, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"updated {res_path}")
    con.close()


# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("HLL_PG_DSN", "host=localhost port=5433 dbname=tpch user=postgres"))
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--ndv", action="store_true")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--runtime", action="store_true")
    ap.add_argument("--sf", type=float, default=1.0)
    ap.add_argument("--draws", type=int, default=None,
                    help=f"hash draws per column (default {DRAWS} for --ndv/--capture; for --analyze, "
                         "the number of raw_d*.json.gz captures present for that scale factor)")
    ap.add_argument("--analyze-draws", type=int, default=DRAWS,
                    help="run EXPLAIN ANALYZE only for the first N draws (plans captured for all)")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()

    draws = a.draws
    if draws is None:
        if a.analyze:
            cap_dir = os.path.join(OUT, f"capture_sf{a.sf:g}")
            draws = len([f for f in os.listdir(cap_dir)]) if os.path.isdir(cap_dir) else 0
            draws = len([f for f in os.listdir(cap_dir) if f.startswith("raw_d") and f.endswith(".json.gz")]) if draws else 0
            if draws == 0:
                sys.exit(f"no raw_d*.json.gz captures under {cap_dir} -- run --capture first (or check --sf)")
        else:
            draws = DRAWS

    if a.check_only:
        sys.exit(check_only(a.dsn))
    if a.ndv:
        return ndv_compute(a.dsn, a.sf, draws, a.jobs)
    if a.capture:
        return capture(a.dsn, a.sf, draws, a.analyze_draws)
    if a.analyze:
        return analyze(a.sf, draws)
    if a.runtime:
        return runtime(a.dsn, a.sf)

    print("Nothing to do. Use --check-only / --ndv / --capture / --analyze / --runtime.")
    sys.exit(2)


if __name__ == "__main__":
    main()
