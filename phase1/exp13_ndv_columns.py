#!/usr/bin/env python3
"""NDV statistics for query optimization: the clip on real table columns.

Motivation. Our source audit found the clip living in exactly one place in a real
system: DuckDB's table-statistics path, where it clamps the HLL estimate at the
insertion count. That is not a coincidence. When a database computes per-column NDV
statistics (ANALYZE), the row count R of the table or partition is known exactly and
for free, so L = R requires no extra counter at all, and key or near-key columns are
precisely the d ~ 1 regime where the value-window law says the clip pays in full.

This experiment measures that. For each column of a real table we emulate ANALYZE on
successive row blocks: the exact NDV of the block is the ground truth, the block's row
count is L, and a production HyperLogLog (DataSketches, lg_k = 14) supplies the
estimate. We then compare the raw estimate against min(estimate, L).

Tables:
  * NASA-HTTP request log (1.89M rows) parsed into five genuine columns spanning the
    whole duplication range -- from near-key (timestamp) to categorical (status).
  * The PyPI package namespace (827,798 rows) as a single-column primary-key table.
  * TPC-H columns via DuckDB, when the optional duckdb package is available.

The prediction: gain tracks the per-column duplication factor d = R/NDV through the
value-window law, so key-like columns get the full benefit and categorical columns get
nothing -- and the optimizer can tell which is which for free, since it already knows R.

Writes exp13_ndv_columns.json.
"""
import json
import os
import re

import numpy as np
import datasketches as ds
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE
NASA = os.path.join(HERE, "realworld_data", "nasa_jul95")
PYPI = os.path.join(HERE, "realworld_data", "pypi_names.txt")

LG_K = 14                 # m = 16384, deployment precision
BLOCK = 50_000            # rows per ANALYZE block
MAX_BLOCKS = 30
TARGET_TRIALS = 400       # per column; salted draws scale up for small tables so that
                          # every column is estimated from a comparable number of trials

phi, Phi, Sf = norm.pdf, norm.cdf, norm.sf
LINE_RE = re.compile(r'^(\S+) \S+ \S+ \[([^\]]+)\] "(?:GET|POST|HEAD) (\S+)[^"]*" (\d+) (\S+)')


def g(a):
    return Phi(a) - a * phi(a) + a * a * Sf(a)


def rel_rmse(est, truth):
    e = np.asarray(est, float)
    t = np.asarray(truth, float)
    return float(np.sqrt(np.mean(((e - t) / t) ** 2)))


def analyze_column(values, label, table):
    """Emulate ANALYZE over successive row blocks: exact NDV vs HLL, with L = row count."""
    raws, clips, truths, ds_ = [], [], [], []
    nblocks = min(MAX_BLOCKS, len(values) // BLOCK)
    if nblocks == 0:
        return None
    # small tables yield few blocks; add hash draws so every column gets ~TARGET_TRIALS
    salts = max(1, -(-TARGET_TRIALS // nblocks))
    for b in range(nblocks):
        blk = values[b * BLOCK:(b + 1) * BLOCK]
        n = len(set(blk))                 # ground-truth NDV of the block
        L = len(blk)                      # row count: free and exact in a DBMS
        for s_i in range(salts):
            salt = f"|{s_i}"              # independent hash draw of the same column
            sk = ds.hll_sketch(LG_K)
            for v in blk:
                sk.update(v + salt)
            est = sk.get_estimate()
            raws.append(est)
            clips.append(min(est, L))
            truths.append(n)
            ds_.append(L / n)
    r_raw, r_clip = rel_rmse(raws, truths), rel_rmse(clips, truths)
    gain = 100 * (1 - r_clip / r_raw) if r_raw > 0 else 0.0
    d_mean = float(np.mean(ds_))
    cell = {"table": table, "column": label, "blocks": nblocks, "block_rows": BLOCK, "salts": salts,
            "trials": nblocks * salts,
            "ndv_mean": float(np.mean(truths)), "d_mean": d_mean,
            "rmse_raw": r_raw, "rmse_clip": r_clip, "gain_pct": gain,
            "clip_worse_than_raw": bool(r_clip > r_raw * 1.0001),
            "clip_binds_frac": float(np.mean([c < r - 1e-9 for c, r in zip(clips, raws)]))}
    print(f"  {table:9s} {label:12s} NDV~{cell['ndv_mean']:9.0f}  d={d_mean:8.2f} | "
          f"raw={r_raw:.5f} clip={r_clip:.5f}  gain={gain:6.2f}%  "
          f"impossible={cell['clip_binds_frac']*100:5.1f}%  (T={nblocks*salts})", flush=True)
    return cell


def nasa_columns():
    hosts, stamps, urls, status, nbytes = [], [], [], [], []
    with open(NASA, encoding="latin-1") as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue
            hosts.append(m.group(1))
            stamps.append(m.group(2))
            urls.append(m.group(3))
            status.append(m.group(4))
            nbytes.append(m.group(5))
    return {"timestamp": stamps, "url": urls, "host": hosts,
            "bytes": nbytes, "status": status}


def tpch_columns():
    """Optional: real TPC-H columns via DuckDB (the canonical optimizer benchmark).

    Yields one column at a time, in a fixed order, so that only one column's values
    are in memory beside the (memory-bounded) DuckDB instance. The results do not
    depend on any of this: dbgen is deterministic, DuckDB preserves insertion order,
    and analyze_column() reads only the first MAX_BLOCKS*BLOCK rows of a column. An
    earlier version held all nine extracted columns next to an unbounded in-memory
    sf=0.5 database and peaked near 2 GB RSS, which small hosts OOM-kill."""
    try:
        import duckdb
    except ImportError:
        print("  (duckdb not installed -- skipping TPC-H)", flush=True)
        return
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="hll_exp13_duck_")
    con = None
    try:
        con = duckdb.connect(os.path.join(tmp, "tpch.duckdb"))   # file-backed: spills
        con.execute("PRAGMA memory_limit='768MB'")
        con.execute(f"PRAGMA temp_directory='{tmp}'")
        con.execute("INSTALL tpch; LOAD tpch;")
        con.execute("CALL dbgen(sf=0.5)")
        for tbl, col in [("orders", "o_orderkey"), ("customer", "c_custkey"),
                         ("part", "p_partkey"), ("lineitem", "l_orderkey"),
                         ("lineitem", "l_partkey"), ("lineitem", "l_shipdate"),
                         ("lineitem", "l_quantity"), ("lineitem", "l_returnflag"),
                         ("orders", "o_custkey")]:
            vals = [str(r[0]) for r in
                    con.execute(f"SELECT {col} FROM {tbl} LIMIT {MAX_BLOCKS * BLOCK}").fetchall()]
            yield f"{tbl}.{col}", vals
            del vals
    except Exception as e:
        print(f"  (TPC-H generation failed: {type(e).__name__} -- skipping)", flush=True)
    finally:
        if con is not None:
            con.close()
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    cells = []

    print("NASA-HTTP table (real request log, five columns):", flush=True)
    if os.path.exists(NASA):
        cols = nasa_columns()
        print(f"  parsed {len(cols['url'])} rows", flush=True)
        for name in ("timestamp", "url", "host", "bytes", "status"):
            c = analyze_column(cols[name], name, "nasa")
            if c:
                cells.append(c)
        del cols          # free the five parsed columns before the next table
    else:
        print("  NASA trace absent -- skipping", flush=True)

    print("PyPI namespace as a primary-key column:", flush=True)
    if os.path.exists(PYPI):
        names = [l.strip() for l in open(PYPI, encoding="utf-8", errors="replace") if l.strip()]
        c = analyze_column(names, "package_name", "pypi")
        if c:
            cells.append(c)
        del names
    else:
        print("  PyPI snapshot absent -- skipping", flush=True)

    print("TPC-H columns (optional):", flush=True)
    for label, vals in tpch_columns():
        c = analyze_column(vals, label.split(".")[1], "tpch")
        if c:
            cells.append(c)
        del vals

    if cells:
        keyish = [c for c in cells if c["d_mean"] < 1.05]
        catish = [c for c in cells if c["d_mean"] > 10]
        print()
        print(f"key-like columns (d<1.05): n={len(keyish)}, "
              f"gain {min((c['gain_pct'] for c in keyish), default=0):.1f}"
              f"--{max((c['gain_pct'] for c in keyish), default=0):.1f}%")
        print(f"categorical columns (d>10): n={len(catish)}, "
              f"gain {min((c['gain_pct'] for c in catish), default=0):.1f}"
              f"--{max((c['gain_pct'] for c in catish), default=0):.1f}%")
        print(f"any column where the clip is worse than raw? "
              f"{any(c['clip_worse_than_raw'] for c in cells)}")

    # Do NOT overwrite the shipped full result with a reduced one. If any source was
    # unavailable (no duckdb, no NASA trace, no PyPI snapshot) this run covers fewer
    # columns than the paper reports, and silently replacing the released JSON would
    # make the artifact degrade itself -- the figures would then render from the
    # reduced file and disagree with the paper.
    EXPECTED = 15
    name = "exp13_ndv_columns.json" if len(cells) >= EXPECTED else "exp13_ndv_columns_partial.json"
    if name.endswith("_partial.json"):
        print(f"\n  WARNING: only {len(cells)} of {EXPECTED} columns were produced "
              f"(a data source was unavailable -- see the skip messages above).")
        print(f"  Writing {name} and LEAVING the released exp13_ndv_columns.json intact.")
    with open(os.path.join(OUT, name), "w") as f:
        json.dump({"params": {"lg_k": LG_K, "block_rows": BLOCK,
                              "max_blocks": MAX_BLOCKS,
                              "columns_expected": EXPECTED, "columns_produced": len(cells)},
                   "cells": cells,
                   "note": "ANALYZE emulation: L = block row count (free in a DBMS)"},
                  f, indent=2)
    print(f"wrote {name}")


if __name__ == "__main__":
    main()
