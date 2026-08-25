#!/usr/bin/env python3
"""Fill the middle of the value window with REAL key values.

The as-is columns of exp13 land at the two ends of the curve: primary keys sit at
a = 0 and everything else is far outside the window at a >> 1. Nothing lies in the
transition region 0 < a < 3, which is precisely where the closed form makes its
strongest claim, so from exp13 alone a reader cannot judge whether the theory tracks
real data through the descent.

This experiment populates that region without leaving real data. We take the same
real key columns and introduce a controlled number of duplicate rows: for a target
duplication factor d we keep n = round(B/d) distinct real values in a block of B rows
and fill the remainder by repeating values already present. The result is exactly a
near-key column -- an identifier column with a handful of repeated entries, which is
what most real "almost unique" columns are -- built from genuine keys rather than
synthetic ones. L is the block's row count, free and exact as before.

Targets are chosen in units of the sketch's own relative error so the points land
evenly across the descent: d - 1 = a * sigma for a in {0.25 ... 3}.

WHY THE INJECTION IS NOT A FREE PARAMETER. A HyperLogLog register holds a maximum over
the hashed items, so the sketch state -- and therefore nhat -- depends only on the SET of
distinct values and the order in which they FIRST arrive, never on how often each occurs
(this is exactly the duplicate insensitivity of Sec. 2; note the DataSketches HIP estimate
IS order-dependent, which is why we fix the arrival order rather than claim invariance to
it -- see the join discussion in the paper). The clip's other input, L, is the row count. So which
rows we duplicate cannot affect any number here: every duplicate pattern with the same
row count yields the identical estimator, whether the repeats are uniform (as below),
Zipfian, or all piled on one value. The injection therefore sets d and nothing else, which
is what makes this a controlled experiment rather than a modelling choice. We keep the
uniform draw only because it is the simplest statement of "add B - n duplicate rows".

Writes exp15_ndv_sweep.json.
"""
import json
import os

import numpy as np
from scipy.stats import norm
import datasketches as ds

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE
PYPI = os.path.join(HERE, "realworld_data", "pypi_names.txt")

LG_K = 14
BLOCK = 50_000
A_TARGETS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
DRAWS = 400
BOOT = 2000


def theory_gain(a):
    g = norm.cdf(a) - a * norm.pdf(a) + a * a * norm.sf(a)
    return 100.0 * (1.0 - np.sqrt(g))


def rel_rmse(est, truth):
    e = np.asarray(est, float)
    return float(np.sqrt(np.mean(((e - truth) / truth) ** 2)))


def boot_ci(raw_err, clip_err, rng, reps=BOOT):
    """Percentile bootstrap CI on the gain, resampling whole draws."""
    raw_err = np.asarray(raw_err, float)
    clip_err = np.asarray(clip_err, float)
    idx = rng.integers(0, raw_err.size, size=(reps, raw_err.size))
    r = np.sqrt(np.mean(raw_err[idx] ** 2, axis=1))
    c = np.sqrt(np.mean(clip_err[idx] ** 2, axis=1))
    gains = 100.0 * (1.0 - c / r)
    return float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))


def sigma_of(values, draws=DRAWS):
    """The sketch's own relative standard error on this column, at this block size."""
    est = []
    for i in range(draws):
        sk = ds.hll_sketch(LG_K)
        for v in values[:BLOCK]:
            sk.update(v + f"|s{i}")
        est.append(sk.get_estimate())
    return float(np.std(np.asarray(est) / BLOCK))


def sweep(values, label, rng):
    base = values[:BLOCK]
    sig = sigma_of(base)
    print(f"  {label}: sigma = {sig:.5f}", flush=True)
    rows = []
    for a in A_TARGETS:
        d = 1.0 + a * sig
        n = int(round(BLOCK / d))                 # distinct values kept
        raw, clip = [], []
        for i in range(DRAWS):
            keep = base[:n]
            extra = [keep[j] for j in rng.integers(0, n, size=BLOCK - n)]
            blk = keep + extra                    # L = BLOCK rows, NDV = n
            sk = ds.hll_sketch(LG_K)
            salt = f"|d{i}"
            for v in blk:
                sk.update(v + salt)
            e = sk.get_estimate()
            raw.append(e)
            clip.append(min(e, BLOCK))            # L = row count, free and exact
        r_raw, r_clip = rel_rmse(raw, n), rel_rmse(clip, n)
        gain = 100 * (1 - r_clip / r_raw)
        a_obs = (BLOCK / n - 1) / sig
        lo, hi = boot_ci((np.asarray(raw) - n) / n, (np.asarray(clip) - n) / n, rng)
        rows.append({"column": label, "a_target": a, "d": BLOCK / n, "ndv": n,
                     "L": BLOCK, "sigma": sig, "a": a_obs,
                     "rmse_raw": r_raw, "rmse_clip": r_clip, "gain_pct": gain,
                     "gain_ci_lo": lo, "gain_ci_hi": hi,
                     "theory_gain_pct": float(theory_gain(a_obs)),
                     "draws": DRAWS,
                     "clip_worse_than_raw": bool(r_clip > r_raw * 1.0001)})
        print(f"    a={a_obs:4.2f} (d={rows[-1]['d']:.5f}, NDV={n}) "
              f"gain={gain:5.1f}% [{lo:5.1f},{hi:5.1f}]  theory={theory_gain(a_obs):5.1f}%",
              flush=True)
    return rows


def pattern_check(values, rng, a=1.0, draws=20):
    """Verify the claim in the docstring: the duplicate PATTERN cannot matter.

    Same row count, same distinct set, three very different duplicate patterns
    (uniform, all repeats on a single value, a Zipf-like skew). If HLL is duplicate
    insensitive the estimates must agree exactly, not approximately.
    """
    base = values[:BLOCK]
    sig = sigma_of(base, draws=draws)
    n = int(round(BLOCK / (1.0 + a * sig)))
    keep, k = base[:n], BLOCK - n
    zipf = [keep[j % max(1, n // 100)] for j in range(k)]          # skewed onto 1% of keys
    patterns = {"uniform": [keep[j] for j in rng.integers(0, n, size=k)],
                "single": [keep[0]] * k,
                "zipf": zipf}
    est = {}
    for name, extra in patterns.items():
        e = []
        for i in range(draws):
            sk = ds.hll_sketch(LG_K)
            for v in keep + extra:
                sk.update(v + f"|p{i}")
            e.append(sk.get_estimate())
        est[name] = e
    ref = est["uniform"]
    identical = all(est[p] == ref for p in patterns)
    print(f"  duplicate-pattern invariance (n={n}, {k} duplicate rows, {draws} draws): "
          f"{'IDENTICAL' if identical else 'DIFFER -- INVESTIGATE'}", flush=True)
    return {"a": a, "ndv": n, "duplicate_rows": k, "draws": draws,
            "patterns": list(patterns), "estimates_identical": bool(identical)}


def main():
    rng = np.random.default_rng(20260810)
    cells = []
    invariance = None

    if os.path.exists(PYPI):
        names = [l.strip() for l in open(PYPI, encoding="utf-8", errors="replace")
                 if l.strip()]
        print("PyPI package_name (real primary-key values):", flush=True)
        invariance = pattern_check(names, rng)
        cells += sweep(names, "pypi.package_name", rng)
    else:
        print("PyPI snapshot absent -- skipping", flush=True)

    try:
        import duckdb
        con = duckdb.connect()
        con.execute("INSTALL tpch; LOAD tpch;")
        con.execute("CALL dbgen(sf=0.3)")
        vals = [str(r[0]) for r in
                con.execute("SELECT o_orderkey FROM orders").fetchall()]
        print("TPC-H orders.o_orderkey (real primary-key values):", flush=True)
        cells += sweep(vals, "tpch.o_orderkey", rng)
    except Exception as e:
        print(f"  (TPC-H unavailable: {type(e).__name__} -- skipping)", flush=True)

    if cells:
        print(f"\nany cell where the clip is worse than raw? "
              f"{any(c['clip_worse_than_raw'] for c in cells)}")

    # Same self-degradation guard as exp13: a run missing PyPI or duckdb covers fewer
    # cells than the paper reports, and must not replace the released file.
    EXPECTED = 12
    name = "exp15_ndv_sweep.json" if len(cells) >= EXPECTED else "exp15_ndv_sweep_partial.json"
    if name.endswith("_partial.json"):
        print(f"\n  WARNING: only {len(cells)} of {EXPECTED} cells produced; writing {name} "
              f"and leaving the released exp15_ndv_sweep.json intact.")
    with open(os.path.join(OUT, name), "w") as f:
        json.dump({"params": {"lg_k": LG_K, "block_rows": BLOCK,
                              "a_targets": A_TARGETS, "draws": DRAWS,
                              "bootstrap_reps": BOOT},
                   "cells": cells,
                   "duplicate_pattern_invariance": invariance,
                   "note": "real key values with controlled duplicate rows; L = block rows; "
                           "the duplicate pattern provably cannot affect nhat (see docstring)"},
                  f, indent=2)
    print(f"wrote {name}")


if __name__ == "__main__":
    main()
