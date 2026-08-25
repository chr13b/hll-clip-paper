#!/usr/bin/env python3
"""Does the clip gain persist on modern production sketches with
lower base error?

Arms (Apache DataSketches python bindings, v4.1.0):
- hll_hip: hll_sketch(lg_k=10) -- get_estimate() on a stream-built sketch is the
  HIP (historic inverse probability / martingale-class) estimator. Lower error than
  the classic FFGM estimator (~0.84/sqrt(m) vs 1.04/sqrt(m)).
- cpc: cpc_sketch(lg_k=10) -- Lang's Compressed Probabilistic Counting; the best
  accuracy-per-byte production sketch in the library.

Scope notes: the bindings expose no composite estimator
(HLL++-style baselines are covered by exp2); UltraLogLog has no python bindings
(hash4j is Java) -- not run, gap documented.

Design: each trial feeds n DISTINCT integer keys (sequential ints from a per-run
disjoint range; the library hashes internally, so trials are iid). Duplicates never
change either sketch (both are idempotent; HIP updates only on state change --
verified in the source audit), so the length-L stream's sketch equals the
n-distinct-key sketch and L enters only the estimators: same exactness argument as
the original prototype.

TPM noise model: c_sigma per (arm, n) calibrated on a SEPARATE seed (--calibrate,
seed 555, disjoint key range) as std(raw/n) over 300 trials -- evaluation seeds
never see calibration data. This is the 'calibrated-sigma' TPM that exp2 showed is
required (nominal HLL sigma is wrong for HIP/CPC).

Pass criterion (mirrors exp1): at d=1.0 clip dRMSE meaningfully positive on every
arm and never worse anywhere; the quantitative question is whether the ~30% gain
persists when base error is already low.
"""

import argparse
import json
import os
import time

import numpy as np
import datasketches as ds

from hll_common import cell_metrics, truncated_posterior_mean

LG_K = 10
NS = [2000, 8000, 32000]
DS_GRID = [1.0, 1.02, 1.1, 2.0]
ARMS = ["hll_hip", "cpc"]
CAL_SEED = 555
CAL_T = 300

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE  # smoke runs redirect outputs
CAL_FILE = os.path.join(OUT, "exp3_calibration.json")


def make_sketch(arm):
    return ds.hll_sketch(LG_K) if arm == "hll_hip" else ds.cpc_sketch(LG_K)


def raw_estimates(arm, n, T, key_base):
    """T trials; each feeds n distinct ints from a disjoint range; returns raw
    estimates and the advanced key_base."""
    raw = np.empty(T)
    for t in range(T):
        sk = make_sketch(arm)
        for i in range(key_base, key_base + n):
            sk.update(i)
        key_base += n
        raw[t] = sk.get_estimate()
    return raw, key_base


def calibrate():
    key_base = CAL_SEED * 10 ** 9
    cal = {}
    for arm in ARMS:
        for n in NS:
            raw, key_base = raw_estimates(arm, n, CAL_T, key_base)
            cal[f"{arm}|{n}"] = float(np.std(raw / n, ddof=1))
            print(f"calibrated {arm} n={n}: c_sigma={cal[f'{arm}|{n}']:.5f}", flush=True)
    with open(CAL_FILE, "w") as f:
        json.dump({"seed": CAL_SEED, "T": CAL_T, "c_sigma": cal}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    if args.calibrate:
        calibrate()
        return

    with open(CAL_FILE) as f:
        cal = json.load(f)["c_sigma"]

    t0 = time.time()
    key_base = args.seed * 10 ** 9
    cells = []
    arrays = {}
    for arm in ARMS:
        for n in NS:
            for d in DS_GRID:
                L = int(round(n * d))
                raw, key_base = raw_estimates(arm, n, args.T, key_base)
                clip = np.minimum(raw, L)
                tpm = truncated_posterior_mean(raw, L, cal[f"{arm}|{n}"])
                met = cell_metrics(n, L, raw, clip, tpm)
                cell = {"arm": arm, "n": n, "d": d, "L": L, "T": args.T,
                        "c_sigma": cal[f"{arm}|{n}"]}
                cell.update(met)
                cells.append(cell)
                ck = f"{arm}|n{n}|d{d}"
                arrays[f"{ck}|raw"] = raw
                arrays[f"{ck}|clip"] = clip
                arrays[f"{ck}|tpm"] = tpm
            print(f"{arm} n={n} done ({time.time()-t0:.0f}s)", flush=True)

    out = os.path.join(OUT, f"exp3_seed{args.seed}")
    np.savez_compressed(out + ".npz", **arrays)

    lines = [f"exp3 modern sketches  (lg_k={LG_K}, T={args.T}, seed={args.seed}, "
             f"datasketches 4.1.0, tpm sigma calibrated on seed {CAL_SEED})", ""]
    hdr = (f"{'arm':>8} {'n':>6} {'d':>5} | {'rmse_raw':>9} {'rmse_clip':>9} "
           f"{'rmse_tpm':>9} | {'dCLIP%':>7} {'dTPM%':>7} | {'z_clip':>7} {'z_tpm':>7} | {'bind%':>6}")
    lines += [hdr, "-" * len(hdr)]
    for c in cells:
        ic = 100 * (1 - c["rmse_clip"] / c["rmse_raw"])
        it = 100 * (1 - c["rmse_tpm"] / c["rmse_raw"])
        lines.append(f"{c['arm']:>8} {c['n']:>6} {c['d']:>5.2f} | {c['rmse_raw']:>9.5f} "
                     f"{c['rmse_clip']:>9.5f} {c['rmse_tpm']:>9.5f} | {ic:>7.2f} {it:>7.2f} | "
                     f"{c['z_clip']:>7.1f} {c['z_tpm']:>7.1f} | {100*c['bind_rate']:>6.2f}")
    lines.append("")
    for arm in ARMS:
        d1 = [c for c in cells if c["arm"] == arm and c["d"] == 1.0]
        imp = float(np.mean([1 - c["rmse_clip"] / c["rmse_raw"] for c in d1]))
        lines.append(f"  {arm:>8}: d=1.0 avg clip dRMSE = {100*imp:6.2f}%")
    worse = [(c["arm"], c["n"], c["d"]) for c in cells
             if c["rmse_clip"] / c["rmse_raw"] > 1.01]
    lines.append(f"cells where clip >1% worse: {worse if worse else 'NONE'}")
    lines.append(f"runtime {time.time()-t0:.0f}s")

    txt = "\n".join(lines)
    print(txt)
    with open(out + ".txt", "w") as f:
        f.write(txt + "\n")
    with open(out + ".json", "w") as f:
        json.dump({"params": {"lg_k": LG_K, "T": args.T, "seed": args.seed,
                              "ns": NS, "ds": DS_GRID, "cal_seed": CAL_SEED},
                   "cells": cells}, f, indent=2)


if __name__ == "__main__":
    main()
