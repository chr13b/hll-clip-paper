#!/usr/bin/env python3
"""Real-world evaluation: the clip on a genuine production workload.

Dataset: the full PyPI package namespace (827,798 distinct package names,
downloaded from https://pypi.org/simple/ on 2026-06-13) -- a canonical
"count distinct identifiers" stream where keys are naturally near-unique.
Sketch: Apache DataSketches production HLL (lg_k=14 -> m=16384, realistic
deployment precision), fed the actual name STRINGS, hashed internally by the
library (real production hashing -- no synthetic PRNG). Nothing in the path is
simulated: real keys, real hashing, real library, end to end.

Protocol: for each n we draw T random n-subsets of the real namespace (disjoint
chunks of a reshuffled index, reshuffling when exhausted) and record the raw HLL
estimate per subset. Duplicates do not change the sketch, so raw estimates at a
given n are reused across the duplication grid d (only L=round(n*d) changes) --
the same exactness argument as the synthetic experiments. The d>1 cells model a
real near-duplicate-free stream that contains a small fraction of repeats.
TPM sigma is calibrated per n on a disjoint held-out shuffle (seed 555).

Also reports the single full-corpus operating point (n=827,798, d=1.0).
"""

import argparse
import json
import os

import numpy as np
import datasketches as ds

from hll_common import cell_metrics, truncated_posterior_mean

LG_K = 14
M = 1 << LG_K
NS = [16384, 65536, 262144]            # n/m = 1, 4, 16 (dense regime)
DS_GRID = [1.0, 1.01, 1.02, 1.03, 1.05, 1.10, 1.20]
ARM = "pypi_hll"
CAL_T = 200
CAL_SEED = 555

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE
NAMES_FILE = os.path.join(HERE, "realworld_data", "pypi_names.txt")
CAL_FILE = os.path.join(OUT, "exp5_calibration.json")


def load_names():
    with open(NAMES_FILE, encoding="utf-8", errors="replace") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


class IndexSampler:
    def __init__(self, size, rng):
        self.size, self.rng, self.pos, self.perm = size, rng, size, None

    def take(self, k):
        if self.pos + k > self.size:
            self.perm = self.rng.permutation(self.size)
            self.pos = 0
        s = self.perm[self.pos:self.pos + k]
        self.pos += k
        return s


def raw_estimates(names, sampler, n, T):
    raw = np.empty(T)
    for t in range(T):
        sk = ds.hll_sketch(LG_K)
        for idx in sampler.take(n):
            sk.update(names[idx])
        raw[t] = sk.get_estimate()
    return raw


def calibrate(names):
    rng = np.random.default_rng(CAL_SEED)
    sampler = IndexSampler(len(names), rng)
    cal = {}
    for n in NS:
        raw = raw_estimates(names, sampler, n, CAL_T)
        cal[str(n)] = float(np.std(raw / n, ddof=1))
        print(f"calibrated pypi n={n}: c_sigma={cal[str(n)]:.5f}", flush=True)
    with open(CAL_FILE, "w") as f:
        json.dump({"T": CAL_T, "lg_k": LG_K, "c_sigma": cal}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    names = load_names()
    print(f"loaded {len(names)} PyPI package names", flush=True)

    if args.calibrate:
        calibrate(names)
        return

    with open(CAL_FILE) as f:
        cal = json.load(f)["c_sigma"]

    rng = np.random.default_rng(args.seed)
    sampler = IndexSampler(len(names), rng)
    cells, arrays = [], {}
    for n in NS:
        raw = raw_estimates(names, sampler, n, args.T)
        for d in DS_GRID:
            L = int(round(n * d))
            clip = np.minimum(raw, L)
            tpm = truncated_posterior_mean(raw, L, cal[str(n)])
            met = cell_metrics(n, L, raw, clip, tpm)
            cell = {"arm": ARM, "n": n, "d": d, "L": L, "T": args.T,
                    "c_sigma": cal[str(n)], "lg_k": LG_K}
            cell.update(met)
            cells.append(cell)
            ck = f"{ARM}|n{n}|d{d}"
            arrays[f"{ck}|raw"], arrays[f"{ck}|clip"], arrays[f"{ck}|tpm"] = raw, clip, tpm
        g = 100 * (1 - cells[-len(DS_GRID)]["rmse_clip"] / cells[-len(DS_GRID)]["rmse_raw"])
        print(f"pypi n={n}: rmse_raw={cells[-len(DS_GRID)]['rmse_raw']:.5f} "
              f"clip dRMSE@d=1.0={g:.2f}% bind={cells[-len(DS_GRID)]['bind_rate']:.2f}",
              flush=True)

    # single full-corpus operating point (deterministic given the corpus)
    sk = ds.hll_sketch(LG_K)
    for nm in names:
        sk.update(nm)
    full_raw = sk.get_estimate()
    full = {"n": len(names), "raw": float(full_raw),
            "clip": float(min(full_raw, len(names))),
            "rel_err_raw": float((full_raw - len(names)) / len(names)),
            "rel_err_clip": float((min(full_raw, len(names)) - len(names)) / len(names))}
    print(f"FULL corpus n={full['n']}: raw={full_raw:.0f} "
          f"(rel_err {100*full['rel_err_raw']:+.2f}%) clip rel_err "
          f"{100*full['rel_err_clip']:+.2f}%", flush=True)

    out = os.path.join(OUT, f"exp5_seed{args.seed}")
    np.savez_compressed(out + ".npz", **arrays)
    with open(out + ".json", "w") as f:
        json.dump({"params": {"lg_k": LG_K, "m": M, "T": args.T, "seed": args.seed,
                              "ns": NS, "ds": DS_GRID, "dataset": "PyPI simple index",
                              "n_corpus": len(names)},
                   "cells": cells, "full_corpus": full}, f, indent=2)


if __name__ == "__main__":
    main()
