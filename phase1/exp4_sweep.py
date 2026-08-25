#!/usr/bin/env python3
"""m-scaling and fine-grained d sweep with the HLL++ baseline.

Grid: m in {256, 1024, 4096, 16384} (p = 8/10/12/14), n = 8m (clean dense regime),
d in a fine [1.0, 1.2] grid. Arms: xxh64 over binary integer keys + ideal-PRNG
control. T trials/cell per seed; intended for >=3 seeds pooled with bootstrap CIs
(ci.py).

Purpose: decay curves gain(d) per m; test the scaling-collapse prediction that the
gain depends on m and d only through x = (d-1)/sigma(m) (sigma measured empirically
from the d=1.2 cell, where the clip never binds).
"""

import argparse
import json
import os
import time

import numpy as np

from hll_common import (ChunkSampler, build_registers, cell_metrics,
                        hllpp_estimate, ideal_registers, precompute_idx_rank,
                        truncated_posterior_mean)
from exp2_hllpp_window import build_int_pool_hashes, POOL_BITS

PS = [8, 10, 12, 14]
RATIO = 8
D_GRID = [1.0, 1.005, 1.01, 1.02, 1.03, 1.05, 1.07, 1.10, 1.15, 1.20]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE  # smoke runs redirect outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--T", type=int, default=600)
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    print("hashing integer pool...", flush=True)
    h = build_int_pool_hashes()
    pools = {p: precompute_idx_rank(h, p) for p in PS}
    del h
    print(f"pools ready ({time.time()-t0:.0f}s)", flush=True)

    cells = []
    arrays = {}
    for arm in ["xxh64-ints", "ideal"]:
        for p in PS:
            m = 1 << p
            n = m * RATIO
            sampler = ChunkSampler(1 << POOL_BITS, rng)
            for d in D_GRID:
                L = int(round(n * d))
                raw = np.empty(args.T)
                for t in range(args.T):
                    if arm == "ideal":
                        regs = ideal_registers(rng, n, m, 64 - p + 1)
                    else:
                        idx_pool, rank_pool = pools[p]
                        regs = build_registers(idx_pool, rank_pool,
                                               sampler.take(n), m)
                    raw[t], _ = hllpp_estimate(regs, p)
                clip = np.minimum(raw, L)
                tpm = truncated_posterior_mean(raw, L, 1.04 / np.sqrt(m))
                met = cell_metrics(n, L, raw, clip, tpm)
                cell = {"arm": arm, "p": p, "m": m, "n": n, "d": d, "L": L,
                        "T": args.T}
                cell.update(met)
                cells.append(cell)
                ck = f"{arm}|p{p}|d{d}"
                arrays[f"{ck}|raw"] = raw
                arrays[f"{ck}|clip"] = clip
                arrays[f"{ck}|tpm"] = tpm
            print(f"{arm} p={p} done ({time.time()-t0:.0f}s)", flush=True)

    out = os.path.join(OUT, f"exp4_seed{args.seed}")
    np.savez_compressed(out + ".npz", **arrays)
    with open(out + ".json", "w") as f:
        json.dump({"params": {"ps": PS, "ratio": RATIO, "ds": D_GRID,
                              "T": args.T, "seed": args.seed},
                   "cells": cells}, f, indent=2)
    print(f"done, runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
