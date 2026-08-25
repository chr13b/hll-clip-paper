#!/usr/bin/env python3
"""HLL++ baseline (bias correction + LC thresholds per Heule et al.
2013) and the (n, m) value-window map for the clip.

Grid: p in {10, 12, 14} (m = 1024/4096/16384), n/m ratio in {0.05 ... 32},
d in {1.0, 1.02, 1.1}. Two arms: xxh64 over binary integer keys (real hash) and
ideal-PRNG control. Records which estimator branch fired per trial (lc / bias / raw)
and flags the zone real implementations cover with near-exact sparse mode
(switch band n in [m/4, 3m/4]; sparse LC at p'=25 has ~0.01% rel. error, so the
clip is irrelevant there by construction -- we mark, not simulate, that zone).
"""

import argparse
import json
import os
import time

import numpy as np

from hll_common import (ChunkSampler, build_registers, cell_metrics,
                        hllpp_estimate, ideal_registers, precompute_idx_rank,
                        truncated_posterior_mean)
import struct
import xxhash

PS = [10, 12, 14]
RATIOS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
DS = [1.0, 1.02, 1.1]
POOL_BITS = 22  # 4.2M integer keys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE  # smoke runs redirect outputs


def build_int_pool_hashes():
    n = 1 << POOL_BITS
    out = np.empty(n, dtype=np.uint64)
    f = xxhash.xxh64_intdigest
    pk = struct.pack
    for i in range(n):
        out[i] = f(pk("<q", i))
    return out


def run_cell(arm, pools, sampler, rng, p, n, d, T):
    m = 1 << p
    L = int(round(n * d))
    raw = np.empty(T)
    branches = {"lc": 0, "bias": 0, "raw": 0}
    for t in range(T):
        if arm == "ideal":
            regs = ideal_registers(rng, n, m, 64 - p + 1)
        else:
            sel = sampler.take(n)
            idx_pool, rank_pool = pools[p]
            regs = build_registers(idx_pool, rank_pool, sel, m)
        raw[t], br = hllpp_estimate(regs, p)
        branches[br] += 1
    clip = np.minimum(raw, L)
    tpm = truncated_posterior_mean(raw, L, 1.04 / np.sqrt(m))
    return L, raw, clip, tpm, branches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--T", type=int, default=500)
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    print("hashing 4.2M integer keys with xxh64...", flush=True)
    h = build_int_pool_hashes()
    pools = {p: precompute_idx_rank(h, p) for p in PS}
    del h
    print(f"pools ready in {time.time()-t0:.0f}s", flush=True)

    cells = []
    arrays = {}
    for arm in ["xxh64-ints", "ideal"]:
        for p in PS:
            m = 1 << p
            sampler = ChunkSampler(1 << POOL_BITS, rng)
            for r in RATIOS:
                n = int(round(m * r))
                for d in DS:
                    L, raw, clip, tpm, br = run_cell(arm, pools, sampler, rng,
                                                     p, n, d, args.T)
                    met = cell_metrics(n, L, raw, clip, tpm)
                    cell = {"arm": arm, "p": p, "m": m, "ratio": r, "n": n,
                            "d": d, "L": L, "T": args.T,
                            "branch_lc": br["lc"] / args.T,
                            "branch_bias": br["bias"] / args.T,
                            "branch_raw": br["raw"] / args.T,
                            "sparse_zone": n <= 0.75 * m}
                    cell.update(met)
                    cells.append(cell)
                    ck = f"{arm}|p{p}|r{r}|d{d}"
                    arrays[f"{ck}|raw"] = raw
                    arrays[f"{ck}|clip"] = clip
                    arrays[f"{ck}|tpm"] = tpm
                print(f"{arm} p={p} ratio={r:5.2f} done ({time.time()-t0:.0f}s)",
                      flush=True)

    out = os.path.join(OUT, f"exp2_seed{args.seed}")
    np.savez_compressed(out + ".npz", **arrays)

    lines = [f"exp2 HLL++ window map  (T={args.T}, seed={args.seed}, hash=xxh64/ints + ideal control)",
             "branch = fraction of trials resolved by linear counting / bias-corrected / raw estimator",
             "sparse* = zone a real HLL++ impl covers with near-exact sparse mode (switch band ~[m/4, 3m/4])", ""]
    hdr = (f"{'arm':>10} {'m':>6} {'n/m':>6} {'n':>7} {'d':>5} | {'rmse_raw':>9} "
           f"{'rmse_clip':>9} {'rmse_tpm':>9} | {'dCLIP%':>7} {'dTPM%':>7} | "
           f"{'z_clip':>7} | {'bind%':>6} | {'lc/bias/raw':>14} | sp")
    lines += [hdr, "-" * len(hdr)]
    for c in cells:
        ic = 100 * (1 - c["rmse_clip"] / c["rmse_raw"])
        it = 100 * (1 - c["rmse_tpm"] / c["rmse_raw"])
        lines.append(
            f"{c['arm']:>10} {c['m']:>6} {c['ratio']:>6.2f} {c['n']:>7} {c['d']:>5.2f} | "
            f"{c['rmse_raw']:>9.5f} {c['rmse_clip']:>9.5f} {c['rmse_tpm']:>9.5f} | "
            f"{ic:>7.2f} {it:>7.2f} | {c['z_clip']:>7.1f} | {100*c['bind_rate']:>6.2f} | "
            f"{c['branch_lc']:>4.2f}/{c['branch_bias']:>4.2f}/{c['branch_raw']:>4.2f} | "
            f"{'*' if c['sparse_zone'] else ' '}")

    worse = [(c["arm"], c["m"], c["ratio"], c["d"])
             for c in cells if c["rmse_clip"] / c["rmse_raw"] > 1.01]
    lines.append("")
    lines.append(f"cells where clip >1% worse than raw: {worse if worse else 'NONE'}")
    lines.append(f"runtime {time.time()-t0:.0f}s")

    txt = "\n".join(lines)
    print(txt)
    with open(out + ".txt", "w") as f:
        f.write(txt + "\n")
    with open(out + ".json", "w") as f:
        json.dump({"params": {"ps": PS, "ratios": RATIOS, "ds": DS, "T": args.T,
                              "seed": args.seed, "pool_bits": POOL_BITS},
                   "cells": cells}, f, indent=2)


if __name__ == "__main__":
    main()
