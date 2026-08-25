#!/usr/bin/env python3
"""Does the clip effect survive real hash functions on real keys?

Arms: ideal-hash control (prototype replication) + {words, urls, ints} x
{xxh64, mmh3, sha256}. Same cell grid as the original prototype: n in {2000, 8000, 32000},
d in {1.0, 1.02, 1.1, 2.0}, T trials/cell, classic Flajolet estimator with
linear-counting small-range correction, m=1024 (p=10).

Pass criterion (the effect must survive):
at d=1.0, clip dRMSE >= 18% for EVERY (key type, hash) arm, and no arm/cell
where clip is >1% worse than raw.
"""

import argparse
import json
import os
import time

import numpy as np

from hll_common import (ChunkSampler, build_ints, build_registers, build_urls,
                        cell_metrics, classic_estimate, hash_corpus,
                        ideal_registers, load_words, precompute_idx_rank,
                        truncated_posterior_mean)

P = 10
M = 1 << P
C_SIGMA = 1.04 / np.sqrt(M)
RANK_CAP = 63  # ideal arm only, as in the original prototype
NS = [2000, 8000, 32000]
DS = [1.0, 1.02, 1.1, 2.0]
KEYTYPES = ["words", "urls", "ints"]
HASHES = ["xxh64", "mmh3", "sha256"]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE  # smoke runs redirect outputs; data stays in HERE


def build_pools():
    words = load_words(os.path.join(HERE, "data", "words_alpha.txt"))
    corpora = {
        "words": words,
        "urls": build_urls(words, 300_000),
        "ints": build_ints(400_000),
    }
    pools = {}
    for kt, keys in corpora.items():
        for hn in HASHES:
            h = hash_corpus(keys, hn)
            pools[(kt, hn)] = precompute_idx_rank(h, P)
    return pools


def run_cell_real(pools, sampler, kt, hn, n, d, T):
    idx_pool, rank_pool = pools[(kt, hn)]
    L = int(round(n * d))
    raw = np.empty(T)
    for t in range(T):
        sel = sampler.take(n)
        raw[t] = classic_estimate(build_registers(idx_pool, rank_pool, sel, M), M)
    clip = np.minimum(raw, L)
    tpm = truncated_posterior_mean(raw, L, C_SIGMA)
    return L, raw, clip, tpm


def run_cell_ideal(rng, n, d, T):
    L = int(round(n * d))
    raw = np.empty(T)
    for t in range(T):
        raw[t] = classic_estimate(ideal_registers(rng, n, M, RANK_CAP), M)
    clip = np.minimum(raw, L)
    tpm = truncated_posterior_mean(raw, L, C_SIGMA)
    return L, raw, clip, tpm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--T", type=int, default=1000)
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    print("building corpora + hashing...", flush=True)
    pools = build_pools()
    print(f"pools ready in {time.time()-t0:.1f}s", flush=True)

    arms = [("ideal", "prng")] + [(kt, hn) for kt in KEYTYPES for hn in HASHES]
    cells = []
    arrays = {}
    for kt, hn in arms:
        samplers = {}
        for n in NS:
            for d in DS:
                if kt == "ideal":
                    L, raw, clip, tpm = run_cell_ideal(rng, n, d, args.T)
                else:
                    key = (kt, hn)
                    if key not in samplers:
                        samplers[key] = ChunkSampler(len(pools[key][0]), rng)
                    L, raw, clip, tpm = run_cell_real(pools, samplers[key], kt, hn,
                                                      n, d, args.T)
                met = cell_metrics(n, L, raw, clip, tpm)
                cell = {"arm": f"{kt}-{hn}", "n": n, "d": d, "L": L, "T": args.T}
                cell.update(met)
                cells.append(cell)
                ck = f"{kt}-{hn}|n{n}|d{d}"
                arrays[f"{ck}|raw"] = raw
                arrays[f"{ck}|clip"] = clip
                arrays[f"{ck}|tpm"] = tpm
        d1 = [c for c in cells if c["arm"] == f"{kt}-{hn}" and c["d"] == 1.0]
        imp = np.mean([1 - c["rmse_clip"] / c["rmse_raw"] for c in d1])
        print(f"arm {kt}-{hn:7s} done  d=1.0 clip dRMSE = {100*imp:6.2f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)

    out = os.path.join(OUT, f"exp1_seed{args.seed}")
    np.savez_compressed(out + ".npz", **arrays)

    # per-arm verdicts
    lines = [f"exp1 real-hash  (m={M}, T={args.T}, seed={args.seed})",
             "All RMSE/bias relative to true n.", ""]
    hdr = (f"{'arm':>14} {'n':>6} {'d':>5} | {'rmse_raw':>9} {'rmse_clip':>9} "
           f"{'rmse_tpm':>9} | {'dCLIP%':>7} {'dTPM%':>7} | {'z_clip':>7} {'z_tpm':>7} | {'bind%':>6}")
    lines += [hdr, "-" * len(hdr)]
    for c in cells:
        ic = 100 * (1 - c["rmse_clip"] / c["rmse_raw"])
        it = 100 * (1 - c["rmse_tpm"] / c["rmse_raw"])
        lines.append(f"{c['arm']:>14} {c['n']:>6} {c['d']:>5.2f} | {c['rmse_raw']:>9.5f} "
                     f"{c['rmse_clip']:>9.5f} {c['rmse_tpm']:>9.5f} | {ic:>7.2f} {it:>7.2f} | "
                     f"{c['z_clip']:>7.1f} {c['z_tpm']:>7.1f} | {100*c['bind_rate']:>6.2f}")

    lines.append("")
    fails, worse = [], []
    for kt, hn in arms:
        a = f"{kt}-{hn}"
        d1 = [c for c in cells if c["arm"] == a and c["d"] == 1.0]
        imp = float(np.mean([1 - c["rmse_clip"] / c["rmse_raw"] for c in d1]))
        lines.append(f"  {a:>14}: d=1.0 avg clip dRMSE = {100*imp:6.2f}%")
        if kt != "ideal" and imp < 0.18:
            fails.append((a, imp))
        for c in [c for c in cells if c["arm"] == a]:
            if c["rmse_clip"] / c["rmse_raw"] > 1.01:
                worse.append((a, c["n"], c["d"]))
    lines.append("")
    if fails:
        lines.append(f"ARMS BELOW 18%: {fails}")
    if worse:
        lines.append(f"CELLS WHERE CLIP >1% WORSE: {worse}")
    verdict = "SURVIVES" if not fails and not worse else "FAILS"
    lines.append(f"VERDICT (this seed): {verdict}")
    lines.append(f"runtime {time.time()-t0:.0f}s")

    txt = "\n".join(lines)
    print(txt)
    with open(out + ".txt", "w") as f:
        f.write(txt + "\n")
    with open(out + ".json", "w") as f:
        json.dump({"params": {"m": M, "T": args.T, "seed": args.seed,
                              "ns": NS, "ds": DS, "c_sigma": C_SIGMA},
                   "cells": cells, "verdict": verdict}, f, indent=2)


if __name__ == "__main__":
    main()
