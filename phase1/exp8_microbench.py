#!/usr/bin/env python3
"""Cost microbenchmark. Quantifies the two costs the clip adds.

Build side: maintaining L is one integer increment per insert. We time inserting
N items into a production DataSketches HLL with and without the side counter; the
overhead is below the noise floor of the sketch update (which hashes + branches).

Query side: the hard clip is a single comparison min(n_hat, L); the truncated-
posterior mean (TPM) is a numerical posterior-mean integral. We time both per query.
This is why the hard clip is the always-on default and the TPM is the opt-in variant.
"""

import json
import os
import time

import numpy as np
import datasketches as ds

from hll_common import truncated_posterior_mean

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE
N = 2_000_000
QREPS = 20000


def time_inserts(with_counter):
    sk = ds.hll_sketch(14)
    L = 0
    t0 = time.perf_counter()
    if with_counter:
        for i in range(N):
            sk.update(i)
            L += 1
    else:
        for i in range(N):
            sk.update(i)
    return time.perf_counter() - t0


def main():
    # build side: median of 3 each, interleaved to share cache/JIT warmth
    base, ctr = [], []
    for _ in range(3):
        base.append(time_inserts(False))
        ctr.append(time_inserts(True))
    t_base, t_ctr = float(np.median(base)), float(np.median(ctr))
    rate_base = N / t_base
    overhead = 100 * (t_ctr - t_base) / t_base

    # query side: clip vs TPM
    rng = np.random.default_rng(0)
    n = 10000
    raws = n * (1 + 0.005 * rng.standard_normal(QREPS))
    L = n
    t0 = time.perf_counter()
    for r in raws:
        _ = min(r, L)
    t_clip = (time.perf_counter() - t0) / QREPS

    t0 = time.perf_counter()
    for r in raws:
        _ = truncated_posterior_mean(np.array([r]), L, 0.005)
    t_tpm = (time.perf_counter() - t0) / QREPS

    res = {
        "insert": {"N": N, "items_per_sec_base": rate_base,
                   "t_base_s": t_base, "t_counter_s": t_ctr,
                   "counter_overhead_pct": overhead},
        "query": {"clip_ns": t_clip * 1e9, "tpm_us": t_tpm * 1e6,
                  "tpm_over_clip": t_tpm / t_clip},
    }
    print(f"insert: {rate_base/1e6:.2f} M items/s; "
          f"side-counter overhead = {overhead:+.2f}%")
    print(f"query: clip = {t_clip*1e9:.1f} ns; TPM = {t_tpm*1e6:.1f} us "
          f"({t_tpm/t_clip:.0f}x the clip)")
    with open(os.path.join(OUT, "exp8_microbench.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("wrote exp8_microbench.json")


if __name__ == "__main__":
    main()
