#!/usr/bin/env python3
"""Is min(n_hat, L) better than the other ways of using L?

The natural competitor that also uses the stream length L is sampling-based
distinct-value estimation (Haas et al. VLDB'95; Charikar et al. PODS'00 GEE),
which keeps a uniform sample of the L-item stream and extrapolates to the
population size L:
    n_GEE = sqrt(L/s) * f1 + sum_{j>=2} f_j,
with f_j = number of values seen exactly j times in a sample of size s. This is
the canonical "use L" estimator. We compare it head-to-head with HLL+clip at a
matched memory budget (bits), on near-duplicate-free streams -- the regime where
the clip helps.

Memory accounting: a production HLL register is ~5 bits (4-bit dense + overhead);
a sampled 64-bit key is 64 bits. So a sketch of m=2^p registers is matched by a
sample of s = round(5 * 2^p / 64) keys -- generous to GEE (we ignore the per-value
bookkeeping a real sample needs).

Result we expect and report honestly: in the sub-linear-memory streaming regime
(s << L), GEE cannot see enough of the stream to count near-distinct cardinalities
-- the sqrt(L/s) singleton term forces a large underestimate -- while HLL+clip is
accurate. This is the textbook reason sketches exist; the clip inherits HLL's
advantage. GEE is excellent in its own regime (large sampling fraction), which the
sketch setting is not.
"""

import json
import os

import numpy as np
import datasketches as ds

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE
PS = [10, 12, 14]
NS = [8000, 32000, 128000]
DS_GRID = [1.0, 1.02, 1.1]
T = 300
BITS_PER_REGISTER = 5
BITS_PER_SAMPLE = 64


def gee_estimate(n, d, s, rng):
    """One GEE trial. Stream = n distinct items, uniform duplication to L=round(n*d).

    The sample is s ROWS drawn WITHOUT replacement from the L-row stream, which is what
    a sampling-based estimator actually gets. Drawing with replacement instead would be
    valid only for s << L; here s/L reaches 0.16, and the spurious repeat-collisions it
    creates deflate f_1 and so understate GEE -- i.e. they would flatter the clip, the
    method we are comparing against. Row r carries item r % n (uniform duplication).
    """
    L = int(round(n * d))
    s = min(s, L)
    items = rng.choice(L, size=s, replace=False) % n
    _, counts = np.unique(items, return_counts=True)
    f1 = np.sum(counts == 1)
    rest = np.sum(counts >= 2)      # Sum_{j>=2} f_j: VALUES seen twice or more,
                                    # not rows -- the GEE formula counts values.
    return np.sqrt(L / s) * f1 + rest


def hll_clip_estimate(n, d, p, key_base, hasher_salt):
    L = int(round(n * d))
    sk = ds.hll_sketch(p)
    for i in range(key_base, key_base + n):
        sk.update(i ^ hasher_salt)
    raw = sk.get_estimate()
    return raw, min(raw, L), key_base + n


def rel_rmse(est, n):
    est = np.asarray(est, float)
    return float(np.sqrt(np.mean(((est - n) / n) ** 2)))


def main():
    rng = np.random.default_rng(20260614)
    key_base = 10 ** 9
    cells = []
    for p in PS:
        m = 1 << p
        s = max(1, round(BITS_PER_REGISTER * m / BITS_PER_SAMPLE))
        for n in NS:
            for d in DS_GRID:
                L = int(round(n * d))
                hll_raw, hll_clip, gee = [], [], []
                for t in range(T):
                    raw, clip, key_base = hll_clip_estimate(n, d, p, key_base, t * 2654435761 & 0xFFFFFFFF)
                    hll_raw.append(raw)
                    hll_clip.append(clip)
                    gee.append(gee_estimate(n, d, s, rng))
                cell = {
                    "p": p, "m": m, "sample_s": s, "n": n, "d": d, "L": L,
                    "rmse_hll_raw": rel_rmse(hll_raw, n),
                    "rmse_hll_clip": rel_rmse(hll_clip, n),
                    "rmse_gee": rel_rmse(gee, n),
                    "gee_over_clip": rel_rmse(gee, n) / rel_rmse(hll_clip, n),
                }
                cells.append(cell)
                print(f"p={p} m={m:5d} s={s:4d} n={n:7d} d={d:.2f} | "
                      f"HLL+clip={cell['rmse_hll_clip']:.4f}  GEE={cell['rmse_gee']:.4f}  "
                      f"(GEE is {cell['gee_over_clip']:.0f}x worse)", flush=True)

    with open(os.path.join(OUT, "exp9_competitors.json"), "w") as f:
        json.dump({"params": {"ps": PS, "ns": NS, "ds": DS_GRID, "T": T,
                              "bits_per_register": BITS_PER_REGISTER,
                              "bits_per_sample": BITS_PER_SAMPLE}, "cells": cells},
                  f, indent=2)
    d1 = [c for c in cells if c["d"] == 1.0]
    print(f"\nnear-distinct (d=1.0): GEE is {np.median([c['gee_over_clip'] for c in d1]):.0f}x "
          f"worse than HLL+clip (median) at matched memory")
    print("wrote exp9_competitors.json")


if __name__ == "__main__":
    main()
