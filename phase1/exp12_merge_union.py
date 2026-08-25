#!/usr/bin/env python3
"""The value window under sketch MERGING and union queries.

Question (raised in review): sketches are mergeable, so what happens to the clip's
value window when two sketches are unioned?

Analysis. Merging carries the counters along: L = L1 + L2, and L >= n_union always,
so the wall stays VALID (Prop. 1 still applies). But the wall gets LOOSER, because an
item present in both inputs is counted twice in L and once in n_union. Writing
o = |S1 ^ S2| for the overlap and taking both inputs duplicate-free,

    d_union - 1 = (L1 + L2)/n_union - 1 = o / n_union,

so under merging the OVERLAP fraction plays exactly the role the duplicate fraction
plays for a single stream: the clip retains >= 5% RMSE while o/n_union <~ 1.6/sqrt(m).
Two corollaries: merging DISJOINT partitions (o = 0) preserves the window exactly,
while merging heavily-overlapping sketches destroys it (merging a sketch with itself
gives d_union = 2 regardless of the inputs).

This experiment verifies that law directly: build two duplicate-free sketches with a
controlled overlap, merge them, and compare the measured clip gain against the
single-stream curve evaluated at a = (o/n_union)/sigma.

Writes exp12_merge_union.json.
"""
import json
import os

import numpy as np
import datasketches as ds
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE

LG_K = 12                      # m = 4096
N_EACH = 20000                 # distinct items per input sketch
OVERLAP_FRACS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
T = 200

phi, Phi, Sf = norm.pdf, norm.cdf, norm.sf


def g(a):
    return Phi(a) - a * phi(a) + a * a * Sf(a)


def rel_rmse(est, truth):
    e = np.asarray(est, float)
    return float(np.sqrt(np.mean(((e - truth) / truth) ** 2)))


def main():
    rng = np.random.default_rng(20260803)
    cells = []

    # Calibrate sigma on the MERGED estimator itself. A union of two sketches has a
    # larger relative standard error than a single sketch built from the same items
    # (0.0140 vs 0.0120 here), and the value-window law is stated in units of the
    # estimator's own sigma, so standardizing with a single-sketch sigma would misplace
    # every point on the curve.
    cal = []
    for t in range(200):
        base = t * 1_000_003
        s1, s2 = ds.hll_sketch(LG_K), ds.hll_sketch(LG_K)
        for i in range(N_EACH):
            s1.update(base + i)
        for i in range(N_EACH):
            s2.update(base + N_EACH + i)
        u = ds.hll_union(LG_K)
        u.update(s1); u.update(s2)
        cal.append(u.get_result().get_estimate())
    sigma = float(np.std(np.asarray(cal) / (2 * N_EACH)))
    print(f"calibrated sigma of the MERGED estimator at n={2*N_EACH}, "
          f"lg_k={LG_K}: {sigma:.5f}", flush=True)

    for of in OVERLAP_FRACS:
        raw, clip = [], []
        for t in range(T):
            base = t * 7_000_003
            o = int(round(of * N_EACH))
            # S1 = [0, N_EACH) ; S2 shares the first o items, then fresh ones
            s1 = ds.hll_sketch(LG_K)
            s2 = ds.hll_sketch(LG_K)
            for i in range(N_EACH):
                s1.update(base + i)
            for i in range(o):
                s2.update(base + i)                       # overlapping items
            for i in range(N_EACH - o):
                s2.update(base + N_EACH + i)              # disjoint items
            union = ds.hll_union(LG_K)
            union.update(s1)
            union.update(s2)
            est = union.get_result().get_estimate()

            n_union = 2 * N_EACH - o
            L = 2 * N_EACH                                 # counters add on merge
            raw.append(est)
            clip.append(min(est, L))

        n_union = 2 * N_EACH - int(round(of * N_EACH))
        L = 2 * N_EACH
        d_union = L / n_union
        a = (d_union - 1) / sigma
        r_raw, r_clip = rel_rmse(raw, n_union), rel_rmse(clip, n_union)
        gain = 100 * (1 - r_clip / r_raw)
        pred = 100 * (1 - np.sqrt(g(a)))
        # record the union estimator's own first two moments. NOTE: we deliberately do
        # NOT "predict" the gain by clipping these same samples -- min(est,L) is exactly
        # min(rel,w), so such a "prediction" would reproduce the measurement identically
        # and mean nothing. The first-order curve below is the genuine prediction.
        rel = (np.asarray(raw, float) - n_union) / n_union
        beta, sig_m = float(np.mean(rel)), float(np.std(rel))
        cells.append({"overlap_frac": of, "n_union": n_union, "L": L,
                      "d_union": d_union, "a": a, "beta": beta, "sigma_measured": sig_m,
                      "rmse_raw": r_raw, "rmse_clip": r_clip,
                      "gain_pct": gain, "gain_pred_pct": pred,
                      "clip_worse_than_raw": bool(r_clip > r_raw * 1.0001)})
        print(f"overlap={of*100:5.1f}%  d_union={d_union:.4f}  a={a:5.2f} | "
              f"gain={gain:5.1f}%  1st-order={pred:5.1f}%  "
              f"(bias={beta*100:+.2f}%, sigma={sig_m:.5f})", flush=True)

    dev = [c["gain_pct"] - c["gain_pred_pct"] for c in cells]
    rms = float(np.sqrt(np.mean(np.square(dev))))
    any_worse = any(c["clip_worse_than_raw"] for c in cells)
    print(f"\nRMS deviation from the single-stream law: {rms:.2f} points")
    print(f"clip ever worse than raw after merging? {any_worse}")

    with open(os.path.join(OUT, "exp12_merge_union.json"), "w") as f:
        json.dump({"params": {"lg_k": LG_K, "n_each": N_EACH, "T": T, "sigma": sigma},
                   "cells": cells, "rms_dev_points": rms, "any_worse": any_worse,
                   "note": "merged L = L1+L2 ; overlap fraction replaces duplicate fraction"},
                  f, indent=2)
    print("wrote exp12_merge_union.json")


if __name__ == "__main__":
    main()
