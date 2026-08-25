#!/usr/bin/env python3
"""Verify the second-order, skew-aware clip theory.

First-order (Gaussian) model:  MSE_clip/MSE_raw = g(a) = Phi(a) - a phi(a) + a^2 Phibar(a),
gain1(a) = 1 - sqrt(g(a)).  This leaves a small positive residual (measured gain a bit
above prediction) because the real relative error is right-skewed.

Second-order (Edgeworth, skewness gamma1) closed form, derived in THEORY.md:
   g2(a) = g(a) - (gamma1/3) (a^2 + 1) phi(a).
At a=0:  g2(0) = 1/2 - gamma1/(3 sqrt(2 pi)).

We also include the (small) standardized bias c = beta/sigma via a full numerical
model, to confirm skew is the dominant correction. For each swept cell we measure
beta, sigma, gamma1 from the pooled raw estimates and compare measured gain to:
  - gain1: Gaussian, zero mean (current paper)
  - gain2: skew-only closed form g2(a)
  - gain3: full numerical model (measured beta + gamma1, Edgeworth)
Verdict: does the second-order prediction shrink the residual vs first-order?
"""

import json
import os

import numpy as np
from scipy.stats import norm, skew

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE  # honor smoke-run output dir
phi, Phi, Phibar = norm.pdf, norm.cdf, norm.sf
SEEDS = ["20260612", "987654321", "31337"]


def g1(a):
    return Phi(a) - a * phi(a) + a ** 2 * Phibar(a)


def gain1(a):
    return 100 * (1 - np.sqrt(g1(a)))


def g2(a, gamma1):
    return g1(a) - (gamma1 / 3.0) * (a ** 2 + 1.0) * phi(a)


def gain2(a, gamma1):
    return 100 * (1 - np.sqrt(np.maximum(g2(a, gamma1), 1e-9)))


# full numerical model: r = beta + sigma Z, Z ~ Edgeworth(gamma1); clip wall at w=d-1
_Z = np.linspace(-12, 12, 24001)
_DZ = _Z[1] - _Z[0]


def edgeworth_pdf(z, gamma1):
    return phi(z) * (1.0 + (gamma1 / 6.0) * (z ** 3 - 3.0 * z))


def gain_full(beta, sigma, gamma1, w):
    f = edgeworth_pdf(_Z, gamma1)
    f = np.clip(f, 0, None)              # asymptotic expansion can dip <0 in far tails
    f /= np.sum(f) * _DZ
    r = beta + sigma * _Z
    clipped = np.minimum(r, w)
    mse_raw = np.sum(r ** 2 * f) * _DZ
    mse_clip = np.sum(clipped ** 2 * f) * _DZ
    return 100 * (1 - np.sqrt(mse_clip / mse_raw))


def main():
    with open(os.path.join(OUT, "exp4_pooled_ci.json")) as f:
        cells = json.load(f)
    npz = {s: np.load(os.path.join(OUT, f"exp4_seed{s}.npz")) for s in SEEDS}

    rows = []
    for c in cells:
        n = c["n"]
        key = f"{c['arm']}|p{c['p']}|d{c['d']}"
        raw = np.concatenate([npz[s][f"{key}|raw"] for s in SEEDS])
        r = (raw - n) / n
        beta, sigma, g1m = float(np.mean(r)), float(np.std(r)), float(skew(r))
        w = c["d"] - 1.0
        a = w / sigma
        meas = c["dclip"]
        rows.append({
            "arm": c["arm"], "m": c["m"], "d": c["d"], "a": a,
            "beta": beta, "sigma": sigma, "gamma1": g1m, "meas": meas,
            "g1": float(gain1(a)), "g2": float(gain2(a, g1m)),
            "g3": float(gain_full(beta, sigma, g1m, w)),
        })

    # headline: d=1.0 cells
    print("=== d=1.0 cells: measured skew and gain vs predictions ===")
    print(f"{'arm':>11} {'m':>6} {'gamma1':>7} | {'meas%':>6} {'1st%':>6} {'2nd%':>6} {'full%':>6}")
    d1 = [r for r in rows if abs(r["d"] - 1.0) < 1e-9]
    for r in d1:
        print(f"{r['arm']:>11} {r['m']:>6} {r['gamma1']:>7.3f} | {r['meas']:>6.2f} "
              f"{r['g1']:>6.2f} {r['g2']:>6.2f} {r['g3']:>6.2f}")
    gam_mean = np.mean([r["gamma1"] for r in d1])
    print(f"\nmean skewness at d=1.0: gamma1 = {gam_mean:.3f}")
    print(f"  => closed-form g2(0): MSE ratio = {0.5 - gam_mean/(3*np.sqrt(2*np.pi)):.4f}, "
          f"gain = {100*(1-np.sqrt(0.5 - gam_mean/(3*np.sqrt(2*np.pi)))):.2f}%")

    # residuals over ALL cells (the collapse)
    def rms(key):
        return float(np.sqrt(np.mean([(r["meas"] - r[key]) ** 2 for r in rows])))

    def mean_dev(key):
        return float(np.mean([r["meas"] - r[key] for r in rows]))

    print(f"\n=== residual over all {len(rows)} swept cells (measured - prediction) ===")
    for key, lab in [("g1", "1st-order Gaussian"), ("g2", "2nd-order skew (closed form)"),
                     ("g3", "full numeric (bias+skew)")]:
        print(f"  {lab:>30}: mean {mean_dev(key):+.3f}pp  RMS {rms(key):.3f}pp")

    with open(os.path.join(OUT, "skew_theory.json"), "w") as f:
        json.dump({"cells": rows, "gamma1_d1_mean": float(gam_mean),
                   "rms_g1": rms("g1"), "rms_g2": rms("g2"), "rms_g3": rms("g3")},
                  f, indent=2)
    print("\nwrote skew_theory.json")


if __name__ == "__main__":
    main()
