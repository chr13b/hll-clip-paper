#!/usr/bin/env python3
"""The refined clip: stream-length conditioning meets
heavy hitters, unifying the clip with duplicate-subtraction.

The duplicate-subtraction view: n = L - D, where D = sum_i (f_i - 1) is the
duplicate mass. A Misra-Gries summary with k counters estimates the duplicate
mass of the CONCENTRATED (heavy) items, D_hat, and -- because MG counters never
exceed the true frequency -- UNDERcounts it: D_hat <= D. Therefore
    U = L - D_hat   satisfies   n <= U <= L,
a valid upper bound on n that is TIGHTER than L. Clipping at U gives the

    refined clip:  n_refined = min(n_hat, L - D_hat).

Properties (proved in THEORY.md):
  * never worse than raw (clips at a valid upper bound n <= U);
  * pointwise dominates the basic clip min(n_hat, L) (tighter wall U <= L);
  * reduces to the basic clip when there are no heavy duplicates (D_hat = 0);
  * extends the benefit to heavy-tailed streams: when a few items carry the
    duplicate mass, U ~ n even though L >> n, so the clip bites where the basic
    clip (wall L) does nothing.

This experiment verifies all four on two stream models at matched-ish memory.
"""

import json
import os

import numpy as np
import datasketches as ds

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE
LG_K = 12                      # HLL m = 4096
MG_K = 256                     # Misra-Gries counters (add-on; ~256*96 bits ~ 3 KB)
NS = [8000, 32000]
T = 120


def misra_gries(stream, k):
    """Standard Misra-Gries. Returns {item: counter} with counter <= true freq."""
    c = {}
    for x in stream:
        if x in c:
            c[x] += 1
        elif len(c) < k:
            c[x] = 1
        else:
            dead = []
            for key in c:
                v = c[key] - 1
                if v == 0:
                    dead.append(key)
                else:
                    c[key] = v
            for key in dead:
                del c[key]
    return c


def dup_mass_hat(mg):
    return sum(v - 1 for v in mg.values() if v >= 2)


def make_uniform(n, d, rng):
    """Each of n items repeated to reach L = round(n*d); shuffled stream."""
    L = int(round(n * d))
    reps = np.ones(n, dtype=np.int64)
    extra = L - n
    if extra > 0:
        idx = rng.integers(0, n, size=extra)
        np.add.at(reps, idx, 1)
    stream = np.repeat(np.arange(n), reps)
    rng.shuffle(stream)
    return stream, L


def make_heavy(n, d, h, rng):
    """n-h singletons + h heavy items carrying all duplicate mass (concentrated)."""
    L = int(round(n * d))
    extra = L - n                       # duplicate occurrences to distribute over h heavies
    reps = np.ones(n, dtype=np.int64)
    if extra > 0 and h > 0:
        heavies = rng.choice(n, size=h, replace=False)
        add = np.full(h, extra // h)
        add[: extra % h] += 1
        reps[heavies] += add
    stream = np.repeat(np.arange(n), reps)
    rng.shuffle(stream)
    return stream, L


def hll_estimate(n_distinct, salt):
    sk = ds.hll_sketch(LG_K)
    for i in range(n_distinct):
        sk.update(i ^ salt)
    return sk.get_estimate()


def run_cell(model, n, d, h, rng, T):
    raw, basic, refined, dupsub = [], [], [], []
    Dhats = []
    for t in range(T):
        if model == "uniform":
            stream, L = make_uniform(n, d, rng)
        else:
            stream, L = make_heavy(n, d, h, rng)
        mg = misra_gries(stream.tolist(), MG_K)
        Dhat = dup_mass_hat(mg)
        Dhats.append(Dhat)
        U = L - Dhat
        est = hll_estimate(n, t * 2654435761 & 0xFFFFFFFF)
        raw.append(est)
        basic.append(min(est, L))
        refined.append(min(est, U))
        dupsub.append(U)
    raw, basic, refined, dupsub = map(lambda x: np.array(x, float),
                                      (raw, basic, refined, dupsub))

    def rr(e):
        return float(np.sqrt(np.mean(((e - n) / n) ** 2)))

    return {
        "model": model, "n": n, "d": d, "h": h, "L": int(round(n * d)),
        "rmse_raw": rr(raw), "rmse_basic": rr(basic), "rmse_refined": rr(refined),
        "rmse_dupsub": rr(dupsub),
        "Dhat_over_D": float(np.mean(Dhats)) / max(1, int(round(n * d)) - n),
        "refined_worse_than_raw": bool(rr(refined) > rr(raw) * 1.0001),
        "refined_worse_than_basic": bool(rr(refined) > rr(basic) * 1.0001),
    }


def main():
    rng = np.random.default_rng(20260614)
    cells = []
    grid = []
    # uniform: clip's home turf -- refined should ~match basic
    for n in NS:
        for d in [1.0, 1.01, 1.05, 1.5, 3.0]:
            grid.append(("uniform", n, d, 0))
    # heavy-tailed: refined extends the benefit when duplicates are concentrated
    for n in NS:
        for d in [2.0, 5.0]:
            for h in [1, 3, 10, 100]:
                grid.append(("heavy", n, d, h))

    for model, n, d, h in grid:
        c = run_cell(model, n, d, h, rng, T)
        cells.append(c)
        tag = f"{model}" + (f"/h={h}" if model == "heavy" else "")
        print(f"{tag:>12} n={n:6d} d={d:4.2f} | raw={c['rmse_raw']:.4f} "
              f"basic={c['rmse_basic']:.4f} refined={c['rmse_refined']:.4f} "
              f"(D_hat/D={c['Dhat_over_D']:.2f})", flush=True)

    any_worse_raw = any(c["refined_worse_than_raw"] for c in cells)
    any_worse_basic = any(c["refined_worse_than_basic"] for c in cells)
    print(f"\nrefined ever worse than raw?   {any_worse_raw}")
    print(f"refined ever worse than basic? {any_worse_basic}")
    heavy = [c for c in cells if c["model"] == "heavy" and c["h"] <= 10]
    impr = np.median([100 * (1 - c["rmse_refined"] / c["rmse_basic"]) for c in heavy])
    print(f"heavy-tailed (h<=10): refined cuts {impr:.0f}% of basic-clip RMSE (median)")

    with open(os.path.join(OUT, "exp10_refined.json"), "w") as f:
        json.dump({"params": {"lg_k": LG_K, "mg_k": MG_K, "T": T},
                   "cells": cells, "any_worse_raw": any_worse_raw,
                   "any_worse_basic": any_worse_basic}, f, indent=2)
    print("wrote exp10_refined.json")


if __name__ == "__main__":
    main()
