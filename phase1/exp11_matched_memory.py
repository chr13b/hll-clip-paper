#!/usr/bin/env python3
"""The fair-budget matched-memory comparison reported in
section "A Tighter Wall, and Its Memory Cost" (the refined clip vs a larger plain HLL).

Question: the refined clip min(n_hat, L - D_hat) tightens the wall with a
Misra-Gries (MG) heavy-hitter sketch, but MG costs memory. At MATCHED TOTAL
memory, is that memory better spent on MG counters (refined clip) or on more
HLL registers (a larger plain sketch)?

We answer it as two memory--accuracy frontiers on the refined clip's MOST
FAVORABLE input -- one item carrying nearly all the duplicate mass (h=1),
d=5, n=32000 -- where the basic wall L is useless but the refined wall
U = L - D_hat collapses onto n:

  * PLAIN frontier:   plain HLL at lg_k = 11..15 (no clip needed; on this
    stream the raw estimate already tracks n, duplicates are invisible to HLL).
    x = updatable serialized bytes (a function of lg_k only -- the in-RAM
    footprint), y = relative RMSE.
  * REFINED frontier: HLL(lg_k=12) + MG(mg_k) using min(n_hat, L - D_hat).
    x = HLL bytes + MG bytes (mg_k * 12: an 8-byte key + 4-byte counter each).
    With a single heavy item any mg_k >= 2 captures it, but Misra-Gries
    decrements the heavy counter once per k singletons, so D_hat = D - n/k and
    the refined wall sits at n(1 + 1/k): 4.9 sigma above n at k=16 (never
    binds), 1.2 sigma at k=64, 0.3 sigma at k=256. The refined accuracy only
    approaches the clip's once 1/k << sigma, i.e. a few hundred counters -- by
    which point the same bytes buy a larger plain sketch. That is why the
    refined frontier descends slowly against the faster-descending plain one.

Memory accounting is deterministic (no data-dependent compact sizes): HLL uses
datasketches' updatable serialized size; MG uses 12 bytes/counter.

Reproducible: fixed seed. Writes exp11_matched_memory.json, whose numbers (refined clip
RMSE vs the plain-HLL frontier, and the crossover budget) are cited in that section.
"""
import json
import os

import numpy as np
import datasketches as ds

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE

N = 32000
D = 5.0          # L = n*d
H = 1            # one heavy item carries the duplicate mass (refined clip's best case)
PLAIN_LGK = [11, 12, 13, 14, 15]
REF_LGK = 12
REF_MGK = [16, 64, 256, 1024]
T_PLAIN = 200    # HLL builds are cheap -> many trials for a smooth frontier
T_REF = 60       # MG passes are O(L*k) -> fewer trials
MG_BYTES_PER_COUNTER = 12   # 8-byte key + 4-byte count


def hll_bytes(lgk):
    # dense (fully-populated) updatable footprint -- a function of lg_k only,
    # so the memory budget is deterministic and data-independent.
    return ds.hll_sketch.get_max_updatable_serialization_bytes(
        lgk, ds.tgt_hll_type.HLL_4)


def misra_gries(stream, k):
    """Standard Misra-Gries. Returns {item: counter}, counter <= true freq."""
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


def make_heavy(n, d, h, rng):
    """n-h singletons + h heavy items carrying all duplicate mass (concentrated)."""
    L = int(round(n * d))
    extra = L - n
    reps = np.ones(n, dtype=np.int64)
    if extra > 0 and h > 0:
        heavies = rng.choice(n, size=h, replace=False)
        add = np.full(h, extra // h)
        add[: extra % h] += 1
        reps[heavies] += add
    stream = np.repeat(np.arange(n), reps)
    rng.shuffle(stream)
    return stream, L


def hll_estimate(n_distinct, lgk, salt):
    sk = ds.hll_sketch(lgk)
    for i in range(n_distinct):
        sk.update(i ^ salt)
    return sk.get_estimate()


def rel_rmse(ests, n):
    e = np.asarray(ests, float)
    return float(np.sqrt(np.mean(((e - n) / n) ** 2)))


def main():
    rng = np.random.default_rng(20260621)
    L = int(round(N * D))

    # --- plain frontier: raw HLL at several register counts ---
    plain = []
    for lgk in PLAIN_LGK:
        ests = [hll_estimate(N, lgk, (t * 2654435761) & 0xFFFFFFFF)
                for t in range(T_PLAIN)]
        plain.append({"lgk": lgk, "bytes": hll_bytes(lgk),
                      "rmse": rel_rmse(ests, N)})
        print(f"plain  lg_k={lgk:2d}  bytes={plain[-1]['bytes']:6d}  "
              f"rmse={plain[-1]['rmse']:.5f}", flush=True)

    # --- refined frontier: HLL(REF_LGK) + MG(mg_k), min(n_hat, L - D_hat) ---
    hb = hll_bytes(REF_LGK)
    refined = []
    for mgk in REF_MGK:
        ests, dhat_frac = [], []
        for t in range(T_REF):
            stream, Lt = make_heavy(N, D, H, rng)
            Dhat = dup_mass_hat(misra_gries(stream.tolist(), mgk))
            dhat_frac.append(Dhat / max(1, Lt - N))
            est = hll_estimate(N, REF_LGK, (t * 40503) & 0xFFFFFFFF)
            ests.append(min(est, Lt - Dhat))
        refined.append({"mgk": mgk, "bytes": hb + mgk * MG_BYTES_PER_COUNTER,
                        "rmse": rel_rmse(ests, N),
                        "Dhat_over_D": float(np.mean(dhat_frac))})
        print(f"refine mg_k={mgk:4d}  bytes={refined[-1]['bytes']:6d}  "
              f"rmse={refined[-1]['rmse']:.5f}  "
              f"Dhat/D={refined[-1]['Dhat_over_D']:.3f}", flush=True)

    # --- the memory value-window: compare each refined point against the BEST
    # PLAIN SKETCH THAT FITS THE SAME BUDGET (plain HLL memory is quantized at
    # powers of two, so the lower envelope is a step function). The refined clip
    # is the better use of memory only until the next register-doubling fits. ---
    def best_affordable_plain(budget):
        fits = [p for p in plain if p["bytes"] <= budget]
        return min(fits, key=lambda p: p["rmse"]) if fits else None
    window_hi = 0
    print()
    for r in refined:
        bp = best_affordable_plain(r["bytes"])
        win = bp is not None and r["rmse"] < bp["rmse"]
        if win:
            window_hi = max(window_hi, r["bytes"])
        print(f"refine {r['bytes']:6d}B rmse={r['rmse']:.5f}  vs best plain<=budget "
              f"(lg{bp['lgk']}, {bp['rmse']:.5f}) -> "
              f"{'REFINED wins' if win else 'plain wins'}")
    # first plain doubling that overtakes the refined clip's saturated accuracy
    ref_floor = float(min(r["rmse"] for r in refined))
    overtake = min((p["bytes"] for p in plain if p["rmse"] < ref_floor), default=None)
    print(f"\nrefined clip saturates near rmse={ref_floor:.5f}; a plain HLL first "
          f"beats that at {overtake} bytes (lg_k where it fits).")
    print(f"value-window: refined clip is the better use of memory up to "
          f"~{window_hi} bytes, dominated by a larger plain sketch above.")

    out = {"plain": plain, "refined": refined,
           "params": {"n": N, "d": D, "h": H, "L": L,
                      "ref_lgk": REF_LGK, "T_plain": T_PLAIN, "T_ref": T_REF,
                      "mg_bytes_per_counter": MG_BYTES_PER_COUNTER},
           "ref_rmse_floor": ref_floor, "window_hi_bytes": window_hi,
           "plain_overtakes_at_bytes": overtake,
           "note": "h=1,d=5 heavy-tailed; HLL bytes = max updatable (dense) size; "
                   "MG bytes = mg_k*12"}
    with open(os.path.join(OUT, "exp11_matched_memory.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote exp11_matched_memory.json")


if __name__ == "__main__":
    main()
