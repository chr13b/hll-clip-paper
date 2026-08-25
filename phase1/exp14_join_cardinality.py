#!/usr/bin/env python3
"""Does the corrected NDV propagate into better join-cardinality estimates?

The per-column result (exp13) shows the clip makes NDV statistics more accurate on key
columns. An accurate NDV is a means rather than an end, so the next question is whether
the correction survives one step downstream. This experiment answers it for the textbook
join-cardinality estimator that optimizers actually use,

    |R join S|  ~=  |R| * |S| / max(NDV_R, NDV_S),

by feeding it three different NDV inputs -- exact, raw HLL, and clipped HLL -- on real
TPC-H primary-key/foreign-key joins, and comparing each against the true join cardinality
measured by SQL.

Two effects are worth separating.

1. MAGNITUDE. For a PK-FK join the PK side has NDV_R = |R| exactly, so the true answer is
   |S|. An HLL that overestimates NDV_R by a fraction eps returns |S|/(1+eps): the
   relative error of the join estimate is inherited directly from the relative error of
   the NDV estimate. The clip removes the overestimates, so it can only help, but the
   size of the effect is bounded by the NDV error itself and is therefore small at
   deployment precision. We report it rather than dress it up.

2. VALIDITY. On a key column the raw estimate frequently exceeds the ROW COUNT, which is
   semantically impossible: it asserts more distinct values than there are rows. That
   breaks the invariant NDV <= |R| that key detection and selectivity bounds rely on
   (a selectivity 1/NDV below 1/|R| implies a key more selective than a primary key).
   The clip restores the invariant for free. We count how often this happens.

Writes exp14_join_cardinality.json.
"""
import json
import os

import numpy as np
import datasketches as ds

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE

LG_K = 14
SF = 0.3
DRAWS = 200          # independent hash draws per column. At 40 the bootstrap intervals
                     # on the gain were several points wide -- an RMSE over T draws carries
                     # ~1/sqrt(2T) relative error, which the RMSE RATIO then amplifies. 200
                     # narrows them by ~sqrt(5) at 5x the runtime, which is affordable
                     # off-line even though the FK side has millions of rows.
# (PK table, PK column) joined with (FK table, FK column)
JOINS = [("customer", "c_custkey", "orders", "o_custkey"),
         ("orders", "o_orderkey", "lineitem", "l_orderkey"),
         ("part", "p_partkey", "lineitem", "l_partkey")]


def ndv_draws(values, draws):
    """Independent HLL estimates of the distinct count of `values`."""
    out = []
    for i in range(draws):
        salt = f"|{i}"
        sk = ds.hll_sketch(LG_K)
        for v in values:
            sk.update(v + salt)
        out.append(sk.get_estimate())
    return np.asarray(out, float)


def main():
    try:
        import duckdb
    except ImportError:
        print("duckdb not installed -- cannot run the TPC-H join study")
        return
    boot_rng = np.random.default_rng(20260824)
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={SF})")
    print(f"TPC-H sf={SF} generated", flush=True)

    cells = []
    for pk_t, pk_c, fk_t, fk_c in JOINS:
        R = con.execute(f"SELECT count(*) FROM {pk_t}").fetchone()[0]
        S = con.execute(f"SELECT count(*) FROM {fk_t}").fetchone()[0]
        true_join = con.execute(
            f"SELECT count(*) FROM {pk_t} JOIN {fk_t} ON {pk_c}={fk_c}").fetchone()[0]
        ndv_pk_exact = con.execute(
            f"SELECT count(DISTINCT {pk_c}) FROM {pk_t}").fetchone()[0]
        ndv_fk_exact = con.execute(
            f"SELECT count(DISTINCT {fk_c}) FROM {fk_t}").fetchone()[0]

        # Estimate BOTH sides with sketches, as an optimizer actually does. (Estimating
        # only the PK side and using an exact FK NDV is not merely optimistic, it is
        # degenerate: for these TPC-H joins NDV_fk happens to equal |R| exactly, so
        # clipping the PK side would force the max() onto the exact value and yield a
        # perfect join estimate -- an artifact of the setup, not a property of the clip.)
        pk_vals = [str(r[0]) for r in con.execute(f"SELECT {pk_c} FROM {pk_t}").fetchall()]
        fk_vals = [str(r[0]) for r in con.execute(f"SELECT {fk_c} FROM {fk_t}").fetchall()]
        raw_pk, raw_fk = ndv_draws(pk_vals, DRAWS), ndv_draws(fk_vals, DRAWS)
        clip_pk = np.minimum(raw_pk, R)          # L = row count of the PK table
        clip_fk = np.minimum(raw_fk, S)          # L = row count of the FK table
        # CONSTRAINT-AWARE clip: under referential integrity every foreign-key value
        # occurs in the referenced key, so NDV_fk <= NDV_pk <= |R|. A catalogue that
        # knows the constraint can therefore clip the FK side at the REFERENCED table's
        # row count, which is far tighter than the FK table's own.
        clip_fk_ri = np.minimum(raw_fk, R)

        def est(ndv_pk, ndv_fk):
            return R * S / np.maximum(ndv_pk, ndv_fk)

        e_exact = est(np.array([float(ndv_pk_exact)]),
                      np.array([float(ndv_fk_exact)]))[0]
        e_raw, e_clip = est(raw_pk, raw_fk), est(clip_pk, clip_fk)
        e_ri = est(clip_pk, clip_fk_ri)

        def relerr(e):
            return (np.asarray(e, float) - true_join) / true_join

        rr_raw = float(np.sqrt(np.mean(relerr(e_raw) ** 2)))
        rr_clip = float(np.sqrt(np.mean(relerr(e_clip) ** 2)))
        rr_ri = float(np.sqrt(np.mean(relerr(e_ri) ** 2)))
        gain = 100 * (1 - rr_clip / rr_raw) if rr_raw > 0 else 0.0
        gain_ri = 100 * (1 - rr_ri / rr_raw) if rr_raw > 0 else 0.0

        # Paired percentile bootstrap over the draws, so the quoted gains carry an
        # interval rather than a bare point estimate at a small draw count.
        def boot(err_num, err_den, reps=2000):
            a = np.asarray(err_num, float)
            b = np.asarray(err_den, float)
            idx = boot_rng.integers(0, a.size, size=(reps, a.size))
            num = np.sqrt(np.mean(a[idx] ** 2, axis=1))
            den = np.sqrt(np.mean(b[idx] ** 2, axis=1))
            g = 100 * (1 - num / den)
            return float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))

        ci_clip = boot(relerr(e_clip), relerr(e_raw))
        ci_ri = boot(relerr(e_ri), relerr(e_raw))
        impossible = float(np.mean(raw_pk > R))   # NDV > row count: semantically invalid
        impossible_fk = float(np.mean(raw_fk > S))

        cells.append({"pk": f"{pk_t}.{pk_c}", "fk": f"{fk_t}.{fk_c}",
                      "rows_pk": R, "rows_fk": S, "true_join": true_join,
                      "ndv_pk_exact": ndv_pk_exact, "ndv_fk_exact": ndv_fk_exact,
                      "join_est_exact_ndv": float(e_exact),
                      "rel_rmse_join_raw": rr_raw, "rel_rmse_join_clip": rr_clip,
                      "rel_rmse_join_ri": rr_ri,
                      "join_gain_pct": gain, "join_gain_ri_pct": gain_ri,
                      "join_gain_ci": ci_clip, "join_gain_ri_ci": ci_ri,
                      "impossible_ndv_rate": impossible,
                      "impossible_ndv_rate_fk": impossible_fk,
                      "clip_ever_worse": bool(rr_clip > rr_raw * 1.0001),
                      "draws": DRAWS})
        print(f"{pk_t}.{pk_c} |><| {fk_t}.{fk_c}: true={true_join:,} "
              f"| join rel-RMSE raw={rr_raw:.5f} clip={rr_clip:.5f} "
              f"gain={gain:5.1f}% [{ci_clip[0]:.1f},{ci_clip[1]:.1f}]  "
              f"RI-aware={gain_ri:5.1f}% [{ci_ri[0]:.1f},{ci_ri[1]:.1f}] | "
              f"NDV>rows: PK {impossible*100:.0f}%", flush=True)

    if cells:
        print()
        print(f"join-estimate gain (plain clip):        "
              f"{min(c['join_gain_pct'] for c in cells):5.1f}"
              f"--{max(c['join_gain_pct'] for c in cells):.1f}%")
        print(f"join-estimate gain (constraint-aware):   "
              f"{min(c['join_gain_ri_pct'] for c in cells):5.1f}"
              f"--{max(c['join_gain_ri_pct'] for c in cells):.1f}%")
        print(f"impossible NDV (raw) on key columns: "
              f"{min(c['impossible_ndv_rate'] for c in cells)*100:.0f}"
              f"--{max(c['impossible_ndv_rate'] for c in cells)*100:.0f}% of draws; "
              f"the clip removes all of them by construction")
        print(f"clip ever worse on any join? "
              f"{any(c['clip_ever_worse'] for c in cells)}")

    with open(os.path.join(OUT, "exp14_join_cardinality.json"), "w") as f:
        json.dump({"params": {"lg_k": LG_K, "sf": SF, "draws": DRAWS,
                              "estimator": "|R||S|/max(NDV_R,NDV_S)"},
                   "cells": cells}, f, indent=2)
    print("wrote exp14_join_cardinality.json")


if __name__ == "__main__":
    main()
