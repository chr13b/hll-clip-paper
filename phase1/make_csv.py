#!/usr/bin/env python3
"""Aggregate every pooled cell from all six experiments into one
tidy CSV (results_all.csv) for the paper's tables/figures.

Columns: experiment, arm, estimator_family, m, n, d, L, T_per_seed, rmse_raw,
rmse_clip, rmse_tpm, dRMSE_clip_pct, dRMSE_tpm_pct, ci_clip_lo, ci_clip_hi,
ci_tpm_lo, ci_tpm_hi, z_clip, z_tpm, bind_rate, a_std (= (d-1)/sigma_emp where
available), clip_worse_than_raw (bool).
"""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE  # smoke runs read/write under HLL_OUT

FAMILY = {
    "ideal-prng": "ideal", "ideal": "ideal",
    "words-xxh64": "classic-FFGM", "words-mmh3": "classic-FFGM",
    "words-sha256": "classic-FFGM", "urls-xxh64": "classic-FFGM",
    "urls-mmh3": "classic-FFGM", "urls-sha256": "classic-FFGM",
    "ints-xxh64": "classic-FFGM", "ints-mmh3": "classic-FFGM",
    "ints-sha256": "classic-FFGM",
    "xxh64-ints": "HLL++", "hll_hip": "HIP-martingale", "cpc": "CPC",
    "ull_fgra": "UltraLogLog", "pypi_hll": "HLL++ (real-world PyPI)",
}

ROWS = []


def add(exp, c, has_ci, sigma_lookup=None):
    arm = c.get("arm", "?")
    m = c.get("m")
    if m is None and "p" in c:
        m = 1 << c["p"]
    if m is None and "lg_k" in c:
        m = 1 << c["lg_k"]      # exp5 (PyPI, lg_k=14)
    if m is None:
        m = 1024  # exp3/exp6 fixed m (lg_k/p = 10)
    rr = c["rmse_raw"]
    dclip = c.get("dclip", 100 * (1 - c["rmse_clip"] / rr))
    dtpm = c.get("dtpm", 100 * (1 - c["rmse_tpm"] / rr))
    ci_c = c.get("ci_clip", ["", ""])
    ci_t = c.get("ci_tpm", ["", ""])
    a_std = ""
    if sigma_lookup is not None:
        s = sigma_lookup.get((arm, m))
        if s:
            a_std = round((c["d"] - 1) / s, 3)
    ROWS.append({
        "experiment": exp,
        "arm": arm,
        "estimator_family": FAMILY.get(arm, "?"),
        "m": m,
        "n": c["n"],
        "d": c["d"],
        "L": c.get("L", int(round(c["n"] * c["d"]))),
        # per-SEED trial count, not the pooled total; blank where the source
        # experiment does not record it (exp1, exp2).
        "T_per_seed": c.get("T", ""),
        "rmse_raw": round(rr, 6),
        "rmse_clip": round(c["rmse_clip"], 6),
        "rmse_tpm": round(c["rmse_tpm"], 6),
        "dRMSE_clip_pct": round(dclip, 3),
        "dRMSE_tpm_pct": round(dtpm, 3),
        "ci_clip_lo": round(ci_c[0], 3) if ci_c[0] != "" else "",
        "ci_clip_hi": round(ci_c[1], 3) if ci_c[1] != "" else "",
        "ci_tpm_lo": round(ci_t[0], 3) if ci_t[0] != "" else "",
        "ci_tpm_hi": round(ci_t[1], 3) if ci_t[1] != "" else "",
        "z_clip": round(c["z_clip"], 3),
        "z_tpm": round(c["z_tpm"], 3),
        "bind_rate": round(c.get("bind", c.get("bind_rate", 0)), 4),
        "a_std": a_std,
        "clip_worse_than_raw": c["rmse_clip"] / rr > 1.01,
    })


def main():
    # exp1, exp2: pooled (no CI) from pool.py
    for exp in ("exp1", "exp2"):
        with open(os.path.join(OUT, f"{exp}_pooled.json")) as f:
            for c in json.load(f):
                add(exp, c, has_ci=False)
    # exp3: pooled CI
    with open(os.path.join(OUT, "exp3_pooled_ci.json")) as f:
        for c in json.load(f):
            add("exp3", c, has_ci=True)
    # exp4: pooled CI, with empirical sigma per (arm,m) from d=1.20
    with open(os.path.join(OUT, "exp4_pooled_ci.json")) as f:
        e4 = json.load(f)
    sig = {(c["arm"], c["m"]): c["rmse_raw"] for c in e4 if abs(c["d"] - 1.20) < 1e-9}
    for c in e4:
        add("exp4", c, has_ci=True, sigma_lookup=sig)
    # exp5 (real-world PyPI) + exp6 (UltraLogLog): pooled CI, optional
    for exp in ("exp5", "exp6"):
        path = os.path.join(OUT, f"{exp}_pooled_ci.json")
        if os.path.exists(path):
            with open(path) as f:
                for c in json.load(f):
                    add(exp, c, has_ci=True)

    # canonical run -> repo root; smoke run (HLL_OUT set) -> stay inside OUT
    out = (os.path.join(OUT, "results_all.csv") if os.environ.get("HLL_OUT")
           else os.path.join(os.path.dirname(HERE), "results_all.csv"))
    cols = list(ROWS[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(ROWS)
    n_worse = sum(r["clip_worse_than_raw"] for r in ROWS)
    print(f"wrote {out}: {len(ROWS)} pooled cells, "
          f"{n_worse} cells where clip >1% worse than raw")


if __name__ == "__main__":
    main()
