#!/usr/bin/env python3
"""Pool per-trial estimates across seeds (paired analysis, as in
the original prototype's pooled analysis): for each cell, concatenate trials from all seeds,
report pooled RMSE per estimator and z = mean(err_x^2 - err_raw^2) / SE.

Usage: pool.py exp1 20260612 987654321
       pool.py exp2 20260612 987654321
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE  # smoke runs read/write under HLL_OUT


def main():
    exp = sys.argv[1]
    seeds = sys.argv[2:]
    metas = []
    datas = []
    for s in seeds:
        with open(os.path.join(OUT, f"{exp}_seed{s}.json")) as f:
            metas.append(json.load(f))
        datas.append(np.load(os.path.join(OUT, f"{exp}_seed{s}.npz")))

    cells = metas[0]["cells"]
    lines = [f"POOLED {exp} over {len(seeds)} seeds "
             f"(T={sum(m['params']['T'] for m in metas)} trials/cell). "
             f"z = mean(err_x^2 - err_raw^2)/SE", ""]
    if exp == "exp1":
        hdr = (f"{'arm':>14} {'n':>6} {'d':>5} | {'rmse_raw':>9} {'rmse_clip':>9} "
               f"{'rmse_tpm':>9} | {'dCLIP%':>7} {'dTPM%':>7} | {'z_clip':>8} {'z_tpm':>8} | {'bind%':>6}")
    else:
        hdr = (f"{'arm':>10} {'m':>6} {'n/m':>6} {'d':>5} | {'rmse_raw':>9} {'rmse_clip':>9} "
               f"{'rmse_tpm':>9} | {'dCLIP%':>7} {'dTPM%':>7} | {'z_clip':>8} {'z_tpm':>8} | {'bind%':>6}")
    lines += [hdr, "-" * len(hdr)]

    pooled_cells = []
    for c in cells:
        if exp == "exp1":
            ck = f"{c['arm'].replace('ideal-prng', 'ideal-prng')}|n{c['n']}|d{c['d']}"
            ck = f"{c['arm']}|n{c['n']}|d{c['d']}"
        else:
            ck = f"{c['arm']}|p{c['p']}|r{c['ratio']}|d{c['d']}"
        n = c["n"]
        L = c["L"]
        est = {}
        for name in ("raw", "clip", "tpm"):
            est[name] = np.concatenate([d[f"{ck}|{name}"] for d in datas])
        raw = est["raw"]

        def rel_rmse(x):
            return float(np.sqrt(np.mean((x - n) ** 2)) / n)

        def z(x):
            dd = (x - n) ** 2 - (raw - n) ** 2
            sd = np.std(dd, ddof=1)
            return 0.0 if sd == 0 else float(np.mean(dd) / (sd / np.sqrt(len(dd))))

        rr, rc, rt = rel_rmse(raw), rel_rmse(est["clip"]), rel_rmse(est["tpm"])
        pc = {"key": ck, "n": n, "d": c["d"], "rmse_raw": rr, "rmse_clip": rc,
              "rmse_tpm": rt, "z_clip": z(est["clip"]), "z_tpm": z(est["tpm"]),
              "bind": float(np.mean(raw > L))}
        if exp == "exp2":
            pc.update({"m": c["m"], "ratio": c["ratio"], "arm": c["arm"],
                       "sparse_zone": c["sparse_zone"],
                       "branch_lc": c["branch_lc"]})
        else:
            pc["arm"] = c["arm"]
        pooled_cells.append(pc)
        ic = 100 * (1 - rc / rr)
        it = 100 * (1 - rt / rr)
        if exp == "exp1":
            lines.append(f"{c['arm']:>14} {n:>6} {c['d']:>5.2f} | {rr:>9.5f} {rc:>9.5f} "
                         f"{rt:>9.5f} | {ic:>7.2f} {it:>7.2f} | {pc['z_clip']:>8.2f} "
                         f"{pc['z_tpm']:>8.2f} | {100*pc['bind']:>6.2f}")
        else:
            lines.append(f"{c['arm']:>10} {c['m']:>6} {c['ratio']:>6.2f} {c['d']:>5.2f} | "
                         f"{rr:>9.5f} {rc:>9.5f} {rt:>9.5f} | {ic:>7.2f} {it:>7.2f} | "
                         f"{pc['z_clip']:>8.2f} {pc['z_tpm']:>8.2f} | {100*pc['bind']:>6.2f}")

    lines.append("")
    if exp == "exp1":
        arms = sorted(set(p["arm"] for p in pooled_cells))
        fails, worse = [], []
        for a in arms:
            d1 = [p for p in pooled_cells if p["arm"] == a and p["d"] == 1.0]
            imp = float(np.mean([1 - p["rmse_clip"] / p["rmse_raw"] for p in d1]))
            lines.append(f"  {a:>14}: pooled d=1.0 avg clip dRMSE = {100*imp:6.2f}%")
            if a != "ideal-prng" and imp < 0.18:
                fails.append((a, round(imp, 4)))
            for p in [p for p in pooled_cells if p["arm"] == a]:
                if p["rmse_clip"] / p["rmse_raw"] > 1.01:
                    worse.append((a, p["n"], p["d"]))
        lines.append("")
        lines.append(f"arms below 18% at d=1.0: {fails if fails else 'NONE'}")
        lines.append(f"cells where clip >1% worse: {worse if worse else 'NONE'}")
        lines.append(f"POOLED VERDICT: {'SURVIVES' if not fails and not worse else 'FAILS'}")
    else:
        worse = [(p["arm"], p["m"], p["ratio"], p["d"]) for p in pooled_cells
                 if p["rmse_clip"] / p["rmse_raw"] > 1.01]
        lines.append(f"cells where clip >1% worse: {worse if worse else 'NONE'}")

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, f"{exp}_pooled.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(OUT, f"{exp}_pooled.json"), "w") as f:
        json.dump(pooled_cells, f, indent=2)


if __name__ == "__main__":
    main()
