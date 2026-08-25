#!/usr/bin/env python3
"""Pool exp3/exp4 across seeds with paired bootstrap CIs.

For each cell: concatenate trials over seeds; report pooled relative RMSE per
estimator, dRMSE% for clip/tpm with 95% percentile bootstrap CIs (B=2000 paired
resamples of trials), bind rate, paired z.

exp4 extras: per (arm, m) empirical sigma (= rmse_raw at d=1.20, clip never binds
there), the scaling variable x = (d-1)/sigma, and the interpolated d* where the
clip gain crosses 5%.

Usage: ci.py exp3 20260612 987654321 31337
       ci.py exp4 20260612 987654321 31337
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE  # smoke runs read/write under HLL_OUT
B = 2000


def cell_key(exp, c):
    if exp in ("exp3", "exp5", "exp6"):
        return f"{c['arm']}|n{c['n']}|d{c['d']}"
    if exp == "exp4":
        return f"{c['arm']}|p{c['p']}|d{c['d']}"
    raise ValueError(exp)


def boot_ci(err_raw2, err_x2, rng):
    """95% CI for 100*(1 - sqrt(mean(err_x2))/sqrt(mean(err_raw2))) via paired
    bootstrap over trials."""
    T = len(err_raw2)
    idx = rng.integers(0, T, size=(B, T))
    g = 100 * (1 - np.sqrt(np.mean(err_x2[idx], axis=1) /
                           np.mean(err_raw2[idx], axis=1)))
    return float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))


def main():
    exp = sys.argv[1]
    seeds = sys.argv[2:]
    metas, datas = [], []
    for s in seeds:
        with open(os.path.join(OUT, f"{exp}_seed{s}.json")) as f:
            metas.append(json.load(f))
        datas.append(np.load(os.path.join(OUT, f"{exp}_seed{s}.npz")))
    cells = metas[0]["cells"]
    rng = np.random.default_rng(0)

    pooled = []
    for c in cells:
        ck = cell_key(exp, c)
        n = c["n"]
        L = c["L"]
        est = {name: np.concatenate([d[f"{ck}|{name}"] for d in datas])
               for name in ("raw", "clip", "tpm")}
        raw = est["raw"]
        e2 = {k: (v - n) ** 2 for k, v in est.items()}

        def rel_rmse(k):
            return float(np.sqrt(np.mean(e2[k])) / n)

        def z(k):
            dd = e2[k] - e2["raw"]
            sd = np.std(dd, ddof=1)
            return 0.0 if sd == 0 else float(np.mean(dd) / (sd / np.sqrt(len(dd))))

        rr = rel_rmse("raw")
        pc = dict(c)
        pc.update({
            "key": ck, "rmse_raw": rr, "rmse_clip": rel_rmse("clip"),
            "rmse_tpm": rel_rmse("tpm"),
            "dclip": 100 * (1 - rel_rmse("clip") / rr),
            "dtpm": 100 * (1 - rel_rmse("tpm") / rr),
            "ci_clip": boot_ci(e2["raw"], e2["clip"], rng),
            "ci_tpm": boot_ci(e2["raw"], e2["tpm"], rng),
            "z_clip": z("clip"), "z_tpm": z("tpm"),
            "bind": float(np.mean(raw > L)),
        })
        pooled.append(pc)

    lines = [f"POOLED {exp} over {len(seeds)} seeds "
             f"(T={sum(m['params']['T'] for m in metas)}/cell), "
             f"95% paired bootstrap CIs (B={B})", ""]
    if exp in ("exp3", "exp5", "exp6"):
        hdr = (f"{'arm':>8} {'n':>6} {'d':>5} | {'rmse_raw':>9} | "
               f"{'dCLIP% [95% CI]':>24} {'dTPM% [95% CI]':>24} | {'z_clip':>8} | {'bind%':>6}")
        lines += [hdr, "-" * len(hdr)]
        for c in pooled:
            lines.append(
                f"{c['arm']:>8} {c['n']:>6} {c['d']:>5.2f} | {c['rmse_raw']:>9.5f} | "
                f"{c['dclip']:>7.2f} [{c['ci_clip'][0]:>6.2f},{c['ci_clip'][1]:>6.2f}] "
                f"{c['dtpm']:>7.2f} [{c['ci_tpm'][0]:>6.2f},{c['ci_tpm'][1]:>6.2f}] | "
                f"{c['z_clip']:>8.2f} | {100*c['bind']:>6.2f}")
    else:
        # empirical sigma per (arm, m) from the d=1.20 cell
        sig = {(c["arm"], c["m"]): c["rmse_raw"] for c in pooled if c["d"] == 1.2}
        hdr = (f"{'arm':>10} {'m':>6} {'d':>6} {'x=(d-1)/s':>9} | {'rmse_raw':>9} | "
               f"{'dCLIP% [95% CI]':>24} {'dTPM% [95% CI]':>24} | {'z_clip':>8} | {'bind%':>6}")
        lines += [hdr, "-" * len(hdr)]
        for c in pooled:
            s = sig[(c["arm"], c["m"])]
            x = (c["d"] - 1) / s
            c["x"] = x
            lines.append(
                f"{c['arm']:>10} {c['m']:>6} {c['d']:>6.3f} {x:>9.2f} | {c['rmse_raw']:>9.5f} | "
                f"{c['dclip']:>7.2f} [{c['ci_clip'][0]:>6.2f},{c['ci_clip'][1]:>6.2f}] "
                f"{c['dtpm']:>7.2f} [{c['ci_tpm'][0]:>6.2f},{c['ci_tpm'][1]:>6.2f}] | "
                f"{c['z_clip']:>8.2f} | {100*c['bind']:>6.2f}")
        lines.append("")
        lines.append("empirical sigma (rmse_raw at d=1.20) and 5%-gain window edge d*:")
        for (arm, m), s in sorted(sig.items()):
            cs = sorted([c for c in pooled if c["arm"] == arm and c["m"] == m],
                        key=lambda c: c["d"])
            dstar = None
            for a, b in zip(cs, cs[1:]):
                if a["dclip"] >= 5 > b["dclip"]:
                    f = (a["dclip"] - 5) / (a["dclip"] - b["dclip"])
                    dstar = a["d"] + f * (b["d"] - a["d"])
                    break
            ds_txt = f"{dstar:.4f}" if dstar else "n/a"
            lines.append(f"  {arm:>10} m={m:>6}: sigma={s:.5f}  "
                         f"d*(5%)={ds_txt}  (d*-1)/sigma="
                         f"{((dstar-1)/s if dstar else float('nan')):.2f}")

    worse = [(c["key"],) for c in pooled if c["rmse_clip"] / c["rmse_raw"] > 1.01]
    lines.append("")
    lines.append(f"cells where clip >1% worse than raw: {worse if worse else 'NONE'}")

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, f"{exp}_pooled_ci.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(OUT, f"{exp}_pooled_ci.json"), "w") as f:
        json.dump([{k: v for k, v in c.items()} for c in pooled], f, indent=2)


if __name__ == "__main__":
    main()
