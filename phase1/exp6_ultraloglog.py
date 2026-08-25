#!/usr/bin/env python3
"""UltraLogLog, the current
state-of-the-art sketch (Ertl, VLDB 2024), via Dynatrace's reference Java
implementation hash4j (no Python bindings exist). Mirrors exp3's protocol exactly
(n in {2k,8k,32k}, d in {1.0,1.02,1.1,2.0}, p=10 -> m=1024 for apples-to-apples
with the datasketches HLL/CPC arms, calibrated-sigma TPM) so the four modern
estimators line up in one table.

The Java driver (ull/UllDriver.java) emits RAW estimates per (n) cell using real
wyhash hashing; duplicates never change the sketch so L enters only the Python-side
estimators (same exactness argument as the other experiments). Raw estimates at a
given n are reused across d (the sketch is independent of d); only L = round(n*d)
changes. TPM sigma is calibrated per n on a disjoint key range (separate offset).
"""

import argparse
import json
import os
import shutil
import subprocess

import numpy as np

from hll_common import cell_metrics, truncated_posterior_mean

P = 10
NS = [2000, 8000, 32000]
DS_GRID = [1.0, 1.02, 1.1, 2.0]
ARM = "ull_fgra"
CAL_T = 300
CAL_OFFSET = 900_000_000_000
SEED_OFFSETS = {20260612: 1_000_000_000, 987654321: 2_000_000_000, 31337: 3_000_000_000}

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE
ULL = os.path.join(HERE, "ull")
def _find_java():
    """The JVM that setup_ull.sh compiled the driver with (recorded in ull/.java_path);
    else the portable JDK, $JAVA_HOME, or the first java on PATH."""
    rec = os.path.join(ULL, ".java_path")
    if os.path.exists(rec):
        p = open(rec).read().strip()
        if os.access(p, os.X_OK):
            return p
    cands = [os.path.join(ULL, "jdk", "bin", "java")]
    if os.environ.get("JAVA_HOME"):
        cands.append(os.path.join(os.environ["JAVA_HOME"], "bin", "java"))
    for c in cands:
        if os.access(c, os.X_OK):
            return c
    found = shutil.which("java")
    if found:
        return found
    raise SystemExit("exp6: no java found -- run `bash phase1/ull/setup_ull.sh` or install a JDK 17+")

JAVA = _find_java()
CP = f"{os.path.join(ULL, 'hash4j-0.30.0.jar')}:{ULL}"
CAL_FILE = os.path.join(OUT, "exp6_calibration.json")


def run_driver(T, offset, ns):
    """Return {n: np.array(T raw estimates)}."""
    nlist = ",".join(str(n) for n in ns)
    res = subprocess.run([JAVA, "-cp", CP, "UllDriver", str(P), str(T),
                          str(offset), nlist],
                         capture_output=True, text=True, check=True)
    out = {n: [] for n in ns}
    for line in res.stdout.splitlines():
        n_s, est_s = line.split(",")
        out[int(n_s)].append(float(est_s))
    return {n: np.array(v) for n, v in out.items()}


def calibrate():
    raws = run_driver(CAL_T, CAL_OFFSET, NS)
    cal = {str(n): float(np.std(raws[n] / n, ddof=1)) for n in NS}
    with open(CAL_FILE, "w") as f:
        json.dump({"T": CAL_T, "p": P, "c_sigma": cal}, f, indent=2)
    for n in NS:
        print(f"calibrated ull n={n}: c_sigma={cal[str(n)]:.5f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    if args.calibrate:
        calibrate()
        return

    with open(CAL_FILE) as f:
        cal = json.load(f)["c_sigma"]

    raws = run_driver(args.T, SEED_OFFSETS[args.seed], NS)
    cells = []
    arrays = {}
    for n in NS:
        raw = raws[n]
        for d in DS_GRID:
            L = int(round(n * d))
            clip = np.minimum(raw, L)
            tpm = truncated_posterior_mean(raw, L, cal[str(n)])
            met = cell_metrics(n, L, raw, clip, tpm)
            cell = {"arm": ARM, "n": n, "d": d, "L": L, "T": args.T,
                    "c_sigma": cal[str(n)]}
            cell.update(met)
            cells.append(cell)
            ck = f"{ARM}|n{n}|d{d}"
            arrays[f"{ck}|raw"] = raw
            arrays[f"{ck}|clip"] = clip
            arrays[f"{ck}|tpm"] = tpm

    out = os.path.join(OUT, f"exp6_seed{args.seed}")
    np.savez_compressed(out + ".npz", **arrays)
    with open(out + ".json", "w") as f:
        json.dump({"params": {"p": P, "T": args.T, "seed": args.seed, "ns": NS,
                              "ds": DS_GRID, "sketch": "UltraLogLog(hash4j 0.30.0, FGRA)"},
                   "cells": cells}, f, indent=2)

    for n in NS:
        d1 = next(c for c in cells if c["n"] == n and c["d"] == 1.0)
        g = 100 * (1 - d1["rmse_clip"] / d1["rmse_raw"])
        print(f"ull n={n}: rmse_raw={d1['rmse_raw']:.5f} clip dRMSE@d=1.0={g:.2f}% "
              f"bind={d1['bind_rate']:.2f}", flush=True)


if __name__ == "__main__":
    main()
