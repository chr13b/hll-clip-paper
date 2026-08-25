#!/usr/bin/env bash
# One-command reproduction of all phase-1 results with pinned seeds (Table 4 / phase 2 is
# separate, see phase2/README.md). Runtime is hours, not minutes, at full T on a laptop
# (exp1-6 ~1.5 h extrapolated from measured smoke timings; exp13-15, TPC-H in DuckDB, are the
# long tail and run at full size in BOTH modes). Needs CPython 3.8-3.11 and >= 2 GB free RAM
# (measured peak RSS on a fresh clone: exp14 1.05 GB, exp13 0.84 GB). No GPU, no API keys.
# exp6 (UltraLogLog) additionally downloads a pinned, checksummed portable JDK (~190 MB) +
# the checksummed hash4j jar; if that setup fails (offline, etc.) exp6 is skipped.
# Every dataset is verified by SHA-256; a mismatch stops the run unless ALLOW_DATA_MISMATCH=1.
#
#   bash repro.sh            # run everything (exp1-15 + theory + CSV + figures)
#   bash repro.sh fast       # smoke (smaller T for exp1-6, sandboxed to phase1/_smoke/; ~65 min measured end to end)
#
# Outputs (all under phase1/ unless noted): exp{1..6}_seed*.{json,npz},
# pooled tables exp{1,2}_pooled.txt and exp{3,4,5,6}_pooled_ci.txt, theory check
# stdout, and results_all.csv (repo root).

set -euo pipefail
cd "$(dirname "$0")/phase1"

SEEDS=(20260612 987654321 31337)
if [[ "${1:-}" == "fast" ]]; then
  T1=100; T2=80; T3=80; T4=100; T5=60
  export HLL_OUT="$PWD/_smoke"   # smoke outputs are sandboxed; canonical results untouched
  mkdir -p "$HLL_OUT"
  echo ">>> FAST smoke mode (reduced T for exp1-6, outputs in phase1/_smoke/; numbers will NOT match the paper; exp9-15 still run at full size)"
else
  T1=1000; T2=500; T3=500; T4=600; T5=300  # exp5 shipped at T=300/seed
fi
RES="${HLL_OUT:-$PWD}"   # where pooled tables land

echo ">>> [0/6] dependency check"
python3 - <<'PY'
import sys
if not ((3, 8) <= sys.version_info[:2] <= (3, 11)):
    sys.exit(f"  !! Python {sys.version.split()[0]}: the pinned wheels in requirements.txt exist for CPython 3.8-3.11 only (scipy 1.10 requires <3.12). Use a 3.8-3.11 interpreter.")
from importlib.metadata import version
import numpy, scipy, xxhash, mmh3, datasketch, datasketches
print("  python", sys.version.split()[0], "| numpy", numpy.__version__, "| scipy", scipy.__version__,
      "| xxhash", xxhash.VERSION, "| mmh3 ok | datasketch", datasketch.__version__,
      "| datasketches", version("datasketches"))
PY

echo ">>> data corpus"
if [[ ! -f data/words_alpha.txt ]]; then
  mkdir -p data
  curl -fsSL https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt \
    -o data/words_alpha.txt
fi
echo "  words_alpha.txt: $(wc -l < data/words_alpha.txt) lines"
WORDS_SHA=3ed0c94610d8bcf7c11bbb49c56aa49c7234d32b66824df91f554169e572da48
if echo "$WORDS_SHA  data/words_alpha.txt" | sha256sum -c --quiet - 2>/dev/null; then
  echo "  words_alpha.txt matches the pinned corpus (370,105 words)"
elif [[ "${ALLOW_DATA_MISMATCH:-0}" == 1 ]]; then
  echo "  !! words_alpha.txt differs from the pinned corpus -- continuing (ALLOW_DATA_MISMATCH=1); exp1 word/URL arms may differ from the paper"
else
  echo "  !! words_alpha.txt differs from the pinned corpus (SHA-256 mismatch). Delete data/words_alpha.txt to re-download, or set ALLOW_DATA_MISMATCH=1 to run anyway."; exit 1
fi
if [[ ! -f realworld_data/pypi_names.txt ]]; then
  mkdir -p realworld_data
  echo "  downloading PyPI simple index (real-world dataset for exp5) ..."
  curl -fsSL --max-time 180 -H "Accept: text/html" https://pypi.org/simple/ \
    -o realworld_data/pypi_simple.html
  python3 - <<'PY'
import re
html = open('realworld_data/pypi_simple.html', encoding='utf-8', errors='replace').read()
names = [n.strip() for n in re.findall(r'<a[^>]*>([^<]+)</a>', html) if n.strip()]
open('realworld_data/pypi_names.txt','w').write('\n'.join(names))
print(f'  pypi_names.txt: {len(names)} package names')
PY
fi

# verify the pinned real-world snapshot (paper numbers depend on it)
PYPI_SHA=63b21acdd134a72fdddea00860be66f120f8f0fb74db138474542d2536adb917
if echo "$PYPI_SHA  realworld_data/pypi_names.txt" | sha256sum -c --quiet - 2>/dev/null; then
  echo "  pypi_names.txt matches the paper's pinned 2026-06-13 snapshot (827,798 names)"
elif [[ "${ALLOW_DATA_MISMATCH:-0}" == 1 ]]; then
  echo "  !! pypi_names.txt differs from the pinned snapshot -- continuing (ALLOW_DATA_MISMATCH=1); real-world numbers may differ from the paper"
else
  echo "  !! pypi_names.txt differs from the pinned snapshot (SHA-256 mismatch). Restore the committed file (git checkout -- phase1/realworld_data/pypi_names.txt), or set ALLOW_DATA_MISMATCH=1 to run anyway."; exit 1
fi

echo ">>> [1/6] exp1 real-hash survival (classic FFGM, 10 arms)"
for s in "${SEEDS[@]}"; do python3 exp1_real_hash.py --seed "$s" --T "$T1"; done

echo ">>> [2/6] exp2 HLL++ value-window map"
for s in "${SEEDS[@]}"; do python3 exp2_hllpp_window.py --seed "$s" --T "$T2"; done

echo ">>> [3/6] exp3 modern sketches (HIP + CPC, calibrated TPM)"
python3 exp3_modern_sketches.py --calibrate
for s in "${SEEDS[@]}"; do python3 exp3_modern_sketches.py --seed "$s" --T "$T3"; done

echo ">>> [4/6] exp4 m-scaling + fine d-sweep"
for s in "${SEEDS[@]}"; do python3 exp4_sweep.py --seed "$s" --T "$T4"; done

echo ">>> [5a] exp5 real-world (PyPI namespace, production DataSketches HLL)"
python3 exp5_realworld.py --calibrate
for s in "${SEEDS[@]}"; do python3 exp5_realworld.py --seed "$s" --T "$T5"; done

echo ">>> [5b] exp6 UltraLogLog (hash4j; downloads portable JDK if needed)"
if bash ull/setup_ull.sh; then
  python3 exp6_ultraloglog.py --calibrate
  for s in "${SEEDS[@]}"; do python3 exp6_ultraloglog.py --seed "$s" --T "$T3"; done
  EXP6_OK=1
else
  echo "  !! ULL setup failed (offline / no JDK) -- skipping exp6; rest of run is unaffected"
  EXP6_OK=0
fi

echo ">>> [6a] pooling (paired analysis + bootstrap CIs)"
python3 pool.py exp1 "${SEEDS[@]}"
python3 pool.py exp2 "${SEEDS[@]}"
python3 ci.py   exp3 "${SEEDS[@]}"
python3 ci.py   exp4 "${SEEDS[@]}"
python3 ci.py   exp5 "${SEEDS[@]}"
[[ "$EXP6_OK" == 1 ]] && python3 ci.py exp6 "${SEEDS[@]}"

echo ">>> [6b] theory verification (closed forms + second-order skew model)"
python3 theory_check.py
python3 skew_theory.py

echo ">>> [6c] real-world trace + competitor + cost (optional, need datasets/network)"
if [[ -f realworld_data/nasa_jul95 ]] || curl -sfL --max-time 300 -o realworld_data/nasa_jul95.gz \
     "https://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz" 2>/dev/null; then
  [[ -f realworld_data/nasa_jul95 ]] || gunzip -kf realworld_data/nasa_jul95.gz
  if echo "96551161b5bdcaacbc3c17fa108191c478fb35dfe87895c16e34a8f6552bf29a  realworld_data/nasa_jul95" \
       | sha256sum -c --quiet - 2>/dev/null; then
    echo "  nasa_jul95 matches the paper's copy"
  elif [[ "${ALLOW_DATA_MISMATCH:-0}" == 1 ]]; then
    echo "  !! NASA trace checksum differs from the paper's copy -- continuing (ALLOW_DATA_MISMATCH=1)"
  else
    echo "  !! NASA trace checksum differs from the paper's copy (SHA-256 mismatch). Delete realworld_data/nasa_jul95* to re-download, or set ALLOW_DATA_MISMATCH=1."; exit 1
  fi
  python3 exp7_nasa.py
else
  echo "  !! NASA trace unavailable -- skipping exp7"
fi
python3 exp9_competitors.py
python3 exp10_refined_clip.py
python3 exp11_matched_memory.py
python3 exp12_merge_union.py
python3 exp13_ndv_columns.py   # NDV per column (TPC-H sf=0.5 in a memory-bounded DuckDB, peak ~0.85 GB RSS; fetches DuckDB's tpch extension on first use)
python3 exp14_join_cardinality.py  # join-cardinality propagation (TPC-H sf=0.3 in DuckDB)
python3 exp15_ndv_sweep.py         # real key values carried through the value window
python3 exp8_microbench.py

echo ">>> [6d] aggregate results_all.csv"
python3 make_csv.py

echo ">>> [6e] regenerate paper figures from the produced data"
python3 ../paper/make_figures.py

echo ">>> DONE. Headline verdicts:"
grep -h "POOLED VERDICT\|d=1.0 avg clip" "$RES/exp1_pooled.txt" | sed 's/^/    /'
echo "    (full numbers: $RES/exp{1,2}_pooled.txt, $RES/exp{3,4,5,6}_pooled_ci.txt, results_all.csv)"

echo ">>> phase 2 (PostgreSQL catalogue injection, Table 4 of the paper) is reproduced separately: see phase2/README.md"
