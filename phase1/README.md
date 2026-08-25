# Experiments: reproduction and per-experiment notes

One-command reproduction (from repo root):

```bash
pip install -r requirements.txt
bash repro.sh          # ~45-60 min, laptop, single thread, no GPU/API
bash repro.sh fast     # ~8 min smoke test (reduced T; numbers won't match paper)
```

## What the clip is

A cardinality sketch (HyperLogLog and relatives) returns an estimate `n_hat` of the
number of distinct items. Every system also knows `L`, the total number of items
inserted (a counter), and `n <= L` always. The **clip** is `min(n_hat, L)`; the
**truncated-posterior mean (TPM)** uses `L` as a soft prior instead of a hard wall.

## Experiments (each: 3 seeds, paired analysis)

| script | question | grid |
|---|---|---|
| `exp1_real_hash.py` | does the gain survive real hashes/keys? | classic FFGM, m=1024, {words,urls,ints}×{xxh64,mmh3,sha256}+ideal, n∈{2k,8k,32k}, d∈{1.0,1.02,1.1,2.0} |
| `exp2_hllpp_window.py` | gain vs strongest baseline (HLL++) + value window | HLL++ dense (Heule et al. bias tables), m∈{1024,4096,16384}, n/m∈{0.05…32}, d∈{1.0,1.02,1.1} |
| `exp3_modern_sketches.py` | gain on low-error production sketches? | Apache DataSketches HIP/martingale + CPC, calibrated TPM |
| `exp4_sweep.py` | decay curves, m-scaling, scaling collapse | HLL++, m∈{256…16384}, n=8m, fine d-grid [1.0,1.20] |
| `exp5_realworld.py` | does it matter on a real workload? | PyPI namespace (827,798 names) → production DataSketches HLL lg_k=14, internal hashing |
| `exp6_ultraloglog.py` | does it persist on the SOTA sketch? | UltraLogLog via hash4j 0.30.0 (Java; `ull/` harness), FGRA estimator, calibrated TPM |

Pooling/analysis: `pool.py` (paired z), `ci.py` (paired bootstrap 95% CIs),
`theory_check.py` (closed forms vs data — see `../THEORY.md`), `make_csv.py`
(→ `../results_all.csv`). Shared machinery in `hll_common.py`. The UltraLogLog harness
(`ull/UllDriver.java`, `ull/setup_ull.sh`) needs a JDK; `setup_ull.sh` fetches a
portable one if none is present.

## Headline results (3 seeds, pooled)

- **Survives real hashing:** every (key,hash) arm 26–36% RMSE reduction at d=1.0,
  no arm below 18%, no cell worse than raw.
- **Survives HLL++ and all modern sketches:** ~26–31% at d=1.0 on HLL++, HIP, CPC, and
  UltraLogLog — invariant to base error level across five estimator families.
- **Real-world (PyPI, 827,798 names):** the production HLL overestimates by +0.66%;
  the clip corrects to +0.00%. 24–27% RMSE reduction at d=1.0 on real keys.
- **Never-worse:** 0 / 437 pooled cells where clip is >1% worse than raw.
- **Theory matches:** measured 30.0% vs predicted `1 - 1/sqrt(2)` = 29.29%; the whole
  decay surface collapses onto the closed-form curve `1 - sqrt(Phi(a)-a phi(a)+a^2
  Phibar(a))`, `a=(d-1)/sigma`, RMS deviation 1.96 pp.
- **Value window:** clip worth ≥5% RMSE while duplicate fraction `(d-1) ≲ 1.6/sqrt(m)`
  — confirmed on the real PyPI workload (narrow window at lg_k=14, as predicted).
- **Calibrated TPM** dominates the hard clip in the lightly-duplicated window
  (20–25% vs 5–17% at d=1.02 on HIP/CPC; 14–16% vs 2% at d=1.01 on real PyPI data).

## Known scope limits (honest)

- Estimator families covered: classic FFGM, HLL++, HIP/martingale, CPC, UltraLogLog
  (the SOTA sketch). Theta/CPC-of-other-libs and ExaLogLog left as future work.
- Duplicates enter only through `L` (exact, since the sketches are idempotent);
  no adversarial duplicate placement tested.
- TPM requires a calibrated noise model; the hard clip is parameter-free.
