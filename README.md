# Stream-Length Conditioning for Cardinality Sketches — Reproducibility Artifact

Reproducibility package for the PVLDB paper *When Is the Free Counter Worth It? A
Value-Window Study of Stream-Length Conditioning for Cardinality Sketches*
(Experiment, Analysis & Benchmark).

**The result.** HyperLogLog and its descendants estimate distinct counts but ignore `L`,
the number of items inserted, even though the true count `n` can never exceed `L`. Clipping
the estimate at `L` — `min(estimate, L)` — deletes the provably impossible half of the error
on near-duplicate-free streams: a **29.3% RMSE reduction that is mathematically never
worse**. This repository regenerates every number, table, and figure in the paper.

## Requirements

- **Python 3.8–3.11** — the pinned wheels in `requirements.txt` do not cover 3.12+;
  `repro.sh` checks the interpreter and stops otherwise.
- **≥ 2 GB free RAM** and a few CPU cores. No GPU, no API keys.
- **Java** for the UltraLogLog experiment: `phase1/ull/setup_ull.sh` fetches a pinned,
  SHA-256-verified Temurin JDK 17 on Linux/x86-64, or uses a system JDK 17+.
- **Network on first run:** the words corpus, the NASA-HTTP trace, and DuckDB's TPC-H
  extension are downloaded and verified by SHA-256 (a mismatch stops the run).
- **PostgreSQL 16** for the catalogue-injection study (Table 4) is installed in user space
  by `phase2/pg_local_setup.sh` — no root required.

## Reproduce

```bash
pip install -r requirements.txt
bash repro.sh          # phase 1: ~2–2.5 h; regenerates every number, table, and figure except Table 4
```

`bash repro.sh fast` runs the same pipeline with reduced trial counts for exp1–6 (~65 min;
the numbers will not match the paper). Table 4 (the PostgreSQL catalogue-injection study) is
reproduced separately:

```bash
cd phase2
bash pg_local_setup.sh                                # user-space PostgreSQL 16
python3 exp16_catalog_injection.py --analyze --sf 1   # regenerate Table 4 from the shipped plan captures
python3 exp16_catalog_injection.py --analyze --sf 10
```

See `phase2/README.md` for the full capture-and-analyze pipeline and the positive control.

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | `make_figures.py` (regenerates every figure) and the generated `figures/` |
| `phase1/` | experiments `exp1`–`exp15`: five sketch families, three hash functions, the PyPI and NASA-HTTP workloads, pooling, bootstrap CIs, and the closed-form theory checks |
| `phase1/ull/` | the UltraLogLog (Java / `hash4j`) harness and its setup script |
| `phase1/exp{1..6}_seed*.npz` | the per-trial estimate arrays behind the pooled tables and confidence intervals |
| `phase2/` | the PostgreSQL 16 catalogue-injection study (paper §7, Table 4) with its shipped plan captures |
| `results_all.csv` | all 437 pooled result cells in one table |
| `THEORY.md` | the truncated-normal analysis, verified against the measurements |
| `repro.sh` | one command for all of phase 1 |

## Paper → data

Every reported number is traceable to a shipped file: Tables 1–3 and the figures come from
`phase1/*_pooled*.json` and `results_all.csv`; Table 4 from `phase2/exp16_results.json`.
`paper/make_figures.py` regenerates the seven figures from those files. The PyPI namespace
snapshot (827,798 names) is committed at `phase1/realworld_data/pypi_names.txt` so the
reported numbers reproduce exactly.

## Citation and license

Please cite the paper and this artifact — see `CITATION.cff`. Released under CC BY 4.0
(`LICENSE`). Third-party inputs (the NASA-HTTP trace, the JDK, the PostgreSQL build) are
fetched by checksum rather than redistributed.
