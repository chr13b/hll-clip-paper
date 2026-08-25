# Phase 2 — catalogue injection into PostgreSQL

**Question.** Does a corrected distinct-count (NDV) statistic change what a real query
optimizer does? This is the study behind the paragraph "Does it change what the database
does?" and Table 4 of the paper.

**Method.** TPC-H statistics are built once with a plain `ANALYZE`. Each condition then
overwrites only the per-column distinct count (`pg_statistic.stadistinct`, the field the
supported `n_distinct` override populates) with one of four catalogues — `exact`, `raw`
(DataSketches HLL, lg_k = 14), `clipped` = min(raw, rows), `clip_ri` = min(raw, rows of the
table referenced by a declared foreign key) — so the catalogues differ in nothing else.
For every hash draw and condition the 22 TPC-H queries are `EXPLAIN`ed, plans are
canonicalised (join order, physical join operator, scan type), per-node Q-error is taken
from `EXPLAIN (ANALYZE)`, and runtime is measured only where a plan changed. PostgreSQL's
own sampled statistic (`pg_default`) is captured on identical histograms as the positive
control. Injecting the exact catalogue twice must reproduce every plan bit for bit; the
scripts check this before reporting any plan-change count.

**Requirements.** PostgreSQL 16 (`pg_local_setup.sh` fetches a user-space 16.14 build, no
root needed), `psycopg2-binary`, `duckdb` (TPC-H generation), and about 8 GB of RAM for
scale factor 10.

```bash
bash phase2/pg_local_setup.sh                          # user-space PostgreSQL 16
python3 phase2/tpch_load.py --sf 1                     # dbgen via DuckDB -> COPY; extracts the 22 queries
python3 phase2/exp16_catalog_injection.py --check-only # five feasibility probes (GO / NO-GO)
python3 phase2/exp16_catalog_injection.py --ndv     --sf 1 --draws 20
python3 phase2/exp16_catalog_injection.py --capture --sf 1 --analyze-draws 10   # EXPLAIN ANALYZE on draws 0-9
python3 phase2/exp16_catalog_injection.py --analyze --sf 1
python3 phase2/exp16_catalog_injection.py --runtime --sf 1
# scale factor 10: same steps with --sf 10 --draws 10 --analyze-draws 0 (plans only; hours, not minutes).
# Q-error is an sf=1 measurement. The Q20 control timing (exact vs pg_default, 220 s vs 694 s) is a
# single paired run with a warm cache, recorded in q20_confirmatory.json; the scripted sf=10 suite
# run of pg_default hit the harness's 600 s statement timeout, which is why it was timed by hand.
```

**Outputs.** `exp16_ndv_sf{1,10}.json` (per-column NDV under each catalogue),
`capture_sf1/` (canonical plans and node estimates per condition and draw, gzipped), and
`exp16_results.json`, which holds every number the paper quotes: the exact-vs-exact gate,
plan-change counts per condition, pairwise estimate differences, Q-error distributions,
the changed-plan runtimes, and the `pg_default` control.
