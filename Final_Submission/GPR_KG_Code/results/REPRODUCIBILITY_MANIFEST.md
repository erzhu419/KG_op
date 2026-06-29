# Reproducibility Manifest

This file records the result sources used by the current manuscript tables and
figures. Paths are relative to the `Final_Submission` directory.

## Manuscript Table Audit

- Audit script: `GPR_KG_Code/experiments/audit_manuscript_tables.py`
- Audit CSV: `GPR_KG_Code/results/audit/manuscript_table_audit.csv`
- Audit JSON: `GPR_KG_Code/results/audit/manuscript_table_audit.json`
- Current audit status: 193 manuscript table values checked, 0 rows flagged.

Run:

```bash
python GPR_KG_Code/experiments/audit_manuscript_tables.py
```

## Figure Regeneration

- Figure script: `GPR_KG_Code/experiments/plot_checkpointed_rzdt_figures.py`
- Output directory: `GPR_KG_Code/results/figures`
- Regenerated manuscript figures:
  - `fig_rzdt_gprkg_recommendation.{pdf,png}`
  - `fig2_hv_convergence.{pdf,png}`
  - `fig3_rmse.{pdf,png}`
  - `fig4_infeas.{pdf,png}`
  - `fig5_vepm_ablation.{pdf,png}`
  - `fig5a_variance.{pdf,png}`

Run:

```bash
python GPR_KG_Code/experiments/plot_checkpointed_rzdt_figures.py
```

## Synthetic Benchmark Result Sources

### Checkpointed GPR-KG

- Result directory: `server311_checkpointed_full_20260519`
- Summary files:
  - `server311_checkpointed_full_20260519/summary_by_problem.json`
  - `server311_checkpointed_full_20260519/summary_runs.csv`
- Per-run files:
  - `result.json`
  - `run_meta.json`
  - `checkpoint.pkl`
  - `iteration_snapshots.jsonl`
- Final-recommendation source:
  - `server311_checkpointed_full_20260519/postprocessing_compact/recommendation_summary.json`
  - `server311_checkpointed_full_20260519/postprocessing_compact/recommendation_rows.csv`

### Checkpointed GPR-KG-nV

- Result directory: `server311_nv_full_20260520`
- Summary files:
  - `server311_nv_full_20260520/summary_by_problem.json`
  - `server311_nv_full_20260520/summary_runs.csv`
- Per-run files:
  - `result.json`
  - `run_meta.json`
  - `checkpoint.pkl`
  - `iteration_snapshots.jsonl`
- Final-recommendation source:
  - `server311_nv_full_20260520/postprocessing_compact/recommendation_summary.json`
  - `server311_nv_full_20260520/postprocessing_compact/recommendation_rows.csv`

### Baseline and Comparison Algorithms

- RZDT1/RZDT2 comparison rows:
  - `GPR_KG_Code/results/d5_v2/RZDT1`
  - `GPR_KG_Code/results/d5_v2/RZDT2`
- RZDT5_RR comparison rows:
  - `GPR_KG_Code/results/rzdt5rr/RZDT5_RR`
- These directories provide the saved per-replication JSON logs for cEHVI,
  cParEGO, NSGA-II-K, NSGA-II-D, and RS.

## Ablation and Diagnostic Sources

- RZDT1 candidate/VEPM ablation:
  - `GPR_KG_Code/results/sec66/summary.json`
- Alignment diagnostic table:
  - Recomputed analytically from the RZDT noise functions and from the
    ingolstadt21 heteroscedastic pilot. The values are retained in the
    manuscript as diagnostic summaries; the table-audit JSON records the
    associated case-study source files.
- RESCO ingolstadt21 heteroscedasticity diagnostic:
  - `GPR_KG_Code/results/ingolstadt21/hetero_test.json`
  - `GPR_KG_Code/results/ingolstadt21/hetero_test.plans.json`
- RESCO ingolstadt21 algorithmic diagnostics:
  - `GPR_KG_Code/results/ingolstadt21/case_study_aggregate_summary.json`
  - `GPR_KG_Code/results/ingolstadt21/case_study_aggregate_table.tex`
  - `GPR_KG_Code/results/ingolstadt21/vepm_convergence_summary.json`

## Experiment Runner Scripts

- Checkpointed synthetic runs:
  - `GPR_KG_Code/experiments/run_rzdt_checkpointed.py`
- Offline final-recommendation analysis:
  - `GPR_KG_Code/experiments/analyze_final_recommendations.py`
- Legacy same-budget baseline runs:
  - `GPR_KG_Code/experiments/run_d5_v2.py`
  - `GPR_KG_Code/experiments/run_rzdt5rr.py`
- RZDT1 ablation:
  - `GPR_KG_Code/experiments/run_all.py`
  - `GPR_KG_Code/results/sec66/summary.json`

## Notes

- The checkpointed GPR-KG and GPR-KG-nV runs save intermediate iteration
  information through `iteration_snapshots.jsonl` and resumable state through
  `checkpoint.pkl`.
- The offline final-recommendation post-processing uses only saved checkpoints
  and result logs. It does not consume additional simulation replications.
- NSGA-II-D is a 10x simulation-budget reference in the manuscript tables.
