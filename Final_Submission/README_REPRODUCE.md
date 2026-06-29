# Reproducing the Manuscript Results

This document describes how to reproduce the numerical tables and figures in
the current manuscript from the saved experiment logs.  Commands are intended
to be run from the `Final_Submission` directory.

## Environment

The code is Python-based and uses NumPy/SciPy/Matplotlib plus the repository's
local modules in `GPR_KG_Code`.  The manuscript is compiled with `pdflatex`.

The synthetic RZDT experiments in the paper use:

- dimension `d=5`
- heteroscedastic noise scale `sigma=0.04`
- strict chance-constraint threshold `tau=0`
- 10 macro-replications for the main checkpointed GPR-KG/GPR-KG-nV runs
- `N=150`, `n0=30` for the synthetic benchmark budget

The RESCO ingolstadt21 case study uses saved SUMO/RESCO logs and is not meant
to be rerun by the lightweight reproduction commands below.

## Fast Integrity Check

Run the table audit:

```bash
python GPR_KG_Code/experiments/audit_manuscript_tables.py
```

Expected ending:

```text
Audited 193 manuscript values; CHECK rows: 0
```

The audit outputs are:

- `GPR_KG_Code/results/audit/manuscript_table_audit.csv`
- `GPR_KG_Code/results/audit/manuscript_table_audit.json`

These files map manuscript values to saved JSON/CSV result sources.

## Regenerate Manuscript Figures

Run:

```bash
python GPR_KG_Code/experiments/plot_checkpointed_rzdt_figures.py
```

This regenerates:

- `GPR_KG_Code/results/figures/fig_rzdt_gprkg_recommendation.{pdf,png}`
- `GPR_KG_Code/results/figures/fig2_hv_convergence.{pdf,png}`
- `GPR_KG_Code/results/figures/fig3_rmse.{pdf,png}`
- `GPR_KG_Code/results/figures/fig4_infeas.{pdf,png}`
- `GPR_KG_Code/results/figures/fig5_vepm_ablation.{pdf,png}`
- `GPR_KG_Code/results/figures/fig5a_variance.{pdf,png}`

## Compile the Manuscript

Run twice after table/figure changes:

```bash
pdflatex -interaction=nonstopmode -halt-on-error Final_Revised_Manuscript_OR.tex
pdflatex -interaction=nonstopmode -halt-on-error Final_Revised_Manuscript_OR.tex
```

Check that the final log does not contain undefined references or citation
warnings.

## Saved Result Sources Used by the Paper

The current manuscript uses the following saved result directories:

- `server311_checkpointed_full_20260519`
  - checkpointed GPR-KG results
  - includes per-run `result.json`, `run_meta.json`, `checkpoint.pkl`, and
    `iteration_snapshots.jsonl`
  - includes offline final-recommendation summaries in
    `postprocessing_compact`
- `server311_nv_full_20260520`
  - checkpointed GPR-KG-nV results
  - same checkpoint and final-recommendation structure as above
- `GPR_KG_Code/results/d5_v2`
  - saved comparison results for RZDT1/RZDT2 baselines
- `GPR_KG_Code/results/rzdt5rr`
  - saved comparison results for RZDT5_RR baselines
- `GPR_KG_Code/results/sec66/summary.json`
  - RZDT1 configuration ablation table
- `GPR_KG_Code/results/ingolstadt21`
  - RESCO ingolstadt21 heteroscedasticity and case-study summaries

The full source mapping is recorded in:

- `GPR_KG_Code/results/REPRODUCIBILITY_MANIFEST.md`

## Re-running Full Synthetic Experiments

The full synthetic runs are computationally heavier than the audit and figure
steps.  The checkpointed runner is:

```bash
python GPR_KG_Code/experiments/run_rzdt_checkpointed.py --help
```

The important reproducibility features are:

- per-run `checkpoint.pkl`
- per-iteration `iteration_snapshots.jsonl`
- final `result.json`
- `run_meta.json` recording run configuration
- resumable output directories

For a publication artifact, the saved logs above are the exact logs used by
the current manuscript; rerunning may produce statistically comparable but not
bitwise-identical results unless every software dependency and random seed is
held fixed.

## Files Not Used for Manuscript Reproduction

Several root-level files are server-deployment scratch files created while
launching or diagnosing remote runs. They are not required for manuscript
reproduction and should not be included in the archival reproducibility package
unless a separate operational log is desired. See
`REPRODUCIBILITY_PACKAGE_FILES.md` for the recommended include/exclude list.
