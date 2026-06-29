# Recommended Reproducibility Package Boundary

This file separates files that should be included in a clean journal
reproducibility archive from temporary server-launch and debugging artifacts.

## Include

### Manuscript

- `Final_Revised_Manuscript_OR.tex`
- `Final_Revised_Manuscript_OR.pdf`
- `references.bib`
- `Final_Revised_Manuscript_OR.bbl`
- `README_REPRODUCE.md`

### Core Code

- `GPR_KG_Code/gpr_kg.py`
- `GPR_KG_Code/metrics.py`
- `GPR_KG_Code/methods/`
- `GPR_KG_Code/experiments/run_rzdt_checkpointed.py`
- `GPR_KG_Code/experiments/analyze_final_recommendations.py`
- `GPR_KG_Code/experiments/plot_checkpointed_rzdt_figures.py`
- `GPR_KG_Code/experiments/audit_manuscript_tables.py`
- `GPR_KG_Code/experiments/run_d5_v2.py`
- `GPR_KG_Code/experiments/run_rzdt5rr.py`
- `GPR_KG_Code/experiments/run_all.py`
- `GPR_KG_Code/experiments/build_extended_tables.py`

### Manuscript Figures

- `GPR_KG_Code/results/figures/fig_rzdt_gprkg_recommendation.pdf`
- `GPR_KG_Code/results/figures/fig2_hv_convergence.pdf`
- `GPR_KG_Code/results/figures/fig3_rmse.pdf`
- `GPR_KG_Code/results/figures/fig4_infeas.pdf`
- `GPR_KG_Code/results/figures/fig5_vepm_ablation.pdf`
- `GPR_KG_Code/results/figures/fig5a_variance.pdf`
- `GPR_KG_Code/results/figures/figH_hetero_ingolstadt21.pdf`
- PNG counterparts may be included for visual inspection but are not required
  by the LaTeX manuscript.

### Synthetic Results Entering the Manuscript

- `server311_checkpointed_full_20260519/`
- `server311_nv_full_20260520/`
- `GPR_KG_Code/results/d5_v2/RZDT1/`
- `GPR_KG_Code/results/d5_v2/RZDT2/`
- `GPR_KG_Code/results/rzdt5rr/RZDT5_RR/`
- `GPR_KG_Code/results/sec66/summary.json`
- `GPR_KG_Code/results/audit/`
- `GPR_KG_Code/results/REPRODUCIBILITY_MANIFEST.md`

### RESCO ingolstadt21 Results Entering the Manuscript

- `GPR_KG_Code/results/ingolstadt21/hetero_test.json`
- `GPR_KG_Code/results/ingolstadt21/hetero_test.plans.json`
- `GPR_KG_Code/results/ingolstadt21/case_study_aggregate_summary.json`
- `GPR_KG_Code/results/ingolstadt21/case_study_aggregate_table.tex`
- `GPR_KG_Code/results/ingolstadt21/vepm_convergence_summary.json`
- `GPR_KG_Code/results/ingolstadt21/baseline.json`

## Exclude From Clean Journal Archive

These files/directories are useful operational scratch materials but are not
needed to reproduce the manuscript tables and figures.

### Remote-Server Launch and Debug Scripts

- `cpu2_deploy.py`
- `cpu2_fetch_updated_results.py`
- `cpu2_keepalive_run.py`
- `cpu2_run_theory_compat_pilot.py`
- `server_check_updated_rzdt_params.py`
- `server_diag_updated_rzdt.ps1`
- `server_launch_updated_rzdt*.ps1`
- `server_run_targeted_pilot.py`
- `server_run_targeted_pilot_wait.ps1`
- `server_run_theory_compat_pilot.py`
- `server_run_theory_compat_pilot_wait.ps1`
- `server_run_updated_rzdt_wait.ps1`
- `server_test_*.ps1`
- `local_launch_server3112080*.ps1`
- `local_keepalive_server311_checkpointed_full.ps1`
- `server311_launch_*.py`
- `server311_launch_*.ps1`
- `server311_register_*.ps1`
- `server311_run_*.bat`
- `server311_run_theory_compat_pilot.py`

### Pilot, Targeted, and Diagnostic Result Directories

- `server311_pilot_results/`
- `server311_targeted_results/`
- `cpu2_updated_results_20260518_120624/`
- `GPR_KG_Code/results/local_validation/`

### Local Logs and Generated Build Byproducts

- `*.aux`
- `*.log`
- `*.out`
- `*.blg`
- `__pycache__/`
- `GPR_KG_Code/__pycache__/`
- root-level `*.err.log`
- root-level `*.out.log`

## Commit Recommendation

For the next clean commit, include:

- manuscript `.tex` and `.pdf`
- regenerated figure files used by the manuscript
- `GPR_KG_Code/experiments/audit_manuscript_tables.py`
- `GPR_KG_Code/experiments/plot_checkpointed_rzdt_figures.py`
- `GPR_KG_Code/results/audit/`
- `GPR_KG_Code/results/REPRODUCIBILITY_MANIFEST.md`
- `README_REPRODUCE.md`
- `REPRODUCIBILITY_PACKAGE_FILES.md`

Do not include the excluded scratch scripts unless the goal is to preserve the
remote execution history rather than prepare a clean journal archive.
