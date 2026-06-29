# Revision actions for OR resubmission

This file records the remaining evidence needed before the manuscript can make
submission-quality empirical and policy claims.

## Completed in the current manuscript pass

- Aligned the traffic case study with the saved RESCO ingolstadt21 code and
  logs: fixed route file, no bus/pedestrian routes, no dynamic rerouting,
  SUMO-seed stochasticity, `d=44`, `N=300`, `n0=100`, and `tau=1.0`.
- Removed placeholder case-study figures/tables and replaced them with saved
  10-seed diagnostics from `GPR_KG_Code/results/ingolstadt21`.
- Recast VEPM claims around feature-variance alignment: strong gains on aligned
  RZDT settings, graceful degradation toward pooled variance when alignment
  fails on RESCO ingolstadt21.
- Softened theory and abstract language from unconditional publication-grade
  guarantees to conditional finite-budget and asymptotic statements.
- Fixed missing bibliography aliases and rebuilt the PDF successfully.
- Added a reproducible RESCO ingolstadt21 aggregation script:
  `python -m experiments.ingolstadt21.summarize_case_study`. It reads the
  40 saved optimization logs and regenerates the JSON audit summary and LaTeX
  table fragment used by the manuscript.
- Added the fresh-seed validation runner:
  `python -m experiments.ingolstadt21.validate_oos_feasibility --R 100`.
  A dry run currently finds 28 unique final Pareto candidates across the saved
  case-study logs.
- Fixed LaTeX submission hygiene issues that could affect the generated PDF:
  appendix hyperref anchors are now unique after counter resets, math in PDF
  bookmarks is protected, and the widest equations/tables are constrained to
  the manuscript text block.
- Audited Sections 6 and 7 against saved CSV/JSON results. The reported RZDT1,
  RZDT2, and RZDT5_RR tables match `results/extended_tables`; unsupported
  RZDT4 scalability/constraint-tightness placeholder material was removed;
  the ablation table was replaced with saved RZDT1 `sec66` results; and the
  RESCO ingolstadt21 improvement claims were reframed as sample-log diagnostics
  from `case_study_aggregate_summary.json`.
- Rechecked paired Wilcoxon statements from saved replication JSON files and
  softened unsupported significance claims: RZDT2's GPR-KG vs GPR-KG-nV and
  GPR-KG vs NSGA-II-D gaps are now described as numerical, and RZDT5_RR no
  longer claims significant superiority over all same-budget competitors.

## Required before a strong OR resubmission

1. Run independent out-of-sample validation for each reported RESCO
   ingolstadt21 Pareto point with at least `R_oos=100` fresh SUMO seeds.
   Report empirical `Pr(f3 <= tau)` with binomial confidence intervals. The
   validation script now exists, but it has not been executed because it reruns
   SUMO.
2. Recompute final case-study Pareto fronts using out-of-sample means and feasibility
   probabilities, not only optimization-log sample Pareto diagnostics.
3. Add a sensitivity grid over `(tau, alpha)` for the emission chance
   constraint. At minimum use three `tau` levels around the baseline cap and
   two alpha values such as `0.05` and `0.10`.
4. Add one real-case comparator outside the GPR-KG variance-estimator family if
   wall time allows, preferably random search or NSGA-II with the same total
   SUMO budget and the same feasibility post-processing.
5. Audit all synthetic tables against raw result files and regenerate any
   figures whose captions make stronger claims than the saved numerical
   evidence supports.
6. Finish low-level layout polishing before submission: the blocking
   duplicate-anchor/bookmark warnings are fixed, but several modest overfull
   paragraph warnings remain in long technical passages.

## Can be done without rerunning SUMO

1. Regenerate the current case-study diagnostics:
   `cd GPR_KG_Code && python -m experiments.ingolstadt21.summarize_case_study`.
2. Audit synthetic tables against saved JSON summaries under
   `GPR_KG_Code/results/sec62` through `sec66`.
3. Regenerate figures from saved logs and verify that every caption matches the
   corresponding JSON source.

## Suggested framing

The strongest defensible framing is:

- The algorithmic contribution is a parametric GPR-KG framework with a VEPM
  variance module.
- VEPM is valuable when partition features align with the true variance field.
- The RESCO case study is not evidence of universal VEPM dominance; it is a
  valuable stress test showing alignment failure and graceful degradation.
- Policy claims should remain conditional until the fresh-seed feasibility
  validation is complete.
