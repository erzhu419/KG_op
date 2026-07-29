# Revised Paper Evidence and Artifact Plan

## What to retain from the legacy manuscript

The legacy manuscript used a strong visual grammar: one complete main table,
paired budget-convergence curves, calibration/feasibility diagnostics,
component ablations, and a separate traffic case study. The revision retains
that evidence order while replacing hypervolume-only and saved-log claims with
single-objective chance-constrained metrics and fresh out-of-sample audits.

## Main-text artifacts

1. **Method diagram.** One source-to-target object: source-only
   dimension-equivariant structural proposal, a replaceable online backend
   (canonical SAASBO in the headline implementation), and a separate
   conservative verifier. Cumulative HVD is shown as an optional mechanistic
   risk head rather than a required source of optimization gain.
2. **Main comparison table.** True-feasible count, feasible regret, certificate
   coverage, false certificates, and wall time. Every regret is accompanied by
   its feasibility denominator.
3. **Dimension/budget frontier.** True-feasible rate and median feasible regret
   against `d/N` for `d=200,1000`, with `d=10000` admitted only after the
   registered `d=1000` gate.
4. **Target convergence.** Incumbent feasible rate and incumbent feasible
   regret against charged target calls. Synthetic truth is joined only after
   the complete decision sequence is frozen.
5. **Proposal-to-adaptation table.** Initial feasible count, final feasible
   count, rescue/loss, and median `n0-best -> final` regret change. This
   separates the front end from the online optimizer.
6. **HVD causal audit.** Log-variance RMSE, variance-shape correlation,
   variance-upper coverage, feasibility, regret, and false certification for
   pooled versus cumulative factor models under the same proposal, backend,
   and verifier. The five-seed gate is a pilot only; the paper decision uses
   the separately frozen 20-seed replay.
7. **Evaluate-or-replicate allocation.** Replication action fraction over
   target budget by domain, paired with the HVD calibration result.
8. **Certification budget.** Certified/evaluated coverage and false
   certification over target and replication budgets; a vacuous certificate
   is reported as a failed empirical obligation, not hidden.

## Supplement and case-study artifacts

- independently retrained structural-prior causal matrix;
- transfer baselines with the identical frozen 384-call source archive and
  paired target initial design;
- target-only SOTA at both `N` and total-cost `384+N` budgets;
- legacy RZDT scalarized bridge under the original problem definitions;
- strict no-history SUMO using fresh trajectory CSV and fresh out-of-sample
  certification seeds, with descriptor-conditioned transfer as the main
  external-validity track and domain-blind transfer as a negative control;
- manifold/SSL/Transformer representation rows as background ablations only.

## Statistical contract

- at least 20 paired seeds for primary synthetic cells;
- failures/timeouts remain in the denominator;
- lexicographic paired comparison: feasibility first, regret second;
- bootstrap 95% intervals and Holm-corrected paired tests;
- report source calls, target calls, and total calls separately;
- report conditional regret only beside feasibility and failure-aware regret;
- no checkpoint, pickle, model weight, or raw scheduler profile is needed for
  figure generation.

`performance/export_paper_audit_render_inputs.py` emits the final `rows.csv`
and `grouped_summary.csv` only from tracks that passed the frozen experiment
registry audit; failures and timeouts remain in those files.
`performance/paper_convergence_extract.py` emits the post-run joined
`traces.csv`. `performance/render_paper_artifacts.py` verifies the audit-export
manifest and embeds the already preregistered paired-statistics artifact
without recomputing a competing test family before producing LaTeX tables,
PDF/PNG figures, and a hash-addressed artifact manifest. The generic
`aggregate_completed_matrix.py` remains useful for exploratory diagnostics but
is not admissible for the final release. Figure exports include editable-text
PDF/SVG plus 300-dpi PNG and 600-dpi TIFF from the same Python backend.
