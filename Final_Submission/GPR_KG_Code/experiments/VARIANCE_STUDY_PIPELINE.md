# Variance-Study Revision and Execution Manual

This manual defines the next revision plan and the unified execution
pipeline for the OR submission. It is intentionally written in ASCII to
avoid Windows console and Git encoding ambiguity. The scientific content
should be used as the source of truth for future code, experiment, and
manuscript updates.

Core claim after revision:

> VEPM is a conditional variance-calibration module. It is valuable when
> the variance structure is aligned with decision features that shape the
> chance-constrained feasible Pareto boundary. In low-sensitivity or
> misaligned regimes, it can reduce to pooled-variance-like behavior.

All future benchmark additions, reruns, result analyses, and manuscript
updates should follow this manual.

## 1. Revision Goal

The revised paper should build a three-layer evidence chain.

1. Original RZDT suite

   Keep `RZDT1`, `RZDT2`, and `RZDT5_RR` as moderate or realistic
   heteroscedasticity tests. These tests should support moderate gains,
   comparable behavior, or conditional advantages, not universal dominance.

2. Variance-critical suite

   Add variance-critical benchmarks where the feasible Pareto boundary
   crosses both low-noise and high-noise regions. These problems should make
   pooled or no-variance baselines prone to chance-constraint
   misclassification.

3. Calibration diagnostics

   Report variance and feasibility calibration metrics in addition to
   `HV`, `IGD`, `CVR`, and `ND`. The goal is to show that performance gains,
   when present, are driven by better variance and feasibility modeling.

## 2. Manuscript Revision Policy

### 2.1 Claim Adjustment

All VEPM-related claims should follow these rules.

- Do not claim that VEPM universally and significantly dominates pooled or
  nV variance.
- State the effective regime: variance-feature alignment, sufficiently
  sampled partition cells, and variance-critical chance constraints.
- State the degradation regime: low variance sensitivity, misaligned
  variance structure, sparse high-dimensional partitions, or overly
  conservative pooled baselines.
- Use the original RZDT suite as the moderate heteroscedasticity evidence.
- Use the variance-critical suite as the main evidence for heteroscedastic
  modeling value.
- Use the oracle-variance experiment to separate benchmark sensitivity from
  variance-estimator limitation.

### 2.2 Sections to Update

After the new experiments are complete, update:

- Abstract: soften universal superiority language and emphasize calibrated
  heteroscedastic treatment.
- Introduction: frame contributions around variance-aware learning,
  variance calibration, and diagnostic methodology.
- Algorithm section: describe VEPM, pooled variance, oracle variance, and
  guarded variance surrogate as implementations of one variance-provider
  interface.
- Theory section: keep the finite-candidate, consistent-variance,
  sufficient-sampling, and partition-alignment conditions explicit. Explain
  low-dimensional or adaptive partitioning as a theory-compatible
  engineering realization.
- Numerical experiments: reorganize around original suite,
  variance-critical suite, oracle-gap diagnostics, calibration diagnostics,
  and sensitivity analysis.
- Conclusion: state effective conditions, failure modes, and engineering
  guidance.

## 3. Method Taxonomy

All GPR-KG variants should be controlled by a unified `variance_mode`.

| Method label | variance_mode | Purpose | Paper status |
| --- | --- | --- | --- |
| `GPR-KG-VEPM` | `vepm` | Main algorithm with VEPM variance estimates | Main |
| `GPR-KG-AVB` | `vepm` plus adaptive VEPM and additive boundary-aware acquisition | Exploratory; too aggressive in the first RCZDT pilot |
| `GPR-KG-AVB-lite` | `vepm` plus adaptive VEPM, shrinkage, and conservative boundary-candidate augmentation | Candidate upgraded main algorithm |
| `GPR-KG-AVB-safe` | `AVB-lite` plus a posterior-feasibility safety buffer for boundary candidates | Conservative CVR-control pilot |
| `GPR-KG-VEPM-robust` | `vepm` plus robust update flags | Finite-budget guard against one-shot residual contamination | Candidate main or sensitivity |
| `GPR-KG-pooled-pre` | `pooled_pre` | Pooled variance estimated from initial or pre-sample residuals | Main baseline |
| `GPR-KG-pooled-online` | `pooled_online` | Online pooled residual variance | Appendix |
| `GPR-KG-fixed-safe` | `fixed_safe` | Historical fixed safe variance, e.g. 0.01 | Appendix only |
| `GPR-KG-oracleV` | `oracle` | True variance, upper-bound diagnostic | Main diagnostic |
| `GPR-KG-HVS-guarded` | `hvs_guarded` | Guarded log-variance surrogate | Appendix or exploratory |
| `cEHVI` | baseline | Same-budget surrogate baseline | Main baseline |
| `cParEGO` | baseline | Same-budget surrogate baseline | Main baseline |
| `NSGA-II-K` | baseline | Surrogate-assisted evolutionary baseline | Main baseline |
| `NSGA-II-D` | baseline | Direct-evaluation evolutionary diagnostic | Appendix/context |
| `RS` | baseline | Random-search lower benchmark | Appendix/context |

Important: the main pooled/nV baseline should not use fixed `0.01` as the
default variance. Fixed `0.01` is a historical conservative baseline and
belongs in sensitivity or appendix results only.

`GPR-KG-AVB-lite` is the current preferred upgraded algorithm to test before
rewriting the manuscript as a new main method.  It keeps the same GPR-KG
belief model and variance-provider interface, but conservatively closes the
loop among:

- adaptive low-dimensional VEPM feature selection,
- shrinkage of sparse VEPM cell estimates toward pooled pre-sample variance,
- posterior-feasible chance-boundary candidate augmentation,
- standard Pareto-KG selection over the enriched candidate set,
- optional explicit random-grid exploration for theory-compatible coverage.

The earlier `GPR-KG-AVB` variant also adds a boundary score directly to both
KG coordinates.  The first RCZDT pilot found that this additive bonus selected
boundary-scored candidates in roughly 29 of 50 adaptive iterations per run,
which was too aggressive and did not yield stable feasible-front gains.

`GPR-KG-AVB-safe` is a conservative follow-up to `AVB-lite`.  It keeps the
same enriched-candidate idea, but a `chance_feasible` boundary candidate must
lie on the safe posterior side with a buffer:

```text
mu_3(x) + z_(1-alpha) sigma_3_hat(x) - tau
    <= -boundary_candidate_feasibility_buffer * sigma_3_hat(x).
```

The default safe buffer is 0.5 in `run_variance_study.py`.  This option is
intended to diagnose and control false feasible recommendations, especially on
benchmarks such as `RCZDT-MisalignedV` where the posterior chance boundary can
be biased even when the variance feature is correctly selected.

Default AVB-lite runner behavior:

```text
--methods GPR-KG-AVB-lite
--adaptive_vepm
--vepm_shrinkage_kappa 10
--boundary_candidate_policy chance_feasible
--boundary_candidate_count 10
--boundary_acquisition_weight 0.0
```

Default AVB-safe runner behavior:

```text
--methods GPR-KG-AVB-safe
--adaptive_vepm
--vepm_shrinkage_kappa 10
--boundary_candidate_policy chance_feasible
--boundary_candidate_count 10
--boundary_candidate_feasibility_buffer 0.5
--boundary_acquisition_weight 0.0
```

The above defaults are method-level defaults in `run_variance_study.py`.
Per-run `result.json` and `run_meta.json` record the realized adaptive
features and AVB-lite parameters after pre-sampling.

## 4. Problem Suites

### 4.1 Original Suite

Keep:

- `RZDT1`
- `RZDT2`
- `RZDT5_RR`

Role:

- Moderate or realistic heteroscedasticity.
- Stability and comparability evidence.
- Not sufficient alone to claim strong VEPM superiority.

### 4.2 Variance-Critical Suite

Add:

- `RZDT1_VC`
- `RZDT2_VC`
- `RZDT5_RR_VC`

Design requirements:

- The feasible Pareto boundary crosses both low-noise and high-noise zones.
- True variance directly affects chance-constraint feasibility.
- Pooled variance has a systematic risk of feasibility misclassification.
- Oracle variance should clearly outperform pooled variance.
- VEPM partition features should be aligned with the true variance features.

### 4.3 RCZDT Main Variance-Critical Suite

Add the following new benchmarks as the preferred main variance-critical
suite:

- `RCZDT-Curve2D`
- `RCZDT-MisalignedV`
- `RCZDT-StepV`

Rationale:

- `RCZDT-Curve2D` breaks the old axis-front limitation. Its theoretical
  Pareto set is approximately `(x1, L-x1, 0, 0, ...)`, and the noise field is
  center-peaked along the two-coordinate Pareto curve.
- `RCZDT-MisalignedV` tests high-dimensional heteroscedastic feature
  selection. Its mean Pareto structure uses `x1`, `x2`, and `x3`, while the
  noise field is mainly governed by `x3`.
- `RCZDT-StepV` tests region-type or operating-regime heteroscedasticity.
  This is where a partition variance estimator should be more natural than a
  pooled or overly smooth variance surrogate.

All three problems have analytic/enumerable true Pareto solution sets through
`true_pareto_solutions()` and true feasible Pareto fronts through
`true_pareto_front()`.

Current truth summary at `d=5`, `L=100`, `sigma=0.04`, `alpha=0.05`:

| Problem | Theoretical Pareto structure | Feasible Pareto count | Std. ratio on full theoretical Pareto set |
| --- | --- | ---: | ---: |
| `RCZDT-Curve2D` | `(x1, 100-x1, 0, 0, 0)` | 41 | 12.00 |
| `RCZDT-MisalignedV` | `(x1, round(100 v*(u)), round(100 r*(u)), 0, 0)` | 52 | 4.57 |
| `RCZDT-StepV` | `(x1, 100-x1, 25/75, 0, 0)` | 51 | 5.93 |

These problems should replace the earlier `RZDT*_VC` variants as the main
variance-critical evidence if pilot results confirm a clear oracle gap and a
recoverable VEPM/AVB gap.

### 4.4 Problem Registry Requirements

Each problem must be registered with:

```text
name
dimension
integer_grid_bounds
objective_function
constraint_function
true_variance_function
true_feasibility_function
reference_point
true_pareto_or_reference_front
variance_features
recommended_partition_features
```

New problems should be added through the registry, not hard-coded in the
runner.

## 5. Variance Modeling Implementation Plan

### 5.1 Unified Variance Provider

Implement a unified variance-provider interface:

```text
fit(initial_samples, observations, residuals)
update(new_sample, new_observation, residual)
predict_variance(x, objective_index)
diagnostics(evaluation_pool)
```

Different `variance_mode` choices should swap the variance provider only,
not the main GPR-KG logic.

### 5.2 VEPM Partition Policy

Default policy:

- Do not use full-dimensional `2d` binary partition as the main default.
- Use `partition_features=auto`, resolved through
  `recommended_partition_features` in the problem registry.
- Keep `partition_features=all` as appendix sensitivity.
- Add `adaptive_min_cell_size` to avoid many singleton cells.
- Log cell occupancy, visited cells, median cell size, and singleton-cell
  ratio at each run.

Suggested CLI:

```text
--partition_features auto | all | 0 | 0,1
--adaptive_vepm
--adaptive_vepm_max_features 2
--adaptive_vepm_min_score 0.0
--vepm_shrinkage_kappa 10
```

Implementation status:

- `VEPM(..., feature_indices=...)` supports low-dimensional variance-feature
  partitioning.
- `--partition_features auto` uses `recommended_partition_features` from the
  problem registry. For the current RZDT suite this resolves to `[0]`.
- `--partition_features all` preserves the historical full-coordinate
  partition and is for sensitivity analysis or old-result replication.
- For `d=5`, RZDT `auto` gives 2 binary features and 4 partitions; `all` gives
  10 binary features and 1024 partitions.
- Adaptive VEPM screens only within the candidate feature set resolved by
  `partition_features`. Thus `partition_features=auto` uses registry-relevant
  candidate features, while `partition_features=all` performs all-coordinate
  data-driven screening.

### 5.3 Boundary-Aware KG Closure

The upgraded algorithm should make variance learning affect sampling
decisions, not only posterior feasibility evaluation.  The current
implementation adds a posterior-only chance-boundary score:

```text
margin(x) = mu_3(x) + z_(1-alpha) * sigma_3_hat(x) - tau
BV(x) = exp(-0.5 * (margin(x) / (scale * sigma_3_hat(x)))^2)
        * sigma_3_hat(x)
```

The normalized boundary score enters selection as:

```text
acquisition_pair(x) =
    KG_pair(x) + boundary_acquisition_weight * normalized_BV(x) * (1, 1)
```

This keeps the two-objective Pareto-KG selection geometry but gives priority
to candidates for which local variance calibration can change chance
feasibility.  The raw `kg_pairs`, boundary scores, acquisition-efficient
indices, selected score, and candidate-set additions are all logged in each
iteration.

### 5.4 Robust Finite-Budget VEPM Policy

The 2026-06-23 auto-partition pilot showed that low-dimensional partitioning
removes the singleton-cell problem but does not by itself prevent variance
overestimation.  The dominant failure mode was residual contamination:
one-shot residuals at newly sampled high-dimensional points sometimes contain
large surrogate mean error, and VEPM can mistake this error for simulation
noise.

Robust VEPM is therefore an engineering guard around the same VEPM recursion,
not a replacement of the variance model.  It should be reported as a
finite-budget robustness option.

Implementation status:

- `--robust_vepm` enables robust VEPM updates.
- `--vepm_residual_clip_factor c` clips each squared residual at `c` times the
  current local/cell variance scale before it enters the VEPM recursion.
- `--vepm_new_point_weight eta` gives the first observation at a new solution
  fractional residual weight `eta`; revisits keep full residual weight.
- `--vepm_partition_weight_floor w` optionally enforces a minimum evidence
  weight when averaging solution-level variances into a partition.
- Each iteration logs `vepm_update_details`, including raw residual,
  effective residual, clipping threshold, clipping indicator, and residual
  update weight.

Recommended pilot settings:

```text
--robust_vepm
--vepm_residual_clip_factor 16
--vepm_new_point_weight 0.2
--vepm_partition_weight_floor 0
```

Decision rule:

- If robust VEPM reduces variance log-RMSE and improves feasibility F1 on
  `RZDT2_VC` without damaging `RZDT5_RR`, it becomes the main finite-budget
  VEPM implementation.
- If it only improves calibration but not HV/IGD, present it as a robustness
  diagnostic and keep the paper claims conditional.
- If it worsens both calibration and performance, keep only the diagnostic
  result and move robust VEPM to future-work discussion.

### 5.5 Pooled Variance Policy

The main baseline should use `pooled_pre`.

Rules:

- Estimate one pooled variance per objective/constraint from initial or
  pre-sample residuals.
- Do not use a manually fixed safe value as the main baseline.
- If residual count is small, apply small-sample correction and a numerical
  lower bound.

Keep `fixed_safe` only as an appendix/historical sensitivity case.

### 5.6 Oracle Variance Policy

`GPR-KG-oracleV` uses the true variance function.

Purpose:

- Test whether the benchmark is variance-sensitive.
- Compute oracle gaps.
- Separate benchmark-design limitations from VEPM-estimator limitations.

The oracle variant is a diagnostic upper bound, not an implementable
decision algorithm.

## 6. Unified Execution Pipeline

Target runner:

```text
GPR_KG_Code/experiments/run_variance_study.py
```

Target analysis tools:

```text
GPR_KG_Code/experiments/analyze_variance_study.py
GPR_KG_Code/experiments/postprocess_variance_recommendations.py
GPR_KG_Code/experiments/diagnose_variance_calibration.py
GPR_KG_Code/experiments/plot_variance_study.py
GPR_KG_Code/experiments/update_variance_study_tables.py
```

### 6.1 Runner Interface

The runner should support:

```text
--suite original_rzdt | variance_critical | all
--problems RZDT1 RZDT2 RZDT5_RR RZDT1_VC RZDT2_VC RZDT5_RR_VC
--methods GPR-KG-VEPM GPR-KG-pooled-pre GPR-KG-oracleV cEHVI cParEGO NSGA-II-K
--initial_design common_random | structured
--N 150
--n0 30
--n_reps 10
--seed_base 1000
--sigma 0.04
--alpha 0.05
--partition_features auto
--checkpoint_every 1
--resume
--force
--run_id RUN_ID
--results_root GPR_KG_Code/results/variance_study
```

### 6.2 Output Directory Schema

Each experiment writes to:

```text
GPR_KG_Code/results/variance_study/{run_id}/
  manifest.json
  config.json
  runs.csv
  summary_by_problem_method.csv
  checkpoints/
  traces/
  diagnostics/
  postprocessing_common/
  figures/
  tables/
  logs/
```

Each method/problem/seed run writes:

```text
{problem}/{method}/rep_{seed}/
  run_meta.json
  checkpoint_iter_000.pkl
  checkpoint_iter_001.pkl
  ...
  checkpoint_latest.pkl
  iteration_snapshots.jsonl
  result.json
  diagnostics.json
```

The original `result.json` recommendation is the algorithm's native final
recommendation. It is useful, but it mixes two effects:

- the sequential policy's ability to visit useful regions;
- the final recommendation candidate pool available to that policy.

For every pilot and full run, also run common-pool final recommendation:

```text
python GPR_KG_Code/experiments/postprocess_variance_recommendations.py \
  --run_dir GPR_KG_Code/results/variance_study/{run_id} \
  --kappas 1.0 \
  --random_pool_size 1000 \
  --neighbor_radius 1
```

This restores each method checkpoint and asks all methods to recommend a final
Pareto set on the same deterministic candidate pool for each
`(problem, replication)`.

Common-pool outputs:

```text
postprocessing_common/
  common_recommendation_rows.csv
  common_recommendation_summary.csv
  common_oracle_gap.csv
  details/
```

`common_generic` is the main fairness diagnostic. It uses the union of observed
and reported points across methods, local neighbors, and a deterministic random
finite-grid sample. `common_axis` additionally adds a synthetic Pareto-axis
scan. It is a diagnostic pool for RZDT-style benchmark validation, not a
replacement for the native recommendation metric.

### 6.3 Manifest Requirements

`manifest.json` must include:

```text
git_commit
git_dirty_status
timestamp
hostname
python_version
numpy_version
sklearn_version
command
run_id
suite
problems
methods
N
n0
n_reps
seed_base
sigma
alpha
initial_design
partition_features
variance_mode
```

If `git_dirty_status` is non-empty, record it explicitly.

### 6.4 Per-Iteration Trace Requirements

`iteration_snapshots.jsonl` should include:

```text
iteration
n_evaluations
selected_x
selected_by
variance_mode
estimated_variance_at_selected_x
true_variance_at_selected_x
posterior_feasibility_at_selected_x
current_sample_pareto_size
current_recommended_pareto_size
HV
IGD
CVR
ND
cell_id
cell_size
```

If a method does not use VEPM, use null for VEPM-specific fields.

## 7. Metrics and Diagnostics

### 7.1 Performance Metrics

Main performance metrics:

- `HV`
- `IGD`
- `CVR`
- `ND`
- `TPOS`
- runtime

Report these metrics for both:

- native final recommendation from `result.json`;
- common-pool final recommendation from
  `postprocessing_common/common_recommendation_summary.csv`.

The native metric evaluates the full sequential algorithm. The common-pool
metric checks whether final recommendation differences survive after every
method is given the same candidate pool.

### 7.2 Variance Calibration Metrics

Add:

- variance RMSE
- variance MAE
- log-variance RMSE
- relative variance error
- calibration by low-noise/high-noise region

Run:

```text
python GPR_KG_Code/experiments/diagnose_variance_calibration.py \
  --run_dir GPR_KG_Code/results/variance_study/{run_id} \
  --problems RZDT2_VC
```

The script restores checkpoints and writes:

```text
variance_diagnostics/
  variance_calibration_rows.csv
  variance_calibration_summary.csv
  feasibility_calibration_rows.csv
  feasibility_calibration_summary.csv
  vepm_cell_diagnostics_rows.csv
  vepm_cell_diagnostics_summary.csv
```

Before interpreting a VEPM loss, inspect `vepm_cell_diagnostics_summary.csv`.
If most visited cells are singleton or most evaluation cells are unseen, the
result primarily diagnoses partition sparsity.

### 7.3 Feasibility Calibration Metrics

Add:

- feasibility precision
- feasibility recall
- feasibility F1
- false feasible rate
- false infeasible rate
- posterior feasible set size
- true feasible set coverage

### 7.4 Chance-Boundary Diagnostics

For variance-critical problems, also diagnose whether a weak result comes from
variance calibration or from posterior chance-boundary location error.

Run:

```text
python GPR_KG_Code/experiments/diagnose_boundary_errors.py \
  --run_dir GPR_KG_Code/results/variance_study/{run_id} \
  --problems RZDT2_VC \
  --random_pool_size 1000 \
  --detail_pool axis \
  --detail_all
```

The script restores checkpoints and writes:

```text
boundary_diagnostics/
  boundary_subset_rows.csv
  boundary_subset_summary.csv
  sampling_allocation_rows.csv
  sampling_allocation_summary.csv
  boundary_point_rows.csv
```

Interpretation order:

1. Inspect `near_true_boundary` in `boundary_subset_summary.csv`.
   If `var2_ratio_mean` is close to one but F1 is still poor, VEPM is no
   longer the dominant failure; the posterior chance-margin sign is wrong.
2. Compare `fp` and `fn` subsets.  False positives have positive true chance
   margin but negative posterior chance margin; false negatives have the
   reverse.  This distinguishes over-acceptance from over-rejection.
3. Inspect `sampling_allocation_summary.csv`.  If the observed near-boundary
   fraction is much lower than the candidate near-boundary fraction, a
   boundary-replication or active-boundary policy is justified.  If sampling
   is adequate but classification remains poor, prioritize constraint-mean
   modeling or final chance-margin calibration.

Boundary replication must be budget guarded.  An unbounded boundary
replication rule can consume nearly all adaptive iterations, reduce the number
of unique evaluated solutions, and damage HV/IGD even if it improves some
feasibility diagnostics.  When testing replication, set:

```text
--replication_policy boundary
--replication_max_per_solution 3
--replication_score_threshold 0.0005
--replication_boundary_scale 1.0
--replication_budget_fraction 0.10  # or 0.15 for pilot diagnostics
```

Report both the number of selected replications and the number of unique
observed solutions.  If guarded replication improves feasibility F1 but not
HV/IGD, move to new-point boundary exploration or final recommendation
calibration rather than increasing the replication budget.

New-point boundary exploration is preferable to replication when the goal is to
increase coverage of the chance boundary without spending the simulation budget
on old points.  The diagnostic implementation is controlled by:

```text
--replication_policy none
--boundary_candidate_policy chance_feasible
--boundary_candidate_count 40
--boundary_candidate_pool_size 500
--boundary_candidate_margin_scale 1.0
```

The `chance_feasible` policy only augments the candidate set with unobserved
posterior-feasible points that are close to the estimated chance boundary.  It
does not change the KG-Pareto selection rule and does not create replications.
The 2026-06-24 `RZDT2_VC`, `N=80`, three-seed pilot showed slight native
HV/IGD improvement over robust VEPM without boundary candidates, but no stable
gain after common post-processing.  Therefore it remains a sensitivity or
diagnostic option, not the default main-experiment policy.

Current main-experiment default:

```text
--replication_policy none
--boundary_candidate_policy none
```

### 7.5 Oracle Gap Metrics

Add:

```text
oracle_gain = metric(oracleV) - metric(pooled_pre)
vepm_gain = metric(VEPM) - metric(pooled_pre)
captured_oracle_ratio = vepm_gain / oracle_gain
unrealized_gap = metric(oracleV) - metric(VEPM)
```

For lower-is-better metrics such as IGD, define the sign consistently before
computing gains.

## 8. Execution Stages

### 8.1 Stage A: Local Smoke Test

Purpose: verify code, config, checkpoint, resume, metrics, and figures.

```text
suite = rczdt
problems = RCZDT-Curve2D
methods = GPR-KG-AVB, GPR-KG-VEPM, GPR-KG-pooled-pre, GPR-KG-oracleV
N = 40
n0 = 20
n_reps = 1
sigma = 0.04
alpha = 0.05
initial_design = common_random
```

Pass criteria:

- All runs finish.
- Checkpoints restore successfully.
- Per-iteration traces are complete.
- Variance diagnostics are non-empty.
- Analysis scripts produce summary tables.
- Common-pool postprocessing restores every checkpoint and produces
  `common_recommendation_summary.csv` and `common_oracle_gap.csv`.

### 8.2 Stage B: Server Pilot

Purpose: test whether the variance-critical benchmark triggers variance
modeling value.

```text
suite = rczdt
problems = RCZDT-Curve2D, RCZDT-MisalignedV, RCZDT-StepV
methods = GPR-KG-AVB, GPR-KG-VEPM, GPR-KG-pooled-pre, GPR-KG-oracleV
N = 80
n0 = 30
n_reps = 3
seed_base = 1000
sigma = 0.04
alpha = 0.05
initial_design = common_random
```

Decision gate:

- If oracleV does not outperform pooled_pre under both native and common-pool
  metrics, the benchmark is not variance-critical enough.
- If oracleV outperforms pooled_pre but VEPM does not, improve VEPM partition
  or variance estimation.
- If auto-partition VEPM loses because of residual contamination, run a robust
  VEPM pilot before any N=150 production batch.
- If a high-dimensional run used `partition_features=all` while the benchmark
  variance depends on low-dimensional features, rerun the pilot with
  `partition_features=auto` before drawing algorithmic conclusions.
- If common-axis shows a gain but common-generic does not, the benchmark may be
  variance-sensitive but the final recommendation pool or search trajectory is
  still too weak; do not claim algorithmic dominance from that evidence alone.
- If oracleV and VEPM both outperform pooled_pre under native and common-generic
  metrics, proceed to full run.
- If oracleV outperforms pooled_pre and AVB closes most of the oracle gap while
  plain VEPM does not, make AVB the upgraded main algorithm and present plain
  VEPM as an ablation.

### 8.3 Stage C: Full Synthetic Study

Purpose: produce the main manuscript tables.

```text
suite = original_rzdt + rczdt
methods = GPR-KG-AVB, GPR-KG-VEPM, GPR-KG-pooled-pre, GPR-KG-oracleV, cEHVI, cParEGO, NSGA-II-K
N = 150
n0 = 30
n_reps = 10
seed_base = 1000
sigma = 0.04
alpha = 0.05
initial_design = common_random
partition_features = auto
```

Only common random initialization should be used in the main tables.

### 8.4 Stage D: Appendix Sensitivity

Appendix experiments:

```text
initial_design = structured
variance_mode = pooled_online
variance_mode = fixed_safe
variance_mode = hvs_guarded
partition_features = all
sigma = 0.02, 0.04, 0.08
alpha = 0.01, 0.05, 0.10
```

The appendix should explain robustness and failure modes, not create the
main claim.

## 9. Server Execution Policy

Server runs must:

- Use a detached process or scheduled task so that experiments continue after
  the SSH session ends.
- Save stdout and stderr under `logs/`.
- Save a checkpoint at every iteration.
- Support `--resume` from `checkpoint_latest.pkl`.
- Sync code and manifest before the run.
- Pull back the complete result directory and a zip archive after completion.

Preferred server:

```text
server3112080
```

After pulling results back, run locally:

```text
analyze_variance_study.py
plot_variance_study.py
update_variance_study_tables.py
```

## 10. Manuscript Synchronization Workflow

After every experimental batch:

1. Run analysis scripts and generate tables/figures.
2. Audit generated tables against raw result files.
3. Update the numerical-experiment tables.
4. Update the corresponding text and remove stale interpretations.
5. Update appendix sensitivity or failure-mode discussion.
6. Compile LaTeX.
7. Check references, figure/table numbering, and formula numbering.
8. Commit the changes.

No manuscript table value should be edited manually without a traceable raw
result file and generation script.

## 11. Adding a New Benchmark

Checklist:

1. Add the problem class and metadata to the problem registry.
2. Provide the true variance function.
3. Provide the true feasibility evaluator.
4. Specify the reference point.
5. Specify recommended partition features.
6. Run Stage A local smoke test.
7. Run oracleV vs pooled_pre pilot.
8. Enter full run only if the oracle gap is meaningful.
9. Generate calibration diagnostics.
10. Update problem description and experiment discussion in the manuscript.

## 12. Immediate Implementation Checklist

Implement in this order:

1. Add a problem registry for original and variance-critical RZDT problems.
2. Add the unified variance-provider abstraction.
3. Implement `pooled_pre`, `pooled_online`, `fixed_safe`, `oracle`, and
   `vepm` modes.
4. Add low-dimensional/adaptive VEPM partition options.
5. Implement `run_variance_study.py`.
6. Implement `analyze_variance_study.py`.
7. Implement `postprocess_variance_recommendations.py`.
8. Implement `plot_variance_study.py`.
9. Run Stage A local smoke test.
10. Run Stage A common-pool postprocessing.
11. Deploy Stage B server pilot.
12. Use the decision gate before any full N=150 rerun.
13. Update manuscript text only after diagnostics support the claim.

## 13. Interpretation Rules

- If VEPM does not reliably outperform pooled_pre on the original RZDT suite,
  this does not by itself invalidate the algorithm. It may indicate low
  variance sensitivity or an already strong pooled baseline.
- If oracleV does not outperform pooled_pre, the benchmark is not
  variance-critical.
- If oracleV outperforms pooled_pre but VEPM does not, the issue is likely
  the VEPM estimator or partition policy.
- VEPM advantage is strongest when VEPM outperforms pooled_pre and
  calibration metrics also improve.
- If HVS-guarded helps while VEPM does not, treat it as future enhancement or
  appendix evidence rather than replacing the main theory.
- If structured initial design helps, report it as sensitivity only; main
  tables should use common random initialization.

## 14. Commit Policy

Recommended commit granularity:

1. Pipeline/manual documentation.
2. Problem registry and variance-critical benchmarks.
3. Variance-provider abstraction and pooled/oracle modes.
4. Unified runner and checkpoint schema.
5. Diagnostics and plotting scripts.
6. Local smoke-test results.
7. Server pilot results.
8. Manuscript updates.

Each commit should have one clear theme. Avoid mixing code, results, and
large manuscript rewrites unless the files are logically inseparable.
