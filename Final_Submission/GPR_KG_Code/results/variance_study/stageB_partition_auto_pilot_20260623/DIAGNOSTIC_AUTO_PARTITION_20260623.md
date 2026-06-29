# Auto-Partition VEPM Diagnostic, 2026-06-23

Run directory:
`GPR_KG_Code/results/variance_study/stageB_partition_auto_pilot_20260623`

This diagnostic supplements the N=80, 3-seed pilot for `RZDT2_VC` and
`RZDT5_RR` under `partition_features=auto`.  All 18 checkpoint files were
recovered from the server and the diagnostic script
`experiments/diagnose_variance_calibration.py` was run locally with
`--random_pool_size 1000`.

## Main Finding

The auto partition fixed the earlier sparsity problem, but it did not fix the
variance-estimation problem.  VEPM now uses only feature 0, producing 4 binary
cells, 3 visited cells on average, zero singleton-cell fraction, and zero
unseen-cell fraction on the diagnostic pools.  Therefore the poor performance
is not caused by empty or singleton partitions in this pilot.

The remaining failure mode is residual contamination: VEPM estimates
observation variance from squared residuals `(Y - mu_before)^2`.  At newly
sampled points, these residuals often contain large surrogate mean error, not
only simulation noise.  A small number of such outliers permanently inflates
the solution-level and cell-level variance estimates.

## Key Evidence

### RZDT2_VC, objective 2, random pool

| method | predicted variance | true variance | RMSE | mean ratio | log-RMSE |
|---|---:|---:|---:|---:|---:|
| GPR-KG / VEPM | 1.0641 | 0.0060 | 1.8241 | 952.0 | 4.0119 |
| pooled-pre | 0.0044 | 0.0060 | 0.0054 | 2.76 | 1.2942 |
| oracleV | 0.0060 | 0.0060 | 0.0000 | 1.00 | 0.0000 |

Feasibility calibration on the random pool:

| method | precision | recall | F1 |
|---|---:|---:|---:|
| GPR-KG / VEPM | 0.095 | 0.086 | 0.090 |
| pooled-pre | 0.308 | 0.407 | 0.350 |
| oracleV | 0.691 | 0.717 | 0.666 |

The selected-point residual diagnostic shows the same mechanism.  Across 150
main-loop samples, objective-2 residual squared divided by true variance has
median 2.20, mean 3748.50, 95th percentile 371.35, and maximum 394722.11.

### RZDT5_RR, objective 2, random pool

| method | predicted variance | true variance | RMSE | mean ratio | log-RMSE |
|---|---:|---:|---:|---:|---:|
| GPR-KG / VEPM | 0.0574 | 0.0021 | 0.0767 | 67.9 | 2.6323 |
| pooled-pre | 0.0008 | 0.0021 | 0.0027 | 1.66 | 1.3493 |
| oracleV | 0.0021 | 0.0021 | 0.0000 | 1.00 | 0.0000 |

Feasibility calibration on the random pool:

| method | precision | recall | F1 |
|---|---:|---:|---:|
| GPR-KG / VEPM | 0.997 | 0.646 | 0.666 |
| pooled-pre | 0.987 | 0.966 | 0.976 |
| oracleV | 0.994 | 0.994 | 0.994 |

VEPM is conservative on this problem: precision remains high, but recall drops
because inflated variance rejects many truly feasible points.

## Cell-Level Diagnosis

For `RZDT2_VC`, seed 1, one newly sampled point
`[96, 86, 100, 38, 70]` produced an objective-2 individual variance estimate
of 91.8346 while the true variance was 0.000465.  This single point was enough
to inflate the final cell-level objective-2 variance for cell `(1, 1)` to
8.3529, while the true cell mean was 0.002155.

For `RZDT5_RR`, seed 1, cell `(0, 0)` had true objective-2 mean variance
0.000388 but final VEPM common variance 0.09044.  This again traces to large
single-sample residuals at newly visited points.

## Interpretation

The current VEPM implementation is aligned with the paper's recursive formula
in the sense that it uses the stage-n posterior mean before the Kalman update.
However, the finite-budget engineering behavior is fragile: in high-dimensional
discrete spaces, most selected points are new points, and a single observation
at a new point cannot separate observation noise from surrogate mean bias.

This explains why:

1. `oracleV` can be much better on `RZDT2_VC`, so variance information is useful.
2. `pooled-pre` can outperform VEPM, because it is less exposed to sequential
   new-point surrogate bias.
3. Auto partition alone does not help enough: it removes sparsity, but not
   residual contamination.

## Recommended Next Step

Do not run a full N=150 production batch with this VEPM variant yet.  The next
engineering fix should make VEPM robust to model-bias contamination while
remaining theory-compatible.  Candidate options are:

1. Update partition variance mainly from replicated observations, and use
   one-shot new-point residuals only with robust clipping or low weight.
2. Clip squared residuals by a posterior-predictive scale before adding them to
   VEPM, so deterministic surrogate error cannot be interpreted entirely as
   observation noise.
3. Use a guarded smooth log-variance surrogate as a shrinkage target, not as an
   unconstrained replacement.
4. Add a replication trigger near the chance boundary to intentionally collect
   repeated observations for variance calibration.

