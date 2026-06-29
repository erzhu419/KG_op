# Robust VEPM Pilot Analysis, 2026-06-23

Run directory:
`GPR_KG_Code/results/variance_study/stageB_robust_vepm_pilot_20260623`

Configuration:

```text
problems = RZDT2_VC, RZDT5_RR
method = GPR-KG-VEPM
N = 80
n0 = 30
n_reps = 3
seed_base = 1000
sigma = 0.04
alpha = 0.05
partition_features = auto
robust_vepm = true
vepm_residual_clip_factor = 16
vepm_new_point_weight = 0.2
vepm_partition_weight_floor = 0
```

The pilot finished successfully on the server with `RUN_EXIT=0`,
`ANALYZE_EXIT=0`, and `POST_EXIT=0`.  All 6 checkpoints were recovered locally
and `experiments/diagnose_variance_calibration.py` was run with
`--random_pool_size 1000`.

## Main Result

Robust VEPM substantially reduced the residual-contamination failure diagnosed
in the auto-partition VEPM pilot.  The improvement is strongest in variance
and feasibility calibration; native HV/IGD also improved relative to the
unprotected auto-partition VEPM.

## Native Metrics

| problem | method | HV | IGD | CVR | ND |
| --- | --- | ---: | ---: | ---: | ---: |
| RZDT2_VC | auto VEPM | 0.2620 | 1.0886 | 0.4952 | 5.67 |
| RZDT2_VC | robust VEPM | 0.3683 | 1.0009 | 0.5000 | 4.00 |
| RZDT2_VC | pooled-pre | 0.2936 | 1.1164 | 0.3631 | 7.33 |
| RZDT2_VC | oracleV | 0.7877 | 0.4689 | 0.0000 | 2.33 |
| RZDT5_RR | auto VEPM | 2.0928 | 0.0643 | 0.0000 | 7.67 |
| RZDT5_RR | robust VEPM | 2.1205 | 0.0434 | 0.0238 | 12.00 |
| RZDT5_RR | pooled-pre | 2.1011 | 0.0478 | 0.0000 | 10.00 |
| RZDT5_RR | oracleV | 2.1015 | 0.0539 | 0.0000 | 11.67 |

## Common-Generic Postprocessing

| problem | method | HV | IGD | CVR | feasibility F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| RZDT2_VC | auto VEPM | 0.2620 | 1.0096 | 0.5686 | 0.0624 |
| RZDT2_VC | robust VEPM | 0.4476 | 0.7512 | 0.4212 | 0.2863 |
| RZDT2_VC | pooled-pre | 0.5971 | 0.7332 | 0.5157 | 0.2780 |
| RZDT2_VC | oracleV | 0.8290 | 0.4248 | 0.0000 | 0.6932 |
| RZDT5_RR | auto VEPM | 2.1102 | 0.0494 | 0.0000 | 0.7348 |
| RZDT5_RR | robust VEPM | 2.1312 | 0.0395 | 0.0152 | 0.9822 |
| RZDT5_RR | pooled-pre | 2.1251 | 0.0419 | 0.0573 | 0.9892 |
| RZDT5_RR | oracleV | 2.1133 | 0.0467 | 0.0000 | 0.9961 |

## Variance Calibration, Objective 2, Random Pool

| problem | method | predicted variance | true variance | RMSE | mean ratio | log-RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RZDT2_VC | auto VEPM | 1.0641 | 0.0060 | 1.8241 | 952.0 | 4.0119 |
| RZDT2_VC | robust VEPM | 0.0047 | 0.0060 | 0.0053 | 2.72 | 1.2710 |
| RZDT5_RR | auto VEPM | 0.0574 | 0.0021 | 0.0767 | 67.9 | 2.6323 |
| RZDT5_RR | robust VEPM | 0.0009 | 0.0021 | 0.0023 | 1.34 | 1.1610 |

## Feasibility Calibration, Random Pool

| problem | method | precision | recall | F1 |
| --- | --- | ---: | ---: | ---: |
| RZDT2_VC | auto VEPM | 0.0951 | 0.0863 | 0.0899 |
| RZDT2_VC | robust VEPM | 0.3438 | 0.3188 | 0.3277 |
| RZDT5_RR | auto VEPM | 0.9970 | 0.6465 | 0.6665 |
| RZDT5_RR | robust VEPM | 0.9676 | 0.9935 | 0.9803 |

## Update Diagnostics

The robust update logged 450 objective-level VEPM updates per problem.

For `RZDT2_VC`, mean raw residuals were reduced to much smaller effective
residuals:

```text
objective 0: clip rate 0.040, mean raw 0.2878, mean effective 0.0111
objective 1: clip rate 0.133, mean raw 0.9635, mean effective 0.0179
objective 2: clip rate 0.100, mean raw 0.2057, mean effective 0.0132
```

For `RZDT5_RR`, objective 1 remains the most surrogate-biased channel, with
clip rate 0.680, mean raw residual 2.0192, and mean effective residual 0.0745.
This is consistent with the earlier diagnosis that VEPM was mainly hurt by
surrogate mean misspecification rather than partition sparsity.

## Interpretation

Robust VEPM is a meaningful finite-budget improvement.  It keeps the
theory-aligned VEPM structure but prevents one-shot residual outliers at new
points from dominating cell-level variance estimates.  This directly addresses
the failure mode found in the auto-partition diagnostic.

The improvement does not yet close the gap to oracleV on `RZDT2_VC`, so the
paper should not claim that VEPM fully recovers true heteroscedastic structure.
However, it now supports a more credible claim: finite-budget robust VEPM can
convert a fragile residual-based variance estimator into a calibrated
variance provider, especially when paired with variance-feature-aligned
partitions.

Recommended next step: run one slightly larger pilot with robust VEPM against
`pooled-pre` and `oracleV` in the same run directory, or proceed to N=150 only
after confirming that the robust advantage is stable over more seeds.

