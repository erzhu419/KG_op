# Stage B Variance Pilot Analysis

Run id: `stageB_variance_pilot_20260622_breakaway`

Server: `tf290q6n.zjz-service.cn:23565`, host `DESKTOP-HVSQKOI`

Configuration:

- Problems: `RZDT1`, `RZDT2`, `RZDT5_RR`, `RZDT1_VC`, `RZDT2_VC`
- Methods: `GPR-KG-VEPM`, `GPR-KG-pooled-pre`, `GPR-KG-oracleV`
- `N=80`, `n0=30`, `n_reps=3`, `sigma=0.04`, `alpha=0.05`
- Initial design: `common_random`
- Postprocessing: native recommendation plus common-pool recommendation

Completion:

- `RUN_EXIT=0`
- `ANALYZE_EXIT=0`
- `POST_EXIT=0`
- `45/45` sequential runs completed.

## Native Recommendation Summary

High HV and low IGD are better. Low CVR is better.

| Problem | Best native HV | Best native IGD | Best native CVR | Main observation |
| --- | --- | --- | --- | --- |
| RZDT1 | oracleV | oracleV | oracleV | True variance helps, but VEPM is worse than pooled-pre. |
| RZDT1_VC | pooled-pre | pooled-pre | pooled-pre | The variance-critical design does not yet favor oracle variance. |
| RZDT2 | pooled-pre | pooled-pre | VEPM/pooled-pre tie | Pooled-pre is strongest; VEPM is weak. |
| RZDT2_VC | oracleV | oracleV | oracleV | The benchmark has an oracle-variance signal, but VEPM does not capture it. |
| RZDT5_RR | VEPM | VEPM | VEPM | VEPM has a small but consistent advantage. |

Native mean HV:

| Problem | VEPM | pooled-pre | oracleV |
| --- | ---: | ---: | ---: |
| RZDT1 | 0.5188 | 0.9118 | 1.1112 |
| RZDT1_VC | 1.1611 | 1.4753 | 0.9772 |
| RZDT2 | 0.5684 | 0.9670 | 0.8630 |
| RZDT2_VC | 0.2224 | 0.6711 | 0.9469 |
| RZDT5_RR | 2.1146 | 2.0730 | 2.1069 |

Native mean IGD:

| Problem | VEPM | pooled-pre | oracleV |
| --- | ---: | ---: | ---: |
| RZDT1 | 0.7780 | 0.5939 | 0.3079 |
| RZDT1_VC | 0.2873 | 0.0638 | 0.5060 |
| RZDT2 | 0.8272 | 0.2651 | 0.5164 |
| RZDT2_VC | 1.0682 | 0.5257 | 0.3286 |
| RZDT5_RR | 0.0494 | 0.0670 | 0.0534 |

## Seed-Level Native Wins

By seed-level HV wins:

- `RZDT1`: oracleV wins `3/3`.
- `RZDT1_VC`: pooled-pre wins `2/3`, oracleV wins `1/3`.
- `RZDT2`: pooled-pre wins `2/3`, oracleV wins `1/3`.
- `RZDT2_VC`: oracleV wins `2/3`, pooled-pre wins `1/3`.
- `RZDT5_RR`: VEPM wins `2/3`, oracleV wins `1/3`.

By seed-level IGD wins:

- `RZDT1`: oracleV wins `3/3`.
- `RZDT1_VC`: pooled-pre wins `3/3`.
- `RZDT2`: pooled-pre wins `3/3`.
- `RZDT2_VC`: oracleV wins `2/3`, pooled-pre wins `1/3`.
- `RZDT5_RR`: VEPM wins `2/3`, oracleV wins `1/3`.

## Common-Pool Recommendation Summary

The common-pool diagnostic gives every method the same final candidate pool.
This separates final recommendation quality from the sequential policy's
visited candidate set.

Common-generic mean HV:

| Problem | VEPM | pooled-pre | oracleV |
| --- | ---: | ---: | ---: |
| RZDT1 | 0.7135 | 1.3372 | 1.1656 |
| RZDT1_VC | 1.2968 | 1.5680 | 1.2190 |
| RZDT2 | 0.6507 | 1.0709 | 1.1056 |
| RZDT2_VC | 0.4911 | 0.8204 | 1.1015 |
| RZDT5_RR | 2.1181 | 2.0899 | 2.1233 |

Common-generic mean IGD:

| Problem | VEPM | pooled-pre | oracleV |
| --- | ---: | ---: | ---: |
| RZDT1 | 0.5926 | 0.2048 | 0.2862 |
| RZDT1_VC | 0.2247 | 0.0402 | 0.2694 |
| RZDT2 | 0.6645 | 0.2045 | 0.1972 |
| RZDT2_VC | 0.7766 | 0.4054 | 0.2068 |
| RZDT5_RR | 0.0454 | 0.0557 | 0.0461 |

Common-pool interpretation:

- `RZDT1` and `RZDT1_VC`: pooled-pre remains strongest after common-pool
  postprocessing. These are not clean variance-critical cases.
- `RZDT2` and `RZDT2_VC`: oracleV becomes strongest or near strongest. These
  cases contain an oracle-variance signal, but VEPM still fails to realize it.
- `RZDT5_RR`: VEPM and oracleV are both better than pooled-pre, with very small
  gaps. This is the only current case where VEPM is competitive in a stable
  way.

## Diagnosis

1. The current `pooled-pre` baseline is strong.

   It is estimated from pre-sample residuals rather than a fixed unsafe value,
   so it is not an artificially weak baseline. On several problems it is
   conservative enough to avoid feasibility failures while still recommending
   good final points.

2. The current `RZDT1_VC` benchmark is not variance-critical enough.

   If oracleV does not beat pooled-pre, the benchmark does not isolate the
   value of accurate heteroscedastic variance. `RZDT1_VC` fails this gate in
   both native and common-pool diagnostics.

3. `RZDT2_VC` is variance-sensitive, but VEPM is not yet strong enough.

   oracleV outperforms pooled-pre on native mean HV/IGD and common-generic
   HV/IGD. VEPM is still worse than pooled-pre, so the issue is likely the
   VEPM estimator or its partition policy rather than the benchmark alone.

4. `RZDT5_RR` supports a moderate VEPM claim.

   VEPM has the best native HV, best native IGD, zero CVR, and wins `2/3`
   seeds. In common-generic postprocessing, oracleV has slightly higher HV,
   but VEPM has slightly better IGD and remains above pooled-pre.

5. The evidence is not sufficient for a strong universal superiority claim.

   The manuscript should not claim that VEPM generally dominates pooled
   variance. The defensible claim is conditional: VEPM can help when variance
   structure is aligned with the feasible Pareto boundary and the partition
   receives enough samples. The current Stage B pilot shows this only weakly
   and mainly on `RZDT5_RR`.

## Variance-Calibration Diagnosis

After the Stage B results were pulled back, `RZDT2_VC` checkpoints were
restored and evaluated with:

```text
GPR_KG_Code/experiments/diagnose_variance_calibration.py
```

The diagnostic output is in:

```text
variance_diagnostics/
  variance_calibration_summary.csv
  feasibility_calibration_summary.csv
  vepm_cell_diagnostics_summary.csv
```

Key `RZDT2_VC` findings for the constraint variance (`objective_index=2`):

| Method | Pool | Variance RMSE | Log RMSE | Ratio mean | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| VEPM | axis | 0.0212 | 1.7156 | 10.96 | 0.0966 | 0.1282 |
| pooled-pre | axis | 0.0054 | 1.2789 | 2.69 | 0.1417 | 0.1538 |
| oracleV | axis | 0.0000 | 0.0000 | 1.00 | 0.9818 | 0.3462 |
| VEPM | combined | 4.1044 | 1.9293 | 504.39 | 0.2798 | 0.2701 |
| pooled-pre | combined | 0.0054 | 1.3082 | 2.84 | 0.2947 | 0.3761 |
| oracleV | combined | 0.0000 | 0.0000 | 1.00 | 0.9266 | 0.5297 |

On the axis pool, VEPM substantially overestimates the low-`t1` constraint
variance:

| Region | VEPM ratio mean | pooled-pre ratio mean |
| --- | ---: | ---: |
| low `t1` | 25.84 | 3.82 |
| mid `t1` | 1.55 | 0.36 |
| high `t1` | 5.22 | 3.82 |

The cell diagnostics explain why. The current Stage B implementation used the
historical full-coordinate binary VEPM partition:

| Method | n features | total partitions | visited cells | visited fraction | singleton visited cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| VEPM | 10 | 1024 | 45.67 | 0.0446 | 0.6636 |

This is misaligned with the benchmark design. `RZDT2_VC` variance depends only
on `x1`, but the implementation partitioned over all five coordinates. With
only `N=80` observations, most cells are singleton or unseen, so partition
sharing is weak and the VEPM variance estimates become noisier than the
pooled-pre baseline.

## Code Correction After Diagnosis

The VEPM implementation has been updated to support theory-compatible
variance-feature partitioning:

```text
--partition_features auto
--partition_features all
--partition_features 0,1
```

`auto` reads `problem.recommended_partition_features` from the problem
registry. For the current RZDT problems this resolves to `[0]`, reducing
binary VEPM from 10 partition features and 1024 cells to 2 partition features
and 4 cells. The historical full-coordinate partition remains available as
`--partition_features all` for sensitivity analysis and old-result
replication.

Local smoke test:

```text
run_id = local_partition_auto_smoke_20260623_v2
problem = RZDT2_VC
N = 13, n0 = 12, n_reps = 1
partition_features = auto
```

The smoke result metadata confirms:

```text
partition_features_resolved = [0]
vepm_n_features = 2
vepm_partitions = 4
```

## Recommended Next Step

Before any full `N=150`, `10`-replication rerun, rerun a small Stage B-style
pilot with the corrected partition:

```text
problems = RZDT2_VC, RZDT5_RR
methods = GPR-KG-VEPM, GPR-KG-pooled-pre, GPR-KG-oracleV
N = 80
n_reps = 3
partition_features = auto
```

Then compare against the old full-coordinate Stage B results.

If VEPM still fails on `RZDT2_VC`, continue with benchmark and estimator
improvements:

1. Redesign `RZDT1_VC` so oracleV reliably beats pooled-pre.
2. Use `RZDT2_VC` as the main debugging case because oracleV already has a
   clear signal there.
3. Add variance calibration diagnostics on a fixed evaluation grid:
   variance RMSE, log-variance RMSE, feasibility precision, feasibility recall,
   and low-noise/high-noise region breakdown.
4. Investigate VEPM partition sparsity:
   cell sizes, residual variance estimates by cell, shrinkage strength, and
   whether `partition_features=(0,)` is too coarse or too brittle.
5. Only after oracleV and VEPM both beat pooled-pre under common-generic
   diagnostics should the full synthetic rerun be started.
