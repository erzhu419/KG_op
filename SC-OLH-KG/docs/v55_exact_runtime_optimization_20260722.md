# V55 exact posterior-update runtime optimization

## Scope

This optimization track must preserve the V55 action set, nested RQMC sample
plan, posterior update, both terminal functionals, score arrays, and selected
action. It may change only execution reuse and parallel scheduling. A change is
accepted only when all six risk/certificate high/prefix score arrays agree
within `1e-10` and the selected action fingerprint is unchanged.

The reference workload is the promoted one-step `d=1000, N=11, n0=10`,
13-new-action V55 decision with 18 active evaluate-or-replicate actions.

## Optimization 1: nested-prefix reuse

`factorized_rqmc_nested` and `antithetic_nested` already place the low-MC
sample set at the start of the high-MC sample plan. The historical guard ran
the high pass and then repeated the prefix posterior refits. The optimized
path aggregates the prefix terminal values while processing the high pass and
returns both estimates from one set of fantasies. The legacy second pass
remains available through `exact_kg_reuse_nested_prefix=False`.

Serial and `process_fork` chunked tests match risk, certificate, prefix, and
selected-action outputs within `1e-12`. The ideal fantasy-work reduction is
20 percent for MC128 (`160 -> 128` total samples) and about 5.9 percent for
MC512 (`544 -> 512`).

## Optimization 2: real worker scaling

The submission script previously increased scheduler `cpu` reservations while
leaving `--exact-jobs` frozen at 12. It now exposes `--exact-jobs`, rejects a
worker count larger than the CPU reservation, and records the actual worker
count in each result. The preregistered scaling run is:

`scolh_v55_prefix_reuse_core_scaling_mc128_c{12,24,36,48}_s1_20260722_01`.

Each scaling point contains the same seed and frozen initial design for
FactorShock, Inventory, and Queue. `analyze_v55_runtime_scaling.py` verifies
score/action equivalence before reporting speedup and parallel efficiency.

## Optimization 2b: balanced LCM chunk waves

The historical process scheduler used
`ceil(worker_count / active_action_count)` chunks per action. This can create a
nearly empty final wave. With 18 actions and 48 workers, for example, it made
54 chunks: 48 ran in the first wave and only 6 in the second. The
`balanced_lcm` schedule uses

`chunks_per_action = workers / gcd(workers, active_actions)`

subject to the available MC samples. The total number of equal-size chunks is
then a multiple of the worker count. For 18 actions, 12/24/36/48/72 workers use
2/4/2/8/4 chunks per action. The legacy schedule remains selectable. Fixed-CRN
process tests verify numerical and action equivalence after recombination.

## Optimization 3: redundant primary-update removal

Under the promoted `task_joint` authority, the Bayes-risk terminal and the
certificate-deficit terminal are functions of the updated task ensemble. The
separate primary GPR/HVD clone and fantasy update is not read by either
functional. `exact_kg_skip_redundant_primary_update=True` removes this shadow
computation only when every active terminal head is in the proven
ensemble-only set. Split authorities, absent task ensembles, and every other
terminal mode retain the historical update.

The gate has unit coverage that makes the shadow update raise if it is called
on the ensemble-only path, and verifies that disabling the optimization
restores the call. The scheduler-level same-seed comparison below also passes
for every domain and score head.

## Completed worker-scaling gate

All 12 FactorShock/Inventory/Queue cells completed. Relative to the controlled
12-worker prefix-reuse reference, median KG time changed as follows:

| exact workers | median KG time | speedup | median parallel efficiency |
| ---: | ---: | ---: | ---: |
| 12 | 2795.464 s | 1.00x | 1.000 |
| 24 | 1514.464 s | 1.83x | 0.916 |
| 36 | 867.035 s | 3.34x | 1.114 |
| 48 | 700.881 s | 4.13x | 1.033 |

The efficiency above one at 36/48 workers reflects reduced tail-wave and host
contention effects, not fewer posterior fantasies. Every cell preserved the
active-action set, selected action, and all six score arrays within `1e-10`;
the largest observed absolute difference was `7.11e-15`.

Against the earlier 12-worker implementation before nested-prefix reuse, the
48-worker KG speedup is `5.38x` at the median. The controlled result separates
that gain into prefix reuse plus a `4.13x` worker-scaling gain. Median total
algorithm speedup against the earlier implementation is `3.15x`, because
Inventory and Queue retain non-KG initialization/finalization work.

The complete machine-readable report is
`scolh_v55_prefix_reuse_core_scaling_mc128_c48_s1_20260722_01/`
`runtime_scaling_summary.json`. Worker scaling and prefix reuse pass their
promotion gate.

## Combined execution-optimization gate

Redundant primary-update removal and balanced chunking were tested in
`scolh_v55_runtime_opt_balanced_shadow_mc128_c48_s1_20260722_01` and
`scolh_v55_runtime_opt_balanced_shadow_mc128_c72_s1_20260722_01`. Every
FactorShock/Inventory/Queue cell passed score/action equivalence.

| configuration | median KG time | KG speedup vs old 12-worker | total speedup |
| --- | ---: | ---: | ---: |
| old 12-worker MC128 | 3764.085 s | 1.00x | 1.00x |
| optimized 48-worker MC128 | 464.656 s | 8.06x | 3.84x |
| optimized 72-worker MC128 | 364.766 s | 10.32x | 3.97x |

At MC128, moving from 48 to 72 workers gives another `1.24x` median KG
speedup but only a small total-runtime gain because fixed initialization and
final diagnostics now dominate Inventory and Queue. The largest score-array
difference against the old implementation is `7.11e-15`.

## Production-fidelity MC512 validation

The production-fidelity 72-worker run is
`scolh_v55_runtime_opt_balanced_shadow_mc512_c72_s1_20260722_01`. It preserves
the full 512-sample posterior update and the 32-sample nested prefix.

| domain | old total | optimized total | optimized KG | total speedup |
| --- | ---: | ---: | ---: | ---: |
| FactorShock | 217.1 min | 29.0 min | 23.9 min | 7.48x |
| Inventory | 237.3 min | 39.0 min | 24.4 min | 6.08x |
| Queue | 211.6 min | 34.5 min | 23.7 min | 6.14x |

Median KG speedup is `8.87x`; median end-to-end speedup is `6.14x`. Every
selected action and active-action fingerprint is unchanged, and the maximum
absolute difference across the six high/prefix risk/certificate score arrays
is `3.55e-15`.

Use `exact_jobs=72` for MC512 validation when node capacity permits. Use 48
workers for throughput-heavy matrices where scheduling density matters: it
retains most of the gain while consuming one third fewer cores. The next
optimization target is fixed initialization/finalization, not fewer MC draws.
