# V32 Fixed-Universe Adaptive Expert Race

> Retrospective information audit (2026-07-14): V32 used the
> `lodo_teacher` line.  Its held-out target was never queried through a truth
> oracle, but source training used analytic `true_sigma`, `true_outputs`, and
> source-domain structural teacher hooks.  V32 is therefore a privileged
> source-oracle upper bound, not the oracle-free LODO mainline.  New promotion
> gates reject such rows even when their target-domain results are strong.

## Single Change

V32 retains V31's history-measurable score refresh but freezes the finite
candidate universe when the reserved suffix begins. Later labels update every
GPR, HVD, and task posterior, then rerank exactly that same action set. No new
candidate can enter the race after seeing reserved-stage outcomes.

This is the finite transductive experiment implied by the proof contract:

```text
U = terminal_pool(D_start)
x_t = nomination(U, D_t)
observe Y_t at x_t
D_{t+1} = D_t union {(x_t, Y_t)}
```

The universe `U` is persisted in checkpoints. Expert nominations may change,
but candidate support cannot disappear because a recommendation RNG generated
a different pool.

## Gate

- Unit tests must show that refresh ignores a different later pool, fixed
  universe state resumes exactly, and simulator calls remain inside `N`.
- Repeat the same FactorShock seed 0 and Inventory seeds 0/1 smoke.
- Only if all three are feasible may the unchanged 7+7 gate run.
- V31 remains available as the drifting-universe ablation; V32 is enabled only
  by `finalist_replication_fixed_universe=True` together with the adaptive
  race.

## Theory

Because `U` is fixed before the suffix labels, all later actions belong to one
pre-observation finite set. The archive is a subset of `U`, its cardinality is
bounded by `|U|`, and a finite union confidence allocation can be fixed before
the adaptive observations. This is stronger than V31's changing-universe
cardinality bound.

## Controlled Smoke Result

The predeclared three-seed smoke passes:

| Domain / seed | True margin | Feasible regret |
|---|---:|---:|
| FactorShock / 0 | `-0.03320` | `0.00825` |
| Inventory / 0 | `-0.03612` | `0.00569` |
| Inventory / 1 | `-0.05090` | `0.00564` |

Inventory seed 1 isolates the mechanism. Stage 17 tests an unsafe ordered
nomination once. After that charged label, stage 18 reranks the same
324-action universe and switches to a safe null-universal nomination. Stage
19 retains that action until it reaches two observations; the completed race
then selects it. The drifting-universe V31 run never retained this candidate.

This result authorizes, but does not replace, the unchanged 7+7 promotion
gate. V32 remains unpromoted until that gate passes.

## Full Gate And Promotion

V32 passes the unchanged 7+7 gate:

| Held-out domain | Truly feasible | False certificate | Median feasible regret | Mean violation |
|---|---:|---:|---:|---:|
| FactorShock | `7/7` | `0/7` | `0.00825` | `0.00000` |
| Inventory | `5/7` | `0/7` | `0.00569` | `0.001582` |

All fourteen nested group-ridge selections are valid. Relative to V30,
FactorShock quality is unchanged, Inventory improves from `3/7` to `5/7`,
Inventory mean violation falls from `0.05306` to `0.001582`, and median wall
time does not regress. V32 is therefore promoted as the new LODO baseline in
`performance/baselines/lodo_current.json`.

## Frozen Queue Gate 2 Result

V32 was frozen before Queue was opened as a held-out target. The only change in
run `lodo_v32_queue_n20_gate2_frozen_20260712` was the held-out domain; all
seven seeds used FactorShock and Inventory as sources and the promoted V32
settings. Tasks `t29249` through `t29255` ran only on `node001-node006`.

| Held-out domain | Truly feasible | False certificate | Median feasible regret | Mean violation | Median true margin |
|---|---:|---:|---:|---:|---:|
| Queue | `3/7` | `0/7` | `0.00455` | `0.05767` | `+0.01829` |

Every terminal pool contained at least one truly feasible action, but no seed
had a posterior-certified action. The four failures are therefore
selection/calibration failures rather than missing-candidate failures. This
rejects a three-domain stability claim for V32. No Queue-specific repair is
permitted; the result opens only the shadow joint task-latent audit.
