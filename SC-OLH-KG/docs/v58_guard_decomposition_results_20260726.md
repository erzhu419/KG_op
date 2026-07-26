# V58 Guard-Decomposition Gate Result

## Registered Matrix

- Domains: FactorShock, Inventory, Queue.
- Seeds: `0..4`.
- Search budget: `N=13`, `n0=10`.
- Control: promoted V51.
- Challenger: V58 guard-decomposed action support with MC512 independent
  confirmation.
- Completed: `30/30`, with no retries or missing rows.

## Result

V58 passed its formal implementation gate but failed promotion.

- Paired performance: `1 win / 14 ties / 0 losses`.
- Feasibility losses: `0`.
- False posterior certificates: `0`.
- Minimum posterior chance margin improved in `13/15` pairs and worsened in
  `2/15`; median change was `-0.0714138`.
- True-feasible recommendations were `5/5` FactorShock, `5/5` Inventory, and
  `4/5` Queue, unchanged from V51.
- Posterior chance certificates remained `0/15`.

Median minimum posterior chance margins changed as follows:

| Domain | V51 | V58 |
|---|---:|---:|
| FactorShock | 0.2884 | 0.0590 |
| Inventory | 0.6409 | 0.3439 |
| Queue | 0.4547 | 0.2112 |

The guard decomposition diagnosed different bottlenecks. FactorShock was
mostly aleatoric-limited. Inventory and Queue were epistemic-limited, and the
robust task/source term inflated the nominal expert uncertainty. Thus V58
made substantial progress toward the boundary but did not make the posterior
certificate nonvacuous.

## Decision

Do not promote V58 as the optimization baseline. The result rules out simply
adding more certificate-directed search heuristics at `N=13`. The next
registered test separates optimization from verification: preserve the V51
search trajectory exactly, freeze its terminal policy, then spend an
independent and fully counted replication budget on direct finite-sample
chance certification.
