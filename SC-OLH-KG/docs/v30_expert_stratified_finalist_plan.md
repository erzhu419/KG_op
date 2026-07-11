# V30 Expert-Stratified Finalist Nomination

## Evidence

V29 fails because its stage-17 finalist set contains no feasible Inventory
policy. A frozen stage-18 audit evaluates one safety nomination per finite task
expert after all predictions and the pool are fixed. The dominant universal
expert nominates an unsafe point with true margin `+0.0551`. In contrast, the
nearly posterior-deleted ordered cumulative expert nominates a feasible point
with margin `-0.0361`, objective `0.282`, and the smallest expert-specific
predicted positive violation (`0.0101`) among all six experts.

The ordered model learned a useful safe direction; mixture-posterior collapse
prevented that direction from reaching the finite terminal race.

## Single Controlled Change

V30 keeps V29's budget, replicated upper margin, certification precedence,
and final safety-first rule. It changes only the second finalist source:

1. finalist 1 is the current minimum Bayes-risk action;
2. every supported structural expert nominates its own minimum predicted
   positive-violation action from the same frozen terminal pool;
3. finalist 2 is the best of those expert-specific nominations, ranked by its
   own predicted violation, without multiplying by posterior expert mass.

If more finalists are requested, the remaining distinct expert nominations
are inserted in that same score order before mixture-level fallbacks. This is
source-prior support at the decision-set level: a small target sample cannot
erase an entire structural family before direct finalist evidence is charged.

No expert name, domain label, analytic boundary, target truth, or uncharged
simulator call enters the selector. The option is default-off so V29 remains an
exact ablation.

## Gate

- Repeat FactorShock/Inventory seed 0 with `N=20`, `R=3`, two finalists, and
  two observations per finalist.
- FactorShock must remain feasible with zero violation.
- Inventory must become truly feasible without a false theory certificate.
- Only after the paired smoke passes may the unchanged 7+7 gate run.

## Proof Obligation

For every finite expert with source-prior support, its nomination exists in the
stratified nomination set. Selecting finalists by the finite expert-specific
score therefore cannot delete a family merely because its task-posterior mass
is small. V29's replicated-margin theorem continues to control the final
empirical decision.

## Controlled Smoke Result

The paired seed-0 smoke passes without changing `N=20` or using target truth:

| Held-out domain | Selected finalist | True margin | Feasible regret |
|---|---|---:|---:|
| FactorShock | minimum Bayes risk | `-0.03320` | `0.00825` |
| Inventory | `ordered_cumulative` expert nomination | `-0.03612` | `0.00569` |

Inventory is the decisive case. The selected ordered nomination is the same
safe structural action identified by the frozen stage-18 audit even though
that expert had posterior mass about `3.3e-5`. The fallback reports no theory
or empirical certificate, so the run does not turn a feasible empirical
outcome into a false certification claim. FactorShock retains its previous
safe recommendation.

This is smoke evidence only. V30 is not promoted until the unchanged 7+7
FactorShock/Inventory gate passes.

## Gate-1 Result

The unchanged 7+7 gate rejects V30:

| Held-out domain | True feasible | False certificate | Median feasible regret |
|---|---:|---:|---:|
| FactorShock | `7/7` | `0/7` | `0.00825` |
| Inventory | `3/7` | `0/7` | `0.00569` |

All group-ridge complexity selections are valid. Inventory misses the
predeclared `4/7` threshold by one seed, so the seed-0 smoke is not promoted.
One Inventory seed also leaves a finalist below the requested replication
count; V30's all-or-nothing terminal check then discards the evidence from the
completed finalist.

A frozen stage-18 audit shows that every failed Inventory pool still contains
five to eight truly feasible actions. For seeds 1 and 5, an expert's first
safety nomination is feasible after one charged finalist update. For seeds 2
and 4, the local-risk expert's second-ranked nomination is feasible. Thus the
learned family has safe support, but V30 compresses each expert to one action
once at stage 17 and never refreshes that support after new labels arrive.

V30 is retained only as the frozen-support ablation. The next controlled
change is a history-measurable adaptive finalist race; no static prior mixture
or additional representation is reintroduced.
