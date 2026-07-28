# V27 Task-Adaptive Group-Ridge Complexity

## Hypothesis

The held-out task should determine how much ordered curvature and shared
exposure complexity it needs.  One global rank fraction is not transferable:
FactorShock observations support a low effective dimension, while Inventory
observations support an approximately eight-dimensional full diagonal model.

## Posterior

The ordered expert keeps the interpretable feature blocks

```text
[A, A^2, N].
```

It learns one isotropic ridge precision per block from the fixed grid

```text
{1e-4, 1e-2, 1e-1, 1, 10, 100, 1000}.
```

Selection uses full nested leave-one-out refits on charged target observations
and a safety-weighted prediction loss.  It does not use target truth, task
labels, recommendation outcomes, or an oracle.  Coordinate descent starts from
both the source scale and a neutral ridge and records every tested model.

The Gaussian posterior is then refit with the selected block precisions.  Its
effective dimension is the ridge smoother trace, not a hard coefficient count.
Exact-KG fantasy clones repeat the same nested selector after each hypothetical
observation, so complexity-parameter VOI remains in the posterior update.

## Controls

- retain V25 latent local/ordered structure and exactly six experts;
- retain ordered frequencies, factor HVD, theory certification, IID exact-MC2;
- retain the same initial design, candidate pool, terminal pool, and budget;
- disable V26 group spike/slab and every semiparametric residual;
- report nested-LOO validity, selected penalties, effective df, and oracle-use
  boolean in every shard.

## Gate

1. Paired smoke: FactorShock seed 0 and Inventory seed 0 must complete without
   material runtime blow-up or invalid complexity selection.
2. Full gate: FactorShock 7/7 true feasible with zero violation; Inventory at
   least 4/7 true feasible with at most one false feasible; all 14 nested
   selectors valid and oracle-free.
3. Queue remains closed until the paired full gate passes.

The finite-grid oracle inequality and ridge effective-dimension bounds are
formalized in `proof/SCOLHKG/Real/GroupRidgeComplexity.lean`.

## Paired Smoke Result

The seed-0 paired smoke completes without retry, invalid selection, or a
material runtime increase.  FactorShock is truly feasible with zero violation
and feasible regret `0.00825`; Inventory is truly feasible with zero violation
and feasible regret `0.02345`.  Wall times are 16.1 and 14.5 minutes,
respectively, compared with 17.0 and 15.2 minutes for V26b seed 0.

The learned constraint complexity differs in the intended way.  Inventory
selects penalties `(1e-4, 1e-4, 1000)` for `[A, A^2, N]`, giving effective
dimension `8.86`; it therefore does not reproduce the V26 hard-cap exclusion
of curvature.  FactorShock's ordered branch also selects a high-dimensional
constraint fit on its altered sequential design, but the latent task posterior
assigns that branch only `7e-6` decision weight and preserves the stable local
recommendation.  This is why both the continuous complexity learner and the
latent structural posterior are required.

The smoke is functional evidence only.  Promotion remains conditional on the
predeclared 14-shard paired gate; no quality claim is inferred from one seed.

## Full Gate Result

The 14-shard gate completes without failure, invalid complexity selection, or
oracle use, but does not pass the quality rule.

| Held-out domain | True feasible | False feasible | Mean violation | Median feasible regret |
|---|---:|---:|---:|---:|
| FactorShock | 7/7 | 0/7 | 0.00000 | 0.00825 |
| Inventory | 3/7 | 0/7 | 0.00840 | 0.00847 |

Inventory remains below the predeclared 4/7 threshold.  It is therefore not
promoted, Queue remains unopened, and no result is pushed as a new baseline.

The failure is no longer a capacity-selection failure.  All seven Inventory
terminal pools contain truly feasible, low-regret candidates, but the same
four seeds rank an observed truly infeasible point above them.  Per-expert
truth-only audits show that the ordered cumulative expert has median safety
ranking AUC `0.882`, yet predictive-likelihood task weights can assign most
decision mass to an expert with poor boundary ranking on the held-out seed.

A frozen-posterior counterfactual tests penalties 5--100, minimum robust
violation, minimum theory margin, prior/posterior averaging, median/trimmed
expert violation, and rank consensus.  None exceeds 3/7; rank consensus merely
swaps one successful seed for another.  V27 is therefore rejected as a
capacity fix that does not repair task-aligned sequential learning.
