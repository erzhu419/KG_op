# V29 Budgeted Finalist Replication

## Failure Identified By V28

V28 learned a materially different safe-decision posterior on held-out
Inventory (`TV(Q_pred,Q_safe)=0.784`) but did not pass the seed-0 smoke. The
selected policy had true chance margin `+0.000551`; it was not falsely
certified because the theory-certified set was empty. A frozen post-run audit
found 11 truly feasible policies in the same terminal pool, including one
with margin `-0.0509`. The failure is therefore selection risk among terminal
actions, not candidate coverage or a missing representation.

The V27 structured action was feasible, while V28 moved most safe-decision
mass to `universal_coordinate` and selected a better-objective but marginally
unsafe observed action after only one noisy evaluation. Static terminal
penalties had already failed the seven-seed counterfactual audit. V29 changes
the information allocation rather than tuning another terminal weight.

## Budgeted Ranking-And-Selection Contract

V29 reserves the last `R` evaluations from the existing total budget `N`; it
does not add simulator calls. At stage `N-R`, it freezes a finite finalist set
using only the current posterior:

1. the current minimum posterior Bayes-risk action;
2. the action with minimum nominal expected positive violation under
   `Q_safe`;
3. if requested, the minimum robust expected-violation actions needed to fill
   the finite set.

Targets are fixed before their new labels are observed. Allocation first
evaluates unobserved finalists and then balances replicate counts until every
target reaches `r_min`. If all targets already have enough evidence, ordinary
exact KG continues. Domain names, analytic boundaries, target truth, and
uncharged simulator calls are absent from the selector.

For finalist `x` with `r_x` charged observations, V29 computes

```text
U_x = ybar_g(x)
      + z_alpha sigma_shrunk(x)
      + z_delta sigma_shrunk(x) / sqrt(r_x)
      - tau.
```

`sigma_shrunk` combines replicate variance with the configured noise-floor
prior. A theory-certified action still has priority. Only when the robust
theory set is empty may the replicated fallback select the lowest empirical
objective among finalists with `U_x <= 0`; if none satisfy that bound, it
chooses the smallest `U_x` lexicographically before objective. The fallback
is reported as empirical replicated evidence, never as a theory certificate.

## Controlled Gate

- Default behavior remains off, so V28 and every earlier result are exact
  ablations.
- Unit tests must cover budget accounting, frozen targets, checkpoint resume,
  conservative replicated margin arithmetic, and no-oracle diagnostics.
- FactorShock/Inventory seed-0 is run first with `R=3`, two finalists, and two
  observations per finalist.
- The unchanged full gate is FactorShock 7/7 with zero violation and Inventory
  at least 4/7 with at most one false-feasible result.
- Queue remains closed until the full gate passes.

## Proof Obligations

- the replicated upper margin implies the true chance constraint on the joint
  mean-confidence and variance-upper event;
- the finite safety-first rule minimizes the replicated upper margin whenever
  no finalist passes it;
- target freezing prevents post-label selection leakage;
- forced allocations consume at most the reserved in-budget stages.

## Result

The seed-0 challenger is rejected. FactorShock remains feasible, but the
Inventory finalist set frozen at stage 17 contains no truly feasible member.
Both targets reach the required two observations and the empirical
safety-first rule behaves consistently; it cannot recover an action excluded
before replication began. No 7+7 gate is dispatched and V29 is not promoted.

Any successor must replace one-time freezing with a predictable adaptive
active set, ideally with source-prior-supported nominations from every finite
structural expert and a family-wise confidence guarantee. Changing only
`delta`, the replicate count, or the terminal penalty does not address the
identified failure.
