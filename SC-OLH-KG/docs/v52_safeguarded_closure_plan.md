# V52 Safeguarded Statistical Closure

## Purpose

V51 remains the immutable promoted baseline. V52 is not allowed to obtain
certificate coverage by relaxing the theory bound, deleting replications, or
overriding the observed-terminal Bayes decision. It targets four measured
closure errors while preserving a literal V51 fallback:

1. certificate nonvacuity;
2. active-HVD excitation;
3. finite candidate-pool coverage in the observable risk coordinate;
4. sequential one-step myopia.

The completed V51 audit separates the failure mechanisms. FactorShock lacks
safe-good candidates in part of the generated pool. Inventory and Queue
already contain safe-good candidates, but their robust constraint-mean
epistemic radius is too large. Active cumulative-HVD designs are also
ill-conditioned after removing feature scale, although improving HVD
conditioning alone cannot solve the mean-radius failure.

## Immutable Baseline

The V51 action set consists of:

- the canonical Sobol continuation;
- three lowest posterior-Bayes-risk unobserved policies;
- every observed policy still eligible for replication.

Its fantasy update refits the same GPR, source-discrepancy posterior, robust
mean head, and cumulative HVD used by the terminal decision. Its final action
is selected from charged observed policies. V52 does not alter any of these
rules when its policy-improvement switch is disabled.

## V52 Action Superset

V52 appends at most two unobserved policies to the V51 action set:

- `certificate_depth`: the smallest current theory margin;
- `psi_coverage`: the largest standardized distance from observed cumulative
  risk coordinates.

The V51 subset is recorded explicitly. Exact one-step VOI is evaluated on the
union with common predictive samples. Let `a_0` maximize estimated VOI over
the V51 subset and `a_+` maximize it over the union. V52 chooses `a_+` only if

`estimated_VOI(a_+) > estimated_VOI(a_0) + 2 * eta_action`.

Otherwise it executes `a_0`. On the event that every estimated VOI differs
from exact VOI by at most `eta_action`, a switched action is no worse than the
V51 action in exact one-step posterior value.

## Guarded Rollout

The optional rollout compares the V51 fallback with at most three challenger
actions from the expanded set. It uses the same observed-terminal Bayes-risk
value and the same GPR, task-posterior, robust-mean, and HVD fantasy updates as
V51. It never invokes the legacy finalist override.

For estimated depth-two values, lower is better. A rollout challenger replaces
the V51 fallback only when

`estimated_value(challenger) + 2 * eta_rollout < estimated_value(V51)`.

Thus a switch is posterior-value noninferior under the declared uniform error
event. This is a conditional policy-improvement statement, not an
unconditional simulator-performance guarantee.

## Excitation Audit

The previous raw excitation constant changes under column rescaling. V52
therefore reports both:

- the original active Gram spectrum and Lean-compatible raw `kappa`;
- a column-RMS-normalized Gram spectrum, normalized `kappa`, feature radius,
  and condition number.

The normalized quantities are diagnostic only. They do not change the fitted
HVD or certification bound.

## Promotion Gate

The first paired gate uses the same frozen source archive, initial policies,
target seeds, `d=1000`, `N=20`, and `n0=10` as V51. It compares:

- immutable V51;
- action-superset V52;
- guarded-rollout V52;
- joint V52.

Promotion requires all of the following on the three-domain sentinel matrix:

- 15/15 true-feasible recommendations;
- zero adaptive losses and zero false certificates;
- median feasible regret no worse than V51;
- paired losses no greater than paired wins;
- improved FactorShock safe-good pool support or demonstrated rescue;
- improved normalized excitation without worse variance calibration;
- positive certificate coverage, or a documented posterior-depth reduction
  if the `N=20` certificate remains information-theoretically vacuous.

Only a passing challenger proceeds to `N=80` and broader-seed evidence. The
promoted V51 record and its historical results are never relabeled as V52.
