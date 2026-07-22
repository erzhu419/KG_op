# V53 Constrained Certificate-Deficit Policy

## Decision object

V53 keeps the promoted V51 observed-action Bayes-risk terminal functional
unchanged. It adds one second posterior terminal functional over the same
history-measurable observed action universe:

```text
C(D) = max(min_{x in O_t} M_D(x), 0)
```

Here `M_D(x)` is the canonical theory chance margin and `O_t` is the
observed terminal action set used by V51. Lower `C(D)` is better. The
certificate score of an evaluate/replicate action is the expected one-step
reduction in `C`.

The candidate set is the literal V52 action superset, but V53 does not use a
rollout. Every candidate is an ordinary paid target action: either evaluate a
new policy or replicate an observed policy.

## Constrained posterior improvement

Let `b` be the highest estimated V51 Bayes-risk-reduction action in the
literal V51 subset. A supplemental action `a` is risk-admissible only if

```text
risk_hat(a) - risk_hat(b) > 2 * eta_risk.
```

Among the risk-admissible actions and `b`, V53 chooses the action with the
largest estimated certificate-deficit reduction. It replaces `b` only if

```text
certificate_hat(a) - certificate_hat(b) > 2 * eta_certificate.
```

Otherwise V53 executes `b`. Thus V51 remains an explicit fallback. The
implementation updates the same GPR, task posterior, and cumulative HVD clone
under each fantasy. It does not read target truth or oracle feasibility.

## Numerical fidelity

The first disjoint gate used `antithetic_nested` common random numbers with a
sampled finite task expert. It completed all 30 pairs, but MC8 failed the
pre-registered stability thresholds: risk/certificate pairwise agreement was
`0.870/0.882`, and certificate top-1 agreement was `0.567`. It therefore did
not authorize a V53 sentinel.

Full finite-expert marginalization remains the exact reference estimator, but
the deployment pilot exposed 49 mean/HVD product experts: MC8 and MC32 expanded
to 392 and 1,568 posterior refits per action. That run was cancelled before it
could consume multiple node-hours and is not fidelity evidence.

The operational estimator is `factorized_rqmc_nested`. A stage-keyed scrambled
Sobol net jointly samples the two Gaussian innovations, the mean expert, and
the HVD expert. It preserves the product posterior without flattening it into a
row-major categorical coordinate, uses exactly MC8 or MC32 refits per action,
and gives MC8 as an exact prefix of MC32. The implementation records the
empirical selector law's L1 distance from the exact finite posterior.
`SCOLHKG.Real.StratifiedExpertKG.finite_rqmc_error_le_conditional_plus_selector_l1`
bounds total integration error by the conditional numerical error plus this
L1 discrepancy times a finite terminal-value bound. The revised disjoint gate
runs

- `d=1000`, `N=11`, `n0=10`;
- FactorShock, Inventory, and Queue;
- frozen-design seeds 0 through 9;
- paired MC8 and MC32 estimators.

Actions are matched by integer-design fingerprint. The analyzer reports
maximum score differences, top-action agreement, and pairwise ranking
agreement. It recommends

```text
eta = max(1e-12, 1.25 * max_observed_abs(MC8 - MC32)).
```

This is an empirical nested-MC calibration, not a proof that MC32 equals the
exact integral. The Lean theorem remains conditional on genuine uniform error
events. If MC8 ranking stability fails, the sequential sentinel uses MC32.

## Sentinel

After freezing nonero `eta_risk` and `eta_certificate`, run the paired
V51, V52 action-superset, and V53 variants with `d=1000`, `N=20`,
`n0=10`, and seeds 0 through 4. The preregistered V53 checks are:

1. all 15 paired runs complete with byte-identical frozen initial designs;
2. 15/15 true-feasible final recommendations;
3. zero adaptive losses and zero false certificates;
4. zero paired losses and noninferior median feasible regret versus V51;
5. domainwise certified log-variance RMSE within 5 percent of V51;
6. FactorShock pool safe-good support and normalized excitation both improve;
7. posterior certificate coverage increases or mean certificate deficit falls;
8. no rollout action is executed.

Passing this sentinel does not promote V53. Promotion still requires the
paper-scale `N=80`, 20-seed gate.

## Formal scope

`SCOLHKG.Real.ConstrainedCertificateDeficit` proves:

- nonnegativity and zero-set semantics of `C(D)`;
- a two-eta estimated score gap implies strict exact score improvement on a
  uniform numerical-error event;
- a V51 fallback or guarded V53 switch is jointly noninferior in exact
  Bayes-risk reduction and exact certificate-deficit reduction.

The theorem is posterior-model internal. Simulator calibration, candidate-pool
coverage, source-target transfer, and certificate nonvacuity remain separate
empirical or probabilistic obligations.


## Post-audit V53-v3 correction

The V53-v2 MC32/MC128 ranking gate passed, but its maximum-error `2 eta` guard
was empty in all 30 audited cells. A fallback-only sentinel is prohibited.
V53-v3 therefore uses bounded per-fantasy improvement utility while preserving
the raw-score V51 fallback, the observed terminal universe, and both terminal
functionals. The preregistered order is now bounded MC32/MC128 fidelity,
nonempty-guard audit, then the original 5-seed sentinel checks. Results from
v1, v2, and v3 keep distinct implementation and theory contract IDs.

## V53-v3 gate outcome

The complete 30-pair bounded MC32/MC128 audit passed numerical fidelity and
recommended MC32, but failed the independently preregistered nonempty-guard
condition. The calibrated risk threshold was `2 eta_risk=0.12072`; the largest
supplemental risk advantage over the literal V51 fallback was only `0.06121`.
Consequently no supplemental action was admissible in any of the 30 cells and
there were zero certificate-directed switches. The sentinel is prohibited
because it would be behaviorally identical to V51. This closes V53-v3 without
promotion and prevents a fidelity-only success from being reported as policy
evidence.

## V54 paired-difference correction

Offline replay of the completed V53-v3 MC32/MC128 pairs showed why merely
replacing the global maximum error by a looser constant would be invalid. Only
two MC32 cells appeared jointly positive in risk and certificate, and both
risk advantages became negative at MC128. V54 therefore keeps the literal V51
fallback and the same two posterior terminal heads, but calibrates each
challenger against that fallback with nested common random numbers:

`r_a = kappa * |Delta_high(a,b) - Delta_prefix(a,b)|`.

A switch requires both the high-fidelity risk advantage and certificate
advantage to exceed their own action-specific radii. Prefix/high score arrays,
radii, admissible indices, and the selected action are retained in the
immutable trace. The Lean contract proves joint posterior improvement
conditional on pairwise radius coverage; MC32/128/512 contraction must validate
that condition empirically.

Before running the V54 selector, the diagnostic
`scolh_v54_action_support_mc128_a12_s5_20260722_01` expands the supplemental
new-point support from two to eight while retaining all four canonical V51
new-point actions and every replicate action. It is a V53-v3-core diagnostic,
not V54 outcome evidence. If no high-MC action jointly dominates the fallback,
the candidate policy must be repaired before any selector sentinel is allowed.

### V54 action-support diagnosis and Pareto repair

The completed 15-cell MC128 support diagnostic retained four literal V51 new
actions and every eligible replication, but its eight additional actions
jointly dominated the fallback in only one cell, Queue seed 4. FactorShock
certificate-directed actions systematically lost Bayes-risk utility;
Inventory usually improved neither head. Therefore no MC512 selector run was
authorized from that action set.

The repaired support policy is
`canonical_plus_posterior_pareto_support`. It keeps the literal V51 subset and
adds oracle-free extreme points for Bayes-risk EI, constrained EI,
chance-boundary proximity and uncertainty, certificate depth, constraint-mean
information, cumulative-HVD information, joint margin information, and
observable `psi=(A,N)` coverage. These quantities only discretize the finite
action set; the shared-fantasy exact posterior refit remains the decision
authority. Run `scolh_v54_pareto_support_mc128_a13_s5_20260722_01` is the
replacement MC128 support diagnostic. MC512 remains prohibited unless all
three domains contain a supplemental joint dominator.

### Joint terminal-head execution reuse

V54 evaluates the unchanged Bayes-risk and certificate-deficit functionals
after each identical posterior fantasy update, but clones and updates the GPR,
task posterior, and HVD only once. The historical two-pass path remains
selectable for equivalence testing. Fixed-CRN regression matches every risk,
certificate, expected-terminal, and selected-action output within `1e-12`;
the Linux `process_fork` path is also covered. A warm local MC32 probe gave
a `1.25x` median speedup. Scheduler task-ensemble timing remains the
paper-relevant performance check.

## V54 Pareto-support outcome and V55 correction

The replacement run
`scolh_v54_pareto_support_mc128_a13_s5_20260722_02` completed all 15 cells
after launch-stage transport failures in the superseded `_01` attempt. The
expanded support produced supplemental fallback-relative joint dominators in
Inventory and Queue, but none in FactorShock. A convex-hull audit also found
that no randomized mixture of the finite actions could jointly dominate the
V51 risk-best fallback in FactorShock: the five max-min gains were all
negative. This is a conflict between two different actions, not a missing
candidate direction or an MC-radius problem. V54 is therefore closed without
an MC512 run or promotion.

The same traces contain a more useful fact in every one of the 15 cells:
there are actions with positive absolute reduction of both current terminal
costs. V55 consequently stops requiring one action to dominate the
single-objective V51 risk-best action in the certificate head. For each action
and each unchanged terminal head it estimates the current-relative reduction
with nested common random numbers and defines

```text
r_R(a) = kappa * |S_R,high(a) - S_R,prefix(a)|
r_C(a) = kappa * |S_C,high(a) - S_C,prefix(a)|
LCB_R(a) = S_R,high(a) - r_R(a)
LCB_C(a) = S_C,high(a) - r_C(a).
```

An action is admissible only when both lower bounds are positive. Among such
actions V55 maximizes the smaller lower bound, then their sum, then the
certificate lower bound. If the admissible set is empty, it executes the
literal V51 risk-best fallback. This is a current-state Pareto improvement
contract; it deliberately does not claim to dominate the V51 fallback in both
heads. `SCOLHKG.Real.ConstrainedCertificateDeficit` proves that a selected
action decreases both exact current terminal costs whenever its two recorded
radii cover the exact numerical errors.

The preregistered sequence is MC32/MC128 activation, disjoint MC512
radius-coverage audit, then a paired V51/V55 sentinel. MC512 is an empirical
reference rather than an exact integral. No V55 performance or promotion
claim is permitted before all three gates pass.

## V55 activation and MC512 reference result

Run `scolh_v55_current_relative_mc128_a13_s5_20260722_01` completed all 15
cells. Every domain selected a positive two-head lower-bound action in all five
seeds, so the preregistered activation gate passed and authorized the disjoint
MC512 reference run.

Run `scolh_v55_current_relative_mc512_a13_s5_20260722_01` also completed all
15 cells. Every action selected at MC128 retained positive Bayes-risk and
certificate-deficit reduction at MC512, and no selected action regressed in
either head. The single nested MC32-prefix radius was not calibrated, however:
it covered only `10/15 = 0.6667` selected actions and `0.6815` of all
action/head comparisons, below the preregistered `1.0/0.95` thresholds. Two
selected cells would require multipliers near `64` and `160` because their
MC32/MC128 difference was accidentally close to zero. Increasing one global
multiplier would therefore be vacuous.

V55 is not promoted from this gate. The empirical direction remains alive
because all 15 selected actions retained the desired sign at MC512, but the
single-prefix difference is rejected as a confidence-radius estimator. The
next numerical contract must use independent scrambled blocks or a
pilot-selection/independent-confirmation design with explicit error spending.
Runtime optimization is tracked separately in
`v55_exact_runtime_optimization_20260722.md`.
