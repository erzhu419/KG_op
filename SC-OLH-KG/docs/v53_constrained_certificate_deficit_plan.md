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

The revised numerical path uses `stratified_expert_nested`. Finite expert
identity is summed exactly with posterior mass, as formalized in
`SCOLHKG.Real.StratifiedExpertKG`; only the conditional two-dimensional
Gaussian innovation is approximated. Stage-keyed antithetic rows are reused by
the risk and certificate heads, and MC8 is an exact prefix of MC32 at a fixed
posterior state. The revised disjoint gate runs

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

After freezing nonzero `eta_risk` and `eta_certificate`, run the paired
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
