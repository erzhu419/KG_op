# V53 Numerical-Fidelity Audit, 2026-07-22

Run `scolh_v53_mc_fidelity_s10_20260722_02` completed 60/60 tasks with
`exit_code=0` and a `DONE` marker. It paired MC8 and MC32 at one charged online
decision for FactorShock, Inventory, and Queue, using frozen source-informed
initial designs for seeds 0 through 9.

The audit passed all structural checks:

- 30/30 expected pairs were present;
- implementation/theory contracts matched;
- initial-design and source-archive fingerprints matched;
- active action fingerprints were identical;
- no target oracle entered the online trace.

It did not pass the MC8 stability gate:

- risk top-1 agreement: `0.9333`;
- risk pairwise agreement: `0.8697`;
- certificate top-1 agreement: `0.5667`;
- certificate pairwise agreement: `0.8818`.

The maximum absolute MC8/MC32 differences were `274.9484` for Bayes-risk
reduction and `0.177734` for certificate-deficit reduction. The risk maximum
came from a posterior state whose ranking still agreed, showing that one
cross-domain absolute error constant is scale-sensitive and would make the
guard vacuous.

No V53 promotion or sentinel result is claimed from this run. Full exact
finite-expert marginalization remains a reference estimator; the operational
replacement is nested factorized RQMC with an explicit selector-discrepancy
diagnostic and error term.

## Exact-enumeration deployment audit

Run `scolh_v53_stratified_mc_fidelity_s10_20260722_01` was launched only as a
runtime pilot and then cancelled 60/60 without failures. The product posterior
contained 49 mean/HVD expert pairs, so MC8 and MC32 required 392 and 1,568
fantasy refits per action. Representative tasks remained healthy at full
12-core utilization, but projected wall time was incompatible with a mainline
gate. No score or fidelity claim is drawn from this cancelled run.

The replacement run uses `factorized_rqmc_nested`, keeps the same target
posterior and active action set, and reports selector L1 discrepancy alongside
MC8/MC32 score agreement.

## Factorized-RQMC result

Run `scolh_v53_rqmc_mc_fidelity_s10_20260722_01` completed 60/60 tasks with no
failures and restored operational wall time. The selector-plan, frozen-design,
contract, and active-action checks all passed. Relative to random expert
sampling, factorized RQMC improved certificate top-1 agreement from `0.5667`
to `0.7333`, certificate pairwise agreement from `0.8818` to `0.9297`, and
risk pairwise agreement from `0.8697` to `0.8921`. Risk top-1 agreement
remained `0.9333`.

MC8 still failed the preregistered stability thresholds. The largest raw
MC8/MC32 score differences were `525.6693` for Bayes risk and `0.106468` for
certificate deficit. Applying one global raw-unit multiplier would require
`eta_risk=657.0866`, which is cross-domain scale sensitive and would make the
V53 switch guard nearly vacuous. This run therefore closes V53-v1 as a failed
numerical contract; it is not promoted.

## V53-v2 normalized contract

V53-v2 divides each score by the positive deterministic scale
`max(1, |current terminal value|_infinity)` computed from the same frozen
pre-update posterior. Both action rankings and the exact two-eta guard are
unchanged when the error radius is expressed in normalized units. Raw scores,
raw-equivalent error radii, and both scales remain in the trace.

The next registered fidelity run is
`scolh_v53_normalized_rqmc_mc32_mc128_s10_20260722_01`. It compares nested MC32
against MC128, rather than treating MC32 as accurate merely because MC8 failed.
No V53-v2 sentinel or promotion claim is permitted before that gate completes.


## V53-v2 completed MC32/MC128 gate

Run `scolh_v53_normalized_rqmc_mc32_mc128_s10_20260722_01` completed 60/60
jobs and all 30 low/high pairs. Frozen initial designs, active action sets,
selector plans, declared contracts, and target-oracle exclusions all matched.
MC32 versus MC128 achieved risk/certificate top-action agreement of
`0.9333/0.9667` and pairwise agreement of `0.9564/0.9764`, so the registered
ranking-stability thresholds passed.

That pass is not sufficient for policy activation. Every current-terminal
normalization scale equaled `1.0`, while the maximum risk-score discrepancy was
`275.0097`; the registered `1.25` multiplier therefore produced
`eta_risk=343.7621`. The certificate radius was `0.03334`. Replaying the exact
V53 guard on all 30 MC32 cells admitted a supplemental risk action in `0/30`
cells and switched in `0/30`. A sentinel with these radii would be a trivial
V51 fallback, not evidence that certificate-directed policy improvement works.
V53-v2 is therefore closed as ranking-stable but guard-vacuous and is not sent
to the sentinel.

## V53-v3 bounded per-fantasy utility

V53-v3 keeps the literal raw-score V51 fallback and the final posterior Bayes
decision unchanged. For each fantasy terminal value `L_a(omega)`, it evaluates

```text
Delta_a(omega) = clip(
    (L_current - L_a(omega)) / max(1, |L_current|_infinity), -1, 1)
```

and integrates `Delta_a` before applying the two-error guards. The same
construction is used independently for Bayes-risk and certificate-deficit
scores. Clipping is inside the expectation; it is not cosmetic clipping of an
already unstable Monte Carlo mean. Raw terminal values and raw VOI scores are
retained beside bounded policy scores in every trace. The operational RQMC
error radius remains an empirical nested-prefix calibration, while Lean proves
the per-fantasy score is in `[-1,1]` and the guarded noninferiority implication
on the declared uniform-error events.

The next disjoint run is
`scolh_v53_bounded_rqmc_mc32_mc128_s10_20260722_01`. Only a complete,
nonvacuous MC32/MC128 gate may authorize
`scolh_v53_bounded_constrained_certificate_s5_20260722_01`.

The bounded run completed 60/60 tasks and all 30 disjoint MC32/MC128 pairs.
All implementation contracts, frozen initial designs, active action sets, and
selector plans matched. Bounded scores were finite and in range. Risk top-1
and pairwise agreement were `0.8667` and `0.9515`; certificate top-1 and
pairwise agreement were `0.9667` and `0.9764`. The maximum nested-prefix
differences were `0.04829` for risk and `0.02667` for certificate, giving the
preregistered radii `eta_risk=0.06036` and `eta_certificate=0.03334` and a
numerical recommendation of MC32.

The separate nonempty-guard audit nevertheless failed. With the required
`2 eta_risk=0.12072` threshold, no supplemental action was risk-admissible in
any of the 30 MC32 cells and no certificate-directed switch was possible. The
largest supplemental risk advantage was `0.04015` on FactorShock, `0.06121`
on Inventory, and `-0.13923` on Queue. Thus bounded utility repaired numerical
scale and fidelity but did not repair policy nonvacuity. V53-v3 is rejected as
a challenger, its five-seed sentinel is not launched, and V51 remains the
promoted baseline.

## V54 support result and V55 numerical contract

The completed 15-cell Pareto-support replacement improved finite action
coverage, but no FactorShock action or randomized action mixture jointly beat
the literal V51 risk-best fallback in both terminal heads. V54 is closed as a
structurally infeasible fallback-relative contract; MC512 was not launched.

V55 uses the same bounded scores, joint fantasy update, oracle-free action
support, and nested factorized RQMC, but forms an action-specific absolute
current-relative lower confidence bound for each terminal reduction. A
positive two-head bound is sufficient for exact current-state Pareto
improvement on the declared numerical-error event. The MC128 activation gate
must select a positive-bound action in at least three of five seeds in every
domain. Only then may a disjoint MC512 run test 100 percent selected-action
coverage, at least 95 percent all-action joint coverage, and zero selected
reference regressions. These thresholds were frozen before V55 target runs.

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
