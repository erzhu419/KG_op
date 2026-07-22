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
