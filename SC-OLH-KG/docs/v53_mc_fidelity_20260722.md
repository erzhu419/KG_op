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

No V53 promotion or sentinel result is claimed from this run. The next
fidelity gate replaces sampled finite expert identity with exact posterior
marginalization (`stratified_expert_nested`). This is the estimator already
covered by `SCOLHKG.Real.StratifiedExpertKG`; only the conditional Gaussian
expectation remains numerical.
