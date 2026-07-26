# V61 Asymmetric-Power Shortlist Verification Gate

## Purpose

V60 preserved V51 search, improved three terminal outcomes, and raised strict
certificate coverage from 11/15 to 14/15 with zero false certificates. Its
only remaining failure was a safe rank-2 Inventory policy whose 48-call
confidence bound lacked power by `0.00613`.

V61 changes only the precommitted replication schedule. It does not change
search, posterior ranking, the family-wise error level, or the deployment
rule.

## Registered Protocol

1. Pair V51 and V61 under the same source archive, source-informed `n0=10`,
   target dimension `d=1000`, search budget `N_search=13`, and target seeds
   `0..19`.
2. Freeze posterior Bayes-risk ranks 1 and 2 before verification.
3. Split family-wise `delta=0.05` into `0.025` per candidate.
4. Verify rank 1 with 48 independent replications.
5. Only if rank 1 fails, verify rank 2 with 96 replications on its disjoint
   candidate-specific stream.
6. Deploy the first certified policy; otherwise retain the optimization
   recommendation and report abstention.
7. Keep every verification label outside GPR, HVD, source-expert adaptation,
   candidate generation, ranking, and search.

The charged target budget is 61 calls after a rank-1 certificate and 157
calls after a rank-2 attempt. The false-deployment probability remains at
most `0.025+0.025=0.05`; unequal sample counts do not alter the union-bound
argument.

Seeds `0..4` are the development stratum exposed to V59/V60 analysis. Seeds
`5..19` are the fresh confirmatory stratum for the 48/96 schedule.

## Registered Gate

- Complete V51/V61 pairing for three domains and 20 seeds.
- Search design, online action sequence, and frozen rank-1 policy match V51.
- Zero false terminal certificates and zero feasibility losses.
- Pairwise feasible performance is noninferior to V51.
- All 60 V61 policies certify.
- The same four conditions also hold on fresh seeds `5..19` alone.
- Actual verification calls and maximum budgets are reported separately.
- No target oracle enters any charged decision.

## Completed Result

The paired matrix completed all 120 runs with no failures or retries. Search
identity and both implementation contracts passed.

- Terminal certificates: `55/60`, with `0` false certificates.
- Fresh seeds `5..19`: `40/45`, with `0` false certificates.
- Paired performance: `16 win / 3 loss / 41 tie`.
- Fresh performance: `12 win / 3 loss / 30 tie`.
- Feasibility rescues: 3 overall and 2 on fresh seeds.
- Feasibility losses: 0.

The three regret losses occurred when a truly safe rank-1 policy missed the
componentwise Student-t/chi-square certificate by a small positive margin and
the protocol deployed a slightly worse certified rank-2 policy. The five
abstentions also retained truly feasible V51 policies. Thus V61 rejected no
unsafe deployment; its registered gate failed because the componentwise
Bonferroni construction remained underpowered.

All seeds `0..19` are development data after this analysis. They cannot be
used as confirmatory evidence for a successor certificate.
