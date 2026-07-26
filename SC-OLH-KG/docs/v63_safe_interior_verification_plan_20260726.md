# V63 Cumulative-Risk Safe-Interior Verification Gate

## Motivation

V62 removed all recommendation losses but certified `57/60` fresh cases.
The remaining failures were not failures of the noncentral-t quantile bound:
the frozen two-policy shortlist omitted a deeper safe point already present
in the charged initial atlas.

V63 keeps the V51 search, posterior Bayes-risk primary, V62 certificate,
familywise error budget, and `64/96` replication schedule unchanged. It
replaces posterior rank 2 by one frozen safe-interior support policy.

## Frozen Selector

Let `I0` be the charged, source-informed initial atlas and let `p_D(x)` be the
posterior probability that the chance constraint is violated. Define

`E_epsilon = {x in I0 : p_D(x) <= min_(u in I0) p_D(u) + epsilon}`.

With `epsilon=0.05`, V63 selects from `E_epsilon` the policy farthest from the
Bayes-risk primary after coordinatewise standardization of the common
cumulative-risk coordinate `psi(x)=(A(x),N(x))`.

The selector reads only the frozen posterior, the already charged initial
atlas, and the oracle-free cumulative-risk provider. It reads no target truth
or verification sample. Both policies are frozen before independent
verification.

## Statistical Contract

- Candidate 1 and candidate 2 each receive error probability `0.025`.
- Candidate 1 uses 64 independent replications.
- Candidate 2 is tested only after candidate 1 fails and uses 96 independent
  replications.
- The first certified candidate is deployed; if neither certifies, the V51
  primary is retained.
- The familywise false-deployment probability remains at most `0.05` for any
  pre-verification frozen selector.
- All verification calls are charged to the target budget.

## Registered Fresh Protocol

1. Materialize source-only designs for seeds `0..59` and verify byte identity
   of seeds `0..39`.
2. Pair V51 and V63 on untouched target seeds `40..59`.
3. Use `d=1000`, `n0=10`, and `N_search=13`.
4. Require identical initial designs, online actions, and primary policies.
5. Require exact selector, provider, verification, and budget contracts.

## Promotion Gate

- Complete `3 domains x 20 seeds x 2 variants`.
- `60/60` terminal certificates and zero false certificates.
- Zero feasibility losses and zero feasible-regret losses versus V51.
- At least one strict paired gain.
- Identical V51/V63 search trajectories and primary actions.
