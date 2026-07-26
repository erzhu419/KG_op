# V62 Exact Gaussian-Quantile Verification Gate

## Motivation

V61 separately upper-bounded the Gaussian mean and standard deviation, then
added the two bounds inside the chance constraint. This is sound but spends
each candidate's error budget twice through a Bonferroni split.

V62 instead directly bounds the target quantile

`q_alpha = mu + z_(1-alpha) sigma`.

For `n` iid Gaussian replications with sample mean `Ybar` and unbiased sample
standard deviation `S`,

`sqrt(n) * (q_alpha - Ybar) / S`

has a noncentral Student-t distribution with `n-1` degrees of freedom and
noncentrality `z_(1-alpha) sqrt(n)`. Therefore

`Ybar + k_(n,alpha,delta) S`

is an exact one-sided `(1-delta)` upper confidence bound for the chance
quantile, where

`k = nct.ppf(1-delta, n-1, z_(1-alpha)sqrt(n)) / sqrt(n)`.

This is a different finite-sample certificate, not a relaxed threshold.

## Development Audit

Using only the V61 frozen shortlist and its precommitted independent
verification streams, the V62 design `rank1=64`, `rank2=96` produced:

- `60/60` certificates and `0` false certificates;
- `15 win / 0 loss / 45 tie` relative to V51;
- 3 feasibility rescues and 0 feasibility losses;
- rank counts of `20/0` for FactorShock, `8/12` for Inventory, and `17/3`
  for Queue.

This audit selected the V62 schedule. It is development evidence only.

## Registered Fresh Protocol

1. Use the same frozen source archive, source-informed `n0=10`, `d=1000`,
   and search budget `N_search=13`.
2. Pair V51 and V62 on previously unused target seeds `20..39`.
3. Freeze posterior Bayes-risk ranks 1 and 2 before verification.
4. Allocate familywise `delta=0.05` as `0.025` to each frozen candidate.
5. Verify rank 1 with 64 independent replications.
6. If rank 1 fails, verify rank 2 with 96 replications on its disjoint stream.
7. Deploy the first certified policy; otherwise retain the V51 optimization
   recommendation and record abstention.
8. Do not update GPR, HVD, source experts, proposal, shortlist, or search from
   verification samples.

The charged target budget is 77 calls after a rank-1 certificate and 173
calls after a rank-2 attempt.

## Promotion Gate

- Complete V51/V62 pairing for 3 domains and 20 fresh seeds.
- Identical target design, online action sequence, and frozen rank-1 policy.
- Exact implementation and theory contracts on every V62 row.
- `60/60` terminal certificates and 0 false certificates.
- 0 feasibility losses and 0 feasible-regret losses against V51.
- At least one strict paired improvement.
- Every verification call is included in the reported target budget.

## Fresh Result

The registered seeds `20..39` completed all `120/120` paired tasks:

- search identity and all implementation contracts passed;
- performance was `10 win / 0 loss / 50 tie`;
- there were no feasibility losses and no false certificates;
- `57/60` terminal policies were certified.

The formal gate therefore failed only its `60/60` coverage requirement. The
three abstentions were Inventory seeds 21 and 32 and Queue seed 37. Post-rank
truth audit showed that the Inventory pools contained a deeper safe point at
posterior rank 3, while Queue contained one at rank 4. V62 is retained as the
exact verification method; V63 changes only the pre-verification shortlist
selector.
