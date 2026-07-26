# V60 Frozen Ordered-Shortlist Verification Gate

## Purpose

V59 produced 11/15 sound fixed-policy certificates and zero false
certificates. Its four uncertified runs already contained a deeper posterior
rank-2 policy in the terminal action universe. V60 tests whether a finite,
precommitted posterior shortlist can remove certificate vacuity without
changing V51 search or relaxing statistical coverage.

## Registered Protocol

1. Run the exact V51 search contract with `N_search=13`, `n0=10`,
   `exact_mc_samples=8`, and the same frozen source-informed archive.
2. Before drawing any verification sample, freeze the two lowest posterior
   Bayes-risk observed policies in order.
3. Allocate family-wise error `delta=0.05` equally:
   `delta_1=delta_2=0.025`.
4. Verify rank 1 using 48 independent iid Gaussian replications and the V59
   Student-t/chi-square bound.
5. If rank 1 certifies, deploy it and stop. Otherwise verify rank 2 on a
   disjoint candidate-specific stream and deploy it only if it certifies.
6. If neither policy certifies, retain the optimization recommendation but
   report `abstained_uncertified`; do not call it certified.
7. Verification samples never update the GPR, HVD, source weights, candidate
   generator, posterior ranking, or search recommendation.

The charged target budget is 61 calls when rank 1 certifies and 109 calls
when rank 2 must be tested. The union bound controls the false deployed
certificate probability by `delta_1+delta_2=0.05`; independence between the
two certification events is not required for this guarantee.

## Registered Gate

- All 15 V51/V60 paired keys are complete.
- Search design, online action sequence, and pre-verification rank-1 policy
  exactly match V51.
- All 15 deployed policies receive a terminal certificate.
- No false terminal certificate and no feasibility loss occur.
- Feasible regret is pairwise noninferior to V51.
- Every search and verification call is separately counted.
- No target oracle enters ranking, verification, stopping, or deployment.

V60 may promote the terminal deployment protocol, but it does not replace
V51 as an equal-total-budget optimization baseline because the verification
suffix has a variable additional cost.
