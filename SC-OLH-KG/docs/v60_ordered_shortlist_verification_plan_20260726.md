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

## Completed Result

Run `scolh_v60_ordered_shortlist_n13_r48_s5_20260726_01` completed all 15
tasks without retry.

- Search design and online action fingerprints matched V51 in all 15 pairs.
- 14/15 deployed policies received a terminal certificate, with zero false
  certificates.
- Three rank-2 switches produced three wins, zero losses, and one Queue
  feasibility rescue.
- FactorShock certified 5/5 at rank 1.
- Inventory certified 4/5: two at rank 1 and two at rank 2.
- Queue certified 5/5: four at rank 1 and one at rank 2.
- The only uncertified run was Inventory seed 4. Its rank-2 policy was truly
  feasible, but its 48-replication upper margin remained `+0.00613`.

A post-gate development-only power audit, using the same precommitted
candidate-specific stream, put that margin at `-0.00281` with 96
replications. V61 therefore pre-registers 48 calls for rank 1 and 96 calls
for rank 2, then evaluates that schedule on fresh seeds; it does not relax
`delta`, change the shortlist, or reuse verification labels.
