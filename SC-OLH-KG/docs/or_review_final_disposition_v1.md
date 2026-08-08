# OR Review Final Evidence Disposition V1

This document freezes the scientific interpretation of the OR review
remediation experiments. It is not manuscript prose and may not be relaxed in
response to presentation preferences.

## Final method identity

The candidate contribution is a **source-scored structural initial design for
ordered policy profiles**, followed by a replaceable target optimizer and an
independent terminal verifier. The source archive selects a finite subset from
a preconstructed, dimension-equivariant profile library. The method is not a
general-purpose high-dimensional Bayesian optimizer.

KG, posterior sampling, functional SCBO, and SAASBO are backend comparisons.
Cumulative HVD is a secondary calibration result because its matched causal
comparison improved variance-shape estimation but did not improve feasible
recovery, regret, false certification, or verification cost.

## Energy decision

Energy is necessary as a negative external control. Removing it would conceal
the most informative applicability-boundary result. Tuning the atlas after
observing Energy V2 or V3 would invalidate the frozen holdout, so no such tuning
is allowed.

- Energy V2: source atlas certified 60/90 decisions; generic DCT certified
  74/90 and target-only functional SCBO certified 62/90.
- Energy V3: source atlas certified 60/90; generic DCT certified 70/90,
  random low frequency 61/90, natural constant grid 57/90, raw Sobol 52/90,
  and target-only functional SCBO 58/90.
- In V3, the region-level source-minus-generic-DCT safe-rate difference was
  -0.0787 with bootstrap 95% CI [-0.1373,-0.0200], with zero region wins,
  three losses, and two ties.
- All Energy arms had zero observed false certificates.
- The chronological and physically nonoverlapping audits are descriptive.
  They do not upgrade the empirical-distribution certificate into an iid
  future-calendar guarantee.

Energy therefore supports profile structure and independent verification, but
does not support a claim that source scoring universally improves a generic
structured design.

## Benchmark-fit audit

The benchmark-overfitting concern is reduced, not eliminated.

1. Code, task laws, analysis endpoints, and claim gates were frozen before the
   randomized confirmatory outcomes.
2. Eight target regimes independently vary rank, frequency support, coordinate
   order, grid regularity, smoothness, sparse high-frequency activity, and
   target misspecification.
3. The primary matrix contains 2,880 cells and the OFAT matrix 8,640 cells.
4. Source atlas achieved mean task feasible rate 0.9167 and certified-success
   rate 0.4729, versus 0.8021 and 0.3000 for raw Sobol. Its mean penalized loss
   was 0.1627 versus 0.3371 for raw Sobol.
5. The advantage is not universal. Irregular-grid, misspecified-target, and
   sparse-high-frequency regimes expose failures or reversals. Target-only
   functional SCBO also produced three empirical false certificates, all in
   frequency-support-shift tasks.
6. Schema-blind and coordinate controls show that no target oracle label is
   required, but declared profile semantics remain a substantive information
   assumption. Descriptor conditioning did not help and is excluded from the
   primary method.
7. Energy independently reverses the source-versus-generic-DCT ordering.

The admissible claim is therefore task-law conditional: source scoring helps
when held-out targets share enough profile-coordinate and low-frequency
structure with the registered source population. It is not a distribution-free
or arbitrary-vector-space result.

## Cost and baseline disposition

The 384-call source archive is always reported separately from target search
and independent verification.

- At the one-target equal preverification cost, the source atlas used 384
  source plus 10 target calls and achieved certified-success rate 0.4625.
- Target-only controls using 394 target calls achieved 0.5375 for generic DCT,
  0.4563 for random low frequency, 0.5313 for natural structure, and 0.5375
  for raw Sobol.
- The source archive is consequently an amortized multi-target investment, not
  a one-target sample-efficiency victory.
- Eight native transfer pipelines were run in 480 matched cells. RGPE certified
  30/60 and MetaBO 28/60; all eight methods were 0/20 on Queue. These are
  end-to-end transfer comparisons, while shared-atlas backend tables are named
  as backend comparisons only.

One target-only functional SCBO equal-cost cell failed because canonical
posterior sampling returned duplicate points. The cell remains an unsuccessful
outcome in the denominator; it is not rerun or silently dropped.

## Verifier and theory disposition

The primary verifier is the preregistered exact all-success binomial rule over
a frozen shortlist with Bonferroni spending. At required safety probability
0.95, shortlist size 3, and familywise error 0.05, its minimum valid budget is
80. Its power is low near the threshold and decreases with larger fixed budgets
when the true probability is below one. Exact Clopper--Pearson thresholds are
reported only as preregistered sensitivity and future protocol, not selected
post hoc.

Lean formalization establishes the finite implementation bridges for profile
coordinates, rank recovery, farthest-first coverage, task-law coverage,
binomial certification, and finite decision guarantees without
`sorry`, `admit`, or project axioms. The task-coverage theorem remains
conditional on a declared task law, source-target discrepancy, and safety
margin. Those assumptions are empirical scope conditions, not proved facts
about arbitrary future domains.

## Frozen claim gates

The manuscript must:

- call the decision object an ordered policy profile;
- report nominal grid dimension separately from effective profile rank;
- separate source, target-search, verification, total, and amortized calls;
- state the Energy negative result in the same paragraph as the randomized
  positive result;
- distinguish task-level inference from within-task algorithmic seeds;
- describe cumulative HVD as secondary calibration evidence;
- label d=10,000 as discretization refinement of a structured profile, not as
  unconstrained 10,000-dimensional optimization;
- retain adverse regimes, the three false functional certificates, and the one
  algorithmic failure;
- avoid universal, distribution-free, causal, or high-dimensional-SOTA claims.

The evidence package is publication-ready only under these restrictions. Its
compact registry is `paper_artifacts/or_review/final_evidence_registry_v1.json`.
