# GPT Math V1 Gap Audit

## Scope

This audit compares `markdown/GPT_math_v1.md` with the current promoted V51
implementation, Lean workspace, and frozen experiment gates. It is a theory
and implementation-contract document, not manuscript text.

The central correction is statistical rather than architectural. Promoted V51
already has one posterior state, one observed-terminal Bayes loss, and one
evaluate-or-replicate action space. The missing layer was a finite-sample path
from replicated cumulative-risk observations to HVD estimation, certificate
nonvacuity, and true safe regret.

## Version Boundary

| Contract | Decision behavior | Diagnostics | Experiments |
|---|---|---|---|
| `v51_conditional_v1` | Promoted observed-terminal V51 | Previous diagnostics | Historical V51, Gate B at launch, Gate C |
| `v51_statistical_closure_v2` | Identical to promoted V51 | Adds active-HVD excitation and explicit contract IDs | New audit runs only |

The v2 work does not alter candidate generation, posterior updates, exact VOI,
evaluate-or-replicate selection, or recommendation. It adds proofs and
read-only diagnostics. An old result may be *audited using* a later theorem,
but it must not be relabeled as having run under that later contract.

`performance/manifests/theory_contract_registry.json` is the contract ledger.
The paper matrix and every new v2 audit command carry both contract IDs before
the first target evaluation; the result row repeats them. A resumed legacy
checkpoint keeps `unversioned` rather than inheriting a later theory label.

## Point-by-Point Finding

### 1. Shortlist and Monte Carlo rates

**Partly outdated.** `Measure/ExactMCConcentration.lean` already proves a
finite-pool sub-Gaussian union bound, defines an explicit `ExactMCSchedule`, and
gives the resulting pool-level failure probability. Gate C subsequently froze
32 antithetic MC samples and a 32-action shortlist.

**Still empirical.** Coverage of an arbitrary full or continuous action space
by the generated finite pool cannot follow from MC concentration. It remains a
candidate/proposal coverage term and is reported explicitly in the final
regret theorem.

### 2. HVD identification, estimation, and misspecification

**Genuinely missing in conditional v1; added in statistical v2.**

`Real/FiniteSampleHVD.lean` now proves:

- identifiability under active exposure-design excitation;
- a finite-sample ridge parameter inequality from replicated variance-target
  accuracy;
- prediction error controlled by active parameter error and feature size;
- an explicit orthogonal/misspecification remainder;
- both a sound variance upper bound and an over-conservatism upper bound.

`Measure/FiniteSampleHVDConcentration.lean` additionally proves that a finite
replication schedule with sub-Gaussian variance-target error yields the exact
uniform event consumed by that oracle inequality, with a finite-union failure
budget over all active target records.

The theorem uses the **active calibration dimension**. With few target
replications, Python freezes a source-learned cumulative-risk shape and learns
a scalar or small source-shape mixture. It does not claim to recover every raw
coefficient. `OrthogonalHVD.diagnostics()` now reports the raw and projected
Gram geometry, rank, minimum eigenvalue, effective replication degrees of
freedom, and whether the active law is identifiable.

### 3. Certificate nonvacuity

**Logical gap fixed; empirical question remains open.**

`Real/CertificateNonvacuity.lean` proves that a point with true safety depth
`Delta > 0` is certified after explicit mean and replication budgets. The
result separates mean estimation from cumulative-HVD variance error. Without
a positive variance-floor smoothness assumption, an `r^(-1/2)` variance rate
becomes an `r^(-1/4)` standard-deviation rate, yielding a conservative
`Delta^(-4)` replication threshold. This is deliberately exposed rather than
hidden.

Gate B subsequently passed at `N=80` with replication cap 20: all three domains
issued at least one true certificate, with three certified points and zero
false certificates across 15 runs. Coverage remains low (3/339 evaluated
points), so the evidence establishes nonvacuity, not broad certificate recall.

### 4. True finite-budget safe regret

**Missing in conditional v1; added for the finite observed terminal pool.**

`Real/EndToEndSafeRegret.lean` compares the recommendation with a true safe
optimum and decomposes regret into:

1. representation error;
2. HVD error;
3. transfer error;
4. observed-pool comparator coverage;
5. action-shortlist error;
6. twice the uniform MC error;
7. sequential/myopia error.

It also proves true feasibility from the implementation certificate.
`Measure/StatisticalClosure.lean` supplies the joint finite-union probability
wrapper, so component concentration bounds imply a high-probability safe
regret statement. Pool coverage and sequential myopia remain explicit because
they cannot be removed by algebra.

### 5. Source-to-target transfer

**Partly fixed, with an unavoidable condition made explicit.**

`Real/TransferGeneralization.lean` combines the finite-task PAC-Bayes event
with an expert-wise source-target domain-shift term. It yields a target-risk
bound containing source empirical risk, KL complexity, source sample count,
and weighted domain discrepancy.

There is no unconditional negative-transfer theorem. Such a theorem is
impossible when source models agree on the source design but differ on an
unseen target region; `Real/BoundaryExcitation.lean` formalizes that lower
bound. The current posterior has a null expert and target-only discrepancy
updates, but claiming domination of target-only learning would require extra
bounded-score or target-excitation assumptions not satisfied automatically by
the Gaussian generalized score.

## Code-Theory Map

| Mathematical quantity | Python evidence | Lean object |
|---|---|---|
| Active cumulative design | `cumulative_statistical_design` | `ActiveHVDExcitation` |
| Replicated variance accuracy | replication-only HVD records and residual tail diagnostics | `UniformReplicatedVarianceAccuracy` plus residual-square concentration |
| Projected ridge fit | source-shape scalar/mixture or full projected IRLS | `ApproximateActiveHVDRidgeFit` |
| Coordinate misspecification | residual/source-shape guard | `active_hvd_misspecification_absolute` |
| Certificate depth and budget | Gate B budget curve | `FiniteCertificateBudget` |
| Terminal score bridge | observed-terminal posterior Bayes risk | `UniformTerminalScoreBridge` |
| Pool coverage | post-run truth audit only | `SafeObservedComparatorCoverage` |
| MC uniformity | Gate C nested antithetic audit | `ExactMCSchedule` |
| Source transfer | finite task posterior and source discrepancy | `finite_source_to_target_pac_bayes` |

## Remaining Obligations

The mathematical gap is no longer “missing theorem statements.” The remaining
obligations are empirical or assumption audits:

- Gate B passed only at `N=80, replication_cap=20`; lower-budget certificates
  remain vacuous and the observed 3/339 coverage must be reported.
- New v2 diagnostic runs must report positive active excitation where HVD is
  claimed identifiable.
- The main matrix must estimate pool-coverage and sequential terms rather than
  silently set them to zero.
- A sharper `Delta^(-2)` certificate rate would require proving and validating
  a positive variance-floor Lipschitz condition for the HVD standard deviation.
- No-negative-transfer dominance remains conditional on observable target
  discrepancy/excitation; it is not claimed unconditionally.

## Current Assessment

The GPT review correctly identified the missing statistical layer, but its MC
assessment predates the explicit schedule and Gate C. Statistical closure v2
raises the theory from conditional decision consistency to a finite-sample,
truth-relative theorem family. It does not by itself make a vacuous empirical
certificate useful; Gate B and the separate v2 diagnostic run decide that.
