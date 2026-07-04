# SC-OLH-KG Lean4 Proof Workspace

This directory is the local proof workspace for the current `KG_op` project.
It is intentionally kept at the project root instead of being moved into any
external proof directory.

Lean4 is the source of truth for formal proofs.  Markdown files are only the
human-readable roadmap and code-to-theory map.

## Build

```bash
cd proof
lake build
```

The project now depends on mathlib `v4.31.0`.  The first build may download
mathlib and its cache; later builds should be fast.

## Core Files

- `SCOLHKG/Variance.lean`: formal cumulative-risk algebra, including the
  `A^T Lambda A + N^T B N + N^T omega + floor` decomposition and truncation
  bookkeeping lemmas.
- `SCOLHKG/Information.lean`: formal algebraic core of the information
  refinement proposition.
- `SCOLHKG/Certification.lean`: formal ordered-arithmetic core of conservative
  chance certification.
- `SCOLHKG/Scalarization.lean`: inherited bi-objective-to-scalar bridge:
  weak Pareto dominance is preserved by nonnegative weighted scalarization.
- `SCOLHKG/HVD.lean`: deterministic oracle-inequality and conservative
  variance-estimation skeleton for HVD.
- `SCOLHKG/Optimization.lean`: safe recommendation and simple-regret
  consequences once mean confidence and variance upper bounds hold.
- `SCOLHKG/KG.lean`: exact KG maximizer bookkeeping and the condition under
  which an additive KG proxy is exact.
- `SCOLHKG/Real/CumulativeRisk.lean`: real-valued cumulative-risk algebra.
- `SCOLHKG/Real/ConditionalVariance.lean`: finite-partition law of total
  variance over real numbers, proved by algebraic expansion.
- `SCOLHKG/Real/GPRUpdate.lean`: code-level rank-one GPR update bridge, showing
  the KG slope used in `core/kg.py` is the standard-shock posterior mean
  response.
- `SCOLHKG/Real/OccupancyDecomposition.lean`: policy trajectory mixture risk
  decomposition into occupancy cumulative risk, occupancy remainder, and
  between-trajectory explained variance.
- `SCOLHKG/Real/Certification.lean`: real-valued GP-confidence plus variance
  upper-bound certification implication.
- `SCOLHKG/Real/HVD.lean`: real-valued residual-square concentration event to
  HVD oracle-inequality implication.
- `SCOLHKG/Real/HVDImplementation.lean`: code-level HVD bridges for residual
  square records, nonnegative cumulative beta predictions, clipping, and
  certification variance.
- `SCOLHKG/Real/PosteriorRecommendation.lean`: robust posterior recommendation
  logic used in `SingleOLHKGAlgorithm._solve_posterior_recommendation`.
- `SCOLHKG/Real/RidgeHVD.lean`: concrete ridge empirical minimizer plus
  residual-square uniform concentration to HVD oracle inequality.
- `SCOLHKG/Real/KG.lean`: real-valued exact KG and additive-proxy relation.
- `SCOLHKG/Real/LineEnvelopeKG.lean`: certificate-level line-envelope KG
  theorem for the `compute_h` calculation once active hull intervals are
  certified.
- `SCOLHKG/Real/LineEnvelopeStack.lean`: endpoint and tail-slope bridge showing
  that the Python `validate_h_certificate` checks imply active-line
  certificates.
- `SCOLHKG/Real/LineEnvelopeAlgorithm.lean`: step-level stack-loop
  formalization for the Python `cuts.pop()` and `cuts.append(z)` mutations.
- `SCOLHKG/Real/LineEnvelopeGlobal.lean`: final global stack dominance
  invariant implies atom certificates and exact line-envelope KG, without a
  Python runtime-validator assumption.
- `SCOLHKG/Real/LineEnvelopeIntersection.lean`: concrete `compute_h`
  intersection arithmetic, popped-cell takeover, and right-tail split
  certificates for the stack loop.
- `SCOLHKG/Real/AdditiveApproxKG.lean`: uniform additive-acquisition
  approximation implies a `2 eta` exact-KG optimality gap.
- `SCOLHKG/Real/InformationGainRegret.lean`: information-gain radius and
  finite-budget regret accounting.
- `SCOLHKG/Real/FiniteKernelInformationGain.lean`: finite-kernel scalar
  information-gain accumulation and uniform-cap bound.
- `SCOLHKG/Real/SafeRegret.lean`: real-valued finite-budget safe simple-regret
  implication.
- `SCOLHKG/Measure/ProbabilityEvents.lean`: mathlib
  `ProbabilityTheory` layer connecting conditional variance, Chebyshev, and
  finite union bounds to GP/residual concentration events.
- `SCOLHKG/Measure/GPKernelConfidence.lean`: finite-kernel posterior error as
  a weighted sum of independent sub-Gaussian noise, with explicit
  `sum_i w_i^2 c_i` parameter and finite/adaptive confidence.
- `SCOLHKG/Measure/SubGaussianConfidence.lean`: sub-Gaussian one-sided and
  centered confidence events over finite and adaptive candidate sets.
- `SCOLHKG/Measure/ResidualSquareConcentration.lean`: bounded residual-square
  distribution constants and finite concentration events for HVD.
- `SCOLHKG/Measure/ResidualSquareTail.lean`: generic sub-exponential or sharper
  residual-square tail interface, closed-form default radius inversion, and
  finite HVD concentration.
- `SCOLHKG/Measure/PosteriorKG.lean`: posterior expected terminal gain defined
  as a Bochner integral.
- `SCOLHKG/Measure/PosteriorUpdateKG.lean`: exact posterior-update SC-OLH-KG
  value as an integral over updated terminal certified value.
- `SCOLHKG/Measure/PosteriorSamplingCandidates.lean`: random posterior-sampled
  candidate sets controlled by deterministic finite envelope pools.
- `SCOLHKG/Measure/SafeRegretEvent.lean`: high-probability transfer from bad
  events to safe-regret failure events.
- `theory.md`: theorem statements, assumptions, and proof sketches for the
  manuscript-level theory.
- `code_map.md`: mapping from mathematical objects to the current
  `SC-OLH-KG/` implementation.

## Formalization Status

Implemented in Lean4 without `sorry`, `admit`, or `axiom`:

1. Fixed-trajectory cumulative variance decomposition algebra.
2. Information-refinement reduction of apparent variance as algebraic lemma.
3. Low-rank/effective-risk truncation bookkeeping lemmas.
4. Conservative chance-feasibility certification arithmetic.
5. Weighted scalarization monotonicity for the inherited bi-objective bridge.
6. Deterministic HVD oracle-inequality skeleton.
7. Safe recommendation and safe simple-regret implication.
8. Exact KG/additive-surrogate bookkeeping.
9. Real-valued cumulative-risk decomposition and truncation lemmas.
10. Two-cell and arbitrary finite-partition laws of total variance, including
    the information-refinement variance-reduction corollary.
11. Real-valued chance certification from GP confidence plus conservative
    standard-deviation upper bound.
12. Real-valued residual-square concentration event to HVD oracle bound.
13. Real-valued exact KG/additive-proxy theorem.
14. Real-valued finite-budget safe simple-regret implication.
15. General mathlib law of total variance via `condVar`.
16. Chebyshev-derived GP confidence bad-event probability.
17. Finite-candidate simultaneous confidence via union bound.
18. Chebyshev-derived residual-square concentration bad-event probability.
19. Posterior exact KG expected gain as an integral.
20. High-probability safe-regret event transfer.
21. Sub-Gaussian one-sided tail confidence from mathlib Chernoff bound.
22. Two-sided centered sub-Gaussian GP-confidence events.
23. Finite and adaptive candidate-set sub-Gaussian union bounds.
24. Ridge-HVD residual-square oracle inequality from a concrete ridge
    minimizer and uniform concentration event.
25. Additive acquisition to exact KG approximation gap (`2 eta` theorem).
26. Information-gain radius to finite-budget safe-regret accounting.
27. Finite-kernel posterior error sub-Gaussian parameter
    `sum_i w_i(x)^2 c_i`.
28. Finite/adaptive candidate confidence for that finite-kernel GP posterior
    error model.
29. Bounded residual-square HVD concentration constants via Hoeffding's lemma.
30. Policy trajectory occupancy-risk decomposition with explicit occupancy
    remainder.
31. Exact posterior-update SC-OLH-KG expected value theorem and maximizer
    optimality.
32. Code-level rank-one GPR update/KG-slope identity.
33. Code-level HVD residual-square, nonnegative beta, clipping, and
    certification-variance guards.
34. Robust posterior recommendation implication.
35. Finite-kernel scalar information-gain accumulation and cap bound.
36. Certificate-level line-envelope KG exactness for `compute_h`.
37. Random posterior-sampled candidate events controlled by deterministic
    envelope pools.
38. Generic sub-exponential/sharper residual-square tail interface.
39. Default sub-exponential residual-square radius wrapper.
40. Stack-hull endpoint/tail certificate bridge for `compute_h`.
41. Stack-loop pop/push cut-order preservation for `compute_h`.
42. Closed-form sub-exponential default radius inversion.
43. Final global stack dominance invariant to exact line-envelope KG.
44. Concrete `compute_h` intersection arithmetic:
    `z=(a_old-a_new)/(b_new-b_old)` gives old-line dominance on the left and
    new-line dominance on the right.
45. Popped finite envelope cells are certificate-preservingly taken over by
    the new line under the Python pop branch condition, with all processed
    lines dominated at every point of the popped interval.
46. Right-tail split branch constructs certified old finite and new right-tail
    cells under the Python break/push branch condition, with all processed
    lines dominated on the finite left piece and the whole right tail.
47. Posterior-score candidate selection from sampled coefficients is contained
    in the deterministic finite candidate pool, so its bad event inherits the
    adaptive sub-Gaussian envelope bound.
48. Finite-kernel information gain equals a determinant/log-product-style cap
    for the finite product ratio, and this cap feeds the regret accounting.

Still to formalize at the deeper mathlib probability/analysis layer:

1. Complete the recursive list/fold theorem that the full sorted-line
   imperative `compute_h` stack, after all candidates are processed, produces
   the final global dominance invariant.  The intersection arithmetic and the
   pop/split certificate-preserving branch steps are now Lean-proved.
2. Formalize a full multivariate-normal coefficient sampling distribution for
   `posterior_sample_candidates`; the posterior-score selector containment and
   random-candidate envelope event are already Lean-proved.
3. Use the optional Python exact posterior-update estimator in large-seed
   benchmarks before deciding whether it replaces the additive runner.
4. Kernel/feature-specific determinant upper bounds for the selected feature
   spaces, beyond the current finite product-ratio identity.
5. Traffic trajectory encoder/log model formalization, after the empirical
   traffic case is rebuilt with fresh seeds.

## Current Math-Depth Assessment

This is still not a complete 10/10 mathematical package, but the gap has
narrowed: finite-kernel GP confidence, bounded residual-square constants,
trajectory occupancy decomposition, ridge-HVD oracle step, additive-to-exact KG
approximation, posterior-update exact KG, line-envelope KG certificates,
validator-level stack-hull certificates, stack-loop cut-order preservation,
final-stack global dominance to exact KG, concrete intersection/pop/split
branch certificates, closed-form residual-square tail radii, posterior-score
candidate containment, random candidate envelope events, finite product-ratio
information gain, and information-gain regret accounting are now Lean-proved.
The remaining hard work is narrower: finish the full recursive sorted-stack
fold/output theorem, add a complete multivariate-normal sampler layer if the
paper needs that exact distribution claim, and decide empirically whether the
optional exact estimator should replace the additive default.
