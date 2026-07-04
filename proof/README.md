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
- `SCOLHKG/Real/Certification.lean`: real-valued GP-confidence plus variance
  upper-bound certification implication.
- `SCOLHKG/Real/HVD.lean`: real-valued residual-square concentration event to
  HVD oracle-inequality implication.
- `SCOLHKG/Real/RidgeHVD.lean`: concrete ridge empirical minimizer plus
  residual-square uniform concentration to HVD oracle inequality.
- `SCOLHKG/Real/KG.lean`: real-valued exact KG and additive-proxy relation.
- `SCOLHKG/Real/AdditiveApproxKG.lean`: uniform additive-acquisition
  approximation implies a `2 eta` exact-KG optimality gap.
- `SCOLHKG/Real/InformationGainRegret.lean`: information-gain radius and
  finite-budget regret accounting.
- `SCOLHKG/Real/SafeRegret.lean`: real-valued finite-budget safe simple-regret
  implication.
- `SCOLHKG/Measure/ProbabilityEvents.lean`: mathlib
  `ProbabilityTheory` layer connecting conditional variance, Chebyshev, and
  finite union bounds to GP/residual concentration events.
- `SCOLHKG/Measure/SubGaussianConfidence.lean`: sub-Gaussian one-sided and
  centered confidence events over finite and adaptive candidate sets.
- `SCOLHKG/Measure/PosteriorKG.lean`: posterior expected terminal gain defined
  as a Bochner integral.
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

Still to formalize at the deeper mathlib probability/analysis layer:

1. Random-policy occupancy-risk decomposition using conditional expectation
   and the trajectory exposure maps `A(T),N(T)`.
2. Instantiation that the chosen GP posterior kernel and noise model satisfy
   the sub-Gaussian parameters used by `SubGaussianConfidence.lean`.
3. Distribution-specific residual-square concentration constants for the HVD
   ridge estimator, beyond the deterministic ridge oracle step.
4. Exact SC-OLH-KG one-step value-of-information theorem tied to the
   implemented update equations.
5. Kernel-specific information-gain upper bounds for the selected feature
   spaces and adaptive candidate policy.

## Current Math-Depth Assessment

This is still not a complete 10/10 mathematical package, but the gap has
narrowed: the generic sub-Gaussian confidence, adaptive candidate union bounds,
ridge-HVD oracle step, additive-to-exact KG approximation, and information-gain
regret accounting are now Lean-proved.  The remaining hard work is model
instantiation: the exact GP posterior kernel assumptions, distribution-specific
HVD concentration constants, posterior-update exact KG theorem, and trajectory
occupancy decomposition.
