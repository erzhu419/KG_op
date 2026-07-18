import Mathlib

namespace SCOLHKG.Real

open scoped BigOperators

/-!
Implementation-facing facts for the hierarchical source-shape HVD challenger.
Each source domain contributes a cumulative-risk shape that is nonnegative at
every policy because its local coefficients are nonnegative and its shared
quadratic block is PSD. Target variance evidence updates only nonnegative
mixture weights and their posterior covariance. For ordinary online new-point
evaluations, the prequential squared innovation is conservative in expectation;
replications continue to use within-policy sample variance.
-/

noncomputable def sourceShapeMixture {k : ℕ}
    (weight shape : Fin k → ℝ) : ℝ :=
  ∑ i, weight i * shape i

theorem sourceShapeMixture_nonnegative {k : ℕ}
    (weight shape : Fin k → ℝ)
    (hWeight : ∀ i, 0 ≤ weight i)
    (hShape : ∀ i, 0 ≤ shape i) :
    0 ≤ sourceShapeMixture weight shape := by
  unfold sourceShapeMixture
  exact Finset.sum_nonneg fun i _ => mul_nonneg (hWeight i) (hShape i)

/- The no-transfer component is a constant pooled target variance.  Adding it
to the source dictionary preserves the same cone used by cumulative HVD. -/
noncomputable def sourceShapeWithTargetNull {k : ℕ}
    (sourceWeight sourceShape : Fin k → ℝ)
    (nullWeight targetPooledVariance : ℝ) : ℝ :=
  sourceShapeMixture sourceWeight sourceShape
    + nullWeight * targetPooledVariance

theorem sourceShapeWithTargetNull_nonnegative {k : ℕ}
    (sourceWeight sourceShape : Fin k → ℝ)
    (nullWeight targetPooledVariance : ℝ)
    (hWeight : ∀ i, 0 ≤ sourceWeight i)
    (hShape : ∀ i, 0 ≤ sourceShape i)
    (hNullWeight : 0 ≤ nullWeight)
    (hPooled : 0 ≤ targetPooledVariance) :
    0 ≤ sourceShapeWithTargetNull
      sourceWeight sourceShape nullWeight targetPooledVariance := by
  unfold sourceShapeWithTargetNull
  exact add_nonneg
    (sourceShapeMixture_nonnegative sourceWeight sourceShape hWeight hShape)
    (mul_nonneg hNullWeight hPooled)

/- `sharedTaskPosteriorExpectation` is the common latent-task expectation used
by both the coefficient-mean mixture and the source HVD dictionary.  The two
models may expose different component values, but not different task mass. -/
noncomputable def sharedTaskPosteriorExpectation {k : ℕ}
    (posteriorWeight componentValue : Fin k → ℝ) : ℝ :=
  ∑ i, posteriorWeight i * componentValue i

theorem shared_task_posterior_preserves_nonnegative_shape {k : ℕ}
    (posteriorWeight varianceShape : Fin k → ℝ)
    (hWeight : ∀ i, 0 ≤ posteriorWeight i)
    (hShape : ∀ i, 0 ≤ varianceShape i) :
    0 ≤ sharedTaskPosteriorExpectation posteriorWeight varianceShape := by
  unfold sharedTaskPosteriorExpectation
  exact Finset.sum_nonneg fun i _ =>
    mul_nonneg (hWeight i) (hShape i)

noncomputable def scalarShapePosteriorVariance
    (priorInformation targetInformation : ℝ) : ℝ :=
  1 / (priorInformation + targetInformation)

theorem scalarShapePosteriorVariance_nonnegative
    (priorInformation targetInformation : ℝ)
    (hPrior : 0 < priorInformation)
    (hTarget : 0 ≤ targetInformation) :
    0 ≤ scalarShapePosteriorVariance priorInformation targetInformation := by
  unfold scalarShapePosteriorVariance
  positivity

theorem target_information_shrinks_scalar_shape_variance
    (priorInformation targetInformation addedInformation : ℝ)
    (hPrior : 0 < priorInformation)
    (hTarget : 0 ≤ targetInformation)
    (hAdded : 0 ≤ addedInformation) :
    scalarShapePosteriorVariance
        priorInformation (targetInformation + addedInformation)
      ≤ scalarShapePosteriorVariance priorInformation targetInformation := by
  unfold scalarShapePosteriorVariance
  have hDenominator : 0 < priorInformation + targetInformation := by linarith
  apply one_div_le_one_div_of_le hDenominator
  linarith

noncomputable def sourceShapeUpperGuard
    (z posteriorVariance : ℝ) : ℝ :=
  z * Real.sqrt posteriorVariance

theorem sourceShapeUpperGuard_nonnegative
    (z posteriorVariance : ℝ)
    (hZ : 0 ≤ z) :
    0 ≤ sourceShapeUpperGuard z posteriorVariance := by
  unfold sourceShapeUpperGuard
  exact mul_nonneg hZ (Real.sqrt_nonneg posteriorVariance)

theorem sourceShapeUpperGuard_mono
    (z oldVariance newVariance : ℝ)
    (hZ : 0 ≤ z)
    (hVariance : newVariance ≤ oldVariance) :
    sourceShapeUpperGuard z newVariance
      ≤ sourceShapeUpperGuard z oldVariance := by
  unfold sourceShapeUpperGuard
  exact mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hVariance) hZ

theorem target_information_shrinks_scalar_shape_guard
    (z priorInformation targetInformation addedInformation : ℝ)
    (hZ : 0 ≤ z)
    (hPrior : 0 < priorInformation)
    (hTarget : 0 ≤ targetInformation)
    (hAdded : 0 ≤ addedInformation) :
    sourceShapeUpperGuard z
        (scalarShapePosteriorVariance
          priorInformation (targetInformation + addedInformation))
      ≤ sourceShapeUpperGuard z
        (scalarShapePosteriorVariance priorInformation targetInformation) := by
  apply sourceShapeUpperGuard_mono z _ _ hZ
  exact target_information_shrinks_scalar_shape_variance
    priorInformation targetInformation addedInformation hPrior hTarget hAdded

/- Both terms of the unified evaluate-or-replicate VOI are measured as a
decrease of a square-root confidence radius.  This elementary bridge prevents
adding a raw GPR variance reduction to an HVD parameter-variance reduction. -/
noncomputable def squareRootRadiusReduction
    (scale oldVariance newVariance : ℝ) : ℝ :=
  scale * (Real.sqrt oldVariance - Real.sqrt newVariance)

theorem squareRootRadiusReduction_nonnegative
    (scale oldVariance newVariance : ℝ)
    (hScale : 0 ≤ scale)
    (hReduction : newVariance ≤ oldVariance) :
    0 ≤ squareRootRadiusReduction scale oldVariance newVariance := by
  unfold squareRootRadiusReduction
  exact mul_nonneg hScale (sub_nonneg.mpr (Real.sqrt_le_sqrt hReduction))

/- A prequential predictor is fixed before the new observation.  For a finite
zero-mean noise law, its squared innovation has second moment equal to the true
noise second moment plus squared mean-prediction bias.  This is the exact
finite-law bridge used by `prequential_upper`; the extra bias can only make the
variance evidence conservative. -/
noncomputable def weightedNoiseSecondMoment {k : ℕ}
    (probability noise : Fin k → ℝ) : ℝ :=
  ∑ i, probability i * noise i ^ 2

noncomputable def weightedPrequentialResidualSecondMoment {k : ℕ}
    (probability noise : Fin k → ℝ) (bias : ℝ) : ℝ :=
  ∑ i, probability i * (noise i + bias) ^ 2

theorem weighted_prequential_residual_second_moment_identity {k : ℕ}
    (probability noise : Fin k → ℝ) (bias : ℝ)
    (hMass : (∑ i, probability i) = 1)
    (hCentered : (∑ i, probability i * noise i) = 0) :
    weightedPrequentialResidualSecondMoment probability noise bias
      = weightedNoiseSecondMoment probability noise + bias ^ 2 := by
  unfold weightedPrequentialResidualSecondMoment weightedNoiseSecondMoment
  calc
    (∑ i, probability i * (noise i + bias) ^ 2)
        = ∑ i, (probability i * noise i ^ 2
            + 2 * bias * (probability i * noise i)
            + bias ^ 2 * probability i) := by
              apply Finset.sum_congr rfl
              intro i hi
              ring
    _ = (∑ i, probability i * noise i ^ 2) + bias ^ 2 := by
          rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
          rw [← Finset.mul_sum, ← Finset.mul_sum]
          rw [hCentered, hMass]
          ring

theorem true_variance_le_prequential_residual_second_moment {k : ℕ}
    (probability noise : Fin k → ℝ) (bias : ℝ)
    (hMass : (∑ i, probability i) = 1)
    (hCentered : (∑ i, probability i * noise i) = 0) :
    weightedNoiseSecondMoment probability noise
      ≤ weightedPrequentialResidualSecondMoment probability noise bias := by
  rw [weighted_prequential_residual_second_moment_identity
    probability noise bias hMass hCentered]
  exact le_add_of_nonneg_right (sq_nonneg bias)

end SCOLHKG.Real
