import Mathlib
import SCOLHKG.Real.TaskPosterior

namespace SCOLHKG.Real

open scoped BigOperators

/-!
Finite joint task-latent posterior used by
`representation.task_posterior.FiniteTaskLatentPosterior`.

The structural index bundles alignment, basis, GPR, and cumulative HVD. The
sensitivity index controls empirical predictive scale and decision loss. The
source prior may start as a product, while the generalized-Bayes likelihood is
allowed to couple the indices after charged target observations.
-/

def productTaskLatentPrior
    {ι κ : Type*}
    (structurePrior : ι → ℝ)
    (sensitivityPrior : κ → ℝ)
    (z : ι × κ) : ℝ :=
  structurePrior z.1 * sensitivityPrior z.2

noncomputable def jointTaskLatentMass
    {ι κ : Type*}
    (structurePrior : ι → ℝ)
    (sensitivityPrior : κ → ℝ)
    (score : ι × κ → ℝ)
    (eta : ℝ)
    (z : ι × κ) : ℝ :=
  generalizedBayesMass
    (productTaskLatentPrior structurePrior sensitivityPrior)
    score eta z

theorem product_task_latent_prior_sum_eq_one
    {ι κ : Type*} [Fintype ι] [Fintype κ]
    {structurePrior : ι → ℝ}
    {sensitivityPrior : κ → ℝ}
    (hStructure : ∑ i, structurePrior i = 1)
    (hSensitivity : ∑ c, sensitivityPrior c = 1) :
    ∑ z : ι × κ,
      productTaskLatentPrior structurePrior sensitivityPrior z = 1 := by
  rw [Fintype.sum_prod_type]
  simp only [productTaskLatentPrior, ← Finset.mul_sum]
  rw [hSensitivity]
  simp [hStructure]

theorem joint_task_latent_mass_pos
    {ι κ : Type*}
    {structurePrior : ι → ℝ}
    {sensitivityPrior : κ → ℝ}
    {score : ι × κ → ℝ}
    {eta : ℝ}
    (hStructure : ∀ i, 0 < structurePrior i)
    (hSensitivity : ∀ c, 0 < sensitivityPrior c)
    (z : ι × κ) :
    0 < jointTaskLatentMass
      structurePrior sensitivityPrior score eta z := by
  unfold jointTaskLatentMass generalizedBayesMass productTaskLatentPrior
  exact mul_pos (mul_pos (hStructure z.1) (hSensitivity z.2))
    (Real.exp_pos _)

theorem joint_task_latent_normalized_support
    {ι κ : Type*} [Fintype ι] [Fintype κ]
    [Nonempty ι] [Nonempty κ]
    {structurePrior : ι → ℝ}
    {sensitivityPrior : κ → ℝ}
    {score : ι × κ → ℝ}
    {eta : ℝ}
    (hStructure : ∀ i, 0 < structurePrior i)
    (hSensitivity : ∀ c, 0 < sensitivityPrior c)
    (z : ι × κ) :
    0 < normalizeFiniteWeights
      (jointTaskLatentMass
        structurePrior sensitivityPrior score eta) z := by
  apply generalizedBayes_normalized_support
  intro pair
  exact mul_pos (hStructure pair.1) (hSensitivity pair.2)

noncomputable def structureTaskMarginal
    {ι κ : Type*} [Fintype κ]
    (q : ι × κ → ℝ)
    (i : ι) : ℝ :=
  ∑ c, q (i, c)

noncomputable def sensitivityTaskMarginal
    {ι κ : Type*} [Fintype ι]
    (q : ι × κ → ℝ)
    (c : κ) : ℝ :=
  ∑ i, q (i, c)

theorem structure_task_marginal_sum_eq_one
    {ι κ : Type*} [Fintype ι] [Fintype κ]
    {q : ι × κ → ℝ}
    (hJoint : ∑ z, q z = 1) :
    ∑ i, structureTaskMarginal q i = 1 := by
  simp only [structureTaskMarginal]
  rw [← Fintype.sum_prod_type]
  exact hJoint

theorem sensitivity_task_marginal_sum_eq_one
    {ι κ : Type*} [Fintype ι] [Fintype κ]
    {q : ι × κ → ℝ}
    (hJoint : ∑ z, q z = 1) :
    ∑ c, sensitivityTaskMarginal q c = 1 := by
  simp only [sensitivityTaskMarginal]
  rw [← Fintype.sum_prod_type_right]
  exact hJoint

theorem structure_task_marginal_nonneg
    {ι κ : Type*} [Fintype κ]
    {q : ι × κ → ℝ}
    (hJoint : ∀ z, 0 ≤ q z)
    (i : ι) :
    0 ≤ structureTaskMarginal q i := by
  exact Finset.sum_nonneg (fun c _hc => hJoint (i, c))

theorem sensitivity_task_marginal_nonneg
    {ι κ : Type*} [Fintype ι]
    {q : ι × κ → ℝ}
    (hJoint : ∀ z, 0 ≤ q z)
    (c : κ) :
    0 ≤ sensitivityTaskMarginal q c := by
  exact Finset.sum_nonneg (fun i _hi => hJoint (i, c))

/- The sensitivity argument is deliberately absent from the formula. -/
noncomputable def jointTaskTheoryMargin
    {κ : Type*}
    (_sensitivity : κ)
    (mu epistemic aleatoric beta zAlpha tau : ℝ) : ℝ :=
  mu + Real.sqrt beta * Real.sqrt epistemic
    + zAlpha * Real.sqrt aleatoric - tau

theorem task_sensitivity_does_not_change_theory_margin
    {κ : Type*}
    (c₁ c₂ : κ)
    (mu epistemic aleatoric beta zAlpha tau : ℝ) :
    jointTaskTheoryMargin c₁ mu epistemic aleatoric beta zAlpha tau =
      jointTaskTheoryMargin c₂ mu epistemic aleatoric beta zAlpha tau := by
  rfl

noncomputable def authoritativeTaskTheoryMargin
    (sensitivityScale mu epistemic aleatoric beta zAlpha tau : ℝ) : ℝ :=
  mu + Real.sqrt beta
      * Real.sqrt (epistemic * (max 1 sensitivityScale) ^ 2)
    + zAlpha * Real.sqrt aleatoric - tau

def taskBiasAdjustedDecisionMean
    (mu referenceSd signedBias : ℝ) : ℝ :=
  mu + referenceSd * signedBias

noncomputable def authoritativeTaskTheoryMarginWithBias
    (_signedBias sensitivityScale mu epistemic aleatoric beta zAlpha tau : ℝ) : ℝ :=
  authoritativeTaskTheoryMargin
    sensitivityScale mu epistemic aleatoric beta zAlpha tau

theorem authoritative_theory_margin_ignores_signed_bias
    (bias₁ bias₂ sensitivityScale mu epistemic aleatoric beta zAlpha tau : ℝ) :
    authoritativeTaskTheoryMarginWithBias
        bias₁ sensitivityScale mu epistemic aleatoric beta zAlpha tau
      = authoritativeTaskTheoryMarginWithBias
        bias₂ sensitivityScale mu epistemic aleatoric beta zAlpha tau := by
  rfl

theorem conservative_sensitivity_scale_sq_ge_one
    (sensitivityScale : ℝ) :
    1 ≤ (max 1 sensitivityScale) ^ 2 := by
  have hScale : 1 ≤ max 1 sensitivityScale := le_max_left _ _
  nlinarith

theorem authoritative_sensitivity_cannot_relax_theory_margin
    (sensitivityScale mu epistemic aleatoric beta zAlpha tau : ℝ)
    (hEpistemic : 0 ≤ epistemic) :
    jointTaskTheoryMargin Unit.unit
        mu epistemic aleatoric beta zAlpha tau
      ≤ authoritativeTaskTheoryMargin
        sensitivityScale mu epistemic aleatoric beta zAlpha tau := by
  have hScale : 1 ≤ (max 1 sensitivityScale) ^ 2 :=
    conservative_sensitivity_scale_sq_ge_one sensitivityScale
  have hVariance :
      epistemic ≤ epistemic * (max 1 sensitivityScale) ^ 2 := by
    nlinarith
  have hSqrt :
      Real.sqrt epistemic
        ≤ Real.sqrt (epistemic * (max 1 sensitivityScale) ^ 2) :=
    Real.sqrt_le_sqrt hVariance
  have hBeta : 0 ≤ Real.sqrt beta := Real.sqrt_nonneg _
  unfold jointTaskTheoryMargin authoritativeTaskTheoryMargin
  nlinarith

def expertCalibrationPrecisionUpdate
    (priorPrecision observationWeight feature : ℝ) : ℝ :=
  priorPrecision + observationWeight * feature ^ 2

theorem expert_calibration_precision_update_ge_prior
    (priorPrecision observationWeight feature : ℝ)
    (hWeight : 0 ≤ observationWeight) :
  priorPrecision ≤ expertCalibrationPrecisionUpdate
      priorPrecision observationWeight feature := by
  unfold expertCalibrationPrecisionUpdate
  nlinarith [sq_nonneg feature]

noncomputable def authoritativeAdaptiveTaskTheoryMargin
    (_calibrationMean sensitivityScale calibrationVariance
      mu epistemic aleatoric beta zAlpha tau : ℝ) : ℝ :=
  mu + Real.sqrt beta
      * Real.sqrt (
        epistemic * (max 1 sensitivityScale) ^ 2
          + calibrationVariance)
    + zAlpha * Real.sqrt aleatoric - tau

theorem adaptive_theory_margin_ignores_calibration_mean
    (mean₁ mean₂ sensitivityScale calibrationVariance
      mu epistemic aleatoric beta zAlpha tau : ℝ) :
    authoritativeAdaptiveTaskTheoryMargin
        mean₁ sensitivityScale calibrationVariance
        mu epistemic aleatoric beta zAlpha tau
      = authoritativeAdaptiveTaskTheoryMargin
        mean₂ sensitivityScale calibrationVariance
        mu epistemic aleatoric beta zAlpha tau := by
  rfl

theorem adaptive_calibration_covariance_cannot_relax_theory_margin
    (calibrationMean sensitivityScale calibrationVariance
      mu epistemic aleatoric beta zAlpha tau : ℝ)
    (hCalibration : 0 ≤ calibrationVariance) :
    authoritativeTaskTheoryMargin
        sensitivityScale mu epistemic aleatoric beta zAlpha tau
      ≤ authoritativeAdaptiveTaskTheoryMargin
        calibrationMean sensitivityScale calibrationVariance
        mu epistemic aleatoric beta zAlpha tau := by
  have hVariance :
      epistemic * (max 1 sensitivityScale) ^ 2
        ≤ epistemic * (max 1 sensitivityScale) ^ 2
          + calibrationVariance := by
    linarith
  have hSqrt := Real.sqrt_le_sqrt hVariance
  have hBeta : 0 ≤ Real.sqrt beta := Real.sqrt_nonneg _
  unfold authoritativeTaskTheoryMargin
    authoritativeAdaptiveTaskTheoryMargin
  nlinarith

end SCOLHKG.Real
