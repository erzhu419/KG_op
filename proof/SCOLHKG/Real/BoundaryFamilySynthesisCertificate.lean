import Mathlib

namespace SCOLHKG.Real

/-!
Implementation contract for TCB-V4 continuous boundary-family synthesis.

The source-frozen atoms are combined with nonnegative target coefficients.
The target posterior covariance and residual scale enter certification only
through a nonnegative radius.
-/

noncomputable def boundaryFamilySynthesis
    {Family : Type*} [Fintype Family]
    (intercept : ℝ)
    (coefficient atom : Family → ℝ) : ℝ :=
  intercept + ∑ family, coefficient family * atom family

theorem nonnegative_boundary_family_synthesis_monotone
    {Family : Type*} [Fintype Family]
    (intercept : ℝ)
    (coefficient firstAtom secondAtom : Family → ℝ)
    (hCoefficient : ∀ family, 0 ≤ coefficient family)
    (hAtom : ∀ family, firstAtom family ≤ secondAtom family) :
    boundaryFamilySynthesis intercept coefficient firstAtom ≤
      boundaryFamilySynthesis intercept coefficient secondAtom := by
  unfold boundaryFamilySynthesis
  apply add_le_add (le_refl intercept)
  apply Finset.sum_le_sum
  intro family _
  exact mul_le_mul_of_nonneg_left
    (hAtom family) (hCoefficient family)

noncomputable def synthesisParameterVariance
    {Parameter : Type*} [Fintype Parameter]
    (loading : Parameter → ℝ) : ℝ :=
  ∑ parameter, loading parameter ^ 2

theorem synthesisParameterVariance_nonnegative
    {Parameter : Type*} [Fintype Parameter]
    (loading : Parameter → ℝ) :
    0 ≤ synthesisParameterVariance loading := by
  unfold synthesisParameterVariance
  positivity

noncomputable def synthesisPredictiveVariance
    {Parameter : Type*} [Fintype Parameter]
    (loading : Parameter → ℝ)
    (residualScale : ℝ) : ℝ :=
  synthesisParameterVariance loading + residualScale ^ 2

theorem synthesisPredictiveVariance_nonnegative
    {Parameter : Type*} [Fintype Parameter]
    (loading : Parameter → ℝ)
    (residualScale : ℝ) :
    0 ≤ synthesisPredictiveVariance loading residualScale := by
  unfold synthesisPredictiveVariance
  exact add_nonneg
    (synthesisParameterVariance_nonnegative loading)
    (sq_nonneg residualScale)

noncomputable def synthesisBoundaryUpper
    {Parameter : Type*} [Fintype Parameter]
    (mean : ℝ)
    (loading : Parameter → ℝ)
    (residualScale quantile : ℝ) : ℝ :=
  mean + quantile * Real.sqrt (
    synthesisPredictiveVariance loading residualScale)

theorem synthesisBoundaryUpper_ge_mean
    {Parameter : Type*} [Fintype Parameter]
    (mean : ℝ)
    (loading : Parameter → ℝ)
    (residualScale quantile : ℝ)
    (hQuantile : 0 ≤ quantile) :
    mean ≤ synthesisBoundaryUpper
      mean loading residualScale quantile := by
  unfold synthesisBoundaryUpper
  exact le_add_of_nonneg_right
    (mul_nonneg hQuantile (Real.sqrt_nonneg _))

theorem synthesis_certified_recommendation_is_safe
    {Parameter : Type*} [Fintype Parameter]
    (trueMargin mean : ℝ)
    (loading : Parameter → ℝ)
    (residualScale quantile : ℝ)
    (hCoverage :
      trueMargin ≤ synthesisBoundaryUpper
        mean loading residualScale quantile)
    (hCertified :
      synthesisBoundaryUpper mean loading residualScale quantile ≤ 0) :
    trueMargin ≤ 0 := by
  exact hCoverage.trans hCertified

def sourceFrozenSynthesisCoefficient
    {Pilot TargetName Family : Type*}
    (fitFromPilots : (Pilot → ℝ) → Family → ℝ)
    (pilotMargin : Pilot → ℝ)
    (_targetName : TargetName)
    (family : Family) : ℝ :=
  fitFromPilots pilotMargin family

theorem sourceFrozenSynthesisCoefficient_target_name_independent
    {Pilot TargetName Family : Type*}
    (fitFromPilots : (Pilot → ℝ) → Family → ℝ)
    (pilotMargin : Pilot → ℝ)
    (firstTargetName secondTargetName : TargetName)
    (family : Family) :
    sourceFrozenSynthesisCoefficient
        fitFromPilots pilotMargin firstTargetName family =
      sourceFrozenSynthesisCoefficient
        fitFromPilots pilotMargin secondTargetName family := by
  rfl

end SCOLHKG.Real
