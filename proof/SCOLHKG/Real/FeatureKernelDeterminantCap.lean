import Mathlib
import SCOLHKG.Real.KernelDeterminantBridge

namespace SCOLHKG.Real

open scoped BigOperators

/-!
Feature/kernel-specific finite information-gain caps.

For the finite feature maps used in the implementation, the posterior variance
ratio can be bounded by a concrete scalar cap, e.g. from a feature-norm bound
and an observation-noise floor.  These lemmas turn that code-facing ratio bound
into the determinant/log-product cap consumed by the safe-regret theorem.
-/

noncomputable def uniformRatioDeterminantCap
    {Time : Type*}
    (steps : Finset Time)
    (ratioCap : ℝ) : ℝ :=
  ∏ _t ∈ steps, (1 + ratioCap)

theorem finiteKernelProductRatio_le_uniform_ratio_cap
    {Time : Type*}
    (steps : Finset Time)
    (variance noise : Time → ℝ)
    {ratioCap : ℝ}
    (hpos :
      ∀ t ∈ steps, 0 < 1 + variance t / noise t)
    (hratio :
      ∀ t ∈ steps, variance t / noise t ≤ ratioCap) :
    finiteKernelProductRatio steps variance noise
      ≤ uniformRatioDeterminantCap steps ratioCap := by
  unfold finiteKernelProductRatio uniformRatioDeterminantCap
  exact Finset.prod_le_prod
    (fun t ht ↦ le_of_lt (hpos t ht))
    (fun t ht ↦ by
      have h := hratio t ht
      linarith)

theorem finiteInformationGain_le_uniform_ratio_log_cap
    {Time : Type*}
    (steps : Finset Time)
    (variance noise : Time → ℝ)
    {ratioCap : ℝ}
    (hpos :
      ∀ t ∈ steps, 0 < 1 + variance t / noise t)
    (hratio :
      ∀ t ∈ steps, variance t / noise t ≤ ratioCap) :
    finiteInformationGain steps variance noise
      ≤ steps.card * ((1 / 2 : ℝ) * Real.log (1 + ratioCap)) := by
  apply finiteInformationGain_le_uniform_cap
    (steps := steps)
    (variance := variance)
    (noise := noise)
    (cap := (1 / 2 : ℝ) * Real.log (1 + ratioCap))
  intro t ht
  have hEach :=
    scalarInformationGain_le_of_ratio
      (variance := variance t)
      (noise := noise t)
      (varianceCap := ratioCap)
      (noiseFloor := 1)
      (hposLeft := hpos t ht)
      (hratio := by
        simpa using hratio t ht)
  simpa [scalarInformationGain] using hEach

theorem finiteInformationGain_le_uniform_determinant_cap
    {Time : Type*}
    (steps : Finset Time)
    (variance noise : Time → ℝ)
    {ratioCap : ℝ}
    (hpos :
      ∀ t ∈ steps, 0 < 1 + variance t / noise t)
    (hratio :
      ∀ t ∈ steps, variance t / noise t ≤ ratioCap) :
    finiteInformationGain steps variance noise
      ≤ determinantInformationGain
          (uniformRatioDeterminantCap steps ratioCap) := by
  have hEq :=
    finiteInformationGain_eq_determinantInformationGain_product
      steps variance noise hpos
  rw [hEq]
  apply determinantInformationGain_mono
  · unfold finiteKernelProductRatio
    exact Finset.prod_pos (fun t ht ↦ hpos t ht)
  · exact finiteKernelProductRatio_le_uniform_ratio_cap
      steps variance noise hpos hratio

structure FeatureNormKernelCap where
  posteriorVarianceCap : ℝ
  observationNoiseFloor : ℝ
  ratioCap : ℝ

def FeatureNormKernelCap.Valid (c : FeatureNormKernelCap) : Prop :=
  0 < c.observationNoiseFloor ∧
  0 ≤ c.posteriorVarianceCap ∧
  c.ratioCap = c.posteriorVarianceCap / c.observationNoiseFloor

theorem finiteInformationGain_le_feature_norm_kernel_cap
    {Time : Type*}
    (steps : Finset Time)
    (variance noise : Time → ℝ)
    (cap : FeatureNormKernelCap)
    (_hvalid : cap.Valid)
    (hpos :
      ∀ t ∈ steps, 0 < 1 + variance t / noise t)
    (hratio :
      ∀ t ∈ steps,
        variance t / noise t ≤ cap.ratioCap) :
    finiteInformationGain steps variance noise
      ≤ steps.card * ((1 / 2 : ℝ) * Real.log (1 + cap.ratioCap)) := by
  exact finiteInformationGain_le_uniform_ratio_log_cap
    steps variance noise hpos hratio

end SCOLHKG.Real
