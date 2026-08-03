import Mathlib

namespace SCOLHKG.Real

/-!
# Observable-provider cumulative-HVD scale calibration

The repaired source head fits the cumulative cone in the observable
`psi = (A, N)` coordinate.  Source-task replication then calibrates one
positive multiplicative scale.  This file records the deterministic inversion
and max-over-source-tasks steps used by the Python implementation.
-/

def providerVariance (shape scale : ℝ) : ℝ :=
  scale * shape

def providerVarianceUpper (shape scaleUpper : ℝ) : ℝ :=
  scaleUpper * shape

noncomputable def aggregateScaleUpper (aggregate lowerQuantile : ℝ) : ℝ :=
  aggregate / lowerQuantile

theorem aggregate_scale_upper_sound
    {trueScale aggregate lowerQuantile : ℝ}
    (hQuantile : 0 < lowerQuantile)
    (hAggregate : lowerQuantile * trueScale ≤ aggregate) :
    trueScale ≤ aggregateScaleUpper aggregate lowerQuantile := by
  unfold aggregateScaleUpper
  exact (le_div_iff₀ hQuantile).2 (by
    simpa [mul_comm] using hAggregate)

theorem provider_variance_upper_sound
    {shape trueScale scaleUpper : ℝ}
    (hShape : 0 ≤ shape)
    (hScale : trueScale ≤ scaleUpper) :
    providerVariance shape trueScale
      ≤ providerVarianceUpper shape scaleUpper := by
  unfold providerVariance providerVarianceUpper
  exact mul_le_mul_of_nonneg_right hScale hShape

theorem two_source_task_max_scale_sound_left
    {trueScale firstUpper secondUpper : ℝ}
    (hFirst : trueScale ≤ firstUpper) :
    trueScale ≤ max firstUpper secondUpper := by
  exact le_trans hFirst (le_max_left _ _)

theorem two_source_task_max_scale_sound_right
    {trueScale firstUpper secondUpper : ℝ}
    (hSecond : trueScale ≤ secondUpper) :
    trueScale ≤ max firstUpper secondUpper := by
  exact le_trans hSecond (le_max_right _ _)

theorem two_source_task_max_provider_variance_sound
    {shape trueScale firstUpper secondUpper : ℝ}
    (hShape : 0 ≤ shape)
    (hFirst : trueScale ≤ firstUpper) :
    providerVariance shape trueScale
      ≤ providerVarianceUpper shape (max firstUpper secondUpper) := by
  exact provider_variance_upper_sound hShape
    (two_source_task_max_scale_sound_left hFirst)

theorem aggregate_scale_then_provider_variance_sound
    {shape trueScale aggregate lowerQuantile : ℝ}
    (hShape : 0 ≤ shape)
    (hQuantile : 0 < lowerQuantile)
    (hAggregate : lowerQuantile * trueScale ≤ aggregate) :
    providerVariance shape trueScale
      ≤ providerVarianceUpper shape
          (aggregateScaleUpper aggregate lowerQuantile) := by
  exact provider_variance_upper_sound hShape
    (aggregate_scale_upper_sound hQuantile hAggregate)

theorem max_calibrated_provider_variance_nonnegative
    {shape firstUpper secondUpper : ℝ}
    (hShape : 0 ≤ shape)
    (hFirst : 0 ≤ firstUpper) :
    0 ≤ providerVarianceUpper shape (max firstUpper secondUpper) := by
  unfold providerVarianceUpper
  exact mul_nonneg (le_trans hFirst (le_max_left _ _)) hShape

end SCOLHKG.Real
