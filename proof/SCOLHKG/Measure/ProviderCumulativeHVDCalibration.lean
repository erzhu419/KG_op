import Mathlib
import SCOLHKG.Real.ProviderCumulativeHVDCalibration

namespace SCOLHKG.Measure

open MeasureTheory

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

def AggregateScaleLowerTailFailure
    (trueScale lowerQuantile : ℝ)
    (aggregate : Ω → ℝ) : Set Ω :=
  {ω | aggregate ω < lowerQuantile * trueScale}

def AggregateScaleUpperFailure
    (trueScale lowerQuantile : ℝ)
    (aggregate : Ω → ℝ) : Set Ω :=
  {ω |
    SCOLHKG.Real.aggregateScaleUpper
        (aggregate ω) lowerQuantile < trueScale}

theorem aggregate_scale_upper_failure_subset_lower_tail
    (trueScale lowerQuantile : ℝ)
    (aggregate : Ω → ℝ)
    (hQuantile : 0 < lowerQuantile) :
    AggregateScaleUpperFailure
        trueScale lowerQuantile aggregate
      ⊆ AggregateScaleLowerTailFailure
        trueScale lowerQuantile aggregate := by
  intro ω hFailure
  change
    SCOLHKG.Real.aggregateScaleUpper
        (aggregate ω) lowerQuantile < trueScale at hFailure
  change aggregate ω < lowerQuantile * trueScale
  unfold SCOLHKG.Real.aggregateScaleUpper at hFailure
  simpa [mul_comm] using (div_lt_iff₀ hQuantile).1 hFailure

theorem aggregate_scale_upper_failure_probability_le
    (trueScale lowerQuantile delta : ℝ)
    (aggregate : Ω → ℝ)
    [IsFiniteMeasure μ]
    (hQuantile : 0 < lowerQuantile)
    (hLowerTail :
      μ.real (AggregateScaleLowerTailFailure
        trueScale lowerQuantile aggregate) ≤ delta) :
    μ.real (AggregateScaleUpperFailure
        trueScale lowerQuantile aggregate) ≤ delta := by
  exact le_trans
    (measureReal_mono
      (aggregate_scale_upper_failure_subset_lower_tail
        trueScale lowerQuantile aggregate hQuantile))
    hLowerTail

def TwoSourceMaxScaleFailure
    (trueScale : ℝ)
    (firstUpper secondUpper : Ω → ℝ) : Set Ω :=
  {ω | max (firstUpper ω) (secondUpper ω) < trueScale}

def SourceScaleFailure
    (trueScale : ℝ)
    (upper : Ω → ℝ) : Set Ω :=
  {ω | upper ω < trueScale}

theorem two_source_max_scale_failure_subset_union
    (trueScale : ℝ)
    (firstUpper secondUpper : Ω → ℝ) :
    TwoSourceMaxScaleFailure trueScale firstUpper secondUpper
      ⊆ SourceScaleFailure trueScale firstUpper
        ∪ SourceScaleFailure trueScale secondUpper := by
  intro ω hFailure
  have hFirst : firstUpper ω < trueScale := by
    exact lt_of_le_of_lt (le_max_left _ _) hFailure
  exact Set.mem_union_left _ hFirst

theorem two_source_max_scale_failure_probability_le
    (trueScale firstDelta secondDelta : ℝ)
    (firstUpper secondUpper : Ω → ℝ)
    [IsFiniteMeasure μ]
    (hFirst :
      μ.real (SourceScaleFailure trueScale firstUpper) ≤ firstDelta)
    (hSecond :
      μ.real (SourceScaleFailure trueScale secondUpper) ≤ secondDelta) :
    μ.real (TwoSourceMaxScaleFailure
        trueScale firstUpper secondUpper)
      ≤ firstDelta + secondDelta := by
  calc
    μ.real (TwoSourceMaxScaleFailure
        trueScale firstUpper secondUpper)
      ≤ μ.real (
          SourceScaleFailure trueScale firstUpper
            ∪ SourceScaleFailure trueScale secondUpper) :=
        measureReal_mono
          (two_source_max_scale_failure_subset_union
            trueScale firstUpper secondUpper)
    _ ≤ μ.real (SourceScaleFailure trueScale firstUpper)
          + μ.real (SourceScaleFailure trueScale secondUpper) :=
        measureReal_union_le _ _
    _ ≤ firstDelta + secondDelta := by
        linarith

theorem two_source_bonferroni_scale_failure_le_familywise
    (trueScale sourceDelta familywiseDelta : ℝ)
    (firstUpper secondUpper : Ω → ℝ)
    [IsFiniteMeasure μ]
    (hFirst :
      μ.real (SourceScaleFailure trueScale firstUpper) ≤ sourceDelta)
    (hSecond :
      μ.real (SourceScaleFailure trueScale secondUpper) ≤ sourceDelta)
    (hSpend : sourceDelta + sourceDelta ≤ familywiseDelta) :
    μ.real (TwoSourceMaxScaleFailure
        trueScale firstUpper secondUpper)
      ≤ familywiseDelta := by
  exact le_trans
    (two_source_max_scale_failure_probability_le
      trueScale sourceDelta sourceDelta firstUpper secondUpper
      hFirst hSecond)
    hSpend

end SCOLHKG.Measure
