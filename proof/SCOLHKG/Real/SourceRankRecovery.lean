import Mathlib

namespace SCOLHKG.Real

/-!
# Finite-error source-score ranking

Uniform source-score estimation error recovers every pair whose true score gap
exceeds twice the error radius. This is the deterministic bridge from the
sub-Gaussian replicated-mean event to the percentile ordering consumed by the
source-scored atlas.
-/

def UniformFiniteScoreError
    {X : Type*} (estimate truth : X → ℝ) (radius : ℝ) : Prop :=
  ∀ x, |estimate x - truth x| ≤ radius

private theorem chance_margin_error_from_mean_scale
    {estimatedMean trueMean estimatedScale trueScale z tau : ℝ}
    {meanRadius scaleRadius : ℝ}
    (hMean : |estimatedMean - trueMean| ≤ meanRadius)
    (hScale : |estimatedScale - trueScale| ≤ scaleRadius)
    (hZ : 0 ≤ z) :
    |(estimatedMean + z * estimatedScale - tau)
        - (trueMean + z * trueScale - tau)|
      ≤ meanRadius + z * scaleRadius := by
  calc
    |(estimatedMean + z * estimatedScale - tau)
        - (trueMean + z * trueScale - tau)|
      = |(estimatedMean - trueMean)
          + z * (estimatedScale - trueScale)| := by ring_nf
    _ ≤ |estimatedMean - trueMean|
          + |z * (estimatedScale - trueScale)| := abs_add_le _ _
    _ = |estimatedMean - trueMean|
          + z * |estimatedScale - trueScale| := by
            rw [abs_mul, abs_of_nonneg hZ]
    _ ≤ meanRadius + z * scaleRadius := by
          gcongr

noncomputable def unbiasedVarianceInflation
    (replicationCount : ℕ) : ℝ :=
  (replicationCount : ℝ) / ((replicationCount : ℝ) - 1)

noncomputable def finiteSampleMean
    {replicationCount : ℕ}
    (sample : Fin replicationCount → ℝ) : ℝ :=
  (∑ replication, sample replication) / replicationCount

noncomputable def finiteResidualSecondMoment
    {replicationCount : ℕ}
    (sample : Fin replicationCount → ℝ)
    (trueMean : ℝ) : ℝ :=
  (∑ replication, (sample replication - trueMean) ^ 2) / replicationCount

noncomputable def finiteUnbiasedSampleVariance
    {replicationCount : ℕ}
    (sample : Fin replicationCount → ℝ) : ℝ :=
  (∑ replication,
      (sample replication - finiteSampleMean sample) ^ 2)
    / ((replicationCount : ℝ) - 1)

theorem finite_residual_sum_eq_count_mul_mean_error
    {replicationCount : ℕ}
    {sample : Fin replicationCount → ℝ}
    {trueMean : ℝ}
    (hReplicationCount : 0 < replicationCount) :
    (∑ replication, (sample replication - trueMean))
      = replicationCount * (finiteSampleMean sample - trueMean) := by
  have hCount : (replicationCount : ℝ) ≠ 0 := by
    exact_mod_cast hReplicationCount.ne'
  rw [Finset.sum_sub_distrib]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul]
  unfold finiteSampleMean
  field_simp

theorem finite_unbiased_sample_variance_identity
    {replicationCount : ℕ}
    {sample : Fin replicationCount → ℝ}
    {trueMean : ℝ}
    (hReplicationCount : 1 < replicationCount) :
    finiteUnbiasedSampleVariance sample
      = unbiasedVarianceInflation replicationCount
          * (finiteResidualSecondMoment sample trueMean
            - (finiteSampleMean sample - trueMean) ^ 2) := by
  have hCountPositive : 0 < replicationCount :=
    Nat.zero_lt_of_lt hReplicationCount
  have hCount : (replicationCount : ℝ) ≠ 0 := by
    exact_mod_cast hCountPositive.ne'
  have hCountMinus : (replicationCount : ℝ) - 1 ≠ 0 := by
    have hCountReal : (1 : ℝ) < replicationCount := by
      exact_mod_cast hReplicationCount
    linarith
  let mean := finiteSampleMean sample
  have hResidualSum :
      (∑ replication, (sample replication - trueMean))
        = replicationCount * (mean - trueMean) := by
    simpa [mean] using
      (finite_residual_sum_eq_count_mul_mean_error
        (sample := sample) (trueMean := trueMean) hCountPositive)
  have hCross :
      (∑ replication,
          2 * (sample replication - trueMean) * (mean - trueMean))
        = 2 * (mean - trueMean)
            * (∑ replication, (sample replication - trueMean)) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro replication _hreplication
    ring
  have hCenteredSum :
      (∑ replication, (sample replication - mean) ^ 2)
        = (∑ replication, (sample replication - trueMean) ^ 2)
          - replicationCount * (mean - trueMean) ^ 2 := by
    calc
      (∑ replication, (sample replication - mean) ^ 2)
          = ∑ replication,
              ((sample replication - trueMean) - (mean - trueMean)) ^ 2 := by
                apply Finset.sum_congr rfl
                intro replication _hreplication
                congr 1
                ring
      _ = (∑ replication, (sample replication - trueMean) ^ 2)
          - 2 * (mean - trueMean)
              * (∑ replication, (sample replication - trueMean))
          + replicationCount * (mean - trueMean) ^ 2 := by
            simp_rw [sub_sq]
            rw [Finset.sum_add_distrib, Finset.sum_sub_distrib]
            rw [hCross]
            simp only [Finset.sum_const, Finset.card_univ,
              Fintype.card_fin, nsmul_eq_mul]
      _ = (∑ replication, (sample replication - trueMean) ^ 2)
          - replicationCount * (mean - trueMean) ^ 2 := by
            rw [hResidualSum]
            ring
  unfold finiteUnbiasedSampleVariance finiteResidualSecondMoment
  change
    (∑ replication, (sample replication - mean) ^ 2)
          / ((replicationCount : ℝ) - 1)
      = (replicationCount : ℝ) / ((replicationCount : ℝ) - 1)
          * ((∑ replication, (sample replication - trueMean) ^ 2)
              / replicationCount
            - (mean - trueMean) ^ 2)
  rw [hCenteredSum]
  field_simp

theorem unbiasedVarianceInflation_nonnegative
    {replicationCount : ℕ}
    (hReplicationCount : 1 < replicationCount) :
    0 ≤ unbiasedVarianceInflation replicationCount := by
  have hCount : (1 : ℝ) < replicationCount := by
    exact_mod_cast hReplicationCount
  exact div_nonneg (by positivity) (sub_nonneg.mpr hCount.le)

theorem one_le_unbiasedVarianceInflation
    {replicationCount : ℕ}
    (hReplicationCount : 1 < replicationCount) :
    1 ≤ unbiasedVarianceInflation replicationCount := by
  have hCount : (1 : ℝ) < replicationCount := by
    exact_mod_cast hReplicationCount
  unfold unbiasedVarianceInflation
  rw [le_div_iff₀ (sub_pos.mpr hCount)]
  linarith

theorem max_variance_floor_error_le
    {estimate truth floor : ℝ}
    (hFloor : floor ≤ truth) :
    |max estimate floor - truth| ≤ |estimate - truth| := by
  rcases le_total estimate floor with hEstimateFloor | hFloorEstimate
  · rw [max_eq_right hEstimateFloor]
    have hEstimateTruth : estimate ≤ truth := hEstimateFloor.trans hFloor
    rw [abs_of_nonpos (sub_nonpos.mpr hFloor),
      abs_of_nonpos (sub_nonpos.mpr hEstimateTruth)]
    linarith
  · rw [max_eq_left hFloorEstimate]

theorem sqrt_error_le_sqrt_absolute_error
    {first second : ℝ}
    (hFirst : 0 ≤ first)
    (hSecond : 0 ≤ second) :
    |Real.sqrt first - Real.sqrt second| ≤ Real.sqrt |first - second| := by
  apply Real.abs_le_sqrt
  rcases le_total first second with hFirstSecond | hSecondFirst
  · have hSqrt : Real.sqrt first ≤ Real.sqrt second :=
      Real.sqrt_le_sqrt hFirstSecond
    have hMul := mul_le_mul_of_nonneg_left hSqrt (Real.sqrt_nonneg first)
    rw [abs_of_nonpos (sub_nonpos.mpr hFirstSecond)]
    nlinarith [Real.sq_sqrt hFirst, Real.sq_sqrt hSecond]
  · have hSqrt : Real.sqrt second ≤ Real.sqrt first :=
      Real.sqrt_le_sqrt hSecondFirst
    have hMul := mul_le_mul_of_nonneg_left hSqrt (Real.sqrt_nonneg second)
    rw [abs_of_nonneg (sub_nonneg.mpr hSecondFirst)]
    nlinarith [Real.sq_sqrt hFirst, Real.sq_sqrt hSecond]

theorem unbiased_sample_variance_error_le_of_residual_moments
    {replicationCount : ℕ}
    {sampleMean trueMean residualSecondMoment sampleVariance trueVariance : ℝ}
    {meanRadius secondMomentRadius varianceUpper : ℝ}
    (hReplicationCount : 1 < replicationCount)
    (hMeanRadius : 0 ≤ meanRadius)
    (hTrueVariance : 0 ≤ trueVariance)
    (hVarianceUpper : trueVariance ≤ varianceUpper)
    (hMean : |sampleMean - trueMean| ≤ meanRadius)
    (hSecondMoment :
      |residualSecondMoment - trueVariance| ≤ secondMomentRadius)
    (hSampleVarianceIdentity :
      sampleVariance = unbiasedVarianceInflation replicationCount
        * (residualSecondMoment - (sampleMean - trueMean) ^ 2)) :
    |sampleVariance - trueVariance|
      ≤ unbiasedVarianceInflation replicationCount
          * (secondMomentRadius + meanRadius ^ 2)
        + (unbiasedVarianceInflation replicationCount - 1) * varianceUpper := by
  let inflation := unbiasedVarianceInflation replicationCount
  have hInflation : 0 ≤ inflation :=
    unbiasedVarianceInflation_nonnegative hReplicationCount
  have hInflationOne : 1 ≤ inflation :=
    one_le_unbiasedVarianceInflation hReplicationCount
  have hInflationMinus : 0 ≤ inflation - 1 := sub_nonneg.mpr hInflationOne
  have hMeanSquare :
      (sampleMean - trueMean) ^ 2 ≤ meanRadius ^ 2 := by
    rw [sq_le_sq]
    simpa [abs_of_nonneg hMeanRadius] using hMean
  have hInner :
      |residualSecondMoment - (sampleMean - trueMean) ^ 2 - trueVariance|
        ≤ secondMomentRadius + meanRadius ^ 2 := by
    calc
      |residualSecondMoment - (sampleMean - trueMean) ^ 2 - trueVariance|
          = |(residualSecondMoment - trueVariance)
              - (sampleMean - trueMean) ^ 2| := by ring_nf
      _ ≤ |residualSecondMoment - trueVariance|
          + |(sampleMean - trueMean) ^ 2| := abs_sub _ _
      _ = |residualSecondMoment - trueVariance|
          + (sampleMean - trueMean) ^ 2 := by
            apply congrArg (fun value : ℝ ↦
              |residualSecondMoment - trueVariance| + value)
            exact abs_of_nonneg (sq_nonneg (sampleMean - trueMean))
      _ ≤ secondMomentRadius + meanRadius ^ 2 :=
            add_le_add hSecondMoment hMeanSquare
  rw [hSampleVarianceIdentity]
  have hRewrite :
      inflation
          * (residualSecondMoment - (sampleMean - trueMean) ^ 2)
          - trueVariance
        = inflation
            * (residualSecondMoment - (sampleMean - trueMean) ^ 2
              - trueVariance)
          + (inflation - 1) * trueVariance := by ring
  rw [hRewrite]
  calc
    |inflation
          * (residualSecondMoment - (sampleMean - trueMean) ^ 2
            - trueVariance)
        + (inflation - 1) * trueVariance|
      ≤ |inflation
          * (residualSecondMoment - (sampleMean - trueMean) ^ 2
            - trueVariance)|
        + |(inflation - 1) * trueVariance| := abs_add_le _ _
    _ = inflation
          * |residualSecondMoment - (sampleMean - trueMean) ^ 2
            - trueVariance|
        + (inflation - 1) * trueVariance := by
          rw [abs_mul, abs_mul, abs_of_nonneg hInflation,
            abs_of_nonneg hInflationMinus, abs_of_nonneg hTrueVariance]
    _ ≤ inflation * (secondMomentRadius + meanRadius ^ 2)
        + (inflation - 1) * varianceUpper := by
          exact add_le_add
            (mul_le_mul_of_nonneg_left hInner hInflation)
            (mul_le_mul_of_nonneg_left hVarianceUpper hInflationMinus)

theorem empirical_chance_margin_error_from_replicated_moments
    {replicationCount : ℕ}
    {sampleMean trueMean residualSecondMoment sampleVariance trueVariance : ℝ}
    {meanRadius secondMomentRadius varianceUpper floor z tau : ℝ}
    (hReplicationCount : 1 < replicationCount)
    (hMeanRadius : 0 ≤ meanRadius)
    (hFloorNonnegative : 0 ≤ floor)
    (hFloor : floor ≤ trueVariance)
    (hTrueVariance : 0 ≤ trueVariance)
    (hVarianceUpper : trueVariance ≤ varianceUpper)
    (hZ : 0 ≤ z)
    (hMean : |sampleMean - trueMean| ≤ meanRadius)
    (hSecondMoment :
      |residualSecondMoment - trueVariance| ≤ secondMomentRadius)
    (hSampleVarianceIdentity :
      sampleVariance = unbiasedVarianceInflation replicationCount
        * (residualSecondMoment - (sampleMean - trueMean) ^ 2)) :
    |(sampleMean + z * Real.sqrt (max sampleVariance floor) - tau)
        - (trueMean + z * Real.sqrt trueVariance - tau)|
      ≤ meanRadius
        + z * Real.sqrt (
            unbiasedVarianceInflation replicationCount
                * (secondMomentRadius + meanRadius ^ 2)
              + (unbiasedVarianceInflation replicationCount - 1)
                * varianceUpper) := by
  have hVariance := unbiased_sample_variance_error_le_of_residual_moments
    hReplicationCount hMeanRadius hTrueVariance
    hVarianceUpper hMean hSecondMoment hSampleVarianceIdentity
  have hFlooredVariance :=
    (max_variance_floor_error_le hFloor).trans hVariance
  have hMaxNonnegative : 0 ≤ max sampleVariance floor :=
    le_max_of_le_right hFloorNonnegative
  have hScaleBase :
      |Real.sqrt (max sampleVariance floor) - Real.sqrt trueVariance|
        ≤ Real.sqrt |max sampleVariance floor - trueVariance| :=
    sqrt_error_le_sqrt_absolute_error hMaxNonnegative hTrueVariance
  have hScale :
      |Real.sqrt (max sampleVariance floor) - Real.sqrt trueVariance|
        ≤ Real.sqrt (
            unbiasedVarianceInflation replicationCount
                * (secondMomentRadius + meanRadius ^ 2)
              + (unbiasedVarianceInflation replicationCount - 1)
                * varianceUpper) := by
    exact hScaleBase.trans (Real.sqrt_le_sqrt hFlooredVariance)
  exact chance_margin_error_from_mean_scale hMean hScale hZ

theorem empirical_chance_margin_error_le
    {estimatedMean trueMean estimatedScale trueScale z tau : ℝ}
    {meanRadius scaleRadius : ℝ}
    (hMean : |estimatedMean - trueMean| ≤ meanRadius)
    (hScale : |estimatedScale - trueScale| ≤ scaleRadius)
    (hZ : 0 ≤ z) :
    |(estimatedMean + z * estimatedScale - tau)
        - (trueMean + z * trueScale - tau)|
      ≤ meanRadius + z * scaleRadius := by
  calc
    |(estimatedMean + z * estimatedScale - tau)
        - (trueMean + z * trueScale - tau)|
      = |(estimatedMean - trueMean)
          + z * (estimatedScale - trueScale)| := by ring_nf
    _ ≤ |estimatedMean - trueMean|
          + |z * (estimatedScale - trueScale)| := abs_add_le _ _
    _ = |estimatedMean - trueMean|
          + z * |estimatedScale - trueScale| := by
            rw [abs_mul, abs_of_nonneg hZ]
    _ ≤ meanRadius + z * scaleRadius := by
          gcongr

theorem floored_empirical_chance_margin_error_le
    {estimatedMean trueMean estimatedScale trueScale scaleFloor z tau : ℝ}
    {meanRadius scaleRadius : ℝ}
    (hMean : |estimatedMean - trueMean| ≤ meanRadius)
    (hScale : |estimatedScale - trueScale| ≤ scaleRadius)
    (hFloor : scaleFloor ≤ trueScale)
    (hZ : 0 ≤ z) :
    |(estimatedMean + z * max estimatedScale scaleFloor - tau)
        - (trueMean + z * trueScale - tau)|
      ≤ meanRadius + z * scaleRadius := by
  have hFlooredScale :
      |max estimatedScale scaleFloor - max trueScale scaleFloor|
        ≤ scaleRadius :=
    (abs_max_sub_max_le_abs estimatedScale trueScale scaleFloor).trans hScale
  have hMargin := chance_margin_error_from_mean_scale
    (tau := tau) hMean hFlooredScale hZ
  simpa [max_eq_left hFloor] using hMargin

theorem separated_source_pair_order_recovered
    {X : Type*}
    {estimate truth : X → ℝ}
    {radius : ℝ}
    {x y : X}
    (hUniform : UniformFiniteScoreError estimate truth radius)
    (hGap : truth x + 2 * radius < truth y) :
    estimate x < estimate y := by
  have hx := hUniform x
  have hy := hUniform y
  obtain ⟨hxLower, hxUpper⟩ := abs_le.mp hx
  obtain ⟨hyLower, hyUpper⟩ := abs_le.mp hy
  linarith

def UniqueFiniteScoreMinimizer
    {X : Type*} [DecidableEq X]
    (library : Finset X) (score : X → ℝ) (best : X) (gap : ℝ) : Prop :=
  best ∈ library ∧
    ∀ x ∈ library, x ≠ best → score best + gap < score x

theorem separated_source_minimizer_recovered
    {X : Type*} [DecidableEq X]
    {library : Finset X}
    {estimate truth : X → ℝ}
    {radius : ℝ}
    {best : X}
    (hUniform : UniformFiniteScoreError estimate truth radius)
    (hBest : UniqueFiniteScoreMinimizer library truth best (2 * radius)) :
    best ∈ library ∧
      ∀ x ∈ library, x ≠ best → estimate best < estimate x := by
  refine ⟨hBest.1, ?_⟩
  intro x hx hne
  exact separated_source_pair_order_recovered
    hUniform (hBest.2 x hx hne)

end SCOLHKG.Real
