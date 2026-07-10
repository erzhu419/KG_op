import Mathlib

namespace SCOLHKG.Real

def expectedSpikeSlabPrecision
    (pip slabPrecision spikePrecision : ℝ) : ℝ :=
  pip * slabPrecision + (1 - pip) * spikePrecision

theorem expectedSpikeSlabPrecision_nonnegative
    (pip slabPrecision spikePrecision : ℝ)
    (hPipNonnegative : 0 ≤ pip)
    (hPipAtMostOne : pip ≤ 1)
    (hSlab : 0 ≤ slabPrecision)
    (hSpike : 0 ≤ spikePrecision) :
    0 ≤ expectedSpikeSlabPrecision pip slabPrecision spikePrecision := by
  unfold expectedSpikeSlabPrecision
  exact add_nonneg
    (mul_nonneg hPipNonnegative hSlab)
    (mul_nonneg (sub_nonneg.mpr hPipAtMostOne) hSpike)

def coefficientMaskResidual {n : ℕ}
    (scale : ℝ)
    (pip slabVariance spikeVariance feature : Fin n → ℝ) : ℝ :=
  scale * ∑ i,
    pip i * (1 - pip i) * (slabVariance i - spikeVariance i) * feature i ^ 2

theorem coefficientMaskResidual_nonnegative
    {n : ℕ}
    (scale : ℝ)
    (pip slabVariance spikeVariance feature : Fin n → ℝ)
    (hScale : 0 ≤ scale)
    (hPipNonnegative : ∀ i, 0 ≤ pip i)
    (hPipAtMostOne : ∀ i, pip i ≤ 1)
    (hVarianceOrder : ∀ i, spikeVariance i ≤ slabVariance i) :
    0 ≤ coefficientMaskResidual scale pip slabVariance spikeVariance feature := by
  unfold coefficientMaskResidual
  apply mul_nonneg hScale
  apply Finset.sum_nonneg
  intro i _
  have hOneMinusPip : 0 ≤ 1 - pip i := sub_nonneg.mpr (hPipAtMostOne i)
  have hVarianceDifference : 0 ≤ slabVariance i - spikeVariance i :=
    sub_nonneg.mpr (hVarianceOrder i)
  exact mul_nonneg
    (mul_nonneg
      (mul_nonneg (hPipNonnegative i) hOneMinusPip)
      hVarianceDifference)
    (sq_nonneg (feature i))

def adaptivePredictiveVariance
    (conditionalVariance maskResidual : ℝ) : ℝ :=
  conditionalVariance + maskResidual

theorem adaptivePredictiveVariance_ge_conditional
    (conditionalVariance maskResidual : ℝ)
    (hResidual : 0 ≤ maskResidual) :
    conditionalVariance ≤
      adaptivePredictiveVariance conditionalVariance maskResidual := by
  unfold adaptivePredictiveVariance
  linarith

theorem adaptivePredictiveVariance_nonnegative
    (conditionalVariance maskResidual : ℝ)
    (hConditional : 0 ≤ conditionalVariance)
    (hResidual : 0 ≤ maskResidual) :
    0 ≤ adaptivePredictiveVariance conditionalVariance maskResidual := by
  unfold adaptivePredictiveVariance
  exact add_nonneg hConditional hResidual

def dampedInclusionProbability
    (damping current proposal : ℝ) : ℝ :=
  damping * current + (1 - damping) * proposal

theorem dampedInclusionProbability_mem
    (lower upper damping current proposal : ℝ)
    (hDampingNonnegative : 0 ≤ damping)
    (hDampingAtMostOne : damping ≤ 1)
    (hCurrentLower : lower ≤ current)
    (hCurrentUpper : current ≤ upper)
    (hProposalLower : lower ≤ proposal)
    (hProposalUpper : proposal ≤ upper) :
    lower ≤ dampedInclusionProbability damping current proposal ∧
      dampedInclusionProbability damping current proposal ≤ upper := by
  unfold dampedInclusionProbability
  constructor <;> nlinarith

theorem source_spike_can_escape_under_target_update
    (lower source posterior : ℝ)
    (hLowerSource : lower ≤ source)
    (hSourcePosterior : source < posterior) :
    lower < posterior := by
  linarith

noncomputable def collapsedGaussianLogBayesFactor
    (logDetPenalty score slabGain spikeGain : ℝ) : ℝ :=
  -logDetPenalty / 2 + score ^ 2 * (slabGain - spikeGain) / 2

theorem collapsedGaussianLogBayesFactor_monotone_score_sq
    (logDetPenalty score₁ score₂ slabGain spikeGain : ℝ)
    (hGain : 0 ≤ slabGain - spikeGain)
    (hScore : score₁ ^ 2 ≤ score₂ ^ 2) :
    collapsedGaussianLogBayesFactor
        logDetPenalty score₁ slabGain spikeGain ≤
      collapsedGaussianLogBayesFactor
        logDetPenalty score₂ slabGain spikeGain := by
  unfold collapsedGaussianLogBayesFactor
  have hProduct := mul_le_mul_of_nonneg_right hScore hGain
  have hDivided :=
    (div_le_div_iff_of_pos_right (by norm_num : (0 : ℝ) < 2)).2 hProduct
  exact add_le_add_right hDivided (-logDetPenalty / 2)

def fixedPrefixInclusion {n : ℕ}
    (fixed : Finset (Fin n))
    (pip : Fin n → ℝ)
    (i : Fin n) : ℝ :=
  if i ∈ fixed then 1 else pip i

theorem fixedPrefixInclusion_eq_one
    {n : ℕ}
    (fixed : Finset (Fin n))
    (pip : Fin n → ℝ)
    (i : Fin n)
    (hFixed : i ∈ fixed) :
    fixedPrefixInclusion fixed pip i = 1 := by
  simp [fixedPrefixInclusion, hFixed]

theorem fixedPrefix_mask_uncertainty_zero
    {n : ℕ}
    (fixed : Finset (Fin n))
    (pip : Fin n → ℝ)
    (i : Fin n)
    (hFixed : i ∈ fixed)
    (slabVariance spikeVariance feature : ℝ) :
    fixedPrefixInclusion fixed pip i *
        (1 - fixedPrefixInclusion fixed pip i) *
        (slabVariance - spikeVariance) * feature ^ 2 = 0 := by
  simp [fixedPrefixInclusion, hFixed]

def pilotAdmittedInclusion {n : ℕ}
    (admitted : Finset (Fin n))
    (floor : ℝ)
    (proposal : Fin n → ℝ)
    (i : Fin n) : ℝ :=
  if i ∈ admitted then proposal i else floor

theorem excluded_direction_stays_at_spike_floor
    {n : ℕ}
    (admitted : Finset (Fin n))
    (floor : ℝ)
    (proposal : Fin n → ℝ)
    (i : Fin n)
    (hExcluded : i ∉ admitted) :
    pilotAdmittedInclusion admitted floor proposal i = floor := by
  simp [pilotAdmittedInclusion, hExcluded]

theorem excluded_direction_cannot_activate
    {n : ℕ}
    (admitted : Finset (Fin n))
    (floor threshold : ℝ)
    (proposal : Fin n → ℝ)
    (i : Fin n)
    (hExcluded : i ∉ admitted)
    (hFloor : floor < threshold) :
    pilotAdmittedInclusion admitted floor proposal i < threshold := by
  rw [excluded_direction_stays_at_spike_floor admitted floor proposal i hExcluded]
  exact hFloor

def effectiveSparseDimension {n : ℕ} (pip : Fin n → ℝ) : ℝ :=
  ∑ i, pip i

theorem effectiveSparseDimension_nonnegative
    {n : ℕ}
    (pip : Fin n → ℝ)
    (hPip : ∀ i, 0 ≤ pip i) :
    0 ≤ effectiveSparseDimension pip := by
  unfold effectiveSparseDimension
  exact Finset.sum_nonneg fun i _ => hPip i

theorem cardinality_projection_preserves_budget
    {n : ℕ}
    (pip : Fin n → ℝ)
    (budget : ℝ)
    (hProjected : (∑ i, pip i) ≤ budget) :
    effectiveSparseDimension pip ≤ budget := by
  exact hProjected

end SCOLHKG.Real
