import Mathlib
import SCOLHKG.Real.CumulativeRisk

namespace SCOLHKG.Real

open scoped BigOperators

/-!
Finite ordered cumulative-risk coordinates used by the V21 representation.
The policy/path positions are finite, and each selected positional basis
function aggregates the same local exposure signal.  The constant basis is
ordinary pooled occupancy; selected low frequencies retain ordered exposure.
-/

def positionalExposure
    {Position Frequency : Type*} [Fintype Position]
    (basis : Frequency → Position → ℝ)
    (localExposure : Position → ℝ)
    (frequency : Frequency) : ℝ :=
  ∑ position, basis frequency position * localExposure position

def constantPositionalBasis
    {Frequency Position : Type*}
    [DecidableEq Frequency]
    (zeroFrequency : Frequency) : Frequency → Position → ℝ :=
  fun frequency _position => if frequency = zeroFrequency then 1 else 0

theorem zero_frequency_is_aggregate_exposure
    {Position Frequency : Type*}
    [Fintype Position] [DecidableEq Frequency]
    (zeroFrequency : Frequency)
    (localExposure : Position → ℝ) :
    positionalExposure
        (constantPositionalBasis zeroFrequency)
        localExposure
        zeroFrequency =
      ∑ position, localExposure position := by
  unfold positionalExposure constantPositionalBasis
  simp

theorem positionalExposure_add
    {Position Frequency : Type*} [Fintype Position]
    (basis : Frequency → Position → ℝ)
    (left right : Position → ℝ)
    (frequency : Frequency) :
    positionalExposure basis (fun position =>
        left position + right position) frequency =
      positionalExposure basis left frequency
        + positionalExposure basis right frequency := by
  unfold positionalExposure
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro position _hposition
  ring

theorem positionalExposure_smul
    {Position Frequency : Type*} [Fintype Position]
    (basis : Frequency → Position → ℝ)
    (localExposure : Position → ℝ)
    (scale : ℝ)
    (frequency : Frequency) :
    positionalExposure basis (fun position =>
        scale * localExposure position) frequency =
      scale * positionalExposure basis localExposure frequency := by
  unfold positionalExposure
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro position _hposition
  ring

def selectedOrderedExposure
    {Position Frequency : Type*}
    [Fintype Position] [Fintype Frequency] [DecidableEq Frequency]
    (selected : Finset Frequency)
    (basis : Frequency → Position → ℝ)
    (localExposure : Position → ℝ) : Frequency → ℝ :=
  fun frequency =>
    if frequency ∈ selected then
      positionalExposure basis localExposure frequency
    else 0

theorem unselected_frequency_is_zero
    {Position Frequency : Type*}
    [Fintype Position] [Fintype Frequency] [DecidableEq Frequency]
    (selected : Finset Frequency)
    (basis : Frequency → Position → ℝ)
    (localExposure : Position → ℝ)
    (frequency : Frequency)
    (hNotSelected : frequency ∉ selected) :
    selectedOrderedExposure selected basis localExposure frequency = 0 := by
  simp [selectedOrderedExposure, hNotSelected]

theorem selected_frequency_is_positional_exposure
    {Position Frequency : Type*}
    [Fintype Position] [Fintype Frequency] [DecidableEq Frequency]
    (selected : Finset Frequency)
    (basis : Frequency → Position → ℝ)
    (localExposure : Position → ℝ)
    (frequency : Frequency)
    (hSelected : frequency ∈ selected) :
    selectedOrderedExposure selected basis localExposure frequency =
      positionalExposure basis localExposure frequency := by
  simp [selectedOrderedExposure, hSelected]

def orderedCumulativeRisk
    (orderedLocal shared lambda : Exposure)
    (shock : RiskMatrix)
    (linear : Exposure)
    (floor : ℝ) : CumulativeRisk where
  A := orderedLocal
  N := shared
  Lambda := lambda
  B := shock
  omega := linear
  floor := floor

theorem ordered_coordinate_uses_cumulative_risk_decomposition
    (orderedLocal shared lambda : Exposure)
    (shock : RiskMatrix)
    (linear : Exposure)
    (floor : ℝ) :
    totalVariance (orderedCumulativeRisk
        orderedLocal shared lambda shock linear floor) =
      floor
        + independentRisk (orderedCumulativeRisk
          orderedLocal shared lambda shock linear floor)
        + sharedShockRisk (orderedCumulativeRisk
          orderedLocal shared lambda shock linear floor)
        + linearRisk (orderedCumulativeRisk
          orderedLocal shared lambda shock linear floor) := by
  exact fixedTrajectoryVarianceDecomposition _

end SCOLHKG.Real
