import Mathlib

namespace SCOLHKG.Real

def weightedInner {n : ℕ}
    (weight left right : Fin n → ℝ) : ℝ :=
  ∑ i, weight i * left i * right i

noncomputable def scalarProjectionCoefficient {n : ℕ}
    (weight base group : Fin n → ℝ) : ℝ :=
  weightedInner weight base group / weightedInner weight base base

noncomputable def projectionResidual {n : ℕ}
    (weight base group : Fin n → ℝ) : Fin n → ℝ :=
  fun i => group i - scalarProjectionCoefficient weight base group * base i

theorem weightedInner_projectionResidual_zero
    {n : ℕ}
    (weight base group : Fin n → ℝ)
    (hBase : weightedInner weight base base ≠ 0) :
    weightedInner weight base (projectionResidual weight base group) = 0 := by
  have hBase' : (∑ i, weight i * base i * base i) ≠ 0 := by
    simpa [weightedInner] using hBase
  have hBaseSq : (∑ i, weight i * base i ^ 2) ≠ 0 := by
    convert hBase' using 1
    apply Finset.sum_congr rfl
    intro i _
    ring
  unfold weightedInner projectionResidual scalarProjectionCoefficient
  calc
    (∑ i, weight i * base i *
      (group i -
        ((∑ j, weight j * base j * group j) /
          ∑ j, weight j * base j * base j) * base i)) =
        ∑ i, (weight i * base i * group i -
          ((∑ j, weight j * base j * group j) /
            ∑ j, weight j * base j * base j) *
              (weight i * base i * base i)) := by
          apply Finset.sum_congr rfl
          intro i _
          ring
    _ = (∑ i, weight i * base i * group i) -
        ∑ i, ((∑ j, weight j * base j * group j) /
          ∑ j, weight j * base j * base j) *
            (weight i * base i * base i) := by
          rw [Finset.sum_sub_distrib]
    _ = (∑ i, weight i * base i * group i) -
        ((∑ j, weight j * base j * group j) /
          ∑ j, weight j * base j * base j) *
            ∑ i, weight i * base i * base i := by
          rw [Finset.mul_sum]
    _ = 0 := by
      field_simp [hBaseSq]
      ring

theorem orthogonality_survives_group_selection
    {n m : ℕ}
    (weight : Fin n → ℝ)
    (feature : Fin m → Fin n → ℝ)
    (available selected : Finset (Fin m))
    (hSelected : selected ⊆ available)
    (hOrthogonal : ∀ i ∈ available, ∀ j ∈ available,
      i ≠ j → weightedInner weight (feature i) (feature j) = 0) :
    ∀ i ∈ selected, ∀ j ∈ selected,
      i ≠ j → weightedInner weight (feature i) (feature j) = 0 := by
  intro i hi j hj hij
  exact hOrthogonal i (hSelected hi) j (hSelected hj) hij

def supportClip (lower upper value : ℝ) : ℝ :=
  max lower (min upper value)

theorem supportClip_mem
    (lower upper value : ℝ)
    (hOrder : lower ≤ upper) :
    lower ≤ supportClip lower upper value ∧
      supportClip lower upper value ≤ upper := by
  unfold supportClip
  constructor
  · exact le_max_left lower (min upper value)
  · exact max_le hOrder (min_le_left upper value)

noncomputable def additiveComplexityPenalty
    (coefficient groupCount logSampleSize sampleSize : ℝ) : ℝ :=
  coefficient * groupCount * logSampleSize / sampleSize

theorem additiveComplexityPenalty_nonnegative
    (coefficient groupCount logSampleSize sampleSize : ℝ)
    (hCoefficient : 0 ≤ coefficient)
    (hCount : 0 ≤ groupCount)
    (hLog : 0 ≤ logSampleSize)
    (hSample : 0 ≤ sampleSize) :
    0 ≤ additiveComplexityPenalty
      coefficient groupCount logSampleSize sampleSize := by
  unfold additiveComplexityPenalty
  exact div_nonneg (mul_nonneg (mul_nonneg hCoefficient hCount) hLog) hSample

def gatedAdditiveValue
    (base additive : ℝ)
    (selected : Bool) : ℝ :=
  if selected then base + additive else base

theorem rejected_additive_group_is_exact_fallback
    (base additive : ℝ) :
    gatedAdditiveValue base additive false = base := by
  simp [gatedAdditiveValue]

end SCOLHKG.Real
