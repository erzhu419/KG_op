import Mathlib

namespace SCOLHKG.Real

open scoped BigOperators

/-!
A finite two-cell law of total variance.

This is the first nontrivial probability-algebra layer: the total variance is
expanded into expected within-cell variance plus between-cell explained
variance.  The proof uses `ring`; it is not a definitional equality trick.
-/

def twoCellMean (p m0 m1 : ℝ) : ℝ :=
  p * m0 + (1 - p) * m1

def twoCellTotalVariance (p m0 m1 v0 v1 : ℝ) : ℝ :=
  let m := twoCellMean p m0 m1
  p * (v0 + (m0 - m) ^ 2) + (1 - p) * (v1 + (m1 - m) ^ 2)

def twoCellExpectedConditionalVariance (p v0 v1 : ℝ) : ℝ :=
  p * v0 + (1 - p) * v1

def twoCellExplainedVariance (p m0 m1 : ℝ) : ℝ :=
  let m := twoCellMean p m0 m1
  p * (m0 - m) ^ 2 + (1 - p) * (m1 - m) ^ 2

theorem twoCell_law_total_variance
    (p m0 m1 v0 v1 : ℝ) :
    twoCellTotalVariance p m0 m1 v0 v1 =
      twoCellExpectedConditionalVariance p v0 v1
        + twoCellExplainedVariance p m0 m1 := by
  unfold twoCellTotalVariance
    twoCellExpectedConditionalVariance
    twoCellExplainedVariance
    twoCellMean
  ring

theorem twoCell_refinement_reduces_apparent_variance
    {p m0 m1 v0 v1 : ℝ}
    (hp0 : 0 ≤ p)
    (hp1 : p ≤ 1) :
    twoCellExpectedConditionalVariance p v0 v1 ≤
      twoCellTotalVariance p m0 m1 v0 v1 := by
  rw [twoCell_law_total_variance]
  have h1mp : 0 ≤ 1 - p := by linarith
  have hsq0 : 0 ≤ (m0 - twoCellMean p m0 m1) ^ 2 := sq_nonneg _
  have hsq1 : 0 ≤ (m1 - twoCellMean p m0 m1) ^ 2 := sq_nonneg _
  have hexpl : 0 ≤ twoCellExplainedVariance p m0 m1 := by
    unfold twoCellExplainedVariance
    exact add_nonneg (mul_nonneg hp0 hsq0) (mul_nonneg h1mp hsq1)
  linarith

theorem twoCell_no_explained_variance_preserves_variance
    {p m0 m1 v0 v1 : ℝ}
    (hzero : twoCellExplainedVariance p m0 m1 = 0) :
    twoCellTotalVariance p m0 m1 v0 v1 =
      twoCellExpectedConditionalVariance p v0 v1 := by
  rw [twoCell_law_total_variance, hzero, add_zero]

def finiteMean {ι : Type*} [Fintype ι] (p m : ι → ℝ) : ℝ :=
  ∑ i, p i * m i

def finiteTotalVariance {ι : Type*} [Fintype ι]
    (p m v : ι → ℝ) : ℝ :=
  let μ := finiteMean p m
  ∑ i, p i * (v i + (m i - μ) ^ 2)

def finiteExpectedConditionalVariance {ι : Type*} [Fintype ι]
    (p v : ι → ℝ) : ℝ :=
  ∑ i, p i * v i

def finiteExplainedVariance {ι : Type*} [Fintype ι]
    (p m : ι → ℝ) : ℝ :=
  let μ := finiteMean p m
  ∑ i, p i * (m i - μ) ^ 2

theorem finite_law_total_variance
    {ι : Type*} [Fintype ι]
    (p m v : ι → ℝ) :
    finiteTotalVariance p m v =
      finiteExpectedConditionalVariance p v + finiteExplainedVariance p m := by
  unfold finiteTotalVariance
    finiteExpectedConditionalVariance
    finiteExplainedVariance
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _hi
  ring

theorem finite_refinement_reduces_apparent_variance
    {ι : Type*} [Fintype ι]
    {p m v : ι → ℝ}
    (hp : ∀ i, 0 ≤ p i) :
    finiteExpectedConditionalVariance p v ≤ finiteTotalVariance p m v := by
  rw [finite_law_total_variance]
  have hexpl : 0 ≤ finiteExplainedVariance p m := by
    unfold finiteExplainedVariance
    exact Finset.sum_nonneg (by
      intro i _hi
      exact mul_nonneg (hp i) (sq_nonneg _))
  linarith

end SCOLHKG.Real
