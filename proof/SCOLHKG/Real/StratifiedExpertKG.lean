import Mathlib

namespace SCOLHKG.Real

open scoped BigOperators

/-!
Rao-Blackwellized finite-expert expectation used by the stratified exact-KG
estimator.  Expert identity is summed with its posterior mass; only the
within-expert Gaussian expectation is approximated numerically.
-/

noncomputable def finiteStratifiedExpectation
    {Expert : Type*} [Fintype Expert]
    (q conditionalValue : Expert → ℝ) : ℝ :=
  ∑ expert, q expert * conditionalValue expert

theorem finite_stratified_identity_has_no_categorical_error
    {Expert : Type*} [Fintype Expert]
    (q conditionalValue : Expert → ℝ) :
    finiteStratifiedExpectation q conditionalValue =
      ∑ expert, q expert * conditionalValue expert := by
  rfl

theorem finite_stratified_error_le_conditional_error
    {Expert : Type*} [Fintype Expert]
    {q exact conditionalEstimate : Expert → ℝ}
    {epsilon : ℝ}
    (hq : ∀ expert, 0 ≤ q expert)
    (hqNorm : ∑ expert, q expert = 1)
    (hConditional :
      ∀ expert, |conditionalEstimate expert - exact expert| ≤ epsilon) :
    |finiteStratifiedExpectation q conditionalEstimate
        - finiteStratifiedExpectation q exact| ≤ epsilon := by
  have hDifference :
      finiteStratifiedExpectation q conditionalEstimate
          - finiteStratifiedExpectation q exact =
        ∑ expert, q expert *
          (conditionalEstimate expert - exact expert) := by
    simp only [finiteStratifiedExpectation]
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro expert _hexpert
    ring
  rw [hDifference]
  calc
    |∑ expert, q expert *
        (conditionalEstimate expert - exact expert)|
        ≤ ∑ expert, |q expert *
          (conditionalEstimate expert - exact expert)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ expert, q expert *
          |conditionalEstimate expert - exact expert| := by
      apply Finset.sum_congr rfl
      intro expert _hexpert
      rw [abs_mul, abs_of_nonneg (hq expert)]
    _ ≤ ∑ expert, q expert * epsilon := by
      exact Finset.sum_le_sum (fun expert _hexpert =>
        mul_le_mul_of_nonneg_left (hConditional expert) (hq expert))
    _ = epsilon := by
      rw [← Finset.sum_mul, hqNorm, one_mul]

theorem finite_rqmc_error_le_conditional_plus_selector_l1
    {Expert : Type*} [Fintype Expert]
    {q qHat exact conditionalEstimate : Expert → ℝ}
    {epsilon bound : ℝ}
    (hqHat : ∀ expert, 0 ≤ qHat expert)
    (hqHatNorm : ∑ expert, qHat expert = 1)
    (hConditional :
      ∀ expert, |conditionalEstimate expert - exact expert| ≤ epsilon)
    (hExactBound : ∀ expert, |exact expert| ≤ bound) :
    |finiteStratifiedExpectation qHat conditionalEstimate
        - finiteStratifiedExpectation q exact| ≤
      epsilon + (∑ expert, |qHat expert - q expert|) * bound := by
  have hConditionalError :
      |finiteStratifiedExpectation qHat conditionalEstimate
          - finiteStratifiedExpectation qHat exact| ≤ epsilon :=
    finite_stratified_error_le_conditional_error
      hqHat hqHatNorm hConditional
  have hWeightIdentity :
      finiteStratifiedExpectation qHat exact
          - finiteStratifiedExpectation q exact =
        ∑ expert, (qHat expert - q expert) * exact expert := by
    simp only [finiteStratifiedExpectation]
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro expert _hexpert
    ring
  have hWeightError :
      |finiteStratifiedExpectation qHat exact
          - finiteStratifiedExpectation q exact| ≤
        (∑ expert, |qHat expert - q expert|) * bound := by
    rw [hWeightIdentity]
    calc
      |∑ expert, (qHat expert - q expert) * exact expert|
          ≤ ∑ expert, |(qHat expert - q expert) * exact expert| :=
        Finset.abs_sum_le_sum_abs _ _
      _ = ∑ expert, |qHat expert - q expert| * |exact expert| := by
        apply Finset.sum_congr rfl
        intro expert _hexpert
        rw [abs_mul]
      _ ≤ ∑ expert, |qHat expert - q expert| * bound := by
        exact Finset.sum_le_sum (fun expert _hexpert =>
          mul_le_mul_of_nonneg_left
            (hExactBound expert) (abs_nonneg _))
      _ = (∑ expert, |qHat expert - q expert|) * bound := by
        rw [Finset.sum_mul]
  calc
    |finiteStratifiedExpectation qHat conditionalEstimate
        - finiteStratifiedExpectation q exact|
        = |(finiteStratifiedExpectation qHat conditionalEstimate
              - finiteStratifiedExpectation qHat exact)
            + (finiteStratifiedExpectation qHat exact
              - finiteStratifiedExpectation q exact)| := by
          ring_nf
    _ ≤ |finiteStratifiedExpectation qHat conditionalEstimate
            - finiteStratifiedExpectation qHat exact|
          + |finiteStratifiedExpectation qHat exact
            - finiteStratifiedExpectation q exact| := abs_add_le _ _
    _ ≤ epsilon
          + (∑ expert, |qHat expert - q expert|) * bound :=
      add_le_add hConditionalError hWeightError

theorem finite_stratified_exact_conditionals_are_exact
    {Expert : Type*} [Fintype Expert]
    {q exact conditionalEstimate : Expert → ℝ}
    (hConditional : ∀ expert, conditionalEstimate expert = exact expert) :
    finiteStratifiedExpectation q conditionalEstimate =
      finiteStratifiedExpectation q exact := by
  unfold finiteStratifiedExpectation
  apply Finset.sum_congr rfl
  intro expert _hexpert
  rw [hConditional expert]

end SCOLHKG.Real
