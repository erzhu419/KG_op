import Mathlib

namespace SCOLHKG.Real

def diagonalCoefficientVariance {n : ℕ}
    (variance coefficient : Fin n → ℝ) : ℝ :=
  ∑ i, variance i * coefficient i ^ 2

theorem diagonal_shrinkage_variance_le_isotropic
    {n : ℕ}
    (baseVariance : ℝ)
    (weight coefficient : Fin n → ℝ)
    (hBaseVariance : 0 ≤ baseVariance)
    (hWeightNonnegative : ∀ i, 0 ≤ weight i)
    (hWeightAtMostOne : ∀ i, weight i ≤ 1) :
    diagonalCoefficientVariance
        (fun i => baseVariance * weight i ^ 2) coefficient
      ≤ diagonalCoefficientVariance (fun _ => baseVariance) coefficient := by
  unfold diagonalCoefficientVariance
  apply Finset.sum_le_sum
  intro i _
  have hWeightSquare : weight i ^ 2 ≤ 1 := by
    nlinarith [
      hWeightNonnegative i,
      hWeightAtMostOne i,
      sq_nonneg (weight i),
      sq_nonneg (weight i - 1)]
  have hCoefficientSquare : 0 ≤ coefficient i ^ 2 := sq_nonneg _
  have hScaledWeight :
      baseVariance * weight i ^ 2 ≤ baseVariance := by
    nlinarith [hBaseVariance]
  exact mul_le_mul_of_nonneg_right hScaledWeight hCoefficientSquare

end SCOLHKG.Real
