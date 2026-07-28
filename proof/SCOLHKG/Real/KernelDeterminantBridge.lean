import Mathlib
import SCOLHKG.Real.FiniteKernelInformationGain

namespace SCOLHKG.Real

open scoped BigOperators

/-!
Kernel/feature determinant information-gain bridge.

The implementation uses finite feature maps.  Once a numerical or analytic
kernel-specific argument bounds the product ratio by a determinant ratio, the
finite information-gain term used in regret accounting is bounded by the same
determinant cap.
-/

structure DeterminantRatioBound where
  productRatio : ℝ
  determinantRatio : ℝ

def DeterminantRatioBound.Valid (b : DeterminantRatioBound) : Prop :=
  0 < b.productRatio ∧ b.productRatio ≤ b.determinantRatio

theorem determinantInformationGain_mono
    {productRatio determinantRatio : ℝ}
    (hprod : 0 < productRatio)
    (hle : productRatio ≤ determinantRatio) :
    determinantInformationGain productRatio
      ≤ determinantInformationGain determinantRatio := by
  unfold determinantInformationGain
  exact mul_le_mul_of_nonneg_left
    (Real.log_le_log hprod hle)
    (by norm_num)

theorem finiteInformationGain_le_determinant_cap
    {Time : Type*}
    (steps : Finset Time)
    (variance noise : Time → ℝ)
    (bound : DeterminantRatioBound)
    (hpos :
      ∀ step ∈ steps, 0 < 1 + variance step / noise step)
    (hvalid : bound.Valid)
    (hProduct :
      finiteKernelProductRatio steps variance noise
        = bound.productRatio) :
    finiteInformationGain steps variance noise
      ≤ determinantInformationGain bound.determinantRatio := by
  have hEq :=
    finiteInformationGain_eq_determinantInformationGain_product
      steps variance noise hpos
  unfold DeterminantRatioBound.Valid at hvalid
  rw [hEq, hProduct]
  exact determinantInformationGain_mono hvalid.1 hvalid.2

theorem finiteInformationGain_regret_budget_from_determinant_cap
    {Time : Type*}
    (steps : Finset Time)
    (variance noise : Time → ℝ)
    (terms : InformationGainRegretTerms)
    (bound : DeterminantRatioBound)
    {eps : ℝ}
    (hpos :
      ∀ step ∈ steps, 0 < 1 + variance step / noise step)
    (hvalid : bound.Valid)
    (hProduct :
      finiteKernelProductRatio steps variance noise
        = bound.productRatio)
    (hRadius :
      finiteInformationGain steps variance noise
        ≤ determinantInformationGain bound.determinantRatio →
        informationGainRadius terms.beta terms.gammaT
          ≤ informationGainRadius terms.beta
            (determinantInformationGain bound.determinantRatio))
    (hRegret : InformationGainRegretBound terms)
    (hBudget :
      informationGainRadius terms.beta
          (determinantInformationGain bound.determinantRatio)
        + terms.candidateSetError
        + terms.certificationError
        + terms.kgApproximationError ≤ eps) :
    terms.actualRegret ≤ eps := by
  apply finiteInformationGain_regret_budget
    (steps := steps)
    (variance := variance)
    (noise := noise)
    (t := terms)
    (cap := determinantInformationGain bound.determinantRatio)
    (eps := eps)
  · exact finiteInformationGain_le_determinant_cap
      steps variance noise bound hpos hvalid hProduct
  · exact hRadius
  · exact hRegret
  · exact hBudget

end SCOLHKG.Real
