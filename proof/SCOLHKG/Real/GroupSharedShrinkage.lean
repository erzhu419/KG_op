import Mathlib
import SCOLHKG.Real.AdaptiveCoefficientSparsity

namespace SCOLHKG.Real

open scoped BigOperators

/-!
The V26 ordered expert does not transfer a source-domain direction inside an
exposure block.  It transfers one inclusion probability and one isotropic
spike/slab scale for the whole block; held-out target data learn the coefficient
direction.  Consequently the prior penalty is invariant to every orthogonal
change of coordinates within that semantic block.
-/

noncomputable def sharedGroupPenalty {n : ℕ}
    (pip slabPrecision spikePrecision : ℝ)
    (coefficient : EuclideanSpace ℝ (Fin n)) : ℝ :=
  expectedSpikeSlabPrecision pip slabPrecision spikePrecision
    * ‖coefficient‖ ^ 2

theorem sharedGroupPenalty_nonnegative {n : ℕ}
    (pip slabPrecision spikePrecision : ℝ)
    (coefficient : EuclideanSpace ℝ (Fin n))
    (hPipNonnegative : 0 ≤ pip)
    (hPipAtMostOne : pip ≤ 1)
    (hSlab : 0 ≤ slabPrecision)
    (hSpike : 0 ≤ spikePrecision) :
    0 ≤ sharedGroupPenalty
      pip slabPrecision spikePrecision coefficient := by
  unfold sharedGroupPenalty
  exact mul_nonneg
    (expectedSpikeSlabPrecision_nonnegative
      pip slabPrecision spikePrecision
      hPipNonnegative hPipAtMostOne hSlab hSpike)
    (sq_nonneg ‖coefficient‖)

theorem sharedGroupPenalty_rotation_invariant {n : ℕ}
    (pip slabPrecision spikePrecision : ℝ)
    (rotation :
      EuclideanSpace ℝ (Fin n) ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin n))
    (coefficient : EuclideanSpace ℝ (Fin n)) :
    sharedGroupPenalty pip slabPrecision spikePrecision
        (rotation coefficient) =
      sharedGroupPenalty pip slabPrecision spikePrecision coefficient := by
  simp [sharedGroupPenalty]

def sharedGroupInclusion {n : ℕ}
    (pip : ℝ) (_coordinate : Fin n) : ℝ :=
  pip

def sharedGroupEffectiveDimension {n : ℕ} (pip : ℝ) : ℝ :=
  ∑ coordinate : Fin n, sharedGroupInclusion pip coordinate

theorem sharedGroupEffectiveDimension_eq {n : ℕ} (pip : ℝ) :
    sharedGroupEffectiveDimension (n := n) pip = (n : ℝ) * pip := by
  simp [sharedGroupEffectiveDimension, sharedGroupInclusion]

theorem sharedGroupEffectiveDimension_nonnegative {n : ℕ}
    (pip : ℝ)
    (hPip : 0 ≤ pip) :
    0 ≤ sharedGroupEffectiveDimension (n := n) pip := by
  rw [sharedGroupEffectiveDimension_eq]
  positivity

theorem shared_group_budget_controls_effective_dimension {n : ℕ}
    (pip budget : ℝ)
    (hBudget : (n : ℝ) * pip ≤ budget) :
    sharedGroupEffectiveDimension (n := n) pip ≤ budget := by
  rwa [sharedGroupEffectiveDimension_eq]

end SCOLHKG.Real
