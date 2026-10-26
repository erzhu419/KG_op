import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.MeasureTheory.Measure.Real
import Mathlib.Tactic.Common
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Finiteness
import Lean.Elab.Tactic.Omega

/-!
# Covariance semantics of the current AFTER / POINT controls

This module proves the finite-matrix congruence and prescribed diagonal used by
mainline_role_after_init.py and mainline_role_point_only.py.
It concerns the decision moment matrix, not a joint Bayesian posterior, the
subsequent jittered sampler, or floating-point positivity.
-/

namespace SCOLHKG.Real.CurrentCovariance

open scoped BigOperators

variable {ι : Type*} [Fintype ι]

def quadraticForm (C : ι → ι → ℝ) (z : ι → ℝ) : ℝ :=
  ∑ i, ∑ j, z i * C i j * z j

def CovariancePSD (C : ι → ι → ℝ) : Prop :=
  (∀ i j, C i j = C j i) ∧ ∀ z, 0 ≤ quadraticForm C z

def scaledCovariance (C : ι → ι → ℝ) (s : ι → ℝ) (i j : ι) : ℝ :=
  s i * C i j * s j

theorem scaled_quadratic_form (C : ι → ι → ℝ) (s z : ι → ℝ) :
    quadraticForm (scaledCovariance C s) z =
      quadraticForm C (fun i => s i * z i) := by
  unfold quadraticForm scaledCovariance
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem scaled_covariance_psd {C : ι → ι → ℝ}
    (hC : CovariancePSD C) (s : ι → ℝ) :
    CovariancePSD (scaledCovariance C s) := by
  constructor
  · intro i j
    unfold scaledCovariance
    rw [hC.1 i j]
    ring
  · intro z
    rw [scaled_quadratic_form]
    exact hC.2 _

noncomputable def marginalScale (C : ι → ι → ℝ) (v : ι → ℝ) (i : ι) : ℝ :=
  Real.sqrt (v i / C i i)

noncomputable def fixedCorrelationCovariance
    (C : ι → ι → ℝ) (v : ι → ℝ) : ι → ι → ℝ :=
  scaledCovariance C (marginalScale C v)

theorem fixed_correlation_covariance_psd {C : ι → ι → ℝ}
    (hC : CovariancePSD C) (v : ι → ℝ) :
    CovariancePSD (fixedCorrelationCovariance C v) :=
  scaled_covariance_psd hC _

theorem fixed_correlation_diagonal
    (C : ι → ι → ℝ) (v : ι → ℝ)
    (hdiag : ∀ i, 0 < C i i) (hv : ∀ i, 0 ≤ v i) (i : ι) :
    fixedCorrelationCovariance C v i i = v i := by
  have hsqrt : (Real.sqrt (v i / C i i)) ^ 2 = v i / C i i :=
    Real.sq_sqrt (div_nonneg (hv i) (le_of_lt (hdiag i)))
  unfold fixedCorrelationCovariance scaledCovariance marginalScale
  calc
    Real.sqrt (v i / C i i) * C i i * Real.sqrt (v i / C i i) =
        C i i * (Real.sqrt (v i / C i i)) ^ 2 := by ring
    _ = v i := by
      rw [hsqrt]
      field_simp [ne_of_gt (hdiag i)]

theorem scaled_correlation_preserved
    (C : ι → ι → ℝ) (s : ι → ℝ) (i j : ι)
    (hi : 0 < C i i) (hj : 0 < C j j)
    (hsi : 0 < s i) (hsj : 0 < s j) :
    scaledCovariance C s i j /
        Real.sqrt (scaledCovariance C s i i *
          scaledCovariance C s j j) =
      C i j / Real.sqrt (C i i * C j j) := by
  have hs : 0 < s i * s j := mul_pos hsi hsj
  have hroot : Real.sqrt (scaledCovariance C s i i *
        scaledCovariance C s j j) =
      (s i * s j) * Real.sqrt (C i i * C j j) := by
    have heq : scaledCovariance C s i i * scaledCovariance C s j j =
        (s i * s j) ^ 2 * (C i i * C j j) := by
      unfold scaledCovariance
      ring
    rw [heq, Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq_eq_abs,
      abs_of_pos hs]
  rw [hroot]
  unfold scaledCovariance
  have hbase : Real.sqrt (C i i * C j j) ≠ 0 :=
    ne_of_gt (Real.sqrt_pos.2 (mul_pos hi hj))
  field_simp [ne_of_gt hsi, ne_of_gt hsj, hbase]
  <;> ring

theorem zero_local_variance_zero_row
    (C : ι → ι → ℝ) (v : ι → ℝ) (i j : ι)
    (hi : v i = 0) :
    fixedCorrelationCovariance C v i j = 0 := by
  simp [fixedCorrelationCovariance, scaledCovariance, marginalScale, hi]

end SCOLHKG.Real.CurrentCovariance
