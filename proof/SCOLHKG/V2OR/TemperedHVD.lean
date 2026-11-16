import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.MeasureTheory.Measure.Real
import Mathlib.Tactic.Common
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Finiteness

/-!
# Theorem 2 Formal Core: Mean-Shift-to-Aleatoric Leakage in Untempered Transfer vs.
# Exact Shift Invariance & Oracle Risk of Prequential Tempered HVD
-/

namespace SCOLHKG.V2OR.TemperedHVD

set_option linter.unusedSectionVars false

open MeasureTheory
open scoped BigOperators

/-- Sample average of `m + 1` replications indexed by `Fin (m + 1)`. -/
noncomputable def sampleMean {m : ℕ} (Y : Fin (m + 1) → ℝ) : ℝ :=
  (∑ r : Fin (m + 1), Y r) / ((m + 1 : ℕ) : ℝ)

/-- Within-policy sum of squared deviations `SSQ = sum_r (Y_r - Y_bar)^2`. -/
noncomputable def withinPolicySSQ {m : ℕ} (Y : Fin (m + 1) → ℝ) : ℝ :=
  ∑ r : Fin (m + 1), (Y r - sampleMean Y) ^ 2

/-- **Exact Mean-Shift Invariance**: shifting target mean by ANY `mu_0 + Delta_g` leaves
    within-policy dispersion `withinPolicySSQ` identically equal to pure noise `SSQ(eps)`. -/
theorem within_policy_ssq_shift_invariant
    {m : ℕ} (mu0 deltaG : ℝ) (eps : Fin (m + 1) → ℝ) :
    withinPolicySSQ (fun r => mu0 + deltaG + eps r) = withinPolicySSQ eps := by
  unfold withinPolicySSQ sampleMean
  have hm : ((m + 1 : ℕ) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.succ_ne_zero m)
  have hMean :
      (∑ r : Fin (m + 1), (mu0 + deltaG + eps r)) / ((m + 1 : ℕ) : ℝ) =
        mu0 + deltaG + (∑ r : Fin (m + 1), eps r) / ((m + 1 : ℕ) : ℝ) := by
    rw [Finset.sum_add_distrib]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp [hm]
    try ring
  rw [hMean]
  apply Finset.sum_congr rfl
  intro r _
  ring

/-- **Exact Second-Moment Decomposition of Untempered Raw Prior-Residual Square**:
    Under any normalized probability law `w` (`∑ w = 1`) with zero-mean noise
    (`∑ w * eps = 0`), `E[(mu0 + deltaG + eps - mu0)^2] = deltaG^2 + E[eps^2]`. -/
theorem untempered_raw_residual_second_moment
    {Ω : Type*} [Fintype Ω]
    (w : Ω → ℝ) (mu0 deltaG : ℝ) (eps : Ω → ℝ)
    (hProb : (∑ ω, w ω) = 1)
    (hZeroMean : (∑ ω, w ω * eps ω) = 0) :
    (∑ ω, w ω * ((mu0 + deltaG + eps ω) - mu0) ^ 2) =
      deltaG ^ 2 + ∑ ω, w ω * (eps ω) ^ 2 := by
  have hExpand :
      (∑ ω, w ω * ((mu0 + deltaG + eps ω) - mu0) ^ 2) =
        (∑ ω, w ω * deltaG ^ 2) + (∑ ω, (2 * deltaG) * (w ω * eps ω)) +
          ∑ ω, w ω * (eps ω) ^ 2 := by
    rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro ω _
    ring
  rw [hExpand, ← Finset.sum_mul, ← Finset.mul_sum, hProb, hZeroMean]
  ring

/-- **Exponential Variance Blow-Up Lower Bound under Log-Normal Moment Coupling**. -/
theorem untempered_exponential_variance_blowup
    (muEta0 muEtaN sigmaEtaSq gamma deltaG : ℝ)
    (hSigma : 0 ≤ sigmaEtaSq)
    (hLeakage : muEta0 + gamma * deltaG ^ 2 ≤ muEtaN) :
    Real.exp muEta0 * Real.exp (gamma * deltaG ^ 2) ≤
      Real.exp (muEtaN + sigmaEtaSq / 2) := by
  rw [← Real.exp_add]
  apply Real.exp_le_exp.mpr
  linarith

/-- **Hard-Cutoff (`AFTER`) Non-Vanishing Regret Lower Bound**. -/
theorem frozen_after_nonvanishing_error_lower_bound
    {X : Type*} [Fintype X]
    (w err : X → ℝ) (xStar : X) (deltaPerp : ℝ)
    (hwNonneg : ∀ x, 0 ≤ w x)
    (hSqLower : deltaPerp ^ 2 ≤ (err xStar) ^ 2) :
    w xStar * deltaPerp ^ 2 ≤ ∑ x, w x * (err x) ^ 2 := by
  have hSingle :
      w xStar * (err xStar) ^ 2 ≤ ∑ x, w x * (err x) ^ 2 :=
    Finset.single_le_sum (fun x _ => mul_nonneg (hwNonneg x) (sq_nonneg (err x)))
      (Finset.mem_univ xStar)
  have hPoint : w xStar * deltaPerp ^ 2 ≤ w xStar * (err xStar) ^ 2 :=
    mul_le_mul_of_nonneg_left hSqLower (hwNonneg xStar)
  linarith

/-- Tempered posterior MSE `(tau^2 * Delta^2 + dataPrec) / (tau + dataPrec)^2`. -/
noncomputable def temperedPosteriorMSE (tau dataPrec delta : ℝ) : ℝ :=
  (tau ^ 2 * delta ^ 2 + dataPrec) / (tau + dataPrec) ^ 2

/-- **Oracle Zero-Shift Strict Gain** (`Delta = 0`, `tau > 0`). -/
theorem tempered_strictly_beats_target_only_at_zero_shift
    (tau dataPrec : ℝ) (hTau : 0 < tau) (hData : 0 < dataPrec) :
    temperedPosteriorMSE tau dataPrec 0 < 1 / dataPrec := by
  unfold temperedPosteriorMSE
  have hPosSum : 0 < tau + dataPrec := by linarith
  have hDenom : 0 < (tau + dataPrec) ^ 2 := sq_pos_of_pos hPosSum
  have hId : (tau ^ 2 * 0 ^ 2 + dataPrec) / (tau + dataPrec) ^ 2 = dataPrec / (tau + dataPrec) ^ 2 := by ring
  rw [hId]
  have hGap : 0 < (1 / dataPrec) - (dataPrec / (tau + dataPrec) ^ 2) := by
    have hDiff : (1 / dataPrec) - (dataPrec / (tau + dataPrec) ^ 2) =
        (tau ^ 2 + 2 * tau * dataPrec) / (dataPrec * (tau + dataPrec) ^ 2) := by
      field_simp [ne_of_gt hData, ne_of_gt hDenom]
      ring
    rw [hDiff]
    positivity
  linarith

/-- **Oracle Optimal Tempering Inequality** (`tau* = 1 / Delta^2`). -/
theorem optimal_tempering_oracle_inequality
    (dataPrec delta : ℝ) (hData : 0 < dataPrec) (hDelta : delta ≠ 0) :
    temperedPosteriorMSE (1 / delta ^ 2) dataPrec delta ≤ min (delta ^ 2) (1 / dataPrec) := by
  unfold temperedPosteriorMSE
  have hd2 : 0 < delta ^ 2 := sq_pos_of_ne_zero hDelta
  have hTauPos : 0 < 1 / delta ^ 2 := div_pos one_pos hd2
  have hSumPos : 0 < 1 / delta ^ 2 + dataPrec := add_pos hTauPos hData
  have hSimplify :
      ((1 / delta ^ 2) ^ 2 * delta ^ 2 + dataPrec) / (1 / delta ^ 2 + dataPrec) ^ 2 =
        1 / (1 / delta ^ 2 + dataPrec) := by
    field_simp [ne_of_gt hd2, ne_of_gt hSumPos]
    try ring
  rw [hSimplify]
  apply le_min
  · have hDiff : delta ^ 2 - 1 / (1 / delta ^ 2 + dataPrec) =
        (delta ^ 2 * dataPrec) / (1 / delta ^ 2 + dataPrec) := by
      field_simp [ne_of_gt hd2, ne_of_gt hSumPos]
      try ring
    have hNonneg : 0 ≤ (delta ^ 2 * dataPrec) / (1 / delta ^ 2 + dataPrec) := by positivity
    linarith
  · have hDiff2 : 1 / dataPrec - 1 / (1 / delta ^ 2 + dataPrec) =
        (1 / delta ^ 2) / (dataPrec * (1 / delta ^ 2 + dataPrec)) := by
      field_simp [ne_of_gt hData, ne_of_gt hSumPos]
      try ring
    have hNonneg2 : 0 ≤ (1 / delta ^ 2) / (dataPrec * (1 / delta ^ 2 + dataPrec)) := by positivity
    linarith

end SCOLHKG.V2OR.TemperedHVD
