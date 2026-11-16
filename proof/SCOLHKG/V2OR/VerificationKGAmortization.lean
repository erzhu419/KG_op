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
# Theorem 3 Formal Core: Verification-Aware KG Nonnegativity, Objective-Guarded 3-Slot
# Exact Finite-Sample Safety (`FWER <= delta`), and Cross-Task Amortization Threshold `M_break`
-/

namespace SCOLHKG.V2OR.VerificationKGAmortization

set_option linter.unusedSectionVars false

open MeasureTheory
open scoped BigOperators

section VKG

/-- **One-Step `V-KG` Nonnegativity (Jensen / Tower Law for Any Monotone Expectation Operator)**:
    Applies to conditional expectation `E_n[.]` as well as positive-weight Gauss-Hermite quadrature. -/
theorem v_kg_information_value_nonneg
    {Ω J : Type*}
    (expectOp : (Ω → ℝ) → ℝ)
    (E : Finset J) (hNE : E.Nonempty)
    (VCurr : J → ℝ) (VNext : J → Ω → ℝ)
    (hMono : ∀ f g : Ω → ℝ, (∀ ω, f ω ≤ g ω) → expectOp f ≤ expectOp g)
    (hTower : ∀ j ∈ E, expectOp (VNext j) = VCurr j) :
    0 ≤ expectOp (fun ω => E.sup' hNE (fun j => VNext j ω)) - E.sup' hNE VCurr := by
  have hLe : E.sup' hNE VCurr ≤ expectOp (fun ω => E.sup' hNE (fun j => VNext j ω)) := by
    apply Finset.sup'_le hNE
    intro j hj
    rw [← hTower j hj]
    apply hMono
    intro ω
    exact Finset.le_sup' (fun k => VNext k ω) hj
  linarith

end VKG

section TerminalGate

variable {Ω : Type*} [MeasurableSpace Ω]

/-- Decision tree of `verify_and_deploy_shortlist` (`v2_or/verification/gated_verifier_v2.py`). -/
structure V2TerminalOutcome where
  certified : Bool
  selectedSlot : Fin 3
  calls : ℕ
  deriving DecidableEq

def v2TerminalDecision (a0 a1 a2 obj0Better : Bool) : V2TerminalOutcome :=
  if a0 then
    if a1 then
      if obj0Better then ⟨true, 0, 240⟩ else ⟨true, 1, 240⟩
    else ⟨true, 0, 240⟩
  else if a1 then ⟨true, 1, 240⟩
  else if a2 then ⟨true, 2, 380⟩
  else ⟨false, 0, 380⟩

/-- Event that a certified deployment selects an unsafe policy (`isUnsafe k = true`). -/
def UnsafeCertifiedDeployment
    (a0 a1 a2 obj0Better : Ω → Bool) (isUnsafe : Fin 3 → Bool) : Set Ω :=
  {ω |
    let out := v2TerminalDecision (a0 ω) (a1 ω) (a2 ω) (obj0Better ω)
    out.certified = true ∧ isUnsafe out.selectedSlot = true}

/-- **Key Structural Lemma**: Deploying an unsafe slot `k` requires its own safety gate `a_k`
    to pass; `obj0Better` adds ZERO false-certificate probability! -/
theorem v2_objective_guarded_unsafe_deployment_subset
    (a0 a1 a2 obj0Better : Ω → Bool) (isUnsafe : Fin 3 → Bool) :
    UnsafeCertifiedDeployment a0 a1 a2 obj0Better isUnsafe ⊆
      (if isUnsafe 0 then {ω | a0 ω = true} else ∅) ∪
        ((if isUnsafe 1 then {ω | a1 ω = true} else ∅) ∪
          (if isUnsafe 2 then {ω | a2 ω = true} else ∅)) := by
  intro ω hω
  change
    (v2TerminalDecision (a0 ω) (a1 ω) (a2 ω) (obj0Better ω)).certified = true ∧
      isUnsafe (v2TerminalDecision (a0 ω) (a1 ω) (a2 ω) (obj0Better ω)).selectedSlot = true at hω
  cases h0 : a0 ω <;> cases h1 : a1 ω <;> cases h2 : a2 ω <;>
    cases hObj : obj0Better ω <;>
    simp [v2TerminalDecision, h0, h1, h2, hObj] at hω ⊢
  all_goals
    simp [hω]

/-- **Exact Finite-Sample FWER Bound `<= d0 + d1 + d2`**. -/
theorem v2_terminal_deployment_fwer_le_delta
    (μ : Measure Ω) [IsFiniteMeasure μ]
    (a0 a1 a2 obj0Better : Ω → Bool) (isUnsafe : Fin 3 → Bool)
    (d0 d1 d2 : ℝ) (hd0 : 0 ≤ d0) (hd1 : 0 ≤ d1) (hd2 : 0 ≤ d2)
    (hGate0 : isUnsafe 0 = true → μ.real {ω | a0 ω = true} ≤ d0)
    (hGate1 : isUnsafe 1 = true → μ.real {ω | a1 ω = true} ≤ d1)
    (hGate2 : isUnsafe 2 = true → μ.real {ω | a2 ω = true} ≤ d2) :
    μ.real (UnsafeCertifiedDeployment a0 a1 a2 obj0Better isUnsafe) ≤ d0 + d1 + d2 := by
  have hSub := v2_objective_guarded_unsafe_deployment_subset a0 a1 a2 obj0Better isUnsafe
  have hMono := measureReal_mono (μ := μ) hSub
  have hU1 := measureReal_union_le (μ := μ)
    (if isUnsafe 0 then {ω | a0 ω = true} else ∅)
    ((if isUnsafe 1 then {ω | a1 ω = true} else ∅) ∪ (if isUnsafe 2 then {ω | a2 ω = true} else ∅))
  have hU2 := measureReal_union_le (μ := μ)
    (if isUnsafe 1 then {ω | a1 ω = true} else ∅) (if isUnsafe 2 then {ω | a2 ω = true} else ∅)
  have hb0 : μ.real (if isUnsafe 0 then {ω | a0 ω = true} else ∅) ≤ d0 := by
    cases u0 : isUnsafe 0
    · simp; exact hd0
    · simp; exact hGate0 u0
  have hb1 : μ.real (if isUnsafe 1 then {ω | a1 ω = true} else ∅) ≤ d1 := by
    cases u1 : isUnsafe 1
    · simp; exact hd1
    · simp; exact hGate1 u1
  have hb2 : μ.real (if isUnsafe 2 then {ω | a2 ω = true} else ∅) ≤ d2 := by
    cases u2 : isUnsafe 2
    · simp; exact hd2
    · simp; exact hGate2 u2
  linarith

/-- **Asymmetric 5% Familywise Error Budget (`0.025 + 0.015 + 0.010 = 0.05`)**. -/
theorem v2_terminal_five_percent_asymmetric_spend
    (μ : Measure Ω) [IsFiniteMeasure μ]
    (a0 a1 a2 obj0Better : Ω → Bool) (isUnsafe : Fin 3 → Bool)
    (hGate0 : isUnsafe 0 = true → μ.real {ω | a0 ω = true} ≤ (25 : ℝ) / 1000)
    (hGate1 : isUnsafe 1 = true → μ.real {ω | a1 ω = true} ≤ (15 : ℝ) / 1000)
    (hGate2 : isUnsafe 2 = true → μ.real {ω | a2 ω = true} ≤ (10 : ℝ) / 1000) :
    μ.real (UnsafeCertifiedDeployment a0 a1 a2 obj0Better isUnsafe) ≤ (1 : ℝ) / 20 := by
  have h := v2_terminal_deployment_fwer_le_delta μ a0 a1 a2 obj0Better isUnsafe
    ((25 : ℝ) / 1000) ((15 : ℝ) / 1000) ((10 : ℝ) / 1000)
    (by norm_num) (by norm_num) (by norm_num) hGate0 hGate1 hGate2
  linarith

/-- Distribution-free binomial zero-violation monotonicity: for any unsafe policy with
    `alpha <= pViol <= 1`, `(1 - pViol)^K <= (1 - alpha)^K`. -/
theorem binomial_zero_violation_power_bound
    (K : ℕ) (alpha pViol : ℝ)
    (hLe1 : pViol ≤ 1)
    (hUnsafe : alpha ≤ pViol) :
    (1 - pViol) ^ K ≤ (1 - alpha) ^ K := by
  have hNonneg : 0 ≤ 1 - pViol := by linarith
  have hMono : 1 - pViol ≤ 1 - alpha := by linarith
  gcongr

/-- Verification call cap `<= 380` across all branches of `v2TerminalDecision`. -/
theorem v2_terminal_calls_le_cap (a0 a1 a2 obj0Better : Bool) :
    (v2TerminalDecision a0 a1 a2 obj0Better).calls ≤ 380 := by
  cases a0 <;> cases a1 <;> cases a2 <;> cases obj0Better <;>
    simp [v2TerminalDecision]

end TerminalGate

section Amortization

/-- **Exact Cross-Task Source-Cost Amortization Break-Even Theorem (`M_break`)**. -/
theorem amortized_cost_le_cold_start_iff
    (B_src B_tls B_cold M : ℝ)
    (hM : 0 < M)
    (hSaving : B_tls < B_cold) :
    B_tls + B_src / M ≤ B_cold ↔ B_src / (B_cold - B_tls) ≤ M := by
  have hDiff : 0 < B_cold - B_tls := sub_pos.mpr hSaving
  have hMNe : M ≠ 0 := ne_of_gt hM
  have hDiffNe : B_cold - B_tls ≠ 0 := ne_of_gt hDiff
  constructor
  · intro hLe
    have h1 : B_src / M ≤ B_cold - B_tls := by linarith
    have h2 : M * (B_src / M) ≤ M * (B_cold - B_tls) :=
      mul_le_mul_of_nonneg_left h1 hM.le
    have h3 : M * (B_src / M) = B_src := by field_simp [hMNe]
    rw [h3] at h2
    have hId : M - B_src / (B_cold - B_tls) =
        (M * (B_cold - B_tls) - B_src) / (B_cold - B_tls) := by
      field_simp [hDiffNe]
    have hNonneg : 0 ≤ (M * (B_cold - B_tls) - B_src) / (B_cold - B_tls) :=
      div_nonneg (by linarith) hDiff.le
    linarith
  · intro hGe
    have h1 : (B_cold - B_tls) * (B_src / (B_cold - B_tls)) ≤ (B_cold - B_tls) * M :=
      mul_le_mul_of_nonneg_left hGe hDiff.le
    have h2 : (B_cold - B_tls) * (B_src / (B_cold - B_tls)) = B_src := by field_simp [hDiffNe]
    rw [h2] at h1
    have hId : B_cold - (B_tls + B_src / M) =
        ((B_cold - B_tls) * M - B_src) / M := by
      field_simp [hMNe]
      try ring
    have hNonneg : 0 ≤ ((B_cold - B_tls) * M - B_src) / M :=
      div_nonneg (by linarith) hM.le
    linarith

end Amortization

end SCOLHKG.V2OR.VerificationKGAmortization
