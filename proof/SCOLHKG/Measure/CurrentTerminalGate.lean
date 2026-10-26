import SCOLHKG.Measure.GaussianReplicationCertificate

namespace SCOLHKG.Measure

open MeasureTheory

/-!
Current 80/128/128 three-policy terminal, with an optional eight-pair comparison.

The statistical interface is the integrated conditional-error inequality
`μ.real (B ∩ E) ≤ δ * μ.real B`.  It follows from a conditional bound on the
fresh block, but this module does not formalize Gaussian sampling, noncentral-t
pivots, quantile computation, or random-stream independence.  Those facts must
supply the four local error bounds; the final familywise bound is derived here.
The event algebra and return/cost tree use a genuine probability measure.
-/

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

/-- The fallback safety block runs only when both initial certificates fail. -/
def CurrentThirdSafetyGate (A₁ A₂ : Set Ω) : Set Ω := A₁ᶜ ∩ A₂ᶜ

/-- The objective comparison runs only when both initial certificates pass. -/
def CurrentComparisonGate (A₁ A₂ : Set Ω) : Set Ω := A₁ ∩ A₂

theorem current_terminal_gates_disjoint (A₁ A₂ : Set Ω) :
    Disjoint (CurrentThirdSafetyGate A₁ A₂) (CurrentComparisonGate A₁ A₂) := by
  refine Set.disjoint_left.2 ?_
  intro ω h₃ hc
  exact h₃.1 hc.1

/-- Disjoint measurable gates can spend at most one unit of probability mass. -/
theorem current_terminal_gate_mass_le_one
    [IsProbabilityMeasure μ]
    (A₁ A₂ : Set Ω) (hA₁ : MeasurableSet A₁) (hA₂ : MeasurableSet A₂) :
    μ.real (CurrentThirdSafetyGate A₁ A₂)
      + μ.real (CurrentComparisonGate A₁ A₂) ≤ 1 := by
  rw [← measureReal_union (current_terminal_gates_disjoint A₁ A₂) (hA₁.inter hA₂)]
  simpa only [probReal_univ] using (measureReal_mono
    (Set.subset_univ (CurrentThirdSafetyGate A₁ A₂ ∪ CurrentComparisonGate A₁ A₂)) :
      μ.real (CurrentThirdSafetyGate A₁ A₂ ∪ CurrentComparisonGate A₁ A₂) ≤ μ.real Set.univ)

/-- Two mutually exclusive fresh-block tests share the larger local budget. -/
theorem current_terminal_gated_errors_probability_le
    [IsProbabilityMeasure μ]
    (A₁ A₂ E₃ Ec : Set Ω) (hA₁ : MeasurableSet A₁) (hA₂ : MeasurableSet A₂)
    (δ₃ δc : ℝ) (hδ₃ : 0 ≤ δ₃) (hδc : 0 ≤ δc)
    (h₃ : μ.real (CurrentThirdSafetyGate A₁ A₂ ∩ E₃)
      ≤ δ₃ * μ.real (CurrentThirdSafetyGate A₁ A₂))
    (hc : μ.real (CurrentComparisonGate A₁ A₂ ∩ Ec)
      ≤ δc * μ.real (CurrentComparisonGate A₁ A₂)) :
    μ.real ((CurrentThirdSafetyGate A₁ A₂ ∩ E₃)
      ∪ (CurrentComparisonGate A₁ A₂ ∩ Ec)) ≤ max δ₃ δc := by
  have hm := current_terminal_gate_mass_le_one (μ := μ) A₁ A₂ hA₁ hA₂
  have hmax : 0 ≤ max δ₃ δc := le_trans hδ₃ (le_max_left _ _)
  have h₃' := mul_le_mul_of_nonneg_right (le_max_left δ₃ δc)
    (show 0 ≤ μ.real (CurrentThirdSafetyGate A₁ A₂) from ENNReal.toReal_nonneg)
  have hc' := mul_le_mul_of_nonneg_right (le_max_right δ₃ δc)
    (show 0 ≤ μ.real (CurrentComparisonGate A₁ A₂) from ENNReal.toReal_nonneg)
  calc
    μ.real ((CurrentThirdSafetyGate A₁ A₂ ∩ E₃)
        ∪ (CurrentComparisonGate A₁ A₂ ∩ Ec))
      ≤ μ.real (CurrentThirdSafetyGate A₁ A₂ ∩ E₃)
        + μ.real (CurrentComparisonGate A₁ A₂ ∩ Ec) := measureReal_union_le _ _
    _ ≤ max δ₃ δc * (μ.real (CurrentThirdSafetyGate A₁ A₂)
        + μ.real (CurrentComparisonGate A₁ A₂)) := by nlinarith
    _ ≤ max δ₃ δc * 1 := mul_le_mul_of_nonneg_left hm hmax
    _ = max δ₃ δc := mul_one _

/-- Errors of issued certificates or an invoked objective-improvement test. -/
def CurrentTerminalJointError (A₁ A₂ E₁ E₂ E₃ Ec : Set Ω) : Set Ω :=
  (E₁ ∪ E₂) ∪ ((CurrentThirdSafetyGate A₁ A₂ ∩ E₃)
    ∪ (CurrentComparisonGate A₁ A₂ ∩ Ec))

theorem current_terminal_joint_error_probability_le
    [IsProbabilityMeasure μ]
    (A₁ A₂ E₁ E₂ E₃ Ec : Set Ω)
    (hA₁ : MeasurableSet A₁) (hA₂ : MeasurableSet A₂)
    (δ₁ δ₂ δ₃ δc : ℝ) (hδ₃ : 0 ≤ δ₃) (hδc : 0 ≤ δc)
    (h₁ : μ.real E₁ ≤ δ₁) (h₂ : μ.real E₂ ≤ δ₂)
    (h₃ : μ.real (CurrentThirdSafetyGate A₁ A₂ ∩ E₃)
      ≤ δ₃ * μ.real (CurrentThirdSafetyGate A₁ A₂))
    (hc : μ.real (CurrentComparisonGate A₁ A₂ ∩ Ec)
      ≤ δc * μ.real (CurrentComparisonGate A₁ A₂)) :
    μ.real (CurrentTerminalJointError A₁ A₂ E₁ E₂ E₃ Ec)
      ≤ δ₁ + δ₂ + max δ₃ δc := by
  have hlate := current_terminal_gated_errors_probability_le
    (μ := μ) A₁ A₂ E₃ Ec hA₁ hA₂ δ₃ δc hδ₃ hδc h₃ hc
  have hfirst := measureReal_union_le (μ := μ) E₁ E₂
  have hall := measureReal_union_le (μ := μ) (E₁ ∪ E₂)
    ((CurrentThirdSafetyGate A₁ A₂ ∩ E₃) ∪ (CurrentComparisonGate A₁ A₂ ∩ Ec))
  unfold CurrentTerminalJointError
  linarith

/-- Three equal shares of 1/60 yield the joint 1/20 error guarantee. -/
theorem current_terminal_five_percent
    [IsProbabilityMeasure μ]
    (A₁ A₂ E₁ E₂ E₃ Ec : Set Ω)
    (hA₁ : MeasurableSet A₁) (hA₂ : MeasurableSet A₂)
    (h₁ : μ.real E₁ ≤ (1 : ℝ) / 60) (h₂ : μ.real E₂ ≤ (1 : ℝ) / 60)
    (h₃ : μ.real (CurrentThirdSafetyGate A₁ A₂ ∩ E₃)
      ≤ ((1 : ℝ) / 60) * μ.real (CurrentThirdSafetyGate A₁ A₂))
    (hc : μ.real (CurrentComparisonGate A₁ A₂ ∩ Ec)
      ≤ ((1 : ℝ) / 60) * μ.real (CurrentComparisonGate A₁ A₂)) :
    μ.real (CurrentTerminalJointError A₁ A₂ E₁ E₂ E₃ Ec) ≤ (1 : ℝ) / 20 := by
  have h := current_terminal_joint_error_probability_le (μ := μ)
    A₁ A₂ E₁ E₂ E₃ Ec hA₁ hA₂ (1 / 60) (1 / 60) (1 / 60) (1 / 60)
    (by norm_num) (by norm_num) h₁ h₂ h₃ hc
  norm_num at h ⊢
  exact h

/-- The completed terminal tree returns policy 0 on abstention, without a certificate.
Its call count excludes additional infrastructure-failure/recovery costs. -/
structure CurrentTerminalDecision where
  policy : Fin 3
  certified : Bool
  improvementDeclared : Bool
  calls : ℕ
  deriving DecidableEq

def currentTerminalDecision (a₁ a₂ a₃ improves : Bool) : CurrentTerminalDecision :=
  if a₁ then
    if a₂ then
      if improves then ⟨0, true, true, 224⟩ else ⟨1, true, false, 224⟩
    else ⟨0, true, false, 208⟩
  else if a₂ then ⟨1, true, false, 208⟩
    else if a₃ then ⟨2, true, false, 336⟩ else ⟨0, false, false, 336⟩

theorem current_terminal_abstention (improves : Bool) :
    currentTerminalDecision false false false improves = ⟨0, false, false, 336⟩ := by
  simp [currentTerminalDecision]

theorem current_terminal_calls_exact (a₁ a₂ a₃ improves : Bool) :
    (currentTerminalDecision a₁ a₂ a₃ improves).calls = 208
      ∨ (currentTerminalDecision a₁ a₂ a₃ improves).calls = 224
      ∨ (currentTerminalDecision a₁ a₂ a₃ improves).calls = 336 := by
  cases a₁ <;> cases a₂ <;> cases a₃ <;> cases improves <;>
    simp [currentTerminalDecision]

theorem current_terminal_calls_le_registered_cap (a₁ a₂ a₃ improves : Bool) :
    (currentTerminalDecision a₁ a₂ a₃ improves).calls ≤ 352 := by
  rcases current_terminal_calls_exact a₁ a₂ a₃ improves with h | h | h <;> omega

theorem current_terminal_calls_branch_identity (a₁ a₂ a₃ improves : Bool) :
    (currentTerminalDecision a₁ a₂ a₃ improves).calls
      = 208 + (if a₁ && a₂ then 16 else 0) + (if !a₁ && !a₂ then 128 else 0) := by
  cases a₁ <;> cases a₂ <;> cases a₃ <;> cases improves <;>
    simp [currentTerminalDecision]

/-- A certificate error for a fixed policy.  The unsafe proposition is about truth. -/
def CurrentSafetyError (accepted : Ω → Bool) (isUnsafe : Prop) : Set Ω :=
  {ω | accepted ω = true ∧ isUnsafe}

def CurrentIssuedFalseCertificate (a₁ a₂ a₃ : Ω → Bool) (isUnsafe : Fin 3 → Prop) : Set Ω :=
  (CurrentSafetyError a₁ (isUnsafe 0) ∪ CurrentSafetyError a₂ (isUnsafe 1))
    ∪ (CurrentThirdSafetyGate {ω | a₁ ω = true} {ω | a₂ ω = true}
      ∩ CurrentSafetyError a₃ (isUnsafe 2))

def CurrentUnsafeCertifiedReturn
    (a₁ a₂ a₃ improves : Ω → Bool) (isUnsafe : Fin 3 → Prop) : Set Ω :=
  {ω | (currentTerminalDecision (a₁ ω) (a₂ ω) (a₃ ω) (improves ω)).certified = true
    ∧ isUnsafe (currentTerminalDecision (a₁ ω) (a₂ ω) (a₃ ω) (improves ω)).policy}

theorem current_unsafe_certified_return_subset_issued_false_certificate
    (a₁ a₂ a₃ improves : Ω → Bool) (isUnsafe : Fin 3 → Prop) :
    CurrentUnsafeCertifiedReturn a₁ a₂ a₃ improves isUnsafe
      ⊆ CurrentIssuedFalseCertificate a₁ a₂ a₃ isUnsafe := by
  intro ω h
  cases h₁ : a₁ ω <;> cases h₂ : a₂ ω <;> cases h₃ : a₃ ω <;>
    cases hj : improves ω <;>
    simp_all [CurrentUnsafeCertifiedReturn, CurrentIssuedFalseCertificate,
      CurrentSafetyError, CurrentThirdSafetyGate, currentTerminalDecision]

/-- A false declared improvement has both initial certificates and a false test. -/
theorem current_false_improvement_subset_comparison_gate
    (a₁ a₂ a₃ improves : Ω → Bool) (notBetter : Prop) :
    {ω | (currentTerminalDecision (a₁ ω) (a₂ ω) (a₃ ω) (improves ω)).improvementDeclared
        = true ∧ notBetter}
      ⊆ CurrentComparisonGate {ω | a₁ ω = true} {ω | a₂ ω = true}
        ∩ {ω | improves ω = true ∧ notBetter} := by
  intro ω h
  cases h₁ : a₁ ω <;> cases h₂ : a₂ ω <;> cases h₃ : a₃ ω <;>
    cases hj : improves ω <;>
    simp_all [CurrentComparisonGate, currentTerminalDecision]

/-- Connect the actual returned certificate/declaration to the error-budget event. -/
theorem current_return_or_false_improvement_subset_joint_error
    (a₁ a₂ a₃ improves : Ω → Bool) (isUnsafe : Fin 3 → Prop) (notBetter : Prop) :
    CurrentUnsafeCertifiedReturn a₁ a₂ a₃ improves isUnsafe
      ∪ {ω | (currentTerminalDecision (a₁ ω) (a₂ ω) (a₃ ω) (improves ω)).improvementDeclared
        = true ∧ notBetter}
      ⊆ CurrentTerminalJointError {ω | a₁ ω = true} {ω | a₂ ω = true}
        (CurrentSafetyError a₁ (isUnsafe 0)) (CurrentSafetyError a₂ (isUnsafe 1))
        (CurrentSafetyError a₃ (isUnsafe 2)) {ω | improves ω = true ∧ notBetter} := by
  intro ω h
  rcases h with hr | hw
  · have hi := current_unsafe_certified_return_subset_issued_false_certificate
      a₁ a₂ a₃ improves isUnsafe hr
    rcases hi with hfirst | hthird
    · exact Or.inl hfirst
    · exact Or.inr (Or.inl hthird)
  · exact Or.inr (Or.inr
      (current_false_improvement_subset_comparison_gate a₁ a₂ a₃ improves notBetter hw))

/-- Joint control for the certificate actually returned and the statement actually made. -/
theorem current_certified_return_or_false_improvement_five_percent
    [IsProbabilityMeasure μ]
    (a₁ a₂ a₃ improves : Ω → Bool) (isUnsafe : Fin 3 → Prop) (notBetter : Prop)
    (hA₁ : MeasurableSet {ω | a₁ ω = true})
    (hA₂ : MeasurableSet {ω | a₂ ω = true})
    (h₁ : μ.real (CurrentSafetyError a₁ (isUnsafe 0)) ≤ (1 : ℝ) / 60)
    (h₂ : μ.real (CurrentSafetyError a₂ (isUnsafe 1)) ≤ (1 : ℝ) / 60)
    (h₃ : μ.real (CurrentThirdSafetyGate {ω | a₁ ω = true} {ω | a₂ ω = true}
        ∩ CurrentSafetyError a₃ (isUnsafe 2))
      ≤ ((1 : ℝ) / 60)
        * μ.real (CurrentThirdSafetyGate {ω | a₁ ω = true} {ω | a₂ ω = true}))
    (hc : μ.real (CurrentComparisonGate {ω | a₁ ω = true} {ω | a₂ ω = true}
        ∩ {ω | improves ω = true ∧ notBetter})
      ≤ ((1 : ℝ) / 60)
        * μ.real (CurrentComparisonGate {ω | a₁ ω = true} {ω | a₂ ω = true})) :
    μ.real (CurrentUnsafeCertifiedReturn a₁ a₂ a₃ improves isUnsafe
      ∪ {ω | (currentTerminalDecision (a₁ ω) (a₂ ω) (a₃ ω) (improves ω)).improvementDeclared
        = true ∧ notBetter}) ≤ (1 : ℝ) / 20 := by
  exact (measureReal_mono
    (current_return_or_false_improvement_subset_joint_error a₁ a₂ a₃ improves isUnsafe notBetter)).trans
    (current_terminal_five_percent (μ := μ)
      {ω | a₁ ω = true} {ω | a₂ ω = true}
      (CurrentSafetyError a₁ (isUnsafe 0)) (CurrentSafetyError a₂ (isUnsafe 1))
      (CurrentSafetyError a₃ (isUnsafe 2)) {ω | improves ω = true ∧ notBetter}
      hA₁ hA₂ h₁ h₂ h₃ hc)

end SCOLHKG.Measure
