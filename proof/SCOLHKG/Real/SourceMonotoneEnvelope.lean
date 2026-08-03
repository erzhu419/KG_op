import Mathlib

namespace SCOLHKG.Real

/-!
# Fail-closed source-monotone envelope

The promoted V3 front end may reserve one proposal slot for a boundary
endpoint.  The endpoint is admitted only when every frozen source task has a
rank-correlation magnitude above the registered threshold and all signs
agree.  Source agreement is evidence, not by itself a target guarantee.  A
target-feasibility conclusion additionally requires the corresponding
source-to-target monotonicity condition.

If agreement fails, the implementation returns no endpoint and the original
atlas is preserved exactly.  This file records both sides of that contract.
-/

def SourceNegativeAgreement {Source : Type*} [Fintype Source]
    (correlation : Source → ℝ) (threshold : ℝ) : Prop :=
  2 ≤ Fintype.card Source ∧ ∀ source, correlation source ≤ -threshold

def SourcePositiveAgreement {Source : Type*} [Fintype Source]
    (correlation : Source → ℝ) (threshold : ℝ) : Prop :=
  2 ≤ Fintype.card Source ∧ ∀ source, threshold ≤ correlation source

def SourceMonotoneEnvelopeAdmitted {Source : Type*} [Fintype Source]
    (correlation : Source → ℝ) (threshold : ℝ) : Prop :=
  SourceNegativeAgreement correlation threshold ∨
    SourcePositiveAgreement correlation threshold

def CoordinateNonincreasingMargin {X : Type*}
    (coordinate margin : X → ℝ) : Prop :=
  ∀ x y, coordinate x ≤ coordinate y → margin y ≤ margin x

def CoordinateNondecreasingMargin {X : Type*}
    (coordinate margin : X → ℝ) : Prop :=
  ∀ x y, coordinate x ≤ coordinate y → margin x ≤ margin y

def failClosedEnvelopeProposal {X : Type*}
    (baseline : List X) (endpoint : X) (admitted : Bool) : List X :=
  if admitted then
    match baseline with
    | [] => []
    | _ :: tail => endpoint :: tail
  else baseline

theorem rejected_envelope_preserves_baseline {X : Type*}
    (baseline : List X) (endpoint : X) :
    failClosedEnvelopeProposal baseline endpoint false = baseline := by
  simp [failClosedEnvelopeProposal]

theorem fail_closed_envelope_preserves_budget {X : Type*}
    (baseline : List X) (endpoint : X) (admitted : Bool) :
    (failClosedEnvelopeProposal baseline endpoint admitted).length =
      baseline.length := by
  cases admitted <;> cases baseline <;> simp [failClosedEnvelopeProposal]

theorem admitted_envelope_occupies_one_existing_slot {X : Type*}
    (first endpoint : X) (tail : List X) :
    endpoint ∈ failClosedEnvelopeProposal (first :: tail) endpoint true := by
  simp [failClosedEnvelopeProposal]

theorem upper_endpoint_safe_of_transferred_nonincreasing_margin
    {X : Type*} {coordinate margin : X → ℝ} {upper : X}
    (hMonotone : CoordinateNonincreasingMargin coordinate margin)
    (hUpper : ∀ x, coordinate x ≤ coordinate upper)
    (hSafeWitness : ∃ x, margin x ≤ 0) :
    margin upper ≤ 0 := by
  obtain ⟨x, hSafe⟩ := hSafeWitness
  exact (hMonotone x upper (hUpper x)).trans hSafe

theorem lower_endpoint_safe_of_transferred_nondecreasing_margin
    {X : Type*} {coordinate margin : X → ℝ} {lower : X}
    (hMonotone : CoordinateNondecreasingMargin coordinate margin)
    (hLower : ∀ x, coordinate lower ≤ coordinate x)
    (hSafeWitness : ∃ x, margin x ≤ 0) :
    margin lower ≤ 0 := by
  obtain ⟨x, hSafe⟩ := hSafeWitness
  exact (hMonotone lower x (hLower x)).trans hSafe

theorem admitted_source_envelope_safe_under_transferred_direction
    {Source X : Type*} [Fintype Source]
    {correlation : Source → ℝ} {threshold : ℝ}
    {coordinate margin : X → ℝ} {lower upper : X}
    (hAdmitted :
      SourceMonotoneEnvelopeAdmitted correlation threshold)
    (hUpper : ∀ x, coordinate x ≤ coordinate upper)
    (hLower : ∀ x, coordinate lower ≤ coordinate x)
    (hNegativeTransfer :
      SourceNegativeAgreement correlation threshold →
        CoordinateNonincreasingMargin coordinate margin)
    (hPositiveTransfer :
      SourcePositiveAgreement correlation threshold →
        CoordinateNondecreasingMargin coordinate margin)
    (hSafeWitness : ∃ x, margin x ≤ 0) :
    (SourceNegativeAgreement correlation threshold ∧ margin upper ≤ 0)
      ∨
    (SourcePositiveAgreement correlation threshold ∧ margin lower ≤ 0) := by
  rcases hAdmitted with hNegative | hPositive
  · left
    exact ⟨hNegative,
      upper_endpoint_safe_of_transferred_nonincreasing_margin
        (hNegativeTransfer hNegative) hUpper hSafeWitness⟩
  · right
    exact ⟨hPositive,
      lower_endpoint_safe_of_transferred_nondecreasing_margin
        (hPositiveTransfer hPositive) hLower hSafeWitness⟩

def sourceEnvelopeFromFrozenRecords
    {SourceRecords Descriptor TargetLabels X : Type*}
    (select : SourceRecords → Descriptor → Option X)
    (source : SourceRecords) (descriptor : Descriptor)
    (_targetLabels : TargetLabels) : Option X :=
  select source descriptor

theorem source_envelope_target_label_noninterference
    {SourceRecords Descriptor TargetLabels X : Type*}
    (select : SourceRecords → Descriptor → Option X)
    (source : SourceRecords) (descriptor : Descriptor)
    (leftLabels rightLabels : TargetLabels) :
    sourceEnvelopeFromFrozenRecords
        select source descriptor leftLabels =
      sourceEnvelopeFromFrozenRecords
        select source descriptor rightLabels := by
  rfl

end SCOLHKG.Real
