import Mathlib
import SCOLHKG.Real.OrthogonalAdditiveGroups

namespace SCOLHKG.Real

open scoped BigOperators

section Projector

variable {n k : ℕ}

def subspaceProjector
    (basis : Matrix (Fin n) (Fin k) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  basis * basis.transpose

def rotateSubspaceBasis
    (basis : Matrix (Fin n) (Fin k) ℝ)
    (rotation : Matrix (Fin k) (Fin k) ℝ) : Matrix (Fin n) (Fin k) ℝ :=
  basis * rotation

theorem subspaceProjector_rotation_invariant
    (basis : Matrix (Fin n) (Fin k) ℝ)
    (rotation : Matrix (Fin k) (Fin k) ℝ)
    (hRotation : rotation * rotation.transpose = 1) :
    subspaceProjector (rotateSubspaceBasis basis rotation) =
      subspaceProjector basis := by
  unfold subspaceProjector rotateSubspaceBasis
  rw [Matrix.transpose_mul]
  calc
    (basis * rotation) * (rotation.transpose * basis.transpose) =
        basis * (rotation * rotation.transpose) * basis.transpose := by
          simp only [Matrix.mul_assoc]
    _ = basis * basis.transpose := by simp [hRotation]

end Projector

section Whitening

def scaledColumn {n r : ℕ}
    (scale : Fin r → ℝ)
    (column : Fin r → Fin n → ℝ)
    (j : Fin r) : Fin n → ℝ :=
  fun i => scale j * column j i

theorem weightedInner_scaledColumn
    {n r : ℕ}
    (weight : Fin n → ℝ)
    (scale : Fin r → ℝ)
    (column : Fin r → Fin n → ℝ)
    (i j : Fin r) :
    weightedInner weight (scaledColumn scale column i)
        (scaledColumn scale column j) =
      scale i * scale j * weightedInner weight (column i) (column j) := by
  unfold weightedInner scaledColumn
  calc
    (∑ x, weight x * (scale i * column i x) *
      (scale j * column j x)) =
        ∑ x, (scale i * scale j) *
          (weight x * column i x * column j x) := by
            apply Finset.sum_congr rfl
            intro x _
            ring
    _ = scale i * scale j *
        ∑ x, weight x * column i x * column j x := by
          rw [Finset.mul_sum]

theorem retainedWhitening_orthonormal
    {n r : ℕ}
    (weight : Fin n → ℝ)
    (eigenvalue scale : Fin r → ℝ)
    (column : Fin r → Fin n → ℝ)
    (hOrthogonal : ∀ i j,
      weightedInner weight (column i) (column j) =
        if i = j then eigenvalue i else 0)
    (hNormalize : ∀ i, scale i * scale i * eigenvalue i = 1) :
    ∀ i j,
      weightedInner weight (scaledColumn scale column i)
          (scaledColumn scale column j) =
        if i = j then 1 else 0 := by
  intro i j
  rw [weightedInner_scaledColumn, hOrthogonal]
  by_cases hij : i = j
  · subst j
    simp [hNormalize]
  · simp [hij]

end Whitening

section ExpertMixture

def expertMixture {m : ℕ}
    (weight score : Fin m → ℝ) : ℝ :=
  ∑ i, weight i * score i

theorem simplexExpertMixture_mem_interval
    {m : ℕ}
    (lower upper : ℝ)
    (weight score : Fin m → ℝ)
    (hWeight : ∀ i, 0 ≤ weight i)
    (hSum : ∑ i, weight i = 1)
    (hLower : ∀ i, lower ≤ score i)
    (hUpper : ∀ i, score i ≤ upper) :
    lower ≤ expertMixture weight score ∧
      expertMixture weight score ≤ upper := by
  constructor
  · calc
      lower = ∑ i, weight i * lower := by
        calc
          lower = lower * ∑ i, weight i := by rw [hSum, mul_one]
          _ = ∑ i, weight i * lower := by
            rw [Finset.mul_sum]
            apply Finset.sum_congr rfl
            intro i _
            ring
      _ ≤ ∑ i, weight i * score i := by
        apply Finset.sum_le_sum
        intro i _
        exact mul_le_mul_of_nonneg_left (hLower i) (hWeight i)
      _ = expertMixture weight score := rfl
  · calc
      expertMixture weight score = ∑ i, weight i * score i := rfl
      _ ≤ ∑ i, weight i * upper := by
        apply Finset.sum_le_sum
        intro i _
        exact mul_le_mul_of_nonneg_left (hUpper i) (hWeight i)
      _ = upper := by
        calc
          (∑ i, weight i * upper) = upper * ∑ i, weight i := by
            rw [Finset.mul_sum]
            apply Finset.sum_congr rfl
            intro i _
            ring
          _ = upper := by rw [hSum, mul_one]

end ExpertMixture

section NestedLOO

def maskHeldout {n : ℕ} {α : Type}
    (fallback : α)
    (heldout : Fin n)
    (labels : Fin n → α) : Fin n → α :=
  fun i => if i = heldout then fallback else labels i

theorem maskHeldout_invariant
    {n : ℕ} {α : Type}
    (fallback : α)
    (heldout : Fin n)
    (left right : Fin n → α)
    (hTrainingEqual : ∀ i, i ≠ heldout → left i = right i) :
    maskHeldout fallback heldout left =
      maskHeldout fallback heldout right := by
  funext i
  by_cases hi : i = heldout
  · simp [maskHeldout, hi]
  · simp [maskHeldout, hi, hTrainingEqual i hi]

theorem nestedLOO_refit_does_not_read_heldout_label
    {n : ℕ} {α β : Type}
    (fitRepresentation : (Fin n → α) → β)
    (fallback : α)
    (heldout : Fin n)
    (left right : Fin n → α)
    (hTrainingEqual : ∀ i, i ≠ heldout → left i = right i) :
    fitRepresentation (maskHeldout fallback heldout left) =
      fitRepresentation (maskHeldout fallback heldout right) := by
  rw [maskHeldout_invariant fallback heldout left right hTrainingEqual]

end NestedLOO

section StrongHeredity

def StrongHeredity {m : ℕ}
    (main : Finset (Fin m))
    (interaction : Finset (Fin m × Fin m)) : Prop :=
  ∀ pair ∈ interaction, pair.1 ∈ main ∧ pair.2 ∈ main

theorem strongHeredity_survives_interaction_filter
    {m : ℕ}
    (main : Finset (Fin m))
    (available selected : Finset (Fin m × Fin m))
    (hHeredity : StrongHeredity main available)
    (hSubset : selected ⊆ available) :
    StrongHeredity main selected := by
  intro pair hPair
  exact hHeredity pair (hSubset hPair)

noncomputable def sourceSupportGate
    (minimum weight base candidate : ℝ) : ℝ :=
  if minimum ≤ weight then candidate else base

theorem weakSourceSupport_is_exact_fallback
    (minimum weight base candidate : ℝ)
    (hWeak : weight < minimum) :
    sourceSupportGate minimum weight base candidate = base := by
  simp [sourceSupportGate, not_le.mpr hWeak]

end StrongHeredity

section BoundaryEvidenceGate

noncomputable def boundaryEvidenceGate
    (nFeasible nInfeasible : ℕ)
    (base candidate : ℝ) : ℝ :=
  if 0 < nFeasible ∧ 0 < nInfeasible then candidate else base

theorem noFeasibleEvidence_is_exact_fallback
    (nInfeasible : ℕ)
    (base candidate : ℝ) :
    boundaryEvidenceGate 0 nInfeasible base candidate = base := by
  simp [boundaryEvidenceGate]

theorem noInfeasibleEvidence_is_exact_fallback
    (nFeasible : ℕ)
    (base candidate : ℝ) :
    boundaryEvidenceGate nFeasible 0 base candidate = base := by
  simp [boundaryEvidenceGate]

end BoundaryEvidenceGate

section SourceEpisodeAdmission

/-- Source episodes may replace noisy target gain evidence, but the target
two-sided boundary and target safety checks remain mandatory. -/
def SourceEpisodeAlignmentAdmission
    (twoSided targetSafe targetGainStable sourceEpisodeSupported : Prop) : Prop :=
  twoSided ∧ targetSafe ∧ (targetGainStable ∨ sourceEpisodeSupported)

theorem sourceEpisodeAdmission_requires_twoSided
    {twoSided targetSafe targetGainStable sourceEpisodeSupported : Prop}
    (h : SourceEpisodeAlignmentAdmission twoSided targetSafe
      targetGainStable sourceEpisodeSupported) :
    twoSided := by
  exact h.1

theorem sourceEpisodeAdmission_requires_targetSafety
    {twoSided targetSafe targetGainStable sourceEpisodeSupported : Prop}
    (h : SourceEpisodeAlignmentAdmission twoSided targetSafe
      targetGainStable sourceEpisodeSupported) :
    targetSafe := by
  exact h.2.1

theorem sourceEpisodeSupport_replaces_gainEvidence_only
    {twoSided targetSafe targetGainStable sourceEpisodeSupported : Prop}
    (hTwoSided : twoSided)
    (hTargetSafe : targetSafe)
    (hSource : sourceEpisodeSupported) :
    SourceEpisodeAlignmentAdmission twoSided targetSafe
      targetGainStable sourceEpisodeSupported := by
  exact ⟨hTwoSided, hTargetSafe, Or.inr hSource⟩

theorem oneSided_sourceEpisodeAdmission_rejected
    {twoSided targetSafe targetGainStable sourceEpisodeSupported : Prop}
    (hOneSided : ¬ twoSided) :
    ¬ SourceEpisodeAlignmentAdmission twoSided targetSafe
      targetGainStable sourceEpisodeSupported := by
  intro hAccepted
  exact hOneSided (sourceEpisodeAdmission_requires_twoSided hAccepted)

theorem unsafe_sourceEpisodeAdmission_rejected
    {twoSided targetSafe targetGainStable sourceEpisodeSupported : Prop}
    (hUnsafe : ¬ targetSafe) :
    ¬ SourceEpisodeAlignmentAdmission twoSided targetSafe
      targetGainStable sourceEpisodeSupported := by
  intro hAccepted
  exact hUnsafe (sourceEpisodeAdmission_requires_targetSafety hAccepted)

noncomputable def sourceEpisodeAlignmentGate
    (twoSided targetSafe targetGainStable sourceEpisodeSupported : Prop)
    (base candidate : ℝ) : ℝ :=
  by
    classical
    exact if SourceEpisodeAlignmentAdmission twoSided targetSafe
        targetGainStable sourceEpisodeSupported then candidate else base

theorem oneSided_sourceEpisodeGate_is_exact_fallback
    {twoSided targetSafe targetGainStable sourceEpisodeSupported : Prop}
    (base candidate : ℝ)
    (hOneSided : ¬ twoSided) :
    sourceEpisodeAlignmentGate twoSided targetSafe targetGainStable
      sourceEpisodeSupported base candidate = base := by
  simp [sourceEpisodeAlignmentGate,
    oneSided_sourceEpisodeAdmission_rejected hOneSided]

theorem unsafe_sourceEpisodeGate_is_exact_fallback
    {twoSided targetSafe targetGainStable sourceEpisodeSupported : Prop}
    (base candidate : ℝ)
    (hUnsafe : ¬ targetSafe) :
    sourceEpisodeAlignmentGate twoSided targetSafe targetGainStable
      sourceEpisodeSupported base candidate = base := by
  simp [sourceEpisodeAlignmentGate,
    unsafe_sourceEpisodeAdmission_rejected hUnsafe]

/-- A proposal fitted only from frozen source records is invariant to every
change in held-out target labels.  This is the architecture used by source
boundary profile replay. -/
def sourceOnlyProposal {α β γ : Type}
    (fitProposal : α → β)
    (sourceRecords : α)
    (_targetLabels : γ) : β :=
  fitProposal sourceRecords

theorem sourceOnlyProposal_targetLabel_invariant
    {α β γ : Type}
    (fitProposal : α → β)
    (sourceRecords : α)
    (leftTargetLabels rightTargetLabels : γ) :
    sourceOnlyProposal fitProposal sourceRecords leftTargetLabels =
      sourceOnlyProposal fitProposal sourceRecords rightTargetLabels := by
  rfl

end SourceEpisodeAdmission

section TransactionalRepresentationSwitch

/-- Reconstruct a posterior state by applying the recorded updates in their
original order under the newly selected feature semantics. -/
def replayPosterior {State Observation : Type}
    (update : State → Observation → State)
    (initial : State)
    (history : List Observation) : State :=
  history.foldl update initial

theorem replayPosterior_nil
    {State Observation : Type}
    (update : State → Observation → State)
    (initial : State) :
    replayPosterior update initial [] = initial := by
  rfl

theorem replayPosterior_cons
    {State Observation : Type}
    (update : State → Observation → State)
    (initial : State)
    (head : Observation)
    (tail : List Observation) :
    replayPosterior update initial (head :: tail) =
      replayPosterior update (update initial head) tail := by
  rfl

/-- A rejected representation proposal commits the old posterior exactly;
only an admitted proposal commits the replayed state. -/
def transactionalRepresentationSwitch {State : Type}
    (admitted : Bool)
    (oldPosterior replayedPosterior : State) : State :=
  if admitted then replayedPosterior else oldPosterior

theorem rejectedRepresentationSwitch_is_exact_fallback
    {State : Type}
    (oldPosterior replayedPosterior : State) :
    transactionalRepresentationSwitch false oldPosterior replayedPosterior =
      oldPosterior := by
  rfl

theorem admittedRepresentationSwitch_commits_replay
    {State : Type}
    (oldPosterior replayedPosterior : State) :
    transactionalRepresentationSwitch true oldPosterior replayedPosterior =
      replayedPosterior := by
  rfl

end TransactionalRepresentationSwitch

end SCOLHKG.Real
