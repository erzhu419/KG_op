import Mathlib

namespace SCOLHKG.Real

open MeasureTheory

/-!
Finite budgeted finalist replication used by V29.

The theorem is deliberately event-conditional.  Replicate data construct an
upper confidence bound for the latent mean, while the shrunk standard
deviation must upper-bound the true aleatoric standard deviation.  On their
joint event, a nonpositive replicated margin implies the original chance
constraint.  Separate probability lemmas can supply the confidence event for
the selected noise family.
-/

noncomputable def replicatedMeanUpper
    (sampleMean sigma : ℝ)
    (replicateCount : ℕ)
    (zDelta : ℝ) : ℝ :=
  sampleMean + zDelta * sigma / Real.sqrt replicateCount

noncomputable def replicatedFinalistMargin
    (sampleMean sigma : ℝ)
    (replicateCount : ℕ)
    (zAlpha zDelta tau : ℝ) : ℝ :=
  replicatedMeanUpper sampleMean sigma replicateCount zDelta
    + zAlpha * sigma - tau

theorem replicated_finalist_margin_sound_on_joint_event
    {trueMean trueSigma sampleMean sigma zAlpha zDelta tau : ℝ}
    {replicateCount : ℕ}
    (hMean :
      trueMean ≤
        replicatedMeanUpper sampleMean sigma replicateCount zDelta)
    (hSigma : trueSigma ≤ sigma)
    (hZAlpha : 0 ≤ zAlpha)
    (hMargin :
      replicatedFinalistMargin sampleMean sigma replicateCount
        zAlpha zDelta tau ≤ 0) :
    trueMean + zAlpha * trueSigma ≤ tau := by
  have hScaledSigma : zAlpha * trueSigma ≤ zAlpha * sigma :=
    mul_le_mul_of_nonneg_left hSigma hZAlpha
  unfold replicatedFinalistMargin at hMargin
  linarith

def replicatedFinalistEmpiricallyFeasible (upperMargin : ℝ) : Prop :=
  upperMargin ≤ 0

def IsFinalistSafetyFirst
    {Design : Type*}
    (finalists : Finset Design)
    (upperMargin objective : Design → ℝ)
    (chosen : Design) : Prop :=
  chosen ∈ finalists ∧
  ((∃ x ∈ finalists, replicatedFinalistEmpiricallyFeasible (upperMargin x)) →
      replicatedFinalistEmpiricallyFeasible (upperMargin chosen) ∧
      ∀ x ∈ finalists,
        replicatedFinalistEmpiricallyFeasible (upperMargin x) →
          objective chosen ≤ objective x) ∧
  ((¬∃ x ∈ finalists,
      replicatedFinalistEmpiricallyFeasible (upperMargin x)) →
      ∀ x ∈ finalists, upperMargin chosen ≤ upperMargin x)

theorem finalist_safety_first_contract
    {Design : Type*}
    {finalists : Finset Design}
    {upperMargin objective : Design → ℝ}
    {chosen : Design}
    (hChosen : chosen ∈ finalists)
    (hFeasible :
      (∃ x ∈ finalists,
        replicatedFinalistEmpiricallyFeasible (upperMargin x)) →
        replicatedFinalistEmpiricallyFeasible (upperMargin chosen) ∧
        ∀ x ∈ finalists,
          replicatedFinalistEmpiricallyFeasible (upperMargin x) →
            objective chosen ≤ objective x)
    (hInfeasible :
      (¬∃ x ∈ finalists,
        replicatedFinalistEmpiricallyFeasible (upperMargin x)) →
        ∀ x ∈ finalists, upperMargin chosen ≤ upperMargin x) :
    IsFinalistSafetyFirst finalists upperMargin objective chosen := by
  exact ⟨hChosen, hFeasible, hInfeasible⟩

structure FrozenFinalistSet (Design : Type*) where
  targets : Finset Design
  frozenStage : ℕ

def freezeFinalistSet
    {Design : Type*}
    (targets : Finset Design)
    (stage : ℕ) : FrozenFinalistSet Design where
  targets := targets
  frozenStage := stage

theorem frozen_finalists_ignore_future_labels
    {Design Observation : Type*}
    (targets : Finset Design)
    (stage : ℕ)
    (_futureLabels : Observation) :
    (freezeFinalistSet targets stage).targets = targets := by
  rfl

def expertNominationSet
    {Expert Design : Type*}
    [Fintype Expert]
    [DecidableEq Design]
    (nomination : Expert → Design) : Finset Design :=
  Finset.univ.image nomination

theorem every_finite_expert_nomination_is_supported
    {Expert Design : Type*}
    [Fintype Expert]
    [DecidableEq Design]
    (nomination : Expert → Design)
    (expert : Expert) :
    nomination expert ∈ expertNominationSet nomination := by
  exact Finset.mem_image.mpr ⟨expert, Finset.mem_univ expert, rfl⟩

theorem nomination_support_does_not_depend_on_posterior_mass
    {Expert Design : Type*}
    [Fintype Expert]
    [DecidableEq Design]
    (nomination : Expert → Design)
    (_posteriorMass : Expert → ℝ)
    (expert : Expert) :
    nomination expert ∈ expertNominationSet nomination := by
  exact every_finite_expert_nomination_is_supported nomination expert

def replicationDeficit (target count : ℕ) : ℕ :=
  target - count

theorem one_replication_strictly_reduces_positive_deficit
    {target count : ℕ}
    (hPending : count < target) :
    replicationDeficit target (count + 1) + 1 =
      replicationDeficit target count := by
  unfold replicationDeficit
  omega

theorem reserved_finalist_stage_stays_inside_total_budget
    {N R stage : ℕ}
    (hReserved : R ≤ N)
    (hInReservedSuffix : N - R ≤ stage)
    (hBeforeEnd : stage < N) :
    stage + 1 ≤ N ∧ N - stage ≤ R := by
  omega

/-!
V31 replaces the one-time frozen challenger by a bounded adaptive archive.
Every nomination is chosen from the pre-observation history, inserted into the
archive, and may later be excluded from the empirical race if it has not
received the declared minimum number of replicates.
-/

def adaptiveFinalistArchive
    {Design : Type*}
    [DecidableEq Design]
    (initial : Finset Design) : List Design → Finset Design
  | [] => initial
  | target :: rest =>
      adaptiveFinalistArchive (insert target initial) rest

theorem adaptive_archive_contains_initial
    {Design : Type*}
    [DecidableEq Design]
    (initial : Finset Design)
    (nominations : List Design) :
    initial ⊆ adaptiveFinalistArchive initial nominations := by
  induction nominations generalizing initial with
  | nil => simp [adaptiveFinalistArchive]
  | cons target rest ih =>
      intro x hx
      apply ih (insert target initial)
      exact Finset.mem_insert_of_mem hx

theorem adaptive_archive_contains_every_nomination
    {Design : Type*}
    [DecidableEq Design]
    (initial : Finset Design)
    (nominations : List Design)
    {target : Design}
    (hTarget : target ∈ nominations) :
    target ∈ adaptiveFinalistArchive initial nominations := by
  induction nominations generalizing initial with
  | nil => simp at hTarget
  | cons head tail ih =>
      simp only [List.mem_cons] at hTarget
      simp only [adaptiveFinalistArchive]
      rcases hTarget with hHead | hTail
      · subst target
        exact adaptive_archive_contains_initial
          (insert head initial) tail (Finset.mem_insert_self head initial)
      · exact ih (insert head initial) hTail

theorem adaptive_archive_card_le_initial_add_refreshes
    {Design : Type*}
    [DecidableEq Design]
    (initial : Finset Design)
    (nominations : List Design) :
    (adaptiveFinalistArchive initial nominations).card
      ≤ initial.card + nominations.length := by
  induction nominations generalizing initial with
  | nil => simp [adaptiveFinalistArchive]
  | cons target rest ih =>
      simp only [adaptiveFinalistArchive, List.length_cons]
      calc
        (adaptiveFinalistArchive (insert target initial) rest).card
            ≤ (insert target initial).card + rest.length :=
          ih (insert target initial)
        _ ≤ (initial.card + 1) + rest.length := by
          gcongr
          exact Finset.card_insert_le target initial
        _ = initial.card + (rest.length + 1) := by omega

theorem adaptive_archive_subset_fixed_universe
    {Design : Type*}
    [DecidableEq Design]
    (fixedSet initial : Finset Design)
    (nominations : List Design)
    (hInitial : initial ⊆ fixedSet)
    (hNominations : ∀ target ∈ nominations, target ∈ fixedSet) :
    adaptiveFinalistArchive initial nominations ⊆ fixedSet := by
  induction nominations generalizing initial with
  | nil => simpa [adaptiveFinalistArchive] using hInitial
  | cons head tail ih =>
      have hHeadIn : head ∈ fixedSet :=
        hNominations head (by simp)
      simp only [adaptiveFinalistArchive]
      apply ih (insert head initial)
      · intro target hTarget
        simp only [Finset.mem_insert] at hTarget
        rcases hTarget with rfl | hTarget
        · exact hHeadIn
        · exact hInitial hTarget
      · intro target hTarget
        exact hNominations target (by simp [hTarget])

theorem adaptive_archive_card_le_fixed_universe
    {Design : Type*}
    [DecidableEq Design]
    (fixedSet initial : Finset Design)
    (nominations : List Design)
    (hInitial : initial ⊆ fixedSet)
    (hNominations : ∀ target ∈ nominations, target ∈ fixedSet) :
    (adaptiveFinalistArchive initial nominations).card ≤ fixedSet.card := by
  exact Finset.card_le_card (
    adaptive_archive_subset_fixed_universe
      fixedSet initial nominations hInitial hNominations)

def completedFinalists
    {Design : Type*}
    [DecidableEq Design]
    (archive : Finset Design)
    (replicateCount : Design → ℕ)
    (minimum : ℕ) : Finset Design :=
  archive.filter fun target => minimum ≤ replicateCount target

theorem mem_completed_finalists_iff
    {Design : Type*}
    [DecidableEq Design]
    {archive : Finset Design}
    {replicateCount : Design → ℕ}
    {minimum : ℕ}
    {target : Design} :
    target ∈ completedFinalists archive replicateCount minimum ↔
      target ∈ archive ∧ minimum ≤ replicateCount target := by
  simp [completedFinalists]

theorem incomplete_finalist_cannot_enter_completed_race
    {Design : Type*}
    [DecidableEq Design]
    {archive : Finset Design}
    {replicateCount : Design → ℕ}
    {minimum : ℕ}
    {target : Design}
    (hIncomplete : replicateCount target < minimum) :
    target ∉ completedFinalists archive replicateCount minimum := by
  simp [completedFinalists, Nat.not_le.mpr hIncomplete]

theorem completed_adaptive_finalist_sound_on_joint_event
    {Design : Type*}
    [DecidableEq Design]
    {archive : Finset Design}
    {replicateCount : Design → ℕ}
    {minimum : ℕ}
    {chosen : Design}
    {trueMean trueSigma sampleMean sigma : Design → ℝ}
    {zAlpha zDelta tau : ℝ}
    (hChosen :
      chosen ∈ completedFinalists archive replicateCount minimum)
    (hMean :
      trueMean chosen ≤ replicatedMeanUpper
        (sampleMean chosen) (sigma chosen) (replicateCount chosen) zDelta)
    (hSigma : trueSigma chosen ≤ sigma chosen)
    (hZAlpha : 0 ≤ zAlpha)
    (hMargin :
      replicatedFinalistMargin
        (sampleMean chosen) (sigma chosen) (replicateCount chosen)
        zAlpha zDelta tau ≤ 0) :
    chosen ∈ archive ∧ minimum ≤ replicateCount chosen ∧
      trueMean chosen + zAlpha * trueSigma chosen ≤ tau := by
  have hMembership := mem_completed_finalists_iff.mp hChosen
  exact ⟨hMembership.1, hMembership.2,
    replicated_finalist_margin_sound_on_joint_event
      hMean hSigma hZAlpha hMargin⟩

theorem adaptive_finalist_bad_event_le_sum
    {Ω Design : Type*}
    {mΩ : MeasurableSpace Ω}
    (μ : Measure Ω)
    (archive : Finset Design)
    (bad : Design → Set Ω) :
    μ (⋃ target ∈ archive, bad target)
      ≤ ∑ target ∈ archive, μ (bad target) := by
  exact measure_biUnion_finset_le archive bad

end SCOLHKG.Real
