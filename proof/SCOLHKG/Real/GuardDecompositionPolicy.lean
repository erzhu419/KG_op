import Mathlib
import SCOLHKG.Measure.IndependentConfirmation
import SCOLHKG.Real.PromotedV51Closure

namespace SCOLHKG.Real

/-!
# Guard-decomposed V58 policy improvement

The conservative chance margin is separated into a constraint-mean term,
ordinary epistemic uncertainty, source/task epistemic ambiguity, cumulative
aleatoric uncertainty, and a signed favorable coupling correction. The largest
nonnegative obstruction selects an oracle-free action-support family. This
support always contains the promoted V51 shortlist. A challenger can replace
the V51 action only after the independent two-head confirmation controls both
posterior Bayes-risk reduction and certificate-deficit reduction.

These statements are conditional on the posterior model and confirmation
event. They do not assert empirical calibration or certificate nonvacuity.
-/

inductive ChanceGuardMode where
  | epistemic
  | aleatoric
  | interior
deriving DecidableEq, Repr

structure ChanceGuardComponents where
  meanExcess : ℝ
  epistemic : ℝ
  jointEpistemic : ℝ
  aleatoric : ℝ
  favorableCoupling : ℝ

def reconstructedChanceMargin (guard : ChanceGuardComponents) : ℝ :=
  guard.meanExcess
    + guard.epistemic
    + guard.jointEpistemic
    + guard.aleatoric
    + guard.favorableCoupling

noncomputable def dominantChanceGuard
    (guard : ChanceGuardComponents) : ChanceGuardMode :=
  if reconstructedChanceMargin guard ≤ 0 then
    .interior
  else if
      guard.epistemic + guard.jointEpistemic ≥ guard.aleatoric ∧
      guard.epistemic + guard.jointEpistemic ≥ max guard.meanExcess 0 then
    .epistemic
  else if guard.aleatoric ≥ max guard.meanExcess 0 then
    .aleatoric
  else
    .interior

theorem reconstructed_chance_margin_exact
    (guard : ChanceGuardComponents) :
    reconstructedChanceMargin guard =
      guard.meanExcess
        + guard.epistemic
        + guard.jointEpistemic
        + guard.aleatoric
        + guard.favorableCoupling := by
  rfl

theorem nonpositive_margin_selects_interior
    (guard : ChanceGuardComponents)
    (hMargin : reconstructedChanceMargin guard ≤ 0) :
    dominantChanceGuard guard = .interior := by
  simp [dominantChanceGuard, hMargin]

theorem positive_epistemic_dominance_selects_epistemic
    (guard : ChanceGuardComponents)
    (hMargin : 0 < reconstructedChanceMargin guard)
    (hAleatoric :
      guard.aleatoric ≤ guard.epistemic + guard.jointEpistemic)
    (hMean :
      max guard.meanExcess 0 ≤
        guard.epistemic + guard.jointEpistemic) :
    dominantChanceGuard guard = .epistemic := by
  simp [
    dominantChanceGuard,
    not_le.mpr hMargin,
    hAleatoric,
    hMean,
  ]

def RetainsBaselineActions
    {Action : Type*} [DecidableEq Action]
    (baseline support : Finset Action) : Prop :=
  baseline ⊆ support

def MaximizesPosteriorValueOn
    {Action : Type*}
    (support : Finset Action)
    (value : Action → ℝ)
    (selected : Action) : Prop :=
  selected ∈ support ∧
    ∀ action ∈ support, value action ≤ value selected

theorem retained_baseline_action_remains_available
    {Action : Type*} [DecidableEq Action]
    {baseline support : Finset Action}
    {fallback : Action}
    (hRetains : RetainsBaselineActions baseline support)
    (hFallback : fallback ∈ baseline) :
    fallback ∈ support := by
  exact hRetains hFallback

theorem action_superset_cannot_reduce_fallback_value
    {Action : Type*} [DecidableEq Action]
    {baseline support : Finset Action}
    {value : Action → ℝ}
    {fallback selected : Action}
    (hRetains : RetainsBaselineActions baseline support)
    (hFallback : fallback ∈ baseline)
    (hSelected : MaximizesPosteriorValueOn support value selected) :
    value fallback ≤ value selected := by
  exact hSelected.2 fallback
    (retained_baseline_action_remains_available hRetains hFallback)

def PosteriorReduction
    {Action : Type*}
    (terminalCost : Action → ℝ)
    (fallback action : Action) : ℝ :=
  terminalCost fallback - terminalCost action

theorem nonnegative_posterior_reduction_iff_noninferior
    {Action : Type*}
    (terminalCost : Action → ℝ)
    (fallback action : Action) :
    0 ≤ PosteriorReduction terminalCost fallback action ↔
      terminalCost action ≤ terminalCost fallback := by
  simp [PosteriorReduction]

def GuardConfirmedSelection
    {Action : Type*}
    (riskCost certificateDeficit : Action → ℝ)
    (fallback selected : Action) : Prop :=
  selected = fallback ∨
    (0 ≤ PosteriorReduction riskCost fallback selected ∧
      0 ≤ PosteriorReduction certificateDeficit fallback selected)

theorem guard_confirmed_selection_joint_noninferiority
    {Action : Type*}
    {riskCost certificateDeficit : Action → ℝ}
    {fallback selected : Action}
    (hConfirmed :
      GuardConfirmedSelection
        riskCost certificateDeficit fallback selected) :
    riskCost selected ≤ riskCost fallback ∧
      certificateDeficit selected ≤ certificateDeficit fallback := by
  rcases hConfirmed with rfl | ⟨hRisk, hCertificate⟩
  · exact ⟨le_rfl, le_rfl⟩
  · exact ⟨
      (nonnegative_posterior_reduction_iff_noninferior
        riskCost fallback selected).mp hRisk,
      (nonnegative_posterior_reduction_iff_noninferior
        certificateDeficit fallback selected).mp hCertificate,
    ⟩

theorem v58_guard_policy_closure
    {Action : Type*} [DecidableEq Action]
    {baseline support : Finset Action}
    {riskCost certificateDeficit : Action → ℝ}
    {fallback selected : Action}
    (hRetains : RetainsBaselineActions baseline support)
    (hFallback : fallback ∈ baseline)
    (hSelected : selected ∈ support)
    (hConfirmed :
      GuardConfirmedSelection
        riskCost certificateDeficit fallback selected) :
    fallback ∈ support ∧
      selected ∈ support ∧
      riskCost selected ≤ riskCost fallback ∧
      certificateDeficit selected ≤ certificateDeficit fallback := by
  have hJoint :=
    guard_confirmed_selection_joint_noninferiority hConfirmed
  exact ⟨
    retained_baseline_action_remains_available hRetains hFallback,
    hSelected,
    hJoint.1,
    hJoint.2,
  ⟩

end SCOLHKG.Real
