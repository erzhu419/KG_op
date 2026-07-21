import Mathlib

namespace SCOLHKG.Real

/-!
Posterior-value noninferiority for the V52 guarded policy improvement.

The implementation retains the promoted V51 action as a fallback. A
challenger from the action superset or finite-horizon rollout may replace it
only when its estimated lower-is-better terminal value improves by more than
twice the declared uniform numerical error. On that event, the exact posterior
value of the challenger cannot be worse than the exact value of the fallback.

This theorem is deliberately about the common posterior decision model. It
does not turn model misspecification or held-out coordinate sufficiency into an
unconditional simulator-performance guarantee.
-/

def UniformTerminalValueApproximation
    {Action : Type*}
    (exact estimate : Action → ℝ)
    (eta : ℝ) : Prop :=
  ∀ action, |estimate action - exact action| ≤ eta

def PassesTwoEtaImprovementGuard
    {Action : Type*}
    (estimate : Action → ℝ)
    (eta : ℝ)
    (fallback challenger : Action) : Prop :=
  estimate challenger + 2 * eta ≤ estimate fallback

theorem two_eta_guard_implies_exact_noninferiority
    {Action : Type*}
    {exact estimate : Action → ℝ}
    {eta : ℝ}
    {fallback challenger : Action}
    (hUniform : UniformTerminalValueApproximation exact estimate eta)
    (hGuard : PassesTwoEtaImprovementGuard
      estimate eta fallback challenger) :
    exact challenger ≤ exact fallback := by
  have hChallenger := abs_le.mp (hUniform challenger)
  have hFallback := abs_le.mp (hUniform fallback)
  unfold PassesTwoEtaImprovementGuard at hGuard
  linarith

theorem guarded_fallback_or_switch_noninferior
    {Action : Type*}
    {exact estimate : Action → ℝ}
    {eta : ℝ}
    {fallback challenger selected : Action}
    (hUniform : UniformTerminalValueApproximation exact estimate eta)
    (hDecision :
      selected = fallback ∨
        (selected = challenger ∧
          PassesTwoEtaImprovementGuard
            estimate eta fallback challenger)) :
    exact selected ≤ exact fallback := by
  rcases hDecision with rfl | ⟨rfl, hGuard⟩
  · exact le_rfl
  · exact two_eta_guard_implies_exact_noninferiority hUniform hGuard

theorem nested_guarded_improvements_noninferior
    {Action : Type*}
    {exactOne exactRollout estimateOne estimateRollout : Action → ℝ}
    {etaOne etaRollout : ℝ}
    {baseline oneStep rollout : Action}
    (hOneUniform : UniformTerminalValueApproximation
      exactOne estimateOne etaOne)
    (hOneGuard : PassesTwoEtaImprovementGuard
      estimateOne etaOne baseline oneStep)
    (hRolloutUniform : UniformTerminalValueApproximation
      exactRollout estimateRollout etaRollout)
    (hRolloutGuard : PassesTwoEtaImprovementGuard
      estimateRollout etaRollout oneStep rollout) :
    exactOne oneStep ≤ exactOne baseline ∧
      exactRollout rollout ≤ exactRollout oneStep := by
  exact ⟨
    two_eta_guard_implies_exact_noninferiority hOneUniform hOneGuard,
    two_eta_guard_implies_exact_noninferiority hRolloutUniform hRolloutGuard,
  ⟩

end SCOLHKG.Real
