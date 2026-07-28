import Mathlib
import SCOLHKG.Real.SafeRegret

namespace SCOLHKG.Real

/-!
True safe regret for the finite-budget terminal decision.

This result separates seven scientifically different errors instead of hiding
them in one optimizer tolerance:

* representation error in the observable state/risk coordinate;
* cumulative-HVD variance error;
* source-to-target transfer error;
* coverage of the true safe comparator by the observed terminal pool;
* action-shortlist error;
* uniform Monte-Carlo error;
* sequential one-step/myopia error.

The promoted implementation returns a posterior Bayes action from the charged
observed policies.  Therefore a true comparator guarantee necessarily needs an
observed-pool coverage term.  No finite-budget algorithm can remove that term
without an assumption on candidate generation or function regularity.
-/

universe u

noncomputable section

structure StatisticalClosureErrors where
  representation : ℝ
  hvd : ℝ
  transfer : ℝ
  poolCoverage : ℝ
  shortlist : ℝ
  mc : ℝ
  sequential : ℝ

def StatisticalClosureErrors.Valid (e : StatisticalClosureErrors) : Prop :=
  0 ≤ e.representation ∧
  0 ≤ e.hvd ∧
  0 ≤ e.transfer ∧
  0 ≤ e.poolCoverage ∧
  0 ≤ e.shortlist ∧
  0 ≤ e.mc ∧
  0 ≤ e.sequential

def StatisticalClosureErrors.terminalScoreRadius
    (e : StatisticalClosureErrors) : ℝ :=
  (e.representation + e.hvd + e.transfer) / 2

def StatisticalClosureErrors.decisionError
    (e : StatisticalClosureErrors) : ℝ :=
  e.shortlist + 2 * e.mc + e.sequential

def StatisticalClosureErrors.total (e : StatisticalClosureErrors) : ℝ :=
  e.representation + e.hvd + e.transfer
    + e.poolCoverage + e.shortlist + 2 * e.mc + e.sequential

def UniformTerminalScoreBridge
    {Design : Type u} [DecidableEq Design]
    (pool : Finset Design)
    (objective score : Design → ℝ)
    (radius : ℝ) : Prop :=
  ∀ x ∈ pool, |score x - objective x| ≤ radius

def ApproximateTerminalBayesAction
    {Design : Type u} [DecidableEq Design]
    (pool : Finset Design)
    (score : Design → ℝ)
    (decisionError : ℝ)
    (selected : Design) : Prop :=
  selected ∈ pool ∧
    ∀ x ∈ pool, score selected ≤ score x + decisionError

def SafeObservedComparatorCoverage
    {Design : Type u} [DecidableEq Design]
    (p : ChanceOptimization Design)
    (pool : Finset Design)
    (xStar : Design)
    (coverageError : ℝ) : Prop :=
  ∃ proxy ∈ pool,
    TrueChanceFeasible p proxy ∧
      p.objective proxy ≤ p.objective xStar + coverageError

theorem terminal_score_bridge_selected_upper
    {Design : Type u} [DecidableEq Design]
    {pool : Finset Design}
    {objective score : Design → ℝ}
    {radius : ℝ}
    {selected : Design}
    (hBridge : UniformTerminalScoreBridge pool objective score radius)
    (hSelected : selected ∈ pool) :
    objective selected ≤ score selected + radius := by
  have h := hBridge selected hSelected
  rw [abs_le] at h
  linarith

theorem terminal_score_bridge_proxy_upper
    {Design : Type u} [DecidableEq Design]
    {pool : Finset Design}
    {objective score : Design → ℝ}
    {radius : ℝ}
    {proxy : Design}
    (hBridge : UniformTerminalScoreBridge pool objective score radius)
    (hProxy : proxy ∈ pool) :
    score proxy ≤ objective proxy + radius := by
  have h := hBridge proxy hProxy
  rw [abs_le] at h
  linarith

theorem finite_pool_true_objective_regret
    {Design : Type u} [DecidableEq Design]
    (p : ChanceOptimization Design)
    (pool : Finset Design)
    (score : Design → ℝ)
    (errors : StatisticalClosureErrors)
    (selected xStar : Design)
    (hBridge : UniformTerminalScoreBridge
      pool p.objective score errors.terminalScoreRadius)
    (hSelect : ApproximateTerminalBayesAction
      pool score errors.decisionError selected)
    (hCoverage : SafeObservedComparatorCoverage
      p pool xStar errors.poolCoverage) :
    p.objective selected - p.objective xStar ≤ errors.total := by
  obtain ⟨proxy, hProxyPool, _hProxySafe, hProxyObjective⟩ := hCoverage
  have hSelectedUpper := terminal_score_bridge_selected_upper
    hBridge hSelect.1
  have hProxyScoreUpper := terminal_score_bridge_proxy_upper
    hBridge hProxyPool
  have hScoreOrder := hSelect.2 proxy hProxyPool
  unfold StatisticalClosureErrors.terminalScoreRadius at *
  unfold StatisticalClosureErrors.decisionError at *
  unfold StatisticalClosureErrors.total
  linarith

theorem end_to_end_finite_budget_safe_regret
    {Design : Type u} [DecidableEq Design]
    (p : ChanceOptimization Design)
    (pool : Finset Design)
    (score : Design → ℝ)
    (errors : StatisticalClosureErrors)
    (selected xStar : Design)
    (hz : 0 ≤ p.z)
    (hMean :
      p.trueMean selected ≤
        p.posteriorMean selected + p.epistemicSlack selected)
    (hSigma : p.trueSigma selected ≤ p.certSigma selected)
    (hCertified : CertifiedFeasible p selected)
    (hBridge : UniformTerminalScoreBridge
      pool p.objective score errors.terminalScoreRadius)
    (hSelect : ApproximateTerminalBayesAction
      pool score errors.decisionError selected)
    (hCoverage : SafeObservedComparatorCoverage
      p pool xStar errors.poolCoverage) :
    TrueChanceFeasible p selected ∧
      SafeSimpleRegretBound p selected xStar errors.total := by
  constructor
  · exact certified_feasible_sound p selected hz hMean hSigma hCertified
  · exact finite_pool_true_objective_regret
      p pool score errors selected xStar hBridge hSelect hCoverage

theorem statistical_closure_total_expands
    (errors : StatisticalClosureErrors) :
    errors.total =
      errors.representation + errors.hvd + errors.transfer
        + errors.poolCoverage + errors.shortlist
        + 2 * errors.mc + errors.sequential := by
  rfl

theorem statistical_closure_total_nonnegative
    (errors : StatisticalClosureErrors)
    (hValid : errors.Valid) :
    0 ≤ errors.total := by
  unfold StatisticalClosureErrors.Valid at hValid
  unfold StatisticalClosureErrors.total
  rcases hValid with
    ⟨hRepresentation, hHVD, hTransfer, hPool, hShortlist, hMC, hSequential⟩
  positivity

end

end SCOLHKG.Real
