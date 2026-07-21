import Mathlib
import SCOLHKG.Measure.PosteriorUpdateKG
import SCOLHKG.Real.CertificationImplementation
import SCOLHKG.Real.ExactKGImplementation

namespace SCOLHKG.Real

open scoped BigOperators

/-!
End-to-end finite-model contract for the promoted V51 policy.

The policy has one posterior state and one terminal Bayes loss.  A charged
action either evaluates a new policy or replicates an observed policy.  Both
actions use the same posterior update and exact expected reduction in the same
terminal loss.  The implementation maximizes a Monte Carlo estimate on a
posterior-only finite shortlist; the main bound below separates shortlist
coverage error from Monte Carlo error.  Certification remains a distinct,
conservative terminal statement.
-/

inductive EvaluateOrReplicateAction (Design : Type*) where
  | evaluate : Design → EvaluateOrReplicateAction Design
  | replicate : Design → EvaluateOrReplicateAction Design
deriving DecidableEq

def evaluateOrReplicateDesign {Design : Type*} :
    EvaluateOrReplicateAction Design → Design
  | .evaluate x => x
  | .replicate x => x

def evaluateOrReplicateCost {Design : Type*}
    (_action : EvaluateOrReplicateAction Design) : ℕ :=
  1

theorem evaluate_or_replicate_has_unit_target_cost
    {Design : Type*} (action : EvaluateOrReplicateAction Design) :
    evaluateOrReplicateCost action = 1 := by
  rfl

def admissibleEvaluateOrReplicateAction {Design : Type*}
    [DecidableEq Design]
    (observed : Finset Design) : EvaluateOrReplicateAction Design → Prop
  | .evaluate x => x ∉ observed
  | .replicate x => x ∈ observed

def observedAfterAction {Design : Type*} [DecidableEq Design]
    (observed : Finset Design)
    (action : EvaluateOrReplicateAction Design) : Finset Design :=
  insert (evaluateOrReplicateDesign action) observed

theorem action_design_is_terminally_eligible_after_update
    {Design : Type*} [DecidableEq Design]
    (observed : Finset Design)
    (action : EvaluateOrReplicateAction Design) :
    evaluateOrReplicateDesign action ∈ observedAfterAction observed action := by
  simp [observedAfterAction]

theorem admissible_replication_preserves_terminal_universe
    {Design : Type*} [DecidableEq Design]
    (observed : Finset Design) (x : Design)
    (hAdmissible :
      admissibleEvaluateOrReplicateAction observed (.replicate x)) :
    observedAfterAction observed (.replicate x) = observed := by
  exact Finset.insert_eq_of_mem hAdmissible

theorem admissible_evaluation_expands_terminal_universe
    {Design : Type*} [DecidableEq Design]
    (observed : Finset Design) (x : Design)
    (hAdmissible :
      admissibleEvaluateOrReplicateAction observed (.evaluate x)) :
    observed ⊂ observedAfterAction observed (.evaluate x) := by
  refine Finset.ssubset_iff_subset_ne.mpr ?_
  constructor
  · exact Finset.subset_insert x observed
  · intro hEq
    have hx : x ∈ observedAfterAction observed (.evaluate x) := by
      simp [observedAfterAction, evaluateOrReplicateDesign]
    have hxObserved : x ∈ observed := by
      rw [hEq]
      exact hx
    exact hAdmissible hxObserved

def ObservedTerminalBayesAction {Design : Type*} [DecidableEq Design]
    (observed : Finset Design) (risk : Design → ℝ) (chosen : Design) : Prop :=
  chosen ∈ observed ∧ ∀ x ∈ observed, risk chosen ≤ risk x

theorem observed_terminal_action_cannot_return_unobserved
    {Design : Type*} [DecidableEq Design]
    {observed : Finset Design} {risk : Design → ℝ} {chosen : Design}
    (hAction : ObservedTerminalBayesAction observed risk chosen) :
    chosen ∈ observed := by
  exact hAction.1

def promotedPosteriorBayesRisk
    (objective penalty expectedPositiveMargin : ℝ) : ℝ :=
  objective + penalty * expectedPositiveMargin

theorem promoted_posterior_bayes_risk_has_nonnegative_safety_excess
    (objective penalty expectedPositiveMargin : ℝ)
    (hPenalty : 0 ≤ penalty) (hMargin : 0 ≤ expectedPositiveMargin) :
    objective ≤ promotedPosteriorBayesRisk
      objective penalty expectedPositiveMargin := by
  unfold promotedPosteriorBayesRisk
  exact le_add_of_nonneg_right (mul_nonneg hPenalty hMargin)

def ShortlistCoversVOI {Action : Type*} [DecidableEq Action]
    (full shortlist : Finset Action) (voi : Action → ℝ) (epsilon : ℝ) : Prop :=
  shortlist ⊆ full ∧
    ∀ action ∈ full,
      ∃ proxy ∈ shortlist, voi action ≤ voi proxy + epsilon

def UniformVOIApproximationOn {Action : Type*}
    (shortlist : Finset Action)
    (exact estimate : Action → ℝ) (eta : ℝ) : Prop :=
  ∀ action ∈ shortlist, |exact action - estimate action| ≤ eta

def MaximizesVOIOn {Action : Type*}
    (shortlist : Finset Action) (estimate : Action → ℝ)
    (selected : Action) : Prop :=
  selected ∈ shortlist ∧
    ∀ action ∈ shortlist, estimate action ≤ estimate selected

theorem shortlist_mc_maximizer_full_action_gap
    {Action : Type*} [DecidableEq Action]
    {full shortlist : Finset Action}
    {exact estimate : Action → ℝ}
    {epsilon eta : ℝ} {selected : Action}
    (hCover : ShortlistCoversVOI full shortlist exact epsilon)
    (hUniform : UniformVOIApproximationOn shortlist exact estimate eta)
    (hMax : MaximizesVOIOn shortlist estimate selected) :
    ∀ action ∈ full,
      exact action ≤ exact selected + epsilon + 2 * eta := by
  intro action hAction
  obtain ⟨proxy, hProxy, hCovered⟩ := hCover.2 action hAction
  have hProxyError := hUniform proxy hProxy
  have hSelectedError := hUniform selected hMax.1
  have hProxyUpper : exact proxy ≤ estimate proxy + eta := by
    have := (abs_le.mp hProxyError).2
    linarith
  have hSelectedLower : estimate selected ≤ exact selected + eta := by
    have := (abs_le.mp hSelectedError).1
    linarith
  have hEstimatedOrder := hMax.2 proxy hProxy
  linarith

theorem promoted_evaluate_or_replicate_one_step_gap
    {State Design Ω : Type*}
    [MeasurableSpace Ω]
    [DecidableEq Design]
    (kg : SCOLHKG.Measure.PosteriorUpdateKG
      State (EvaluateOrReplicateAction Design) Ω)
    (μ : MeasureTheory.Measure Ω) (state : State)
    (full shortlist : Finset (EvaluateOrReplicateAction Design))
    (estimate : EvaluateOrReplicateAction Design → ℝ)
    (epsilon eta : ℝ)
    (selected : EvaluateOrReplicateAction Design)
    (hCover : ShortlistCoversVOI full shortlist
      (SCOLHKG.Measure.posteriorUpdateExpectedGain kg μ state) epsilon)
    (hUniform : UniformVOIApproximationOn shortlist
      (SCOLHKG.Measure.posteriorUpdateExpectedGain kg μ state)
      estimate eta)
    (hMax : MaximizesVOIOn shortlist estimate selected) :
    ∀ action ∈ full,
      SCOLHKG.Measure.posteriorUpdateExpectedGain kg μ state action
        ≤ SCOLHKG.Measure.posteriorUpdateExpectedGain kg μ state selected
          + epsilon + 2 * eta := by
  exact shortlist_mc_maximizer_full_action_gap hCover hUniform hMax

theorem posterior_value_reductions_telescope
    (risk : ℕ → ℝ) (budget : ℕ) :
    (∑ t ∈ Finset.range budget, (risk t - risk (t + 1)))
      = risk 0 - risk budget := by
  induction budget with
  | zero => simp
  | succ budget ih =>
      rw [Finset.sum_range_succ, ih]
      ring

theorem finite_budget_approximate_voi_accounting
    (risk bestVOI selectedVOI error : ℕ → ℝ) (budget : ℕ)
    (hBellman : ∀ t < budget,
      selectedVOI t = risk t - risk (t + 1))
    (hApprox : ∀ t < budget,
      bestVOI t ≤ selectedVOI t + error t) :
    (∑ t ∈ Finset.range budget, bestVOI t)
      ≤ risk 0 - risk budget + ∑ t ∈ Finset.range budget, error t := by
  calc
    (∑ t ∈ Finset.range budget, bestVOI t)
        ≤ ∑ t ∈ Finset.range budget, (selectedVOI t + error t) := by
          apply Finset.sum_le_sum
          intro t ht
          exact hApprox t (Finset.mem_range.mp ht)
    _ = (∑ t ∈ Finset.range budget, selectedVOI t)
          + ∑ t ∈ Finset.range budget, error t := by
          exact Finset.sum_add_distrib
    _ = (∑ t ∈ Finset.range budget, (risk t - risk (t + 1)))
          + ∑ t ∈ Finset.range budget, error t := by
          congr 1
          apply Finset.sum_congr rfl
          intro t ht
          exact hBellman t (Finset.mem_range.mp ht)
    _ = risk 0 - risk budget
          + ∑ t ∈ Finset.range budget, error t := by
          rw [posterior_value_reductions_telescope]

theorem promoted_v51_one_step_and_terminal_certificate
    {State Design Ω : Type*}
    [MeasurableSpace Ω]
    [DecidableEq Design]
    (kg : SCOLHKG.Measure.PosteriorUpdateKG
      State (EvaluateOrReplicateAction Design) Ω)
    (μ : MeasureTheory.Measure Ω) (state : State)
    (full shortlist : Finset (EvaluateOrReplicateAction Design))
    (estimate : EvaluateOrReplicateAction Design → ℝ)
    (epsilon eta : ℝ)
    (selected : EvaluateOrReplicateAction Design)
    (hCover : ShortlistCoversVOI full shortlist
      (SCOLHKG.Measure.posteriorUpdateExpectedGain kg μ state) epsilon)
    (hUniform : UniformVOIApproximationOn shortlist
      (SCOLHKG.Measure.posteriorUpdateExpectedGain kg μ state)
      estimate eta)
    (hMax : MaximizesVOIOn shortlist estimate selected)
    {trueMean postMean beta epistemicVar z trueSigma vC tau : ℝ}
    (hz : 0 ≤ z)
    (hMean : trueMean ≤ postMean
      + implementationEpistemicSlack beta epistemicVar)
    (hSigma : trueSigma ≤ implementationCertSigma vC)
    (hMargin : theoryCertificationMargin
      postMean beta epistemicVar z vC tau ≤ 0) :
    (∀ action ∈ full,
      SCOLHKG.Measure.posteriorUpdateExpectedGain kg μ state action
        ≤ SCOLHKG.Measure.posteriorUpdateExpectedGain kg μ state selected
          + epsilon + 2 * eta)
      ∧ trueMean + z * trueSigma ≤ tau := by
  constructor
  · exact promoted_evaluate_or_replicate_one_step_gap
      kg μ state full shortlist estimate epsilon eta selected
      hCover hUniform hMax
  · exact implementation_certifies_true_quantile
      hz hMean hSigma hMargin

end SCOLHKG.Real
