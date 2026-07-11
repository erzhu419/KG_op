import Mathlib
import SCOLHKG.Real.TaskPosterior

namespace SCOLHKG.Real

open scoped BigOperators

/-!
Implementation-facing theory for the two-posterior V28 task layer.

`Q_pred` and `Q_safe` are normalized from the same frozen source prior but
different strictly prequential scores. Objective aggregation may retain
`Q_pred`; candidate allocation, certification, and exact KG use `Q_safe`.
The clipped Bernoulli and pairwise probability scores have a finite loss
radius. The Gaussian constraint-score contribution is covered separately by
the exponential-moment hypotheses in `TaskPosterior` and `TaskPACBayes`.
-/

def clippedUnitProbability (epsilon probability : ℝ) : ℝ :=
  max epsilon (min (1 - epsilon) probability)

theorem epsilon_le_clippedUnitProbability
    (epsilon probability : ℝ) :
    epsilon ≤ clippedUnitProbability epsilon probability := by
  exact le_max_left _ _

theorem clippedUnitProbability_le_one_sub_epsilon
    {epsilon probability : ℝ}
    (hClip : epsilon ≤ 1 - epsilon) :
    clippedUnitProbability epsilon probability ≤ 1 - epsilon := by
  exact max_le hClip (min_le_left _ _)

theorem clippedUnitProbability_pos
    {epsilon probability : ℝ}
    (hEpsilon : 0 < epsilon) :
    0 < clippedUnitProbability epsilon probability := by
  exact lt_of_lt_of_le hEpsilon
    (epsilon_le_clippedUnitProbability epsilon probability)

theorem clippedUnitProbability_le_one
    {epsilon probability : ℝ}
    (hEpsilon : 0 < epsilon)
    (hClip : epsilon ≤ 1 - epsilon) :
    clippedUnitProbability epsilon probability ≤ 1 := by
  exact le_trans
    (clippedUnitProbability_le_one_sub_epsilon hClip)
    (sub_le_self 1 (le_of_lt hEpsilon))

noncomputable def clippedLogLoss
    (epsilon probability : ℝ) : ℝ :=
  -Real.log (clippedUnitProbability epsilon probability)

theorem clipped_probability_log_score_bounded
    {epsilon probability : ℝ}
    (hEpsilon : 0 < epsilon)
    (hClip : epsilon ≤ 1 - epsilon) :
    Real.log epsilon ≤
        Real.log (clippedUnitProbability epsilon probability) ∧
      Real.log (clippedUnitProbability epsilon probability) ≤ 0 := by
  constructor
  · exact Real.log_le_log hEpsilon
      (epsilon_le_clippedUnitProbability epsilon probability)
  · exact Real.log_nonpos
      (le_of_lt (clippedUnitProbability_pos hEpsilon))
      (clippedUnitProbability_le_one hEpsilon hClip)

theorem clipped_probability_log_loss_bounded
    {epsilon probability : ℝ}
    (hEpsilon : 0 < epsilon)
    (hClip : epsilon ≤ 1 - epsilon) :
    0 ≤ clippedLogLoss epsilon probability ∧
      clippedLogLoss epsilon probability ≤ -Real.log epsilon := by
  obtain ⟨hLowerScore, hUpperScore⟩ :=
    clipped_probability_log_score_bounded hEpsilon hClip
  unfold clippedLogLoss
  exact ⟨neg_nonneg.mpr hUpperScore, neg_le_neg hLowerScore⟩

def safeCompositeLoss
    (constraintLoss boundaryLoss pairwiseLoss : ℝ)
    (boundaryWeight pairwiseWeight : ℝ) : ℝ :=
  constraintLoss
    + boundaryWeight * boundaryLoss
    + pairwiseWeight * pairwiseLoss

theorem safe_composite_loss_upper_bound
    {constraintLoss boundaryLoss pairwiseLoss : ℝ}
    {constraintUpper boundaryUpper pairwiseUpper : ℝ}
    {boundaryWeight pairwiseWeight : ℝ}
    (hBoundaryWeight : 0 ≤ boundaryWeight)
    (hPairwiseWeight : 0 ≤ pairwiseWeight)
    (hConstraint : constraintLoss ≤ constraintUpper)
    (hBoundary : boundaryLoss ≤ boundaryUpper)
    (hPairwise : pairwiseLoss ≤ pairwiseUpper) :
    safeCompositeLoss constraintLoss boundaryLoss pairwiseLoss
        boundaryWeight pairwiseWeight ≤
      safeCompositeLoss constraintUpper boundaryUpper pairwiseUpper
        boundaryWeight pairwiseWeight := by
  unfold safeCompositeLoss
  gcongr

noncomputable def predictiveTaskMass
    {ι : Type*} [Fintype ι]
    (prior predictiveScore : ι → ℝ)
    (eta : ℝ) (i : ι) : ℝ :=
  normalizeFiniteWeights
    (generalizedBayesMass prior predictiveScore eta) i

noncomputable def safeDecisionTaskMass
    {ι : Type*} [Fintype ι]
    (prior safeScore : ι → ℝ)
    (eta : ℝ) (i : ι) : ℝ :=
  normalizeFiniteWeights
    (generalizedBayesMass prior safeScore eta) i

structure DualTaskMass (ι : Type*) where
  predictiveMass : ι → ℝ
  safeDecisionMass : ι → ℝ

noncomputable def dualGeneralizedBayesTaskMass
    {ι : Type*} [Fintype ι]
    (prior predictiveScore safeScore : ι → ℝ)
    (eta : ℝ) : DualTaskMass ι where
  predictiveMass := predictiveTaskMass prior predictiveScore eta
  safeDecisionMass := safeDecisionTaskMass prior safeScore eta

theorem predictiveTaskMass_sum_eq_one
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {prior predictiveScore : ι → ℝ}
    {eta : ℝ}
    (hPrior : ∀ i, 0 < prior i) :
    ∑ i, predictiveTaskMass prior predictiveScore eta i = 1 := by
  have hEach : ∀ i, 0 <
      generalizedBayesMass prior predictiveScore eta i :=
    generalizedBayesMass_pos hPrior
  have hSum : 0 < ∑ i,
      generalizedBayesMass prior predictiveScore eta i := by
    exact Finset.sum_pos (fun i _hi => hEach i) Finset.univ_nonempty
  exact normalizeFiniteWeights_sum_eq_one (ne_of_gt hSum)

theorem safeDecisionTaskMass_sum_eq_one
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {prior safeScore : ι → ℝ}
    {eta : ℝ}
    (hPrior : ∀ i, 0 < prior i) :
    ∑ i, safeDecisionTaskMass prior safeScore eta i = 1 := by
  have hEach : ∀ i, 0 <
      generalizedBayesMass prior safeScore eta i :=
    generalizedBayesMass_pos hPrior
  have hSum : 0 < ∑ i,
      generalizedBayesMass prior safeScore eta i := by
    exact Finset.sum_pos (fun i _hi => hEach i) Finset.univ_nonempty
  exact normalizeFiniteWeights_sum_eq_one (ne_of_gt hSum)

theorem predictiveTaskMass_pos
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {prior predictiveScore : ι → ℝ}
    {eta : ℝ}
    (hPrior : ∀ i, 0 < prior i)
    (i : ι) :
    0 < predictiveTaskMass prior predictiveScore eta i := by
  exact generalizedBayes_normalized_support hPrior i

theorem safeDecisionTaskMass_pos
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {prior safeScore : ι → ℝ}
    {eta : ℝ}
    (hPrior : ∀ i, 0 < prior i)
    (i : ι) :
    0 < safeDecisionTaskMass prior safeScore eta i := by
  exact generalizedBayes_normalized_support hPrior i

theorem safeDecisionMass_independent_of_predictive_score
    {ι : Type*} [Fintype ι]
    (prior safeScore predictiveScore₁ predictiveScore₂ : ι → ℝ)
    (eta : ℝ) :
    (dualGeneralizedBayesTaskMass
      prior predictiveScore₁ safeScore eta).safeDecisionMass =
    (dualGeneralizedBayesTaskMass
      prior predictiveScore₂ safeScore eta).safeDecisionMass := by
  rfl

theorem safe_generalized_pac_bayes_bound_on_moment_event
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {qSafe prior gap : ι → ℝ}
    {rho delta n : ℝ}
    (hqSafe : ∀ i, 0 < qSafe i)
    (hPrior : ∀ i, 0 < prior i)
    (hqSafeNorm : ∑ i, qSafe i = 1)
    (hPriorNorm : ∑ i, prior i = 1)
    (hKL : finiteTaskKL qSafe prior ≤ rho)
    (hDelta : 0 < delta)
    (hN : 0 < n)
    (hMoment :
      (∑ i, prior i * Real.exp (n * gap i)) ≤ 1 / delta) :
    (∑ i, qSafe i * gap i) ≤
      (rho + Real.log (1 / delta)) / n := by
  exact finite_pac_bayes_bound_on_moment_event
    hqSafe hPrior hqSafeNorm hPriorNorm hKL hDelta hN hMoment

structure DualTaskBelief (Expert ModelState : Type*) where
  predictiveWeight : Expert → ℝ
  safeDecisionWeight : Expert → ℝ
  objectiveState : Expert → ModelState
  constraintState : Expert → ModelState
  varianceState : Expert → ModelState

def safeDecisionJointBelief
    {Expert ModelState : Type*}
    (belief : DualTaskBelief Expert ModelState) :
    JointTaskBelief Expert ModelState where
  taskWeight := belief.safeDecisionWeight
  objectiveState := belief.objectiveState
  constraintState := belief.constraintState
  varianceState := belief.varianceState

theorem safeDecisionJointBelief_uses_safe_weight
    {Expert ModelState : Type*}
    (belief : DualTaskBelief Expert ModelState)
    (expert : Expert) :
    (safeDecisionJointBelief belief).taskWeight expert =
      belief.safeDecisionWeight expert := by
  rfl

def dualTaskExactGain
    {Expert ModelState Design Observation : Type*}
    (terminal : DualTaskBelief Expert ModelState → ℝ)
    (update : DualTaskBelief Expert ModelState →
      Design → Observation → DualTaskBelief Expert ModelState)
    (belief : DualTaskBelief Expert ModelState)
    (x : Design)
    (y : Observation) : ℝ :=
  terminal belief - terminal (update belief x y)

theorem dual_task_exact_mc_zero_error_is_one_step_optimal
    {Design : Type*}
    {exact estimate : Design → ℝ}
    {x : Design}
    (hEstimator : ExactMCEstimator exact estimate 0)
    (hMax : ∀ y, estimate y ≤ estimate x) :
    KGMaximizer { expectedTerminalGain := exact } x := by
  exact exact_mc_zero_error_recovers_exact_maximizer hEstimator hMax

end SCOLHKG.Real
