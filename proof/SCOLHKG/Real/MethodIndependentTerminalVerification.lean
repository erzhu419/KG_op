import Mathlib
import SCOLHKG.Measure.GaussianReplicationCertificate

namespace SCOLHKG.Real

/-!
# Method-independent terminal verification

The optimizer supplies a frozen posterior shortlist.  Independent terminal
verification then treats the optimizer identity as inert metadata: the same
ordered rule, candidate-wise error spending, and budget accounting apply to
SC-OLH-KG and every comparator.
-/

structure MethodFrozenShortlist
    (Optimizer Design : Type*) where
  optimizer : Optimizer
  shortlist : FrozenSafeInteriorShortlist Design

structure MethodFrozenObjectiveShortlist
    (Optimizer Design : Type*) where
  optimizer : Optimizer
  shortlist : FrozenObjectiveChallengerShortlist Design

structure PaperGradeBudget where
  sourceCalls : ℕ
  searchCalls : ℕ
  verificationCalls : ℕ

def PaperGradeBudget.targetCalls
    (budget : PaperGradeBudget) : ℕ :=
  budget.searchCalls + budget.verificationCalls

def PaperGradeBudget.totalCalls
    (budget : PaperGradeBudget) : ℕ :=
  budget.sourceCalls + budget.targetCalls

theorem paper_grade_budget_exact_decomposition
    (budget : PaperGradeBudget) :
    budget.totalCalls =
      budget.sourceCalls + budget.searchCalls + budget.verificationCalls := by
  simp [PaperGradeBudget.totalCalls, PaperGradeBudget.targetCalls, Nat.add_assoc]

theorem powered_shortlist_total_budget_le
    (sourceCalls searchCalls : ℕ)
    (firstCertified : Bool) :
    sourceCalls
        + searchCalls
        + (if firstCertified then 80 else 80 + 96)
      ≤ sourceCalls + searchCalls + 80 + 96 := by
  have h :=
    ordered_two_policy_powered_safe_interior_budget_le
      searchCalls firstCertified
  omega

theorem objective_guard_switch_is_correct_on_upper_coverage
    {trueDifference upperBound : Real}
    (hCoverage : trueDifference <= upperBound)
    (hSwitch : upperBound < 0) :
    trueDifference < 0 := by
  exact lt_of_le_of_lt hCoverage hSwitch

end SCOLHKG.Real

namespace SCOLHKG.Measure

open MeasureTheory

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

theorem optimizer_agnostic_false_deployment_probability_le
    {Optimizer Design : Type*}
    (method : SCOLHKG.Real.MethodFrozenShortlist Optimizer Design)
    (isUnsafe : Design → Prop)
    (certifiedFirst certifiedSecond : Ω → Prop)
    (deltaFirst deltaSecond familywiseDelta : ℝ)
    [IsFiniteMeasure μ]
    (hFirst :
      μ.real (
        CandidateFalseCertificate
          (isUnsafe method.shortlist.primary) certifiedFirst)
        ≤ deltaFirst)
    (hSecond :
      μ.real (
        CandidateFalseCertificate
          (isUnsafe method.shortlist.support) certifiedSecond)
        ≤ deltaSecond)
    (hSpend : deltaFirst + deltaSecond ≤ familywiseDelta) :
    μ.real (
      FalseOrderedTwoPolicyDeployment
        (isUnsafe method.shortlist.primary)
        (isUnsafe method.shortlist.support)
        certifiedFirst certifiedSecond)
      ≤ familywiseDelta := by
  exact
    false_frozen_safe_interior_deployment_probability_le
      method.shortlist
      isUnsafe
      certifiedFirst
      certifiedSecond
      deltaFirst
      deltaSecond
      familywiseDelta
      hFirst
      hSecond
      hSpend

theorem every_finite_method_retains_familywise_control
    {Optimizer Design : Type*}
    (methods : List (SCOLHKG.Real.MethodFrozenShortlist Optimizer Design))
    (isUnsafe : Design → Prop)
    (certifiedFirst certifiedSecond :
      SCOLHKG.Real.MethodFrozenShortlist Optimizer Design → Ω → Prop)
    (deltaFirst deltaSecond familywiseDelta : ℝ)
    [IsFiniteMeasure μ]
    (hFirst :
      ∀ method ∈ methods,
        μ.real (
          CandidateFalseCertificate
            (isUnsafe method.shortlist.primary)
            (certifiedFirst method))
          ≤ deltaFirst)
    (hSecond :
      ∀ method ∈ methods,
        μ.real (
          CandidateFalseCertificate
            (isUnsafe method.shortlist.support)
            (certifiedSecond method))
          ≤ deltaSecond)
    (hSpend : deltaFirst + deltaSecond ≤ familywiseDelta) :
    ∀ method ∈ methods,
      μ.real (
        FalseOrderedTwoPolicyDeployment
          (isUnsafe method.shortlist.primary)
          (isUnsafe method.shortlist.support)
          (certifiedFirst method)
          (certifiedSecond method))
        ≤ familywiseDelta := by
  intro method hMethod
  exact
    optimizer_agnostic_false_deployment_probability_le
      method
      isUnsafe
      (certifiedFirst method)
      (certifiedSecond method)
      deltaFirst
      deltaSecond
      familywiseDelta
      (hFirst method hMethod)
      (hSecond method hMethod)
      hSpend

def FalseObjectiveSwitch
    (trueDifference : Real) (upperBound : Ω -> Real) : Set Ω :=
  {omega | upperBound omega < 0 ∧ 0 <= trueDifference}

theorem false_objective_switch_subset_upper_coverage_failure
    (trueDifference : Real) (upperBound : Ω -> Real) :
    FalseObjectiveSwitch trueDifference upperBound
      ⊆ {omega | upperBound omega < trueDifference} := by
  intro omega hFalse
  exact lt_of_lt_of_le hFalse.1 hFalse.2

theorem false_objective_switch_probability_le
    (trueDifference : Real) (upperBound : Ω -> Real) (delta : Real)
    [IsFiniteMeasure μ]
    (hCoverageFailure :
      μ.real {omega | upperBound omega < trueDifference} <= delta) :
    μ.real (FalseObjectiveSwitch trueDifference upperBound) <= delta := by
  exact (measureReal_mono
    (false_objective_switch_subset_upper_coverage_failure
      trueDifference upperBound)).trans hCoverageFailure

theorem optimizer_agnostic_three_policy_false_deployment_probability_le
    {Optimizer Design : Type*}
    (method :
      SCOLHKG.Real.MethodFrozenObjectiveShortlist Optimizer Design)
    (isUnsafe : Design -> Prop)
    (certifiedFirst certifiedSecond certifiedThird : Ω -> Prop)
    (deltaFirst deltaSecond deltaThird familywiseDelta : Real)
    [IsFiniteMeasure μ]
    (hFirst :
      μ.real (
        CandidateFalseCertificate
          (isUnsafe method.shortlist.challenger) certifiedFirst)
        <= deltaFirst)
    (hSecond :
      μ.real (
        CandidateFalseCertificate
          (isUnsafe method.shortlist.primary) certifiedSecond)
        <= deltaSecond)
    (hThird :
      μ.real (
        CandidateFalseCertificate
          (isUnsafe method.shortlist.support) certifiedThird)
        <= deltaThird)
    (hSpend :
      deltaFirst + deltaSecond + deltaThird <= familywiseDelta) :
    μ.real (
      FalseOrderedThreePolicyDeployment
        (isUnsafe method.shortlist.challenger)
        (isUnsafe method.shortlist.primary)
        (isUnsafe method.shortlist.support)
        certifiedFirst certifiedSecond certifiedThird)
      <= familywiseDelta := by
  exact false_frozen_objective_challenger_deployment_probability_le
    method.shortlist isUnsafe
    certifiedFirst certifiedSecond certifiedThird
    deltaFirst deltaSecond deltaThird familywiseDelta
    hFirst hSecond hThird hSpend

theorem optimizer_agnostic_three_policy_and_objective_guard_failure_le
    {Optimizer Design : Type*}
    (method :
      SCOLHKG.Real.MethodFrozenObjectiveShortlist Optimizer Design)
    (isUnsafe : Design -> Prop)
    (certifiedFirst certifiedSecond certifiedThird : Ω -> Prop)
    (trueObjectiveDifference : Real)
    (objectiveDifferenceUpper : Ω -> Real)
    (deltaFirst deltaSecond deltaThird familywiseDelta objectiveDelta : Real)
    [IsFiniteMeasure μ]
    (hFirst :
      μ.real (
        CandidateFalseCertificate
          (isUnsafe method.shortlist.challenger) certifiedFirst)
        <= deltaFirst)
    (hSecond :
      μ.real (
        CandidateFalseCertificate
          (isUnsafe method.shortlist.primary) certifiedSecond)
        <= deltaSecond)
    (hThird :
      μ.real (
        CandidateFalseCertificate
          (isUnsafe method.shortlist.support) certifiedThird)
        <= deltaThird)
    (hSpend :
      deltaFirst + deltaSecond + deltaThird <= familywiseDelta)
    (hObjectiveCoverageFailure :
      μ.real {omega |
        objectiveDifferenceUpper omega < trueObjectiveDifference}
        <= objectiveDelta) :
    μ.real (
      FalseOrderedThreePolicyDeployment
          (isUnsafe method.shortlist.challenger)
          (isUnsafe method.shortlist.primary)
          (isUnsafe method.shortlist.support)
          certifiedFirst certifiedSecond certifiedThird
        ∪
      FalseObjectiveSwitch
        trueObjectiveDifference objectiveDifferenceUpper)
      <= familywiseDelta + objectiveDelta := by
  have hSafety :=
    optimizer_agnostic_three_policy_false_deployment_probability_le
      method isUnsafe certifiedFirst certifiedSecond certifiedThird
      deltaFirst deltaSecond deltaThird familywiseDelta
      hFirst hSecond hThird hSpend
  have hObjective := false_objective_switch_probability_le
    trueObjectiveDifference objectiveDifferenceUpper objectiveDelta
    hObjectiveCoverageFailure
  exact (measureReal_union_le _ _).trans (add_le_add hSafety hObjective)

end SCOLHKG.Measure
