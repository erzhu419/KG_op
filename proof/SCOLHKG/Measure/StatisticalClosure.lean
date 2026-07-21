import Mathlib
import SCOLHKG.Measure.ProbabilityEvents
import SCOLHKG.Real.EndToEndSafeRegret

namespace SCOLHKG.Measure

open MeasureTheory
open scoped ENNReal BigOperators

/-!
Joint high-probability wrapper for the statistical paper theorem.

Each component event is discharged by its own distributional result: GP
confidence, residual-square/HVD concentration, source-task PAC-Bayes transfer,
finite-pool coverage, and exact-MC concentration.  A finite union bound then
turns the deterministic end-to-end theorem into a frequentist statement.
-/

inductive StatisticalFailureComponent where
  | meanConfidence
  | hvdEstimation
  | transfer
  | poolCoverage
  | shortlist
  | monteCarlo
  | sequential
deriving DecidableEq, Fintype

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

def JointStatisticalBadEvent
    (bad : StatisticalFailureComponent → Set Ω) : Set Ω :=
  ⋃ component ∈ (Finset.univ : Finset StatisticalFailureComponent),
    bad component

theorem joint_statistical_bad_event_le_sum
    (bad : StatisticalFailureComponent → Set Ω) :
    μ (JointStatisticalBadEvent bad) ≤
      ∑ component, μ (bad component) := by
  unfold JointStatisticalBadEvent
  exact finite_union_bad_event_le_sum
    (μ := μ) Finset.univ bad

theorem joint_statistical_bad_event_le_delta_sum
    (bad : StatisticalFailureComponent → Set Ω)
    (delta : StatisticalFailureComponent → ℝ)
    (hEach : ∀ component,
      μ (bad component) ≤ ENNReal.ofReal (delta component)) :
    μ (JointStatisticalBadEvent bad) ≤
      ∑ component, ENNReal.ofReal (delta component) := by
  calc
    μ (JointStatisticalBadEvent bad)
      ≤ ∑ component, μ (bad component) :=
        joint_statistical_bad_event_le_sum bad
    _ ≤ ∑ component, ENNReal.ofReal (delta component) := by
      exact Finset.sum_le_sum fun component _hcomponent => hEach component

def EndToEndStatisticalFailure
    {Design : Type*}
    (p : SCOLHKG.Real.ChanceOptimization Design)
    (recommendation : Ω → Design)
    (xStar : Design)
    (error : ℝ) : Set Ω :=
  {ω |
    ¬ (SCOLHKG.Real.TrueChanceFeasible p (recommendation ω) ∧
      SCOLHKG.Real.SafeSimpleRegretBound
        p (recommendation ω) xStar error)}

theorem end_to_end_statistical_failure_le
    {Design : Type*}
    {p : SCOLHKG.Real.ChanceOptimization Design}
    {recommendation : Ω → Design}
    {xStar : Design}
    {error : ℝ}
    (bad : StatisticalFailureComponent → Set Ω)
    (delta : StatisticalFailureComponent → ℝ)
    (hEach : ∀ component,
      μ (bad component) ≤ ENNReal.ofReal (delta component))
    (hOutside : ∀ ω,
      ω ∉ JointStatisticalBadEvent bad →
        SCOLHKG.Real.TrueChanceFeasible p (recommendation ω) ∧
        SCOLHKG.Real.SafeSimpleRegretBound
          p (recommendation ω) xStar error) :
    μ (EndToEndStatisticalFailure p recommendation xStar error) ≤
      ∑ component, ENNReal.ofReal (delta component) := by
  have hSubset :
      EndToEndStatisticalFailure p recommendation xStar error
        ⊆ JointStatisticalBadEvent bad := by
    intro ω hFailure
    by_contra hNotBad
    exact hFailure (hOutside ω hNotBad)
  exact (measure_mono hSubset).trans
    (joint_statistical_bad_event_le_delta_sum bad delta hEach)

end SCOLHKG.Measure
