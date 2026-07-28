import Mathlib
import SCOLHKG.Real.Certification

namespace SCOLHKG.Real

/-!
Code-level bridge for `SingleOLHKGAlgorithm._solve_posterior_recommendation`.

The implementation recommends the lowest posterior objective among designs
whose robust chance margin is nonpositive.  The theorem below records the
deterministic guarantee used by the certification layer.
-/

universe u

structure PosteriorRecommendationProblem (Design : Type u) where
  posteriorObjective : Design → ℝ
  posteriorConstraintMean : Design → ℝ
  certificationSigma : Design → ℝ
  z : ℝ
  tau : ℝ
  safetyBuffer : Design → ℝ

def posteriorChanceMargin
    {Design : Type u}
    (p : PosteriorRecommendationProblem Design)
    (x : Design) : ℝ :=
  p.posteriorConstraintMean x + p.z * p.certificationSigma x - p.tau

def robustChanceMargin
    {Design : Type u}
    (p : PosteriorRecommendationProblem Design)
    (x : Design) : ℝ :=
  posteriorChanceMargin p x + p.safetyBuffer x

def RobustPosteriorFeasible
    {Design : Type u}
    (p : PosteriorRecommendationProblem Design)
    (x : Design) : Prop :=
  robustChanceMargin p x ≤ 0

def PosteriorCertifiedFeasible
    {Design : Type u}
    (p : PosteriorRecommendationProblem Design)
    (x : Design) : Prop :=
  Certified (p.posteriorConstraintMean x) 0 p.z (p.certificationSigma x) p.tau

theorem robust_feasible_implies_posterior_certified
    {Design : Type u}
    (p : PosteriorRecommendationProblem Design)
    (x : Design)
    (hSafety : 0 ≤ p.safetyBuffer x)
    (hRobust : RobustPosteriorFeasible p x) :
    PosteriorCertifiedFeasible p x := by
  unfold RobustPosteriorFeasible robustChanceMargin posteriorChanceMargin
    PosteriorCertifiedFeasible Certified chanceUpper at *
  linarith

theorem robust_argmin_is_objective_minimizer_on_robust_set
    {Design : Type u}
    (p : PosteriorRecommendationProblem Design)
    (x : Design)
    (hArgmin :
      ∀ y, RobustPosteriorFeasible p y →
        p.posteriorObjective x ≤ p.posteriorObjective y) :
    ∀ y, RobustPosteriorFeasible p y →
      p.posteriorObjective x ≤ p.posteriorObjective y := by
  exact hArgmin

end SCOLHKG.Real
