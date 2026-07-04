import Mathlib
import SCOLHKG.Real.Certification

namespace SCOLHKG.Real

/-!
Finite-budget safe simple regret over real-valued objectives.

This theorem is the final deterministic layer: if the recommendation is
certified and its optimization error is bounded by `ε`, then it is truly safe
and has simple regret at most `ε` against the comparator.
-/

universe u

structure ChanceOptimization (Design : Type u) where
  objective : Design → ℝ
  trueMean : Design → ℝ
  posteriorMean : Design → ℝ
  epistemicSlack : Design → ℝ
  trueSigma : Design → ℝ
  certSigma : Design → ℝ
  z : ℝ
  tau : ℝ

def TrueChanceFeasible
    {Design : Type u}
    (p : ChanceOptimization Design)
    (x : Design) : Prop :=
  p.trueMean x + p.z * p.trueSigma x ≤ p.tau

def CertifiedFeasible
    {Design : Type u}
    (p : ChanceOptimization Design)
    (x : Design) : Prop :=
  Certified (p.posteriorMean x) (p.epistemicSlack x) p.z (p.certSigma x) p.tau

def SafeSimpleRegretBound
    {Design : Type u}
    (p : ChanceOptimization Design)
    (xRec xStar : Design)
    (eps : ℝ) : Prop :=
  p.objective xRec - p.objective xStar ≤ eps

theorem certified_feasible_sound
    {Design : Type u}
    (p : ChanceOptimization Design)
    (x : Design)
    (hz : 0 ≤ p.z)
    (hMean : p.trueMean x ≤ p.posteriorMean x + p.epistemicSlack x)
    (hSigma : p.trueSigma x ≤ p.certSigma x)
    (hCert : CertifiedFeasible p x) :
    TrueChanceFeasible p x := by
  unfold TrueChanceFeasible CertifiedFeasible at *
  exact certified_implies_true_quantile_bound hz hMean hSigma hCert

theorem finite_budget_safe_simple_regret
    {Design : Type u}
    (p : ChanceOptimization Design)
    (xRec xStar : Design)
    (eps : ℝ)
    (hz : 0 ≤ p.z)
    (hMean : p.trueMean xRec ≤ p.posteriorMean xRec + p.epistemicSlack xRec)
    (hSigma : p.trueSigma xRec ≤ p.certSigma xRec)
    (hCert : CertifiedFeasible p xRec)
    (hOpt : p.objective xRec - p.objective xStar ≤ eps) :
    TrueChanceFeasible p xRec ∧ SafeSimpleRegretBound p xRec xStar eps := by
  constructor
  · exact certified_feasible_sound p xRec hz hMean hSigma hCert
  · exact hOpt

theorem optimization_error_from_upper_objective_bound
    {Design : Type u}
    (p : ChanceOptimization Design)
    (xRec xStar : Design)
    (eps : ℝ)
    (hObj : p.objective xRec ≤ p.objective xStar + eps) :
    SafeSimpleRegretBound p xRec xStar eps := by
  unfold SafeSimpleRegretBound
  linarith

end SCOLHKG.Real

