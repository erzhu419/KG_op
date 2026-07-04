import SCOLHKG.Certification

namespace SCOLHKG

/-!
Safe recommendation and simple-regret consequences.

This is the inherited optimization layer: once mean confidence and variance
over-certification are available, a certified recommendation is truly feasible;
once the certified feasible set is searched well enough, safe simple regret
follows.
-/

universe u

structure ChanceOptimization (Design : Type u) where
  objective : Design → Nat
  trueMean : Design → Nat
  posteriorMean : Design → Nat
  epistemicSlack : Design → Nat
  trueSigma : Design → Nat
  certSigma : Design → Nat
  tau : Nat

def TrueChanceFeasible
    {Design : Type u}
    (p : ChanceOptimization Design)
    (x : Design) : Prop :=
  p.trueMean x + p.trueSigma x ≤ p.tau

def CertifiedFeasible
    {Design : Type u}
    (p : ChanceOptimization Design)
    (x : Design) : Prop :=
  Certified (p.posteriorMean x) (p.epistemicSlack x) (p.certSigma x) p.tau

def SafeSimpleRegretBound
    {Design : Type u}
    (p : ChanceOptimization Design)
    (xRec xStar : Design)
    (eps : Nat) : Prop :=
  p.objective xRec ≤ p.objective xStar + eps

def BestFeasible
    {Design : Type u}
    (p : ChanceOptimization Design)
    (xStar : Design) : Prop :=
  TrueChanceFeasible p xStar ∧
    ∀ y, TrueChanceFeasible p y → p.objective xStar ≤ p.objective y

theorem certifiedFeasible_sound
    {Design : Type u}
    (p : ChanceOptimization Design)
    (x : Design)
    (hMean : p.trueMean x ≤ p.posteriorMean x + p.epistemicSlack x)
    (hSigma : p.trueSigma x ≤ p.certSigma x)
    (hCert : CertifiedFeasible p x) :
    TrueChanceFeasible p x := by
  unfold TrueChanceFeasible CertifiedFeasible at *
  exact certification_sound_with_variance_upper hMean hSigma hCert

theorem zero_safe_regret_of_best_feasible_recommendation
    {Design : Type u}
    (p : ChanceOptimization Design)
    (xRec xStar : Design)
    (hRecBest :
      ∀ y, TrueChanceFeasible p y → p.objective xRec ≤ p.objective y)
    (hStarFeasible : TrueChanceFeasible p xStar) :
    SafeSimpleRegretBound p xRec xStar 0 := by
  unfold SafeSimpleRegretBound
  simpa using hRecBest xStar hStarFeasible

theorem certified_recommendation_safe_and_regret
    {Design : Type u}
    (p : ChanceOptimization Design)
    (xRec xStar : Design)
    (eps : Nat)
    (hMean : p.trueMean xRec ≤ p.posteriorMean xRec + p.epistemicSlack xRec)
    (hSigma : p.trueSigma xRec ≤ p.certSigma xRec)
    (hCert : CertifiedFeasible p xRec)
    (hRegret : p.objective xRec ≤ p.objective xStar + eps) :
    TrueChanceFeasible p xRec ∧ SafeSimpleRegretBound p xRec xStar eps := by
  constructor
  · exact certifiedFeasible_sound p xRec hMean hSigma hCert
  · exact hRegret

end SCOLHKG

