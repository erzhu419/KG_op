import Mathlib

namespace SCOLHKG.Real

/-!
Real-valued chance certification.

The theorem consumes two events that later proofs must establish:

* posterior mean upper confidence;
* conservative variance/standard-deviation upper bound.

It then proves the certified quantile condition implies true chance-feasibility
at the arithmetic level.
-/

def chanceUpper (mean epistemicSlack z sigma : ℝ) : ℝ :=
  mean + epistemicSlack + z * sigma

def Certified (mean epistemicSlack z certSigma tau : ℝ) : Prop :=
  chanceUpper mean epistemicSlack z certSigma ≤ tau

theorem certified_implies_true_quantile_bound
    {trueMean postMean epistemicSlack z trueSigma certSigma tau : ℝ}
    (hz : 0 ≤ z)
    (hMean : trueMean ≤ postMean + epistemicSlack)
    (hSigma : trueSigma ≤ certSigma)
    (hCert : Certified postMean epistemicSlack z certSigma tau) :
    trueMean + z * trueSigma ≤ tau := by
  unfold Certified chanceUpper at hCert
  have hSigmaMul : z * trueSigma ≤ z * certSigma := by
    exact mul_le_mul_of_nonneg_left hSigma hz
  linarith

structure GPConfidenceEvent where
  trueMean : ℝ
  posteriorMean : ℝ
  epistemicSlack : ℝ

def GPConfidenceEvent.Valid (e : GPConfidenceEvent) : Prop :=
  e.trueMean ≤ e.posteriorMean + e.epistemicSlack

structure VarianceUpperEvent where
  trueSigma : ℝ
  certSigma : ℝ

def VarianceUpperEvent.Valid (e : VarianceUpperEvent) : Prop :=
  e.trueSigma ≤ e.certSigma

theorem gp_confidence_and_variance_upper_certify
    {meanEvent : GPConfidenceEvent}
    {varEvent : VarianceUpperEvent}
    {z tau : ℝ}
    (hz : 0 ≤ z)
    (hMean : meanEvent.Valid)
    (hVar : varEvent.Valid)
    (hCert :
      Certified meanEvent.posteriorMean meanEvent.epistemicSlack
        z varEvent.certSigma tau) :
    meanEvent.trueMean + z * varEvent.trueSigma ≤ tau := by
  exact certified_implies_true_quantile_bound hz hMean hVar hCert

end SCOLHKG.Real

