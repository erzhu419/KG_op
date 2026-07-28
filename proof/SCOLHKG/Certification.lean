namespace SCOLHKG

/-!
Algebraic core of conservative chance certification.

The full paper theorem will instantiate `sigmaQuantile` as
`z_{1-alpha} * sqrt(v_C^+)` over the reals.  Here we prove the ordered
arithmetic implication that matters for certification: if the true mean and
true standard-deviation term are both upper-bounded by certified quantities,
then the true quantile bound is also below the threshold.
-/

def chanceUpper (mean epistemicSlack sigmaQuantile : Nat) : Nat :=
  mean + epistemicSlack + sigmaQuantile

def Certified
    (mean epistemicSlack sigmaQuantile tau : Nat) : Prop :=
  chanceUpper mean epistemicSlack sigmaQuantile ≤ tau

theorem certified_implies_upper_le_tau
    {mean epistemicSlack sigmaQuantile tau : Nat}
    (h : Certified mean epistemicSlack sigmaQuantile tau) :
    chanceUpper mean epistemicSlack sigmaQuantile ≤ tau := by
  exact h

theorem chanceUpper_sigma_monotone
    {mean epistemicSlack sigmaSmall sigmaBig : Nat}
    (hSigma : sigmaSmall ≤ sigmaBig) :
    chanceUpper mean epistemicSlack sigmaSmall ≤
      chanceUpper mean epistemicSlack sigmaBig := by
  unfold chanceUpper
  exact Nat.add_le_add_left hSigma (mean + epistemicSlack)

theorem certification_sound_algebra
    {trueMean mean epistemicSlack sigmaQuantile tau : Nat}
    (hMean : trueMean ≤ mean + epistemicSlack)
    (hCert : Certified mean epistemicSlack sigmaQuantile tau) :
    trueMean + sigmaQuantile ≤ tau := by
  unfold Certified chanceUpper at hCert
  exact Nat.le_trans (Nat.add_le_add_right hMean sigmaQuantile) hCert

theorem certification_sound_with_variance_upper
    {trueMean mean epistemicSlack trueSigma certSigma tau : Nat}
    (hMean : trueMean ≤ mean + epistemicSlack)
    (hSigma : trueSigma ≤ certSigma)
    (hCert : Certified mean epistemicSlack certSigma tau) :
    trueMean + trueSigma ≤ tau := by
  unfold Certified chanceUpper at hCert
  have hMeanAdd :
      trueMean + trueSigma ≤ mean + epistemicSlack + trueSigma := by
    exact Nat.add_le_add_right hMean trueSigma
  have hSigmaAdd :
      mean + epistemicSlack + trueSigma ≤
        mean + epistemicSlack + certSigma := by
    exact Nat.add_le_add_left hSigma (mean + epistemicSlack)
  exact Nat.le_trans hMeanAdd (Nat.le_trans hSigmaAdd hCert)

theorem certified_with_larger_sigma_implies_certified_with_smaller_sigma
    {mean epistemicSlack sigmaSmall sigmaBig tau : Nat}
    (hSigma : sigmaSmall ≤ sigmaBig)
    (hCert : Certified mean epistemicSlack sigmaBig tau) :
    Certified mean epistemicSlack sigmaSmall tau := by
  exact Nat.le_trans (chanceUpper_sigma_monotone hSigma) hCert

end SCOLHKG

