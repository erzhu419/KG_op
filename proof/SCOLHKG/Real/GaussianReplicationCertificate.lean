import Mathlib

namespace SCOLHKG.Real

/-!
# Frozen-policy Gaussian replication certificate

After search freezes a policy, an independent replication batch supplies a
one-sided upper bound for its Gaussian mean and standard deviation. The
certificate itself only needs those two coverage events. Student-t and
chi-square quantiles are the Python mechanism used to construct the bounds.
-/

def gaussianReplicationMargin
    (meanUpper sigmaUpper zAlpha tau : ℝ) : ℝ :=
  meanUpper + zAlpha * sigmaUpper - tau

def chanceSafetyDepth
    (trueMean trueSigma zAlpha tau : ℝ) : ℝ :=
  tau - (trueMean + zAlpha * trueSigma)

def replicationBoundExcess
    (trueMean trueSigma meanUpper sigmaUpper zAlpha : ℝ) : ℝ :=
  (meanUpper - trueMean) + zAlpha * (sigmaUpper - trueSigma)

theorem gaussian_replication_margin_sound
    {trueMean trueSigma meanUpper sigmaUpper zAlpha tau : ℝ}
    (hMean : trueMean ≤ meanUpper)
    (hSigma : trueSigma ≤ sigmaUpper)
    (hZ : 0 ≤ zAlpha)
    (hCertificate :
      gaussianReplicationMargin meanUpper sigmaUpper zAlpha tau ≤ 0) :
    trueMean + zAlpha * trueSigma ≤ tau := by
  have hScaledSigma :
      zAlpha * trueSigma ≤ zAlpha * sigmaUpper :=
    mul_le_mul_of_nonneg_left hSigma hZ
  unfold gaussianReplicationMargin at hCertificate
  linarith

theorem gaussian_replication_margin_decomposition
    (trueMean trueSigma meanUpper sigmaUpper zAlpha tau : ℝ) :
    gaussianReplicationMargin meanUpper sigmaUpper zAlpha tau =
      replicationBoundExcess
          trueMean trueSigma meanUpper sigmaUpper zAlpha
        - chanceSafetyDepth trueMean trueSigma zAlpha tau := by
  simp [
    gaussianReplicationMargin,
    replicationBoundExcess,
    chanceSafetyDepth,
  ]
  ring

theorem gaussian_replication_certificate_nonvacuous_of_excess_le_depth
    {trueMean trueSigma meanUpper sigmaUpper zAlpha tau : ℝ}
    (hContraction :
      replicationBoundExcess
          trueMean trueSigma meanUpper sigmaUpper zAlpha
        ≤ chanceSafetyDepth trueMean trueSigma zAlpha tau) :
    gaussianReplicationMargin meanUpper sigmaUpper zAlpha tau ≤ 0 := by
  rw [
    gaussian_replication_margin_decomposition
      trueMean trueSigma meanUpper sigmaUpper zAlpha tau
  ]
  linarith

structure FrozenTerminalPolicy (Design : Type*) where
  policy : Design
  searchBudget : ℕ

def freezeTerminalPolicy
    {Design : Type*}
    (policy : Design)
    (searchBudget : ℕ) : FrozenTerminalPolicy Design where
  policy := policy
  searchBudget := searchBudget

theorem independent_verification_does_not_change_frozen_policy
    {Design Sample : Type*}
    (policy : Design)
    (searchBudget : ℕ)
    (_verification : List Sample) :
    (freezeTerminalPolicy policy searchBudget).policy = policy := by
  rfl

theorem search_plus_verification_budget_accounting
    (searchBudget verificationBudget : ℕ) :
    searchBudget + verificationBudget =
      (freezeTerminalPolicy Unit.unit searchBudget).searchBudget
        + verificationBudget := by
  rfl

end SCOLHKG.Real
