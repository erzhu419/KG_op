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

def gaussianQuantileToleranceUpper
    (sampleMean sampleStd toleranceFactor : ℝ) : ℝ :=
  sampleMean + toleranceFactor * sampleStd

def gaussianQuantileToleranceMargin
    (quantileUpper tau : ℝ) : ℝ :=
  quantileUpper - tau

theorem gaussian_quantile_tolerance_margin_sound
    {trueQuantile quantileUpper tau : ℝ}
    (hCoverage : trueQuantile ≤ quantileUpper)
    (hCertificate :
      gaussianQuantileToleranceMargin quantileUpper tau ≤ 0) :
    trueQuantile ≤ tau := by
  unfold gaussianQuantileToleranceMargin at hCertificate
  linarith

theorem gaussian_quantile_tolerance_upper_sound
    {trueMean trueSigma zAlpha sampleMean sampleStd toleranceFactor tau : ℝ}
    (hCoverage :
      trueMean + zAlpha * trueSigma
        ≤ gaussianQuantileToleranceUpper
            sampleMean sampleStd toleranceFactor)
    (hCertificate :
      gaussianQuantileToleranceMargin
          (gaussianQuantileToleranceUpper
            sampleMean sampleStd toleranceFactor)
          tau
        ≤ 0) :
    trueMean + zAlpha * trueSigma ≤ tau := by
  exact gaussian_quantile_tolerance_margin_sound
    hCoverage hCertificate

theorem gaussian_quantile_tolerance_nonvacuous_of_upper_le
    {quantileUpper tau : ℝ}
    (hUpper : quantileUpper ≤ tau) :
    gaussianQuantileToleranceMargin quantileUpper tau ≤ 0 := by
  unfold gaussianQuantileToleranceMargin
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

structure FrozenTerminalShortlist (Design : Type*) where
  policies : List Design
  searchBudget : ℕ
  verificationBudgetPerPolicy : ℕ

def freezeTerminalShortlist
    {Design : Type*}
    (policies : List Design)
    (searchBudget verificationBudgetPerPolicy : ℕ) :
    FrozenTerminalShortlist Design where
  policies := policies
  searchBudget := searchBudget
  verificationBudgetPerPolicy := verificationBudgetPerPolicy

def firstCertified?
    {Design : Type*}
    (certified : Design → Bool) :
    List Design → Option Design
  | [] => none
  | policy :: remaining =>
      if certified policy then
        some policy
      else
        firstCertified? certified remaining

theorem firstCertified_mem_and_certified
    {Design : Type*}
    (certified : Design → Bool)
    (policies : List Design)
    (selected : Design)
    (hSelected : firstCertified? certified policies = some selected) :
    selected ∈ policies ∧ certified selected = true := by
  induction policies with
  | nil =>
      simp [firstCertified?] at hSelected
  | cons policy remaining inductionHypothesis =>
      by_cases hPolicy : certified policy = true
      · simp [firstCertified?, hPolicy] at hSelected
        subst selected
        simp [hPolicy]
      · have hPolicyFalse : certified policy = false := by
          exact Bool.eq_false_of_not_eq_true hPolicy
        simp [firstCertified?, hPolicyFalse] at hSelected
        have hRemaining :=
          inductionHypothesis hSelected
        exact ⟨by simp [hRemaining.1], hRemaining.2⟩

theorem firstCertified_safe
    {Design : Type*}
    (certified : Design → Bool)
    (safe : Design → Prop)
    (policies : List Design)
    (selected : Design)
    (hSelected : firstCertified? certified policies = some selected)
    (hCertificateSound :
      ∀ policy, certified policy = true → safe policy) :
    safe selected := by
  exact hCertificateSound selected (
    firstCertified_mem_and_certified
      certified policies selected hSelected
  ).2

theorem ordered_shortlist_verification_budget_le
    (searchBudget verificationBudgetPerPolicy tested shortlistSize : ℕ)
    (hTested : tested ≤ shortlistSize) :
    searchBudget + tested * verificationBudgetPerPolicy
      ≤ searchBudget + shortlistSize * verificationBudgetPerPolicy := by
  exact Nat.add_le_add_left
    (Nat.mul_le_mul_right verificationBudgetPerPolicy hTested)
    searchBudget

theorem ordered_two_policy_asymmetric_verification_budget_le
    (searchBudget firstBudget secondBudget : ℕ)
    (firstCertified : Bool) :
    searchBudget
        + (if firstCertified then firstBudget
           else firstBudget + secondBudget)
      ≤ searchBudget + firstBudget + secondBudget := by
  cases firstCertified <;> simp [Nat.add_assoc]

theorem ordered_two_policy_exact_tolerance_budget_le
    (searchBudget : ℕ)
    (firstCertified : Bool) :
    searchBudget
        + (if firstCertified then 64 else 64 + 96)
      ≤ searchBudget + 64 + 96 := by
  exact ordered_two_policy_asymmetric_verification_budget_le
    searchBudget 64 96 firstCertified

theorem ordered_two_policy_powered_safe_interior_budget_le
    (searchBudget : ℕ)
    (firstCertified : Bool) :
    searchBudget
        + (if firstCertified then 80 else 80 + 96)
      ≤ searchBudget + 80 + 96 := by
  exact ordered_two_policy_asymmetric_verification_budget_le
    searchBudget 80 96 firstCertified

theorem ordered_shortlist_freeze_is_verification_invariant
    {Design Sample : Type*}
    (policies : List Design)
    (searchBudget verificationBudgetPerPolicy : ℕ)
    (_verification : List Sample) :
    (freezeTerminalShortlist
      policies searchBudget verificationBudgetPerPolicy).policies
      = policies := by
  rfl

structure FrozenSafeInteriorShortlist (Design : Type*) where
  primary : Design
  support : Design
  supportFromInitialAtlas : Prop
  supportInPosteriorViolationSublevel : Prop
  searchBudget : ℕ

def freezeSafeInteriorShortlist
    {Design : Type*}
    (primary support : Design)
    (supportFromInitialAtlas : Prop)
    (supportInPosteriorViolationSublevel : Prop)
    (searchBudget : ℕ) :
    FrozenSafeInteriorShortlist Design where
  primary := primary
  support := support
  supportFromInitialAtlas := supportFromInitialAtlas
  supportInPosteriorViolationSublevel :=
    supportInPosteriorViolationSublevel
  searchBudget := searchBudget

def safeInteriorPolicies
    {Design : Type*}
    (shortlist : FrozenSafeInteriorShortlist Design) : List Design :=
  [shortlist.primary, shortlist.support]

theorem safe_interior_shortlist_contains_primary_and_support
    {Design : Type*}
    (shortlist : FrozenSafeInteriorShortlist Design) :
    safeInteriorPolicies shortlist =
      [shortlist.primary, shortlist.support] := by
  rfl

theorem safe_interior_selector_contract
    {Design : Type*}
    (primary support : Design)
    (supportFromInitialAtlas : Prop)
    (supportInPosteriorViolationSublevel : Prop)
    (searchBudget : ℕ)
    (hAtlas : supportFromInitialAtlas)
    (hSublevel : supportInPosteriorViolationSublevel) :
    let shortlist :=
      freezeSafeInteriorShortlist
        primary support
        supportFromInitialAtlas
        supportInPosteriorViolationSublevel
        searchBudget
    shortlist.supportFromInitialAtlas
      ∧ shortlist.supportInPosteriorViolationSublevel := by
  exact ⟨hAtlas, hSublevel⟩

theorem safe_interior_shortlist_freeze_is_verification_invariant
    {Design Sample : Type*}
    (shortlist : FrozenSafeInteriorShortlist Design)
    (_verification : List Sample) :
    safeInteriorPolicies shortlist =
      [shortlist.primary, shortlist.support] := by
  rfl

end SCOLHKG.Real
