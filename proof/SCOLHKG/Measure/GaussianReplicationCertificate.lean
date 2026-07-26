import Mathlib
import SCOLHKG.Real.GaussianReplicationCertificate

namespace SCOLHKG.Measure

open MeasureTheory

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

def GaussianMeanCoverageFailure
    (trueMean : ℝ)
    (meanUpper : Ω → ℝ) : Set Ω :=
  {ω | meanUpper ω < trueMean}

def GaussianSigmaCoverageFailure
    (trueSigma : ℝ)
    (sigmaUpper : Ω → ℝ) : Set Ω :=
  {ω | sigmaUpper ω < trueSigma}

def FalseGaussianReplicationCertificate
    (trueMean trueSigma zAlpha tau : ℝ)
    (meanUpper sigmaUpper : Ω → ℝ) : Set Ω :=
  {ω |
    SCOLHKG.Real.gaussianReplicationMargin
        (meanUpper ω) (sigmaUpper ω) zAlpha tau ≤ 0 ∧
      tau < trueMean + zAlpha * trueSigma}

theorem false_gaussian_replication_certificate_subset_coverage_failure
    (trueMean trueSigma zAlpha tau : ℝ)
    (meanUpper sigmaUpper : Ω → ℝ)
    (hZ : 0 ≤ zAlpha) :
    FalseGaussianReplicationCertificate
        trueMean trueSigma zAlpha tau meanUpper sigmaUpper
      ⊆ GaussianMeanCoverageFailure trueMean meanUpper
        ∪ GaussianSigmaCoverageFailure trueSigma sigmaUpper := by
  intro ω hFalse
  by_contra hGood
  have hMeanNot :
      ω ∉ GaussianMeanCoverageFailure trueMean meanUpper := by
    intro hMean
    exact hGood (Set.mem_union_left _ hMean)
  have hSigmaNot :
      ω ∉ GaussianSigmaCoverageFailure trueSigma sigmaUpper := by
    intro hSigma
    exact hGood (Set.mem_union_right _ hSigma)
  have hMean : trueMean ≤ meanUpper ω := by
    exact not_lt.mp hMeanNot
  have hSigma : trueSigma ≤ sigmaUpper ω := by
    exact not_lt.mp hSigmaNot
  have hSound :=
    SCOLHKG.Real.gaussian_replication_margin_sound
      hMean hSigma hZ hFalse.1
  exact (not_lt_of_ge hSound) hFalse.2

theorem false_gaussian_replication_certificate_probability_le
    (trueMean trueSigma zAlpha tau : ℝ)
    (meanUpper sigmaUpper : Ω → ℝ)
    (meanDelta sigmaDelta : ℝ)
    [IsFiniteMeasure μ]
    (hZ : 0 ≤ zAlpha)
    (hMean :
      μ.real (GaussianMeanCoverageFailure trueMean meanUpper)
        ≤ meanDelta)
    (hSigma :
      μ.real (GaussianSigmaCoverageFailure trueSigma sigmaUpper)
        ≤ sigmaDelta) :
    μ.real (FalseGaussianReplicationCertificate
      trueMean trueSigma zAlpha tau meanUpper sigmaUpper)
      ≤ meanDelta + sigmaDelta := by
  calc
    μ.real (FalseGaussianReplicationCertificate
        trueMean trueSigma zAlpha tau meanUpper sigmaUpper)
      ≤ μ.real (
          GaussianMeanCoverageFailure trueMean meanUpper
            ∪ GaussianSigmaCoverageFailure trueSigma sigmaUpper) :=
        measureReal_mono (
          false_gaussian_replication_certificate_subset_coverage_failure
            trueMean trueSigma zAlpha tau meanUpper sigmaUpper hZ)
    _ ≤ μ.real (GaussianMeanCoverageFailure trueMean meanUpper)
          + μ.real (GaussianSigmaCoverageFailure trueSigma sigmaUpper) :=
        measureReal_union_le _ _
    _ ≤ meanDelta + sigmaDelta := by
      linarith

theorem false_gaussian_replication_certificate_probability_le_delta
    (trueMean trueSigma zAlpha tau : ℝ)
    (meanUpper sigmaUpper : Ω → ℝ)
    (meanDelta sigmaDelta delta : ℝ)
    [IsFiniteMeasure μ]
    (hZ : 0 ≤ zAlpha)
    (hMean :
      μ.real (GaussianMeanCoverageFailure trueMean meanUpper)
        ≤ meanDelta)
    (hSigma :
      μ.real (GaussianSigmaCoverageFailure trueSigma sigmaUpper)
        ≤ sigmaDelta)
    (hSplit : meanDelta + sigmaDelta ≤ delta) :
    μ.real (FalseGaussianReplicationCertificate
      trueMean trueSigma zAlpha tau meanUpper sigmaUpper)
      ≤ delta := by
  exact le_trans (
    false_gaussian_replication_certificate_probability_le
      trueMean trueSigma zAlpha tau meanUpper sigmaUpper
      meanDelta sigmaDelta hZ hMean hSigma
  ) hSplit

def GaussianQuantileCoverageFailure
    (trueQuantile : ℝ)
    (quantileUpper : Ω → ℝ) : Set Ω :=
  {ω | quantileUpper ω < trueQuantile}

def FalseGaussianQuantileToleranceCertificate
    (trueQuantile tau : ℝ)
    (quantileUpper : Ω → ℝ) : Set Ω :=
  {ω |
    SCOLHKG.Real.gaussianQuantileToleranceMargin
        (quantileUpper ω) tau ≤ 0
      ∧ tau < trueQuantile}

theorem false_gaussian_quantile_tolerance_certificate_subset
    (trueQuantile tau : ℝ)
    (quantileUpper : Ω → ℝ) :
    FalseGaussianQuantileToleranceCertificate
        trueQuantile tau quantileUpper
      ⊆ GaussianQuantileCoverageFailure
        trueQuantile quantileUpper := by
  intro ω hFalse
  change
    SCOLHKG.Real.gaussianQuantileToleranceMargin
        (quantileUpper ω) tau ≤ 0
      ∧ tau < trueQuantile at hFalse
  have hUpper : quantileUpper ω ≤ tau := by
    unfold SCOLHKG.Real.gaussianQuantileToleranceMargin at hFalse
    linarith [hFalse.1]
  change quantileUpper ω < trueQuantile
  exact lt_of_le_of_lt hUpper hFalse.2

theorem false_gaussian_quantile_tolerance_certificate_probability_le
    (trueQuantile tau : ℝ)
    (quantileUpper : Ω → ℝ)
    (delta : ℝ)
    [IsFiniteMeasure μ]
    (hCoverage :
      μ.real (GaussianQuantileCoverageFailure
        trueQuantile quantileUpper) ≤ delta) :
    μ.real (FalseGaussianQuantileToleranceCertificate
      trueQuantile tau quantileUpper) ≤ delta := by
  exact le_trans (
    measureReal_mono (
      false_gaussian_quantile_tolerance_certificate_subset
        trueQuantile tau quantileUpper)
  ) hCoverage

def CandidateFalseCertificate
    (isUnsafe : Prop)
    (certified : Ω → Prop) : Set Ω :=
  {ω | certified ω ∧ isUnsafe}

def FalseOrderedTwoPolicyDeployment
    (unsafeFirst unsafeSecond : Prop)
    (certifiedFirst certifiedSecond : Ω → Prop) : Set Ω :=
  {ω |
    (certifiedFirst ω ∧ unsafeFirst)
      ∨
    (¬ certifiedFirst ω ∧ certifiedSecond ω ∧ unsafeSecond)}

theorem false_ordered_two_policy_deployment_subset
    (unsafeFirst unsafeSecond : Prop)
    (certifiedFirst certifiedSecond : Ω → Prop) :
    FalseOrderedTwoPolicyDeployment
        unsafeFirst unsafeSecond certifiedFirst certifiedSecond
      ⊆ CandidateFalseCertificate unsafeFirst certifiedFirst
        ∪ CandidateFalseCertificate unsafeSecond certifiedSecond := by
  intro ω hFalse
  rcases hFalse with hFirst | hSecond
  · exact Set.mem_union_left _ hFirst
  · exact Set.mem_union_right _ ⟨hSecond.2.1, hSecond.2.2⟩

theorem false_ordered_two_policy_deployment_probability_le
    (unsafeFirst unsafeSecond : Prop)
    (certifiedFirst certifiedSecond : Ω → Prop)
    (deltaFirst deltaSecond : ℝ)
    [IsFiniteMeasure μ]
    (hFirst :
      μ.real (CandidateFalseCertificate unsafeFirst certifiedFirst)
        ≤ deltaFirst)
    (hSecond :
      μ.real (CandidateFalseCertificate unsafeSecond certifiedSecond)
        ≤ deltaSecond) :
    μ.real (FalseOrderedTwoPolicyDeployment
      unsafeFirst unsafeSecond certifiedFirst certifiedSecond)
      ≤ deltaFirst + deltaSecond := by
  calc
    μ.real (FalseOrderedTwoPolicyDeployment
        unsafeFirst unsafeSecond certifiedFirst certifiedSecond)
      ≤ μ.real (
          CandidateFalseCertificate unsafeFirst certifiedFirst
            ∪ CandidateFalseCertificate unsafeSecond certifiedSecond) :=
        measureReal_mono (
          false_ordered_two_policy_deployment_subset
            unsafeFirst unsafeSecond certifiedFirst certifiedSecond)
    _ ≤ μ.real (CandidateFalseCertificate unsafeFirst certifiedFirst)
          + μ.real (
              CandidateFalseCertificate unsafeSecond certifiedSecond) :=
        measureReal_union_le _ _
    _ ≤ deltaFirst + deltaSecond := by
      linarith

theorem false_ordered_two_policy_deployment_probability_le_familywise_delta
    (unsafeFirst unsafeSecond : Prop)
    (certifiedFirst certifiedSecond : Ω → Prop)
    (deltaFirst deltaSecond familywiseDelta : ℝ)
    [IsFiniteMeasure μ]
    (hFirst :
      μ.real (CandidateFalseCertificate unsafeFirst certifiedFirst)
        ≤ deltaFirst)
    (hSecond :
      μ.real (CandidateFalseCertificate unsafeSecond certifiedSecond)
        ≤ deltaSecond)
    (hSpend : deltaFirst + deltaSecond ≤ familywiseDelta) :
    μ.real (FalseOrderedTwoPolicyDeployment
      unsafeFirst unsafeSecond certifiedFirst certifiedSecond)
      ≤ familywiseDelta := by
  exact le_trans (
    false_ordered_two_policy_deployment_probability_le
      unsafeFirst unsafeSecond certifiedFirst certifiedSecond
      deltaFirst deltaSecond hFirst hSecond
  ) hSpend

theorem false_ordered_two_policy_quantile_deployment_probability_le
    (unsafeFirst unsafeSecond : Prop)
    (certifiedFirst certifiedSecond : Ω → Prop)
    (deltaFirst deltaSecond familywiseDelta : ℝ)
    [IsFiniteMeasure μ]
    (hFirst :
      μ.real (CandidateFalseCertificate unsafeFirst certifiedFirst)
        ≤ deltaFirst)
    (hSecond :
      μ.real (CandidateFalseCertificate unsafeSecond certifiedSecond)
        ≤ deltaSecond)
    (hSpend : deltaFirst + deltaSecond ≤ familywiseDelta) :
    μ.real (FalseOrderedTwoPolicyDeployment
      unsafeFirst unsafeSecond certifiedFirst certifiedSecond)
      ≤ familywiseDelta := by
  exact
    false_ordered_two_policy_deployment_probability_le_familywise_delta
      unsafeFirst unsafeSecond certifiedFirst certifiedSecond
      deltaFirst deltaSecond familywiseDelta
      hFirst hSecond hSpend

theorem false_frozen_safe_interior_deployment_probability_le
    {Design : Type*}
    (shortlist : SCOLHKG.Real.FrozenSafeInteriorShortlist Design)
    (isUnsafe : Design → Prop)
    (certifiedFirst certifiedSecond : Ω → Prop)
    (deltaFirst deltaSecond familywiseDelta : ℝ)
    [IsFiniteMeasure μ]
    (hFirst :
      μ.real (
        CandidateFalseCertificate
          (isUnsafe shortlist.primary) certifiedFirst)
        ≤ deltaFirst)
    (hSecond :
      μ.real (
        CandidateFalseCertificate
          (isUnsafe shortlist.support) certifiedSecond)
        ≤ deltaSecond)
    (hSpend : deltaFirst + deltaSecond ≤ familywiseDelta) :
    μ.real (
      FalseOrderedTwoPolicyDeployment
        (isUnsafe shortlist.primary)
        (isUnsafe shortlist.support)
        certifiedFirst certifiedSecond)
      ≤ familywiseDelta := by
  exact
    false_ordered_two_policy_deployment_probability_le_familywise_delta
      (isUnsafe shortlist.primary)
      (isUnsafe shortlist.support)
      certifiedFirst certifiedSecond
      deltaFirst deltaSecond familywiseDelta
      hFirst hSecond hSpend

end SCOLHKG.Measure
