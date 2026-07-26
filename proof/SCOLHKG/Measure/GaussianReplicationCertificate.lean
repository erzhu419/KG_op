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

end SCOLHKG.Measure
