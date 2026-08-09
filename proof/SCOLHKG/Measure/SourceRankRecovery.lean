import Mathlib
import SCOLHKG.Measure.GPKernelConfidence
import SCOLHKG.Measure.SubGaussianConfidence
import SCOLHKG.Real.SourceRankRecovery

namespace SCOLHKG.Measure

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal ProbabilityTheory BigOperators

variable {Omega Rep Profile : Type*}
  {mOmega : MeasurableSpace Omega} {mu : Measure Omega}

noncomputable def replicatedSourceMeanError
    [Fintype Rep]
    (noise : Rep → Omega → ℝ) : Omega → ℝ :=
  finiteKernelPosteriorError
    Finset.univ
    (fun (_ : Unit) (_ : Rep) => 1 / (Fintype.card Rep : ℝ))
    noise
    ()

noncomputable def replicatedSourceMeanProxy
    [Fintype Rep]
    (proxy : Rep → NNReal) : NNReal :=
  finiteKernelSubGaussianParam
    Finset.univ
    (fun (_ : Unit) (_ : Rep) => 1 / (Fintype.card Rep : ℝ))
    proxy
    ()

theorem replicatedSourceMeanError_subGaussian
    [Fintype Rep]
    (noise : Rep → Omega → ℝ)
    (proxy : Rep → NNReal)
    (hindep : iIndepFun noise mu)
    (hnoise : ∀ replication,
      HasSubgaussianMGF (noise replication) (proxy replication) mu) :
    HasSubgaussianMGF
      (replicatedSourceMeanError noise)
      (replicatedSourceMeanProxy proxy)
      mu := by
  simpa [
    replicatedSourceMeanError,
    replicatedSourceMeanProxy,
    finiteKernelPosteriorError,
    finiteKernelSubGaussianParam,
  ] using finiteKernelPosteriorError_subGaussian
    (μ := mu)
    (active := Finset.univ)
    (weight := fun (_ : Unit) (_ : Rep) =>
      1 / (Fintype.card Rep : ℝ))
    (noise := noise)
    (c := proxy)
    (x := ())
    hindep
    (fun replication _hreplication => hnoise replication)

theorem finite_source_profile_mean_bad_event_le_sum
    [Fintype Rep]
    [IsFiniteMeasure mu]
    (profiles : Finset Profile)
    (noise : Profile → Rep → Omega → ℝ)
    (proxy : Profile → Rep → NNReal)
    (radius delta : Profile → ℝ)
    (hindep : ∀ profile ∈ profiles, iIndepFun (noise profile) mu)
    (hnoise : ∀ profile ∈ profiles, ∀ replication,
      HasSubgaussianMGF
        (noise profile replication) (proxy profile replication) mu)
    (hradius : ∀ profile ∈ profiles, 0 ≤ radius profile)
    (htail : ∀ profile ∈ profiles,
      2 * Real.exp (
        -(radius profile) ^ 2
          / (2 * (replicatedSourceMeanProxy (proxy profile) : ℝ)))
        ≤ delta profile) :
    mu.real
        (⋃ profile ∈ profiles,
          CenteredSubGaussianBadEvent
            (replicatedSourceMeanError (noise profile))
            (radius profile))
      ≤ ∑ profile ∈ profiles, delta profile := by
  exact centeredSubGaussian_finite_candidate_bad_event_le_sum
    (μ := mu)
    (s := profiles)
    (X := fun profile => replicatedSourceMeanError (noise profile))
    (c := fun profile => replicatedSourceMeanProxy (proxy profile))
    (radius := radius)
    (delta := delta)
    (fun profile hprofile =>
      replicatedSourceMeanError_subGaussian
        (mu := mu)
        (noise profile)
        (proxy profile)
        (hindep profile hprofile)
        (hnoise profile hprofile))
    hradius
    htail

theorem floored_margin_bad_event_subset_mean_union_scale_bad
    (estimatedMean trueMean estimatedScale trueScale : Omega → ℝ)
    (meanRadius scaleRadius scaleFloor z tau : ℝ)
    (hFloor : ∀ omega, scaleFloor ≤ trueScale omega)
    (hZ : 0 ≤ z) :
    {omega |
      meanRadius + z * scaleRadius <
        |(estimatedMean omega + z * max (estimatedScale omega) scaleFloor - tau)
          - (trueMean omega + z * trueScale omega - tau)|}
      ⊆
    {omega | meanRadius < |estimatedMean omega - trueMean omega|} ∪
      {omega | scaleRadius < |estimatedScale omega - trueScale omega|} := by
  intro omega hMargin
  by_contra hOutside
  simp only [Set.mem_union, Set.mem_setOf_eq, not_or, not_lt] at hOutside
  have hBound := SCOLHKG.Real.floored_empirical_chance_margin_error_le
    (tau := tau) hOutside.1 hOutside.2 (hFloor omega) hZ
  exact (not_le_of_gt hMargin) hBound

theorem floored_margin_bad_event_measure_le
    [IsFiniteMeasure mu]
    (estimatedMean trueMean estimatedScale trueScale : Omega → ℝ)
    (meanRadius scaleRadius scaleFloor z tau deltaMean deltaScale : ℝ)
    (hFloor : ∀ omega, scaleFloor ≤ trueScale omega)
    (hZ : 0 ≤ z)
    (hMeanTail :
      mu.real {omega | meanRadius <
        |estimatedMean omega - trueMean omega|} ≤ deltaMean)
    (hScaleTail :
      mu.real {omega | scaleRadius <
        |estimatedScale omega - trueScale omega|} ≤ deltaScale) :
    mu.real {omega |
      meanRadius + z * scaleRadius <
        |(estimatedMean omega + z * max (estimatedScale omega) scaleFloor - tau)
          - (trueMean omega + z * trueScale omega - tau)|}
      ≤ deltaMean + deltaScale := by
  calc
    mu.real {omega |
        meanRadius + z * scaleRadius <
          |(estimatedMean omega + z * max (estimatedScale omega) scaleFloor - tau)
            - (trueMean omega + z * trueScale omega - tau)|}
      ≤ mu.real (
          {omega | meanRadius < |estimatedMean omega - trueMean omega|} ∪
          {omega | scaleRadius < |estimatedScale omega - trueScale omega|}) :=
        measureReal_mono
          (floored_margin_bad_event_subset_mean_union_scale_bad
            estimatedMean trueMean estimatedScale trueScale
            meanRadius scaleRadius scaleFloor z tau hFloor hZ)
    _ ≤ mu.real {omega | meanRadius <
          |estimatedMean omega - trueMean omega|}
        + mu.real {omega | scaleRadius <
          |estimatedScale omega - trueScale omega|} :=
        measureReal_union_le _ _
    _ ≤ deltaMean + deltaScale := add_le_add hMeanTail hScaleTail

end SCOLHKG.Measure
