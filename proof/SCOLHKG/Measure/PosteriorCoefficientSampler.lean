import SCOLHKG.Measure.PosteriorSamplingCandidates

namespace SCOLHKG.Measure

open MeasureTheory ProbabilityTheory
open scoped MeasureTheory ProbabilityTheory BigOperators

/-!
Bridge for the multivariate posterior coefficient sampler used by
`posterior_sample_candidates`.

The implementation samples a coefficient vector, scores a finite raw pool, and
keeps the designs passing a score/feasibility threshold.  The distributional
details of the multivariate sampler are intentionally kept in the `draw` field;
the theorem needed downstream is that the selected random set is contained in
the deterministic raw pool, so the existing finite adaptive concentration bound
applies.
-/

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

structure PosteriorCoefficientSampler
    (Ω Coeff : Type*)
    [MeasurableSpace Ω] where
  draw : Ω → Coeff

noncomputable def posteriorCoefficientSelected
    {Design Coeff : Type*}
    [DecidableEq Design]
    (pool : Finset Design)
    (sampler : PosteriorCoefficientSampler Ω Coeff)
    (score : Design → Coeff → ℝ)
    (threshold : Ω → ℝ)
    (ω : Ω) : Finset Design :=
  posteriorScoreSelected pool score sampler.draw threshold ω

theorem posteriorCoefficientSelected_subset_pool
    {Design Coeff : Type*}
    [DecidableEq Design]
    (pool : Finset Design)
    (sampler : PosteriorCoefficientSampler Ω Coeff)
    (score : Design → Coeff → ℝ)
    (threshold : Ω → ℝ)
    (ω : Ω) :
    posteriorCoefficientSelected pool sampler score threshold ω ⊆ pool := by
  exact posteriorScoreSelected_subset_pool pool score sampler.draw threshold ω

theorem posteriorCoefficientSampler_bad_event_le_sum
    {Design Coeff : Type*}
    [DecidableEq Design]
    (pool : ℕ → Finset Design)
    (sampler : ℕ → PosteriorCoefficientSampler Ω Coeff)
    (score : ℕ → Design → Coeff → ℝ)
    (threshold : ℕ → Ω → ℝ)
    (X : ℕ → Design → Ω → ℝ)
    (c : ℕ → Design → NNReal)
    (radius delta : ℕ → Design → ℝ)
    (T : ℕ)
    [IsFiniteMeasure μ]
    (hsub :
      ∀ t ∈ Finset.range T, ∀ x ∈ pool t,
        HasSubgaussianMGF (X t x) (c t x) μ)
    (hradius :
      ∀ t ∈ Finset.range T, ∀ x ∈ pool t,
        0 ≤ radius t x)
    (htail :
      ∀ t ∈ Finset.range T, ∀ x ∈ pool t,
        2 * Real.exp (-(radius t x) ^ 2 / (2 * (c t x : ℝ))) ≤ delta t x) :
    μ.real
        (RandomAdaptiveBadEventUpTo
          (fun t ω ↦
            posteriorCoefficientSelected
              (pool t) (sampler t) (score t) (threshold t) ω)
          (fun t x ↦ CenteredSubGaussianBadEvent (X t x) (radius t x)) T)
      ≤ ∑ t ∈ Finset.range T, ∑ x ∈ pool t, delta t x := by
  exact randomAdaptiveCenteredSubGaussian_bad_event_le_sum
    (μ := μ)
    (randomCandidates := fun t ω ↦
      posteriorCoefficientSelected
        (pool t) (sampler t) (score t) (threshold t) ω)
    (envelope := pool)
    (X := X)
    (c := c)
    (radius := radius)
    (delta := delta)
    (T := T)
    (fun ω t _ht ↦
      posteriorCoefficientSelected_subset_pool
        (pool t) (sampler t) (score t) (threshold t) ω)
    hsub
    hradius
    htail

end SCOLHKG.Measure
