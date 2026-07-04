import Mathlib.Probability.Moments.SubGaussian
import SCOLHKG.Measure.SubGaussianConfidence

namespace SCOLHKG.Measure

open MeasureTheory ProbabilityTheory
open scoped MeasureTheory ProbabilityTheory BigOperators

/-!
Finite-kernel GP posterior confidence.

For the implementation, posterior errors on a finite candidate set are linear
combinations of observation/simulation noise through kernel posterior weights.
This file proves the concrete sub-Gaussian parameter for that linear
combination: independent sub-Gaussian noises with parameters `c_i` give a
posterior error parameter `sum_i w_i(x)^2 c_i`.
-/

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

universe u v

def squareNNReal (w : ℝ) : NNReal :=
  ⟨w ^ 2, sq_nonneg w⟩

def finiteKernelPosteriorError
    {Index : Type u}
    {Design : Type v}
    (active : Finset Index)
    (weight : Design → Index → ℝ)
    (noise : Index → Ω → ℝ)
    (x : Design) : Ω → ℝ :=
  fun ω ↦ ∑ i ∈ active, weight x i * noise i ω

def finiteKernelSubGaussianParam
    {Index : Type u}
    {Design : Type v}
    (active : Finset Index)
    (weight : Design → Index → ℝ)
    (c : Index → NNReal)
    (x : Design) : NNReal :=
  ∑ i ∈ active, squareNNReal (weight x i) * c i

theorem finiteKernelPosteriorError_subGaussian
    {Index : Type u}
    {Design : Type v}
    (active : Finset Index)
    (weight : Design → Index → ℝ)
    (noise : Index → Ω → ℝ)
    (c : Index → NNReal)
    (x : Design)
    (hindep : iIndepFun noise μ)
    (hnoise : ∀ i ∈ active, HasSubgaussianMGF (noise i) (c i) μ) :
    HasSubgaussianMGF
      (finiteKernelPosteriorError active weight noise x)
      (finiteKernelSubGaussianParam active weight c x)
      μ := by
  let scaledNoise : Index → Ω → ℝ :=
    fun i ω ↦ weight x i * noise i ω
  have hindepScaled : iIndepFun scaledNoise μ := by
    have h := hindep.comp
      (fun i y ↦ weight x i * y)
      (by
        intro i
        fun_prop)
    simpa [scaledNoise, Function.comp_def] using h
  have hsubScaled :
      ∀ i ∈ active,
        HasSubgaussianMGF
          (scaledNoise i)
          (squareNNReal (weight x i) * c i)
          μ := by
    intro i hi
    simpa [scaledNoise, squareNNReal] using
      (hnoise i hi).const_mul (weight x i)
  have hsum :=
    HasSubgaussianMGF.sum_of_iIndepFun
      (X := scaledNoise)
      (c := fun i ↦
        squareNNReal (weight x i) * c i)
      (s := active)
      hindepScaled
      hsubScaled
  change
    HasSubgaussianMGF
      (fun ω ↦ ∑ i ∈ active, weight x i * noise i ω)
      (∑ i ∈ active, squareNNReal (weight x i) * c i)
      μ
  simpa [scaledNoise] using hsum

theorem finiteKernelPosteriorError_centered_confidence
    {Index : Type u}
    {Design : Type v}
    (active : Finset Index)
    (candidates : Finset Design)
    (weight : Design → Index → ℝ)
    (noise : Index → Ω → ℝ)
    (c : Index → NNReal)
    (radius delta : Design → ℝ)
    [IsFiniteMeasure μ]
    (hindep : iIndepFun noise μ)
    (hnoise : ∀ i ∈ active, HasSubgaussianMGF (noise i) (c i) μ)
    (hradius : ∀ x ∈ candidates, 0 ≤ radius x)
    (htail :
      ∀ x ∈ candidates,
        2 * Real.exp
          (-(radius x) ^ 2
            / (2 * (finiteKernelSubGaussianParam active weight c x : ℝ)))
          ≤ delta x) :
    μ.real
        (⋃ x ∈ candidates,
          CenteredSubGaussianBadEvent
            (finiteKernelPosteriorError active weight noise x)
            (radius x))
      ≤ ∑ x ∈ candidates, delta x := by
  exact centeredSubGaussian_finite_candidate_bad_event_le_sum
    (μ := μ)
    (s := candidates)
    (X := fun x ↦ finiteKernelPosteriorError active weight noise x)
    (c := fun x ↦ finiteKernelSubGaussianParam active weight c x)
    (radius := radius)
    (delta := delta)
    (fun x _hx ↦
      finiteKernelPosteriorError_subGaussian
        (μ := μ)
        active weight noise c x hindep hnoise)
    hradius
    htail

theorem adaptiveFiniteKernelPosteriorError_centered_confidence
    {Index : Type u}
    {Design : Type v}
    (active : Finset Index)
    (candidates : ℕ → Finset Design)
    (weight : ℕ → Design → Index → ℝ)
    (noise : Index → Ω → ℝ)
    (c : Index → NNReal)
    (radius delta : ℕ → Design → ℝ)
    (T : ℕ)
    [IsFiniteMeasure μ]
    (hindep : iIndepFun noise μ)
    (hnoise : ∀ i ∈ active, HasSubgaussianMGF (noise i) (c i) μ)
    (hradius :
      ∀ t ∈ Finset.range T, ∀ x ∈ candidates t,
        0 ≤ radius t x)
    (htail :
      ∀ t ∈ Finset.range T, ∀ x ∈ candidates t,
        2 * Real.exp
          (-(radius t x) ^ 2
            / (2 * (finiteKernelSubGaussianParam active (weight t) c x : ℝ)))
          ≤ delta t x) :
    μ.real
        (AdaptiveBadEventUpTo candidates
          (fun t x ↦
            CenteredSubGaussianBadEvent
              (finiteKernelPosteriorError active (weight t) noise x)
              (radius t x))
          T)
      ≤ ∑ t ∈ Finset.range T, ∑ x ∈ candidates t, delta t x := by
  exact adaptive_centeredSubGaussian_bad_event_le_sum
    (μ := μ)
    (candidates := candidates)
    (X := fun t x ↦ finiteKernelPosteriorError active (weight t) noise x)
    (c := fun t x ↦ finiteKernelSubGaussianParam active (weight t) c x)
    (radius := radius)
    (delta := delta)
    (T := T)
    (fun t _ht x _hx ↦
      finiteKernelPosteriorError_subGaussian
        (μ := μ)
        active (weight t) noise c x hindep hnoise)
    hradius
    htail

end SCOLHKG.Measure
