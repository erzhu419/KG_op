import SCOLHKG.Measure.SubGaussianConfidence
import SCOLHKG.Real.ExactKGImplementation

namespace SCOLHKG.Measure

open MeasureTheory ProbabilityTheory
open scoped MeasureTheory ProbabilityTheory BigOperators

/-!
Concentration bridge for the exact-MC posterior-update KG estimator.

The implementation evaluates a finite candidate pool.  If the Monte-Carlo
estimator error for each candidate is centered sub-Gaussian, then a finite
union bound gives a uniform-error event.  On that event, the deterministic
`ExactMCEstimator` bridge in `Real/ExactKGImplementation.lean` yields the
same `2 eta` maximizer gap.
-/

variable {Ω Design : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

def ExactMCUniformErrorEvent
    (candidates : Finset Design)
    (exact : Design → ℝ)
    (estimate : Ω → Design → ℝ)
    (eta : ℝ) : Set Ω :=
  {ω | ∀ x ∈ candidates, |estimate ω x - exact x| ≤ eta}

def ExactMCBadEvent
    (candidates : Finset Design)
    (exact : Design → ℝ)
    (estimate : Ω → Design → ℝ)
    (eta : ℝ) : Set Ω :=
  ⋃ x ∈ candidates,
    CenteredSubGaussianBadEvent
      (fun ω ↦ estimate ω x - exact x)
      eta

theorem exactMC_uniform_error_of_not_bad
    (candidates : Finset Design)
    (exact : Design → ℝ)
    (estimate : Ω → Design → ℝ)
    {eta : ℝ}
    {ω : Ω}
    (hnot : ω ∉ ExactMCBadEvent candidates exact estimate eta) :
    ω ∈ ExactMCUniformErrorEvent candidates exact estimate eta := by
  intro x hx
  by_contra hgt
  have heta : eta ≤ |estimate ω x - exact x| := le_of_lt (not_le.mp hgt)
  have hmem :
      ω ∈ CenteredSubGaussianBadEvent
        (fun ω ↦ estimate ω x - exact x) eta := by
    exact heta
  exact hnot (by
    unfold ExactMCBadEvent
    exact Set.mem_iUnion₂.mpr ⟨x, hx, hmem⟩)

theorem exactMC_uniform_error_implies_estimator
    (candidates : Finset Design)
    (exact : Design → ℝ)
    (estimate : Ω → Design → ℝ)
    {eta : ℝ}
    {ω : Ω}
    (hevent : ω ∈ ExactMCUniformErrorEvent candidates exact estimate eta)
    (heta : 0 ≤ eta)
    (houtside : ∀ x, x ∉ candidates → estimate ω x = exact x) :
    SCOLHKG.Real.ExactMCEstimator exact (estimate ω) eta := by
  intro x
  by_cases hx : x ∈ candidates
  · simpa [abs_sub_comm] using hevent x hx
  · simp [houtside x hx]
    exact heta

theorem exactMC_bad_event_le_sum
    (candidates : Finset Design)
    (exact : Design → ℝ)
    (estimate : Ω → Design → ℝ)
    (c : Design → NNReal)
    (eta delta : Design → ℝ)
    [IsFiniteMeasure μ]
    (hsub :
      ∀ x ∈ candidates,
        HasSubgaussianMGF
          (fun ω ↦ estimate ω x - exact x)
          (c x)
          μ)
    (heta : ∀ x ∈ candidates, 0 ≤ eta x)
    (htail :
      ∀ x ∈ candidates,
        2 * Real.exp (-(eta x) ^ 2 / (2 * (c x : ℝ))) ≤ delta x) :
    μ.real
        (⋃ x ∈ candidates,
          CenteredSubGaussianBadEvent
            (fun ω ↦ estimate ω x - exact x)
            (eta x))
      ≤ ∑ x ∈ candidates, delta x := by
  exact centeredSubGaussian_finite_candidate_bad_event_le_sum
    (μ := μ)
    (s := candidates)
    (X := fun x ω ↦ estimate ω x - exact x)
    (c := c)
    (radius := eta)
    (delta := delta)
    hsub
    heta
    htail

theorem exactMC_constant_radius_bad_event_le_sum
    (candidates : Finset Design)
    (exact : Design → ℝ)
    (estimate : Ω → Design → ℝ)
    (c : Design → NNReal)
    {eta : ℝ}
    (delta : Design → ℝ)
    [IsFiniteMeasure μ]
    (hsub :
      ∀ x ∈ candidates,
        HasSubgaussianMGF
          (fun ω ↦ estimate ω x - exact x)
          (c x)
          μ)
    (heta : 0 ≤ eta)
    (htail :
      ∀ x ∈ candidates,
        2 * Real.exp (-eta ^ 2 / (2 * (c x : ℝ))) ≤ delta x) :
    μ.real (ExactMCBadEvent candidates exact estimate eta)
      ≤ ∑ x ∈ candidates, delta x := by
  unfold ExactMCBadEvent
  exact exactMC_bad_event_le_sum
    (μ := μ)
    candidates
    exact
    estimate
    c
    (fun _ ↦ eta)
    delta
    hsub
    (fun x hx ↦ heta)
    (fun x hx ↦ by simpa using htail x hx)

end SCOLHKG.Measure
