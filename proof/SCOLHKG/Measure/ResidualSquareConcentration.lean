import Mathlib.Probability.Moments.SubGaussian
import SCOLHKG.Measure.SubGaussianConfidence

namespace SCOLHKG.Measure

open MeasureTheory ProbabilityTheory
open scoped MeasureTheory ProbabilityTheory BigOperators

/-!
Distribution constants for residual-square HVD concentration.

The ridge-HVD proof consumes a uniform residual-square concentration event.
This file gives a concrete way to obtain that event: bounded residual-square
observations are centered sub-Gaussian by Hoeffding's lemma, with parameter
`((upper - lower)/2)^2` in `NNReal` form.
-/

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

noncomputable def boundedResidualSquareConstant (lower upper : ℝ) : NNReal :=
  ((‖upper - lower‖₊ / 2) ^ 2 : NNReal)

noncomputable def ResidualSquareCenteredBadEvent
    (Z : Ω → ℝ)
    (radius : ℝ) : Set Ω :=
  CenteredSubGaussianBadEvent (fun ω ↦ Z ω - μ[Z]) radius

theorem boundedResidualSquare_centered_subGaussian
    {Z : Ω → ℝ}
    {lower upper : ℝ}
    [IsProbabilityMeasure μ]
    (hZ : AEMeasurable Z μ)
    (hBound : ∀ᵐ ω ∂μ, Z ω ∈ Set.Icc lower upper) :
    HasSubgaussianMGF
      (fun ω ↦ Z ω - μ[Z])
      (boundedResidualSquareConstant lower upper)
      μ := by
  simpa [boundedResidualSquareConstant] using
    (hasSubgaussianMGF_of_mem_Icc
      (μ := μ)
      (X := Z)
      (a := lower)
      (b := upper)
      hZ
      hBound)

theorem boundedResidualSquare_centered_bad_event_le
    {Z : Ω → ℝ}
    {lower upper radius : ℝ}
    [IsProbabilityMeasure μ]
    (hZ : AEMeasurable Z μ)
    (hBound : ∀ᵐ ω ∂μ, Z ω ∈ Set.Icc lower upper)
    (hradius : 0 ≤ radius) :
    μ.real (ResidualSquareCenteredBadEvent (μ := μ) Z radius)
      ≤
        2 * Real.exp
          (-radius ^ 2
            / (2 * (boundedResidualSquareConstant lower upper : ℝ))) := by
  exact centeredSubGaussian_abs_bad_event_le
    (μ := μ)
    (X := fun ω ↦ Z ω - μ[Z])
    (c := boundedResidualSquareConstant lower upper)
    (radius := radius)
    (boundedResidualSquare_centered_subGaussian
      (μ := μ)
      (Z := Z)
      (lower := lower)
      (upper := upper)
      hZ
      hBound)
    hradius

theorem boundedResidualSquare_finite_concentration
    {Param : Type*}
    (s : Finset Param)
    (Z : Param → Ω → ℝ)
    (lower upper radius delta : Param → ℝ)
    [IsProbabilityMeasure μ]
    (hZ : ∀ theta ∈ s, AEMeasurable (Z theta) μ)
    (hBound :
      ∀ theta ∈ s,
        ∀ᵐ ω ∂μ, Z theta ω ∈ Set.Icc (lower theta) (upper theta))
    (hradius : ∀ theta ∈ s, 0 ≤ radius theta)
    (htail :
      ∀ theta ∈ s,
        2 * Real.exp
          (-(radius theta) ^ 2
            / (2 * (boundedResidualSquareConstant
                (lower theta) (upper theta) : ℝ)))
          ≤ delta theta) :
    μ.real
        (⋃ theta ∈ s,
          ResidualSquareCenteredBadEvent (μ := μ) (Z theta) (radius theta))
      ≤ ∑ theta ∈ s, delta theta := by
  let bad : Param → Set Ω :=
    fun theta ↦ ResidualSquareCenteredBadEvent (μ := μ) (Z theta) (radius theta)
  have hEach : ∀ theta ∈ s, μ.real (bad theta) ≤ delta theta := by
    intro theta htheta
    exact (boundedResidualSquare_centered_bad_event_le
      (μ := μ)
      (Z := Z theta)
      (lower := lower theta)
      (upper := upper theta)
      (radius := radius theta)
      (hZ theta htheta)
      (hBound theta htheta)
      (hradius theta htheta)).trans
      (htail theta htheta)
  calc
    μ.real
        (⋃ theta ∈ s,
          ResidualSquareCenteredBadEvent (μ := μ) (Z theta) (radius theta))
      = μ.real (⋃ theta ∈ s, bad theta) := by
        rfl
    _ ≤ ∑ theta ∈ s, μ.real (bad theta) := by
        exact finite_real_union_bad_event_le_sum (μ := μ) s bad
    _ ≤ ∑ theta ∈ s, delta theta := by
      exact Finset.sum_le_sum (by
        intro theta htheta
        exact hEach theta htheta)

theorem outside_finite_residual_square_bad_event_uniform
    {Param : Type*}
    (s : Finset Param)
    (Z : Param → Ω → ℝ)
    (radius : Param → ℝ)
    {ω : Ω}
    (hOutside :
      ω ∉
        (⋃ theta ∈ s,
          ResidualSquareCenteredBadEvent (μ := μ) (Z theta) (radius theta))) :
    ∀ theta ∈ s,
      |Z theta ω - μ[Z theta]| < radius theta := by
  intro theta htheta
  by_contra hnot
  have hbad : radius theta ≤ |Z theta ω - μ[Z theta]| := le_of_not_gt hnot
  have hin :
      ω ∈
        (⋃ theta ∈ s,
          ResidualSquareCenteredBadEvent (μ := μ) (Z theta) (radius theta)) := by
    exact Set.mem_iUnion.2
      ⟨theta, Set.mem_iUnion.2
        ⟨htheta, hbad⟩⟩
  exact hOutside hin

end SCOLHKG.Measure
