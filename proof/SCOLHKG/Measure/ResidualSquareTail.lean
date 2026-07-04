import SCOLHKG.Measure.ResidualSquareConcentration

namespace SCOLHKG.Measure

open MeasureTheory ProbabilityTheory
open scoped MeasureTheory ProbabilityTheory BigOperators

/-!
Sharper residual-square tail interface.

The bounded Hoeffding result is one usable concentration theorem.  If the final
manuscript chooses a sharper Gaussian-square or sub-exponential residual-square
model, it can instantiate this tail interface directly.
-/

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

def ResidualSquareTailBound
    (Z : Ω → ℝ)
    (tail : ℝ → ℝ)
    (radius : ℝ) : Prop :=
  μ.real (ResidualSquareCenteredBadEvent (μ := μ) Z radius) ≤ tail radius

noncomputable def subExponentialResidualSquareTail
    (nu b radius : ℝ) : ℝ :=
  2 * Real.exp (-min (radius ^ 2 / (2 * nu ^ 2)) (radius / (2 * b)))

noncomputable def subExponentialResidualSquareRadius
    (nu b delta : ℝ) : ℝ :=
  max
    (Real.sqrt (2 * nu ^ 2 * Real.log (2 / delta)))
    (2 * b * Real.log (2 / delta))

def HasSubExponentialResidualSquareTail
    (Z : Ω → ℝ)
    (nu b : ℝ) : Prop :=
  ∀ radius, 0 ≤ radius →
    ResidualSquareTailBound (μ := μ) Z
      (subExponentialResidualSquareTail nu b)
      radius

theorem residualSquare_finite_concentration_from_tail
    {Param : Type*}
    (s : Finset Param)
    (Z : Param → Ω → ℝ)
    (radius delta : Param → ℝ)
    (tail : Param → ℝ → ℝ)
    (hTail :
      ∀ theta ∈ s,
        ResidualSquareTailBound (μ := μ) (Z theta) (tail theta) (radius theta))
    (hDelta :
      ∀ theta ∈ s, tail theta (radius theta) ≤ delta theta) :
    μ.real
        (⋃ theta ∈ s,
          ResidualSquareCenteredBadEvent (μ := μ) (Z theta) (radius theta))
      ≤ ∑ theta ∈ s, delta theta := by
  let bad : Param → Set Ω :=
    fun theta ↦ ResidualSquareCenteredBadEvent (μ := μ) (Z theta) (radius theta)
  have hEach : ∀ theta ∈ s, μ.real (bad theta) ≤ delta theta := by
    intro theta htheta
    exact (hTail theta htheta).trans (hDelta theta htheta)
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

theorem residualSquare_finite_concentration_from_subExponential_tail
    {Param : Type*}
    (s : Finset Param)
    (Z : Param → Ω → ℝ)
    (nu b radius delta : Param → ℝ)
    (hradius : ∀ theta ∈ s, 0 ≤ radius theta)
    (hTail :
      ∀ theta ∈ s,
        HasSubExponentialResidualSquareTail
          (μ := μ) (Z theta) (nu theta) (b theta))
    (hDelta :
      ∀ theta ∈ s,
        subExponentialResidualSquareTail
          (nu theta) (b theta) (radius theta) ≤ delta theta) :
    μ.real
        (⋃ theta ∈ s,
          ResidualSquareCenteredBadEvent (μ := μ) (Z theta) (radius theta))
      ≤ ∑ theta ∈ s, delta theta := by
  exact residualSquare_finite_concentration_from_tail
    (μ := μ)
    s
    Z
    radius
    delta
    (fun theta ↦ subExponentialResidualSquareTail (nu theta) (b theta))
    (fun theta htheta ↦ hTail theta htheta (radius theta) (hradius theta htheta))
    hDelta

theorem residualSquare_finite_concentration_from_subExponential_default_radius
    {Param : Type*}
    (s : Finset Param)
    (Z : Param → Ω → ℝ)
    (nu b delta : Param → ℝ)
    (hradius :
      ∀ theta ∈ s,
        0 ≤ subExponentialResidualSquareRadius
          (nu theta) (b theta) (delta theta))
    (hTail :
      ∀ theta ∈ s,
        HasSubExponentialResidualSquareTail
          (μ := μ) (Z theta) (nu theta) (b theta))
    (hDelta :
      ∀ theta ∈ s,
        subExponentialResidualSquareTail
          (nu theta) (b theta)
          (subExponentialResidualSquareRadius
            (nu theta) (b theta) (delta theta))
          ≤ delta theta) :
    μ.real
        (⋃ theta ∈ s,
          ResidualSquareCenteredBadEvent (μ := μ) (Z theta)
            (subExponentialResidualSquareRadius
              (nu theta) (b theta) (delta theta)))
      ≤ ∑ theta ∈ s, delta theta := by
  exact residualSquare_finite_concentration_from_subExponential_tail
    (μ := μ)
    s
    Z
    nu
    b
    (fun theta ↦ subExponentialResidualSquareRadius
      (nu theta) (b theta) (delta theta))
    delta
    hradius
    hTail
    hDelta

end SCOLHKG.Measure
