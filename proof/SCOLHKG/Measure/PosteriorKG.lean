import Mathlib.Probability.Moments.Basic
import SCOLHKG.Real.KG

namespace SCOLHKG.Measure

open MeasureTheory
open scoped MeasureTheory

noncomputable section

/-!
Posterior expected terminal gain.

This is the measure-theoretic source of exact KG: expected terminal gains are
Bochner integrals under the posterior predictive measure.
-/

universe u v

structure PosteriorKG (Design : Type u) (Ω : Type v) [MeasurableSpace Ω] where
  terminalGain : Design → Ω → ℝ

def posteriorExpectedGain
    {Design : Type u}
    {Ω : Type v}
    [MeasurableSpace Ω]
    (kg : PosteriorKG Design Ω)
    (μ : Measure Ω)
    (x : Design) : ℝ :=
  ∫ ω, kg.terminalGain x ω ∂μ

def toExactKG
    {Design : Type u}
    {Ω : Type v}
    [MeasurableSpace Ω]
    (kg : PosteriorKG Design Ω)
    (μ : Measure Ω) : SCOLHKG.Real.ExactKG Design :=
  {
    expectedTerminalGain := posteriorExpectedGain kg μ
  }

theorem posterior_expected_gain_is_exact_gain
    {Design : Type u}
    {Ω : Type v}
    [MeasurableSpace Ω]
    (kg : PosteriorKG Design Ω)
    (μ : Measure Ω)
    (x : Design) :
    (toExactKG kg μ).expectedTerminalGain x =
      ∫ ω, kg.terminalGain x ω ∂μ := by
  rfl

theorem posterior_kg_maximizer_optimal
    {Design : Type u}
    {Ω : Type v}
    [MeasurableSpace Ω]
    (kg : PosteriorKG Design Ω)
    (μ : Measure Ω)
    (x y : Design)
    (hMax :
      ∀ z,
        posteriorExpectedGain kg μ z ≤ posteriorExpectedGain kg μ x) :
    posteriorExpectedGain kg μ y ≤ posteriorExpectedGain kg μ x := by
  exact hMax y

theorem posterior_kg_maximizer_is_exact_kg_maximizer
    {Design : Type u}
    {Ω : Type v}
    [MeasurableSpace Ω]
    (kg : PosteriorKG Design Ω)
    (μ : Measure Ω)
    (x : Design)
    (hMax :
      ∀ z,
        posteriorExpectedGain kg μ z ≤ posteriorExpectedGain kg μ x) :
    SCOLHKG.Real.KGMaximizer (toExactKG kg μ) x := by
  intro y
  exact hMax y

end

end SCOLHKG.Measure
