import Mathlib.Probability.Moments.Basic
import SCOLHKG.Measure.PosteriorKG
import SCOLHKG.Real.AdditiveApproxKG

namespace SCOLHKG.Measure

open MeasureTheory
open scoped MeasureTheory

noncomputable section

/-!
Exact posterior-update SC-OLH-KG value theorem.

The exact one-step acquisition is the posterior expectation of the terminal
certified value improvement after applying the Bayesian/HVD update induced by
sampling a design.  This file keeps the update equation abstract but explicit:
`update state x observation`.
-/

universe u v w

structure PosteriorUpdateKG
    (State : Type u)
    (Design : Type v)
    (Ω : Type w)
    [MeasurableSpace Ω] where
  currentValue : State → ℝ
  terminalValue : State → ℝ
  update : State → Design → Ω → State

def posteriorUpdateGain
    {State : Type u}
    {Design : Type v}
    {Ω : Type w}
    [MeasurableSpace Ω]
    (kg : PosteriorUpdateKG State Design Ω)
    (state : State)
    (x : Design) : Ω → ℝ :=
  fun ω ↦ kg.currentValue state - kg.terminalValue (kg.update state x ω)

def posteriorUpdateExpectedGain
    {State : Type u}
    {Design : Type v}
    {Ω : Type w}
    [MeasurableSpace Ω]
    (kg : PosteriorUpdateKG State Design Ω)
    (μ : Measure Ω)
    (state : State)
    (x : Design) : ℝ :=
  ∫ ω, posteriorUpdateGain kg state x ω ∂μ

def posteriorUpdateToPosteriorKG
    {State : Type u}
    {Design : Type v}
    {Ω : Type w}
    [MeasurableSpace Ω]
    (kg : PosteriorUpdateKG State Design Ω)
    (state : State) : PosteriorKG Design Ω :=
  {
    terminalGain := posteriorUpdateGain kg state
  }

def posteriorUpdateToExactKG
    {State : Type u}
    {Design : Type v}
    {Ω : Type w}
    [MeasurableSpace Ω]
    (kg : PosteriorUpdateKG State Design Ω)
    (μ : Measure Ω)
    (state : State) : SCOLHKG.Real.ExactKG Design :=
  {
    expectedTerminalGain := posteriorUpdateExpectedGain kg μ state
  }

theorem posterior_update_expected_gain_is_integral
    {State : Type u}
    {Design : Type v}
    {Ω : Type w}
    [MeasurableSpace Ω]
    (kg : PosteriorUpdateKG State Design Ω)
    (μ : Measure Ω)
    (state : State)
    (x : Design) :
    posteriorUpdateExpectedGain kg μ state x =
      ∫ ω, kg.currentValue state
        - kg.terminalValue (kg.update state x ω) ∂μ := by
  rfl

theorem posterior_update_exact_kg_matches_posterior_kg
    {State : Type u}
    {Design : Type v}
    {Ω : Type w}
    [MeasurableSpace Ω]
    (kg : PosteriorUpdateKG State Design Ω)
    (μ : Measure Ω)
    (state : State) :
    posteriorUpdateToExactKG kg μ state =
      toExactKG (posteriorUpdateToPosteriorKG kg state) μ := by
  rfl

theorem posterior_update_kg_maximizer_optimal
    {State : Type u}
    {Design : Type v}
    {Ω : Type w}
    [MeasurableSpace Ω]
    (kg : PosteriorUpdateKG State Design Ω)
    (μ : Measure Ω)
    (state : State)
    (x y : Design)
    (hMax :
      ∀ z,
        posteriorUpdateExpectedGain kg μ state z
          ≤ posteriorUpdateExpectedGain kg μ state x) :
    posteriorUpdateExpectedGain kg μ state y
      ≤ posteriorUpdateExpectedGain kg μ state x := by
  exact hMax y

theorem posterior_update_kg_maximizer_is_exact_kg_maximizer
    {State : Type u}
    {Design : Type v}
    {Ω : Type w}
    [MeasurableSpace Ω]
    (kg : PosteriorUpdateKG State Design Ω)
    (μ : Measure Ω)
    (state : State)
    (x : Design)
    (hMax :
      ∀ z,
        posteriorUpdateExpectedGain kg μ state z
          ≤ posteriorUpdateExpectedGain kg μ state x) :
    SCOLHKG.Real.KGMaximizer
      (posteriorUpdateToExactKG kg μ state)
      x := by
  intro y
  exact hMax y

theorem additive_posterior_update_proxy_gap
    {State : Type u}
    {Design : Type v}
    {Ω : Type w}
    [MeasurableSpace Ω]
    (kg : PosteriorUpdateKG State Design Ω)
    (μ : Measure Ω)
    (state : State)
    (proxy : Design → ℝ)
    {eta : ℝ}
    {x : Design}
    (hApprox :
      SCOLHKG.Real.UniformKGApprox
        (posteriorUpdateExpectedGain kg μ state)
        proxy
        eta)
    (hMax : ∀ y, proxy y ≤ proxy x) :
    ∀ y,
      posteriorUpdateExpectedGain kg μ state y
        ≤ posteriorUpdateExpectedGain kg μ state x + 2 * eta := by
  exact SCOLHKG.Real.proxy_maximizer_exact_gap_le_two_eta
    (exact := posteriorUpdateExpectedGain kg μ state)
    (proxy := proxy)
    (eta := eta)
    (x := x)
    hApprox
    hMax

end

end SCOLHKG.Measure
