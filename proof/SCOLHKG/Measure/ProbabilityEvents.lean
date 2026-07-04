import Mathlib.Probability.CondVar
import Mathlib.Probability.Moments.Variance
import SCOLHKG.Real.Certification
import SCOLHKG.Real.HVD

namespace SCOLHKG.Measure

open MeasureTheory ProbabilityTheory
open scoped MeasureTheory ProbabilityTheory ENNReal BigOperators

/-!
Measure-theoretic probability events for SC-OLH-KG.

This file connects the real-valued deterministic layer to mathlib's probability
library.  It uses mathlib's conditional variance and Chebyshev inequality to
derive high-probability GP-confidence and residual-square concentration events.
-/

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

theorem law_total_variance
    {m : MeasurableSpace Ω}
    {hm : m ≤ mΩ}
    {X : Ω → ℝ}
    [IsProbabilityMeasure μ]
    (hX : MemLp X 2 μ) :
    μ[Var[X; μ | m]] + Var[μ[X | m]; μ] = Var[X; μ] := by
  exact ProbabilityTheory.integral_condVar_add_variance_condExp
    (m := m) (μ := μ) hm hX

theorem chebyshev_bad_event_le
    {X : Ω → ℝ}
    [IsFiniteMeasure μ]
    (hX : MemLp X 2 μ)
    {radius : ℝ}
    (hradius : 0 < radius) :
    μ {ω | radius ≤ |X ω - μ[X]|} ≤
      ENNReal.ofReal (Var[X; μ] / radius ^ 2) := by
  exact ProbabilityTheory.meas_ge_le_variance_div_sq hX hradius

theorem chebyshev_bad_event_le_of_ratio
    {X : Ω → ℝ}
    [IsFiniteMeasure μ]
    (hX : MemLp X 2 μ)
    {radius delta : ℝ}
    (hradius : 0 < radius)
    (hratio : Var[X; μ] / radius ^ 2 ≤ delta) :
    μ {ω | radius ≤ |X ω - μ[X]|} ≤ ENNReal.ofReal delta := by
  exact (chebyshev_bad_event_le (μ := μ) hX hradius).trans
    (ENNReal.ofReal_le_ofReal hratio)

def GPBadEvent (X : Ω → ℝ) (radius : ℝ) : Set Ω :=
  {ω | radius ≤ |X ω - μ[X]|}

def GPGoodEvent (X : Ω → ℝ) (radius : ℝ) : Set Ω :=
  {ω | |X ω - μ[X]| < radius}

theorem gp_bad_event_probability_le
    {X : Ω → ℝ}
    [IsFiniteMeasure μ]
    (hX : MemLp X 2 μ)
    {radius delta : ℝ}
    (hradius : 0 < radius)
    (hratio : Var[X; μ] / radius ^ 2 ≤ delta) :
    μ (GPBadEvent (μ := μ) X radius) ≤ ENNReal.ofReal delta := by
  unfold GPBadEvent
  exact chebyshev_bad_event_le_of_ratio (μ := μ) hX hradius hratio

theorem gp_good_event_pointwise_confidence
    {X : Ω → ℝ}
    {radius : ℝ}
    {ω : Ω}
    (hgood : ω ∈ GPGoodEvent (μ := μ) X radius) :
    X ω ≤ μ[X] + radius ∧ μ[X] ≤ X ω + radius := by
  unfold GPGoodEvent at hgood
  have hlt : |X ω - μ[X]| < radius := hgood
  have hle : |X ω - μ[X]| ≤ radius := le_of_lt hlt
  constructor
  · have h1 : X ω - μ[X] ≤ radius := (le_abs_self _).trans hle
    linarith
  · have h2 : -(X ω - μ[X]) ≤ radius := (neg_le_abs _).trans hle
    linarith

theorem finite_union_bad_event_le_sum
    {ι : Type*}
    (s : Finset ι)
    (bad : ι → Set Ω) :
    μ (⋃ i ∈ s, bad i) ≤ ∑ i ∈ s, μ (bad i) := by
  exact measure_biUnion_finset_le s bad

theorem gp_finite_candidate_bad_event_le_sum
    {ι : Type*}
    (s : Finset ι)
    (X : ι → Ω → ℝ)
    (radius delta : ι → ℝ)
    [IsFiniteMeasure μ]
    (hX : ∀ i ∈ s, MemLp (X i) 2 μ)
    (hradius : ∀ i ∈ s, 0 < radius i)
    (hratio : ∀ i ∈ s, Var[X i; μ] / radius i ^ 2 ≤ delta i) :
    μ (⋃ i ∈ s, GPBadEvent (μ := μ) (X i) (radius i))
      ≤ ∑ i ∈ s, ENNReal.ofReal (delta i) := by
  calc
    μ (⋃ i ∈ s, GPBadEvent (μ := μ) (X i) (radius i))
      ≤ ∑ i ∈ s, μ (GPBadEvent (μ := μ) (X i) (radius i)) := by
        exact finite_union_bad_event_le_sum (μ := μ) s
          (fun i ↦ GPBadEvent (μ := μ) (X i) (radius i))
    _ ≤ ∑ i ∈ s, ENNReal.ofReal (delta i) := by
      exact Finset.sum_le_sum (by
        intro i hi
        exact gp_bad_event_probability_le
          (μ := μ)
          (X := X i)
          (hX i hi)
          (hradius i hi)
          (hratio i hi))

def ResidualSquareBadEvent (estimator centeredAt : Ω → ℝ) (radius : ℝ) : Set Ω :=
  {ω | radius ≤ |estimator ω - centeredAt ω|}

theorem residual_square_bad_event_le_of_chebyshev
    {estimator centeredAt : Ω → ℝ}
    [IsFiniteMeasure μ]
    (hZ : MemLp (fun ω ↦ estimator ω - centeredAt ω) 2 μ)
    {radius delta : ℝ}
    (hradius : 0 < radius)
    (hcenter :
      μ[fun ω ↦ estimator ω - centeredAt ω] = 0)
    (hratio :
      Var[fun ω ↦ estimator ω - centeredAt ω; μ] / radius ^ 2 ≤ delta) :
    μ (ResidualSquareBadEvent estimator centeredAt radius) ≤ ENNReal.ofReal delta := by
  unfold ResidualSquareBadEvent
  simpa [hcenter] using
    (chebyshev_bad_event_le_of_ratio
      (μ := μ)
      (X := fun ω ↦ estimator ω - centeredAt ω)
      hZ
      hradius
      hratio)

theorem residual_concentration_event_yields_hvd_event
    {estimationError complexity concentrationRadius : ℝ}
    (h :
      estimationError ≤ complexity + concentrationRadius) :
    SCOLHKG.Real.ResidualSquareConcentration.Valid
      {
        estimationError := estimationError
        complexity := complexity
        concentrationRadius := concentrationRadius
      } := by
  exact h

end SCOLHKG.Measure
