import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
import SCOLHKG.Real.SafeRegret

namespace SCOLHKG.Measure

open MeasureTheory

/-!
High-probability safe-regret event transfer.

If all failures are contained in a bad event whose probability is controlled,
then the safe-regret conclusion holds outside a high-probability set.
-/

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

theorem failure_probability_le_of_subset_bad
    {bad failure : Set Ω}
    {delta : ℝ}
    (hsub : failure ⊆ bad)
    (hbad : μ bad ≤ ENNReal.ofReal delta) :
    μ failure ≤ ENNReal.ofReal delta := by
  exact (measure_mono hsub).trans hbad

def SafeRegretFailure
    {Design : Type*}
    (p : SCOLHKG.Real.ChanceOptimization Design)
    (xStar : Design)
    (eps : ℝ)
    (rec : Ω → Design) : Set Ω :=
  {ω | ¬ SCOLHKG.Real.SafeSimpleRegretBound p (rec ω) xStar eps}

theorem safe_regret_failure_le_bad
    {Design : Type*}
    {p : SCOLHKG.Real.ChanceOptimization Design}
    {xStar : Design}
    {eps delta : ℝ}
    {rec : Ω → Design}
    {bad : Set Ω}
    (hOutside :
      ∀ ω, ω ∉ bad →
        SCOLHKG.Real.SafeSimpleRegretBound p (rec ω) xStar eps)
    (hbad : μ bad ≤ ENNReal.ofReal delta) :
    μ (SafeRegretFailure p xStar eps rec)
      ≤ ENNReal.ofReal delta := by
  apply failure_probability_le_of_subset_bad (bad := bad)
  · intro ω hω
    unfold SafeRegretFailure at hω
    by_contra hnot
    exact hω (hOutside ω hnot)
  · exact hbad

end SCOLHKG.Measure
