import Mathlib

namespace SCOLHKG.Measure

open MeasureTheory

/-!
Exact finite-product bridge for independent frozen proposal draws.

For `n0` IID draws from a probability measure, the product-measure mass of
missing one measurable feasible set in every coordinate is exactly the
one-draw miss mass raised to `n0`.
-/

theorem iid_all_miss_probability
    {X : Type*} [MeasurableSpace X]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (feasible : Set X)
    (n0 : ℕ) :
    Measure.pi (fun _ : Fin n0 => μ)
      (Set.univ.pi (fun _ : Fin n0 => feasibleᶜ)) =
        (μ feasibleᶜ) ^ n0 := by
  rw [Measure.pi_pi]
  simp

theorem iid_at_least_one_hit_probability
    {X : Type*} [MeasurableSpace X]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (feasible : Set X)
    (hFeasible : MeasurableSet feasible)
    (n0 : ℕ) :
    Measure.pi (fun _ : Fin n0 => μ)
      (Set.univ.pi (fun _ : Fin n0 => feasibleᶜ))ᶜ =
        1 - (μ feasibleᶜ) ^ n0 := by
  have hMissMeasurable :
      MeasurableSet (Set.univ.pi (fun _ : Fin n0 => feasibleᶜ)) :=
    MeasurableSet.pi (Set.to_countable Set.univ) (fun _ _ => hFeasible.compl)
  rw [measure_compl hMissMeasurable (measure_ne_top _ _)]
  rw [measure_univ, iid_all_miss_probability μ feasible n0]

end SCOLHKG.Measure
