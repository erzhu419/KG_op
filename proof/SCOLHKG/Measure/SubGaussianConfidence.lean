import Mathlib.MeasureTheory.Measure.Real
import Mathlib.Probability.Moments.SubGaussian
import SCOLHKG.Measure.ProbabilityEvents

namespace SCOLHKG.Measure

open MeasureTheory ProbabilityTheory
open scoped MeasureTheory ProbabilityTheory ENNReal BigOperators

/-!
Sub-Gaussian confidence events for finite and adaptive candidate sets.

This is the first measure-theoretic bridge from a model-specific posterior
assumption to the deterministic certification layer: once the posterior error
at a candidate is sub-Gaussian, mathlib's Chernoff bound controls the bad event.
Finite and adaptive candidate sets are then handled by finite union bounds.
-/

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

def SubGaussianBadEvent (X : Ω → ℝ) (radius : ℝ) : Set Ω :=
  {ω | radius ≤ X ω}

theorem subGaussian_right_tail_le
    {X : Ω → ℝ}
    {c : NNReal}
    {radius : ℝ}
    (hsub : HasSubgaussianMGF X c μ)
    (hradius : 0 ≤ radius) :
    μ.real (SubGaussianBadEvent X radius) ≤
      Real.exp (-radius ^ 2 / (2 * (c : ℝ))) := by
  unfold SubGaussianBadEvent
  exact hsub.measure_ge_le hradius

def CenteredSubGaussianBadEvent (X : Ω → ℝ) (radius : ℝ) : Set Ω :=
  {ω | radius ≤ |X ω|}

theorem centeredSubGaussian_abs_bad_event_le
    {X : Ω → ℝ}
    {c : NNReal}
    {radius : ℝ}
    [IsFiniteMeasure μ]
    (hsub : HasSubgaussianMGF X c μ)
    (hradius : 0 ≤ radius) :
    μ.real (CenteredSubGaussianBadEvent X radius) ≤
      2 * Real.exp (-radius ^ 2 / (2 * (c : ℝ))) := by
  have hSubset :
      CenteredSubGaussianBadEvent X radius
        ⊆ SubGaussianBadEvent X radius
          ∪ SubGaussianBadEvent (fun ω ↦ -X ω) radius := by
    intro ω hω
    unfold CenteredSubGaussianBadEvent at hω
    by_cases hx : 0 ≤ X ω
    · left
      unfold SubGaussianBadEvent
      simpa [abs_of_nonneg hx] using hω
    · right
      unfold SubGaussianBadEvent
      have hxlt : X ω < 0 := lt_of_not_ge hx
      simpa [abs_of_neg hxlt] using hω
  calc
    μ.real (CenteredSubGaussianBadEvent X radius)
      ≤ μ.real
          (SubGaussianBadEvent X radius
            ∪ SubGaussianBadEvent (fun ω ↦ -X ω) radius) := by
        exact measureReal_mono hSubset
    _ ≤ μ.real (SubGaussianBadEvent X radius)
          + μ.real (SubGaussianBadEvent (fun ω ↦ -X ω) radius) := by
        exact measureReal_union_le _ _
    _ ≤ Real.exp (-radius ^ 2 / (2 * (c : ℝ)))
          + Real.exp (-radius ^ 2 / (2 * (c : ℝ))) := by
        gcongr
        · exact subGaussian_right_tail_le
            (μ := μ) (X := X) (c := c) (radius := radius) hsub hradius
        · exact subGaussian_right_tail_le
            (μ := μ) (X := fun ω ↦ -X ω) (c := c) (radius := radius)
            hsub.neg hradius
    _ = 2 * Real.exp (-radius ^ 2 / (2 * (c : ℝ))) := by
        ring

theorem finite_real_union_bad_event_le_sum
    {ι : Type*}
    (s : Finset ι)
    (bad : ι → Set Ω) :
    μ.real (⋃ i ∈ s, bad i) ≤ ∑ i ∈ s, μ.real (bad i) := by
  exact measureReal_biUnion_finset_le (μ := μ) s bad

theorem subGaussian_finite_candidate_bad_event_le_sum
    {ι : Type*}
    (s : Finset ι)
    (X : ι → Ω → ℝ)
    (c : ι → NNReal)
    (radius delta : ι → ℝ)
    (hsub : ∀ i ∈ s, HasSubgaussianMGF (X i) (c i) μ)
    (hradius : ∀ i ∈ s, 0 ≤ radius i)
    (htail :
      ∀ i ∈ s,
        Real.exp (-(radius i) ^ 2 / (2 * (c i : ℝ))) ≤ delta i) :
    μ.real (⋃ i ∈ s, SubGaussianBadEvent (X i) (radius i))
      ≤ ∑ i ∈ s, delta i := by
  calc
    μ.real (⋃ i ∈ s, SubGaussianBadEvent (X i) (radius i))
      ≤ ∑ i ∈ s, μ.real (SubGaussianBadEvent (X i) (radius i)) := by
        exact finite_real_union_bad_event_le_sum (μ := μ) s
          (fun i ↦ SubGaussianBadEvent (X i) (radius i))
    _ ≤ ∑ i ∈ s, delta i := by
      exact Finset.sum_le_sum (by
        intro i hi
        exact (subGaussian_right_tail_le
          (μ := μ)
          (X := X i)
          (c := c i)
          (radius := radius i)
          (hsub i hi)
          (hradius i hi)).trans (htail i hi))

theorem centeredSubGaussian_finite_candidate_bad_event_le_sum
    {ι : Type*}
    (s : Finset ι)
    (X : ι → Ω → ℝ)
    (c : ι → NNReal)
    (radius delta : ι → ℝ)
    [IsFiniteMeasure μ]
    (hsub : ∀ i ∈ s, HasSubgaussianMGF (X i) (c i) μ)
    (hradius : ∀ i ∈ s, 0 ≤ radius i)
    (htail :
      ∀ i ∈ s,
        2 * Real.exp (-(radius i) ^ 2 / (2 * (c i : ℝ))) ≤ delta i) :
    μ.real (⋃ i ∈ s, CenteredSubGaussianBadEvent (X i) (radius i))
      ≤ ∑ i ∈ s, delta i := by
  calc
    μ.real (⋃ i ∈ s, CenteredSubGaussianBadEvent (X i) (radius i))
      ≤ ∑ i ∈ s, μ.real (CenteredSubGaussianBadEvent (X i) (radius i)) := by
        exact finite_real_union_bad_event_le_sum (μ := μ) s
          (fun i ↦ CenteredSubGaussianBadEvent (X i) (radius i))
    _ ≤ ∑ i ∈ s, delta i := by
      exact Finset.sum_le_sum (by
        intro i hi
        exact (centeredSubGaussian_abs_bad_event_le
          (μ := μ)
          (X := X i)
          (c := c i)
          (radius := radius i)
          (hsub i hi)
          (hradius i hi)).trans (htail i hi))

def AdaptiveStageBadEvent
    {ι : Type*}
    (candidates : ℕ → Finset ι)
    (bad : ℕ → ι → Set Ω)
    (t : ℕ) : Set Ω :=
  ⋃ i ∈ candidates t, bad t i

def AdaptiveBadEventUpTo
    {ι : Type*}
    (candidates : ℕ → Finset ι)
    (bad : ℕ → ι → Set Ω)
    (T : ℕ) : Set Ω :=
  ⋃ t ∈ Finset.range T, AdaptiveStageBadEvent candidates bad t

theorem adaptive_bad_event_real_le_sum
    {ι : Type*}
    (candidates : ℕ → Finset ι)
    (bad : ℕ → ι → Set Ω)
    (T : ℕ) :
    μ.real (AdaptiveBadEventUpTo candidates bad T)
      ≤ ∑ t ∈ Finset.range T, ∑ i ∈ candidates t, μ.real (bad t i) := by
  calc
    μ.real (AdaptiveBadEventUpTo candidates bad T)
      ≤ ∑ t ∈ Finset.range T,
          μ.real (AdaptiveStageBadEvent candidates bad t) := by
        unfold AdaptiveBadEventUpTo
        exact finite_real_union_bad_event_le_sum (μ := μ) (Finset.range T)
          (fun t ↦ AdaptiveStageBadEvent candidates bad t)
    _ ≤ ∑ t ∈ Finset.range T, ∑ i ∈ candidates t, μ.real (bad t i) := by
      exact Finset.sum_le_sum (by
        intro t ht
        unfold AdaptiveStageBadEvent
        exact finite_real_union_bad_event_le_sum (μ := μ) (candidates t)
          (fun i ↦ bad t i))

theorem adaptive_subGaussian_bad_event_le_sum
    {ι : Type*}
    (candidates : ℕ → Finset ι)
    (X : ℕ → ι → Ω → ℝ)
    (c : ℕ → ι → NNReal)
    (radius delta : ℕ → ι → ℝ)
    (T : ℕ)
    (hsub :
      ∀ t ∈ Finset.range T, ∀ i ∈ candidates t,
        HasSubgaussianMGF (X t i) (c t i) μ)
    (hradius :
      ∀ t ∈ Finset.range T, ∀ i ∈ candidates t,
        0 ≤ radius t i)
    (htail :
      ∀ t ∈ Finset.range T, ∀ i ∈ candidates t,
        Real.exp (-(radius t i) ^ 2 / (2 * (c t i : ℝ))) ≤ delta t i) :
    μ.real
        (AdaptiveBadEventUpTo candidates
          (fun t i ↦ SubGaussianBadEvent (X t i) (radius t i)) T)
      ≤ ∑ t ∈ Finset.range T, ∑ i ∈ candidates t, delta t i := by
  calc
    μ.real
        (AdaptiveBadEventUpTo candidates
          (fun t i ↦ SubGaussianBadEvent (X t i) (radius t i)) T)
      ≤ ∑ t ∈ Finset.range T,
          μ.real
            (AdaptiveStageBadEvent candidates
              (fun t i ↦ SubGaussianBadEvent (X t i) (radius t i)) t) := by
        unfold AdaptiveBadEventUpTo
        exact finite_real_union_bad_event_le_sum (μ := μ) (Finset.range T)
          (fun t ↦ AdaptiveStageBadEvent candidates
            (fun t i ↦ SubGaussianBadEvent (X t i) (radius t i)) t)
    _ ≤ ∑ t ∈ Finset.range T, ∑ i ∈ candidates t, delta t i := by
      exact Finset.sum_le_sum (by
        intro t ht
        unfold AdaptiveStageBadEvent
        exact subGaussian_finite_candidate_bad_event_le_sum
          (μ := μ)
          (s := candidates t)
          (X := X t)
          (c := c t)
          (radius := radius t)
          (delta := delta t)
          (fun i hi ↦ hsub t ht i hi)
          (fun i hi ↦ hradius t ht i hi)
          (fun i hi ↦ htail t ht i hi))

theorem adaptive_centeredSubGaussian_bad_event_le_sum
    {ι : Type*}
    (candidates : ℕ → Finset ι)
    (X : ℕ → ι → Ω → ℝ)
    (c : ℕ → ι → NNReal)
    (radius delta : ℕ → ι → ℝ)
    (T : ℕ)
    [IsFiniteMeasure μ]
    (hsub :
      ∀ t ∈ Finset.range T, ∀ i ∈ candidates t,
        HasSubgaussianMGF (X t i) (c t i) μ)
    (hradius :
      ∀ t ∈ Finset.range T, ∀ i ∈ candidates t,
        0 ≤ radius t i)
    (htail :
      ∀ t ∈ Finset.range T, ∀ i ∈ candidates t,
        2 * Real.exp (-(radius t i) ^ 2 / (2 * (c t i : ℝ))) ≤ delta t i) :
    μ.real
        (AdaptiveBadEventUpTo candidates
          (fun t i ↦ CenteredSubGaussianBadEvent (X t i) (radius t i)) T)
      ≤ ∑ t ∈ Finset.range T, ∑ i ∈ candidates t, delta t i := by
  calc
    μ.real
        (AdaptiveBadEventUpTo candidates
          (fun t i ↦ CenteredSubGaussianBadEvent (X t i) (radius t i)) T)
      ≤ ∑ t ∈ Finset.range T,
          μ.real
            (AdaptiveStageBadEvent candidates
              (fun t i ↦ CenteredSubGaussianBadEvent (X t i) (radius t i)) t) := by
        unfold AdaptiveBadEventUpTo
        exact finite_real_union_bad_event_le_sum (μ := μ) (Finset.range T)
          (fun t ↦ AdaptiveStageBadEvent candidates
            (fun t i ↦ CenteredSubGaussianBadEvent (X t i) (radius t i)) t)
    _ ≤ ∑ t ∈ Finset.range T, ∑ i ∈ candidates t, delta t i := by
      exact Finset.sum_le_sum (by
        intro t ht
        unfold AdaptiveStageBadEvent
        exact centeredSubGaussian_finite_candidate_bad_event_le_sum
          (μ := μ)
          (s := candidates t)
          (X := X t)
          (c := c t)
          (radius := radius t)
          (delta := delta t)
          (fun i hi ↦ hsub t ht i hi)
          (fun i hi ↦ hradius t ht i hi)
          (fun i hi ↦ htail t ht i hi))

end SCOLHKG.Measure
