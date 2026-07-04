import SCOLHKG.Measure.SubGaussianConfidence

namespace SCOLHKG.Measure

open MeasureTheory ProbabilityTheory
open scoped MeasureTheory ProbabilityTheory BigOperators

/-!
Posterior-sampling candidate generator events.

`core/candidates.py::posterior_sample_candidates` produces random finite
candidate sets by sampling parametric coefficients and keeping good candidates
from a finite pool.  The proof bridge below says: if every random candidate set
is contained in a deterministic envelope pool, then the bad event for the
random generator is contained in the deterministic adaptive bad event already
covered by `SubGaussianConfidence.lean`.
-/

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

def RandomAdaptiveBadEventUpTo
    {ι : Type*}
    (randomCandidates : ℕ → Ω → Finset ι)
    (bad : ℕ → ι → Set Ω)
    (T : ℕ) : Set Ω :=
  {ω | ∃ t ∈ Finset.range T, ∃ x ∈ randomCandidates t ω, ω ∈ bad t x}

theorem randomAdaptiveBadEvent_subset_envelope
    {ι : Type*}
    (randomCandidates : ℕ → Ω → Finset ι)
    (envelope : ℕ → Finset ι)
    (bad : ℕ → ι → Set Ω)
    (T : ℕ)
    (hSubset :
      ∀ ω t, t ∈ Finset.range T →
        randomCandidates t ω ⊆ envelope t) :
    RandomAdaptiveBadEventUpTo randomCandidates bad T
      ⊆ AdaptiveBadEventUpTo envelope bad T := by
  intro ω hω
  rcases hω with ⟨t, ht, x, hx, hbad⟩
  unfold AdaptiveBadEventUpTo AdaptiveStageBadEvent
  exact Set.mem_iUnion.2
    ⟨t, Set.mem_iUnion.2
      ⟨ht, Set.mem_iUnion.2
        ⟨x, Set.mem_iUnion.2
          ⟨hSubset ω t ht hx, hbad⟩⟩⟩⟩

theorem randomAdaptiveBadEvent_probability_le_envelope
    {ι : Type*}
    (randomCandidates : ℕ → Ω → Finset ι)
    (envelope : ℕ → Finset ι)
    (bad : ℕ → ι → Set Ω)
    (T : ℕ)
    {delta : ℝ}
    [IsFiniteMeasure μ]
    (hSubset :
      ∀ ω t, t ∈ Finset.range T →
        randomCandidates t ω ⊆ envelope t)
    (hEnvelope :
      μ.real (AdaptiveBadEventUpTo envelope bad T) ≤ delta) :
    μ.real (RandomAdaptiveBadEventUpTo randomCandidates bad T) ≤ delta := by
  exact (measureReal_mono
    (randomAdaptiveBadEvent_subset_envelope
      randomCandidates envelope bad T hSubset)).trans hEnvelope

theorem randomAdaptiveCenteredSubGaussian_bad_event_le_sum
    {ι : Type*}
    (randomCandidates : ℕ → Ω → Finset ι)
    (envelope : ℕ → Finset ι)
    (X : ℕ → ι → Ω → ℝ)
    (c : ℕ → ι → NNReal)
    (radius delta : ℕ → ι → ℝ)
    (T : ℕ)
    [IsFiniteMeasure μ]
    (hSubset :
      ∀ ω t, t ∈ Finset.range T →
        randomCandidates t ω ⊆ envelope t)
    (hsub :
      ∀ t ∈ Finset.range T, ∀ x ∈ envelope t,
        HasSubgaussianMGF (X t x) (c t x) μ)
    (hradius :
      ∀ t ∈ Finset.range T, ∀ x ∈ envelope t,
        0 ≤ radius t x)
    (htail :
      ∀ t ∈ Finset.range T, ∀ x ∈ envelope t,
        2 * Real.exp (-(radius t x) ^ 2 / (2 * (c t x : ℝ))) ≤ delta t x) :
    μ.real
        (RandomAdaptiveBadEventUpTo randomCandidates
          (fun t x ↦ CenteredSubGaussianBadEvent (X t x) (radius t x)) T)
      ≤ ∑ t ∈ Finset.range T, ∑ x ∈ envelope t, delta t x := by
  apply randomAdaptiveBadEvent_probability_le_envelope
    (randomCandidates := randomCandidates)
    (envelope := envelope)
    (bad := fun t x ↦ CenteredSubGaussianBadEvent (X t x) (radius t x))
    (T := T)
    (hSubset := hSubset)
  exact adaptive_centeredSubGaussian_bad_event_le_sum
    (μ := μ)
    (candidates := envelope)
    (X := X)
    (c := c)
    (radius := radius)
    (delta := delta)
    (T := T)
    hsub
    hradius
    htail

end SCOLHKG.Measure
