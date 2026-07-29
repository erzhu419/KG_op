import Mathlib

namespace SCOLHKG.Real

/-!
No-free-lunch result for a finite target-label-free proposal.

If a deterministic atlas does not enumerate the whole policy space, there is
always a nonempty target feasible set that avoids every atlas member. Hence no
source-only finite proposal can guarantee arbitrary held-out feasibility
without a source-to-target coverage, discrepancy, or structural assumption.
-/

theorem proper_finite_atlas_misses_some_nonempty_feasible_set
    {X : Type*} [Fintype X] [DecidableEq X]
    (atlas : Finset X)
    (hProper : atlas ≠ Finset.univ) :
    ∃ feasible : X → Prop,
      (∃ x, feasible x) ∧
        ¬ ∃ candidate ∈ atlas, feasible candidate := by
  have hOutside : ∃ x, x ∉ atlas := by
    by_contra hNoOutside
    push Not at hNoOutside
    apply hProper
    ext x
    simp [hNoOutside x]
  obtain ⟨outside, hOutsideAtlas⟩ := hOutside
  refine ⟨fun x => x = outside, ⟨outside, rfl⟩, ?_⟩
  rintro ⟨candidate, hCandidate, rfl⟩
  exact hOutsideAtlas hCandidate

theorem finite_budget_no_unconditional_target_coverage
    {X : Type*} [Fintype X] [DecidableEq X]
    (atlas : Finset X)
    (n0 : ℕ)
    (hBudget : atlas.card ≤ n0)
    (hSpaceLarger : n0 < Fintype.card X) :
    atlas.card ≤ n0 ∧
      ∃ feasible : X → Prop,
        (∃ x, feasible x) ∧
          ¬ ∃ candidate ∈ atlas, feasible candidate := by
  refine ⟨hBudget, proper_finite_atlas_misses_some_nonempty_feasible_set
    atlas ?_⟩
  intro hAll
  have hCard : atlas.card = Fintype.card X := by
    simp [hAll]
  omega

end SCOLHKG.Real
