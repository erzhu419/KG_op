import SCOLHKG.Real.LineEnvelopeStack

namespace SCOLHKG.Real

/-!
Step-level formalization of the Python `compute_h` stack loop.

The imperative implementation maintains a stack of active lines and a parallel
stack of lower cut points.  The `while` loop has only two structural mutations:

* pop the last line/cut while the new intersection violates monotonicity;
* append the new line/cut once the new cut is larger than every existing cut
  (in Python this is tested against the last cut because the stack is already
  ordered).

This file proves those two mutations preserve the cut-order invariant used by
`validate_h_certificate` and the Lean certificate bridge in
`LineEnvelopeStack.lean`.
-/

open List

def StrictCuts (cuts : List ℝ) : Prop :=
  cuts.Pairwise (· < ·)

structure StackShape where
  slopes : List ℝ
  cuts : List ℝ

def StackShape.WellFormed (s : StackShape) : Prop :=
  s.slopes.Pairwise (· < ·) ∧ StrictCuts s.cuts ∧ s.slopes.length = s.cuts.length

def StackShape.popLast (s : StackShape) : StackShape :=
  { slopes := s.slopes.dropLast, cuts := s.cuts.dropLast }

def StackShape.push (s : StackShape) (slope cut : ℝ) : StackShape :=
  { slopes := s.slopes ++ [slope], cuts := s.cuts ++ [cut] }

theorem strictCuts_dropLast
    {cuts : List ℝ}
    (h : StrictCuts cuts) :
    StrictCuts cuts.dropLast := by
  exact List.Pairwise.sublist (List.IsPrefix.sublist (List.dropLast_prefix cuts)) h

theorem pairwise_append_singleton_of_forall_lt
    {xs : List ℝ}
    {x : ℝ}
    (h : xs.Pairwise (· < ·))
    (hall : ∀ y ∈ xs, y < x) :
    (xs ++ [x]).Pairwise (· < ·) := by
  rw [List.pairwise_append]
  refine ⟨h, by simp, ?_⟩
  intro y hy z hz
  simp at hz
  subst z
  exact hall y hy

theorem stack_pop_preserves_wellFormed
    {s : StackShape}
    (h : s.WellFormed) :
    s.popLast.WellFormed := by
  rcases h with ⟨hSlopes, hCuts, hLen⟩
  refine ⟨?_, ?_, ?_⟩
  · exact List.Pairwise.sublist
      (List.IsPrefix.sublist (List.dropLast_prefix s.slopes))
      hSlopes
  · exact strictCuts_dropLast hCuts
  · unfold StackShape.popLast
    simp [hLen]

theorem stack_push_preserves_wellFormed
    {s : StackShape}
    {newSlope newCut : ℝ}
    (h : s.WellFormed)
    (hSlope : ∀ old ∈ s.slopes, old < newSlope)
    (hCut : ∀ old ∈ s.cuts, old < newCut) :
    (s.push newSlope newCut).WellFormed := by
  rcases h with ⟨hSlopes, hCuts, hLen⟩
  refine ⟨?_, ?_, ?_⟩
  · exact pairwise_append_singleton_of_forall_lt hSlopes hSlope
  · exact pairwise_append_singleton_of_forall_lt hCuts hCut
  · unfold StackShape.push
    simp [hLen]

theorem python_pop_branch_preserves_cut_order
    {s : StackShape}
    (h : s.WellFormed) :
    StrictCuts s.popLast.cuts :=
  (stack_pop_preserves_wellFormed h).2.1

theorem python_break_branch_push_preserves_cut_order
    {s : StackShape}
    {newSlope newCut : ℝ}
    (h : s.WellFormed)
    (hSlope : ∀ old ∈ s.slopes, old < newSlope)
    (hCut : ∀ old ∈ s.cuts, old < newCut) :
    StrictCuts (s.push newSlope newCut).cuts :=
  (stack_push_preserves_wellFormed h hSlope hCut).2.1

end SCOLHKG.Real
