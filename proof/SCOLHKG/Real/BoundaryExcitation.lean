import Mathlib

namespace SCOLHKG.Real

/-!
Source-only learning of a signed chance-boundary coordinate requires an
excited source design. The results below separate two facts:

* a design containing observations on both strict sides of zero has explicit
  boundary excitation;
* when two candidate margin functions agree on every source design point, no
  deterministic source-only estimator can distinguish them away from that
  design. At least one of its two errors is half the unseen gap.

The second statement is an identifiability lower bound, not a guarantee that
two-sided excitation alone is sufficient for a chosen regression model.
-/

def StrictTwoSidedBoundaryExcitation
    {X : Type*} (sourceDesign : Set X) (margin : X → ℝ) : Prop :=
  (∃ x ∈ sourceDesign, margin x < 0) ∧
    (∃ x ∈ sourceDesign, 0 < margin x)

theorem nonnegative_source_not_strictly_excited
    {X : Type*} {sourceDesign : Set X} {margin : X → ℝ}
    (hNonnegative : ∀ x ∈ sourceDesign, 0 ≤ margin x) :
    ¬ StrictTwoSidedBoundaryExcitation sourceDesign margin := by
  intro hExcited
  rcases hExcited.1 with ⟨x, hx, hNegative⟩
  exact (not_lt_of_ge (hNonnegative x hx)) hNegative

theorem nonpositive_source_not_strictly_excited
    {X : Type*} {sourceDesign : Set X} {margin : X → ℝ}
    (hNonpositive : ∀ x ∈ sourceDesign, margin x ≤ 0) :
    ¬ StrictTwoSidedBoundaryExcitation sourceDesign margin := by
  intro hExcited
  rcases hExcited.2 with ⟨x, hx, hPositive⟩
  exact (not_lt_of_ge (hNonpositive x hx)) hPositive

def sourceRestriction
    {X : Type*} (sourceDesign : Set X) (margin : X → ℝ) :
    sourceDesign → ℝ :=
  fun x => margin x.1

theorem sourceRestriction_eq_of_agreeOn
    {X : Type*} {sourceDesign : Set X} {first second : X → ℝ}
    (hAgree : Set.EqOn first second sourceDesign) :
    sourceRestriction sourceDesign first =
      sourceRestriction sourceDesign second := by
  funext x
  exact hAgree x.2

theorem half_gap_le_max_error (prediction first second : ℝ) :
    |first - second| / 2 ≤
      max |prediction - first| |prediction - second| := by
  have hTriangle :
      |first - second| ≤
        |prediction - first| + |prediction - second| := by
    calc
      |first - second| = |(first - prediction) + (prediction - second)| := by
        ring_nf
      _ ≤ |first - prediction| + |prediction - second| := abs_add_le _ _
      _ = |prediction - first| + |prediction - second| := by
        rw [abs_sub_comm first prediction]
  have hFirst :
      |prediction - first| ≤
        max |prediction - first| |prediction - second| :=
    le_max_left _ _
  have hSecond :
      |prediction - second| ≤
        max |prediction - first| |prediction - second| :=
    le_max_right _ _
  linarith

theorem source_indistinguishability_lower_bound
    {X : Type*}
    {sourceDesign : Set X}
    {first second : X → ℝ}
    (estimate : (sourceDesign → ℝ) → X → ℝ)
    (hAgree : Set.EqOn first second sourceDesign)
    (x : X) :
    |first x - second x| / 2 ≤
      max
        |estimate (sourceRestriction sourceDesign first) x - first x|
        |estimate (sourceRestriction sourceDesign second) x - second x| := by
  have hRestriction := sourceRestriction_eq_of_agreeOn hAgree
  rw [← hRestriction]
  exact half_gap_le_max_error
    (estimate (sourceRestriction sourceDesign first) x)
    (first x)
    (second x)

theorem source_only_decision_cannot_separate_agreeing_models
    {X Decision : Type*}
    {sourceDesign : Set X}
    {first second : X → ℝ}
    (decide : (sourceDesign → ℝ) → Decision)
    (hAgree : Set.EqOn first second sourceDesign) :
    decide (sourceRestriction sourceDesign first) =
      decide (sourceRestriction sourceDesign second) := by
  rw [sourceRestriction_eq_of_agreeOn hAgree]

end SCOLHKG.Real
