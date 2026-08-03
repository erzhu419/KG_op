import Mathlib

namespace SCOLHKG.Real

/-!
Coverage of a deterministic source-ranked proposal atlas.

Absolute chance thresholds need not agree across domains.  The transferable
object is instead a normalized risk rank.  Uniform source/target rank
alignment, one-sided atlas coverage in source rank, and a sufficiently deep
target-safe policy together force a feasible member to appear in the frozen
atlas.  This is the proposal contract used by the deployed risk-objective
atlas; it makes no independence claim between atlas members.
-/

def UniformRiskRankAlignment
    {X : Type*}
    (sourceRank targetRank : X → ℝ) (epsilon : ℝ) : Prop :=
  ∀ x, |sourceRank x - targetRank x| ≤ epsilon

def OneSidedRiskRankAtlasCover
    {X : Type*} [DecidableEq X]
    (atlas : Finset X) (sourceRank : X → ℝ) (coverError : ℝ) : Prop :=
  ∀ x, ∃ y ∈ atlas, sourceRank y ≤ sourceRank x + coverError

def RiskRankImpliesFeasible
    {X : Type*}
    (targetRank : X → ℝ) (feasible : X → Prop) (threshold : ℝ) : Prop :=
  ∀ x, targetRank x ≤ threshold → feasible x

theorem aligned_rank_atlas_contains_feasible
    {X : Type*} [DecidableEq X]
    {atlas : Finset X}
    {sourceRank targetRank : X → ℝ}
    {feasible : X → Prop}
    {epsilon coverError threshold : ℝ}
    (hAlignment :
      UniformRiskRankAlignment sourceRank targetRank epsilon)
    (hCover :
      OneSidedRiskRankAtlasCover atlas sourceRank coverError)
    (hSafe :
      RiskRankImpliesFeasible targetRank feasible threshold)
    (hInterior :
      ∃ x, targetRank x + 2 * epsilon + coverError ≤ threshold) :
    ∃ y ∈ atlas, feasible y := by
  obtain ⟨x, hDeep⟩ := hInterior
  obtain ⟨y, hYAtlas, hYCover⟩ := hCover x
  have hAlignX := hAlignment x
  have hAlignY := hAlignment y
  have hSourceX :
      sourceRank x ≤ targetRank x + epsilon := by
    obtain ⟨_, hUpper⟩ := abs_le.mp hAlignX
    linarith
  have hTargetY :
      targetRank y ≤ sourceRank y + epsilon := by
    obtain ⟨hLower, _⟩ := abs_le.mp hAlignY
    linarith
  have hYThreshold : targetRank y ≤ threshold := by
    linarith
  exact ⟨y, hYAtlas, hSafe y hYThreshold⟩

theorem finite_rank_aligned_atlas_coverage
    {X : Type*} [DecidableEq X]
    {atlas : Finset X}
    {sourceRank targetRank : X → ℝ}
    {feasible : X → Prop}
    {epsilon coverError threshold : ℝ}
    {n0 : ℕ}
    (hAtlasSize : atlas.card ≤ n0)
    (hAlignment :
      UniformRiskRankAlignment sourceRank targetRank epsilon)
    (hCover :
      OneSidedRiskRankAtlasCover atlas sourceRank coverError)
    (hSafe :
      RiskRankImpliesFeasible targetRank feasible threshold)
    (hInterior :
      ∃ x, targetRank x + 2 * epsilon + coverError ≤ threshold) :
    atlas.card ≤ n0 ∧ ∃ y ∈ atlas, feasible y := by
  exact ⟨hAtlasSize, aligned_rank_atlas_contains_feasible
    hAlignment hCover hSafe hInterior⟩

/-! ## Budgeted universal sentinel reservation

The lower-envelope challenger reserves one of the fixed `n0` proposal slots
for a policy defined only from normalized bounds. It does not enlarge the
charged initial design: the remaining source atlas has at most `n0 - 1`
members. The result below records both the budget and coverage consequences
without claiming that replacing a source member is cost-free.
-/

def ReservedSentinelAtlas
    {X : Type*} [DecidableEq X]
    (sourceAtlas : Finset X) (sentinel : X) : Finset X :=
  insert sentinel sourceAtlas

theorem reserved_sentinel_atlas_card_le
    {X : Type*} [DecidableEq X]
    {sourceAtlas : Finset X}
    {sentinel : X}
    {n0 : ℕ}
    (hReservedBudget : sourceAtlas.card + 1 ≤ n0) :
    (ReservedSentinelAtlas sourceAtlas sentinel).card ≤ n0 := by
  calc
    (ReservedSentinelAtlas sourceAtlas sentinel).card
        ≤ sourceAtlas.card + 1 := by
          simpa [ReservedSentinelAtlas, Nat.add_comm] using
            Finset.card_insert_le sentinel sourceAtlas
    _ ≤ n0 := hReservedBudget

theorem reserved_feasible_sentinel_establishes_atlas_coverage
    {X : Type*} [DecidableEq X]
    {sourceAtlas : Finset X}
    {sentinel : X}
    {feasible : X → Prop}
    (hSentinelFeasible : feasible sentinel) :
    ∃ x ∈ ReservedSentinelAtlas sourceAtlas sentinel, feasible x := by
  exact ⟨sentinel, by simp [ReservedSentinelAtlas], hSentinelFeasible⟩

theorem finite_reserved_sentinel_coverage
    {X : Type*} [DecidableEq X]
    {sourceAtlas : Finset X}
    {sentinel : X}
    {feasible : X → Prop}
    {n0 : ℕ}
    (hReservedBudget : sourceAtlas.card + 1 ≤ n0)
    (hSentinelFeasible : feasible sentinel) :
    (ReservedSentinelAtlas sourceAtlas sentinel).card ≤ n0 ∧
      ∃ x ∈ ReservedSentinelAtlas sourceAtlas sentinel, feasible x := by
  exact ⟨
    reserved_sentinel_atlas_card_le hReservedBudget,
    reserved_feasible_sentinel_establishes_atlas_coverage hSentinelFeasible,
  ⟩

end SCOLHKG.Real
