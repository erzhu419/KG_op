import Mathlib

namespace SCOLHKG.Real

/-!
The oracle-free LODO challenger ranks a shared source policy library within
each source domain, aggregates those ranks, and freezes the resulting proposal
before seeing the held-out target. Its terminal suffix commits enough charged
replications to a fixed shortlist of observed safety challengers before
switching to posterior-only experts.
-/

noncomputable def sourceConsensusScore
    (meanRank worstRank disagreement : ℝ) : ℝ :=
  meanRank + (1 / 4 : ℝ) * worstRank + (1 / 4 : ℝ) * disagreement

theorem sourceConsensusScore_nonnegative
    {meanRank worstRank disagreement : ℝ}
    (hMean : 0 ≤ meanRank)
    (hWorst : 0 ≤ worstRank)
    (hDisagreement : 0 ≤ disagreement) :
    0 ≤ sourceConsensusScore meanRank worstRank disagreement := by
  unfold sourceConsensusScore
  positivity

theorem positiveAffine_order_invariant
    {scale shift first second : ℝ}
    (hScale : 0 < scale) :
    scale * first + shift ≤ scale * second + shift ↔ first ≤ second := by
  constructor <;> intro h
  · nlinarith
  · nlinarith

theorem strictlyMonotone_order_invariant
    {f : ℝ → ℝ} (hMono : StrictMono f) {first second : ℝ} :
    f first ≤ f second ↔ first ≤ second := by
  constructor
  · intro h
    by_contra hNot
    have hReverse : second < first := lt_of_not_ge hNot
    exact (not_lt_of_ge h) (hMono hReverse)
  · intro h
    exact hMono.monotone h

noncomputable def sourceSafetyObjectiveScore
    (weight safetyRank objectiveRank : ℝ) : ℝ :=
  weight * safetyRank + (1 - weight) * objectiveRank

theorem sourceSafetyObjectiveScore_nonnegative
    {weight safetyRank objectiveRank : ℝ}
    (hWeightLower : 0 ≤ weight)
    (hWeightUpper : weight ≤ 1)
    (hSafety : 0 ≤ safetyRank)
    (hObjective : 0 ≤ objectiveRank) :
    0 ≤ sourceSafetyObjectiveScore weight safetyRank objectiveRank := by
  unfold sourceSafetyObjectiveScore
  nlinarith

theorem sourceSafetyObjectiveScore_pareto_monotone
    {weight safetyFirst safetySecond objectiveFirst objectiveSecond : ℝ}
    (hWeightLower : 0 ≤ weight)
    (hWeightUpper : weight ≤ 1)
    (hSafety : safetyFirst ≤ safetySecond)
    (hObjective : objectiveFirst ≤ objectiveSecond) :
    sourceSafetyObjectiveScore weight safetyFirst objectiveFirst ≤
      sourceSafetyObjectiveScore weight safetySecond objectiveSecond := by
  unfold sourceSafetyObjectiveScore
  nlinarith

theorem sourceSafetyObjectiveScore_strict_pareto_monotone
    {weight safetyFirst safetySecond objectiveFirst objectiveSecond : ℝ}
    (hWeightLower : 0 < weight)
    (hWeightUpper : weight < 1)
    (hSafety : safetyFirst ≤ safetySecond)
    (hObjective : objectiveFirst ≤ objectiveSecond)
    (hStrict :
      safetyFirst < safetySecond ∨ objectiveFirst < objectiveSecond) :
    sourceSafetyObjectiveScore weight safetyFirst objectiveFirst <
      sourceSafetyObjectiveScore weight safetySecond objectiveSecond := by
  unfold sourceSafetyObjectiveScore
  rcases hStrict with hStrict | hStrict <;> nlinarith

def sourceFrozenProposal
    {SourceArchive TargetName Proposal : Type*}
    (select : SourceArchive → Proposal)
    (source : SourceArchive)
    (_target : TargetName) : Proposal :=
  select source

theorem source_frozen_proposal_target_noninterference
    {SourceArchive TargetName Proposal : Type*}
    (select : SourceArchive → Proposal)
    (source : SourceArchive)
    (firstTarget secondTarget : TargetName) :
    sourceFrozenProposal select source firstTarget =
      sourceFrozenProposal select source secondTarget := by
  rfl

def rankSpanningDesign {Template : Type*}
    (first : Template) (interior : List Template) (last : Template) :
    List Template :=
  first :: interior ++ [last]

theorem rankSpanningDesign_contains_first
    {Template : Type*} (first last : Template) (interior : List Template) :
    first ∈ rankSpanningDesign first interior last := by
  simp [rankSpanningDesign]

theorem rankSpanningDesign_contains_last
    {Template : Type*} (first last : Template) (interior : List Template) :
    last ∈ rankSpanningDesign first interior last := by
  simp [rankSpanningDesign]

theorem restricted_challenger_is_in_source_design
    {Policy : Type*} {sourceDesign charged : Set Policy} {chosen : Policy}
    (hChosen : chosen ∈ sourceDesign ∩ charged) :
    chosen ∈ sourceDesign := by
  exact hChosen.1

theorem committed_replication_exact
    {current minimum : ℕ}
    (hCurrent : current ≤ minimum) :
    current + (minimum - current) = minimum := by
  omega

theorem committed_replication_within_reserved_budget
    {current minimum reserved : ℕ}
    (hCurrent : current ≤ minimum)
    (hEnough : minimum - current ≤ reserved) :
    minimum ≤ current + reserved := by
  omega

theorem one_challenger_commit_prevents_incomplete_switch
    {current minimum : ℕ}
    (hIncomplete : current < minimum) :
    current + 1 ≤ minimum := by
  omega

theorem bounded_error_preserves_two_challenger_order
    {trueFirst trueSecond estimateFirst estimateSecond epsilon : ℝ}
    (hGap : trueFirst + 2 * epsilon < trueSecond)
    (hFirst : |estimateFirst - trueFirst| ≤ epsilon)
    (hSecond : |estimateSecond - trueSecond| ≤ epsilon) :
    estimateFirst < estimateSecond := by
  rw [abs_le] at hFirst hSecond
  linarith

theorem two_challenger_commit_exact
    {currentFirst currentSecond minimum : ℕ}
    (hFirst : currentFirst ≤ minimum)
    (hSecond : currentSecond ≤ minimum) :
    currentFirst + (minimum - currentFirst) = minimum ∧
      currentSecond + (minimum - currentSecond) = minimum := by
  omega

theorem two_challenger_commit_within_reserved_budget
    {currentFirst currentSecond minimum reserved : ℕ}
    (hFirst : currentFirst ≤ minimum)
    (hSecond : currentSecond ≤ minimum)
    (hEnough :
      (minimum - currentFirst) + (minimum - currentSecond) ≤ reserved) :
    minimum + minimum ≤ currentFirst + currentSecond + reserved := by
  omega

end SCOLHKG.Real
