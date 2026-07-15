import SCOLHKG.Measure.SubGaussianConfidence
import SCOLHKG.Real.TwoStageDecision

namespace SCOLHKG.Measure

open MeasureTheory ProbabilityTheory
open scoped MeasureTheory ProbabilityTheory BigOperators

/-!
# Probability layer for two-stage verification

The frozen finalist universe permits one simultaneous allocation before the
verification labels are observed.  Margin and objective errors are controlled
uniformly over that universe.  Search, proposal-coverage, and verification
failures are then combined without assuming independence.
-/

variable {Ω Design : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

def FiniteVerificationBadEvent
    (finalists : Finset Design)
    (trueMargin trueObjective : Design → ℝ)
    (estimatedMargin estimatedObjective : Ω → Design → ℝ)
    (marginRadius objectiveRadius : ℝ) : Set Ω :=
  (⋃ x ∈ finalists,
      CenteredSubGaussianBadEvent
        (fun ω => estimatedMargin ω x - trueMargin x)
        marginRadius) ∪
  (⋃ x ∈ finalists,
      CenteredSubGaussianBadEvent
        (fun ω => estimatedObjective ω x - trueObjective x)
        objectiveRadius)

theorem finite_verification_bad_event_le
    (finalists : Finset Design)
    (trueMargin trueObjective : Design → ℝ)
    (estimatedMargin estimatedObjective : Ω → Design → ℝ)
    (marginProxy objectiveProxy : Design → NNReal)
    {marginRadius objectiveRadius : ℝ}
    (marginDelta objectiveDelta : Design → ℝ)
    [IsFiniteMeasure μ]
    (hMarginSubgaussian :
      ∀ x ∈ finalists,
        HasSubgaussianMGF
          (fun ω => estimatedMargin ω x - trueMargin x)
          (marginProxy x) μ)
    (hObjectiveSubgaussian :
      ∀ x ∈ finalists,
        HasSubgaussianMGF
          (fun ω => estimatedObjective ω x - trueObjective x)
          (objectiveProxy x) μ)
    (hMarginRadius : 0 ≤ marginRadius)
    (hObjectiveRadius : 0 ≤ objectiveRadius)
    (hMarginTail :
      ∀ x ∈ finalists,
        2 * Real.exp
          (-marginRadius ^ 2 / (2 * (marginProxy x : ℝ)))
          ≤ marginDelta x)
    (hObjectiveTail :
      ∀ x ∈ finalists,
        2 * Real.exp
          (-objectiveRadius ^ 2 / (2 * (objectiveProxy x : ℝ)))
          ≤ objectiveDelta x) :
    μ.real (FiniteVerificationBadEvent finalists
      trueMargin trueObjective estimatedMargin estimatedObjective
      marginRadius objectiveRadius)
      ≤ (∑ x ∈ finalists, marginDelta x)
        + ∑ x ∈ finalists, objectiveDelta x := by
  unfold FiniteVerificationBadEvent
  calc
    μ.real
        ((⋃ x ∈ finalists,
            CenteredSubGaussianBadEvent
              (fun ω => estimatedMargin ω x - trueMargin x)
              marginRadius) ∪
          (⋃ x ∈ finalists,
            CenteredSubGaussianBadEvent
              (fun ω => estimatedObjective ω x - trueObjective x)
              objectiveRadius))
      ≤ μ.real
          (⋃ x ∈ finalists,
            CenteredSubGaussianBadEvent
              (fun ω => estimatedMargin ω x - trueMargin x)
              marginRadius)
        + μ.real
          (⋃ x ∈ finalists,
            CenteredSubGaussianBadEvent
              (fun ω => estimatedObjective ω x - trueObjective x)
              objectiveRadius) := measureReal_union_le _ _
    _ ≤ (∑ x ∈ finalists, marginDelta x)
          + ∑ x ∈ finalists, objectiveDelta x := by
      gcongr
      · exact centeredSubGaussian_finite_candidate_bad_event_le_sum
          (μ := μ) finalists
          (fun x ω => estimatedMargin ω x - trueMargin x)
          marginProxy (fun _ => marginRadius) marginDelta
          hMarginSubgaussian
          (fun _ _ => hMarginRadius)
          hMarginTail
      · exact centeredSubGaussian_finite_candidate_bad_event_le_sum
          (μ := μ) finalists
          (fun x ω => estimatedObjective ω x - trueObjective x)
          objectiveProxy (fun _ => objectiveRadius) objectiveDelta
          hObjectiveSubgaussian
          (fun _ _ => hObjectiveRadius)
          hObjectiveTail

theorem uniform_verification_errors_of_not_bad
    (finalists : Finset Design)
    (trueMargin trueObjective : Design → ℝ)
    (estimatedMargin estimatedObjective : Ω → Design → ℝ)
    {marginRadius objectiveRadius : ℝ}
    {ω : Ω}
    (hGood : ω ∉ FiniteVerificationBadEvent finalists
      trueMargin trueObjective estimatedMargin estimatedObjective
      marginRadius objectiveRadius) :
    (∀ x ∈ finalists,
      |estimatedMargin ω x - trueMargin x| ≤ marginRadius) ∧
    (∀ x ∈ finalists,
      |estimatedObjective ω x - trueObjective x| ≤ objectiveRadius) := by
  constructor
  · intro x hx
    have hNot :
        ¬ marginRadius ≤ |estimatedMargin ω x - trueMargin x| := by
      intro hBad
      apply hGood
      apply Set.mem_union_left
      exact Set.mem_iUnion₂.mpr ⟨x, hx, hBad⟩
    exact le_of_lt (lt_of_not_ge hNot)
  · intro x hx
    have hNot :
        ¬ objectiveRadius ≤ |estimatedObjective ω x - trueObjective x| := by
      intro hBad
      apply hGood
      apply Set.mem_union_right
      exact Set.mem_iUnion₂.mpr ⟨x, hx, hBad⟩
    exact le_of_lt (lt_of_not_ge hNot)

def TwoStageBadEvent
    (searchBad proposalBad verificationBad : Set Ω) : Set Ω :=
  searchBad ∪ proposalBad ∪ verificationBad

theorem two_stage_bad_event_le_sum
    (searchBad proposalBad verificationBad : Set Ω) :
    μ.real (TwoStageBadEvent searchBad proposalBad verificationBad)
      ≤ μ.real searchBad + μ.real proposalBad + μ.real verificationBad := by
  unfold TwoStageBadEvent
  have hFirst :
      μ.real (searchBad ∪ proposalBad)
        ≤ μ.real searchBad + μ.real proposalBad :=
    measureReal_union_le _ _
  have hSecond :
      μ.real ((searchBad ∪ proposalBad) ∪ verificationBad)
        ≤ μ.real (searchBad ∪ proposalBad) + μ.real verificationBad :=
    measureReal_union_le _ _
  linarith

def TwoStageConclusionFailure (conclusion : Ω → Prop) : Set Ω :=
  {ω | ¬ conclusion ω}

theorem two_stage_conclusion_failure_le_bad
    {bad : Set Ω}
    {conclusion : Ω → Prop}
    {delta : ℝ}
    [IsFiniteMeasure μ]
    (hOutside : ∀ ω, ω ∉ bad → conclusion ω)
    (hBad : μ.real bad ≤ delta) :
    μ.real (TwoStageConclusionFailure conclusion) ≤ delta := by
  apply (measureReal_mono ?_).trans hBad
  intro ω hFailure
  by_contra hGood
  exact hFailure (hOutside ω hGood)

theorem two_stage_safe_regret_failure_probability_le
    (searchBad proposalBad verificationBad : Set Ω)
    (rec : Ω → Design)
    (trueMargin objective : Design → ℝ)
    (xStar : Design)
    (epsilon searchDelta proposalDelta verificationDelta : ℝ)
    [IsFiniteMeasure μ]
    (hSearch : μ.real searchBad ≤ searchDelta)
    (hProposal : μ.real proposalBad ≤ proposalDelta)
    (hVerification : μ.real verificationBad ≤ verificationDelta)
    (hOutside :
      ∀ ω, ω ∉ TwoStageBadEvent searchBad proposalBad verificationBad →
        trueMargin (rec ω) ≤ 0 ∧
        objective (rec ω) - objective xStar ≤ epsilon) :
    μ.real (TwoStageConclusionFailure (fun ω =>
      trueMargin (rec ω) ≤ 0 ∧
      objective (rec ω) - objective xStar ≤ epsilon))
      ≤ searchDelta + proposalDelta + verificationDelta := by
  apply two_stage_conclusion_failure_le_bad hOutside
  calc
    μ.real (TwoStageBadEvent searchBad proposalBad verificationBad)
      ≤ μ.real searchBad + μ.real proposalBad + μ.real verificationBad :=
        two_stage_bad_event_le_sum searchBad proposalBad verificationBad
    _ ≤ searchDelta + proposalDelta + verificationDelta := by
      linarith

end SCOLHKG.Measure
