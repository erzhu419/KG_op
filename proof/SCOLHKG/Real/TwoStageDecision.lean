import Mathlib
import SCOLHKG.Real.FinalistReplication

namespace SCOLHKG.Real

/-!
# Two-stage search and confirmatory selection

This file is the deterministic core of the deployed decision architecture.
The first `N - R` observations are charged to state-coupled KG search.  The
last `R` observations are charged to heteroscedastic ranking-and-selection on
a finite universe frozen before those observations are seen.

The terminal report has two semantically distinct states.  A certified report
may claim chance feasibility on a simultaneous upper-coverage event.  A
fallback report only minimizes an upper-risk score and never inherits that
claim.  This distinction lets the useful least-risk behavior remain in the
algorithm without turning a two-replicate ranking score into a theorem-level
certificate.
-/

structure TwoStageBudget where
  totalBudget : ℕ
  verificationBudget : ℕ

def TwoStageBudget.Valid (b : TwoStageBudget) : Prop :=
  b.verificationBudget ≤ b.totalBudget

def TwoStageBudget.searchBudget (b : TwoStageBudget) : ℕ :=
  b.totalBudget - b.verificationBudget

def TwoStageBudget.initialDesignBudget
    (b : TwoStageBudget) (n0 : ℕ) : ℕ :=
  min n0 b.searchBudget

def TwoStageBudget.adaptiveSearchBudget
    (b : TwoStageBudget) (n0 : ℕ) : ℕ :=
  b.searchBudget - b.initialDesignBudget n0

def TwoStageBudget.IsSearchStage (b : TwoStageBudget) (stage : ℕ) : Prop :=
  stage < b.searchBudget

def TwoStageBudget.IsInitialDesignStage
    (b : TwoStageBudget) (n0 stage : ℕ) : Prop :=
  stage < b.initialDesignBudget n0

def TwoStageBudget.IsAdaptiveSearchStage
    (b : TwoStageBudget) (n0 stage : ℕ) : Prop :=
  b.initialDesignBudget n0 ≤ stage ∧ stage < b.searchBudget

def TwoStageBudget.IsVerificationStage
    (b : TwoStageBudget) (stage : ℕ) : Prop :=
  b.searchBudget ≤ stage ∧ stage < b.totalBudget

theorem search_add_verification_eq_total
    (b : TwoStageBudget)
    (hValid : b.Valid) :
    b.searchBudget + b.verificationBudget = b.totalBudget := by
  unfold TwoStageBudget.Valid TwoStageBudget.searchBudget at *
  omega

theorem initial_add_adaptive_add_verification_eq_total
    (b : TwoStageBudget)
    (n0 : ℕ)
    (hValid : b.Valid) :
    b.initialDesignBudget n0 + b.adaptiveSearchBudget n0
      + b.verificationBudget = b.totalBudget := by
  have hInitialLe : b.initialDesignBudget n0 ≤ b.searchBudget := by
    exact Nat.min_le_right n0 b.searchBudget
  have hSearchPartition :
      b.initialDesignBudget n0 + b.adaptiveSearchBudget n0
        = b.searchBudget := by
    unfold TwoStageBudget.adaptiveSearchBudget
    omega
  rw [hSearchPartition]
  exact search_add_verification_eq_total b hValid

theorem search_stage_partition
    (b : TwoStageBudget)
    (n0 : ℕ)
    {stage : ℕ}
    (hSearch : b.IsSearchStage stage) :
    b.IsInitialDesignStage n0 stage ∨
      b.IsAdaptiveSearchStage n0 stage := by
  unfold TwoStageBudget.IsSearchStage at hSearch
  unfold TwoStageBudget.IsInitialDesignStage
    TwoStageBudget.IsAdaptiveSearchStage
  by_cases hInitial : stage < b.initialDesignBudget n0
  · exact Or.inl hInitial
  · exact Or.inr ⟨Nat.le_of_not_gt hInitial, hSearch⟩

theorem initial_and_adaptive_stages_disjoint
    (b : TwoStageBudget)
    (n0 : ℕ)
    {stage : ℕ}
    (hInitial : b.IsInitialDesignStage n0 stage)
    (hAdaptive : b.IsAdaptiveSearchStage n0 stage) : False := by
  exact (Nat.not_lt_of_ge hAdaptive.1) hInitial

theorem stage_partition
    (b : TwoStageBudget)
    {stage : ℕ}
    (hInside : stage < b.totalBudget) :
    b.IsSearchStage stage ∨ b.IsVerificationStage stage := by
  unfold TwoStageBudget.IsSearchStage
    TwoStageBudget.IsVerificationStage
  by_cases hSearch : stage < b.searchBudget
  · exact Or.inl hSearch
  · exact Or.inr ⟨Nat.le_of_not_gt hSearch, hInside⟩

theorem search_and_verification_stages_disjoint
    (b : TwoStageBudget)
    {stage : ℕ}
    (hSearch : b.IsSearchStage stage)
    (hVerification : b.IsVerificationStage stage) : False := by
  exact (Nat.not_lt_of_ge hVerification.1) hSearch

inductive TerminalDecisionStatus where
  | certified
  | fallback
deriving DecidableEq, Repr

structure TerminalReport (Design : Type*) where
  chosen : Design
  status : TerminalDecisionStatus

def ValidTerminalReport
    {Design : Type*}
    (finalists : Finset Design)
    (upperMargin estimatedObjective : Design → ℝ)
    (report : TerminalReport Design) : Prop :=
  report.chosen ∈ finalists ∧
  match report.status with
  | .certified =>
      upperMargin report.chosen ≤ 0 ∧
      ∀ x ∈ finalists, upperMargin x ≤ 0 →
        estimatedObjective report.chosen ≤ estimatedObjective x
  | .fallback =>
      (¬ ∃ x ∈ finalists, upperMargin x ≤ 0) ∧
      ∀ x ∈ finalists, upperMargin report.chosen ≤ upperMargin x

theorem safety_first_yields_certified_report
    {Design : Type*}
    {finalists : Finset Design}
    {upperMargin estimatedObjective : Design → ℝ}
    {chosen : Design}
    (hSafetyFirst :
      IsFinalistSafetyFirst finalists upperMargin estimatedObjective chosen)
    (hExists : ∃ x ∈ finalists, upperMargin x ≤ 0) :
    ValidTerminalReport finalists upperMargin estimatedObjective
      { chosen := chosen, status := .certified } := by
  rcases hSafetyFirst with ⟨hChosen, hCertified, _⟩
  rcases hCertified hExists with ⟨hMargin, hObjective⟩
  exact ⟨hChosen, hMargin, hObjective⟩

theorem safety_first_yields_fallback_report
    {Design : Type*}
    {finalists : Finset Design}
    {upperMargin estimatedObjective : Design → ℝ}
    {chosen : Design}
    (hSafetyFirst :
      IsFinalistSafetyFirst finalists upperMargin estimatedObjective chosen)
    (hNone : ¬ ∃ x ∈ finalists, upperMargin x ≤ 0) :
    ValidTerminalReport finalists upperMargin estimatedObjective
      { chosen := chosen, status := .fallback } := by
  rcases hSafetyFirst with ⟨hChosen, _, hFallback⟩
  exact ⟨hChosen, hNone, hFallback hNone⟩

theorem certified_report_has_nonpositive_upper_margin
    {Design : Type*}
    {finalists : Finset Design}
    {upperMargin estimatedObjective : Design → ℝ}
    {chosen : Design}
    (hValid :
      ValidTerminalReport finalists upperMargin estimatedObjective
        { chosen := chosen, status := .certified }) :
    upperMargin chosen ≤ 0 := by
  exact hValid.2.1

theorem fallback_report_does_not_claim_certification
    {Design : Type*}
    {finalists : Finset Design}
    {upperMargin estimatedObjective : Design → ℝ}
    {chosen : Design}
    (hValid :
      ValidTerminalReport finalists upperMargin estimatedObjective
        { chosen := chosen, status := .fallback }) :
    ¬ upperMargin chosen ≤ 0 := by
  intro hMargin
  exact hValid.2.1 ⟨chosen, hValid.1, hMargin⟩

theorem certified_terminal_sound_on_coverage_event
    {Design : Type*}
    {finalists : Finset Design}
    {trueMargin upperMargin estimatedObjective : Design → ℝ}
    {chosen : Design}
    (hValid :
      ValidTerminalReport finalists upperMargin estimatedObjective
        { chosen := chosen, status := .certified })
    (hCoverage : trueMargin chosen ≤ upperMargin chosen) :
    trueMargin chosen ≤ 0 := by
  exact hCoverage.trans hValid.2.1

theorem replicated_terminal_sound_on_joint_event
    {Design : Type*}
    {finalists : Finset Design}
    {sampleMean sigma estimatedObjective : Design → ℝ}
    {replicateCount : Design → ℕ}
    {trueMean trueSigma : Design → ℝ}
    {zAlpha zDelta tau : ℝ}
    {chosen : Design}
    (hValid :
      ValidTerminalReport finalists
        (fun x => replicatedFinalistMargin
          (sampleMean x) (sigma x) (replicateCount x)
          zAlpha zDelta tau)
        estimatedObjective
        { chosen := chosen, status := .certified })
    (hMean :
      trueMean chosen ≤ replicatedMeanUpper
        (sampleMean chosen) (sigma chosen) (replicateCount chosen) zDelta)
    (hSigma : trueSigma chosen ≤ sigma chosen)
    (hZAlpha : 0 ≤ zAlpha) :
    trueMean chosen + zAlpha * trueSigma chosen ≤ tau := by
  exact replicated_finalist_margin_sound_on_joint_event
    hMean hSigma hZAlpha hValid.2.1

theorem uniform_error_preserves_strict_safety
    {trueMargin upperMargin epsilon : ℝ}
    (hError : |upperMargin - trueMargin| ≤ epsilon)
    (hBuffer : trueMargin ≤ -epsilon) :
    upperMargin ≤ 0 := by
  rw [abs_le] at hError
  linarith

theorem estimated_minimizer_true_objective_le_two_error
    {estimatedChosen estimatedComparator : ℝ}
    {trueChosen trueComparator epsilon : ℝ}
    (hOrder : estimatedChosen ≤ estimatedComparator)
    (hChosenError : |estimatedChosen - trueChosen| ≤ epsilon)
    (hComparatorError :
      |estimatedComparator - trueComparator| ≤ epsilon) :
    trueChosen ≤ trueComparator + 2 * epsilon := by
  rw [abs_le] at hChosenError hComparatorError
  linarith

theorem certified_report_near_strictly_safe_comparator
    {Design : Type*}
    {finalists : Finset Design}
    {trueMargin upperMargin trueObjective estimatedObjective : Design → ℝ}
    {chosen comparator : Design}
    {marginError objectiveError : ℝ}
    (hValid :
      ValidTerminalReport finalists upperMargin estimatedObjective
        { chosen := chosen, status := .certified })
    (hComparator : comparator ∈ finalists)
    (hStrictSafety : trueMargin comparator ≤ -marginError)
    (hMarginError :
      |upperMargin comparator - trueMargin comparator| ≤ marginError)
    (hChosenObjectiveError :
      |estimatedObjective chosen - trueObjective chosen| ≤ objectiveError)
    (hComparatorObjectiveError :
      |estimatedObjective comparator - trueObjective comparator|
        ≤ objectiveError) :
    trueObjective chosen ≤ trueObjective comparator + 2 * objectiveError := by
  have hComparatorCertified : upperMargin comparator ≤ 0 :=
    uniform_error_preserves_strict_safety hMarginError hStrictSafety
  have hEstimatedOrder :
      estimatedObjective chosen ≤ estimatedObjective comparator :=
    hValid.2.2 comparator hComparator hComparatorCertified
  exact estimated_minimizer_true_objective_le_two_error
    hEstimatedOrder hChosenObjectiveError hComparatorObjectiveError

theorem least_risk_fallback_true_margin_le_comparator
    {Design : Type*}
    {finalists : Finset Design}
    {trueMargin upperMargin estimatedObjective : Design → ℝ}
    {chosen comparator : Design}
    {epsilon : ℝ}
    (hValid :
      ValidTerminalReport finalists upperMargin estimatedObjective
        { chosen := chosen, status := .fallback })
    (hComparator : comparator ∈ finalists)
    (hChosenError : |upperMargin chosen - trueMargin chosen| ≤ epsilon)
    (hComparatorError :
      |upperMargin comparator - trueMargin comparator| ≤ epsilon) :
    trueMargin chosen ≤ trueMargin comparator + 2 * epsilon := by
  have hUpperOrder : upperMargin chosen ≤ upperMargin comparator :=
    hValid.2.2 comparator hComparator
  rw [abs_le] at hChosenError hComparatorError
  linarith

structure TwoStageRegretTerms where
  searchError : ℝ
  proposalError : ℝ
  verificationError : ℝ

def TwoStageRegretTerms.total (t : TwoStageRegretTerms) : ℝ :=
  t.searchError + t.proposalError + t.verificationError

theorem two_stage_regret_decomposition
    {objective : Design → ℝ}
    {xStar xSearch xFinalist chosen : Design}
    (terms : TwoStageRegretTerms)
    (hSearch :
      objective xSearch - objective xStar ≤ terms.searchError)
    (hProposal :
      objective xFinalist - objective xSearch ≤ terms.proposalError)
    (hVerification :
      objective chosen - objective xFinalist ≤ terms.verificationError) :
    objective chosen - objective xStar ≤ terms.total := by
  unfold TwoStageRegretTerms.total
  linarith

theorem two_stage_certified_safe_regret
    {Design : Type*}
    {finalists : Finset Design}
    {trueMargin upperMargin estimatedObjective objective : Design → ℝ}
    {chosen xFinalist xSearch xStar : Design}
    (terms : TwoStageRegretTerms)
    (hValid :
      ValidTerminalReport finalists upperMargin estimatedObjective
        { chosen := chosen, status := .certified })
    (hCoverage : trueMargin chosen ≤ upperMargin chosen)
    (hSearch :
      objective xSearch - objective xStar ≤ terms.searchError)
    (hProposal :
      objective xFinalist - objective xSearch ≤ terms.proposalError)
    (hVerification :
      objective chosen - objective xFinalist ≤ terms.verificationError) :
    trueMargin chosen ≤ 0 ∧
      objective chosen - objective xStar ≤ terms.total := by
  constructor
  · exact certified_terminal_sound_on_coverage_event hValid hCoverage
  · exact two_stage_regret_decomposition terms hSearch hProposal hVerification

end SCOLHKG.Real
