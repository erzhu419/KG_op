import SCOLHKG.Measure.IndependentConfirmation
import SCOLHKG.Real.BoundaryCoordinateSufficiency

namespace SCOLHKG.Measure

open MeasureTheory
open scoped BigOperators ENNReal

/-!
# V57 action-confirmation and terminal-switch composition

V57 retains V56's finite two-head confirmation event and adds a sequential
posterior-incumbent switch event. The two guarantees concern different
failures, so the implementation records separate budgets. A finite union
bound composes them without assuming independence. The runtime divides the
terminal-switch budget by the number of charged online stages.
-/

theorem finite_switch_horizon_error_spending
    {Stage Omega : Type*}
    [MeasurableSpace Omega]
    {mu : Measure Omega}
    (stages : Finset Stage)
    (badSwitch : Stage → Set Omega)
    (alphaSwitch : Stage → ℝ≥0∞)
    (deltaSwitch : ℝ≥0∞)
    (hCell : ∀ stage ∈ stages,
      mu (badSwitch stage) ≤ alphaSwitch stage)
    (hBudget :
      ∑ stage ∈ stages, alphaSwitch stage ≤ deltaSwitch) :
    mu (⋃ stage ∈ stages, badSwitch stage) ≤ deltaSwitch := by
  calc
    mu (⋃ stage ∈ stages, badSwitch stage) ≤
        ∑ stage ∈ stages, mu (badSwitch stage) := by
      exact measure_biUnion_finset_le stages badSwitch
    _ ≤ ∑ stage ∈ stages, alphaSwitch stage := by
      exact Finset.sum_le_sum (fun stage hStage =>
        hCell stage hStage)
    _ ≤ deltaSwitch := hBudget

theorem confirmation_and_terminal_switch_error_compose
    {Stage Head Look Omega : Type*}
    [MeasurableSpace Omega]
    {mu : Measure Omega}
    (stages : Finset Stage)
    (heads : Finset Head)
    (looks : Finset Look)
    (badConfirmation : Stage → Head → Look → Set Omega)
    (badSwitch : Stage → Set Omega)
    (alphaConfirmation : Stage → Head → Look → ℝ≥0∞)
    (alphaSwitch : Stage → ℝ≥0∞)
    (deltaConfirmation deltaSwitch : ℝ≥0∞)
    (hConfirmationCell :
      ∀ stage ∈ stages, ∀ head ∈ heads, ∀ look ∈ looks,
        mu (badConfirmation stage head look)
          ≤ alphaConfirmation stage head look)
    (hConfirmationBudget :
      ∑ stage ∈ stages, ∑ head ∈ heads, ∑ look ∈ looks,
        alphaConfirmation stage head look ≤ deltaConfirmation)
    (hSwitchCell : ∀ stage ∈ stages,
      mu (badSwitch stage) ≤ alphaSwitch stage)
    (hSwitchBudget :
      ∑ stage ∈ stages, alphaSwitch stage ≤ deltaSwitch) :
    mu (
      (⋃ stage ∈ stages, ⋃ head ∈ heads, ⋃ look ∈ looks,
        badConfirmation stage head look)
      ∪
      (⋃ stage ∈ stages, badSwitch stage)
    ) ≤ deltaConfirmation + deltaSwitch := by
  have hConfirmation :=
    finite_two_head_horizon_look_error_spending
      stages heads looks badConfirmation alphaConfirmation
      deltaConfirmation hConfirmationCell hConfirmationBudget
  have hSwitch :=
    finite_switch_horizon_error_spending
      stages badSwitch alphaSwitch deltaSwitch
      hSwitchCell hSwitchBudget
  calc
    mu (
      (⋃ stage ∈ stages, ⋃ head ∈ heads, ⋃ look ∈ looks,
        badConfirmation stage head look)
      ∪
      (⋃ stage ∈ stages, badSwitch stage)
    ) ≤
        mu (⋃ stage ∈ stages, ⋃ head ∈ heads, ⋃ look ∈ looks,
          badConfirmation stage head look)
        + mu (⋃ stage ∈ stages, badSwitch stage) := by
      exact measure_union_le _ _
    _ ≤ deltaConfirmation + deltaSwitch :=
      add_le_add hConfirmation hSwitch

end SCOLHKG.Measure
