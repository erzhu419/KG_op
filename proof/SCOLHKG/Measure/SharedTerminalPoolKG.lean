import Mathlib.Data.Finset.Union
import Mathlib.Probability.Moments.Basic
import SCOLHKG.Measure.PosteriorUpdateKG

namespace SCOLHKG.Measure

open MeasureTheory
open scoped MeasureTheory

noncomputable section

/-!
Implementation bridge for a history-measurable terminal action pool.

The pool is chosen from the pre-observation state.  The current terminal value
and every hypothetical posterior-update value are evaluated on that same pool.
This is the contract enforced by `SingleOLHKGAlgorithm.run` before exact-MC KG
scores are computed.
-/

universe u v w p

structure SharedTerminalPoolKG
    (State : Type u)
    (Design : Type v)
    (Observation : Type w)
    (Pool : Type p)
    [MeasurableSpace Observation] where
  terminalPool : State → Pool
  terminalValue : State → Pool → ℝ
  update : State → Design → Observation → State

def sharedTerminalPoolGain
    {State : Type u}
    {Design : Type v}
    {Observation : Type w}
    {Pool : Type p}
    [MeasurableSpace Observation]
    (kg : SharedTerminalPoolKG State Design Observation Pool)
    (state : State)
    (x : Design) : Observation → ℝ :=
  fun observation ↦
    kg.terminalValue state (kg.terminalPool state)
      - kg.terminalValue
          (kg.update state x observation)
          (kg.terminalPool state)

def sharedTerminalPoolExpectedGain
    {State : Type u}
    {Design : Type v}
    {Observation : Type w}
    {Pool : Type p}
    [MeasurableSpace Observation]
    (kg : SharedTerminalPoolKG State Design Observation Pool)
    (μ : Measure Observation)
    (state : State)
    (x : Design) : ℝ :=
  ∫ observation, sharedTerminalPoolGain kg state x observation ∂μ

theorem shared_terminal_pool_gain_uses_pre_state_pool
    {State : Type u}
    {Design : Type v}
    {Observation : Type w}
    {Pool : Type p}
    [MeasurableSpace Observation]
    (kg : SharedTerminalPoolKG State Design Observation Pool)
    (state : State)
    (x : Design)
    (observation : Observation) :
    sharedTerminalPoolGain kg state x observation =
      kg.terminalValue state (kg.terminalPool state)
        - kg.terminalValue
            (kg.update state x observation)
            (kg.terminalPool state) := by
  rfl

theorem shared_terminal_pool_expected_gain_is_integral
    {State : Type u}
    {Design : Type v}
    {Observation : Type w}
    {Pool : Type p}
    [MeasurableSpace Observation]
    (kg : SharedTerminalPoolKG State Design Observation Pool)
    (μ : Measure Observation)
    (state : State)
    (x : Design) :
    sharedTerminalPoolExpectedGain kg μ state x =
      ∫ observation,
        kg.terminalValue state (kg.terminalPool state)
          - kg.terminalValue
              (kg.update state x observation)
              (kg.terminalPool state) ∂μ := by
  rfl

theorem shared_terminal_pool_maximizer_is_one_step_optimal
    {State : Type u}
    {Design : Type v}
    {Observation : Type w}
    {Pool : Type p}
    [MeasurableSpace Observation]
    (kg : SharedTerminalPoolKG State Design Observation Pool)
    (μ : Measure Observation)
    (state : State)
    (x : Design)
    (hMax : ∀ y,
      sharedTerminalPoolExpectedGain kg μ state y
        ≤ sharedTerminalPoolExpectedGain kg μ state x) :
    ∀ y,
      sharedTerminalPoolExpectedGain kg μ state y
        ≤ sharedTerminalPoolExpectedGain kg μ state x := by
  exact hMax

variable {Design : Type v} [DecidableEq Design]

def closeExperimentActions
    (experiments frontier : Finset Design) : Finset Design :=
  experiments ∪ frontier

theorem terminal_frontier_subset_closed_experiments
    (experiments frontier : Finset Design) :
    frontier ⊆ closeExperimentActions experiments frontier := by
  exact Finset.subset_union_right

theorem original_experiments_subset_closed_experiments
    (experiments frontier : Finset Design) :
    experiments ⊆ closeExperimentActions experiments frontier := by
  exact Finset.subset_union_left

end

end SCOLHKG.Measure
