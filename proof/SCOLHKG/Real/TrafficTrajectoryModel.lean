import Mathlib

namespace SCOLHKG.Real

/-!
Finite fresh-seed traffic trajectory risk model.

This file formalizes the traffic-log side of the plan at the level currently
implemented by `TrafficTrajectoryEncoder`: state-action occupancy cells provide
local queue/wait/flow exposure, and demand-shock coordinates provide shared
shock exposure.  The empirical CSV parser is an implementation of these finite
objects once real fresh-seed trajectory logs are available.
-/

open scoped BigOperators

structure TrafficTrajectoryRiskModel
    (Cell Shock : Type*)
    [DecidableEq Cell] [DecidableEq Shock] where
  cells : Finset Cell
  shocks : Finset Shock
  occupancy : Cell → ℝ
  queue : Cell → ℝ
  wait : Cell → ℝ
  flow : Cell → ℝ
  queueWeight : ℝ
  waitWeight : ℝ
  flowWeight : ℝ
  sharedExposure : Shock → ℝ
  sharedCovariance : Shock → Shock → ℝ
  sharedLinear : Shock → ℝ
  floor : ℝ

namespace TrafficTrajectoryRiskModel

variable {Cell Shock : Type*} [DecidableEq Cell] [DecidableEq Shock]

def localRisk (m : TrafficTrajectoryRiskModel Cell Shock) : ℝ :=
  ∑ c ∈ m.cells,
    m.occupancy c *
      (m.queueWeight * m.queue c ^ 2
        + m.waitWeight * m.wait c ^ 2
        + m.flowWeight * m.flow c ^ 2)

def sharedShockRisk (m : TrafficTrajectoryRiskModel Cell Shock) : ℝ :=
  ∑ i ∈ m.shocks, ∑ j ∈ m.shocks,
    m.sharedExposure i * m.sharedCovariance i j * m.sharedExposure j

def linearShockRisk (m : TrafficTrajectoryRiskModel Cell Shock) : ℝ :=
  ∑ i ∈ m.shocks, m.sharedExposure i * m.sharedLinear i

def totalRisk (m : TrafficTrajectoryRiskModel Cell Shock) : ℝ :=
  m.floor + m.localRisk + m.sharedShockRisk + m.linearShockRisk

theorem totalRisk_decomposition
    (m : TrafficTrajectoryRiskModel Cell Shock) :
    m.totalRisk =
      m.floor + m.localRisk + m.sharedShockRisk + m.linearShockRisk := by
  rfl

theorem sharedShock_omission_underestimates
    (m : TrafficTrajectoryRiskModel Cell Shock)
    (hshared : 0 ≤ m.sharedShockRisk) :
    m.floor + m.localRisk + m.linearShockRisk ≤ m.totalRisk := by
  unfold totalRisk
  linarith

theorem localRisk_nonnegative
    (m : TrafficTrajectoryRiskModel Cell Shock)
    (hOcc : ∀ c ∈ m.cells, 0 ≤ m.occupancy c)
    (hQ : 0 ≤ m.queueWeight)
    (hW : 0 ≤ m.waitWeight)
    (hF : 0 ≤ m.flowWeight) :
    0 ≤ m.localRisk := by
  unfold localRisk
  exact Finset.sum_nonneg (by
    intro c hc
    have hinside :
        0 ≤
          m.queueWeight * m.queue c ^ 2
            + m.waitWeight * m.wait c ^ 2
            + m.flowWeight * m.flow c ^ 2 := by
      positivity
    exact mul_nonneg (hOcc c hc) hinside)

theorem linearShockRisk_nonnegative
    (m : TrafficTrajectoryRiskModel Cell Shock)
    (hExposure : ∀ i ∈ m.shocks, 0 ≤ m.sharedExposure i)
    (hLinear : ∀ i ∈ m.shocks, 0 ≤ m.sharedLinear i) :
    0 ≤ m.linearShockRisk := by
  unfold linearShockRisk
  exact Finset.sum_nonneg (by
    intro i hi
    exact mul_nonneg (hExposure i hi) (hLinear i hi))

theorem totalRisk_nonnegative
    (m : TrafficTrajectoryRiskModel Cell Shock)
    (hfloor : 0 ≤ m.floor)
    (hlocal : 0 ≤ m.localRisk)
    (hshared : 0 ≤ m.sharedShockRisk)
    (hlinear : 0 ≤ m.linearShockRisk) :
    0 ≤ m.totalRisk := by
  unfold totalRisk
  positivity

structure FreshSeedCoverage
    (Policy Seed : Type*)
    [DecidableEq Policy] [DecidableEq Seed] where
  policies : Finset Policy
  seeds : Finset Seed
  evaluated : Policy → Seed → Prop

def FreshSeedCoverage.Complete
    {Policy Seed : Type*}
    [DecidableEq Policy] [DecidableEq Seed]
    (c : FreshSeedCoverage Policy Seed) : Prop :=
  ∀ p ∈ c.policies, ∀ s ∈ c.seeds, c.evaluated p s

theorem complete_fresh_seed_coverage_has_policy_seed_eval
    {Policy Seed : Type*}
    [DecidableEq Policy] [DecidableEq Seed]
    (c : FreshSeedCoverage Policy Seed)
    (h : c.Complete)
    {p : Policy}
    {s : Seed}
    (hp : p ∈ c.policies)
    (hs : s ∈ c.seeds) :
    c.evaluated p s := by
  exact h p hp s hs

end TrafficTrajectoryRiskModel

structure TrafficLogSchemaRow where
  policyId : String
  seed : String
  time : ℕ
  state : String
  action : String
  occupancy : ℝ
  queue : ℝ
  wait : ℝ
  flow : ℝ
  demandShock : ℝ

namespace TrafficLogSchemaRow

def cellKey (r : TrafficLogSchemaRow) : String :=
  r.state ++ "|" ++ r.action

def localExposure (r : TrafficLogSchemaRow) : ℝ × ℝ × ℝ :=
  (r.queue, r.wait, max r.flow 0)

def sharedExposure (r : TrafficLogSchemaRow) : ℝ :=
  r.demandShock

def hasRequiredFields (r : TrafficLogSchemaRow) : Prop :=
  r.policyId ≠ "" ∧ r.state ≠ "" ∧ r.action ≠ ""

theorem localExposure_first_eq_queue
    (r : TrafficLogSchemaRow) :
    r.localExposure.1 = r.queue := by
  rfl

theorem localExposure_second_eq_wait
    (r : TrafficLogSchemaRow) :
    r.localExposure.2.1 = r.wait := by
  rfl

theorem localExposure_flow_nonnegative
    (r : TrafficLogSchemaRow) :
    0 ≤ r.localExposure.2.2 := by
  unfold localExposure
  exact le_max_right r.flow 0

theorem required_fields_give_nonempty_policy_state_action
    (r : TrafficLogSchemaRow)
    (h : r.hasRequiredFields) :
    r.policyId ≠ "" ∧ r.state ≠ "" ∧ r.action ≠ "" := by
  exact h

end TrafficLogSchemaRow

end SCOLHKG.Real
