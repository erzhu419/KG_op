import SCOLHKG.Real.ConditionalVariance
import SCOLHKG.Real.CumulativeRisk

namespace SCOLHKG.Real

open scoped BigOperators

/-!
Trajectory occupancy-risk decomposition.

For a policy/design, the trajectory index is the finite latent state.  Each
trajectory has a fixed cumulative-risk variance decomposition.  The policy-level
variance is the law of total variance over trajectories: expected fixed
trajectory risk plus explained between-trajectory variation.  A state/occupancy
encoder approximates the expected fixed trajectory risk by one cumulative-risk
object; the difference is the occupancy remainder.
-/

def trajectoryExpectedFixedRisk
    {Trajectory : Type*}
    [Fintype Trajectory]
    (p : Trajectory → ℝ)
    (risk : Trajectory → CumulativeRisk) : ℝ :=
  finiteExpectedConditionalVariance p (fun τ ↦ totalVariance (risk τ))

def policyTrajectoryTotalVariance
    {Trajectory : Type*}
    [Fintype Trajectory]
    (p mean : Trajectory → ℝ)
    (risk : Trajectory → CumulativeRisk) : ℝ :=
  finiteTotalVariance p mean (fun τ ↦ totalVariance (risk τ))

def occupancyRemainder
    {Trajectory : Type*}
    [Fintype Trajectory]
    (p : Trajectory → ℝ)
    (risk : Trajectory → CumulativeRisk)
    (occupancyRisk : CumulativeRisk) : ℝ :=
  trajectoryExpectedFixedRisk p risk - totalVariance occupancyRisk

theorem expected_fixed_trajectory_risk_decomposes
    {Trajectory : Type*}
    [Fintype Trajectory]
    (p : Trajectory → ℝ)
    (risk : Trajectory → CumulativeRisk) :
    trajectoryExpectedFixedRisk p risk =
      ∑ τ, (p τ *
        (risk τ).floor
          + p τ * independentRisk (risk τ)
          + p τ * sharedShockRisk (risk τ)
          + p τ * linearRisk (risk τ)) := by
  unfold trajectoryExpectedFixedRisk finiteExpectedConditionalVariance
  apply Finset.sum_congr rfl
  intro τ _hτ
  change p τ * totalVariance (risk τ) =
    p τ * (risk τ).floor
      + p τ * independentRisk (risk τ)
      + p τ * sharedShockRisk (risk τ)
      + p τ * linearRisk (risk τ)
  rw [fixedTrajectoryVarianceDecomposition]
  ring

theorem policy_trajectory_occupancy_decomposition
    {Trajectory : Type*}
    [Fintype Trajectory]
    (p mean : Trajectory → ℝ)
    (risk : Trajectory → CumulativeRisk)
    (occupancyRisk : CumulativeRisk) :
    policyTrajectoryTotalVariance p mean risk =
      totalVariance occupancyRisk
        + occupancyRemainder p risk occupancyRisk
        + finiteExplainedVariance p mean := by
  unfold policyTrajectoryTotalVariance occupancyRemainder trajectoryExpectedFixedRisk
  rw [finite_law_total_variance]
  ring

theorem occupancy_component_le_policy_total_variance
    {Trajectory : Type*}
    [Fintype Trajectory]
    {p mean : Trajectory → ℝ}
    {risk : Trajectory → CumulativeRisk}
    {occupancyRisk : CumulativeRisk}
    (hp : ∀ τ, 0 ≤ p τ) :
    totalVariance occupancyRisk + occupancyRemainder p risk occupancyRisk
      ≤ policyTrajectoryTotalVariance p mean risk := by
  rw [policy_trajectory_occupancy_decomposition]
  have hexpl : 0 ≤ finiteExplainedVariance p mean := by
    unfold finiteExplainedVariance
    exact Finset.sum_nonneg (by
      intro τ _hτ
      exact mul_nonneg (hp τ) (sq_nonneg _))
  linarith

def OccupancyApproximationBound
    {Trajectory : Type*}
    [Fintype Trajectory]
    (p : Trajectory → ℝ)
    (risk : Trajectory → CumulativeRisk)
    (occupancyRisk : CumulativeRisk)
    (tol : ℝ) : Prop :=
  |occupancyRemainder p risk occupancyRisk| ≤ tol

theorem policy_trajectory_occupancy_decomposition_with_bound
    {Trajectory : Type*}
    [Fintype Trajectory]
    (p mean : Trajectory → ℝ)
    (risk : Trajectory → CumulativeRisk)
    (occupancyRisk : CumulativeRisk)
    {tol : ℝ}
    (hBound : OccupancyApproximationBound p risk occupancyRisk tol) :
    |policyTrajectoryTotalVariance p mean risk
        - (totalVariance occupancyRisk + finiteExplainedVariance p mean)|
      ≤ tol := by
  have hEq :
      policyTrajectoryTotalVariance p mean risk
        - (totalVariance occupancyRisk + finiteExplainedVariance p mean)
        = occupancyRemainder p risk occupancyRisk := by
    rw [policy_trajectory_occupancy_decomposition]
    ring
  rw [hEq]
  exact hBound

end SCOLHKG.Real
