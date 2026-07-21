import SCOLHKG.Measure.ExactMCConcentration
import SCOLHKG.Real.FiniteSampleHVD

namespace SCOLHKG.Measure

open MeasureTheory ProbabilityTheory
open scoped MeasureTheory ProbabilityTheory BigOperators

/-!
Finite-replication probability bridge for the active cumulative-HVD theorem.

The deterministic oracle inequality in `Real/FiniteSampleHVD.lean` consumes a
uniformly accurate vector of variance targets. Here each target is a random
replicate-based estimator. If its centered error is sub-Gaussian with proxy
`varianceProxy / replicates`, a finite union bound yields exactly the uniform
event consumed by the active-HVD theorem. Thus target replication count is a
declared statistical resource rather than an informal calibration heuristic.
-/

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

structure ActiveHVDReplicationSchedule where
  replicates : ℕ
  perReplicateVarianceProxy : ℝ
  poolDelta : ℝ
  radius : ℝ

def ActiveHVDReplicationSchedule.Valid
    (schedule : ActiveHVDReplicationSchedule)
    (targetCount : ℕ) : Prop :=
  0 < schedule.replicates ∧
  0 < schedule.perReplicateVarianceProxy ∧
  0 ≤ schedule.radius ∧
  0 ≤ schedule.poolDelta ∧
  2 * Real.exp
      (-(schedule.radius) ^ 2 /
        (2 * (schedule.perReplicateVarianceProxy / schedule.replicates)))
    ≤ schedule.poolDelta / targetCount

def ActiveHVDReplicationSchedule.toExactMCSchedule
    (schedule : ActiveHVDReplicationSchedule) : ExactMCSchedule where
  mcSamples := schedule.replicates
  perDrawVarianceProxy := schedule.perReplicateVarianceProxy
  poolDelta := schedule.poolDelta
  radius := schedule.radius

theorem activeHVDReplicationSchedule_toExact_valid
    (schedule : ActiveHVDReplicationSchedule)
    (targetCount : ℕ)
    (hValid : schedule.Valid targetCount) :
    schedule.toExactMCSchedule.Valid targetCount := by
  exact hValid

def ActiveHVDTargetBadEvent
    {n q : ℕ}
    (design : Fin n → Fin q → ℝ)
    (observed : Ω → Fin n → ℝ)
    (oracleParameter : Fin q → ℝ)
    (radius : ℝ) : Set Ω :=
  ExactMCBadEvent
    Finset.univ
    (fun i => SCOLHKG.Real.hvdLinearPrediction
      (design i) oracleParameter)
    observed
    radius

theorem outside_activeHVDTargetBadEvent_uniform
    {n q : ℕ}
    (design : Fin n → Fin q → ℝ)
    (observed : Ω → Fin n → ℝ)
    (oracleParameter : Fin q → ℝ)
    {radius : ℝ}
    {ω : Ω}
    (hOutside : ω ∉ ActiveHVDTargetBadEvent
      design observed oracleParameter radius) :
    SCOLHKG.Real.UniformReplicatedVarianceAccuracy
      design (observed ω) oracleParameter radius := by
  have hUniform := exactMC_uniform_error_of_not_bad
    Finset.univ
    (fun i => SCOLHKG.Real.hvdLinearPrediction
      (design i) oracleParameter)
    observed
    hOutside
  intro i
  simpa [abs_sub_comm] using hUniform i (Finset.mem_univ i)

theorem activeHVDTargetBadEvent_le_poolDelta
    {n q : ℕ}
    (design : Fin n → Fin q → ℝ)
    (observed : Ω → Fin n → ℝ)
    (oracleParameter : Fin q → ℝ)
    (schedule : ActiveHVDReplicationSchedule)
    [IsFiniteMeasure μ]
    (hValid : schedule.Valid n)
    (hSubGaussian : ∀ i,
      HasSubgaussianMGF
        (fun ω => observed ω i -
          SCOLHKG.Real.hvdLinearPrediction (design i) oracleParameter)
        ⟨schedule.perReplicateVarianceProxy / schedule.replicates,
          by
            exact (div_pos hValid.2.1
              (Nat.cast_pos.mpr hValid.1)).le⟩
        μ) :
    μ.real (ActiveHVDTargetBadEvent
      design observed oracleParameter schedule.radius)
      ≤ schedule.poolDelta := by
  unfold ActiveHVDTargetBadEvent
  exact exactMC_schedule_bad_event_le_pool_delta
    (μ := μ)
    Finset.univ
    (fun i => SCOLHKG.Real.hvdLinearPrediction
      (design i) oracleParameter)
    observed
    schedule.toExactMCSchedule
    (by
      simpa using
        (activeHVDReplicationSchedule_toExact_valid schedule n hValid))
    (fun i _hi => hSubGaussian i)

end SCOLHKG.Measure
