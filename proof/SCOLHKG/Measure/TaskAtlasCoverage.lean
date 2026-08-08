import Mathlib.Probability.Distributions.Binomial
import SCOLHKG.Measure.ExactBinomialCertificate

namespace SCOLHKG.Measure

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal ProbabilityTheory unitInterval

variable {Omega : Type*} {mOmega : MeasurableSpace Omega}
  {mu : Measure Omega}

/-!
# Task-distribution atlas coverage calibration

After fitting an atlas, evaluate its hit indicator on independent calibration
tasks from the declared meta-distribution. If all `m` calibration tasks are hit,
then falsely asserting a task-level coverage probability above `pLower` has
probability at most `pLower ^ m`. Training tasks cannot be reused for this
statement.
-/

theorem unsafe_task_coverage_claim_all_success_probability_le
    (calibrationHitCount : Omega → Nat)
    (calibrationTaskCount : Nat)
    (trueTaskHitProbability claimedLowerProbability : unitInterval)
    (hLaw : HasLaw calibrationHitCount
      Bin(calibrationTaskCount, trueTaskHitProbability) mu)
    (hFalseClaim :
      trueTaskHitProbability ≤ claimedLowerProbability) :
    mu.real {
      omega | calibrationHitCount omega = calibrationTaskCount
    } ≤ (claimedLowerProbability : Real) ^ calibrationTaskCount := by
  exact unsafe_all_success_certificate_probability_le
    calibrationHitCount
    calibrationTaskCount
    trueTaskHitProbability
    claimedLowerProbability
    hLaw
    hFalseClaim

theorem calibrated_task_coverage_false_claim_le_delta
    (calibrationHitCount : Omega → Nat)
    (calibrationTaskCount : Nat)
    (trueTaskHitProbability claimedLowerProbability : unitInterval)
    (delta : Real)
    (hLaw : HasLaw calibrationHitCount
      Bin(calibrationTaskCount, trueTaskHitProbability) mu)
    (hFalseClaim :
      trueTaskHitProbability ≤ claimedLowerProbability)
    (hSpend :
      (claimedLowerProbability : Real) ^ calibrationTaskCount ≤ delta) :
    mu.real {
      omega | calibrationHitCount omega = calibrationTaskCount
    } ≤ delta := by
  exact (unsafe_task_coverage_claim_all_success_probability_le
    calibrationHitCount
    calibrationTaskCount
    trueTaskHitProbability
    claimedLowerProbability
    hLaw
    hFalseClaim).trans hSpend

end SCOLHKG.Measure
