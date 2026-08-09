import Mathlib.Probability.Distributions.Binomial
import SCOLHKG.Measure.ExactBinomialCertificate
import SCOLHKG.Measure.GPKernelConfidence
import SCOLHKG.Measure.ResidualSquareConcentration

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

/-!
The all-success result above is exact but applies only to a perfect calibration
sample.  The following bounded-hit construction gives a finite-task guarantee
for every empirical hit rate.  The atlas is fixed before these tasks are drawn;
the random variables below are its held-out task hit indicators encoded in
`[0,1]`.  No claim is made outside their common registered task distribution.
-/

noncomputable def taskCoverageMeanError
    {Task : Type*} [Fintype Task]
    (hit : Task → Omega → ℝ) : Omega → ℝ :=
  finiteKernelPosteriorError
    Finset.univ
    (fun (_ : Unit) (_ : Task) => 1 / (Fintype.card Task : ℝ))
    (fun task omega => hit task omega - mu[hit task])
    ()

noncomputable def taskCoverageMeanProxy
    (Task : Type*) [Fintype Task] : NNReal :=
  finiteKernelSubGaussianParam
    Finset.univ
    (fun (_ : Unit) (_ : Task) => 1 / (Fintype.card Task : ℝ))
    (fun (_ : Task) => boundedResidualSquareConstant 0 1)
    ()

noncomputable def inverseFourTaskCountProxy
    (Task : Type*) [Fintype Task] : NNReal :=
  ⟨1 / (4 * (Fintype.card Task : ℝ)), by positivity⟩

theorem taskCoverageMeanProxy_eq_inverseFourTaskCountProxy
    (Task : Type*) [Fintype Task] [Nonempty Task] :
    taskCoverageMeanProxy Task = inverseFourTaskCountProxy Task := by
  have hCardNNReal : (Fintype.card Task : NNReal) ≠ 0 := by
    exact_mod_cast Fintype.card_ne_zero
  have hWeight :
      squareNNReal (1 / (Fintype.card Task : ℝ))
        = ((Fintype.card Task : NNReal)⁻¹) ^ 2 := by
    apply NNReal.eq
    simp [squareNNReal]
    rfl
  have hBound :
      boundedResidualSquareConstant 0 1 = (4 : NNReal)⁻¹ := by
    apply NNReal.eq
    norm_num [boundedResidualSquareConstant]
  have hTarget :
      inverseFourTaskCountProxy Task
        = ((4 : NNReal) * (Fintype.card Task : NNReal))⁻¹ := by
    apply NNReal.eq
    simp [inverseFourTaskCountProxy]
    rfl
  unfold taskCoverageMeanProxy finiteKernelSubGaussianParam
  rw [Finset.sum_const]
  simp only [Finset.card_univ, nsmul_eq_mul]
  rw [hWeight, hBound, hTarget]
  field_simp [hCardNNReal]

theorem taskCoverageMeanError_subGaussian
    {Task : Type*} [Fintype Task] [Nonempty Task]
    [IsProbabilityMeasure mu]
    (hit : Task → Omega → ℝ)
    (hIndependent : iIndepFun hit mu)
    (hMeasurable : ∀ task, AEMeasurable (hit task) mu)
    (hUnit : ∀ task, ∀ᵐ omega ∂mu, hit task omega ∈ Set.Icc (0 : ℝ) 1) :
    HasSubgaussianMGF
      (taskCoverageMeanError (mu := mu) hit)
      (taskCoverageMeanProxy Task)
      mu := by
  let centered : Task → Omega → ℝ :=
    fun task omega => hit task omega - mu[hit task]
  have hIndependentCentered : iIndepFun centered mu := by
    have hComposed := hIndependent.comp
      (fun task value => value - mu[hit task])
      (by
        intro task
        fun_prop)
    simpa [centered, Function.comp_def] using hComposed
  have hEach : ∀ task ∈ (Finset.univ : Finset Task),
      HasSubgaussianMGF
        (centered task)
        (boundedResidualSquareConstant 0 1)
        mu := by
    intro task _htask
    simpa [centered, boundedResidualSquareConstant] using
      (hasSubgaussianMGF_of_mem_Icc
        (μ := mu)
        (X := hit task)
        (a := (0 : ℝ))
        (b := (1 : ℝ))
        (hMeasurable task)
        (hUnit task))
  simpa [taskCoverageMeanError, taskCoverageMeanProxy, centered] using
    (finiteKernelPosteriorError_subGaussian
      (μ := mu)
      (active := (Finset.univ : Finset Task))
      (weight := fun (_ : Unit) (_ : Task) =>
        1 / (Fintype.card Task : ℝ))
      (noise := centered)
      (c := fun (_ : Task) => boundedResidualSquareConstant 0 1)
      (x := ())
      hIndependentCentered
      hEach)

theorem taskCoverageMeanError_abs_tail_le
    {Task : Type*} [Fintype Task] [Nonempty Task]
    [IsProbabilityMeasure mu]
    (hit : Task → Omega → ℝ)
    (radius : ℝ)
    (hIndependent : iIndepFun hit mu)
    (hMeasurable : ∀ task, AEMeasurable (hit task) mu)
    (hUnit : ∀ task, ∀ᵐ omega ∂mu, hit task omega ∈ Set.Icc (0 : ℝ) 1)
    (hRadius : 0 ≤ radius) :
    mu.real {
      omega | radius ≤ |taskCoverageMeanError (mu := mu) hit omega|
    } ≤
      2 * Real.exp (
        -radius ^ 2 / (2 * (taskCoverageMeanProxy Task : ℝ))) := by
  exact centeredSubGaussian_abs_bad_event_le
    (μ := mu)
    (X := taskCoverageMeanError (mu := mu) hit)
    (c := taskCoverageMeanProxy Task)
    (radius := radius)
      (taskCoverageMeanError_subGaussian
        (mu := mu) hit hIndependent hMeasurable hUnit)
      hRadius

theorem taskCoverageMeanError_abs_tail_inverseFourTaskCount
    {Task : Type*} [Fintype Task] [Nonempty Task]
    [IsProbabilityMeasure mu]
    (hit : Task → Omega → ℝ)
    (radius : ℝ)
    (hIndependent : iIndepFun hit mu)
    (hMeasurable : ∀ task, AEMeasurable (hit task) mu)
    (hUnit : ∀ task, ∀ᵐ omega ∂mu, hit task omega ∈ Set.Icc (0 : ℝ) 1)
    (hRadius : 0 ≤ radius) :
    mu.real {
      omega | radius ≤ |taskCoverageMeanError (mu := mu) hit omega|
    } ≤
      2 * Real.exp (
        -radius ^ 2 / (2 * (inverseFourTaskCountProxy Task : ℝ))) := by
  rw [← taskCoverageMeanProxy_eq_inverseFourTaskCountProxy]
  exact taskCoverageMeanError_abs_tail_le
    (mu := mu) hit radius hIndependent hMeasurable hUnit hRadius

theorem taskCoverageMeanError_eq_empirical_sub_expectation
    {Task : Type*} [Fintype Task] [Nonempty Task]
    (hit : Task → Omega → ℝ)
    (omega : Omega) :
    taskCoverageMeanError (mu := mu) hit omega =
      (∑ task, hit task omega) / Fintype.card Task
        - (∑ task, mu[hit task]) / Fintype.card Task := by
  have hCard : (Fintype.card Task : ℝ) ≠ 0 := by
    exact_mod_cast Fintype.card_ne_zero
  unfold taskCoverageMeanError finiteKernelPosteriorError
  simp_rw [mul_sub]
  rw [Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  field_simp

/-!
The registered stress experiment is a fixed collection of regime strata, not
an IID draw from one pooled law.  The weighted form below needs independence
and boundedness only; task distributions may differ.  Setting every task in
stratum `s` to weight `w_s / n_s` gives the usual stratified Hoeffding proxy
`(1/4) * sum_s w_s^2 / n_s`.
-/

noncomputable def weightedTaskCoverageError
    {Task : Type*} [Fintype Task]
    (weight : Task → ℝ) (hit : Task → Omega → ℝ) : Omega → ℝ :=
  finiteKernelPosteriorError
    Finset.univ
    (fun (_ : Unit) task => weight task)
    (fun task omega => hit task omega - mu[hit task])
    ()

noncomputable def weightedTaskCoverageProxy
    {Task : Type*} [Fintype Task]
    (weight : Task → ℝ) : NNReal :=
  finiteKernelSubGaussianParam
    Finset.univ
    (fun (_ : Unit) task => weight task)
    (fun (_ : Task) => boundedResidualSquareConstant 0 1)
    ()

theorem weightedTaskCoverageError_subGaussian
    {Task : Type*} [Fintype Task]
    [IsProbabilityMeasure mu]
    (weight : Task → ℝ)
    (hit : Task → Omega → ℝ)
    (hIndependent : iIndepFun hit mu)
    (hMeasurable : ∀ task, AEMeasurable (hit task) mu)
    (hUnit : ∀ task, ∀ᵐ omega ∂mu, hit task omega ∈ Set.Icc (0 : ℝ) 1) :
    HasSubgaussianMGF
      (weightedTaskCoverageError (mu := mu) weight hit)
      (weightedTaskCoverageProxy weight)
      mu := by
  let centered : Task → Omega → ℝ :=
    fun task omega => hit task omega - mu[hit task]
  have hIndependentCentered : iIndepFun centered mu := by
    have hComposed := hIndependent.comp
      (fun task value => value - mu[hit task])
      (by
        intro task
        fun_prop)
    simpa [centered, Function.comp_def] using hComposed
  have hEach : ∀ task ∈ (Finset.univ : Finset Task),
      HasSubgaussianMGF
        (centered task)
        (boundedResidualSquareConstant 0 1)
        mu := by
    intro task _hTask
    simpa [centered, boundedResidualSquareConstant] using
      (hasSubgaussianMGF_of_mem_Icc
        (μ := mu)
        (X := hit task)
        (a := (0 : ℝ))
        (b := (1 : ℝ))
        (hMeasurable task)
        (hUnit task))
  simpa [weightedTaskCoverageError, weightedTaskCoverageProxy, centered] using
    (finiteKernelPosteriorError_subGaussian
      (μ := mu)
      (active := (Finset.univ : Finset Task))
      (weight := fun (_ : Unit) task => weight task)
      (noise := centered)
      (c := fun (_ : Task) => boundedResidualSquareConstant 0 1)
      (x := ())
      hIndependentCentered
      hEach)

theorem weightedTaskCoverageError_abs_tail_le
    {Task : Type*} [Fintype Task]
    [IsProbabilityMeasure mu]
    (weight : Task → ℝ)
    (hit : Task → Omega → ℝ)
    (radius : ℝ)
    (hIndependent : iIndepFun hit mu)
    (hMeasurable : ∀ task, AEMeasurable (hit task) mu)
    (hUnit : ∀ task, ∀ᵐ omega ∂mu, hit task omega ∈ Set.Icc (0 : ℝ) 1)
    (hRadius : 0 ≤ radius) :
    mu.real {omega |
      radius ≤ |weightedTaskCoverageError (mu := mu) weight hit omega|}
      ≤ 2 * Real.exp (
        -radius ^ 2 / (2 * (weightedTaskCoverageProxy weight : ℝ))) := by
  exact centeredSubGaussian_abs_bad_event_le
    (μ := mu)
    (X := weightedTaskCoverageError (mu := mu) weight hit)
    (c := weightedTaskCoverageProxy weight)
    (radius := radius)
    (weightedTaskCoverageError_subGaussian
      (mu := mu) weight hit hIndependent hMeasurable hUnit)
    hRadius

end SCOLHKG.Measure
