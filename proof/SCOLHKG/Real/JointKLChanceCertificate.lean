import Mathlib

namespace SCOLHKG.Real

open scoped BigOperators

/-!
The legacy task-robust certificate maximized the posterior mean, epistemic
variance, and aleatoric variance under three independent task laws.  The
implementation now uses one common task law and a square-root tangent
envelope.  This file records the deterministic inequalities used by that
implementation; the PAC-Bayes/KL event supplying the robust expectation is
formalized separately in `SCOLHKG.Measure.TaskPACBayes`.
-/

noncomputable def taskWeightedMean {k : ℕ}
    (q value : Fin k → ℝ) : ℝ :=
  ∑ i, q i * value i

noncomputable def taskWeightedVariance {k : ℕ}
    (q value : Fin k → ℝ) : ℝ :=
  taskWeightedMean q fun i =>
    (value i - taskWeightedMean q value) ^ 2

noncomputable def centeredTaskSecondMoment {k : ℕ}
    (q value : Fin k → ℝ) (center : ℝ) : ℝ :=
  taskWeightedMean q fun i => (value i - center) ^ 2

theorem taskWeightedMean_nonnegative {k : ℕ}
    (q value : Fin k → ℝ)
    (hQ : ∀ i, 0 ≤ q i)
    (hValue : ∀ i, 0 ≤ value i) :
    0 ≤ taskWeightedMean q value := by
  unfold taskWeightedMean
  exact Finset.sum_nonneg fun i _ => mul_nonneg (hQ i) (hValue i)

theorem taskWeightedMean_add {k : ℕ}
    (q first second : Fin k → ℝ) :
    taskWeightedMean q (fun i => first i + second i)
      = taskWeightedMean q first + taskWeightedMean q second := by
  unfold taskWeightedMean
  simp only [mul_add, Finset.sum_add_distrib]

theorem taskWeightedVariance_nonnegative {k : ℕ}
    (q value : Fin k → ℝ)
    (hQ : ∀ i, 0 ≤ q i) :
    0 ≤ taskWeightedVariance q value := by
  unfold taskWeightedVariance
  apply taskWeightedMean_nonnegative
  · exact hQ
  · intro i
    exact sq_nonneg _

theorem sqrt_le_tangent (variance scale : ℝ)
    (hVariance : 0 ≤ variance)
    (hScale : 0 < scale) :
    Real.sqrt variance ≤ variance / (2 * scale) + scale / 2 := by
  have hTwoScale : 0 < 2 * scale := by positivity
  have hSquare : 0 ≤ (Real.sqrt variance - scale) ^ 2 := sq_nonneg _
  have hSqrtSquare : (Real.sqrt variance) ^ 2 = variance := by
    exact Real.sq_sqrt hVariance
  calc
    Real.sqrt variance ≤ (variance + scale ^ 2) / (2 * scale) := by
      apply (le_div_iff₀ hTwoScale).2
      nlinarith
    _ = variance / (2 * scale) + scale / 2 := by
      field_simp

theorem taskWeightedVariance_le_centeredSecondMoment {k : ℕ}
    (q value : Fin k → ℝ) (center : ℝ)
    (hMass : ∑ i, q i = 1) :
    taskWeightedVariance q value
      ≤ centeredTaskSecondMoment q value center := by
  let mean := taskWeightedMean q value
  have hCentered : ∑ i, q i * (value i - mean) = 0 := by
    simp only [mul_sub, Finset.sum_sub_distrib]
    rw [show (∑ i, q i * value i) = mean by rfl]
    rw [← Finset.sum_mul]
    rw [hMass]
    ring
  have hIdentity :
      centeredTaskSecondMoment q value center
        = taskWeightedVariance q value + (mean - center) ^ 2 := by
    unfold centeredTaskSecondMoment taskWeightedVariance taskWeightedMean
    change
      (∑ i, q i * (value i - center) ^ 2)
        = (∑ i, q i * (value i - mean) ^ 2) + (mean - center) ^ 2
    calc
      (∑ i, q i * (value i - center) ^ 2)
          = ∑ i, (
              q i * (value i - mean) ^ 2
              + 2 * (mean - center) * (q i * (value i - mean))
              + q i * (mean - center) ^ 2) := by
            apply Finset.sum_congr rfl
            intro i hi
            ring
      _ = (∑ i, q i * (value i - mean) ^ 2)
            + 2 * (mean - center) *
                (∑ i, q i * (value i - mean))
            + (∑ i, q i) * (mean - center) ^ 2 := by
            simp only [Finset.sum_add_distrib]
            rw [Finset.mul_sum]
            rw [Finset.sum_mul]
      _ = (∑ i, q i * (value i - mean) ^ 2)
            + (mean - center) ^ 2 := by
            rw [hCentered, hMass]
            ring
  rw [hIdentity]
  exact le_add_of_nonneg_right (sq_nonneg _)

noncomputable def taskChanceMargin
    (mean epistemic aleatoric betaRadius zRadius threshold : ℝ) : ℝ :=
  mean
    + betaRadius * Real.sqrt epistemic
    + zRadius * Real.sqrt aleatoric
    - threshold

noncomputable def jointTaskTangentUpper
    (mean epistemicUpper aleatoric betaRadius zRadius threshold
      epistemicScale aleatoricScale : ℝ) : ℝ :=
  mean
    + betaRadius * (
        epistemicUpper / (2 * epistemicScale) + epistemicScale / 2)
    + zRadius * (
        aleatoric / (2 * aleatoricScale) + aleatoricScale / 2)
    - threshold

noncomputable def jointTaskPayoff {k : ℕ}
    (mean epistemicPayoff aleatoric : Fin k → ℝ)
    (betaRadius zRadius epistemicScale aleatoricScale : ℝ)
    (i : Fin k) : ℝ :=
  mean i
    + betaRadius * epistemicPayoff i / (2 * epistemicScale)
    + zRadius * aleatoric i / (2 * aleatoricScale)

theorem weighted_jointTaskPayoff {k : ℕ}
    (q mean epistemicPayoff aleatoric : Fin k → ℝ)
    (betaRadius zRadius epistemicScale aleatoricScale : ℝ) :
    taskWeightedMean q (jointTaskPayoff mean epistemicPayoff aleatoric
      betaRadius zRadius epistemicScale aleatoricScale)
      = taskWeightedMean q mean
        + betaRadius * taskWeightedMean q epistemicPayoff
            / (2 * epistemicScale)
        + zRadius * taskWeightedMean q aleatoric
            / (2 * aleatoricScale) := by
  unfold taskWeightedMean jointTaskPayoff
  calc
    (∑ i, q i *
        (mean i
          + betaRadius * epistemicPayoff i / (2 * epistemicScale)
          + zRadius * aleatoric i / (2 * aleatoricScale)))
        = ∑ i, (
            q i * mean i
              + betaRadius / (2 * epistemicScale)
                  * (q i * epistemicPayoff i)
              + zRadius / (2 * aleatoricScale)
                  * (q i * aleatoric i)) := by
            apply Finset.sum_congr rfl
            intro i hi
            ring
    _ = (∑ i, q i * mean i)
          + betaRadius / (2 * epistemicScale)
              * (∑ i, q i * epistemicPayoff i)
          + zRadius / (2 * aleatoricScale)
              * (∑ i, q i * aleatoric i) := by
            simp only [Finset.sum_add_distrib]
            rw [Finset.mul_sum, Finset.mul_sum]
    _ = (∑ i, q i * mean i)
          + betaRadius * (∑ i, q i * epistemicPayoff i)
              / (2 * epistemicScale)
          + zRadius * (∑ i, q i * aleatoric i)
              / (2 * aleatoricScale) := by
            ring

theorem taskChanceMargin_le_jointTangentUpper
    (mean epistemic epistemicUpper aleatoric betaRadius zRadius threshold
      epistemicScale aleatoricScale : ℝ)
    (hEpistemic : 0 ≤ epistemic)
    (hEpistemicUpper : epistemic ≤ epistemicUpper)
    (hAleatoric : 0 ≤ aleatoric)
    (hBeta : 0 ≤ betaRadius)
    (hZ : 0 ≤ zRadius)
    (hEpistemicScale : 0 < epistemicScale)
    (hAleatoricScale : 0 < aleatoricScale) :
    taskChanceMargin mean epistemic aleatoric betaRadius zRadius threshold
      ≤ jointTaskTangentUpper mean epistemicUpper aleatoric betaRadius
          zRadius threshold epistemicScale aleatoricScale := by
  have hEpistemicUpperNonnegative : 0 ≤ epistemicUpper :=
    hEpistemic.trans hEpistemicUpper
  have hEpiSqrt :
      Real.sqrt epistemic
        ≤ epistemicUpper / (2 * epistemicScale) + epistemicScale / 2 := by
    calc
      Real.sqrt epistemic ≤ Real.sqrt epistemicUpper :=
        Real.sqrt_le_sqrt hEpistemicUpper
      _ ≤ epistemicUpper / (2 * epistemicScale) + epistemicScale / 2 :=
        sqrt_le_tangent epistemicUpper epistemicScale
          hEpistemicUpperNonnegative hEpistemicScale
  have hAleaSqrt :
      Real.sqrt aleatoric
        ≤ aleatoric / (2 * aleatoricScale) + aleatoricScale / 2 :=
    sqrt_le_tangent aleatoric aleatoricScale hAleatoric hAleatoricScale
  unfold taskChanceMargin jointTaskTangentUpper
  nlinarith [
    mul_le_mul_of_nonneg_left hEpiSqrt hBeta,
    mul_le_mul_of_nonneg_left hAleaSqrt hZ
  ]

theorem shared_task_law_chance_le_robust_tangent {k : ℕ}
    (q mean expertEpistemic expertAleatoric : Fin k → ℝ)
    (center betaRadius zRadius threshold epistemicScale aleatoricScale
      robustPayoffUpper : ℝ)
    (hMass : ∑ i, q i = 1)
    (hQ : ∀ i, 0 ≤ q i)
    (hExpertEpistemic : ∀ i, 0 ≤ expertEpistemic i)
    (hExpertAleatoric : ∀ i, 0 ≤ expertAleatoric i)
    (hBeta : 0 ≤ betaRadius)
    (hZ : 0 ≤ zRadius)
    (hEpistemicScale : 0 < epistemicScale)
    (hAleatoricScale : 0 < aleatoricScale)
    (hRobustPayoff :
      taskWeightedMean q (jointTaskPayoff mean
        (fun i => expertEpistemic i + (mean i - center) ^ 2)
        expertAleatoric betaRadius zRadius epistemicScale aleatoricScale)
        ≤ robustPayoffUpper) :
    taskChanceMargin
        (taskWeightedMean q mean)
        (taskWeightedMean q expertEpistemic + taskWeightedVariance q mean)
        (taskWeightedMean q expertAleatoric)
        betaRadius zRadius threshold
      ≤ robustPayoffUpper
          + betaRadius * epistemicScale / 2
          + zRadius * aleatoricScale / 2
          - threshold := by
  let epistemicPayoff : Fin k → ℝ := fun i =>
    expertEpistemic i + (mean i - center) ^ 2
  have hEpistemicMean : 0 ≤ taskWeightedMean q expertEpistemic :=
    taskWeightedMean_nonnegative q expertEpistemic hQ hExpertEpistemic
  have hVariance : 0 ≤ taskWeightedVariance q mean :=
    taskWeightedVariance_nonnegative q mean hQ
  have hAleatoricMean : 0 ≤ taskWeightedMean q expertAleatoric :=
    taskWeightedMean_nonnegative q expertAleatoric hQ hExpertAleatoric
  have hVarianceUpper :=
    taskWeightedVariance_le_centeredSecondMoment q mean center hMass
  have hEpistemicPayoffIdentity :
      taskWeightedMean q epistemicPayoff
        = taskWeightedMean q expertEpistemic
          + centeredTaskSecondMoment q mean center := by
    unfold epistemicPayoff centeredTaskSecondMoment
    exact taskWeightedMean_add q expertEpistemic
      (fun i => (mean i - center) ^ 2)
  have hEpistemicUpper :
      taskWeightedMean q expertEpistemic + taskWeightedVariance q mean
        ≤ taskWeightedMean q epistemicPayoff := by
    rw [hEpistemicPayoffIdentity]
    linarith
  have hTangent := taskChanceMargin_le_jointTangentUpper
    (taskWeightedMean q mean)
    (taskWeightedMean q expertEpistemic + taskWeightedVariance q mean)
    (taskWeightedMean q epistemicPayoff)
    (taskWeightedMean q expertAleatoric)
    betaRadius zRadius threshold epistemicScale aleatoricScale
    (add_nonneg hEpistemicMean hVariance)
    hEpistemicUpper hAleatoricMean hBeta hZ
    hEpistemicScale hAleatoricScale
  calc
    taskChanceMargin
        (taskWeightedMean q mean)
        (taskWeightedMean q expertEpistemic + taskWeightedVariance q mean)
        (taskWeightedMean q expertAleatoric)
        betaRadius zRadius threshold
        ≤ jointTaskTangentUpper
            (taskWeightedMean q mean)
            (taskWeightedMean q epistemicPayoff)
            (taskWeightedMean q expertAleatoric)
            betaRadius zRadius threshold epistemicScale aleatoricScale :=
          hTangent
    _ = taskWeightedMean q (jointTaskPayoff mean epistemicPayoff
          expertAleatoric betaRadius zRadius epistemicScale aleatoricScale)
          + betaRadius * epistemicScale / 2
          + zRadius * aleatoricScale / 2
          - threshold := by
        rw [weighted_jointTaskPayoff]
        unfold jointTaskTangentUpper
        ring
    _ ≤ robustPayoffUpper
          + betaRadius * epistemicScale / 2
          + zRadius * aleatoricScale / 2
          - threshold := by
        have hRobust :
            taskWeightedMean q (jointTaskPayoff mean epistemicPayoff
              expertAleatoric betaRadius zRadius epistemicScale
              aleatoricScale) ≤ robustPayoffUpper := by
          simpa [epistemicPayoff] using hRobustPayoff
        linarith

theorem minimum_of_two_joint_bounds_is_valid
    (margin first second : ℝ)
    (hFirst : margin ≤ first)
    (hSecond : margin ≤ second) :
    margin ≤ min first second := by
  exact le_min hFirst hSecond

theorem finite_minimum_of_joint_bounds_is_valid
    (margin : ℝ) (bounds : Finset ℝ) (hBounds : bounds.Nonempty)
    (hUpper : ∀ upper ∈ bounds, margin ≤ upper) :
    margin ≤ bounds.min' hBounds := by
  exact Finset.le_min' bounds hBounds margin hUpper

end SCOLHKG.Real
