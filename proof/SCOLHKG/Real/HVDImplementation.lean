import Mathlib
import SCOLHKG.Real.CumulativeRisk

namespace SCOLHKG.Real

/-!
Code-level bridge for `SC-OLH-KG/variance/orthogonal_hvd.py`.

The implementation records squared residuals, projects cumulative-risk ridge
coefficients to the nonnegative orthant, clips predicted variances at a floor,
and forms certification variance by adding model uncertainty and a class-level
guard.
-/

def residualSquare (y mu : ℝ) : ℝ :=
  (y - mu) ^ 2

theorem residualSquare_nonnegative (y mu : ℝ) :
    0 ≤ residualSquare y mu := by
  unfold residualSquare
  exact sq_nonneg _

def replicateSampleVarianceNumerator (values : List ℝ) (mean : ℝ) : ℝ :=
  (values.map fun value => (value - mean) ^ 2).sum

theorem replicateSampleVarianceNumerator_nonnegative
    (values : List ℝ)
    (mean : ℝ) :
    0 ≤ replicateSampleVarianceNumerator values mean := by
  unfold replicateSampleVarianceNumerator
  apply List.sum_nonneg
  intro value hvalue
  simp only [List.mem_map] at hvalue
  obtain ⟨source, _, rfl⟩ := hvalue
  exact sq_nonneg (source - mean)

noncomputable def replicateSampleVariance
    (values : List ℝ)
    (mean : ℝ) : ℝ :=
  replicateSampleVarianceNumerator values mean / values.length

theorem replicateSampleVariance_nonnegative
    (values : List ℝ)
    (mean : ℝ) :
    0 ≤ replicateSampleVariance values mean := by
  unfold replicateSampleVariance
  exact div_nonneg
    (replicateSampleVarianceNumerator_nonnegative values mean)
    (Nat.cast_nonneg values.length)

def replicateVarianceDof (replicateCount : ℕ) : ℕ :=
  replicateCount - 1

theorem replicateVarianceDof_mono
    {first second : ℕ}
    (hCount : first ≤ second) :
    replicateVarianceDof first ≤ replicateVarianceDof second := by
  unfold replicateVarianceDof
  exact Nat.sub_le_sub_right hCount 1

def effectiveVarianceDof (replicateCounts : List ℕ) : ℕ :=
  (replicateCounts.map replicateVarianceDof).sum

theorem effectiveVarianceDof_append
    (replicateCounts : List ℕ)
    (replicateCount : ℕ) :
    effectiveVarianceDof (replicateCounts ++ [replicateCount]) =
      effectiveVarianceDof replicateCounts + replicateVarianceDof replicateCount := by
  simp [effectiveVarianceDof]

theorem effectiveVarianceDof_append_ge
    (replicateCounts : List ℕ)
    (replicateCount : ℕ) :
    effectiveVarianceDof replicateCounts ≤
      effectiveVarianceDof (replicateCounts ++ [replicateCount]) := by
  rw [effectiveVarianceDof_append]
  exact Nat.le_add_right _ _

def clippedVariance (floor pred : ℝ) : ℝ :=
  max pred floor

theorem clippedVariance_ge_floor (floor pred : ℝ) :
    floor ≤ clippedVariance floor pred := by
  unfold clippedVariance
  exact le_max_right pred floor

theorem clippedVariance_ge_prediction (floor pred : ℝ) :
    pred ≤ clippedVariance floor pred := by
  unfold clippedVariance
  exact le_max_left pred floor

def certificationVarianceCode
    (base modelUncertainty classVariance floor : ℝ) : ℝ :=
  max (max (base + modelUncertainty) classVariance) floor

theorem certificationVariance_ge_base
    {base modelUncertainty classVariance floor : ℝ}
    (hUncertainty : 0 ≤ modelUncertainty) :
    base ≤ certificationVarianceCode base modelUncertainty classVariance floor := by
  unfold certificationVarianceCode
  have h1 : base ≤ base + modelUncertainty := by linarith
  have h2 : base + modelUncertainty ≤ max (base + modelUncertainty) classVariance :=
    le_max_left _ _
  have h3 :
      max (base + modelUncertainty) classVariance
        ≤ max (max (base + modelUncertainty) classVariance) floor :=
    le_max_left _ _
  exact h1.trans (h2.trans h3)

theorem certificationVariance_sound_from_model_uncertainty
    {trueVariance base modelUncertainty classVariance floor : ℝ}
    (hUpper : trueVariance ≤ base + modelUncertainty) :
    trueVariance ≤ certificationVarianceCode base modelUncertainty classVariance floor := by
  unfold certificationVarianceCode
  have h2 : base + modelUncertainty ≤ max (base + modelUncertainty) classVariance :=
    le_max_left _ _
  have h3 :
      max (base + modelUncertainty) classVariance
        ≤ max (max (base + modelUncertainty) classVariance) floor :=
    le_max_left _ _
  exact hUpper.trans (h2.trans h3)

theorem dot_nonnegative_of_entries_nonnegative
    {xs ys : List ℝ}
    (hxs : ∀ x ∈ xs, 0 ≤ x)
    (hys : ∀ y ∈ ys, 0 ≤ y) :
    0 ≤ dot xs ys := by
  induction xs generalizing ys with
  | nil =>
      simp [dot]
  | cons x xs ih =>
      cases ys with
      | nil =>
          simp [dot]
      | cons y ys =>
          have hx : 0 ≤ x := hxs x (by simp)
          have hy : 0 ≤ y := hys y (by simp)
          have hxsTail : ∀ z ∈ xs, 0 ≤ z := by
            intro z hz
            exact hxs z (by simp [hz])
          have hysTail : ∀ z ∈ ys, 0 ≤ z := by
            intro z hz
            exact hys z (by simp [hz])
          simp [dot]
          exact add_nonneg (mul_nonneg hx hy) (ih hxsTail hysTail)

theorem cumulative_linear_prediction_nonnegative
    {features beta : List ℝ}
    (hFeatures : ∀ x ∈ features, 0 ≤ x)
    (hBeta : ∀ b ∈ beta, 0 ≤ b) :
    0 ≤ dot features beta := by
  exact dot_nonnegative_of_entries_nonnegative hFeatures hBeta

def weightedSquaredResidualRisk
    (weights residuals : List ℝ) : ℝ :=
  dot weights (residuals.map fun residual => residual ^ 2)

theorem weightedSquaredResidualRisk_nonnegative
    {weights residuals : List ℝ}
    (hWeights : ∀ weight ∈ weights, 0 ≤ weight) :
    0 ≤ weightedSquaredResidualRisk weights residuals := by
  unfold weightedSquaredResidualRisk
  apply dot_nonnegative_of_entries_nonnegative hWeights
  intro value hvalue
  simp only [List.mem_map] at hvalue
  obtain ⟨residual, _, rfl⟩ := hvalue
  exact sq_nonneg residual

noncomputable def acceptedProjectedStep
    {Param : Type*}
    (objective : Param → ℝ)
    (project : Param → Param)
    (candidate current : Param) : Param :=
  if objective (project candidate) ≤ objective current then
    project candidate
  else
    current

theorem acceptedProjectedStep_objective_le
    {Param : Type*}
    (objective : Param → ℝ)
    (project : Param → Param)
    (candidate current : Param) :
    objective (acceptedProjectedStep objective project candidate current) ≤
      objective current := by
  unfold acceptedProjectedStep
  split_ifs with hAccept
  · exact hAccept
  · exact le_rfl

theorem acceptedProjectedStep_feasible
    {Param : Type*}
    (objective : Param → ℝ)
    (project : Param → Param)
    (Feasible : Param → Prop)
    (hProject : ∀ candidate, Feasible (project candidate))
    {candidate current : Param}
    (hCurrent : Feasible current) :
    Feasible (acceptedProjectedStep objective project candidate current) := by
  unfold acceptedProjectedStep
  split_ifs
  · exact hProject candidate
  · exact hCurrent

noncomputable def acceptedProjectedIterations
    {Param : Type*}
    (objective : Param → ℝ)
    (project : Param → Param)
    (candidate : ℕ → Param → Param) :
    ℕ → Param → Param
  | 0, current => current
  | steps + 1, current =>
      acceptedProjectedIterations objective project candidate steps
        (acceptedProjectedStep objective project (candidate steps current) current)

theorem acceptedProjectedIterations_objective_le
    {Param : Type*}
    (objective : Param → ℝ)
    (project : Param → Param)
    (candidate : ℕ → Param → Param)
    (steps : ℕ)
    (initial : Param) :
    objective
        (acceptedProjectedIterations objective project candidate steps initial) ≤
      objective initial := by
  induction steps generalizing initial with
  | zero =>
      simp [acceptedProjectedIterations]
  | succ steps ih =>
      simp only [acceptedProjectedIterations]
      exact (ih _).trans
        (acceptedProjectedStep_objective_le
          objective project (candidate steps initial) initial)

theorem acceptedProjectedIterations_feasible
    {Param : Type*}
    (objective : Param → ℝ)
    (project : Param → Param)
    (candidate : ℕ → Param → Param)
    (Feasible : Param → Prop)
    (hProject : ∀ value, Feasible (project value))
    (steps : ℕ)
    {initial : Param}
    (hInitial : Feasible initial) :
    Feasible
      (acceptedProjectedIterations objective project candidate steps initial) := by
  induction steps generalizing initial with
  | zero =>
      simpa [acceptedProjectedIterations] using hInitial
  | succ steps ih =>
      simp only [acceptedProjectedIterations]
      apply ih
      exact acceptedProjectedStep_feasible
        objective project Feasible hProject hInitial

end SCOLHKG.Real
