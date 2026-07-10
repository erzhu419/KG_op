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

end SCOLHKG.Real
