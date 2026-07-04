import SCOLHKG.Measure.ResidualSquareConcentration

namespace SCOLHKG.Measure

open MeasureTheory ProbabilityTheory
open scoped MeasureTheory ProbabilityTheory BigOperators

/-!
Sharper residual-square tail interface.

The bounded Hoeffding result is one usable concentration theorem.  If the final
manuscript chooses a sharper Gaussian-square or sub-exponential residual-square
model, it can instantiate this tail interface directly.
-/

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

def ResidualSquareTailBound
    (Z : Ω → ℝ)
    (tail : ℝ → ℝ)
    (radius : ℝ) : Prop :=
  μ.real (ResidualSquareCenteredBadEvent (μ := μ) Z radius) ≤ tail radius

noncomputable def subExponentialResidualSquareTail
    (nu b radius : ℝ) : ℝ :=
  2 * Real.exp (-min (radius ^ 2 / (2 * nu ^ 2)) (radius / (2 * b)))

noncomputable def subExponentialResidualSquareRadius
    (nu b delta : ℝ) : ℝ :=
  max
    (Real.sqrt (2 * nu ^ 2 * Real.log (2 / delta)))
    (2 * b * Real.log (2 / delta))

theorem subExponentialResidualSquareRadius_nonnegative
    {nu b delta : ℝ}
    (hb : 0 ≤ b)
    (hdelta0 : 0 < delta)
    (hdelta : delta ≤ 2) :
    0 ≤ subExponentialResidualSquareRadius nu b delta := by
  unfold subExponentialResidualSquareRadius
  have hlog : 0 ≤ Real.log (2 / delta) := by
    exact Real.log_nonneg ((one_le_div₀ hdelta0).2 hdelta)
  exact le_max_of_le_right (mul_nonneg (mul_nonneg (by norm_num) hb) hlog)

theorem subExponentialResidualSquare_tail_default_radius_le
    {nu b delta : ℝ}
    (hnu : 0 < nu)
    (hb : 0 < b)
    (hdelta0 : 0 < delta)
    (hdelta2 : delta ≤ 2) :
    subExponentialResidualSquareTail
      nu b (subExponentialResidualSquareRadius nu b delta) ≤ delta := by
  let L := Real.log (2 / delta)
  have hL_nonneg : 0 ≤ L := by
    exact Real.log_nonneg ((one_le_div₀ hdelta0).2 hdelta2)
  have hdelta_pos : 0 < 2 / delta := div_pos (by norm_num) hdelta0
  have hExpNegL : Real.exp (-L) = delta / 2 := by
    have hlog : Real.exp L = 2 / delta := by
      simpa [L] using Real.exp_log hdelta_pos
    have hne : Real.exp L ≠ 0 := (Real.exp_pos L).ne'
    calc
      Real.exp (-L) = (Real.exp L)⁻¹ := by
        simp [Real.exp_neg]
      _ = (2 / delta)⁻¹ := by rw [hlog]
      _ = delta / 2 := by field_simp [hdelta0.ne']
  have hRadius_ge_sqrt :
      Real.sqrt (2 * nu ^ 2 * L) ≤
        subExponentialResidualSquareRadius nu b delta := by
    unfold subExponentialResidualSquareRadius
    exact le_max_left _ _
  have hRadius_ge_lin :
      2 * b * L ≤ subExponentialResidualSquareRadius nu b delta := by
    unfold subExponentialResidualSquareRadius
    exact le_max_right _ _
  have hA_nonneg : 0 ≤ 2 * nu ^ 2 * L := by
    exact mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg nu)) hL_nonneg
  have hRadius_nonneg :
      0 ≤ subExponentialResidualSquareRadius nu b delta :=
    subExponentialResidualSquareRadius_nonnegative hb.le hdelta0 hdelta2
  have hQuad :
      L ≤
        (subExponentialResidualSquareRadius nu b delta) ^ 2
          / (2 * nu ^ 2) := by
    have hsq :
        2 * nu ^ 2 * L ≤
          (subExponentialResidualSquareRadius nu b delta) ^ 2 := by
      calc
        2 * nu ^ 2 * L
            = (Real.sqrt (2 * nu ^ 2 * L)) ^ 2 := by
                rw [Real.sq_sqrt hA_nonneg]
        _ ≤ (subExponentialResidualSquareRadius nu b delta) ^ 2 := by
                exact sq_le_sq'
                  (by
                    exact (neg_nonpos.mpr hRadius_nonneg).trans
                      (Real.sqrt_nonneg _))
                  hRadius_ge_sqrt
    have hden_pos : 0 < 2 * nu ^ 2 := by positivity
    exact (le_div_iff₀ hden_pos).2 (by
      calc
        L * (2 * nu ^ 2) = 2 * nu ^ 2 * L := by ring
        _ ≤ (subExponentialResidualSquareRadius nu b delta) ^ 2 := hsq)
  have hLin :
      L ≤ subExponentialResidualSquareRadius nu b delta / (2 * b) := by
    have hden_pos : 0 < 2 * b := by positivity
    exact (le_div_iff₀ hden_pos).2 (by
      calc
        L * (2 * b) = 2 * b * L := by ring
        _ ≤ subExponentialResidualSquareRadius nu b delta := hRadius_ge_lin)
  have hMin :
      L ≤
        min
          ((subExponentialResidualSquareRadius nu b delta) ^ 2 / (2 * nu ^ 2))
          (subExponentialResidualSquareRadius nu b delta / (2 * b)) := by
    exact le_min hQuad hLin
  unfold subExponentialResidualSquareTail
  have hExp :
      Real.exp
        (-
          min
            ((subExponentialResidualSquareRadius nu b delta) ^ 2 / (2 * nu ^ 2))
            (subExponentialResidualSquareRadius nu b delta / (2 * b)))
        ≤ Real.exp (-L) := by
    exact Real.exp_le_exp.2 (neg_le_neg hMin)
  calc
    2 *
        Real.exp
          (-
            min
              ((subExponentialResidualSquareRadius nu b delta) ^ 2 / (2 * nu ^ 2))
              (subExponentialResidualSquareRadius nu b delta / (2 * b)))
        ≤ 2 * Real.exp (-L) := by
          exact mul_le_mul_of_nonneg_left hExp (by norm_num)
    _ = delta := by
          rw [hExpNegL]
          ring

def HasSubExponentialResidualSquareTail
    (Z : Ω → ℝ)
    (nu b : ℝ) : Prop :=
  ∀ radius, 0 ≤ radius →
    ResidualSquareTailBound (μ := μ) Z
      (subExponentialResidualSquareTail nu b)
      radius

theorem residualSquare_finite_concentration_from_tail
    {Param : Type*}
    (s : Finset Param)
    (Z : Param → Ω → ℝ)
    (radius delta : Param → ℝ)
    (tail : Param → ℝ → ℝ)
    (hTail :
      ∀ theta ∈ s,
        ResidualSquareTailBound (μ := μ) (Z theta) (tail theta) (radius theta))
    (hDelta :
      ∀ theta ∈ s, tail theta (radius theta) ≤ delta theta) :
    μ.real
        (⋃ theta ∈ s,
          ResidualSquareCenteredBadEvent (μ := μ) (Z theta) (radius theta))
      ≤ ∑ theta ∈ s, delta theta := by
  let bad : Param → Set Ω :=
    fun theta ↦ ResidualSquareCenteredBadEvent (μ := μ) (Z theta) (radius theta)
  have hEach : ∀ theta ∈ s, μ.real (bad theta) ≤ delta theta := by
    intro theta htheta
    exact (hTail theta htheta).trans (hDelta theta htheta)
  calc
    μ.real
        (⋃ theta ∈ s,
          ResidualSquareCenteredBadEvent (μ := μ) (Z theta) (radius theta))
      = μ.real (⋃ theta ∈ s, bad theta) := by
        rfl
    _ ≤ ∑ theta ∈ s, μ.real (bad theta) := by
        exact finite_real_union_bad_event_le_sum (μ := μ) s bad
    _ ≤ ∑ theta ∈ s, delta theta := by
        exact Finset.sum_le_sum (by
          intro theta htheta
          exact hEach theta htheta)

theorem residualSquare_finite_concentration_from_subExponential_tail
    {Param : Type*}
    (s : Finset Param)
    (Z : Param → Ω → ℝ)
    (nu b radius delta : Param → ℝ)
    (hradius : ∀ theta ∈ s, 0 ≤ radius theta)
    (hTail :
      ∀ theta ∈ s,
        HasSubExponentialResidualSquareTail
          (μ := μ) (Z theta) (nu theta) (b theta))
    (hDelta :
      ∀ theta ∈ s,
        subExponentialResidualSquareTail
          (nu theta) (b theta) (radius theta) ≤ delta theta) :
    μ.real
        (⋃ theta ∈ s,
          ResidualSquareCenteredBadEvent (μ := μ) (Z theta) (radius theta))
      ≤ ∑ theta ∈ s, delta theta := by
  exact residualSquare_finite_concentration_from_tail
    (μ := μ)
    s
    Z
    radius
    delta
    (fun theta ↦ subExponentialResidualSquareTail (nu theta) (b theta))
    (fun theta htheta ↦ hTail theta htheta (radius theta) (hradius theta htheta))
    hDelta

theorem residualSquare_finite_concentration_from_subExponential_default_radius
    {Param : Type*}
    (s : Finset Param)
    (Z : Param → Ω → ℝ)
    (nu b delta : Param → ℝ)
    (hradius :
      ∀ theta ∈ s,
        0 ≤ subExponentialResidualSquareRadius
          (nu theta) (b theta) (delta theta))
    (hTail :
      ∀ theta ∈ s,
        HasSubExponentialResidualSquareTail
          (μ := μ) (Z theta) (nu theta) (b theta))
    (hDelta :
      ∀ theta ∈ s,
        subExponentialResidualSquareTail
          (nu theta) (b theta)
          (subExponentialResidualSquareRadius
            (nu theta) (b theta) (delta theta))
          ≤ delta theta) :
    μ.real
        (⋃ theta ∈ s,
          ResidualSquareCenteredBadEvent (μ := μ) (Z theta)
            (subExponentialResidualSquareRadius
              (nu theta) (b theta) (delta theta)))
      ≤ ∑ theta ∈ s, delta theta := by
  exact residualSquare_finite_concentration_from_subExponential_tail
    (μ := μ)
    s
    Z
    nu
    b
    (fun theta ↦ subExponentialResidualSquareRadius
      (nu theta) (b theta) (delta theta))
    delta
    hradius
    hTail
    hDelta

theorem residualSquare_finite_concentration_from_subExponential_default_radius_closed
    {Param : Type*}
    (s : Finset Param)
    (Z : Param → Ω → ℝ)
    (nu b delta : Param → ℝ)
    (hnu : ∀ theta ∈ s, 0 < nu theta)
    (hb : ∀ theta ∈ s, 0 < b theta)
    (hdelta0 : ∀ theta ∈ s, 0 < delta theta)
    (hdelta2 : ∀ theta ∈ s, delta theta ≤ 2)
    (hTail :
      ∀ theta ∈ s,
        HasSubExponentialResidualSquareTail
          (μ := μ) (Z theta) (nu theta) (b theta)) :
    μ.real
        (⋃ theta ∈ s,
          ResidualSquareCenteredBadEvent (μ := μ) (Z theta)
            (subExponentialResidualSquareRadius
              (nu theta) (b theta) (delta theta)))
      ≤ ∑ theta ∈ s, delta theta := by
  exact residualSquare_finite_concentration_from_subExponential_default_radius
    (μ := μ)
    s
    Z
    nu
    b
    delta
    (fun theta htheta ↦
      subExponentialResidualSquareRadius_nonnegative
        (hb theta htheta).le
        (hdelta0 theta htheta)
        (hdelta2 theta htheta))
    hTail
    (fun theta htheta ↦
      subExponentialResidualSquare_tail_default_radius_le
        (hnu theta htheta)
        (hb theta htheta)
        (hdelta0 theta htheta)
        (hdelta2 theta htheta))

end SCOLHKG.Measure
