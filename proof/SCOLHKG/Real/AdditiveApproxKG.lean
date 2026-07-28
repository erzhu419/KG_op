import Mathlib
import SCOLHKG.Real.KG

namespace SCOLHKG.Real

/-!
Approximation bound from additive acquisition to exact KG.

The implemented acquisition is additive.  The theorem here records the precise
price of using that additive proxy: if it uniformly approximates exact KG within
`eta`, any proxy maximizer is `2 * eta`-optimal for exact one-step KG.
-/

universe u

def UniformKGApprox
    {Design : Type u}
    (exact proxy : Design → ℝ)
    (eta : ℝ) : Prop :=
  ∀ x, |exact x - proxy x| ≤ eta

theorem proxy_maximizer_exact_gap_le_two_eta
    {Design : Type u}
    {exact proxy : Design → ℝ}
    {eta : ℝ}
    {x : Design}
    (hApprox : UniformKGApprox exact proxy eta)
    (hMax : ∀ y, proxy y ≤ proxy x) :
    ∀ y, exact y ≤ exact x + 2 * eta := by
  intro y
  have hyAbs := abs_le.mp (hApprox y)
  have hxAbs := abs_le.mp (hApprox x)
  have hy : exact y ≤ proxy y + eta := by
    linarith [hyAbs.2]
  have hx : proxy x ≤ exact x + eta := by
    linarith [hxAbs.1]
  have hProxy : proxy y ≤ proxy x := hMax y
  linarith

def AdditiveKGUniformApprox
    {Design : Type u}
    (kg : AdditiveKG Design)
    (eta : ℝ) : Prop :=
  UniformKGApprox kg.exact (additiveScore kg) eta

theorem additive_proxy_maximizer_exact_gap_le_two_eta
    {Design : Type u}
    (kg : AdditiveKG Design)
    {eta : ℝ}
    {x : Design}
    (hApprox : AdditiveKGUniformApprox kg eta)
    (hMax : ∀ y, additiveScore kg y ≤ additiveScore kg x) :
    ∀ y, kg.exact y ≤ kg.exact x + 2 * eta := by
  exact proxy_maximizer_exact_gap_le_two_eta
    (exact := kg.exact)
    (proxy := additiveScore kg)
    (eta := eta)
    (x := x)
    hApprox
    hMax

end SCOLHKG.Real
