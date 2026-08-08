import Mathlib

namespace SCOLHKG.Real

/-!
# Finite-error source-score ranking

Uniform source-score estimation error recovers every pair whose true score gap
exceeds twice the error radius. This is the deterministic bridge from the
sub-Gaussian replicated-mean event to the percentile ordering consumed by the
source-scored atlas.
-/

def UniformFiniteScoreError
    {X : Type*} (estimate truth : X → ℝ) (radius : ℝ) : Prop :=
  ∀ x, |estimate x - truth x| ≤ radius

theorem empirical_chance_margin_error_le
    {estimatedMean trueMean estimatedScale trueScale z tau : ℝ}
    {meanRadius scaleRadius : ℝ}
    (hMean : |estimatedMean - trueMean| ≤ meanRadius)
    (hScale : |estimatedScale - trueScale| ≤ scaleRadius)
    (hZ : 0 ≤ z) :
    |(estimatedMean + z * estimatedScale - tau)
        - (trueMean + z * trueScale - tau)|
      ≤ meanRadius + z * scaleRadius := by
  calc
    |(estimatedMean + z * estimatedScale - tau)
        - (trueMean + z * trueScale - tau)|
      = |(estimatedMean - trueMean)
          + z * (estimatedScale - trueScale)| := by ring_nf
    _ ≤ |estimatedMean - trueMean|
          + |z * (estimatedScale - trueScale)| := abs_add_le _ _
    _ = |estimatedMean - trueMean|
          + z * |estimatedScale - trueScale| := by
            rw [abs_mul, abs_of_nonneg hZ]
    _ ≤ meanRadius + z * scaleRadius := by
          gcongr

theorem separated_source_pair_order_recovered
    {X : Type*}
    {estimate truth : X → ℝ}
    {radius : ℝ}
    {x y : X}
    (hUniform : UniformFiniteScoreError estimate truth radius)
    (hGap : truth x + 2 * radius < truth y) :
    estimate x < estimate y := by
  have hx := hUniform x
  have hy := hUniform y
  obtain ⟨hxLower, hxUpper⟩ := abs_le.mp hx
  obtain ⟨hyLower, hyUpper⟩ := abs_le.mp hy
  linarith

def UniqueFiniteScoreMinimizer
    {X : Type*} [DecidableEq X]
    (library : Finset X) (score : X → ℝ) (best : X) (gap : ℝ) : Prop :=
  best ∈ library ∧
    ∀ x ∈ library, x ≠ best → score best + gap < score x

theorem separated_source_minimizer_recovered
    {X : Type*} [DecidableEq X]
    {library : Finset X}
    {estimate truth : X → ℝ}
    {radius : ℝ}
    {best : X}
    (hUniform : UniformFiniteScoreError estimate truth radius)
    (hBest : UniqueFiniteScoreMinimizer library truth best (2 * radius)) :
    best ∈ library ∧
      ∀ x ∈ library, x ≠ best → estimate best < estimate x := by
  refine ⟨hBest.1, ?_⟩
  intro x hx hne
  exact separated_source_pair_order_recovered
    hUniform (hBest.2 x hx hne)

end SCOLHKG.Real
