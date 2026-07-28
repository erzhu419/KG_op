import Mathlib

namespace SCOLHKG.Real

/-!
Implementation contract for the TCB-V2 hierarchical signed-distance model.

The source model supplies one shared boundary shape.  A held-out target learns
only a location and a positive scale.  Parameter covariance and residual
uncertainty enter the upper margin through a nonnegative square-root radius.
The same upper margin is then consumed by frontier nomination, terminal value,
and final recommendation.
-/

noncomputable def hierarchicalBoundaryMean
    (location logScale shape : ℝ) : ℝ :=
  location + Real.exp logScale * shape

theorem hierarchicalBoundaryScale_pos (logScale : ℝ) :
    0 < Real.exp logScale := by
  exact Real.exp_pos logScale

noncomputable def choleskyParameterVariance
    (loading0 loading1 shape : ℝ) : ℝ :=
  (loading0 + shape * loading1) ^ 2

theorem choleskyParameterVariance_nonnegative
    (loading0 loading1 shape : ℝ) :
    0 ≤ choleskyParameterVariance loading0 loading1 shape := by
  unfold choleskyParameterVariance
  positivity

noncomputable def hierarchicalBoundaryVariance
    (loading0 loading1 shape residualScale : ℝ) : ℝ :=
  choleskyParameterVariance loading0 loading1 shape + residualScale ^ 2

theorem hierarchicalBoundaryVariance_nonnegative
    (loading0 loading1 shape residualScale : ℝ) :
    0 ≤ hierarchicalBoundaryVariance
      loading0 loading1 shape residualScale := by
  unfold hierarchicalBoundaryVariance
  exact add_nonneg
    (choleskyParameterVariance_nonnegative loading0 loading1 shape)
    (sq_nonneg residualScale)

noncomputable def hierarchicalBoundaryUpper
    (location logScale shape loading0 loading1 residualScale quantile : ℝ) : ℝ :=
  hierarchicalBoundaryMean location logScale shape
    + quantile * Real.sqrt (
      hierarchicalBoundaryVariance loading0 loading1 shape residualScale)

theorem hierarchicalBoundaryUpper_ge_mean
    {location logScale shape loading0 loading1 residualScale quantile : ℝ}
    (hQuantile : 0 ≤ quantile) :
    hierarchicalBoundaryMean location logScale shape ≤
      hierarchicalBoundaryUpper
        location logScale shape loading0 loading1 residualScale quantile := by
  unfold hierarchicalBoundaryUpper
  have hRadius :
      0 ≤ quantile * Real.sqrt (
        hierarchicalBoundaryVariance
          loading0 loading1 shape residualScale) := by
    exact mul_nonneg hQuantile (Real.sqrt_nonneg _)
  linarith

theorem covariance_radius_cannot_relax_certificate
    {mean radius quantile : ℝ}
    (hRadius : 0 ≤ radius)
    (hQuantile : 0 ≤ quantile) :
    mean ≤ mean + quantile * radius := by
  nlinarith [mul_nonneg hQuantile hRadius]

noncomputable def planarRotationFirst (angle x y : ℝ) : ℝ :=
  Real.cos angle * x - Real.sin angle * y

noncomputable def planarRotationSecond (angle x y : ℝ) : ℝ :=
  Real.sin angle * x + Real.cos angle * y

theorem planarRotation_preserves_squared_norm (angle x y : ℝ) :
    planarRotationFirst angle x y ^ 2
      + planarRotationSecond angle x y ^ 2 = x ^ 2 + y ^ 2 := by
  unfold planarRotationFirst planarRotationSecond
  have hTrig := Real.sin_sq_add_cos_sq angle
  nlinarith

noncomputable def extendedHierarchicalBoundaryVariance
    (baseVariance rotationLoading residualLoading : ℝ) : ℝ :=
  baseVariance + rotationLoading ^ 2 + residualLoading ^ 2

theorem extendedHierarchicalBoundaryVariance_nonnegative
    {baseVariance rotationLoading residualLoading : ℝ}
    (hBase : 0 ≤ baseVariance) :
    0 ≤ extendedHierarchicalBoundaryVariance
      baseVariance rotationLoading residualLoading := by
  unfold extendedHierarchicalBoundaryVariance
  positivity

theorem adding_rotation_or_residual_uncertainty_cannot_lower_upper_margin
    {mean baseVariance rotationLoading residualLoading quantile : ℝ}
    (hQuantile : 0 ≤ quantile) :
    mean + quantile * Real.sqrt baseVariance ≤
      mean + quantile * Real.sqrt (
        extendedHierarchicalBoundaryVariance
          baseVariance rotationLoading residualLoading) := by
  have hVariance :
      baseVariance ≤ extendedHierarchicalBoundaryVariance
        baseVariance rotationLoading residualLoading := by
    unfold extendedHierarchicalBoundaryVariance
    nlinarith [sq_nonneg rotationLoading, sq_nonneg residualLoading]
  have hSqrt := Real.sqrt_le_sqrt hVariance
  nlinarith [mul_le_mul_of_nonneg_left hSqrt hQuantile]

def coverageReservedFrontier
    {Design : Type*}
    (bayes certificate robust nominal : Design)
    (expertNominations : List Design) : List Design :=
  [bayes, certificate, robust, nominal] ++ expertNominations

@[simp] theorem coverageReservedFrontier_take_four
    {Design : Type*}
    (bayes certificate robust nominal : Design)
    (expertNominations : List Design) :
    (coverageReservedFrontier
      bayes certificate robust nominal expertNominations).take 4 =
      [bayes, certificate, robust, nominal] := by
  simp [coverageReservedFrontier]

structure ThreeLayerBoundaryMargin (Design : Type*) where
  frontier : Design → ℝ
  terminal : Design → ℝ
  recommendation : Design → ℝ

def ThreeLayerBoundaryMargin.Coherent
    {Design : Type*}
    (margin : ThreeLayerBoundaryMargin Design) : Prop :=
  ∀ x, margin.frontier x = margin.terminal x ∧
    margin.terminal x = margin.recommendation x

def sharedThreeLayerBoundaryMargin
    {Design : Type*}
    (upper : Design → ℝ) : ThreeLayerBoundaryMargin Design where
  frontier := upper
  terminal := upper
  recommendation := upper

theorem sharedThreeLayerBoundaryMargin_coherent
    {Design : Type*}
    (upper : Design → ℝ) :
    (sharedThreeLayerBoundaryMargin upper).Coherent := by
  intro x
  exact ⟨rfl, rfl⟩

theorem coherent_frontier_equals_recommendation
    {Design : Type*}
    {margin : ThreeLayerBoundaryMargin Design}
    (hCoherent : margin.Coherent)
    (x : Design) :
    margin.frontier x = margin.recommendation x := by
  rcases hCoherent x with ⟨hFrontier, hTerminal⟩
  exact hFrontier.trans hTerminal

theorem coherent_recommendation_is_safe
    {Design : Type*}
    {margin : ThreeLayerBoundaryMargin Design}
    {trueMargin : Design → ℝ}
    (hCoherent : margin.Coherent)
    (hCoverage : ∀ x, trueMargin x ≤ margin.frontier x)
    {x : Design}
    (hRecommended : margin.recommendation x ≤ 0) :
    trueMargin x ≤ 0 := by
  have hSame := coherent_frontier_equals_recommendation hCoherent x
  have hUpper := hCoverage x
  linarith

structure CertifiedLexValue where
  uncertifiedProbability : ℝ
  positiveUpperMargin : ℝ
  objective : ℝ

def CertifiedLexValue.LexLE
    (left right : CertifiedLexValue) : Prop :=
  left.uncertifiedProbability < right.uncertifiedProbability ∨
  (left.uncertifiedProbability = right.uncertifiedProbability ∧
    (left.positiveUpperMargin < right.positiveUpperMargin ∨
      (left.positiveUpperMargin = right.positiveUpperMargin ∧
        left.objective ≤ right.objective)))

def certifiedTerminalValue (objective : ℝ) : CertifiedLexValue where
  uncertifiedProbability := 0
  positiveUpperMargin := 0
  objective := objective

def uncertifiedTerminalValue
    (positiveUpperMargin objective : ℝ) : CertifiedLexValue where
  uncertifiedProbability := 1
  positiveUpperMargin := positiveUpperMargin
  objective := objective

theorem certified_terminal_dominates_uncertified_regardless_objective
    (certifiedObjective uncertifiedMargin uncertifiedObjective : ℝ) :
    (certifiedTerminalValue certifiedObjective).LexLE
      (uncertifiedTerminalValue
        uncertifiedMargin uncertifiedObjective) := by
  left
  norm_num [certifiedTerminalValue, uncertifiedTerminalValue]

theorem lower_positive_margin_dominates_objective_when_uncertified
    {leftMargin rightMargin leftObjective rightObjective : ℝ}
    (hMargin : leftMargin < rightMargin) :
    (uncertifiedTerminalValue leftMargin leftObjective).LexLE
      (uncertifiedTerminalValue rightMargin rightObjective) := by
  right
  constructor
  · rfl
  · exact Or.inl hMargin

theorem objective_breaks_tie_only_after_certificate_components
    {margin leftObjective rightObjective : ℝ}
    (hObjective : leftObjective ≤ rightObjective) :
    (uncertifiedTerminalValue margin leftObjective).LexLE
      (uncertifiedTerminalValue margin rightObjective) := by
  right
  exact ⟨rfl, Or.inr ⟨rfl, hObjective⟩⟩

end SCOLHKG.Real
