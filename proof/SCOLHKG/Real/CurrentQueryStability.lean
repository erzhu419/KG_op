import Mathlib.Data.Real.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

namespace SCOLHKG.Real.CurrentQueryStability

open scoped BigOperators

noncomputable section

/-!
# Common-pool constrained Thompson-sampling stability

The two selection predicates below encode the two `constrained_ts` branches in
`acquisition/decision_backends.py`. The feasible branch uses its actual finite
sentinel; the empty-feasible-set branch includes the normalized objective
machine-epsilon term. Scores and draws are real arithmetic abstractions.

One fixed candidate pool and one common realized random vector are arguments of
the theorems. They do not identify different dynamic pools, different sampled
experts, or different random streams. No KG optimality claim is made.
-/

def UniformPoolError {X : Type*} (pool : Finset X)
    (estimate baseline : X → ℝ) (radius : ℝ) : Prop :=
  ∀ x ∈ pool, |estimate x - baseline x| ≤ radius

def SampledFeasible {X : Type*} [DecidableEq X]
    (pool : Finset X) (margin : X → ℝ) : Finset X :=
  pool.filter fun x => margin x ≤ 0

theorem separated_pool_pair_order {X : Type*} {pool : Finset X}
    {estimate baseline : X → ℝ} {radius : ℝ} {x y : X}
    (hError : UniformPoolError pool estimate baseline radius)
    (hx : x ∈ pool) (hy : y ∈ pool)
    (hGap : baseline x + 2 * radius < baseline y) :
    estimate x < estimate y := by
  obtain ⟨hxLower, hxUpper⟩ := abs_le.mp (hError x hx)
  obtain ⟨hyLower, hyUpper⟩ := abs_le.mp (hError y hy)
  linarith

theorem sampled_feasibility_iff {X : Type*} {pool : Finset X}
    {estimate baseline : X → ℝ} {radius : ℝ}
    (hError : UniformPoolError pool estimate baseline radius)
    (hMargin : ∀ x ∈ pool, radius < |baseline x|)
    {x : X} (hx : x ∈ pool) :
    (estimate x ≤ 0 ↔ baseline x ≤ 0) := by
  obtain ⟨hLower, hUpper⟩ := abs_le.mp (hError x hx)
  have hSep := hMargin x hx
  by_cases hBase : baseline x ≤ 0
  · rw [abs_of_nonpos hBase] at hSep
    have hEstimate : estimate x ≤ 0 := by linarith
    exact iff_of_true hEstimate hBase
  · have hPositive : 0 < baseline x := lt_of_not_ge hBase
    rw [abs_of_pos hPositive] at hSep
    have hEstimate : ¬ estimate x ≤ 0 := by linarith
    exact iff_of_false hEstimate hBase

theorem sampled_feasible_set_eq {X : Type*} [DecidableEq X]
    {pool : Finset X} {estimate baseline : X → ℝ} {radius : ℝ}
    (hError : UniformPoolError pool estimate baseline radius)
    (hMargin : ∀ x ∈ pool, radius < |baseline x|) :
    SampledFeasible pool estimate = SampledFeasible pool baseline := by
  ext x
  simp only [SampledFeasible, Finset.mem_filter]
  by_cases hx : x ∈ pool
  · simp only [hx, true_and]
    exact sampled_feasibility_iff hError hMargin hx
  · simp only [hx, false_and]

theorem empty_feasible_branch_preserved {X : Type*} {pool : Finset X}
    {estimate baseline : X → ℝ} {radius : ℝ}
    (hError : UniformPoolError pool estimate baseline radius)
    (hMargin : ∀ x ∈ pool, radius < |baseline x|)
    (hEmpty : ∀ x ∈ pool, 0 < baseline x) :
    ∀ x ∈ pool, 0 < estimate x := by
  intro x hx
  apply lt_of_not_ge
  intro hEstimate
  have hBaseline := (sampled_feasibility_iff hError hMargin hx).1 hEstimate
  exact (not_le.mpr (hEmpty x hx)) hBaseline

def FeasibleBranchSelects {X : Type*} (pool : Finset X)
    (objective margin : X → ℝ) (sentinel : ℝ) (chosen : X) : Prop :=
  chosen ∈ pool ∧ ∀ x ∈ pool,
    (if margin x ≤ 0 then -objective x else sentinel) ≤
      (if margin chosen ≤ 0 then -objective chosen else sentinel)

theorem feasible_branch_selection_eq {X : Type*} {pool : Finset X}
    {objective margin : X → ℝ} {sentinel : ℝ} {best chosen : X}
    (hBest : best ∈ pool) (hFeasible : margin best ≤ 0)
    (hGap : ∀ x ∈ pool, x ≠ best → margin x ≤ 0 →
      objective best < objective x)
    (hSentinel : sentinel < -objective best)
    (hSelect : FeasibleBranchSelects pool objective margin sentinel chosen) :
    chosen = best := by
  by_contra hne
  have hCompare := hSelect.2 best hBest
  by_cases hChosen : margin chosen ≤ 0
  · simp only [if_pos hFeasible, if_pos hChosen] at hCompare
    have hStrict := hGap chosen hSelect.1 hne hChosen
    linarith
  · simp only [if_pos hFeasible, if_neg hChosen] at hCompare
    linarith

theorem feasible_branch_query_stable {X : Type*} {pool : Finset X}
    {objective objective' margin margin' : X → ℝ}
    {objectiveRadius marginRadius sentinel : ℝ} {best chosen chosen' : X}
    (hRadius : 0 ≤ objectiveRadius)
    (hObjective : UniformPoolError pool objective' objective objectiveRadius)
    (hMargin : UniformPoolError pool margin' margin marginRadius)
    (hSeparated : ∀ x ∈ pool, marginRadius < |margin x|)
    (hBest : best ∈ pool) (hFeasible : margin best ≤ 0)
    (hGap : ∀ x ∈ pool, x ≠ best → margin x ≤ 0 →
      objective best + 2 * objectiveRadius < objective x)
    (hSentinel : sentinel < -objective best)
    (hSentinel' : sentinel < -objective' best)
    (hSelect : FeasibleBranchSelects pool objective margin sentinel chosen)
    (hSelect' : FeasibleBranchSelects pool objective' margin' sentinel chosen') :
    chosen = chosen' := by
  have hOld : chosen = best := feasible_branch_selection_eq hBest hFeasible
    (by
      intro x hx hne hfx
      have h := hGap x hx hne hfx
      linarith) hSentinel hSelect
  have hNewFeasible : margin' best ≤ 0 :=
    (sampled_feasibility_iff hMargin hSeparated hBest).2 hFeasible
  have hNew : chosen' = best := feasible_branch_selection_eq hBest hNewFeasible
    (by
      intro x hx hne hfx
      have hOriginalFeasible :=
        (sampled_feasibility_iff hMargin hSeparated hx).1 hfx
      exact separated_pool_pair_order hObjective hBest hx
        (hGap x hx hne hOriginalFeasible)) hSentinel' hSelect'
  exact hOld.trans hNew.symm

def fallbackLoss {X : Type*} (margin objective : X → ℝ)
    (machineEpsilon objectiveScale : ℝ) (x : X) : ℝ :=
  margin x + machineEpsilon * (objective x / objectiveScale)

def MinimumViolationSelects {X : Type*} (pool : Finset X)
    (margin objective : X → ℝ) (machineEpsilon objectiveScale : ℝ)
    (chosen : X) : Prop :=
  chosen ∈ pool ∧ (∀ x ∈ pool, 0 < margin x) ∧
    ∀ x ∈ pool, fallbackLoss margin objective machineEpsilon objectiveScale chosen ≤
      fallbackLoss margin objective machineEpsilon objectiveScale x

theorem fallback_loss_error_le {X : Type*}
    {margin margin' objective objective' : X → ℝ}
    {machineEpsilon scale scale' marginRadius normalizedRadius : ℝ} {x : X}
    (hEpsilon : 0 ≤ machineEpsilon)
    (hMargin : |margin' x - margin x| ≤ marginRadius)
    (hNormalized : |objective' x / scale' - objective x / scale| ≤ normalizedRadius) :
    |fallbackLoss margin' objective' machineEpsilon scale' x -
        fallbackLoss margin objective machineEpsilon scale x| ≤
      marginRadius + machineEpsilon * normalizedRadius := by
  unfold fallbackLoss
  calc
    |margin' x + machineEpsilon * (objective' x / scale') -
        (margin x + machineEpsilon * (objective x / scale))| =
        |(margin' x - margin x) +
          machineEpsilon * (objective' x / scale' - objective x / scale)| := by ring_nf
    _ ≤ |margin' x - margin x| +
        |machineEpsilon * (objective' x / scale' - objective x / scale)| := abs_add_le _ _
    _ = |margin' x - margin x| +
        machineEpsilon * |objective' x / scale' - objective x / scale| := by
          rw [abs_mul, abs_of_nonneg hEpsilon]
    _ ≤ marginRadius + machineEpsilon * normalizedRadius :=
      add_le_add hMargin (mul_le_mul_of_nonneg_left hNormalized hEpsilon)

theorem minimum_violation_query_stable {X : Type*} {pool : Finset X}
    {margin margin' objective objective' : X → ℝ}
    {machineEpsilon scale scale' radius : ℝ} {best chosen chosen' : X}
    (hRadius : 0 ≤ radius) (hBest : best ∈ pool)
    (hError : UniformPoolError pool
      (fallbackLoss margin' objective' machineEpsilon scale')
      (fallbackLoss margin objective machineEpsilon scale) radius)
    (hGap : ∀ x ∈ pool, x ≠ best →
      fallbackLoss margin objective machineEpsilon scale best + 2 * radius <
        fallbackLoss margin objective machineEpsilon scale x)
    (hSelect : MinimumViolationSelects pool margin objective machineEpsilon scale chosen)
    (hSelect' : MinimumViolationSelects pool margin' objective' machineEpsilon scale' chosen') :
    chosen = chosen' := by
  have hOld : chosen = best := by
    by_contra hne
    have hLower := hSelect.2.2 best hBest
    have hStrict := hGap chosen hSelect.1 hne
    linarith
  have hNew : chosen' = best := by
    by_contra hne
    have hLower := hSelect'.2.2 best hBest
    have hStrict := separated_pool_pair_order hError hBest hSelect'.1
      (hGap chosen' hSelect'.1 hne)
    linarith
  exact hOld.trans hNew.symm

def coupledDraw {X J : Type*} [Fintype J]
    (mean : X → ℝ) (loading : X → J → ℝ) (randomVector : J → ℝ) (x : X) : ℝ :=
  mean x + ∑ j, loading x j * randomVector j

theorem coupled_draw_error_le {X J : Type*} [Fintype J]
    {mean mean' : X → ℝ} {loading loading' : X → J → ℝ}
    {randomVector : J → ℝ} {meanRadius : ℝ} {loadingRadius : J → ℝ} {x : X}
    (hMean : |mean' x - mean x| ≤ meanRadius)
    (hLoading : ∀ j, |loading' x j - loading x j| ≤ loadingRadius j) :
    |coupledDraw mean' loading' randomVector x - coupledDraw mean loading randomVector x| ≤
      meanRadius + ∑ j, loadingRadius j * |randomVector j| := by
  have hDifference : coupledDraw mean' loading' randomVector x -
      coupledDraw mean loading randomVector x =
      (mean' x - mean x) + ∑ j, (loading' x j - loading x j) * randomVector j := by
    simp only [coupledDraw, sub_mul, Finset.sum_sub_distrib]
    ring
  rw [hDifference]
  calc
    |(mean' x - mean x) + ∑ j, (loading' x j - loading x j) * randomVector j| ≤
        |mean' x - mean x| + |∑ j, (loading' x j - loading x j) * randomVector j| :=
      abs_add_le _ _
    _ ≤ meanRadius + ∑ j, |(loading' x j - loading x j) * randomVector j| :=
      add_le_add hMean (Finset.abs_sum_le_sum_abs _ _)
    _ = meanRadius + ∑ j, |loading' x j - loading x j| * |randomVector j| := by
      simp only [abs_mul]
    _ ≤ meanRadius + ∑ j, loadingRadius j * |randomVector j| := by
      exact add_le_add le_rfl (Finset.sum_le_sum fun j _ =>
        mul_le_mul_of_nonneg_right (hLoading j) (abs_nonneg _))

theorem coupled_draw_pool_error_le {X J : Type*} [Fintype J]
    {pool : Finset X} {mean mean' : X → ℝ} {loading loading' : X → J → ℝ}
    {randomVector : J → ℝ} {meanRadius : ℝ} {loadingRadius : J → ℝ}
    (hMean : ∀ x ∈ pool, |mean' x - mean x| ≤ meanRadius)
    (hLoading : ∀ x ∈ pool, ∀ j, |loading' x j - loading x j| ≤ loadingRadius j) :
    UniformPoolError pool
      (coupledDraw mean' loading' randomVector)
      (coupledDraw mean loading randomVector)
      (meanRadius + ∑ j, loadingRadius j * |randomVector j|) := by
  intro x hx
  exact coupled_draw_error_le (hMean x hx) (hLoading x hx)

theorem chance_margin_error_le
    {draw draw' noiseScale noiseScale' quantile threshold drawRadius noiseRadius : ℝ}
    (hDraw : |draw' - draw| ≤ drawRadius)
    (hNoise : |noiseScale' - noiseScale| ≤ noiseRadius) :
    |(draw' + quantile * noiseScale' - threshold) -
        (draw + quantile * noiseScale - threshold)| ≤
      drawRadius + |quantile| * noiseRadius := by
  calc
    |(draw' + quantile * noiseScale' - threshold) -
        (draw + quantile * noiseScale - threshold)| =
      |(draw' - draw) + quantile * (noiseScale' - noiseScale)| := by ring_nf
    _ ≤ |draw' - draw| + |quantile * (noiseScale' - noiseScale)| := abs_add_le _ _
    _ = |draw' - draw| + |quantile| * |noiseScale' - noiseScale| := by rw [abs_mul]
    _ ≤ drawRadius + |quantile| * noiseRadius :=
      add_le_add hDraw (mul_le_mul_of_nonneg_left hNoise (abs_nonneg _))

end

end SCOLHKG.Real.CurrentQueryStability
