import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.MeasureTheory.Measure.Real
import Mathlib.Tactic.Common
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Positivity

/-!
# Theorem 1 Formal Core: Debiased Reference Inner Product, Orthogonal Parseval Decomposition,
# and Pruned High-Frequency Residual Safety Envelope

Formalizes the functional-space guarantees of `DebiasedSpectralMetaPrior` (`v2_or/priors/debiased_spectral_prior.py`).
-/

namespace SCOLHKG.V2OR.DebiasedSubspace

set_option linter.unusedSectionVars false

open scoped BigOperators

variable {X K : Type*} [Fintype X] [Fintype K] [DecidableEq K]

/-- Weighted inner product on exposure manifold `X` under measure weights `mu`. -/
def weightedInner (mu : X → ℝ) (f g : X → ℝ) : ℝ :=
  ∑ x, mu x * f x * g x

/-- Weighted squared L2 norm on exposure manifold `X`. -/
def weightedNormSq (mu : X → ℝ) (f : X → ℝ) : ℝ :=
  ∑ x, mu x * (f x) ^ 2

/-- Clipped inverse-occupancy importance weight `min(wBar, nu(x) / q(x))`. -/
noncomputable def clippedOccupancyWeight (nu q : X → ℝ) (wBar : ℝ) (x : X) : ℝ :=
  min wBar (nu x / q x)

/-- Effective reweighted measure `q(x) * clippedOccupancyWeight(x)`. -/
noncomputable def debiasedMeasure (nu q : X → ℝ) (wBar : ℝ) (x : X) : ℝ :=
  q x * clippedOccupancyWeight nu q wBar x

/-- Unclipped bounded-ratio condition recovers the exact reference inner product with zero bias. -/
theorem debiased_inner_exact_when_bounded
    (nu q f g : X → ℝ) (wBar : ℝ)
    (hq : ∀ x, q x ≠ 0)
    (hBound : ∀ x, nu x / q x ≤ wBar) :
    weightedInner (debiasedMeasure nu q wBar) f g = weightedInner nu f g := by
  unfold weightedInner debiasedMeasure clippedOccupancyWeight
  apply Finset.sum_congr rfl
  intro x _
  rw [min_eq_right (hBound x)]
  have hqx := hq x
  field_simp [hqx]

/-- Exact truncation bias bound when clipping extreme inverse-occupancy weights at `wBar`. -/
theorem debiased_inner_clipping_bias_le
    (nu q f g : X → ℝ) (wBar : ℝ)
    (hqPos : ∀ x, 0 < q x) :
    |weightedInner (debiasedMeasure nu q wBar) f g - weightedInner nu f g| ≤
      ∑ x, max 0 (nu x - wBar * q x) * |f x * g x| := by
  unfold weightedInner debiasedMeasure clippedOccupancyWeight
  have hDiff :
      (∑ x, q x * min wBar (nu x / q x) * f x * g x) - (∑ x, nu x * f x * g x) =
        ∑ x, (q x * min wBar (nu x / q x) - nu x) * (f x * g x) := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro x _
    ring
  rw [hDiff]
  calc
    |∑ x, (q x * min wBar (nu x / q x) - nu x) * (f x * g x)| ≤
        ∑ x, |(q x * min wBar (nu x / q x) - nu x) * (f x * g x)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ x, |q x * min wBar (nu x / q x) - nu x| * |f x * g x| := by
      apply Finset.sum_congr rfl
      intro x _
      exact abs_mul _ _
    _ ≤ ∑ x, max 0 (nu x - wBar * q x) * |f x * g x| := by
      apply Finset.sum_le_sum
      intro x _
      have hqx : 0 < q x := hqPos x
      have hqxNe : q x ≠ 0 := ne_of_gt hqx
      have hDiv : q x * (nu x / q x) = nu x := by field_simp [hqxNe]
      have hAbs : |q x * min wBar (nu x / q x) - nu x| = max 0 (nu x - wBar * q x) := by
        by_cases hle : nu x / q x ≤ wBar
        · rw [min_eq_right hle, hDiv, sub_self, abs_zero]
          have hMul : q x * (nu x / q x) ≤ q x * wBar :=
            mul_le_mul_of_nonneg_left hle hqx.le
          rw [hDiv] at hMul
          symm
          exact max_eq_left (by linarith)
        · have hgt : wBar < nu x / q x := lt_of_not_ge hle
          rw [min_eq_left hgt.le]
          have hMul : q x * wBar ≤ q x * (nu x / q x) :=
            mul_le_mul_of_nonneg_left hgt.le hqx.le
          rw [hDiv] at hMul
          have hNonpos : q x * wBar - nu x ≤ 0 := by linarith
          rw [abs_of_nonpos hNonpos]
          symm
          calc
            max 0 (nu x - wBar * q x) = nu x - wBar * q x := max_eq_right (by linarith)
            _ = -(q x * wBar - nu x) := by ring
      rw [hAbs]

/-- Weighted orthonormality predicate `<phi_j, phi_k>_mu = delta_{jk}`. -/
def IsWeightedOrthonormal (mu : X → ℝ) (phi : K → X → ℝ) : Prop :=
  ∀ j k, weightedInner mu (phi j) (phi k) = if j = k then 1 else 0

/-- Linear expansion over a subset `S` of basis coordinates. -/
def basisExpansion (S : Finset K) (theta : K → ℝ) (phi : K → X → ℝ) (x : X) : ℝ :=
  ∑ k ∈ S, theta k * phi k x

/-- Weighted Parseval identity on any subset `S` of orthonormal coordinates. -/
theorem orthonormal_subset_parseval
    (mu : X → ℝ) (phi : K → X → ℝ) (theta : K → ℝ) (S : Finset K)
    (hOrth : IsWeightedOrthonormal mu phi) :
    weightedNormSq mu (basisExpansion S theta phi) = ∑ k ∈ S, (theta k) ^ 2 := by
  unfold weightedNormSq basisExpansion
  have hExpand :
      (∑ x, mu x * (∑ k ∈ S, theta k * phi k x) ^ 2) =
        ∑ j ∈ S, ∑ k ∈ S, theta j * theta k * weightedInner mu (phi j) (phi k) := by
    calc
      (∑ x, mu x * (∑ k ∈ S, theta k * phi k x) ^ 2) =
          ∑ x, ∑ j ∈ S, ∑ k ∈ S, mu x * (theta j * phi j x) * (theta k * phi k x) := by
        apply Finset.sum_congr rfl
        intro x _
        rw [pow_two, Finset.sum_mul_sum, Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro j _
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro k _
        ring
      _ = ∑ j ∈ S, ∑ x, ∑ k ∈ S, mu x * (theta j * phi j x) * (theta k * phi k x) :=
        Finset.sum_comm
      _ = ∑ j ∈ S, ∑ k ∈ S, ∑ x, mu x * (theta j * phi j x) * (theta k * phi k x) := by
        apply Finset.sum_congr rfl
        intro j _
        exact Finset.sum_comm
      _ = ∑ j ∈ S, ∑ k ∈ S, theta j * theta k * weightedInner mu (phi j) (phi k) := by
        apply Finset.sum_congr rfl
        intro j _
        apply Finset.sum_congr rfl
        intro k _
        unfold weightedInner
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro x _
        ring
  rw [hExpand]
  apply Finset.sum_congr rfl
  intro j hj
  have hInner :
      (∑ k ∈ S, theta j * theta k * weightedInner mu (phi j) (phi k)) =
        ∑ k ∈ S, if j = k then (theta j) ^ 2 else 0 := by
    apply Finset.sum_congr rfl
    intro k _
    rw [hOrth j k]
    split_ifs with hjk
    · subst hjk; ring
    · ring
  rw [hInner]
  simp [hj]

/-- Active + pruned orthogonal energy decomposition (zero cross-term leakage). -/
theorem active_pruned_orthogonal_decomposition
    (mu : X → ℝ) (phi : K → X → ℝ) (theta : K → ℝ) (S : Finset K)
    (hOrth : IsWeightedOrthonormal mu phi) :
    weightedNormSq mu (basisExpansion Finset.univ theta phi) =
      (∑ j ∈ S, (theta j) ^ 2) + (∑ k ∈ Sᶜ, (theta k) ^ 2) := by
  rw [orthonormal_subset_parseval mu phi theta Finset.univ hOrth]
  rw [← Finset.sum_add_sum_compl S]

/-- Elementary finite-sum quadratic bound `(∑ k ∈ P, c k)^2 ≤ |P| * ∑ k ∈ P, (c k)^2`
    proved directly via `0 ≤ ∑ j ∈ P, ∑ k ∈ P, (c j - c k)^2`. -/
theorem finset_sq_sum_le_card_mul_sum_sq (P : Finset K) (c : K → ℝ) :
    (∑ k ∈ P, c k) ^ 2 ≤ (P.card : ℝ) * ∑ k ∈ P, (c k) ^ 2 := by
  have hNonneg : 0 ≤ ∑ j ∈ P, ∑ k ∈ P, (c j - c k) ^ 2 :=
    Finset.sum_nonneg (fun j _ => Finset.sum_nonneg (fun k _ => sq_nonneg _))
  have hId :
      (∑ j ∈ P, ∑ k ∈ P, (c j - c k) ^ 2) =
        2 * ((P.card : ℝ) * ∑ k ∈ P, (c k) ^ 2) - 2 * (∑ k ∈ P, c k) ^ 2 := by
    have h1 :
        (∑ j ∈ P, ∑ k ∈ P, (c j - c k) ^ 2) =
          ∑ j ∈ P, ((P.card : ℝ) * (c j) ^ 2 - 2 * c j * (∑ k ∈ P, c k) + ∑ k ∈ P, (c k) ^ 2) := by
      apply Finset.sum_congr rfl
      intro j _
      have hSub :
          (∑ k ∈ P, (c j - c k) ^ 2) =
            (∑ _k ∈ P, (c j) ^ 2) - (∑ k ∈ P, 2 * c j * c k) + ∑ k ∈ P, (c k) ^ 2 := by
        rw [← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
        apply Finset.sum_congr rfl
        intro k _
        ring
      rw [hSub, Finset.sum_const, nsmul_eq_mul]
      have hMid : (∑ k ∈ P, 2 * c j * c k) = 2 * c j * (∑ k ∈ P, c k) :=
        (Finset.mul_sum P c (2 * c j)).symm
      rw [hMid]
    rw [h1, Finset.sum_add_distrib, Finset.sum_sub_distrib]
    rw [Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum]
    have hCross : (∑ j ∈ P, 2 * c j * (∑ k ∈ P, c k)) = 2 * (∑ k ∈ P, c k) ^ 2 := by
      rw [pow_two, Finset.mul_sum, Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro j _
      ring
    rw [hCross]
    ring
  linarith

/-- Pointwise safety bound justifying the `R_perp(x)` conservative residual floor. -/
theorem pruned_residual_pointwise_square_bound
    (P : Finset K) (theta : K → ℝ) (phi : K → X → ℝ)
    (energyCap : K → ℝ) (highFloor : ℝ) (x : X)
    (hFloor : 0 ≤ highFloor)
    (hCap : ∀ k ∈ P, (P.card : ℝ) * (theta k) ^ 2 ≤ energyCap k) :
    (basisExpansion P theta phi x) ^ 2 ≤
      (∑ k ∈ P, energyCap k * (phi k x) ^ 2) + highFloor := by
  unfold basisExpansion
  have hCS := finset_sq_sum_le_card_mul_sum_sq P (fun k => theta k * phi k x)
  have hDom :
      (P.card : ℝ) * (∑ k ∈ P, (theta k * phi k x) ^ 2) ≤
        ∑ k ∈ P, energyCap k * (phi k x) ^ 2 := by
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro k hk
    have h1 : (P.card : ℝ) * (theta k * phi k x) ^ 2 =
        ((P.card : ℝ) * (theta k) ^ 2) * (phi k x) ^ 2 := by ring
    rw [h1]
    exact mul_le_mul_of_nonneg_right (hCap k hk) (sq_nonneg _)
  linarith

/-- Self-normalized clipped inverse-occupancy probability measure `mu_c(x) / ∑_y mu_c(y)`
    matching the normalized weights `w / np.sum(w)` in `DebiasedSpectralMetaPrior`. -/
noncomputable def normalizedDebiasedMeasure (nu q : X → ℝ) (wBar : ℝ) (x : X) : ℝ :=
  debiasedMeasure nu q wBar x / (∑ y, debiasedMeasure nu q wBar y)

/-- Pointwise mass deficit identity `nu(x) - mu_c(x) = max(0, nu(x) - wBar * q(x))`. -/
lemma debiased_measure_deficit_eq
    (nu q : X → ℝ) (wBar : ℝ) (x : X)
    (hqPos : 0 < q x) :
    nu x - debiasedMeasure nu q wBar x = max 0 (nu x - wBar * q x) := by
  unfold debiasedMeasure clippedOccupancyWeight
  have hqxNe : q x ≠ 0 := ne_of_gt hqPos
  have hDiv : q x * (nu x / q x) = nu x := by field_simp [hqxNe]
  by_cases hle : nu x / q x ≤ wBar
  · rw [min_eq_right hle, hDiv, sub_self]
    have hMul : q x * (nu x / q x) ≤ q x * wBar :=
      mul_le_mul_of_nonneg_left hle hqPos.le
    rw [hDiv] at hMul
    symm
    exact max_eq_left (by linarith)
  · have hgt : wBar < nu x / q x := lt_of_not_ge hle
    rw [min_eq_left hgt.le]
    have hMul : q x * wBar ≤ q x * (nu x / q x) :=
      mul_le_mul_of_nonneg_left hgt.le hqPos.le
    rw [hDiv] at hMul
    symm
    calc
      max 0 (nu x - wBar * q x) = nu x - wBar * q x := max_eq_right (by linarith)
      _ = nu x - q x * wBar := by ring

/-- Exact self-normalized clipping bias bound for `w_i / ∑_j w_j` with bounded integrands
    `|f(x) g(x)| ≤ B`: the normalizing constant `Z_c = ∑_y mu_c(y) > 0` cancels cleanly! -/
theorem normalized_debiased_inner_clipping_bias_le
    (nu q f g : X → ℝ) (wBar B : ℝ)
    (hqPos : ∀ x, 0 < q x)
    (hNuNonneg : ∀ x, 0 ≤ nu x)
    (hwBarNonneg : 0 ≤ wBar)
    (hNuSum : (∑ x, nu x) = 1)
    (hZPos : 0 < ∑ x, debiasedMeasure nu q wBar x)
    (hBNonneg : 0 ≤ B)
    (hProdBound : ∀ x, |f x * g x| ≤ B) :
    |weightedInner (normalizedDebiasedMeasure nu q wBar) f g - weightedInner nu f g| ≤
      2 * B * (∑ x, max 0 (nu x - wBar * q x)) := by
  set Z := ∑ x, debiasedMeasure nu q wBar x
  set Ic := weightedInner (debiasedMeasure nu q wBar) f g
  set Inu := weightedInner nu f g
  set Delta := ∑ x, max 0 (nu x - wBar * q x)
  have hZNe : Z ≠ 0 := ne_of_gt hZPos
  have hNormInner : weightedInner (normalizedDebiasedMeasure nu q wBar) f g = Ic / Z := by
    unfold weightedInner normalizedDebiasedMeasure
    simp only [Ic, Z, weightedInner, div_eq_mul_inv, Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro x _
    ring
  rw [hNormInner]
  have hOneSubZ : 1 - Z = Delta := by
    rw [← hNuSum]
    simp only [Z, Delta, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro x _
    exact debiased_measure_deficit_eq nu q wBar x (hqPos x)
  have hDeltaNonneg : 0 ≤ Delta :=
    Finset.sum_nonneg (fun x _ => le_max_left 0 _)
  have hMuNonneg : ∀ x, 0 ≤ debiasedMeasure nu q wBar x := by
    intro x
    unfold debiasedMeasure clippedOccupancyWeight
    exact mul_nonneg (hqPos x).le (le_min hwBarNonneg (div_nonneg (hNuNonneg x) (hqPos x).le))
  have hIcBound : |Ic| ≤ Z * B := by
    simp only [Ic, weightedInner]
    calc
      |∑ x, debiasedMeasure nu q wBar x * f x * g x| ≤
          ∑ x, |debiasedMeasure nu q wBar x * f x * g x| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ x, debiasedMeasure nu q wBar x * |f x * g x| := by
        apply Finset.sum_congr rfl
        intro x _
        have hAssoc : debiasedMeasure nu q wBar x * f x * g x =
            debiasedMeasure nu q wBar x * (f x * g x) := by ring
        rw [hAssoc, abs_mul, abs_of_nonneg (hMuNonneg x)]
      _ ≤ ∑ x, debiasedMeasure nu q wBar x * B := by
        apply Finset.sum_le_sum
        intro x _
        exact mul_le_mul_of_nonneg_left (hProdBound x) (hMuNonneg x)
      _ = Z * B := by simp only [Z, ← Finset.sum_mul]
  have hTerm1 : |Ic / Z - Ic| ≤ B * Delta := by
    have hFactor : Ic / Z - Ic = ((1 - Z) / Z) * Ic := by
      field_simp [hZNe]
      try ring
    have hFracNonneg : 0 ≤ (1 - Z) / Z := by
      rw [hOneSubZ]
      exact div_nonneg hDeltaNonneg hZPos.le
    rw [hFactor, abs_mul, abs_of_nonneg hFracNonneg]
    have hMul := mul_le_mul_of_nonneg_left hIcBound hFracNonneg
    have hCancel : ((1 - Z) / Z) * (Z * B) = B * Delta := by
      rw [hOneSubZ]
      field_simp [hZNe]
      try ring
    linarith
  have hTerm2 : |Ic - Inu| ≤ B * Delta := by
    have hRaw := debiased_inner_clipping_bias_le nu q f g wBar hqPos
    have hStep : (∑ x, max 0 (nu x - wBar * q x) * |f x * g x|) ≤ Delta * B := by
      simp only [Delta, Finset.sum_mul]
      apply Finset.sum_le_sum
      intro x _
      exact mul_le_mul_of_nonneg_left (hProdBound x) (le_max_left 0 _)
    linarith
  have hTri : |Ic / Z - Inu| ≤ |Ic / Z - Ic| + |Ic - Inu| := by
    have hSplit : Ic / Z - Inu = (Ic / Z - Ic) + (Ic - Inu) := by ring
    rw [hSplit]
    exact abs_add_le _ _
  linarith

end SCOLHKG.V2OR.DebiasedSubspace
