import Mathlib

namespace SCOLHKG.Real

open MeasureTheory
open scoped BigOperators

/-!
# Ordered-profile coordinate consistency

The implemented profile coordinate is a weighted finite cosine functional.
The first theorem controls any coefficient by the sup-norm reconstruction
error. The second makes the inverse-grid rate explicit. For a Lipschitz
continuous profile, midpoint or linear reconstruction supplies the pointwise
`C / d` premise with a basis-dependent constant; the theorem then transfers it
to every retained coefficient without dependence on nominal policy dimension.
-/

def weightedProfileCoefficient
    {ι : Type*} [Fintype ι]
    (weight profile basis : ι → ℝ) : ℝ :=
  ∑ i, weight i * profile i * basis i

noncomputable def continuousProfileCoefficient
    (profile basis : ℝ → ℝ) : ℝ :=
  ∫ x in (0 : ℝ)..1, profile x * basis x ∂volume

theorem continuousProfileCoefficient_error_le
    {first second basis : ℝ → ℝ}
    {epsilon basisBound : ℝ}
    (hFirst : IntervalIntegrable
      (fun x => first x * basis x) volume 0 1)
    (hSecond : IntervalIntegrable
      (fun x => second x * basis x) volume 0 1)
    (hError : ∀ x ∈ Set.uIoc (0 : ℝ) 1,
      |first x - second x| ≤ epsilon)
    (hBasis : ∀ x ∈ Set.uIoc (0 : ℝ) 1, |basis x| ≤ basisBound)
    (hEpsilon : 0 ≤ epsilon) :
    |continuousProfileCoefficient first basis
        - continuousProfileCoefficient second basis|
      ≤ epsilon * basisBound := by
  have hIntegral := intervalIntegral.norm_integral_le_of_norm_le_const
    (a := (0 : ℝ)) (b := 1) (C := epsilon * basisBound)
    (f := fun x => first x * basis x - second x * basis x)
    (fun x hx => by
      simp only [Real.norm_eq_abs]
      rw [← sub_mul, abs_mul]
      exact mul_le_mul
        (hError x hx)
        (hBasis x hx)
        (abs_nonneg _)
        hEpsilon)
  have hSub := intervalIntegral.integral_sub hFirst hSecond
  calc
    |continuousProfileCoefficient first basis
        - continuousProfileCoefficient second basis|
      = |∫ x in (0 : ℝ)..1,
          first x * basis x - second x * basis x ∂volume| := by
            unfold continuousProfileCoefficient
            exact congrArg (fun z : ℝ => |z|) hSub.symm
    _ = ‖∫ x in (0 : ℝ)..1,
          first x * basis x - second x * basis x ∂volume‖ := by
            rw [Real.norm_eq_abs]
    _ ≤ (epsilon * basisBound) * |(1 : ℝ) - 0| := hIntegral
    _ = epsilon * basisBound := by ring

theorem continuousProfileCoefficient_inverse_grid_rate
    {first second basis : ℝ → ℝ}
    {constant basisBound : ℝ}
    {dimension : ℕ}
    (hDimension : 0 < dimension)
    (hConstant : 0 ≤ constant)
    (hFirst : IntervalIntegrable
      (fun x => first x * basis x) volume 0 1)
    (hSecond : IntervalIntegrable
      (fun x => second x * basis x) volume 0 1)
    (hError : ∀ x ∈ Set.uIoc (0 : ℝ) 1,
      |first x - second x| ≤ constant / dimension)
    (hBasis : ∀ x ∈ Set.uIoc (0 : ℝ) 1, |basis x| ≤ basisBound) :
    |continuousProfileCoefficient first basis
        - continuousProfileCoefficient second basis|
      ≤ constant * basisBound / dimension := by
  have hDimensionReal : (0 : ℝ) < dimension := by exact_mod_cast hDimension
  have hRateNonnegative : 0 ≤ constant / (dimension : ℝ) :=
    div_nonneg hConstant hDimensionReal.le
  have hBound := continuousProfileCoefficient_error_le
    hFirst hSecond hError hBasis hRateNonnegative
  calc
    |continuousProfileCoefficient first basis
        - continuousProfileCoefficient second basis|
      ≤ (constant / (dimension : ℝ)) * basisBound := hBound
    _ = constant * basisBound / dimension := by ring

theorem weightedProfileCoefficient_error_le
    {ι : Type*} [Fintype ι]
    {weight first second basis : ι → ℝ}
    {epsilon basisMass : ℝ}
    (hError : ∀ i, |first i - second i| ≤ epsilon)
    (hMass : ∑ i, |weight i| * |basis i| ≤ basisMass)
    (hEpsilon : 0 ≤ epsilon) :
    |weightedProfileCoefficient weight first basis
        - weightedProfileCoefficient weight second basis|
      ≤ epsilon * basisMass := by
  have hRewrite :
      weightedProfileCoefficient weight first basis
          - weightedProfileCoefficient weight second basis =
        ∑ i, weight i * (first i - second i) * basis i := by
    simp only [weightedProfileCoefficient, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _hi
    ring
  rw [hRewrite]
  calc
    |∑ i, weight i * (first i - second i) * basis i|
        ≤ ∑ i, |weight i * (first i - second i) * basis i| :=
          Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, |weight i| * |first i - second i| * |basis i| := by
          apply Finset.sum_congr rfl
          intro i _hi
          simp only [abs_mul]
    _ ≤ ∑ i, epsilon * (|weight i| * |basis i|) := by
          apply Finset.sum_le_sum
          intro i _hi
          have hWeight : 0 ≤ |weight i| := abs_nonneg _
          have hBasis : 0 ≤ |basis i| := abs_nonneg _
          calc
            |weight i| * |first i - second i| * |basis i|
                ≤ |weight i| * epsilon * |basis i| := by
                  gcongr
                  exact hError i
            _ = epsilon * (|weight i| * |basis i|) := by ring
    _ = epsilon * ∑ i, |weight i| * |basis i| := by
          rw [Finset.mul_sum]
    _ ≤ epsilon * basisMass :=
          mul_le_mul_of_nonneg_left hMass hEpsilon

theorem weightedProfileCoefficient_inverse_grid_rate
    {ι : Type*} [Fintype ι]
    {weight first second basis : ι → ℝ}
    {constant basisMass : ℝ}
    {dimension : ℕ}
    (hDimension : 0 < dimension)
    (hConstant : 0 ≤ constant)
    (hPointwise :
      ∀ i, |first i - second i| ≤ constant / dimension)
    (hMass : ∑ i, |weight i| * |basis i| ≤ basisMass) :
    |weightedProfileCoefficient weight first basis
        - weightedProfileCoefficient weight second basis|
      ≤ constant * basisMass / dimension := by
  have hDimensionReal : (0 : ℝ) < dimension := by exact_mod_cast hDimension
  have hRateNonnegative : 0 ≤ constant / (dimension : ℝ) :=
    div_nonneg hConstant hDimensionReal.le
  have hBound := weightedProfileCoefficient_error_le
    hPointwise hMass hRateNonnegative
  calc
    |weightedProfileCoefficient weight first basis
        - weightedProfileCoefficient weight second basis|
        ≤ (constant / (dimension : ℝ)) * basisMass := hBound
    _ = constant * basisMass / dimension := by ring

theorem lipschitz_sample_reconstruction_error
    {h : ℝ → ℝ}
    {L x node inverseDimension : ℝ}
    (hLipschitz : |h x - h node| ≤ L * |x - node|)
    (hNode : |x - node| ≤ inverseDimension)
    (hL : 0 ≤ L) :
    |h x - h node| ≤ L * inverseDimension := by
  exact hLipschitz.trans (mul_le_mul_of_nonneg_left hNode hL)

theorem lipschitz_voronoi_coefficient_inverse_grid_rate
    {profile reconstruction basis node : ℝ → ℝ}
    {L basisBound : ℝ}
    {dimension : ℕ}
    (hDimension : 0 < dimension)
    (hL : 0 ≤ L)
    (hProfile : IntervalIntegrable
      (fun x => profile x * basis x) volume 0 1)
    (hReconstructionIntegrable : IntervalIntegrable
      (fun x => reconstruction x * basis x) volume 0 1)
    (hReconstruction : ∀ x ∈ Set.uIoc (0 : ℝ) 1,
      reconstruction x = profile (node x))
    (hLipschitz : ∀ x ∈ Set.uIoc (0 : ℝ) 1,
      |profile x - profile (node x)| ≤ L * |x - node x|)
    (hVoronoiRadius : ∀ x ∈ Set.uIoc (0 : ℝ) 1,
      |x - node x| ≤ 1 / (2 * (dimension : ℝ)))
    (hBasis : ∀ x ∈ Set.uIoc (0 : ℝ) 1,
      |basis x| ≤ basisBound) :
    |continuousProfileCoefficient profile basis
        - continuousProfileCoefficient reconstruction basis|
      ≤ L * basisBound / (2 * dimension) := by
  have hPointwise : ∀ x ∈ Set.uIoc (0 : ℝ) 1,
      |profile x - reconstruction x| ≤ (L / 2) / dimension := by
    intro x hx
    rw [hReconstruction x hx]
    calc
      |profile x - profile (node x)|
          ≤ L * |x - node x| := hLipschitz x hx
      _ ≤ L * (1 / (2 * (dimension : ℝ))) :=
        mul_le_mul_of_nonneg_left (hVoronoiRadius x hx) hL
      _ = (L / 2) / dimension := by ring
  have hRate := continuousProfileCoefficient_inverse_grid_rate
    (constant := L / 2)
    hDimension
    (div_nonneg hL (by norm_num : (0 : ℝ) ≤ 2))
    hProfile
    hReconstructionIntegrable
    hPointwise
    hBasis
  calc
    |continuousProfileCoefficient profile basis
        - continuousProfileCoefficient reconstruction basis|
      ≤ (L / 2) * basisBound / dimension := hRate
    _ = L * basisBound / (2 * dimension) := by ring

theorem frequency_penalty_cannot_increase_coefficient_error
    {first second penalty : ℝ}
    (hPenalty : 1 ≤ penalty) :
    |first / penalty - second / penalty| ≤ |first - second| := by
  have hPositive : 0 < penalty := lt_of_lt_of_le zero_lt_one hPenalty
  rw [div_sub_div_same, abs_div]
  simpa [abs_of_pos hPositive] using
    (div_le_self (abs_nonneg (first - second)) hPenalty)

end SCOLHKG.Real
