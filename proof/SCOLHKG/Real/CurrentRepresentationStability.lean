import SCOLHKG.Real.CurrentPropagation

namespace SCOLHKG.Real.CurrentRepresentationStability

open scoped BigOperators

/-!
Finite-coordinate stability for the current representation chain.

The midpoint cosine library, common-grid interpolation and observable-control
descriptor supply the reconstruction/quadrature premises below. Frozen mixing,
positive source standardization, quadratic interactions and clipped radial
normalization then propagate them. The actual mean coordinate is the first ten
columns of the normalized 65-column map; it is not an unnormalized DCT vector.
The risk coordinate also retains the actual control-group/occupancy descriptor,
so its encoder error cannot be inferred from a curve-only interpolation error.

No claim of sufficiency for arbitrary high-dimensional functions is made.
Encoder constants, retained spectral scales, profile reconstruction and
quadrature errors are explicit assumptions, rather than fitted quantities or
new computations. All identities concern exact real arithmetic.
-/

def linearForm {I : Type*} [Fintype I]
    (coefficient coordinate : I → ℝ) : ℝ :=
  ∑ i, coefficient i * coordinate i

def bilinearKernel {I J : Type*} [Fintype I] [Fintype J]
    (covariance : I → J → ℝ) (left : I → ℝ) (right : J → ℝ) : ℝ :=
  ∑ i, ∑ j, left i * covariance i j * right j

theorem common_reference_error
    {I : Type*} {first second reference : I → ℝ}
    {epsilonFirst epsilonSecond quadratureFirst quadratureSecond : ℝ}
    (hFirst : ∀ i, |first i - reference i| ≤ epsilonFirst + quadratureFirst)
    (hSecond : ∀ i, |second i - reference i| ≤ epsilonSecond + quadratureSecond) :
    ∀ i, |first i - second i| ≤
      epsilonFirst + epsilonSecond + quadratureFirst + quadratureSecond := by
  intro i
  calc
    |first i - second i| ≤ |first i - reference i| + |reference i - second i| :=
      abs_sub_le _ _ _
    _ = |first i - reference i| + |second i - reference i| := by
      rw [abs_sub_comm (reference i) (second i)]
    _ ≤ (epsilonFirst + quadratureFirst) + (epsilonSecond + quadratureSecond) :=
      add_le_add (hFirst i) (hSecond i)
    _ = epsilonFirst + epsilonSecond + quadratureFirst + quadratureSecond := by ring

theorem linearForm_error_le
    {I : Type*} [Fintype I] {coefficient first second : I → ℝ}
    {epsilon : ℝ} (hError : ∀ i, |first i - second i| ≤ epsilon) :
    |linearForm coefficient first - linearForm coefficient second| ≤
      (∑ i, |coefficient i|) * epsilon := by
  classical
  have hIdentity : linearForm coefficient first - linearForm coefficient second =
      ∑ i, coefficient i * (first i - second i) := by
    simp only [linearForm, ← Finset.sum_sub_distrib, mul_sub]
  rw [hIdentity]
  calc
    |∑ i, coefficient i * (first i - second i)| ≤
        ∑ i, |coefficient i * (first i - second i)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, |coefficient i| * epsilon := by
      apply Finset.sum_le_sum
      intro i _
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left (hError i) (abs_nonneg _)
    _ = (∑ i, |coefficient i|) * epsilon := by rw [Finset.sum_mul]

theorem frozen_linear_map_error
    {I J : Type*} [Fintype I] {mixing : J → I → ℝ}
    {first second : I → ℝ} {epsilon mass : ℝ}
    (hEpsilon : 0 ≤ epsilon)
    (hError : ∀ i, |first i - second i| ≤ epsilon)
    (hMass : ∀ j, ∑ i, |mixing j i| ≤ mass) :
    ∀ j, |linearForm (mixing j) first - linearForm (mixing j) second| ≤ mass * epsilon := by
  intro j
  exact (linearForm_error_le hError).trans
    (mul_le_mul_of_nonneg_right (hMass j) hEpsilon)

theorem positive_standardization_error
    {first second center scale epsilon : ℝ}
    (hScale : 0 < scale) (hError : |first - second| ≤ epsilon) :
    |(first - center) / scale - (second - center) / scale| ≤ epsilon / scale := by
  have hIdentity : (first - center) / scale - (second - center) / scale =
      (first - second) / scale := by ring
  rw [hIdentity, abs_div, abs_of_pos hScale]
  exact div_le_div_of_nonneg_right hError hScale.le

theorem retained_spectral_scale_error
    {first second center eigenvalue epsilon : ℝ}
    (hEigenvalue : 0 < eigenvalue) (hError : |first - second| ≤ epsilon) :
    |(first - center) / Real.sqrt eigenvalue -
        (second - center) / Real.sqrt eigenvalue| ≤ epsilon / Real.sqrt eigenvalue := by
  exact positive_standardization_error (Real.sqrt_pos.2 hEigenvalue) hError

theorem positive_standardization_uniform_error
    {I : Type*} {first second center scale : I → ℝ}
    {epsilon minimumScale : ℝ}
    (hEpsilon : 0 ≤ epsilon) (hMinimumScale : 0 < minimumScale)
    (hScale : ∀ i, minimumScale ≤ scale i)
    (hError : ∀ i, |first i - second i| ≤ epsilon) :
    ∀ i, |(first i - center i) / scale i - (second i - center i) / scale i| ≤
      epsilon / minimumScale := by
  intro i
  exact (positive_standardization_error
    (lt_of_lt_of_le hMinimumScale (hScale i)) (hError i)).trans
      (div_le_div_of_nonneg_left hEpsilon hMinimumScale (hScale i))

theorem product_feature_error
    {firstLeft secondLeft firstRight secondRight : ℝ}
    {epsilonLeft epsilonRight boundLeft boundRight : ℝ}
    (hEpsilonLeft : 0 ≤ epsilonLeft) (hEpsilonRight : 0 ≤ epsilonRight)
    (hLeft : |firstLeft - secondLeft| ≤ epsilonLeft)
    (hRight : |firstRight - secondRight| ≤ epsilonRight)
    (hBoundLeft : |secondLeft| ≤ boundLeft)
    (hBoundRight : |firstRight| ≤ boundRight) :
    |firstLeft * firstRight - secondLeft * secondRight| ≤
      epsilonLeft * boundRight + boundLeft * epsilonRight := by
  have hIdentity : firstLeft * firstRight - secondLeft * secondRight =
      (firstLeft - secondLeft) * firstRight + secondLeft * (firstRight - secondRight) := by ring
  rw [hIdentity]
  calc
    |(firstLeft - secondLeft) * firstRight + secondLeft * (firstRight - secondRight)| ≤
        |(firstLeft - secondLeft) * firstRight| + |secondLeft * (firstRight - secondRight)| :=
      abs_add_le _ _
    _ = |firstLeft - secondLeft| * |firstRight| +
        |secondLeft| * |firstRight - secondRight| := by rw [abs_mul, abs_mul]
    _ ≤ epsilonLeft * boundRight + boundLeft * epsilonRight := by
      exact add_le_add
        (mul_le_mul hLeft hBoundRight (abs_nonneg _) hEpsilonLeft)
        (mul_le_mul hBoundLeft hRight (abs_nonneg _) (le_trans (abs_nonneg _) hBoundLeft))

theorem quadratic_features_error
    {I : Type*} {first second : I → ℝ} {epsilon bound : ℝ}
    (hEpsilon : 0 ≤ epsilon) (hError : ∀ i, |first i - second i| ≤ epsilon)
    (hFirst : ∀ i, |first i| ≤ bound) (hSecond : ∀ i, |second i| ≤ bound) :
    ∀ i j, |first i * first j - second i * second j| ≤ 2 * bound * epsilon := by
  intro i j
  have h := product_feature_error hEpsilon hEpsilon (hError i) (hError j)
    (hSecond i) (hFirst j)
  calc
    |first i * first j - second i * second j| ≤ epsilon * bound + bound * epsilon := h
    _ = 2 * bound * epsilon := by ring

theorem clipped_radius_error
    {firstRadius secondRadius epsilon floor : ℝ}
    (hError : |firstRadius - secondRadius| ≤ epsilon) :
    |max floor firstRadius - max floor secondRadius| ≤ epsilon := by
  have h := abs_le.mp hError
  apply abs_le.mpr
  constructor
  · by_cases hFirst : floor ≤ firstRadius <;> by_cases hSecond : floor ≤ secondRadius
    · simp only [max_eq_right hFirst, max_eq_right hSecond]
      exact h.1
    · rw [max_eq_right hFirst, max_eq_left (le_of_not_ge hSecond)]
      linarith
    · rw [max_eq_left (le_of_not_ge hFirst), max_eq_right hSecond]
      linarith
    · simp only [max_eq_left (le_of_not_ge hFirst), max_eq_left (le_of_not_ge hSecond), sub_self]
      have : 0 ≤ epsilon := le_trans (abs_nonneg _) hError
      linarith
  · by_cases hFirst : floor ≤ firstRadius <;> by_cases hSecond : floor ≤ secondRadius
    · simp only [max_eq_right hFirst, max_eq_right hSecond]
      exact h.2
    · rw [max_eq_right hFirst, max_eq_left (le_of_not_ge hSecond)]
      linarith
    · rw [max_eq_left (le_of_not_ge hFirst), max_eq_right hSecond]
      linarith
    · simp only [max_eq_left (le_of_not_ge hFirst), max_eq_left (le_of_not_ge hSecond), sub_self]
      exact le_trans (abs_nonneg _) hError

theorem radial_normalization_coordinate_error
    {first second firstScale secondScale epsilon radiusError : ℝ}
    (hEpsilon : 0 ≤ epsilon) (hRadiusError : 0 ≤ radiusError)
    (hFirstScale : 1 ≤ firstScale) (hSecondScale : 1 ≤ secondScale)
    (hCoordinate : |first - second| ≤ epsilon)
    (hRadius : |firstScale - secondScale| ≤ radiusError)
    (hSecondCoordinate : |second| ≤ secondScale) :
    |first / firstScale - second / secondScale| ≤ epsilon + radiusError := by
  have hFirstPositive : 0 < firstScale := lt_of_lt_of_le zero_lt_one hFirstScale
  have hSecondPositive : 0 < secondScale := lt_of_lt_of_le zero_lt_one hSecondScale
  have hRatio : |second / secondScale| ≤ 1 := by
    rw [abs_div, abs_of_pos hSecondPositive]
    exact (div_le_one hSecondPositive).2 hSecondCoordinate
  have hIdentity : first / firstScale - second / secondScale =
      (first - second + (second / secondScale) * (secondScale - firstScale)) / firstScale := by
    field_simp [ne_of_gt hFirstPositive, ne_of_gt hSecondPositive]
    <;> ring
  rw [hIdentity, abs_div, abs_of_pos hFirstPositive]
  calc
    |first - second + (second / secondScale) * (secondScale - firstScale)| / firstScale ≤
        (|first - second| + |second / secondScale| * |secondScale - firstScale|) / firstScale := by
      apply div_le_div_of_nonneg_right _ hFirstPositive.le
      simpa only [abs_mul] using abs_add_le (first - second)
        ((second / secondScale) * (secondScale - firstScale))
    _ ≤ (epsilon + radiusError) / firstScale := by
      apply div_le_div_of_nonneg_right _ hFirstPositive.le
      apply add_le_add hCoordinate
      have hRadius' : |secondScale - firstScale| ≤ radiusError := by simpa only [abs_sub_comm] using hRadius
      simpa only [one_mul] using mul_le_mul hRatio hRadius' (abs_nonneg _) (by norm_num : (0 : ℝ) ≤ 1)
    _ ≤ epsilon + radiusError := div_le_self (add_nonneg hEpsilon hRadiusError) hFirstScale

theorem source_encoder_prediction_error
    {I J : Type*} [Fintype J] {first second : I → ℝ}
    {encoder : (I → ℝ) → J → ℝ} {coefficient : J → ℝ}
    {epsilon encoderConstant : ℝ}
    (hRepresentation : ∀ i, |first i - second i| ≤ epsilon)
    (hEncoder : (∀ i, |first i - second i| ≤ epsilon) →
      ∀ j, |encoder first j - encoder second j| ≤ encoderConstant * epsilon) :
    |linearForm coefficient (encoder first) - linearForm coefficient (encoder second)| ≤
      (∑ j, |coefficient j|) * (encoderConstant * epsilon) := by
  exact linearForm_error_le (hEncoder hRepresentation)

theorem cross_dimension_source_prediction_error
    {I J : Type*} [Fintype J] {first second reference : I → ℝ}
    {encoder : (I → ℝ) → J → ℝ} {coefficient : J → ℝ}
    {epsilonFirst epsilonSecond quadratureFirst quadratureSecond encoderConstant : ℝ}
    (hFirst : ∀ i, |first i - reference i| ≤ epsilonFirst + quadratureFirst)
    (hSecond : ∀ i, |second i - reference i| ≤ epsilonSecond + quadratureSecond)
    (hEncoder : (∀ i, |first i - second i| ≤
        epsilonFirst + epsilonSecond + quadratureFirst + quadratureSecond) →
      ∀ j, |encoder first j - encoder second j| ≤ encoderConstant *
        (epsilonFirst + epsilonSecond + quadratureFirst + quadratureSecond)) :
    |linearForm coefficient (encoder first) - linearForm coefficient (encoder second)| ≤
      (∑ j, |coefficient j|) * (encoderConstant *
        (epsilonFirst + epsilonSecond + quadratureFirst + quadratureSecond)) := by
  exact source_encoder_prediction_error (common_reference_error hFirst hSecond)
    hEncoder

theorem bilinear_kernel_error
    {I J : Type*} [Fintype I] [Fintype J]
    {covariance : I → J → ℝ} {firstLeft secondLeft : I → ℝ}
    {firstRight secondRight : J → ℝ}
    {epsilonLeft epsilonRight boundLeft boundRight : ℝ}
    (hEpsilonLeft : 0 ≤ epsilonLeft) (hEpsilonRight : 0 ≤ epsilonRight)
    (hLeft : ∀ i, |firstLeft i - secondLeft i| ≤ epsilonLeft)
    (hRight : ∀ j, |firstRight j - secondRight j| ≤ epsilonRight)
    (hBoundLeft : ∀ i, |secondLeft i| ≤ boundLeft)
    (hBoundRight : ∀ j, |firstRight j| ≤ boundRight) :
    |bilinearKernel covariance firstLeft firstRight -
        bilinearKernel covariance secondLeft secondRight| ≤
      (∑ i, ∑ j, |covariance i j|) *
        (epsilonLeft * boundRight + boundLeft * epsilonRight) := by
  classical
  have hIdentity : bilinearKernel covariance firstLeft firstRight -
      bilinearKernel covariance secondLeft secondRight =
      ∑ i, ∑ j, covariance i j * (firstLeft i * firstRight j - secondLeft i * secondRight j) := by
    simp only [bilinearKernel, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [hIdentity]
  calc
    |∑ i, ∑ j, covariance i j * (firstLeft i * firstRight j - secondLeft i * secondRight j)| ≤
        ∑ i, ∑ j, |covariance i j * (firstLeft i * firstRight j - secondLeft i * secondRight j)| :=
      (Finset.abs_sum_le_sum_abs _ _).trans (by
        apply Finset.sum_le_sum
        intro i _
        exact Finset.abs_sum_le_sum_abs _ _)
    _ ≤ ∑ i, ∑ j, |covariance i j| *
        (epsilonLeft * boundRight + boundLeft * epsilonRight) := by
      apply Finset.sum_le_sum
      intro i _
      apply Finset.sum_le_sum
      intro j _
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left
        (product_feature_error hEpsilonLeft hEpsilonRight (hLeft i) (hRight j)
          (hBoundLeft i) (hBoundRight j)) (abs_nonneg _)
    _ = (∑ i, ∑ j, |covariance i j|) *
        (epsilonLeft * boundRight + boundLeft * epsilonRight) := by simp only [Finset.sum_mul]

theorem positive_diagonal_reparameterization_mean
    {I : Type*} [Fintype I] {feature coefficient scale : I → ℝ}
    (hScale : ∀ i, 0 < scale i) :
    linearForm (fun i => scale i * coefficient i) (fun i => feature i / scale i) =
      linearForm coefficient feature := by
  classical
  apply Finset.sum_congr rfl
  intro i _
  dsimp
  field_simp [ne_of_gt (hScale i)]
  <;> ring

theorem positive_diagonal_reparameterization_kernel
    {I : Type*} [Fintype I] {left right scale : I → ℝ} {covariance : I → I → ℝ}
    (hScale : ∀ i, 0 < scale i) :
    bilinearKernel (fun i j => scale i * covariance i j * scale j)
        (fun i => left i / scale i) (fun j => right j / scale j) =
      bilinearKernel covariance left right := by
  classical
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  dsimp
  field_simp [ne_of_gt (hScale i), ne_of_gt (hScale j)]
  <;> ring

end SCOLHKG.Real.CurrentRepresentationStability
