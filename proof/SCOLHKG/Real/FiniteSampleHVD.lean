import Mathlib

namespace SCOLHKG.Real

open scoped BigOperators

/-!
Finite-sample identification of the active cumulative-HVD calibration law.

The Python implementation does not attempt to identify every raw cumulative
risk coefficient from a handful of target replications.  It freezes a
source-learned risk shape and fits either a scalar calibration or a small
source-shape mixture.  Consequently the relevant statistical dimension is the
active calibration dimension `q`, not the raw policy dimension and not
necessarily the full cumulative-feature dimension.

The results below formalize that exact finite-dimensional problem.  Exposure
excitation supplies coercivity of the active design.  A ridge basic inequality
and uniformly accurate replicated variance targets then imply parameter and
out-of-sample variance bounds.  A separate nonnegative remainder records
coordinate misspecification instead of silently assuming that the active
source span is exact.
-/

noncomputable section

def hvdLinearPrediction {q : ℕ}
    (feature parameter : Fin q → ℝ) : ℝ :=
  ∑ j, feature j * parameter j

def hvdSquaredNorm {q : ℕ} (parameter : Fin q → ℝ) : ℝ :=
  ∑ j, parameter j ^ 2

def hvdFeatureL1 {q : ℕ} (feature : Fin q → ℝ) : ℝ :=
  ∑ j, |feature j|

def ActiveHVDExcitation {n q : ℕ}
    (feature : Fin n → Fin q → ℝ) (kappa : ℝ) : Prop :=
  ∀ direction,
    kappa * hvdSquaredNorm direction ≤
      ∑ i, (hvdLinearPrediction (feature i) direction) ^ 2

def UniformReplicatedVarianceAccuracy {n q : ℕ}
    (feature : Fin n → Fin q → ℝ)
    (observed : Fin n → ℝ)
    (oracleParameter : Fin q → ℝ)
    (radius : ℝ) : Prop :=
  ∀ i,
    |hvdLinearPrediction (feature i) oracleParameter - observed i| ≤ radius

def ApproximateActiveHVDRidgeFit {n q : ℕ}
    (feature : Fin n → Fin q → ℝ)
    (observed : Fin n → ℝ)
    (ridge optimizationSlack : ℝ)
    (fitted oracleParameter : Fin q → ℝ) : Prop :=
  (∑ i,
      (hvdLinearPrediction (feature i) fitted - observed i) ^ 2)
      + ridge * hvdSquaredNorm fitted
    ≤
  (∑ i,
      (hvdLinearPrediction (feature i) oracleParameter - observed i) ^ 2)
      + ridge * hvdSquaredNorm oracleParameter
      + optimizationSlack

theorem hvdSquaredNorm_nonnegative {q : ℕ} (parameter : Fin q → ℝ) :
    0 ≤ hvdSquaredNorm parameter := by
  unfold hvdSquaredNorm
  exact Finset.sum_nonneg fun _ _ => sq_nonneg _

theorem hvdFeatureL1_nonnegative {q : ℕ} (feature : Fin q → ℝ) :
    0 ≤ hvdFeatureL1 feature := by
  unfold hvdFeatureL1
  exact Finset.sum_nonneg fun _ _ => abs_nonneg _

theorem hvdFeatureL1_le_dimension_mul
    {q : ℕ}
    (feature : Fin q → ℝ)
    {featureBound : ℝ}
    (hBound : ∀ j, |feature j| ≤ featureBound) :
    hvdFeatureL1 feature ≤ (q : ℝ) * featureBound := by
  unfold hvdFeatureL1
  calc
    (∑ j, |feature j|) ≤ ∑ _j : Fin q, featureBound := by
      exact Finset.sum_le_sum fun j _hj => hBound j
    _ = (q : ℝ) * featureBound := by simp

theorem hvdLinearPrediction_sub {q : ℕ}
    (feature first second : Fin q → ℝ) :
    hvdLinearPrediction feature (fun j => first j - second j) =
      hvdLinearPrediction feature first -
        hvdLinearPrediction feature second := by
  simp [hvdLinearPrediction, mul_sub, Finset.sum_sub_distrib]

theorem uniform_replicated_variance_squared_risk_le
    {n q : ℕ}
    {feature : Fin n → Fin q → ℝ}
    {observed : Fin n → ℝ}
    {oracleParameter : Fin q → ℝ}
    {radius : ℝ}
    (hUniform : UniformReplicatedVarianceAccuracy
      feature observed oracleParameter radius) :
    (∑ i,
      (hvdLinearPrediction (feature i) oracleParameter - observed i) ^ 2)
      ≤ n * radius ^ 2 := by
  calc
    (∑ i,
        (hvdLinearPrediction (feature i) oracleParameter - observed i) ^ 2)
        ≤ ∑ _i : Fin n, radius ^ 2 := by
          apply Finset.sum_le_sum
          intro i _hi
          have h := hUniform i
          rw [abs_le] at h
          nlinarith
    _ = n * radius ^ 2 := by simp

theorem active_hvd_prediction_difference_squared_le
    {n q : ℕ}
    (feature : Fin n → Fin q → ℝ)
    (fitted oracleParameter : Fin q → ℝ) :
    (∑ i,
      (hvdLinearPrediction (feature i)
        (fun j => fitted j - oracleParameter j)) ^ 2)
      ≤
      2 * (∑ i,
        (hvdLinearPrediction (feature i) fitted) ^ 2)
      + 2 * (∑ i,
        (hvdLinearPrediction (feature i) oracleParameter) ^ 2) := by
  calc
    (∑ i,
      (hvdLinearPrediction (feature i)
        (fun j => fitted j - oracleParameter j)) ^ 2)
        = ∑ i,
          (hvdLinearPrediction (feature i) fitted -
            hvdLinearPrediction (feature i) oracleParameter) ^ 2 := by
              apply Finset.sum_congr rfl
              intro i _hi
              rw [hvdLinearPrediction_sub]
    _ ≤ ∑ i,
        (2 * (hvdLinearPrediction (feature i) fitted) ^ 2
          + 2 * (hvdLinearPrediction (feature i) oracleParameter) ^ 2) := by
            apply Finset.sum_le_sum
            intro i _hi
            nlinarith [sq_nonneg
              (hvdLinearPrediction (feature i) fitted
                + hvdLinearPrediction (feature i) oracleParameter)]
    _ =
      2 * (∑ i, (hvdLinearPrediction (feature i) fitted) ^ 2)
        + 2 * (∑ i,
          (hvdLinearPrediction (feature i) oracleParameter) ^ 2) := by
            simp [Finset.sum_add_distrib, Finset.mul_sum]

theorem active_hvd_parameter_oracle_inequality
    {n q : ℕ}
    {feature : Fin n → Fin q → ℝ}
    {observed : Fin n → ℝ}
    {fitted oracleParameter : Fin q → ℝ}
    {kappa ridge radius optimizationSlack : ℝ}
    (hExcitation : ActiveHVDExcitation feature kappa)
    (hUniform : UniformReplicatedVarianceAccuracy
      feature observed oracleParameter radius)
    (hFit : ApproximateActiveHVDRidgeFit
      feature observed ridge optimizationSlack fitted oracleParameter)
    (hRidge : 0 ≤ ridge) :
    kappa * hvdSquaredNorm (fun j => fitted j - oracleParameter j)
      ≤
      4 * n * radius ^ 2
        + 2 * ridge * hvdSquaredNorm oracleParameter
        + 2 * optimizationSlack := by
  let fittedResidual : Fin n → ℝ := fun i =>
    hvdLinearPrediction (feature i) fitted - observed i
  let oracleResidual : Fin n → ℝ := fun i =>
    hvdLinearPrediction (feature i) oracleParameter - observed i
  have hOracle :
      (∑ i, (oracleResidual i) ^ 2) ≤ n * radius ^ 2 := by
    exact uniform_replicated_variance_squared_risk_le hUniform
  have hFitResidual :
      (∑ i, (fittedResidual i) ^ 2)
        ≤ (∑ i, (oracleResidual i) ^ 2)
          + ridge * hvdSquaredNorm oracleParameter
          + optimizationSlack := by
    have hPenalty : 0 ≤ ridge * hvdSquaredNorm fitted :=
      mul_nonneg hRidge (hvdSquaredNorm_nonnegative fitted)
    dsimp [ApproximateActiveHVDRidgeFit] at hFit
    dsimp [fittedResidual, oracleResidual]
    linarith
  have hDifference :
      (∑ i,
        (hvdLinearPrediction (feature i)
          (fun j => fitted j - oracleParameter j)) ^ 2)
        ≤
        2 * (∑ i, (fittedResidual i) ^ 2)
          + 2 * (∑ i, (oracleResidual i) ^ 2) := by
    have hIdentity : ∀ i,
        hvdLinearPrediction (feature i)
            (fun j => fitted j - oracleParameter j)
          = fittedResidual i - oracleResidual i := by
      intro i
      simp only [fittedResidual, oracleResidual]
      rw [hvdLinearPrediction_sub]
      ring
    calc
      (∑ i,
        (hvdLinearPrediction (feature i)
          (fun j => fitted j - oracleParameter j)) ^ 2)
          = ∑ i, (fittedResidual i - oracleResidual i) ^ 2 := by
              apply Finset.sum_congr rfl
              intro i _hi
              rw [hIdentity i]
      _ ≤ ∑ i,
          (2 * (fittedResidual i) ^ 2
            + 2 * (oracleResidual i) ^ 2) := by
              apply Finset.sum_le_sum
              intro i _hi
              nlinarith [sq_nonneg (fittedResidual i + oracleResidual i)]
      _ = 2 * (∑ i, (fittedResidual i) ^ 2)
          + 2 * (∑ i, (oracleResidual i) ^ 2) := by
              simp [Finset.sum_add_distrib, Finset.mul_sum]
  have hUpper :
      (∑ i,
        (hvdLinearPrediction (feature i)
          (fun j => fitted j - oracleParameter j)) ^ 2)
      ≤
      4 * n * radius ^ 2
        + 2 * ridge * hvdSquaredNorm oracleParameter
        + 2 * optimizationSlack := by
    linarith
  exact (hExcitation (fun j => fitted j - oracleParameter j)).trans hUpper

theorem active_hvd_identifiable
    {n q : ℕ}
    {feature : Fin n → Fin q → ℝ}
    {first second : Fin q → ℝ}
    {kappa : ℝ}
    (hKappa : 0 < kappa)
    (hExcitation : ActiveHVDExcitation feature kappa)
    (hEqual : ∀ i,
      hvdLinearPrediction (feature i) first =
        hvdLinearPrediction (feature i) second) :
    first = second := by
  funext j
  have hZeroPrediction :
      (∑ i,
        (hvdLinearPrediction (feature i)
          (fun k => first k - second k)) ^ 2) = 0 := by
    apply Finset.sum_eq_zero
    intro i _hi
    rw [hvdLinearPrediction_sub, hEqual i, sub_self, zero_pow]
    norm_num
  have hNorm :
      kappa * hvdSquaredNorm (fun k => first k - second k) ≤ 0 := by
    simpa [hZeroPrediction] using
      (hExcitation (fun k => first k - second k))
  have hNormZero :
      hvdSquaredNorm (fun k => first k - second k) = 0 := by
    have hNonnegative :=
      hvdSquaredNorm_nonnegative (fun k => first k - second k)
    nlinarith
  have hCoordinate : (first j - second j) ^ 2 = 0 := by
    have hTerm :
        (first j - second j) ^ 2 ≤
          hvdSquaredNorm (fun k => first k - second k) := by
      unfold hvdSquaredNorm
      exact Finset.single_le_sum
        (fun k _hk => sq_nonneg (first k - second k))
        (Finset.mem_univ j)
    nlinarith
  nlinarith

theorem hvd_prediction_error_le_l1_radius
    {q : ℕ}
    (feature fitted oracleParameter : Fin q → ℝ)
    {radius : ℝ}
    (hCoordinate : ∀ j, |fitted j - oracleParameter j| ≤ radius) :
    |hvdLinearPrediction feature fitted -
        hvdLinearPrediction feature oracleParameter|
      ≤ hvdFeatureL1 feature * radius := by
  calc
    |hvdLinearPrediction feature fitted -
        hvdLinearPrediction feature oracleParameter|
      = |∑ j, feature j * (fitted j - oracleParameter j)| := by
          rw [← hvdLinearPrediction_sub]
          rfl
    _ ≤ ∑ j, |feature j * (fitted j - oracleParameter j)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ j, |feature j| * |fitted j - oracleParameter j| := by
      apply Finset.sum_congr rfl
      intro j _hj
      exact abs_mul _ _
    _ ≤ ∑ j, |feature j| * radius := by
      apply Finset.sum_le_sum
      intro j _hj
      exact mul_le_mul_of_nonneg_left (hCoordinate j) (abs_nonneg _)
    _ = hvdFeatureL1 feature * radius := by
      simp [hvdFeatureL1, Finset.sum_mul]

theorem active_hvd_coordinate_error_le
    {q : ℕ}
    {fitted oracleParameter : Fin q → ℝ}
    {kappa errorBudget : ℝ}
    (hKappa : 0 < kappa)
    (hParameter :
      kappa * hvdSquaredNorm (fun j => fitted j - oracleParameter j)
        ≤ errorBudget) :
    ∀ j,
      |fitted j - oracleParameter j| ≤
        Real.sqrt (errorBudget / kappa) := by
  intro j
  have hNorm :
      hvdSquaredNorm (fun k => fitted k - oracleParameter k)
        ≤ errorBudget / kappa := by
    exact (le_div_iff₀ hKappa).2 (by
      simpa [mul_comm] using hParameter)
  have hCoordinate :
      (fitted j - oracleParameter j) ^ 2 ≤
        hvdSquaredNorm (fun k => fitted k - oracleParameter k) := by
    unfold hvdSquaredNorm
    exact Finset.single_le_sum
      (fun k _hk => sq_nonneg (fitted k - oracleParameter k))
      (Finset.mem_univ j)
  exact Real.abs_le_sqrt (hCoordinate.trans hNorm)

theorem active_hvd_misspecification_upper
    {q : ℕ}
    {feature fitted oracleParameter : Fin q → ℝ}
    {trueVariance misspecification parameterRadius : ℝ}
    (hCoordinate : ∀ j,
      |fitted j - oracleParameter j| ≤ parameterRadius)
    (hTruth :
      trueVariance ≤
        hvdLinearPrediction feature oracleParameter + misspecification) :
    trueVariance ≤
      hvdLinearPrediction feature fitted
        + hvdFeatureL1 feature * parameterRadius
        + misspecification := by
  have hPrediction := hvd_prediction_error_le_l1_radius
    feature fitted oracleParameter hCoordinate
  have hOracleUpper :
      hvdLinearPrediction feature oracleParameter ≤
        hvdLinearPrediction feature fitted
          + hvdFeatureL1 feature * parameterRadius := by
    rw [abs_le] at hPrediction
    linarith
  linarith

theorem active_hvd_misspecification_absolute
    {q : ℕ}
    {feature fitted oracleParameter : Fin q → ℝ}
    {trueVariance misspecification parameterRadius : ℝ}
    (hCoordinate : ∀ j,
      |fitted j - oracleParameter j| ≤ parameterRadius)
    (hTruth :
      |trueVariance - hvdLinearPrediction feature oracleParameter|
        ≤ misspecification) :
    |trueVariance - hvdLinearPrediction feature fitted| ≤
      hvdFeatureL1 feature * parameterRadius + misspecification := by
  have hPrediction := hvd_prediction_error_le_l1_radius
    feature fitted oracleParameter hCoordinate
  have hTriangle :
      |trueVariance - hvdLinearPrediction feature fitted| ≤
        |trueVariance - hvdLinearPrediction feature oracleParameter|
          + |hvdLinearPrediction feature oracleParameter
              - hvdLinearPrediction feature fitted| := by
    calc
      |trueVariance - hvdLinearPrediction feature fitted|
        = |(trueVariance - hvdLinearPrediction feature oracleParameter)
            + (hvdLinearPrediction feature oracleParameter
              - hvdLinearPrediction feature fitted)| := by ring_nf
      _ ≤ |trueVariance - hvdLinearPrediction feature oracleParameter|
          + |hvdLinearPrediction feature oracleParameter
              - hvdLinearPrediction feature fitted| := abs_add_le _ _
  rw [abs_sub_comm
    (hvdLinearPrediction feature oracleParameter)
    (hvdLinearPrediction feature fitted)] at hTriangle
  linarith

theorem active_hvd_certification_overcoverage_upper
    {q : ℕ}
    {feature fitted oracleParameter : Fin q → ℝ}
    {trueVariance certVariance misspecification parameterRadius guard : ℝ}
    (hCoordinate : ∀ j,
      |fitted j - oracleParameter j| ≤ parameterRadius)
    (hTruth :
      |trueVariance - hvdLinearPrediction feature oracleParameter|
        ≤ misspecification)
    (hCert :
      certVariance ≤ hvdLinearPrediction feature fitted + guard) :
    certVariance ≤ trueVariance
      + hvdFeatureL1 feature * parameterRadius
      + misspecification + guard := by
  have hAbsolute := active_hvd_misspecification_absolute
    hCoordinate hTruth
  rw [abs_le] at hAbsolute
  linarith

theorem active_hvd_finite_sample_variance_upper
    {n q : ℕ}
    {design : Fin n → Fin q → ℝ}
    {observed : Fin n → ℝ}
    {feature fitted oracleParameter : Fin q → ℝ}
    {kappa ridge radius optimizationSlack : ℝ}
    {trueVariance misspecification : ℝ}
    (hKappa : 0 < kappa)
    (hRidge : 0 ≤ ridge)
    (hOptimization : 0 ≤ optimizationSlack)
    (hExcitation : ActiveHVDExcitation design kappa)
    (hUniform : UniformReplicatedVarianceAccuracy
      design observed oracleParameter radius)
    (hFit : ApproximateActiveHVDRidgeFit
      design observed ridge optimizationSlack fitted oracleParameter)
    (hTruth :
      trueVariance ≤
        hvdLinearPrediction feature oracleParameter + misspecification) :
    trueVariance ≤
      hvdLinearPrediction feature fitted
        + hvdFeatureL1 feature *
          Real.sqrt (
            (4 * n * radius ^ 2
              + 2 * ridge * hvdSquaredNorm oracleParameter
              + 2 * optimizationSlack) / kappa)
        + misspecification := by
  let errorBudget :=
    4 * n * radius ^ 2
      + 2 * ridge * hvdSquaredNorm oracleParameter
      + 2 * optimizationSlack
  have hBudget : 0 ≤ errorBudget := by
    dsimp [errorBudget]
    exact add_nonneg
      (add_nonneg
        (mul_nonneg
          (mul_nonneg (by norm_num) (Nat.cast_nonneg n))
          (sq_nonneg radius))
        (mul_nonneg
          (mul_nonneg (by norm_num) hRidge)
          (hvdSquaredNorm_nonnegative oracleParameter)))
      (mul_nonneg (by norm_num) hOptimization)
  have hParameter :
      kappa * hvdSquaredNorm (fun j => fitted j - oracleParameter j)
        ≤ errorBudget := by
    exact active_hvd_parameter_oracle_inequality
      hExcitation hUniform hFit hRidge
  have hCoordinate := active_hvd_coordinate_error_le
    hKappa hParameter
  exact active_hvd_misspecification_upper
    hCoordinate hTruth

end

end SCOLHKG.Real
