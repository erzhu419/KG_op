import SCOLHKG.Real.CurrentCovariance
import Mathlib.MeasureTheory.Integral.Bochner.Basic

/-!
# Fixed affine source rules: relationship mismatch and mean bias

The covariance and cross-covariance below are actual Bochner integrals on an
arbitrary probability space. Explicit integrability assumptions prevent the
totalized integral from supplying a spurious second moment. No Gaussian law or
matrix inverse is needed: the reference coefficient satisfies the normal
equations. This concerns a fixed affine rule, not the nonlinear mixture AFTER.
-/

namespace SCOLHKG.Real.CurrentAffineMismatch

open MeasureTheory
open scoped BigOperators
open CurrentCovariance

variable {ι Ω : Type*} [Fintype ι] [MeasurableSpace Ω]

def linearCombination (a : ι → ℝ) (Y : ι → Ω → ℝ) (ω : Ω) : ℝ :=
  ∑ i, a i * Y i ω

noncomputable def centered (μ : Measure Ω) (X : Ω → ℝ) (ω : Ω) : ℝ :=
  X ω - ∫ t, X t ∂μ

noncomputable def covariance (μ : Measure Ω) (Y : ι → Ω → ℝ) (i j : ι) : ℝ :=
  ∫ ω, centered μ (Y i) ω * centered μ (Y j) ω ∂μ

noncomputable def crossCovariance (μ : Measure Ω) (Y : ι → Ω → ℝ)
    (U : Ω → ℝ) (i : ι) : ℝ :=
  ∫ ω, centered μ (Y i) ω * centered μ U ω ∂μ

noncomputable def targetVariance (μ : Measure Ω) (U : Ω → ℝ) : ℝ :=
  ∫ ω, (centered μ U ω) ^ 2 ∂μ

noncomputable def sourceBias (μ : Measure Ω) (Y : ι → Ω → ℝ)
    (U : Ω → ℝ) (a mY : ι → ℝ) (mU : ℝ) : ℝ :=
  mU - (∫ ω, U ω ∂μ) + ∑ i, a i * ((∫ ω, Y i ω ∂μ) - mY i)

theorem quadratic_difference_expansion (S : ι → ι → ℝ) (a b : ι → ℝ) :
    quadraticForm S (fun i => a i - b i) =
      quadraticForm S a - (∑ i, ∑ j, a i * S i j * b j) -
        (∑ i, ∑ j, b i * S i j * a j) + quadraticForm S b := by
  simp only [quadraticForm, sub_mul, mul_sub, Finset.sum_sub_distrib]
  ring

theorem normal_equation_cross (S : ι → ι → ℝ) (a aStar c : ι → ℝ)
    (hSymm : ∀ i j, S i j = S j i)
    (hNormal : ∀ j, ∑ i, aStar i * S i j = c j) :
    (∑ i, ∑ j, a i * S i j * aStar j) = ∑ i, a i * c i := by
  apply Finset.sum_congr rfl
  intro i _
  rw [← hNormal i, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  rw [hSymm i j]
  ring

theorem affine_risk_completed_square (S : ι → ι → ℝ)
    (a aStar c : ι → ℝ) (v b : ℝ)
    (hSymm : ∀ i j, S i j = S j i)
    (hNormal : ∀ j, ∑ i, aStar i * S i j = c j) :
    v + quadraticForm S a - 2 * (∑ i, a i * c i) + b ^ 2 =
      v - quadraticForm S aStar +
        quadraticForm S (fun i => a i - aStar i) + b ^ 2 := by
  have hLeft := normal_equation_cross S a aStar c hSymm hNormal
  have hRight : (∑ i, ∑ j, aStar i * S i j * a j) = ∑ i, a i * c i := by
    rw [Finset.sum_comm]
    calc
      (∑ j, ∑ i, aStar i * S i j * a j) =
          ∑ j, (∑ i, aStar i * S i j) * a j := by
        simp only [Finset.sum_mul]
      _ = ∑ j, a j * c j := by
        apply Finset.sum_congr rfl
        intro j _
        rw [hNormal j]
        ring
  rw [quadratic_difference_expansion, hLeft, hRight]
  ring

theorem relationship_mismatch_nonnegative (S : ι → ι → ℝ)
    (hS : CovariancePSD S) (a aStar : ι → ℝ) :
    0 ≤ quadraticForm S (fun i => a i - aStar i) :=
  hS.2 _

theorem centered_integrable (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Ω → ℝ) (hX : Integrable X μ) : Integrable (centered μ X) μ :=
  hX.sub (integrable_const _)

theorem centered_integral_zero (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Ω → ℝ) (hX : Integrable X μ) :
    (∫ ω, centered μ X ω ∂μ) = 0 := by
  unfold centered
  rw [integral_sub hX (integrable_const _)]
  simp

theorem linearCombination_integrable (μ : Measure Ω)
    (a : ι → ℝ) (Y : ι → Ω → ℝ) (hY : ∀ i, Integrable (Y i) μ) :
    Integrable (linearCombination a Y) μ := by
  exact integrable_finsetSum Finset.univ (fun i _ => (hY i).const_mul (a i))

theorem integral_linearCombination (μ : Measure Ω)
    (a : ι → ℝ) (Y : ι → Ω → ℝ) (hY : ∀ i, Integrable (Y i) μ) :
    (∫ ω, linearCombination a Y ω ∂μ) = ∑ i, a i * (∫ ω, Y i ω ∂μ) := by
  unfold linearCombination
  rw [integral_finsetSum Finset.univ (fun i _ => (hY i).const_mul (a i))]
  simp only [integral_const_mul]

theorem linearCombination_square_moments (μ : Measure Ω)
    (a : ι → ℝ) (Y : ι → Ω → ℝ)
    (hYY : ∀ i j, Integrable (fun ω => Y i ω * Y j ω) μ) :
    Integrable (fun ω => (linearCombination a Y ω) ^ 2) μ ∧
      (∫ ω, (linearCombination a Y ω) ^ 2 ∂μ) =
        quadraticForm (fun i j => ∫ ω, Y i ω * Y j ω ∂μ) a := by
  have heq : (fun ω => (linearCombination a Y ω) ^ 2) =
      (fun ω => ∑ i, ∑ j, (a i * a j) * (Y i ω * Y j ω)) := by
    funext ω
    simp only [linearCombination, pow_two, Fintype.sum_mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    apply Finset.sum_congr rfl
    intro j _
    ring
  have hTerm := fun i j => (hYY i j).const_mul (a i * a j)
  have hRow := fun i => integrable_finsetSum Finset.univ (fun j _ => hTerm i j)
  rw [heq]
  constructor
  · exact integrable_finsetSum Finset.univ (fun i _ => hRow i)
  · rw [integral_finsetSum Finset.univ (fun i _ => hRow i)]
    unfold quadraticForm
    apply Finset.sum_congr rfl
    intro i _
    rw [integral_finsetSum Finset.univ (fun j _ => hTerm i j)]
    apply Finset.sum_congr rfl
    intro j _
    rw [integral_const_mul]
    ring

theorem linearCombination_cross_moments (μ : Measure Ω)
    (a : ι → ℝ) (Y : ι → Ω → ℝ) (U : Ω → ℝ)
    (hYU : ∀ i, Integrable (fun ω => Y i ω * U ω) μ) :
    Integrable (fun ω => linearCombination a Y ω * U ω) μ ∧
      (∫ ω, linearCombination a Y ω * U ω ∂μ) =
        ∑ i, a i * (∫ ω, Y i ω * U ω ∂μ) := by
  have heq : (fun ω => linearCombination a Y ω * U ω) =
      (fun ω => ∑ i, a i * (Y i ω * U ω)) := by
    funext ω
    simp only [linearCombination, Finset.sum_mul, mul_assoc]
  rw [heq]
  constructor
  · exact integrable_finsetSum Finset.univ (fun i _ => (hYU i).const_mul (a i))
  · rw [integral_finsetSum Finset.univ (fun i _ => (hYU i).const_mul (a i))]
    simp only [integral_const_mul]

theorem centered_residual_moments (μ : Measure Ω)
    (a : ι → ℝ) (Y : ι → Ω → ℝ) (U : Ω → ℝ)
    (hYY : ∀ i j, Integrable (fun ω => Y i ω * Y j ω) μ)
    (hYU : ∀ i, Integrable (fun ω => Y i ω * U ω) μ)
    (hU2 : Integrable (fun ω => (U ω) ^ 2) μ) :
    Integrable (fun ω => (linearCombination a Y ω - U ω) ^ 2) μ ∧
      (∫ ω, (linearCombination a Y ω - U ω) ^ 2 ∂μ) =
        (∫ ω, (U ω) ^ 2 ∂μ) +
          quadraticForm (fun i j => ∫ ω, Y i ω * Y j ω ∂μ) a -
          2 * (∑ i, a i * (∫ ω, Y i ω * U ω ∂μ)) := by
  obtain ⟨hL2, hL2eq⟩ := linearCombination_square_moments μ a Y hYY
  obtain ⟨hLU, hLUeq⟩ := linearCombination_cross_moments μ a Y U hYU
  have heq : (fun ω => (linearCombination a Y ω - U ω) ^ 2) =
      (fun ω => ((linearCombination a Y ω) ^ 2 + (U ω) ^ 2) -
        2 * (linearCombination a Y ω * U ω)) := by
    funext ω
    ring
  rw [heq]
  constructor
  · exact (hL2.add hU2).sub (hLU.const_mul 2)
  · rw [integral_sub
      (f := fun ω => (linearCombination a Y ω) ^ 2 + (U ω) ^ 2)
      (g := fun ω => 2 * (linearCombination a Y ω * U ω))
      (hL2.add hU2) (hLU.const_mul 2),
      integral_add hL2 hU2, integral_const_mul, hL2eq, hLUeq]
    ring

theorem centered_scalar_bias (μ : Measure Ω) [IsProbabilityMeasure μ]
    (R : Ω → ℝ) (b : ℝ) (hR : Integrable R μ)
    (hR2 : Integrable (fun ω => (R ω) ^ 2) μ)
    (hZero : (∫ ω, R ω ∂μ) = 0) :
    (∫ ω, (R ω + b) ^ 2 ∂μ) = (∫ ω, (R ω) ^ 2 ∂μ) + b ^ 2 := by
  have heq : (fun ω => (R ω + b) ^ 2) =
      (fun ω => ((R ω) ^ 2 + (2 * b) * R ω) + b ^ 2) := by
    funext ω
    ring
  rw [heq, integral_add
    (f := fun ω => (R ω) ^ 2 + (2 * b) * R ω)
    (g := fun _ => b ^ 2)
    (hR2.add (hR.const_mul _)) (integrable_const _),
    integral_add hR2 (hR.const_mul _), integral_const_mul]
  simp [hZero]

theorem covariance_symmetric (μ : Measure Ω) (Y : ι → Ω → ℝ) (i j : ι) :
    covariance μ Y i j = covariance μ Y j i := by
  unfold covariance
  congr 1
  funext ω
  ring

theorem covariance_psd (μ : Measure Ω) (Y : ι → Ω → ℝ)
    (hYY : ∀ i j, Integrable
      (fun ω => centered μ (Y i) ω * centered μ (Y j) ω) μ) :
    CovariancePSD (covariance μ Y) := by
  refine ⟨covariance_symmetric μ Y, ?_⟩
  intro a
  have hMoment := (linearCombination_square_moments μ a
    (fun i => centered μ (Y i)) hYY).2
  change 0 ≤ quadraticForm
    (fun i j => ∫ ω, centered μ (Y i) ω * centered μ (Y j) ω ∂μ) a
  rw [← hMoment]
  exact integral_nonneg (fun ω => sq_nonneg _)

/-- Formula (11), with covariance and risk defined by integrals rather than
assumed risk identities. The reference row is any solution of `aStar S = c`.
The mixed second-moment assumptions are explicit; no Gaussianity is used. -/
theorem fixed_affine_mse_decomposition (μ : Measure Ω) [IsProbabilityMeasure μ]
    (Y : ι → Ω → ℝ) (U : Ω → ℝ) (a aStar mY : ι → ℝ) (mU : ℝ)
    (hY : ∀ i, Integrable (Y i) μ) (hU : Integrable U μ)
    (hYY : ∀ i j, Integrable
      (fun ω => centered μ (Y i) ω * centered μ (Y j) ω) μ)
    (hYU : ∀ i, Integrable
      (fun ω => centered μ (Y i) ω * centered μ U ω) μ)
    (hU2 : Integrable (fun ω => (centered μ U ω) ^ 2) μ)
    (hNormal : ∀ j, ∑ i, aStar i * covariance μ Y i j = crossCovariance μ Y U j) :
    (∫ ω, (mU + (∑ i, a i * (Y i ω - mY i)) - U ω) ^ 2 ∂μ) =
      targetVariance μ U - quadraticForm (covariance μ Y) aStar +
        quadraticForm (covariance μ Y) (fun i => a i - aStar i) +
        (sourceBias μ Y U a mY mU) ^ 2 := by
  let Y0 := fun i => centered μ (Y i)
  let U0 := centered μ U
  let R := fun ω => linearCombination a Y0 ω - U0 ω
  have hY0 : ∀ i, Integrable (Y0 i) μ := fun i => centered_integrable μ _ (hY i)
  have hU0 : Integrable U0 μ := centered_integrable μ _ hU
  have hR : Integrable R μ := (linearCombination_integrable μ a Y0 hY0).sub hU0
  have hYZero : ∀ i, (∫ ω, Y0 i ω ∂μ) = 0 :=
    fun i => centered_integral_zero μ _ (hY i)
  have hUZero : (∫ ω, U0 ω ∂μ) = 0 := centered_integral_zero μ _ hU
  have hRZero : (∫ ω, R ω ∂μ) = 0 := by
    change (∫ ω, linearCombination a Y0 ω - U0 ω ∂μ) = 0
    rw [integral_sub (linearCombination_integrable μ a Y0 hY0) hU0,
      integral_linearCombination μ a Y0 hY0]
    simp only [hYZero, hUZero, mul_zero, Finset.sum_const_zero, sub_self]
  obtain ⟨hR2, hR2eq⟩ := centered_residual_moments μ a Y0 U0 hYY hYU hU2
  have heq : ∀ ω, mU + (∑ i, a i * (Y i ω - mY i)) - U ω =
      R ω + sourceBias μ Y U a mY mU := by
    intro ω
    simp only [R, Y0, U0, linearCombination, centered, sourceBias,
      mul_sub, Finset.sum_sub_distrib]
    ring
  simp_rw [heq]
  rw [centered_scalar_bias μ R _ hR hR2 hRZero, hR2eq]
  exact affine_risk_completed_square (covariance μ Y) a aStar
    (crossCovariance μ Y U) (targetVariance μ U) _
    (covariance_symmetric μ Y) hNormal

end SCOLHKG.Real.CurrentAffineMismatch
