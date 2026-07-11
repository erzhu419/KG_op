import Mathlib

namespace SCOLHKG.Real

open scoped BigOperators

/-!
Finite-design orthogonality for the ordered semiparametric expert.  The
ordered dictionary is fitted first.  V23 used a least-squares residual; V24
instead projects bounded kernel-center coefficients into the nullspace of the
ordered/kernel cross matrix.  The latter remains orthogonal on the frozen
unlabelled design without subtracting an unbounded ordered polynomial away
from that design.
-/

def finiteFeatureDot
    {Index : Type*} [Fintype Index]
    (left right : Index → ℝ) : ℝ :=
  ∑ i, left i * right i

def finiteSpanPrediction
    {Index Feature : Type*} [Fintype Feature]
    (basis : Feature → Index → ℝ)
    (coefficient : Feature → ℝ)
    (i : Index) : ℝ :=
  ∑ feature, coefficient feature * basis feature i

def finiteSemiparametricResidual
    {Index Feature : Type*} [Fintype Feature]
    (localFeature : Index → ℝ)
    (basis : Feature → Index → ℝ)
    (coefficient : Feature → ℝ)
    (i : Index) : ℝ :=
  localFeature i - finiteSpanPrediction basis coefficient i

theorem finiteFeatureDot_sub
    {Index : Type*} [Fintype Index]
    (left right projection : Index → ℝ) :
    finiteFeatureDot left (fun i => right i - projection i) =
      finiteFeatureDot left right - finiteFeatureDot left projection := by
  unfold finiteFeatureDot
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _hi
  ring

theorem finiteFeatureDot_spanPrediction
    {Index Feature : Type*}
    [Fintype Index] [Fintype Feature]
    (basis : Feature → Index → ℝ)
    (coefficient : Feature → ℝ)
    (feature : Feature) :
    finiteFeatureDot (basis feature)
        (finiteSpanPrediction basis coefficient) =
      ∑ other, coefficient other
        * finiteFeatureDot (basis feature) (basis other) := by
  unfold finiteFeatureDot finiteSpanPrediction
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro other _hother
  apply Finset.sum_congr rfl
  intro i _hi
  ring

theorem residual_orthogonal_of_normal_equations
    {Index Feature : Type*}
    [Fintype Index] [Fintype Feature]
    (localFeature : Index → ℝ)
    (basis : Feature → Index → ℝ)
    (coefficient : Feature → ℝ)
    (hNormal : ∀ feature,
      finiteFeatureDot (basis feature) localFeature =
        ∑ other, coefficient other
          * finiteFeatureDot (basis feature) (basis other)) :
    ∀ feature,
      finiteFeatureDot (basis feature)
        (finiteSemiparametricResidual localFeature basis coefficient) = 0 := by
  intro feature
  change finiteFeatureDot (basis feature)
      (fun i => localFeature i - finiteSpanPrediction basis coefficient i) = 0
  rw [finiteFeatureDot_sub]
  rw [finiteFeatureDot_spanPrediction]
  rw [hNormal feature]
  ring

def finiteKernelCombination
    {Index Center : Type*} [Fintype Center]
    (kernel : Center → Index → ℝ)
    (coefficient : Center → ℝ)
    (i : Index) : ℝ :=
  ∑ center, coefficient center * kernel center i

theorem finiteFeatureDot_kernelCombination
    {Index Center : Type*}
    [Fintype Index] [Fintype Center]
    (orderedFeature : Index → ℝ)
    (kernel : Center → Index → ℝ)
    (coefficient : Center → ℝ) :
    finiteFeatureDot orderedFeature
        (finiteKernelCombination kernel coefficient) =
      ∑ center, coefficient center
        * finiteFeatureDot orderedFeature (kernel center) := by
  unfold finiteFeatureDot finiteKernelCombination
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro center _hcenter
  apply Finset.sum_congr rfl
  intro i _hi
  ring

theorem coefficientNullspace_orthogonal
    {Index Center : Type*}
    [Fintype Index] [Fintype Center]
    (orderedFeature : Index → ℝ)
    (kernel : Center → Index → ℝ)
    (coefficient : Center → ℝ)
    (hNull :
      ∑ center, coefficient center
        * finiteFeatureDot orderedFeature (kernel center) = 0) :
    finiteFeatureDot orderedFeature
        (finiteKernelCombination kernel coefficient) = 0 := by
  rw [finiteFeatureDot_kernelCombination]
  exact hNull

theorem coefficientNullspace_orthogonal_all
    {Index Ordered Center Residual : Type*}
    [Fintype Index] [Fintype Center]
    (orderedFeature : Ordered → Index → ℝ)
    (kernel : Center → Index → ℝ)
    (projection : Center → Residual → ℝ)
    (hNull : ∀ ordered residual,
      ∑ center, projection center residual
        * finiteFeatureDot (orderedFeature ordered) (kernel center) = 0) :
    ∀ ordered residual,
      finiteFeatureDot (orderedFeature ordered)
        (finiteKernelCombination kernel
          (fun center => projection center residual)) = 0 := by
  intro ordered residual
  exact coefficientNullspace_orthogonal
    (orderedFeature ordered) kernel
    (fun center => projection center residual)
    (hNull ordered residual)

theorem finiteKernelCombination_abs_le_l1
    {Index Center : Type*} [Fintype Center]
    (kernel : Center → Index → ℝ)
    (coefficient : Center → ℝ)
    (i : Index)
    (hKernel : ∀ center, |kernel center i| ≤ 1) :
    |finiteKernelCombination kernel coefficient i| ≤
      ∑ center, |coefficient center| := by
  unfold finiteKernelCombination
  calc
    |∑ center, coefficient center * kernel center i| ≤
        ∑ center, |coefficient center * kernel center i| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ center, |coefficient center| * |kernel center i| := by
      apply Finset.sum_congr rfl
      intro center _hcenter
      rw [abs_mul]
    _ ≤ ∑ center, |coefficient center| := by
      apply Finset.sum_le_sum
      intro center _hcenter
      nlinarith [abs_nonneg (coefficient center), hKernel center]

theorem finiteKernelCombination_abs_le_card
    {Index Center : Type*} [Fintype Center]
    (kernel : Center → Index → ℝ)
    (coefficient : Center → ℝ)
    (i : Index)
    (hKernel : ∀ center, |kernel center i| ≤ 1)
    (hCoefficient : ∀ center, |coefficient center| ≤ 1) :
    |finiteKernelCombination kernel coefficient i| ≤
      (Fintype.card Center : ℝ) := by
  calc
    |finiteKernelCombination kernel coefficient i| ≤
        ∑ center, |coefficient center| :=
      finiteKernelCombination_abs_le_l1 kernel coefficient i hKernel
    _ ≤ ∑ _center : Center, (1 : ℝ) := by
      apply Finset.sum_le_sum
      intro center _hcenter
      exact hCoefficient center
    _ = (Fintype.card Center : ℝ) := by simp

end SCOLHKG.Real
