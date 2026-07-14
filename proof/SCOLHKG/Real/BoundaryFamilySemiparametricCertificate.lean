import SCOLHKG.Real.BoundaryFamilySynthesisCertificate
import SCOLHKG.Real.OrthogonalSemiparametric

namespace SCOLHKG.Real

/-!
Finite implementation contract for TCB-V5.  A nonnegative source-family
synthesis and a frozen orthogonal kernel residual form a direct-sum mean.
Both posterior covariance blocks and the remaining residual scale enter the
upper margin through nonnegative squared radii.
-/

noncomputable def semiparametricBoundaryMean
    (synthesis residual : ℝ) : ℝ :=
  synthesis + residual

noncomputable def semiparametricBoundaryVariance
    {SynthesisParameter ResidualParameter : Type*}
    [Fintype SynthesisParameter] [Fintype ResidualParameter]
    (synthesisLoading : SynthesisParameter → ℝ)
    (residualLoading : ResidualParameter → ℝ)
    (noiseScale : ℝ) : ℝ :=
  synthesisParameterVariance synthesisLoading
    + synthesisParameterVariance residualLoading
    + noiseScale ^ 2

theorem semiparametricBoundaryVariance_nonnegative
    {SynthesisParameter ResidualParameter : Type*}
    [Fintype SynthesisParameter] [Fintype ResidualParameter]
    (synthesisLoading : SynthesisParameter → ℝ)
    (residualLoading : ResidualParameter → ℝ)
    (noiseScale : ℝ) :
    0 ≤ semiparametricBoundaryVariance
      synthesisLoading residualLoading noiseScale := by
  unfold semiparametricBoundaryVariance
  exact add_nonneg
    (add_nonneg
      (synthesisParameterVariance_nonnegative synthesisLoading)
      (synthesisParameterVariance_nonnegative residualLoading))
    (sq_nonneg noiseScale)

noncomputable def semiparametricBoundaryUpper
    {SynthesisParameter ResidualParameter : Type*}
    [Fintype SynthesisParameter] [Fintype ResidualParameter]
    (synthesis residual : ℝ)
    (synthesisLoading : SynthesisParameter → ℝ)
    (residualLoading : ResidualParameter → ℝ)
    (noiseScale quantile : ℝ) : ℝ :=
  semiparametricBoundaryMean synthesis residual
    + quantile * Real.sqrt (
      semiparametricBoundaryVariance
        synthesisLoading residualLoading noiseScale)

theorem semiparametricBoundaryUpper_ge_mean
    {SynthesisParameter ResidualParameter : Type*}
    [Fintype SynthesisParameter] [Fintype ResidualParameter]
    (synthesis residual : ℝ)
    (synthesisLoading : SynthesisParameter → ℝ)
    (residualLoading : ResidualParameter → ℝ)
    (noiseScale quantile : ℝ)
    (hQuantile : 0 ≤ quantile) :
    semiparametricBoundaryMean synthesis residual ≤
      semiparametricBoundaryUpper
        synthesis residual synthesisLoading residualLoading
        noiseScale quantile := by
  unfold semiparametricBoundaryUpper
  exact le_add_of_nonneg_right
    (mul_nonneg hQuantile (Real.sqrt_nonneg _))

theorem semiparametric_certified_recommendation_is_safe
    {SynthesisParameter ResidualParameter : Type*}
    [Fintype SynthesisParameter] [Fintype ResidualParameter]
    (trueMargin synthesis residual : ℝ)
    (synthesisLoading : SynthesisParameter → ℝ)
    (residualLoading : ResidualParameter → ℝ)
    (noiseScale quantile : ℝ)
    (hCoverage :
      trueMargin ≤ semiparametricBoundaryUpper
        synthesis residual synthesisLoading residualLoading
        noiseScale quantile)
    (hCertified :
      semiparametricBoundaryUpper
        synthesis residual synthesisLoading residualLoading
        noiseScale quantile ≤ 0) :
    trueMargin ≤ 0 := by
  exact hCoverage.trans hCertified

theorem frozen_nullspace_residual_is_orthogonal
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
  exact coefficientNullspace_orthogonal_all
    orderedFeature kernel projection hNull

end SCOLHKG.Real
