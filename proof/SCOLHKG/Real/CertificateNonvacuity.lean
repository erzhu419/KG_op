import Mathlib
import SCOLHKG.Real.CertificationImplementation

namespace SCOLHKG.Real

/-!
Finite-budget nonvacuity of the implementation certificate.

Soundness and nonvacuity are different statements.  Soundness says that an
issued certificate is correct.  Nonvacuity additionally needs a policy with
positive true safety depth and enough target evidence for both the constraint
mean and the cumulative-HVD variance upper bound.  This file makes that budget
dependence explicit.

The HVD estimator controls variance.  Without an additional lower-variance
regularity assumption, converting an `O(r^{-1/2})` variance error into a
standard-deviation error costs a square root.  The resulting conservative
replication threshold scales as `depth^{-4}`.  Recording this rate is more
honest than silently treating variance and standard deviation errors as the
same quantity.
-/

noncomputable section

def trueSafetyDepth
    (trueMean trueVariance z tau : ℝ) : ℝ :=
  tau - (trueMean + z * Real.sqrt trueVariance)

def certificateMeanRate (constant samples : ℝ) : ℝ :=
  constant / Real.sqrt samples

def certificateVarianceRate (constant replicates : ℝ) : ℝ :=
  constant / Real.sqrt replicates

def FiniteCertificateBudget
    (depth meanConstant varianceConstant z meanSamples replicates : ℝ) : Prop :=
  4 * meanConstant ^ 2 ≤ depth ^ 2 * meanSamples
    ∧ 16 * z ^ 4 * varianceConstant ^ 2 ≤ depth ^ 4 * replicates

theorem sqrt_upper_of_variance_upper
    {trueVariance certVariance varianceGap : ℝ}
    (hTrue : 0 ≤ trueVariance)
    (hGap : 0 ≤ varianceGap)
    (hUpper : certVariance ≤ trueVariance + varianceGap) :
    Real.sqrt certVariance ≤
      Real.sqrt trueVariance + Real.sqrt varianceGap := by
  apply (Real.sqrt_le_iff).2
  constructor
  · exact add_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  · have hTrueSqrt := Real.sq_sqrt hTrue
    have hGapSqrt := Real.sq_sqrt hGap
    have hCross :
        0 ≤ 2 * Real.sqrt trueVariance * Real.sqrt varianceGap := by
      positivity
    nlinarith

theorem theory_certificate_nonvacuous_of_depth
    {trueMean trueVariance postMean beta epistemicVar z certVariance tau : ℝ}
    {meanRadius varianceGap depth : ℝ}
    (hTrueVariance : 0 ≤ trueVariance)
    (hVarianceGap : 0 ≤ varianceGap)
    (hZ : 0 ≤ z)
    (hDepth :
      depth = trueSafetyDepth trueMean trueVariance z tau)
    (hMeanUpper :
      postMean + implementationEpistemicSlack beta epistemicVar
        ≤ trueMean + meanRadius)
    (hVarianceUpper :
      certVariance ≤ trueVariance + varianceGap)
    (hErrorBudget :
      meanRadius + z * Real.sqrt varianceGap ≤ depth) :
    theoryCertificationMargin
      postMean beta epistemicVar z certVariance tau ≤ 0 := by
  have hSigmaUpper := sqrt_upper_of_variance_upper
    hTrueVariance hVarianceGap hVarianceUpper
  rw [theoryCertificationMargin_le_zero_iff_bound]
  unfold implementationCertSigma
  unfold trueSafetyDepth at hDepth
  nlinarith

theorem certificate_mean_rate_le_half_depth
    {depth meanConstant meanSamples : ℝ}
    (hDepth : 0 < depth)
    (hConstant : 0 ≤ meanConstant)
    (hSamples : 0 < meanSamples)
    (hBudget :
      4 * meanConstant ^ 2 ≤ depth ^ 2 * meanSamples) :
    certificateMeanRate meanConstant meanSamples ≤ depth / 2 := by
  unfold certificateMeanRate
  have hSqrt : 0 < Real.sqrt meanSamples := Real.sqrt_pos.2 hSamples
  have hProduct :
      2 * meanConstant ≤ depth * Real.sqrt meanSamples := by
    have hLeft : 0 ≤ 2 * meanConstant := by positivity
    have hRight : 0 ≤ depth * Real.sqrt meanSamples := by positivity
    have hRightSquare :
        (depth * Real.sqrt meanSamples) ^ 2 =
          depth ^ 2 * meanSamples := by
      rw [mul_pow, Real.sq_sqrt hSamples.le]
    nlinarith
  exact (div_le_iff₀ hSqrt).2 (by nlinarith)

theorem certificate_variance_rate_sigma_le_half_depth
    {depth varianceConstant z replicates : ℝ}
    (hDepth : 0 < depth)
    (hConstant : 0 ≤ varianceConstant)
    (hZ : 0 ≤ z)
    (hReplicates : 0 < replicates)
    (hBudget :
      16 * z ^ 4 * varianceConstant ^ 2
        ≤ depth ^ 4 * replicates) :
    z * Real.sqrt (certificateVarianceRate varianceConstant replicates)
      ≤ depth / 2 := by
  unfold certificateVarianceRate
  have hSqrtRep : 0 < Real.sqrt replicates := Real.sqrt_pos.2 hReplicates
  have hRateNonnegative :
      0 ≤ varianceConstant / Real.sqrt replicates :=
    div_nonneg hConstant hSqrtRep.le
  have hIntermediate :
      4 * z ^ 2 * varianceConstant
        ≤ depth ^ 2 * Real.sqrt replicates := by
    have hLeft : 0 ≤ 4 * z ^ 2 * varianceConstant := by positivity
    have hRight : 0 ≤ depth ^ 2 * Real.sqrt replicates := by positivity
    have hLeftSquare :
        (4 * z ^ 2 * varianceConstant) ^ 2 =
          16 * z ^ 4 * varianceConstant ^ 2 := by ring
    have hRightSquare :
        (depth ^ 2 * Real.sqrt replicates) ^ 2 =
          depth ^ 4 * replicates := by
      rw [mul_pow, Real.sq_sqrt hReplicates.le]
      ring
    nlinarith
  have hSquared :
      (2 * z * Real.sqrt
        (varianceConstant / Real.sqrt replicates)) ^ 2
        ≤ depth ^ 2 := by
    have hDiv :
        4 * z ^ 2 * varianceConstant / Real.sqrt replicates
          ≤ depth ^ 2 := by
      exact (div_le_iff₀ hSqrtRep).2 (by
        simpa [mul_assoc] using hIntermediate)
    calc
      (2 * z * Real.sqrt
          (varianceConstant / Real.sqrt replicates)) ^ 2
        = 4 * z ^ 2 * varianceConstant / Real.sqrt replicates := by
            rw [mul_pow, mul_pow, Real.sq_sqrt hRateNonnegative]
            ring
      _ ≤ depth ^ 2 := hDiv
  have hLeftNonnegative :
      0 ≤ 2 * z * Real.sqrt
        (varianceConstant / Real.sqrt replicates) := by positivity
  nlinarith

theorem finite_budget_certificate_nonvacuous
    {trueMean trueVariance postMean beta epistemicVar z certVariance tau : ℝ}
    {depth meanConstant varianceConstant meanSamples replicates : ℝ}
    (hTrueVariance : 0 ≤ trueVariance)
    (hZ : 0 ≤ z)
    (hDepthPositive : 0 < depth)
    (hMeanConstant : 0 ≤ meanConstant)
    (hVarianceConstant : 0 ≤ varianceConstant)
    (hMeanSamples : 0 < meanSamples)
    (hReplicates : 0 < replicates)
    (hDepth : depth =
      trueSafetyDepth trueMean trueVariance z tau)
    (hMeanUpper :
      postMean + implementationEpistemicSlack beta epistemicVar
        ≤ trueMean + certificateMeanRate meanConstant meanSamples)
    (hVarianceUpper :
      certVariance ≤ trueVariance
        + certificateVarianceRate varianceConstant replicates)
    (hBudget : FiniteCertificateBudget
      depth meanConstant varianceConstant z meanSamples replicates) :
    theoryCertificationMargin
      postMean beta epistemicVar z certVariance tau ≤ 0 := by
  have hMeanHalf := certificate_mean_rate_le_half_depth
    hDepthPositive hMeanConstant hMeanSamples hBudget.1
  have hVarianceHalf := certificate_variance_rate_sigma_le_half_depth
    hDepthPositive hVarianceConstant hZ hReplicates hBudget.2
  have hVarianceRateNonnegative :
      0 ≤ certificateVarianceRate varianceConstant replicates := by
    unfold certificateVarianceRate
    exact div_nonneg hVarianceConstant (Real.sqrt_nonneg _)
  apply theory_certificate_nonvacuous_of_depth
    hTrueVariance hVarianceRateNonnegative hZ hDepth
    hMeanUpper hVarianceUpper
  linarith

def TheoryCertifiedSet {Design : Type*}
    (postMean epistemicVar certVariance : Design → ℝ)
    (beta z tau : ℝ) : Set Design :=
  {x | theoryCertificationMargin
    (postMean x) beta (epistemicVar x) z (certVariance x) tau ≤ 0}

theorem certified_set_nonempty_of_finite_budget_point
    {Design : Type*}
    {postMean epistemicVar certVariance : Design → ℝ}
    {beta z tau : ℝ}
    {x : Design}
    (hPoint : theoryCertificationMargin
      (postMean x) beta (epistemicVar x) z (certVariance x) tau ≤ 0) :
    (TheoryCertifiedSet postMean epistemicVar certVariance beta z tau).Nonempty := by
  exact ⟨x, hPoint⟩

end

end SCOLHKG.Real
