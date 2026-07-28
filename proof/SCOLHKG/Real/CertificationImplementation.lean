import Mathlib
import SCOLHKG.Real.Certification

namespace SCOLHKG.Real

/-!
Implementation bridge for `core/certification.py`.

The Python implementation uses

`mu_g + sqrt(beta_g) * sqrt(epistemic_var_g) + z_alpha * sqrt(v_C_plus) <= tau`.

This file proves that the code-level margin is exactly the arithmetic
certificate consumed by `Certification.lean`, and that theory mode is never
less conservative than the legacy aleatoric-only bound.
-/

noncomputable def theoryCertificationMargin
    (mu beta epistemicVar z vC tau : ℝ) : ℝ :=
  mu + Real.sqrt beta * Real.sqrt epistemicVar + z * Real.sqrt vC - tau

noncomputable def legacyCertificationMargin
    (mu z vC tau : ℝ) : ℝ :=
  mu + z * Real.sqrt vC - tau

noncomputable def implementationEpistemicSlack
    (beta epistemicVar : ℝ) : ℝ :=
  Real.sqrt beta * Real.sqrt epistemicVar

noncomputable def implementationCertSigma (vC : ℝ) : ℝ :=
  Real.sqrt vC

theorem theoryCertificationMargin_le_zero_iff_bound
    {mu beta epistemicVar z vC tau : ℝ} :
    theoryCertificationMargin mu beta epistemicVar z vC tau ≤ 0
      ↔ mu + implementationEpistemicSlack beta epistemicVar
          + z * implementationCertSigma vC ≤ tau := by
  unfold theoryCertificationMargin implementationEpistemicSlack implementationCertSigma
  constructor <;> intro h <;> linarith

theorem theory_margin_certificate_matches_Certified
    {mu beta epistemicVar z vC tau : ℝ}
    (h :
      theoryCertificationMargin mu beta epistemicVar z vC tau ≤ 0) :
    Certified
      mu
      (implementationEpistemicSlack beta epistemicVar)
      z
      (implementationCertSigma vC)
      tau := by
  unfold Certified chanceUpper
  exact (theoryCertificationMargin_le_zero_iff_bound.mp h)

theorem legacy_margin_le_theory_margin
    (mu beta epistemicVar z vC tau : ℝ) :
    legacyCertificationMargin mu z vC tau
      ≤ theoryCertificationMargin mu beta epistemicVar z vC tau := by
  unfold legacyCertificationMargin theoryCertificationMargin
  have hterm :
      0 ≤ Real.sqrt beta * Real.sqrt epistemicVar := by
    exact mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  linarith

theorem theory_margin_eq_legacy_when_beta_zero
    (mu epistemicVar z vC tau : ℝ) :
    theoryCertificationMargin mu 0 epistemicVar z vC tau
      = legacyCertificationMargin mu z vC tau := by
  unfold theoryCertificationMargin legacyCertificationMargin
  simp

theorem implementation_certifies_true_quantile
    {trueMean mu beta epistemicVar z trueSigma vC tau : ℝ}
    (hz : 0 ≤ z)
    (hMean :
      trueMean ≤
        mu + implementationEpistemicSlack beta epistemicVar)
    (hSigma : trueSigma ≤ implementationCertSigma vC)
    (hMargin :
      theoryCertificationMargin mu beta epistemicVar z vC tau ≤ 0) :
    trueMean + z * trueSigma ≤ tau := by
  exact certified_implies_true_quantile_bound
    (trueMean := trueMean)
    (postMean := mu)
    (epistemicSlack := implementationEpistemicSlack beta epistemicVar)
    (z := z)
    (trueSigma := trueSigma)
    (certSigma := implementationCertSigma vC)
    (tau := tau)
    hz
    hMean
    hSigma
    (theory_margin_certificate_matches_Certified hMargin)

theorem oracle_mean_certifiable_requires_epistemic_radius_le_safety_depth
    {trueChanceMargin beta epistemicVar : ℝ}
    (hCertificate :
      trueChanceMargin
          + implementationEpistemicSlack beta epistemicVar ≤ 0) :
    implementationEpistemicSlack beta epistemicVar
      ≤ -trueChanceMargin := by
  linarith

theorem oracle_mean_certificate_vacuous_of_shallow_safety
    {trueChanceMargin beta epistemicVar : ℝ}
    (hShallow :
      -trueChanceMargin
        < implementationEpistemicSlack beta epistemicVar) :
    0 < trueChanceMargin
        + implementationEpistemicSlack beta epistemicVar := by
  linarith

theorem oracle_mean_certifiable_requires_epistemic_variance_threshold
    {trueChanceMargin beta epistemicVar : ℝ}
    (hBeta : 0 ≤ beta)
    (hEpistemic : 0 ≤ epistemicVar)
    (hCertificate :
      trueChanceMargin
          + implementationEpistemicSlack beta epistemicVar ≤ 0) :
    beta * epistemicVar ≤ (-trueChanceMargin) ^ 2 := by
  have hRadiusNonnegative :
      0 ≤ implementationEpistemicSlack beta epistemicVar := by
    exact mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  have hDepthNonnegative : 0 ≤ -trueChanceMargin := by
    have hRadiusDepth :=
      oracle_mean_certifiable_requires_epistemic_radius_le_safety_depth
        hCertificate
    linarith
  have hRadiusDepth :
      implementationEpistemicSlack beta epistemicVar
        ≤ -trueChanceMargin :=
    oracle_mean_certifiable_requires_epistemic_radius_le_safety_depth
      hCertificate
  have hSquared :
      (implementationEpistemicSlack beta epistemicVar) ^ 2
        ≤ (-trueChanceMargin) ^ 2 := by
    nlinarith
  have hRadiusSquare :
      (implementationEpistemicSlack beta epistemicVar) ^ 2
        = beta * epistemicVar := by
    unfold implementationEpistemicSlack
    rw [mul_pow, Real.sq_sqrt hBeta, Real.sq_sqrt hEpistemic]
  linarith

end SCOLHKG.Real
