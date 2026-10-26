import SCOLHKG.Real.GPRUpdate
import SCOLHKG.Real.TaskPosterior

namespace SCOLHKG.Real.CurrentPropagation

open scoped BigOperators

/-!
Finite-pool moment algebra for the current raw-response working model.

`loading` represents the local solve `C[:, o] S⁻¹`, and `deltaMean` and
`deltaCovariance` represent changes of the local moments. These definitions
formalize the completion performed after that solve; they do not certify the
local numerical integration, Gaussian regression identity, or multi-round
Bayesian distribution. In particular, there is no unknown-noise covariance
contraction assumption. Component identity/node weights are updated explicitly
from their predictive evidence.
-/

def targetContrast {Pool : Type*} (mean : Pool → ℝ) (u v : Pool) : ℝ :=
  mean u - mean v

def targetContrastVariance {Pool : Type*}
    (covariance : Pool → Pool → ℝ) (u v : Pool) : ℝ :=
  covariance u u + covariance v v - covariance u v - covariance v u

def localCompletedMean {Pool Local : Type*} [Fintype Local]
    (mean : Pool → ℝ) (loading : Pool → Local → ℝ)
    (deltaMean : Local → ℝ) (u : Pool) : ℝ :=
  mean u + ∑ p, loading u p * deltaMean p

def localCompletedCovariance {Pool Local : Type*} [Fintype Local]
    (covariance : Pool → Pool → ℝ) (loading : Pool → Local → ℝ)
    (deltaCovariance : Local → Local → ℝ) (u v : Pool) : ℝ :=
  covariance u v + ∑ p, ∑ q,
    loading u p * deltaCovariance p q * loading v q

/-- The load relevant to distinguishing two policies is the difference of
their local regression rows, rather than either row separately. -/
theorem local_completion_contrast_mean
    {Pool Local : Type*} [Fintype Local]
    (mean : Pool → ℝ) (loading : Pool → Local → ℝ)
    (deltaMean : Local → ℝ) (u v : Pool) :
    targetContrast (localCompletedMean mean loading deltaMean) u v -
        targetContrast mean u v =
      ∑ p, (loading u p - loading v p) * deltaMean p := by
  simp only [targetContrast, localCompletedMean, sub_mul,
    Finset.sum_sub_distrib]
  ring

theorem local_completion_contrast_variance
    {Pool Local : Type*} [Fintype Local]
    (covariance : Pool → Pool → ℝ) (loading : Pool → Local → ℝ)
    (deltaCovariance : Local → Local → ℝ) (u v : Pool) :
    targetContrastVariance
        (localCompletedCovariance covariance loading deltaCovariance) u v -
        targetContrastVariance covariance u v =
      ∑ p, ∑ q, (loading u p - loading v p) * deltaCovariance p q *
        (loading u q - loading v q) := by
  simp only [targetContrastVariance, localCompletedCovariance,
    sub_mul, mul_sub, Finset.sum_sub_distrib]
  ring

theorem equal_local_load_preserves_contrast_mean
    {Pool Local : Type*} [Fintype Local]
    (mean : Pool → ℝ) (loading : Pool → Local → ℝ)
    (deltaMean : Local → ℝ) (u v : Pool)
    (hLoad : ∀ p, loading u p = loading v p) :
    targetContrast (localCompletedMean mean loading deltaMean) u v =
      targetContrast mean u v := by
  have h := local_completion_contrast_mean mean loading deltaMean u v
  simp only [hLoad, sub_self, zero_mul, Finset.sum_const_zero] at h
  exact sub_eq_zero.mp h

theorem equal_local_load_preserves_contrast_variance
    {Pool Local : Type*} [Fintype Local]
    (covariance : Pool → Pool → ℝ) (loading : Pool → Local → ℝ)
    (deltaCovariance : Local → Local → ℝ) (u v : Pool)
    (hLoad : ∀ p, loading u p = loading v p) :
    targetContrastVariance
        (localCompletedCovariance covariance loading deltaCovariance) u v =
      targetContrastVariance covariance u v := by
  have h := local_completion_contrast_variance covariance loading deltaCovariance u v
  simp only [hLoad, sub_self, zero_mul, mul_zero,
    Finset.sum_const_zero] at h
  exact sub_eq_zero.mp h

noncomputable def evidencePosterior
    {Component : Type*} [Fintype Component]
    (weight evidence : Component → ℝ) : Component → ℝ :=
  normalizeFiniteWeights (fun j => weight j * evidence j)

theorem evidencePosterior_normalized
    {Component : Type*} [Fintype Component]
    (weight evidence : Component → ℝ)
    (hEvidence : (∑ j, weight j * evidence j) ≠ 0) :
    ∑ j, evidencePosterior weight evidence j = 1 := by
  exact normalizeFiniteWeights_sum_eq_one hEvidence

theorem evidencePosterior_nonnegative
    {Component : Type*} [Fintype Component]
    (weight evidence : Component → ℝ)
    (hWeight : ∀ j, 0 ≤ weight j)
    (hEvidence : ∀ j, 0 ≤ evidence j)
    (hTotal : 0 < ∑ j, weight j * evidence j) (j : Component) :
    0 ≤ evidencePosterior weight evidence j := by
  exact normalizeFiniteWeights_nonneg
    (fun i => mul_nonneg (hWeight i) (hEvidence i)) hTotal j

/-- This finite identity separates a within-component change from a change
in the component weights. Neither term is assumed to vanish. -/
theorem finite_mixture_reweighting_decomposition
    {Component : Type*} [Fintype Component]
    (weight updatedWeight contrast updatedContrast : Component → ℝ) :
    finiteMean updatedWeight updatedContrast - finiteMean weight contrast =
      (∑ j, updatedWeight j * (updatedContrast j - contrast j)) +
        ∑ j, (updatedWeight j - weight j) * contrast j := by
  simp only [finiteMean, mul_sub, sub_mul, Finset.sum_sub_distrib]
  ring

/-- The current component completion and evidence update jointly yield the
two propagation channels for an unobserved pair's objective contrast. -/
theorem evidence_local_completion_contrast_decomposition
    {Pool Local Component : Type*} [Fintype Local] [Fintype Component]
    (weight evidence : Component → ℝ)
    (mean : Component → Pool → ℝ)
    (loading : Component → Pool → Local → ℝ)
    (deltaMean : Component → Local → ℝ) (u v : Pool) :
    finiteMean (evidencePosterior weight evidence)
        (fun j => targetContrast
          (localCompletedMean (mean j) (loading j) (deltaMean j)) u v) -
        finiteMean weight (fun j => targetContrast (mean j) u v) =
      (∑ j, evidencePosterior weight evidence j *
        (∑ p, (loading j u p - loading j v p) * deltaMean j p)) +
        ∑ j, (evidencePosterior weight evidence j - weight j) *
          targetContrast (mean j) u v := by
  rw [finite_mixture_reweighting_decomposition]
  congr 1
  apply Finset.sum_congr rfl
  intro j _hj
  rw [local_completion_contrast_mean]

/-!
Known-noise single-Gaussian algebra. The input `RankOneGPRState` is the existing
rank-one interface. A positive predictive variance is an explicit hypothesis
where needed; the square-over-variance decrement is not asserted for the
unknown-noise mixture above.
-/

noncomputable def rankOneCompletedCovariance
    {Pool : Type*} (s : RankOneGPRState Pool) (sample : Pool)
    (u v : Pool) : ℝ :=
  s.covariance u v -
    s.covariance u sample * s.covariance v sample / predictiveVariance s sample

theorem rank_one_contrast_mean_change
    {Pool : Type*} (s : RankOneGPRState Pool) (sample u v : Pool) (y : ℝ) :
    targetContrast (updatedPosteriorMean s sample y) u v -
        targetContrast s.mean u v =
      (s.covariance u sample - s.covariance v sample) /
        predictiveVariance s sample * (y - s.mean sample) := by
  simp only [targetContrast, updatedPosteriorMean, kalmanGain, div_eq_mul_inv]
  ring

theorem rank_one_contrast_variance_decrement
    {Pool : Type*} (s : RankOneGPRState Pool) (sample u v : Pool) :
    targetContrastVariance s.covariance u v -
        targetContrastVariance (rankOneCompletedCovariance s sample) u v =
      (s.covariance u sample - s.covariance v sample) ^ 2 /
        predictiveVariance s sample := by
  simp only [targetContrastVariance, rankOneCompletedCovariance, div_eq_mul_inv]
  ring

theorem rank_one_contrast_variance_nonincreasing
    {Pool : Type*} (s : RankOneGPRState Pool) (sample u v : Pool)
    (hPredictive : 0 < predictiveVariance s sample) :
    targetContrastVariance (rankOneCompletedCovariance s sample) u v ≤
      targetContrastVariance s.covariance u v := by
  have hDecrement := rank_one_contrast_variance_decrement s sample u v
  have hNonnegative : 0 ≤
      (s.covariance u sample - s.covariance v sample) ^ 2 /
        predictiveVariance s sample :=
    div_nonneg (sq_nonneg _) (le_of_lt hPredictive)
  linarith

theorem equal_kernel_relation_preserves_contrast_mean
    {Pool : Type*} (s : RankOneGPRState Pool) (sample u v : Pool) (y : ℝ)
    (hRelation : s.covariance u sample = s.covariance v sample) :
    targetContrast (updatedPosteriorMean s sample y) u v =
      targetContrast s.mean u v := by
  have h := rank_one_contrast_mean_change s sample u v y
  rw [hRelation, sub_self, zero_div, zero_mul] at h
  exact sub_eq_zero.mp h

theorem equal_kernel_relation_preserves_contrast_variance
    {Pool : Type*} (s : RankOneGPRState Pool) (sample u v : Pool)
    (hRelation : s.covariance u sample = s.covariance v sample) :
    targetContrastVariance (rankOneCompletedCovariance s sample) u v =
      targetContrastVariance s.covariance u v := by
  have h := rank_one_contrast_variance_decrement s sample u v
  rw [hRelation, sub_self, zero_pow (by decide : 2 ≠ 0), zero_div] at h
  exact (sub_eq_zero.mp h).symm

end SCOLHKG.Real.CurrentPropagation
