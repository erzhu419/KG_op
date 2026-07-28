import Mathlib
import SCOLHKG.Real.TaskPosterior

namespace SCOLHKG.Real

open scoped BigOperators

/-!
Source-to-target generalization for the finite task posterior.

The source archive cannot imply target performance without a domain-shift
assumption.  This theorem keeps that unavoidable quantity explicit.  A
PAC-Bayes moment event controls the posterior-weighted source generalization
gap, while `domainShift` records the remaining source-to-held-out-target
discrepancy expert by expert.
-/

noncomputable section

def sourceTargetGap
    {Expert : Type*}
    (sourceRisk targetRisk domainShift : Expert → ℝ)
    (expert : Expert) : ℝ :=
  targetRisk expert - sourceRisk expert - domainShift expert

def transferGeneralizationRadius
    (rho delta sourceSamples : ℝ) : ℝ :=
  (rho + Real.log (1 / delta)) / sourceSamples

theorem finite_source_to_target_pac_bayes
    {Expert : Type*} [Fintype Expert] [Nonempty Expert]
    {posterior prior : Expert → ℝ}
    {sourceRisk targetRisk domainShift : Expert → ℝ}
    {rho delta sourceSamples : ℝ}
    (hPosterior : ∀ i, 0 < posterior i)
    (hPrior : ∀ i, 0 < prior i)
    (hPosteriorNorm : ∑ i, posterior i = 1)
    (hPriorNorm : ∑ i, prior i = 1)
    (hKL : finiteTaskKL posterior prior ≤ rho)
    (hDelta : 0 < delta)
    (hSamples : 0 < sourceSamples)
    (hMoment :
      (∑ i, prior i * Real.exp
        (sourceSamples * sourceTargetGap
          sourceRisk targetRisk domainShift i)) ≤ 1 / delta) :
    (∑ i, posterior i * targetRisk i) ≤
      (∑ i, posterior i * sourceRisk i)
        + (∑ i, posterior i * domainShift i)
        + transferGeneralizationRadius rho delta sourceSamples := by
  have hPAC := finite_pac_bayes_bound_on_moment_event
    (q := posterior)
    (p := prior)
    (gap := sourceTargetGap sourceRisk targetRisk domainShift)
    hPosterior hPrior hPosteriorNorm hPriorNorm hKL hDelta hSamples hMoment
  unfold sourceTargetGap at hPAC
  unfold transferGeneralizationRadius
  have hExpand :
      (∑ i, posterior i *
        (targetRisk i - sourceRisk i - domainShift i)) =
        (∑ i, posterior i * targetRisk i)
          - (∑ i, posterior i * sourceRisk i)
          - (∑ i, posterior i * domainShift i) := by
    calc
      (∑ i, posterior i *
          (targetRisk i - sourceRisk i - domainShift i))
        = ∑ i,
            (posterior i * targetRisk i
              - posterior i * sourceRisk i
              - posterior i * domainShift i) := by
                apply Finset.sum_congr rfl
                intro i _hi
                ring
      _ = (∑ i, posterior i * targetRisk i)
          - (∑ i, posterior i * sourceRisk i)
          - (∑ i, posterior i * domainShift i) := by
            rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib]
  rw [hExpand] at hPAC
  linarith

theorem finite_source_to_target_uniform_shift
    {Expert : Type*} [Fintype Expert] [Nonempty Expert]
    {posterior prior : Expert → ℝ}
    {sourceRisk targetRisk : Expert → ℝ}
    {uniformShift rho delta sourceSamples : ℝ}
    (hPosterior : ∀ i, 0 < posterior i)
    (hPrior : ∀ i, 0 < prior i)
    (hPosteriorNorm : ∑ i, posterior i = 1)
    (hPriorNorm : ∑ i, prior i = 1)
    (hKL : finiteTaskKL posterior prior ≤ rho)
    (hDelta : 0 < delta)
    (hSamples : 0 < sourceSamples)
    (hMoment :
      (∑ i, prior i * Real.exp
        (sourceSamples * sourceTargetGap
          sourceRisk targetRisk (fun _ => uniformShift) i)) ≤ 1 / delta) :
    (∑ i, posterior i * targetRisk i) ≤
      (∑ i, posterior i * sourceRisk i)
        + uniformShift
        + transferGeneralizationRadius rho delta sourceSamples := by
  have hGeneral := finite_source_to_target_pac_bayes
    hPosterior hPrior hPosteriorNorm hPriorNorm hKL hDelta hSamples hMoment
  have hShift :
      (∑ i, posterior i * uniformShift) = uniformShift := by
    rw [← Finset.sum_mul, hPosteriorNorm, one_mul]
  simpa [hShift] using hGeneral

end

end SCOLHKG.Real
