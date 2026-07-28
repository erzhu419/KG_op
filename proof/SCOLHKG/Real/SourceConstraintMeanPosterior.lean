import Mathlib
import SCOLHKG.Real.CertificationImplementation

namespace SCOLHKG.Real

open scoped BigOperators

/-!
Implementation bridge for the source-informed constraint-mean posterior.

The frozen observable coordinate `eta` is learned from source archives.  In
any coefficient direction, source within-domain uncertainty, between-domain
coefficient disagreement, and a positive covariance floor define the held-out
prior variance.  Charged target observations then perform the same conjugate
rank-one update as `ParametricGPR.update`.
-/

noncomputable def hierarchicalSourceDirectionalVariance {k : ℕ}
    (weight within disagreement : Fin k → ℝ)
    (floor : ℝ) : ℝ :=
  (∑ i, weight i * (within i + (disagreement i) ^ 2)) + floor

theorem hierarchicalSourceDirectionalVariance_nonnegative {k : ℕ}
    (weight within disagreement : Fin k → ℝ)
    (floor : ℝ)
    (hWeight : ∀ i, 0 ≤ weight i)
    (hWithin : ∀ i, 0 ≤ within i)
    (hFloor : 0 ≤ floor) :
    0 ≤ hierarchicalSourceDirectionalVariance
      weight within disagreement floor := by
  unfold hierarchicalSourceDirectionalVariance
  apply add_nonneg
  · exact Finset.sum_nonneg fun i _ =>
      mul_nonneg (hWeight i) (add_nonneg (hWithin i) (sq_nonneg _))
  · exact hFloor

noncomputable def sourceTargetPosteriorDirectionalVariance
    (priorVariance observationVariance : ℝ) : ℝ :=
  priorVariance * observationVariance /
    (priorVariance + observationVariance)

theorem sourceTargetPosteriorDirectionalVariance_nonnegative
    {priorVariance observationVariance : ℝ}
    (hPrior : 0 ≤ priorVariance)
    (hObservation : 0 < observationVariance) :
    0 ≤ sourceTargetPosteriorDirectionalVariance
      priorVariance observationVariance := by
  unfold sourceTargetPosteriorDirectionalVariance
  exact div_nonneg
    (mul_nonneg hPrior (le_of_lt hObservation))
    (le_of_lt (add_pos_of_nonneg_of_pos hPrior hObservation))

theorem target_conditioning_shrinks_source_directional_variance
    {priorVariance observationVariance : ℝ}
    (hPrior : 0 ≤ priorVariance)
    (hObservation : 0 < observationVariance) :
    sourceTargetPosteriorDirectionalVariance
        priorVariance observationVariance
      ≤ priorVariance := by
  have hDenominator : 0 < priorVariance + observationVariance :=
    add_pos_of_nonneg_of_pos hPrior hObservation
  unfold sourceTargetPosteriorDirectionalVariance
  rw [div_le_iff₀ hDenominator]
  nlinarith

noncomputable def sourceTargetPosteriorMean
    (sourceMean targetObservation priorVariance observationVariance : ℝ) : ℝ :=
  sourceMean
    + priorVariance / (priorVariance + observationVariance)
      * (targetObservation - sourceMean)

theorem sourceTargetPosteriorMean_eq_precision_weighted
    {sourceMean targetObservation priorVariance observationVariance : ℝ}
    (hDenominator : priorVariance + observationVariance ≠ 0) :
    sourceTargetPosteriorMean
        sourceMean targetObservation priorVariance observationVariance
      = (observationVariance * sourceMean
          + priorVariance * targetObservation)
          / (priorVariance + observationVariance) := by
  unfold sourceTargetPosteriorMean
  field_simp [hDenominator]
  ring

theorem zero_prior_variance_freezes_source_mean
    (sourceMean targetObservation observationVariance : ℝ) :
    sourceTargetPosteriorMean
        sourceMean targetObservation 0 observationVariance
      = sourceMean := by
  simp [sourceTargetPosteriorMean]

noncomputable def finiteMixtureDirectionalMean {k : ℕ}
    (weight mean : Fin k → ℝ) : ℝ :=
  ∑ i, weight i * mean i

noncomputable def finiteMixtureDirectionalVariance {k : ℕ}
    (weight mean withinVariance : Fin k → ℝ) : ℝ :=
  let mixtureMean := finiteMixtureDirectionalMean weight mean
  ∑ i, weight i *
    (withinVariance i + (mean i - mixtureMean) ^ 2)

theorem finiteMixtureDirectionalVariance_nonnegative {k : ℕ}
    (weight mean withinVariance : Fin k → ℝ)
    (hWeight : ∀ i, 0 ≤ weight i)
    (hWithin : ∀ i, 0 ≤ withinVariance i) :
    0 ≤ finiteMixtureDirectionalVariance
      weight mean withinVariance := by
  unfold finiteMixtureDirectionalVariance
  exact Finset.sum_nonneg fun i _ =>
    mul_nonneg (hWeight i) (add_nonneg (hWithin i) (sq_nonneg _))

theorem mixture_disagreement_cannot_reduce_directional_variance {k : ℕ}
    (weight mean withinVariance : Fin k → ℝ)
    (hWeight : ∀ i, 0 ≤ weight i) :
    (∑ i, weight i * withinVariance i)
      ≤ finiteMixtureDirectionalVariance weight mean withinVariance := by
  unfold finiteMixtureDirectionalVariance
  apply Finset.sum_le_sum
  intro i hi
  have hSquare : 0 ≤ (mean i - finiteMixtureDirectionalMean weight mean) ^ 2 :=
    sq_nonneg _
  nlinarith [hWeight i]

noncomputable def sequentialMixtureEvidenceMass {k : ℕ}
    (weight likelihood : Fin k → ℝ) : ℝ :=
  ∑ i, weight i * likelihood i

noncomputable def sequentialMixtureWeight {k : ℕ}
    (weight likelihood : Fin k → ℝ) (i : Fin k) : ℝ :=
  weight i * likelihood i /
    sequentialMixtureEvidenceMass weight likelihood

theorem sequentialMixtureWeight_nonnegative {k : ℕ}
    (weight likelihood : Fin k → ℝ)
    (hWeight : ∀ i, 0 ≤ weight i)
    (hLikelihood : ∀ i, 0 ≤ likelihood i)
    (hMass : 0 < sequentialMixtureEvidenceMass weight likelihood) :
    ∀ i, 0 ≤ sequentialMixtureWeight weight likelihood i := by
  intro i
  unfold sequentialMixtureWeight
  exact div_nonneg
    (mul_nonneg (hWeight i) (hLikelihood i))
    (le_of_lt hMass)

theorem sequentialMixtureWeight_sum_one {k : ℕ}
    (weight likelihood : Fin k → ℝ)
    (hMass : sequentialMixtureEvidenceMass weight likelihood ≠ 0) :
    ∑ i, sequentialMixtureWeight weight likelihood i = 1 := by
  unfold sequentialMixtureWeight sequentialMixtureEvidenceMass
  rw [← Finset.sum_div]
  exact div_self hMass

theorem source_posterior_certificate_sound
    {trueMean posteriorMean epistemicVariance trueSigma certVariance
      beta z tau : ℝ}
    (hz : 0 ≤ z)
    (hMean :
      trueMean ≤ posteriorMean
        + implementationEpistemicSlack beta epistemicVariance)
    (hSigma :
      trueSigma ≤ implementationCertSigma certVariance)
    (hMargin :
      theoryCertificationMargin
          posteriorMean beta epistemicVariance z certVariance tau
        ≤ 0) :
    trueMean + z * trueSigma ≤ tau := by
  exact implementation_certifies_true_quantile hz hMean hSigma hMargin

end SCOLHKG.Real
