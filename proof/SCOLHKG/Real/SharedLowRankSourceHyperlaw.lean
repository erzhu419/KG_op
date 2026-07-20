import Mathlib

namespace SCOLHKG.Real

/-!
V42's source-to-target coefficient law separates uncertainty in the shared
source mean from genuine target-domain discrepancy.  These directional
identities are the finite-feature contract implemented by
`ExchangeableBoundaryMeanCoordinate`.
-/

noncomputable def weightedSharedEstimationVariance {s : ℕ}
    (weight estimationVariance : Fin s → ℝ) : ℝ :=
  ∑ source, (weight source) ^ 2 * estimationVariance source

noncomputable def legacyTransferredEstimationVariance {s : ℕ}
    (weight estimationVariance : Fin s → ℝ) : ℝ :=
  ∑ source, weight source * estimationVariance source

theorem weighted_shared_estimation_variance_nonnegative {s : ℕ}
    (weight estimationVariance : Fin s → ℝ)
    (hVariance : ∀ source, 0 ≤ estimationVariance source) :
    0 ≤ weightedSharedEstimationVariance weight estimationVariance := by
  unfold weightedSharedEstimationVariance
  exact Finset.sum_nonneg fun source _ =>
    mul_nonneg (sq_nonneg _) (hVariance source)

theorem shared_estimation_variance_le_legacy_transfer {s : ℕ}
    (weight estimationVariance : Fin s → ℝ)
    (hWeightNonnegative : ∀ source, 0 ≤ weight source)
    (hWeightAtMostOne : ∀ source, weight source ≤ 1)
    (hVariance : ∀ source, 0 ≤ estimationVariance source) :
    weightedSharedEstimationVariance weight estimationVariance
      ≤ legacyTransferredEstimationVariance weight estimationVariance := by
  unfold weightedSharedEstimationVariance
    legacyTransferredEstimationVariance
  apply Finset.sum_le_sum
  intro source _
  apply mul_le_mul_of_nonneg_right _ (hVariance source)
  nlinarith [hWeightNonnegative source, hWeightAtMostOne source]

noncomputable def betweenDomainProjectedVariance {s : ℕ}
    (weight projectedCoefficient : Fin s → ℝ)
    (sharedCoefficient : ℝ) : ℝ :=
  ∑ source, weight source
    * (projectedCoefficient source - sharedCoefficient) ^ 2

theorem between_domain_projected_variance_nonnegative {s : ℕ}
    (weight projectedCoefficient : Fin s → ℝ)
    (sharedCoefficient : ℝ)
    (hWeight : ∀ source, 0 ≤ weight source) :
    0 ≤ betweenDomainProjectedVariance
      weight projectedCoefficient sharedCoefficient := by
  unfold betweenDomainProjectedVariance
  exact Finset.sum_nonneg fun source _ =>
    mul_nonneg (hWeight source) (sq_nonneg _)

noncomputable def sourceDomainDiscrepancyProjectedVariance {p s : ℕ}
    (feature : Fin p → ℝ)
    (loading : Fin p → Fin (s - 1) → ℝ) : ℝ :=
  ∑ factor : Fin (s - 1),
    (∑ coefficient, feature coefficient * loading coefficient factor) ^ 2

theorem source_domain_discrepancy_projected_variance_nonnegative {p s : ℕ}
    (feature : Fin p → ℝ)
    (loading : Fin p → Fin (s - 1) → ℝ) :
    0 ≤ sourceDomainDiscrepancyProjectedVariance feature loading := by
  unfold sourceDomainDiscrepancyProjectedVariance
  exact Finset.sum_nonneg fun _ _ => sq_nonneg _

theorem one_source_has_zero_domain_discrepancy {p : ℕ}
    (feature : Fin p → ℝ)
    (loading : Fin p → Fin (1 - 1) → ℝ) :
    sourceDomainDiscrepancyProjectedVariance feature loading = 0 := by
  simp [sourceDomainDiscrepancyProjectedVariance]

noncomputable def sharedLowRankHyperlawProjectedVariance {s : ℕ}
    (weight estimationVariance : Fin s → ℝ)
    (channelRoleVariance domainDiscrepancyVariance : ℝ) : ℝ :=
  weightedSharedEstimationVariance weight estimationVariance
    + channelRoleVariance
    + domainDiscrepancyVariance

theorem shared_low_rank_hyperlaw_projected_variance_nonnegative {s : ℕ}
    (weight estimationVariance : Fin s → ℝ)
    (channelRoleVariance domainDiscrepancyVariance : ℝ)
    (hEstimation : ∀ source, 0 ≤ estimationVariance source)
    (hRole : 0 ≤ channelRoleVariance)
    (hDomain : 0 ≤ domainDiscrepancyVariance) :
    0 ≤ sharedLowRankHyperlawProjectedVariance
      weight estimationVariance channelRoleVariance
      domainDiscrepancyVariance := by
  unfold sharedLowRankHyperlawProjectedVariance
  exact add_nonneg
    (add_nonneg
      (weighted_shared_estimation_variance_nonnegative _ _ hEstimation) hRole)
    hDomain

theorem shared_low_rank_separates_estimation_from_target_variation {s : ℕ}
    (weight estimationVariance : Fin s → ℝ)
    (channelRoleVariance domainDiscrepancyVariance : ℝ) :
    sharedLowRankHyperlawProjectedVariance
        weight estimationVariance channelRoleVariance
        domainDiscrepancyVariance
      - channelRoleVariance
      - domainDiscrepancyVariance
      = weightedSharedEstimationVariance weight estimationVariance := by
  unfold sharedLowRankHyperlawProjectedVariance
  ring

theorem shared_low_rank_no_wider_than_legacy_when_discrepancy_is_fixed
    {s : ℕ}
    (weight estimationVariance : Fin s → ℝ)
    (channelRoleVariance domainDiscrepancyVariance : ℝ)
    (hWeightNonnegative : ∀ source, 0 ≤ weight source)
    (hWeightAtMostOne : ∀ source, weight source ≤ 1)
    (hVariance : ∀ source, 0 ≤ estimationVariance source) :
    sharedLowRankHyperlawProjectedVariance
        weight estimationVariance channelRoleVariance
        domainDiscrepancyVariance
      ≤ legacyTransferredEstimationVariance weight estimationVariance
        + channelRoleVariance
        + domainDiscrepancyVariance := by
  unfold sharedLowRankHyperlawProjectedVariance
  linarith [shared_estimation_variance_le_legacy_transfer
    weight estimationVariance hWeightNonnegative hWeightAtMostOne hVariance]

/-!
For normalized source weights let `c = sum_s w_s^2`.  If source-task
coefficients are exchangeable, the weighted population covariance estimates
`(1-c) Sigma`, while a newly drawn target differs from the weighted source
mean by `(1+c) Sigma`.  V43 therefore multiplies the observed between-source
covariance by `(1+c)/(1-c)`.  The correction changes scale but keeps the same
`Fin (s - 1)` discrepancy factors above.
-/

noncomputable def finiteSourcePredictiveMultiplier (concentration : ℝ) : ℝ :=
  (1 + concentration) / (1 - concentration)

theorem finite_source_predictive_multiplier_nonnegative
    (concentration : ℝ)
    (hNonnegative : 0 ≤ concentration)
    (hNondegenerate : concentration < 1) :
    0 ≤ finiteSourcePredictiveMultiplier concentration := by
  unfold finiteSourcePredictiveMultiplier
  exact div_nonneg (by linarith) (by linarith)

theorem finite_source_predictive_multiplier_at_least_one
    (concentration : ℝ)
    (hNonnegative : 0 ≤ concentration)
    (hNondegenerate : concentration < 1) :
    1 ≤ finiteSourcePredictiveMultiplier concentration := by
  unfold finiteSourcePredictiveMultiplier
  rw [le_div_iff₀ (by linarith : 0 < 1 - concentration)]
  linarith

noncomputable def finiteSourcePredictiveProjectedVariance
    (concentration populationProjectedVariance : ℝ) : ℝ :=
  finiteSourcePredictiveMultiplier concentration
    * populationProjectedVariance

theorem finite_source_predictive_projected_variance_nonnegative
    (concentration populationProjectedVariance : ℝ)
    (hConcentration : 0 ≤ concentration)
    (hNondegenerate : concentration < 1)
    (hPopulation : 0 ≤ populationProjectedVariance) :
    0 ≤ finiteSourcePredictiveProjectedVariance
      concentration populationProjectedVariance := by
  unfold finiteSourcePredictiveProjectedVariance
  exact mul_nonneg
    (finite_source_predictive_multiplier_nonnegative
      concentration hConcentration hNondegenerate)
    hPopulation

theorem finite_source_predictive_dominates_population_discrepancy
    (concentration populationProjectedVariance : ℝ)
    (hConcentration : 0 ≤ concentration)
    (hNondegenerate : concentration < 1)
    (hPopulation : 0 ≤ populationProjectedVariance) :
    populationProjectedVariance ≤
      finiteSourcePredictiveProjectedVariance
        concentration populationProjectedVariance := by
  unfold finiteSourcePredictiveProjectedVariance
  nlinarith [finite_source_predictive_multiplier_at_least_one
    concentration hConcentration hNondegenerate]

theorem finite_source_predictive_preserves_zero_discrepancy
    (concentration : ℝ) :
    finiteSourcePredictiveProjectedVariance concentration 0 = 0 := by
  simp [finiteSourcePredictiveProjectedVariance]

noncomputable def finiteSourcePredictiveLowRankVariance {p s : ℕ}
    (concentration : ℝ)
    (feature : Fin p → ℝ)
    (loading : Fin p → Fin (s - 1) → ℝ) : ℝ :=
  finiteSourcePredictiveMultiplier concentration
    * sourceDomainDiscrepancyProjectedVariance feature loading

theorem finite_source_predictive_low_rank_variance_nonnegative {p s : ℕ}
    (concentration : ℝ)
    (feature : Fin p → ℝ)
    (loading : Fin p → Fin (s - 1) → ℝ)
    (hConcentration : 0 ≤ concentration)
    (hNondegenerate : concentration < 1) :
    0 ≤ finiteSourcePredictiveLowRankVariance
      concentration feature loading := by
  unfold finiteSourcePredictiveLowRankVariance
  exact mul_nonneg
    (finite_source_predictive_multiplier_nonnegative
      concentration hConcentration hNondegenerate)
    (source_domain_discrepancy_projected_variance_nonnegative
      feature loading)

theorem one_source_predictive_domain_discrepancy_is_zero {p : ℕ}
    (concentration : ℝ)
    (feature : Fin p → ℝ)
    (loading : Fin p → Fin (1 - 1) → ℝ) :
    finiteSourcePredictiveLowRankVariance
      concentration feature loading = 0 := by
  simp [finiteSourcePredictiveLowRankVariance,
    sourceDomainDiscrepancyProjectedVariance]

/-!
V45 redistributes a fixed offline record budget over more exchangeable task
episodes.  The simulator budget is unchanged when the per-episode allocation
partitions the old per-base-domain budget.  More task episodes enlarge only
the maximum discrepancy-factor capacity; they do not assert that the observed
coefficient matrix attains that rank.
-/

def sourceSimulatorCallBudget
    (baseDomains recordsPerBase replicates : ℕ) : ℕ :=
  baseDomains * recordsPerBase * replicates

def sourceEpisodeTaskCount (baseDomains episodesPerBase : ℕ) : ℕ :=
  baseDomains * episodesPerBase

def sourceEpisodeDiscrepancyCapacity
    (baseDomains episodesPerBase : ℕ) : ℕ :=
  sourceEpisodeTaskCount baseDomains episodesPerBase - 1

theorem equal_episode_allocation_preserves_source_call_budget
    (baseDomains episodesPerBase recordsPerEpisode replicates : ℕ) :
    sourceEpisodeTaskCount baseDomains episodesPerBase
        * recordsPerEpisode * replicates
      =
    sourceSimulatorCallBudget
      baseDomains (episodesPerBase * recordsPerEpisode) replicates := by
  simp [sourceEpisodeTaskCount, sourceSimulatorCallBudget, Nat.mul_assoc]

theorem source_episode_capacity_monotone
    (baseDomains firstEpisodes secondEpisodes : ℕ)
    (hEpisodes : firstEpisodes ≤ secondEpisodes) :
    sourceEpisodeDiscrepancyCapacity baseDomains firstEpisodes
      ≤ sourceEpisodeDiscrepancyCapacity baseDomains secondEpisodes := by
  unfold sourceEpisodeDiscrepancyCapacity sourceEpisodeTaskCount
  exact Nat.sub_le_sub_right (Nat.mul_le_mul_left baseDomains hEpisodes) 1

theorem one_episode_per_base_capacity
    (baseDomains : ℕ) :
    sourceEpisodeDiscrepancyCapacity baseDomains 1
      = baseDomains - 1 := by
  simp [sourceEpisodeDiscrepancyCapacity, sourceEpisodeTaskCount]

theorem two_base_single_episode_has_rank_one_capacity :
    sourceEpisodeDiscrepancyCapacity 2 1 = 1 := by
  decide

theorem episode_discrepancy_loading_uses_capacity {p baseDomains episodes : ℕ}
    (feature : Fin p → ℝ)
    (loading :
      Fin p → Fin (sourceEpisodeDiscrepancyCapacity baseDomains episodes) → ℝ) :
    0 ≤ ∑ factor,
      (∑ coefficient, feature coefficient * loading coefficient factor) ^ 2 := by
  exact Finset.sum_nonneg fun _ _ => sq_nonneg _

/-!
V46 does not treat augmented episodes as independent source domains.  A
base-domain mean captures between-domain variation, while centered episode
contrasts capture within-base task variation.  Source-fit covariance belongs
only to the estimated shared mean.  The finite-dimensional identities below
are the scalar projection of that grouped Gaussian hyperlaw.
-/

noncomputable def withinBaseTaskProjectedVariance {b e : ℕ}
    (episodeCoefficient : Fin b → Fin e → ℝ)
    (baseMean : Fin b → ℝ) : ℝ :=
  ∑ base, ∑ episode,
    (episodeCoefficient base episode - baseMean base) ^ 2

theorem within_base_task_projected_variance_nonnegative {b e : ℕ}
    (episodeCoefficient : Fin b → Fin e → ℝ)
    (baseMean : Fin b → ℝ) :
    0 ≤ withinBaseTaskProjectedVariance episodeCoefficient baseMean := by
  unfold withinBaseTaskProjectedVariance
  exact Finset.sum_nonneg fun _ _ =>
    Finset.sum_nonneg fun _ _ => sq_nonneg _

theorem within_base_task_contrast_invariant_to_base_offset {b e : ℕ}
    (episodeCoefficient : Fin b → Fin e → ℝ)
    (baseMean offset : Fin b → ℝ) :
    withinBaseTaskProjectedVariance
        (fun base episode => episodeCoefficient base episode + offset base)
        (fun base => baseMean base + offset base)
      =
    withinBaseTaskProjectedVariance episodeCoefficient baseMean := by
  unfold withinBaseTaskProjectedVariance
  apply Finset.sum_congr rfl
  intro base _
  apply Finset.sum_congr rfl
  intro episode _
  ring

noncomputable def groupedSourceTaskProjectedVariance
    (sharedEstimation channelRole betweenBase withinBase : ℝ) : ℝ :=
  sharedEstimation + channelRole + betweenBase + withinBase

theorem grouped_source_task_projected_variance_nonnegative
    (sharedEstimation channelRole betweenBase withinBase : ℝ)
    (hShared : 0 ≤ sharedEstimation)
    (hRole : 0 ≤ channelRole)
    (hBetween : 0 ≤ betweenBase)
    (hWithin : 0 ≤ withinBase) :
    0 ≤ groupedSourceTaskProjectedVariance
      sharedEstimation channelRole betweenBase withinBase := by
  unfold groupedSourceTaskProjectedVariance
  positivity

theorem grouped_source_task_separates_shared_estimation
    (sharedEstimation channelRole betweenBase withinBase : ℝ) :
    groupedSourceTaskProjectedVariance
        sharedEstimation channelRole betweenBase withinBase
      - channelRole - betweenBase - withinBase
      = sharedEstimation := by
  unfold groupedSourceTaskProjectedVariance
  ring

def withinBaseTaskDiscrepancyCapacity
    (baseDomains episodesPerBase : ℕ) : ℕ :=
  baseDomains * (episodesPerBase - 1)

def groupedTaskDiscrepancyCapacity
    (baseDomains episodesPerBase : ℕ) : ℕ :=
  (baseDomains - 1)
    + withinBaseTaskDiscrepancyCapacity baseDomains episodesPerBase

theorem grouped_task_capacity_is_between_plus_within
    (baseDomains episodesPerBase : ℕ) :
    groupedTaskDiscrepancyCapacity baseDomains episodesPerBase
      =
    (baseDomains - 1)
      + baseDomains * (episodesPerBase - 1) := by
  rfl

theorem two_base_four_episode_grouped_capacity :
    groupedTaskDiscrepancyCapacity 2 4 = 7 := by
  decide

theorem grouped_episode_loading_uses_capacity
    {p baseDomains episodesPerBase : ℕ}
    (feature : Fin p → ℝ)
    (loading :
      Fin p →
        Fin (groupedTaskDiscrepancyCapacity baseDomains episodesPerBase) → ℝ) :
    0 ≤ ∑ factor,
      (∑ coefficient, feature coefficient * loading coefficient factor) ^ 2 := by
  exact Finset.sum_nonneg fun _ _ => sq_nonneg _

/-!
V47 removes source-fit noise from the observed channel-role, between-base,
and within-base random-effect covariance before projecting back to the
positive-semidefinite cone.  In any fixed feature direction, PSD projection
reduces to truncating `observed - fitNoise` at zero.
-/

noncomputable def randomEffectsDeconvolvedProjectedVariance
    (observed fitNoise : ℝ) : ℝ :=
  max (observed - fitNoise) 0

theorem random_effects_deconvolved_projected_variance_nonnegative
    (observed fitNoise : ℝ) :
    0 ≤ randomEffectsDeconvolvedProjectedVariance observed fitNoise := by
  unfold randomEffectsDeconvolvedProjectedVariance
  exact le_max_right _ _

theorem random_effects_deconvolution_no_larger_than_observed
    (observed fitNoise : ℝ)
    (hObserved : 0 ≤ observed)
    (hNoise : 0 ≤ fitNoise) :
    randomEffectsDeconvolvedProjectedVariance observed fitNoise
      ≤ observed := by
  unfold randomEffectsDeconvolvedProjectedVariance
  exact max_le (by linarith) hObserved

theorem random_effects_deconvolution_recovers_latent_variance
    (latent fitNoise : ℝ)
    (hLatent : 0 ≤ latent) :
    randomEffectsDeconvolvedProjectedVariance
        (latent + fitNoise) fitNoise
      = latent := by
  simp [randomEffectsDeconvolvedProjectedVariance, hLatent]

theorem random_effects_deconvolution_removes_noise_dominated_direction
    (observed fitNoise : ℝ)
    (hDominated : observed ≤ fitNoise) :
    randomEffectsDeconvolvedProjectedVariance observed fitNoise = 0 := by
  unfold randomEffectsDeconvolvedProjectedVariance
  rw [max_eq_right]
  linarith

end SCOLHKG.Real
