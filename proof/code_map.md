# Code-To-Theory Map

## Mathematical Objects

| Theory object | Current code object | Status |
| --- | --- | --- |
| Decision vector `x` | `problem` candidates, integer tuples | Implemented |
| Policy-state summary `s(x)` | `policy_state(x)`, `SyntheticPolicyStateEncoder`, `SelfSupervisedPolicyStateEncoder` | Synthetic and learned encoder paths implemented |
| Provider coordinate `psi(x)=(A(x),N(x))` | `core.cumulative_risk.CumulativeRiskFeatureProvider` / `RiskExposure` | Implemented as the main interface |
| Observable constraint-mean coordinate `eta(x)` | `representation.observable_coordinate.SourceLearnedObservableCoordinate` / `ObservableConstraintMeanBasis` | Frozen from ordinary replicated source simulations; target coefficients use only charged observations |
| Source boundary excitation | `LearnedMetaPrior.source_design_mode=universal_mixture`, frozen generic low-frequency policy library, and source sign-support diagnostics | Oracle-free structural design; source calls are charged and random-source control is retained |
| Idiosyncratic exposure `A(x)` | `risk_exposures(x).A` | Implemented for factor synthetic, inventory, queue, and traffic proxy/CSV |
| Shared-shock exposure `N(x)` | `risk_exposures(x).N` | Implemented for factor synthetic, inventory, queue, and traffic proxy/CSV |
| `Lambda`, `B`, `omega` | `cumulative_risk_parameters()` | Implemented for synthetic |
| `v_C(x)` | `true_cumulative_risk_decomposition()["total"]` | Implemented for synthetic |
| HVD predictor `\hat v_C(x)` | `OrthogonalHVD(mode="factor")` cumulative beta | Implemented for synthetic |
| Certified bound | `core.certification.conservative_chance_margin` with `predict_certification_variance()` | Implemented for theory/legacy modes |
| SC candidate generation | `state_anchor_points()` and `inverse_state_anchor()` | Synthetic implemented |
| Self-supervised trajectory representation | `SelfSupervisedTrajectoryEncoder`, `TransformerTrajectoryEncoder` | Implemented for masked, contrastive, and attention-style pooling ablations |
| Boundary-aligned LODO representation | `BoundaryAlignedRiskSubspaces`, compact source-expert mixture, frozen source-boundary episode admission, target nested-LOO diagnostics | Implemented; paired N=40 KG promotion matrix is running |
| Traffic occupancy encoder | `TrafficTrajectoryEncoder` plus `sumo_sim.py` trajectory logger | Implemented for fresh-seed CSV schema; large trajectory table requires server-generated logs |
| Adaptive Stage-I exact KG | `SingleOLHKGAlgorithm._exact_posterior_update_scores` | Adaptive search acquisition after `n0` via `acquisition_mode=exact_mc`; it is not a claim about the initial design or reserved verification suffix |
| Two-stage terminal decision | `finalist_replication_budget`, fixed finalist universe, `_replicated_finalist_recommendation_index`, `two_stage_decision` diagnostics | Main deployed decision architecture; certified and uncertified outputs have distinct theorem claims |
| Finite task-structure posterior `Q_t(xi)` | `representation.task_posterior.FiniteTaskPosterior` | Implemented behind `task_posterior_mode=finite`; FactorShock N=20 Gate 1 promoted on 2026-07-11, cross-domain Gate 2 pending |
| Expert-specific surrogate state | `FiniteTaskModelEnsemble` / `TaskExpertState` | Frozen source expert basis plus independent GPR/HVD per expert |
| Task-robust cumulative certificate | `FiniteTaskModelEnsemble.robust_moments_many` | Within/between variance plus forward-KL robust upper moments |

## Implementation Notes

`factor` HVD is now the only mode that consumes cumulative-risk linear
features.  `pooled`, `class`, and `orthogonal` remain pointwise residual
variance estimators, which makes the ablation clean:

- `pooled`: no heteroscedastic structure.
- `class`: regime-level heteroscedasticity.
- `orthogonal`: smooth low-dimensional log-variance.
- `factor`: cumulative trajectory/meta variance with shared shocks.

The current high-dependence path uses two noninterchangeable coordinates.
Factor synthetic, inventory, queue, and traffic expose `psi(x)=(A,N)` through
`CumulativeRiskFeatureProvider`; factor-HVD, state candidate anchors,
certification variance, and exact-KG variance updates consume `psi`.  A frozen
source-learned observable coordinate `eta(x)` drives only the constraint-mean
GPR and its epistemic uncertainty.  They meet in one chance margin, rather than
assuming that `psi` must also be sufficient for the conditional mean.  The
factor synthetic is the oracle-clean case where the true constraint
variance is exactly

```text
v_C(x) = A(x)^T Lambda A(x) + N(x)^T B N(x) + N(x)^T omega + floor.
```

This gives the proof and experiments one place where the shared-shock term
`N^T B N` is not merely a metaphor.

The default paper-style feasibility certificate is now the theory bound

```text
mu_g(x) + sqrt(beta_g) s_g(x) + z_alpha sqrt(v_C^+(x)) <= tau,
```

where `s_g` is the constraint GPR posterior standard deviation and `v_C^+`
comes from HVD certification variance.  Legacy mode remains available only as
an ablation.

## Code-Level Proof Bridges

| Python implementation | Formula implemented | Lean bridge |
| --- | --- | --- |
| `core.gpr.ParametricGPR.update` | rank-one Kalman mean update `m'(u)=m(u)+cov(u,x)/(sigma2+cov(x,x))*(y-m(x))` | `SCOLHKG.Real.GPRUpdate.rank_one_update_standard_shock_slope` |
| `core.kg.compute_kg_vectorized` | KG slope `sigma_tilde(u;x)=cov(u,x)/sqrt(sigma2+cov(x,x))` under a standard predictive shock | `SCOLHKG.Real.GPRUpdate.kg_sigma_tilde_matches_code_formula` plus `rank_one_update_standard_shock_slope` |
| `SingleOLHKGAlgorithm._replication_candidates` | an observed point with epistemic variance `q` and observation variance `r` offers posterior variance reduction `q^2/(q+r)`; exact KG, rather than a heuristic bonus, decides whether to repeat it | `SCOLHKG.Real.GPRUpdate.replication_variance_update_identity`, `replication_variance_reduction_nonnegative`, and `replication_variance_reduction_le_epistemic` |
| `core.gpr.ParametricGPR._project_covariance_psd` and stabilized `update` denominator | a negative numerical quadratic form is clipped to zero before adding strictly positive observation variance | `SCOLHKG.Real.GPRUpdate.stabilizedQuadraticVariance_nonnegative`, `stabilizedQuadraticVariance_eq_self`, and `stabilizedPredictiveVariance_positive` |
| `core.kg.compute_h` | `E[max_j a_j+b_j Z]-max_j a_j` line-envelope KG value | `SCOLHKG.Real.LineEnvelopeKG.certified_lineEnvelopeKG_exact` proves exactness once the active hull certificate is available |
| `core.kg.compute_h_certificate`, `validate_h_certificate` | active stack hull, cuts, Gaussian masses, endpoint/tail-slope dominance checks | `SCOLHKG.Real.LineEnvelopeStack` proves the validator conditions imply active-line certificates |
| `core.kg.compute_h_certificate(...).trace` | per-step stack snapshots for candidate, break, pop, and push actions | `SCOLHKG.Real.LineEnvelopeAlgorithm` proves pop/push preserve slope and cut order under the Python branch conditions |
| final `compute_h` hull state | active atoms whose lines dominate all original/processed lines at interval endpoints | `SCOLHKG.Real.LineEnvelopeGlobal` proves final global dominance implies atom certificates and exact line-envelope KG without a runtime validator |
| `core.kg._build_line_envelope` intersection `z=(a_old-a_new)/(b_new-b_old)` | old active line dominates left of the cut; new line dominates right of the cut; pop and right-tail split preserve certificates over every point of the affected interval/tail | `SCOLHKG.Real.LineEnvelopeIntersection` proves the concrete intersection arithmetic, popped-cell takeover, right-tail finite/tail cell construction, and interval/tail dominance |
| `representation.risk_aligned_subspace.BoundaryAlignedRiskSubspaces` | transfer projectors instead of named latent axes; orthogonal source/target rotations leave `UU^T` unchanged | `SCOLHKG.Real.RiskAlignedRepresentation.subspaceProjector_rotation_invariant` |
| `TransferableSpectralBasis` retained-eigenvalue whitening | discard numerically null directions and whiten only the identifiable rank | `SCOLHKG.Real.RiskAlignedRepresentation.retainedWhitening_orthonormal` |
| compact source risk-expert ensemble | nonnegative weights summing to one mix normalized source chance-boundary scores | `SCOLHKG.Real.RiskAlignedRepresentation.simplexExpertMixture_mem_interval` |
| `PilotGatedMetaPriorBasis._nested_alignment_loo_predictions` | refit the complete target alignment after removing each held-out pilot label | `SCOLHKG.Real.RiskAlignedRepresentation.nestedLOO_refit_does_not_read_heldout_label` |
| guarded additive group gate | strong heredity plus source-support floor; rejected groups return the exact prior model | `SCOLHKG.Real.RiskAlignedRepresentation.strongHeredity_survives_interaction_filter` and `weakSourceSupport_is_exact_fallback` |
| target alignment/frequency evidence gate | require observed feasible and infeasible chance margins; one-sided pilots and rejected adapters leave the Stage-1 basis unchanged, then may be reconsidered after a fixed observation interval | `SCOLHKG.Real.RiskAlignedRepresentation.noFeasibleEvidence_is_exact_fallback` and `noInfeasibleEvidence_is_exact_fallback` |
| `SourceBoundaryEpisodePrior` plus frozen profile replay | source-only, disjoint pilot/evaluation episodes may replace unstable target gain evidence, but cannot replace two-sided target support or target safety checks; proposals do not read target labels | `SCOLHKG.Real.RiskAlignedRepresentation.sourceEpisodeAdmission_requires_twoSided`, `sourceEpisodeAdmission_requires_targetSafety`, `sourceEpisodeSupport_replaces_gainEvidence_only`, and `sourceOnlyProposal_targetLabel_invariant` |
| `benchmark_lodo_meta_prior.meta_source_seed`, paper mode | source histories and the learned prior are fixed across held-out target evaluation seeds; target-seed-dependent source retraining is sensitivity-only | `SCOLHKG.Real.RiskAlignedRepresentation.sourceOnlyProposal_targetLabel_invariant` |
| replicated `universal_mixture` source design | one target-formula-free low-frequency policy library is simulated in each source domain; one-sided source margins fail strict excitation, and source-only estimators cannot distinguish margin functions agreeing on the full source design | `SCOLHKG.Real.BoundaryExcitation.nonnegative_source_not_strictly_excited`, `source_indistinguishability_lower_bound`, and `source_only_decision_cannot_separate_agreeing_models` |
| `SingleOLHKGAlgorithm._refresh_sequential_basis` | freeze the Stage-1 fallback; an admitted feature switch rebuilds the GPR from the initial empirical prior and replays every recorded rank-one update, while rejection commits the old posterior exactly | `SCOLHKG.Real.RiskAlignedRepresentation.replayPosterior_cons`, `rejectedRepresentationSwitch_is_exact_fallback`, and `admittedRepresentationSwitch_commits_replay` |
| sorted/collapsed `_build_line_envelope` while-loop fold | recursive insert loop over active lines; popped lines remain pointwise dominated by the final output stack; final output endpoint checks lift to original input lines | `SCOLHKG.Real.LineEnvelopeFold.foldLoop_dominates_input`, `foldLoop_output_endpoint_dominance_to_finalInvariant`, and `foldLoop_lineEnvelopeKG_exact_from_output_endpoint_dominance` |
| `variance.OrthogonalHVD.update` | residual record `resid2=(y-mu)^2` | `SCOLHKG.Real.HVDImplementation.residualSquare_nonnegative` |
| `variance.OrthogonalHVD.update(..., replicate_variance=...)` | repeated evaluations replace singleton innovation residuals with nonnegative within-policy sample variance | `SCOLHKG.Real.HVDImplementation.replicateSampleVariance_nonnegative` |
| `meta_component_stage="spectral_hvd"` and factor-HVD target update | source-only boundary-aligned dimensionless variance shape; unlabeled target-measure normalization; prior-centered ridge with nonnegative hierarchical scale `(n*c_target + rho*c_source)/(n+rho)`; target variance-shape updates require within-policy replications | `SCOLHKG.Real.RidgeHVD.priorCenteredPenalty_nonnegative`, `hierarchicalVarianceScale_nonnegative`, and `SCOLHKG.Real.HVDImplementation.replicateSampleVariance_nonnegative` |
| `variance.OrthogonalHVD._fit_output`, factor mode | provider cumulative ridge fit, PSD-project `B`, nonnegative-project `Lambda/floor/omega`, and predict `floor + A^T Lambda A + N^T B N + N^T omega` | `SCOLHKG.Real.RidgeHVD.ridge_hvd_residual_square_oracle`, `SCOLHKG.Real.HVDImplementation.cumulative_linear_prediction_nonnegative`, `clippedVariance_ge_floor`, and `SCOLHKG.Real.CumulativeRiskImplementation.providerRiskBlocks_vCPlus_conservative` |
| `variance.OrthogonalHVD.predict_certification_variance` | `base + model_uncertainty`, guarded by class variance and floor | `SCOLHKG.Real.HVDImplementation.certificationVariance_sound_from_model_uncertainty` |
| `variance.OrthogonalHVD.predict_decomposition`, factor mode | provider block diagnostics `floor/independent/shared/linear/tail_guard/v_C_plus` | `SCOLHKG.Real.CumulativeRiskImplementation.factorShockBlocks_total_eq_components`, `factorShockBlocks_shared_omission_underestimates`, `providerRiskBlocks_total_eq_components`, and `providerRiskBlocks_vCPlus_conservative` |
| `core.certification.conservative_chance_margin` | `mu_g + sqrt(beta_g)s_g + z sqrt(v_C^+) - tau` | `SCOLHKG.Real.CertificationImplementation.implementation_certifies_true_quantile` |
| `ObservableConstraintMeanBasis` plus expert `CumulativeRiskFeatureProvider` | Separate `eta` mean/epistemic and `psi` cumulative-risk coordinates joined only by the certified margin | `SCOLHKG.Real.MeanRiskCoordinateSeparation.separated_certificate_sound` and joint-coordinate noninterference lemmas |
| `acquisition.OLHKGAcquisition.score` | additive proxy retained for ablation | `SCOLHKG.Real.AdditiveApproxKG.additive_proxy_maximizer_exact_gap_le_two_eta` |
| `algorithms.SingleOLHKGAlgorithm._solve_posterior_recommendation` | choose lowest posterior objective among robust chance-feasible candidates | `SCOLHKG.Real.PosteriorRecommendation.robust_feasible_implies_posterior_certified` and `robust_argmin_is_objective_minimizer_on_robust_set` |
| `core.candidates.posterior_sample_candidates` | finite posterior candidate pool from sampled parametric coefficients | `SCOLHKG.Measure.PosteriorCoefficientSampler.posteriorCoefficientSampler_bad_event_le_sum` and `SCOLHKG.Measure.PosteriorSamplingCandidates.randomAdaptiveCenteredSubGaussian_bad_event_le_sum` control random candidate sets by deterministic envelope pools |
| posterior coefficient draw law | sampled parametric coefficient vector with mean/covariance from GPR posterior | `SCOLHKG.Measure.PosteriorMultivariateGaussian` uses mathlib `multivariateGaussian` to prove the draw law, mean, covariance, and Gaussian linear scores |
| finite candidate/kernel budget | scalar information gain `0.5 log(1+var/noise)` accumulated over finite steps | `SCOLHKG.Real.FiniteKernelInformationGain.finiteInformationGain_le_uniform_cap`, `finiteInformationGain_eq_determinantInformationGain_product`, `SCOLHKG.Real.KernelDeterminantBridge.finiteInformationGain_le_determinant_cap`, and `SCOLHKG.Real.FeatureKernelDeterminantCap.finiteInformationGain_le_feature_map_norm_cap` |
| `SingleOLHKGAlgorithm._exact_posterior_update_scores` | MC estimate of current terminal certified value minus updated terminal certified value after GPR/HVD update | `SCOLHKG.Measure.PosteriorUpdateKG.posterior_update_kg_maximizer_is_exact_kg_maximizer` defines the exact target; `SCOLHKG.Real.ExactKGImplementation.exact_mc_estimator_maximizer_gap` bridges uniformly accurate MC estimates; `SCOLHKG.Measure.ExactMCConcentration.exactMC_constant_radius_bad_event_le_sum` gives finite-pool concentration |
| shared `terminal_pool` plus posterior frontier experiment actions in `SingleOLHKGAlgorithm.run` | the current recommendation, every hypothetical update, and the realized post-update recommendation use the same history-measurable terminal pool; posterior-only frontier actions are included in the experiment set | `SCOLHKG.Measure.SharedTerminalPoolKG.shared_terminal_pool_gain_uses_pre_state_pool`, `shared_terminal_pool_maximizer_is_one_step_optimal`, `terminal_frontier_subset_closed_experiments`, and `original_experiments_subset_closed_experiments` |
| `FiniteTaskPosterior.update_from_predictive` | positive generalized-Bayes expert mass followed by simplex normalization | `SCOLHKG.Real.TaskPosterior.generalizedBayes_normalized_support` and `normalizeFiniteWeights_sum_eq_one` |
| `FiniteTaskLatentPosterior` shadow/authoritative joint update | positive normalized generalized-Bayes mass on `(structural expert, signed-bias-function/scale/loss class)`, normalized nonnegative marginals, source-frozen scalar or `b(psi)` bias used only by Bayes decisions, and authoritative epistemic scaling by `max(1,c_scale)^2` that cannot relax the theory margin | `SCOLHKG.Real.JointTaskLatentPosterior` |
| V4 expert-conditional calibration posterior | charged standardized residuals update one conjugate precision/information pair per structural expert; posterior coefficient means affect Bayes ranking only, while `predictive_sd^2 phi^T P^{-1} phi >= 0` is added to theory epistemic variance and exact-KG clones the same state | `SCOLHKG.Real.JointTaskLatentPosterior.expert_calibration_precision_update_ge_prior`, `adaptive_theory_margin_ignores_calibration_mean`, and `adaptive_calibration_covariance_cannot_relax_theory_margin` |
| V28 `Q_pred` / `Q_safe` update and decision split | two independently normalized full-support generalized-Bayes masses; clipped threshold/pairwise losses lie in `[0,-log(epsilon)]`; objective aggregation uses `Q_pred`, while proposals/certification/exact KG use `Q_safe` | `SCOLHKG.Real.SafeGeneralizedTaskPosterior.predictiveTaskMass_sum_eq_one`, `safeDecisionTaskMass_sum_eq_one`, `clipped_probability_log_loss_bounded`, `safe_generalized_pac_bayes_bound_on_moment_event`, and `safeDecisionJointBelief_uses_safe_weight` |
| `FiniteTaskPosterior.proposal_weights` plus expert-mixture initial/sequential/terminal proposals | `(1-epsilon) Q_t + epsilon Pi` is normalized and retains at least `epsilon * Pi(xi)` mass on every prior-supported expert | `SCOLHKG.Real.TaskPosterior.finite_task_proposal_normalized`, `finite_task_proposal_preserves_prior_support`, and `finite_task_proposal_positive_of_prior_positive` |
| `FiniteTaskPosterior.mixture_moments` | `E_Q[s_k^2] + Var_Q[m_k] + E_Q[v_k]` | `SCOLHKG.Real.TaskPosterior.task_total_variance_is_within_between_aleatoric` |
| `FiniteTaskPosterior.kl_robust_expectation` and robust certification | entropic dual upper bound over `KL(q||p) <= rho`, followed by robust moment certification | `SCOLHKG.Real.TaskPosterior.finiteTaskKL_nonnegative`, `kl_ball_entropic_upper`, `finite_pac_bayes_bound_on_moment_event`, `SCOLHKG.Measure.TaskPACBayes`, and `robust_certificate_holds_for_every_admissible_task_posterior` |
| finite task-posterior exact-MC branch | sample expert identity, clone/update task weights plus every expert GPR/HVD, then recompute robust terminal value | `SCOLHKG.Real.TaskPosterior.joint_task_exact_mc_zero_error_is_one_step_optimal` plus the existing MC concentration layer |
| `exact_kg_sampling_mode=stratified_expert` | enumerate every finite expert with its posterior decision weight and use common antithetic Gaussian innovations within expert | `SCOLHKG.Real.StratifiedExpertKG.finite_stratified_identity_has_no_categorical_error` and `finite_stratified_error_le_conditional_error` prove that the categorical layer is exact and only conditional Gaussian error remains |
| `LearnedMetaPrior.ordered_cumulative_risk_exposure` and the `ordered_cumulative` finite expert | source-selected global plus low-frequency positional exposure is used by the same expert GPR, factor-HVD, certificate, proposal family, and exact KG update | `SCOLHKG.Real.OrderedCumulativeExposure.zero_frequency_is_aggregate_exposure`, `selected_frequency_is_positional_exposure`, and `ordered_coordinate_uses_cumulative_risk_decomposition` |
| `FixedTaskExpertBasis` ordered semiparametric coefficient-nullspace projection | bounded local RBF center coefficients are projected into the nullspace of the finite ordered/kernel cross matrix before entering the same capped expert | `SCOLHKG.Real.OrthogonalSemiparametric.coefficientNullspace_orthogonal_all` proves finite-design orthogonality, while `finiteKernelCombination_abs_le_card` gives a candidate-independent finite amplitude bound |
| `AdaptiveSpikeSlabPosterior.fit` total-rank projection | reserves the fixed prefix inside the `max_effective_fraction * N` budget, then projects optional inclusion probabilities onto the remaining budget | `SCOLHKG.Real.AdaptiveCoefficientSparsity.fixed_prefix_and_optional_budget_control_total_dimension` and `fraction_budget_controls_total_dimension` |
| adaptive PIP cardinality bracketing and damping | dynamically reaches the minimum-PIP endpoint before bisection, then damps the previous and proposed budget-feasible vectors | `SCOLHKG.Real.AdaptiveCoefficientSparsity.constant_inclusion_floor_is_budget_feasible` and `damping_preserves_effective_dimension_budget` prove endpoint feasibility and preservation under damping |
| `AdaptiveGroupRidgePosterior` nested target refits | full LOO refits choose one isotropic ridge penalty per `A`, `A^2`, and `N` group from a fixed finite grid; analytic target truth is never read | `SCOLHKG.Real.GroupRidgeComplexity.finiteRidgeEffectiveDimension_le_feature_count` bounds learned effective df and `finite_nested_selector_oracle_bound` gives the finite-selector `2 epsilon` oracle inequality |
| V29 budgeted finalist replication | freeze posterior-only finalists before new labels, spend only the final reserved stages to balance replicate counts, and use `ybar_g + z_alpha sigma + z_delta sigma/sqrt(r) - tau` only when the theory-certified set is empty | `SCOLHKG.Real.FinalistReplication.replicated_finalist_margin_sound_on_joint_event`, `frozen_finalists_ignore_future_labels`, `one_replication_strictly_reduces_positive_deficit`, and `reserved_finalist_stage_stays_inside_total_budget` |
| V30 expert-stratified safety nomination | every finite structural expert nominates its own minimum predicted-violation action before posterior-mass-free finalist ranking | `SCOLHKG.Real.FinalistReplication.every_finite_expert_nomination_is_supported` and `nomination_support_does_not_depend_on_posterior_mass` |
| V31 history-measurable adaptive finalist race | refresh the expert challenger before each paid suffix observation, archive every tested action, and compare only candidates that meet the replication contract | `SCOLHKG.Real.FinalistReplication.adaptive_archive_contains_every_nomination`, `adaptive_archive_card_le_initial_add_refreshes`, `incomplete_finalist_cannot_enter_completed_race`, `completed_adaptive_finalist_sound_on_joint_event`, and `adaptive_finalist_bad_event_le_sum` |
| V32 fixed-universe adaptive race | freeze the finite terminal action universe before suffix labels and rerank only that universe after each paid update | `SCOLHKG.Real.FinalistReplication.adaptive_archive_subset_fixed_universe` and `adaptive_archive_card_le_fixed_universe` |
| Oracle-free two-stage source-consensus baseline | use `n0` source-informed initial-design calls and `N-R-n0` adaptive state-coupled exact-KG calls, freeze the terminal universe, spend `R` charged calls on heteroscedastic ranking-and-selection, and report posterior/replication certification separately from least-risk fallback | `SCOLHKG.Real.TwoStageDecision` proves the budget partition, terminal semantics, certified safety, fallback relative-risk bound, and deterministic regret decomposition; `SCOLHKG.Measure.TwoStageDecision` proves finite verification concentration and high-probability event transfer |
| TCB-V2 hierarchical boundary plus V33 three-layer repair | `HierarchicalSignedDistancePosterior` learns one source boundary shape with target location, positive scale, optional planar rotation, and orthogonal low-rank residual; `decision_contract_mode=certified_lexicographic` makes main exact KG, coverage-reserved finalist nomination, suffix KG, and final recommendation consume the same authoritative upper margin, while every fantasy refits the same adapter | `SCOLHKG.Real.HierarchicalBoundaryCertificate` proves positive scale, planar-rotation norm preservation, nonnegative predictive covariance, upper-margin monotonicity under rotation/residual uncertainty, reserved-frontier order, frontier/terminal/recommendation coherence, recommendation safety under upper coverage, and lexicographic certified-action dominance |
| TCB-V3 finite boundary-family posterior | `BoundaryFamilyMixturePosterior` supports broad and atomic source-frozen libraries, updates only family mass from leave-one-pilot-out target evidence, ranks by posterior mean, and certifies with a `1-delta_family` credible-family upper envelope plus nonnegative guard | `SCOLHKG.Real.BoundaryFamilyMixtureCertificate` proves credible-mass accounting, envelope coverage when the true family remains credible, guard monotonicity, safe recommendation, target-name noninterference, and failure probability at most `delta_family + alpha` |
| TCB-V4 continuous boundary-family synthesis | `BoundaryFamilySynthesisPosterior` freezes one canonical signed-distance atom per source domain, learns a source coefficient prior, updates a held-out intercept and nonnegative atom coefficients from ordinary pilots, and adds coefficient covariance plus residual uncertainty to its Student-t upper margin | `SCOLHKG.Real.BoundaryFamilySynthesisCertificate` proves nonnegative-synthesis monotonicity, nonnegative predictive variance, upper-margin non-relaxation, recommendation safety under coverage, and target-name noninterference |
| TCB-V5 orthogonal semiparametric boundary | `BoundaryFamilySemiparametricPosterior` adds one or two frozen RBF residual coordinates in source-family score space, projects their center coefficients into the source-design cross nullspace, updates only low-dimensional target coefficients, and includes both covariance blocks plus remaining noise in the upper margin | `SCOLHKG.Real.OrthogonalSemiparametric` proves coefficient-nullspace orthogonality and bounded kernel combinations; `SCOLHKG.Real.BoundaryFamilySemiparametricCertificate` proves nonnegative direct-sum predictive variance and recommendation safety under coverage |
| Noise-limited certifiability audit | `core.oracle_certification.oracle_certifiability_metrics` evaluates `m + q sigma / sqrt(R)` and the closed-form known-variance replication burden; `benchmark_certifiability_coordinate_audit.py` keeps strict, target-oracle, and provider-upper-bound strata separate | `SCOLHKG.Real.OracleCertifiability` proves radius nonnegativity/replication monotonicity, squared-budget sufficiency, and persistence of certification at larger budgets |
| Frozen source-consensus proposal and committed safety shortlist | `LearnedMetaPrior._fit_source_consensus_templates` aggregates within-source empirical chance-margin ranks over one shared universal archive; `initial_universal_candidates` freezes a structural sentinel plus a rank-spanning source design; `observed_safety_reserved` restricts its protected shortlist to charged members of that design, preserves a safety alias when the first arm duplicates minimum Bayes risk, and `commit_before_switch` finishes all reserved arms before changing to posterior-only experts | `SCOLHKG.Real.SourceConsensusCommit.positiveAffine_order_invariant` proves rank-order scale invariance, the rank-spanning lemmas record endpoint coverage, `source_frozen_proposal_target_noninterference` records target-name noninterference, `bounded_error_preserves_two_challenger_order` gives the simultaneous-error ordering condition, and the two-challenger commitment theorems prove exact completion within the reserved suffix budget |
| `AdaptiveSpikeSlabPosterior.fit` semantic group-shared shrinkage | one joint Bayes factor, PIP, and isotropic slab/spike scale are shared within each declared `A^2` or `N` block, while target data learn the coefficient direction | `SCOLHKG.Real.GroupSharedShrinkage.sharedGroupPenalty_rotation_invariant` and `sharedGroupEffectiveDimension_eq` prove within-group coordinate invariance and full coefficient-budget accounting |
| `TrafficTrajectoryEncoder` fresh CSV aggregate | state-action occupancy plus queue/wait/flow and demand-shock exposure | `SCOLHKG.Real.TrafficTrajectoryModel.totalRisk_decomposition`, `sharedShock_omission_underestimates`, and `TrafficLogSchemaRow` formalize the finite traffic risk model and CSV schema semantics |

The exact posterior-update SC-OLH-KG object is formalized in
`SCOLHKG.Measure.PosteriorUpdateKG`, and the Python runner now defaults to the
MC estimator through `acquisition_mode=exact_mc` and `exact_kg_mc_samples=8`.
The additive proxy above is now explicitly an ablation; if exact-MC is too
expensive in a table, the manuscript must cite the `2 eta` approximation lemma
instead of presenting additive as the main mathematical object.

## Code-To-Theory Closure Notes

1. `compute_h` now emits and validates a checkable certificate, the validator
   conditions are Lean-bridged, pop/push cut-order preservation is Lean-proved,
   concrete intersection/pop/split branch certificates are Lean-proved, and
   final global dominance implies exact KG without runtime validation.  The
   full recursive sorted-line fold is also Lean-proved: popped active lines are
   pointwise dominated by the final output stack, and output endpoint dominance
   lifts to `FinalEnvelopeStackInvariant` over all original input lines.
2. The posterior-sampling candidate generator now has both pieces: mathlib
   multivariate-Gaussian coefficient law and finite selector/envelope
   containment.  The remaining manuscript choice is only the exact final
   posterior covariance parameterization to describe in prose.
3. Bounded and generic sub-exponential residual-square interfaces are
   available, and the default radius is exposed in code/proof with a
   closed-form inversion theorem.
4. The exact KG estimator is the Stage-I search acquisition and has both a
   finite-pool concentration theorem and an MC-schedule variance theorem. The
   reserved suffix is separately modeled as finite heteroscedastic
   ranking-and-selection; the new `two_stage_decision` result block audits this
   implementation contract without pretending the full run is exact KG.
5. The traffic encoder/log parser, SUMO trajectory logger, schema-row
   contract, and finite traffic-risk Lean model are implemented.  The remaining
   work is to generate the server-side fresh-seed CSV artifact and include its
   encoded table in the paper package.
