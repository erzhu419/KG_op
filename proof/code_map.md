# Code-To-Theory Map

## Mathematical Objects

| Theory object | Current code object | Status |
| --- | --- | --- |
| Decision vector `x` | `problem` candidates, integer tuples | Implemented |
| Policy-state summary `s(x)` | `policy_state(x)`, `SyntheticPolicyStateEncoder`, `SelfSupervisedPolicyStateEncoder` | Synthetic and learned encoder paths implemented |
| Provider coordinate `psi(x)=(A(x),N(x))` | `core.cumulative_risk.CumulativeRiskFeatureProvider` / `RiskExposure` | Implemented as the main interface |
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
| Exact terminal KG | `SingleOLHKGAlgorithm._exact_posterior_update_scores` | Main default via `acquisition_mode=exact_mc`; additive is an ablation/proxy |

## Implementation Notes

`factor` HVD is now the only mode that consumes cumulative-risk linear
features.  `pooled`, `class`, and `orthogonal` remain pointwise residual
variance estimators, which makes the ablation clean:

- `pooled`: no heteroscedastic structure.
- `class`: regime-level heteroscedasticity.
- `orthogonal`: smooth low-dimensional log-variance.
- `factor`: cumulative trajectory/meta variance with shared shocks.

The current high-dependence path is deliberately controlled around one provider
coordinate.  Factor synthetic, inventory, queue, and traffic all expose
`psi(x)=(A,N)` through `CumulativeRiskFeatureProvider`; factor-HVD, state
candidate anchors, GPR state basis, certification and exact KG all consume that
same coordinate.  The factor synthetic is the oracle-clean case where the true
constraint
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
| `SingleOLHKGAlgorithm._refresh_sequential_basis` | freeze the Stage-1 fallback; an admitted feature switch rebuilds the GPR from the initial empirical prior and replays every recorded rank-one update, while rejection commits the old posterior exactly | `SCOLHKG.Real.RiskAlignedRepresentation.replayPosterior_cons`, `rejectedRepresentationSwitch_is_exact_fallback`, and `admittedRepresentationSwitch_commits_replay` |
| sorted/collapsed `_build_line_envelope` while-loop fold | recursive insert loop over active lines; popped lines remain pointwise dominated by the final output stack; final output endpoint checks lift to original input lines | `SCOLHKG.Real.LineEnvelopeFold.foldLoop_dominates_input`, `foldLoop_output_endpoint_dominance_to_finalInvariant`, and `foldLoop_lineEnvelopeKG_exact_from_output_endpoint_dominance` |
| `variance.OrthogonalHVD.update` | residual record `resid2=(y-mu)^2` | `SCOLHKG.Real.HVDImplementation.residualSquare_nonnegative` |
| `variance.OrthogonalHVD.update(..., replicate_variance=...)` | repeated evaluations replace singleton innovation residuals with nonnegative within-policy sample variance | `SCOLHKG.Real.HVDImplementation.replicateSampleVariance_nonnegative` |
| `meta_component_stage="spectral_hvd"` and factor-HVD target update | source-only boundary-aligned dimensionless variance shape; unlabeled target-measure normalization; prior-centered ridge with nonnegative hierarchical scale `(n*c_target + rho*c_source)/(n+rho)`; target variance-shape updates require within-policy replications | `SCOLHKG.Real.RidgeHVD.priorCenteredPenalty_nonnegative`, `hierarchicalVarianceScale_nonnegative`, and `SCOLHKG.Real.HVDImplementation.replicateSampleVariance_nonnegative` |
| `variance.OrthogonalHVD._fit_output`, factor mode | provider cumulative ridge fit, PSD-project `B`, nonnegative-project `Lambda/floor/omega`, and predict `floor + A^T Lambda A + N^T B N + N^T omega` | `SCOLHKG.Real.RidgeHVD.ridge_hvd_residual_square_oracle`, `SCOLHKG.Real.HVDImplementation.cumulative_linear_prediction_nonnegative`, `clippedVariance_ge_floor`, and `SCOLHKG.Real.CumulativeRiskImplementation.providerRiskBlocks_vCPlus_conservative` |
| `variance.OrthogonalHVD.predict_certification_variance` | `base + model_uncertainty`, guarded by class variance and floor | `SCOLHKG.Real.HVDImplementation.certificationVariance_sound_from_model_uncertainty` |
| `variance.OrthogonalHVD.predict_decomposition`, factor mode | provider block diagnostics `floor/independent/shared/linear/tail_guard/v_C_plus` | `SCOLHKG.Real.CumulativeRiskImplementation.factorShockBlocks_total_eq_components`, `factorShockBlocks_shared_omission_underestimates`, `providerRiskBlocks_total_eq_components`, and `providerRiskBlocks_vCPlus_conservative` |
| `core.certification.conservative_chance_margin` | `mu_g + sqrt(beta_g)s_g + z sqrt(v_C^+) - tau` | `SCOLHKG.Real.CertificationImplementation.implementation_certifies_true_quantile` |
| `acquisition.OLHKGAcquisition.score` | additive proxy retained for ablation | `SCOLHKG.Real.AdditiveApproxKG.additive_proxy_maximizer_exact_gap_le_two_eta` |
| `algorithms.SingleOLHKGAlgorithm._solve_posterior_recommendation` | choose lowest posterior objective among robust chance-feasible candidates | `SCOLHKG.Real.PosteriorRecommendation.robust_feasible_implies_posterior_certified` and `robust_argmin_is_objective_minimizer_on_robust_set` |
| `core.candidates.posterior_sample_candidates` | finite posterior candidate pool from sampled parametric coefficients | `SCOLHKG.Measure.PosteriorCoefficientSampler.posteriorCoefficientSampler_bad_event_le_sum` and `SCOLHKG.Measure.PosteriorSamplingCandidates.randomAdaptiveCenteredSubGaussian_bad_event_le_sum` control random candidate sets by deterministic envelope pools |
| posterior coefficient draw law | sampled parametric coefficient vector with mean/covariance from GPR posterior | `SCOLHKG.Measure.PosteriorMultivariateGaussian` uses mathlib `multivariateGaussian` to prove the draw law, mean, covariance, and Gaussian linear scores |
| finite candidate/kernel budget | scalar information gain `0.5 log(1+var/noise)` accumulated over finite steps | `SCOLHKG.Real.FiniteKernelInformationGain.finiteInformationGain_le_uniform_cap`, `finiteInformationGain_eq_determinantInformationGain_product`, `SCOLHKG.Real.KernelDeterminantBridge.finiteInformationGain_le_determinant_cap`, and `SCOLHKG.Real.FeatureKernelDeterminantCap.finiteInformationGain_le_feature_map_norm_cap` |
| `SingleOLHKGAlgorithm._exact_posterior_update_scores` | MC estimate of current terminal certified value minus updated terminal certified value after GPR/HVD update | `SCOLHKG.Measure.PosteriorUpdateKG.posterior_update_kg_maximizer_is_exact_kg_maximizer` defines the exact target; `SCOLHKG.Real.ExactKGImplementation.exact_mc_estimator_maximizer_gap` bridges uniformly accurate MC estimates; `SCOLHKG.Measure.ExactMCConcentration.exactMC_constant_radius_bad_event_le_sum` gives finite-pool concentration |
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
4. The exact KG estimator is benchmark-wired as `exact_mc`/`blend` and now has
   both a finite-pool concentration theorem and an MC-schedule variance theorem.
   It is not yet empirically promoted over the additive default because current
   probes show a large wall-time multiplier; the large benchmark matrix will
   decide whether main text uses `exact_mc`, `blend`, or additive plus `2 eta`.
5. The traffic encoder/log parser, SUMO trajectory logger, schema-row
   contract, and finite traffic-risk Lean model are implemented.  The remaining
   work is to generate the server-side fresh-seed CSV artifact and include its
   encoded table in the paper package.
