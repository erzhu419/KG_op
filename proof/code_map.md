# Code-To-Theory Map

## Mathematical Objects

| Theory object | Current code object | Status |
| --- | --- | --- |
| Decision vector `x` | `problem` candidates, integer tuples | Implemented |
| Policy-state summary `s(x)` | `policy_state(x)`, `SyntheticPolicyStateEncoder` | Synthetic implemented |
| Idiosyncratic exposure `A(x)` | `FactorShockStatePolicyRZDT1.risk_exposures(x)[0]` | Implemented for synthetic |
| Shared-shock exposure `N(x)` | `FactorShockStatePolicyRZDT1.risk_exposures(x)[1]` | Implemented for synthetic |
| `Lambda`, `B`, `omega` | `cumulative_risk_parameters()` | Implemented for synthetic |
| `v_C(x)` | `true_cumulative_risk_decomposition()["total"]` | Implemented for synthetic |
| HVD predictor `\hat v_C(x)` | `OrthogonalHVD(mode="factor")` cumulative beta | Implemented for synthetic |
| Certified bound | `predict_certification_variance()` in chance margin | Partial |
| SC candidate generation | `state_anchor_points()` and `inverse_state_anchor()` | Synthetic implemented |
| Exact terminal KG | `SingleOLHKGAlgorithm._exact_posterior_update_scores` | Optional MC estimator implemented; additive remains default |

## Implementation Notes

`factor` HVD is now the only mode that consumes cumulative-risk linear
features.  `pooled`, `class`, and `orthogonal` remain pointwise residual
variance estimators, which makes the ablation clean:

- `pooled`: no heteroscedastic structure.
- `class`: regime-level heteroscedasticity.
- `orthogonal`: smooth low-dimensional log-variance.
- `factor`: cumulative trajectory/meta variance with shared shocks.

The current factor synthetic is deliberately controlled: the true constraint
variance is exactly

```text
v_C(x) = A(x)^T Lambda A(x) + N(x)^T B N(x) + N(x)^T omega + floor.
```

This gives the proof and experiments one place where the shared-shock term
`N^T B N` is not merely a metaphor.

## Code-Level Proof Bridges

| Python implementation | Formula implemented | Lean bridge |
| --- | --- | --- |
| `core.gpr.ParametricGPR.update` | rank-one Kalman mean update `m'(u)=m(u)+cov(u,x)/(sigma2+cov(x,x))*(y-m(x))` | `SCOLHKG.Real.GPRUpdate.rank_one_update_standard_shock_slope` |
| `core.kg.compute_kg_vectorized` | KG slope `sigma_tilde(u;x)=cov(u,x)/sqrt(sigma2+cov(x,x))` under a standard predictive shock | `SCOLHKG.Real.GPRUpdate.kg_sigma_tilde_matches_code_formula` plus `rank_one_update_standard_shock_slope` |
| `core.kg.compute_h` | `E[max_j a_j+b_j Z]-max_j a_j` line-envelope KG value | `SCOLHKG.Real.LineEnvelopeKG.certified_lineEnvelopeKG_exact` proves exactness once the active hull certificate is available |
| `variance.OrthogonalHVD.update` | residual record `resid2=(y-mu)^2` | `SCOLHKG.Real.HVDImplementation.residualSquare_nonnegative` |
| `variance.OrthogonalHVD._fit_output`, factor mode | cumulative ridge fit, then `beta=max(beta,0)` and `pred=max(F beta,floor)` | `SCOLHKG.Real.RidgeHVD.ridge_hvd_residual_square_oracle`, `SCOLHKG.Real.HVDImplementation.cumulative_linear_prediction_nonnegative`, `clippedVariance_ge_floor` |
| `variance.OrthogonalHVD.predict_certification_variance` | `base + model_uncertainty`, guarded by class variance and floor | `SCOLHKG.Real.HVDImplementation.certificationVariance_sound_from_model_uncertainty` |
| `acquisition.OLHKGAcquisition.score` | additive proxy `KG_obj + lambda_f KG_feas + lambda_v KG_var + lambda_m KG_mean + lambda_rho KG_coupling` | `SCOLHKG.Real.AdditiveApproxKG.additive_proxy_maximizer_exact_gap_le_two_eta` |
| `algorithms.SingleOLHKGAlgorithm._solve_posterior_recommendation` | choose lowest posterior objective among robust chance-feasible candidates | `SCOLHKG.Real.PosteriorRecommendation.robust_feasible_implies_posterior_certified` and `robust_argmin_is_objective_minimizer_on_robust_set` |
| `core.candidates.posterior_sample_candidates` | finite posterior candidate pool from sampled parametric coefficients | `SCOLHKG.Measure.PosteriorSamplingCandidates.randomAdaptiveCenteredSubGaussian_bad_event_le_sum` controls random candidate sets by deterministic envelope pools |
| finite candidate/kernel budget | scalar information gain `0.5 log(1+var/noise)` accumulated over finite steps | `SCOLHKG.Real.FiniteKernelInformationGain.finiteInformationGain_le_uniform_cap` |
| `SingleOLHKGAlgorithm._exact_posterior_update_scores` | MC estimate of current terminal certified value minus updated terminal certified value after GPR/HVD update | `SCOLHKG.Measure.PosteriorUpdateKG.posterior_update_kg_maximizer_is_exact_kg_maximizer` defines the exact target; MC estimator is optional code path |

The exact posterior-update SC-OLH-KG object is formalized in
`SCOLHKG.Measure.PosteriorUpdateKG`, and the Python runner now has an optional
MC estimator through `exact_kg_mc_samples`.  The default remains the additive
proxy above, so the manuscript has two clean paths:

1. claim additive OLH-KG and use the `2 eta` approximation theorem;
2. promote the sampled exact posterior-update estimator after performance and
   quality validation, then connect it to
   `PosteriorUpdateKG.posterior_update_kg_maximizer_is_exact_kg_maximizer`.

## Remaining Code-To-Theory Gaps

1. `compute_h` has a certificate-level theorem, but the imperative stack
   construction has not been proved to emit the certificate.
2. The posterior-sampling candidate generator is covered by random-set envelope
   containment, but its multivariate-normal coefficient sampling distribution
   has not been formalized.
3. Bounded and generic sub-exponential residual-square interfaces are
   available; the manuscript still needs to choose the final tail class.
4. The optional exact KG estimator is not yet benchmark-promoted over the
   additive default.
5. The traffic encoder/log model is still synthetic-only in code and therefore
   not yet formalized as a real traffic trajectory theorem.
