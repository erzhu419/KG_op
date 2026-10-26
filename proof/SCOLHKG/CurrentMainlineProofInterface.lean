import SCOLHKG.Real.CurrentPropagation
import SCOLHKG.Real.CurrentCovariance
import SCOLHKG.Measure.CurrentTerminalGate
import SCOLHKG.Real.CurrentRepresentationStability
import SCOLHKG.Real.CurrentQueryStability
import SCOLHKG.Real.CurrentSourceKernel
import SCOLHKG.Real.CurrentAffineMismatch

/-!
# Current mainline proof interface

This interface is separate from the historical Atlas PaperProofInterface.
The declarations certify finite working-moment algebra, covariance congruence
and the actual terminal event/return/cost tree. They do not certify Gaussian
quadrature, multi-round Bayesian exactness, noncentral-t sampling pivots,
floating-point arithmetic, or simulation stream independence.

The terminal measure theorem assumes local certificate error bounds and
fresh-block gated bounds, then derives (rather than assumes) the final 1/20.

Representation results propagate explicit coordinate, encoder and scale error
bounds. Query results compare the same finite pool and realized random vectors,
using strict feasibility and score gaps for the current constrained-TS branches.

Source-prior results construct finite PSD covariances and restrictions.
The fixed affine MSE identity uses actual Bochner integrals with explicit
moment integrability and normal equations, independently of Gaussianity.
-/

#check SCOLHKG.Real.CurrentPropagation.local_completion_contrast_mean
#check SCOLHKG.Real.CurrentPropagation.local_completion_contrast_variance
#check SCOLHKG.Real.CurrentPropagation.evidencePosterior_normalized
#check SCOLHKG.Real.CurrentPropagation.evidencePosterior_nonnegative
#check SCOLHKG.Real.CurrentPropagation.evidence_local_completion_contrast_decomposition
#check SCOLHKG.Real.CurrentPropagation.equal_local_load_preserves_contrast_mean
#check SCOLHKG.Real.CurrentPropagation.rank_one_contrast_variance_decrement
#check SCOLHKG.Real.CurrentPropagation.rank_one_contrast_variance_nonincreasing

#check SCOLHKG.Real.CurrentCovariance.scaled_covariance_psd
#check SCOLHKG.Real.CurrentCovariance.fixed_correlation_diagonal
#check SCOLHKG.Real.CurrentCovariance.scaled_correlation_preserved
#check SCOLHKG.Real.CurrentCovariance.zero_local_variance_zero_row

#check SCOLHKG.Measure.current_terminal_gated_errors_probability_le
#check SCOLHKG.Measure.current_certified_return_or_false_improvement_five_percent
#check SCOLHKG.Measure.current_terminal_abstention
#check SCOLHKG.Measure.current_terminal_calls_branch_identity
#check SCOLHKG.Measure.current_terminal_calls_le_registered_cap

#check SCOLHKG.Real.CurrentRepresentationStability.common_reference_error
#check SCOLHKG.Real.CurrentRepresentationStability.frozen_linear_map_error
#check SCOLHKG.Real.CurrentRepresentationStability.retained_spectral_scale_error
#check SCOLHKG.Real.CurrentRepresentationStability.positive_standardization_uniform_error
#check SCOLHKG.Real.CurrentRepresentationStability.quadratic_features_error
#check SCOLHKG.Real.CurrentRepresentationStability.radial_normalization_coordinate_error
#check SCOLHKG.Real.CurrentRepresentationStability.source_encoder_prediction_error
#check SCOLHKG.Real.CurrentRepresentationStability.cross_dimension_source_prediction_error
#check SCOLHKG.Real.CurrentRepresentationStability.bilinear_kernel_error
#check SCOLHKG.Real.CurrentRepresentationStability.positive_diagonal_reparameterization_kernel

#check SCOLHKG.Real.CurrentQueryStability.sampled_feasible_set_eq
#check SCOLHKG.Real.CurrentQueryStability.feasible_branch_query_stable
#check SCOLHKG.Real.CurrentQueryStability.minimum_violation_query_stable
#check SCOLHKG.Real.CurrentQueryStability.coupled_draw_error_le
#check SCOLHKG.Real.CurrentQueryStability.coupled_draw_pool_error_le
#check SCOLHKG.Real.CurrentQueryStability.chance_margin_error_le

#check SCOLHKG.Real.CurrentSourceKernel.source_covariance_psd
#check SCOLHKG.Real.CurrentSourceKernel.mixture_covariance_psd
#check SCOLHKG.Real.CurrentSourceKernel.physical_white_eq_index_white
#check SCOLHKG.Real.CurrentSourceKernel.source_covariance_restriction
#check SCOLHKG.Real.CurrentSourceKernel.pullback_covariance_psd
#check SCOLHKG.Real.CurrentSourceKernel.role_replacement_covariance_psd

#check SCOLHKG.Real.CurrentAffineMismatch.affine_risk_completed_square
#check SCOLHKG.Real.CurrentAffineMismatch.covariance_psd
#check SCOLHKG.Real.CurrentAffineMismatch.relationship_mismatch_nonnegative
#check SCOLHKG.Real.CurrentAffineMismatch.fixed_affine_mse_decomposition

#print axioms SCOLHKG.Real.CurrentPropagation.evidence_local_completion_contrast_decomposition
#print axioms SCOLHKG.Real.CurrentCovariance.fixed_correlation_covariance_psd
#print axioms SCOLHKG.Measure.current_certified_return_or_false_improvement_five_percent
#print axioms SCOLHKG.Real.CurrentRepresentationStability.bilinear_kernel_error
#print axioms SCOLHKG.Real.CurrentRepresentationStability.radial_normalization_coordinate_error
#print axioms SCOLHKG.Real.CurrentQueryStability.feasible_branch_query_stable
#print axioms SCOLHKG.Real.CurrentQueryStability.minimum_violation_query_stable
#print axioms SCOLHKG.Real.CurrentSourceKernel.source_covariance_psd
#print axioms SCOLHKG.Real.CurrentSourceKernel.mixture_covariance_psd
#print axioms SCOLHKG.Real.CurrentSourceKernel.role_replacement_covariance_psd
#print axioms SCOLHKG.Real.CurrentAffineMismatch.covariance_psd
#print axioms SCOLHKG.Real.CurrentAffineMismatch.fixed_affine_mse_decomposition
