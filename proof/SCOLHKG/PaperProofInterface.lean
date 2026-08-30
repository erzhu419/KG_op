import SCOLHKG.Measure.ExactBinomialCertificate
import SCOLHKG.Measure.SourceRankRecovery
import SCOLHKG.Measure.TaskAtlasCoverage
import SCOLHKG.Real.FarthestFirstKCenter
import SCOLHKG.Real.GeometricAtlasCoverage
import SCOLHKG.Real.MethodIndependentTerminalVerification
import SCOLHKG.Real.ProfileCoordinateConsistency
import SCOLHKG.Real.ProposalNoFreeLunch
import SCOLHKG.Real.RiskAlignedRepresentation
import SCOLHKG.Real.SourceRankRecovery

/-!
# Final paper proof interface

This file is the compiled, paper-facing proof manifest. Every declaration below
is cited by exact name in the Online Supplement. Other SCOLHKG declarations are
historical or optional results and are not claims of the final paper.
-/

#check SCOLHKG.Real.sourceOnlyProposal_targetLabel_invariant

#check SCOLHKG.Real.continuousProfileCoefficient_error_le
#check SCOLHKG.Real.lipschitz_voronoi_coefficient_inverse_grid_rate
#check SCOLHKG.Real.lipschitz_linear_interpolation_inverse_grid_rate
#check SCOLHKG.Real.frequency_penalty_cannot_increase_coefficient_error

#check SCOLHKG.Real.GonzalezWitnessCertificate.two_approx_every_k_center
#check SCOLHKG.Real.projected_coordinate_atlas_covers

#check SCOLHKG.Measure.finite_source_profile_mean_bad_event_le_sum
#check SCOLHKG.Measure.floored_margin_bad_event_measure_le
#check SCOLHKG.Real.floored_empirical_chance_margin_error_le
#check SCOLHKG.Real.separated_source_pair_order_recovered

#check SCOLHKG.Real.finite_projected_aligned_geometric_lipschitz_atlas_coverage

#check SCOLHKG.Measure.weightedTaskCoverageError_subGaussian
#check SCOLHKG.Measure.weightedTaskCoverageError_abs_tail_le

#check SCOLHKG.Measure.unsafe_all_success_certificate_probability_le
#check SCOLHKG.Measure.finite_all_success_certificates_familywise

#check SCOLHKG.Real.paper_grade_budget_exact_decomposition
#check SCOLHKG.Real.paper_grade_budget_amortized_decomposition

#check SCOLHKG.Real.proper_finite_atlas_misses_some_nonempty_feasible_set
#check SCOLHKG.Real.finite_budget_no_unconditional_target_coverage
