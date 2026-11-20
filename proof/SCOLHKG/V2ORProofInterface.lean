import SCOLHKG.V2OR.DebiasedSubspace
import SCOLHKG.V2OR.TemperedHVD
import SCOLHKG.V2OR.VerificationKGAmortization
import SCOLHKG.V2OR.FunctionalTransferRisk
import SCOLHKG.V2OR.SequentialTheory

/-!
# Revised *Operations Research* Proof Interface (`V2ORProofInterface`)

Machine-checked Lean 4.31.0 verification manifest linking supporting lemmas and the
focused algorithm-aligned transfer dominance & `k* = 1` binomial certification theorems.
-/

-- Focused Algorithm-Aligned Transfer Boundary & `k*=1` Gate Theorems (`FunctionalTransferRisk`)
#check SCOLHKG.V2OR.FunctionalTransferRisk.effective_mismatch_second_moment_with_finite_sources
#check SCOLHKG.V2OR.FunctionalTransferRisk.transfer_dominates_target_only_iff
#check SCOLHKG.V2OR.FunctionalTransferRisk.realizable_grid_selector_total_risk_le
#check SCOLHKG.V2OR.FunctionalTransferRisk.shrinkageMSE_zero_eq_target_only
#check SCOLHKG.V2OR.FunctionalTransferRisk.grid_with_release_bounded_by_target_only
#check SCOLHKG.V2OR.FunctionalTransferRisk.binomial_one_violation_gate_power_bound
#check SCOLHKG.V2OR.FunctionalTransferRisk.status_quo_loss_roster_invariant

-- Supporting Subspace & Occupancy Lemmas (`DebiasedSubspace`)
#check SCOLHKG.V2OR.DebiasedSubspace.debiased_inner_exact_when_bounded
#check SCOLHKG.V2OR.DebiasedSubspace.debiased_inner_clipping_bias_le
#check SCOLHKG.V2OR.DebiasedSubspace.orthonormal_subset_parseval
#check SCOLHKG.V2OR.DebiasedSubspace.active_pruned_orthogonal_decomposition
#check SCOLHKG.V2OR.DebiasedSubspace.finset_sq_sum_le_card_mul_sum_sq
#check SCOLHKG.V2OR.DebiasedSubspace.pruned_residual_pointwise_square_bound

-- Supporting Within-Policy Variance Shift-Invariance & Shrinkage Lemmas (`TemperedHVD`)
#check SCOLHKG.V2OR.TemperedHVD.within_policy_ssq_shift_invariant
#check SCOLHKG.V2OR.TemperedHVD.untempered_raw_residual_second_moment
#check SCOLHKG.V2OR.TemperedHVD.untempered_exponential_variance_blowup
#check SCOLHKG.V2OR.TemperedHVD.frozen_after_nonvanishing_error_lower_bound
#check SCOLHKG.V2OR.TemperedHVD.tempered_strictly_beats_target_only_at_zero_shift
#check SCOLHKG.V2OR.TemperedHVD.optimal_tempering_oracle_inequality

-- Supporting Terminal Certification & Amortization Lemmas (`VerificationKGAmortization`)
#check SCOLHKG.V2OR.VerificationKGAmortization.v_kg_information_value_nonneg
#check SCOLHKG.V2OR.VerificationKGAmortization.v2_objective_guarded_unsafe_deployment_subset
#check SCOLHKG.V2OR.VerificationKGAmortization.v2_terminal_deployment_fwer_le_delta
#check SCOLHKG.V2OR.VerificationKGAmortization.v2_terminal_five_percent_asymmetric_spend
#check SCOLHKG.V2OR.VerificationKGAmortization.binomial_zero_violation_power_bound
#check SCOLHKG.V2OR.VerificationKGAmortization.v2_terminal_calls_le_cap
#check SCOLHKG.V2OR.VerificationKGAmortization.amortized_cost_le_cold_start_iff

-- Flagship Self-Normalized SURE, Submartingale, Consistency & Bellman Cascade Theorems (`SequentialTheory`)
#check SCOLHKG.V2OR.SequentialTheory.sure_excess_loss_minus_three_halves_empirical_gap
#check SCOLHKG.V2OR.SequentialTheory.sure_selector_zero_shift_excess_loss_bound
#check SCOLHKG.V2OR.SequentialTheory.sure_selector_uniform_shift_excess_loss_bound
#check SCOLHKG.V2OR.SequentialTheory.expected_unit_projection_noise_le_sigma_div_rank
#check SCOLHKG.V2OR.SequentialTheory.hinge_submartingale_step
#check SCOLHKG.V2OR.SequentialTheory.factored_gate_utility_submartingale
#check SCOLHKG.V2OR.SequentialTheory.bgp_kg_nonneg_from_posterior_tower
#check SCOLHKG.V2OR.SequentialTheory.bgp_kg_lipschitz_in_step_innovations
#check SCOLHKG.V2OR.SequentialTheory.greedy_bgp_kg_cannot_starve_informative_arm
#check SCOLHKG.V2OR.SequentialTheory.three_slot_cascade_bellman_identity
#check SCOLHKG.V2OR.SequentialTheory.bellman_slot0_minimizes_expected_deployed_loss

-- Axiom audits on core theorems (must depend strictly on `[propext, Classical.choice, Quot.sound]`)
#print axioms SCOLHKG.V2OR.FunctionalTransferRisk.effective_mismatch_second_moment_with_finite_sources
#print axioms SCOLHKG.V2OR.FunctionalTransferRisk.transfer_dominates_target_only_iff
#print axioms SCOLHKG.V2OR.FunctionalTransferRisk.grid_with_release_bounded_by_target_only
#print axioms SCOLHKG.V2OR.FunctionalTransferRisk.binomial_one_violation_gate_power_bound
#print axioms SCOLHKG.V2OR.SequentialTheory.sure_selector_uniform_shift_excess_loss_bound
#print axioms SCOLHKG.V2OR.SequentialTheory.expected_unit_projection_noise_le_sigma_div_rank
#print axioms SCOLHKG.V2OR.SequentialTheory.bgp_kg_nonneg_from_posterior_tower
#print axioms SCOLHKG.V2OR.SequentialTheory.bgp_kg_lipschitz_in_step_innovations
#print axioms SCOLHKG.V2OR.SequentialTheory.greedy_bgp_kg_cannot_starve_informative_arm
#print axioms SCOLHKG.V2OR.SequentialTheory.bellman_slot0_minimizes_expected_deployed_loss
#print axioms SCOLHKG.V2OR.TemperedHVD.within_policy_ssq_shift_invariant
#print axioms SCOLHKG.V2OR.VerificationKGAmortization.v2_terminal_five_percent_asymmetric_spend
#print axioms SCOLHKG.V2OR.VerificationKGAmortization.amortized_cost_le_cold_start_iff
