import SCOLHKG.V2OR.DebiasedSubspace
import SCOLHKG.V2OR.TemperedHVD
import SCOLHKG.V2OR.VerificationKGAmortization

/-!
# Path-B Flagship *Operations Research* Proof Interface (`V2ORProofInterface`)

Authoritative Lean 4.31.0 verification manifest for **TLS-OLH-KG** (`SC-OLH-KG/v2_or`).
-/

-- Theorem 1 checks
#check SCOLHKG.V2OR.DebiasedSubspace.debiased_inner_exact_when_bounded
#check SCOLHKG.V2OR.DebiasedSubspace.debiased_inner_clipping_bias_le
#check SCOLHKG.V2OR.DebiasedSubspace.orthonormal_subset_parseval
#check SCOLHKG.V2OR.DebiasedSubspace.active_pruned_orthogonal_decomposition
#check SCOLHKG.V2OR.DebiasedSubspace.finset_sq_sum_le_card_mul_sum_sq
#check SCOLHKG.V2OR.DebiasedSubspace.pruned_residual_pointwise_square_bound

-- Theorem 2 checks
#check SCOLHKG.V2OR.TemperedHVD.within_policy_ssq_shift_invariant
#check SCOLHKG.V2OR.TemperedHVD.untempered_raw_residual_second_moment
#check SCOLHKG.V2OR.TemperedHVD.untempered_exponential_variance_blowup
#check SCOLHKG.V2OR.TemperedHVD.frozen_after_nonvanishing_error_lower_bound
#check SCOLHKG.V2OR.TemperedHVD.tempered_strictly_beats_target_only_at_zero_shift
#check SCOLHKG.V2OR.TemperedHVD.optimal_tempering_oracle_inequality

-- Theorem 3 checks
#check SCOLHKG.V2OR.VerificationKGAmortization.v_kg_information_value_nonneg
#check SCOLHKG.V2OR.VerificationKGAmortization.v2_objective_guarded_unsafe_deployment_subset
#check SCOLHKG.V2OR.VerificationKGAmortization.v2_terminal_deployment_fwer_le_delta
#check SCOLHKG.V2OR.VerificationKGAmortization.v2_terminal_five_percent_asymmetric_spend
#check SCOLHKG.V2OR.VerificationKGAmortization.binomial_zero_violation_power_bound
#check SCOLHKG.V2OR.VerificationKGAmortization.v2_terminal_calls_le_cap
#check SCOLHKG.V2OR.VerificationKGAmortization.amortized_cost_le_cold_start_iff

-- Axiom audits on headline theorems
#print axioms SCOLHKG.V2OR.DebiasedSubspace.debiased_inner_clipping_bias_le
#print axioms SCOLHKG.V2OR.DebiasedSubspace.active_pruned_orthogonal_decomposition
#print axioms SCOLHKG.V2OR.TemperedHVD.within_policy_ssq_shift_invariant
#print axioms SCOLHKG.V2OR.TemperedHVD.untempered_exponential_variance_blowup
#print axioms SCOLHKG.V2OR.TemperedHVD.optimal_tempering_oracle_inequality
#print axioms SCOLHKG.V2OR.VerificationKGAmortization.v_kg_information_value_nonneg
#print axioms SCOLHKG.V2OR.VerificationKGAmortization.v2_terminal_five_percent_asymmetric_spend
#print axioms SCOLHKG.V2OR.VerificationKGAmortization.amortized_cost_le_cold_start_iff
