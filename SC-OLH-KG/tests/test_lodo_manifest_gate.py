import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.summarize_lodo_manifest_gate import summarize  # noqa: E402


def row(domain, seed, feasible=True, expert_name="ordered_cumulative"):
    return {
        "_cell": "ordered_iid",
        "heldout": domain,
        "seed": seed,
        "true_feasible": feasible,
        "posterior_feasible": feasible,
        "false_feasible": False,
        "initial_has_true_feasible": True,
        "feasible_simple_regret": 0.1 if feasible else None,
        "constraint_violation": 0.0 if feasible else 0.02,
        "true_chance_margin": -0.1 if feasible else 0.02,
        "algorithm_time_sec": 2.0,
        "wall_time_sec": 3.0,
        "audit_admissible_mainline": True,
        "audit": {
            "admissible_mainline": True,
            "source_oracle_aided": False,
            "source_observation_mode": "replicated",
            "source_simulator_calls": 384,
        },
        "task_posterior": {
            "posterior": {
                "decision_by_expert": {expert_name: 0.4},
                "posterior_by_expert": {expert_name: 0.7},
                "safe_posterior_by_expert": {expert_name: 0.3},
                "posterior_weights": [0.7, 0.3],
                "safe_posterior_weights": [0.3, 0.7],
                "safe_generalized": True,
                "safe_effective_experts": 1.8,
                "last_update": {
                    "safe_pairwise_pairs": 4,
                    "safe_pairwise_effective_weight": 1.5,
                },
            },
            "safe_history_count": 10,
            "task_latent_posterior": {
                "inference_mode": "shadow_joint_generalized_bayes",
                "structure_sensitivity_mutual_information": 0.12,
                "safe_structure_sensitivity_mutual_information": 0.18,
                "legacy_structure_total_variation": 0.07,
            },
            "experts": [{
                "name": expert_name,
                "gpr_adaptive_sparsity": [
                    {"status": "fit", "effective_dimension": 6.0,
                     "max_effective_dimension": 7.0},
                    {"status": "fit", "effective_dimension": 6.5,
                     "max_effective_dimension": 7.0,
                     "shared_shrinkage_groups": {
                         "0": {"posterior_pip": 0.6},
                         "1": {"posterior_pip": 0.8},
                     }},
                ],
                "basis": {
                    "ordered_residual_projection": (
                        {
                            "residual_dim": 6,
                            "orthogonality_relative": 1e-14,
                        }
                        if expert_name == "ordered_semiparametric"
                        else None
                    ),
                },
            }],
        },
        "task_meta_coherence": {
            "status": "audited",
            "joint_and_robust_reference_select_same": True,
            "selected_candidate_expert_support_mass": 0.65,
            "selected_candidate_feasible_mass": 0.75,
            "mean_margin_sign_agreement": 0.8,
            "normalized_margin_disagreement": 0.2,
            "cumulative_hvd_active_mass": 0.9,
        },
        "meta_prior": {
            "ordered_cumulative_exposure": {
                "selected_frequencies": [1, 2],
            },
        },
    }


class LODOManifestGateSummaryTests(unittest.TestCase):
    def test_missing_queue_domain_cannot_complete_current_gate(self):
        rows = [
            row("FactorShockStatePolicyRZDT1", seed)
            for seed in range(7)
        ] + [
            row("InventorySupplyChain", seed)
            for seed in range(7)
        ]
        gate = summarize(rows)["gates"][0]
        self.assertFalse(gate["complete"])
        self.assertFalse(gate["passed"])

    def test_complete_gate_uses_all_three_domains(self):
        rows = [
            row("FactorShockStatePolicyRZDT1", seed)
            for seed in range(7)
        ] + [
            row("InventorySupplyChain", seed)
            for seed in range(7)
        ] + [
            row("QueueResourceControl", seed, feasible=seed < 5)
            for seed in range(7)
        ]
        result = summarize(rows)
        self.assertEqual(len(result["gates"]), 1)
        gate = result["gates"][0]
        self.assertTrue(gate["complete"])
        self.assertTrue(gate["passed"])
        inventory = next(
            summary for summary in result["summaries"]
            if summary["heldout"] == "InventorySupplyChain")
        self.assertEqual(inventory["true_feasible_count"], 7)
        self.assertEqual(inventory["ordered_frequency_counts"], {"1,2": 7})
        self.assertEqual(
            inventory["ordered_constraint_effective_dimension_median"], 6.5)
        self.assertEqual(
            inventory["ordered_constraint_dimension_cap_median"], 7.0)
        self.assertEqual(
            inventory["ordered_curvature_group_pip_median"], 0.6)
        self.assertEqual(
            inventory["ordered_shared_group_pip_median"], 0.8)
        self.assertEqual(
            inventory["ordered_dimension_budget_checked_count"], 7)
        self.assertEqual(
            inventory["ordered_dimension_budget_violation_count"], 0)
        self.assertEqual(inventory["safe_generalized_count"], 7)
        self.assertEqual(inventory["safe_history_count_median"], 10.0)
        self.assertEqual(inventory["safe_pairwise_pairs_median"], 4.0)
        self.assertAlmostEqual(
            inventory["predictive_safe_total_variation_median"], 0.4)
        self.assertEqual(
            inventory["predictive_ordered_expert_weight_median"], 0.7)
        self.assertEqual(
            inventory["safe_ordered_expert_weight_median"], 0.3)
        self.assertEqual(inventory["joint_shadow_count"], 7)
        self.assertEqual(inventory["joint_posterior_count"], 7)
        self.assertEqual(inventory["joint_authoritative_count"], 0)
        self.assertEqual(inventory["coherence_audited_count"], 7)
        self.assertEqual(
            inventory["coherence_joint_reference_agreement_count"], 7)
        self.assertAlmostEqual(
            inventory["joint_mutual_information_median"], 0.12)
        self.assertAlmostEqual(
            inventory["coherence_selected_support_mass_median"], 0.65)
        self.assertAlmostEqual(
            inventory["coherence_cumulative_hvd_active_mass_median"], 0.9)

    def test_any_effective_dimension_violation_rejects_gate(self):
        rows = [
            row("FactorShockStatePolicyRZDT1", seed)
            for seed in range(7)
        ] + [
            row("InventorySupplyChain", seed)
            for seed in range(7)
        ] + [
            row("QueueResourceControl", seed, feasible=seed < 5)
            for seed in range(7)
        ]
        adaptive = rows[0]["task_posterior"]["experts"][0][
            "gpr_adaptive_sparsity"][1]
        adaptive["effective_dimension"] = 7.05
        result = summarize(rows)
        self.assertFalse(result["gates"][0]["passed"])
        factor = next(
            summary for summary in result["summaries"]
            if summary["heldout"] == "FactorShockStatePolicyRZDT1")
        self.assertEqual(
            factor["ordered_dimension_budget_violation_count"], 1)

    def test_source_oracle_aided_rows_cannot_pass_mainline_gate(self):
        rows = [
            row("FactorShockStatePolicyRZDT1", seed)
            for seed in range(7)
        ] + [
            row("InventorySupplyChain", seed)
            for seed in range(7)
        ] + [
            row("QueueResourceControl", seed)
            for seed in range(7)
        ]
        for item in rows:
            item["audit_admissible_mainline"] = False
            item["audit"]["admissible_mainline"] = False
            item["audit"]["source_oracle_aided"] = True
            item["audit"]["uses_source_true_sigma"] = True
        result = summarize(rows)
        self.assertFalse(result["gates"][0]["passed"])
        self.assertEqual(
            result["gates"][0]["factor_source_oracle_aided"], 7)

    def test_group_ridge_uses_nested_selection_validity_not_hard_cap(self):
        rows = [
            row("FactorShockStatePolicyRZDT1", seed)
            for seed in range(7)
        ] + [
            row("InventorySupplyChain", seed)
            for seed in range(7)
        ] + [
            row("QueueResourceControl", seed, feasible=seed < 5)
            for seed in range(7)
        ]
        for item in rows:
            item["_cell"] = (
                "ordered_latent_structure_group_ridge_"
                "diag_sparse_add_iid"
            )
            adaptive = item["task_posterior"]["experts"][0][
                "gpr_adaptive_sparsity"][1]
            adaptive.update({
                "method": "nested_loo_group_ridge",
                "effective_dimension": 8.0,
                "max_effective_dimension": None,
                "complexity_selection_valid": True,
                "groups": {
                    "0": {"selected_penalty": 0.01},
                    "1": {"selected_penalty": 0.01},
                    "2": {"selected_penalty": 100.0},
                },
            })
        result = summarize(rows)
        self.assertTrue(result["gates"][0]["passed"])
        inventory = next(
            summary for summary in result["summaries"]
            if summary["heldout"] == "InventorySupplyChain")
        self.assertEqual(
            inventory["ordered_dimension_budget_checked_count"], 0)
        self.assertEqual(
            inventory["ordered_complexity_selection_checked_count"], 7)
        self.assertEqual(
            inventory["ordered_complexity_selection_invalid_count"], 0)
        self.assertEqual(
            inventory["ordered_curvature_group_penalty_median"], 0.01)

    def test_semiparametric_ordered_expert_diagnostics_are_summarized(self):
        rows = [
            row(
                "FactorShockStatePolicyRZDT1",
                seed,
                expert_name="ordered_semiparametric",
            )
            for seed in range(7)
        ] + [
            row(
                "InventorySupplyChain",
                seed,
                feasible=seed < 4,
                expert_name="ordered_semiparametric",
            )
            for seed in range(7)
        ]
        for item in rows:
            item["_cell"] = "ordered_semiparametric_diag_sparse_replace_iid"
            adaptive = item["task_posterior"]["experts"][0][
                "gpr_adaptive_sparsity"][1]
            adaptive["posterior_pip"] = [1.0] * 4 + [0.1] * 7 + [0.2] * 6
        result = summarize(rows)
        inventory = next(
            summary for summary in result["summaries"]
            if summary["heldout"] == "InventorySupplyChain")
        self.assertAlmostEqual(
            inventory["ordered_local_inclusion_mass_median"], 1.2)
        self.assertEqual(
            inventory["ordered_projection_error_median"], 1e-14)


if __name__ == "__main__":
    unittest.main()
