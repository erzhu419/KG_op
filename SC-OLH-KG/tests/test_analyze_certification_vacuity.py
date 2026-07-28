import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from performance.analyze_certification_vacuity import (
    oracle_variance_certifiability,
    summarize,
)


class CertificationVacuityAnalysisTests(unittest.TestCase):
    def test_oracle_variance_audit_separates_epistemic_blocker(self):
        row = {
            "heldout": "InventorySupplyChain",
            "recommendation_best_true_feasible_x": [14, 14, 15, 9, 12],
            "recommendation_best_true_feasible_mu_con": -0.20,
            "recommendation_best_true_feasible_epistemic_var": 0.04,
        }
        audit = oracle_variance_certifiability(row, {
            "d": 5,
            "L": 30,
            "sigma": 0.04,
            "alpha": 0.05,
            "weights": "0.5,0.5",
            "beta_g": 2.0,
        })
        self.assertEqual(audit["status"], "audited")
        self.assertTrue(audit["mean_aleatoric_certified"])
        self.assertFalse(audit["oracle_variance_certified"])
        self.assertTrue(audit["target_oracle_used_for_post_run_audit"])
        self.assertFalse(audit["target_oracle_used_for_decision"])
        self.assertIn("posterior_mean_bias", audit)
        self.assertIn("true_chance_margin", audit)
        self.assertIn("oracle_mean_with_epistemic_margin", audit)
        self.assertIn("epistemic_variance_contraction_factor", audit)
        self.assertGreater(audit["epistemic_variance_contraction_factor"], 1.0)

    def test_summary_identifies_expert_guard_as_vacuity_layer(self):
        row = {
            "heldout": "InventorySupplyChain",
            "hvd_ablation_profile": "factor_cumulative",
            "source_discrepancy_update": True,
            "decision_backend": "sobol_new",
            "experiment_variant": "gate/certificate_depth_search/joint_tangent",
            "target_shared_shock_scale": 1.0,
            "truth_pool_diagnostics": {
                "mean_pool_min_true_margin": -0.2,
                "mean_pool_min_posterior_margin": 0.3,
                "mean_selected_true_margin": 0.1,
            },
            "certification_margin_decomposition": {
                "n_certified": 0,
                "minimum_margin": {
                    "observation_nominal": {
                        "margin": -0.2,
                    },
                    "expert_certified": {
                        "margin": 0.1,
                    },
                    "task_robust": {
                        "margin": 0.2,
                    },
                    "final_certificate": {
                        "margin": 0.2,
                        "mean_minus_tau": -0.5,
                        "epistemic_radius": 0.2,
                        "aleatoric_radius": 0.5,
                    },
                    "expert_to_observation_aleatoric_ratio": 4.0,
                    "robust_to_expert_aleatoric_ratio": 1.5,
                },
            },
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(row))
            result = summarize([path])[0]
        self.assertEqual(result["nominal_safe_expert_unsafe"], 1)
        self.assertEqual(result["expert_safe_robust_unsafe"], 0)
        self.assertEqual(result["median_expert_observation_aleatoric_ratio"], 4.0)
        self.assertEqual(result["median_pool_min_true_margin"], -0.2)
        self.assertEqual(result["median_pool_min_posterior_margin"], 0.3)
        self.assertEqual(result["median_selected_true_margin"], 0.1)
        self.assertEqual(result["backend"], "certificate_depth_search")

    def test_summary_separates_adaptive_source_mean_mixture(self):
        row = {
            "heldout": "QueueResourceControl",
            "hvd_ablation_profile": "factor_hierarchical",
            "meta_observable_mean_coordinate": True,
            "source_constraint_mean_coefficient_prior": True,
            "source_constraint_mean_adaptation_mode": "evidence_mixture",
            "meta_observable_mean_mode": "consensus",
            "hvd_source_task_weight_mode": "constraint_mean",
            "source_discrepancy_update": True,
            "decision_backend": "sobol_new",
            "target_shared_shock_scale": 1.0,
            "gpr_numerics": [{}, {
                "source_parametric_prior": {
                    "adaptation_mode": "target_evidence_mixture",
                    "target_only_posterior_weight": 0.75,
                    "source_posterior_weight": 0.25,
                    "selected_component": "target:null",
                },
            }],
            "variance_diagnostics": {
                "cumulative_prior_component_domains": {
                    "1": ["source_a", "target:null"],
                },
                "cumulative_prior_component_weights": {
                    "1": [0.25, 0.75],
                },
            },
            "certification_margin_decomposition": {
                "n_certified": 0,
                "minimum_margin": {},
            },
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(row))
            result = summarize([path])[0]
        self.assertEqual(result["mean_profile"], "eta_source_adaptive")
        self.assertEqual(result["observable_mean_mode"], "consensus")
        self.assertEqual(result["hvd_task_weight_mode"], "constraint_mean")
        self.assertEqual(result["source_mean_mixture_count"], 1)
        self.assertEqual(result["median_target_only_posterior_weight"], 0.75)
        self.assertEqual(result["median_source_posterior_weight"], 0.25)
        self.assertEqual(result["target_only_selected_count"], 1)
        self.assertEqual(result["median_hvd_target_null_mass"], 0.75)

    def test_summary_labels_sequential_source_mean_mixture(self):
        row = {
            "heldout": "QueueResourceControl",
            "hvd_ablation_profile": "factor_hierarchical",
            "meta_observable_mean_coordinate": True,
            "source_constraint_mean_coefficient_prior": True,
            "source_constraint_mean_adaptation_mode": (
                "sequential_evidence_mixture"),
            "meta_observable_mean_mode": "latent",
            "hvd_source_task_weight_mode": "constraint_mean",
            "source_discrepancy_update": True,
            "decision_backend": "sobol_new",
            "target_shared_shock_scale": 1.0,
            "gpr_numerics": [{}, {
                "source_parametric_prior": {
                    "adaptation_mode": (
                        "sequential_target_evidence_mixture"),
                    "target_only_posterior_weight": 0.4,
                    "source_posterior_weight": 0.6,
                    "selected_component": "source:inventory",
                },
            }],
            "certification_margin_decomposition": {
                "n_certified": 0,
                "minimum_margin": {},
            },
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(row))
            result = summarize([path])[0]
        self.assertEqual(result["mean_profile"], "eta_source_sequential")
        self.assertEqual(result["source_mean_mixture_count"], 1)
        self.assertEqual(result["median_target_only_posterior_weight"], 0.4)
        self.assertEqual(result["median_source_posterior_weight"], 0.6)

    def test_summary_separates_prequential_target_shape_evidence(self):
        common = {
            "heldout": "QueueResourceControl",
            "hvd_ablation_profile": "factor_hierarchical",
            "meta_observable_mean_coordinate": True,
            "source_constraint_mean_coefficient_prior": True,
            "source_constraint_mean_adaptation_mode": "evidence_mixture",
            "meta_observable_mean_mode": "latent",
            "hvd_source_task_weight_mode": "constraint_mean",
            "source_discrepancy_update": True,
            "decision_backend": "sobol_new",
            "target_shared_shock_scale": 1.0,
            "certification_margin_decomposition": {
                "n_certified": 0,
                "minimum_margin": {},
            },
        }
        control = dict(common)
        control.update({
            "hvd_cumulative_target_evidence_mode": "replication_only",
            "variance_diagnostics": {
                "cumulative_prior_shape_target_dof": {"1": 0.0},
                "prequential_upper_solution_count": {"1": 0},
                "cumulative_prior_target_weight": {"1": 0},
                "cumulative_prior_scale_source": {
                    "1": "source_prior_fallback",
                },
            },
        })
        challenger = dict(common)
        challenger.update({
            "hvd_cumulative_target_evidence_mode": "prequential_upper",
            "variance_diagnostics": {
                "cumulative_prior_shape_target_dof": {"1": 10.0},
                "prequential_upper_solution_count": {"1": 10},
                "cumulative_prior_target_weight": {"1": 10},
                "cumulative_prior_scale_source": {"1": "prequential_upper"},
            },
        })
        with TemporaryDirectory() as directory:
            root = Path(directory)
            control_path = root / "control.json"
            challenger_path = root / "challenger.json"
            control_path.write_text(json.dumps(control))
            challenger_path.write_text(json.dumps(challenger))
            results = summarize([control_path, challenger_path])

        self.assertEqual(len(results), 2)
        by_mode = {
            result["hvd_target_evidence_mode"]: result for result in results
        }
        self.assertEqual(
            by_mode["replication_only"]["median_hvd_target_shape_dof"], 0.0)
        self.assertEqual(
            by_mode["prequential_upper"]["median_hvd_target_shape_dof"], 10.0)
        self.assertEqual(
            by_mode["prequential_upper"][
                "median_prequential_upper_solution_count"],
            10.0,
        )
        self.assertEqual(
            by_mode["prequential_upper"]["median_hvd_target_weight"], 10.0)
        self.assertEqual(
            by_mode["prequential_upper"]["hvd_scale_sources"],
            ["prequential_upper"],
        )


if __name__ == "__main__":
    unittest.main()
