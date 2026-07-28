import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from performance.benchmark_tcb_v2_source_gate import (  # noqa: E402
    build_parser,
    run_gate,
)
from performance.summarize_tcb_v2_source_gate import summarize  # noqa: E402


SUBMIT_SCRIPT = REPO / "scripts/submit_scolhkg_tcb_v2_source_gate_scheduler.py"
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "tcb_v2_source_submit", SUBMIT_SCRIPT)
SUBMIT = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(SUBMIT)


class TCBV2SourceGateTests(unittest.TestCase):
    def _tiny_args(self, output):
        args = build_parser().parse_args([
            "--domains",
            (
                "RZDT1,StatePolicyRZDT1,InventorySupplyChain,"
                "QueueResourceControl"
            ),
            "--heldout", "InventorySupplyChain",
            "--target-seed", "3",
            "--source-records", "10",
            "--source-replicates", "2",
            "--pilot-pool", "14",
            "--evaluation-pool", "18",
            "--n0", "4",
            "--target-replicates", "2",
            "--pilot-policies", "random,source_boundary",
            "--coordinate", "boundary_latent",
            "--geometry", "linear_monotone",
            "--rank", "1",
            "--hierarchy-iterations", "2",
            "--out", str(output),
        ])
        args.domains = (
            "RZDT1",
            "StatePolicyRZDT1",
            "InventorySupplyChain",
            "QueueResourceControl",
        )
        args.pilot_policies = ("random", "source_boundary")
        return args

    def test_gate_uses_only_noisy_target_pilots_for_two_dimensional_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._tiny_args(Path(directory) / "result.json")
            result = run_gate(args, "InventorySupplyChain", 3)
        self.assertEqual(
            result["model"]["model_version"],
            "tcb_v2_hierarchical_signed_distance",
        )
        self.assertEqual(len(result["rows"]), 2)
        for row in result["rows"]:
            self.assertFalse(row["target_oracle_used_for_fit"])
            self.assertFalse(
                row["target_oracle_used_for_hyperparameter_selection"])
            self.assertTrue(row["evaluation_oracle_used_after_fit"])
            self.assertEqual(row["adapter_effective_dimension"], 2)
            self.assertEqual(row["target_evaluations_used"], 8)
            self.assertEqual(row["metrics"]["evaluation_count"], 18)
            self.assertEqual(row["source_inner_lodo"]["fold_count"], 3)
            self.assertTrue(row["source_inner_lodo"][
                "target_domain_excluded_from_training_and_selection"])

    def test_summary_promotes_only_complete_nonvacuous_safe_gate(self):
        row = {
            "descriptor_mode": "learned_risk",
            "coordinate": "boundary_latent",
            "geometry": "linear_monotone",
            "rank": 1,
            "ridge": 1e-3,
            "domain_penalty": 0.5,
            "adaptation_ridge": 1.0,
            "effect_ridge": 1.0,
            "rotation_mode": "none",
            "rotation_ridge": 5.0,
            "target_residual_rank": 0,
            "residual_ridge": 5.0,
            "upper_alpha": 0.01,
            "pilot_policy": "random",
            "heldout": "RZDT1",
            "target_seed": 0,
            "adapter_effective_dimension": 2,
            "target_oracle_used_for_fit": False,
            "target_oracle_used_for_hyperparameter_selection": False,
            "rank_improved": True,
            "false_safe_nonworse": True,
            "nonvacuous_safe_set": True,
            "metrics": {
                "evaluation_count": 100,
                "coverage_count": 98,
                "predicted_safe_count": 20,
                "false_safe_count": 0,
                "spearman": 0.8,
                "safe_recall": 0.7,
                "boundary_mae": 0.1,
            },
            "frozen_metrics": {
                "predicted_safe_count": 10,
                "false_safe_count": 1,
                "spearman": 0.4,
            },
            "source_inner_lodo": {
                "fold_count": 2,
                "coverage_rate": 0.98,
                "predicted_safe_count": 20,
                "false_safe_count": 0,
                "false_safe_conditional_rate": 0.0,
                "frozen_predicted_safe_count": 10,
                "frozen_false_safe_count": 1,
                "frozen_false_safe_conditional_rate": 0.1,
                "mean_spearman": 0.8,
                "frozen_mean_spearman": 0.4,
                "rank_win_rate": 1.0,
                "nonvacuous_rate": 1.0,
                "target_domain_excluded_from_training_and_selection": True,
                "target_oracle_used": False,
            },
        }
        args = SimpleNamespace(
            expected_domains=1,
            expected_seeds=1,
            minimum_coverage=0.95,
            false_safe_tolerance=0.01,
            minimum_spearman=0.35,
            minimum_nonvacuous_rate=0.50,
        )
        second = copy.deepcopy(row)
        second["pilot_policy"] = "source_boundary"
        result = summarize([row, second], args)
        self.assertTrue(result["gate_pass"])
        self.assertIsNotNone(result["promoted_candidate"])

        provider = copy.deepcopy(row)
        provider["descriptor_mode"] = "provider_risk"
        provider_second = copy.deepcopy(provider)
        provider_second["pilot_policy"] = "source_boundary"
        provider_result = summarize([provider, provider_second], args)
        self.assertFalse(provider_result["gate_pass"])
        self.assertFalse(provider_result["groups"][0]["checks"][
            "strict_lodo_descriptor"])

    def test_focused_source_gate_is_five_auditable_configurations(self):
        args = SimpleNamespace(preset="focused_v2")
        grid = SUBMIT.configuration_grid(args)
        self.assertEqual(len(grid), 5)
        self.assertTrue(all(
            "provider_" not in row["descriptor_mode"] for row in grid
        ))
        self.assertEqual(
            {row["rotation_mode"] for row in grid}, {"none", "planar"})
        self.assertEqual(
            {row["target_residual_rank"] for row in grid}, {0, 1})


if __name__ == "__main__":
    unittest.main()
