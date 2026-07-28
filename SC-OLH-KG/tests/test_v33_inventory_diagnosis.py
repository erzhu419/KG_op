import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.diagnose_v33_inventory_terminal import diagnose  # noqa: E402


class V33InventoryDiagnosisTests(unittest.TestCase):
    def test_posthoc_audit_separates_support_and_certificate_failure(self):
        feasible = [55, 55, 35, 35, 45, 45]
        unsafe = [35, 35, 25, 25, 30, 30]
        config = {
            "d": 6,
            "L": 100,
            "sigma": 0.04,
            "alpha": 0.05,
            "weights": "0.5,0.5",
        }

        def result_row(variant):
            terminal = variant.startswith("terminal_kg")
            return {
                "x_recommended": unsafe if terminal else feasible,
                "posterior_feasible": False,
                "posterior_theory_chance_margin": 0.4,
                "replicated_finalist_used": variant == "v32",
                "replicated_finalist_reason": "fixture",
                "truth_pool_diagnostics": {
                    "pool_has_true_feasible_rate": 1.0,
                    "best_true_feasible_posterior_feasible_rate": 0.0,
                    "mean_best_true_feasible_posterior_margin": 0.4,
                    "missed_true_feasible_rate": 1.0,
                },
                "finalist_replication": {
                    "targets": [feasible, unsafe],
                    "labels": ["safe", "unsafe"],
                    "replicate_counts": [2, 1],
                    "statistics": [
                        {
                            "upper_chance_margin": 0.3,
                            "constraint_mean": -0.05,
                        },
                        {
                            "upper_chance_margin": 0.6,
                            "constraint_mean": 0.1,
                        },
                    ],
                    "terminal_kg_rows": ([{
                        "terminal_kg_selected_index": 1,
                        "terminal_kg_arms": [feasible, unsafe],
                        "terminal_kg_depth": 1,
                        "terminal_kg_selected_gain": 0.1,
                    }] if terminal else []),
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for variant in ("v32", "terminal_kg_1step", "terminal_kg_depth3"):
                path = (
                    root / variant / "InventorySupplyChain" / "seed0"
                    / "result.json"
                )
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "config": config,
                    "rows": [result_row(variant)],
                }))
            payload = diagnose(root, seeds=[0])

        self.assertTrue(payload["posthoc_truth_audit"])
        self.assertFalse(payload["truth_used_by_optimizer"])
        self.assertFalse(
            payload["mechanism"]["candidate_pool_support_failure"])
        self.assertTrue(payload["mechanism"]["certificate_vacuity"])
        self.assertTrue(payload["mechanism"]["terminal_action_misranking"])
        self.assertEqual(
            payload["paired_failure"]["v32_uncertified_override_rescue_seeds"],
            [0],
        )


if __name__ == "__main__":
    unittest.main()
