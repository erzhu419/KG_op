import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.diagnose_v33_frontier_repair import diagnose  # noqa: E402


class V33FrontierRepairDiagnosisTests(unittest.TestCase):
    def test_posthoc_diagnosis_separates_support_and_ranking_failure(self):
        feasible = [55, 55, 35, 35, 45, 45]
        unsafe = [35, 35, 25, 25, 30, 30]
        variants = (
            "v32",
            "v33_legacy_4",
            "v33_coherent_coverage_4",
            "v33_coherent_coverage_8",
        )
        config = {
            "d": 6,
            "L": 100,
            "sigma": 0.04,
            "alpha": 0.05,
            "weights": "0.5,0.5",
        }

        def row(variant):
            coherent = variant.startswith("v33_coherent")
            return {
                "x_recommended": unsafe if coherent else feasible,
                "posterior_feasible": False,
                "posterior_theory_chance_margin": 0.4,
                "finalist_replication": {
                    "targets": [feasible, unsafe],
                    "labels": ["safe", "unsafe"],
                    "target_oracle_used": False,
                    "terminal_kg_rows": ([{
                        "terminal_kg_selected_index": 1,
                        "terminal_kg_arms": [feasible, unsafe],
                    }] if coherent else []),
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for variant in variants:
                path = (
                    root / variant / "InventorySupplyChain" / "seed0"
                    / "result.json"
                )
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "config": config,
                    "rows": [row(variant)],
                }))
            result = diagnose(
                root,
                variants=variants,
                domains=("InventorySupplyChain",),
                seeds=(0,),
            )

        challenger = next(
            item for item in result["summaries"]
            if item["variant"] == "v33_coherent_coverage_8"
        )
        self.assertTrue(result["posthoc_truth_audit"])
        self.assertFalse(result["truth_used_by_optimizer"])
        self.assertEqual(challenger["target_sets_with_true_feasible"], 1)
        self.assertEqual(challenger["support_failure_count"], 0)
        self.assertEqual(challenger["terminal_misranking_count"], 1)
        self.assertEqual(challenger["final_ranking_failure_count"], 1)
        self.assertEqual(challenger["certificate_vacuity_count"], 1)


if __name__ == "__main__":
    unittest.main()
