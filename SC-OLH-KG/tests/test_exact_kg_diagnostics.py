import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.summarize_exact_kg_diagnostics import (  # noqa: E402
    spearman,
    summarize,
    top_k_overlap,
)


def audit(seed, reference_scores, challenger_scores):
    def row(mc, scores):
        selected = max(range(len(scores)), key=scores.__getitem__)
        feasible_index = len(scores) - 1
        order = sorted(range(len(scores)), key=lambda index: -scores[index])
        feasible_rank = order.index(feasible_index) + 1
        return {
            "sampling_mode": "iid",
            "mc_samples": mc,
            "selected_index": selected,
            "selected_source": "test",
            "selected_true_feasible": selected == feasible_index,
            "selected_true_margin": -0.1 if selected == feasible_index else 0.1,
            "raw_scores": scores,
            "raw_negative_fraction": 0.0,
            "elapsed_sec": float(mc),
            "highest_score_true_feasible_rank": feasible_rank,
            "best_true_feasible_rank": feasible_rank,
            "selected_minus_highest_feasible_score": (
                scores[selected] - scores[feasible_index]),
        }

    return {
        "offline_only": True,
        "simulator_calls": 0,
        "seed": seed,
        "candidate_table": [
            {
                "index": index,
                "true_chance_margin": -0.1 if index == 3 else 0.1,
                "true_objective": float(index),
            }
            for index in range(4)
        ],
        "rows": [row(2, reference_scores), row(8, challenger_scores)],
    }


class ExactKGDiagnosticSummaryTests(unittest.TestCase):
    def test_rank_metrics(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertAlmostEqual(top_k_overlap([3, 2, 1], [3, 1, 2], 2), 0.5)

    def test_predeclared_estimator_noise_verdict(self):
        audits = [
            audit(seed, [4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0])
            for seed in range(3)
        ]
        result = summarize(audits)
        self.assertEqual(result["predeclared_verdict"], "estimator_noise_primary")
        self.assertEqual(result["best_challenger"], {
            "sampling_mode": "iid",
            "mc_samples": 8,
        })
        challenger = next(
            row for row in result["variants"] if row["mc_samples"] == 8)
        self.assertEqual(challenger["selected_true_feasible_row_count"], 3)
        self.assertEqual(challenger["near_top_safe_count"], 3)


if __name__ == "__main__":
    unittest.main()
