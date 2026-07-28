import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.summarize_boundary_coordinate_screen import (  # noqa: E402
    BASELINE_VARIANT_ID,
    _source_gate,
)


DOMAINS = ["a", "b", "c", "d", "e"]


def result(rank_losses, *, false_safe=0.0, collapse=False):
    return {
        "aggregate": {
            "all_finite": True,
            "worst_false_safe_rate": false_safe,
            "mean_false_safe_rate": false_safe,
            "median_rank_loss": 0.2,
            "median_boundary_rmse": 1.0,
            "minimum_upper_coverage": 0.9,
            "adaptation_dimension_admissible": True,
            "single_domain_collapse": collapse,
        },
        "protocol": {
            "target_evaluation_used_for_fit": False,
            "target_oracle_used": False,
        },
        "folds": [
            {
                "heldout": domain,
                "rank_loss": loss,
                "evaluation_feasible_rate": 0.5,
                "false_safe_rate": false_safe,
                "false_unsafe_rate": (
                    0.5 if collapse else 0.3),
            }
            for domain, loss in zip(DOMAINS, rank_losses)
        ],
    }


class BoundaryScreenSummaryTests(unittest.TestCase):
    def test_source_gate_requires_four_of_five_rank_wins(self):
        baseline = result([0.5] * 5, collapse=True)
        four_wins = result([0.4, 0.4, 0.4, 0.4, 0.6])
        three_wins = result([0.4, 0.4, 0.4, 0.6, 0.6])
        self.assertTrue(_source_gate(four_wins, baseline)["passed"])
        self.assertFalse(_source_gate(three_wins, baseline)["passed"])

    def test_source_gate_rejects_false_safe_increase_and_collapse(self):
        baseline = result([0.5] * 5, false_safe=0.0, collapse=True)
        unsafe = result([0.4] * 5, false_safe=0.01)
        collapsed = result([0.4] * 5, collapse=True)
        self.assertFalse(_source_gate(unsafe, baseline)["passed"])
        self.assertFalse(_source_gate(collapsed, baseline)["passed"])

    def test_baseline_identifier_is_frozen(self):
        self.assertEqual(
            BASELINE_VARIANT_ID,
            "learned_psi__linear_monotone__frozen__r2",
        )


if __name__ == "__main__":
    unittest.main()
