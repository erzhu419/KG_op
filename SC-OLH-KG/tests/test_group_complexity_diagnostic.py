import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.diagnose_group_complexity_checkpoint import (  # noqa: E402
    _effective_degrees_of_freedom,
    _fit_family,
    _group_penalties,
)


class GroupComplexityDiagnosticTests(unittest.TestCase):
    def test_group_penalty_grid_has_one_penalty_per_feature(self):
        rows = _group_penalties([2, 3], ridges=(0.1, 1.0))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(len(penalty) == 5 for _, penalty in rows))
        np.testing.assert_allclose(rows[0][1], [0.1] * 5)
        np.testing.assert_allclose(rows[-1][1], [1.0] * 5)

    def test_stronger_ridge_reduces_effective_degrees_of_freedom(self):
        rng = np.random.default_rng(271)
        X = rng.normal(size=(20, 6))
        weak = _effective_degrees_of_freedom(X, np.full(6, 0.01))
        strong = _effective_degrees_of_freedom(X, np.full(6, 100.0))
        self.assertGreater(weak, strong)
        self.assertGreaterEqual(strong, 0.0)
        self.assertLessEqual(weak, 6.0 + 1e-10)

    def test_nested_loo_family_reports_admissible_and_oracle_rows(self):
        rng = np.random.default_rng(272)
        train_x = rng.normal(size=(16, 3))
        train_y = 1.5 * train_x[:, 0] + 0.05 * rng.normal(size=16)
        test_x = rng.normal(size=(40, 3))
        true_mean = 1.5 * test_x[:, 0]
        result = _fit_family(
            train_x,
            train_y,
            test_x,
            _group_penalties([3], ridges=(0.01, 1.0, 100.0)),
            true_mean,
            np.full(len(test_x), 0.05),
            tau=0.0,
            z_value=1.64,
        )
        self.assertEqual(result["n_penalty_models"], 3)
        self.assertFalse(result["oracle_used_for_selection"])
        self.assertIn("loo_loss", result["selected_by_nested_loo"])
        self.assertIn("oracle_loss", result["oracle_best_after_freeze"])


if __name__ == "__main__":
    unittest.main()
