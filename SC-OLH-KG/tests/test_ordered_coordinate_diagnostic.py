import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.diagnose_ordered_coordinate import (  # noqa: E402
    _auc,
    _fit_decomposed_audit,
    _fit_audit,
    _ordered_group_feature_maps,
)
from core.cumulative_risk import RiskExposure  # noqa: E402


class OrderedCoordinateDiagnosticTests(unittest.TestCase):
    def test_auc_and_ridge_audit(self):
        self.assertAlmostEqual(
            _auc([False, False, True, True], [0.0, 1.0, 2.0, 3.0]),
            1.0,
        )
        x = np.linspace(-1.0, 1.0, 30)[:, None]
        features = np.column_stack([x[:, 0], x[:, 0] ** 2])
        margin = 0.5 * x[:, 0] + 0.2 * x[:, 0] ** 2 - 0.1
        result = _fit_audit(
            features,
            margin,
            np.arange(18),
            np.arange(18, 24),
            np.arange(24, 30),
        )
        self.assertGreater(result["test_r2"], 0.99)
        self.assertEqual(result["feature_dim"], 2)

    def test_group_maps_separate_invariant_and_semantic_features(self):
        exposures = [
            RiskExposure([1.0, 2.0, 3.0], [4.0, 5.0]),
            RiskExposure([3.0, 1.0, 2.0], [5.0, 4.0]),
        ]
        maps = _ordered_group_feature_maps(exposures)
        self.assertEqual(maps["ordered_fully_invariant"].shape, (2, 4))
        self.assertEqual(maps["ordered_curvature_grouped"].shape, (2, 6))
        self.assertEqual(maps["ordered_shared_grouped"].shape, (2, 8))
        self.assertEqual(maps["ordered_both_grouped"].shape, (2, 6))
        np.testing.assert_allclose(
            maps["ordered_fully_invariant"][0],
            maps["ordered_fully_invariant"][1],
        )
        np.testing.assert_allclose(
            maps["ordered_both_grouped"][:, -2:],
            np.repeat(
                maps["ordered_both_grouped"][0:1, -2:], 2, axis=0),
        )

    def test_decomposed_audit_reconstructs_mean_plus_risk_buffer(self):
        x = np.linspace(-1.0, 1.0, 60)
        mean_features = np.column_stack([x, np.ones_like(x)])
        variance_features = np.column_stack([x ** 2, np.ones_like(x)])
        mean_margin = 0.2 * x - 0.1
        variance = 0.02 + 0.01 * x ** 2
        result = _fit_decomposed_audit(
            mean_features,
            variance_features,
            mean_margin,
            variance,
            1.64,
            np.arange(36),
            np.arange(36, 48),
            np.arange(48, 60),
        )
        self.assertGreater(result["test_mean_r2"], 0.99)
        self.assertGreater(result["test_variance_r2"], 0.99)
        self.assertGreater(result["test_chance_r2"], 0.99)


if __name__ == "__main__":
    unittest.main()
