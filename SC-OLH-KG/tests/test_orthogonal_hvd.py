import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.rzdt import RegimeRZDT1  # noqa: E402
from variance.orthogonal_hvd import OrthogonalHVD  # noqa: E402


class OrthogonalHVDTests(unittest.TestCase):
    def setUp(self):
        self.problem = RegimeRZDT1(d=3, L=90, sigma=0.05)

    def test_variance_is_nonnegative_all_modes(self):
        X = [(0, 0, 0), (10, 0, 0), (40, 0, 0), (70, 0, 0), (90, 0, 0)]
        residuals = [0.01, 0.02, 0.08, 0.09, 0.15]
        for mode in ["pooled", "class", "orthogonal", "factor"]:
            model = OrthogonalHVD(mode=mode, n_outputs=1)
            model.fit_from_residuals(X, residuals, 0, self.problem)
            for x in X:
                self.assertGreaterEqual(model.predict_variance(0, x, self.problem), 0.0)
            self.assertEqual(model.diagnostics()["mode"], mode)

    def test_class_hvd_recovers_low_high_regime_order(self):
        X = []
        residuals = []
        for x1, resid in [(5, 0.01), (10, 0.015), (20, 0.02),
                          (70, 0.20), (80, 0.22), (90, 0.24)]:
            X.append((x1, 0, 0))
            residuals.append(resid)
        model = OrthogonalHVD(mode="class", n_outputs=1, shrinkage_kappa=0.0)
        model.fit_from_residuals(X, residuals, 0, self.problem)
        low = model.predict_variance(0, (10, 0, 0), self.problem)
        high = model.predict_variance(0, (80, 0, 0), self.problem)
        self.assertGreater(high, low)
        self.assertGreater(high / max(low, 1e-12), 10.0)

    def test_update_returns_diagnostics(self):
        model = OrthogonalHVD(mode="class", n_outputs=1)
        detail = model.update(0, np.array([10, 0, 0]), 1.0, 0.8, problem=self.problem)
        self.assertEqual(detail["mode"], "class")
        self.assertIn("new_variance", detail)
        self.assertGreaterEqual(detail["new_variance"], 0.0)


if __name__ == "__main__":
    unittest.main()
