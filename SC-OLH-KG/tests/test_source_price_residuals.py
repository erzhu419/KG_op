import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.inventory_cost_plan import InventoryPlan
from representation.source_price_residuals import source_price_residual_covariance, pooled_residual_covariance


class SourceResidualTests(unittest.TestCase):
    def test_public_offsets_and_quantity_scaling_leave_price_covariance_unchanged(self):
        rng = np.random.default_rng(10056)
        q = InventoryPlan().coordinates_many(rng.uniform(.05, .95, (24, 4)))[:, [3, 5, 4, 6]]
        offsets = np.linspace(-1.3, -.8, len(q))
        prices = rng.uniform(.4, .9, (8, 4))
        labels = [q @ p + offsets for p in prices]
        noises = [np.full(len(q), .003)] * len(prices)
        covariance, info = source_price_residual_covariance(q, labels, noises, offsets, (.4, .9))
        shifted, _ = source_price_residual_covariance(q, [y + 10 for y in labels], noises, offsets + 10, (.4, .9))
        scaled, _ = source_price_residual_covariance(3 * q, [3 * y for y in labels],
            [9 * n for n in noises], 3 * offsets, (.4, .9))
        np.testing.assert_allclose(covariance, shifted, atol=1e-14)
        np.testing.assert_allclose(covariance, scaled, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0)
        self.assertAlmostEqual(np.trace(covariance), np.trace(pooled_residual_covariance(covariance)))
        self.assertEqual(info["target_residual_mean"], [0.] * 4)

    def test_source_coefficient_uncertainty_remains_when_point_estimates_agree(self):
        design = np.eye(4)
        labels = [np.full(4, .65)] * 8
        covariance, _ = source_price_residual_covariance(design, labels, [np.full(4, .1)] * 8, np.zeros(4), (.4, .9))
        parent = .5 ** 2 / 12
        conditional = 1 / (1 / parent + 1 / .1)
        np.testing.assert_allclose(covariance, np.eye(4) * (parent + 8 * conditional) / 9)


if __name__ == "__main__":
    unittest.main()
