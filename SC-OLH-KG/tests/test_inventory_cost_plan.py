import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.special import digamma, polygamma

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cumulative_risk import cumulative_feature_vector
from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan, sample_cost_law


class InventoryCostPlanTests(unittest.TestCase):
    def test_material_balance_and_feedback_use_current_stock(self):
        plan = InventoryPlan(demand=(3., 9., 5.), capacity=10., initial_stock=6.)
        trajectory = plan.trajectory([.8, .2, .5, 1.])
        np.testing.assert_allclose(trajectory.stock_before, [6., 3., 0.])
        np.testing.assert_allclose(trajectory.orders, [0., 0., 8.])
        np.testing.assert_allclose(trajectory.lost_sales, [0., 6., 0.])
        np.testing.assert_allclose(trajectory.stock_after, [3., 0., 3.])
        np.testing.assert_allclose(trajectory.stock_before + trajectory.orders,
                                  np.array(plan.demand) - trajectory.lost_sales + trajectory.stock_after)
        self.assertEqual(trajectory.coordinates(plan.capacity).shape, (8,))

    def test_public_coordinates_do_not_use_unknown_prices_or_risk(self):
        plan = InventoryPlan()
        policies = np.random.default_rng(402).uniform(size=(10, 4))
        worlds = [InventoryCostSimulator(plan, sample_cost_law(s)) for s in (10, 11)]
        np.testing.assert_array_equal(worlds[0].plan.coordinates_many(policies),
                                      worlds[1].plan.coordinates_many(policies))
        self.assertFalse(np.array_equal(worlds[0].oracle_moments(policies[0]),
                                       worlds[1].oracle_moments(policies[0])))
        self.assertEqual([w.episode_calls for w in worlds], [0, 0])

    def test_cumulative_risk_matches_independent_episode_shocks(self):
        world = InventoryCostSimulator(InventoryPlan(), sample_cost_law(403))
        policy = np.array([.85, .25, .55, .75])
        trajectory = world.plan.trajectory(policy)
        exposure, risk = trajectory.risk_exposures(), world._law.risk
        coefficients = np.r_[risk.floor, risk.Lambda, risk.B[0, 0], risk.B[0, 1],
                             risk.B[1, 1], risk.omega]
        truth = world.oracle_moments(policy)
        self.assertAlmostEqual(float(cumulative_feature_vector(exposure) @ coefficients), np.exp(truth[2]))
        # This local distribution check detects misplaced shared shocks and
        # linear/quadratic exposure errors; it is not a performance experiment.
        episodes = world.episodes(policy, 20000, np.random.default_rng(404))
        np.testing.assert_allclose(episodes.mean(axis=0), truth[:2], atol=.008)
        self.assertAlmostEqual(float(np.var(episodes[:, 1], ddof=1)), np.exp(truth[2]),
                               delta=.04 * np.exp(truth[2]))

    def test_risk_label_uses_paid_sample_variance_and_counts_replicates(self):
        world = InventoryCostSimulator(InventoryPlan(), sample_cost_law(405))
        policy = [.5] * 4
        rng = np.random.default_rng(406)
        reference = InventoryCostSimulator(world.plan, world._law).episodes(policy, 12, rng)
        with patch.object(world, "oracle_moments", side_effect=AssertionError("oracle label leak")):
            response, noise = world.observe(policy, 12, np.random.default_rng(406))
        variance = np.var(reference[:, 1], ddof=1)
        expected_eta = np.log(variance) - digamma(5.5) - np.log(2 / 11)
        np.testing.assert_allclose(response, np.r_[reference.mean(axis=0), expected_eta])
        np.testing.assert_allclose(noise, [.12 ** 2 / 12, variance / 12, polygamma(1, 5.5)])
        self.assertEqual(world.episode_calls, 12)


if __name__ == "__main__":
    unittest.main()
