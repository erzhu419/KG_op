import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig  # noqa: E402
from encoders.policy_state_encoder import SyntheticPolicyStateEncoder  # noqa: E402
from problems.rzdt import HighDimStatePolicyRZDT1, RegimeRZDT1, StatePolicyRZDT1  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


class StateEncoderTests(unittest.TestCase):
    def test_coupling_scores_reward_coverage_gaps(self):
        problem = RegimeRZDT1(d=3, L=100, sigma=0.05)
        encoder = SyntheticPolicyStateEncoder(problem)
        candidates = [(0, 0, 0), (50, 0, 0), (100, 0, 0)]
        observed = [(0, 0, 0), (100, 0, 0)]
        scores = encoder.propagation_scores(candidates, observed)
        self.assertLess(scores[0], 1e-12)
        self.assertLess(scores[2], 1e-12)
        self.assertGreater(scores[1], 0.99)

    def test_coupling_scores_are_zero_when_all_candidates_equally_covered(self):
        problem = RegimeRZDT1(d=3, L=100, sigma=0.05)
        encoder = SyntheticPolicyStateEncoder(problem)
        candidates = [(0, 0, 0), (100, 0, 0)]
        observed = [(0, 0, 0), (100, 0, 0)]
        scores = encoder.propagation_scores(candidates, observed)
        self.assertTrue((scores == 0.0).all())

    def test_state_policy_problem_exposes_occupancy_grid(self):
        base = StatePolicyRZDT1(d=5, L=100, sigma=0.04)
        problem = ScalarizedProblem(base)
        grid = problem.all_axis_solutions()
        best_x, best_y = problem.true_best_feasible()
        self.assertGreater(len(grid), 1000)
        self.assertIsNotNone(best_x)
        self.assertTrue(problem.is_truly_feasible(best_x))
        self.assertLess(best_y, 0.45)
        self.assertTrue(10 <= best_x[0] <= 40)
        tail_mean = sum(best_x[1:]) / max(len(best_x) - 1, 1)
        self.assertTrue(55 <= tail_mean <= 85)

    def test_state_policy_structured_candidates_cover_low_spread_pocket(self):
        base = StatePolicyRZDT1(d=5, L=100, sigma=0.04)
        candidates = base.structured_candidates(n=42, rng=np.random.default_rng(0))
        self.assertIn((25, 70, 70, 70, 70), candidates)
        self.assertNotEqual(
            base.risk_class((25, 70, 70, 70, 70)),
            base.risk_class((25, 100, 100, 50, 30)),
        )

    def test_state_policy_initial_samples_cover_structure_without_true_best(self):
        base = StatePolicyRZDT1(d=5, L=100, sigma=0.04)
        samples = base.initial_samples(n=5)
        self.assertIn((0, 70, 70, 70, 70), samples)
        self.assertIn((50, 70, 70, 70, 70), samples)
        self.assertNotIn((24, 70, 70, 70, 70), samples)
        self.assertNotIn((25, 70, 70, 70, 70), samples)

    def test_state_space_candidates_invert_meta_anchors(self):
        base = StatePolicyRZDT1(d=5, L=100, sigma=0.04)
        problem = ScalarizedProblem(base)
        encoder = SyntheticPolicyStateEncoder(problem)
        candidates = encoder.state_space_candidates(
            n_anchors=12,
            inverse_neighbors=2,
            rng=np.random.default_rng(1),
        )
        self.assertGreaterEqual(len(candidates), 12)
        tail_spreads = [
            np.std(problem.normalize(x)[1:])
            for x in candidates
        ]
        self.assertLess(float(np.median(tail_spreads)), 0.08)

    def test_coupling_scores_prefer_promising_feasible_states(self):
        base = StatePolicyRZDT1(d=5, L=100, sigma=0.04)
        problem = ScalarizedProblem(base)
        encoder = SyntheticPolicyStateEncoder(problem)
        good = (20, 70, 70, 70, 70)
        far = (90, 0, 0, 0, 0)
        history = [
            (good, np.array([0.4, -0.2], dtype=float)),
            (far, np.array([1.5, 0.5], dtype=float)),
        ]
        scores = encoder.coupling_scores(
            [(21, 70, 70, 70, 70), (90, 0, 0, 0, 0)],
            history,
        )
        self.assertGreater(float(scores[0]), float(scores[1]))

    def test_high_dim_state_policy_has_low_dimensional_basis_and_true_best(self):
        base = HighDimStatePolicyRZDT1(d=1000, L=100, sigma=0.04)
        problem = ScalarizedProblem(base)
        best_x, best_y = problem.true_best_feasible()
        self.assertIsNotNone(best_x)
        self.assertEqual(len(best_x), 1000)
        self.assertTrue(problem.is_truly_feasible(best_x))
        self.assertLess(best_y, 0.35)
        u, q, spread = base.policy_state(best_x)
        self.assertTrue(0.15 <= u <= 0.30)
        self.assertTrue(0.65 <= q <= 0.78)
        self.assertLess(spread, 1e-12)
        self.assertEqual(len(problem.hvd_features(best_x)), 9)
        self.assertEqual(problem.recommendation_random_pool_size(), 0)

        alg = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(N=5, n0=5, K1=4, K2=0, seed=1),
        )
        self.assertEqual(alg.gpr[0].p, 12)
        self.assertLess(alg.gpr[0].p, 2 * problem.d + 1)

    def test_high_dim_state_inverse_returns_raw_policy_with_low_spread(self):
        base = HighDimStatePolicyRZDT1(d=1000, L=100, sigma=0.04)
        problem = ScalarizedProblem(base)
        encoder = SyntheticPolicyStateEncoder(problem)
        candidates = encoder.state_space_candidates(
            n_anchors=8,
            inverse_neighbors=2,
            rng=np.random.default_rng(2),
        )
        self.assertGreaterEqual(len(candidates), 8)
        self.assertTrue(all(len(x) == 1000 for x in candidates))
        spreads = [base.policy_state(x)[2] for x in candidates]
        self.assertLess(float(np.median(spreads)), 1e-12)


if __name__ == "__main__":
    unittest.main()
