import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders.policy_state_encoder import SyntheticPolicyStateEncoder  # noqa: E402
from problems.rzdt import RegimeRZDT1, StatePolicyRZDT1  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
