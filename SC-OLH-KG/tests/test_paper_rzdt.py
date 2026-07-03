import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import SingleOLHKGConfig  # noqa: E402
from problems.rzdt import PaperRZDT1, make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


class PaperRZDTTests(unittest.TestCase):
    def test_paper_rzdt1_uses_submitted_constraint_output(self):
        problem = PaperRZDT1(d=5, L=100, sigma=0.04)
        self.assertAlmostEqual(problem.true_objectives((50, 0, 0, 0, 0))[2], 0.04)
        self.assertAlmostEqual(problem.true_objectives((0, 0, 0, 0, 0))[2], -0.21)
        self.assertAlmostEqual(problem.sigma_func((100, 0, 0, 0, 0)), 0.1)

    def test_paper_problem_registry_keeps_prototype_rzdt1_separate(self):
        paper = make_problem("PaperRZDT1", d=5, L=100, sigma=0.04)
        proto = make_problem("RZDT1", d=5, L=100, sigma=0.04)
        self.assertNotEqual(paper.problem_name, proto.problem_name)
        self.assertNotEqual(
            paper.true_objectives((50, 0, 0, 0, 0))[2],
            proto.true_objectives((50, 0, 0, 0, 0))[2],
        )

    def test_recommendation_axis_oracle_can_be_disabled(self):
        problem = ScalarizedProblem(make_problem("PaperRZDT2", d=5, L=100, sigma=0.04))
        self.assertTrue(SingleOLHKGConfig().recommendation_axis_oracle)
        self.assertTrue(hasattr(problem, "all_axis_solutions"))
        self.assertFalse(SingleOLHKGConfig(recommendation_axis_oracle=False).recommendation_axis_oracle)

    def test_paper_rzdt5_exposes_hyperbolic_surrogate_basis(self):
        problem = ScalarizedProblem(make_problem("PaperRZDT5_RR", d=5, L=100, sigma=0.04))
        basis = problem.surrogate_basis_map()
        self.assertIsNotNone(basis)
        refinement = problem.recommendation_refinement_candidates()
        self.assertIn((9, 0, 0, 0, 0), refinement)
        self.assertIn((10, 0, 0, 0, 0), refinement)
        train = [
            (0, 0, 0, 0, 0),
            (5, 0, 0, 0, 0),
            (9, 0, 0, 0, 0),
            (10, 0, 0, 0, 0),
            (14, 0, 0, 0, 0),
            (25, 0, 0, 0, 0),
            (50, 0, 0, 0, 0),
            (100, 0, 0, 0, 0),
            (9, 10, 0, 0, 0),
            (25, 10, 0, 0, 0),
            (9, 0, 10, 0, 0),
            (25, 0, 10, 0, 0),
            (9, 0, 0, 10, 0),
            (25, 0, 0, 10, 0),
            (9, 0, 0, 0, 10),
        ]
        Phi = np.vstack([
            np.concatenate([[1.0], basis.features(x)])
            for x in train
        ])
        y = np.array([problem.true_objective(x) for x in train], dtype=float)
        beta = np.linalg.lstsq(Phi, y, rcond=None)[0]
        pred_9 = float(np.concatenate([[1.0], basis.features((9, 0, 0, 0, 0))]) @ beta)
        pred_25 = float(np.concatenate([[1.0], basis.features((25, 0, 0, 0, 0))]) @ beta)
        self.assertLess(pred_9, pred_25)
        self.assertAlmostEqual(pred_9, problem.true_objective((9, 0, 0, 0, 0)), places=8)


if __name__ == "__main__":
    unittest.main()
