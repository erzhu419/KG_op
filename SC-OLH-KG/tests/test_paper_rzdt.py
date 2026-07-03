import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
