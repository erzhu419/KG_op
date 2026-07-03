import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.candidates import axis_candidates, boundary_solutions  # noqa: E402
from problems.rzdt import RZDT2, StatePolicyRZDT1  # noqa: E402


class CandidateTests(unittest.TestCase):
    def test_boundary_solutions_prioritize_global_extremes(self):
        problem = RZDT2(d=5, L=100, sigma=0.04)
        rows = boundary_solutions(problem)
        self.assertEqual(rows[0], (0, 0, 0, 0, 0))
        self.assertEqual(rows[1], (100, 100, 100, 100, 100))
        self.assertEqual(rows[2], (50, 50, 50, 50, 50))
        self.assertEqual(rows[3], (100, 0, 0, 0, 0))
        self.assertEqual(rows[4], (25, 0, 0, 0, 0))
        self.assertEqual(rows[5], (50, 0, 0, 0, 0))
        self.assertEqual(rows[6], (75, 0, 0, 0, 0))
        self.assertEqual(len(rows), len(set(rows)))

    def test_axis_candidates_empty_without_problem_axis_oracle(self):
        problem = RZDT2(d=5, L=100, sigma=0.04)
        rows = axis_candidates(problem, n=9)
        self.assertEqual(rows, [])

    def test_axis_candidates_cover_problem_axis_quantiles(self):
        problem = StatePolicyRZDT1(d=5, L=100, sigma=0.04)
        rows = axis_candidates(problem, n=9)
        self.assertIn((0, 0, 0, 0, 0), rows)
        self.assertIn((100, 100, 100, 100, 100), rows)
        self.assertEqual(len(rows), 9)
        self.assertEqual(len(rows), len(set(rows)))


if __name__ == "__main__":
    unittest.main()
