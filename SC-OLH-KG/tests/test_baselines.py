import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.baseline_algorithms import BaselineConfig, SequentialBaseline  # noqa: E402
from problems.rzdt import StatePolicyRZDT1  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


class BaselineTests(unittest.TestCase):
    def test_lightweight_baselines_run(self):
        problem = ScalarizedProblem(StatePolicyRZDT1(d=5, L=100, sigma=0.04))
        for method in (
            "random",
            "sobol",
            "turbo_lite",
            "scbo_lite",
            "hetgp_lite",
            "rahbo_lite",
            "safeopt_lite",
            "legacy_vepm_lite",
        ):
            config = BaselineConfig(N=8, n0=4, seed=1, method=method)
            result = SequentialBaseline(problem, config).run()
            self.assertEqual(result["n_simulations"], 8)
            self.assertIn("x_recommended", result)
            self.assertIn("true_feasible", result)


if __name__ == "__main__":
    unittest.main()
