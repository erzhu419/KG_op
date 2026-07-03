import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.botorch_adapters import (  # noqa: E402
    BoTorchBaseline,
    BoTorchBaselineConfig,
    is_botorch_available,
)
from problems.rzdt import StatePolicyRZDT1  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


@unittest.skipUnless(is_botorch_available(), "BoTorch is not installed")
class BoTorchAdapterTests(unittest.TestCase):
    def _problem(self):
        return ScalarizedProblem(StatePolicyRZDT1(d=5, L=100, sigma=0.04))

    def test_turbo_and_scbo_run_real_botorch_path(self):
        for method in ("botorch_turbo", "botorch_scbo"):
            config = BoTorchBaselineConfig(
                N=6,
                n0=4,
                seed=7,
                method=method,
                raw_samples=8,
                num_restarts=2,
                maxiter=10,
                batch_candidates=16,
            )
            result = BoTorchBaseline(self._problem(), config).run()
            self.assertEqual(result["backend"], "botorch")
            self.assertEqual(result["method"], method)
            self.assertEqual(result["n_simulations"], 6)
            self.assertIn("x_recommended", result)

    def test_saasbo_runs_with_tiny_nuts_budget(self):
        config = BoTorchBaselineConfig(
            N=5,
            n0=4,
            seed=9,
            method="botorch_saasbo",
            raw_samples=8,
            num_restarts=2,
            maxiter=10,
            saas_warmup_steps=4,
            saas_num_samples=4,
            saas_thinning=1,
            saas_max_tree_depth=2,
            saas_mc_samples=16,
        )
        result = BoTorchBaseline(self._problem(), config).run()
        self.assertEqual(result["backend"], "botorch")
        self.assertEqual(result["method"], "botorch_saasbo")
        self.assertEqual(result["n_simulations"], 5)
        self.assertTrue(result["saas_constrained"])


if __name__ == "__main__":
    unittest.main()

