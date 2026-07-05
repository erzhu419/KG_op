import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig  # noqa: E402
from performance import benchmark_exact_kg  # noqa: E402
from problems.rzdt import RZDT1  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


class ExactKGTests(unittest.TestCase):
    def test_optional_exact_posterior_update_scores_are_nonnegative(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        config = SingleOLHKGConfig(
            N=5,
            n0=4,
            K1=4,
            K2=0,
            exact_kg_mc_samples=2,
            exact_kg_use_score=True,
            eval_pool_size=12,
            seed=7,
        )
        alg = SingleOLHKGAlgorithm(problem, config)
        samples = alg._initial_samples()
        alg._fit_initial_belief(samples)
        candidates = [(0, 0, 0), (10, 0, 0), (20, 0, 0)]
        pool = alg._recommendation_pool()
        scores = alg._exact_posterior_update_scores(candidates, pool)
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(scores >= 0.0))
        self.assertEqual(len(alg.history), config.n0)

    def test_optional_exact_posterior_update_runner_smoke(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        config = SingleOLHKGConfig(
            N=6,
            n0=4,
            K1=4,
            K2=0,
            exact_kg_mc_samples=1,
            acquisition_mode="exact_mc",
            eval_pool_size=10,
            seed=11,
        )
        alg = SingleOLHKGAlgorithm(problem, config)
        result = alg.run()
        self.assertEqual(result["n_simulations"], 6)
        self.assertIn("x_recommended", result)
        self.assertTrue(any(
            "exact_kg_selected" in row
            for row in alg.iteration_log
        ))
        self.assertTrue(all(
            row.get("acquisition_mode") == "exact_mc"
            for row in alg.iteration_log
        ))

    def test_exact_mc_mode_supplies_default_sample_count(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        config = SingleOLHKGConfig(
            N=5,
            n0=4,
            K1=4,
            K2=0,
            acquisition_mode="exact_mc",
            exact_kg_mc_samples=0,
            eval_pool_size=10,
            seed=13,
        )
        alg = SingleOLHKGAlgorithm(problem, config)
        self.assertEqual(alg._effective_exact_kg_mc_samples(), 4)

    def test_additive_mode_ignores_exact_samples_unless_requested(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        config = SingleOLHKGConfig(
            N=5,
            n0=4,
            K1=4,
            K2=0,
            acquisition_mode="additive",
            exact_kg_mc_samples=5,
            exact_kg_use_score=False,
            exact_kg_blend=0.0,
            seed=17,
        )
        alg = SingleOLHKGAlgorithm(problem, config)
        self.assertEqual(alg._effective_exact_kg_mc_samples(), 0)

    def test_exact_benchmark_methods_set_acquisition_mode(self):
        class Args:
            exact_mc_samples = 3

        exact = benchmark_exact_kg._method_args(Args(), "exact")
        self.assertEqual(exact.acquisition_mode, "exact_mc")
        self.assertTrue(exact.exact_kg_use_score)
        blend = benchmark_exact_kg._method_args(Args(), "blend0.25")
        self.assertEqual(blend.acquisition_mode, "blend")
        self.assertEqual(blend.exact_kg_blend, 0.25)


if __name__ == "__main__":
    unittest.main()
