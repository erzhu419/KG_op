import copy
import math
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
    def test_exact_kg_gpr_clone_skips_non_picklable_handles(self):
        class ModuleHoldingBasisMap:
            feature_dim = 1

            def __init__(self):
                self.module_handle = math

            def features(self, x):
                return np.array([float(np.asarray(x, dtype=float)[0])])

        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        alg = SingleOLHKGAlgorithm(problem, SingleOLHKGConfig(seed=23))
        alg.gpr[0].basis_map = ModuleHoldingBasisMap()
        with self.assertRaises(TypeError):
            copy.deepcopy(alg.gpr[0])
        clone = alg._clone_gpr_for_exact_kg(alg.gpr[0])
        self.assertIs(clone.basis_map, alg.gpr[0].basis_map)
        self.assertIsNot(clone.a, alg.gpr[0].a)
        self.assertIsNot(clone.C, alg.gpr[0].C)

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

    def test_parallel_exact_scores_match_serial_scores(self):
        problem_a = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        problem_b = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        base = dict(
            N=5,
            n0=4,
            K1=4,
            K2=0,
            exact_kg_mc_samples=3,
            exact_kg_use_score=True,
            eval_pool_size=12,
            seed=19,
        )
        alg_serial = SingleOLHKGAlgorithm(
            problem_a,
            SingleOLHKGConfig(**base, exact_kg_jobs=1),
        )
        alg_parallel = SingleOLHKGAlgorithm(
            problem_b,
            SingleOLHKGConfig(**base, exact_kg_jobs=2),
        )
        samples_serial = alg_serial._initial_samples()
        samples_parallel = alg_parallel._initial_samples()
        self.assertEqual(samples_serial, samples_parallel)
        alg_serial._fit_initial_belief(samples_serial)
        alg_parallel._fit_initial_belief(samples_parallel)
        candidates = [(0, 0, 0), (10, 0, 0), (20, 0, 0), (15, 2, 1)]
        scores_serial = alg_serial._exact_posterior_update_scores(
            candidates, alg_serial._recommendation_pool())
        scores_parallel = alg_parallel._exact_posterior_update_scores(
            candidates, alg_parallel._recommendation_pool())
        np.testing.assert_allclose(scores_serial, scores_parallel, rtol=0.0, atol=1e-12)

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
        self.assertEqual(alg._effective_exact_kg_mc_samples(), 8)

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
