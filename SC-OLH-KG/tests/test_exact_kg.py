import copy
import math
import multiprocessing
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
    def test_iid_exact_kg_sample_plan_has_equal_normalized_weights(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                exact_kg_sampling_mode="iid",
                seed=221,
            ),
        )
        z_rows, expert_uniforms, weights = algorithm._exact_kg_sample_plan(5)
        self.assertEqual(z_rows.shape, (5, 2))
        self.assertEqual(expert_uniforms.shape, (5,))
        np.testing.assert_allclose(weights, np.full(5, 0.2))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=12)

    def test_gaussian_positive_part_closed_form(self):
        values = SingleOLHKGAlgorithm._normal_positive_part(
            np.asarray([-1.0, 0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 0.0, 4.0]),
        )
        np.testing.assert_allclose(values[:3], [0.0, 0.0, 1.0])
        self.assertAlmostEqual(
            values[3], 2.0 / np.sqrt(2.0 * np.pi), places=12)

    def test_bayes_terminal_uses_fixed_unscaled_risk(self):
        class FakeGPR:
            def __init__(self, means, variances):
                self.means = means
                self.variances = variances

            def posterior_mean_many(self, pool):
                return np.asarray([
                    self.means[int(np.asarray(x)[0])] for x in pool
                ], dtype=float)

            def posterior_var_many(self, pool):
                return np.asarray([
                    self.variances[int(np.asarray(x)[0])] for x in pool
                ], dtype=float)

        class FakeVariance:
            @staticmethod
            def predict_certification_variance_many(output_index, pool, problem):
                del output_index, problem
                return np.full(len(pool), 0.01, dtype=float)

        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                exact_kg_terminal_mode="bayes_risk",
                terminal_bayes_violation_penalty=5.0,
                seed=222,
            ),
        )
        objective = FakeGPR(
            {0: 0.2, 1: 0.4, 2: 100.0}, {0: 0.0, 1: 0.0, 2: 0.0})
        constraint = FakeGPR(
            {0: -0.3, 1: 0.2, 2: 100.0},
            {0: 0.01, 1: 0.01, 2: 100.0},
        )
        base_pool = [(0, 0, 0), (1, 0, 0)]
        expanded_pool = base_pool + [(2, 0, 0)]
        base = algorithm._terminal_value_from_models(
            [objective, constraint], FakeVariance(), base_pool)
        expanded = algorithm._terminal_value_from_models(
            [objective, constraint], FakeVariance(), expanded_pool)
        self.assertAlmostEqual(base, expanded, places=12)

    def test_bayes_risk_exact_runner_and_recommendation_share_terminal(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        config = SingleOLHKGConfig(
            N=5,
            n0=4,
            K1=4,
            K2=0,
            exact_kg_mc_samples=1,
            acquisition_mode="exact_mc",
            exact_kg_terminal_mode="bayes_risk",
            terminal_bayes_violation_penalty=7.0,
            recommendation_infeasible_strategy="bayes_risk",
            recommendation_slack_initial=1e6,
            recommendation_observed_fallback=False,
            eval_pool_size=10,
            seed=223,
        )
        algorithm = SingleOLHKGAlgorithm(problem, config)
        result = algorithm.run()
        self.assertEqual(
            algorithm.iteration_log[0]["exact_kg_terminal_mode"],
            "bayes_risk",
        )
        self.assertTrue(result["posterior_bayes_risk_used"])
        self.assertIsNotNone(result["posterior_bayes_expected_violation"])
        self.assertGreaterEqual(
            result["posterior_bayes_expected_violation"], 0.0)
        row = algorithm.iteration_log[0]
        self.assertTrue(row["terminal_pool_shared"])
        self.assertEqual(row["rec_n_pool"], row["terminal_pool_size"])
        self.assertTrue(result["terminal_pool_shared"])
        self.assertEqual(result["n_pool"], result["terminal_pool_size"])

    def test_terminal_frontier_covers_risk_and_disagreement_axes(self):
        mu_obj = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4])
        margins = np.asarray([0.5, 0.4, 0.1, 0.3, 0.2])
        components = {
            "risk": np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
            "expected_violation": np.asarray([0.4, 0.3, 0.2, 0.1, 0.5]),
            "model_disagreement": np.asarray([0.0, 0.1, 0.9, 0.2, 1.0]),
        }
        indices, labels = SingleOLHKGAlgorithm._terminal_frontier_indices(
            mu_obj,
            margins,
            chosen=0,
            count=4,
            bayes_components=components,
        )
        self.assertEqual(indices, [0, 2, 3, 1])
        self.assertEqual(labels, [
            "bayes_action",
            "minimum_theory_margin",
            "minimum_expected_violation",
            "risk_frontier_fill",
        ])

    def test_terminal_frontier_candidates_join_experiment_actions(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        config = SingleOLHKGConfig(
            N=5,
            n0=4,
            K1=2,
            K2=0,
            acquisition_mode="exact_mc",
            exact_kg_mc_samples=1,
            exact_kg_terminal_mode="bayes_risk",
            recommendation_infeasible_strategy="bayes_risk",
            recommendation_slack_initial=1e6,
            terminal_frontier_candidate_count=4,
            eval_pool_size=12,
            seed=224,
        )
        algorithm = SingleOLHKGAlgorithm(problem, config)
        algorithm.run()
        row = algorithm.iteration_log[0]
        self.assertEqual(row["terminal_frontier_candidate_count"], 4)
        self.assertEqual(
            row["terminal_frontier_candidates_in_action_set"], 4)
        self.assertLessEqual(
            row["terminal_frontier_candidate_count"], row["n_candidates"])

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

    def test_raw_exact_scores_are_not_silently_clipped(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        config = SingleOLHKGConfig(
            N=5,
            n0=4,
            K1=4,
            K2=0,
            exact_kg_mc_samples=2,
            exact_kg_use_score=True,
            exact_kg_clip_negative=False,
            eval_pool_size=12,
            seed=8,
        )
        algorithm = SingleOLHKGAlgorithm(problem, config)
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)
        candidates = [(0, 0, 0), (10, 0, 0), (20, 0, 0)]
        scores = algorithm._exact_posterior_update_scores(
            candidates, algorithm._recommendation_pool())
        np.testing.assert_allclose(
            scores, algorithm._last_exact_kg_raw_scores, atol=0.0, rtol=0.0)

    def test_antithetic_common_samples_are_paired(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                exact_kg_sampling_mode="antithetic",
                seed=9,
            ),
        )
        z_rows, uniforms = algorithm._exact_kg_common_samples(5)
        np.testing.assert_allclose(z_rows[0], -z_rows[1])
        np.testing.assert_allclose(z_rows[2], -z_rows[3])
        np.testing.assert_allclose(z_rows[4], np.zeros(2))
        self.assertAlmostEqual(uniforms[0] + uniforms[1], 1.0)
        self.assertAlmostEqual(uniforms[2] + uniforms[3], 1.0)
        self.assertAlmostEqual(uniforms[4], 0.5)

    def test_stratified_expert_plan_integrates_discrete_weights(self):
        class Posterior:
            @staticmethod
            def decision_weights():
                return np.asarray([0.2, 0.3, 0.5], dtype=float)

        class Ensemble:
            posterior = Posterior()

        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                exact_kg_sampling_mode="stratified_expert",
                seed=9,
            ),
        )
        algorithm.task_ensemble = Ensemble()
        z_rows, uniforms, weights = algorithm._exact_kg_sample_plan(2)
        self.assertEqual(z_rows.shape, (6, 2))
        np.testing.assert_allclose(z_rows[0], -z_rows[1])
        np.testing.assert_allclose(z_rows[:2], z_rows[2:4])
        np.testing.assert_allclose(z_rows[:2], z_rows[4:6])
        np.testing.assert_allclose(
            uniforms, [0.1, 0.1, 0.35, 0.35, 0.75, 0.75])
        np.testing.assert_allclose(
            weights, [0.1, 0.1, 0.15, 0.15, 0.25, 0.25])
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)

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

    @unittest.skipUnless(
        "fork" in multiprocessing.get_all_start_methods(),
        "fork backend requires Linux",
    )
    def test_fork_process_exact_scores_match_serial_scores(self):
        problem_a = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        problem_b = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        base = dict(
            N=5,
            n0=4,
            K1=4,
            K2=0,
            exact_kg_mc_samples=2,
            exact_kg_use_score=True,
            eval_pool_size=12,
            seed=29,
        )
        serial = SingleOLHKGAlgorithm(
            problem_a,
            SingleOLHKGConfig(**base, exact_kg_jobs=1),
        )
        forked = SingleOLHKGAlgorithm(
            problem_b,
            SingleOLHKGConfig(
                **base,
                exact_kg_jobs=3,
                exact_kg_parallel_backend="process_fork",
            ),
        )
        serial_samples = serial._initial_samples()
        forked_samples = forked._initial_samples()
        self.assertEqual(serial_samples, forked_samples)
        serial._fit_initial_belief(serial_samples)
        forked._fit_initial_belief(forked_samples)
        candidates = [(0, 0, 0), (10, 0, 0), (20, 0, 0), (15, 2, 1)]
        serial_scores = serial._exact_posterior_update_scores(
            candidates, serial._recommendation_pool())
        forked_scores = forked._exact_posterior_update_scores(
            candidates, forked._recommendation_pool())
        np.testing.assert_allclose(
            serial_scores, forked_scores, rtol=0.0, atol=1e-12)

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
