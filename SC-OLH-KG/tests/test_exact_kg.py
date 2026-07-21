import copy
import math
import multiprocessing
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig  # noqa: E402
from performance import benchmark_exact_kg  # noqa: E402
from problems.rzdt import RZDT1  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


class ExactKGTests(unittest.TestCase):
    def test_confidence_only_sandwich_separates_ranking_from_certificate(self):
        class ObjectiveGPR:
            @staticmethod
            def posterior_mean_many(pool):
                return np.zeros(len(pool), dtype=float)

            @staticmethod
            def posterior_var_many(pool):
                return np.full(len(pool), 0.01, dtype=float)

        class ConstraintGPR:
            @staticmethod
            def posterior_mean_many(pool):
                return np.full(len(pool), -0.1, dtype=float)

            @staticmethod
            def posterior_var_many(pool):
                return np.full(len(pool), 1.0, dtype=float)

            @staticmethod
            def decision_posterior_var_many(pool):
                return np.full(len(pool), 0.01, dtype=float)

        class Variance:
            @staticmethod
            def predict_certification_variance_many(
                output_index, pool, problem,
            ):
                del output_index, problem
                return np.full(len(pool), 0.01, dtype=float)

        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                certification_head_authority="split_gpr_cumulative_hvd",
                source_constraint_mean_misspecification_mode=(
                    "predictive_scale_sandwich_hc3_confidence"),
            ),
        )
        pool = [(1, 1, 1), (2, 2, 2)]
        models = [ObjectiveGPR(), ConstraintGPR()]
        risk = algorithm._terminal_bayes_risk_components(
            models, Variance(), pool)
        certificate = algorithm._terminal_certificate_components(
            models, Variance(), pool)
        z_alpha = float(norm.ppf(1 - problem.alpha))
        margin_mean = -0.1 + z_alpha * 0.1 - problem.tau
        expected = algorithm._normal_positive_part(
            np.full(2, margin_mean), np.full(2, 0.01))
        np.testing.assert_allclose(risk["expected_violation"], expected)
        np.testing.assert_allclose(certificate["epistemic"], 1.0)
        self.assertTrue(np.all(risk["violation_variance"] > 0.01))

    def test_split_certificate_authority_is_shared_by_filter_and_terminal(self):
        class FakeGPR:
            @staticmethod
            def posterior_mean_many(pool):
                return np.full(len(pool), 0.20, dtype=float)

            @staticmethod
            def posterior_var_many(pool):
                return np.full(len(pool), 0.04, dtype=float)

        class FakeVariance:
            @staticmethod
            def predict_certification_variance_many(
                output_index, pool, problem,
            ):
                del output_index, problem
                return np.full(len(pool), 0.01, dtype=float)

        class FakeTaskEnsemble:
            joint_calls = 0
            posterior = SimpleNamespace(
                safe_generalized=False,
                decision_weights=lambda: np.asarray([1.0]),
                posterior_weights=lambda: np.asarray([1.0]),
            )

            @classmethod
            def robust_moments_many(
                cls, output_index, pool, certification=True,
            ):
                del cls, output_index, certification
                n = len(pool)
                return SimpleNamespace(
                    mean_upper=np.full(n, 9.0, dtype=float),
                    epistemic_upper=np.full(n, 4.0, dtype=float),
                    aleatoric_upper=np.full(n, 0.25, dtype=float),
                    nominal=SimpleNamespace(
                        between_mean=np.zeros(n, dtype=float)),
                )

            @classmethod
            def robust_chance_margin_many(cls, pool, **kwargs):
                del kwargs
                cls.joint_calls += 1
                return SimpleNamespace(
                    upper=np.full(len(pool), -99.0, dtype=float),
                    separable_upper=np.full(len(pool), 99.0, dtype=float),
                )

            @staticmethod
            def mixture_moments_many(output_index, pool, certification=False):
                del output_index, certification
                return SimpleNamespace(mean=np.zeros(len(pool), dtype=float))

            @staticmethod
            def expert_moments_many(output_index, pool, certification=False):
                del certification
                if output_index != 0:
                    raise AssertionError(
                        "split Bayes risk must not request constraint experts")
                n = len(pool)
                return (
                    np.zeros((1, n), dtype=float),
                    np.full((1, n), 0.02, dtype=float),
                    np.zeros((1, n), dtype=float),
                )

        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        pool = [(2, 2, 2), (4, 4, 4)]
        expected = {
            "split_gpr_task_hvd": (
                0.25, "split_aggregate_gpr_task_hvd"),
            "split_gpr_cumulative_hvd": (
                0.01, "split_aggregate_gpr_cumulative_hvd"),
        }
        for authority, (variance, source) in expected.items():
            with self.subTest(authority=authority):
                algorithm = SingleOLHKGAlgorithm(
                    problem,
                    SingleOLHKGConfig(
                        task_posterior_robust_certificate_mode="joint_tangent",
                        certification_head_authority=authority,
                        recommendation_slack_initial=0.0,
                        seed=245,
                    ),
                )
                gpr = FakeGPR()
                variance_model = FakeVariance()
                ensemble = FakeTaskEnsemble()
                algorithm.gpr[1] = gpr
                algorithm.variance_model = variance_model
                algorithm.task_ensemble = ensemble
                FakeTaskEnsemble.joint_calls = 0

                direct = algorithm._certification_result(
                    np.full(2, 100.0),
                    pool,
                    np.full(2, 100.0),
                    epistemic=np.full(2, 100.0),
                )
                terminal = algorithm._terminal_certificate_components(
                    [None, gpr],
                    variance_model,
                    pool,
                    task_ensemble=ensemble,
                )
                bayes = algorithm._terminal_bayes_risk_components(
                    [None, gpr],
                    variance_model,
                    pool,
                    task_ensemble=ensemble,
                )

                np.testing.assert_allclose(direct.mu, 0.20)
                np.testing.assert_allclose(direct.epistemic_var, 0.04)
                np.testing.assert_allclose(direct.aleatoric_var, variance)
                np.testing.assert_allclose(terminal["mu_con"], direct.mu)
                np.testing.assert_allclose(
                    terminal["epistemic"], direct.epistemic_var)
                np.testing.assert_allclose(
                    terminal["aleatoric"], direct.aleatoric_var)
                np.testing.assert_allclose(terminal["margin"], direct.margin)
                self.assertEqual(terminal["source"], source)
                self.assertEqual(
                    terminal["certification_head_authority"], authority)
                self.assertEqual(
                    bayes["certification_head_authority"], authority)
                self.assertEqual(
                    bayes["constraint_posterior_source"], source)
                self.assertTrue(np.all(np.isfinite(
                    bayes["expected_violation"])))
                self.assertEqual(FakeTaskEnsemble.joint_calls, 0)

    def test_joint_task_certificate_is_shared_by_filter_and_terminal_value(self):
        class FakeTaskEnsemble:
            @staticmethod
            def robust_moments_many(output_index, pool, certification=True):
                del output_index, certification
                n = len(pool)
                nominal = SimpleNamespace(
                    between_mean=np.zeros(n, dtype=float))
                return SimpleNamespace(
                    mean_upper=np.full(n, 0.10, dtype=float),
                    epistemic_upper=np.full(n, 0.16, dtype=float),
                    aleatoric_upper=np.full(n, 0.09, dtype=float),
                    nominal=nominal,
                )

            @staticmethod
            def robust_chance_margin_many(
                pool, *, beta_g, z_alpha, tau, certification=True,
            ):
                del beta_g, z_alpha, tau, certification
                n = len(pool)
                return SimpleNamespace(
                    upper=np.full(n, -0.20, dtype=float),
                    separable_upper=np.full(n, 1.0, dtype=float),
                )

            @staticmethod
            def mixture_moments_many(output_index, pool, certification=False):
                del output_index, certification
                return SimpleNamespace(mean=np.arange(len(pool), dtype=float))

        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                task_posterior_robust_certificate_mode="joint_tangent",
                recommendation_slack_initial=0.0,
                seed=244,
            ),
        )
        ensemble = FakeTaskEnsemble()
        algorithm.task_ensemble = ensemble
        pool = [(2, 2, 2), (4, 4, 4)]
        direct = algorithm._certification_result(
            np.zeros(2), pool, np.ones(2))
        terminal = algorithm._terminal_certificate_components(
            [None, None], None, pool, task_ensemble=ensemble)
        np.testing.assert_allclose(direct.margin, [-0.20, -0.20])
        np.testing.assert_allclose(terminal["margin"], direct.margin)
        self.assertEqual(terminal["source"], "task_joint_kl_hvd")
        self.assertTrue(np.all(terminal["separable_margin"] > 0.0))

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

    def test_gaussian_positive_part_variance_closed_form(self):
        values = SingleOLHKGAlgorithm._normal_positive_part_variance(
            np.asarray([-1.0, 0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 0.0, 4.0]),
        )
        np.testing.assert_allclose(values[:3], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(values[3], 2.0 - 2.0 / np.pi, places=12)

    def test_cantelli_dominance_requires_mean_gain_and_low_uncertainty(self):
        certain = SingleOLHKGAlgorithm._cantelli_dominance_lower_bound(
            1.0, 0.0, 1e-4, 1e-4)
        uncertain = SingleOLHKGAlgorithm._cantelli_dominance_lower_bound(
            1.0, 0.0, 1.0, 1.0)
        no_gain = SingleOLHKGAlgorithm._cantelli_dominance_lower_bound(
            0.0, 1.0, 0.0, 0.0)
        self.assertGreater(
            certain["posterior_dominance_lower_bound"], 0.95)
        self.assertLess(
            uncertain["posterior_dominance_lower_bound"], 0.95)
        self.assertEqual(no_gain["posterior_dominance_lower_bound"], 0.0)

    def test_posterior_dominance_retains_uncertain_and_accepts_certain(self):
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
                return np.zeros(len(pool), dtype=float)

        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                posterior_dominance_enabled=True,
                posterior_dominance_delta=0.05,
                terminal_bayes_violation_penalty=5.0,
            ),
        )
        constraint = FakeGPR(
            {0: -100.0, 1: -100.0},
            {0: 0.0, 1: 0.0},
        )
        incumbent = (0, 0, 0)
        challenger = (1, 0, 0)
        uncertain_objective = FakeGPR(
            {0: 1.0, 1: 0.0}, {0: 1.0, 1: 1.0})
        retained, _, retained_info = (
            algorithm._posterior_dominance_decision_from_models(
                [uncertain_objective, constraint],
                FakeVariance(),
                [incumbent, challenger],
                incumbent,
            )
        )
        self.assertEqual(retained, incumbent)
        self.assertFalse(retained_info["switch_accepted"])

        certain_objective = FakeGPR(
            {0: 1.0, 1: 0.0}, {0: 1e-4, 1: 1e-4})
        switched, _, switched_info = (
            algorithm._posterior_dominance_decision_from_models(
                [certain_objective, constraint],
                FakeVariance(),
                [incumbent, challenger],
                incumbent,
            )
        )
        self.assertEqual(switched, challenger)
        self.assertTrue(switched_info["switch_accepted"])
        self.assertLessEqual(
            switched_info["false_switch_posterior_bound"], 0.05)
        self.assertFalse(switched_info["target_oracle_used"])

        unobserved = (2, 0, 0)
        objective_with_unobserved = FakeGPR(
            {0: 1.0, 1: 0.0, 2: -100.0},
            {0: 1e-4, 1: 1e-4, 2: 1e-4},
        )
        constraint_with_unobserved = FakeGPR(
            {0: -100.0, 1: -100.0, 2: -100.0},
            {0: 0.0, 1: 0.0, 2: 0.0},
        )
        algorithm._posterior_dominance_incumbent = incumbent
        algorithm.observations = {
            incumbent: [np.asarray([1.0, -100.0])],
        }
        observed_value = algorithm._terminal_value_from_models(
            [objective_with_unobserved, constraint_with_unobserved],
            FakeVariance(),
            [incumbent, unobserved],
            observations=algorithm.observations,
        )
        self.assertAlmostEqual(observed_value, 1.0, places=10)

    def test_posterior_dominance_initializes_from_canonical_certificate(self):
        class FakeGPR:
            def __init__(self, means):
                self.means = means

            def posterior_mean_many(self, pool):
                return np.asarray([
                    self.means[int(np.asarray(x)[0])] for x in pool
                ], dtype=float)

            def posterior_var_many(self, pool):
                return np.zeros(len(pool), dtype=float)

        class FakeVariance:
            @staticmethod
            def predict_certification_variance_many(output_index, pool, problem):
                del output_index, problem
                return np.zeros(len(pool), dtype=float)

        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                posterior_dominance_enabled=True,
                posterior_dominance_initialization=(
                    "certificate_lexicographic"),
                terminal_bayes_violation_penalty=5.0,
            ),
        )
        unsafe_low_objective = (0, 0, 0)
        safe_higher_objective = (1, 0, 0)
        objective = FakeGPR({0: -100.0, 1: 0.0})
        constraint = FakeGPR({0: 1.0, 1: -1.0})
        selected, _, details = (
            algorithm._posterior_dominance_decision_from_models(
                [objective, constraint],
                FakeVariance(),
                [unsafe_low_objective, safe_higher_objective],
                None,
            )
        )
        self.assertEqual(selected, safe_higher_objective)
        self.assertEqual(details["status"], "initialized_certified")
        self.assertEqual(details["initial_certified_count"], 1)
        self.assertEqual(
            details["posterior_dominance_initialization"],
            "certificate_lexicographic",
        )
        self.assertLessEqual(details["selected_certificate_margin"], 0.0)
        self.assertFalse(details["target_oracle_used"])

    def test_certified_only_dominance_stays_uninitialized_without_certificate(
        self,
    ):
        class FakeGPR:
            def __init__(self, means):
                self.means = means

            def posterior_mean_many(self, pool):
                return np.asarray([
                    self.means[int(np.asarray(x)[0])] for x in pool
                ], dtype=float)

            def posterior_var_many(self, pool):
                return np.zeros(len(pool), dtype=float)

        class FakeVariance:
            @staticmethod
            def predict_certification_variance_many(output_index, pool, problem):
                del output_index, problem
                return np.zeros(len(pool), dtype=float)

        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                posterior_dominance_enabled=True,
                posterior_dominance_initialization="certified_only",
            ),
        )
        pool = [(0, 0, 0), (1, 0, 0)]
        objective = FakeGPR({0: -1.0, 1: 0.0})
        constraint = FakeGPR({0: 2.0, 1: 1.5})
        selected, value, details = (
            algorithm._posterior_dominance_decision_from_models(
                [objective, constraint], FakeVariance(), pool, None)
        )
        self.assertIsNone(selected)
        self.assertTrue(np.isinf(value))
        self.assertEqual(details["status"], "uninitialized_no_certificate")
        self.assertEqual(details["initial_certified_count"], 0)
        self.assertTrue(details["terminal_fallback_required"])
        self.assertFalse(details["switch_accepted"])
        self.assertFalse(details["target_oracle_used"])

    def test_terminal_bayes_pool_audit_freezes_rank_before_truth_join(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(truth_pool_diagnostics=True),
        )
        pool = [(0, 0, 0), (1, 0, 0)]
        algorithm._true_chance_margin = lambda x: {
            pool[0]: -0.2,
            pool[1]: 0.1,
        }[tuple(x)]
        audit = algorithm._terminal_bayes_pool_audit(
            pool,
            {"risk": np.asarray([1.0, 2.0])},
            pool[1],
        )
        self.assertEqual(audit["selected_risk_rank"], 2)
        self.assertFalse(audit["selected_matches_counterfactual_bayes"])
        self.assertFalse(audit["selected_true_feasible"])
        self.assertTrue(audit["counterfactual_bayes_true_feasible"])
        self.assertFalse(audit["target_oracle_used_for_ranking"])
        self.assertFalse(audit["truth_admissible_decision_input"])
        self.assertFalse(audit["target_oracle_used_for_decision"])
        self.assertEqual(len(audit["posterior_ranked_candidates"]), 2)
        self.assertEqual(
            audit["best_true_feasible_post_rank"]["posterior_rank"], 1)

    def test_terminal_probability_loss_uses_central_hvd_only_for_decision(self):
        class FakeGPR:
            def __init__(self, mean, variance):
                self.mean = float(mean)
                self.variance = float(variance)

            def posterior_mean_many(self, pool):
                return np.full(len(pool), self.mean, dtype=float)

            def posterior_var_many(self, pool):
                return np.full(len(pool), self.variance, dtype=float)

        class SplitVariance:
            @staticmethod
            def predict_variance_many(output_index, pool, problem):
                del output_index, problem
                return np.full(len(pool), 0.01, dtype=float)

            @staticmethod
            def predict_certification_variance_many(
                output_index, pool, problem,
            ):
                del output_index, problem
                return np.full(len(pool), 1.0, dtype=float)

        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                certification_head_authority="split_gpr_cumulative_hvd",
                decision_aleatoric_mode="posterior_central",
                decision_violation_loss_mode="failure_probability",
                terminal_bayes_violation_penalty=7.0,
            ),
        )
        pool = [(0, 0, 0), (1, 0, 0)]
        components = algorithm._terminal_bayes_risk_components(
            [FakeGPR(0.2, 0.01), FakeGPR(0.1, 0.04)],
            SplitVariance(),
            pool,
        )
        np.testing.assert_allclose(
            components["violation_loss"],
            components["probability_violation"],
        )
        np.testing.assert_allclose(
            components["risk"],
            components["objective"]
            + 7.0 * components["probability_violation"],
        )
        self.assertEqual(
            components["decision_aleatoric_mode"], "posterior_central")
        self.assertEqual(
            components["violation_loss_mode"], "failure_probability")

    def test_terminal_nominal_task_mixture_separates_bayes_risk_from_robustness(self):
        class Posterior:
            safe_generalized = False

            @staticmethod
            def decision_weights():
                return np.asarray([0.5, 0.5], dtype=float)

            @staticmethod
            def posterior_weights():
                return np.asarray([0.5, 0.5], dtype=float)

            @staticmethod
            def kl_robust_expectation(values, radius):
                assert radius == 1.0
                return np.max(np.asarray(values, dtype=float), axis=0)

        class Ensemble:
            posterior = Posterior()
            task_latent_authoritative = False

            @staticmethod
            def effective_kl_radius():
                return 1.0

            @staticmethod
            def expert_moments_many(output_index, pool, certification=False):
                del certification
                n = len(pool)
                if int(output_index) == 0:
                    return (
                        np.vstack([np.full(n, 0.2), np.full(n, 0.2)]),
                        np.full((2, n), 0.01),
                        np.zeros((2, n)),
                    )
                return (
                    np.asarray([[0.0, 0.2], [0.4, 0.2]], dtype=float),
                    np.full((2, n), 0.01),
                    np.full((2, n), 0.01),
                )

        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        pool = [(0, 0, 0), (1, 0, 0)]

        def components(mode):
            algorithm = SingleOLHKGAlgorithm(
                problem,
                SingleOLHKGConfig(
                    certification_head_authority="task_joint",
                    decision_aleatoric_mode="posterior_central",
                    decision_ambiguity_mode=mode,
                    terminal_bayes_violation_penalty=5.0,
                ),
            )
            return algorithm._terminal_bayes_risk_components(
                [object(), object()],
                object(),
                pool,
                task_ensemble=Ensemble(),
            )

        robust = components("kl_robust")
        nominal = components("posterior_nominal")
        np.testing.assert_allclose(
            nominal["expected_violation"],
            nominal["nominal_expected_violation"],
        )
        np.testing.assert_allclose(
            robust["expected_violation"],
            robust["robust_expected_violation"],
        )
        self.assertTrue(np.all(
            nominal["risk"] <= robust["risk"] + 1e-12))
        self.assertEqual(
            nominal["decision_ambiguity_mode"], "posterior_nominal")

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

    def test_promoted_exact_voi_and_final_decision_share_observed_action_pool(self):
        class FakeGPR:
            def __init__(self, means):
                self.means = means

            def posterior_mean_many(self, pool):
                return np.asarray([
                    self.means[int(np.asarray(x)[0])] for x in pool
                ], dtype=float)

            @staticmethod
            def posterior_var_many(pool):
                return np.full(len(pool), 0.01, dtype=float)

        class FakeVariance:
            @staticmethod
            def predict_variance_many(output_index, pool, problem):
                del output_index, problem
                return np.full(len(pool), 0.01, dtype=float)

        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                decision_backend="sobol_exact_joint_voi",
                exact_kg_terminal_mode="bayes_risk",
                decision_aleatoric_mode="posterior_central",
                decision_ambiguity_mode="posterior_nominal",
                decision_recommend_observed_only=True,
                decision_risk_penalty=5.0,
                terminal_bayes_violation_penalty=999.0,
            ),
        )
        observed = (0, 0, 0)
        new = (1, 0, 0)
        pool = [observed, new]
        current_observations = {
            observed: [np.asarray([1.0, -10.0], dtype=float)],
        }
        future_observations = {
            **current_observations,
            new: [np.asarray([-10.0, -10.0], dtype=float)],
        }
        models = [
            FakeGPR({0: 1.0, 1: -10.0}),
            FakeGPR({0: -10.0, 1: -10.0}),
        ]
        current = algorithm._terminal_value_from_models(
            models,
            FakeVariance(),
            pool,
            observations=current_observations,
        )
        future = algorithm._terminal_value_from_models(
            models,
            FakeVariance(),
            pool,
            observations=future_observations,
        )
        observed_risk = algorithm._terminal_bayes_risk_components(
            models,
            FakeVariance(),
            [observed],
            risk_penalty=5.0,
        )["risk"][0]
        self.assertAlmostEqual(current, observed_risk, places=12)
        self.assertLess(future, current)
        self.assertEqual(
            algorithm._terminal_action_pool(
                pool, observations=current_observations),
            [observed],
        )
        self.assertEqual(
            algorithm._terminal_action_pool(
                pool, observations=future_observations),
            [observed, new],
        )
        self.assertIn(
            "observed_actions",
            algorithm._terminal_value_contract_id(),
        )

    def test_authoritative_task_latent_defines_terminal_bayes_risk(self):
        class AuthoritativeEnsemble:
            task_latent_authoritative = True

            @staticmethod
            def joint_terminal_risk_many(X, *, tau, alpha):
                self.assertEqual(X, [(0, 0, 0), (1, 0, 0)])
                self.assertTrue(np.isfinite(tau))
                self.assertGreater(alpha, 0.0)
                return {
                    "objective": np.asarray([0.2, 0.1]),
                    "expected_violation": np.asarray([0.01, 0.2]),
                    "nominal_expected_violation": np.asarray([0.01, 0.2]),
                    "risk": np.asarray([0.3, 2.1]),
                    "model_disagreement": np.asarray([0.0, 0.1]),
                    "kl_radius": 0.0,
                    "task_latent_authoritative": True,
                }

        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(exact_kg_terminal_mode="bayes_risk"),
        )
        pool = [(0, 0, 0), (1, 0, 0)]
        components = algorithm._terminal_bayes_risk_components(
            None,
            None,
            pool,
            task_ensemble=AuthoritativeEnsemble(),
        )
        np.testing.assert_allclose(components["risk"], [0.3, 2.1])
        value = algorithm._terminal_value_from_models(
            None,
            None,
            pool,
            task_ensemble=AuthoritativeEnsemble(),
        )
        self.assertAlmostEqual(value, 0.3)

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

    def test_adaptive_replication_and_dominance_share_exact_terminal(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        config = SingleOLHKGConfig(
            N=5,
            n0=4,
            K1=4,
            K2=0,
            decision_backend="exact_kg",
            acquisition_mode="exact_mc",
            exact_kg_mc_samples=1,
            exact_kg_terminal_mode="bayes_risk_dominance",
            adaptive_replication_voi=True,
            replication_candidate_count=2,
            replication_max_per_solution=5,
            posterior_dominance_enabled=True,
            posterior_dominance_delta=0.10,
            finalist_replication_budget=0,
            certification_recheck_top_k=0,
            eval_pool_size=10,
            seed=227,
        )
        algorithm = SingleOLHKGAlgorithm(problem, config)
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)
        candidates, sources = algorithm._generate_candidates(config.n0)
        replication_points = algorithm._replication_candidates()
        self.assertTrue(replication_points)
        self.assertTrue(all(point in candidates for point in replication_points))
        self.assertIn("replication", set(sources.values()))

        algorithm = SingleOLHKGAlgorithm(
            ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03)), config)
        result = algorithm.run()
        replication = result["adaptive_replication_voi"]
        self.assertTrue(replication["enabled"])
        self.assertTrue(replication["unified_exact_voi"])
        self.assertEqual(
            replication["selected_replication_count"]
            + replication["selected_new_point_count"],
            config.N - config.n0,
        )
        self.assertEqual(replication["forced_recheck_count"], 0)
        self.assertTrue(result["posterior_dominance"]["enabled"])
        self.assertTrue(result["posterior_dominance_terminal_used"])
        self.assertEqual(
            result["decision_backend_contract"]["terminal_rule"],
            "posterior_dominance",
        )
        self.assertTrue(result["decision_backend_contract"]["coherent"])

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

    def test_terminal_frontier_reserves_tcb_boundary_actions(self):
        indices, labels = SingleOLHKGAlgorithm._terminal_frontier_indices(
            np.asarray([0.0, 0.1, 0.2, 0.3]),
            np.asarray([0.4, 0.3, 0.2, 0.1]),
            chosen=0,
            count=3,
            tcb_upper=np.asarray([0.8, 0.4, -0.1, 0.2]),
            tcb_count=2,
        )
        self.assertEqual(indices, [0, 2, 3])
        self.assertEqual(labels, [
            "bayes_action",
            "minimum_tcb_upper",
            "closest_tcb_boundary",
        ])

    def test_tcb_v2_certificate_is_authoritative_at_recommendation(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        config = SingleOLHKGConfig(
            N=5,
            n0=4,
            K1=2,
            K2=0,
            tcb_v2_mode="certified",
            recommendation_observed_fallback=False,
            recommendation_calibration=False,
            certification_calibration=False,
            seed=226,
        )
        algorithm = SingleOLHKGAlgorithm(problem, config)
        algorithm._fit_initial_belief(algorithm._initial_samples())
        pool = [(0, 0, 0), (20, 20, 20)]
        algorithm._objective_posterior_mean_many = lambda candidates: (
            np.asarray([0.0, 1.0], dtype=float)
        )
        algorithm._observed_nominal_incumbent = lambda: None

        def fake_tcb(candidates, **kwargs):
            del candidates, kwargs
            return {
                "mean": np.asarray([0.3, -0.3]),
                "upper": np.asarray([0.4, -0.2]),
                "adapter_diagnostics": {"effective_dimension": 2},
                "pilot_points": 4,
                "target_oracle_used": False,
            }

        algorithm._tcb_v2_margin_many = fake_tcb
        selected, diagnostics = algorithm._solve_posterior_recommendation(
            pool=pool)
        self.assertEqual(selected, pool[1])
        self.assertTrue(diagnostics["tcb_v2_authoritative"])
        self.assertEqual(
            diagnostics["posterior_certification_source"],
            "tcb_v2_hierarchical",
        )
        self.assertLess(diagnostics["posterior_chance_margin"], 0.0)

    def test_lexicographic_terminal_index_does_not_scalarize(self):
        values = np.asarray([
            [0.2, 0.0, -1000.0],
            [0.1, 100.0, 1000.0],
            [0.1, 2.0, 10.0],
            [0.1, 2.0, 5.0],
        ])
        self.assertEqual(
            SingleOLHKGAlgorithm._terminal_value_index(values), 3)

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

    def test_exact_action_scope_refits_only_declared_candidates(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        config = SingleOLHKGConfig(
            N=5,
            n0=4,
            K1=4,
            K2=0,
            decision_backend="sobol_exact_joint_voi",
            adaptive_replication_voi=True,
            exact_kg_mc_samples=2,
            exact_kg_jobs=1,
            eval_pool_size=12,
            seed=71,
        )
        algorithm = SingleOLHKGAlgorithm(problem, config)
        algorithm._fit_initial_belief(algorithm._initial_samples())
        candidates = [(0, 0, 0), (10, 0, 0), (20, 0, 0)]
        scores = algorithm._exact_posterior_update_scores_for_actions(
            candidates,
            algorithm._recommendation_pool(),
            [0, 2],
        )
        self.assertEqual(scores.shape, (3,))
        self.assertLess(scores[1], -1e250)
        self.assertTrue(np.all(np.isfinite(scores[[0, 2]])))
        self.assertTrue(np.isnan(
            algorithm._last_exact_kg_raw_scores[1]))
        np.testing.assert_array_equal(
            algorithm._last_exact_kg_active_indices, [0, 2])

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

    def test_nested_antithetic_plans_have_exact_prefixes(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                exact_kg_sampling_mode="antithetic_nested",
                seed=9,
            ),
        )
        z2, u2 = algorithm._exact_kg_common_samples(2)
        z8, u8 = algorithm._exact_kg_common_samples(8)
        z32, u32 = algorithm._exact_kg_common_samples(32)
        np.testing.assert_allclose(z2, z8[:2])
        np.testing.assert_allclose(z8, z32[:8])
        np.testing.assert_allclose(u2, u8[:2])
        np.testing.assert_allclose(u8, u32[:8])
        np.testing.assert_allclose(z32[0::2], -z32[1::2])
        np.testing.assert_allclose(u32[0::2] + u32[1::2], 1.0)

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

    def test_nested_stratified_expert_plans_have_exact_prefixes(self):
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
                exact_kg_sampling_mode="stratified_expert_nested",
                seed=9,
            ),
        )
        algorithm.task_ensemble = Ensemble()
        z8, u8, w8 = algorithm._exact_kg_sample_plan(8)
        z32, u32, w32 = algorithm._exact_kg_sample_plan(32)
        self.assertEqual(z8.shape, (24, 2))
        self.assertEqual(z32.shape, (96, 2))
        for expert in range(3):
            np.testing.assert_allclose(
                z8[expert * 8:(expert + 1) * 8],
                z32[expert * 32:expert * 32 + 8],
            )
        np.testing.assert_allclose(z8[0::2], -z8[1::2])
        np.testing.assert_allclose(
            [np.sum(w8[index * 8:(index + 1) * 8])
             for index in range(3)],
            [0.2, 0.3, 0.5],
        )
        np.testing.assert_allclose(
            [np.sum(w32[index * 32:(index + 1) * 32])
             for index in range(3)],
            [0.2, 0.3, 0.5],
        )
        np.testing.assert_allclose(u8[:8], np.full(8, 0.1))
        np.testing.assert_allclose(u32[:32], np.full(32, 0.1))

    def test_stratified_plan_integrates_mean_variance_product_posterior(self):
        class Ensemble:
            @staticmethod
            def predictive_selector_weights():
                return np.asarray([0.08, 0.32, 0.12, 0.48], dtype=float)

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
        self.assertEqual(z_rows.shape, (8, 2))
        np.testing.assert_allclose(
            uniforms,
            [0.04, 0.04, 0.24, 0.24, 0.46, 0.46, 0.76, 0.76],
        )
        np.testing.assert_allclose(
            weights,
            [0.04, 0.04, 0.16, 0.16, 0.06, 0.06, 0.24, 0.24],
        )
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
                exact_kg_jobs=8,
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
        self.assertEqual(forked._last_exact_kg_parallel_workers, 8)
        self.assertEqual(forked._last_exact_kg_chunks_per_candidate, 2)

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

    def test_terminal_replication_kg_is_reproducible_and_side_effect_free(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        config = SingleOLHKGConfig(
            N=7,
            n0=4,
            K1=4,
            K2=0,
            exact_kg_terminal_mode="bayes_risk",
            exact_kg_jobs=1,
            finalist_terminal_mc_samples=2,
            finalist_replication_policy="terminal_kg_1step",
            seed=223,
        )
        algorithm = SingleOLHKGAlgorithm(problem, config)
        algorithm._fit_initial_belief(algorithm._initial_samples())
        arms = [(0, 0, 0), (20, 20, 20)]
        terminal_pool = arms + [(10, 10, 10)]
        history_before = copy.deepcopy(algorithm.history)
        observations_before = copy.deepcopy(algorithm.observations)
        first, first_info = algorithm._terminal_replication_kg_candidate(
            arms, terminal_pool, depth=1, stage=4)
        second, second_info = algorithm._terminal_replication_kg_candidate(
            arms, terminal_pool, depth=1, stage=4)
        self.assertEqual(first, second)
        np.testing.assert_allclose(
            first_info["terminal_kg_expected_values"],
            second_info["terminal_kg_expected_values"],
            rtol=0.0,
            atol=1e-12,
        )
        self.assertEqual(len(algorithm.history), len(history_before))
        for (x_after, y_after), (x_before, y_before) in zip(
            algorithm.history, history_before
        ):
            self.assertEqual(x_after, x_before)
            np.testing.assert_allclose(y_after, y_before)
        self.assertEqual(set(algorithm.observations), set(observations_before))
        for key in observations_before:
            np.testing.assert_allclose(
                algorithm.observations[key], observations_before[key])
        self.assertFalse(first_info["terminal_kg_target_oracle_used"])

    def test_tcb_v2_terminal_kg_updates_same_lexicographic_certificate(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        config = SingleOLHKGConfig(
            N=7,
            n0=4,
            K1=4,
            K2=0,
            tcb_v2_mode="certified",
            finalist_terminal_value_mode="certified_lexicographic",
            exact_kg_jobs=1,
            finalist_terminal_mc_samples=2,
            finalist_replication_policy="terminal_kg_1step",
            seed=227,
        )
        algorithm = SingleOLHKGAlgorithm(problem, config)
        algorithm._fit_initial_belief(algorithm._initial_samples())
        arms = [(7, 7, 7), (13, 13, 13)]
        initial_counts = {
            point: len(algorithm.observations.get(point, []))
            for point in arms
        }

        def fake_tcb(candidates, *, observations=None, **kwargs):
            del kwargs
            observations = algorithm.observations if observations is None else observations
            upper = []
            for candidate in candidates:
                point = tuple(int(v) for v in candidate)
                base = 1.0 if point == arms[0] else 0.1
                added = len(observations.get(point, [])) - initial_counts.get(point, 0)
                upper.append(base - 0.2 * added)
            upper = np.asarray(upper, dtype=float)
            return {
                "mean": upper - 0.05,
                "upper": upper,
                "adapter_diagnostics": {"effective_dimension": 2},
                "pilot_points": len(observations),
                "target_oracle_used": False,
            }

        algorithm._tcb_v2_margin_many = fake_tcb
        selected, diagnostics = algorithm._terminal_replication_kg_candidate(
            arms,
            arms,
            depth=1,
            stage=4,
        )
        self.assertEqual(selected, arms[1])
        self.assertEqual(
            diagnostics["terminal_kg_value_mode"],
            "certified_lexicographic",
        )
        self.assertEqual(
            np.asarray(diagnostics["terminal_kg_expected_values"]).shape,
            (2, 3),
        )
        self.assertEqual(diagnostics["terminal_kg_selected_index"], 1)
        self.assertFalse(diagnostics["terminal_kg_target_oracle_used"])

    def test_tcb_v2_main_exact_kg_uses_same_fantasy_updated_lexicographic_value(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        config = SingleOLHKGConfig(
            N=6,
            n0=4,
            K1=2,
            K2=0,
            acquisition_mode="exact_mc",
            exact_kg_mc_samples=2,
            exact_kg_jobs=1,
            exact_kg_sampling_mode="antithetic",
            exact_kg_terminal_mode="tcb_certified_lexicographic",
            tcb_v2_mode="certified",
            seed=229,
        )
        algorithm = SingleOLHKGAlgorithm(problem, config)
        algorithm._fit_initial_belief(algorithm._initial_samples())
        arms = [(7, 7, 7), (13, 13, 13)]
        initial_counts = {
            point: len(algorithm.observations.get(point, []))
            for point in arms
        }

        def fake_tcb(candidates, *, observations=None, **kwargs):
            del kwargs
            observations = algorithm.observations if observations is None else observations
            upper = []
            for candidate in candidates:
                point = tuple(int(v) for v in candidate)
                base = 0.8 if point == arms[0] else 0.1
                added = len(observations.get(point, [])) - initial_counts.get(point, 0)
                upper.append(base - 0.2 * added)
            upper = np.asarray(upper, dtype=float)
            return {
                "mean": upper - 0.05,
                "upper": upper,
                "adapter_diagnostics": {"effective_dimension": 2},
                "pilot_points": len(observations),
                "target_oracle_used": False,
            }

        algorithm._tcb_v2_margin_many = fake_tcb
        scores = algorithm._exact_posterior_update_scores(arms, arms)
        self.assertGreater(scores[1], scores[0])
        self.assertEqual(
            np.asarray(algorithm._last_exact_kg_current_value).shape, (3,))
        self.assertEqual(
            np.asarray(algorithm._last_exact_kg_expected_values).shape, (2, 3))
        self.assertLess(
            algorithm._last_exact_kg_expected_values[1, 0],
            algorithm._last_exact_kg_expected_values[0, 0],
        )

    def test_coverage_reserved_frontier_cannot_be_crowded_by_experts(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                N=8,
                n0=4,
                finalist_replication_budget=4,
                finalist_replication_count=2,
                finalist_replication_policy="terminal_kg_1step",
                finalist_replication_expert_stratified=True,
                finalist_frontier_policy="coverage_reserved",
                finalist_terminal_max_arms=4,
                decision_contract_mode="certified_lexicographic",
                seed=233,
            ),
        )
        pool = [
            (index, index, index) for index in range(6)
        ]
        algorithm.task_ensemble = object()
        algorithm._terminal_bayes_risk_components = lambda *args, **kwargs: {
            "objective": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            "risk": np.asarray([0.0, 4.0, 3.0, 2.0, 5.0, 6.0]),
            "expected_violation": np.asarray([4.0, 3.0, 0.0, 2.0, 5.0, 6.0]),
            "nominal_expected_violation": np.asarray(
                [4.0, 3.0, 2.0, 0.0, 5.0, 6.0]),
            "model_disagreement": np.asarray(
                [0.1, 0.2, 0.3, 0.4, 1.0, 0.9]),
            "kl_radius": 0.0,
        }
        algorithm._terminal_certificate_components = lambda *args, **kwargs: {
            "objective": np.arange(6, dtype=float),
            "margin": np.asarray([3.0, -1.0, 2.0, 1.0, 4.0, 5.0]),
            "source": "theory_hvd",
            "tcb_v2": None,
        }
        algorithm._finalist_expert_safety_nominations = lambda candidates: [
            (0.0, 0, "expert_a", 4),
            (0.1, 1, "expert_b", 5),
        ]
        initialization = algorithm._initialize_finalist_replication_targets(
            stage=4, pool=pool)
        self.assertEqual(
            initialization["labels"],
            [
                "minimum_bayes_risk",
                "minimum_certificate_margin",
                "minimum_robust_expected_violation",
                "minimum_nominal_expected_violation",
            ],
        )
        self.assertEqual(initialization["targets"], [
            list(pool[index]) for index in range(4)
        ])
        self.assertEqual(
            initialization["frontier_policy"], "coverage_reserved")
        self.assertEqual(
            initialization["certificate_source"], "theory_hvd")

    def test_shadow_tcb_cannot_change_coherent_terminal_certificate(self):
        class FakeGPR:
            def __init__(self, mean, variance):
                self.mean = float(mean)
                self.variance = float(variance)

            def posterior_mean_many(self, pool):
                return np.full(len(pool), self.mean, dtype=float)

            def posterior_var_many(self, pool):
                return np.full(len(pool), self.variance, dtype=float)

        class FakeVariance:
            @staticmethod
            def predict_certification_variance_many(output_index, pool, problem):
                del output_index, problem
                return np.full(len(pool), 0.01, dtype=float)

        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                tcb_v2_mode="shadow",
                decision_contract_mode="certified_lexicographic",
                seed=234,
            ),
        )
        calls = []

        def fake_tcb(*args, **kwargs):
            calls.append((args, kwargs))
            return {
                "mean": np.asarray([-10.0]),
                "upper": np.asarray([-9.0]),
                "target_oracle_used": False,
            }

        algorithm._tcb_v2_margin_many = fake_tcb
        certificate = algorithm._terminal_certificate_components(
            [FakeGPR(0.2, 0.01), FakeGPR(0.0, 0.01)],
            FakeVariance(),
            [(4, 4, 4)],
        )
        self.assertEqual(certificate["source"], "theory_hvd")
        self.assertEqual(calls, [])
        self.assertEqual(
            algorithm._effective_exact_terminal_mode(),
            "tcb_certified_lexicographic",
        )
        self.assertEqual(
            algorithm._finalist_terminal_value_mode(),
            "certified_lexicographic",
        )

    def test_coherent_recommendation_disables_empirical_reranking(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                N=5,
                n0=4,
                K1=4,
                K2=0,
                decision_contract_mode="certified_lexicographic",
                recommendation_calibration=True,
                recommendation_infeasible_strategy="bayes_risk",
                seed=235,
            ),
        )
        algorithm._fit_initial_belief(algorithm._initial_samples())

        def fail_if_called(*args, **kwargs):
            raise AssertionError("empirical reranker escaped coherent contract")

        algorithm._calibrated_recommendation_index = fail_if_called
        _, diagnostics = algorithm._solve_posterior_recommendation(
            pool=[(0, 0, 0), (10, 10, 10), (20, 20, 20)])
        self.assertTrue(diagnostics["decision_contract_coherent"])
        self.assertFalse(diagnostics["calibrated_recommendation_used"])
        self.assertEqual(
            diagnostics["calibrated_recommendation_reason"],
            "disabled_by_coherent_certificate_contract",
        )
        audit = diagnostics["certification_margin_decomposition"]
        self.assertEqual(audit["schema_version"], 1)
        self.assertEqual(audit["n_pool"], 3)
        selected = audit["selected"]["final_certificate"]
        reconstructed = (
            selected["mean_minus_tau"]
            + selected["epistemic_radius"]
            + selected["aleatoric_radius"]
            + selected["extra_guard"]
        )
        self.assertAlmostEqual(selected["margin"], reconstructed)
        self.assertAlmostEqual(
            selected["margin"], diagnostics["posterior_chance_margin"])

    def test_decision_contract_audit_distinguishes_closed_from_legacy(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        closed = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                N=5,
                n0=4,
                K1=4,
                K2=0,
                acquisition_mode="exact_mc",
                exact_kg_mc_samples=2,
                decision_contract_mode="certified_lexicographic",
                finalist_replication_budget=0,
                finalist_empirical_override="off",
                seed=236,
            ),
        ).run()["finalist_replication"]
        self.assertTrue(closed["sampling_terminal_contract_closed"])
        self.assertTrue(closed["recommendation_override_closed"])
        self.assertTrue(closed["mathematically_closed"])

        legacy = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                N=5,
                n0=4,
                K1=4,
                K2=0,
                acquisition_mode="exact_mc",
                exact_kg_mc_samples=2,
                finalist_replication_budget=1,
                finalist_replication_policy="commit_before_switch",
                finalist_empirical_override="legacy",
                decision_contract_mode="legacy",
                seed=237,
            ),
        ).run()["finalist_replication"]
        self.assertFalse(legacy["sampling_terminal_contract_closed"])
        self.assertFalse(legacy["recommendation_override_closed"])
        self.assertFalse(legacy["mathematically_closed"])

    @unittest.skipUnless(
        "fork" in multiprocessing.get_all_start_methods(),
        "terminal rollout process backend requires fork",
    )
    def test_tcb_v2_depth3_flattened_fork_matches_serial(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        base = SingleOLHKGConfig(
            N=7,
            n0=4,
            K1=4,
            K2=0,
            tcb_v2_mode="certified",
            finalist_terminal_value_mode="certified_lexicographic",
            finalist_terminal_mc_samples=2,
            finalist_replication_policy="terminal_kg_depth3",
            seed=228,
        )
        serial_config = copy.deepcopy(base)
        serial_config.exact_kg_jobs = 1
        fork_config = copy.deepcopy(base)
        fork_config.exact_kg_jobs = 8
        fork_config.exact_kg_parallel_backend = "process_fork"
        serial = SingleOLHKGAlgorithm(problem, serial_config)
        forked = SingleOLHKGAlgorithm(problem, fork_config)
        samples = serial._initial_samples()
        serial._fit_initial_belief(samples)
        forked._fit_initial_belief(samples)
        arms = [(7, 7, 7), (13, 13, 13)]

        def attach_tcb(algorithm):
            initial_counts = {
                point: len(algorithm.observations.get(point, []))
                for point in arms
            }

            def fake_tcb(candidates, *, observations=None, **kwargs):
                del kwargs
                observations = (
                    algorithm.observations
                    if observations is None else observations
                )
                upper = []
                for candidate in candidates:
                    point = tuple(int(v) for v in candidate)
                    base_margin = 1.0 if point == arms[0] else 0.1
                    added = (
                        len(observations.get(point, []))
                        - initial_counts.get(point, 0)
                    )
                    upper.append(base_margin - 0.2 * added)
                upper = np.asarray(upper, dtype=float)
                return {
                    "mean": upper - 0.05,
                    "upper": upper,
                    "adapter_diagnostics": {"effective_dimension": 2},
                    "pilot_points": len(observations),
                    "target_oracle_used": False,
                }

            algorithm._tcb_v2_margin_many = fake_tcb

        attach_tcb(serial)
        attach_tcb(forked)
        serial_selected, serial_info = (
            serial._terminal_replication_kg_candidate(
                arms, arms, depth=3, stage=4))
        fork_selected, fork_info = (
            forked._terminal_replication_kg_candidate(
                arms, arms, depth=3, stage=4))
        self.assertEqual(serial_selected, fork_selected)
        np.testing.assert_allclose(
            serial_info["terminal_kg_expected_values"],
            fork_info["terminal_kg_expected_values"],
            rtol=0.0,
            atol=1e-12,
        )
        self.assertEqual(fork_info["terminal_kg_jobs"], 8)

    @unittest.skipUnless(
        "fork" in multiprocessing.get_all_start_methods(),
        "terminal rollout process backend requires fork",
    )
    def test_terminal_replication_fork_matches_serial(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        base = SingleOLHKGConfig(
            N=7,
            n0=4,
            K1=4,
            K2=0,
            exact_kg_terminal_mode="bayes_risk",
            finalist_terminal_mc_samples=2,
            finalist_replication_policy="terminal_kg_1step",
            seed=224,
        )
        serial_config = copy.deepcopy(base)
        serial_config.exact_kg_jobs = 1
        fork_config = copy.deepcopy(base)
        fork_config.exact_kg_jobs = 2
        fork_config.exact_kg_parallel_backend = "process_fork"
        serial = SingleOLHKGAlgorithm(problem, serial_config)
        forked = SingleOLHKGAlgorithm(problem, fork_config)
        samples = serial._initial_samples()
        serial._fit_initial_belief(samples)
        forked._fit_initial_belief(samples)
        arms = [(0, 0, 0), (20, 20, 20)]
        terminal_pool = arms + [(10, 10, 10)]
        serial_selected, serial_info = (
            serial._terminal_replication_kg_candidate(
                arms, terminal_pool, depth=1, stage=4))
        fork_selected, fork_info = (
            forked._terminal_replication_kg_candidate(
                arms, terminal_pool, depth=1, stage=4))
        self.assertEqual(serial_selected, fork_selected)
        np.testing.assert_allclose(
            serial_info["terminal_kg_expected_values"],
            fork_info["terminal_kg_expected_values"],
            rtol=0.0,
            atol=1e-12,
        )

    @unittest.skipUnless(
        "fork" in multiprocessing.get_all_start_methods(),
        "terminal rollout process backend requires fork",
    )
    def test_terminal_depth3_flattened_fork_matches_serial(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        base = SingleOLHKGConfig(
            N=7,
            n0=4,
            K1=4,
            K2=0,
            exact_kg_terminal_mode="bayes_risk",
            finalist_terminal_mc_samples=2,
            finalist_replication_policy="terminal_kg_depth3",
            seed=225,
        )
        serial_config = copy.deepcopy(base)
        serial_config.exact_kg_jobs = 1
        fork_config = copy.deepcopy(base)
        fork_config.exact_kg_jobs = 8
        fork_config.exact_kg_parallel_backend = "process_fork"
        serial = SingleOLHKGAlgorithm(problem, serial_config)
        forked = SingleOLHKGAlgorithm(problem, fork_config)
        samples = serial._initial_samples()
        serial._fit_initial_belief(samples)
        forked._fit_initial_belief(samples)
        arms = [(0, 0, 0), (20, 20, 20)]
        terminal_pool = arms + [(10, 10, 10)]
        serial_selected, serial_info = (
            serial._terminal_replication_kg_candidate(
                arms, terminal_pool, depth=3, stage=4))
        fork_selected, fork_info = (
            forked._terminal_replication_kg_candidate(
                arms, terminal_pool, depth=3, stage=4))
        self.assertEqual(serial_selected, fork_selected)
        self.assertEqual(fork_info["terminal_kg_jobs"], 8)
        np.testing.assert_allclose(
            serial_info["terminal_kg_expected_values"],
            fork_info["terminal_kg_expected_values"],
            rtol=0.0,
            atol=1e-12,
        )

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
