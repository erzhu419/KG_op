import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from representation.task_posterior import (  # noqa: E402
    FiniteTaskLatentPosterior,
    FiniteTaskModelEnsemble,
    FiniteTaskPosterior,
    FiniteTaskSensitivityPosterior,
    TaskExpertState,
)
from algorithms.single_olhkg import (  # noqa: E402
    SingleOLHKGAlgorithm,
    SingleOLHKGConfig,
)
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from representation.meta_prior import (  # noqa: E402
    LearnedMetaPrior,
    MetaPriorProblemAdapter,
)


class FiniteTaskPosteriorTests(unittest.TestCase):
    def test_joint_task_latent_posterior_learns_structure_sensitivity_coupling(self):
        structure = FiniteTaskPosterior(
            ["broad_error", "local_fit"],
            [0.5, 0.5],
            temperature=1.0,
            temperature_decay=0.0,
            output_score_weights=[0.25, 1.0],
            safe_generalized=True,
        )
        sensitivity = FiniteTaskSensitivityPosterior(
            class_names=("stable", "sensitive"),
            scales=(0.5, 2.0),
            decision_penalties=(2.0, 20.0),
            empirical_trust=(1.0, 0.0),
            prior_weights=(0.5, 0.5),
            temperature=1.0,
            temperature_decay=0.0,
        )
        latent = FiniteTaskLatentPosterior(structure, sensitivity)
        update = latent.update_from_predictive(
            [0.0, 2.0],
            means=np.asarray([
                [0.0, 0.0],
                [0.0, 1.9],
            ]),
            epistemic_vars=np.asarray([
                [0.01, 1.0],
                [0.01, 0.01],
            ]),
            aleatoric_vars=np.full((2, 2), 0.01),
            tau=0.0,
        )
        joint = latent.posterior_weights()
        self.assertAlmostEqual(float(np.sum(joint)), 1.0, places=12)
        self.assertGreater(latent.mutual_information(), 1e-5)
        self.assertGreater(joint[0, 1], joint[0, 0])
        self.assertFalse(update["target_oracle_used"])
        diagnostics = latent.diagnostics()
        self.assertFalse(diagnostics["used_for_decision"])
        self.assertFalse(diagnostics["affects_theory_certificate"])

        clone = latent.clone()
        clone.update_from_predictive(
            [0.0, 0.0],
            means=np.zeros((2, 2)),
            epistemic_vars=np.full((2, 2), 0.1),
            aleatoric_vars=np.full((2, 2), 0.1),
            tau=0.0,
        )
        self.assertFalse(np.allclose(
            clone.posterior_weights(), latent.posterior_weights()))

    def test_expert_ridge_calibration_updates_each_structure_separately(self):
        structure = FiniteTaskPosterior(
            ["broad", "local"],
            [0.5, 0.5],
            temperature=1.0,
            temperature_decay=0.0,
            safe_generalized=True,
        )
        sensitivity = FiniteTaskSensitivityPosterior(
            class_names=("balanced",),
            scales=(1.0,),
            decision_penalties=(5.0,),
            empirical_trust=(0.25,),
            prior_weights=(1.0,),
        )
        latent = FiniteTaskLatentPosterior(
            structure,
            sensitivity,
            calibration_mode="expert_ridge",
            adaptive_bias_prior={
                "status": "test_prior",
                "mean": [0.0, 0.0],
                "precision": [[1.0, 0.0], [0.0, 1.0]],
                "feature_names": ["intercept", "risk"],
            },
        )
        features = np.asarray([[1.0, -0.5], [1.0, -0.5]])
        before_variance = latent.adaptive_bias_variance_many(
            np.full((2, 1), 0.01),
            np.full((2, 1), 0.01),
            features,
        )
        update = latent.update_from_predictive(
            [0.0, 1.0],
            means=np.asarray([[0.0, 0.0], [0.0, 0.8]]),
            epistemic_vars=np.full((2, 2), 0.01),
            aleatoric_vars=np.full((2, 2), 0.01),
            tau=0.0,
            bias_features=features,
        )
        after_variance = latent.adaptive_bias_variance_many(
            np.full((2, 1), 0.01),
            np.full((2, 1), 0.01),
            features,
        )
        self.assertEqual(
            update["adaptive_bias_update"]["status"], "updated")
        self.assertTrue(np.all(after_variance < before_variance))
        self.assertFalse(np.allclose(
            latent.adaptive_bias_mean[0],
            latent.adaptive_bias_mean[1],
        ))
        self.assertTrue(np.all(latent.adaptive_bias_n_updates == 1))
        self.assertTrue(np.all(
            latent.adaptive_bias_kl_by_structure() >= 0.0))
        self.assertGreater(latent.kl_from_prior(safe=True), 0.0)
        diagnostics = latent.diagnostics()
        self.assertEqual(diagnostics["calibration_mode"], "expert_ridge")
        self.assertTrue(diagnostics[
            "bias_covariance_affects_theory_certificate"])
        self.assertFalse(diagnostics[
            "bias_mean_affects_theory_certificate"])

        clone = latent.clone()
        clone.update_from_predictive(
            [0.0, -0.5],
            means=np.zeros((2, 2)),
            epistemic_vars=np.full((2, 2), 0.01),
            aleatoric_vars=np.full((2, 2), 0.01),
            tau=0.0,
            bias_features=features,
        )
        self.assertFalse(np.allclose(
            clone.adaptive_bias_mean, latent.adaptive_bias_mean))

    def test_meta_coherence_audit_uses_only_surrogate_views(self):
        class VectorModel:
            def __init__(self, means, variances):
                self.means = np.asarray(means, dtype=float)
                self.variances = np.asarray(variances, dtype=float)

            def posterior_mean_many(self, X):
                return self.means[[int(x[0]) for x in X]]

            def posterior_var_many(self, X):
                return self.variances[[int(x[0]) for x in X]]

        class VectorVariance:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=float)

            def predict_variance_many(self, output_index, X, problem):
                del output_index, problem
                return self.values[[int(x[0]) for x in X]]

            def predict_certification_variance_many(
                self, output_index, X, problem,
            ):
                return self.predict_variance_many(output_index, X, problem)

            @staticmethod
            def diagnostics():
                return {
                    "cumulative_active": {"1": True},
                    "cumulative_provider_active": {"1": True},
                }

        class Provider:
            @staticmethod
            def cumulative_risk_provider_status():
                return {
                    "status": "available",
                    "coordinate": "psi=(A,N)",
                    "source_domains": ["source-a", "source-b"],
                    "target_data_used": False,
                }

        posterior = FiniteTaskPosterior(
            ["risk_aligned_coordinate", "ordered_cumulative"],
            [0.6, 0.4],
            safe_generalized=True,
        )
        sensitivity = FiniteTaskSensitivityPosterior(
            class_names=("balanced",),
            scales=(1.0,),
            decision_penalties=(5.0,),
            empirical_trust=(0.25,),
            prior_weights=(1.0,),
        )
        states = [
            TaskExpertState(
                "risk_aligned_coordinate",
                [
                    VectorModel([0.0, 1.0], [0.01, 0.01]),
                    VectorModel([-1.0, 1.0], [0.01, 0.01]),
                ],
                VectorVariance([0.01, 0.01]),
                Provider(),
            ),
            TaskExpertState(
                "ordered_cumulative",
                [
                    VectorModel([0.2, 0.8], [0.01, 0.01]),
                    VectorModel([-0.8, 0.8], [0.01, 0.01]),
                ],
                VectorVariance([0.01, 0.01]),
                Provider(),
            ),
        ]
        ensemble = FiniteTaskModelEnsemble(
            states,
            posterior,
            sensitivity_posterior=sensitivity,
            kl_radius_numerator=0.0,
            maximum_kl_radius=0.0,
        )
        audit = ensemble.meta_coherence_diagnostics(
            [(0,), (1,)],
            tau=0.0,
            alpha=0.05,
            beta_g=2.0,
            algorithm_selected_x=(0,),
        )
        self.assertEqual(audit["status"], "audited")
        self.assertEqual(audit["algorithm_selected_index"], 0)
        self.assertEqual(audit["algorithm_selected_x"], [0])
        self.assertEqual(
            len(audit["robust_reference_selected_x"]), 1)
        self.assertEqual(len(audit["joint_risk_selected_x"]), 1)
        self.assertAlmostEqual(
            audit["selected_candidate_expert_support_mass"], 1.0)
        self.assertAlmostEqual(audit["cumulative_hvd_active_mass"], 1.0)
        self.assertTrue(audit["source_domain_sets_consistent"])
        self.assertFalse(audit["used_for_decision"])
        self.assertFalse(audit["target_oracle_used"])

        del ensemble.task_latent_posterior
        upgraded = ensemble.diagnostics()["task_latent_posterior"]
        self.assertEqual(upgraded["status"], "initialized")
        self.assertFalse(upgraded["used_for_decision"])

    def test_authoritative_joint_latent_controls_loss_without_relaxing_scale(self):
        class EmptyVariance:
            @staticmethod
            def diagnostics():
                return {"status": "empty"}

        structure = FiniteTaskPosterior(
            ["broad", "local"],
            [0.5, 0.5],
            temperature=1.0,
            temperature_decay=0.0,
            safe_generalized=True,
        )
        sensitivity = FiniteTaskSensitivityPosterior(
            class_names=("stable", "sensitive"),
            scales=(0.5, 2.0),
            decision_penalties=(2.0, 20.0),
            empirical_trust=(1.0, 0.0),
            prior_weights=(0.5, 0.5),
        )
        ensemble = FiniteTaskModelEnsemble(
            [
                TaskExpertState("broad", [], EmptyVariance(), None),
                TaskExpertState("local", [], EmptyVariance(), None),
            ],
            structure,
            sensitivity_posterior=sensitivity,
            task_latent_inference_mode="authoritative",
        )
        ensemble._task_latent().update_from_predictive(
            [0.0, 2.0],
            means=np.asarray([[0.0, 0.0], [0.0, 1.9]]),
            epistemic_vars=np.asarray([[0.01, 1.0], [0.01, 0.01]]),
            aleatoric_vars=np.full((2, 2), 0.01),
            tau=0.0,
        )
        self.assertTrue(ensemble.task_latent_authoritative)
        self.assertAlmostEqual(
            float(np.sum(ensemble.structure_weights())), 1.0)
        self.assertTrue(np.all(
            ensemble._task_latent()
            .conditional_epistemic_scale_squared() >= 1.0
        ))
        risk = ensemble._task_latent().positive_margin_decision_risk_many(
            np.asarray([[0.1, -0.2], [0.2, -0.1]]),
            np.full((2, 2), 0.1),
            np.full((2, 2), 0.02),
            tau=0.0,
            z_alpha=1.64,
        )
        self.assertTrue(np.all(
            risk["posterior_expected_decision_loss"] >= 0.0))
        diagnostics = ensemble.diagnostics()
        self.assertTrue(diagnostics["task_latent_authoritative"])
        self.assertTrue(
            diagnostics["task_latent_posterior"]["used_for_decision"])
        clone = ensemble.clone()
        np.testing.assert_allclose(
            clone.inference_weights(), ensemble.inference_weights())

    def test_sensitivity_class_is_learned_from_prequential_residuals(self):
        stable = FiniteTaskSensitivityPosterior(
            prior_weights=[1.0, 1.0, 1.0],
            temperature=1.0,
            temperature_decay=0.0,
        )
        stable.update_from_predictive(
            observation=0.05,
            mean=0.0,
            epistemic_var=1.0,
            aleatoric_var=0.01,
            tau=0.0,
        )
        self.assertEqual(
            int(np.argmax(stable.posterior_weights())), 0)

        sensitive = FiniteTaskSensitivityPosterior(
            prior_weights=[1.0, 1.0, 1.0],
            temperature=1.0,
            temperature_decay=0.0,
        )
        update = sensitive.update_from_predictive(
            observation=4.0,
            mean=0.0,
            epistemic_var=1.0,
            aleatoric_var=0.01,
            tau=0.0,
        )
        self.assertEqual(
            int(np.argmax(sensitive.posterior_weights())), 2)
        self.assertGreater(
            sensitive.expected_decision_penalty(),
            stable.expected_decision_penalty(),
        )
        self.assertFalse(update["target_oracle_used"])
        self.assertFalse(
            sensitive.diagnostics()["affects_theory_certificate"])

    def test_signed_bias_class_learns_direction_without_relaxing_certificate(self):
        posterior = FiniteTaskSensitivityPosterior(
            class_names=("negative_bias", "positive_bias"),
            scales=(1.0, 1.0),
            biases=(-1.0, 1.0),
            decision_penalties=(5.0, 5.0),
            empirical_trust=(0.25, 0.25),
            prior_weights=(0.5, 0.5),
            temperature=1.0,
            temperature_decay=0.0,
        )
        posterior.update_from_predictive(
            observation=-1.0,
            mean=0.0,
            epistemic_var=1.0,
            aleatoric_var=0.0,
            tau=0.0,
        )
        self.assertGreater(
            posterior.posterior_weights()[0],
            posterior.posterior_weights()[1],
        )
        risk = posterior.posterior_violation_decision_risk(
            [0.0], 1.0, [0.0], tau=0.0)
        self.assertLess(
            risk["class_violation_probability"][0, 0],
            risk["class_violation_probability"][0, 1],
        )
        self.assertFalse(risk["affects_theory_certificate"])

    def test_functional_bias_profile_uses_risk_features(self):
        posterior = FiniteTaskSensitivityPosterior(
            class_names=("decreasing", "increasing"),
            scales=(1.0, 1.0),
            biases=(0.0, 0.0),
            bias_coefficients=((-1.0, 0.5), (1.0, -0.5)),
            bias_feature_names=("intercept", "risk"),
            decision_penalties=(5.0, 5.0),
            empirical_trust=(0.25, 0.25),
            prior_weights=(0.5, 0.5),
            temperature=1.0,
            temperature_decay=0.0,
        )
        with self.assertRaises(ValueError):
            posterior.update_from_predictive(
                -1.0, 0.0, 1.0, 0.0, tau=0.0)
        posterior.update_from_predictive(
            -1.0,
            0.0,
            1.0,
            0.0,
            tau=0.0,
            bias_features=(1.0, 0.0),
        )
        self.assertGreater(
            posterior.posterior_weights()[0],
            posterior.posterior_weights()[1],
        )
        self.assertEqual(
            posterior.diagnostics()["bias_feature_names"],
            ["intercept", "risk"],
        )
        np.testing.assert_allclose(
            posterior.bias_offsets(2.0, bias_features=(1.0, 0.0)),
            [-2.0, 2.0],
        )

    def test_latent_sensitivity_defines_posterior_violation_loss(self):
        stable = FiniteTaskSensitivityPosterior(
            prior_weights=[1.0, 0.0, 0.0])
        sensitive = FiniteTaskSensitivityPosterior(
            prior_weights=[0.0, 0.0, 1.0])
        stable_risk = stable.posterior_violation_decision_risk(
            [-0.2, 0.1], 0.1, [1.0, 1.0], tau=0.0)
        sensitive_risk = sensitive.posterior_violation_decision_risk(
            [-0.2, 0.1], 0.1, [1.0, 1.0], tau=0.0)
        self.assertGreater(
            sensitive_risk["posterior_expected_decision_risk"][0],
            stable_risk["posterior_expected_decision_risk"][0],
        )
        self.assertGreater(
            stable_risk["posterior_violation_probability"][1],
            stable_risk["posterior_violation_probability"][0],
        )
        cumulative_risk = stable.posterior_violation_decision_risk(
            [-0.2], 0.1, [1.0], tau=0.0,
            aleatoric_variance=[0.2],
        )
        self.assertGreater(
            cumulative_risk["posterior_violation_probability"][0],
            stable_risk["posterior_violation_probability"][0],
        )
        self.assertFalse(sensitive_risk["affects_theory_certificate"])

    def test_predictive_evidence_moves_mass_to_supported_expert(self):
        posterior = FiniteTaskPosterior(
            ["supported", "misspecified", "null"],
            [0.4, 0.4, 0.2],
            temperature=1.0,
            temperature_decay=0.0,
            output_score_weights=[0.25, 1.0],
        )
        before = posterior.posterior_weights()
        update = posterior.update_from_predictive(
            [0.05, -0.02],
            means=np.array([
                [0.0, 0.0],
                [1.5, 1.0],
                [0.2, 0.3],
            ]),
            epistemic_vars=np.full((3, 2), 0.02),
            aleatoric_vars=np.full((3, 2), 0.01),
            tau=0.0,
        )
        after = posterior.posterior_weights()
        self.assertGreater(after[0], before[0])
        self.assertLess(after[1], before[1])
        self.assertAlmostEqual(float(np.sum(after)), 1.0, places=12)
        self.assertFalse(update["target_oracle_used"])

    def test_safe_generalized_posterior_separates_fit_from_decision(self):
        posterior = FiniteTaskPosterior(
            ["objective_fit", "boundary_fit"],
            [0.5, 0.5],
            temperature=1.0,
            temperature_decay=0.0,
            output_score_weights=[4.0, 1.0],
            safe_generalized=True,
            safe_boundary_score_weight=1.0,
            safe_pairwise_score_weight=1.0,
        )
        update = posterior.update_from_predictive(
            [0.0, -0.5],
            means=np.asarray([
                [0.0, 0.4],
                [2.0, -0.5],
            ]),
            epistemic_vars=np.full((2, 2), 0.01),
            aleatoric_vars=np.full((2, 2), 0.01),
            tau=0.0,
            safe_pairwise_log_score=[-4.0, -0.01],
            safe_pairwise_pairs=1,
            safe_pairwise_effective_weight=0.8,
        )
        predictive = posterior.posterior_weights()
        safe = posterior.safe_posterior_weights()
        self.assertGreater(predictive[0], predictive[1])
        self.assertGreater(safe[1], safe[0])
        np.testing.assert_allclose(posterior.decision_weights(), safe)
        self.assertEqual(update["safe_pairwise_pairs"], 1)
        self.assertAlmostEqual(
            update["safe_pairwise_effective_weight"], 0.8)
        self.assertFalse(update["target_oracle_used"])

    def test_safe_pairwise_update_is_normalized_and_clone_isolated(self):
        posterior = FiniteTaskPosterior(
            ["wrong_order", "right_order"],
            [0.5, 0.5],
            temperature=0.5,
            temperature_decay=0.0,
            safe_generalized=True,
            safe_boundary_score_weight=0.0,
            safe_pairwise_score_weight=2.0,
        )
        clone = posterior.clone()
        clone.update_from_predictive(
            [0.0, 0.1],
            means=np.zeros((2, 2)),
            epistemic_vars=np.full((2, 2), 0.1),
            aleatoric_vars=np.full((2, 2), 0.1),
            tau=0.0,
            safe_pairwise_log_score=[np.log(1e-6), np.log(0.9)],
            safe_pairwise_pairs=3,
            safe_pairwise_effective_weight=1.5,
        )
        np.testing.assert_allclose(
            posterior.safe_posterior_weights(), [0.5, 0.5])
        self.assertGreater(
            clone.safe_posterior_weights()[1],
            clone.safe_posterior_weights()[0],
        )
        self.assertAlmostEqual(
            float(np.sum(clone.safe_posterior_weights())), 1.0, places=12)
        diagnostics = clone.diagnostics()
        self.assertTrue(diagnostics["safe_generalized"])
        self.assertEqual(
            diagnostics["last_update"]["safe_pairwise_pairs"], 3)

    def test_objective_uses_predictive_mass_and_constraint_uses_safe_mass(self):
        class ConstantModel:
            def __init__(self, mean):
                self.mean = float(mean)

            def posterior_mean_many(self, X):
                return np.full(len(X), self.mean, dtype=float)

            @staticmethod
            def posterior_var_many(X):
                return np.zeros(len(X), dtype=float)

        class ZeroVariance:
            @staticmethod
            def predict_variance_many(output_index, X, problem):
                del output_index, problem
                return np.zeros(len(X), dtype=float)

            @staticmethod
            def predict_certification_variance_many(output_index, X, problem):
                del output_index, problem
                return np.zeros(len(X), dtype=float)

        posterior = FiniteTaskPosterior(
            ["predictive", "safe"],
            [0.5, 0.5],
            safe_generalized=True,
        )
        posterior._log_weights = np.log([0.9, 0.1])
        posterior._log_safe_weights = np.log([0.1, 0.9])
        ensemble = FiniteTaskModelEnsemble(
            [
                TaskExpertState(
                    "predictive",
                    [ConstantModel(0.0), ConstantModel(0.0)],
                    ZeroVariance(),
                    object(),
                ),
                TaskExpertState(
                    "safe",
                    [ConstantModel(10.0), ConstantModel(10.0)],
                    ZeroVariance(),
                    object(),
                ),
            ],
            posterior,
            kl_radius_numerator=0.0,
            maximum_kl_radius=0.0,
        )
        X = [(0,), (1,)]
        objective = ensemble.mixture_moments_many(
            0, X, certification=False)
        constraint = ensemble.mixture_moments_many(
            1, X, certification=True)
        np.testing.assert_allclose(objective.mean, [1.0, 1.0])
        np.testing.assert_allclose(constraint.mean, [9.0, 9.0])

    def test_prequential_pairwise_score_rewards_correct_boundary_order(self):
        posterior = FiniteTaskPosterior(
            ["wrong", "right"],
            [0.5, 0.5],
            safe_generalized=True,
        )
        ensemble = FiniteTaskModelEnsemble(
            [
                TaskExpertState("wrong", [], None, None),
                TaskExpertState("right", [], None, None),
            ],
            posterior,
            safe_pairwise_max_history=4,
            safe_pairwise_probability_floor=1e-6,
            safe_history=[{
                "x": [0],
                "observation": -1.0,
                "means": [1.0, -1.0],
                "variances": [0.2, 0.2],
                "observation_source": "budgeted_target_evaluation",
                "target_oracle_used": False,
            }],
        )
        score, diagnostics = ensemble._safe_pairwise_log_score(
            1.0,
            means=np.asarray([[0.0, -1.0], [0.0, 1.0]]),
            epistemic=np.full((2, 2), 0.1),
            aleatoric=np.full((2, 2), 0.1),
            tau=0.0,
            constraint_index=1,
        )
        self.assertGreater(score[1], score[0])
        self.assertTrue(np.all(np.isfinite(score)))
        self.assertTrue(np.all(score <= 0.0))
        self.assertGreaterEqual(
            float(np.min(score)), np.log(1e-6) - 1e-12)
        self.assertEqual(diagnostics["pair_count"], 1)
        self.assertGreater(diagnostics["effective_weight"], 0.0)

    def test_decision_prior_protection_prevents_finite_sample_collapse(self):
        posterior = FiniteTaskPosterior(
            ["winner", "discarded"],
            [0.5, 0.5],
            decision_prior_protection_numerator=1.0,
            decision_prior_protection_max=0.5,
        )
        posterior._log_weights = np.log([1.0 - 1e-12, 1e-12])
        posterior.n_updates = 16
        decision = posterior.decision_weights()
        self.assertAlmostEqual(posterior.decision_prior_mix(), 0.25)
        self.assertAlmostEqual(float(np.sum(decision)), 1.0, places=12)
        self.assertGreaterEqual(decision[1], 0.125 - 1e-12)

        moments = posterior.mixture_moments(
            means=np.asarray([[0.0], [10.0]]),
            epistemic_vars=np.zeros((2, 1)),
            aleatoric_vars=np.zeros((2, 1)),
        )
        self.assertAlmostEqual(moments.mean[0], 10.0 * decision[1])
        robust_zero = posterior.kl_robust_expectation(
            np.asarray([[0.0], [10.0]]), radius=0.0)
        self.assertAlmostEqual(robust_zero[0], moments.mean[0])
        self.assertEqual(
            posterior.diagnostics()["decision_by_expert"]["discarded"],
            decision[1],
        )

    def test_hierarchical_variance_is_within_plus_between(self):
        posterior = FiniteTaskPosterior(
            ["left", "right"], [0.25, 0.75])
        moments = posterior.mixture_moments(
            means=np.array([[0.0, 2.0], [2.0, 4.0]]),
            epistemic_vars=np.array([[1.0, 0.5], [3.0, 1.5]]),
            aleatoric_vars=np.array([[0.2, 0.4], [0.6, 0.8]]),
        )
        np.testing.assert_allclose(
            moments.epistemic,
            moments.within_epistemic + moments.between_mean,
        )
        np.testing.assert_allclose(
            moments.total,
            moments.epistemic + moments.aleatoric,
        )
        np.testing.assert_allclose(moments.between_mean, [0.75, 0.75])

    def test_kl_robust_moments_cannot_be_more_optimistic(self):
        posterior = FiniteTaskPosterior(
            ["safe", "unsafe", "null"], [0.6, 0.2, 0.2])
        means = np.array([[-0.3], [0.5], [0.1]])
        epistemic = np.array([[0.02], [0.08], [0.04]])
        aleatoric = np.array([[0.01], [0.12], [0.05]])
        nominal = posterior.robust_mixture_moments(
            means, epistemic, aleatoric, radius=0.0)
        robust = posterior.robust_mixture_moments(
            means, epistemic, aleatoric, radius=0.5)
        self.assertGreaterEqual(robust.mean_upper[0], nominal.mean_upper[0])
        self.assertGreaterEqual(
            robust.epistemic_upper[0], nominal.epistemic_upper[0])
        self.assertGreaterEqual(
            robust.aleatoric_upper[0], nominal.aleatoric_upper[0])
        authoritative_center = np.asarray([0.1, 0.8, 0.1])
        centered = posterior.robust_mixture_moments(
            means,
            epistemic,
            aleatoric,
            radius=0.0,
            weights=authoritative_center,
        )
        self.assertAlmostEqual(
            centered.mean_upper[0],
            float(authoritative_center @ means[:, 0]),
        )

    def test_batched_entropic_dual_dominates_sampled_kl_ball(self):
        posterior = FiniteTaskPosterior(
            ["a", "b", "c"], [0.5, 0.3, 0.2],
            robust_dual_grid_size=65,
        )
        values = np.array([
            [-0.5, 0.2, 1.0],
            [0.4, -0.1, 0.7],
            [1.2, 0.8, -0.2],
        ])
        radius = 0.35
        upper = posterior.kl_robust_expectation(values, radius)
        prior = posterior.posterior_weights()
        rng = np.random.default_rng(77)
        samples = rng.dirichlet(np.ones(3), size=20000)
        kl = np.sum(samples * (
            np.log(np.maximum(samples, 1e-300))
            - np.log(prior[None, :])
        ), axis=1)
        admissible = samples[kl <= radius]
        sampled_values = admissible @ values
        self.assertTrue(np.all(
            np.max(sampled_values, axis=0) <= upper + 1e-10))

    def test_pac_bayes_radius_matches_proved_schedule(self):
        posterior = FiniteTaskPosterior(["a", "b"], [0.5, 0.5])
        ensemble = FiniteTaskModelEnsemble(
            [
                TaskExpertState("a", [], None, None),
                TaskExpertState("b", [], None, None),
            ],
            posterior,
            kl_radius_numerator=0.5,
            confidence_delta=0.05,
            maximum_kl_radius=10.0,
        )
        self.assertAlmostEqual(
            ensemble.effective_kl_radius(),
            0.5 + np.log(20.0),
            places=12,
        )

    def test_proposal_mixture_preserves_prior_support_and_budget(self):
        posterior = FiniteTaskPosterior(
            ["dominant", "rare", "null"], [0.7, 0.2, 0.1],
            temperature=2.0,
            temperature_decay=0.0,
        )
        for _ in range(4):
            posterior.update_from_predictive(
                [0.0, 0.0],
                means=np.array([
                    [0.0, 0.0],
                    [2.0, 2.0],
                    [1.0, 1.0],
                ]),
                epistemic_vars=np.full((3, 2), 0.01),
                aleatoric_vars=np.full((3, 2), 0.01),
                tau=0.0,
            )
        epsilon = 0.2
        proposal = posterior.proposal_weights(exploration=epsilon)
        np.testing.assert_array_less(
            epsilon * posterior.prior_weights() - 1e-15,
            proposal,
        )
        allocation = posterior.proposal_allocation(
            11,
            exploration=epsilon,
            minimum_per_expert=1,
        )
        self.assertEqual(sum(allocation.values()), 11)
        self.assertTrue(all(count >= 1 for count in allocation.values()))

    def test_clone_update_does_not_mutate_original(self):
        posterior = FiniteTaskPosterior(["a", "b"], [0.5, 0.5])
        clone = posterior.clone()
        clone.update_from_predictive(
            [0.0, 0.0],
            means=np.array([[0.0, 0.0], [1.0, 1.0]]),
            epistemic_vars=np.full((2, 2), 0.01),
            aleatoric_vars=np.full((2, 2), 0.01),
            tau=0.0,
        )
        np.testing.assert_allclose(
            posterior.posterior_weights(), [0.5, 0.5])
        self.assertFalse(np.allclose(
            clone.posterior_weights(), posterior.posterior_weights()))

    def test_exact_kg_updates_finite_task_posterior_end_to_end(self):
        def problem(name):
            return ScalarizedProblem(make_problem(
                name, d=5, L=30, sigma=0.04))

        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral_hvd",
            spectral_active_dim=3,
            spectral_risk_alignment=True,
            seed=901,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", problem("InventorySupplyChain")),
                ("QueueResourceControl", problem("QueueResourceControl")),
            ],
            n_records_per_domain=8,
            rng=np.random.default_rng(902),
        )
        target = MetaPriorProblemAdapter(
            problem("FactorShockStatePolicyRZDT1"), prior)
        local_specs = target.task_posterior_expert_specs(
            include_local_kernel=True)
        self.assertIn(
            "local_risk_kernel",
            [spec["name"] for spec in local_specs],
        )
        local_basis = target.task_expert_basis_map(
            "local_risk_kernel", output_index=1)
        local_basis_again = target.task_expert_basis_map(
            "local_risk_kernel", output_index=1)
        local_rows = [target.sample_random(np.random.default_rng(seed))
                      for seed in (911, 912, 913)]
        local_features = local_basis.features_many(local_rows)
        np.testing.assert_allclose(
            local_features,
            local_basis_again.features_many(local_rows),
        )
        self.assertEqual(local_features.shape, (3, 6))
        self.assertTrue(np.all(np.isfinite(local_features)))
        self.assertTrue(np.all((0.0 <= local_features) & (local_features <= 1.0)))
        lo, hi = target.int_bounds()
        target.task_boundary_bracket_candidates = lambda n, **_: [
            tuple(np.rint(
                np.asarray(lo, dtype=float)
                + quantile
                * (np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float))
            ).astype(int))
            for quantile in np.linspace(0.2, 0.8, int(n))
        ]
        for spec in target.task_posterior_expert_specs():
            proposals = target.task_expert_proposal_candidates(
                spec["name"],
                n=3,
                rng=np.random.default_rng(910),
                pool_size=32,
            )
            self.assertEqual(len(proposals), 3)
        config_values = dict(
            N=5,
            n0=4,
            K1=2,
            K2=0,
            posterior_pool_size=8,
            posterior_keep=2,
            axis_candidate_count=0,
            state_candidate_count=0,
            eval_pool_size=8,
            evaluate_interval=0,
            use_problem_initial_samples=True,
            use_boundary_initial_samples=False,
            use_recommendation_refinement=False,
            recommendation_axis_oracle=False,
            recommendation_axis_candidate_count=0,
            recommendation_calibration=True,
            recommendation_calibration_min_obs=4,
            recommendation_infeasible_strategy="task_adaptive",
            certification_calibration=False,
            acquisition_mode="exact_mc",
            exact_kg_mc_samples=1,
            exact_kg_jobs=2,
            exact_kg_parallel_backend="process_fork",
            task_posterior_mode="finite",
            task_posterior_safe_generalized=True,
            task_posterior_safe_pairwise_max_history=8,
            task_posterior_sensitivity_mode="fixed",
            task_posterior_local_kernel_expert=True,
            task_posterior_boundary_bracket_fraction=0.5,
            task_posterior_mandatory_universal_count=2,
            task_posterior_candidate_count=5,
            task_posterior_recommendation_count=5,
            task_posterior_proposal_pool_size=32,
            seed=903,
        )
        with tempfile.TemporaryDirectory() as tmp:
            config_values.update({
                "checkpoint_dir": tmp,
                "checkpoint_resume": True,
                "checkpoint_interval": 1,
            })
            algorithm = SingleOLHKGAlgorithm(
                target, SingleOLHKGConfig(**config_values))
            result = algorithm.run()
            diagnostics = result["task_posterior"]
            bayes_components = algorithm._terminal_bayes_risk_components(
                None,
                None,
                algorithm._recommendation_pool(),
                task_ensemble=algorithm.task_ensemble,
            )
            self.assertTrue(np.all(np.isfinite(
                bayes_components["risk"])))
            self.assertTrue(np.all(
                bayes_components["expected_violation"] >= 0.0))
            self.assertGreater(
                bayes_components["kl_radius"], 0.0)
            self.assertIsNotNone(diagnostics)
            self.assertIn(
                "local_risk_kernel",
                diagnostics["posterior"]["expert_names"],
            )
            self.assertEqual(
                result["task_initial_design"]["status"], "generated")
            self.assertFalse(
                result["task_initial_design"]["target_oracle_used"])
            self.assertEqual(diagnostics["posterior"]["n_updates"], 2)
            self.assertEqual(
                diagnostics["sensitivity_posterior"]["n_updates"], 2)
            self.assertEqual(
                diagnostics["sensitivity_posterior"]["class_names"], ["fixed"])
            self.assertEqual(
                len(diagnostics["task_latent_posterior"][
                    "sensitivity_names"]),
                9,
            )
            self.assertEqual(
                len(diagnostics["task_latent_posterior"][
                    "sensitivity_biases"]),
                9,
            )
            self.assertEqual(
                len(diagnostics["task_latent_posterior"][
                    "sensitivity_bias_coefficients"]),
                9,
            )
            self.assertIsNone(
                diagnostics["task_latent_posterior"]
                ["legacy_sensitivity_total_variation"])
            self.assertEqual(diagnostics["pilot_count"], 3)
            self.assertAlmostEqual(
                sum(diagnostics["posterior"]["posterior_weights"]),
                1.0,
                places=10,
            )
            self.assertTrue(
                diagnostics["posterior"]["safe_generalized"])
            self.assertAlmostEqual(
                sum(diagnostics["posterior"]["safe_posterior_weights"]),
                1.0,
                places=10,
            )
            self.assertEqual(diagnostics["safe_history_count"], 2)
            self.assertGreaterEqual(
                algorithm.iteration_log[0]["task_posterior_update"]
                ["posterior"]["safe_pairwise_pairs"],
                1,
            )
            self.assertIn(
                "exact_kg_task_entropy_gain_selected",
                algorithm.iteration_log[0],
            )
            self.assertEqual(
                algorithm.iteration_log[0]["task_expert_proposals"]["status"],
                "generated",
            )
            self.assertFalse(diagnostics["target_oracle_used"])
            self.assertEqual(
                result["task_meta_coherence"]["status"], "audited")
            self.assertFalse(
                result["task_meta_coherence"]["used_for_decision"])
            self.assertTrue(
                result["recommendation_calibration_features_standardized"])
            self.assertGreater(
                result["recommendation_calibration_selected_ridge"], 0.0)
            self.assertTrue(
                result["recommendation_calibration_nested_refit"])
            self.assertTrue(
                result["recommendation_calibration_rank_cap_satisfied"])
            self.assertLessEqual(
                result["recommendation_calibration_effective_rank"],
                result["recommendation_calibration_effective_rank_cap"],
            )
            self.assertGreaterEqual(
                result["task_initial_design"][
                    "boundary_bracket_generated"],
                1,
            )
            self.assertEqual(
                result["task_initial_design"][
                    "mandatory_universal_generated"],
                2,
            )
            first_weights = np.asarray(
                diagnostics["posterior"]["posterior_weights"], dtype=float)
            first_safe_weights = np.asarray(
                diagnostics["posterior"]["safe_posterior_weights"],
                dtype=float,
            )
            checkpoint_state = algorithm._task_ensemble_checkpoint_state()
            self.assertIn("task_latent_posterior", checkpoint_state)
            self.assertIn("sensitivity_posterior", checkpoint_state)
            self.assertEqual(
                checkpoint_state["task_latent_inference_mode"], "shadow")
            self.assertEqual(
                checkpoint_state["task_latent_calibration_mode"],
                "source_profiles",
            )

            resumed_values = dict(config_values)
            resumed_values["N"] = 6
            resumed = SingleOLHKGAlgorithm(
                MetaPriorProblemAdapter(
                    problem("FactorShockStatePolicyRZDT1"), prior),
                SingleOLHKGConfig(**resumed_values),
            )
            resumed_result = resumed.run()
            resumed_diag = resumed_result["task_posterior"]["posterior"]
            self.assertEqual(resumed_diag["n_updates"], 3)
            self.assertEqual(len(resumed.history), 6)
            self.assertEqual(len(resumed.iteration_log), 2)
            checkpoint_before = np.asarray(
                resumed.iteration_log[1]["task_posterior_before"][
                    "posterior_weights"],
                dtype=float,
            )
            np.testing.assert_allclose(
                checkpoint_before, first_weights, atol=0.0, rtol=0.0)
            checkpoint_safe_before = np.asarray(
                resumed.iteration_log[1]["task_posterior_before"]
                ["safe_posterior_weights"],
                dtype=float,
            )
            np.testing.assert_allclose(
                checkpoint_safe_before,
                first_safe_weights,
                atol=0.0,
                rtol=0.0,
            )
            self.assertEqual(
                resumed_result["task_posterior"]["safe_history_count"], 3)

    def test_expert_ridge_calibration_is_cloned_and_updated_by_exact_kg(self):
        def problem(name):
            return ScalarizedProblem(make_problem(
                name, d=5, L=30, sigma=0.04))

        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral_hvd",
            spectral_active_dim=3,
            spectral_risk_alignment=True,
            seed=921,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", problem("InventorySupplyChain")),
                ("QueueResourceControl", problem("QueueResourceControl")),
            ],
            n_records_per_domain=8,
            rng=np.random.default_rng(922),
        )
        target = MetaPriorProblemAdapter(
            problem("FactorShockStatePolicyRZDT1"), prior)
        algorithm = SingleOLHKGAlgorithm(
            target,
            SingleOLHKGConfig(
                N=5,
                n0=4,
                K1=1,
                K2=0,
                posterior_pool_size=6,
                posterior_keep=2,
                axis_candidate_count=0,
                state_candidate_count=0,
                eval_pool_size=6,
                evaluate_interval=0,
                use_problem_initial_samples=True,
                use_boundary_initial_samples=False,
                use_recommendation_refinement=False,
                recommendation_axis_oracle=False,
                recommendation_axis_candidate_count=0,
                recommendation_calibration=False,
                acquisition_mode="exact_mc",
                exact_kg_mc_samples=1,
                exact_kg_jobs=1,
                task_posterior_mode="finite",
                task_posterior_safe_generalized=True,
                task_posterior_sensitivity_mode="fixed",
                task_latent_inference_mode="authoritative",
                task_latent_calibration_mode="expert_ridge",
                seed=923,
            ),
        )
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)
        ensemble = algorithm.task_ensemble
        latent = ensemble._task_latent()
        self.assertTrue(latent.adaptive_bias_enabled)
        self.assertEqual(len(latent.sensitivity_names), 3)
        self.assertTrue(np.all(latent.adaptive_bias_n_updates == 1))
        raw = ensemble.expert_moments_many(
            1, samples[:2], certification=True)
        mixture = ensemble.mixture_moments_many(
            1, samples[:2], certification=True)
        self.assertTrue(np.all(mixture.epistemic >= 0.0))
        calibration_variance = latent.adaptive_bias_variance_many(
            raw[1], raw[2], ensemble.task_bias_features_many(samples[:2]))
        self.assertTrue(np.all(calibration_variance >= 0.0))
        self.assertTrue(np.any(calibration_variance > 0.0))

        clone = ensemble.clone(
            gpr_cloner=algorithm._clone_gpr_for_exact_kg,
            variance_cloner=algorithm._clone_variance_model_for_exact_kg,
        )
        mean_before = clone._task_latent().adaptive_bias_mean.copy()
        y, _ = ensemble.predictive_sample(samples[0], [0.0, 0.0], 0.5)
        clone.update(samples[0], y, tau=target.tau)
        self.assertFalse(np.allclose(
            clone._task_latent().adaptive_bias_mean, mean_before))
        self.assertFalse(np.shares_memory(
            clone._task_latent().adaptive_bias_precision,
            latent.adaptive_bias_precision,
        ))
        scores = algorithm._exact_posterior_update_scores(
            samples[:1], samples[:2])
        self.assertEqual(scores.shape, (1,))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(
            latent.adaptive_bias_n_updates == 1))

    def test_ordered_semiparametric_replacement_updates_sparse_fantasy_gpr(self):
        def problem(name):
            return ScalarizedProblem(make_problem(
                name, d=8, L=30, sigma=0.04))

        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            ordered_cumulative_exposure=True,
            ordered_exposure_max_frequency=6,
            ordered_exposure_active_dim=2,
            ordered_exposure_basis_mode="diagonal_quadratic",
            ordered_exposure_adaptive_sparsity=True,
            ordered_exposure_replace_local_kernel=True,
            ordered_exposure_semiparametric_residual=True,
            seed=931,
        ).fit_from_source_problems(
            [
                ("FactorShockStatePolicyRZDT1", problem(
                    "FactorShockStatePolicyRZDT1")),
                ("QueueResourceControl", problem("QueueResourceControl")),
            ],
            n_records_per_domain=8,
            rng=np.random.default_rng(932),
        )
        target = MetaPriorProblemAdapter(
            problem("InventorySupplyChain"), prior)
        algorithm = SingleOLHKGAlgorithm(
            target,
            SingleOLHKGConfig(
                N=4,
                n0=4,
                K1=2,
                K2=0,
                task_posterior_mode="finite",
                task_posterior_local_kernel_expert=True,
                task_posterior_initial_design=False,
                use_problem_initial_samples=True,
                axis_candidate_count=0,
                state_candidate_count=0,
                acquisition_mode="exact_mc",
                exact_kg_mc_samples=1,
                exact_kg_jobs=1,
                seed=933,
            ),
        )
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)
        names = [state.name for state in algorithm.task_ensemble.states]
        self.assertIn("ordered_semiparametric", names)
        self.assertNotIn("ordered_cumulative", names)
        self.assertNotIn("local_risk_kernel", names)
        ordered = next(
            state for state in algorithm.task_ensemble.states
            if state.name == "ordered_semiparametric")
        for model in ordered.gpr_models:
            self.assertIsNotNone(model._adaptive_sparsity)
            diagnostics = model.adaptive_sparsity_diagnostics()
            self.assertEqual(diagnostics["status"], "fit")
            self.assertLessEqual(
                diagnostics["effective_dimension"],
                diagnostics["max_effective_dimension"] + 1e-8,
            )
        ordered_diagnostics = next(
            expert for expert in algorithm.task_ensemble.diagnostics()["experts"]
            if expert["name"] == "ordered_semiparametric")
        self.assertEqual(
            len(ordered_diagnostics["gpr_adaptive_sparsity"]), 2)
        self.assertTrue(all(
            item["status"] == "fit"
            for item in ordered_diagnostics["gpr_adaptive_sparsity"]
        ))
        self.assertLess(
            ordered_diagnostics["basis"]["ordered_residual_projection"][
                "orthogonality_relative"],
            1e-10,
        )
        self.assertEqual(
            ordered_diagnostics["basis"]["ordered_residual_projection"][
                "residualization_mode"],
            "bounded_coefficient_nullspace",
        )
        candidates = samples[:2]
        scores = algorithm._exact_posterior_update_scores(
            candidates, candidates)
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertEqual(len(algorithm.history), 4)


if __name__ == "__main__":
    unittest.main()
