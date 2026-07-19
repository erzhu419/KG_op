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
    ReplicationVarianceTaskPosterior,
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
from core.gpr import (  # noqa: E402
    ParametricGPR,
    posterior_mixture_weights,
)


class FiniteTaskPosteriorTests(unittest.TestCase):
    @staticmethod
    def _hierarchical_mixture_fixture(
        misspecification_mode="hierarchical_predictive_scale",
        misspecification_ridge=1.0,
    ):
        parent = ParametricGPR(d=2, lambda_i=0.1, prior_var=1.0)
        source = ParametricGPR(d=2, lambda_i=0.1, prior_var=1.0)
        target = ParametricGPR(d=2, lambda_i=0.1, prior_var=1.0)
        source_prior = {
            "name": "source:wrong",
            "mean": np.zeros(parent.p, dtype=float),
            "covariance": 0.01 * np.eye(parent.p, dtype=float),
            "deviation_variance": 0.01,
            "prior_weight": 0.5,
            "diagnostics": {},
        }
        target_prior = {
            "name": "target:null",
            "mean": np.zeros(parent.p, dtype=float),
            "covariance": 10.0 * np.eye(parent.p, dtype=float),
            "deviation_variance": 1.0,
            "prior_weight": 0.5,
            "diagnostics": {},
        }
        parent.set_hierarchical_misspecification_posterior(
            [source, target],
            [source_prior, target_prior],
            [0.5, 0.5],
            [(0, 0), (100, 100)],
            [4.0, 5.0],
            [0.01, 0.01],
            diagnostics={
                "component_names": ["source:wrong", "target:null"],
                "evidence_temperature": 1.0,
                "online_mixture_update_count": 0,
            },
            prior_df=2.0,
            max_scale=20.0,
            misspecification_mode=misspecification_mode,
            misspecification_ridge=misspecification_ridge,
        )
        return parent

    def test_gaussian_loo_score_matches_direct_conditionals(self):
        residual = np.asarray([0.4, -0.2, 0.7], dtype=float)
        covariance = np.asarray([
            [1.4, 0.3, 0.1],
            [0.3, 1.1, 0.2],
            [0.1, 0.2, 0.9],
        ], dtype=float)
        score, diagnostics = ParametricGPR._gaussian_loo_log_score(
            residual, covariance)
        direct = 0.0
        for index in range(len(residual)):
            keep = [row for row in range(len(residual)) if row != index]
            cross = covariance[index, keep]
            block = covariance[np.ix_(keep, keep)]
            conditional_mean = float(
                cross @ np.linalg.solve(block, residual[keep]))
            conditional_variance = float(
                covariance[index, index]
                - cross @ np.linalg.solve(block, cross))
            error = float(residual[index] - conditional_mean)
            direct += -0.5 * (
                np.log(2.0 * np.pi * conditional_variance)
                + error ** 2 / conditional_variance
            )
        self.assertAlmostEqual(score, direct, places=8)
        self.assertEqual(diagnostics["loo_count"], 3)
        self.assertTrue(np.isfinite(
            diagnostics["loo_mean_log_score"]))

    def test_grouped_mixture_updates_conditionals_not_group_masses(self):
        prior = np.asarray([0.4, 0.1, 0.1, 0.4], dtype=float)
        evidence = np.asarray([3.0, -2.0, -4.0, 2.0], dtype=float)
        labels = ["role-a", "role-a", "role-b", "role-b"]
        posterior, masses = ParametricGPR.group_mass_preserving_weights(
            prior,
            evidence,
            labels,
            {"role-a": 0.65, "role-b": 0.35},
            temperature=1.0,
        )

        self.assertAlmostEqual(float(np.sum(posterior)), 1.0, places=12)
        self.assertAlmostEqual(float(np.sum(posterior[:2])), 0.65, places=12)
        self.assertAlmostEqual(float(np.sum(posterior[2:])), 0.35, places=12)
        self.assertEqual(masses, {"role-a": 0.65, "role-b": 0.35})
        self.assertGreater(posterior[0] / posterior[1], prior[0] / prior[1])
        self.assertGreater(posterior[3] / posterior[2], prior[3] / prior[2])

    def test_grouped_mixture_online_update_keeps_assignment_marginal(self):
        parent = ParametricGPR(d=2, lambda_i=0.1, prior_var=1.0)
        components = []
        for intercept in (0.0, 5.0, -5.0, 0.0):
            component = ParametricGPR(d=2, lambda_i=0.1, prior_var=1.0)
            mean = np.zeros(component.p, dtype=float)
            mean[0] = intercept
            component.set_parametric_prior(
                mean,
                0.01,
                0.01 * np.eye(component.p),
            )
            components.append(component)
        names = [
            "source:a|role_assignment=0-1",
            "target:null|role_assignment=0-1",
            "source:b|role_assignment=1-0",
            "target:null|role_assignment=1-0",
        ]
        labels = ["0-1", "0-1", "1-0", "1-0"]
        parent.set_group_mass_preserving_posterior(
            components,
            [0.35, 0.35, 0.15, 0.15],
            labels,
            {"0-1": 0.7, "1-0": 0.3},
            diagnostics={
                "component_names": names,
                "evidence_temperature": 1.0,
                "posterior_target_data_used": False,
            },
        )
        before = np.asarray(parent._finite_mixture_weights, dtype=float)
        parent.update((0, 0), 0.0, 0.01)
        after = np.asarray(parent._finite_mixture_weights, dtype=float)
        diagnostics = parent.source_parametric_prior_diagnostics

        self.assertAlmostEqual(float(np.sum(after[:2])), 0.7, places=12)
        self.assertAlmostEqual(float(np.sum(after[2:])), 0.3, places=12)
        self.assertFalse(np.allclose(before, after))
        self.assertTrue(diagnostics["assignment_group_masses_fixed"])
        self.assertFalse(diagnostics[
            "target_role_assignment_target_labels_used_for_update"])
        self.assertTrue(diagnostics[
            "target_role_assignment_conditional_expert_uses_target_labels"])
        self.assertEqual(
            diagnostics["target_role_assignment_update_scope"],
            "frozen_assignment_marginal_conditional_expert_only",
        )
        self.assertFalse(diagnostics["target_oracle_used_for_group_masses"])

    def test_cross_validated_structure_posterior_refits_online(self):
        parent = ParametricGPR(d=2, lambda_i=0.1, prior_var=1.0)
        good = ParametricGPR(d=2, lambda_i=0.1, prior_var=1.0)
        bad = ParametricGPR(d=2, lambda_i=0.1, prior_var=1.0)
        points = [(0, 0), (100, 0), (0, 100), (100, 100)]
        truth = np.linspace(-0.2, 0.3, parent.p)
        targets = parent.basis_matrix(points) @ truth
        covariance = 0.05 * np.eye(parent.p, dtype=float)
        priors = [
            {
                "name": "source:good|role_assignment=0-1",
                "mean": truth.copy(),
                "covariance": covariance.copy(),
                "deviation_variance": 0.01,
                "prior_weight": 0.5,
                "diagnostics": {},
            },
            {
                "name": "target:null|role_assignment=1-0",
                "mean": -truth,
                "covariance": covariance.copy(),
                "deviation_variance": 0.01,
                "prior_weight": 0.5,
                "diagnostics": {},
            },
        ]
        parent.set_cross_validated_structure_posterior(
            [good, bad],
            priors,
            [0.5, 0.5],
            points,
            targets,
            np.full(len(points), 0.01),
            diagnostics={
                "component_names": [row["name"] for row in priors],
                "evidence_temperature": 1.0,
            },
        )
        initial = parent.source_parametric_prior_diagnostics
        self.assertEqual(
            initial["structure_score_mode"], "loo_predictive")
        self.assertTrue(initial["structure_score_cross_fitted"])
        self.assertGreater(
            initial["component_posterior_weights"][0],
            initial["component_posterior_weights"][1],
        )
        self.assertTrue(initial[
            "target_role_assignment_posterior_active"])
        self.assertFalse(initial[
            "target_oracle_used_for_structure_score"])

        parent.update((50, 50), float(
            parent.basis_matrix([(50, 50)])[0] @ truth), 0.01)
        updated = parent.source_parametric_prior_diagnostics
        self.assertEqual(updated["online_mixture_update_count"], 1)
        self.assertEqual(updated["target_observation_count"], 5)
        self.assertEqual(len(parent.sampled_set), 5)
        self.assertTrue(all(
            row["loo_count"] == 5
            for row in updated["component_loo_predictive_diagnostics"]
        ))

    def test_hierarchical_misspecification_refits_from_frozen_source_law(self):
        model = self._hierarchical_mixture_fixture()
        prior_before = np.asarray(
            model._finite_mixture_component_priors[0]["covariance"]
        ).copy()
        initial = model.source_parametric_prior_diagnostics
        initial_components = {
            row["name"]: row
            for row in initial["component_deviation_diagnostics"]
        }
        self.assertGreater(
            initial_components["source:wrong"][
                "source_mean_misspecification_scale"],
            1.0,
        )
        self.assertEqual(
            initial_components["target:null"][
                "source_mean_misspecification_scale"],
            1.0,
        )

        model.update((50, 50), 8.0, 0.01)
        diagnostics = model.source_parametric_prior_diagnostics
        components = {
            row["name"]: row
            for row in diagnostics["component_deviation_diagnostics"]
        }
        self.assertEqual(diagnostics["online_mixture_update_count"], 1)
        self.assertEqual(diagnostics["target_observation_count"], 3)
        self.assertEqual(len(model._finite_mixture_target_history), 3)
        self.assertEqual(len(diagnostics[
            "source_mean_misspecification_scale_trajectory"]), 2)
        self.assertGreaterEqual(
            components["source:wrong"][
                "source_mean_misspecification_scale"],
            1.0,
        )
        self.assertEqual(
            components["target:null"][
                "source_mean_misspecification_scale"],
            1.0,
        )
        np.testing.assert_allclose(
            model._finite_mixture_component_priors[0]["covariance"],
            prior_before,
        )
        self.assertFalse(diagnostics["target_oracle_used"])

    def test_hierarchical_misspecification_exact_kg_clone_is_independent(self):
        model = self._hierarchical_mixture_fixture()
        algorithm = object.__new__(SingleOLHKGAlgorithm)
        clone = algorithm._clone_gpr_for_exact_kg(model)
        clone.update((25, 25), 9.0, 0.01)
        self.assertEqual(model._finite_mixture_update_count, 0)
        self.assertEqual(len(model._finite_mixture_target_history), 2)
        self.assertEqual(clone._finite_mixture_update_count, 1)
        self.assertEqual(len(clone._finite_mixture_target_history), 3)
        self.assertIsNot(
            clone._finite_mixture_component_priors,
            model._finite_mixture_component_priors,
        )

    def test_hierarchical_misspecification_checkpoint_round_trip(self):
        algorithm = object.__new__(SingleOLHKGAlgorithm)
        model = self._hierarchical_mixture_fixture(
            misspecification_mode="predictive_scale_directional",
            misspecification_ridge=0.25,
        )
        model.update((25, 25), 9.0, 0.01)
        state = algorithm._gpr_checkpoint_state(model)
        restored = self._hierarchical_mixture_fixture()
        algorithm._restore_gpr_checkpoint_state(restored, state)
        self.assertTrue(
            restored._finite_mixture_hierarchical_misspecification)
        self.assertEqual(restored._finite_mixture_update_count, 1)
        self.assertEqual(len(restored._finite_mixture_target_history), 3)
        self.assertEqual(
            restored._finite_mixture_misspecification_mode,
            "predictive_scale_directional",
        )
        self.assertEqual(restored._finite_mixture_misspecification_ridge, 0.25)
        np.testing.assert_allclose(restored.a, model.a)
        np.testing.assert_allclose(restored.C, model.C)
        np.testing.assert_allclose(
            restored._finite_mixture_prior_weights,
            model._finite_mixture_prior_weights,
        )
        self.assertEqual(
            restored.source_parametric_prior_diagnostics[
                "source_mean_misspecification_scale_trajectory"],
            model.source_parametric_prior_diagnostics[
                "source_mean_misspecification_scale_trajectory"],
        )

    def test_source_mean_misspecification_only_increases_uncertainty(self):
        model = ParametricGPR(d=2, lambda_i=0.1, prior_var=1.0)
        algorithm = object.__new__(SingleOLHKGAlgorithm)
        algorithm.config = SingleOLHKGConfig(
            source_constraint_mean_misspecification_mode=(
                "predictive_scale_directional"),
            source_constraint_mean_misspecification_prior_df=2.0,
            source_constraint_mean_misspecification_ridge=0.1,
            source_constraint_mean_misspecification_max_scale=20.0,
        )
        samples = [(0, 0), (100, 0), (0, 100), (100, 100)]
        component = {
            "name": "source:wrong",
            "mean": np.zeros(model.p, dtype=float),
            "covariance": 0.01 * np.eye(model.p, dtype=float),
            "deviation_variance": 0.01,
            "diagnostics": {},
        }
        calibrated = algorithm._calibrate_source_constraint_misspecification(
            model,
            component,
            samples,
            np.asarray([3.0, 5.0, -2.0, 4.0]),
            0.01,
        )
        covariance_gain = (
            calibrated["covariance"] - component["covariance"])
        self.assertGreaterEqual(
            float(np.min(np.linalg.eigvalsh(
                0.5 * (covariance_gain + covariance_gain.T)))),
            -1e-10,
        )
        self.assertGreaterEqual(
            calibrated["deviation_variance"],
            component["deviation_variance"],
        )
        diagnostics = calibrated["diagnostics"]
        self.assertGreater(
            diagnostics["source_mean_misspecification_scale"], 1.0)
        self.assertGreaterEqual(
            diagnostics["source_mean_misspecification_directional_mass"],
            0.0,
        )
        self.assertFalse(
            diagnostics["target_oracle_used_for_misspecification"])
        np.testing.assert_allclose(component["covariance"], 0.01 * np.eye(
            model.p))

        null = {
            **component,
            "name": "target:null",
            "diagnostics": {},
        }
        untouched = algorithm._calibrate_source_constraint_misspecification(
            model, null, samples, np.ones(4), 0.01)
        np.testing.assert_allclose(
            untouched["covariance"], null["covariance"])
        self.assertFalse(untouched["diagnostics"][
            "source_mean_misspecification_applied"])

    def test_source_contrast_posterior_is_low_rank_psd_and_source_only(self):
        algorithm = object.__new__(SingleOLHKGAlgorithm)
        algorithm.config = SingleOLHKGConfig(
            source_constraint_mean_misspecification_mode="source_contrast",
            source_constraint_mean_contrast_scale=1.5,
        )
        means = (
            np.asarray([0.0, 1.0, -1.0]),
            np.asarray([1.0, -1.0, 0.5]),
            np.asarray([-0.5, 0.0, 1.0]),
        )
        components = [
            {
                "name": f"source:{index}",
                "mean": mean.copy(),
                "covariance": 0.1 * np.eye(3),
                "deviation_variance": 0.2,
                "prior_weight": weight,
                "diagnostics": {},
            }
            for index, (mean, weight) in enumerate(zip(
                means, (0.2, 0.3, 0.5)))
        ]
        calibrated = algorithm._calibrate_source_contrast_posterior(
            components)
        gains = []
        for original, updated in zip(components, calibrated):
            np.testing.assert_allclose(updated["mean"], original["mean"])
            gain = updated["covariance"] - original["covariance"]
            gains.append(gain)
            self.assertGreaterEqual(
                float(np.min(np.linalg.eigvalsh(
                    0.5 * (gain + gain.T)))), -1e-10)
            diagnostics = updated["diagnostics"]
            self.assertLessEqual(
                diagnostics["source_contrast_rank"], len(components) - 1)
            self.assertFalse(diagnostics["source_contrast_uses_target_data"])
            self.assertFalse(
                diagnostics["target_oracle_used_for_misspecification"])
        np.testing.assert_allclose(gains[0], gains[1])
        np.testing.assert_allclose(gains[1], gains[2])
        self.assertLessEqual(
            np.linalg.matrix_rank(gains[0], tol=1e-10),
            len(components) - 1,
        )

    def test_source_contrast_is_conditional_on_role_assignment(self):
        algorithm = object.__new__(SingleOLHKGAlgorithm)
        algorithm.config = SingleOLHKGConfig(
            source_constraint_mean_misspecification_mode="source_contrast",
            source_constraint_mean_contrast_scale=1.0,
        )
        components = []
        for assignment, means in {
            "0-1": (
                np.asarray([0.0, 1.0, -0.5, 0.0, 0.0]),
                np.asarray([0.3, -0.5, 0.8, 0.0, 0.0]),
            ),
            "1-0": (
                np.asarray([0.0, 0.0, 0.0, 1.2, -0.2]),
                np.asarray([-0.2, 0.0, 0.0, -0.4, 0.9]),
            ),
        }.items():
            for source, mean in zip(("left", "right"), means):
                components.append({
                    "name": f"source:{source}|role_assignment={assignment}",
                    "mean": mean,
                    "covariance": 0.01 * np.eye(5),
                    "deviation_variance": 0.1,
                    "prior_weight": 0.25,
                    "diagnostics": {},
                })
        calibrated = algorithm._calibrate_source_contrast_posterior(
            components)
        for original, updated in zip(components, calibrated):
            gain = updated["covariance"] - original["covariance"]
            assignment = original["name"].split("role_assignment=", 1)[1]
            inactive = [3, 4] if assignment == "0-1" else [1, 2]
            np.testing.assert_allclose(
                gain[np.ix_(inactive, range(5))], 0.0, atol=1e-14)
            diagnostics = updated["diagnostics"]
            self.assertTrue(
                diagnostics["source_contrast_assignment_conditional"])
            self.assertEqual(
                diagnostics["source_contrast_assignment_group"], assignment)
            self.assertEqual(
                diagnostics["source_contrast_group_component_count"], 2)
            self.assertLessEqual(diagnostics["source_contrast_rank"], 1)

    def test_latent_source_deviation_split_preserves_reference_variance(self):
        model = ParametricGPR(d=2, lambda_i=0.1, prior_var=1.0)
        algorithm = object.__new__(SingleOLHKGAlgorithm)
        algorithm.config = SingleOLHKGConfig(
            source_constraint_mean_deviation_mode="latent_shared")
        original = 0.8
        component = {
            "mean": np.zeros(model.p, dtype=float),
            "covariance": 0.1 * np.eye(model.p, dtype=float),
            "deviation_variance": original,
            "diagnostics": {"source_record_count": 8},
        }
        calibrated = algorithm._calibrate_source_constraint_deviation(
            model, component)
        covariance_gain = float(np.trace(
            calibrated["covariance"] - component["covariance"]))
        self.assertAlmostEqual(covariance_gain, 0.7, places=10)
        self.assertAlmostEqual(
            calibrated["deviation_variance"], 0.1, places=10)
        self.assertAlmostEqual(
            covariance_gain + calibrated["deviation_variance"],
            original,
            places=10,
        )
        self.assertTrue(np.all(np.linalg.eigvalsh(
            calibrated["covariance"]) >= -1e-12))
        self.assertEqual(
            calibrated["diagnostics"]["source_deviation_mode"],
            "latent_shared",
        )
        self.assertEqual(component["deviation_variance"], original)

        algorithm.config.source_constraint_mean_deviation_mode = (
            "raw_independent")
        unchanged = algorithm._calibrate_source_constraint_deviation(
            model, component)
        np.testing.assert_allclose(
            unchanged["covariance"], component["covariance"])
        self.assertEqual(unchanged["deviation_variance"], original)

    def test_sequential_source_mixture_reweights_and_clones_independently(self):
        left = ParametricGPR(d=1, lambda_i=0.05, prior_var=0.01)
        right = ParametricGPR(d=1, lambda_i=0.05, prior_var=0.01)
        left_mean = np.zeros(left.p, dtype=float)
        right_mean = np.zeros(right.p, dtype=float)
        left_mean[0] = -1.0
        right_mean[0] = 1.0
        left.set_parametric_prior(left_mean, 0.05, 0.01)
        right.set_parametric_prior(right_mean, 0.05, 0.01)

        mixture = ParametricGPR(d=1, lambda_i=0.05, prior_var=1.0)
        mixture.set_moment_matched_posterior(
            [left, right],
            [0.5, 0.5],
            diagnostics={
                "adaptation_mode": "sequential_target_evidence_mixture",
                "component_names": ["source:left", "target:null"],
                "component_posterior_weights": [0.5, 0.5],
                "evidence_temperature": 1.0,
                "target_observation_count": 0,
                "online_mixture_update_count": 0,
            },
            sequential_updates=True,
        )
        mixture.update((0,), 0.9, 0.01)
        diagnostics = mixture.numerical_diagnostics()[
            "source_parametric_prior"]
        weights = np.asarray(
            diagnostics["component_posterior_weights"], dtype=float)

        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=12)
        self.assertGreater(weights[1], weights[0])
        self.assertEqual(diagnostics["online_mixture_update_count"], 1)
        self.assertEqual(diagnostics["target_observation_count"], 1)
        self.assertEqual(
            diagnostics["adaptation_mode"],
            "sequential_target_evidence_mixture",
        )
        self.assertTrue(np.all(np.linalg.eigvalsh(
            0.5 * (mixture.C + mixture.C.T)) >= -1e-10))

        algorithm = object.__new__(SingleOLHKGAlgorithm)
        clone = algorithm._clone_gpr_for_exact_kg(mixture)
        original_weights = mixture._finite_mixture_weights.copy()
        clone.update((1,), -0.9, 0.01)
        np.testing.assert_allclose(
            mixture._finite_mixture_weights, original_weights)
        self.assertFalse(np.allclose(
            clone._finite_mixture_weights, original_weights))

    def test_residual_rank_structure_mass_updates_from_target_evidence(self):
        rank0 = ParametricGPR(d=1, lambda_i=0.01, prior_var=1.0)
        rank1 = ParametricGPR(d=1, lambda_i=0.01, prior_var=1.0)
        mean = np.zeros(rank0.p, dtype=float)
        covariance0 = 1e-6 * np.eye(rank0.p, dtype=float)
        covariance1 = covariance0.copy()
        covariance1[1, 1] = 1.0
        rank0.set_parametric_prior(mean, 0.01, covariance0)
        rank1.set_parametric_prior(mean, 0.01, covariance1)

        mixture = ParametricGPR(d=1, lambda_i=0.01, prior_var=1.0)
        names = [
            "source:a|target_residual_rank=0",
            "source:a|target_residual_rank=1",
        ]
        mixture.set_moment_matched_posterior(
            [rank0, rank1],
            [0.8, 0.2],
            diagnostics={
                "adaptation_mode": "sequential_target_evidence_mixture",
                "component_names": names,
                "component_posterior_weights": [0.8, 0.2],
                "evidence_temperature": 1.0,
                "target_observation_count": 0,
                "online_mixture_update_count": 0,
            },
            sequential_updates=True,
        )
        initial = mixture.source_parametric_prior_diagnostics
        self.assertTrue(initial["target_residual_rank_posterior_active"])
        self.assertAlmostEqual(
            initial["target_residual_rank_conditional_source_mass"]["1"],
            0.2,
        )

        mixture.update((1,), 2.0, 0.01)
        posterior = mixture.source_parametric_prior_diagnostics
        self.assertGreater(
            posterior["target_residual_rank_conditional_source_mass"]["1"],
            0.2,
        )
        self.assertEqual(posterior["target_residual_rank_selected"], 1)
        self.assertTrue(
            posterior[
                "target_residual_rank_target_labels_used_for_update"])
        self.assertFalse(
            posterior["target_residual_rank_target_oracle_used"])

    def test_mixture_update_preserves_tiny_and_zero_support(self):
        prior, posterior = posterior_mixture_weights(
            [1e-60, 1.0, 0.0],
            [30.0, 0.0, 1e6],
        )
        self.assertLess(prior[0], 1e-50)
        self.assertLess(posterior[0], 1e-40)
        self.assertEqual(prior[2], 0.0)
        self.assertEqual(posterior[2], 0.0)
        self.assertAlmostEqual(float(np.sum(posterior)), 1.0, places=14)

    def test_source_constraint_coefficient_prior_conditions_on_target_pilot(self):
        def problem(name):
            return ScalarizedProblem(make_problem(
                name, d=5, L=30, sigma=0.04))

        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="consensus",
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            seed=940,
        ).fit_from_source_problems(
            [
                ("FactorShockStatePolicyRZDT1", problem(
                    "FactorShockStatePolicyRZDT1")),
                ("InventorySupplyChain", problem("InventorySupplyChain")),
            ],
            n_records_per_domain=8,
            rng=np.random.default_rng(941),
        )
        target = MetaPriorProblemAdapter(
            problem("QueueResourceControl"), prior)
        algorithm = SingleOLHKGAlgorithm(
            target,
            SingleOLHKGConfig(
                N=4,
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
                task_posterior_mode="finite",
                source_constraint_mean_coefficient_prior=True,
                source_constraint_mean_adaptation_mode=(
                    "sequential_evidence_mixture"),
                source_constraint_mean_deviation_mode="latent_shared",
                seed=942,
            ),
        )
        samples = algorithm._initial_samples()
        source_mean = target.gpr_basis_map(
            output_index=1).source_parametric_prior()["mean"]
        algorithm._fit_initial_belief(samples)

        constraint = algorithm.gpr[1]
        diagnostics = constraint.numerical_diagnostics()
        self.assertIn("source_parametric_prior", diagnostics)
        source_diagnostics = diagnostics["source_parametric_prior"]
        self.assertEqual(
            source_diagnostics["source_deviation_mode"], "latent_shared")
        component_diagnostics = source_diagnostics[
            "component_deviation_diagnostics"]
        source_components = [
            row for row in component_diagnostics
            if row["name"].startswith("source:")
        ]
        target_null = next(
            row for row in component_diagnostics
            if row["name"] == "target:null")
        self.assertTrue(source_components)
        self.assertTrue(all(
            row["source_deviation_mode"] == "latent_shared"
            for row in source_components))
        self.assertNotIn("source_deviation_mode", target_null)
        self.assertEqual(len(constraint.sampled_set), len(samples))
        self.assertFalse(np.allclose(constraint.a[:constraint.p], source_mean))
        self.assertTrue(np.all(np.linalg.eigvalsh(
            0.5 * (constraint.C + constraint.C.T)) >= -1e-10))
        for state in algorithm.task_ensemble.states:
            expert_constraint = state.gpr_models[1]
            self.assertIn(
                "source_parametric_prior",
                expert_constraint.numerical_diagnostics(),
            )
            self.assertEqual(len(expert_constraint.sampled_set), len(samples))

    def test_target_residual_rank_is_a_target_updated_structure_posterior(self):
        def problem(name):
            return ScalarizedProblem(make_problem(
                name, d=8, L=30, sigma=0.04))

        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="observable_state_exposure",
            observable_mean_descriptor_mode="role_adaptive_ordered",
            observable_mean_feature_mode="linear",
            observable_mean_latent_dim=3,
            observable_mean_latent_transform="source_tanh",
            observable_mean_target_residual_rank=2,
            observable_mean_target_residual_prior_scale=0.25,
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            seed=2940,
        ).fit_from_source_problems(
            [
                ("FactorShockStatePolicyRZDT1", problem(
                    "FactorShockStatePolicyRZDT1")),
                ("InventorySupplyChain", problem("InventorySupplyChain")),
            ],
            n_records_per_domain=8,
            rng=np.random.default_rng(2941),
        )
        target = MetaPriorProblemAdapter(
            problem("QueueResourceControl"), prior)
        algorithm = SingleOLHKGAlgorithm(
            target,
            SingleOLHKGConfig(
                N=4,
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
                task_posterior_mode="finite",
                source_constraint_mean_coefficient_prior=True,
                source_constraint_mean_adaptation_mode=(
                    "sequential_evidence_mixture"),
                source_constraint_mean_deviation_mode="latent_shared",
                source_constraint_mean_residual_rank_posterior=True,
                source_constraint_mean_residual_rank_prior="0.7,0.2,0.1",
                seed=2942,
            ),
        )
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)
        diagnostics = algorithm.gpr[1].source_parametric_prior_diagnostics
        self.assertTrue(
            diagnostics["target_residual_rank_posterior_active"])
        self.assertEqual(
            set(diagnostics["target_residual_rank_posterior_mass"]),
            {"0", "1", "2"},
        )
        self.assertAlmostEqual(
            sum(diagnostics[
                "target_residual_rank_posterior_mass"].values()),
            1.0,
            places=10,
        )
        self.assertTrue(all(
            "|target_residual_rank=" in name
            for name in diagnostics["component_names"]
        ))
        self.assertEqual(
            diagnostics["target_observation_count"], len(samples))
        self.assertFalse(
            diagnostics["target_residual_rank_target_oracle_used"])

    def test_role_assignment_structure_posterior_reaches_constraint_gpr(self):
        def problem(name):
            return ScalarizedProblem(make_problem(
                name, d=8, L=30, sigma=0.04))

        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="observable_state_exposure",
            observable_mean_descriptor_mode="role_adaptive_ordered",
            observable_mean_feature_mode="linear",
            observable_mean_latent_dim=3,
            observable_mean_latent_transform="source_tanh",
            observable_mean_role_assignment_posterior=True,
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            seed=3940,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", problem("InventorySupplyChain")),
                ("QueueResourceControl", problem("QueueResourceControl")),
            ],
            n_records_per_domain=8,
            rng=np.random.default_rng(3941),
        )
        target = MetaPriorProblemAdapter(
            problem("FactorShockStatePolicyRZDT1"), prior)
        algorithm = SingleOLHKGAlgorithm(
            target,
            SingleOLHKGConfig(
                N=4,
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
                task_posterior_mode="off",
                source_constraint_mean_coefficient_prior=True,
                source_constraint_mean_adaptation_mode=(
                    "sequential_evidence_mixture"),
                source_constraint_mean_deviation_mode="latent_shared",
                source_constraint_mean_misspecification_mode=(
                    "hierarchical_predictive_scale"),
                hvd_source_task_weight_mode="independent",
                hvd_cumulative_target_evidence_mode="replication_only",
                seed=3942,
            ),
        )
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)

        diagnostics = (
            algorithm.gpr[1].source_parametric_prior_diagnostics)
        names = diagnostics["component_names"]
        weights = np.asarray(
            diagnostics["component_posterior_weights"], dtype=float)
        self.assertTrue(
            diagnostics["target_role_assignment_posterior_active"])
        self.assertEqual(len(names), 18)
        self.assertEqual(sum(name.startswith("source:") for name in names), 12)
        self.assertEqual(
            sum(name.startswith("target:null") for name in names), 6)
        self.assertTrue(all(
            "|role_assignment=" in name for name in names))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=10)
        self.assertAlmostEqual(
            sum(diagnostics[
                "target_role_assignment_posterior_mass"].values()),
            1.0,
            places=10,
        )
        self.assertIn(
            diagnostics["target_role_assignment_selected"],
            diagnostics["target_role_assignment_posterior_mass"],
        )
        self.assertTrue(diagnostics[
            "target_role_assignment_target_labels_used_for_update"])
        self.assertFalse(diagnostics[
            "target_role_assignment_target_oracle_used"])
        self.assertTrue(diagnostics[
            "target_role_assignment_permutation_equivariant"])
        self.assertAlmostEqual(
            diagnostics["target_role_assignment_structured_source_mass"]
            + diagnostics["target_role_assignment_structured_null_mass"],
            1.0,
            places=10,
        )
        self.assertEqual(
            diagnostics["source_mean_misspecification_mode"],
            "hierarchical_predictive_scale",
        )

        contract = target.mean_risk_coordinate_contract()
        self.assertTrue(contract["channel_role_assignment_posterior"])
        self.assertFalse(contract[
            "channel_role_assignment_hypotheses_use_target_labels"])
        self.assertTrue(contract[
            "channel_role_assignment_weights_use_charged_target_labels"])
        self.assertFalse(contract[
            "channel_role_assignment_weights_use_target_oracle"])
        self.assertIsNone(
            algorithm.variance_model.cumulative_source_task_posterior[1])

    def test_role_assignment_loo_structure_posterior_is_cross_fitted(self):
        def problem(name):
            return ScalarizedProblem(make_problem(
                name, d=8, L=30, sigma=0.04))

        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="observable_state_exposure",
            observable_mean_descriptor_mode="role_adaptive_ordered",
            observable_mean_feature_mode="linear",
            observable_mean_latent_dim=3,
            observable_mean_latent_transform="source_tanh",
            observable_mean_role_assignment_posterior=True,
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            seed=4940,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", problem("InventorySupplyChain")),
                ("QueueResourceControl", problem("QueueResourceControl")),
            ],
            n_records_per_domain=8,
            rng=np.random.default_rng(4941),
        )
        target = MetaPriorProblemAdapter(
            problem("FactorShockStatePolicyRZDT1"), prior)
        algorithm = SingleOLHKGAlgorithm(
            target,
            SingleOLHKGConfig(
                N=4,
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
                task_posterior_mode="off",
                source_constraint_mean_coefficient_prior=True,
                source_constraint_mean_adaptation_mode=(
                    "sequential_evidence_mixture"),
                source_constraint_mean_deviation_mode="latent_shared",
                source_constraint_mean_misspecification_mode="none",
                source_constraint_mean_structure_score_mode="loo_predictive",
                hvd_source_task_weight_mode="independent",
                hvd_cumulative_target_evidence_mode="replication_only",
                seed=4942,
            ),
        )
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)

        constraint = algorithm.gpr[1]
        diagnostics = constraint.source_parametric_prior_diagnostics
        self.assertEqual(diagnostics["structure_score_mode"], "loo_predictive")
        self.assertTrue(diagnostics["structure_score_cross_fitted"])
        self.assertEqual(diagnostics["target_observation_count"], len(samples))
        self.assertTrue(all(
            item["loo_count"] == len(samples)
            for item in diagnostics["component_loo_predictive_diagnostics"]
        ))
        self.assertTrue(
            diagnostics["target_role_assignment_posterior_active"])
        self.assertFalse(diagnostics["target_oracle_used"])
        self.assertFalse(
            diagnostics["target_oracle_used_for_structure_score"])
        self.assertIsNone(
            algorithm.variance_model.cumulative_source_task_posterior[1])

        before = np.asarray(
            diagnostics["component_posterior_weights"], dtype=float)
        x = tuple(samples[0])
        y = target.simulate(x, np.random.default_rng(4943))
        sigma2 = algorithm.variance_model.predict_variance(1, x, target)
        constraint.update(x, float(y[1]), float(sigma2))
        after_diagnostics = constraint.source_parametric_prior_diagnostics
        after = np.asarray(
            after_diagnostics["component_posterior_weights"], dtype=float)
        self.assertEqual(after_diagnostics["online_mixture_update_count"], 1)
        self.assertEqual(
            after_diagnostics["target_observation_count"], len(samples) + 1)
        self.assertTrue(all(
            item["loo_count"] == len(samples) + 1
            for item in after_diagnostics[
                "component_loo_predictive_diagnostics"]
        ))
        self.assertFalse(np.allclose(before, after))
        self.assertFalse(
            after_diagnostics["target_oracle_used_for_structure_score"])
        self.assertIsNone(
            algorithm.variance_model.cumulative_source_task_posterior[1])

    def test_geometry_assignment_marginal_is_frozen_while_experts_adapt(self):
        def problem(name):
            return ScalarizedProblem(make_problem(
                name, d=8, L=30, sigma=0.04))

        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="observable_state_exposure",
            observable_mean_descriptor_mode="role_adaptive_ordered",
            observable_mean_feature_mode="linear",
            observable_mean_latent_dim=3,
            observable_mean_latent_transform="source_tanh",
            observable_mean_role_assignment_posterior=True,
            observable_mean_role_assignment_prior="source_geometry",
            observable_mean_role_assignment_prior_temperature_scale=0.25,
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            seed=5940,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", problem("InventorySupplyChain")),
                ("QueueResourceControl", problem("QueueResourceControl")),
            ],
            n_records_per_domain=8,
            rng=np.random.default_rng(5941),
        )
        target = MetaPriorProblemAdapter(
            problem("FactorShockStatePolicyRZDT1"), prior)
        algorithm = SingleOLHKGAlgorithm(
            target,
            SingleOLHKGConfig(
                N=4,
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
                task_posterior_mode="off",
                source_constraint_mean_coefficient_prior=True,
                source_constraint_mean_adaptation_mode=(
                    "sequential_evidence_mixture"),
                source_constraint_mean_deviation_mode="latent_shared",
                source_constraint_mean_misspecification_mode="none",
                source_constraint_mean_structure_score_mode=(
                    "geometry_conditional"),
                hvd_source_task_weight_mode="independent",
                hvd_cumulative_target_evidence_mode="replication_only",
                seed=5942,
            ),
        )
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)

        constraint = algorithm.gpr[1]
        before_diagnostics = dict(
            constraint.source_parametric_prior_diagnostics)
        before_weights = np.asarray(
            before_diagnostics["component_posterior_weights"], dtype=float)
        before_groups = dict(before_diagnostics["assignment_group_masses"])
        before_role_mass = dict(before_diagnostics[
            "target_role_assignment_posterior_mass"])

        x = tuple(samples[0])
        component_means = np.asarray([
            component.posterior_mean(x)
            for component in constraint._finite_mixture_components
        ], dtype=float)
        y = float(np.max(component_means) + 5.0)
        constraint.update(x, y, 0.01)
        after_diagnostics = constraint.source_parametric_prior_diagnostics
        after_weights = np.asarray(
            after_diagnostics["component_posterior_weights"], dtype=float)

        self.assertEqual(
            after_diagnostics["adaptation_mode"],
            "sequential_assignment_prior_conditional_expert_mixture",
        )
        self.assertEqual(before_groups, after_diagnostics[
            "assignment_group_masses"])
        self.assertEqual(
            set(before_role_mass),
            set(after_diagnostics["target_role_assignment_posterior_mass"]),
        )
        for label, mass in before_role_mass.items():
            self.assertAlmostEqual(
                mass,
                after_diagnostics[
                    "target_role_assignment_posterior_mass"][label],
                places=14,
            )
        self.assertFalse(np.allclose(before_weights, after_weights))
        self.assertFalse(after_diagnostics[
            "target_role_assignment_target_labels_used_for_update"])
        self.assertTrue(after_diagnostics[
            "target_role_assignment_conditional_expert_uses_target_labels"])
        self.assertFalse(after_diagnostics["target_oracle_used"])
        self.assertFalse(after_diagnostics[
            "target_oracle_used_for_group_masses"])
        self.assertIsNone(
            algorithm.variance_model.cumulative_source_task_posterior[1])

        hierarchical_target = MetaPriorProblemAdapter(
            problem("FactorShockStatePolicyRZDT1"), prior)
        hierarchical = SingleOLHKGAlgorithm(
            hierarchical_target,
            SingleOLHKGConfig(
                N=4,
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
                task_posterior_mode="off",
                source_constraint_mean_coefficient_prior=True,
                source_constraint_mean_adaptation_mode=(
                    "sequential_evidence_mixture"),
                source_constraint_mean_deviation_mode="latent_shared",
                source_constraint_mean_misspecification_mode=(
                    "hierarchical_predictive_scale"),
                source_constraint_mean_structure_score_mode=(
                    "geometry_conditional"),
                hvd_source_task_weight_mode="independent",
                hvd_cumulative_target_evidence_mode="replication_only",
                seed=5944,
            ),
        )
        hierarchical_samples = hierarchical._initial_samples()
        hierarchical._fit_initial_belief(hierarchical_samples)
        hierarchical_constraint = hierarchical.gpr[1]
        hierarchical_before = dict(
            hierarchical_constraint.source_parametric_prior_diagnostics)
        hierarchical_groups = dict(
            hierarchical_before["assignment_group_masses"])
        self.assertEqual(
            hierarchical_before["adaptation_mode"],
            "sequential_assignment_prior_conditional_hierarchical_"
            "expert_mixture",
        )
        self.assertTrue(hierarchical_before[
            "source_mean_misspecification_online"])
        self.assertTrue(hierarchical_before[
            "source_mean_misspecification_refit_from_frozen_law"])
        self.assertTrue(all(
            float(component["source_mean_misspecification_scale"]) >= 1.0
            for component in hierarchical_before[
                "component_deviation_diagnostics"]
        ))

        hx = tuple(hierarchical_samples[0])
        hy = hierarchical_target.simulate(
            hx, np.random.default_rng(5945))
        hierarchical_constraint.update(hx, float(hy[1]), 0.01)
        hierarchical_after = (
            hierarchical_constraint.source_parametric_prior_diagnostics)
        for label, mass in hierarchical_groups.items():
            self.assertAlmostEqual(
                mass,
                hierarchical_after["assignment_group_masses"][label],
                places=14,
            )
        self.assertEqual(
            hierarchical_after["online_mixture_update_count"], 1)
        self.assertEqual(
            len(hierarchical_after[
                "source_mean_misspecification_scale_trajectory"]),
            2,
        )
        self.assertFalse(hierarchical_after[
            "target_role_assignment_target_labels_used_for_update"])
        self.assertTrue(hierarchical_after[
            "target_role_assignment_conditional_expert_uses_target_labels"])
        self.assertIsNone(
            hierarchical.variance_model.cumulative_source_task_posterior[1])

    def test_boundary_role_prior_uses_charged_pilot_then_freezes(self):
        def problem(name):
            return ScalarizedProblem(make_problem(
                name, d=8, L=30, sigma=0.04))

        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="observable_state_exposure",
            observable_mean_descriptor_mode="role_adaptive_ordered",
            observable_mean_feature_mode="linear",
            observable_mean_latent_dim=3,
            observable_mean_latent_transform="source_tanh",
            observable_mean_role_assignment_posterior=True,
            observable_mean_role_assignment_prior=(
                "source_geometry_boundary"),
            observable_mean_role_assignment_prior_temperature_scale=1.0,
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            seed=5950,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", problem("InventorySupplyChain")),
                ("QueueResourceControl", problem("QueueResourceControl")),
            ],
            n_records_per_domain=8,
            rng=np.random.default_rng(5951),
        )
        target = MetaPriorProblemAdapter(
            problem("FactorShockStatePolicyRZDT1"), prior)
        algorithm = SingleOLHKGAlgorithm(
            target,
            SingleOLHKGConfig(
                N=4,
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
                task_posterior_mode="off",
                source_constraint_mean_coefficient_prior=True,
                source_constraint_mean_adaptation_mode=(
                    "sequential_evidence_mixture"),
                source_constraint_mean_deviation_mode="latent_shared",
                source_constraint_mean_misspecification_mode="source_contrast",
                source_constraint_mean_structure_score_mode=(
                    "geometry_conditional"),
                hvd_source_task_weight_mode="independent",
                hvd_cumulative_target_evidence_mode="replication_only",
                seed=5952,
            ),
        )
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)
        constraint = algorithm.gpr[1]
        diagnostics = dict(
            constraint.source_parametric_prior_diagnostics)
        self.assertTrue(diagnostics["target_labels_used_for_group_masses"])
        self.assertFalse(diagnostics["target_oracle_used_for_group_masses"])
        self.assertTrue(diagnostics[
            "target_role_assignment_target_labels_used_for_prior"])
        self.assertFalse(diagnostics[
            "target_role_assignment_target_labels_used_for_online_update"])
        self.assertEqual(
            diagnostics["target_role_assignment_update_scope"],
            "charged_pilot_assignment_prior_then_frozen_"
            "conditional_expert_only",
        )
        self.assertAlmostEqual(
            sum(diagnostics["assignment_group_masses"].values()), 1.0)
        source_diagnostics = [
            item for item in diagnostics["component_deviation_diagnostics"]
            if str(item["name"]).startswith("source:")
        ]
        self.assertTrue(source_diagnostics)
        self.assertTrue(all(
            item["source_contrast_assignment_conditional"]
            for item in source_diagnostics
        ))
        self.assertIsNone(
            algorithm.variance_model.cumulative_source_task_posterior[1])

    def test_aggregate_transferability_mixture_updates_from_target_only(self):
        def problem(name):
            return ScalarizedProblem(make_problem(
                name, d=5, L=30, sigma=0.04))

        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="consensus",
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            seed=1940,
        ).fit_from_source_problems(
            [
                ("FactorShockStatePolicyRZDT1", problem(
                    "FactorShockStatePolicyRZDT1")),
                ("InventorySupplyChain", problem("InventorySupplyChain")),
            ],
            n_records_per_domain=8,
            rng=np.random.default_rng(1941),
        )
        target = MetaPriorProblemAdapter(
            problem("QueueResourceControl"), prior)
        algorithm = SingleOLHKGAlgorithm(
            target,
            SingleOLHKGConfig(
                N=4,
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
                task_posterior_mode="finite",
                source_constraint_mean_coefficient_prior=True,
                source_constraint_mean_adaptation_mode=(
                    "sequential_aggregate_mixture"),
                source_constraint_mean_deviation_mode="latent_shared",
                hvd_source_task_weight_mode="independent",
                seed=1942,
            ),
        )
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)
        constraint = algorithm.gpr[1]
        diagnostics = constraint.source_parametric_prior_diagnostics
        self.assertEqual(
            diagnostics["adaptation_mode"],
            "sequential_aggregate_target_evidence_mixture",
        )
        self.assertTrue(diagnostics["aggregate_transferability_latent"])
        self.assertEqual(
            diagnostics["component_names"],
            ["source:aggregate", "target:null"],
        )
        source = diagnostics["component_deviation_diagnostics"][0]
        self.assertTrue(source[
            "aggregate_contains_between_source_disagreement"])
        self.assertFalse(source["target_data_used_to_define_aggregate"])
        self.assertFalse(source["target_oracle_used_to_define_aggregate"])
        weights_before = np.asarray(
            diagnostics["component_posterior_weights"], dtype=float)
        constraint.update(samples[0], 3.0, 0.01)
        updated = constraint.source_parametric_prior_diagnostics
        self.assertEqual(updated["online_mixture_update_count"], 1)
        self.assertFalse(np.allclose(
            weights_before,
            np.asarray(updated["component_posterior_weights"], dtype=float),
        ))

    def test_target_calibrated_boundary_phi_generates_three_proposal_roles(self):
        def problem(name):
            return ScalarizedProblem(make_problem(
                name, d=12, L=30, sigma=0.04))

        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_latent_dim=2,
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            source_universal_fraction=1.0,
            seed=946,
        ).fit_from_source_problems(
            [
                ("FactorShockStatePolicyRZDT1", problem(
                    "FactorShockStatePolicyRZDT1")),
                ("InventorySupplyChain", problem("InventorySupplyChain")),
            ],
            n_records_per_domain=16,
            rng=np.random.default_rng(947),
        )
        target = MetaPriorProblemAdapter(
            problem("QueueResourceControl"), prior)
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
                source_constraint_mean_coefficient_prior=True,
                source_constraint_mean_adaptation_mode="evidence_mixture",
                boundary_coordinate_candidate_count=10,
                boundary_coordinate_pool_size=64,
                truth_pool_diagnostics=True,
                seed=948,
            ),
        )
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)

        true_objective = target.base.true_objective
        true_constraint_mean = target.base.true_constraint_mean
        true_sigma = target.base.true_sigma

        def forbidden(*_args, **_kwargs):
            raise AssertionError("target oracle was called by phi proposal")

        target.base.true_objective = forbidden
        target.base.true_constraint_mean = forbidden
        target.base.true_sigma = forbidden
        batches = algorithm._boundary_coordinate_proposal_batches(
            np.random.default_rng(949), record=True)
        roles = {role for role, _rows in batches}
        self.assertEqual(roles, {"safe", "boundary", "coverage"})
        self.assertEqual(sum(len(rows) for _role, rows in batches), 10)
        diagnostics = algorithm._last_boundary_coordinate_proposal_info
        self.assertEqual(diagnostics["status"], "selected")
        self.assertEqual(
            diagnostics["coordinate"],
            "phi=source_aligned_chance_boundary",
        )
        self.assertFalse(diagnostics["target_oracle_used"])
        algorithm.iteration_log = [{
            "boundary_coordinate_proposal": diagnostics,
        }]
        summary = algorithm._summarize_boundary_coordinate_proposals()
        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["generated_iteration_count"], 1)
        self.assertEqual(summary["selected_candidate_count"], 10)
        target.base.true_objective = true_objective
        target.base.true_constraint_mean = true_constraint_mean
        target.base.true_sigma = true_sigma
        pool = [point for _role, rows in batches for point in rows]
        source = {
            point: f"boundary_phi:{role}"
            for role, rows in batches for point in rows
        }
        audit = algorithm._truth_pool_diagnostics(
            pool,
            prefix="candidate",
            sources=source,
        )
        self.assertTrue(audit["candidate_failure_decomposition_available"])
        self.assertIn(audit["candidate_failure_layer"], {
            "candidate_support",
            "epistemic_or_safety_depth",
            "closed",
            "constraint_mean",
            "cumulative_variance",
        })
        self.assertEqual(
            set(audit["candidate_source_truth_support"]),
            {"boundary_phi:safe", "boundary_phi:boundary",
             "boundary_phi:coverage"},
        )
        self.assertFalse(
            audit["candidate_audit_target_oracle_used_for_decision"])
        self.assertIn("candidate_false_certified_count", audit)
        self.assertIn("candidate_true_certified_count", audit)
        raw_audit = algorithm._boundary_coordinate_raw_pool_audit()
        self.assertEqual(raw_audit["status"], "audited")
        self.assertEqual(
            raw_audit["pool_contract"],
            "universal_low_frequency_no_source_templates",
        )
        self.assertTrue(raw_audit["post_run_only"])
        self.assertTrue(raw_audit["target_truth_used_for_audit"])
        self.assertFalse(raw_audit["target_oracle_used_for_decision"])
        self.assertTrue(raw_audit[
            "boundary_raw_pool_truth_diagnostics_available"])
        self.assertEqual(
            raw_audit,
            algorithm._boundary_coordinate_raw_pool_audit(),
        )

    def test_source_constraint_evidence_mixture_is_oracle_free_and_psd(self):
        def problem(name):
            return ScalarizedProblem(make_problem(
                name, d=5, L=30, sigma=0.04))

        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="latent",
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            seed=943,
        ).fit_from_source_problems(
            [
                ("FactorShockStatePolicyRZDT1", problem(
                    "FactorShockStatePolicyRZDT1")),
                ("QueueResourceControl", problem("QueueResourceControl")),
            ],
            n_records_per_domain=8,
            rng=np.random.default_rng(944),
        )
        target = MetaPriorProblemAdapter(
            problem("InventorySupplyChain"), prior)
        algorithm = SingleOLHKGAlgorithm(
            target,
            SingleOLHKGConfig(
                N=4,
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
                source_constraint_mean_coefficient_prior=True,
                source_constraint_mean_adaptation_mode="evidence_mixture",
                source_constraint_mean_null_weight=0.5,
                seed=945,
            ),
        )
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)
        constraint = algorithm.gpr[1]
        diagnostics = constraint.numerical_diagnostics()[
            "source_parametric_prior"]
        weights = np.asarray(
            diagnostics["component_posterior_weights"], dtype=float)

        self.assertEqual(
            diagnostics["adaptation_mode"], "target_evidence_mixture")
        self.assertEqual(
            diagnostics["posterior_projection"],
            "finite_mixture_moment_match",
        )
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=12)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertIn("target:null", diagnostics["component_names"])
        self.assertTrue(diagnostics["posterior_target_data_used"])
        self.assertFalse(diagnostics["target_oracle_used"])
        self.assertEqual(len(constraint.sampled_set), len(samples))
        self.assertTrue(np.all(np.linalg.eigvalsh(
            0.5 * (constraint.C + constraint.C.T)) >= -1e-10))

    def test_target_pool_null_geometry_is_outcome_free_psd_and_scale_preserving(
        self,
    ):
        def problem(name):
            return ScalarizedProblem(make_problem(
                name, d=12, L=30, sigma=0.04))

        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_latent_dim=3,
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="observable_state_exposure",
            observable_mean_descriptor_mode="role_adaptive_ordered",
            observable_mean_feature_mode="linear",
            observable_mean_latent_transform="source_tanh",
            observable_variance_input_mode="observable_state_exposure",
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            source_universal_fraction=1.0,
            seed=9461,
        ).fit_from_source_problems(
            [
                ("FactorShockStatePolicyRZDT1", problem(
                    "FactorShockStatePolicyRZDT1")),
                ("QueueResourceControl", problem("QueueResourceControl")),
            ],
            n_records_per_domain=10,
            rng=np.random.default_rng(9462),
        )
        target = MetaPriorProblemAdapter(
            problem("InventorySupplyChain"), prior)
        algorithm = SingleOLHKGAlgorithm(
            target,
            SingleOLHKGConfig(
                N=4,
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
                source_constraint_mean_coefficient_prior=True,
                source_constraint_mean_adaptation_mode="evidence_mixture",
                source_constraint_mean_null_weight=0.5,
                source_constraint_mean_null_geometry="target_pool",
                source_constraint_mean_null_geometry_ridge=1e-3,
                seed=9463,
            ),
        )
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)
        diagnostics = algorithm.gpr[1].numerical_diagnostics()[
            "source_parametric_prior"]
        null = next(
            row for row in diagnostics["component_deviation_diagnostics"]
            if row["name"] == "target:null")
        self.assertEqual(null["null_geometry_mode"], "target_pool")
        self.assertTrue(null["average_predictive_scale_preserved"])
        self.assertAlmostEqual(
            null["average_predictive_variance_ratio"], 1.0, places=7)
        self.assertGreaterEqual(null["minimum_covariance_eigenvalue"], 0.0)
        self.assertFalse(null["target_labels_used_for_null_geometry"])
        self.assertFalse(null["target_oracle_used_for_null_geometry"])
        self.assertEqual(
            null["target_geometry_pool_source"],
            "deterministic_unlabeled_role_matching_pool",
        )

    def test_role_transport_matching_uncertainty_only_inflates_epistemic_law(
        self,
    ):
        def problem(name):
            return ScalarizedProblem(make_problem(
                name, d=12, L=30, sigma=0.04))

        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_latent_dim=3,
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="observable_state_exposure",
            observable_mean_descriptor_mode="role_transport",
            observable_mean_feature_mode="linear",
            observable_mean_latent_transform="source_tanh",
            observable_variance_input_mode="observable_state_exposure",
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            source_universal_fraction=1.0,
            seed=9464,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", problem("InventorySupplyChain")),
                ("QueueResourceControl", problem("QueueResourceControl")),
            ],
            n_records_per_domain=10,
            rng=np.random.default_rng(9465),
        )
        target = MetaPriorProblemAdapter(
            problem("FactorShockStatePolicyRZDT1"), prior)
        algorithm = SingleOLHKGAlgorithm(
            target,
            SingleOLHKGConfig(
                N=4,
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
                source_constraint_mean_coefficient_prior=True,
                source_constraint_mean_adaptation_mode="evidence_mixture",
                source_constraint_mean_deviation_mode="latent_shared",
                source_constraint_mean_role_epistemic_mode=(
                    "matching_uncertainty"),
                source_constraint_mean_null_weight=0.5,
                seed=9466,
            ),
        )
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)
        diagnostics = algorithm.gpr[1].numerical_diagnostics()[
            "source_parametric_prior"]
        calibration = diagnostics["source_role_epistemic_calibration"]
        source_rows = [
            row for row in diagnostics["component_deviation_diagnostics"]
            if row["name"] != "target:null"
        ]
        self.assertEqual(calibration["mode"], "matching_uncertainty")
        self.assertEqual(calibration["source_role_trust"], 1.0)
        self.assertGreaterEqual(
            calibration["epistemic_covariance_scale"], 1.0)
        self.assertFalse(calibration["target_labels_used"])
        self.assertFalse(calibration["target_oracle_used"])
        self.assertTrue(source_rows)
        self.assertTrue(all(
            row["role_matching_uncertainty_monotone"]
            and row["role_matching_epistemic_covariance_scale"] >= 1.0
            and not row["role_matching_target_labels_used"]
            and not row["role_matching_target_oracle_used"]
            for row in source_rows
        ))

    def test_sequential_source_mixture_updates_shared_hvd_task_law(self):
        def problem(name):
            return ScalarizedProblem(make_problem(
                name, d=5, L=30, sigma=0.04))

        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="latent",
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            seed=946,
        ).fit_from_source_problems(
            [
                ("FactorShockStatePolicyRZDT1", problem(
                    "FactorShockStatePolicyRZDT1")),
                ("QueueResourceControl", problem("QueueResourceControl")),
            ],
            n_records_per_domain=8,
            rng=np.random.default_rng(947),
        )
        target = MetaPriorProblemAdapter(
            problem("InventorySupplyChain"), prior)
        algorithm = SingleOLHKGAlgorithm(
            target,
            SingleOLHKGConfig(
                N=4,
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
                source_constraint_mean_coefficient_prior=True,
                source_constraint_mean_adaptation_mode=(
                    "sequential_evidence_mixture"),
                hvd_source_task_weight_mode="constraint_mean",
                seed=948,
            ),
        )
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)
        constraint = algorithm.gpr[1]
        before = np.asarray(
            constraint.source_parametric_prior_diagnostics[
                "component_posterior_weights"],
            dtype=float,
        )

        x = tuple(samples[0])
        y = target.simulate(x, np.random.default_rng(949))
        sigma2 = algorithm.variance_model.predict_variance(1, x, target)
        constraint.update(x, float(y[1]), float(sigma2))
        algorithm._configure_hvd_source_task_posterior(
            algorithm.variance_model, algorithm.gpr)
        diagnostics = constraint.source_parametric_prior_diagnostics
        after = np.asarray(
            diagnostics["component_posterior_weights"], dtype=float)
        shared = algorithm.variance_model.cumulative_source_task_posterior[1]

        self.assertEqual(diagnostics["online_mixture_update_count"], 1)
        self.assertFalse(np.allclose(before, after))
        np.testing.assert_allclose(
            shared["posterior_weights"], after, atol=1e-12, rtol=1e-12)
        self.assertFalse(diagnostics["target_oracle_used"])

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

    def test_replication_variance_posterior_ignores_singletons_and_mean(self):
        posterior = ReplicationVarianceTaskPosterior(
            ["low", "high"], [0.5, 0.5], temperature=1.0)
        singleton = posterior.update_from_replication(
            (1, 4, 5), 1000.0, [0.1, 10.0], 1)
        self.assertEqual(singleton["status"], "ignored_singleton")
        np.testing.assert_allclose(
            posterior.posterior_weights(), [0.5, 0.5])

        update = posterior.update_from_replication(
            (1, 4, 5), 8.0, [0.1, 10.0], 4)
        self.assertGreater(
            posterior.posterior_weights()[1],
            posterior.posterior_weights()[0],
        )
        self.assertFalse(update["target_mean_used"])
        self.assertEqual(posterior.evidence_count, 1)
        self.assertEqual(posterior.effective_dof, 3)

        # Replacing the same policy record must not count two nested sample
        # variances as independent evidence.
        posterior.update_from_replication(
            (1, 4, 5), 9.0, [0.1, 10.0], 5)
        self.assertEqual(posterior.evidence_count, 1)
        self.assertEqual(posterior.effective_dof, 4)

    def test_isolated_variance_mixture_is_invariant_to_mean_task_weights(self):
        class ConstantModel:
            def __init__(self, mean, variance=0.0):
                self.mean = float(mean)
                self.variance = float(variance)

            def posterior_mean_many(self, X):
                return np.full(len(X), self.mean, dtype=float)

            def posterior_var_many(self, X):
                return np.full(len(X), self.variance, dtype=float)

            def posterior_mean(self, x):
                del x
                return self.mean

            def posterior_var(self, x):
                del x
                return self.variance

        class ConstantVariance:
            def __init__(self, value):
                self.value = float(value)

            def predict_variance_many(self, output_index, X, problem):
                del output_index, problem
                return np.full(len(X), self.value, dtype=float)

            def predict_certification_variance_many(
                self, output_index, X, problem,
            ):
                del output_index, problem
                return np.full(len(X), self.value, dtype=float)

            def predict_variance(self, output_index, x, problem):
                del output_index, x, problem
                return self.value

            def diagnostics(self):
                return {"status": "constant"}

        mean_posterior = FiniteTaskPosterior(
            ["left", "right"], [0.8, 0.2], safe_generalized=True)
        ensemble = FiniteTaskModelEnsemble(
            [
                TaskExpertState(
                    "left",
                    [ConstantModel(-2.0), ConstantModel(-1.0)],
                    ConstantVariance(1.0),
                    object(),
                ),
                TaskExpertState(
                    "right",
                    [ConstantModel(2.0), ConstantModel(1.0)],
                    ConstantVariance(9.0),
                    object(),
                ),
            ],
            mean_posterior,
            kl_radius_numerator=0.0,
            maximum_kl_radius=0.0,
            variance_structure_posterior_mode="replication_only",
        )
        points = [(0,), (1,)]
        before = ensemble.mixture_moments_many(1, points)
        np.testing.assert_allclose(before.aleatoric, [2.6, 2.6])

        ensemble.posterior._log_weights = np.log([0.01, 0.99])
        ensemble.posterior._log_safe_weights = np.log([0.01, 0.99])
        after = ensemble.mixture_moments_many(1, points)
        self.assertGreater(after.mean[0], before.mean[0])
        np.testing.assert_allclose(after.aleatoric, before.aleatoric)
        np.testing.assert_allclose(
            ensemble.variance_structure_weights(), [0.8, 0.2])
        self.assertEqual(len(ensemble.predictive_selector_weights()), 4)

        robust = ensemble.robust_moments_many(1, points)
        np.testing.assert_allclose(
            robust.aleatoric_upper, after.aleatoric)
        diagnostics = ensemble.diagnostics()
        self.assertEqual(
            diagnostics["variance_structure_posterior"]["status"],
            "frozen_source_prior",
        )
        self.assertFalse(diagnostics[
            "variance_structure_posterior"]["target_mean_used"])
        clone = ensemble.clone()
        clone._replication_variance_posterior().update_from_replication(
            (1, 0), 8.0, [1.0, 9.0], 4)
        np.testing.assert_allclose(
            ensemble.variance_structure_weights(), [0.8, 0.2])
        self.assertFalse(np.allclose(
            clone.variance_structure_weights(),
            ensemble.variance_structure_weights(),
        ))

    def test_precomputed_expert_moments_preserve_all_posterior_outputs(self):
        class CountingModel:
            def __init__(self, mean, variance):
                self.mean = float(mean)
                self.variance = float(variance)
                self.mean_calls = 0
                self.variance_calls = 0

            def posterior_mean_many(self, X):
                self.mean_calls += 1
                return np.full(len(X), self.mean, dtype=float)

            def posterior_var_many(self, X):
                self.variance_calls += 1
                return np.full(len(X), self.variance, dtype=float)

        class ConstantVariance:
            def __init__(self, value):
                self.value = float(value)

            def predict_variance_many(self, output_index, X, problem):
                del output_index, problem
                return np.full(len(X), self.value, dtype=float)

            def predict_certification_variance_many(
                self, output_index, X, problem,
            ):
                del output_index, problem
                return np.full(len(X), 2.0 * self.value, dtype=float)

        models = [
            CountingModel(-0.5, 0.2),
            CountingModel(0.75, 0.3),
        ]
        posterior = FiniteTaskPosterior(
            ["left", "right"], [0.4, 0.6], safe_generalized=True)
        ensemble = FiniteTaskModelEnsemble(
            [
                TaskExpertState(
                    "left",
                    [CountingModel(0.0, 0.1), models[0]],
                    ConstantVariance(0.04),
                    object(),
                ),
                TaskExpertState(
                    "right",
                    [CountingModel(1.0, 0.1), models[1]],
                    ConstantVariance(0.09),
                    object(),
                ),
            ],
            posterior,
            kl_radius_numerator=0.0,
            maximum_kl_radius=0.0,
        )
        X = [(0,), (1,), (2,)]
        raw = ensemble.expert_moments_many(
            1, X, certification=True)
        calls_after_raw = sum(
            model.mean_calls + model.variance_calls for model in models)

        cached_mix = ensemble.mixture_moments_many(
            1, X, certification=True, expert_moments=raw)
        cached_robust = ensemble.robust_moments_many(
            1, X, certification=True, expert_moments=raw)
        cached_joint = ensemble.robust_chance_margin_many(
            X,
            beta_g=2.0,
            z_alpha=1.64,
            tau=0.0,
            certification=True,
            expert_moments=raw,
        )
        self.assertEqual(
            sum(model.mean_calls + model.variance_calls for model in models),
            calls_after_raw,
        )

        uncached_mix = ensemble.mixture_moments_many(
            1, X, certification=True)
        uncached_robust = ensemble.robust_moments_many(
            1, X, certification=True)
        uncached_joint = ensemble.robust_chance_margin_many(
            X,
            beta_g=2.0,
            z_alpha=1.64,
            tau=0.0,
            certification=True,
        )
        for field in (
            "mean", "within_epistemic", "between_mean",
            "epistemic", "aleatoric", "total",
        ):
            np.testing.assert_allclose(
                getattr(cached_mix, field), getattr(uncached_mix, field))
        for field in (
            "mean_upper", "epistemic_upper", "aleatoric_upper",
            "total_upper",
        ):
            np.testing.assert_allclose(
                getattr(cached_robust, field),
                getattr(uncached_robust, field),
            )
        for field in (
            "upper", "nominal", "separable_upper",
            "tangent_epistemic_scale", "tangent_aleatoric_scale",
            "used_separable_upper",
        ):
            np.testing.assert_allclose(
                getattr(cached_joint, field), getattr(uncached_joint, field))

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

    def test_joint_kl_chance_margin_is_tighter_and_bounds_shared_task_law(self):
        posterior = FiniteTaskPosterior(
            ["mean_worst", "epistemic_worst", "aleatoric_worst"],
            [1.0 / 3.0] * 3,
            robust_dual_grid_size=81,
        )
        means = np.asarray([[1.2], [0.0], [0.0]])
        epistemic = np.asarray([[0.01], [1.5], [0.01]])
        aleatoric = np.asarray([[0.01], [0.01], [1.5]])
        beta_g = 2.0
        z_alpha = 1.6448536269514722
        radius = 0.35
        result = posterior.robust_chance_margin(
            means,
            epistemic,
            aleatoric,
            beta_g=beta_g,
            z_alpha=z_alpha,
            tau=0.0,
            radius=radius,
        )
        self.assertLess(result.upper[0], result.separable_upper[0] - 0.5)
        self.assertFalse(result.used_separable_upper[0])

        rng = np.random.default_rng(77)
        samples = rng.dirichlet(np.ones(3), size=100_000)
        center = np.full(3, 1.0 / 3.0)
        kl = np.sum(samples * (
            np.log(np.maximum(samples, 1e-300))
            - np.log(center[None, :])
        ), axis=1)
        admissible = samples[kl <= radius]
        mixture_mean = admissible @ means[:, 0]
        mixture_epistemic = (
            admissible @ epistemic[:, 0]
            + np.sum(
                admissible
                * (means[:, 0][None, :] - mixture_mean[:, None]) ** 2,
                axis=1,
            )
        )
        mixture_aleatoric = admissible @ aleatoric[:, 0]
        sampled_margin = (
            mixture_mean
            + np.sqrt(beta_g) * np.sqrt(mixture_epistemic)
            + z_alpha * np.sqrt(mixture_aleatoric)
        )
        self.assertLessEqual(
            float(np.max(sampled_margin)), result.upper[0] + 1e-10)

        centered = posterior.robust_chance_margin(
            means,
            epistemic,
            aleatoric,
            beta_g=beta_g,
            z_alpha=z_alpha,
            tau=0.0,
            radius=0.0,
        )
        self.assertAlmostEqual(centered.upper[0], centered.nominal[0])
        self.assertFalse(centered.used_separable_upper[0])

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
