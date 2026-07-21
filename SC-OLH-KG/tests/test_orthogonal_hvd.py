import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.rzdt import (  # noqa: E402
    FactorShockStatePolicyRZDT1,
    RegimeRZDT1,
    StatePolicyRZDT1,
    make_problem,
)
from problems.single_objective import ScalarizedProblem  # noqa: E402
from representation.manifold import ManifoldRiskDecomposer, PCAManifoldEncoder  # noqa: E402
from core.cumulative_risk import params_to_beta  # noqa: E402
from variance.orthogonal_hvd import (  # noqa: E402
    OrthogonalHVD,
    gaussian_square_subexp_params,
    sub_exponential_residual_square_radius,
    sub_exponential_sample_mean_radius,
)


class OrthogonalHVDTests(unittest.TestCase):
    def setUp(self):
        self.problem = RegimeRZDT1(d=3, L=90, sigma=0.05)

    def test_variance_is_nonnegative_all_modes(self):
        X = [(0, 0, 0), (10, 0, 0), (40, 0, 0), (70, 0, 0), (90, 0, 0)]
        residuals = [0.01, 0.02, 0.08, 0.09, 0.15]
        for mode in ["pooled", "class", "orthogonal", "factor"]:
            model = OrthogonalHVD(mode=mode, n_outputs=1)
            model.fit_from_residuals(X, residuals, 0, self.problem)
            for x in X:
                self.assertGreaterEqual(model.predict_variance(0, x, self.problem), 0.0)
            self.assertEqual(model.diagnostics()["mode"], mode)

    def test_class_hvd_recovers_low_high_regime_order(self):
        X = []
        residuals = []
        for x1, resid in [(5, 0.01), (10, 0.015), (20, 0.02),
                          (70, 0.20), (80, 0.22), (90, 0.24)]:
            X.append((x1, 0, 0))
            residuals.append(resid)
        model = OrthogonalHVD(mode="class", n_outputs=1, shrinkage_kappa=0.0)
        model.fit_from_residuals(X, residuals, 0, self.problem)
        low = model.predict_variance(0, (10, 0, 0), self.problem)
        high = model.predict_variance(0, (80, 0, 0), self.problem)
        self.assertGreater(high, low)
        self.assertGreater(high / max(low, 1e-12), 10.0)

    def test_update_returns_diagnostics(self):
        model = OrthogonalHVD(mode="class", n_outputs=1)
        detail = model.update(
            0,
            np.array([10, 0, 0]),
            1.0,
            0.8,
            problem=self.problem,
            epistemic_var=0.01,
        )
        self.assertEqual(detail["mode"], "class")
        self.assertIn("new_variance", detail)
        self.assertGreaterEqual(detail["new_variance"], 0.0)
        self.assertAlmostEqual(detail["raw_innovation2"], 0.04)
        self.assertAlmostEqual(detail["epistemic_correction"], 0.01)
        self.assertAlmostEqual(detail["resid2"], 0.03)

    def test_replication_replaces_singleton_residual_with_sample_variance(self):
        model = OrthogonalHVD(mode="factor", n_outputs=1)
        x = np.asarray([10, 0, 0])
        model.update(0, x, 1.0, 0.8, problem=self.problem)
        detail = model.update(
            0,
            x,
            0.9,
            0.8,
            problem=self.problem,
            replicate_variance=0.005,
            replicate_count=8,
        )
        self.assertEqual(detail["variance_source"], "within_solution_replication")
        self.assertEqual(len(model.records[0]), 1)
        self.assertAlmostEqual(model.records[0][0][1], 0.005)
        self.assertEqual(model.diagnostics()["replicated_solution_count"]["0"], 1)
        self.assertEqual(detail["replication_dof"], 7.0)

    def test_replication_degrees_of_freedom_contract_tail_guard(self):
        low_dof = OrthogonalHVD(mode="factor", n_outputs=1)
        high_dof = OrthogonalHVD(mode="factor", n_outputs=1)
        for index in range(12):
            x = np.asarray([index + 1, 0, 0])
            for model, count in ((low_dof, 2), (high_dof, 16)):
                model.update(
                    0,
                    x,
                    0.0,
                    0.0,
                    problem=self.problem,
                    replicate_variance=0.04,
                    replicate_count=count,
                )
        low_diag = low_dof.diagnostics()["residual_square_tail"]["0"]
        high_diag = high_dof.diagnostics()["residual_square_tail"]["0"]
        self.assertEqual(low_diag["effective_dof"], 12.0)
        self.assertEqual(high_diag["effective_dof"], 180.0)
        self.assertLess(high_diag["uncertainty"], low_diag["uncertainty"])

    def test_residual_square_tail_radius_is_exposed(self):
        nu, b = gaussian_square_subexp_params(0.04)
        loose = sub_exponential_residual_square_radius(nu, b, 0.20)
        tight = sub_exponential_residual_square_radius(nu, b, 0.01)
        self.assertGreater(nu, 0.0)
        self.assertGreater(b, 0.0)
        self.assertGreater(tight, loose)
        model = OrthogonalHVD(mode="class", n_outputs=1)
        model.fit_from_residuals([(10, 0, 0), (80, 0, 0)], [0.1, 0.2], 0, self.problem)
        tail = model.diagnostics()["residual_square_tail"]["0"]
        self.assertEqual(tail["delta"], 0.05)
        self.assertGreater(tail["radius"], 0.0)

    def test_sample_mean_tail_uses_bernstein_branch_scaling(self):
        nu, b = gaussian_square_subexp_params(0.04)
        single = sub_exponential_residual_square_radius(nu, b, 0.05)
        radius_4 = sub_exponential_sample_mean_radius(nu, b, 0.05, 4)
        radius_16 = sub_exponential_sample_mean_radius(nu, b, 0.05, 16)
        self.assertLess(radius_4, single)
        self.assertLess(radius_16, radius_4)
        self.assertLessEqual(radius_16, single / np.sqrt(16.0))
        with self.assertRaises(ValueError):
            sub_exponential_sample_mean_radius(nu, b, 0.05, 0)

    def test_orthogonal_delays_activation_until_enough_records(self):
        X = [(5, 0, 0), (10, 0, 0), (70, 0, 0)]
        residuals = [0.01, 0.02, 0.2]
        model = OrthogonalHVD(
            mode="orthogonal",
            n_outputs=1,
            activation_min_records=5,
        )
        model.fit_from_residuals(X, residuals, 0, self.problem)
        diag = model.diagnostics()
        self.assertFalse(diag["orthogonal_active"]["0"])
        pred = model.predict_variance(0, (10, 0, 0), self.problem)
        cert = model.predict_certification_variance(0, (10, 0, 0), self.problem)
        self.assertGreaterEqual(cert, pred)

    def test_orthogonal_certification_uses_class_floor(self):
        X = [(35, 0, 0), (40, 0, 0), (45, 0, 0), (70, 0, 0), (80, 0, 0)]
        residuals = [0.08, 0.09, 0.10, 0.02, 0.02]
        model = OrthogonalHVD(
            mode="orthogonal",
            n_outputs=1,
            activation_min_records=3,
            shrinkage_kappa=0.0,
        )
        model.fit_from_residuals(X, residuals, 0, self.problem)
        x = (42, 0, 0)
        decomposition = model.predict_decomposition(0, x, self.problem)
        self.assertGreaterEqual(
            decomposition["certification_variance"],
            decomposition["class_variance"],
        )
        self.assertTrue(model.diagnostics()["certification_uses_class_floor"])

    def test_problem_residual_cap_limits_single_large_residual(self):
        problem = StatePolicyRZDT1(d=5, L=100, sigma=0.04)
        x = (25, 70, 70, 70, 70)
        model = OrthogonalHVD(mode="class", n_outputs=1, shrinkage_kappa=0.0)
        model.fit_from_residuals([x], [10.0], 0, problem)
        cap = problem.hvd_residual_variance_cap(0)
        self.assertLessEqual(model.predict_variance(0, x, problem), cap)
        self.assertEqual(model.diagnostics()["residual_variance_cap"]["0"], cap)

    def test_update_clips_residual_before_float_overflow(self):
        problem = StatePolicyRZDT1(d=5, L=100, sigma=0.04)
        x = np.array([25, 70, 70, 70, 70])
        model = OrthogonalHVD(mode="class", n_outputs=1, shrinkage_kappa=0.0)
        detail = model.update(0, x, 1e200, 0.0, problem=problem)
        cap = problem.hvd_residual_variance_cap(0)
        self.assertTrue(np.isfinite(detail["resid2"]))
        self.assertLessEqual(detail["resid2"], cap)
        self.assertLessEqual(model.predict_variance(0, tuple(x), problem), cap)

    def test_factor_shock_oracle_sigma_matches_cumulative_formula(self):
        base = FactorShockStatePolicyRZDT1(d=8, L=100, sigma=0.04)
        problem = ScalarizedProblem(base)
        x = (25, 72, 72, 72, 72, 72, 72, 72)
        decomp = problem.true_cumulative_risk_decomposition(x, output_index=1)
        self.assertIsNotNone(decomp)
        self.assertGreater(decomp["shared"], 0.0)
        self.assertAlmostEqual(problem.true_sigma(x)[1] ** 2, decomp["total"])
        features = problem.cumulative_risk_features(x, output_index=1)
        names = problem.cumulative_risk_feature_names(output_index=1)
        self.assertEqual(len(features), len(names))

    def test_problem_factory_exposes_registered_shared_shock_scale(self):
        low = ScalarizedProblem(make_problem(
            "FactorShockStatePolicyRZDT1",
            d=8,
            L=100,
            sigma=0.04,
            shared_shock_scale=0.0,
        ))
        high = ScalarizedProblem(make_problem(
            "FactorShockStatePolicyRZDT1",
            d=8,
            L=100,
            sigma=0.04,
            shared_shock_scale=4.0,
        ))
        point = tuple([50] + [95] * 7)
        self.assertGreater(
            high.true_sigma(point)[1] ** 2,
            low.true_sigma(point)[1] ** 2,
        )

    def test_factor_hvd_learns_cumulative_risk_ordering(self):
        base = FactorShockStatePolicyRZDT1(d=8, L=100, sigma=0.04)
        problem = ScalarizedProblem(base)
        X = []
        residuals = []
        for u in [5, 15, 25, 40, 60, 80, 95]:
            for q in [45, 60, 72, 85, 95]:
                x = tuple([u] + [q] * 7)
                X.append(x)
                residuals.append(np.sqrt(problem.true_sigma(x)[1] ** 2))
        model = OrthogonalHVD(
            mode="factor",
            n_outputs=2,
            activation_min_records=10,
            shrinkage_kappa=0.0,
        )
        model.fit_from_residuals(X, residuals, output_index=1, problem=problem)
        low = tuple([25] + [72] * 7)
        high = tuple([50] + [95] * 7)
        self.assertTrue(model.diagnostics()["cumulative_active"]["1"])
        self.assertGreater(
            model.predict_variance(1, high, problem),
            model.predict_variance(1, low, problem),
        )
        decomposition = model.predict_decomposition(1, high, problem)
        self.assertIsNotNone(decomposition["cumulative"])
        self.assertTrue(decomposition["cumulative"]["active"])
        self.assertIsNotNone(decomposition["cumulative"]["oracle"])
        self.assertIsNotNone(decomposition["cumulative"]["fitted_blocks"])
        blocks = decomposition["cumulative"]["fitted_blocks"]
        self.assertAlmostEqual(
            blocks["total"],
            blocks["floor"] + blocks["independent"] + blocks["shared"] + blocks["linear"],
        )
        diagnostics = model.diagnostics()
        self.assertEqual(
            diagnostics["cumulative_fit_method"]["1"],
            "replication_aware_projected_irls",
        )
        self.assertGreaterEqual(
            diagnostics["cumulative_fit_effective_dof"]["1"],
            float(len(X)),
        )

    def test_source_hvd_shape_uses_scalar_only_target_replication_calibration(self):
        base = ScalarizedProblem(
            FactorShockStatePolicyRZDT1(d=8, L=100, sigma=0.04))
        source_scale = 0.01

        class FrozenShapeProblem:
            def __init__(self):
                self.information_cap = None

            def __getattr__(self, name):
                return getattr(base, name)

            def cumulative_hvd_prior_beta(
                self, output_index=1, feature_dim=None,
            ):
                beta = params_to_beta(
                    base.cumulative_risk_parameters(output_index)) / source_scale
                if feature_dim is not None and len(beta) != int(feature_dim):
                    return None
                return beta

            @staticmethod
            def cumulative_hvd_prior_precision(output_index=1):
                del output_index
                return 1.0

            @staticmethod
            def cumulative_hvd_prior_scale_mean(output_index=1):
                del output_index
                return source_scale

            @staticmethod
            def cumulative_hvd_prior_upper_scale(output_index=1):
                del output_index
                return 1.0

            @staticmethod
            def cumulative_hvd_prior_min_records():
                return 1

            def hvd_residual_variance_cap(self, output_index=0):
                if self.information_cap is not None:
                    return float(self.information_cap)
                return base.hvd_residual_variance_cap(output_index)

        problem = FrozenShapeProblem()
        model = OrthogonalHVD(
            mode="factor",
            n_outputs=2,
            activation_min_records=1,
        )
        target_scale = 0.04
        points = [
            tuple([15] + [55] * 7),
            tuple([35] + [72] * 7),
            tuple([65] + [88] * 7),
            tuple([85] + [95] * 7),
        ]
        source_shape = problem.cumulative_hvd_prior_beta(1)
        for point in points:
            feature = problem.cumulative_risk_features(point, output_index=1)
            model.update(
                1,
                point,
                0.0,
                0.0,
                problem=problem,
                replicate_variance=target_scale * float(feature @ source_shape),
                replicate_count=8,
            )

        diagnostics = model.diagnostics()
        self.assertEqual(
            diagnostics["cumulative_fit_method"]["1"],
            "replication_scalar_calibration",
        )
        fitted = np.asarray(model.cumulative_beta[1], dtype=float)
        fitted_shape = fitted / max(float(model.cumulative_prior_scale[1]), 1e-12)
        np.testing.assert_allclose(fitted_shape, source_shape, rtol=1e-10, atol=1e-12)
        self.assertGreater(model.cumulative_prior_scale[1], source_scale)
        self.assertLess(model.cumulative_prior_scale[1], target_scale * 1.01)
        self.assertEqual(
            diagnostics["cumulative_fit_effective_dof"]["1"], 28.0)
        self.assertEqual(
            diagnostics["cumulative_information_geometry"]["1"],
            "scaled_chi_square_scalar",
        )

        reliability = np.asarray([1.0, 0.25, 0.75, 0.5], dtype=float)
        reference_weights = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float)
        fitted_scale = float(model.cumulative_prior_scale[1])
        precision = (1.0 + 28.0) / (2.0 * fitted_scale ** 2)
        normalized_weights = reference_weights / np.sum(reference_weights)
        shape_values = np.asarray([
            float(problem.cumulative_risk_features(
                point, output_index=1) @ source_shape)
            for point in points
        ])
        unclipped = fitted_scale * shape_values
        ordered_unclipped = np.sort(unclipped)
        problem.information_cap = float(
            0.5 * (ordered_unclipped[1] + ordered_unclipped[2]))
        cap = float(problem.hvd_residual_variance_cap(output_index=1))
        actual_reduction = model.information_reduction_many(
            1,
            points,
            points,
            problem,
            action_reliability=reliability,
            reference_weights=reference_weights,
        )
        active = (unclipped > model.floor) & (unclipped < cap)
        self.assertTrue(np.any(active))
        self.assertTrue(np.any(~active))
        derivative = np.where(active, shape_values, 0.0)
        integrated_shape_square = float(np.sum(
            normalized_weights * derivative ** 2))
        increment = reliability * active / (2.0 * fitted_scale ** 2)
        expected_reduction = (
            increment / (precision * (precision + increment))
            * integrated_shape_square
        )
        np.testing.assert_allclose(
            actual_reduction,
            expected_reduction,
            rtol=1e-12,
            atol=1e-15,
        )
        np.testing.assert_array_equal(actual_reduction[~active], 0.0)

    def test_source_shape_mixture_learns_target_shape_and_shrinks_guard(self):
        base = ScalarizedProblem(
            FactorShockStatePolicyRZDT1(d=8, L=100, sigma=0.04))
        physical = params_to_beta(
            base.cumulative_risk_parameters(output_index=1))
        component_a = physical / max(float(np.mean(physical)), 1e-6)
        component_b = component_a.copy()
        component_b[0] *= 2.5
        component_b[1:4] *= 0.35
        component_b[4:7] *= 1.8

        class FrozenMixtureProblem:
            def __getattr__(self, name):
                return getattr(base, name)

            @staticmethod
            def cumulative_hvd_prior_beta(output_index=1, feature_dim=None):
                del output_index
                beta = 0.5 * (component_a + component_b)
                if feature_dim is not None and len(beta) != int(feature_dim):
                    return None
                return beta

            @staticmethod
            def cumulative_hvd_prior_components(output_index=1, feature_dim=None):
                del output_index
                coefficients = np.vstack([component_a, component_b])
                if feature_dim is not None and coefficients.shape[1] != int(feature_dim):
                    return None
                return {
                    "coefficients": coefficients,
                    "domains": ["source_a", "source_b"],
                }

            @staticmethod
            def cumulative_hvd_prior_precision(output_index=1):
                del output_index
                return 1.0

            @staticmethod
            def cumulative_hvd_prior_scale_mean(output_index=1):
                del output_index
                return 0.02

            @staticmethod
            def cumulative_hvd_prior_upper_scale(output_index=1):
                del output_index
                return 2.0

            @staticmethod
            def cumulative_hvd_prior_min_records():
                return 1

        problem = FrozenMixtureProblem()
        model = OrthogonalHVD(
            mode="factor",
            n_outputs=2,
            activation_min_records=1,
            cumulative_transfer_mode="source_mixture",
        )
        probe = tuple([10] + [45] * 7)
        model.update(1, probe, 0.1, 0.0, problem=problem)
        guard_before = model._source_mixture_guard(1, probe, problem)

        target_weights = np.asarray([0.032, 0.008], dtype=float)
        points = [
            tuple([15] + [55] * 7),
            tuple([35] + [72] * 7),
            tuple([65] + [88] * 7),
            tuple([85] + [95] * 7),
        ]
        for point in points:
            feature = problem.cumulative_risk_features(point, output_index=1)
            component_value = feature @ np.vstack([component_a, component_b]).T
            model.update(
                1,
                point,
                0.0,
                0.0,
                problem=problem,
                replicate_variance=float(component_value @ target_weights),
                replicate_count=8,
            )

        diagnostics = model.diagnostics()
        self.assertEqual(
            diagnostics["cumulative_fit_method"]["1"],
            "replication_source_shape_mixture",
        )
        self.assertEqual(
            diagnostics["cumulative_information_geometry"]["1"],
            "scaled_chi_square_source_shape_mixture",
        )
        self.assertEqual(
            diagnostics["cumulative_prior_component_domains"]["1"],
            ["source_a", "source_b"],
        )
        self.assertEqual(
            diagnostics["cumulative_prior_shape_target_dof"]["1"], 28.0)
        fitted_weights = np.asarray(
            diagnostics["cumulative_prior_component_weights"]["1"])
        self.assertTrue(np.all(fitted_weights >= 0.0))
        self.assertGreater(fitted_weights[0], fitted_weights[1])
        self.assertLess(
            model._source_mixture_guard(1, probe, problem), guard_before)
        for point in points:
            self.assertGreaterEqual(
                model.predict_decomposition(1, point, problem)["cumulative"][
                    "fitted_blocks"]["shared"],
                -1e-12,
            )

    def test_source_prior_singletons_are_independent_of_constraint_mean_head(self):
        base = ScalarizedProblem(
            FactorShockStatePolicyRZDT1(d=8, L=100, sigma=0.04))
        physical = params_to_beta(
            base.cumulative_risk_parameters(output_index=1))
        normalized = physical / max(float(np.mean(physical)), 1e-6)

        class FrozenProblem:
            def __getattr__(self, name):
                return getattr(base, name)

            @staticmethod
            def cumulative_hvd_prior_beta(output_index=1, feature_dim=None):
                del output_index
                if feature_dim is not None and len(normalized) != int(feature_dim):
                    return None
                return normalized

            @staticmethod
            def cumulative_hvd_prior_components(
                output_index=1, feature_dim=None,
            ):
                del output_index
                coefficients = np.vstack([normalized, 1.2 * normalized])
                if feature_dim is not None and coefficients.shape[1] != int(feature_dim):
                    return None
                return {
                    "coefficients": coefficients,
                    "domains": ["source_a", "source_b"],
                }

            @staticmethod
            def cumulative_hvd_prior_precision(output_index=1):
                del output_index
                return 1.0

            @staticmethod
            def cumulative_hvd_prior_scale_mean(output_index=1):
                del output_index
                return 0.02

            @staticmethod
            def cumulative_hvd_prior_upper_scale(output_index=1):
                del output_index
                return 2.0

            @staticmethod
            def cumulative_hvd_prior_min_records():
                return 1

        class ConstantMean:
            def __init__(self, value):
                self.value = float(value)

            def posterior_mean(self, _x):
                return self.value

        problem = FrozenProblem()
        samples = [
            tuple([value] + [45] * 7) for value in (10, 25, 40, 55, 70)
        ]
        observations = {
            point: [np.asarray([0.3, -0.2], dtype=float)]
            for point in samples
        }
        left = OrthogonalHVD(
            mode="factor",
            n_outputs=2,
            activation_min_records=1,
            cumulative_transfer_mode="source_mixture",
            singleton_evidence_mode="source_prior",
        )
        right = OrthogonalHVD(
            mode="factor",
            n_outputs=2,
            activation_min_records=1,
            cumulative_transfer_mode="source_mixture",
            singleton_evidence_mode="source_prior",
        )
        left.initialize(
            samples,
            observations,
            [ConstantMean(-100.0), ConstantMean(-100.0)],
            problem,
        )
        right.initialize(
            samples,
            observations,
            [ConstantMean(100.0), ConstantMean(100.0)],
            problem,
        )

        for output_index in range(2):
            np.testing.assert_allclose(
                [value for _, value in left.records[output_index]],
                [value for _, value in right.records[output_index]],
                atol=0.0,
                rtol=0.0,
            )
            self.assertAlmostEqual(
                left.global_var[output_index], right.global_var[output_index])
            self.assertEqual(
                left.diagnostics()["source_prior_singleton_count"][
                    str(output_index)],
                len(samples),
            )
            self.assertEqual(
                left.diagnostics()["residual_square_tail"][str(output_index)][
                    "effective_dof"],
                0.0,
            )

        probe = samples[0]
        before = left.predict_variance(1, probe, problem)
        detail = left.update(
            1, probe, 1e6, -1e6, problem=problem)
        self.assertEqual(detail["variance_source"], "source_prior_singleton")
        self.assertAlmostEqual(
            left.predict_variance(1, probe, problem), before)
        left.update(
            1,
            probe,
            0.0,
            0.0,
            problem=problem,
            replicate_variance=0.003,
            replicate_count=8,
        )
        self.assertEqual(
            left.diagnostics()["source_prior_singleton_count"]["1"],
            len(samples) - 1,
        )
        self.assertEqual(
            left.diagnostics()["replicated_solution_count"]["1"], 1)

    def test_constraint_mean_task_posterior_controls_hvd_shape_mixture(self):
        base = ScalarizedProblem(
            FactorShockStatePolicyRZDT1(d=8, L=100, sigma=0.04))
        physical = params_to_beta(
            base.cumulative_risk_parameters(output_index=1))
        source_a = physical / max(float(np.mean(physical)), 1e-6)
        source_b = source_a.copy()
        source_b[0] *= 4.0

        class CoupledMixtureProblem:
            def __getattr__(self, name):
                return getattr(base, name)

            @staticmethod
            def cumulative_hvd_prior_beta(output_index=1, feature_dim=None):
                del output_index
                beta = 0.5 * (source_a + source_b)
                if feature_dim is not None and len(beta) != int(feature_dim):
                    return None
                return beta

            @staticmethod
            def cumulative_hvd_prior_components(output_index=1, feature_dim=None):
                del output_index
                coefficients = np.vstack([source_a, source_b])
                if feature_dim is not None and coefficients.shape[1] != int(feature_dim):
                    return None
                return {
                    "coefficients": coefficients,
                    "domains": ["source_a", "source_b"],
                }

            @staticmethod
            def cumulative_hvd_prior_precision(output_index=1):
                del output_index
                return 1.0

            @staticmethod
            def cumulative_hvd_prior_scale_mean(output_index=1):
                del output_index
                return 0.02

            @staticmethod
            def cumulative_hvd_prior_upper_scale(output_index=1):
                del output_index
                return 2.0

            @staticmethod
            def cumulative_hvd_prior_min_records():
                return 1

        problem = CoupledMixtureProblem()
        model = OrthogonalHVD(
            mode="factor",
            n_outputs=2,
            activation_min_records=1,
            cumulative_transfer_mode="source_mixture",
            cumulative_source_task_weight_mode="constraint_mean",
        )
        model.set_source_task_posterior(
            1,
            ["source:source_a", "source:source_b", "target:null"],
            [0.0, 0.0, 1.0],
        )
        probe = tuple([10] + [45] * 7)
        model.update(1, probe, 0.1, 0.0, problem=problem)
        diagnostics = model.diagnostics()
        self.assertEqual(
            diagnostics["cumulative_prior_component_domains"]["1"],
            ["source_a", "source_b", "target:null"],
        )
        np.testing.assert_allclose(
            diagnostics["cumulative_prior_component_weights"]["1"],
            [0.0, 0.0, 0.02],
            atol=1e-12,
        )
        self.assertAlmostEqual(
            model.predict_cumulative_variance(1, probe, problem),
            model.global_var[1],
            places=12,
        )
        self.assertFalse(
            diagnostics["cumulative_source_task_posterior"]["1"][
                "target_oracle_used"])

    def test_prequential_upper_updates_source_shape_without_replication(self):
        base = ScalarizedProblem(
            FactorShockStatePolicyRZDT1(d=8, L=100, sigma=0.04))
        physical = params_to_beta(
            base.cumulative_risk_parameters(output_index=1))
        source_a = physical / max(float(np.mean(physical)), 1e-6)
        source_b = source_a.copy()
        source_b[0] *= 2.0
        components = np.vstack([source_a, source_b])

        class PrequentialProblem:
            def __getattr__(self, name):
                return getattr(base, name)

            @staticmethod
            def cumulative_hvd_prior_beta(output_index=1, feature_dim=None):
                del output_index
                beta = 0.5 * (source_a + source_b)
                if feature_dim is not None and len(beta) != int(feature_dim):
                    return None
                return beta

            @staticmethod
            def cumulative_hvd_prior_components(
                output_index=1, feature_dim=None
            ):
                del output_index
                if feature_dim is not None and components.shape[1] != int(
                    feature_dim
                ):
                    return None
                return {
                    "coefficients": components,
                    "domains": ["source_a", "source_b"],
                }

            @staticmethod
            def cumulative_hvd_prior_precision(output_index=1):
                del output_index
                return 1.0

            @staticmethod
            def cumulative_hvd_prior_scale_mean(output_index=1):
                del output_index
                return 0.02

            @staticmethod
            def cumulative_hvd_prior_upper_scale(output_index=1):
                del output_index
                return 2.0

            @staticmethod
            def cumulative_hvd_prior_min_records():
                return 1

        problem = PrequentialProblem()
        control = OrthogonalHVD(
            mode="factor",
            n_outputs=2,
            activation_min_records=1,
            cumulative_transfer_mode="source_mixture",
            cumulative_target_evidence_mode="replication_only",
        )
        challenger = OrthogonalHVD(
            mode="factor",
            n_outputs=2,
            activation_min_records=1,
            cumulative_transfer_mode="source_mixture",
            cumulative_target_evidence_mode="prequential_upper",
        )
        target_weights = np.asarray([0.03, 0.01], dtype=float)
        points = [
            tuple([15] + [55] * 7),
            tuple([35] + [72] * 7),
            tuple([65] + [88] * 7),
            tuple([85] + [95] * 7),
        ]
        for point in points:
            feature = problem.cumulative_risk_features(
                point, output_index=1)
            variance = float((feature @ components.T) @ target_weights)
            observation = float(np.sqrt(max(variance, 1e-12)))
            for model in (control, challenger):
                model.update(
                    1,
                    point,
                    observation,
                    0.0,
                    problem=problem,
                    epistemic_var=0.0,
                )

        control_diagnostics = control.diagnostics()
        challenger_diagnostics = challenger.diagnostics()
        self.assertEqual(
            control_diagnostics["cumulative_prior_shape_target_dof"]["1"],
            0.0,
        )
        self.assertEqual(
            challenger_diagnostics[
                "cumulative_prior_shape_target_dof"]["1"],
            4.0,
        )
        self.assertEqual(
            challenger_diagnostics["cumulative_fit_method"]["1"],
            "prequential_upper_source_shape_mixture",
        )
        self.assertEqual(
            challenger_diagnostics["prequential_upper_solution_count"]["1"],
            4,
        )
        self.assertEqual(
            challenger_diagnostics["cumulative_prior_scale_source"]["1"],
            "prequential_upper",
        )
        margin_gain = challenger.certification_margin_information_reduction_many(
            1,
            [points[0]],
            points,
            problem,
            action_reliability=[1.0],
            reference_weights=np.ones(len(points)),
            z_alpha=1.6448536269514722,
        )
        self.assertEqual(margin_gain.shape, (1,))
        self.assertTrue(np.all(np.isfinite(margin_gain)))
        self.assertGreaterEqual(float(margin_gain[0]), 0.0)
        current_radius = np.mean(np.sqrt(
            challenger.predict_certification_variance_many(
                1, points, problem))) * 1.6448536269514722
        self.assertLessEqual(float(margin_gain[0]), current_radius + 1e-12)

    def test_factor_certification_includes_residual_tail_guard(self):
        base = FactorShockStatePolicyRZDT1(d=8, L=100, sigma=0.04)
        problem = ScalarizedProblem(base)
        X = []
        residuals = []
        for u in [5, 15, 25, 40, 60, 80, 95]:
            for q in [45, 60, 72, 85, 95]:
                x = tuple([u] + [q] * 7)
                X.append(x)
                residuals.append(problem.true_sigma(x)[1])
        model = OrthogonalHVD(
            mode="factor",
            n_outputs=2,
            activation_min_records=10,
            shrinkage_kappa=0.0,
            residual_tail_delta=0.05,
        )
        model.fit_from_residuals(X, residuals, output_index=1, problem=problem)
        x = tuple([50] + [95] * 7)
        decomposition = model.predict_decomposition(1, x, problem)
        self.assertTrue(decomposition["cumulative"]["active"])
        self.assertGreater(decomposition["residual_tail_uncertainty"], 0.0)
        self.assertGreaterEqual(
            decomposition["certification_variance"],
            decomposition["variance"] + decomposition["residual_tail_uncertainty"],
        )

    def test_hvd_information_reduction_is_reliability_aware(self):
        base = FactorShockStatePolicyRZDT1(d=8, L=100, sigma=0.04)
        problem = ScalarizedProblem(base)
        model = OrthogonalHVD(
            mode="factor",
            n_outputs=2,
            activation_min_records=2,
        )
        points = [
            tuple([15] + [55] * 7),
            tuple([35] + [72] * 7),
            tuple([65] + [88] * 7),
        ]
        for point in points:
            model.update(
                1,
                point,
                0.0,
                0.0,
                problem=problem,
                replicate_variance=problem.true_sigma(point)[1] ** 2,
                replicate_count=2,
            )
        action = [tuple([50] + [80] * 7)]
        clean = model.information_reduction_many(
            1,
            action,
            points,
            problem,
            action_reliability=[1.0],
        )
        noisy = model.information_reduction_many(
            1,
            action,
            points,
            problem,
            action_reliability=[0.2],
        )
        self.assertEqual(clean.shape, (1,))
        self.assertTrue(np.all(np.isfinite(clean)))
        self.assertGreater(clean[0], noisy[0])
        self.assertGreaterEqual(noisy[0], 0.0)

    def test_replication_reduces_future_hvd_information_gain(self):
        base = FactorShockStatePolicyRZDT1(d=8, L=100, sigma=0.04)
        problem = ScalarizedProblem(base)
        model = OrthogonalHVD(
            mode="factor",
            n_outputs=2,
            activation_min_records=2,
        )
        points = [
            tuple([15] + [55] * 7),
            tuple([35] + [72] * 7),
            tuple([65] + [88] * 7),
        ]
        for point in points:
            model.update(
                1,
                point,
                0.0,
                0.0,
                problem=problem,
                replicate_variance=problem.true_sigma(point)[1] ** 2,
                replicate_count=2,
            )
        target = points[1]
        before = model.information_reduction_many(
            1, [target], points, problem, action_reliability=[1.0])
        model.update(
            1,
            target,
            0.0,
            0.0,
            problem=problem,
            replicate_variance=problem.true_sigma(target)[1] ** 2,
            replicate_count=8,
        )
        after = model.information_reduction_many(
            1, [target], points, problem, action_reliability=[1.0])
        self.assertLess(after[0], before[0])

    def test_cumulative_statistical_design_reports_active_excitation(self):
        base = FactorShockStatePolicyRZDT1(d=8, L=100, sigma=0.04)
        problem = ScalarizedProblem(base)
        model = OrthogonalHVD(
            mode="factor",
            n_outputs=2,
            activation_min_records=2,
        )
        points = [
            tuple([10] + [50] * 7),
            tuple([35] + [70] * 7),
            tuple([70] + [90] * 7),
        ]
        for point in points:
            model.update(
                1,
                point,
                0.0,
                0.0,
                problem=problem,
                replicate_variance=problem.true_sigma(point)[1] ** 2,
                replicate_count=4,
            )
        audit = model.diagnostics()["cumulative_statistical_design"]["1"]
        self.assertEqual(audit["theory_contract"],
                         "v51_statistical_closure_v2")
        self.assertEqual(audit["replicated_solution_count"], 3)
        self.assertEqual(audit["target_evidence_solution_count"], 3)
        self.assertGreater(audit["raw_feature_dimension"], 0)
        self.assertGreater(audit["active_calibration_dimension"], 0)
        self.assertLessEqual(
            audit["active_geometry"]["rank"],
            audit["active_calibration_dimension"],
        )
        self.assertGreaterEqual(
            audit["active_geometry"]["minimum_eigenvalue"], 0.0)
        self.assertAlmostEqual(
            audit["lean_excitation_kappa"],
            audit["target_evidence_solution_count"]
            * audit["active_geometry"]["minimum_eigenvalue"],
        )

    def test_statistical_design_diagnostics_are_read_only(self):
        base = FactorShockStatePolicyRZDT1(d=8, L=100, sigma=0.04)
        problem = ScalarizedProblem(base)
        first = OrthogonalHVD(mode="factor", n_outputs=2,
                              activation_min_records=2)
        second = OrthogonalHVD(mode="factor", n_outputs=2,
                               activation_min_records=2)
        points = [
            tuple([10] + [50] * 7),
            tuple([35] + [70] * 7),
            tuple([70] + [90] * 7),
        ]
        for point in points[:2]:
            kwargs = dict(
                i=1,
                x=point,
                y=0.0,
                mu=0.0,
                problem=problem,
                replicate_variance=problem.true_sigma(point)[1] ** 2,
                replicate_count=4,
            )
            first.update(**kwargs)
            second.update(**kwargs)
        first.diagnostics()
        point = points[2]
        kwargs = dict(
            i=1,
            x=point,
            y=0.0,
            mu=0.0,
            problem=problem,
            replicate_variance=problem.true_sigma(point)[1] ** 2,
            replicate_count=4,
        )
        first.update(**kwargs)
        second.update(**kwargs)
        np.testing.assert_allclose(
            first.cumulative_beta[1], second.cumulative_beta[1])
        self.assertEqual(first.cumulative_fit_method[1],
                         second.cumulative_fit_method[1])

    def test_factor_hvd_accepts_manifold_cumulative_blocks(self):
        base = FactorShockStatePolicyRZDT1(d=8, L=100, sigma=0.04)
        problem = ScalarizedProblem(base)
        encoder = PCAManifoldEncoder(
            problem,
            latent_dim=4,
            fit_pool_size=32,
            rng=np.random.default_rng(7),
        )
        problem._scolhkg_representation_encoder = encoder
        problem._scolhkg_use_manifold_hvd = True
        problem._scolhkg_manifold_decomposer = ManifoldRiskDecomposer(encoder)

        X = []
        residuals = []
        for u in [5, 15, 25, 40, 60, 80, 95]:
            for q in [45, 60, 72, 85, 95]:
                x = tuple([u] + [q] * 7)
                X.append(x)
                residuals.append(problem.true_sigma(x)[1])
        model = OrthogonalHVD(
            mode="factor",
            n_outputs=2,
            activation_min_records=10,
            shrinkage_kappa=0.0,
        )
        model.fit_from_residuals(X, residuals, output_index=1, problem=problem)
        x = tuple([50] + [95] * 7)
        decomposition = model.predict_decomposition(1, x, problem)
        cumulative = decomposition["cumulative"]
        self.assertTrue(model.diagnostics()["uses_manifold_hvd_features"])
        self.assertIsNotNone(cumulative["manifold_blocks"])
        blocks = cumulative["manifold_blocks"]
        self.assertAlmostEqual(
            blocks["total"],
            blocks["tangent"] + blocks["normal"] + blocks["shared"] + blocks["residual"],
        )
        self.assertTrue(any(
            str(name).startswith("manifold_")
            for name in cumulative["feature_names"]
        ))

    def test_omitting_shared_shock_can_flip_false_feasible_certificate(self):
        base = FactorShockStatePolicyRZDT1(d=8, L=100, sigma=0.04)
        problem = ScalarizedProblem(base)
        x = tuple([50] + [95] * 7)
        decomp = problem.true_cumulative_risk_decomposition(x, output_index=1)
        self.assertGreater(decomp["shared"], 0.0)
        no_shared = decomp["floor"] + decomp["independent"] + decomp["linear"]
        self.assertLess(no_shared, decomp["total"])

        z_alpha = 1.6448536269514722
        mu_g = -z_alpha * 0.5 * (
            np.sqrt(no_shared) + np.sqrt(decomp["total"]))
        tau = 0.0
        no_shared_margin = mu_g + z_alpha * np.sqrt(no_shared) - tau
        true_margin = mu_g + z_alpha * np.sqrt(decomp["total"]) - tau
        self.assertLessEqual(no_shared_margin, 0.0)
        self.assertGreater(true_margin, 0.0)


if __name__ == "__main__":
    unittest.main()
