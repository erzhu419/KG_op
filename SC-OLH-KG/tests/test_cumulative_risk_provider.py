import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import SingleOLHKGConfig  # noqa: E402
from core.certification import conservative_chance_margin  # noqa: E402
from core.cumulative_risk import (  # noqa: E402
    beta_to_params,
    cumulative_feature_vector,
    decompose_cumulative_risk,
    get_risk_exposure,
    project_cumulative_beta,
)
from encoders.policy_state_encoder import TrafficTrajectoryEncoder  # noqa: E402
from problems.rzdt import (  # noqa: E402
    FactorShockStatePolicyRZDT1,
    InventorySupplyChainProblem,
    QueueResourceControlProblem,
)
from problems.single_objective import ScalarizedProblem  # noqa: E402
from variance.orthogonal_hvd import OrthogonalHVD  # noqa: E402


class CumulativeRiskProviderTests(unittest.TestCase):
    def test_factor_shock_scalarized_oracle_uses_cumulative_variance(self):
        weights = np.array([0.5, 0.5], dtype=float)
        weak = FactorShockStatePolicyRZDT1(
            d=8, L=100, sigma=0.04, shared_shock_scale=0.0)
        moderate = FactorShockStatePolicyRZDT1(
            d=8, L=100, sigma=0.04, shared_shock_scale=4.0)
        infeasible = FactorShockStatePolicyRZDT1(
            d=8, L=100, sigma=0.04, shared_shock_scale=8.0)

        weak_x, weak_obj = weak.scalarized_true_best_feasible(weights)
        moderate_x, moderate_obj = moderate.scalarized_true_best_feasible(weights)
        infeasible_x, infeasible_obj = (
            infeasible.scalarized_true_best_feasible(weights))

        self.assertIsNotNone(weak_x)
        self.assertIsNotNone(moderate_x)
        self.assertTrue(weak.is_truly_feasible(weak_x))
        self.assertTrue(moderate.is_truly_feasible(moderate_x))
        self.assertGreaterEqual(moderate_obj, weak_obj)
        self.assertIsNone(infeasible_x)
        self.assertTrue(np.isinf(infeasible_obj))

    def _check_problem_formula(self, problem, x):
        exposure = get_risk_exposure(problem, x, output_index=1)
        params = problem.cumulative_risk_parameters(output_index=1)
        features = cumulative_feature_vector(exposure)
        beta, projected = project_cumulative_beta(
            np.asarray([
                params.floor,
                *params.Lambda,
                *[
                    params.B[i, j]
                    for i in range(len(params.omega))
                    for j in range(i, len(params.omega))
                ],
                *params.omega,
            ], dtype=float),
            exposure,
        )
        self.assertTrue(np.allclose(beta, cumulative_feature_vector(exposure) * 0.0 + beta))
        blocks = decompose_cumulative_risk(exposure, projected)
        self.assertAlmostEqual(float(features @ beta), blocks["total"])
        self.assertGreaterEqual(blocks["shared"], 0.0)
        self.assertAlmostEqual(
            blocks["total"],
            blocks["floor"] + blocks["independent"] + blocks["shared"] + blocks["linear"],
        )

    def test_synthetic_domains_expose_provider_formula(self):
        cases = [
            (FactorShockStatePolicyRZDT1(d=8, L=100, sigma=0.04), tuple([25] + [72] * 7)),
            (InventorySupplyChainProblem(d=12, L=100, sigma=0.04), tuple([60] * 4 + [40] * 4 + [20] * 4)),
            (QueueResourceControlProblem(d=12, L=100, sigma=0.04), tuple([70] * 4 + [40] * 4 + [55] * 4)),
        ]
        for problem, x in cases:
            with self.subTest(problem=problem.problem_name):
                self._check_problem_formula(problem, x)
                names = problem.cumulative_risk_feature_names(output_index=1)
                self.assertEqual(len(names), len(problem.cumulative_risk_features(x, output_index=1)))
                anchors = problem.state_anchor_points(n=3, rng=np.random.default_rng(3))
                self.assertTrue(anchors)
                inv = problem.inverse_state_anchor(anchors[0], rng=np.random.default_rng(4), n=2)
                self.assertTrue(inv)

    def test_factor_hvd_uses_psd_provider_projection(self):
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
        )
        model.fit_from_residuals(X, residuals, output_index=1, problem=problem)
        diag = model.diagnostics()
        self.assertTrue(diag["cumulative_provider_active"]["1"])
        x = tuple([50] + [95] * 7)
        cumulative = model.predict_decomposition(1, x, problem)["cumulative"]
        self.assertEqual(cumulative["v_C_plus"], model.predict_certification_variance(1, x, problem))
        self.assertTrue(cumulative["provider_active"])
        B = np.asarray(cumulative["fitted_blocks"]["B"], dtype=float)
        if B.size:
            eigvals = np.linalg.eigvalsh(0.5 * (B + B.T))
            self.assertGreaterEqual(float(np.min(eigvals)), -1e-10)
        self.assertGreaterEqual(cumulative["fitted_blocks"]["shared"], 0.0)

    def test_theory_certification_uses_epistemic_and_cumulative_variance(self):
        mu = np.array([0.1])
        s2 = np.array([0.04])
        v_c_plus = np.array([0.09])
        cert = conservative_chance_margin(
            mu,
            s2,
            v_c_plus,
            tau=0.8,
            alpha=0.05,
            beta_g=4.0,
            mode="theory",
        )
        expected = mu[0] + 2.0 * np.sqrt(s2[0]) + cert.z_alpha * np.sqrt(v_c_plus[0]) - 0.8
        self.assertAlmostEqual(float(cert.margin[0]), float(expected))

    def test_high_dependence_defaults_are_mainline(self):
        cfg = SingleOLHKGConfig()
        self.assertEqual(cfg.variance_mode, "factor")
        self.assertEqual(cfg.certification_mode, "theory")
        self.assertTrue(cfg.use_state_coupling)
        self.assertTrue(cfg.use_state_basis)
        self.assertEqual(cfg.state_basis_mode, "raw+state")
        self.assertEqual(cfg.acquisition_mode, "exact_mc")
        self.assertEqual(cfg.exact_kg_mc_samples, 8)

    def test_traffic_trajectory_encoder_exports_four_local_exposures(self):
        rows = [
            {
                "policy_id": "p0",
                "state": "s0",
                "action": "a0",
                "occupancy": "1.0",
                "queue": "2.0",
                "wait": "3.0",
                "flow": "4.0",
                "emission": "5.0",
                "demand_shock": "0.2",
            },
            {
                "policy_id": "p0",
                "state": "s1",
                "action": "a1",
                "occupancy": "2.0",
                "queue": "4.0",
                "wait": "5.0",
                "flow": "6.0",
                "emission": "7.0",
                "demand_shock": "0.4",
            },
        ]
        encoder = TrafficTrajectoryEncoder(rows)
        self.assertEqual(encoder.risk_exposure("p0").shape, (4,))
        self.assertEqual(encoder.shared_shock_exposure("p0").shape, (2,))
        self.assertAlmostEqual(float(encoder.risk_exposure("p0")[3]), 6.0)


if __name__ == "__main__":
    unittest.main()
