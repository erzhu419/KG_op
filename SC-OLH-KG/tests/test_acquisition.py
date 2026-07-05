import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acquisition.olhkg import OLHKGAcquisition  # noqa: E402
from core.gpr import ParametricGPR  # noqa: E402
from core.kg import compute_kg_vectorized  # noqa: E402
from problems.rzdt import RegimeRZDT1  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from variance.orthogonal_hvd import OrthogonalHVD  # noqa: E402


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        base = RegimeRZDT1(d=3, L=30, sigma=0.05)
        self.problem = ScalarizedProblem(base)
        self.obj = ParametricGPR(3, normalize_func=self.problem.normalize)
        self.con = ParametricGPR(3, normalize_func=self.problem.normalize)
        self.candidates = [(0, 0, 0), (10, 0, 0), (20, 0, 0), (30, 0, 0)]
        for model in (self.obj, self.con):
            beta = np.zeros(model.p)
            beta[1] = 0.1
            model.set_parametric_prior(beta, lambda_i=0.1, prior_var=1.0)
            for x in self.candidates[:2]:
                model.dimension_augment(x)
        self.hvd = OrthogonalHVD(mode="pooled", n_outputs=2)
        self.hvd.fit_from_residuals(self.candidates, [0.1, 0.1, 0.1, 0.1], 0, self.problem)
        self.hvd.fit_from_residuals(self.candidates, [0.2, 0.2, 0.2, 0.2], 1, self.problem)

    def test_zero_aux_lambdas_equal_objective_kg(self):
        acq = OLHKGAcquisition(lambda_feas=0.0, lambda_var=0.0, lambda_coupling=0.0)
        score = acq.score(self.candidates, self.obj, self.con, self.hvd, self.problem)
        sig2 = [self.hvd.predict_variance(0, x, self.problem) for x in self.candidates]
        kg = compute_kg_vectorized(self.obj, self.candidates, sig2)
        np.testing.assert_allclose(score["total"], kg, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(score["kg_obj_scaled"], kg, rtol=1e-12, atol=1e-12)

    def test_optional_torch_backend_matches_numpy_kg(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch is not installed")
        torch_obj = ParametricGPR(
            3,
            normalize_func=self.problem.normalize,
            numeric_backend="torch",
            numeric_backend_device="cpu",
            torch_min_rows=1,
        )
        torch_obj.a = self.obj.a.copy()
        torch_obj.C = self.obj.C.copy()
        torch_obj.lambda_i = self.obj.lambda_i
        torch_obj.sampled_set = list(self.obj.sampled_set)
        torch_obj.sol_to_idx = dict(self.obj.sol_to_idx)
        sig2 = [self.hvd.predict_variance(0, x, self.problem) for x in self.candidates]
        np.testing.assert_allclose(
            torch_obj.posterior_mean_many(self.candidates),
            self.obj.posterior_mean_many(self.candidates),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            torch_obj.posterior_var_many(self.candidates),
            self.obj.posterior_var_many(self.candidates),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            compute_kg_vectorized(torch_obj, self.candidates, sig2),
            compute_kg_vectorized(self.obj, self.candidates, sig2),
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertEqual(torch_obj.backend_status()["effective_backend"], "torch")

    def test_auxiliary_scores_use_scaled_objective_kg(self):
        acq = OLHKGAcquisition(lambda_feas=0.25, lambda_var=0.25, lambda_coupling=0.0)
        score = acq.score(self.candidates, self.obj, self.con, self.hvd, self.problem)
        sig2 = [self.hvd.predict_variance(0, x, self.problem) for x in self.candidates]
        kg = compute_kg_vectorized(self.obj, self.candidates, sig2)
        self.assertGreaterEqual(float(np.min(score["kg_obj_scaled"])), 0.0)
        self.assertLessEqual(float(np.max(score["kg_obj_scaled"])), 1.0)
        if float(np.max(kg) - np.min(kg)) > 1e-14:
            self.assertAlmostEqual(float(np.max(score["kg_obj_scaled"])), 1.0)
        expected = (
            score["kg_obj_scaled"]
            + 0.25 * score["kg_feas"]
            + 0.25 * score["kg_var"]
        )
        np.testing.assert_allclose(score["total"], expected, rtol=1e-12, atol=1e-12)

    def test_mean_score_enters_total_when_enabled(self):
        acq = OLHKGAcquisition(
            lambda_feas=0.25,
            lambda_var=0.25,
            lambda_mean=0.10,
            lambda_coupling=0.0,
        )
        score = acq.score(self.candidates, self.obj, self.con, self.hvd, self.problem)
        expected = (
            score["kg_obj_scaled"]
            + 0.25 * score["kg_feas"]
            + 0.25 * score["kg_var"]
            + 0.10 * score["kg_mean"]
        )
        np.testing.assert_allclose(score["total"], expected, rtol=1e-12, atol=1e-12)
        self.assertGreaterEqual(float(np.min(score["kg_mean"])), 0.0)
        self.assertLessEqual(float(np.max(score["kg_mean"])), 1.0)

    def test_chance_margin_formula(self):
        mu = 0.2
        variance = 0.04
        tau = 0.5
        alpha = 0.05
        expected = mu + norm.ppf(1 - alpha) * np.sqrt(variance) - tau
        got = OLHKGAcquisition.chance_margin(mu, variance, tau, alpha)
        self.assertAlmostEqual(got, expected, places=12)

    def test_coupling_gate_penalizes_risky_chance_margins(self):
        acq = OLHKGAcquisition(
            lambda_coupling=0.05,
            coupling_safety_z=0.5,
            coupling_gate_temperature=0.25,
        )
        details = [
            {"chance_margin": -0.2, "variance_g": 0.01},
            {"chance_margin": 0.0, "variance_g": 0.01},
            {"chance_margin": 0.2, "variance_g": 0.01},
        ]
        gate = acq.coupling_feasibility_gate(details)
        self.assertGreater(float(gate[0]), 0.99)
        self.assertLess(float(gate[1]), 0.2)
        self.assertLess(float(gate[2]), 1e-3)


if __name__ == "__main__":
    unittest.main()
