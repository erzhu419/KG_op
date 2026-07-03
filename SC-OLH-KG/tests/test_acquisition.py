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

    def test_chance_margin_formula(self):
        mu = 0.2
        variance = 0.04
        tau = 0.5
        alpha = 0.05
        expected = mu + norm.ppf(1 - alpha) * np.sqrt(variance) - tau
        got = OLHKGAcquisition.chance_margin(mu, variance, tau, alpha)
        self.assertAlmostEqual(got, expected, places=12)


if __name__ == "__main__":
    unittest.main()
