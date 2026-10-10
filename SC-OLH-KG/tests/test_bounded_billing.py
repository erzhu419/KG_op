import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import digamma, logsumexp
from scipy.stats import norm, truncnorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from representation.bounded_billing import bounded_joint, bounded_objective, conditional_eta_update, gaussian_field
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


class BoundedBillingTests(unittest.TestCase):
    def test_conditional_risk_integral_matches_raw_normal_batch(self):
        mu, var, g, y, label, n = -2.6, .7, -.5, -.68, -3.3, 12
        result = conditional_eta_update(np.array([mu]), np.array([var]), np.array([g]), y, label, n)
        k = (n - 1) / 2
        s2 = np.exp(label + digamma(k) - np.log(k))
        residual = (n - 1) * s2 + n * (y - g) ** 2
        def density(eta):
            return norm.pdf(eta, mu, np.sqrt(var)) * np.exp(-n * eta / 2 - residual * np.exp(-eta) / 2 - 12)
        mass = quad(density, mu - 12 * np.sqrt(var), mu + 12 * np.sqrt(var), epsabs=1e-11)[0]
        mean = quad(lambda eta: density(eta) * eta, mu - 12 * np.sqrt(var), mu + 12 * np.sqrt(var))[0] / mass
        variance = quad(lambda eta: density(eta) * (eta - mean) ** 2, mu - 12 * np.sqrt(var), mu + 12 * np.sqrt(var))[0] / mass
        np.testing.assert_allclose([result[0][0], result[1][0]], [mean, variance], atol=2e-7)

    def test_bounded_objective_matches_independent_truncated_gaussians(self):
        center, prior_var, noise = .65, .2, .01
        labels = np.array([.05, .58, .75, 1.2])
        observations = [{"index": i, "response": [float(y)], "noise": [noise]} for i, y in enumerate(labels)]
        result = bounded_objective(np.eye(4), np.zeros(4), observations, np.eye(4) * prior_var, (.4, .9), 9)
        variance = 1 / (1 / prior_var + 1 / noise)
        mean = variance * (center / prior_var + labels / noise)
        sd = np.sqrt(variance)
        a, b = (.4 - mean) / sd, (.9 - mean) / sd
        expected_mean, expected_var = truncnorm.stats(a, b, loc=mean, scale=sd, moments="mv")
        np.testing.assert_allclose(result["price_mean"], expected_mean, atol=7e-4)
        np.testing.assert_allclose(np.diag(result["price_covariance"]), expected_var, atol=2e-4)
        self.assertTrue(np.all((result["price_mean"] > .4) & (result["price_mean"] < .9)))
        expected_evidence = np.sum(norm.logpdf(labels, center, np.sqrt(prior_var + noise))
            + np.log(norm.cdf(b) - norm.cdf(a))
            - np.log(norm.cdf((.9 - center) / np.sqrt(prior_var)) - norm.cdf((.4 - center) / np.sqrt(prior_var))))
        self.assertAlmostEqual(result["log_evidence"], expected_evidence, places=7)

    def test_joint_cross_candidate_moments_match_direct_eta_quadrature(self):
        psi = np.array([[-1.], [1.]])
        basis = FixedFunctionBasis(psi)
        prior = SourceFunctionPrior(basis, np.r_[-2.5, np.zeros(basis.feature_dim)],
            np.diag(np.r_[.1, np.full(basis.feature_dim, .005)]), .02, {})
        design, offsets = np.array([[1.], [.6]]), np.array([-1., -1.])
        observation = {"index": 0, "response": [0., -.42, -2.9], "noise": [0., .004, .2], "replicates": 12}
        result = bounded_joint(design, offsets, [observation], None, (.4, .9), [prior], psi, order=25)
        mu, cov = gaussian_field(prior, psi)
        # Independent two-dimensional integration of one price and one eta.
        from numpy.polynomial.legendre import leggauss
        p, pw = leggauss(80)
        price, pw = .65 + .25 * p, pw / 2
        e, ew = leggauss(100)
        eta, ew = mu[0] + 8 * np.sqrt(cov[0, 0]) * e, ew * 8 * np.sqrt(cov[0, 0])
        k = 5.5
        s2 = np.exp(-2.9 + digamma(k) - np.log(k))
        from representation.physical_risk_prior import log_sample_variance_density
        logs = (np.log(pw[:, None] * ew) + norm.logpdf(eta, mu[0], np.sqrt(cov[0, 0]))
            + norm.logpdf(-.42, price[:, None] - 1, np.exp(.5 * eta) / np.sqrt(12))
            + log_sample_variance_density(eta, s2, 12))
        evidence = logsumexp(logs)
        weights = np.exp(logs - evidence)
        price_mean, eta_mean = np.sum(weights * price[:, None]), np.sum(weights * eta)
        cross = np.sum(weights * (price[:, None] - price_mean) * (eta - eta_mean))
        np.testing.assert_allclose(result["mean"][0], [price_mean - 1, eta_mean], atol=4e-4)
        self.assertAlmostEqual(result["log_evidence"], evidence, delta=1e-4)
        self.assertAlmostEqual(result["covariance"][0, 0, 1, 1], cross * cov[1, 0] / cov[0, 0], delta=2e-5)


if __name__ == "__main__":
    unittest.main()
