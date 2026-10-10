"""Check branch averaging against the raw repeated-label Gaussian model."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import logsumexp, roots_laguerre
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.target_risk_direction_mixture import joint_label_log_scores, predictive_branches
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


class DirectionMixtureTests(unittest.TestCase):
    def setUp(self):
        self.psi = np.random.default_rng(20261011).uniform(-2, 2, (12, 8))
        basis = FixedFunctionBasis(self.psi, [0, 19, 67])
        self.priors = [SourceFunctionPrior(basis, np.r_[mean, np.zeros(3)],
            np.diag([.25, .03, .01, .02]), discrepancy, {})
            for mean, discrepancy in ((-2.3, .025), (-2.6, .06))]
        self.ids = np.array([0, 0, 2, 5, 5, 8])
        self.labels = np.array([-1.9, -2.1, -2.7, -2.2, -2.4, -3.1])
        self.noise = np.array([.17, .2, .21, .19, .18, .22])

    def test_complete_branches_match_raw_repeated_label_conditioning(self):
        locations, quadrature_weights = roots_laguerre(6)
        for candidate in (0, 3):
            weights, means, variances, _, null = predictive_branches(
                self.priors, self.psi, self.ids, self.labels, self.noise, candidate, .1, 6)
            dense_logs, dense_means, dense_variances = [], [], []
            for prior in self.priors:
                design = np.column_stack([np.ones(len(self.ids)), prior.basis.transform(self.psi[self.ids])])
                feature = np.r_[1., prior.basis.transform(self.psi[candidate:candidate + 1])[0]]
                branches = [(None, 0., .5 / 2)] + [(j, .1 * x, .5 * w / (2 * 3))
                    for j in range(3) for x, w in zip(locations, quadrature_weights)]
                for j, delta, mass in branches:
                    covariance = prior.covariance.copy()
                    if j is not None:
                        covariance[j + 1, j + 1] += delta
                    raw = design @ covariance @ design.T + np.diag(self.noise)
                    raw += prior.discrepancy_variance * (self.ids[:, None] == self.ids[None, :])
                    cross = design @ covariance @ feature + prior.discrepancy_variance * (self.ids == candidate)
                    residual = self.labels - design @ prior.mean
                    dense_logs.append(np.log(mass) + multivariate_normal.logpdf(residual, cov=raw))
                    dense_means.append(feature @ prior.mean + cross @ np.linalg.solve(raw, residual))
                    dense_variances.append(feature @ covariance @ feature + prior.discrepancy_variance -
                                           cross @ np.linalg.solve(raw, cross))
            dense_weights = np.asarray(dense_logs) - logsumexp(dense_logs)
            np.testing.assert_allclose(weights, dense_weights, atol=1e-11)
            np.testing.assert_allclose(means, dense_means, atol=1e-11)
            np.testing.assert_allclose(variances, dense_variances, atol=1e-11)
            self.assertAlmostEqual(null, float(np.exp(dense_weights[[0, 19]]).sum()), places=11)

    def test_repeated_held_label_density_matches_dense_gaussian(self):
        means, variances = np.array([-2.3, -1.8, -2.5]), np.array([0., .15, 1.2])
        labels, noise = self.labels[:2], self.noise[:2]
        scores = joint_label_log_scores(means, variances, labels, noise)
        expected = [multivariate_normal.logpdf(labels, mean=np.full(2, m),
            cov=np.full((2, 2), v) + np.diag(noise)) for m, v in zip(means, variances)]
        np.testing.assert_allclose(scores, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
