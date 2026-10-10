import unittest

import numpy as np
from scipy.special import logsumexp, roots_legendre
from scipy.stats import multivariate_normal

from performance.risk_family_reweighting import family_predictive_branches
from performance.scale_preserving_source_covariance import (
    scale_preserving_covariance, selected_scale_forecasts,
)
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


class ScalePreservingSourceCovarianceTests(unittest.TestCase):
    def setUp(self):
        self.psi = np.random.default_rng(20).uniform(-2., 2., (24, 8))
        basis = FixedFunctionBasis(self.psi, [0, 19, 67])
        self.priors = [SourceFunctionPrior(basis, np.r_[mean, np.zeros(3)],
            np.diag([.25, .8, .1, .2]), discrepancy, {})
            for mean, discrepancy in ((-2.3, .02), (-2.6, .06))]

    def test_budget_ratios_and_remaining_directions(self):
        prior = self.priors[0]
        original = prior.covariance.copy()
        for rho in (0., .4, 1.):
            changed = scale_preserving_covariance(prior, [0, 1], rho)
            self.assertAlmostEqual(np.trace(changed[1:, 1:]), 1.1, places=13)
            self.assertEqual(changed[0, 0], .25)
            self.assertAlmostEqual(changed[1, 1] / changed[2, 2], 8., places=13)
            self.assertGreater(changed[3, 3], 0.)
        np.testing.assert_array_equal(scale_preserving_covariance(prior, [0, 1], 0.), original)
        np.testing.assert_array_equal(prior.covariance, original)
        endpoint = np.diag(scale_preserving_covariance(prior, [0, 1], 1.))[1:] / 1.1
        mass = .9 / 1.1
        self.assertAlmostEqual(.5 * np.abs(endpoint - np.diag(original)[1:] / 1.1).sum(),
                               mass * (1. - mass) / (1. + mass), places=13)

    def test_signed_kernel_matches_dense_conditioning_with_repeated_visits(self):
        ids = np.array([0, 0, 2, 5, 5, 8])
        labels = np.array([-1.9, -2.1, -2.7, -2.2, -2.4, -3.1])
        noise = np.array([.17, .2, .21, .19, .18, .22])
        points, qweights = roots_legendre(6)
        for candidate in (0, 3):
            weights, means, variances, _ = family_predictive_branches(
                self.priors, self.psi, ids, labels, noise, candidate, {"selected": [0, 1]}, 6,
                covariance_builder=scale_preserving_covariance)
            logs, dense_means, dense_variances = [], [], []
            for prior in self.priors:
                design = np.column_stack([np.ones(len(ids)), prior.basis.transform(self.psi[ids])])
                feature = np.r_[1., prior.basis.transform(self.psi[candidate:candidate + 1])[0]]
                branches = [(prior.covariance, .25)] + [
                    (scale_preserving_covariance(prior, [0, 1], (x + 1.) / 2.), .25 * w / 2.)
                    for x, w in zip(points, qweights)]
                for covariance, mass in branches:
                    raw = design @ covariance @ design.T + np.diag(noise)
                    raw += prior.discrepancy_variance * (ids[:, None] == ids[None, :])
                    cross = design @ covariance @ feature + prior.discrepancy_variance * (ids == candidate)
                    residual = labels - design @ prior.mean
                    logs.append(np.log(mass) + multivariate_normal.logpdf(residual, cov=raw))
                    dense_means.append(feature @ prior.mean + cross @ np.linalg.solve(raw, residual))
                    dense_variances.append(feature @ covariance @ feature + prior.discrepancy_variance -
                                           cross @ np.linalg.solve(raw, cross))
            np.testing.assert_allclose(weights, np.asarray(logs) - logsumexp(logs), atol=1e-11)
            np.testing.assert_allclose(means, dense_means, atol=1e-11)
            np.testing.assert_allclose(variances, dense_variances, atol=1e-11)

    def test_current_held_label_does_not_choose_branch_odds(self):
        task = {"psi": self.psi, "response": np.zeros((24, 3)), "noise": np.full((24, 3), .2)}
        original = selected_scale_forecasts(self.priors, task, [0, 1], 6)
        task["response"][8, 2] = 4.
        changed = selected_scale_forecasts(self.priors, task, [0, 1], 6)
        self.assertEqual(original[0]["relative_log_evidence"], changed[0]["relative_log_evidence"])
        self.assertNotEqual(original[0]["adapted"], changed[0]["adapted"])
        self.assertNotEqual(original[1]["relative_log_evidence"], changed[1]["relative_log_evidence"])


if __name__ == "__main__":
    unittest.main()
