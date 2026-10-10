"""Check equal variance budgets and raw likelihoods in the oracle contrast."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import logsumexp, roots_laguerre
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.risk_reallocation_diagnostic import oracle_branches, support_base
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


class ReallocationDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.psi = np.random.default_rng(20261015).uniform(-2, 2, (12, 8))
        basis = FixedFunctionBasis(self.psi, [0, 19, 67])
        self.priors = [SourceFunctionPrior(basis, np.r_[mean, np.zeros(3)],
            np.diag([.25, .03, .01, .02]), discrepancy, {})
            for mean, discrepancy in ((-2.3, .025), (-2.6, .06))]
        self.ids = np.array([0, 0, 2, 5, 5, 8])
        self.labels = np.array([-1.9, -2.1, -2.7, -2.2, -2.4, -3.1])
        self.noise = np.array([.17, .2, .21, .19, .18, .22])
        self.support = [0, 2]

    def test_oracle_variants_have_equal_covariance_budget(self):
        prior = self.priors[0]
        original = prior.covariance.copy()
        reallocated = support_base(prior, self.support, True)
        np.testing.assert_array_equal(prior.covariance, original)
        np.testing.assert_array_equal(reallocated.mean, prior.mean)
        self.assertEqual(reallocated.covariance[0, 0], prior.covariance[0, 0])
        self.assertEqual(reallocated.discrepancy_variance, prior.discrepancy_variance)
        self.assertAlmostEqual(np.trace(reallocated.covariance[1:, 1:]), np.trace(original[1:, 1:]), places=12)
        for delta in (.01, .2):
            budgets = []
            for mode in (False, True):
                covariance = support_base(prior, self.support, mode).covariance.copy()
                covariance[np.asarray(self.support) + 1, np.asarray(self.support) + 1] += delta / 2
                budgets.append(np.trace(covariance[1:, 1:]))
            self.assertAlmostEqual(budgets[0], budgets[1], places=12)

    def test_both_complete_mixtures_match_dense_raw_label_conditioning(self):
        locations, qweights = roots_laguerre(6)
        for candidate in (0, 3):
            for mode in (False, True):
                weights, means, variances, null = oracle_branches(self.priors, self.psi, self.ids,
                    self.labels, self.noise, candidate, self.support, .1, mode, 6)
                logs, dense_means, dense_variances = [], [], []
                for prior in self.priors:
                    design = np.column_stack([np.ones(len(self.ids)), prior.basis.transform(self.psi[self.ids])])
                    feature = np.r_[1., prior.basis.transform(self.psi[candidate:candidate + 1])[0]]
                    branches = [(prior.covariance, .25)]
                    for x, w in zip(locations, qweights):
                        covariance = support_base(prior, self.support, mode).covariance.copy()
                        covariance[np.asarray(self.support) + 1, np.asarray(self.support) + 1] += .1 * x / 2
                        branches.append((covariance, .25 * w))
                    for covariance, mass in branches:
                        raw = design @ covariance @ design.T + np.diag(self.noise)
                        raw += prior.discrepancy_variance * (self.ids[:, None] == self.ids[None, :])
                        cross = design @ covariance @ feature + prior.discrepancy_variance * (self.ids == candidate)
                        residual = self.labels - design @ prior.mean
                        logs.append(np.log(mass) + multivariate_normal.logpdf(residual, cov=raw))
                        dense_means.append(feature @ prior.mean + cross @ np.linalg.solve(raw, residual))
                        dense_variances.append(feature @ covariance @ feature + prior.discrepancy_variance -
                                               cross @ np.linalg.solve(raw, cross))
                dense_weights = np.asarray(logs) - logsumexp(logs)
                np.testing.assert_allclose(weights, dense_weights, atol=1e-11)
                np.testing.assert_allclose(means, dense_means, atol=1e-11)
                np.testing.assert_allclose(variances, dense_variances, atol=1e-11)
                self.assertAlmostEqual(null, float(np.exp(dense_weights[[0, 7]]).sum()), places=11)


if __name__ == "__main__":
    unittest.main()
