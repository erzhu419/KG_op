"""Check rank-one arithmetic against dense likelihoods and candidate isolation."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.target_risk_direction import (direction_evidence, direction_priors, fit_direction,
                                              grouped_direction_validation, rank_one_statistics)
from performance.target_risk_residual import grouped_validation, training_matrices
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


class TargetRiskDirectionTests(unittest.TestCase):
    def setUp(self):
        self.psi = np.random.default_rng(20261008).uniform(-2, 2, (12, 8))
        basis = FixedFunctionBasis(self.psi)
        self.priors = [SourceFunctionPrior(basis, np.r_[mean, np.zeros(68)],
            np.diag(np.r_[.25, np.full(68, scale)]), discrepancy, {})
            for mean, scale, discrepancy in ((-2.3, .002, .025), (-2.6, .004, .06))]
        self.ids = np.array([0, 0, 2, 5, 5, 8])
        self.labels = np.array([-1.9, -2.1, -2.7, -2.2, -2.4, -3.1])
        self.noise = np.full(6, .199342)

    def test_rank_one_evidence_matches_dense_component_mixture(self):
        statistics = rank_one_statistics(self.priors, self.psi, self.ids, self.labels, self.noise)
        for direction in (0, 19, 67):
            for delta in (0., .05, 1.3):
                logs = []
                for prior in direction_priors(self.priors, direction, delta):
                    _, residual, _, covariance, _ = training_matrices(
                        prior, self.psi, self.ids, self.labels, self.noise)
                    logs.append(multivariate_normal.logpdf(residual, cov=covariance))
                dense = logsumexp(logs) - np.log(len(logs))
                self.assertAlmostEqual(direction_evidence(statistics, direction, delta), dense, places=11)
        old = self.priors[0].covariance.copy()
        corrected = direction_priors(self.priors, 19, .05)[0]
        expected = np.zeros_like(old)
        expected[20, 20] = .05
        np.testing.assert_allclose(corrected.covariance - old, expected)
        np.testing.assert_array_equal(self.priors[0].covariance, old)
        np.testing.assert_array_equal(corrected.mean, self.priors[0].mean)

    def test_one_component_fit_matches_independent_dense_direction_grid(self):
        fitted = fit_direction(self.priors[:1], self.psi, self.ids, self.labels, self.noise)
        # Independently derive each optimum from dense inverse, including zero.
        prior = self.priors[0]
        _, residual, design, covariance, _ = training_matrices(
            prior, self.psi, self.ids, self.labels, self.noise)
        inverse = np.linalg.inv(covariance)
        candidates = [(multivariate_normal.logpdf(residual, cov=covariance), None, 0.)]
        for j, feature in enumerate(design[:, 1:].T):
            a, b = feature @ inverse @ feature, (feature @ inverse @ residual) ** 2
            delta = max((b - a) / a ** 2, 0.)
            candidates.append((multivariate_normal.logpdf(residual,
                cov=covariance + delta * np.outer(feature, feature)), j, delta))
        score, direction, delta = max(candidates, key=lambda row: row[0])
        self.assertAlmostEqual(fitted["log_evidence"], score, places=11)
        self.assertEqual(fitted["direction"], direction)
        self.assertAlmostEqual(fitted["delta"], delta, places=11)

    def test_held_candidate_cannot_choose_direction_or_scale(self):
        old = grouped_validation(self.priors, self.psi, self.ids, self.labels, self.noise)
        first = grouped_direction_validation(self.priors, self.psi, self.ids, self.labels,
                                             self.noise, old["folds"])
        changed = self.labels.copy()
        changed[self.ids == 0] += 3.
        second = grouped_direction_validation(self.priors, self.psi, self.ids, changed,
                                              self.noise, old["folds"])
        a = next(f for f in first["folds"] if f["held_candidate"] == 0)
        b = next(f for f in second["folds"] if f["held_candidate"] == 0)
        self.assertEqual((a["direction"], a["delta"]), (b["direction"], b["delta"]))
        self.assertNotIn(0, a["training_candidates"])
        self.assertEqual(a["held_batches"], 2)
        self.assertNotEqual(a["direction_joint_log_score"], b["direction_joint_log_score"])

    def test_mixture_fit_beats_independent_dense_coarse_search(self):
        fitted = fit_direction(self.priors, self.psi, self.ids, self.labels, self.noise)
        dense_best = -np.inf
        for j in range(68):
            for delta in (0., .01, .1, 1., 10.):
                logs = []
                for prior in direction_priors(self.priors, j, delta):
                    _, residual, _, covariance, _ = training_matrices(
                        prior, self.psi, self.ids, self.labels, self.noise)
                    logs.append(multivariate_normal.logpdf(residual, cov=covariance))
                dense_best = max(dense_best, float(logsumexp(logs) - np.log(2)))
        self.assertGreaterEqual(fitted["log_evidence"] + 1e-9, dense_best)


if __name__ == "__main__":
    unittest.main()
