"""Verify rank-two branches against raw repeated-label conditioning."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import logsumexp, roots_laguerre
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.target_risk_joint import grouped_joint_validation, joint_predictive_branches
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


class JointRiskTests(unittest.TestCase):
    def setUp(self):
        self.psi = np.random.default_rng(20261013).uniform(-2, 2, (12, 8))
        basis = FixedFunctionBasis(self.psi, [0, 19, 67])
        self.priors = [SourceFunctionPrior(basis, np.r_[mean, np.zeros(3)],
            np.diag([.25, .03, .01, .02]), discrepancy, {})
            for mean, discrepancy in ((-2.3, .025), (-2.6, .06))]
        self.ids = np.array([0, 0, 2, 5, 5, 8])
        self.labels = np.array([-1.9, -2.1, -2.7, -2.2, -2.4, -3.1])
        self.noise = np.array([.17, .2, .21, .19, .18, .22])

    def test_every_branch_matches_dense_raw_gaussian_model(self):
        locations, quadrature_weights = roots_laguerre(6)
        originals = [p.covariance.copy() for p in self.priors]
        for candidate in (0, 3):
            weights, means, variances, _, rank_mass = joint_predictive_branches(
                self.priors, self.psi, self.ids, self.labels, self.noise, candidate, .1, 6)
            dense_logs, dense_means, dense_variances = [], [], []
            for prior in self.priors:
                design = np.column_stack([np.ones(len(self.ids)), prior.basis.transform(self.psi[self.ids])])
                feature = np.r_[1., prior.basis.transform(self.psi[candidate:candidate + 1])[0]]
                branches = [((), 0., .5 / 2)]
                branches += [((j,), .1 * x, .25 * w / (2 * 3))
                    for j in range(3) for x, w in zip(locations, quadrature_weights)]
                branches += [((j, k), .1 * x, .25 * w / (2 * 3))
                    for j, k in zip(*np.triu_indices(3, 1)) for x, w in zip(locations, quadrature_weights)]
                for indices, delta, mass in branches:
                    covariance = prior.covariance.copy()
                    for j in indices:
                        covariance[j + 1, j + 1] += delta / len(indices)
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
            dense_rank = np.zeros(3)
            for start in (0, 37):
                dense_rank += [np.exp(dense_weights[start]), np.exp(dense_weights[start + 1:start + 19]).sum(),
                               np.exp(dense_weights[start + 19:start + 37]).sum()]
            np.testing.assert_allclose(rank_mass, dense_rank, atol=1e-11)
        for prior, original in zip(self.priors, originals):
            np.testing.assert_array_equal(prior.covariance, original)

    def test_all_held_candidate_repeats_are_excluded_from_inference(self):
        old = [{"held_candidate": int(i), "training_candidates": np.unique(self.ids[self.ids != i]).tolist(),
                "old_joint_log_score": 0.} for i in np.unique(self.ids)]
        first = grouped_joint_validation(self.priors, self.psi, self.ids, self.labels, self.noise, old, .1, 6)
        labels = self.labels.copy()
        labels[self.ids == 0] += 3.
        second = grouped_joint_validation(self.priors, self.psi, self.ids, labels, self.noise, old, .1, 6)
        a = next(f for f in first["folds"] if f["held_candidate"] == 0)
        b = next(f for f in second["folds"] if f["held_candidate"] == 0)
        self.assertEqual(a["rank_posterior"], b["rank_posterior"])
        self.assertEqual(a["training_log_evidence"], b["training_log_evidence"])
        self.assertNotIn(0, a["training_candidates"])
        self.assertEqual(a["held_batches"], 2)
        self.assertNotEqual(a["joint_model_log_score"], b["joint_model_log_score"])


if __name__ == "__main__":
    unittest.main()
