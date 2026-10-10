"""Verify variance conservation, signed covariance changes and held-out isolation."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import logsumexp, roots_legendre
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.risk_family_reweighting import family_covariance, family_predictive_branches, grouped_family_validation
from performance.audit_meta_prior_risk_family_prequential import model_log_scores
from performance.target_risk_direction_mixture import joint_label_log_scores
from performance.target_risk_residual import predictive_group_score
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


class FamilyReweightingTests(unittest.TestCase):
    def setUp(self):
        self.psi = np.random.default_rng(20261016).uniform(-2, 2, (12, 8))
        basis = FixedFunctionBasis(self.psi, [0, 19, 67])
        self.priors = [SourceFunctionPrior(basis, np.r_[mean, np.zeros(3)],
            np.diag([.25, .03, .01, .02]), discrepancy, {})
            for mean, discrepancy in ((-2.3, .025), (-2.6, .06))]
        self.ids = np.array([0, 0, 2, 5, 5, 8])
        self.labels = np.array([-1.9, -2.1, -2.7, -2.2, -2.4, -3.1])
        self.noise = np.array([.17, .2, .21, .19, .18, .22])
        self.families = {"linear": [0], "cos_pi": [1], "sin_4pi": [2]}

    def test_reweighting_preserves_budget_and_can_reduce_source_directions(self):
        prior = self.priors[0]
        original = prior.covariance.copy()
        for rho in (0., .4, 1.):
            covariance = family_covariance(prior, [2], rho)
            self.assertAlmostEqual(np.trace(covariance[1:, 1:]), np.trace(original[1:, 1:]), places=12)
            self.assertEqual(covariance[0, 0], original[0, 0])
        self.assertLess(family_covariance(prior, [2], .4)[1, 1], original[1, 1])
        np.testing.assert_array_equal(prior.covariance, original)

    def test_all_branches_match_dense_raw_repeated_label_conditioning(self):
        points, qweights = roots_legendre(6)
        for candidate in (0, 3):
            weights, means, variances, summary = family_predictive_branches(
                self.priors, self.psi, self.ids, self.labels, self.noise, candidate, self.families, 6)
            logs, dense_means, dense_variances = [], [], []
            for prior in self.priors:
                design = np.column_stack([np.ones(len(self.ids)), prior.basis.transform(self.psi[self.ids])])
                feature = np.r_[1., prior.basis.transform(self.psi[candidate:candidate + 1])[0]]
                branches = [(prior.covariance, .25)] + [
                    (family_covariance(prior, indices, (x + 1.) / 2.), .25 * w / (2 * 3))
                    for indices in self.families.values() for x, w in zip(points, qweights)]
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
            self.assertAlmostEqual(summary["null_posterior"], float(np.exp(dense_weights[[0, 19]]).sum()), places=11)
            self.assertAlmostEqual(summary["null_posterior"] + sum(summary["family_posterior"].values()), 1., places=12)

    def test_held_candidate_labels_do_not_set_family_or_reallocation(self):
        old = [{"held_candidate": int(i), "training_candidates": np.unique(self.ids[self.ids != i]).tolist(),
                "old_joint_log_score": 0.} for i in np.unique(self.ids)]
        a = grouped_family_validation(self.priors, self.psi, self.ids, self.labels, self.noise, old, self.families, 6)
        labels = self.labels.copy()
        labels[self.ids == 0] += 3.
        b = grouped_family_validation(self.priors, self.psi, self.ids, labels, self.noise, old, self.families, 6)
        first = next(f for f in a["folds"] if f["held_candidate"] == 0)
        second = next(f for f in b["folds"] if f["held_candidate"] == 0)
        self.assertEqual(first["family_posterior"], second["family_posterior"])
        self.assertEqual(first["mean_rho_posterior"], second["mean_rho_posterior"])
        self.assertEqual(first["training_log_evidence"], second["training_log_evidence"])
        self.assertEqual(first["held_batches"], 2)
        self.assertNotIn(0, first["training_candidates"])
        self.assertNotEqual(first["family_joint_log_score"], second["family_joint_log_score"])

    def test_prequential_source_control_matches_unmodified_source_predictor(self):
        # A9's posterior mass on adaptation must not alter its source control.
        for candidate in (0, 3):
            weights, means, variances, _ = family_predictive_branches(
                self.priors, self.psi, self.ids, self.labels, self.noise, candidate, self.families, 6)
            label, noise = np.array([-2.8]), np.array([.19])
            scores = joint_label_log_scores(means, variances, label, noise)
            source = model_log_scores(weights, scores, 2, 3, 6)["source"]
            original = predictive_group_score(self.priors, self.psi, self.ids, self.labels,
                self.noise, 0., candidate, label, noise)
            self.assertAlmostEqual(source, original, places=11)


if __name__ == "__main__":
    unittest.main()
