"""Verify endpoint optimization, grouped history and the issued predictor."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.risk_adaptation_calibration import (fit_predictive_weight, fit_calibration,
    calibrated_predictive_branches, prequential_forecasts)
from performance.target_risk_residual import predictive_group_score
from performance.target_risk_direction_mixture import joint_label_log_scores
from performance.audit_meta_prior_risk_family_prequential import model_log_scores
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.psi = np.random.default_rng(20261019).uniform(-2., 2., (14, 8))
        basis = FixedFunctionBasis(self.psi, [0, 19, 67])
        self.priors = [SourceFunctionPrior(basis, np.r_[m, np.zeros(3)],
            np.diag([.25, .03, .01, .02]), z, {}) for m, z in ((-2.3, .025), (-2.6, .06))]
        self.ids = np.array([0, 0, 2, 5, 5, 8, 9, 10, 0, 3])
        self.labels = np.array([-1.9, -2.1, -2.7, -2.2, -2.4, -3.1, -2.3, -2.9, -2.2, -2.6])
        self.noise = np.full(10, .2)
        self.families = {"linear": [0], "cos_pi": [1], "sin_4pi": [2]}

    def test_concave_optimum_and_endpoint_solutions(self):
        source, adapted = np.log([.8, .2]), np.log([.2, .8])
        pi, score = fit_predictive_weight(source, adapted)
        self.assertAlmostEqual(pi, .5, places=11)
        self.assertAlmostEqual(score, 2 * np.log(.5), places=11)
        shifted, _ = fit_predictive_weight(source + [2., -5.], adapted + [2., -5.])
        self.assertAlmostEqual(shifted, pi, places=11)
        self.assertEqual(fit_predictive_weight(np.log([.8, .7]), np.log([.2, .3]))[0], 0.)
        self.assertEqual(fit_predictive_weight(np.log([.2, .3]), np.log([.8, .7]))[0], 1.)
        self.assertEqual(fit_predictive_weight(source, source)[0], 0.)

    def test_inner_groups_exclude_repeats_and_match_source_density(self):
        ids, labels, noise = self.ids[:8], self.labels[:8], self.noise[:8]
        fit = fit_calibration(self.priors, self.psi, ids, labels, noise, self.families, 6)
        for fold in fit["folds"]:
            candidate = fold["held_candidate"]
            train = ids != candidate
            self.assertNotIn(candidate, fold["training_candidates"])
            self.assertEqual(fold["held_batches"], int(np.sum(~train)))
            reference = predictive_group_score(self.priors, self.psi, ids[train], labels[train], noise[train],
                0., candidate, labels[~train], noise[~train])
            self.assertAlmostEqual(fold["source_log_score"], reference, places=11)
        self.assertEqual(sum(f["held_batches"] for f in fit["folds"]), 8)

    def test_issued_mixture_matches_two_predictive_densities(self):
        original_covariance = [p.covariance.copy() for p in self.priors]
        weights, means, variances, original, fit = calibrated_predictive_branches(self.priors, self.psi,
            self.ids[:8], self.labels[:8], self.noise[:8], 0, self.families, 6)
        self.assertAlmostEqual(float(logsumexp(weights)), 0., places=12)
        scores = joint_label_log_scores(means, variances, self.labels[8:9], self.noise[8:9])
        models = model_log_scores(original, scores, 2, 3, 6)
        pi = fit["pi"]
        reference = np.log((1. - pi) * np.exp(models["source"]) + pi * np.exp(models["adapted"]))
        self.assertAlmostEqual(float(logsumexp(weights + scores)), reference, places=11)
        for prior, covariance in zip(self.priors, original_covariance):
            np.testing.assert_array_equal(prior.covariance, covariance)

    def test_next_and_future_labels_do_not_change_issued_predictor(self):
        a = prequential_forecasts(self.priors, self.psi, self.ids, self.labels, self.noise,
                                 self.families, 6, [9])
        changed = self.labels.copy()
        changed[8:] += 3.
        b = prequential_forecasts(self.priors, self.psi, self.ids, changed, self.noise,
                                 self.families, 6, [9])
        for key in ("pi", "inner_log_score", "issued_eta_mean", "issued_eta_variance", "a9_null_posterior"):
            self.assertEqual(a[0][key], b[0][key])
        self.assertNotEqual(a[0]["calibrated_log_score"], b[0]["calibrated_log_score"])


if __name__ == "__main__":
    unittest.main()
