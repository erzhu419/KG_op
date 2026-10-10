"""Verify repeated-candidate covariance, zero correction, and held-out isolation."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.target_risk_residual import adjusted_priors, fit_delta, grouped_validation, posterior_components
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior
from representation.transfer_mixture import TransferMixtureGPR


class TargetRiskResidualTests(unittest.TestCase):
    def setUp(self):
        self.psi = np.random.default_rng(20261005).uniform(-2, 2, (12, 8))
        basis = FixedFunctionBasis(self.psi)
        self.priors = [SourceFunctionPrior(basis, np.r_[mean, np.zeros(68)],
            np.diag(np.r_[.25, np.full(68, scale)]), discrepancy, {})
            for mean, scale, discrepancy in ((-2.3, .002, .025), (-2.6, .004, .06))]
        self.ids = np.array([0, 0, 2, 5, 5, 8])
        self.labels = np.array([-1.9, -2.1, -2.7, -2.2, -2.4, -3.1])
        self.noise = np.full(6, .199342)

    def test_zero_delta_batch_posterior_matches_sequential_gpr(self):
        model = TransferMixtureGPR(self.priors, self.psi, self.psi)
        for index, label, noise in zip(self.ids, self.labels, self.noise):
            model.update([index], label, noise)
        for index in (0, 3, 5):
            weights, components = posterior_components(self.priors, self.psi, self.ids,
                self.labels, self.noise, 0., index)
            weights = np.exp(weights)
            mean = sum(w * m for w, (m, v) in zip(weights, components))
            variance = sum(w * (v + (m - mean) ** 2) for w, (m, v) in zip(weights, components))
            self.assertAlmostEqual(mean, model.posterior_mean_many([[index]])[0], places=11)
            self.assertAlmostEqual(variance, model.posterior_var_many([[index]])[0], places=11)
            np.testing.assert_allclose(weights, np.exp(model.log_weights), atol=1e-12)

    def test_group_refit_does_not_use_held_candidate_labels(self):
        first = grouped_validation(self.priors, self.psi, self.ids, self.labels, self.noise)
        changed = self.labels.copy()
        changed[self.ids == 0] += 3.
        second = grouped_validation(self.priors, self.psi, self.ids, changed, self.noise)
        a = next(f for f in first["folds"] if f["held_candidate"] == 0)
        b = next(f for f in second["folds"] if f["held_candidate"] == 0)
        self.assertEqual(a["delta"], b["delta"])
        self.assertNotIn(0, a["training_candidates"])
        self.assertEqual(a["held_batches"], 2)
        self.assertNotEqual(a["residual_joint_log_score"], b["residual_joint_log_score"])

    def test_fitted_residual_includes_zero_and_preserves_source_prior(self):
        zero_labels = np.full(len(self.ids), self.priors[0].mean[0])
        delta, _ = fit_delta(self.priors[:1], self.psi, self.ids, zero_labels, self.noise)
        self.assertEqual(delta, 0.)
        covariance = self.priors[0].covariance.copy()
        corrected = adjusted_priors(self.priors, 1.5)
        np.testing.assert_array_equal(self.priors[0].covariance, covariance)
        np.testing.assert_array_equal(corrected[0].mean, self.priors[0].mean)
        self.assertEqual(corrected[0].covariance[0, 0], covariance[0, 0])
        np.testing.assert_allclose(np.diag(corrected[0].covariance - covariance)[1:], 1.5 / 68)


if __name__ == "__main__":
    unittest.main()
