import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import digamma, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from representation.joint_function_risk import JointFunctionRiskPosterior, local_joint_moments
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


def raw_batch_weights(draws, g_label, eta_label, reps):
    """Independent raw-Normal likelihood, using its residual sum of squares."""
    residual_sum = reps * (g_label - draws[:, 0]) ** 2
    if reps > 1:
        sample_variance = np.exp(eta_label + digamma((reps - 1) / 2) - np.log((reps - 1) / 2))
        residual_sum += (reps - 1) * sample_variance
    logs = -reps * draws[:, 1] / 2 - residual_sum / (2 * np.exp(draws[:, 1]))
    return np.exp(logs - logsumexp(logs))


class JointFunctionRiskTests(unittest.TestCase):
    def test_kernel_initialization_preserves_cross_candidate_filtering(self):
        psi = np.array([[-1.], [1.]])
        basis = FixedFunctionBasis(psi)
        priors = [SourceFunctionPrior(basis, np.r_[mean, np.zeros(basis.feature_dim)],
            np.diag(np.r_[.06, np.full(basis.feature_dim, .01)]), .02, {}) for mean in [-.5, -2.5]]
        original = JointFunctionRiskPosterior([priors[0]], [priors[1]], psi)
        mean, covariance = original.predict(full_cov=True)
        kernel = JointFunctionRiskPosterior.from_moments(mean, covariance)
        for i, value in enumerate([-.68, -.53]):
            a = original.update_batch(i, value, None, 1)
            b = kernel.update_batch(i, value, None, 1)
            self.assertAlmostEqual(a, b, places=12)
        for a, b in zip(original.predict(full_cov=True), kernel.predict(full_cov=True)):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(kernel.paid_episodes, 2)
        np.testing.assert_array_equal(mean, JointFunctionRiskPosterior([priors[0]], [priors[1]], psi).predict()[0])

    def test_single_episode_conditions_latent_risk_without_variance_label(self):
        mean, covariance = np.array([-.5, -3.]), np.array([[.025, -.04], [-.04, .4]])
        result = local_joint_moments(mean, covariance, -.9, None, 1)
        draws = np.random.default_rng(105108).multivariate_normal(mean, covariance, 300000)
        weights = raw_batch_weights(draws, -.9, None, 1)
        expected = weights @ draws
        centered = draws - expected
        np.testing.assert_allclose(result["mean"], expected, atol=.006)
        np.testing.assert_allclose(result["covariance"], (centered.T * weights) @ centered, atol=.004)
        self.assertGreater(abs(result["mean"][1] - mean[1]), .03)
        with self.assertRaises(ValueError):
            local_joint_moments(mean, covariance, -.9, -3., 1)

    def test_correlated_local_update_matches_raw_batch_posterior(self):
        mean, covariance = np.array([-.5, -2.5]), np.array([[.03, -.035], [-.035, .3]])
        result = local_joint_moments(mean, covariance, -.68, -3.3, 12)
        draws = np.random.default_rng(105104).multivariate_normal(mean, covariance, 300000)
        weights = raw_batch_weights(draws, -.68, -3.3, 12)
        expected_mean = weights @ draws
        centered = draws - expected_mean
        expected_covariance = (centered.T * weights) @ centered
        np.testing.assert_allclose(result["mean"], expected_mean, atol=.006)
        np.testing.assert_allclose(result["covariance"], expected_covariance, atol=.004)
        self.assertLess(result["integration_relative_error"], 1e-8)

    def test_small_sample_variance_does_not_become_known_mean_noise(self):
        result = local_joint_moments(np.array([-.56, -2.4]), np.diag([.01, .2]), -.715, -3.529, 12)
        sample_variance = np.exp(-3.529 + digamma(5.5) - np.log(5.5))
        plugin_variance = .01 * (sample_variance / 12) / (.01 + sample_variance / 12)
        self.assertGreater(result["covariance"][0, 0], 1.5 * plugin_variance)
        self.assertGreater(abs(result["covariance"][0, 1]), .001)

    def test_cross_candidate_update_matches_full_raw_batch_posterior(self):
        psi = np.array([[-1.], [1.]])
        basis = FixedFunctionBasis(psi)
        def prior(intercept, scale):
            mean = np.r_[intercept, np.zeros(basis.feature_dim)]
            covariance = np.diag(np.r_[.06, np.full(basis.feature_dim, scale)])
            return SourceFunctionPrior(basis, mean, covariance, .02, {})
        model = JointFunctionRiskPosterior([prior(-.5, .005)], [prior(-2.5, .015)], psi)
        before_mean, before_covariance = model.predict(full_cov=True)
        draws = np.random.default_rng(105105).multivariate_normal(
            before_mean.reshape(-1), before_covariance.reshape(4, 4), 300000)
        weights = raw_batch_weights(draws[:, :2], -.68, -3.3, 12)
        expected_mean = weights @ draws
        centered = draws - expected_mean
        expected_covariance = (centered.T * weights) @ centered
        model.update_batch(0, -.68, -3.3, 12)
        after_mean, after_covariance = model.predict(full_cov=True)
        np.testing.assert_allclose(after_mean.reshape(-1), expected_mean, atol=.006)
        np.testing.assert_allclose(after_covariance.reshape(4, 4), expected_covariance, atol=.004)
        self.assertGreater(abs(after_covariance[1, 0, 1, 1]), 1e-5)
        self.assertGreater(np.linalg.eigvalsh(after_covariance.reshape(4, 4)).min(), 0)
        self.assertEqual(model.paid_batches, 1)
        self.assertEqual(model.paid_episodes, 12)

    def test_joint_component_weights_match_raw_batch_posterior(self):
        psi = np.array([[-1.], [1.]])
        basis = FixedFunctionBasis(psi)
        def prior(intercept, structural):
            return SourceFunctionPrior(basis, np.r_[intercept, np.zeros(basis.feature_dim)],
                np.diag(np.r_[.06, np.full(basis.feature_dim, .005)]), .02,
                {"mode": "source_orthogonal_group_spectral" if structural else "same_source_orthogonal_full"})
        gp, ep = [prior(-.4, True), prior(-.65, False)], [prior(-2.6, True), prior(-1.8, False)]
        model = JointFunctionRiskPosterior(gp, ep, psi)
        rng = np.random.default_rng(105106)
        draws = np.concatenate([rng.multivariate_normal(mean, covariance, 100000)
            for _, mean, covariance in model.local_components(0)])
        weights = raw_batch_weights(draws, -.7, -3.3, 12)
        expected = weights.reshape(4, -1).sum(axis=1)
        model.update_batch(0, -.7, -3.3, 12)
        np.testing.assert_allclose(np.exp(model.log_weights), expected, atol=.005)
        diagnostics = model.diagnostics()
        self.assertAlmostEqual(diagnostics["g_structural_weight"], float(expected[:2].sum()), delta=.005)
        self.assertAlmostEqual(diagnostics["eta_structural_weight"], float(expected[[0, 2]].sum()), delta=.005)
        control = JointFunctionRiskPosterior([gp[1]], [ep[1]], psi)
        self.assertEqual(control.diagnostics()["g_structural_weight"], 0.)


if __name__ == "__main__":
    unittest.main()
