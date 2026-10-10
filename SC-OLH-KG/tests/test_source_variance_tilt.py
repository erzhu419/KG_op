import unittest

import numpy as np
from numpy.polynomial.hermite import hermgauss

from performance.resolve_historical_source_variance import (
    VarianceTiltedPosterior, weighted_moments, weighted_function_moments,
    original_posterior_summary,
)
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.integrate_historical_joint_source_variance import integrate_risk_lines


class SourceVarianceTiltTests(unittest.TestCase):
    def test_uniform_retained_weights_have_finite_precision_diagnostics(self):
        class Model:
            mean = np.zeros(2)

        eta = np.random.default_rng(73).normal(size=(4, 256, 2))
        ds = .3 * eta; cs = np.tile(np.eye(2) * .2, (4, 256, 1, 1))
        spec = {'monte_carlo_gate': {
            'maximum_split_rhat_eta_and_rao_blackwell_f_moments': 1.05,
            'minimum_autocorrelation_ess_eta_and_rao_blackwell_f_moments': 100,
            'maximum_first_second_half_f_variance_relative_difference': .15}}
        with np.errstate(invalid='raise', divide='raise'):
            result = original_posterior_summary(Model(), eta, ds, cs, np.ones((4, 256)), spec)
        self.assertTrue(np.isfinite(result['maximum_split_rhat']))
        self.assertTrue(np.isfinite(result['minimum_autocorrelation_ess']))
        self.assertTrue(result['monte_carlo_gate_passed'])
        expected = .2 * np.eye(2) + np.cov(ds.reshape(-1, 2), rowvar=False, ddof=0)
        np.testing.assert_allclose(result['posterior_covariance'], expected, atol=1e-14)

    def test_joint_integral_corrects_tilted_residual_marginals(self):
        class GaussianTilt:
            mean = np.zeros(2)

            def evaluate(self, point):
                return np.log1p(point[0]**2), point, None

            def log_tilt(self, evaluation):
                return evaluation[0]

            def moments(self, evaluation):
                return evaluation[1], np.diag([.2, .3])

        states = np.array([[2., .3], [-1., -.5]])
        eta, ds, cs, weight, ec, _, _ = integrate_risk_lines(
            GaussianTilt(), states, np.zeros(2), np.eye(2), 0)
        np.testing.assert_allclose(weight, [.5, .5], atol=1e-10)
        np.testing.assert_allclose(eta, [[0., .3], [0., -.5]], atol=1e-10)
        np.testing.assert_allclose(ds, eta, atol=1e-10)
        np.testing.assert_allclose(cs, np.tile(np.diag([1.2, .3]), (2, 1, 1)), atol=1e-10)
        np.testing.assert_allclose(ec, np.tile(np.diag([1., 0.]), (2, 1, 1)), atol=1e-10)

    def test_inverse_tilt_recovers_original_gaussian_moments(self):
        nodes, mass = hermgauss(32); x = np.sqrt(2.) * nodes
        mass = mass / np.sqrt(np.pi); h = 1. + x ** 2
        proposal_mass = mass * h; proposal_mass /= proposal_mass.sum()
        mean, covariance = weighted_moments(x.reshape(1, -1, 1), proposal_mass / h)
        np.testing.assert_allclose(mean, [0.], atol=1e-14)
        np.testing.assert_allclose(covariance, [[1.]], atol=1e-14)
        tilted_variance = weighted_moments(x.reshape(1, -1, 1), proposal_mass)[1]
        self.assertAlmostEqual(tilted_variance[0, 0], 2.)

    def test_function_summary_keeps_within_and_between_variance(self):
        ds = np.array([[[0.], [2.]]]); cs = np.array([[[[.1]], [[.3]]]])
        mean, covariance = weighted_function_moments(ds, cs, np.array([[.75, .25]]))
        np.testing.assert_allclose(mean, [.5], atol=1e-14)
        np.testing.assert_allclose(covariance, [[.9]], atol=1e-14)

    def test_sampling_score_and_retained_weight_cancel_the_same_tilt(self):
        mean = np.array([.2, -.4]); K = np.array([[.5, .15], [.15, .3]])
        labels = [.6, -.1]; s2 = [.01, .04]; center = [.4, -.2]; scale = [.03, .02]
        model = VarianceTiltedPosterior(mean, K, labels, s2, [0, 1], center, scale)
        base = VarianceReferencePosterior(mean, K, labels, s2)
        for eta in (np.array([-3., -2.]), np.array([0., 1.])):
            evaluation = model.evaluate(eta); ds, cs = model.moments(evaluation)
            inverse = model.inverse_weights(ds.reshape(1, 1, 2), cs.reshape(1, 1, 2, 2))[0, 0]
            self.assertAlmostEqual(evaluation[0] + np.log(inverse), base.evaluate(eta)[0], places=12)


if __name__ == '__main__': unittest.main()
