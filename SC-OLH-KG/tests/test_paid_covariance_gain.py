import unittest

import numpy as np

from performance.check_legacy_paid_covariance_gain import fixed_noise_condition


class PaidCovarianceGainTests(unittest.TestCase):
    def test_two_raw_values_match_dense_precision_conditioning(self):
        rng = np.random.default_rng(1267); a = rng.normal(size=(7, 7))
        covariance = a@a.T+.2*np.eye(7); mean = rng.normal(size=7)
        sites, values, noise = [1, 5], [.4, -.7], [.03, .8]
        m, c, gain = fixed_noise_condition(mean, covariance, sites, values, noise)
        precision = np.linalg.inv(covariance); rhs = precision@mean
        for j, y, v in zip(sites, values, noise):
            precision[j, j] += 1/v; rhs[j] += y/v
        expected_covariance = np.linalg.inv(precision)
        np.testing.assert_allclose(c, expected_covariance, atol=1e-11)
        np.testing.assert_allclose(m, expected_covariance@rhs, atol=1e-11)
        self.assertEqual(gain.shape, (7, 2))

    def test_singular_source_field_and_independent_unseen_site(self):
        # A finite basis can give a singular function covariance; noise is still positive.
        direction = np.array([1., -2., .5, 0.]); covariance = np.outer(direction, direction)
        covariance[3, 3] = .7
        m, c, _ = fixed_noise_condition(np.zeros(4), covariance, [0, 1], [2., -1.], [.5, 2.])
        coefficient_variance = 1/(1+1/.5+4/2.)
        coefficient_mean = coefficient_variance*(2/.5+2/2.)
        np.testing.assert_allclose(m[:3], direction[:3]*coefficient_mean, atol=1e-12)
        np.testing.assert_allclose(c[:3, :3], coefficient_variance*covariance[:3, :3], atol=1e-12)
        self.assertEqual(m[3], 0.)
        self.assertEqual(c[3, 3], .7)
        self.assertGreaterEqual(np.linalg.eigvalsh(c).min(), -1e-12)


if __name__ == '__main__':
    unittest.main()
