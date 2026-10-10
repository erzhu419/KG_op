"""The faster Gaussian conditional must retain the original spectral cutoff."""

import unittest

import numpy as np

from core.conditional_terminal_kg import conditional_covariance_inverse


class ConditionalCovarianceTests(unittest.TestCase):
    def test_full_rank_correlated_policy_covariances(self):
        rng = np.random.default_rng(91)
        for dimension in (1, 2, 3):
            loading = rng.normal(size=(128, dimension, dimension))
            covariance = loading @ loading.transpose(0, 2, 1)+.2*np.eye(dimension)
            expected = np.linalg.pinv(covariance, hermitian=True, rcond=1e-12)
            np.testing.assert_allclose(conditional_covariance_inverse(covariance), expected, atol=2e-13, rtol=2e-13)

    def test_shared_and_nearly_constant_eta_keeps_pseudoinverse(self):
        covariance = np.array([[[1., 1.], [1., 1.]], [[1., 0.], [0., 1e-14]]])
        expected = np.linalg.pinv(covariance, hermitian=True, rcond=1e-12)
        np.testing.assert_array_equal(conditional_covariance_inverse(covariance), expected)

    def test_mixed_truncated_and_full_rank_batches(self):
        covariance = np.array([[[1., .3], [.3, .7]], [[1., 0.], [0., 1e-14]]])
        expected = np.linalg.pinv(covariance, hermitian=True, rcond=1e-12)
        np.testing.assert_allclose(conditional_covariance_inverse(covariance), expected, atol=1e-14, rtol=1e-14)


if __name__ == '__main__':
    unittest.main()
