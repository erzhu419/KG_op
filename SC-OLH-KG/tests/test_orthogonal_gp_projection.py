import unittest

import numpy as np

from representation.orthogonal_gp_projection import project_gaussian_field


class OrthogonalGPProjectionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1269)
        self.design = np.column_stack([np.ones(9), rng.normal(size=(9, 2))])
        self.coefficient_mean = np.array([.3, -.5, .7])
        self.coefficient_covariance = np.array([[.4, .1, -.05], [.1, .3, .07], [-.05, .07, .2]])
        self.mean = self.design@self.coefficient_mean
        self.smooth = self.design@self.coefficient_covariance@self.design.T
        self.white = .08

    def test_in_space_field_and_white_discrepancy_are_unchanged(self):
        covariance = self.smooth+self.white*np.eye(9)
        result = project_gaussian_field(self.design, self.mean, covariance, self.white)
        np.testing.assert_allclose(result['mean'], self.mean, atol=1e-12)
        np.testing.assert_allclose(result['covariance'], covariance, atol=1e-12)
        np.testing.assert_allclose(result['coefficient_mean'], self.coefficient_mean, atol=1e-12)
        np.testing.assert_allclose(result['coefficient_covariance'], self.coefficient_covariance, atol=1e-12)
        self.assertLess(abs(result['diagnostics']['discarded_smooth_covariance_trace_fraction']), 1e-12)

    def test_only_outside_smooth_component_is_discarded(self):
        q, _ = np.linalg.qr(self.design, mode='complete'); outside = q[:, 3]
        mean = self.mean+2*outside
        covariance = self.smooth+1.2*np.outer(outside, outside)+self.white*np.eye(9)
        result = project_gaussian_field(self.design, mean, covariance, self.white)
        np.testing.assert_allclose(result['mean'], self.mean, atol=1e-12)
        np.testing.assert_allclose(result['covariance'], self.smooth+self.white*np.eye(9), atol=1e-12)
        np.testing.assert_allclose(result['coefficient_covariance'], self.coefficient_covariance, atol=1e-12)
        expected_fraction = 1.2/(np.trace(self.smooth)+1.2)
        self.assertAlmostEqual(result['diagnostics']['discarded_smooth_covariance_trace_fraction'], expected_fraction)
        self.assertGreaterEqual(np.linalg.eigvalsh(result['covariance']).min(), 0.)


if __name__ == '__main__':
    unittest.main()
