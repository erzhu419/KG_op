import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_source_joint_population import fixture
from representation.source_joint_population import FlatCovariancePopulationLikelihood, fit_flat_common_source_population


class FlatCommonCovarianceTests(unittest.TestCase):
    def test_gradient_likelihood_and_new_task_match_dense_gaussian_on_distinct_sites(self):
        prior, batches, noise = fixture()
        model = FlatCovariancePopulationLikelihood(prior, batches, noise)
        p = np.array([.7, .2, .4, .08])
        value, gradient, state = model.evaluate(p)
        finite = []
        for i in range(4):
            plus, minus = p.copy(), p.copy(); plus[i] += 1e-6; minus[i] -= 1e-6
            finite.append((model.evaluate(plus)[0]-model.evaluate(minus)[0])/2e-6)
        np.testing.assert_allclose(gradient, finite, atol=2e-7, rtol=2e-5)
        designs = [np.column_stack([np.ones(len(b.psi)), prior.basis.transform(b.psi)]) for b in batches]
        D = np.vstack(designs); C = np.diag(state['common_diagonal']); B = np.diag(state['residual_diagonal'])
        K = D @ C @ D.T+block_diag(*[d @ B @ d.T+np.diag(n+state['discrepancy']) for d, n in zip(designs, noise)])
        error = np.concatenate([b.signals for b in batches])-D @ prior.mean
        expected = .5*(len(error)*np.log(2*np.pi)+np.linalg.slogdet(K)[1]+error @ np.linalg.solve(K, error))/len(error)
        self.assertAlmostEqual(value, expected, places=11)
        target = np.array([[-.7, .2], [1.4, -.3]])
        T = np.column_stack([np.ones(len(target)), prior.basis.transform(target)])
        cross = T @ C @ D.T
        np.testing.assert_allclose(T @ state['mean'], T @ prior.mean+cross @ np.linalg.solve(K, error), atol=1e-11)
        actual = T @ (state['mean_covariance']+B) @ T.T+state['discrepancy']*np.eye(len(target))
        expected = T @ (C+B) @ T.T+state['discrepancy']*np.eye(len(target))-cross @ np.linalg.solve(K, cross.T)
        np.testing.assert_allclose(actual, expected, atol=1e-11)
        np.testing.assert_allclose(state['common_diagonal'][1:], prior.diagnostics['source_signal_scale']*(p[0]+.01)/prior.basis.feature_dim)
        np.testing.assert_allclose(state['residual_diagonal'][1:], prior.diagnostics['source_signal_scale']*(p[2]+.01)/prior.basis.feature_dim)

    def test_initialization_preserves_total_excess_variance_and_zero_scales_keep_original_tails(self):
        prior, batches, noise = fixture()
        previous = {'common_group_variances': [.08, .06, .04],
            'source_group_variances': [.03, .02, .01], 'residual_coefficient_diagonal': [.05], 'discrepancy_variance': .02}
        model = FlatCovariancePopulationLikelihood(prior, batches, noise)
        p = model.initial_parameters(previous); _, _, state = model.evaluate(p)
        tail = .01*model.signal/prior.basis.feature_dim
        self.assertAlmostEqual(np.sum(state['common_diagonal'][1:]-tail), .18)
        self.assertAlmostEqual(np.sum(state['residual_diagonal'][1:]-tail), .06)
        _, _, zero = model.evaluate(np.zeros(4))
        np.testing.assert_array_equal(zero['common_diagonal'][1:], np.full(prior.basis.feature_dim, tail))
        np.testing.assert_array_equal(zero['residual_diagonal'][1:], np.full(prior.basis.feature_dim, tail))
        fitted = fit_flat_common_source_population(prior, batches, noise, previous,
            optimizer={'method': 'L-BFGS-B', 'max_iterations': 100, 'ftol': 1e-9, 'gtol': 1e-5})
        np.testing.assert_allclose(fitted.covariance, np.asarray(fitted.diagnostics['population_mean_covariance'])
                                  +np.diag(fitted.diagnostics['residual_coefficient_diagonal']))
        self.assertLessEqual(fitted.diagnostics['source_negative_log_likelihood_per_label'],
                             fitted.diagnostics['initial_source_negative_log_likelihood_per_label'])


if __name__ == '__main__':
    unittest.main()
