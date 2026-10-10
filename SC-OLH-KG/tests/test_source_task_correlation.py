import unittest

import numpy as np

from performance.diagnose_legacy_task_covariance_direction import dense_nll
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_function_prior import SourceFunctionPrior, head_batches
from representation.source_task_correlation import SourceFixedMarginalCorrelationLikelihood
from representation.source_task_covariance import SourceTaskDirectionLikelihood


class SourceTaskCorrelationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1265); reference = rng.normal(size=(40, 2))
        basis = OrthogonalLowOrderBasis(reference); n = 1+basis.feature_dim
        priors = [SourceFunctionPrior(basis, np.linspace(-.1, .1, n)*(h+1), .2*np.eye(n), .03,
            {'source_signal_scale': .2}) for h in range(2)]
        tasks = [{'name': str(i), 'psi': reference[6*i:6*i+6], 'features': reference[6*i:6*i+6],
            'response': rng.normal(size=(6, 2)), 'noise': np.full((6, 2), .04)} for i in range(3)]
        common = [np.full(n, v) for v in (.3, .4)]
        residual = [np.r_[0., np.full(n-1, v)] for v in (.08, .06)]
        direction = np.r_[0., np.linspace(-.1, .1, n-1), 0., np.linspace(.1, .3, n-1)]
        self.base = SourceTaskDirectionLikelihood.from_paired_heads(priors,
            [head_batches(tasks, h) for h in range(2)], [[t['noise'][:, h] for t in tasks] for h in range(2)],
            common, residual, [.03, .07], direction)
        self.model = SourceFixedMarginalCorrelationLikelihood(self.base)

    def test_zero_correlation_reproduces_native_source_posterior(self):
        value, _, state = self.model.evaluate(0.)
        expected, _, native = self.base.evaluate(0.)
        np.testing.assert_allclose(value, expected, atol=1e-12)
        for key in ('mean', 'mean_covariance', 'residual_covariance'):
            np.testing.assert_allclose(state[key], native[key], atol=1e-12)

    def test_dense_likelihood_gradient_fixed_marginals_and_psd(self):
        n = self.model.width; native = np.diag(self.base.residual_diagonal)
        dense = lambda gamma: dense_nll(self.base.blocks, self.base.common_diagonal,
            native+gamma*self.model.derivative, 0.)
        support = self.base.residual_diagonal > 0
        scale = np.sqrt(self.base.residual_diagonal[support])
        for gamma in (-1., -.6, .6, 1.):
            value, gradient, state = self.model.evaluate(gamma)
            np.testing.assert_allclose(value, dense(gamma), atol=1e-10)
            np.testing.assert_allclose(gradient[0], (dense(gamma+1e-5)-dense(gamma-1e-5))/2e-5,
                atol=1e-7, rtol=1e-5)
            covariance = state['residual_covariance']
            np.testing.assert_array_equal(covariance[:n, :n], native[:n, :n])
            np.testing.assert_array_equal(covariance[n:, n:], native[n:, n:])
            whitened = covariance[np.ix_(support, support)]/np.outer(scale, scale)
            self.assertGreaterEqual(np.linalg.eigvalsh(whitened).min(), -1e-12)
            self.assertGreater(np.linalg.norm(state['mean_covariance'][:n, n:]), 0.)


if __name__ == '__main__':
    unittest.main()
