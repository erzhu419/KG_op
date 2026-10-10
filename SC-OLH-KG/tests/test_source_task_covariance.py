import unittest

import numpy as np

from performance.diagnose_legacy_task_covariance_direction import dense_nll
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_function_prior import SourceFunctionPrior, head_batches
from representation.source_task_covariance import SourceTaskDirectionLikelihood


class SourceTaskCovarianceTests(unittest.TestCase):
    def test_paired_heads_reproduce_independent_law_and_correlated_dense_marginal(self):
        rng = np.random.default_rng(1207); reference = rng.normal(size=(40, 2))
        basis = OrthogonalLowOrderBasis(reference); n = 1+basis.feature_dim
        priors = [SourceFunctionPrior(basis, np.linspace(-.1, .1, n)*(h+1), .2*np.eye(n), .03,
            {'source_signal_scale': .2}) for h in range(2)]
        tasks = [{'name': str(i), 'psi': reference[6*i:6*i+6], 'features': reference[6*i:6*i+6],
            'response': rng.normal(size=(6, 2)), 'noise': np.full((6, 2), .04)} for i in range(3)]
        common, residual, discrepancy = [np.full(n, v) for v in (.3, .4)], [np.full(n, v) for v in (.08, .06)], [.03, .07]
        direction = np.r_[np.linspace(-.1, .1, n), np.linspace(.1, .3, n)]
        batches, noises = [head_batches(tasks, h) for h in range(2)], [[t['noise'][:, h] for t in tasks] for h in range(2)]
        model = SourceTaskDirectionLikelihood.from_paired_heads(priors, batches, noises, common, residual, discrepancy, direction)
        independent = [SourceTaskDirectionLikelihood(priors[h], batches[h], noises[h], common[h], residual[h], discrepancy[h], direction[h*n:(h+1)*n])
            for h in range(2)]
        value, _, state = model.evaluate(0.)
        originals = [m.evaluate(0.) for m in independent]
        np.testing.assert_allclose(value, np.mean([r[0] for r in originals]), atol=1e-10)
        np.testing.assert_allclose(state['mean'], np.concatenate([r[2]['mean'] for r in originals]), atol=1e-10)
        np.testing.assert_allclose(state['mean_covariance'][:n, n:], 0., atol=1e-12)
        value, gradient, state = model.evaluate(.8)
        dense = lambda rho: dense_nll(model.blocks, np.concatenate(common),
            np.diag(np.concatenate(residual))+rho*np.outer(direction, direction), 0.)
        np.testing.assert_allclose(value, dense(.8), atol=1e-10)
        np.testing.assert_allclose(gradient[0], (dense(.80001)-dense(.79999))/.00002, atol=1e-7, rtol=1e-5)
        self.assertGreater(np.linalg.norm(state['mean_covariance'][:n, n:]), 0.)

    def test_common_task_likelihood_and_scale_gradient_match_dense_marginal(self):
        rng = np.random.default_rng(854); reference = rng.normal(size=(40, 2))
        basis = OrthogonalLowOrderBasis(reference); n = 1+basis.feature_dim
        prior = SourceFunctionPrior(basis, np.linspace(-.1, .1, n), .2*np.eye(n), .03, {'source_signal_scale': .2})
        tasks = [{'name': str(i), 'psi': reference[6*i:6*i+6], 'features': reference[6*i:6*i+6],
            'response': rng.normal(size=(6, 2)), 'noise': np.full((6, 2), .04)} for i in range(3)]
        common, residual, direction = np.full(n, .3), np.full(n, .08), np.linspace(-.1, .1, n)
        model = SourceTaskDirectionLikelihood(prior, head_batches(tasks, 0), [t['noise'][:, 0] for t in tasks], common, residual, .03, direction)
        for scale in (.3, 1.4):
            value, gradient, _ = model.evaluate(scale)
            matrix = np.diag(residual)+scale*np.outer(direction, direction)
            np.testing.assert_allclose(value, dense_nll(model.blocks, common, matrix, .03), atol=1e-10)
            step = 1e-5
            dense = lambda rho: dense_nll(model.blocks, common, np.diag(residual)+rho*np.outer(direction, direction), .03)
            numerical = (dense(scale+step)-dense(scale-step))/(2*step)
            np.testing.assert_allclose(gradient[0], numerical, atol=1e-7, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
