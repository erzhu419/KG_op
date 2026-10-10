import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, spectral_weights
from representation.source_function_prior import SourceFunctionPrior
from representation.source_joint_population import (
    GaussianPopulationLikelihood, continue_joint_source_population,
    fit_joint_source_population, same_source_full_residual_prior)
from representation.transferable_spectral import SourceDomainBatch


def fixture():
    rng = np.random.default_rng(80781)
    basis = OrthogonalLowOrderBasis(rng.uniform(-1.5, 1.5, (24, 2)))
    groups = np.array([.08, .06, .04])
    weights, _ = spectral_weights(basis, .8)
    covariance = np.diag(np.r_[.25, groups[basis.group_index]*weights+.003/basis.feature_dim])
    prior = SourceFunctionPrior(basis, np.r_[.2, np.zeros(basis.feature_dim)], covariance, .04,
        {'source_signal_scale': .3, 'source_group_variances': groups.tolist(), 'spectral_exponent': .8})
    inputs = [rng.uniform(-1.2, 1.2, (8, 2)) for _ in range(3)]
    batches = [SourceDomainBatch(str(i), x, .5+.3*np.sin(x[:, 0])+.05*i) for i, x in enumerate(inputs)]
    noise = [np.linspace(.015, .03, 8) for _ in batches]
    return prior, batches, noise


class JointPopulationTests(unittest.TestCase):
    def test_block_likelihood_and_posterior_match_dense_joint_gaussian(self):
        prior, batches, noise = fixture()
        model = GaussianPopulationLikelihood(prior, batches, noise)
        params = model.initial_parameters(); params[0] *= .3; params[-1] *= .5
        value, _, state = model.evaluate(params)
        designs = [np.column_stack([np.ones(len(b.psi)), prior.basis.transform(b.psi)]) for b in batches]
        D = np.vstack(designs); y = np.concatenate([b.signals for b in batches])
        B = np.diag(state['residual_diagonal'])
        V = block_diag(*[d @ B @ d.T+np.diag(n+state['discrepancy']) for d, n in zip(designs, noise)])
        K = V+D @ prior.covariance @ D.T
        error = y-D @ prior.mean
        expected = .5*(len(y)*np.log(2*np.pi)+np.linalg.slogdet(K)[1]+error @ np.linalg.solve(K, error))/len(y)
        expected_mean = prior.mean+prior.covariance @ D.T @ np.linalg.solve(K, error)
        expected_cov = prior.covariance-prior.covariance @ D.T @ np.linalg.solve(K, D @ prior.covariance)
        self.assertAlmostEqual(value, expected, places=11)
        np.testing.assert_allclose(state['mean'], expected_mean, atol=1e-11)
        np.testing.assert_allclose(state['mean_covariance'], expected_cov, atol=1e-11)

    def test_integrated_mean_likelihood_gradient(self):
        prior, batches, noise = fixture()
        model = GaussianPopulationLikelihood(prior, batches, noise)
        params = model.initial_parameters()
        _, gradient, _ = model.evaluate(params)
        numerical = []
        for i in range(len(params)):
            plus, minus = params.copy(), params.copy()
            plus[i] += 1e-6; minus[i] -= 1e-6
            numerical.append((model.evaluate(plus)[0]-model.evaluate(minus)[0])/2e-6)
        np.testing.assert_allclose(gradient, numerical, atol=2e-7, rtol=2e-5)

    def test_new_task_covariance_and_full_control_preserve_mean_uncertainty(self):
        prior, batches, noise = fixture()
        joint = fit_joint_source_population(prior, batches, noise, max_iterations=40)
        full = same_source_full_residual_prior(joint)
        H = np.asarray(joint.diagnostics['population_mean_covariance'])
        b = np.asarray(joint.diagnostics['residual_coefficient_diagonal'])
        np.testing.assert_allclose(joint.covariance, H+np.diag(b))
        pooled = b.copy(); pooled[1:] = b[1:].mean()
        np.testing.assert_allclose(full.covariance, H+np.diag(pooled))
        np.testing.assert_array_equal(full.mean, joint.mean)
        self.assertEqual(full.discrepancy_variance, joint.discrepancy_variance)
        self.assertGreater(np.linalg.eigvalsh(H)[0], 0.)
        self.assertGreaterEqual(joint.discrepancy_variance, 0.)
        self.assertTrue(joint.diagnostics['new_task_covariance_includes_population_mean_uncertainty'])

    def test_source_completion_restarts_the_same_recorded_objective(self):
        prior, batches, noise = fixture()
        capped = fit_joint_source_population(prior, batches, noise, max_iterations=2)
        original_mean, original_covariance = capped.mean.copy(), capped.covariance.copy()
        completed = continue_joint_source_population(prior, batches, noise, capped, max_iterations=64)
        self.assertAlmostEqual(completed.diagnostics['continued_from_source_negative_log_likelihood_per_label'],
                               capped.diagnostics['source_negative_log_likelihood_per_label'], places=11)
        self.assertLessEqual(completed.diagnostics['source_negative_log_likelihood_per_label'],
                             capped.diagnostics['source_negative_log_likelihood_per_label']+1e-12)
        np.testing.assert_array_equal(capped.mean, original_mean)
        np.testing.assert_array_equal(capped.covariance, original_covariance)


if __name__ == '__main__':
    unittest.main()
