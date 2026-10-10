import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_source_joint_population import fixture
from representation.intercept_task_population import InterceptTaskPopulationLikelihood
from representation.source_joint_population import GaussianPopulationLikelihood, JointCovariancePopulationLikelihood


class InterceptTaskTests(unittest.TestCase):
    def model(self):
        prior, batches, noise = fixture()
        fixed = GaussianPopulationLikelihood(prior, batches, noise)
        native = JointCovariancePopulationLikelihood(prior, batches, noise)
        full = native.initial_parameters({'optimizer_parameters': fixed.initial_parameters()})
        restricted = InterceptTaskPopulationLikelihood(prior, batches, noise)
        return restricted, restricted.initial_parameters({'optimizer_parameters': full}), batches, noise

    def test_restricted_covariance_matches_independent_dense_gaussian(self):
        model, parameters, batches, noise = self.model()
        value, _, state = model.evaluate(parameters)
        designs = [np.column_stack([np.ones(len(b.psi)), model.basis.transform(b.psi)]) for b in batches]
        D = np.vstack(designs); common = np.diag(state['common_diagonal'])
        task = np.diag(np.r_[model.signal*parameters[-2], np.full(model.basis.feature_dim, model.tail)])
        K = D@common@D.T+block_diag(*[d@task@d.T+np.diag(n+state['discrepancy']) for d, n in zip(designs, noise)])
        error = np.concatenate([b.signals for b in batches])-D@model.prior.mean
        expected = .5*(len(error)*np.log(2*np.pi)+np.linalg.slogdet(K)[1]+error@np.linalg.solve(K, error))/len(error)
        self.assertAlmostEqual(value, expected, places=11)
        expected_mean = model.prior.mean+common@D.T@np.linalg.solve(K, error)
        np.testing.assert_allclose(state['mean'], expected_mean, atol=1e-11)
        np.testing.assert_allclose(state['residual_diagonal'][1:], model.tail, atol=1e-14)

    def test_free_parameter_gradient_matches_finite_differences(self):
        model, parameters, _, _ = self.model()
        gradient = model.evaluate(parameters)[1]; numerical = []
        for i in range(len(parameters)):
            plus, minus = parameters.copy(), parameters.copy()
            plus[i] += 1e-6; minus[i] -= 1e-6
            numerical.append((model.evaluate(plus)[0]-model.evaluate(minus)[0])/2e-6)
        np.testing.assert_allclose(gradient, numerical, atol=2e-7, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
