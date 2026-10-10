import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from representation.source_kernel_population import (KernelPopulationLikelihood,
    predict_kernel_population, rbf_cross, shared_kernel)


class KernelPopulationTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[-1.], [-.5], [.1], [.6], [1.]])
        self.anchor = {'constant': .2, 'constant_variance': .1, 'signal_variance': .9,
                       'lengthscale': [.7], 'normalization': {'mean': 0., 'scale': 1.}}
        self.ys = [np.sin(self.x[:, 0])+.1*i*self.x[:, 0] for i in range(3)]
        self.noises = [np.linspace(.03, .08, len(self.x))*(i+1) for i in range(3)]
        self.parameters = np.array([.3, .08, .05])
        self.likelihood = KernelPopulationLikelihood(self.x, self.ys, self.noises, self.anchor)
        common = shared_kernel(self.x, self.x, self.anchor)
        residual = self.parameters[0]*rbf_cross(self.x, self.x, self.anchor['lengthscale'])+self.parameters[1]
        blocks = [residual+np.diag(v+self.parameters[2]) for v in self.noises]
        self.dense_covariance = np.tile(common, (3, 3))+block_diag(*blocks)
        self.error = np.concatenate(self.ys)-self.anchor['constant']

    def test_whitened_joint_likelihood_and_source_posterior_equal_dense_gaussian(self):
        value, _, state = self.likelihood.evaluate(self.parameters)
        dense = .5*(len(self.error)*np.log(2*np.pi)
            +np.linalg.slogdet(self.dense_covariance)[1]
            +self.error @ np.linalg.solve(self.dense_covariance, self.error))/len(self.error)
        self.assertAlmostEqual(value, dense, places=12)
        common = shared_kernel(self.x, self.x, self.anchor)
        cross = np.tile(common, (1, 3))
        solved = np.linalg.solve(self.dense_covariance, cross.T)
        np.testing.assert_allclose(state['mean'], self.anchor['constant']+solved.T @ self.error, atol=2e-13)
        np.testing.assert_allclose(state['mean_covariance'], common-cross @ solved, atol=2e-13)

    def test_source_residual_gradient_equals_finite_differences(self):
        _, gradient, _ = self.likelihood.evaluate(self.parameters)
        numerical = []
        for i in range(3):
            plus, minus = self.parameters.copy(), self.parameters.copy()
            plus[i] += 1e-6; minus[i] -= 1e-6
            numerical.append((self.likelihood.evaluate(plus)[0]-self.likelihood.evaluate(minus)[0])/2e-6)
        np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-6)

    def test_zero_signal_shared_mean_matches_dense_conditioning_without_variance_floor(self):
        anchor = {**self.anchor, 'signal_variance': 0.}
        model = KernelPopulationLikelihood(self.x, self.ys, self.noises, anchor)
        value, _, state = model.evaluate(self.parameters)
        self.assertEqual(model.root.shape, (len(self.x), 1))
        common = shared_kernel(self.x, self.x, anchor)
        residual = self.parameters[0]*rbf_cross(self.x, self.x, anchor['lengthscale'])+self.parameters[1]
        blocks = [residual+np.diag(v+self.parameters[2]) for v in self.noises]
        dense = np.tile(common, (3, 3))+block_diag(*blocks)
        expected = .5*(len(self.error)*np.log(2*np.pi)+np.linalg.slogdet(dense)[1]
                      +self.error @ np.linalg.solve(dense, self.error))/len(self.error)
        self.assertAlmostEqual(value, expected, places=12)
        target = np.array([[-.7], [1.4]])
        fit = {'anchor': anchor, 'residual_parameters': self.parameters,
               'common_function_whitened_mean': state['whitened_mean'],
               'common_function_whitened_covariance': state['whitened_covariance']}
        mean, covariance = predict_kernel_population(self.x, target, fit)
        cross = np.tile(shared_kernel(target, self.x, anchor), (1, 3))
        solved = np.linalg.solve(dense, cross.T)
        np.testing.assert_allclose(mean, anchor['constant']+solved.T @ self.error, atol=2e-13)
        residual_target = (self.parameters[0]*rbf_cross(target, target, anchor['lengthscale'])
                           +self.parameters[1]+self.parameters[2]*np.eye(len(target)))
        np.testing.assert_allclose(covariance, shared_kernel(target, target, anchor)
                                   -cross @ solved+residual_target, atol=2e-13)

    def test_new_task_posterior_equals_dense_conditioning_and_retains_interpolation(self):
        target = np.array([[-.7], [.3], [1.4]])
        _, _, state = self.likelihood.evaluate(self.parameters)
        fit = {'anchor': self.anchor, 'residual_parameters': self.parameters,
               'common_function_whitened_mean': state['whitened_mean'],
               'common_function_whitened_covariance': state['whitened_covariance']}
        mean, covariance = predict_kernel_population(self.x, target, fit)
        common = shared_kernel(target, target, self.anchor)
        cross = np.tile(shared_kernel(target, self.x, self.anchor), (1, 3))
        solved = np.linalg.solve(self.dense_covariance, cross.T)
        residual = (self.parameters[0]*rbf_cross(target, target, self.anchor['lengthscale'])
                    +self.parameters[1]+self.parameters[2]*np.eye(len(target)))
        np.testing.assert_allclose(mean, self.anchor['constant']+solved.T @ self.error, atol=2e-13)
        np.testing.assert_allclose(covariance, common-cross @ solved+residual, atol=2e-13)
        transfer = np.linalg.solve(self.likelihood.root, shared_kernel(self.x, target, self.anchor)).T
        interpolation = common-transfer @ transfer.T
        self.assertGreater(interpolation[-1, -1], .03)
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0.)


if __name__ == '__main__':
    unittest.main()
