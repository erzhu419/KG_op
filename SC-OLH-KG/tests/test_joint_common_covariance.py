import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_source_joint_population import fixture
from representation.source_joint_population import GaussianPopulationLikelihood, JointCovariancePopulationLikelihood
from representation.source_kernel_population import (JointCommonKernelLikelihood, KernelPopulationLikelihood,
    predict_joint_common_kernel_population, rbf_cross, shared_kernel)


def numerical_gradient(model, parameters):
    result = []
    for i in range(len(parameters)):
        plus, minus = parameters.copy(), parameters.copy()
        plus[i] += 1e-6; minus[i] -= 1e-6
        result.append((model.evaluate(plus)[0]-model.evaluate(minus)[0])/2e-6)
    return result


class JointCommonCovarianceTests(unittest.TestCase):
    def test_structural_initialization_preserves_fixed_common_objective(self):
        prior, batches, noise = fixture()
        old = GaussianPopulationLikelihood(prior, batches, noise)
        b = old.initial_parameters(); b[0] *= .3; b[-1] *= .5
        previous = {'optimizer_parameters': b.tolist()}
        new = JointCovariancePopulationLikelihood(prior, batches, noise)
        initial = new.initial_parameters(previous)
        value, _, state = new.evaluate(initial)
        expected, _, fixed = old.evaluate(b)
        self.assertAlmostEqual(value, expected, places=12)
        np.testing.assert_allclose(state['common_diagonal'], np.diag(prior.covariance), atol=1e-14)
        np.testing.assert_allclose(state['mean'], fixed['mean'], atol=1e-13)

    def test_structural_common_and_residual_gradient_and_new_task_equal_dense_gaussian(self):
        prior, batches, noise = fixture()
        old = GaussianPopulationLikelihood(prior, batches, noise)
        model = JointCovariancePopulationLikelihood(prior, batches, noise)
        p = model.initial_parameters({'optimizer_parameters': old.initial_parameters().tolist()})
        p[:len(prior.basis.group_names)] *= .6
        value, gradient, state = model.evaluate(p)
        np.testing.assert_allclose(gradient, numerical_gradient(model, p), atol=3e-7, rtol=3e-5)
        designs = [np.column_stack([np.ones(len(b.psi)), prior.basis.transform(b.psi)]) for b in batches]
        D = np.vstack(designs); C, B = np.diag(state['common_diagonal']), np.diag(state['residual_diagonal'])
        K = D @ C @ D.T+block_diag(*[d @ B @ d.T+np.diag(n+state['discrepancy'])
                                     for d, n in zip(designs, noise)])
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

    def test_generic_joint_common_gradient_matches_whitened_objective(self):
        x = np.array([[-1.], [-.5], [.1], [.6], [1.]])
        anchor = {'constant': .2, 'constant_variance': .1, 'signal_variance': .9,
                  'lengthscale': [.7], 'normalization': {'mean': .4, 'scale': 1.7}}
        ys = [np.sin(x[:, 0])+.1*i*x[:, 0] for i in range(3)]
        noises = [np.linspace(.03, .08, len(x))*(i+1) for i in range(3)]
        parameters = np.array([.6, .3, .08, .05])
        model = JointCommonKernelLikelihood(x, ys, noises, anchor)
        value, gradient, _ = model.evaluate(parameters)
        whitened = KernelPopulationLikelihood(x, ys, noises, {**anchor, 'signal_variance': parameters[0]})
        self.assertAlmostEqual(value, whitened.evaluate(parameters[1:])[0], places=12)
        np.testing.assert_allclose(gradient, numerical_gradient(model, parameters), atol=2e-8, rtol=2e-6)

    def test_generic_prediction_matches_dense_conditioning_including_zero_common_amplitude(self):
        x = np.array([[-1.], [-.5], [.1], [.6], [1.]])
        anchor = {'constant': .2, 'constant_variance': .1, 'signal_variance': .9,
                  'lengthscale': [.7], 'normalization': {'mean': .4, 'scale': 1.7}}
        ys = [np.sin(x[:, 0])+.1*i*x[:, 0] for i in range(3)]
        noises = [np.linspace(.03, .08, len(x))*(i+1) for i in range(3)]
        model = JointCommonKernelLikelihood(x, ys, noises, anchor)
        target = np.array([[-.7], [1.4]])
        for amplitude in [.6, 0.]:
            p = np.array([amplitude, .3, .08, .05]); _, _, state = model.evaluate(p)
            a = {**anchor, 'signal_variance': amplitude}
            fit = {'anchor': a, 'residual_parameters': p[1:],
                   'common_alpha_sum': state['alpha_sum'], 'common_precision_sum': state['precision_sum']}
            mean, covariance = predict_joint_common_kernel_population(x, target, fit)
            R = p[1]*rbf_cross(x, x, anchor['lengthscale'])+p[2]
            K = np.tile(shared_kernel(x, x, a), (3, 3))+block_diag(*[R+np.diag(n+p[3]) for n in noises])
            cross = np.tile(shared_kernel(target, x, a), (1, 3)); error = np.concatenate(ys)-anchor['constant']
            expected_mean = a['constant']+cross @ np.linalg.solve(K, error)
            Rt = p[1]*rbf_cross(target, target, a['lengthscale'])+p[2]+p[3]*np.eye(len(target))
            expected_covariance = shared_kernel(target, target, a)+Rt-cross @ np.linalg.solve(K, cross.T)
            np.testing.assert_allclose(mean, anchor['normalization']['mean']+1.7*expected_mean, atol=2e-13)
            np.testing.assert_allclose(covariance, 1.7**2*expected_covariance, atol=2e-13)


if __name__ == '__main__':
    unittest.main()
