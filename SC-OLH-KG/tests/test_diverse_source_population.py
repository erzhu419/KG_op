import copy
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from performance.check_legacy_diverse_source_design import encode_diverse_source
from representation.source_kernel_population import (UnequalSiteKernelLikelihood,
    predict_unequal_site_kernel_population, rbf_cross, shared_kernel)
from representation.source_joint_population import (GaussianPopulationLikelihood,
    JointCovariancePopulationLikelihood, continue_joint_common_source_population)
from test_source_joint_population import fixture


class DiverseSourcePopulationTests(unittest.TestCase):
    def test_joint_completion_starts_at_saved_full_covariance_vector(self):
        prior, batches, noise = fixture()
        fixed = GaussianPopulationLikelihood(prior, batches, noise)
        model = JointCovariancePopulationLikelihood(prior, batches, noise)
        parameters = model.initial_parameters({'optimizer_parameters': fixed.initial_parameters()})
        parameters[:len(prior.basis.group_names)] *= .6
        previous = {'optimizer_parameters': parameters.tolist(), 'optimizer_iterations': 4096,
            'source_negative_log_likelihood_per_label': model.evaluate(parameters)[0],
            'fixed_common_source_negative_log_likelihood_per_label': fixed.evaluate(fixed.initial_parameters())[0]}
        result = continue_joint_common_source_population(prior, batches, noise, previous,
            optimizer={'method': 'L-BFGS-B', 'max_iterations': 1, 'ftol': 1e-9, 'gtol': 1e-5})
        np.testing.assert_array_equal(result.diagnostics['optimizer_initial_parameters'], parameters)
        self.assertAlmostEqual(result.diagnostics['initial_source_negative_log_likelihood_per_label'],
                               previous['source_negative_log_likelihood_per_label'], places=12)

    def test_distinct_inputs_likelihood_gradient_and_prediction_equal_dense_conditioning(self):
        xs = [np.array([[-1.], [.1], [.7]]), np.array([[-1.], [-.4], [.5], [1.3]])]
        ys = [np.sin(x[:, 0])+.1*i for i, x in enumerate(xs)]
        noises = [np.linspace(.03, .08, len(x)) for x in xs]
        anchor = {'constant': .2, 'constant_variance': .1, 'signal_variance': .6,
                  'lengthscale': [.7], 'normalization': {'mean': .4, 'scale': 1.7}}
        model = UnequalSiteKernelLikelihood(xs, ys, noises, anchor)
        p = np.array([.6, .3, .08, .05]); value, gradient, _ = model.evaluate(p)
        finite = []
        for i in range(4):
            plus, minus = p.copy(), p.copy(); plus[i] += 1e-6; minus[i] -= 1e-6
            finite.append((model.evaluate(plus)[0]-model.evaluate(minus)[0])/2e-6)
        np.testing.assert_allclose(gradient, finite, atol=2e-8, rtol=2e-6)
        target = np.array([[-.7], [1.4]]); X = np.vstack(xs); error = np.concatenate(ys)-anchor['constant']
        for common in [.6, 0.]:
            p[0] = common; a = {**anchor, 'signal_variance': common}
            value, _, state = model.evaluate(p)
            K = shared_kernel(X, X, a)+block_diag(*[p[1]*rbf_cross(x, x, a['lengthscale'])
                +p[2]+np.diag(n+p[3]) for x, n in zip(xs, noises)])
            expected = .5*(len(error)*np.log(2*np.pi)+np.linalg.slogdet(K)[1]
                          +error @ np.linalg.solve(K, error))/len(error)
            self.assertAlmostEqual(value, expected, places=12)
            fit = {'anchor': a, 'source_features': X, 'residual_parameters': p[1:],
                   'common_alpha': state['alpha'], 'common_precision': state['precision']}
            mean, covariance = predict_unequal_site_kernel_population(target, fit)
            cross = shared_kernel(target, X, a)
            expected_mean = a['constant']+cross @ np.linalg.solve(K, error)
            expected_covariance = (shared_kernel(target, target, a)+p[1]*rbf_cross(target, target, a['lengthscale'])
                +p[2]+p[3]*np.eye(len(target))-cross @ np.linalg.solve(K, cross.T))
            np.testing.assert_allclose(mean, .4+1.7*expected_mean, atol=2e-13)
            np.testing.assert_allclose(covariance, 1.7**2*expected_covariance, atol=2e-13)

    def test_excluded_and_heldout_labels_do_not_enter_encoder_or_normalization(self):
        rng = np.random.default_rng(372)
        data = {'bank': rng.integers(1, 90, (7, 5)).tolist(), 'tasks': [
            {'task_id': f'task{i}', 'tau': .2, 'Y_replicates': rng.normal(size=(7, 3, 2)).tolist()}
            for i in range(3)]}
        desc = {i: rng.normal(size=(7, 12)) for i in range(3)}
        layouts = [[0, 2, 4], [0, 3, 5]]
        first = encode_diverse_source(data, [0, 1], layouts, desc, 100)
        changed = copy.deepcopy(data)
        for i, task in enumerate(changed['tasks']):
            for j in range(7):
                if i == 2 or j not in layouts[i]:
                    task['Y_replicates'][j] = (1000*rng.normal(size=(3, 2))).tolist()
        second = encode_diverse_source(changed, [0, 1], layouts, desc, 100)
        np.testing.assert_array_equal(first[0].transform(desc[2]), second[0].transform(desc[2]))
        for a, b in zip(first[1], second[1]):
            for k in ('psi', 'features', 'response', 'noise'):
                np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(first[2], second[2]); np.testing.assert_array_equal(first[3], second[3])
        self.assertEqual(sum(len(t['response']) for t in first[1])*3, 18)


if __name__ == '__main__':
    unittest.main()
