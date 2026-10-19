"""A missing-direction correction must change the native conditional field."""

from copy import deepcopy
import json
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

from performance import mainline_risk_correction as correction
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture


def field(mean, variance):
    mean = np.asarray(mean)
    covariance = np.diag(np.asarray(variance).reshape(-1))
    return JointFunctionRiskPosterior.from_moments(
        mean, covariance.reshape(len(mean), 2, len(mean), 2))


def model(names=('spectral/head_specific/test',), components=1):
    pairs = []
    for i, _ in enumerate(names):
        f = field([[.2, -4.], [.3, -4.2], [.4, -4.1]],
                  [[.03, .2], [.04, .15], [.05, .25]])
        g = field([[.05+i*.1, -3.5], [-.1, -3.8], [.2, -3.6]],
                  [[.04, .2], [.06, .15], [.05, .25]])
        if components > 1:
            for current in (f, g):
                current.means = np.concatenate([current.means, current.means+.01])
                current.covariances = np.concatenate([current.covariances, 1.1*current.covariances])
                current.log_weights = np.log([.3, .7])
                current.component_pairs = [(0, 0), (0, 0)]
        pairs.append((f, g))
    return JointResponseMixture(list(names), pairs, np.arange(1., len(names)+1))


def independent_raw_reference(mean, covariance, query, observation):
    """Integrate full Gaussian conditioning over lognoise with Hermite nodes.

    This reference conditions the entire vector at each fixed eta node, rather
    than calling the production local integrator or transport implementation.
    """
    nodes, base_weights = hermgauss(80)
    gi, ei = 2*query, 2*query+1
    eta = mean[ei]+np.sqrt(2*covariance[ei, ei])*nodes
    cross_eta = covariance[:, ei]
    conditional_mean = mean[None, :]+np.outer(eta-mean[ei], cross_eta/covariance[ei, ei])
    conditional_covariance = covariance-np.outer(cross_eta, cross_eta)/covariance[ei, ei]
    total = conditional_covariance[gi, gi]+np.exp(eta)
    error = observation-conditional_mean[:, gi]
    logs = np.log(base_weights)-.5*(np.log(total)+error**2/total)
    weights = np.exp(logs-logsumexp(logs))
    cross_g = conditional_covariance[:, gi]
    posterior_means = conditional_mean+np.outer(error/total, cross_g)
    posterior_covariances = (conditional_covariance[None, :, :]
                             - np.outer(cross_g, cross_g)[None, :, :]/total[:, None, None])
    expected_mean = weights@posterior_means
    centered = posterior_means-expected_mean
    expected_covariance = np.einsum('k,kij->ij', weights, posterior_covariances)
    expected_covariance += (centered.T*weights)@centered
    return expected_mean, expected_covariance


class MainlineRiskCorrectionTests(unittest.TestCase):
    residual = np.array([[1., 0., .5, -.25], [.25, 1., -.3, .5], [-.4, .25, 1., .75]])
    beta = np.array([.1, -.2, .05, .03])
    beta_cov = np.diag([.04, .03, .02, .01])

    def test_augmented_gaussian_law_changes_only_selected_native_g(self):
        names = ['spectral/head_specific/a', 'additive/head_specific/a', 'spectral/head_specific/b']
        actual = model(names, components=2)
        before = deepcopy(actual)
        record = correction.add_risk_directions(actual, self.residual, self.beta, self.beta_cov,
                                                'spectral/head_specific/')
        self.assertEqual(record['affected_identities'], [names[0], names[2]])
        # Marginalize the independent augmented random vector [old g, beta].
        observation_matrix = np.column_stack([np.eye(3), self.residual])
        for identity, ((old_f, old_g), (new_f, new_g)) in enumerate(zip(before.components, actual.components)):
            np.testing.assert_array_equal(new_f.means, old_f.means)
            np.testing.assert_array_equal(new_f.covariances, old_f.covariances)
            np.testing.assert_array_equal(new_g.means[:, 1::2], old_g.means[:, 1::2])
            np.testing.assert_array_equal(new_g.covariances[:, 1::2, :], old_g.covariances[:, 1::2, :])
            np.testing.assert_array_equal(new_g.covariances[:, :, 1::2], old_g.covariances[:, :, 1::2])
            if identity == 1:
                np.testing.assert_array_equal(new_g.means, old_g.means)
                np.testing.assert_array_equal(new_g.covariances, old_g.covariances)
            else:
                for k in range(2):
                    joint_mean = np.r_[old_g.means[k, ::2], self.beta]
                    joint_cov = np.zeros((7, 7))
                    joint_cov[:3, :3] = old_g.covariances[k, ::2, ::2]
                    joint_cov[3:, 3:] = self.beta_cov
                    np.testing.assert_allclose(new_g.means[k, ::2], observation_matrix@joint_mean)
                    np.testing.assert_allclose(new_g.covariances[k, ::2, ::2],
                                               observation_matrix@joint_cov@observation_matrix.T)
                self.assertGreater(abs(new_g.covariances[0, 0, 4]), .001)
            for old_field, new_field in ((old_f, new_f), (old_g, new_g)):
                np.testing.assert_array_equal(new_field.log_weights, old_field.log_weights)
                self.assertEqual(new_field.paid_episodes, 0)
                self.assertEqual(new_field.paid_batches, 0)
        np.testing.assert_array_equal(actual.weights, before.weights)
        self.assertEqual(actual.paid_episodes, 0)
        self.assertEqual(actual.history, [])

    def test_raw_observations_propagate_using_native_covariance_and_eta(self):
        actual = model()
        correction.add_risk_directions(actual, self.residual, self.beta, self.beta_cov,
                                       'spectral/head_specific/')
        risk = actual.components[0][1]
        first_unobserved = risk.means[0, [2, 4]].copy()
        for step, observation in enumerate([.4, -.15]):
            mean, covariance = risk.means[0].copy(), risk.covariances[0].copy()
            expected_mean, expected_covariance = independent_raw_reference(mean, covariance, 0, observation)
            actual.update_raw(0, [.3, observation])
            np.testing.assert_allclose(risk.means[0], expected_mean, atol=2e-10, rtol=2e-9)
            np.testing.assert_allclose(risk.covariances[0], expected_covariance, atol=2e-10, rtol=2e-9)
            self.assertEqual(actual.paid_episodes, step+1)
            if step == 0:
                self.assertGreater(np.max(abs(risk.means[0, [2, 4]]-first_unobserved)), .01)
                self.assertGreater(abs(risk.covariances[0, 0, 1]), 1e-5)
        with self.assertRaises(AssertionError):
            correction.add_risk_directions(actual, self.residual, self.beta, self.beta_cov,
                                           'spectral/head_specific/')

    def test_public_residual_removes_frozen_represented_directions(self):
        mean = np.arange(32., dtype=float).reshape(8, 4)/11
        mean[:, 1:] = np.sin(mean[:, 1:])
        design = np.column_stack([np.ones(8), mean[:, 0]])
        projection = np.linalg.lstsq(design, mean, rcond=None)[0]
        calibration = {'layout': '5', 'maps': 'maps', 'selection_plan': 'selection',
                       'projection_coefficients': projection.tolist()}
        coordinates = {'mean': mean, 'risk': np.zeros((8, 10))}
        with patch.object(correction, 'read', side_effect=[{'rows': []}, {}]), \
                patch.object(correction, 'design_for', return_value=design):
            residual = correction.public_mean_residual(coordinates, calibration)
        np.testing.assert_allclose(design.T@residual, 0., atol=1e-14)
        np.testing.assert_allclose(residual[:, 0], 0., atol=1e-14)
        self.assertGreater(np.linalg.norm(residual[:, 1:]), .1)

    def test_real_four_policy_frozen_restore_and_calibration(self):
        from performance.mainline_cross_dimension_prior import restore_cross_dimension_prior
        from performance.mainline_moment_policy import moment_state

        started = time.process_time()
        spec = correction.read('performance/manifests/mainline_d1000_ranked_v1.json')
        reuse = correction.read(spec['reuse_plan'])
        pilot = correction.read(correction.read(reuse['independent_execution_plan'])['pilot_plan'])
        saved = correction.read(spec['query_coordinates'])
        indices = [45, 10, 83, 84]
        coordinates = {key: np.asarray(saved[key])[indices].tolist() for key in ('mean', 'risk')}
        calibration = correction.read('paper_artifacts/mainline_d1000_risk_basis_diagnostic_v1/calibration.json')
        actual, restored = restore_cross_dimension_prior(reuse, pilot, '5', 'source_structural_mixture', coordinates)
        before = moment_state(actual)
        weights = actual.weights.copy()
        record = correction.apply_public_mean_correction(actual, coordinates, calibration)
        after = moment_state(actual)
        self.assertEqual(restored['positive_components_per_head'], [1024]*16)
        self.assertEqual(len(record['affected_identities']), 2)
        mass = sum(weight for name, weight in zip(actual.names, weights)
                   if name.startswith(calibration['affected_identity_prefix']))
        np.testing.assert_allclose(after['mean'][:, 2]-before['mean'][:, 2],
                                   mass*np.asarray(record['mean_shift']), atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(after['mean'][:, [0, 1, 3]], before['mean'][:, [0, 1, 3]], atol=1e-13)
        np.testing.assert_allclose(after['noise'], before['noise'], atol=1e-14)
        np.testing.assert_array_equal(actual.weights, weights)
        self.assertEqual(actual.paid_episodes, 0)
        self.assertEqual(actual.history, [])
        self.assertTrue(np.isfinite(after['mean']).all() and np.isfinite(after['cov']).all())
        print(json.dumps({'real_smoke_python': sys.executable, 'indices': indices,
            'affected_identities': record['affected_identities'], 'paid_episodes': 0,
            'g_mean_shift': (after['mean'][:, 2]-before['mean'][:, 2]).tolist(),
            'new_simulator_calls': 0, 'source_fits': 0,
            'cpu_seconds': time.process_time()-started}), flush=True)


if __name__ == '__main__':
    unittest.main()
