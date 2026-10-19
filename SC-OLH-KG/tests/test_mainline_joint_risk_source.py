import unittest

import numpy as np
from scipy.linalg import block_diag

from performance.mainline_joint_risk_source import (
    joint_prior_moments, SourceEvidence, _source_context, _restore_joint_g, read)
from performance.connect_complete_historical_reuse import restore_head


class JointSourceRiskTests(unittest.TestCase):
    def test_zero_increment_native_head_matches_original_full_source_mixture(self):
        context = _source_context()
        inputs = read(context['source_inputs'])
        old = read(context['connection']['legacy_connection_plan'])
        risk = read(old['query_risk_inputs_by_layout']['5'])
        indices = [45, 10, 63]
        coordinates = {'mean': np.asarray(inputs['public_mean_reference'])[indices],
                       'risk': np.asarray(risk['public_reference_psi'])[indices]}
        for pair in context['pairs']:
            D = np.asarray(pair['law']['source_design'])
            projection = np.linalg.lstsq(D, context['mean_coordinates'], rcond=None)[0]
            row = {'new_variances': [0., 0.], 'new_component_weights': pair['masses'].reshape(-1)}
            actual = _restore_joint_g(pair, context, coordinates, projection, row)
            source = {'sites': [], 'batch_means': context['labels'],
                      'sample_variances': context['sample_variances']}
            states = {'positive_eta_mixture': {'eta_states_by_chain': pair['eta'],
                       'positive_component_weights': pair['masses'].reshape(-1)}}
            expected, _ = restore_head(pair['law'], pair['risk'], source, states,
                (coordinates['mean'], coordinates['risk']), context['maps'], context['selection'])
            np.testing.assert_allclose(actual.means, expected.means, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(actual.covariances, expected.covariances, atol=1e-12, rtol=1e-12)
            np.testing.assert_array_equal(actual.log_weights, expected.log_weights)
            self.assertEqual(len(actual.means), 1024)
            self.assertEqual(actual.paid_episodes, 0)

    def setup_prior(self):
        rng = np.random.default_rng(1701)
        D = np.column_stack([np.ones(12), rng.normal(size=12)])
        H = rng.normal(size=(12, 4))
        common = np.array([[.2, .03], [.03, .1]])
        task = np.diag([.04, .02])
        K = D@common@D.T + block_diag(*[d@task@d.T for d in np.split(D, 4)]) + .01*np.eye(12)
        law = {'source_design': D, 'source_mean': D@np.array([.2, -.1]),
               'source_covariance': K, 'unconditioned_common_mean': np.array([.2, -.1]),
               'unconditioned_common_covariance': common, 'task_covariance': task,
               'point_discrepancy_variance': .01}
        Q = np.array([[1., -.2], [1., .4]])
        J = rng.normal(size=(2, 4))
        return law, H, Q, J

    def test_zero_increment_preserves_prior_component_and_identity_evidence(self):
        law, H, Q, J = self.setup_prior()
        prior = joint_prior_moments(law, H, Q, J, [0., 0.])
        np.testing.assert_array_equal(prior['source_covariance'], law['source_covariance'])
        np.testing.assert_array_equal(prior['query_covariance'],
            Q@(law['unconditioned_common_covariance']+law['task_covariance'])@Q.T+.01*np.eye(2))
        np.testing.assert_array_equal(prior['cross_covariance'], Q@law['unconditioned_common_covariance']@law['source_design'].T)
        rng = np.random.default_rng(1702)
        eta = rng.normal(-4., .3, (4, 8, 12))
        masses = np.arange(1, 33, dtype=float).reshape(4, 8)
        masses /= masses.sum()
        labels = law['source_mean'] + rng.normal(0, .15, 12)
        evidence = SourceEvidence(law, H, labels, eta, masses)
        zero = evidence.summarize([0., 0.])
        self.assertAlmostEqual(zero['log_source_evidence_ratio'], 0., places=13)
        np.testing.assert_allclose(zero['new_component_weights'], masses.reshape(-1), atol=1e-15)
        theta = np.array([.2, .1])
        value, gradient = evidence.objective(theta)
        finite = [(evidence.objective(theta+np.eye(2)[j]*1e-5)[0]
                   - evidence.objective(theta-np.eye(2)[j]*1e-5)[0]) / 2e-5 for j in range(2)]
        np.testing.assert_allclose(gradient, finite, atol=1e-8, rtol=1e-6)

    def test_joint_completion_matches_explicit_old_new_coefficients(self):
        law, H, Q, J = self.setup_prior()
        variances = [.07, .03]
        prior = joint_prior_moments(law, H, Q, J, variances)
        common = block_diag(law['unconditioned_common_covariance'], variances[0]*np.eye(4))
        task = block_diag(law['task_covariance'], variances[1]*np.eye(4))
        coefficient_covariance = block_diag(common, *([task]*5))
        feature = np.column_stack([law['source_design'], H])
        source = np.zeros((12, 36))
        source[:, :6] = feature
        for t in range(4):
            source[3*t:3*t+3, 6+6*t:12+6*t] = feature[3*t:3*t+3]
        query = np.zeros((2, 36))
        query[:, :6] = np.column_stack([Q, J])
        query[:, -6:] = query[:, :6]
        K = source@coefficient_covariance@source.T+.01*np.eye(12)
        C = query@coefficient_covariance@source.T
        V = query@coefficient_covariance@query.T+.01*np.eye(2)
        np.testing.assert_allclose(K, prior['source_covariance'], atol=1e-15)
        np.testing.assert_allclose(C, prior['cross_covariance'], atol=1e-15)
        np.testing.assert_allclose(V, prior['query_covariance'], atol=1e-15)
        noise = np.exp(np.linspace(-4., -3., 12))/3
        labels = law['source_mean'] + np.linspace(-.1, .15, 12)
        inverse = np.linalg.inv(K+np.diag(noise))
        coefficient_gain = coefficient_covariance@source.T@inverse
        post_mean = query@coefficient_gain@(labels-law['source_mean']) + prior['query_mean']
        post_cov = query@(coefficient_covariance-coefficient_gain@source@coefficient_covariance)@query.T+.01*np.eye(2)
        np.testing.assert_allclose(post_mean, prior['query_mean']+C@inverse@(labels-law['source_mean']), atol=1e-14)
        np.testing.assert_allclose(post_cov, V-C@inverse@C.T, atol=1e-14)
        posterior_coefficients = coefficient_covariance-coefficient_gain@source@coefficient_covariance
        self.assertGreater(np.max(abs(posterior_coefficients[:2, 2:6])), 1e-6)


if __name__ == '__main__':
    unittest.main()
