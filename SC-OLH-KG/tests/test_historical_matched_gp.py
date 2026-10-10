import unittest

import numpy as np
from scipy.linalg import block_diag
from scipy.stats import qmc

from performance.connect_historical_matched_gp import controls_from_fit, summarize_query
from performance.connect_historical_prior_posterior import snapshot
from performance.extend_historical_matched_gp import restore_saved_model
from performance.run_legacy_joint_population_matched_baselines import paired_fields
from representation.joint_response_mixture import JointResponseMixture
from representation.source_kernel_population import (
    UnequalSiteKernelLikelihood, predict_unequal_site_kernel_population, rbf_cross,
)


class MatchedGPTests(unittest.TestCase):
    def setUp(self):
        self.sites = [np.array([[-1.], [.2]]), np.array([[-.4], [.7], [1.2]])]
        self.responses = [np.array([-.8, .3]), np.array([-.1, .6, .9])]
        self.noises = [np.array([.03, .06]), np.array([.08, .05, .04])]
        self.anchor = {'constant': .1, 'constant_variance': .07, 'signal_variance': .8,
            'discrepancy_variance': .02, 'lengthscale': [.9],
            'normalization': {'mean': 2., 'scale': 3.}}
        self.parameters = np.array([.8, .3, .05, .02])
        self.likelihood = UnequalSiteKernelLikelihood(self.sites, self.responses, self.noises, self.anchor)

    def test_unequal_site_covariance_and_gradient_equal_dense_reference(self):
        xs = np.vstack(self.sites); p = self.parameters
        unit = rbf_cross(xs, xs, self.anchor['lengthscale'])
        covariance = p[0]*unit+self.anchor['constant_variance']
        covariance += block_diag(*[p[1]*rbf_cross(x, x, self.anchor['lengthscale'])+p[2] for x in self.sites])
        covariance += np.diag(np.concatenate(self.noises)+p[3])
        error = np.concatenate(self.responses)-self.anchor['constant']
        value, gradient, _ = self.likelihood.evaluate(p)
        dense = .5*(len(error)*np.log(2*np.pi)+np.linalg.slogdet(covariance)[1]
            +error @ np.linalg.solve(covariance, error))/len(error)
        self.assertAlmostEqual(value, dense, places=12)
        numeric = []
        for i in range(4):
            plus, minus = p.copy(), p.copy(); plus[i] += 1e-6; minus[i] -= 1e-6
            numeric.append((self.likelihood.evaluate(plus)[0]-self.likelihood.evaluate(minus)[0])/2e-6)
        np.testing.assert_allclose(gradient, numeric, atol=2e-8, rtol=2e-6)

    def test_new_task_retains_common_interpolation_and_task_covariance_once(self):
        xs = np.vstack(self.sites); target = np.array([[-.7], [1.8]])
        p = self.parameters; _, _, state = self.likelihood.evaluate(p)
        fit = {'anchor': self.anchor, 'source_features': xs, 'common_alpha': state['alpha'],
            'common_precision': state['precision'], 'residual_parameters': p[1:]}
        mean, covariance = predict_unequal_site_kernel_population(target, fit)
        cross = p[0]*rbf_cross(target, xs, self.anchor['lengthscale'])+self.anchor['constant_variance']
        common = p[0]*rbf_cross(target, target, self.anchor['lengthscale'])+self.anchor['constant_variance']
        residual = p[1]*rbf_cross(target, target, self.anchor['lengthscale'])+p[2]+p[3]*np.eye(len(target))
        np.testing.assert_allclose(mean, 2.+3.*(self.anchor['constant']+cross @ state['alpha']), atol=2e-13)
        np.testing.assert_allclose(covariance, 9.*(common-cross @ state['precision'] @ cross.T+residual), atol=2e-13)
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0.)

    def test_four_head_units_and_function_variance_head_order(self):
        heads = []
        for h in range(4):
            anchor = dict(self.anchor, normalization={'mean': 10.*h, 'scale': 1.},
                constant=1., constant_variance=.1)
            joint = {'anchor': anchor, 'source_features': [[0.]], 'common_alpha': [0.],
                'common_precision': [[0.]], 'residual_parameters': [.2, .05, .03]}
            heads.append({'anchor': anchor, 'joint_common': joint})
        controls = controls_from_fit({'heads': heads}, np.array([[-1.], [.4]]),
            np.array([[-.5], [.8]]), .2)
        for name in ('source_common_task_rbf', 'source_task_rbf'):
            f, g = controls[name].components[0]
            np.testing.assert_allclose(f.predict()[0], [[1., 31.], [1., 31.]])
            np.testing.assert_allclose(g.predict()[0], [[11., 21.], [11., 21.]])
        f, g = controls['fixed_rbf_shared_coordinates'].components[0]
        for field in (f, g):
            np.testing.assert_allclose(field.predict()[0], [[0., np.log(.04)], [0., np.log(.04)]])

    def test_saved_gp_restore_keeps_paid_state_and_next_episode_update(self):
        heads = [(np.full(2, m), np.eye(2)*v) for m, v in ((.1, .04), (.2, .03), (-3., .2), (-4., .1))]
        original = JointResponseMixture(['test GP'], [paired_fields(heads)])
        original.update_raw(0, [.15, .1]); original.update_raw(1, [.05, .3])
        restored = restore_saved_model({'posterior': snapshot(original)})
        self.assertEqual(snapshot(restored), snapshot(original))
        original.update_raw(1, [.1, .25]); restored.update_raw(1, [.1, .25])
        self.assertEqual(snapshot(restored), snapshot(original))

    def test_three_complement_prefixes_use_the_last_adjacent_pair(self):
        uniforms = qmc.Sobol(2, scramble=True, seed=202610078).random_base2(7)
        values = (np.arange(128)+.1)*1e-5; records = []
        for position, value in enumerate(values):
            raw = {'values': [0., value], 'costs': [1., 1.],
                'predictive_log_evidence_max_error': 0., 'effective_sample_size_min': 128.}
            records.append({'origin_identity': 0, 'identity_fantasy_position': position,
                'shared_identity_weights': [1.], 'projected_values': [0., value],
                'projected_action_index': 1, 'raw': {'12': raw, '14': raw}})
        terminal = [{'remaining_calls': b, 'power': 14, 'best_reward': 0., 'best_action_index': 0} for b in (360, 359)]
        spec = {'raw_powers': [12, 14], 'tail_counts': [4, 8, 16], 'complement_prefixes': [16, 32, 64],
            'low_constraint_probability': .125, 'maximum_terminal_contrast_difference': 1e-4,
            'maximum_fantasy_difference': 1e-4}
        model = JointResponseMixture(['test GP'], [(None, None)])
        result = summarize_query(records, model, terminal, [{}, {}], uniforms, spec)
        body32 = np.flatnonzero(uniforms[:32, 1] >= .125)
        body64 = np.flatnonzero(uniforms[:64, 1] >= .125)
        expected = .875*abs(values[body64].mean()-values[body32].mean())
        self.assertAlmostEqual(result['numerical_changes']['complement_adaptation'], expected, places=15)


if __name__ == '__main__':
    unittest.main()
