import unittest

import numpy as np

from core.gpr import posterior_mixture_weights
from performance.check_legacy_f_risk_amplitude_importance import log_ratios
from performance.reuse_historical_source_risk_states import risk_statistics
from performance.certify_historical_reused_risk_candidate import response_stationarity
from performance.retain_historical_stationary_risk_cases import retain_risk, retain_followup_risk
from performance.refine_historical_stopped_risk_updates import fixed_update
from performance.bridge_historical_final_risk_evidence import compose_intervals


class SourceRiskStateReuseTests(unittest.TestCase):
    def test_followup_hold_requires_two_distinct_actual_precise_laws(self):
        import copy
        summary = {'scaled_95_MC_halfwidth': [.0004, .0005],
            'split_rhat': [1., 1.], 'autocorrelation_ess': [1000., 1000.]}
        previous = {'accepted_fixed_risk_update': True, 'fixed_proposal_qualified': True,
            'new_current_stationarity_diagnostic_passed': True, 'checked_current_scales': [.2, .4],
            'final_risk_scales': [.199, .399], 'fixed_response_scales': [.1, .3],
            'current_score_summaries': [summary] * 3, 'current_score_precision_passed': True,
            'scaled_projected_current_score': [.0002, .0006], 'new_diagnostic_covariance_change': .005}
        final = {'stationarity_diagnostic_passed': True, 'parameters_changed': False,
            'score_summaries': [summary] * 3, 'score_precision_passed': True,
            'scaled_projected_raw_score': [.0003, .0007], 'relative_diagnostic_covariance_change': .006}
        evidence = {'final_risk_scales': [.199, .399], 'fixed_response_scales': [.1, .3],
            'final_risk_diagnostic_from_parent_bank': final}
        spec = {'maximum_scaled_projected_raw_score': .001, 'maximum_relative_covariance_change': .01,
            'score_precision': {'maximum_split_Rhat': 1.05, 'minimum_functional_ESS': 100,
                'maximum_scaled_95_MC_halfwidth': .001}}
        held = retain_followup_risk(previous, evidence, spec)
        self.assertEqual(held['held_risk_scales'], [.199, .399])
        self.assertEqual(held['additional_accepted_risk_updates'], 0)
        invalid = copy.deepcopy(previous); invalid['checked_current_scales'] = invalid['final_risk_scales']
        with self.assertRaises(AssertionError):
            retain_followup_risk(invalid, evidence, spec)
        invalid = copy.deepcopy(evidence)
        invalid['final_risk_diagnostic_from_parent_bank']['score_summaries'][0]['scaled_95_MC_halfwidth'] = [.002, .002]
        with self.assertRaises(AssertionError):
            retain_followup_risk(previous, invalid, spec)

    def test_bridge_interval_does_not_assume_independent_shared_bank_errors(self):
        from scipy.stats import norm
        segments = [{'source_evidence': {'precision_gate_passed': True,
            'log_source_evidence_ratio': .03, 'conditional_MC_interval_95': [.03 - 1.96 * .01, .03 + 1.96 * .01]}}
            for _ in range(2)]
        result = compose_intervals(segments)
        width = 2 * norm.ppf(1. - .05 / 4) * .01
        np.testing.assert_allclose(result['conditional_MC_interval_95'], [.06 - width, .06 + width])
        self.assertGreater(width, 1.96 * np.sqrt(2) * .01)
        segments[1]['source_evidence']['precision_gate_passed'] = False
        segments[1]['source_evidence']['conditional_MC_interval_95'] = None
        self.assertFalse(compose_intervals(segments)['precision_gate_passed'])

    def test_stationary_risk_hold_preserves_the_actual_law_and_failed_proposal(self):
        import copy
        summary = {'scaled_raw_score': [.0002, .0006], 'scaled_95_MC_halfwidth': [.0004, .0005],
            'split_rhat': [1., 1.], 'autocorrelation_ess': [1000., 1000.]}
        previous = {'update_accepted': True, 'stationarity_diagnostic_passed': True,
            'current_theta': [.2, .4], 'proposed_theta': [.199, .399]}
        failed = {'update_accepted': False, 'stationarity_diagnostic_passed': True,
            'current_theta': [.199, .399], 'proposed_theta': [.198, .398],
            'relative_covariance_change': .005, 'old_score_summaries': [summary] * 3,
            'full_and_split_Q_improvements': [1e-5, -1e-6, 2e-5]}
        result = {'attempts': [previous, failed], 'final_risk_scales': [.199, .399],
            'consecutive_risk_stationarity_iterations': 1}
        spec = {'required_consecutive_iterations': 2, 'maximum_scaled_projected_raw_score': .001,
            'maximum_relative_covariance_change': .01, 'score_precision': {'maximum_split_Rhat': 1.05,
                'minimum_functional_ESS': 100, 'maximum_scaled_95_MC_halfwidth': .001}}
        held = retain_risk(result, spec)
        self.assertEqual(held['held_risk_scales'], [.199, .399])
        self.assertFalse(held['further_proposal_accepted'])
        self.assertEqual(held['rejected_proposal']['full_and_split_Q_improvements'], [1e-5, -1e-6, 2e-5])
        self.assertEqual(held['additional_accepted_risk_updates'], 0)
        for field, value in [('consecutive_risk_stationarity_iterations', 0), ('final_risk_scales', [.2, .4])]:
            invalid = copy.deepcopy(result); invalid[field] = value
            with self.assertRaises(AssertionError):
                retain_risk(invalid, spec)

    def test_reweighted_moments_and_score_match_the_new_gaussian_posterior(self):
        noise = np.array([.3, .4]); observed = np.array([.4, -.2])
        old_theta = np.array([.7, .9]); new_theta = np.array([.55, .8]); white = np.array([.1, .2])
        directions = [np.diag(np.r_[1., 0., np.zeros(46)]), np.diag(np.r_[0., 1., np.zeros(46)])]
        def builder(theta):
            return np.diag(np.r_[white + theta, np.ones(46)]), directions, {}
        prior = white + old_theta
        mean = prior / (prior + noise) * observed; variance = prior * noise / (prior + noise)
        nodes, mass = np.polynomial.hermite.hermgauss(20)
        x, y = np.meshgrid(nodes, nodes, indexing='ij'); weights = np.outer(mass, mass).ravel() / np.pi
        points = np.zeros((400, 48)); points[:, 0] = mean[0] + np.sqrt(2 * variance[0]) * x.ravel()
        points[:, 1] = mean[1] + np.sqrt(2 * variance[1]) * y.ravel()
        errors = np.tile(points[None], (4, 1, 1)); weights = np.tile(weights[None], (4, 1)) / 4
        lr = log_ratios(points, builder(old_theta)[0], builder(new_theta)[0])
        _, updated = posterior_mixture_weights(weights.ravel(), np.tile(lr, 4))
        limits = {'score_precision': {'maximum_split_Rhat':1.05,'minimum_functional_ESS':100,
            'maximum_scaled_95_MC_halfwidth':.001}}
        seconds, summaries, _ = risk_statistics(errors, updated.reshape(weights.shape), builder, new_theta, limits)
        prior = white + new_theta; pm = prior / (prior + noise) * observed; pv = prior * noise / (prior + noise)
        expected = np.diag(pv) + np.outer(pm, pm)
        np.testing.assert_allclose(seconds[0][:2, :2], expected, atol=1e-11)
        gradient = .5 * (1 / (prior + noise) - observed ** 2 / (prior + noise) ** 2) / 48
        np.testing.assert_allclose(summaries[0]['scaled_raw_score'], gradient * new_theta, atol=1e-11)
        limits['evidence_precision'] = {'minimum_importance_ESS': 100, 'minimum_chain_functional_ESS': 100,
            'maximum_split_Rhat': 1.05, 'maximum_split_log_ratio_difference': .1}
        _, check = fixed_update(errors, weights, builder, old_theta, new_theta, limits)
        old_prior = white + old_theta; new_prior = white + new_theta
        expected_q = .5 * np.sum(np.log(old_prior / new_prior)
            + (variance + mean ** 2) * (1 / old_prior - 1 / new_prior)) / 48
        expected_log_z = -.5 * np.sum(np.log((new_prior + noise) / (old_prior + noise))
            + observed ** 2 * (1 / (new_prior + noise) - 1 / (old_prior + noise)))
        self.assertAlmostEqual(check['full_and_split_Q_improvements'][0], expected_q, places=11)
        self.assertAlmostEqual(check['source_evidence']['log_source_evidence_ratio'], expected_log_z, places=11)

    def test_response_score_uses_the_changed_risk_posterior_weights(self):
        from scipy.linalg import block_diag
        design = np.tile(np.eye(12), (4, 1)); common = .1 * np.eye(12); task = .3 * np.eye(12)
        directions = [design @ common @ design.T, block_diag(*([task] * 4))]
        law = {'source_design': design.tolist(), 'unconditioned_common_covariance': common.tolist(),
            'task_covariance': task.tolist(), 'point_discrepancy_variance': .4,
            'source_covariance': (.4 * np.eye(48) + sum(directions)).tolist()}
        noise = np.array([.3, .4]); observed = np.array([.4, -.2])
        old_prior = np.array([.8, 1.1]); new_prior = np.array([.65, 1.])
        old_mean = old_prior / (old_prior + noise) * observed
        old_variance = old_prior * noise / (old_prior + noise)
        nodes, mass = np.polynomial.hermite.hermgauss(20)
        x, y = np.meshgrid(nodes, nodes, indexing='ij')
        points = np.column_stack([old_mean[0] + np.sqrt(2 * old_variance[0]) * x.ravel(),
            old_mean[1] + np.sqrt(2 * old_variance[1]) * y.ravel()])
        lr = log_ratios(points, np.diag(old_prior), np.diag(new_prior))
        quadrature = np.tile((np.outer(mass, mass).ravel() / np.pi)[None], (4, 1)) / 4
        _, weights = posterior_mixture_weights(quadrature.ravel(), np.tile(lr, 4))
        mapping = np.zeros((48, 2)); mapping[0, 0] = 1.; mapping[12, 1] = 1.
        ds = np.tile((points @ mapping.T)[None], (4, 1, 1))
        eta = ds.copy(); cs = np.broadcast_to(.2 * np.eye(48), (4, 400, 48, 48))
        theta = np.array([.2, .8])
        result = response_stationarity(None, eta, ds, cs, weights.reshape(4, 400), law, theta,
            {'optimizer': {'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-7},
             'convergence': {'maximum_scaled_projected_raw_score': .001,
                 'maximum_relative_covariance_change': .01}})
        new_mean = new_prior / (new_prior + noise) * observed
        new_variance = new_prior * noise / (new_prior + noise)
        second = .2 * np.eye(48) + mapping @ (np.diag(new_variance)
            + np.outer(new_mean, new_mean)) @ mapping.T
        precision = np.linalg.inv(.4 * np.eye(48) + sum(t * d for t, d in zip(theta, directions)))
        expected = np.array([.5 * (np.trace(precision @ d)
            - np.trace(precision @ d @ precision @ second)) / 48 for d in directions]) * theta
        np.testing.assert_allclose(result['scaled_projected_raw_score'], expected, atol=1e-11)


if __name__ == '__main__':
    unittest.main()
