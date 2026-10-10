import unittest
from itertools import product

import numpy as np

from performance.probe_historical_source_risk_plateau import ratio_summary
from performance.check_legacy_f_risk_amplitude_importance import log_ratios
from performance.retain_historical_stationary_source_heads import retain_stationary
from performance.replay_historical_source_learning_round2 import assemble_source_precision
from performance.compare_historical_source_acceleration import raw_source_log_ratios
from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior


class SourceRiskContinuationTests(unittest.TestCase):
    def test_acceleration_ratio_includes_actual_response_and_risk_covariance_changes(self):
        from scipy.stats import multivariate_normal
        mean = np.array([.1, -.2]); labels = np.array([.6, -.3]); s2 = np.array([.03, .05])
        old_f = np.array([[.3, .05], [.05, .2]]); new_f = np.array([[.4, .08], [.08, .25]])
        risk_mean = np.array([-2., -1.5]); old_risk = np.array([[.6, .1], [.1, .4]])
        new_risk = np.array([[.4, .07], [.07, .35]])
        states = np.array([[-3., -2.], [-1., -.4], [-2.2, -1.3]])
        old = CollapsedRawSourcePosterior(mean, old_f, labels, s2)
        new = CollapsedRawSourcePosterior(mean, new_f, labels, s2)
        actual = raw_source_log_ratios(old, new, states, risk_mean, old_risk, new_risk)
        expected = []
        for eta in states:
            # The unchanged chi-square term cancels;both Gaussian density determinants remain.
            def joint(fc, rc):
                return multivariate_normal.logpdf(labels, mean=mean, cov=fc + np.diag(np.exp(eta) / 3)) \
                    + multivariate_normal.logpdf(eta, mean=risk_mean, cov=rc)
            expected.append(joint(new_f, new_risk) - joint(old_f, old_risk))
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_same_law_certificate_selects_its_actual_weighted_components(self):
        names = ('layout', 'selector', 'support', 'treatment', 'head')
        identities = list(product(('5', '6'), ('additive', 'spectral'),
            ('head_specific', 'joint'), ('observed_task', 'deconvolved_task'), ('f', 'g')))
        post = [{'monte_carlo_gate_passed': True, 'maximum_split_rhat': 1.,
            'minimum_autocorrelation_ess': 400.,
            'maximum_first_second_half_f_variance_relative_difference': .05,
            'candidate_component_file': f'current_weighted_{i}.json', **dict(zip(names, values))}
            for i, values in enumerate(identities)]
        cases = [{**r, 'source_Mstep_pair_passed': True, 'evidence_ratio_precision_passed': True,
            'log_source_evidence_ratio_new_vs_previous': .02} for r in post]
        # A fresh mixture missed the tail;the certificate validates the separately
        # reweighted current-law components,not this failed fresh mixture.
        failed = {**post[0], 'candidate_component_file': 'failed_fresh.json',
            'monte_carlo_gate_passed': False, 'point_mixture_retention_passed': False}
        repair = {**post[0], 'functional_precision_passed': True}
        rows = assemble_source_precision({'all_registered_cases_complete': True, 'evidence_rows': cases},
            {'rows': post}, {'all_registered_cases_complete': True, 'rows': [failed]},
            {'all_registered_cases_complete': True, 'evidence_rows': [], 'functional_repairs': [repair]},
            'learning.json', 'same_law_certificate.json')
        self.assertTrue(rows[0]['positive_mixture_precision_passed'])
        self.assertEqual(rows[0]['positive_mixture_component_file'], 'current_weighted_0.json')
        self.assertEqual(sum(r['positive_mixture_precision_passed'] for r in rows), 32)

    def test_weighted_prior_ratio_matches_analytic_source_evidence(self):
        # A nonuniform Gaussian posterior quadrature catches missing prior determinants
        # or an accidental replacement of positive weights by uniform weights.
        noise, observed, new = .3, .7, .6
        nodes, weights = np.polynomial.hermite.hermgauss(64)
        points = observed / (1 + noise) + np.sqrt(2 * noise / (1 + noise)) * nodes
        ratios = log_ratios(points[:, None], np.eye(1), np.array([[new]]))
        values = np.tile(ratios, (4, 1)); weights = np.tile(weights / np.sqrt(np.pi), (4, 1)) / 4
        limits = {'minimum_importance_ESS': 100, 'minimum_chain_functional_ESS': 100,
            'maximum_split_Rhat': 1.05, 'maximum_split_log_ratio_difference': .1}
        result = ratio_summary(values, weights, limits)
        analytic = -.5 * (np.log((new + noise) / (1 + noise))
            + observed ** 2 * (1 / (new + noise) - 1 / (1 + noise)))
        self.assertAlmostEqual(result['log_source_evidence_ratio'], analytic, places=10)

    def test_stationary_hold_is_exact_and_preserves_failed_proposal(self):
        row = {'head': 'f', 'source_Mstep_gate_passed': False,
            'source_stationarity_diagnostic_passed': True, 'consecutive_source_stationarity_iterations': 2,
            'old_common_task_covariance_scales': [.2, .8], 'common_task_covariance_scales': [.201, .799],
            'old_expected_complete_NLL_per_batch': -1., 'learned_expected_complete_NLL_per_batch': -1.0001,
            'full_and_split_Q_improvements': [1e-8, -1e-9, 2e-8], 'relative_source_covariance_change': .001,
            'scaled_projected_raw_score_at_old_parameters': [2e-5, 8e-5],
            'projected_Q_gradient_at_new_parameters': [1e-8, 2e-8]}
        selected = retain_stationary(row, 2)
        self.assertEqual(selected['common_task_covariance_scales'], row['old_common_task_covariance_scales'])
        self.assertEqual(selected['full_and_split_Q_improvements'], [0., 0., 0.])
        self.assertEqual(selected['proposed_full_and_split_Q_improvements'], row['full_and_split_Q_improvements'])
        self.assertFalse(selected['source_Mstep_gate_passed'])
        self.assertTrue(selected['source_update_accepted'])
        self.assertEqual(row['common_task_covariance_scales'], [.201, .799])
        np.testing.assert_allclose(selected['projected_Q_gradient_at_new_parameters'], [1e-4, 1e-4])
        self.assertEqual(selected['proposed_projected_Q_gradient_at_new_parameters'], [1e-8, 2e-8])
        for field, value in [('consecutive_source_stationarity_iterations', 1), ('head', 'eta_f')]:
            with self.assertRaises(AssertionError):
                retain_stationary({**row, field: value}, 2)


if __name__ == '__main__':
    unittest.main()
