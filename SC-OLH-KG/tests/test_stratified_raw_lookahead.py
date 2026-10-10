import unittest
from unittest.mock import patch

import numpy as np

from core.conditional_terminal_kg import ConditionalTerminalActions
from core.positive_raw_lookahead import evaluate_raw_futures
from core.stratified_raw_lookahead import stratified_record_weights, summarize_stratified_futures
from representation.joint_response_mixture import JointResponseMixture
from test_conditional_terminal_kg import correlated_fields
from test_positive_source_mixture import terminal


class StratifiedRawLookaheadTests(unittest.TestCase):
    def test_original_probability_mass_and_joint_adaptive_maximum(self):
        uniforms = np.array([[.1, .1], [.2, .6], [.9, .2], [.8, .8]])
        records = []
        for identity in range(2):
            for p in range(4):
                low = uniforms[p, 1] < .25; values = [.8, .2] if low else [.2, .8]
                records.append({'origin_identity': identity, 'identity_fantasy_position': p,
                    'projected_values': values, 'projected_action_index': 0 if low else 1,
                    'shared_identity_weights': [.2, .8], 'raw': {'5': {'values': values,
                        'costs': [1., 1.], 'predictive_log_evidence_max_error': 0., 'effective_sample_size_min': 32.}}})
        selected, weights, regions = stratified_record_weights(records, [.2, .8], uniforms, [0, 1], .25, 2, 4)
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertAlmostEqual(weights[regions == 'low_constraint'].sum(), .25)
        row = summarize_stratified_futures(selected, weights, regions, [{'primary': 0}, {'primary': 1}],
            (.4, .39, 0), [5], 0.)[0]
        self.assertAlmostEqual(row['raw_adaptation_value'], .45)
        self.assertAlmostEqual(row['raw_net_reward_control_variate'], .44)
        # Averaging action values before maximizing would incorrectly return .30.
        self.assertNotAlmostEqual(row['raw_adaptation_value'], .30)
        self.assertEqual(row['action_selection_loss'], 0.)
        self.assertAlmostEqual(row['raw_net_control_variate_identity_interval'][0], .44)

    def test_rare_origin_mass_and_terminal_remainder_are_kept(self):
        uniforms = np.array([[.1, .1], [.2, .6]])
        records = []
        for identity in range(2):
            for p in range(2):
                records.append({'origin_identity': identity, 'identity_fantasy_position': p,
                    'projected_values': [.2, .5], 'projected_action_index': 1,
                    'shared_identity_weights': [.1, .9], 'raw': {'5': {'values': [.2, .5],
                        'costs': [1., 1.], 'omitted_posterior_utility_mass': .1 if identity == 0 else 0.,
                        'predictive_log_evidence_max_error': 0., 'effective_sample_size_min': 32.}}})
        selected, weights, regions = stratified_record_weights(records, [.99, .01], uniforms, [0], .25, 1, 2)
        self.assertAlmostEqual(weights[regions == 'retained_rare_origin'].sum(), .01)
        row = summarize_stratified_futures(selected, weights, regions, [{'primary': 0}, {'primary': 1}],
            (.4, .39, 0), [5], .01)[0]
        self.assertAlmostEqual(row['weighted_omitted_terminal_identity_mass'], .099)
        self.assertAlmostEqual(row['identity_and_rare_origin_error_bound'], .109)
        interval = row['raw_net_control_variate_identity_interval']
        self.assertLessEqual(interval[0], .29)
        self.assertGreaterEqual(interval[1], .29)

    def test_sparse_coordinates_reuse_paid_futures(self):
        model = JointResponseMixture(['A', 'B'], [correlated_fields(), correlated_fields()], prior_weights=[.2, .8])
        old = terminal(); current = ConditionalTerminalActions(old.shortlists, old.incumbents, old.protocol)
        args = {'raw_powers': [5], 'projected_power': 5, 'seed': 202610077}
        records, actions = evaluate_raw_futures(model, current, 360, 0, [[.4, -.3], [.9, -.15]], **args)
        extended, _ = evaluate_raw_futures(model, current, 360, 0, [[.6, -.2]],
            fantasy_coordinates=[(1, 7)], completed_records=records, **args)
        self.assertEqual(sorted((r['origin_identity'], r['identity_fantasy_position']) for r in extended), [(0, 0), (1, 0), (1, 7)])
        self.assertEqual(extended[0]['raw'], records[0]['raw'])
        with patch('core.positive_raw_lookahead.raw_fantasy_moments', side_effect=AssertionError('repeated paid moments')):
            replay, _ = evaluate_raw_futures(model, current, 360, 0, [[.6, -.2]],
                fantasy_coordinates=[(1, 7)], completed_records=extended, **args)
        self.assertEqual(replay, extended)
        self.assertEqual(model.paid_episodes, 0)


if __name__ == '__main__':
    unittest.main()
