import copy
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import qmc

from core.conditional_terminal_kg import ConditionalTerminalActions
from core.positive_mixture_lookahead import condition_positive_raw
from core.positive_raw_terminal import raw_terminal_component, raw_terminal_values
from core.positive_raw_lookahead import evaluate_raw_futures, summarize_raw_futures
from core.raw_likelihood_lookahead import objective_terminal_statistics, risk_terminal_statistics
from representation.joint_response_mixture import JointResponseMixture
from test_conditional_terminal_kg import correlated_fields
from test_positive_source_mixture import terminal


def four_policy_fields():
    pair = correlated_fields()
    for field in pair:
        field.means = np.column_stack([field.means, field.means[:, 0]+.1, np.full(len(field.means), -3.8)])
        root = np.zeros((8, 8)); root[:6, :6] = np.linalg.cholesky(field.covariances[0])
        root[6, 1] = .02; root[6, 6] = .06; root[7, 7] = .25
        field.covariances = np.tile(root @ root.T, (len(field.means), 1, 1))
    return pair


class PositiveRawTerminalTests(unittest.TestCase):
    def test_outside_query_and_direct_supports_match_historical_transport(self):
        prior = four_policy_fields(); updated = copy.deepcopy(prior); query = 3; labels = [.45, -.24]
        scores = [field.update_batch(query, value, None, 1) for field, value in zip(updated, labels)]
        current = ConditionalTerminalActions([[0, 1, 2], [1, 0, 2], [2, 0, 1], [3, 0, 1]], [1]*4, terminal().protocol)
        nodes = qmc.Sobol(7, scramble=True, seed=202610077).random_base2(16)
        values, _, actions, diagnostics = raw_terminal_component(prior, updated, current, 359, query, labels, scores, nodes)
        self.assertEqual(len(actions), 20)
        chosen = [next(i for i, a in enumerate(actions) if a['kind'] == 'guarded' and a['primary'] == 0),
            next(i for i, a in enumerate(actions) if a['kind'] == 'direct' and a['primary'] == 0 and a['budgets'] == [359]),
            next(i for i, a in enumerate(actions) if a['kind'] == 'direct' and a['primary'] == query and a['budgets'] == [359])]
        for index in chosen:
            action = actions[index]; points = list(dict.fromkeys([query]+action['shortlist']))
            full_nodes = qmc.Sobol(1+2*len(points), scramble=True, seed=202610077).random_base2(18)
            positions = [points.index(i) for i in action['shortlist']]; stats = []
            for head in (0, 1):
                draws, weights, _ = condition_positive_raw(prior[head], updated[head], query, points,
                    full_nodes, labels[head], scores[head])
                draws = draws.reshape(1, len(full_nodes), len(points), 2)[:, :, positions]
                stats.append(objective_terminal_statistics(draws, weights[None], action['incumbent_position'], current.protocol)[0]
                    if head == 0 else risk_terminal_statistics(draws, weights[None], action['incumbent_position'],
                        action['budgets'], current.protocol)[0][0])
            self.assertAlmostEqual(values[index], stats[0] @ stats[1], delta=2e-4)
        self.assertLess(max(abs(r['predictive_log_evidence_error']) for r in diagnostics), 1e-4)

    def test_shared_identity_and_completed_future_replay(self):
        model = JointResponseMixture(['A', 'B'], [correlated_fields(), correlated_fields()], prior_weights=[.2, .8])
        model.components[1][0].means[:, ::2] += .15
        model.components[1][1].means[:, ::2] += .04
        old = terminal(); current = ConditionalTerminalActions(old.shortlists, old.incumbents, old.protocol)
        labels = [[.4, -.3], [.9, -.15]]
        kwargs = {'raw_powers': [5, 6], 'projected_power': 6, 'seed': 202610077}
        records, actions = evaluate_raw_futures(model, current, 360, 0, labels, **kwargs)
        with patch('core.positive_raw_lookahead.raw_fantasy_moments', side_effect=AssertionError('repeated paid moments')):
            replay, replay_actions = evaluate_raw_futures(model, current, 360, 0, labels, completed_records=records, **kwargs)
        self.assertEqual(replay, records); self.assertEqual(replay_actions, actions)
        self.assertEqual(model.paid_episodes, 0); self.assertEqual(model.history, [])
        future = copy.deepcopy(model); score = future.update_raw(0, labels[0])['head_log_scores']
        actual, _, _, _ = raw_terminal_values(model, future, current, 359, 0, labels[0], score, power=6, seed=202610077)
        components = [raw_terminal_component(old, new, current, 359, 0, labels[0], score[i],
            qmc.Sobol(7, scramble=True, seed=202610077).random_base2(6))[0]
            for i, (old, new) in enumerate(zip(model.components, future.components))]
        np.testing.assert_allclose(actual, future.weights @ components, atol=1e-12, rtol=0.)
        rows = summarize_raw_futures(records, model.weights, 1, actions, (.5, .49, 0), [5, 6])
        for row in rows:
            self.assertGreaterEqual(row['action_selection_loss'], 0.)
            self.assertAlmostEqual(row['raw_net_reward_control_variate']-row['raw_net_reward_of_projected_choice_control_variate'],
                row['action_selection_loss'], places=12)

    def test_omitted_utility_bound_preserves_original_weights(self):
        model = JointResponseMixture(['A', 'B'], [correlated_fields(), correlated_fields()], prior_weights=[.2, .8])
        model.components[1][0].means[:, ::2] += .15
        old = terminal(); current = ConditionalTerminalActions(old.shortlists, old.incumbents, old.protocol)
        future = copy.deepcopy(model); labels = [.4, -.3]
        scores = future.update_raw(0, labels)['head_log_scores']; before = future.weights.copy()
        kwargs = {'power': 7, 'seed': 202610077}
        full, _, _, _ = raw_terminal_values(model, future, current, 359, 0, labels, scores, **kwargs)
        lower, _, _, check = raw_terminal_values(model, future, current, 359, 0, labels, scores,
            utility_identities=[1], **kwargs)
        component = raw_terminal_component(model.components[1], future.components[1], current, 359, 0,
            labels, scores[1], qmc.Sobol(7, scramble=True, seed=kwargs['seed']).random_base2(kwargs['power']))[0]
        np.testing.assert_allclose(lower, before[1]*component, atol=1e-12, rtol=0.)
        self.assertAlmostEqual(check['omitted_posterior_utility_mass'], before[0], places=14)
        np.testing.assert_array_equal(future.weights, before)
        self.assertGreater(np.max(full-lower), 0.)
        self.assertTrue(np.all(full >= lower-1e-12))
        self.assertTrue(np.all(full <= lower+before[0]+1e-12))
        full_contrast = full.max()-full[0]; bounded_contrast = lower.max()-lower[0]
        self.assertLessEqual(abs(full_contrast-bounded_contrast), before[0]+1e-12)
        # One old future per identity survives remapping; only B receives a new future.
        args = {'raw_powers': [5], 'projected_power': 5, 'seed': 202610077}
        first, actions = evaluate_raw_futures(model, current, 360, 0, [labels, [.9, -.15]], **args)
        extended, _ = evaluate_raw_futures(model, current, 360, 0,
            [labels, [.1, -.1], [.9, -.15], [.5, -.2]], completed_records=first,
            utility_identities=[1], new_origin_identities=[1], **args)
        self.assertEqual(sorted(r['position'] for r in extended), [0, 2, 3])
        for original in first:
            retained = next(r for r in extended if (r['origin_identity'], r['identity_fantasy_position']) ==
                (original['origin_identity'], original['identity_fantasy_position']))
            self.assertEqual(retained['raw'], original['raw'])
        self.assertEqual(model.paid_episodes, 0)

    def test_partial_origin_expectation_interval_contains_full_reference(self):
        actions = [{'primary': 0}, {'primary': 1}]; records = []; full_adaptation = 0.
        for origin in range(2):
            for position in range(4):
                lower = np.array([.2, .35]); actual = lower.copy()
                mass = .1 if origin == 1 and position > 0 else 0.
                if mass:
                    actual += [.07, .01]
                if origin == 0 and position > 0:
                    actual = np.array([.1, .5])
                full_adaptation += [.2, .8][origin]/4*(actual.max()-actual[0])
                if origin == 0 and position > 0:
                    continue
                records.append({'position': origin*4+position, 'origin_identity': origin,
                    'identity_fantasy_position': position, 'shared_identity_weights': [.1, .9],
                    'projected_values': [.3, .4], 'projected_action_index': 1,
                    'maximum_raw_integration_relative_error': 0.,
                    'raw': {'5': {'values': lower.tolist(), 'costs': [1., 1.],
                        'omitted_posterior_utility_mass': mass,
                        'predictive_log_evidence_max_error': 0., 'effective_sample_size_min': 16.}}})
        row = summarize_raw_futures(records, np.array([.2, .8]), 4, actions, (.3, .29, 0), [5],
            omitted_origin_identities=[0])[0]
        interval = row['raw_adaptation_identity_interval']
        self.assertLessEqual(interval[0], full_adaptation)
        self.assertGreaterEqual(interval[1], full_adaptation)
        self.assertAlmostEqual(row['omitted_predictive_origin_mass'], .15)
        self.assertAlmostEqual(row['weighted_omitted_posterior_utility_mass'], .06)
        self.assertEqual(row['omitted_joint_futures'], 3)
        self.assertNotIn('expected_model_weights', row)
        np.testing.assert_allclose(row['covered_predictive_expected_model_weights'], [.085, .765])
        with self.assertRaises(AssertionError):
            summarize_raw_futures(records, np.array([.2, .8]), 4, actions, (.3, .29, 0), [5])


if __name__ == '__main__':
    unittest.main()
