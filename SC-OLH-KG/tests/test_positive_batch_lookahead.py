import copy
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import digamma, polygamma
from scipy.stats import qmc

from core.conditional_terminal_kg import ConditionalTerminalActions
from core.positive_mixture_lookahead import (
    condition_positive_raw, predictive_positive_batches, predictive_values)
from core.positive_raw_lookahead import evaluate_raw_futures, summarize_raw_futures
from core.positive_raw_terminal import raw_terminal_values, raw_terminal_component
from core.raw_likelihood_lookahead import objective_terminal_statistics, risk_terminal_statistics
from core.raw_terminal_head_grid import evaluate_head_records, compose_head_grid, summarize_achievable_grid
from representation.joint_response_mixture import JointResponseMixture
from representation.vectorized_joint_function_risk import raw_fantasy_moments, apply_raw_moments
from test_conditional_terminal_kg import correlated_fields
from test_positive_source_mixture import terminal
from test_positive_raw_terminal import four_policy_fields
from performance.historical_batch_head_precision import action_key
from performance.historical_full_bank_development import terminal_rule


def model():
    second = correlated_fields()
    second[0].means[:, ::2] += .15
    second[1].means[:, ::2] += .08
    return JointResponseMixture(['A', 'B'], [correlated_fields(), second], prior_weights=[.3, .7])


class PositiveBatchLookaheadTests(unittest.TestCase):
    def test_native_joint_grid_matches_actual_crossed_paid_batches(self):
        current = model(); query = 1
        batches = [np.array([[.7, -.22], [.78, -.15], [.64, -.31], [.73, -.24]]),
                   np.array([[.3, -.48], [.37, -.4], [.24, -.53], [.33, -.45]])]
        labels = np.asarray([b.mean(axis=0) for b in batches])
        eta = np.asarray([np.log(b.var(axis=0, ddof=1))-digamma(1.5)+np.log(1.5) for b in batches])
        rule = ConditionalTerminalActions([[0, 1, 2], [1, 0, 2]], [1, 1], terminal().protocol)
        heads = [evaluate_head_records(current, rule, 360, query, h, labels[:, h], [0, 1],
            powers=[6], projected_powers=[6, 7], seed=80656, replicates=4, eta_labels=eta[:, h],
            guarded_mean_position=0) for h in (0, 1)]
        projected = [np.asarray([r['projected']['7']['statistics'] for r in head]) for head in heads]
        costs = np.asarray([r['projected']['7']['costs'] for r in heads[1]])
        values, costs, weights, _ = compose_head_grid(*projected, costs,
            *[[r['head_log_scores'] for r in head] for head in heads], current.weights)
        for f in range(2):
            for g in range(2):
                actual = copy.deepcopy(current)
                actual.update_replicates(query, np.column_stack([batches[f][:, 0], batches[g][:, 1]]))
                reference = terminal_rule(actual, rule, 356, [0, 1], 7, 80656)
                np.testing.assert_allclose(values[f, g], reference['values'], atol=2e-13, rtol=0.)
                np.testing.assert_allclose(costs[f, g], reference['costs'], atol=2e-11, rtol=0.)
                np.testing.assert_allclose(weights[f, g], actual.weights, atol=2e-13, rtol=0.)
                self.assertEqual(int(np.lexsort((costs[f, g], -values[f, g]))[0]), reference['best_action_index'])
        with patch('core.raw_terminal_head_grid.raw_terminal_head', side_effect=AssertionError('old raw cache recomputed')):
            extended = evaluate_head_records(current, rule, 360, query, 0, labels[:, 0], [0, 1],
                powers=[6], projected_powers=[6, 7, 8], completed=heads[0], seed=80656, replicates=4,
                eta_labels=eta[:, 0], guarded_mean_position=0)
        self.assertEqual(extended[0]['raw'], heads[0][0]['raw'])
        self.assertEqual(current.paid_episodes, 0)

    def test_achievable_kg_keeps_negative_native_switch_and_budget_price(self):
        raw = np.array([[[.6, .2], [.6, .8]]])
        projected = np.array([[[.5, .9], [.5, .9]]])
        costs = np.full_like(raw, 356.)
        weights = np.array([[.25, .75]])
        row = summarize_achievable_grid(raw, costs, projected, costs, np.ones((1, 2, 1)),
            weights, (.62, .6, 0), 7, 3)
        self.assertAlmostEqual(row['raw_adaptation_value'], .15)
        self.assertAlmostEqual(row['raw_adaptation_of_projected_choice'], .05)
        self.assertAlmostEqual(row['raw_net_reward_of_projected_choice_control_variate'], .03)
        self.assertAlmostEqual(row['action_selection_loss'], .1)
        np.testing.assert_allclose(row['weighted_achievable_contrasts'], [[-.4, .2]], atol=1e-15, rtol=0.)
        failed = summarize_achievable_grid(raw[:, :1], costs[:, :1], projected[:, :1], costs[:, :1],
            np.ones((1, 1, 1)), np.ones((1, 1)), (.62, .6, 0), 7, 2)
        self.assertAlmostEqual(failed['raw_net_reward_of_projected_choice_control_variate'], -.42)

    def test_native_policy_shards_keep_full_shared_mixture_and_budget(self):
        current = model(); observations = np.array([[.7, -.22], [.78, -.15], [.64, -.31], [.73, -.24]])
        current.update_replicates(1, observations)
        rule = ConditionalTerminalActions([[0, 1, 2], [1, 0, 2]], [1, 1], terminal().protocol)
        identities = [0, 1]
        full = terminal_rule(current, rule, 356, identities, 7, 80655)
        positions = {action_key(a): i for i,a in enumerate(full['actions'])}
        values = np.empty(len(positions)); costs = np.empty(len(positions)); covered = []
        for shard in range(2):
            indices = np.arange(shard, len(rule.shortlists), 2)
            part = ConditionalTerminalActions(rule.shortlists[indices], rule.incumbents[indices], rule.protocol)
            row = terminal_rule(current, part, 356, identities, 7, 80655)
            selected = [positions[action_key(a)] for a in row['actions']]
            values[selected] = row['values']; costs[selected] = row['costs']; covered.extend(selected)
        self.assertEqual(sorted(covered), list(range(len(positions))))
        np.testing.assert_allclose(values, full['values'], atol=2e-13, rtol=0.)
        np.testing.assert_allclose(costs, full['costs'], atol=2e-11, rtol=0.)
        self.assertTrue(all(sum(a['budgets'])+(16 if a['incumbent_position'] else 0) <= 356
            for a in full['actions']))

    def test_policy_shards_preserve_every_native_batch_action(self):
        current = model(); query = 1
        observations = np.array([[.7, -.22], [.78, -.15], [.64, -.31], [.73, -.24]])
        labels = observations.mean(axis=0)
        eta = np.log(observations.var(axis=0, ddof=1))-digamma(1.5)+np.log(1.5)
        rule = ConditionalTerminalActions([[0, 1, 2], [1, 0, 2]], [1, 1], terminal().protocol)
        future = copy.deepcopy(current); scores = future.update_replicates(query, observations)['head_log_scores']
        expected, costs, actions, _ = raw_terminal_values(current, future, rule, 356, query, labels, scores,
            power=7, seed=80654, replicates=4, eta_labels=eta, guarded_mean_position=0)
        positions = {action_key(a): i for i,a in enumerate(actions)}
        heads = []
        for head in (0, 1):
            statistics = np.empty((len(current.components), len(actions), 4)); gc = np.empty(statistics.shape[:2])
            covered = []
            for shard in range(2):
                indices = np.arange(shard, len(rule.shortlists), 2)
                partial = ConditionalTerminalActions(rule.shortlists[indices], rule.incumbents[indices], rule.protocol)
                bank = []
                rows = evaluate_head_records(current, partial, 360, query, head, [labels[head]], [0],
                    powers=[7], seed=80654, replicates=4, eta_labels=[eta[head]], guarded_mean_position=0,
                    checkpoint=lambda r,a: bank.extend(a))
                selected = [positions[action_key(a)] for a in bank]; raw = rows[0]['raw']['7']
                statistics[:, selected] = raw['statistics']; gc[:, selected] = raw['costs']; covered.extend(selected)
            self.assertEqual(sorted(covered), list(range(len(actions))))
            heads.append((statistics, gc, rows[0]['head_log_scores']))
        actual, actual_costs, weights, _ = compose_head_grid(heads[0][0][None], heads[1][0][None],
            heads[1][1][None], [heads[0][2]], [heads[1][2]], current.weights)
        np.testing.assert_allclose(actual[0, 0], expected, atol=2e-13, rtol=0.)
        np.testing.assert_allclose(actual_costs[0, 0], costs, atol=2e-11, rtol=0.)
        np.testing.assert_allclose(weights[0, 0], future.weights, atol=2e-13, rtol=0.)

    def test_predictive_batch_law_retains_component_and_shared_identity_dependence(self):
        current = model(); count = 4; query = 1
        # Give source components different variance levels as well as means.
        for pair in current.components:
            for field in pair:
                field.means[:, 1::2] += np.arange(len(field.means))[:, None]*.8
        values, eta, weights = predictive_positive_batches(current, query, count, 15, 80647)
        expectation = np.zeros(2); second = np.zeros(2); em = np.zeros(2); ev = np.zeros(2)
        cross = 0.; within_cross = np.zeros(2)
        for mass, pair in zip(current.weights, current.components):
            head_means = []
            for h, field in enumerate(pair):
                w = np.exp(field.log_weights); mu = field.means[:, 2*query]; e = field.means[:, 2*query+1]
                covariance = field.covariances
                head_means.append(w @ mu)
                expectation[h] += mass*(w @ mu)
                second[h] += mass*(w @ (mu**2+covariance[:, 2*query, 2*query]
                    +np.exp(e+covariance[:, 2*query+1, 2*query+1]/2)/count))
                em[h] += mass*(w @ e)
                ev[h] += mass*(w @ (e**2+covariance[:, 2*query+1, 2*query+1]+polygamma(1, (count-1)/2)))
                within_cross[h] += mass*(w @ (mu*e+covariance[:, 2*query, 2*query+1]))
            cross += mass*np.prod(head_means)
        np.testing.assert_allclose(weights @ values, expectation, atol=.001, rtol=0.)
        np.testing.assert_allclose(weights @ values**2, second, atol=.003, rtol=0.)
        np.testing.assert_allclose(weights @ eta, em, atol=.002, rtol=0.)
        np.testing.assert_allclose(weights @ eta**2, ev, atol=.025, rtol=0.)
        np.testing.assert_allclose(weights @ (values*eta), within_cross, atol=.004, rtol=0.)
        self.assertAlmostEqual(weights @ (values[:, 0]*values[:, 1]), cross, delta=.001)
        self.assertAlmostEqual(weights.sum(), 1., places=14)

    def test_single_episode_predictive_path_is_unchanged(self):
        current = model()
        means, eta, weights = predictive_positive_batches(current, 1, 1, 2, 80648)
        expected, ew = predictive_values(current, 1, 2, 80648)
        self.assertIsNone(eta)
        np.testing.assert_array_equal(means, expected)
        np.testing.assert_array_equal(weights, ew)

    def test_cached_batch_completion_matches_actual_paid_observations(self):
        current = model(); query = 1
        current.update_raw(0, [.45, -.2])
        batches = [np.array([[.7, -.22], [.78, -.15], [.64, -.31], [.73, -.24]]),
                   np.array([[.3, -.48], [.37, -.4], [.24, -.53], [.33, -.45]])]
        means = np.asarray([v.mean(axis=0) for v in batches]); k = 1.5
        etas = np.asarray([np.log(v.var(axis=0, ddof=1))-digamma(k)+np.log(k) for v in batches])
        cache = [[raw_fantasy_moments(field, query, means[:, h], etas[:, h], 4)
                  for h, field in enumerate(pair)] for pair in current.components]
        for p, observations in enumerate(batches):
            actual = copy.deepcopy(current); actual.update_replicates(query, observations)
            cached = copy.deepcopy(current); scores = []
            for identity, pair in enumerate(cached.components):
                scores.append([apply_raw_moments(field, query, cache[identity][h][0][p],
                    cache[identity][h][1], replicates=4) for h, field in enumerate(pair)])
            cached._record_scores(query, scores, 4)
            np.testing.assert_allclose(cached.weights, actual.weights, atol=2e-14, rtol=0.)
            for pair, reference in zip(cached.components, actual.components):
                for field, rf in zip(pair, reference):
                    np.testing.assert_allclose(field.means, rf.means, atol=2e-14, rtol=0.)
                    np.testing.assert_allclose(field.covariances, rf.covariances, atol=2e-14, rtol=0.)
                    np.testing.assert_allclose(field.log_weights, rf.log_weights, atol=2e-14, rtol=0.)
                    self.assertEqual(field.paid_episodes, 5)
                    self.assertEqual(field.paid_batches, 2)

    def test_raw_batch_transport_matches_full_field_posterior_moments(self):
        prior = correlated_fields()[1]; updated = copy.deepcopy(prior)
        observations = np.array([-.12, -.27, -.2, -.24]); count = len(observations); k = (count-1)/2
        eta = np.log(observations.var(ddof=1))-digamma(k)+np.log(k)
        score = updated.update_batch(1, observations.mean(), eta, count)
        nodes = qmc.Sobol(7, scramble=True, seed=80649).random_base2(17)
        draws, weights, evidence = condition_positive_raw(prior, updated, 1, [1, 0, 2], nodes,
            observations.mean(), score, replicates=count, eta_label=eta)
        reordered = draws[:, [2, 3, 0, 1, 4, 5]]
        mean = weights @ reordered; centered = reordered-mean
        expected, covariance = updated.predict(full_cov=True)
        np.testing.assert_allclose(mean, expected.reshape(-1), atol=2e-4, rtol=0.)
        np.testing.assert_allclose(centered.T @ (weights[:, None]*centered), covariance.reshape(6, 6), atol=4e-4, rtol=0.)
        self.assertLess(abs(evidence), 3e-4)

    def test_batch_futures_use_native_updates_full_terminal_and_physical_debit(self):
        current = model(); current.update_raw(0, [.45, -.2]); original = current.predict_joint()
        old = terminal(); rule = ConditionalTerminalActions(old.shortlists, old.incumbents, old.protocol)
        batches = [np.array([[.7, -.22], [.78, -.15], [.64, -.31], [.73, -.24]]),
                   np.array([[.3, -.48], [.37, -.4], [.24, -.53], [.33, -.45]])]
        means = np.asarray([v.mean(axis=0) for v in batches]); k = 1.5
        etas = np.asarray([np.log(v.var(axis=0, ddof=1))-digamma(k)+np.log(k) for v in batches])
        args = dict(raw_powers=[6, 7], projected_power=7, seed=80650, replicates=4, eta_labels=etas,
                    guarded_mean_position=0)
        records, actions = evaluate_raw_futures(current, rule, 360, 1, means, **args)
        for p, observations in enumerate(batches):
            future = copy.deepcopy(current); score = future.update_replicates(1, observations)['head_log_scores']
            expected, costs, expected_actions = future.terminal_values(rule, 356, 7, args['seed'])
            actual, _, _, _ = raw_terminal_values(current, future, rule, 356, 1, means[p], score,
                power=7, seed=args['seed'], replicates=4, eta_labels=etas[p],guarded_mean_position=0)
            self.assertEqual(actions, expected_actions)
            np.testing.assert_allclose(records[p]['projected_values'], expected, atol=2e-13, rtol=0.)
            np.testing.assert_allclose(records[p]['raw']['7']['values'], actual, atol=2e-13, rtol=0.)
            self.assertEqual(records[p]['future_remaining_calls'], 356)
            self.assertTrue(all(sum(a['budgets'])+(16 if a['incumbent_position'] else 0) <= 356 for a in actions))
        with patch('core.positive_raw_lookahead.raw_fantasy_moments', side_effect=AssertionError('duplicate integral')):
            replay, _ = evaluate_raw_futures(current, rule, 360, 1, means, completed_records=records, **args)
        self.assertEqual(replay, records)
        rows = summarize_raw_futures(records, current.weights, 1, actions, (.6, .59, 0), [6, 7])
        for row in rows:
            self.assertAlmostEqual(row['raw_net_reward_control_variate']-
                row['raw_net_reward_of_projected_choice_control_variate'], row['action_selection_loss'], places=13)
        self.assertEqual(current.paid_episodes, 1)
        for before, after in zip(original, current.predict_joint()):
            np.testing.assert_array_equal(before, after)

    def test_cached_head_grid_composes_every_batch_pair_with_native_joint_identity(self):
        current = model(); old = terminal()
        rule = ConditionalTerminalActions(old.shortlists, old.incumbents, old.protocol)
        batches = [np.array([[.7, -.22], [.78, -.15], [.64, -.31], [.73, -.24]]),
                   np.array([[.3, -.48], [.37, -.4], [.24, -.53], [.33, -.45]])]
        heads = []
        for h in (0, 1):
            labels = {p: v[:, h].mean() for p, v in enumerate(batches)}
            eta = {p: np.log(v[:, h].var(ddof=1))-digamma(1.5)+np.log(1.5) for p, v in enumerate(batches)}
            records = evaluate_head_records(current, rule, 360, 1, h, labels, [0, 1], powers=[7], seed=80651,
                guarded_mean_position=0, replicates=4, eta_labels=eta)
            with patch('core.raw_terminal_head_grid.raw_fantasy_moments', side_effect=AssertionError('repeated batch')):
                replay = evaluate_head_records(current, rule, 360, 1, h, labels, [0, 1], powers=[7], seed=80651,
                    guarded_mean_position=0, replicates=4, eta_labels=eta, completed=records)
            self.assertEqual(replay, records)
            self.assertTrue(all(r['future_remaining_calls'] == 356 for r in records))
            heads.append(records)
        f, g = heads
        values, costs, weights, _ = compose_head_grid(
            [r['raw']['7']['statistics'] for r in f], [r['raw']['7']['statistics'] for r in g],
            [r['raw']['7']['costs'] for r in g], [r['head_log_scores'] for r in f],
            [r['head_log_scores'] for r in g], current.weights)
        for i in range(2):
            for j in range(2):
                observations = np.column_stack([batches[i][:, 0], batches[j][:, 1]])
                future = copy.deepcopy(current); score = future.update_replicates(1, observations)['head_log_scores']
                actual, actual_cost, _, _ = raw_terminal_values(current, future, rule, 356, 1,
                    observations.mean(axis=0), score, power=7, seed=80651, guarded_mean_position=0,
                    replicates=4, eta_labels=[f[i]['eta_label'], g[j]['eta_label']])
                np.testing.assert_allclose(values[i, j], actual, atol=2e-13, rtol=0.)
                np.testing.assert_allclose(costs[i, j], actual_cost, atol=2e-12, rtol=0.)
                np.testing.assert_allclose(weights[i, j], future.weights, atol=2e-14, rtol=0.)
        self.assertEqual(current.paid_episodes, 0)

    def test_projected_utility_subset_keeps_original_mass_and_all_paid_fields(self):
        current = model(); old = terminal()
        rule = ConditionalTerminalActions(old.shortlists, old.incumbents, old.protocol)
        observations = np.array([[.7, -.22], [.78, -.15], [.64, -.31], [.73, -.24]])
        future = copy.deepcopy(current); future.update_replicates(1, observations)
        complete, full_costs, actions = future.terminal_values(rule, 356, 7, 80652)
        lower, lower_costs, same_actions = future.terminal_values(rule, 356, 7, 80652, utility_identities=[1])
        native, native_costs, _ = rule.evaluate_posterior(*future.components[1], 356, 7, 80652)
        self.assertEqual(actions, same_actions)
        np.testing.assert_allclose(lower, future.weights[1]*native, atol=1e-14, rtol=0.)
        np.testing.assert_allclose(lower_costs, future.weights[1]*native_costs, atol=1e-12, rtol=0.)
        self.assertTrue(np.all(complete >= lower-1e-14))
        self.assertTrue(np.all(complete <= lower+future.weights[0]+1e-14))
        self.assertTrue(np.all(full_costs <= lower_costs+356*future.weights[0]+1e-12))
        eta = np.log(observations.var(axis=0, ddof=1))-digamma(1.5)+np.log(1.5)
        records, _ = evaluate_raw_futures(current, rule, 360, 1, [observations.mean(axis=0)],
            raw_powers=[7], projected_power=7, seed=80652, replicates=4, eta_labels=[eta],
            fantasy_coordinates=[(0, 0)], utility_identities=[1], projected_utility_identities=[1])
        np.testing.assert_allclose(records[0]['projected_values'], lower, atol=2e-13, rtol=0.)
        self.assertAlmostEqual(records[0]['omitted_projected_utility_mass'], future.weights[0], places=14)
        self.assertTrue(all(f.paid_episodes == g.paid_episodes == 4 for f, g in future.components))

    def test_batch_conditional_terminal_matches_independent_full_raw_transport(self):
        prior = four_policy_fields(); future = copy.deepcopy(prior); query = 3; count = 4
        observations = np.array([[.45, -.24], [.5, -.3], [.39, -.27], [.48, -.18]])
        labels = observations.mean(axis=0)
        eta = np.log(observations.var(axis=0, ddof=1))-digamma(1.5)+np.log(1.5)
        scores = [field.update_batch(query, labels[h], eta[h], count) for h, field in enumerate(future)]
        rule = ConditionalTerminalActions([[0, 1, 2], [3, 0, 1]], [1, 1], terminal().protocol)
        nodes = qmc.Sobol(7, scramble=True, seed=80653).random_base2(16)
        values, _, actions, _ = raw_terminal_component(prior, future, rule, 356, query, labels, scores, nodes,
            replicates=count, eta_labels=eta, guarded_mean_position=0)
        incumbent_values, _, incumbent_actions, _ = raw_terminal_component(prior, future, rule, 356,
            query, labels, scores, nodes, replicates=count, eta_labels=eta, guarded_mean_position=1)
        self.assertEqual(actions, incumbent_actions)
        chosen = [i for i, a in enumerate(actions) if
            (a['kind'] == 'guarded' and a['primary'] == 0 and i < 2) or
            (a['kind'] == 'direct' and a['budgets'] == [356])]
        for i in chosen:
            action = actions[i]; points = list(dict.fromkeys([query]+action['shortlist']))
            positions = [points.index(p) for p in action['shortlist']]
            reference_nodes = qmc.Sobol(1+2*len(points), scramble=True, seed=80653).random_base2(18)
            stats = []
            for h in (0, 1):
                draws, weights, _ = condition_positive_raw(prior[h], future[h], query, points,
                    reference_nodes, labels[h], scores[h], replicates=count, eta_label=eta[h])
                draws = draws.reshape(1, len(draws), len(points), 2)[:, :, positions]
                stats.append(objective_terminal_statistics(draws, weights[None], action['incumbent_position'], rule.protocol)[0]
                    if h == 0 else risk_terminal_statistics(draws, weights[None], action['incumbent_position'],
                        action['budgets'], rule.protocol)[0][0])
            self.assertAlmostEqual(values[i], stats[0] @ stats[1], delta=2e-4)
            self.assertAlmostEqual(incumbent_values[i], stats[0] @ stats[1], delta=2e-4)


if __name__ == '__main__':
    unittest.main()
