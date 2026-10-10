import copy
import unittest
import threading
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import qmc

from core.conditional_terminal_kg import ConditionalTerminalActions
from core.positive_raw_terminal import raw_terminal_component, raw_terminal_values
from core.raw_terminal_head_grid import raw_terminal_head, evaluate_head_records, compose_head_grid, summarize_head_grid, summarize_origin_head_grid, summarize_origin_grid_mixture, summarize_proposal_head_grid
from performance.run_legacy_joint_population_matched_baselines import paired_fields
from representation.joint_response_mixture import JointResponseMixture
import performance.check_historical_remaining_queries as remaining_queries


class RawHeadGridTests(unittest.TestCase):
    def test_bounded_utilities_preserve_all_identity_likelihoods_and_retained_rules(self):
        components=[]
        for shift in (0., .07):
            heads=[(np.full(3,value+shift),np.eye(3)*.02+.003) for value in (1.,-.2,-4.,-4.)]
            components.append(paired_fields(heads))
        model=JointResponseMixture(['first','second'],components)
        protocol={'objective_reference':2.,'chance_alpha':.05,'familywise_delta':.05,
            'comparison_budget':8,'comparison_delta':.05}
        terminal=ConditionalTerminalActions([[1,0,2]],[1],protocol)
        for head,label in ((0,1.1),(1,-.13)):
            args=(model,terminal,360,0,head,{0:label},[0])
            full=evaluate_head_records(*args,powers=[5],projected_powers=[5],seed=714,guarded_mean_position=0)[0]
            bounded=evaluate_head_records(*args,powers=[5],projected_powers=[5],seed=714,
                guarded_mean_position=0,utility_identities=[0])[0]
            self.assertEqual(full['head_log_scores'],bounded['head_log_scores'])
            for kind in ('raw','projected'):
                for key in ('statistics','costs'):
                    np.testing.assert_array_equal(full[kind]['5'][key][0],bounded[kind]['5'][key][0])
                    self.assertEqual(np.max(abs(np.asarray(bounded[kind]['5'][key][1]))),0.)
        self.assertEqual(model.paid_episodes,0)

    def test_parallel_native_anchors_match_serial_and_keep_both_identities(self):
        heads = [(np.full(3, value), np.eye(3)*.02) for value in (1., -.2, -4., -4.)]
        model = JointResponseMixture(['first', 'second'], [paired_fields(heads), paired_fields(heads)])
        protocol = {'objective_reference': 2., 'chance_alpha': .05, 'familywise_delta': .05,
            'comparison_budget': 8, 'comparison_delta': .05}
        terminal = ConditionalTerminalActions([[1, 0, 2]], [1], protocol)
        actions = []
        records = evaluate_head_records(model, terminal, 360, 0, 1, {0: -.13, 1: -.24}, [0, 1],
            powers=[4, 5], seed=714, guarded_mean_position=0,
            checkpoint=lambda rows, a: actions.extend(a) if not actions else None)
        spec = dict(anchor_conditional_power=6, anchor_full_joint_power=8, maximum_anchor_branch_error=1.,
            seed=714, extended_anchor_conditional_power=7, extended_anchor_full_joint_power=9)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); results = []
            for workers in (1, 2):
                directory = root/str(workers); directory.mkdir()
                (directory/'head1.json').write_text(json.dumps({'records': records, 'actions': actions}))
                with patch.object(remaining_queries, 'ROOT', root), \
                     patch.object(remaining_queries, 'read', side_effect=lambda p: json.loads((root/p).read_text())):
                    results.append(remaining_queries.validate_anchors(model, {'terminal_protocol': protocol},
                        360, [.2, .19, 0], directory, [0, 1], 0, [4, 5], spec, workers=workers))
                self.assertEqual(len(json.loads((directory/'anchors.json').read_text())['rows']), 2)
            self.assertEqual(results[0], results[1])

    def test_parallel_native_labels_match_serial_and_preserve_cached_powers(self):
        heads = [(np.full(3, value), np.eye(3)*.02) for value in (1., -.2, -4., -4.)]
        model = JointResponseMixture(['test'], [paired_fields(heads)])
        protocol = {'objective_reference': 2., 'chance_alpha': .05, 'familywise_delta': .05,
            'comparison_budget': 8, 'comparison_delta': .05}
        terminal = ConditionalTerminalActions([[1, 0, 2]], [1], protocol)
        args = (model, terminal, 360, 0, 1, {0: -.13, 1: -.24}, [0, 1])
        old = evaluate_head_records(*args, powers=[4], seed=714, guarded_mean_position=0)
        snapshot = copy.deepcopy(old)
        serial = evaluate_head_records(*args, powers=[4, 5], seed=714, guarded_mean_position=0, completed=old)
        callbacks = []
        parallel = evaluate_head_records(*args, powers=[4, 5], seed=714, guarded_mean_position=0,
            completed=old, workers=2, checkpoint=lambda *args: callbacks.append(threading.get_ident()))
        self.assertEqual(old, snapshot)
        self.assertEqual(callbacks, [threading.get_ident()]*2)
        for a, b, before in zip(serial, parallel, snapshot):
            self.assertEqual({k: v for k, v in a.items() if k != 'cpu_seconds'},
                {k: v for k, v in b.items() if k != 'cpu_seconds'})
            self.assertEqual(b['raw']['4'], before['raw']['4'])

    def test_actual_recommendation_uses_the_same_joint_head_law(self):
        components = []
        for shift in (0., .04):
            heads = [(np.full(4, value+shift), np.eye(4)*.02+.003) for value in (1., -.2, -4., -4.)]
            components.append(paired_fields(heads))
        model = JointResponseMixture(['first', 'second'], components); model.log_weights = np.log([.3, .7])
        labels = [1.1, -.16]; updated = copy.deepcopy(model); scores = updated.update_raw(0, labels)['head_log_scores']
        protocol = {'objective_reference': 2., 'chance_alpha': .05, 'familywise_delta': .05,
            'comparison_budget': 8, 'comparison_delta': .05}
        terminal = ConditionalTerminalActions([[1, 2, 3]], [1], protocol)
        nodes = qmc.Sobol(7, scramble=True, seed=714).random_base2(8)
        f, g, costs = [], [], []
        for identity in range(2):
            heads = [raw_terminal_head(model.components[identity][h], updated.components[identity][h], terminal,
                359, 0, labels[h], scores[identity][h], nodes, h, guarded_mean_position=0) for h in (0, 1)]
            f.append(heads[0][0]); g.append(heads[1][0]); costs.append(heads[1][1])
        values, cost, posterior, _ = compose_head_grid([f], [g], [costs],
            [[s[0] for s in scores]], [[s[1] for s in scores]], model.weights)
        reference, reference_cost, actions, _ = raw_terminal_values(model, updated, terminal, 359, 0, labels,
            scores, power=8, seed=714, guarded_mean_position=0)
        np.testing.assert_allclose(values[0, 0], reference, atol=1e-13)
        np.testing.assert_allclose(cost[0, 0], reference_cost, atol=1e-12)
        np.testing.assert_allclose(posterior[0, 0], updated.weights, atol=1e-15)

    def test_resolution_extension_reuses_old_states_and_matches_one_shot(self):
        heads = [(np.full(3, value), np.eye(3)*.02) for value in (1., -.2, -4., -4.)]
        model = JointResponseMixture(['test'], [paired_fields(heads)])
        protocol = {'objective_reference': 2., 'chance_alpha': .05, 'familywise_delta': .05,
            'comparison_budget': 8, 'comparison_delta': .05}
        terminal = ConditionalTerminalActions([[1, 0, 2]], [1], protocol)
        labels = {0: -.13, 1: -.24}
        args = (model, terminal, 360, 0, 1, labels, [0, 1])
        old = evaluate_head_records(*args, powers=[4], seed=714, guarded_mean_position=0)
        snapshot = copy.deepcopy(old)
        extension = evaluate_head_records(*args, powers=[4, 5], seed=714,
            guarded_mean_position=0, completed=old)
        whole = evaluate_head_records(*args, powers=[4, 5], seed=714, guarded_mean_position=0)
        self.assertEqual(len(extension), len(snapshot))
        for before, after, reference in zip(snapshot, extension, whole):
            self.assertEqual(before['raw']['4'], after['raw']['4'])
            self.assertEqual(after['head_log_scores'], reference['head_log_scores'])
            self.assertEqual(after['raw'], reference['raw'])
        self.assertEqual(model.paid_episodes, 0)

    def test_shared_identity_grid_matches_joint_bayes_and_predictive_mass(self):
        prior = np.array([.3, .7]); fl = np.array([[.9, .1], [.1, .9]]); gl = np.array([[.8, .2], [.2, .8]])
        f = np.array([[[[.9], [.2]], [[.3], [.7]]], [[[.4], [.8]], [[.6], [.1]]]])
        g = np.array([[[[.8], [.3]], [[.2], [.9]]], [[[.5], [.4]], [[.9], [.2]]]])
        cost = np.ones((2, 2, 2))*10.
        values, _, posterior, ratio = compose_head_grid(f, g, cost, np.log(fl), np.log(gl), prior)
        joint_mass = np.outer(fl@prior, gl@prior)*ratio
        expected_mass = np.zeros((2, 2)); expected = np.empty((2, 2, 2))
        for i in range(2):
            for j in range(2):
                weights = prior*fl[i]*gl[j]; expected_mass[i, j] = weights.sum(); weights /= weights.sum()
                for a in range(2):
                    expected[i, j, a] = sum(weights[k]*f[i, k, a, 0]*g[j, k, a, 0] for k in range(2))
        np.testing.assert_allclose(values, expected, atol=1e-15)
        np.testing.assert_allclose(joint_mass, expected_mass, atol=1e-15)
        np.testing.assert_allclose(np.einsum('fg,fgk->k', joint_mass, posterior), prior, atol=1e-15)

    def test_adaptive_maximum_stays_inside_each_joint_fantasy(self):
        def records(vectors):
            return [{'head_log_scores': [0.], 'raw': {'8': {
                'statistics': [[[[v] for v in vector][i] for i in range(2)]], 'costs': [[10., 10.]]}}}
                for vector in vectors]
        f = records([[1., 0.], [0., 1.]]); g = records([[1., 1.], [1., 1.]])
        result = summarize_head_grid(f, g, [.5, .5], [.5, .5], [1.], power=8, baseline=(.5, .5, 0))
        self.assertAlmostEqual(result['raw_adaptation_value'], .5)
        self.assertAlmostEqual(result['quadrature_joint_probability_mass'], 1.)

    def test_same_origin_label_banks_recover_complete_mixture_without_extra_density_ratio(self):
        prior = np.array([.3, .7]); fl = np.array([[.9, .1], [.1, .9]]); gl = np.array([[.8, .2], [.2, .8]])
        f = np.array([[[[.9], [.2]], [[.3], [.7]]], [[[.4], [.8]], [[.6], [.1]]]])
        g = np.array([[[[.8], [.3]], [[.2], [.9]]], [[[.5], [.4]], [[.9], [.2]]]])
        def records(statistics, probabilities):
            return [{'head_log_scores': np.log(p).tolist(), 'raw': {'8': {
                'statistics': s.tolist(), 'costs': np.ones((2, 2)).tolist()}}}
                for s, p in zip(statistics, probabilities)]
        results = [summarize_origin_head_grid(records(f, fl), records(g, gl), fl[:, k], gl[:, k], prior, k,
            power=8, baseline=(.2, .2, 0)) for k in range(2)]
        expected = 0.
        for i in range(2):
            for j in range(2):
                weights = prior*fl[i]*gl[j]; probability = weights.sum(); weights /= probability
                values = [sum(weights[k]*f[i, k, a, 0]*g[j, k, a, 0] for k in range(2)) for a in range(2)]
                expected += probability*(max(values)-values[0])
        self.assertAlmostEqual(sum(r['raw_adaptation_value'] for r in results), expected, places=15)
        np.testing.assert_allclose(np.sum([r['expected_identity_weights'] for r in results], axis=0), prior, atol=1e-15)
        self.assertAlmostEqual(sum(r['quadrature_joint_probability_mass'] for r in results), 1.)

    def test_head_statistics_reproduce_raw_actions_and_primary_refinement(self):
        means = ([1.2, 1.1, 1.3], [-.2, -.1, .05], [-4., -4.1, -3.9], [-3.8, -4., -3.7])
        heads = [(np.asarray(m), np.eye(3)*(.02 if h < 2 else .1)+.005) for h, m in enumerate(means)]
        model = JointResponseMixture(['test'], [paired_fields(heads)])
        future = copy.deepcopy(model); labels = [1.15, -.13]; scores = future.update_raw(0, labels)['head_log_scores'][0]
        protocol = {'objective_reference': 2., 'chance_alpha': .05, 'familywise_delta': .05,
            'comparison_budget': 8, 'comparison_delta': .05}
        terminal = ConditionalTerminalActions([[0, 1, 2], [1, 0, 2], [2, 0, 1]], [1, 0, 2], protocol)
        nodes = qmc.Sobol(7, scramble=True, seed=81471).random_base2(8)
        f, _, actions, _ = raw_terminal_head(model.components[0][0], future.components[0][0],
            terminal, 359, 0, labels[0], scores[0], nodes, 0)
        g, cost, same_actions, _ = raw_terminal_head(model.components[0][1], future.components[0][1],
            terminal, 359, 0, labels[1], scores[1], nodes, 1)
        expected, expected_cost, original_actions, _ = raw_terminal_component(model.components[0], future.components[0],
            terminal, 359, 0, labels, scores, nodes)
        subset = ConditionalTerminalActions([[0, 1, 2]], [1], protocol)
        args = (model.components[0], future.components[0], subset, 359, 0, labels, scores, nodes)
        old, old_cost, small_actions, _ = raw_terminal_component(*args)
        new, new_cost, _, _ = raw_terminal_component(*args, guarded_mean_position=0)
        affected = [actions.index(a) for a in small_actions[:3]]
        expected[affected] += (new-old)[:3]; expected_cost[affected] += (new_cost-old_cost)[:3]
        self.assertEqual(actions, same_actions); self.assertEqual(actions, original_actions)
        np.testing.assert_allclose(np.sum(f*g, axis=1), expected, atol=1e-13)
        np.testing.assert_allclose(cost, expected_cost, atol=1e-12)
        primary, primary_cost, _, _ = raw_terminal_head(model.components[0][1], future.components[0][1],
            terminal, 359, 0, labels[1], scores[1], nodes, 1, guarded_mean_position=0)
        primary_reference, reference_cost, _, _ = raw_terminal_component(model.components[0], future.components[0],
            terminal, 359, 0, labels, scores, nodes, guarded_mean_position=0)
        np.testing.assert_allclose(np.sum(f*primary, axis=1), primary_reference, atol=1e-13)
        np.testing.assert_allclose(primary_cost, reference_cost, atol=1e-12)

    def test_low_identity_and_origin_bounds_keep_mass_and_charge_budget_once(self):
        prior = np.array([.999, .001]); fl = np.array([[.2, .7], [.8, .3]]); gl = fl[::-1]
        f = np.array([[[[.9], [.2]], [[.3], [.7]]], [[[.4], [.8]], [[.6], [.1]]]])
        g = np.array([[[[.8], [.3]], [[.2], [.9]]], [[[.5], [.4]], [[.9], [.2]]]])
        def records(statistics, probabilities, retained):
            return [{'head_log_scores': np.log(p).tolist(), 'utility_identities': retained,
                'raw': {'8': {'statistics': s.tolist(), 'costs': np.ones((2, 2)).tolist()}}}
                for s, p in zip(statistics, probabilities)]
        baseline = (.4, .39, 0)
        full = [summarize_origin_head_grid(records(f, fl, [0, 1]), records(g, gl, [0, 1]),
            fl[:, k], gl[:, k], prior, k, power=8, baseline=baseline) for k in range(2)]
        ff, gg = f.copy(), g.copy(); ff[:, 1] = 0.; gg[:, 1] = 0.
        partial = summarize_origin_head_grid(records(ff, fl, [0]), records(gg, gl, [0]),
            fl[:, 0], gl[:, 0], prior, 0, power=8, baseline=baseline)
        result = summarize_origin_grid_mixture([partial], prior, baseline)
        expected = sum(r['raw_adaptation_value'] for r in full)+baseline[1]-baseline[0]
        lower, upper = result['raw_net_control_variate_identity_interval']
        self.assertLessEqual(lower, expected); self.assertGreaterEqual(upper, expected)
        self.assertAlmostEqual(result['quadrature_joint_probability_mass'], prior[0])
        self.assertAlmostEqual(result['rare_predictive_origin_mass'], prior[1])
        self.assertAlmostEqual(result['verification_budget_opportunity_cost'], .01)
        self.assertEqual(len(result['expected_identity_weights']), 2)

    def test_single_proposal_restores_both_predictive_origins_without_refitting_weights(self):
        prior = np.array([.3, .7]); fl = np.array([[.9, .1], [.1, .9]]); gl = np.array([[.8, .2], [.2, .8]])
        def records(probabilities, vectors):
            return [{'head_log_scores': np.log(p).tolist(), 'raw': {'8': {
                'statistics': s, 'costs': [[1., 1.], [1., 1.]]}}} for p, s in zip(probabilities, vectors)]
        f = records(fl, [[[[.9], [.2]], [[.3], [.7]]], [[[.4], [.8]], [[.6], [.1]]]])
        g = records(gl, [[[[.8], [.3]], [[.2], [.9]]], [[[.5], [.4]], [[.9], [.2]]]])
        original = [summarize_origin_head_grid(f, g, fl[:, k], gl[:, k], prior, k,
            power=8, baseline=(.4, .39, 0)) for k in range(2)]
        proposal = summarize_proposal_head_grid(f, g, fl[:, 0], gl[:, 0], prior, 0, [0, 1],
            power=8, baseline=(.4, .39, 0))
        self.assertAlmostEqual(proposal['raw_adaptation_value'], sum(r['raw_adaptation_value'] for r in original))
        self.assertAlmostEqual(proposal['quadrature_joint_probability_mass'], 1.)
        np.testing.assert_allclose(proposal['expected_identity_weights'], prior, atol=1e-15)
        self.assertAlmostEqual(proposal['verification_budget_opportunity_cost'], .01)
        higher_risk = copy.deepcopy(g)
        for record in higher_risk:
            record['raw']['9'] = record['raw'].pop('8')
        separate_powers = summarize_proposal_head_grid(f, higher_risk, fl[:, 0], gl[:, 0], prior, 0, [0, 1],
            power=9, objective_power=8, baseline=(.4, .39, 0))
        self.assertAlmostEqual(separate_powers['raw_adaptation_value'], proposal['raw_adaptation_value'])
        self.assertEqual(separate_powers['objective_raw_draw_power'], 8)
        self.assertEqual(separate_powers['raw_draw_power'], 9)
        partial = summarize_proposal_head_grid(f, g, fl[:, 0], gl[:, 0], prior, 0, [0],
            power=8, baseline=(.4, .39, 0))
        self.assertAlmostEqual(partial['quadrature_joint_probability_mass'], .3)
        self.assertAlmostEqual(partial['rare_predictive_origin_mass'], .7)
        self.assertAlmostEqual(partial['verification_budget_opportunity_cost'], .01)


if __name__ == '__main__':
    unittest.main()
