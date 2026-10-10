import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import performance.run_historical_independent_kg_pilot as execution
from performance.check_historical_independent_queries import registered_queries
from test_joint_response_mixture import mixture
from test_legacy_one_step_closed_loop import PaidProblem
from test_positive_source_mixture import terminal as toy_terminal
from performance.prepare_historical_independent_kg import actions_at
from core.conditional_terminal_kg import ConditionalTerminalActions


def cohort():
    states = [{'name': arm, 'task_id': 'fresh', 'layout': '5', 'arm': arm,
        'queries': [5, 4]} for arm in ['source', 'gp']]
    cases = [dict(c, processing_complete=True, adaptation_diagnostics_passed=True,
        estimated_net_kg=0.) for c in registered_queries(states)]
    diagnostics = {'all_model_states_ready': True, 'cases': cases}
    baselines = {'all_baseline_diagnostics_passed': True,
        'cases': [{'name': s['name'], 'baseline_diagnostics_passed': True} for s in states]}
    return states, diagnostics, baselines


class ExecutionGateTests(unittest.TestCase):
    def test_model_shard_cannot_create_the_full_verification_barrier(self):
        states, diagnostics, baselines = cohort()
        action = {'kind': 'direct', 'primary': 0, 'shortlist': [0], 'incumbent_position': 0, 'budgets': [360]}
        for state in states:
            state.update(terminal_actions=[action], future_terminal_actions=[dict(action, budgets=[359])])
        for case in baselines['cases']:
            case.update(result='baseline_result', power=12)
        fine = {'rows': [{'power':12, 'remaining_calls':360, 'best_action':action,
            'best_action_index':0, 'best_reward':.5, 'costs':[360]}]}
        pilot = {'policies':[execution.KG, execution.STOP], 'models':['source','gp'], 'verification_seed':714}
        frozen = {'tasks':[{'task_id':'fresh'}], 'banks':[{'layout':'5','policies':[[0.]]}]}
        spec = {'output':'out', 'state_names':['source'], 'baselines':'b', 'query_diagnostics':'q',
            'importance_precision_plan':'importance'}
        data = {'plan':spec, 'b':baselines, 'q':diagnostics, 'importance':{'precision':{}}, 'baseline_result':fine}
        with tempfile.TemporaryDirectory() as name, \
             patch.object(execution, 'ROOT', Path(name)), \
             patch.object(execution, 'inputs', return_value=(pilot,frozen,states,{})), \
             patch.object(execution, 'read', side_effect=data.__getitem__), \
             patch.object(execution, 'restore_target_models', side_effect=lambda p, selected:
                [(s,None,None) for s in selected]):
            result = execution.freeze('plan')
            self.assertEqual(len(result['rows']),2)
            self.assertTrue(result['selected_recommendations_complete'])
            self.assertFalse(result['all_recommendations_frozen'])
            self.assertFalse((Path(name)/'out/recommendations.json').exists())
            with self.assertRaises(ValueError):
                execution.validate_recommendations(result['rows'], states, pilot)

    def test_missing_hold_or_mismatched_query_blocks_search(self):
        states, diagnostics, baselines = cohort()
        for change in ('missing', 'hold', 'wrong_task'):
            bad = copy.deepcopy(diagnostics)
            if change == 'missing':
                bad['cases'].pop()
            elif change == 'hold':
                bad['cases'][0]['adaptation_diagnostics_passed'] = False
            else:
                bad['cases'][0]['task_id'] = 'development'
            with self.subTest(change=change), self.assertRaises(ValueError):
                execution.choose_queries(states, bad, baselines)

    def test_model_own_argmax_frozen_tie_order_and_zero_stopping(self):
        states, diagnostics, baselines = cohort()
        for case, value in zip(diagnostics['cases'], [.1, .1, -.2, .3]):
            case['estimated_net_kg'] = value
        decisions = execution.choose_queries(states, diagnostics, baselines)
        self.assertEqual([d['query'] for d in decisions], [5, 4])
        for case in diagnostics['cases']:
            case['estimated_net_kg'] = 0.
        self.assertTrue(all(d['query'] is None for d in execution.choose_queries(states, diagnostics, baselines)))

    def test_real_entrypoint_refuses_incomplete_cohort_before_problem_creation(self):
        states, diagnostics, baselines = cohort(); diagnostics['cases'].pop()
        with patch.object(execution, 'inputs', return_value=({}, {}, states, {})), \
                patch.object(execution, 'read', side_effect=[{'baselines': 'b', 'query_diagnostics': 'q'}, baselines, diagnostics]), \
                patch.object(execution, 'problem_for') as simulator:
            with self.assertRaises(ValueError):
                execution.freeze()
            simulator.assert_not_called()

    def test_complete_barrier_and_query_debits(self):
        action = {'kind': 'direct', 'primary': 0, 'shortlist': [0], 'incumbent_position': 0, 'budgets': [360]}
        future = dict(action, budgets=[359])
        states = [{'name': 'state', 'task_id': 'fresh', 'layout': '5', 'arm': 'source',
            'terminal_actions': [action], 'future_terminal_actions': [future]}]
        pilot = {'policies': [execution.KG, execution.STOP]}
        rows = [dict(state_name='state', task_id='fresh', layout='5', arm='source', policy=policy,
            query=5 if policy == execution.KG else None, search_calls=3 if policy == execution.KG else 2,
            action=future if policy == execution.KG else action, action_index=0,
            recommendation_diagnostics_passed=True) for policy in pilot['policies']]
        self.assertTrue(execution.validate_recommendations(rows, states, pilot))
        for bad in (rows[:1], [rows[0], rows[0]], [dict(rows[0], search_calls=2), rows[1]],
                    [dict(rows[0], action=action), rows[1]]):
            with self.assertRaises(ValueError):
                execution.validate_recommendations(bad, states, pilot)
        with patch.object(execution, 'inputs', return_value=(pilot, {}, states, {})), \
                patch.object(execution, 'read', side_effect=[{'output': 'unused'},
                    {'all_recommendations_frozen': True, 'rows': rows[:1]}]), \
                patch.object(execution, 'execute_terminal') as verifier:
            with self.assertRaises(ValueError):
                execution.verify()
            verifier.assert_not_called()


class PaidRawRecommendationTests(unittest.TestCase):
    def test_freeze_uses_raw_recommendation_without_verification_and_reuses_barrier(self):
        model = mixture(); model.update_raw(0, [.2, -.4]); model.update_raw(1, [.7, -.4])
        initial = toy_terminal()
        terminal = ConditionalTerminalActions(initial.shortlists, initial.incumbents, initial.protocol)
        state = dict(name='fresh_gp', task_id='fresh', layout='5', arm='gp', queries=[2],
            terminal_actions=actions_at(terminal, 360), future_terminal_actions=actions_at(terminal, 359))
        pilot = dict(models=['gp'], policies=[execution.KG, execution.STOP],
            search_seed=91, verification_seed=92, maximum_identity_integration_error_bound=2e-6,
            maximum_numerical_adaptation_difference=1e-4)
        frozen = {'tasks': [{'task_id': 'fresh'}], 'banks': [{'layout': '5', 'policies': [[0], [1], [2]]}]}
        diagnostics = {'all_model_states_ready': True, 'cases': [dict(registered_queries([state])[0],
            processing_complete=True, adaptation_diagnostics_passed=True, estimated_net_kg=.02)]}
        baseline = {'all_baseline_diagnostics_passed': True, 'cases': [{'name': state['name'],
            'result': 'baseline_case', 'power': 2, 'baseline_diagnostics_passed': True}]}
        fine = dict(power=2, remaining_calls=360, best_action=state['terminal_actions'][0],
            best_action_index=0, best_reward=.1, costs=[300]*len(state['terminal_actions']))
        spec = dict(baselines='baselines', query_diagnostics='diagnostics', output='pilot',
            importance_precision_plan='importance', maximum_freeze_cpu_seconds=10,
            post_query_raw_powers=[2, 3], integration_seed=3)
        data = {'plan': spec, 'baselines': baseline, 'diagnostics': diagnostics,
            'baseline_case': {'rows': [fine]}, 'importance': {'precision': {
                'maximum_log_evidence_error': .02, 'minimum_reference_ESS': 128}}}
        def load(path):
            return data[path] if path in data else json.loads(Path(path).read_text())
        problem = PaidProblem(); problem.tau = .07
        expected_label = np.array([.6, -.5])+.02*np.random.default_rng(np.random.SeedSequence([91, 0])).normal(size=2)
        expected_label[1] -= .07
        def raw(prior, updated, current_terminal, remaining, query, labels, scores, **kwargs):
            self.assertIs(prior, model); self.assertIsNot(updated, model)
            self.assertEqual(prior.paid_episodes, 2); self.assertEqual(updated.paid_episodes, 3)
            self.assertEqual(remaining, 359); self.assertEqual(query, 2)
            np.testing.assert_array_equal(labels, expected_label)
            self.assertEqual(kwargs['guarded_mean_position'], 0)
            np.testing.assert_array_equal(scores, updated.history[-1]['head_log_scores'])
            values = np.zeros(len(state['future_terminal_actions'])); values[1] = .3
            return values, np.full(len(values), 300.), state['future_terminal_actions'], {
                'omitted_posterior_utility_mass': 0., 'predictive_log_evidence_max_error': .001,
                'effective_sample_size_min': 256}
        with tempfile.TemporaryDirectory() as tmp, patch.object(execution, 'ROOT', Path(tmp)), \
                patch.object(execution, 'read', side_effect=load), \
                patch.object(execution, 'inputs', return_value=(pilot, frozen, [state], {})), \
                patch.object(execution, 'restore_target_models', return_value=[(state, model, terminal)]), \
                patch.object(execution, 'problem_for', return_value=problem), \
                patch.object(execution, 'raw_terminal_values', side_effect=raw) as raw_call, \
                patch.object(execution, 'execute_terminal') as verifier, \
                patch.object(model, 'terminal_values', side_effect=AssertionError('projected third update')):
            result = execution.freeze('plan')
            self.assertTrue(result['all_recommendations_frozen'])
            self.assertEqual([r['search_calls'] for r in result['rows']], [3, 2])
            self.assertEqual(result['rows'][0]['action'], state['future_terminal_actions'][1])
            self.assertEqual(raw_call.call_count, 2); self.assertEqual(problem.calls, 1)
            self.assertEqual(execution.freeze('plan'), result)
            self.assertEqual(problem.calls, 1); verifier.assert_not_called()

    def test_paid_record_resumes_without_another_episode_or_truth(self):
        model = mixture(); model.update_raw(0, [.2, -.4]); model.update_raw(1, [.7, -.4])
        problem = PaidProblem(); decision = {'query': 2, 'state_name': 'fresh'}
        candidates = [(0,), (1,), (2,)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'search.json'
            first, record = execution.search_record(model, problem, candidates, decision, 91, 2, path)
            second, saved = execution.search_record(model, problem, candidates, decision, 91, 2, path)
        self.assertEqual(problem.calls, 1)
        self.assertEqual(model.paid_episodes, 2)
        self.assertEqual(first.paid_episodes, 3)
        self.assertEqual(second.history, first.history)
        self.assertEqual(record, saved)
        self.assertEqual(np.asarray(saved['evidence']['head_log_scores']).shape, (2, 2))

    def test_realized_rare_mass_adds_identities_without_renormalization(self):
        class Posterior:
            components = [None]*8
            weights = np.array([.04, .02, .001, .002, .6, .3, .03, .007])
        updated = Posterior(); before = updated.weights.copy()
        selected = execution.utility_identities(updated, 'source_structural_mixture', .002)
        self.assertEqual(selected[:2], [4, 5])
        self.assertLessEqual(updated.weights[[i for i in range(8) if i not in selected]].sum(), .002)
        np.testing.assert_array_equal(updated.weights, before)

    def test_post_query_numerical_hold_cannot_accept_changed_choice_or_poor_importance(self):
        previous = {'values': [.2, .1], 'best_action_index': 0}
        current = {'values': [.20001, .1], 'best_action_index': 0, 'diagnostics': {
            'omitted_posterior_utility_mass': 1e-6, 'predictive_log_evidence_max_error': .001,
            'effective_sample_size_min': 256}}
        pilot = {'maximum_numerical_adaptation_difference': 1e-4, 'maximum_identity_integration_error_bound': 2e-6}
        importance = {'maximum_log_evidence_error': .02, 'minimum_reference_ESS': 128}
        self.assertTrue(execution.recommendation_check(previous, current, pilot, importance)['passed'])
        current['best_action_index'] = 1
        self.assertFalse(execution.recommendation_check(previous, current, pilot, importance)['passed'])
        current['best_action_index'] = 0; current['diagnostics']['effective_sample_size_min'] = 20
        self.assertFalse(execution.recommendation_check(previous, current, pilot, importance)['passed'])


class PairedTaskTests(unittest.TestCase):
    def test_two_layouts_are_averaged_before_whole_task_bootstrap(self):
        pilot = {'models': ['source', 'gp'], 'bootstrap_seed': 1}
        frozen = {'tasks': [{'task_id': f't{i}'} for i in range(3)], 'banks': [{'layout': '5'}, {'layout': '6'}]}
        rows = []
        for i, task in enumerate(frozen['tasks']):
            for layout, offset in [('5', .2), ('6', -.1)]:
                for arm in pilot['models']:
                    for policy in [execution.KG, execution.STOP]:
                        value = offset if arm == 'source' and policy == execution.KG else 0.
                        row = dict(task_id=task['task_id'], layout=layout, arm=arm, policy=policy,
                            scores={m: value+i for m in ['actual_safe_reward', 'false_certificate', 'rejection',
                                'actual_total_calls', 'conditional_safe_reward', 'conditional_total_calls']})
                        rows.append(row)
        comparison = execution.paired_summary(rows, frozen, pilot, 100)[0]
        self.assertEqual(comparison['task_count'], 3)
        metric = comparison['metrics']['actual_safe_reward']
        np.testing.assert_allclose(metric['task_differences'], [.05]*3, atol=1e-15)
        np.testing.assert_allclose(metric['task_bootstrap_95_interval'], [.05, .05], atol=1e-15)


if __name__ == '__main__':
    unittest.main()
