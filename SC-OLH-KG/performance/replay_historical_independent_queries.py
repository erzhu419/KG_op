"""Replay saved independent grids without new terminal integrations or episodes."""

import time

import numpy as np

from performance.check_historical_gauss_risk_tail import summarize
from performance.check_historical_independent_queries import PLAN, cohort_status, registered_queries
from performance.complete_historical_source_heads import ROOT, read, save


def run(plan=None):
    started = time.process_time(); location = read(plan or PLAN)
    execution = read(location['output']+'/results.json')
    plan = execution['execution_plan'] if plan is None else plan
    spec = read(plan); pilot = read(spec['pilot_plan'])
    rule = dict(read(pilot['numerical_plan']), **spec['numerical_overrides'])
    preflight = read(spec['preflight']); states = {s['name']: s for s in preflight['states']}
    baselines = {c['name']: c for c in read(spec['baselines'])['cases']}
    out = ROOT/spec['output']
    registration = read(spec['output']+'/registered_queries.json')
    assert registration['queries'] == registered_queries(list(states.values()))
    assert preflight['all_density_and_rare_origin_prechecks_passed']
    assert all(c['baseline_diagnostics_passed'] for c in baselines.values())
    tests = read(spec['unit_tests']); assert tests['successful']
    assert not execution['independent_outcomes_opened'] and not execution['independent_confirmation']
    compact = []
    for case in execution['cases']:
        state = states[case['state_name']]
        assert all(case[k] == state[k] for k in ('task_id', 'layout', 'arm'))
        assert case['query'] in state['queries']
        assert case['baseline'] == baselines[state['name']]['baseline']
        bank = read(case['bank']); assert bank['source_weights'] == case['source_weights'] == state['source_weights']
        heads = [read(spec['output']+'/'+case['name']+f'/head{h}.json') for h in (0, 1)]
        for head, records in enumerate(heads):
            assert records['actions'] == state['future_terminal_actions']
            labels = bank['heads'][head]; by_position = dict(zip(labels['positions'], labels['labels']))
            assert {r['position'] for r in records['records']} == set(by_position)
            for row in records['records']:
                assert row['head'] == head and row['label'] == by_position[row['position']]
                assert row['guarded_mean_position'] == 0
                identities = [4, 5] if state['arm'] == 'source_structural_mixture' else [0]
                assert row['utility_identities'] == identities
                index = labels['positions'].index(row['position'])
                assert max(abs(row['head_log_scores'][k]-labels['predictive_log_densities'][str(k)][index])
                    for k in identities) < 1e-8
        for stage in case['stages']:
            assert stage['raw_powers'] in [rule['initial_raw_powers'], rule['final_raw_powers']]
            replay = summarize([h['records'] for h in heads], bank['risk_rules'], state['source_weights'],
                dict(rule, raw_powers=stage['raw_powers'], proposal_origin=4 if len(identities) == 2 else 0,
                    predictive_origins=identities), objective_rules=bank['objective_rules'], baseline=case['baseline'])
            assert len(replay['rows']) == len(stage['rows'])
            for original, current in zip(stage['rows'], replay['rows']):
                for key in ('raw_adaptation_value', 'raw_net_reward_control_variate',
                        'raw_fixed_action_tower_residual', 'identity_and_rare_origin_error_bound',
                        'predictive_mass_residual'):
                    assert abs(original[key]-current[key]) < 1e-14
                np.testing.assert_allclose(original['selected_action_probabilities'], current['selected_action_probabilities'],
                    atol=1e-14, rtol=0.)
            assert stage['numerical_changes'] == replay['numerical_changes']
            gates = {'terminal_and_predictive_resolution': replay['numerical_gates_passed'],
                'identity_remainder': max(r['identity_and_rare_origin_error_bound'] for r in replay['rows']) <= pilot['maximum_identity_integration_error_bound'],
                'predictive_mass': max(abs(r['predictive_mass_residual']) for r in replay['rows']) <= pilot['maximum_predictive_mass_residual'],
                'density_ratio': all(pilot['predictive_density_ratio_limits'][0] <= r['predictive_density_ratio_range'][0] and
                    r['predictive_density_ratio_range'][1] <= pilot['predictive_density_ratio_limits'][1] for r in replay['rows'])}
            assert gates == stage['gates'] and all(gates.values()) == stage['grid_diagnostics_passed']
            assert stage['fixed_action_tower_gate_passed'] == replay['fixed_action_tower_gate_passed']
        if case.get('processing_complete', False):
            anchors = case['anchors']
            assert {a['identity'] for a in anchors['rows']} == set(identities)
            for anchor in anchors['rows']:
                error = float(np.max(abs(np.asarray(anchor['actual'])-anchor['full_reference'])))
                assert error == anchor['error'] and anchor['query'] == case['query']
                assert anchor['passed'] == (error <= pilot['maximum_anchor_branch_error'])
            assert anchors['all_passed'] == all(a['passed'] for a in anchors['rows'])
            assert case['adaptation_diagnostics_passed'] == (case['stages'][-1]['grid_diagnostics_passed'] and anchors['all_passed'])
        fine = case['stages'][-1]['rows'][-1]
        compact.append({k: case[k] for k in ('name', 'state_name', 'task_id', 'layout', 'arm', 'query')} | {
            'processing_complete': case.get('processing_complete', False),
            'adaptation_diagnostics_passed': case.get('adaptation_diagnostics_passed', False),
            'grid_diagnostics_passed': case['stages'][-1]['grid_diagnostics_passed'],
            'numerical_changes': case['stages'][-1]['numerical_changes'],
            'estimated_net_kg': fine['raw_net_reward_control_variate'],
            'raw_fixed_action_tower_residual': fine['raw_fixed_action_tower_residual'],
            'fixed_action_tower_gate_passed': case['stages'][-1]['fixed_action_tower_gate_passed'],
            'raw_net_identity_interval': fine['raw_net_control_variate_identity_interval']})
    status = cohort_status(list(states.values()), execution['cases'])
    assert all(execution[k] == value for k, value in status.items())
    result = dict(status, execution_plan=plan, cases=compact,
        initial_grid_passed_cases=sum(c['stages'][0]['grid_diagnostics_passed'] for c in execution['cases']),
        latest_grid_passed_cases=sum(c['grid_diagnostics_passed'] for c in compact),
        raw_fixed_action_tower_hold_cases=[c['name'] for c in compact if not c['fixed_action_tower_gate_passed']],
        cpu_seconds=execution['cpu_seconds'], cpu_budget_stopped=execution['cpu_budget_stopped'],
        successful_tests=tests['tests_run'], cached_replay_cpu_seconds=time.process_time()-started,
        cost=execution['cost'], independent_outcomes_opened=False, independent_confirmation=False,
        decision_ranking_certified=False, limitations=execution['limitations'], next=execution['next'])
    save(out/'phase_results.json', result)
    print('independent query replay', result['initial_query_cases_complete'], '/144,passed',
        result['passed_query_cases'], 'ready models', len(result['ready_model_states']), flush=True)
    return result


if __name__ == '__main__':
    run()
