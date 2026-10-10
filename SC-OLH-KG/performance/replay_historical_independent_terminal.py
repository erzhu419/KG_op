"""Replay baseline diagnostics and merge selective refinements across24 states."""

import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, read, save


def run():
    started = time.process_time(); preflight = read('paper_artifacts/historical_independent_kg_preflight_v1/results.json')
    states = {s['name']: s for s in preflight['states']}; latest = {}; stages = []; old = None
    for version in (1, 2, 3, 4):
        plan = f'performance/manifests/historical_independent_terminal_baselines_v{version}.json'
        spec = read(plan); result = read(spec['output']+'/results.json')
        assert result['all_registered_cases_complete']
        if version <= 2:
            assert {c['name'] for c in result['cases']} == set(states)
        else:
            assert set(spec['case_names']) == {c['name'] for c in old['cases'] if not c['baseline_diagnostics_passed']}
            assert {c['name'] for c in result['cases']} == set(spec['case_names'])
        for case in result['cases']:
            state = states[case['name']]; record = read(spec['output']+'/'+case['name']+'/terminal_rows.json')
            assert record['rows'] == case['rows'] and record['source_weights'] == state['source_weights']
            if version >= 3:
                original = read(spec['reuse_output']+'/'+case['name']+'/terminal_rows.json')
                assert record['rows'][:len(original['rows'])] == original['rows']
            identities = [4, 5] if state['arm'] == 'source_structural_mixture' else [0]
            omitted = sum(w for k, w in enumerate(state['source_weights']) if k not in identities)
            for row in case['rows']:
                actions = state['terminal_actions' if row['remaining_calls'] == 360 else 'future_terminal_actions']
                assert row['utility_identities'] == identities and row['omitted_terminal_identity_mass'] == omitted
                assert row['best_action'] == actions[row['best_action_index']]
                best = int(np.lexsort((row['costs'], -np.asarray(row['values'])))[0])
                assert best == row['best_action_index'] and row['best_reward'] == row['values'][best]
            changes = []; stable = []
            for budget in spec['remaining_budgets']:
                a, b = [next(r for r in case['rows'] if (r['power'], r['remaining_calls']) == (p, budget)) for p in spec['raw_powers']]
                changes.append(float(np.max(abs(np.asarray(a['values'])-b['values']))))
                stable.append(a['best_action_index'] == b['best_action_index'])
            opportunity = [next(r for r in case['rows'] if (r['power'], r['remaining_calls']) == (p, 360))['best_reward']-
                next(r for r in case['rows'] if (r['power'], r['remaining_calls']) == (p, 359))['best_reward'] for p in spec['raw_powers']]
            gates = {'all_action_resolution': max(changes) <= spec['maximum_terminal_action_difference'],
                'opportunity_cost_resolution': abs(opportunity[1]-opportunity[0]) <= spec['maximum_opportunity_cost_difference'],
                'stop_and_debited_choices_stable': all(stable), 'identity_remainder': omitted <= spec['maximum_identity_integration_error_bound']}
            assert changes == case['maximum_action_changes_by_budget'] and opportunity == case['opportunity_cost_by_power']
            assert gates == case['gates'] and all(gates.values()) == case['baseline_diagnostics_passed']
            if version >= 2:
                fine = spec['raw_powers'][-1]
                before, after = [next(r for r in case['rows'] if (r['power'], r['remaining_calls']) == (fine, budget)) for budget in (360, 359)]
                latest[case['name']] = {'name': case['name'], 'task_id': case['task_id'], 'layout': case['layout'], 'arm': case['arm'],
                    'result': spec['output']+'/'+case['name']+'/results.json', 'power': fine, 'gates': gates,
                    'baseline_diagnostics_passed': all(gates.values()), 'maximum_action_difference': max(changes),
                    'opportunity_cost_difference': abs(opportunity[1]-opportunity[0]),
                    'stop_action': before['best_action'], 'baseline': [before['best_reward'], after['best_reward'], after['best_action_index']]}
        stages.append({'plan': plan, 'registered_cases': result['registered_case_count'],
            'passed_cases': result['passed_case_count'], 'cpu_seconds': result['cpu_seconds']})
        old = result
    assert set(latest) == set(states)
    tests = read('paper_artifacts/historical_independent_terminal_baselines_v2/unit_tests.json'); assert tests['successful']
    summary = {'cases': list(latest.values()), 'stages': stages, 'registered_model_states': len(latest),
        'passed_model_states': sum(c['baseline_diagnostics_passed'] for c in latest.values()),
        'all_baseline_diagnostics_passed': all(c['baseline_diagnostics_passed'] for c in latest.values()),
        'all_previous_vectors_preserved': True, 'successful_tests': tests['tests_run'],
        'cpu_seconds': sum(s['cpu_seconds'] for s in stages), 'cached_replay_cpu_seconds': time.process_time()-started,
        'cost': {'new_source_fits': 0, 'new_physical_calls': 0, 'new_search_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0},
        'independent_outcomes_opened': False, 'independent_confirmation': False,
        'limitations': ['Stopping/debited utility diagnostics do not evaluate independent deployment outcomes.',
            'Adjacent-rule gates do not certify strict global integration errors or final query rankings.'],
        'next': 'Use the complete target-specific baselines and cached predictive banks to compute all144 KG queries;freeze48 recommendations before verification.'}
    save(ROOT/'paper_artifacts/historical_independent_terminal_baselines_v4/phase_results.json', summary)
    print('independent baseline replay', summary['passed_model_states'], '/', len(latest), 'CPU', round(summary['cpu_seconds'], 2))
    return summary


if __name__ == '__main__':
    run()
