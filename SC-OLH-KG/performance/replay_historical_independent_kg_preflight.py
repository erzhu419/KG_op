"""Replay cached independent-query prechecks without new physical records."""

import time

from performance.check_historical_remaining_queries import density_gate
from performance.complete_historical_source_heads import ROOT, read, save


def run():
    started = time.process_time(); directory = 'paper_artifacts/historical_independent_kg_preflight_v1'
    result = read(directory+'/results.json'); spec = read(result['pilot_plan']); numerical = read(spec['numerical_plan'])
    tests = read(directory+'/unit_tests.json'); restoration = read(directory+'/restoration_checks.json')
    assert tests['successful'] and restoration['complete_connection_exactly_preserved']
    assert result['all_registered_cases_complete'] and len(result['states']) == 24 and len(result['cases']) == 144
    assert len({r['name'] for r in result['cases']}) == 144
    initial = read(directory+'/initializations.json')['rows']
    assert len(initial) == 6 and sum(r['physical_calls'] for r in initial) == 12
    initial_by_task = {r['task_id']: r for r in initial}
    states = {r['name']: r for r in result['states']}
    for state in states.values():
        observed = initial_by_task[state['task_id']]
        assert state['paid_observations'] == observed['paired_observations']
        assert state['candidate_bank_indices'][:2] == observed['candidate_bank_indices']
        assert state['centered_observations'] == [[v[0], v[1]-observed['tau']] for v in observed['paired_observations']]
        assert state['posterior_moment_diagnostic']['paid_episodes'] == 2
        for budget, key in ((360, 'terminal_actions'), (359, 'future_terminal_actions')):
            assert len(state[key]) == 40
            assert max(sum(a['budgets'])+(16 if a['kind'] == 'guarded' else 0) for a in state[key]) == budget
    for case in result['cases']:
        state = states[case['name'].rsplit('_query', 1)[0]]
        source = case['arm'] == 'source_structural_mixture'; identities, proposal = ([4, 5], 4) if source else ([0], 0)
        bank = read(directory+'/'+case['name']+'/label_banks.json')
        assert bank['source_weights'] == state['source_weights']
        assert max(h['maximum_inverse_cdf_error'] for h in bank['heads']) <= numerical['maximum_inverse_cdf_error']
        density = density_gate(bank, proposal, identities, spec, numerical)
        assert density == case['density_precheck'] and density['passed'] == case['density_precheck_passed']
        rare = sum(w for i, w in enumerate(bank['source_weights']) if i not in identities)
        assert rare == case['rare_predictive_origin_mass']
        assert case['rare_origin_precheck_passed'] == (rare <= spec['maximum_identity_integration_error_bound'])
    original = read(directory+'/initial_fixed_bisection_results.json')
    assert original['cases'] == result['cases'][:len(original['cases'])]
    phase = {'execution_plan': result['execution_plan'], 'pilot_plan': result['pilot_plan'],
        'new_task_count': len(initial), 'restored_model_state_count': len(states), 'query_case_count': len(result['cases']),
        'all_density_and_rare_origin_prechecks_passed': result['all_density_and_rare_origin_prechecks_passed'],
        'all_original17_cases_preserved': True, 'complete_old_source_connection_preserved': True,
        'source_restoration': result['source_restoration'], 'cost': result['cost'],
        'cpu_seconds': result['cpu_seconds'], 'successful_tests': tests['tests_run'],
        'cdf_efficiency_smoke': read(directory+'/cdf_efficiency_smoke.json'),
        'cached_replay_cpu_seconds': time.process_time()-started,
        'independent_outcomes_opened': False, 'independent_confirmation': False,
        'limitations': result['limitations'], 'next': result['next']}
    save(ROOT/directory/'phase_results.json', phase)
    print('independent cached replay', len(result['cases']), 'queries,', len(states), 'states,', len(initial), 'tasks;',
        tests['tests_run'], 'tests;CPU', round(phase['cached_replay_cpu_seconds'], 3))
    return phase


if __name__ == '__main__':
    run()
