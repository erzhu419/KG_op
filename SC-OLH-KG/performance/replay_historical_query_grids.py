"""Replay matched query results without raw conditioning or new predictive draws."""

import time

from performance.check_historical_query_grids import PLAN, grid_rules
from performance.check_historical_gauss_risk_tail import PLAN as TAIL_PLAN, summarize
from performance.complete_historical_source_heads import ROOT, read, save


def run(plan=PLAN):
    started = time.process_time(); spec = read(plan); out = ROOT/spec['output']
    result = read(str((out/'results.json').relative_to(ROOT))); tests = read(str((out/'unit_tests.json').relative_to(ROOT)))
    assert tests['successful']
    for case in result['cases']:
        directory = out/case['name']; heads = []
        for head in (0, 1):
            current = read(str((directory/f'head{head}.json').relative_to(ROOT)))
            bank = read(str((directory/f'labels{head}.json').relative_to(ROOT)))
            labels = dict(zip(bank['positions'], bank['labels']))
            for r in current['records']:
                assert r['label'] == labels[r['position']]
                assert len(r['head_log_scores']) == len(case['source_weights'])
            heads.append(current['records'])
        is_source = case['arm'] == 'source_structural_mixture'
        replay = grid_rules(heads, case['source_weights'], case['baseline'],
            spec['source_proposal_origin'] if is_source else 0, spec['source_predictive_origins'] if is_source else [0], spec)
        assert len(replay['rows']) == len(case['rows'])
        for saved, current in zip(case['rows'], replay['rows']):
            for key in ('raw_adaptation_value', 'raw_net_reward_control_variate', 'predictive_mass_residual'):
                assert abs(saved[key]-current[key]) < 1e-14
        assert replay['numerical_changes'] == case['numerical_changes']
    tail_spec = read(TAIL_PLAN); tail_out = ROOT/tail_spec['output']
    tail = read(str((tail_out/'results.json').relative_to(ROOT)))
    if tail['all_registered_cases_complete']:
        heads = [read(tail_spec['completed_objective_heads'])['records'], read(tail_spec['completed_risk_heads'])['records']+
            read(str((tail_out/'risk_heads.json').relative_to(ROOT)))['records']]
        reference = read(tail_spec['source_reference'])
        replay = summarize(heads, read(str((tail_out/'labels.json').relative_to(ROOT))), reference['source_weights'], tail_spec)
        assert replay['numerical_changes'] == tail['numerical_changes']
        assert replay['fixed_action_tower_gate_passed'] == tail['fixed_action_tower_gate_passed']
    gates = ('numerical_gates_passed', 'identity_remainder_gate_passed', 'proposal_mass_gate_passed', 'proposal_ratio_gate_passed')
    result.update(all_registered_numerical_gates_passed=result['all_registered_cases_complete'] and
        all(c[k] for c in result['cases'] for k in gates),
        cached_replay_cpu_seconds=time.process_time()-started,
        gauss_tail_fixed_action_tower_gate_passed=tail.get('fixed_action_tower_gate_passed', False),
        next='Complete unresolved numerical query cases and remaining frozen queries before independent deployment comparison.')
    save(out/'phase_results.json', result)
    print('cohort complete', result['all_registered_cases_complete'], 'all numerical gates', result['all_registered_numerical_gates_passed'],
        'CPU', result['cpu_seconds'])
    for c in result['cases']:
        print(c['name'], c['rows'][-1]['raw_net_reward_control_variate'], c['numerical_changes'], c['numerical_gates_passed'])
    return result


if __name__ == '__main__':
    run()
