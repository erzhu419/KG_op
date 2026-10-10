"""Replay cached head grids,source weights and unchanged old head records."""

import time

import numpy as np
from scipy.stats import qmc

from performance.complete_historical_source_heads import ROOT, read, save
from performance.check_historical_raw_head_grid import grid_rules, anchor_comparison
from performance.extend_historical_matched_gp import restore_saved_model

PLAN = 'performance/manifests/historical_raw_head_grid_v2.json'


def run(plan=PLAN):
    started = time.process_time(); spec = read(plan); out = ROOT/spec['output']
    result = read(str((out/'results.json').relative_to(ROOT)))
    assert result['all_registered_cases_complete']
    parent_out = ROOT/spec['completed_head_output']
    parent = read(str((parent_out/'results.json').relative_to(ROOT)))
    tests = read(str((parent_out/'unit_tests.json').relative_to(ROOT))); assert tests['successful']
    heads = []
    for head in (0, 1):
        own = out/f'gp_head{head}.json'; path = own if own.exists() else parent_out/f'gp_head{head}.json'
        saved = read(str(path.relative_to(ROOT))); old = read(str((parent_out/f'gp_head{head}.json').relative_to(ROOT)))
        by_position = {r['position']: r for r in saved['records']}
        for row in old['records']:
            assert by_position[row['position']] == row
        heads.append(saved['records'])
    connected = next(r for r in read(spec['gp_states'])['rows'] if
        (r['layout'], r['arm']) == (spec['layout'], 'source_common_task_rbf'))
    model = restore_saved_model(connected)
    uniforms = qmc.Sobol(2, scramble=True, seed=spec['seed']+1).random_base2(spec['cached_predictive_power'])
    replay = grid_rules(heads, model.weights, connected['terminal'], uniforms, spec)
    for current, saved in zip(replay['rows'], result['rows']):
        for key in ('raw_adaptation_value', 'raw_net_reward_control_variate',
                'quadrature_joint_probability_mass', 'raw_fixed_action_tower_residual'):
            assert abs(current[key]-saved[key]) < 1e-14
        assert abs(current['quadrature_joint_probability_mass']-1.) < 1e-12
    assert replay['numerical_changes'] == result['numerical_changes']
    source = read(str((out/'source_anchor_heads.json').relative_to(ROOT)))
    old_source = read(spec['source_anchor_cache']); assert source == old_source
    anchor_rows = read(str((out/'anchors.json').relative_to(ROOT)))['rows']
    assert source['all8_complete_identities'] == 8
    for power in spec['raw_powers']:
        current = anchor_comparison(*source['heads'], source['source_weights'], source['reference'], power)
        old = next(r for r in anchor_rows if (r['model'], r['power']) == ('learned source mixture', power))
        for key in ('maximum_reward_error', 'maximum_shared_weight_error', 'maximum_cost_error'):
            assert abs(current[key]-old[key]) < 1e-14
    gp_baseline = read(spec['gp_completed_results'])
    fine = replay['rows'][-1]
    gp_joint = gp_baseline['rows'][-1]['raw_net_reward_control_variate']
    result.update(all69_old_head_states_preserved=True, all8_source_identities_and_positive_components_preserved=True,
        source_positive_component_counts=source['retained_positive_components'],
        source_positive_component_total=sum(sum(c) for c in source['retained_positive_components']),
        joint_maximization_probability_mass=float(sum(fine['selected_action_probabilities'])),
        finest_joint_pair_count=fine['joint_fantasy_count'],
        finest_required_head_state_count=fine['distinct_head_state_count'],
        old_joint_rule_raw_kg=gp_joint, head_grid_minus_old_joint_rule=fine['raw_net_reward_control_variate']-gp_joint,
        old_failed_head_grid_diagnostics=parent['numerical_changes'],
        maximum_gp_anchor_reward_error=max(r['maximum_reward_error'] for r in anchor_rows if r['model'] == 'matched GP'),
        maximum_source_anchor_reward_error=max(r['maximum_reward_error'] for r in anchor_rows if r['model'] == 'learned source mixture'),
        successful_tests=tests['tests_run'], all_unit_tests_successful=tests['successful'],
        cpu_seconds_including_parent_and_tests=parent['cpu_seconds']+result['cpu_seconds']+tests['cpu_seconds'],
        cached_replay_cpu_seconds=time.process_time()-started,
        next='Extend cached head statistics to the remaining frozen queries. For the learned source mixture, '
            'reuse its existing per-identity predictive label banks:compose each grid under one origin identity, '
            'weight that complete origin with its source mass,and condition all posterior identities jointly. '
            'Apply existing low-identity utility bounds before expensive integrations. Freeze the full '
            'query policy before independent deployment evaluation.')
    save(out/'phase_results.json', result)
    print({k: result[k] for k in ('all_reference_anchors_passed', 'numerical_gates_passed',
        'finest_joint_pair_count', 'finest_required_head_state_count', 'successful_tests',
        'cpu_seconds_including_parent_and_tests')})
    print(fine['raw_net_reward_control_variate'], result['head_grid_minus_old_joint_rule'], result['numerical_changes'])
    return result


if __name__ == '__main__':
    run()
