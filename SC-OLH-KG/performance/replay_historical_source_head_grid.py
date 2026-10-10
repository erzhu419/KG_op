"""Replay the actual meta-prior grid using only saved head statistics."""

import time

from performance.check_historical_source_head_grid import PLAN, grid_rules, anchors_for, baseline_for
from performance.complete_historical_source_heads import ROOT, read, save


def run(plan=PLAN):
    started = time.process_time(); spec = read(plan); out = ROOT/spec['output']
    result = read(str((out/'results.json').relative_to(ROOT))); assert result['all_registered_cases_complete']
    tests = read(str((out/'unit_tests.json').relative_to(ROOT))); assert tests['successful']
    bank = read(spec['predictive_heads']); weights = result['source_weights']; assert weights == bank['identity_weights']
    heads = {}
    reference_actions = read(spec['raw_reference'])['actions']
    for origin in spec['predictive_origins']:
        for head in (0, 1):
            current = read(str((out/f'origin{origin}_head{head}.json').relative_to(ROOT)))
            assert current['actions'] == reference_actions
            labels = next(r['predictive_labels'] for r in bank['heads'] if (r['identity'], r['head']) == (origin, head))
            for record in current['records']:
                assert record['label'] == labels[record['position']]
                assert len(record['head_log_scores']) == len(weights) == 8
                assert record['utility_identities'] == spec['utility_identities']
                for raw in record['raw'].values():
                    assert len(raw['statistics']) == len(weights) and len(raw['statistics'][0]) == 40
            heads[origin, head] = current['records']
    replay = grid_rules(heads, weights, baseline_for(spec), spec)
    for current, saved in zip(replay['rows'], result['rows']):
        for key in ('raw_adaptation_value', 'raw_net_reward_control_variate', 'quadrature_joint_probability_mass',
                'identity_and_rare_origin_error_bound', 'raw_fixed_action_tower_residual'):
            assert abs(current[key]-saved[key]) < 1e-14
        assert abs(current['quadrature_joint_probability_mass']+current['rare_predictive_origin_mass']-1.) < 1e-14
    assert len(replay['rows']) == len(result['rows'])
    assert replay['numerical_changes'] == result['numerical_changes']
    anchors = anchors_for(heads, weights, spec); assert anchors == read(str((out/'anchors.json').relative_to(ROOT)))
    assert anchors['all_anchors_passed']
    fine = replay['rows'][-1]; component_count = sum(sum(c) for c in result['source_positive_component_counts'])
    previous = read(spec['raw_reference'])
    gp = read(spec['gp_grid_results'])
    result.update(source_positive_component_total=component_count, successful_tests=tests['tests_run'],
        all_unit_tests_successful=tests['successful'],
        finest_joint_pair_count=fine['joint_fantasy_count'],
        finest_required_head_state_count=fine['distinct_head_state_count'],
        retained_cached_head_state_count=sum(result['cached_head_counts'].values()),
        source_anchor_head_states_reused=2, cached_raw_reference_futures_preserved=len(previous['records']),
        old_joint_rule_raw_kg=read('paper_artifacts/historical_primary_mean_terminal_v1/phase_results.json')['rows'][-1]['raw_net_reward_control_variate'],
        gp_own_model_raw_kg=gp['rows'][-1]['raw_net_reward_control_variate'],
        maximum_reference_reward_error=max(r['maximum_reward_error'] for r in anchors['rows']),
        maximum_reference_shared_weight_error=max(r['maximum_shared_weight_error'] for r in anchors['rows']),
        cpu_seconds_including_tests=result['cpu_seconds']+tests['cpu_seconds'],
        cached_replay_cpu_seconds=time.process_time()-started,
        next='Freeze and evaluate the remaining query policies on the same source and GP information contract;'
            'then compare independent deployment outcomes.Model-own KG values are not performance differences.')
    if 'maximum_fixed_action_tower_residual' in spec:
        result['fixed_action_tower_gate_passed'] = (abs(fine['raw_fixed_action_tower_residual'])+
            fine['identity_and_rare_origin_error_bound'] <= spec['maximum_fixed_action_tower_residual'])
    save(out/'phase_results.json', result)
    print({k: result[k] for k in ('all_reference_anchors_passed', 'numerical_gates_passed',
        'identity_remainder_gate_passed', 'finest_joint_pair_count', 'finest_required_head_state_count',
        'successful_tests', 'cpu_seconds_including_tests')})
    print(fine['raw_net_reward_control_variate'], fine['raw_net_control_variate_identity_interval'], result['numerical_changes'])
    return result


if __name__ == '__main__':
    run()
