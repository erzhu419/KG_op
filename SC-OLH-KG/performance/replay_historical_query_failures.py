"""Replay selective refinements and freeze decisions for the two paid states."""

import time

import numpy as np

from performance.check_historical_gauss_risk_tail import summarize
from performance.complete_historical_source_heads import ROOT, read, save
from performance.refine_historical_remaining_query_failures import PLAN


def previous_cases(parent):
    """Use the seven already replayed cases, retaining their artifact paths."""
    previous = read(parent['previous_coverage']); cases = []
    for case in previous['cases']:
        layout = case['name'].split('_layout')[1].split('_')[0]
        arm = case['name'].split('_layout')[0]
        cases.append(dict(case, layout=layout, arm=arm, query=0))
    paths = [('5', 'source_structural_mixture', 5, 'historical_gauss_risk_tail_v1/results.json'),
        ('5', 'source_common_task_rbf', 5, 'historical_raw_head_grid_v2/phase_results.json'),
        ('6', 'source_common_task_rbf', 1, 'historical_matched_gp_extension_v2/phase_results.json')]
    for layout, arm, query, path in paths:
        artifact = read('paper_artifacts/'+path)
        result = artifact['layout6_existing_result_preserved'] if layout == '6' else artifact
        fine = result['rows'][-1]
        remainder = fine['identity_and_rare_origin_error_bound'] if arm == 'source_structural_mixture' else 0.
        cases.append({'name': f'{arm}_layout{layout}_query{query}', 'layout': layout, 'arm': arm,
            'query': query, 'result': 'paper_artifacts/'+path,
            'raw_net_reward_control_variate': fine['raw_net_reward_control_variate'],
            'numerical_changes': result['numerical_changes'],
            'adaptation_diagnostics_passed': result['numerical_gates_passed'],
            'fixed_action_tower_gate_passed': abs(fine['raw_fixed_action_tower_residual'])+
                remainder <= parent['maximum_fixed_action_tower_residual'],
            'raw_fixed_action_tower_residual': fine['raw_fixed_action_tower_residual']})
    assert all(c['adaptation_diagnostics_passed'] for c in cases)
    return cases


def run(plan=PLAN):
    started = time.process_time(); spec = read(plan); parent = read(spec['parent_plan'])
    base = read(parent['parent_plan']); original = ROOT/parent['output']; out = ROOT/spec['output']
    execution = read(str((out/'results.json').relative_to(ROOT)))
    cohort = read(str((original/'phase_results.json').relative_to(ROOT)))
    tests = read(str((out/'unit_tests.json').relative_to(ROOT)))
    assert execution['all_registered_cases_complete'] and cohort['all_registered_cases_complete'] and tests['successful']
    refinements = []
    for case in execution['cases']:
        before = original/case['name']; directory = out/case['name']
        old = read(str((before/'results.json').relative_to(ROOT)))
        bank = read(str((before/'label_banks.json').relative_to(ROOT)))
        f = read(str((before/'head0.json').relative_to(ROOT)))['records']
        g = read(str((directory/'risk_heads.json').relative_to(ROOT)))
        old_g = read(str((before/'head1.json').relative_to(ROOT)))
        assert case['source_weights'] == old['source_weights'] == bank['source_weights']
        assert case['baseline'] == old['baseline'] and case['anchors'] == old['anchors']
        assert g['actions'] == old_g['actions']
        index = {r['position']: r for r in g['records']}
        for record in old_g['records']:
            current = index[record['position']]
            for key in ('label', 'head', 'utility_identities', 'guarded_mean_position'):
                assert current[key] == record[key]
            np.testing.assert_allclose(current['head_log_scores'], record['head_log_scores'], atol=1e-10, rtol=0.)
            assert all(current['raw'][power] == value for power, value in record['raw'].items())
        source = case['arm'] == 'source_structural_mixture'; labels = dict(bank['risk_rules'])
        rule = dict(parent, proposal_origin=4 if source else 0, predictive_origins=[4, 5] if source else [0])
        if source:
            added = read(str((directory/'risk_tail64.json').relative_to(ROOT)))
            assert len(added['positions']) == 64 and min(added['weights']) > 0.
            assert abs(sum(added['weights'])-parent['low_constraint_probability']) < 1e-14
            assert added['maximum_inverse_cdf_error'] <= parent['maximum_inverse_cdf_error']
            assert set(index)-{r['position'] for r in old_g['records']} == set(added['positions'])
            for position, label in zip(added['positions'], added['labels']):
                record = index[position]
                assert record['label'] == label and len(record['head_log_scores']) == len(case['source_weights'])
            labels['64'] = added; rule['risk_tail_orders'] = spec['source_risk_tail_orders']
        else:
            assert len(g['records']) == len(old_g['records'])
            rule['objective_raw_power'] = spec['gp_objective_power']
        for stage in case['stages']:
            powers = list(dict.fromkeys(r['raw_draw_power'] for r in stage['rows']))
            current = summarize([f, g['records']], labels, case['source_weights'], dict(rule, raw_powers=powers),
                objective_rules=bank['objective_rules'], baseline=case['baseline'])
            assert len(current['rows']) == len(stage['rows'])
            for key, value in stage['numerical_changes'].items():
                assert abs(current['numerical_changes'][key]-value) < 1e-14
            for saved, replayed in zip(stage['rows'], current['rows']):
                for key in ('raw_adaptation_value', 'raw_net_reward_control_variate', 'predictive_mass_residual',
                        'raw_fixed_action_tower_residual', 'identity_and_rare_origin_error_bound'):
                    assert abs(saved[key]-replayed[key]) < 1e-14
                np.testing.assert_allclose(saved['selected_action_probabilities'], replayed['selected_action_probabilities'], atol=1e-14, rtol=0.)
            assert current['numerical_gates_passed'] == stage['numerical_gates_passed']
            assert current['fixed_action_tower_gate_passed'] == stage['fixed_action_tower_gate_passed']
        for anchor in case['anchors']['rows']:
            error = float(np.max(abs(np.asarray(anchor['actual'])-anchor['full_reference'])))
            assert anchor['query'] == case['query'] and abs(error-anchor['error']) < 1e-14
            assert anchor['passed'] == (error <= parent['maximum_anchor_branch_error'])
        gates = {'terminal_and_predictive_resolution': current['numerical_gates_passed'],
            'identity_remainder': max(r['identity_and_rare_origin_error_bound'] for r in current['rows']) <= base['maximum_identity_integration_error_bound'],
            'predictive_mass': max(abs(r['predictive_mass_residual']) for r in current['rows']) <= base['maximum_predictive_mass_residual'],
            'density_ratio': all(base['predictive_density_ratio_limits'][0] <= r['predictive_density_ratio_range'][0] and
                r['predictive_density_ratio_range'][1] <= base['predictive_density_ratio_limits'][1] for r in current['rows']),
            'native_anchors': all(a['passed'] for a in case['anchors']['rows'])}
        assert gates == case['gates'] and all(gates.values()) == case['adaptation_diagnostics_passed']
        fine = current['rows'][-1]
        refinements.append({'name': case['name'], 'layout': case['layout'], 'arm': case['arm'], 'query': case['query'],
            'result': str((directory/'results.json').relative_to(ROOT)), 'gates': gates,
            'numerical_changes': current['numerical_changes'], 'adaptation_diagnostics_passed': all(gates.values()),
            'raw_net_reward_control_variate': fine['raw_net_reward_control_variate'],
            'fixed_action_tower_gate_passed': current['fixed_action_tower_gate_passed'],
            'raw_fixed_action_tower_residual': fine['raw_fixed_action_tower_residual'],
            'objective_raw_draw_power': fine.get('objective_raw_draw_power', fine['raw_draw_power']),
            'risk_raw_draw_power': fine['raw_draw_power']})
    replacements = {c['name']: c for c in refinements}
    cases = previous_cases(parent)+[replacements.get(c['name'], dict(c,
        result=str((original/c['name']/'results.json').relative_to(ROOT)))) for c in cohort['cases']]
    assert len(cases) == len({c['name'] for c in cases}) == 24
    connection = read(base['connection_plan']); state_path = connection['output']+'/connected_states.json'
    states = read(state_path)['rows']; terminal = read(base['terminal_results'])['rows']
    gp_terminal = read('paper_artifacts/historical_matched_gp_extension_v2/phase_results.json')['terminal_summary']
    coverage = []; policies = []
    for state in states:
        for arm in ('source_structural_mixture', 'source_common_task_rbf'):
            queries = state['action_sets']['4']['active_indices']
            values = [next(c for c in cases if (c['layout'], c['arm'], c['query']) == (state['layout'], arm, q)) for q in queries]
            passed = [c['query'] for c in values if c['adaptation_diagnostics_passed']]
            coverage.append({'layout': state['layout'], 'arm': arm, 'diagnostic_queries': passed,
                'remaining_queries': [q for q in queries if q not in passed]})
            assert len(passed) == len(queries)
            best = max(values, key=lambda c: c['raw_net_reward_control_variate']); net = best['raw_net_reward_control_variate']
            if arm == 'source_structural_mixture':
                stop = next(r for r in terminal if (r['layout'], r['remaining_calls'], r['power']) == (state['layout'], 360, 14))['best_action']
                model = {'artifact': state_path, 'row_layout': state['layout'], 'snapshot': 'posterior'}
            else:
                stop = next(r for r in gp_terminal if (r['layout'], r['arm']) == (state['layout'], arm))['by_budget']['360']['action']
                model = {'artifact': base['gp_states'], 'row_layout': state['layout'], 'row_arm': arm, 'snapshot': 'posterior'}
            policies.append({'layout': state['layout'], 'arm': arm, 'model': model,
                'paid_observations': state['paid_observations'], 'remaining_calls': state['remaining_calls'],
                'terminal_protocol': state['terminal_protocol'], 'stop_action': stop,
                'decision': 'query' if net > 0. else 'stop', 'selected_query': best['query'] if net > 0. else None,
                'maximum_estimated_net_kg': net, 'query_results': values,
                'after_query': {'remaining_calls': state['future_remaining_calls'],
                    'actions': state['future_terminal_actions'], 'likelihood': 'raw joint F/G likelihood; shared posterior identity weights',
                    'recommendation': 'core.positive_raw_terminal.raw_terminal_values', 'guarded_mean_position': 0},
                'decision_ranking_certified': False, 'independent_confirmation': False})
    result = {'execution_plan': plan, 'cases': cases, 'selective_refinements': refinements,
        'frozen_query_coverage': coverage, 'remaining_model_query_cases': sum(len(c['remaining_queries']) for c in coverage),
        'diagnostic_model_query_case_count': sum(len(c['diagnostic_queries']) for c in coverage),
        'all_adaptation_diagnostics_passed': all(c['adaptation_diagnostics_passed'] for c in cases),
        'raw_fixed_action_tower_hold_cases': [c['name'] for c in cases if not c['fixed_action_tower_gate_passed']],
        'original_17_case_phase': parent['output']+'/phase_results.json',
        'cohort_cpu_seconds': cohort['cpu_seconds'], 'selective_refinement_cpu_seconds': execution['cpu_seconds'],
        'cached_replay_cpu_seconds': time.process_time()-started, 'successful_tests': tests['tests_run'],
        'cost': base['cost'], 'independent_confirmation': False, 'decision_ranking_certified': False,
        'limitations': ['Resolution and native-point diagnostics are not strict integration error bounds.',
            'These four decisions apply to the two saved development states; independent targets require their own posterior decisions.',
            'Raw fixed-action tower HOLD cases and earlier paid moment projections remain.',
            'Model-own KG values do not establish independent deployment superiority.']}
    save(out/'phase_results.json', result)
    save(out/'frozen_single_step_policies.json', {'selection_rule': 'argmax estimated net KG in frozen query order; stop if maximum <=0',
        'scope': 'Two saved development states, two models; all24 registered adaptation diagnostics pass.',
        'source_fit_plan': base['connection_plan'], 'coverage_result': str((out/'phase_results.json').relative_to(ROOT)),
        'policies': policies, 'decision_ranking_certified': False, 'independent_confirmation': False})
    print('cached replay', result['diagnostic_model_query_case_count'], '/24; raw tower HOLD', len(result['raw_fixed_action_tower_hold_cases']),
        '; CPU', round(result['cached_replay_cpu_seconds'], 3))
    for policy in policies:
        print(policy['layout'], policy['arm'], policy['decision'], policy['selected_query'], policy['maximum_estimated_net_kg'])
    return result


if __name__ == '__main__':
    run()
