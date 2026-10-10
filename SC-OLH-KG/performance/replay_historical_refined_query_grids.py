"""Replay the resolved query0 cohort and verify precisely what was reused."""

import time

import numpy as np

from performance.check_historical_gauss_risk_tail import summarize
from performance.check_historical_query_grids import grid_rules
from performance.complete_historical_source_heads import ROOT, read, save
from performance.refine_historical_objective_quadrature import PLAN


def verify_risk_reuse(original, gauss, refined, selected):
    old = {r['position']: r for r in original['records']+gauss['records']}
    assert refined['actions'] == original['actions'] == gauss['actions']
    assert len(old) == len(refined['records'])
    untouched = sorted(set(range(len(original['actions'])))-set(selected))
    for record in refined['records']:
        before = old[record['position']]
        assert record['primary_mean_integrated']
        for key in ('label', 'head_log_scores', 'utility_identities'):
            assert record[key] == before[key]
        for power in record['raw']:
            for key in ('statistics', 'costs'):
                previous = np.asarray(before['raw'][power][key])[:, untouched]
                current = np.asarray(record['raw'][power][key])[:, untouched]
                assert np.array_equal(previous, current)


def compare_summary(saved, current):
    assert saved['numerical_changes'] == current['numerical_changes']
    assert saved['fixed_action_tower_gate_passed'] == current['fixed_action_tower_gate_passed']
    assert len(saved['rows']) == len(current['rows'])
    for before, after in zip(saved['rows'], current['rows']):
        for key in ('raw_adaptation_value', 'raw_net_reward_control_variate', 'predictive_mass_residual',
                'raw_fixed_action_tower_residual', 'identity_and_rare_origin_error_bound'):
            assert abs(before[key]-after[key]) < 1e-14


def run(plan=PLAN):
    started = time.process_time(); spec = read(plan); parent = read(spec['parent_plan'])
    risk_spec = read(spec['risk_refinement_plan']); original_out = ROOT/parent['output']
    tests = read(str((original_out/'unit_tests.json').relative_to(ROOT))); assert tests['successful']
    cases = []
    for case in read(str((original_out/'results.json').relative_to(ROOT)))['cases']:
        name = case['name']; original = original_out/name
        if name not in risk_spec['case_names']:
            heads = [read(str((original/f'head{h}.json').relative_to(ROOT)))['records'] for h in (0, 1)]
            current = grid_rules(heads, case['source_weights'], case['baseline'], 0, [0], parent)
            assert current['numerical_changes'] == case['numerical_changes']
            assert current['rows'] == case['rows']
            anchors_passed = True; result_path = original/'results.json'
        else:
            risk = ROOT/risk_spec['output']/name
            original_risk = read(str((original/'head1.json').relative_to(ROOT)))
            gauss = read(str((risk/'gauss_heads.json').relative_to(ROOT)))
            refined = read(str((risk/'refined_risk_heads.json').relative_to(ROOT)))
            primary = original_risk['actions'][case['baseline'][2]]['primary']
            selected = [i for i, a in enumerate(original_risk['actions']) if a['kind'] == 'guarded' and a['primary'] == primary]
            verify_risk_reuse(original_risk, gauss, refined, selected)
            labels = read(str((risk/'labels.json').relative_to(ROOT)))
            label_index = {p: y for rule in labels.values() for p, y in zip(rule['positions'], rule['labels'])}
            label_bank = read(str((original/'labels1.json').relative_to(ROOT)))
            assert label_bank['source_weights'] == case['source_weights']
            label_index.update(zip(label_bank['positions'], label_bank['labels']))
            assert all(r['label'] == label_index[r['position']] for r in refined['records'])
            if name == spec['case']:
                output = ROOT/spec['output']; result_path = output/'results.json'
                f = read(str((output/'objective_heads.json').relative_to(ROOT)))['records']
                objective_rules = read(str((output/'objective_rules.json').relative_to(ROOT)))
                label_index = {p: y for rule in objective_rules.values() for p, y in zip(rule['positions'], rule['labels'])}
                assert all(r['label'] == label_index[r['position']] and len(r['head_log_scores']) == len(case['source_weights']) for r in f)
                rule = dict(spec, source_reference=str((original/'results.json').relative_to(ROOT)))
                precheck = read(str((output/'density_precheck.json').relative_to(ROOT)))
                assert precheck['passed'] and precheck['new_terminal_heads_at_precheck'] == 0
                assert all(abs(sum(r['weights'])-1.) < 1e-14 for r in objective_rules.values())
                assert all(a['passed'] for a in read(spec['anchor_results'])['anchors'])
            else:
                result_path = risk/'results.json'; f = read(str((original/'head0.json').relative_to(ROOT)))['records']
                source = case['arm'] == 'source_structural_mixture'
                rule = dict(risk_spec, source_reference=str((original/'results.json').relative_to(ROOT)),
                    proposal_origin=4 if source else 0, predictive_origins=[4, 5] if source else [0])
            saved = read(str(result_path.relative_to(ROOT)))
            current = summarize([f, refined['records']], labels, case['source_weights'], rule,
                objective_rules=objective_rules if name == spec['case'] else None)
            compare_summary(saved, current); anchors_passed = saved['actual_branch_anchors_passed']
        fine = current['rows'][-1]
        gates = {'terminal_and_predictive_resolution': current['numerical_gates_passed'],
            'identity_remainder': max(r['identity_and_rare_origin_error_bound'] for r in current['rows']) <= parent['maximum_identity_integration_error_bound'],
            'predictive_mass': max(abs(r['predictive_mass_residual']) for r in current['rows']) <= parent['maximum_predictive_mass_residual'],
            'density_ratio': all(parent['predictive_density_ratio_limits'][0] <= r['predictive_density_ratio_range'][0] and
                r['predictive_density_ratio_range'][1] <= parent['predictive_density_ratio_limits'][1] for r in current['rows']),
            'actual_branch_anchors': anchors_passed}
        cases.append({'name': name, 'result': str(result_path.relative_to(ROOT)),
            'raw_net_reward_control_variate': fine['raw_net_reward_control_variate'],
            'numerical_changes': current['numerical_changes'], 'gates': gates,
            'adaptation_diagnostics_passed': all(gates.values()),
            'fixed_action_tower_residual': fine['raw_fixed_action_tower_residual'],
            'fixed_action_tower_gate_passed': abs(fine['raw_fixed_action_tower_residual'])+
                fine['identity_and_rare_origin_error_bound'] <= risk_spec['maximum_fixed_action_tower_residual']})
    source_query5 = read('paper_artifacts/historical_gauss_risk_tail_v1/results.json')
    gp_query5 = read('paper_artifacts/historical_raw_head_grid_v2/phase_results.json')
    gp_queries = read('paper_artifacts/historical_matched_gp_extension_v2/phase_results.json')
    done = {(c['layout'], c['arm'], c['query']) for c, latest in zip(
        read(str((original_out/'results.json').relative_to(ROOT)))['cases'], cases) if latest['adaptation_diagnostics_passed']}
    if source_query5['numerical_gates_passed']:
        done.add(('5', 'source_structural_mixture', 5))
    if gp_query5['numerical_gates_passed']:
        done.add(('5', 'source_common_task_rbf', 5))
    if gp_queries['layout6_existing_result_preserved']['numerical_gates_passed']:
        done.add(('6', 'source_common_task_rbf', 1))
    connection_spec = read(parent['connection_plan'])
    states = read(connection_spec['output']+'/connected_states.json')['rows']
    coverage = [{'layout': state['layout'], 'arm': arm,
        'diagnostic_queries': [q for q in state['action_sets']['4']['active_indices'] if (state['layout'], arm, q) in done],
        'remaining_queries': [q for q in state['action_sets']['4']['active_indices'] if (state['layout'], arm, q) not in done]}
        for state in states for arm in ('source_structural_mixture', 'source_common_task_rbf')]
    result = {'execution_plan': plan, 'cases': cases, 'query0_cohort_complete': len(cases) == len(parent['cases']),
        'all_adaptation_diagnostics_passed': all(c['adaptation_diagnostics_passed'] for c in cases),
        'all_fixed_action_tower_gates_passed': all(c['fixed_action_tower_gate_passed'] for c in cases),
        'successful_tests': tests['tests_run'], 'cached_replay_cpu_seconds': time.process_time()-started,
        'frozen_query_coverage': coverage,
        'remaining_model_query_cases': sum(len(r['remaining_queries']) for r in coverage),
        'cost': parent['cost'], 'independent_confirmation': False, 'decision_ranking_certified': False,
        'limitations': ['Adjacent quadrature diagnostics are not strict integration error bounds.',
            'Model-own expected KG values do not compare realized deployment performance.',
            'Other frozen queries,earlier paid projections and independent decision evidence remain.']}
    save(ROOT/spec['output']/'phase_results.json', result)
    print('query0 cached replay', result['all_adaptation_diagnostics_passed'], 'tower', result['all_fixed_action_tower_gates_passed'],
        'CPU', round(result['cached_replay_cpu_seconds'], 3))
    for case in cases:
        print(case['name'], case['raw_net_reward_control_variate'], case['gates'], 'tower', case['fixed_action_tower_gate_passed'])
    return result


if __name__ == '__main__':
    run()
