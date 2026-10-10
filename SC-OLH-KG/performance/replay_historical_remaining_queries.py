"""Replay all new query stages and merge their coverage with the seven old cases."""

import time

import numpy as np

from performance.check_historical_gauss_risk_tail import summarize
from performance.check_historical_remaining_queries import PLAN, density_gate
from performance.complete_historical_source_heads import ROOT, read, save


def run(plan=PLAN):
    started = time.process_time(); spec = read(plan); parent = read(spec['parent_plan']); out = ROOT/spec['output']
    execution = read(str((out/'results.json').relative_to(ROOT))); tests = read(str((out/'unit_tests.json').relative_to(ROOT)))
    assert tests['successful']; cases = []
    for case in execution['cases']:
        directory = out/case['name']; source = case['arm'] == 'source_structural_mixture'
        identities, proposal = ([4, 5], 4) if source else ([0], 0)
        bank = read(str((directory/'label_banks.json').relative_to(ROOT)))
        assert bank['source_weights'] == case['source_weights']
        density = density_gate(bank, proposal, identities, parent, spec)
        assert density == read(str((directory/'density_precheck.json').relative_to(ROOT)))
        assert density['passed'] == case['density_precheck_passed']
        if not density['passed']:
            cases.append({'name': case['name'], 'layout': case['layout'], 'arm': case['arm'], 'query': case['query'],
                'adaptation_diagnostics_passed': False, 'reason': 'density_precheck_failed'})
            continue
        heads = []
        for head in (0, 1):
            saved = read(str((directory/f'head{head}.json').relative_to(ROOT)))
            labels = bank['heads'][head]; index = {p: i for i, p in enumerate(labels['positions'])}
            assert len(saved['records']) == len(labels['positions'])
            for record in saved['records']:
                i = index[record['position']]
                assert record['label'] == labels['labels'][i] and record['guarded_mean_position'] == 0
                assert record['utility_identities'] == identities and len(record['head_log_scores']) == len(case['source_weights'])
                assert max(abs(record['head_log_scores'][k]-labels['predictive_log_densities'][str(k)][i]) for k in identities) < 1e-8
            heads.append(saved['records'])
        for stage in case['stages']:
            current = summarize(heads, bank['risk_rules'], case['source_weights'],
                dict(spec, raw_powers=stage['raw_powers'], proposal_origin=proposal, predictive_origins=identities),
                objective_rules=bank['objective_rules'], baseline=case['baseline'])
            for key, value in stage['numerical_changes'].items():
                assert abs(current['numerical_changes'][key]-value) < 1e-14
            for before, after in zip(stage['rows'], current['rows']):
                for key in ('raw_adaptation_value', 'raw_net_reward_control_variate', 'predictive_mass_residual',
                        'raw_fixed_action_tower_residual', 'identity_and_rare_origin_error_bound'):
                    assert abs(before[key]-after[key]) < 1e-14
                np.testing.assert_allclose(before['selected_action_probabilities'], after['selected_action_probabilities'], atol=1e-14, rtol=0.)
            assert current['numerical_gates_passed'] == stage['numerical_gates_passed']
        anchors = case.get('anchors', {'rows': [], 'all_passed': False})
        for a in anchors['rows']:
            assert a['query'] == case['query']
            error = float(np.max(abs(np.asarray(a['actual'])-a['full_reference'])))
            assert abs(error-a['error']) < 1e-14 and a['passed'] == (error <= spec['maximum_anchor_branch_error'])
        latest = case['stages'][-1]; passed = latest['grid_diagnostics_passed'] and anchors['all_passed']
        assert passed == case.get('adaptation_diagnostics_passed', False)
        fine = latest['rows'][-1]
        cases.append({'name': case['name'], 'layout': case['layout'], 'arm': case['arm'], 'query': case['query'],
            'raw_net_reward_control_variate': fine['raw_net_reward_control_variate'],
            'numerical_changes': latest['numerical_changes'], 'grid_gates': latest['gates'],
            'native_anchor_gate_passed': anchors['all_passed'], 'adaptation_diagnostics_passed': passed,
            'fixed_action_tower_gate_passed': latest['fixed_action_tower_gate_passed'],
            'raw_fixed_action_tower_residual': fine['raw_fixed_action_tower_residual'],
            'terminal_refinement_performed': len(case['stages']) > 1})
    previous = read(spec['previous_coverage']); coverage = []
    for old in previous['frozen_query_coverage']:
        matched = [c for c in cases if (c['layout'], c['arm']) == (old['layout'], old['arm'])]
        passed = set(old['diagnostic_queries']) | {c['query'] for c in matched if c['adaptation_diagnostics_passed']}
        pending = [q for q in old['remaining_queries'] if q not in passed]
        coverage.append(dict(old, diagnostic_queries=sorted(passed), remaining_queries=pending))
    remaining = sum(len(c['remaining_queries']) for c in coverage)
    result = {'execution_plan': execution['execution_plan'], 'replay_plan': plan, 'cases': cases, 'frozen_query_coverage': coverage,
        'remaining_model_query_cases': remaining, 'diagnostic_model_query_case_count': 24-remaining,
        'all_initial_cases_complete': execution['all_initial_cases_complete'],
        'all_registered_cases_complete': execution['all_registered_cases_complete'],
        'successful_tests': tests['tests_run'], 'cpu_seconds': execution['cpu_seconds'],
        'cached_replay_cpu_seconds': time.process_time()-started, 'cost': parent['cost'],
        'independent_confirmation': False, 'decision_ranking_certified': False,
        'limitations': ['Resolution and native-point diagnostics are not strict global integration error bounds.',
            'Raw fixed-action tower failures remain separate from the known-expectation control variate.',
            'These are model-own development queries;independent deployment benefit is not evaluated.']}
    save(out/'phase_results.json', result)
    print('query coverage', result['diagnostic_model_query_case_count'], '/24, remaining', remaining,
        'CPU', round(execution['cpu_seconds'], 2), 'replay CPU', round(result['cached_replay_cpu_seconds'], 3))
    for case in cases:
        print(case['name'], case.get('raw_net_reward_control_variate'), case['adaptation_diagnostics_passed'], case.get('numerical_changes', case.get('reason')))
    return result


if __name__ == '__main__':
    run()
