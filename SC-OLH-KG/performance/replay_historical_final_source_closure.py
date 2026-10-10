"""Assemble final source candidates from cached certificates without sampling."""

import time
from collections import Counter, defaultdict

import numpy as np

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.retain_historical_stationary_risk_cases import retain_followup_risk
from performance.continue_historical_raw_source_learning import load_mixture

PLAN = 'performance/manifests/historical_final_source_closure_v1.json'
KEY = (*IDENTITY, 'head')


def key(row):
    return tuple(row[k] for k in KEY)


def run(plan=PLAN):
    started = time.process_time(); spec = read(plan)
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    batch = read(spec['final_source_results']); bridge = read(spec['bridge_results'])
    assert batch['all_registered_cases_complete'] and bridge['all_registered_cases_complete']
    assert len(batch['rows']) == len({key(r) for r in batch['rows']}) == 32
    bridged = {key(r): r for r in bridge['rows']}
    same_law_confirmed = {key(r) for r in read(spec['same_law_precision_results'])['rows']
        if r['stationary_current_law_held']}
    assert set(bridged) == {key(r) for r in batch['rows'] if not r['final_vs_parent_evidence_certified']}
    previous = read(spec['last_risk_update']); diagnostic = read(spec['last_risk_diagnostic'])
    assert key(previous) == key(diagnostic)
    retention = {**{k: previous[k] for k in KEY}, **retain_followup_risk(previous, diagnostic, spec),
        'previous_actual_update': spec['last_risk_update'], 'final_source_diagnostic': spec['last_risk_diagnostic']}
    hold_path = str((out / 'last_risk_retention.json').relative_to(ROOT)); save(ROOT / hold_path, retention)
    final_spec = read(spec['final_source_batch']); components = Counter(); complete = defaultdict(set)
    parent_cases = read(spec['parent_learning'])['evidence_rows']
    rows = []; cases = []; certified = []; heads = []
    evidence_cpu = direct_cpu = conditioning_cpu = 0.; posterior_sweeps = evidence_sweeps = 0
    for item, original in zip(final_spec['cases'], batch['rows']):
        candidate = read(item['candidate']); assert key(candidate) == key(original)
        row = dict(original); direct_evidence = read(row['evidence_results'])
        evidence = bridged.get(key(row), direct_evidence)
        assert evidence['final_vs_parent_evidence_certified'] == (
            evidence['source_evidence']['precision_gate_passed'] and
            (evidence['source_evidence']['eligible_improvement_excludes_zero'] or
             evidence.get('unchanged_source_law', False)))
        if key(row) == key(previous):
            assert row['final_risk_scales'] == retention['held_risk_scales']
            row['conditional_risk_stationarity_passed'] = True
            row['risk_stationarity_certificate'] = hold_path
        else:
            row['risk_stationarity_certificate'] = candidate['risk_stationarity_certificate']
        row['same_law_precision_confirmation_passed'] = key(row) in same_law_confirmed
        if row['same_law_precision_confirmation_passed']:
            row['conditional_risk_stationarity_passed'] = False
        row['source_evidence'] = evidence['source_evidence']
        row['source_evidence_method'] = 'chronological source anchors' if key(row) in bridged else 'full parent bank'
        row['selected_evidence_results'] = (read(evidence['execution_plan'])['output'] + '/results.json'
            if key(row) in bridged else row['evidence_results'])
        row['source_evidence_precision_passed'] = evidence['source_evidence']['precision_gate_passed']
        row['final_vs_parent_evidence_certified'] = evidence['final_vs_parent_evidence_certified']
        row['final_candidate_numerically_certified'] = bool(row['posterior_functional_and_retention_certified']
            and row['final_vs_parent_evidence_certified'])
        component = read(row['candidate_component_file']); fixed = component['fixed_candidate']
        assert key(fixed) == key(row)
        assert fixed['eta_common_task_scales'] == row['final_risk_scales']
        assert fixed['f_common_task_scales'] == row['fixed_response_scales']
        certificate = {**{k: row[k] for k in KEY}, 'positive_mixture_precision_passed': True,
            'positive_mixture_component_file': row['candidate_component_file'],
            'positive_weights_results': row['candidate_component_file'],
            'within_identity_evidence_results': row['selected_evidence_results'],
            'posterior_functional_and_retention_results': row['posterior_results']}
        parent = next(c for c in parent_cases if key(c) == key(row))
        case = {**{k: row[k] for k in KEY}, 'source_case': parent['source_case'],
            'f_common_task_scales': row['fixed_response_scales'], 'eta_common_task_scales': row['final_risk_scales']}
        eta, weights = load_mixture(certificate, case)
        n = eta.shape[0] * eta.shape[1]
        assert eta.shape[2] == 48 and weights.size == n and np.all(np.isfinite(weights)) and np.all(weights > 0.)
        np.testing.assert_allclose(weights.sum(), 1., rtol=1e-12)
        expected_n = 512 if (row['layout'], row['selector'], row['support'], row['treatment']) == (
            '6', 'spectral', 'head_specific', 'deconvolved_task') else 1024
        assert n == expected_n; row['mixture_components'] = n; components[str(n)] += 1
        certificate['positive_mixture_component_count'] = n
        certificate['retention_scope'] = row['posterior_method']
        case.update(candidate_component_file=row['candidate_component_file'],
            new_positive_eta_weights=weights.ravel().tolist(), evidence_ratio_precision_passed=row['source_evidence_precision_passed'],
            log_source_evidence_ratio_new_vs_previous=row['source_evidence']['log_source_evidence_ratio'],
            conditional_risk_stationarity_passed=row['conditional_risk_stationarity_passed'],
            posterior_certificate_results=row['posterior_results'], evidence_certificate_results=row['selected_evidence_results'])
        for head, scales in ((row['head'], row['fixed_response_scales']),
            ('eta_g' if row['head'] == 'g' else 'eta_f', row['final_risk_scales'])):
            heads.append({**{k: row[k] for k in IDENTITY}, 'head': head,
                'common_task_covariance_scales': scales,
                'consecutive_source_stationarity_iterations': 0,
                'joint_stationarity_iteration_count_reset': True,
                'source_case_certificate': row['posterior_results'] if head == row['head'] else row['risk_stationarity_certificate']})
        cases.append(case); certified.append(certificate)
        if row['final_candidate_numerically_certified']:
            complete[key(row)[:-1]].add(row['head'])
        direct = read(read(item['direct_plan'])['output'] + '/results.json')
        direct_cpu += direct['cpu_seconds']; evidence_cpu += direct_evidence['cpu_seconds']
        evidence_sweeps += direct_evidence['completed_candidate_MCMC_sweeps']
        if row['posterior_method'] == 'fresh fixed-candidate conditioning':
            posterior = read(row['posterior_results']); conditioning_cpu += posterior['cpu_seconds']
            posterior_sweeps += posterior['new_candidate_MCMC_sweeps']
        rows.append(row)
    risk_batches = [read(p) for p in spec['recorded_risk_batches']]
    risk_cpu = sum(r.get('recorded_CPU_seconds', r.get('recorded_risk_learning_CPU_seconds', 0.)) for r in risk_batches)
    risk_sweeps = sum(r['completed_candidate_MCMC_sweeps'] for r in risk_batches)
    counts = {field: sum(bool(r[field]) for r in rows) for field in (
        'conditional_risk_stationarity_passed', 'source_evidence_precision_passed',
        'final_vs_parent_evidence_certified', 'posterior_functional_and_retention_certified',
        'final_candidate_numerically_certified')}
    registry_created = counts['final_candidate_numerically_certified'] == 32
    learning_path = str((out / 'source_candidates.json').relative_to(ROOT))
    precision_path = str((out / 'source_precision.json').relative_to(ROOT))
    if registry_created:
        save(ROOT / learning_path, {'execution_plan': plan, 'rows': heads, 'evidence_rows': cases,
            'all_registered_cases_complete': True, 'source_learning_converged': False,
            'joint_stationarity_iterations_counted': 0, 'independent_confirmation': False,
            'scope': 'Actual numerically certified final candidates for source continuation. Joint stationarity counts start afresh;conditional-risk iterations are not transferred to joint convergence.'})
        save(ROOT / precision_path, {'execution_plan': plan, 'rows': certified,
            'source_posterior_precision_cases_passed': 32, 'complete_learned_source_identities': 16,
            'all_registered_cases_complete': True, 'source_learning_converged': False,
            'scope': 'Actual final mixtures loaded by the existing source-learning interface;no target model adoption.'})
    result = {'execution_plan': plan, 'rows': rows, 'all_registered_cases_complete': True,
        'registered_cases': 32, 'case_counts': counts, 'source_mixture_component_counts': dict(components),
        'same_law_prefix_precision_confirmations': len(same_law_confirmed),
        'complete_numerically_certified_source_identities': sum(heads == {'f', 'g'} for heads in complete.values()),
        'single_step_response_stationarity_cases_passed': sum(r['response_stationarity_at_final_risk'][
            'single_iteration_stationarity_passed'] for r in rows),
        'remaining_evidence_cases': [{k: r[k] for k in (*KEY, 'selected_evidence_results', 'source_evidence')}
            for r in rows if not r['final_vs_parent_evidence_certified']],
        'recorded_this_stage_CPU_seconds': risk_cpu + evidence_cpu + direct_cpu + conditioning_cpu + bridge['cpu_seconds'],
        'CPU_components': {'risk_precision_and_continuation': risk_cpu, 'full_parent_evidence': evidence_cpu,
            'retained_posterior_checks': direct_cpu, 'fresh_final_posterior_conditioning': conditioning_cpu,
            'chronological_evidence_bridges': bridge['cpu_seconds']},
        'recorded_completed_candidate_MCMC_sweeps': risk_sweeps + evidence_sweeps + posterior_sweeps
            + bridge['completed_candidate_MCMC_sweeps'],
        'CPU_total_is_recorded_lower_bound': True,
        'execution_incident': 'First bridge attempt stopped after sampling because a list was passed to an array score calculation. No valid result was saved. Corrected argument type and replayed registered seeds;failed-attempt CPU was not recorded.',
        'current_certified_source_registry': spec['current_certified_source_registry'],
        'qualified_candidate_registry_created': registry_created,
        'qualified_source_learning_results': learning_path if registry_created else None,
        'qualified_source_precision_results': precision_path if registry_created else None,
        'existing_source_interface_cases_loaded': len(cases),
        'source_registry_replaced': False, 'joint_source_learning_converged': False,
        'new_candidate_MCMC_sweeps_executed_on_replay': 0, 'original_source_cases_resampled': 0,
        'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0,
        'server_jobs': 0, 'independent_confirmation': False,
        'unit_test_results': spec['unit_tests'], 'cached_replay_CPU_seconds': time.process_time() - started,
        'limitations': ['Risk stationarity is conditional on the fixed response covariance;final response checks are single-step diagnostics.',
            'Five same-law overlapping-prefix confirmations validate local precision but do not count as a second optimization iteration.',
            'Within-identity numerical evidence ratios do not provide cross-identity absolute source evidence.',
            'No matched full-GP or independent target decision comparison has adopted these candidates.'],
        'next_work': 'Confirm joint source stationarity and cross-identity evidence,then connect the actual qualified mixtures to matched-control cached target comparisons.'}
    save(out / 'phase_results.json', result)
    print('Cached final source closure', counts, 'complete source identities',
        result['complete_numerically_certified_source_identities'], flush=True)
    return result


if __name__ == '__main__':
    run()
