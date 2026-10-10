"""Replay the completed joint audit and one actual accepted fixed source update."""

import time
from collections import defaultdict

from performance.complete_historical_source_heads import ROOT, read, save
from performance.confirm_historical_joint_source_stationarity import KEY
from performance.continue_historical_raw_source_learning import load_mixture


def key(row):
    return tuple(row[k] for k in KEY)


def run():
    started = time.process_time()
    audit_path = 'paper_artifacts/historical_joint_source_stationarity_v1/results.json'
    update_path = 'paper_artifacts/historical_pending_joint_update_v1/results.json'
    base = read('paper_artifacts/historical_final_source_closure_v1/phase_results.json')
    audit = read(audit_path); update = read(update_path); assert audit['all_registered_cases_complete']
    checks = {key(r): r for r in audit['rows']}; rows = []; complete = defaultdict(set)
    out = ROOT / 'paper_artifacts/historical_joint_source_stationarity_v1'
    for original in base['rows']:
        row = dict(original); diagnostic = checks[key(row)]
        row['current_joint_stationarity_diagnostic_passed'] = diagnostic['current_joint_stationarity_diagnostic_passed']
        row['two_consecutive_actual_law_joint_stationarity_confirmed'] = diagnostic['two_consecutive_actual_law_joint_stationarity_confirmed']
        row['joint_stationarity_results'] = str((out / 'cases' / '_'.join(key(row)) / 'results.json').relative_to(ROOT))
        if key(row) == key(update) and update['accepted_fixed_risk_update']:
            assert row['final_risk_scales'] == update['checked_current_scales']
            row.update(final_risk_scales=update['final_risk_scales'], conditional_risk_stationarity_passed=True,
                same_law_precision_confirmation_passed=False,
                candidate_component_file=update['candidate_component_file'],
                source_evidence=update['final_vs_selected_parent_evidence'],
                selected_evidence_results=update_path, evidence_results=update_path,
                source_evidence_method='original evidence plus previously qualified fixed update',
                posterior_results=update_path, posterior_method='directly certified reweighted retained mixture;no further reduction',
                response_stationarity_at_final_risk=update['new_response_diagnostic'],
                risk_stationarity_certificate=update_path, joint_stationarity_results=update_path,
                two_consecutive_actual_law_joint_stationarity_confirmed=True)
            assert update['response_posterior_summary']['monte_carlo_gate_passed']
            assert row['source_evidence']['precision_gate_passed'] and row['source_evidence']['eligible_improvement_excludes_zero']
            input_path = 'paper_artifacts/historical_pending_joint_update_v1/current_candidate_input.json'
            save(ROOT / input_path, {**{k: row[k] for k in KEY}, 'final_risk_scales': row['final_risk_scales'],
                'fixed_response_scales': row['fixed_response_scales'], 'conditional_risk_stationarity_passed': True,
                'actual_risk_results': update_path, 'risk_stationarity_certificate': update_path,
                'joint_source_learning_converged': False})
            row['candidate_results'] = input_path
        if row['two_consecutive_actual_law_joint_stationarity_confirmed']:
            complete[key(row)[:-1]].add(row['head'])
        rows.append(row)
    learning_path = update['qualified_source_learning_results'] if update['accepted_fixed_risk_update'] else base['qualified_source_learning_results']
    precision_path = update['qualified_source_precision_results'] if update['accepted_fixed_risk_update'] else base['qualified_source_precision_results']
    learning = read(learning_path); precision = read(precision_path)
    for case, certificate, row in zip(learning['evidence_rows'], precision['rows'], rows):
        assert key(case) == key(certificate) == key(row)
        assert case['eta_common_task_scales'] == row['final_risk_scales']
        assert case['f_common_task_scales'] == row['fixed_response_scales']
        eta, _ = load_mixture(certificate, case); assert eta.shape[0] * eta.shape[1] == row['mixture_components']
    remaining = [{k: r[k] for k in (*KEY, 'joint_stationarity_results')}
        for r in rows if not r['two_consecutive_actual_law_joint_stationarity_confirmed']]
    result = {'rows': rows, 'all_registered_cases_complete': True, 'registered_cases': len(rows),
        'source_audit_results': audit_path, 'accepted_fixed_update_results': update_path,
        'current_joint_stationarity_cases_passed': sum(r['current_joint_stationarity_diagnostic_passed'] for r in rows),
        'consecutive_actual_law_joint_stationarity_cases_confirmed': sum(r['two_consecutive_actual_law_joint_stationarity_confirmed'] for r in rows),
        'complete_joint_stationary_source_identities': sum(heads == {'f', 'g'} for heads in complete.values()),
        'final_posterior_and_evidence_cases_certified': sum(r['final_candidate_numerically_certified'] for r in rows),
        'remaining_joint_stationarity_cases': remaining, 'joint_source_learning_converged': not remaining,
        'new_accepted_fixed_risk_updates': int(update['accepted_fixed_risk_update']),
        'qualified_source_learning_results': learning_path, 'qualified_source_precision_results': precision_path,
        'existing_source_interface_cases_loaded': len(rows),
        'unit_test_results': 'paper_artifacts/historical_joint_source_stationarity_v1/unit_tests.json',
        'recorded_CPU_seconds': audit['recorded_CPU_seconds'] + update['cpu_seconds'],
        'recorded_CPU_is_lower_bound': True,
        'execution_incident': 'The fixed-update execution first stopped because its manifest omitted the existing evidence-precision settings. Restored the registered settings and replayed the same seed;failed-attempt CPU was not recorded.',
        'completed_candidate_MCMC_sweeps': audit['completed_candidate_MCMC_sweeps'] + update['completed_candidate_MCMC_sweeps'],
        'compact_replayed_risk_statistics_files': sum(r['current_risk_sufficient_statistics'] is not None for r in audit['rows']) + 1,
        'new_candidate_MCMC_sweeps_executed_on_replay': 0, 'cached_replay_CPU_seconds': time.process_time() - started,
        'original_source_cases_resampled': 0, 'new_source_labels': 0, 'new_target_simulator_calls': 0,
        'new_verifier_calls': 0, 'server_jobs': 0, 'source_registry_replaced': False, 'independent_confirmation': False,
        'limitations': ['Four current laws meet single-point joint diagnostics but lack a second qualified actual law;unaccepted proposals remain HOLD.',
            'Joint confirmations concern registered local score/covariance thresholds on actual risk-update trajectories,not global optimality or physical-factor recovery.',
            'Cross-identity absolute evidence and independent matched-control target decisions remain open.'],
        'next_work': 'Reuse the saved compact statistics and actual candidate registries. Resolve the remaining four actual-law confirmations without favorable-seed search;complete cross-identity source evidence and matched-control cached target comparisons.'}
    save(out / 'phase_results.json', result)
    print('Joint source confirmation', result['consecutive_actual_law_joint_stationarity_cases_confirmed'], '/32;',
        result['current_joint_stationarity_cases_passed'], 'current joint diagnostics;source interfaces', len(rows), flush=True)
    return result


if __name__ == '__main__':
    run()
