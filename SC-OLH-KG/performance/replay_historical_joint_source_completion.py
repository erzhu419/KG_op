"""Load the completed actual-law source certificates without running a sampler."""

import time

from performance.complete_historical_source_heads import ROOT, read, save
from performance.confirm_historical_joint_source_stationarity import KEY
from performance.continue_historical_raw_source_learning import load_mixture


def key(row):
    return tuple(row[k] for k in KEY)


def run():
    started = time.process_time()
    base_path = 'paper_artifacts/historical_previous_source_laws_v1/phase_results.json'
    update_path = 'paper_artifacts/historical_last_joint_precision_v1/results.json'
    base = read(base_path)
    update = read(update_path)
    rows = [dict(r) for r in base['rows']]
    row = next(r for r in rows if key(r) == key(update))
    accepted = update['accepted_fixed_risk_update']
    if accepted:
        assert row['final_risk_scales'] == update['checked_current_scales']
        assert row['fixed_response_scales'] == update['fixed_response_scales']
        assert update['fixed_proposal_qualified'] and min(update['full_and_split_Q_improvements']) > 0.
        assert update['old_risk_diagnostic']['stationarity_diagnostic_passed']
        assert update['new_risk_diagnostic']['stationarity_diagnostic_passed']
        assert update['new_response_diagnostic']['single_iteration_stationarity_passed']
        assert update['response_posterior_summary']['monte_carlo_gate_passed']
        assert update['final_vs_selected_parent_evidence']['eligible_improvement_excludes_zero']
        row.update(final_risk_scales=update['final_risk_scales'], conditional_risk_stationarity_passed=True,
            current_joint_stationarity_diagnostic_passed=True, two_consecutive_actual_law_joint_stationarity_confirmed=True,
            candidate_component_file=update['candidate_component_file'], evidence_results=update_path,
            selected_evidence_results=update_path, source_evidence=update['final_vs_selected_parent_evidence'],
            source_evidence_method='certified parent evidence plus accepted fixed update',
            posterior_results=update_path, posterior_method='certified reweighted retained mixture;component count preserved',
            response_stationarity_at_final_risk=update['new_response_diagnostic'],
            risk_stationarity_certificate=update_path, joint_stationarity_results=update_path)
        input_path = 'paper_artifacts/historical_last_joint_precision_v1/current_candidate_input.json'
        save(ROOT / input_path, {**{k: row[k] for k in KEY}, 'final_risk_scales': row['final_risk_scales'],
            'fixed_response_scales': row['fixed_response_scales'], 'actual_risk_results': update_path,
            'conditional_risk_stationarity_passed': True, 'risk_stationarity_certificate': update_path})
        row['candidate_results'] = input_path
    learning_path = update['qualified_source_learning_results'] if accepted else base['qualified_source_learning_results']
    precision_path = update['qualified_source_precision_results'] if accepted else base['qualified_source_precision_results']
    learning = read(learning_path)
    precision = read(precision_path)
    remaining = [{k: r[k] for k in (*KEY, 'joint_stationarity_results')}
        for r in rows if not r['two_consecutive_actual_law_joint_stationarity_confirmed']]
    converged = not remaining
    for case, certificate, r in zip(learning['evidence_rows'], precision['rows'], rows):
        assert key(case) == key(certificate) == key(r)
        assert case['eta_common_task_scales'] == r['final_risk_scales']
        assert case['f_common_task_scales'] == r['fixed_response_scales']
        eta, _ = load_mixture(certificate, case)
        assert eta.shape[0] * eta.shape[1] == r['mixture_components']
        case.update(joint_stationarity_certificate_results=r['joint_stationarity_results'],
            conditional_risk_stationarity_passed=r['conditional_risk_stationarity_passed'],
            two_consecutive_actual_law_joint_stationarity_confirmed=r['two_consecutive_actual_law_joint_stationarity_confirmed'])
        for head in learning['rows']:
            if key(head)[:-1] == key(case)[:-1] and head['head'] in (case['head'], 'eta_' + case['head']):
                head.update(consecutive_source_stationarity_iterations=2 if r['two_consecutive_actual_law_joint_stationarity_confirmed'] else 1,
                    joint_stationarity_iteration_count_reset=False, joint_stationarity_certificate_results=r['joint_stationarity_results'])
    learning.update(source_learning_converged=converged, joint_stationarity_iterations_counted=2 if converged else 0,
        new_joint_optimization_iterations=0, scope='Two consecutive distinct actual source laws pass registered joint numerical stationarity;one accepted fixed risk update. No independent decision claim.')
    precision['source_learning_converged'] = converged
    out = ROOT / 'paper_artifacts/historical_last_joint_precision_v1'
    save(out / 'source_candidates.json', learning)
    save(out / 'source_precision.json', precision)
    identities = {key(r)[:-1] for r in rows}
    complete = sum(all(r['two_consecutive_actual_law_joint_stationarity_confirmed']
        for r in rows if key(r)[:-1] == identity) for identity in identities)
    phase = {**base, 'rows': rows, 'execution_plan': update['execution_plan'], 'predecessor_phase_results': base_path,
        'accepted_fixed_update_results': update_path, 'remaining_joint_stationarity_cases': remaining,
        'consecutive_actual_law_joint_stationarity_cases_confirmed': len(rows) - len(remaining),
        'complete_joint_stationary_source_identities': complete, 'joint_source_learning_converged': converged,
        'qualified_source_learning_results': str((out / 'source_candidates.json').relative_to(ROOT)),
        'qualified_source_precision_results': str((out / 'source_precision.json').relative_to(ROOT)),
        'new_accepted_fixed_risk_updates': int(accepted), 'parameters_changed': accepted,
        'recorded_CPU_seconds': base['recorded_CPU_seconds'] + update['cpu_seconds'],
        'completed_candidate_MCMC_sweeps': base['completed_candidate_MCMC_sweeps'] + update['completed_candidate_MCMC_sweeps'],
        'additional_sweeps_beyond_registered_banks': update['additional_retained_sweeps_beyond_previous_prefix'],
        'compact_replayed_risk_statistics_files': base['compact_replayed_risk_statistics_files'] + 1,
        'unit_test_results': str((out / 'unit_tests.json').relative_to(ROOT)),
        'new_candidate_MCMC_sweeps_executed_on_replay': 0, 'cached_replay_CPU_seconds': time.process_time() - started}
    save(out / 'phase_results.json', phase)
    print('Joint source completion', len(rows) - len(remaining), '/32;identities', complete,
        '/16;existing source mixtures loaded', len(rows), flush=True)
    return phase


if __name__ == '__main__':
    run()
