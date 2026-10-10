"""Summarize the completed source-risk reuse batch without sampling."""

import time

from performance.complete_historical_source_heads import ROOT, read, save


def run():
    started = time.process_time()
    batch = read('paper_artifacts/historical_source_risk_state_reuse_batch_v1/results.json')
    assert batch['all_registered_cases_complete']
    conditioning = read('paper_artifacts/historical_source_risk_state_reuse_conditioning_v1/results.json')
    failed = read('paper_artifacts/historical_source_risk_state_reuse_certification_v1/results.json')
    retention = read('paper_artifacts/historical_source_risk_stationary_retention_v1/results.json')
    tests = read('paper_artifacts/historical_source_risk_stationary_retention_v1/unit_tests.json')
    prefix = read('paper_artifacts/historical_source_risk_state_reuse_v1/historical_sampler_prefix_check.json')
    statuses = {}; stopped_stationarity_diagnostics = []
    for row in batch['rows']:
        statuses[row['status']] = statuses.get(row['status'], 0) + 1
        if not row['conditional_risk_stationarity_passed']:
            detail = read(row['risk_results'])
            if detail['attempts'] and detail['attempts'][-1]['stationarity_diagnostic_passed']:
                stopped_stationarity_diagnostics.append({k: row[k] for k in
                    ('layout', 'selector', 'support', 'treatment', 'head', 'risk_results', 'status')})
    held_paths = {r['risk_results'] for r in retention['rows']}
    remaining_statuses = {}
    for row in batch['rows']:
        if not row['conditional_risk_stationarity_passed'] and row['risk_results'] not in held_paths:
            remaining_statuses[row['status']] = remaining_statuses.get(row['status'], 0) + 1
    result = {'status': 'complete registered conditional-risk batch; joint source learning open',
        'risk_batch_results': 'paper_artifacts/historical_source_risk_state_reuse_batch_v1/results.json',
        'registered_cases': batch['registered_cases'],
        'conditional_risk_stationarity_cases_passed': retention['conditional_risk_stationarity_cases_passed'],
        'cases_stable_after_accepted_updates': batch['conditional_risk_stationarity_cases_passed'],
        'stationary_current_risk_laws_held': retention['stationary_current_risk_laws_held'],
        'stationary_retention_results': 'paper_artifacts/historical_source_risk_stationary_retention_v1/results.json',
        'original_case_stop_status_counts': statuses, 'remaining_stop_status_counts': remaining_statuses,
        'remaining_cases_with_single_iteration_stationarity_diagnostic':
            [r for r in stopped_stationarity_diagnostics if r['risk_results'] not in held_paths],
        'accepted_risk_updates': batch['accepted_risk_updates'], 'risk_sampling_batches': batch['sampling_batches'],
        'accepted_updates_per_risk_sampling_batch': batch['accepted_risk_updates'] / batch['sampling_batches'],
        'recorded_risk_learning_CPU_seconds': batch['recorded_risk_learning_CPU_seconds'],
        'final_response_conditioning_CPU_seconds': conditioning['cpu_seconds'],
        'failed_parent_reweighting_CPU_seconds': failed['cpu_seconds'],
        'completed_candidate_MCMC_sweeps': batch['completed_candidate_MCMC_sweeps']
            + conditioning['new_candidate_MCMC_sweeps'],
        'historical_sampler_prefix_check': prefix,
        'final_posterior_and_retention_certified_cases': int(conditioning['final_posterior_and_retention_certified']),
        'final_posterior_certification': 'paper_artifacts/historical_source_risk_state_reuse_conditioning_v1/results.json',
        'final_pilot_vs_parent_evidence_precision_passed': False,
        'current_certified_source_registry': 'paper_artifacts/historical_source_acceleration_selection_v1/phase_results.json',
        'current_certified_source_cases': 32, 'current_complete_source_identities': 16,
        'source_mixture_component_counts': {'1024': 30, '512': 2},
        'joint_source_learning_converged': False, 'source_registry_replaced': False,
        'new_candidate_MCMC_sweeps_executed_on_replay': 0,
        'original_source_cases_resampled': 0, 'new_source_labels': 0,
        'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
        'tests_run': tests['tests_run'], 'tests_passed': tests['passed'],
        'cached_replay_CPU_seconds': time.process_time() - started,
        'no_model_adopted': True, 'independent_confirmation': False,
        'limitations': ['Fixed response laws within risk updates; final response posteriors and joint convergence remain incomplete.',
            'The pilot final-law direct parent evidence ratio fails overlap; the successful final response certificate does not fix that normalizer.',
            'Updates per batch measures sampling economy,not measured end-to-end speedup or independent decision benefit.'],
        'next_work': 'Resolve the actual stopped risk cases using their fixed current laws and precision requirements. Certify final response functionals and retention;check response stationarity at final risk laws. Complete source-evidence normalization before replacing the32-case registry or entering matched-control independent decisions.'}
    save(ROOT / 'paper_artifacts/historical_source_risk_state_reuse_batch_v1/phase_results.json', result)
    print('Cached source-risk reuse', result['conditional_risk_stationarity_cases_passed'], '/32 stable;',
        result['accepted_risk_updates'], 'accepted updates;', result['risk_sampling_batches'], 'risk batches', flush=True)
    return result


if __name__ == '__main__':
    run()
