"""Cached replay of complete risk-score precision and source iteration7."""

import statistics

from performance.complete_historical_source_heads import ROOT, read, save
from performance.replay_historical_source_learning_round3 import run as replay


def run():
    phase = replay('performance/manifests/historical_source_learning_round7_replay.json')
    banks = read('paper_artifacts/historical_source_risk_score_banks_v1/results.json')
    rate = read('paper_artifacts/historical_source_risk_score_banks_v1/em_rate.json')
    probe = read('paper_artifacts/historical_source_risk_acceleration_probe_v1/results.json')
    summaries = [read(r['results'])['score_summaries']['full'] for r in banks['rows']]
    mean_errors = [v for s in summaries for v in s['scaled_MCSE']]
    result = {'status': 'All32 risk-score banks precise;seventh source iteration conditioned;convergence remains open',
        'risk_score_precision_cases_passed': banks['risk_score_precision_cases_passed'],
        'maximum_scaled_score_95_MC_halfwidth': max(v for s in summaries for v in s['scaled_95_MC_halfwidth']),
        'median_scaled_score_MCSE': statistics.median(mean_errors),
        'source_posterior_precision_cases_passed': phase['source_posterior_precision_cases_passed'],
        'complete_learned_source_identities': phase['complete_learned_source_identities'],
        'source_covariance_change_by_head': phase['source_covariance_change_by_head'],
        'source_learning_converged': bool(phase['source_posterior_precision_cases_passed'] == 32 and all(
            h['two_consecutive_stationarity_heads'] == 16 for h in phase['source_covariance_change_by_head'].values())),
        'latest_learning': phase['learning_results'],
        'latest_certified_precision': 'paper_artifacts/historical_source_learning_round7/phase_results.json',
        'risk_statistics_registry': 'paper_artifacts/historical_source_risk_score_banks_v1/results.json',
        'median_local_missing_information_fraction': rate['median_local_missing_information_fraction'],
        'fixed_acceleration_Q_cases_passed': probe['Q_cases_passed'],
        'fixed_acceleration_evidence_precision_cases_passed': probe['evidence_precision_cases_passed'],
        'fixed_acceleration_Q_and_evidence_improvement_cases_passed': probe['Q_and_evidence_improvement_cases_passed'],
        'acceleration_parameters_adopted': False,
        'recorded_main_stage_single_thread_CPU_seconds': banks['recorded_new_bank_CPU_seconds']
            + phase['recorded_single_thread_cpu_seconds'] + rate['cpu_seconds'] + probe['cpu_seconds'],
        'candidate_MCMC_sweeps_executed': banks['new_candidate_MCMC_sweeps_executed']
            + phase['new_candidate_MCMC_sweeps_executed'],
        'KG_components_changed': 0, 'original_source_cases_resampled': 0,
        'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
        'tests_run': phase['tests_run'], 'tests_passed': phase['tests_passed'],
        'no_model_adopted': True, 'independent_confirmation': False,
        'limitations': ['The32 precise score banks describe the sixth-law E-step used in iteration7;they cannot be reused as seventh-law moments.',
            'Source7 risk covariance still moves;score precision is distinct from hyperparameter convergence.',
            'The curvature diagnostic supplies local diagonals without curvature error intervals or off-diagonal score covariance.',
            'The fixed acceleration probe compares with source6 at fixed source6 response parameters;it has no changed-law posterior certificate or proven advantage over source7.',
            'Learned candidates have not entered the original12 KG-action comparison or independent target evaluation.'],
        'next_work': 'Compare source-qualified accelerated candidates with the certified seventh law and condition only candidates retaining an improvement. Preserve failed Q/evidence proposals. Then complete source-risk convergence,comparable source evidence,matched completeGP controls and independent decisions.'}
    save(ROOT / 'paper_artifacts/historical_source_risk_score_banks_v1/phase_results.json', result)
    print(result['status'], 'recorded CPU', round(result['recorded_main_stage_single_thread_CPU_seconds'], 2), flush=True)
    return result


if __name__ == '__main__':
    run()
