"""Cached summary of source iterations5/6 and the risk-score precision pilot."""

from performance.complete_historical_source_heads import ROOT, read, save
from performance.replay_historical_source_learning_round3 import run as replay


def run():
    rounds = [replay(f'performance/manifests/historical_source_learning_round{n}_replay.json') for n in (5, 6)]
    scores = read('paper_artifacts/historical_source_risk_score_precision_v1/results.json')
    bank = read('paper_artifacts/historical_source_risk_score_bank_v1/results.json')
    reuse = read('paper_artifacts/historical_source_learning_round6_mixture_reuse_v1/results.json')
    good = [r for r in scores['rows'] if r['diagnostic_precision_passed']]
    import statistics
    result = {'status': 'Both source iterations conditioned;source-risk convergence remains open',
        'source_cases_passed_by_round': [r['source_posterior_precision_cases_passed'] for r in rounds],
        'complete_identities_by_round': [r['complete_learned_source_identities'] for r in rounds],
        'risk_covariance_changes_by_round': [r['source_covariance_change_by_head'] for r in rounds],
        'all_registered_iterations_conditioned': all(r['source_posterior_precision_cases_passed'] == 32 for r in rounds),
        'latest_learning': rounds[-1]['learning_results'],
        'latest_certified_precision': 'paper_artifacts/historical_source_learning_round6/phase_results.json',
        'source_learning_converged': False,
        'current_moment_scaled_MCSE_median': statistics.median(r['moment_scaled_MCSE'] for r in good),
        'current_Stein_scaled_MCSE_median': statistics.median(r['Stein_scaled_MCSE'] for r in good),
        'Stein_over_moment_MCSE_ratio_median': statistics.median(r['Stein_scaled_MCSE'] / r['moment_scaled_MCSE'] for r in good),
        'current_score_diagnostics_passed': len(good),
        'paired_score_estimators_compatible_coordinates': sum(r['moment_Stein_difference_inside_conditional_95_MC_interval'] for r in good),
        'Stein_estimator_adopted': False, 'risk_only_bank_source_cases': 1,
        'risk_only_bank_prefix_matched': bank['same_seed_prefix_matched'],
        'risk_only_bank_score_precision_passed': bank['score_95_MC_halfwidth_below_stationarity_threshold'],
        'risk_only_bank_scaled_95_MC_halfwidth': bank['score_summaries']['full']['scaled_95_MC_halfwidth'],
        'risk_only_bank_scaled_scores': bank['score_summaries']['full']['scaled_scores'],
        'reused_current_law_tail_variance_error': reuse['retained_mixture_variance_error'],
        'KG_components_changed_for_score_bank': bank['KG_components_changed'],
        'recorded_main_stage_single_thread_CPU_seconds': sum(r['recorded_single_thread_cpu_seconds'] for r in rounds)
            + scores['cpu_seconds'] + bank['cpu_seconds'],
        'candidate_MCMC_sweeps_executed': sum(r['new_candidate_MCMC_sweeps_executed'] for r in rounds)
            + bank['new_candidate_MCMC_sweeps_executed'],
        'score_bank_sweeps_beyond_its_existing_prefix': bank['extension_sweeps_beyond_existing_prefix'],
        'original_source_cases_resampled': 0, 'new_source_labels': 0, 'new_target_simulator_calls': 0,
        'new_verifier_calls': 0, 'server_jobs': 0, 'tests_run': rounds[-1]['tests_run'],
        'tests_passed': rounds[-1]['tests_passed'], 'no_model_adopted': True, 'independent_confirmation': False,
        'next_work': 'Use longer source risk statistics at score-precision failures while keeping the KG mixture at1024 components;continue risk hyperparameter updates with unchanged Q,posterior and stationarity requirements. Comparable source evidence and independent matched-control decision tests follow.'}
    save(ROOT / 'paper_artifacts/historical_source_learning_rounds5_6_v1/phase_results.json', result)
    print(result['status'], 'cases', result['source_cases_passed_by_round'],
        'risk-only score pilot', result['risk_only_bank_score_precision_passed'], flush=True)
    return result


if __name__ == '__main__':
    run()
