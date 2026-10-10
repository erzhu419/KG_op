"""Replay the accepted source-acceleration registry using existing artifacts."""

from performance.complete_historical_source_heads import ROOT, read, save
from performance.select_historical_source_acceleration import run as select


def run():
    phase = select()
    comparison = read('paper_artifacts/historical_source_acceleration_comparison_v1/results.json')
    selection = read(phase['learning_results'])
    tests = read('paper_artifacts/historical_source_acceleration_comparison_v1/unit_tests.json')
    result = {'status': phase['status'],
        'registered_acceleration_candidates': comparison['registered_candidates'],
        'source7_evidence_precision_candidates': comparison['source7_evidence_precision_cases_passed'],
        'source7_evidence_improvement_candidates': comparison['source7_evidence_improvement_cases_passed'],
        'changed_law_posterior_candidates_passed': comparison['direct_changed_law_posterior_cases_passed'],
        'accelerated_source_cases_selected': phase['accelerated_source_cases_selected'],
        'source7_cases_retained': phase['source7_cases_retained'],
        'source_posterior_precision_cases_passed': phase['source_posterior_precision_cases_passed'],
        'complete_learned_source_identities': phase['complete_learned_source_identities'],
        'source_mixture_component_counts': phase['source_mixture_component_counts'],
        'latest_learning': phase['learning_results'],
        'latest_certified_precision': 'paper_artifacts/historical_source_acceleration_selection_v1/phase_results.json',
        'source_learning_converged': False, 'new_stationarity_iterations_counted': 0,
        'comparison_CPU_seconds': comparison['cpu_seconds'],
        'registry_replay_CPU_seconds': selection['cpu_seconds'],
        'new_candidate_MCMC_sweeps_executed': 0, 'original_source_cases_resampled': 0,
        'KG_components_changed': 0, 'new_source_labels': 0, 'new_target_simulator_calls': 0,
        'new_verifier_calls': 0, 'server_jobs': 0, 'tests_run': tests['tests_run'], 'tests_passed': tests['passed'],
        'no_model_adopted': True, 'independent_confirmation': False,
        'limitations': phase['limitations'],
        'next_work': 'Continue source learning from the selected actual laws. Reuse full risk-chain states in memory for several source-only updates with qualified importance/evidence and unchanged Q/score requirements,then certify final posterior functionals;do not reuse source6 moments at selected parameters. Source convergence,comparable evidence and independent matched-control decisions remain.'}
    save(ROOT / 'paper_artifacts/historical_source_acceleration_comparison_v1/phase_results.json', result)
    print('source acceleration', result['accelerated_source_cases_selected'], 'selected;32 cases/16 identities',
        'components', result['source_mixture_component_counts'], flush=True)
    return result


if __name__ == '__main__':
    run()
