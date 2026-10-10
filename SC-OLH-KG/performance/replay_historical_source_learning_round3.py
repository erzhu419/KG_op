"""Replay continued source conditioning and risk probes from local JSON."""

import statistics
import time

from performance.complete_historical_source_heads import ROOT, save
from performance.replay_historical_source_learning_round2 import read, key, assemble_source_precision

PLAN = 'performance/manifests/historical_source_learning_round3_replay.json'


def run(plan=PLAN):
    started = time.process_time()
    spec = read(plan); base = spec['base']
    paths = [spec[name] for name in ('proposal', 'learning', 'fresh', 'local', 'probe')]
    loaded = {p: read(p) for p in paths}
    proposal, step, fresh, local, probe = [loaded[p] for p in paths]
    rows = assemble_source_precision(step, read(base + '/posterior_precision.json'), fresh, local,
        spec['learning'], spec['local'])
    passed = {key(r) for r in rows if r['positive_mixture_precision_passed']}
    identities = {key(r)[:-1] for r in rows}
    complete = sum(k + ('f',) in passed and k + ('g',) in passed for k in identities)
    changes = {}
    for head in ('f', 'g', 'eta_f', 'eta_g'):
        current = [r for r in step['rows'] if r['head'] == head]
        changes[head] = {'single_step_stationarity_heads': sum(r['source_stationarity_diagnostic_passed'] for r in current),
            'two_consecutive_stationarity_heads': sum(r['consecutive_source_stationarity_iterations'] >= 2 for r in current),
            'median_relative_covariance_change': statistics.median(r['relative_source_covariance_change'] for r in current),
            'maximum_relative_covariance_change': max(r['relative_source_covariance_change'] for r in current)}
    qualified = [r for r in probe['rows'] if r['relative_common_task_scales'] == [.75, .75] and r['precision_gate_passed']]
    tests = read(spec['unit_tests'])
    result = {'execution_plan': plan, 'status': (spec['step_name'] + ' conditioned;mean heads stationary,risk learning remains open'
            if len(passed) == 32 else spec['step_name'] + ' remains HOLD'),
        'rows': rows, 'source_Mstep_Q_heads_improved': step['source_Mstep_Q_heads_improved'],
        'stationary_mean_parameters_held': step['stationary_parameters_held'],
        'source_updates_accepted': step['source_updates_accepted'],
        'joint_full_and_split_Q_pairs_improved': step['joint_full_and_split_Q_pairs_improved'],
        'source_posterior_precision_cases_passed': len(passed),
        'evidence_ratio_precision_cases_passed': sum(r['evidence_ratio_precision_passed'] for r in rows),
        'complete_learned_source_identities': complete, 'source_covariance_change_by_head': changes,
        'source_learning_converged': False, 'cross_identity_source_evidence_normalized': False,
        'risk_probe_75percent_precision_cases': len(qualified),
        'risk_probe_75percent_improvement_cases': sum(r['eligible_improvement_excludes_zero'] for r in qualified),
        'risk_probe_larger_steps_passed': sum(r['precision_gate_passed'] for r in probe['rows']
            if r['relative_common_task_scales'] in ([.25, .25], [0., 0.], [0., 1.], [1., 0.])),
        'risk_probe_MC_interval_scope': 'Conditional source-state numerical error,not independent task uncertainty.',
        'learning_results': spec['learning'], 'preserved_proposed_updates': spec['proposal'],
        'fresh_results': spec['fresh'], 'functional_results': [spec['local']], 'risk_probe_results': spec['probe'],
        'recorded_single_thread_cpu_seconds': sum(loaded[p]['cpu_seconds'] for p in spec['cost_results']),
        'unrecorded_cpu_stages': fresh.get('unrecorded_cpu_stages', []),
        'new_candidate_MCMC_sweeps_executed': fresh['new_candidate_MCMC_sweeps'],
        'original_source_cases_resampled': 0, 'new_source_labels': 0, 'new_target_simulator_calls': 0,
        'new_verifier_calls': 0, 'server_jobs': 0, 'tests_run': tests['tests_run'], 'tests_passed': tests['passed'],
        'no_model_adopted': True, 'independent_confirmation': False,
        'next_work': 'Continue source-risk amplitudes from these certified mixtures;retain stationary mean heads and use precision-qualified adjacent steps instead of jumps to zero risk. Convergence precedes comparable source evidence,matched completeGP controls and independent target decision tests.',
        'limitations': ['Tiny proposed mean updates failing split Q remain archived;their actual selected changes are zero after at least two registered stationary iterations.',
            'Risk amplitudes continue moving;this numerical source closure is not structural learning convergence.',
            'Fresh reverse normalizers against the original law mostly fail overlap;only actual adjacent-step normalizers enter the source registry.',
            'Candidate source laws have not entered the original12 KG-action comparison or independent target evaluation.'],
        'cached_replay_cpu_seconds': time.process_time() - started}
    save(ROOT / base / 'phase_results.json', result)
    print(result['status'], 'cases', len(passed), 'complete identities', complete,
        'cached CPU', round(result['cached_replay_cpu_seconds'], 3), flush=True)
    return result


if __name__ == '__main__':
    run()
