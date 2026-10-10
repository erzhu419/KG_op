"""Replay the complete second raw-source MCEM step without new computation."""

import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ('layout', 'selector', 'support', 'treatment', 'head')
BASE = 'paper_artifacts/historical_raw_source_amplitude_learning_v2'
FRESH = 'paper_artifacts/historical_changed_candidate_reconditioning_v2/results.json'
LOCAL = 'paper_artifacts/historical_source_learning_round2_completion_v1/results.json'
FINAL = 'paper_artifacts/historical_source_learning_round2_completion_v2/results.json'


def read(path):
    return json.loads((ROOT / path).read_text())


def key(row):
    return tuple(row[k] for k in IDENTITY)


def assemble_source_precision(step, post, fresh, local, learning_file, local_file, final=None, final_file=None):
    """Combine same-candidate posterior estimates and adjacent normalizers."""
    assert step['all_registered_cases_complete'] and fresh['all_registered_cases_complete']
    assert local['all_registered_cases_complete']
    initial = {key(r): r for r in post['rows']}
    refreshed = {key(r): r for r in fresh['rows']}
    adjacent = {key(r): r for r in local['evidence_rows']}
    repairs = {key(r): r for r in local['functional_repairs']}
    if final is not None and final['functional_precision_passed']:
        repairs[key(final)] = final
    rows = []
    for case in step['evidence_rows']:
        k = key(case); row = refreshed.get(k, initial[k])
        if k in repairs and repairs[k]['functional_precision_passed']:
            component = repairs[k]['candidate_component_file']
            if component != row['candidate_component_file']:
                row = initial[k]
                assert row['candidate_component_file'] == component
        passed = row.get('posterior_functional_precision_passed', row['monte_carlo_gate_passed'])
        retention = row.get('point_mixture_retention_passed')
        rhat, ess = row['maximum_split_rhat'], row['minimum_autocorrelation_ess']
        gap = row['maximum_first_second_half_f_variance_relative_difference']
        functional_ref = None
        if k in repairs and repairs[k]['functional_precision_passed']:
            repair = repairs[k]; passed = True; retention = True
            rhat, ess = repair['maximum_split_rhat'], repair['minimum_autocorrelation_ess']
            gap = repair['maximum_first_second_half_f_variance_relative_difference']
            functional_ref = final_file if final is not None and k == key(final) else local_file
        if case['evidence_ratio_precision_passed']:
            evidence = case; origin = learning_file
        else:
            evidence = adjacent[k]; origin = local_file
        combined = bool(case['source_Mstep_pair_passed'] and passed and retention is not False
            and evidence['evidence_ratio_precision_passed'])
        rows.append({**{name: case[name] for name in IDENTITY},
            'source_Mstep_pair_passed': case['source_Mstep_pair_passed'],
            'posterior_functional_precision_passed': passed,
            'point_mixture_retention_comparison_passed': retention,
            'evidence_ratio_precision_passed': evidence['evidence_ratio_precision_passed'],
            'new_source_posterior_precision_passed': combined, 'positive_mixture_precision_passed': combined,
            'maximum_split_rhat': rhat, 'minimum_autocorrelation_ess': ess,
            'maximum_first_second_half_f_variance_relative_difference': gap,
            'positive_mixture_component_file': row['candidate_component_file'],
            'positive_weights_results': row['candidate_component_file'],
            'functional_precision_reference': functional_ref,
            'within_identity_evidence_results': origin,
            'log_source_evidence_ratio_new_vs_previous': evidence['log_source_evidence_ratio_new_vs_previous'],
            'normalizer_scope': 'Z(current source step)/Z(previous source step) within one frozen identity.',
            'retention_scope': ('Directly certified retained-state estimate with no subsequent reduction.'
                if retention is None else 'Positive1024 mixture compared with the certified posterior variance reference.')})
    assert len(rows) == len({key(r) for r in rows}) == 32
    return rows


def run():
    started = time.process_time()
    step = read(BASE + '/results.json'); post = read(BASE + '/posterior_precision.json')
    fresh = read(FRESH); local = read(LOCAL); final = read(FINAL)
    rows = assemble_source_precision(step, post, fresh, local, BASE + '/results.json', LOCAL, final, FINAL)
    passed_keys = {key(r) for r in rows if r['positive_mixture_precision_passed']}
    identities = {key(r)[:-1] for r in rows}
    complete = sum(k + ('f',) in passed_keys and k + ('g',) in passed_keys for k in identities)
    statistics_by_head = {}
    for head in ('f', 'g', 'eta_f', 'eta_g'):
        current = [r for r in step['rows'] if r['head'] == head]
        changes = [r['relative_source_covariance_change'] for r in current]
        statistics_by_head[head] = {'single_step_stationarity_heads': sum(
            r['source_stationarity_diagnostic_passed'] for r in current),
            'median_relative_covariance_change': statistics.median(changes),
            'maximum_relative_covariance_change': max(changes)}
    tests = read('paper_artifacts/historical_source_learning_round2_completion_v1/unit_tests.json')
    result = {'status': ('Second MCEM step conditioned:32 source cases passed;source-risk learning not converged'
            if len(passed_keys) == 32 else 'Second MCEM step complete;changed-law precision remains HOLD'),
        'rows': rows, 'source_Mstep_Q_heads_passed': step['source_Mstep_gates_passed'],
        'source_posterior_precision_cases_passed': len(passed_keys),
        'evidence_ratio_precision_cases_passed': sum(r['evidence_ratio_precision_passed'] for r in rows),
        'complete_learned_source_identities': complete,
        'source_covariance_change_by_head': statistics_by_head,
        'source_stationarity_heads': step['source_stationarity_heads'],
        'source_learning_converged': False, 'required_consecutive_stable_iterations': 2,
        'learning_results': BASE + '/results.json', 'fresh_results': FRESH,
        'functional_results': [LOCAL, FINAL],
        'forward_evidence_cases_passed': step['evidence_ratio_precision_cases_passed'],
        'reverse_adjacent_evidence_cases_passed': local['adjacent_evidence_cases_passed'],
        'recorded_single_thread_cpu_seconds': sum(r['cpu_seconds'] for r in (step, fresh, local, final)),
        'new_candidate_MCMC_sweeps_executed': fresh['new_candidate_MCMC_sweeps'],
        'original_source_cases_resampled': 0, 'new_source_labels': 0,
        'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
        'tests_run': tests['tests_run'], 'tests_passed': tests['passed'],
        'metadata_repair_record': BASE + '/metadata_repair.json',
        'no_model_adopted': True, 'independent_confirmation': False,
        'cross_identity_source_evidence_normalized': False,
        'next_work': 'Continue source-risk covariance learning from these second-step mixtures with the same registered stationarity and posterior gates. Reuse passed estimates and restrict functional integration to actual failures. Comparable source evidence,matched completeGP controls and independent decision validation follow source convergence.',
        'limitations': [
            'Both source steps improve full/split Q;two MCEM updates do not establish structural hyperparameter convergence.',
            'One reverse adjacent normalizer remains HOLD at0.1058 split difference;its registered forward estimate passes. Both estimates are archived.',
            'These are within-identity adjacent-step normalizers,not absolute source evidence across identities.',
            'No new source prior has entered the original12 cached KG-action comparison or independent target evaluation.',
        ], 'cached_replay_cpu_seconds': time.process_time() - started}
    (ROOT / BASE / 'phase_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], 'complete identities', complete,
        'cached CPU', round(result['cached_replay_cpu_seconds'], 3), flush=True)
    return result


if __name__ == '__main__': run()
