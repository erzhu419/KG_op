"""Assemble cached conditioning of the first source-only MCEM candidate."""

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ('layout', 'selector', 'support', 'treatment', 'head')
STAGES = (
    'historical_changed_source_conditional_precision_v1',
    'historical_changed_source_conditional_precision_v2',
    'historical_changed_candidate_reconditioning_v1',
    'historical_changed_candidate_tail_resolution_v1',
    'historical_changed_candidate_tail_resolution_v2',
)
FUNCTIONAL = 'paper_artifacts/historical_changed_candidate_functional_resolution_v4/results.json'
LEARNING = 'paper_artifacts/historical_raw_source_amplitude_learning_v1/results.json'
OUT = ROOT / 'paper_artifacts/historical_changed_candidate_functional_resolution_v4'


def read(path):
    return json.loads((ROOT / path).read_text())


def key(row):
    return tuple(row[k] for k in IDENTITY)


def run():
    started = time.process_time()
    stages = [read(f'paper_artifacts/{name}/results.json') for name in STAGES]
    learning = read(LEARNING)
    candidates = {key(row): row for row in learning['evidence_rows']}
    rows = stages[-1]['rows'] + stages[-1]['reused_passed_cases']
    assert len(rows) == len(candidates) == 32
    assert {key(row) for row in rows} == set(candidates)
    assert all(stage['all_registered_cases_complete'] for stage in stages)
    functional = read(FUNCTIONAL)
    resolved = []
    for row in rows:
        case = candidates[key(row)]
        passed = row['new_source_posterior_precision_passed']
        retention = row.get('point_mixture_retention_passed')
        rhat = row['maximum_split_rhat']
        ess = row['minimum_autocorrelation_ess']
        gap = row['maximum_first_second_half_f_variance_relative_difference']
        variance_error = row.get('point_mixture_maximum_f_variance_relative_error')
        mean_error = row.get('point_mixture_mean_error')
        fixed = key(row) == key(functional['identity']) if 'identity' in functional else False
        if fixed and functional['functional_precision_passed']:
            assert functional['source_evidence_ratio_precision_preserved']
            passed = True
            retention = True
            rhat = max(rhat, functional['fixed_site8_rhat'])
            ess = min(ess, functional['fixed_site8_ESS'])
            gap = max(functional['fixed_site8_variance_split'],
                *(v for i, v in enumerate(row['half_variance_relative_differences']) if i != 8))
            variance_error = functional['retained_mixture_variance_error']
            mean_error = functional['retained_mixture_mean_error']
        resolved.append({
            **{k: row[k] for k in IDENTITY},
            'source_Mstep_pair_passed': case['source_Mstep_pair_passed'],
            'new_source_posterior_precision_passed': passed,
            'evidence_ratio_precision_passed': row['evidence_ratio_precision_passed'],
            'point_mixture_retention_comparison_passed': retention,
            'positive_mixture_precision_passed': bool(passed and retention is not False),
            'retention_scope': ('No additional reduction of the directly certified retained-state estimate.'
                if retention is None else 'Positive retained mixture compared with its certified posterior moments.'),
            'maximum_split_rhat': rhat,
            'minimum_autocorrelation_ess': ess,
            'maximum_first_second_half_f_variance_relative_difference': gap,
            'point_mixture_maximum_f_variance_relative_error': variance_error,
            'point_mixture_mean_error': mean_error,
            'log_source_evidence_ratio_new_vs_old': row.get('log_source_evidence_ratio_new_vs_old',
                case['log_source_evidence_ratio_new_vs_old']),
            'posterior_results': f'paper_artifacts/{STAGES[-1]}/results.json',
            'positive_mixture_component_file': row.get('candidate_component_file', row['source_case']),
            'positive_weights_results': row.get('candidate_component_file', LEARNING),
            'within_identity_evidence_results': row.get('evidence_source',
                f'paper_artifacts/{STAGES[-1]}/results.json'),
            'functional_precision_reference': FUNCTIONAL if fixed else None,
        })
    passed_keys = {key(row) for row in resolved if row['positive_mixture_precision_passed']}
    identities = {k[:-1] for k in candidates}
    complete = sum(k + ('f',) in passed_keys and k + ('g',) in passed_keys for k in identities)
    actions = read('paper_artifacts/historical_control_variate_action_precision_v2/results.json')
    tests = read(str((OUT / 'unit_tests.json').relative_to(ROOT)))
    old_cap = read('paper_artifacts/historical_changed_candidate_functional_resolution_v3/results.json')
    result = {
        'status': ('32 fixed changed-law source cases meet the registered precision gates'
            if len(passed_keys) == 32 else 'Changed-law source precision remains HOLD'),
        'rows': resolved,
        'posterior_precision_cases_passed': sum(r['new_source_posterior_precision_passed'] for r in resolved),
        'evidence_ratio_precision_cases_passed': sum(r['evidence_ratio_precision_passed'] for r in resolved),
        'positive_mixture_precision_cases_passed': len(passed_keys),
        'separate_mixture_retention_comparisons_passed': sum(
            r['point_mixture_retention_comparison_passed'] is True for r in resolved),
        'direct_retained_state_precision_cases': sum(
            r['point_mixture_retention_comparison_passed'] is None for r in resolved),
        'complete_learned_source_identities': complete,
        'source_Mstep_Q_heads_passed': learning['source_Mstep_gates_passed'],
        'source_learning_results': LEARNING,
        'recorded_source_precision_cpu_seconds': sum(s['cpu_seconds'] for s in stages)
            + old_cap['cpu_seconds'] + functional['cpu_seconds'],
        'new_candidate_MCMC_sweeps_executed': sum(s.get('new_candidate_MCMC_sweeps', 0) for s in stages),
        'conditioning_batches': [{'results': f'paper_artifacts/{name}/results.json',
            'cases_passed': stage['posterior_precision_cases_passed'], 'cpu_seconds': stage['cpu_seconds']}
            for name, stage in zip(STAGES, stages)],
        'functional_precision_results': FUNCTIONAL,
        'old_cpu_cap_result': 'paper_artifacts/historical_changed_candidate_functional_resolution_v3/results.json',
        'original32_passed_source_cases_unchanged': True,
        'original_source_cases_resampled': 0,
        'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
        'cached_action_precision_cases_passed': actions['actions_precision_passed'],
        'cached_actions_use_original_source_laws': True,
        'tests_run': tests['tests_run'], 'tests_passed': tests['passed'],
        'source_hyperparameter_learning_converged': False,
        'cross_identity_source_evidence_normalized': False,
        'no_model_adopted': True, 'independent_confirmation': False,
        'next_work': 'Iterate source-only MCEM using these changed-law posteriors; establish comparable source evidence across identities before target adoption. Then evaluate the historical completeGP controls under the same paid-update and terminal contract, followed by independent decision validation.',
        'limitations': [
            'This certifies numerical conditioning of one fixed source-only covariance M-step, not converged learning.',
            'Evidence ratios are Z(new)/Z(old) within each identity; they do not normalize different identities.',
            'The site8 scalar reference replaces only its variance precision estimate; saved components, means and the other47 estimates are unchanged.',
            'Cached12-action comparisons use original source laws at two opened development states; no independent deployment gain is established.',
        ],
        'cached_replay_cpu_seconds': time.process_time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'phase_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], 'complete identities', complete, 'cached CPU',
        round(result['cached_replay_cpu_seconds'], 3), flush=True)
    return result


if __name__ == '__main__':
    run()
