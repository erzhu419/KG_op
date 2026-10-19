"""Replay native posterior effects on the completed second query.

Both states use all 104 policies and remaining budget 359. The before state
deliberately ignores the first search observation while retaining its budget
debit. Only saved raw observations are consumed; no truth or simulator is used.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.historical_common_bank import initialization_batches, replay_raw_batches
from performance.mainline_cross_dimension_prior import restore_cross_dimension_prior
from performance.mainline_joint_risk_source import apply_joint_risk_source
from performance.mainline_moment_policy import available_state, moment_state, query_action
from performance.mainline_propagation import point_covariance, propagation

MANIFEST = 'performance/manifests/mainline_d1000_query_update_replay_v1.json'
OUTPUT = 'paper_artifacts/mainline_d1000_query_update_replay_v1'


def read(path):
    return json.loads((ROOT / path).read_text())


def parity(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0.)
    return float(np.max(abs(actual - expected)))


def snapshot_parity(state, weights, saved):
    return {'mean': parity(state['mean'], saved['mean']),
            'point_cov': parity(point_covariance(state), saved['point_cov']),
            'identity_weights': parity(weights, saved['identity_weights'])}


def rank(scores, index):
    """One-based rank with the selector's smallest-index tie break."""
    return int(np.flatnonzero(np.argsort(-scores, kind='stable') == index)[0]) + 1


def comparison(before, after):
    old, new = np.asarray(before['scores73']), np.asarray(after['scores73'])
    q0, q1 = int(before['query']), int(after['query'])
    assert q0 == int(np.argmax(old)) and q1 == int(np.argmax(new))
    return {'before_query': q0, 'after_query': q1, 'query_changed': q0 != q1,
            'after_query_before_rank': rank(old, q1),
            'after_query_before_score': float(old[q1]),
            'after_query_before_gap_to_maximum': float(old.max() - old[q1]),
            'before_query_after_rank': rank(new, q0),
            'before_query_after_score': float(new[q0]),
            'before_top_score': float(old[q0]), 'after_top_score': float(new[q1]),
            'top_score_change': float(new[q1] - old[q0]),
            'after_query_score_change': float(new[q1] - old[q1]),
            'maximum_absolute_score_change': float(np.max(abs(new - old))),
            'score_changed_count_above_1e_minus12': int(np.sum(abs(new - old) > 1e-12))}


def _timeout(signum, frame):
    raise TimeoutError('Registered replay CPU limit exceeded')


def run_case(task_index, manifest=MANIFEST, output=None):
    started, wall = time.process_time(), time.monotonic()
    spec = read(manifest)
    out = ROOT / (output or spec['output'])
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / f'task{task_index:02d}.json'
    attempt = {'task_index': task_index, 'started_at': datetime.now(timezone.utc).isoformat(),
               'manifest': str(manifest), 'status': 'failed', 'phase': 'inputs',
               'saved_raw_observation_updates': 0, 'query_score_calls': 0,
               'new_simulator_calls': 0, 'new_verification_calls': 0,
               'new_target_truth_calls': 0, 'new_source_labels': 0,
               'new_source_fits': 0, 'new_deployments': 0}
    phases = {}
    signal.signal(signal.SIGPROF, _timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_case_cpu_seconds'])
    try:
        assert task_index in spec['task_indices']
        assert (spec['candidate_count'], spec['remaining'], spec['quadrature_order']) == (104, 359, 3)
        source = ROOT / spec['source_output']
        frozen = read(source / 'frozen_inputs.json')
        plan, protocol = frozen['plan'], frozen['terminal_protocol']
        paths = spec['inputs']
        paid = read(source / paths['initialization'].format(task_index=task_index))
        case = read(source / paths['case'].format(task_index=task_index))
        first_raw = read(source / paths['first_raw'].format(task_index=task_index))
        second = read(source / paths['second_decision'].format(task_index=task_index))
        first = case['rounds'][0]
        assert first_raw['raw_observation'] == first['raw_observation']
        assert first_raw['decision'] == first['decision']
        assert second['decision'] == case['rounds'][1]['decision']
        assert first['paid_episodes'] == 3 and second['paid_episodes'] == 4
        assert len(frozen['bank']) == spec['candidate_count']
        assert frozen['dynamic_candidates']['round_end_indices'][1] == spec['candidate_count']
        actual = second['decision']
        assert actual['order'] == spec['quadrature_order']
        assert actual['available_candidate_count'] == len(actual['scores73']) == spec['candidate_count']
        assert plan['source_joint_risk_fit_file'] == spec['source_joint_risk_fit_file']
        q1, q2 = int(first['decision']['query']), int(actual['query'])
        reuse = read(plan['reuse_plan'])
        pilot = read(read(reuse['independent_execution_plan'])['pilot_plan'])
        coordinates = read(plan['query_coordinates'])

        attempt['phase'] = 'restore_and_initialization'
        phase_started = time.process_time()
        model, restoration = restore_cross_dimension_prior(
            reuse, pilot, plan['layout'], spec['arm'], coordinates)
        restoration['source_joint_risk_fit'] = apply_joint_risk_source(
            model, coordinates, frozen['source_joint_risk_fit'])
        initialization = replay_raw_batches(model, initialization_batches(paid), paid['tau'])
        attempt['saved_raw_observation_updates'] = initialization['paid_episodes']
        assert model.paid_episodes == 2
        before, before_weights = moment_state(model), model.weights.copy()
        checks = {'initial': snapshot_parity(before, before_weights, case['initial'])}
        checks['initial_query_cross_covariance'] = parity(
            before['cov'][:, :, q1, :], first['propagation']['before_query_cross_covariance'])
        phases['restore_and_initialization_cpu_seconds'] = time.process_time() - phase_started

        attempt['phase'] = 'before_query_scores'
        phase_started = time.process_time()
        attempt['query_score_calls'] += 1
        baseline = query_action(available_state(before, spec['candidate_count']), spec['remaining'], protocol)
        baseline['available_candidate_count'] = spec['candidate_count']
        phases['before_query_scores_cpu_seconds'] = time.process_time() - phase_started

        attempt['phase'] = 'replay_first_search_observation'
        phase_started = time.process_time()
        raw = first_raw['raw_observation']
        model.update_raw(q1, [raw[0], raw[1] - paid['tau']])
        attempt['saved_raw_observation_updates'] += 1
        assert model.paid_episodes == 3
        assert plan['total_target_call_cap'] - model.paid_episodes == spec['remaining']
        after, after_weights = moment_state(model), model.weights.copy()
        checks['after_first_raw'] = snapshot_parity(after, after_weights, second['before'])
        checks['second_query_cross_covariance'] = parity(
            after['cov'][:, :, q2, :], second['propagation']['before_query_cross_covariance'])
        phases['replay_first_search_cpu_seconds'] = time.process_time() - phase_started
        del model

        recomputed = task_index == 0
        if recomputed:
            attempt['phase'] = 'after_query_scores_end_to_end_check'
            phase_started = time.process_time()
            attempt['query_score_calls'] += 1
            check = query_action(available_state(after, spec['candidate_count']), spec['remaining'], protocol)
            assert check['query'] == actual['query'] and check['order'] == actual['order']
            checks['after_query_scores'] = {key: parity(check[key], actual[key]) for key in
                ('scores73', 'adaptation_value', 'verification_budget_cost', 'estimated_net_proxy')}
            phases['after_query_scores_cpu_seconds'] = time.process_time() - phase_started

        attempt['phase'] = 'save'
        mean_delta = after['mean'] - before['mean']
        point_delta = point_covariance(after) - point_covariance(before)
        changes = propagation(before, after, q1, paid['candidate_bank_indices'])
        changes.update({'all_policy_mean_delta': mean_delta.tolist(),
                        'all_policy_point_covariance_delta': point_delta.tolist(),
                        'all_policy_exact_episode_noise_delta': (after['noise'] - before['noise']).tolist(),
                        'maximum_absolute_full_covariance_change': float(np.max(abs(after['cov'] - before['cov']))),
                        'maximum_absolute_identity_weight_change': float(np.max(abs(after_weights - before_weights))),
                        'query_point_mean_delta': mean_delta[q1].tolist(),
                        'query_point_covariance_delta': point_delta[q1].tolist()})
        contrast = comparison(baseline, actual)
        observed = set(paid['candidate_bank_indices']) | {q1}
        contrast.update({'before_query_unobserved_after_first_raw': baseline['query'] not in observed,
                         'after_query_unobserved_after_first_raw': q2 not in observed})
        result = {'task_index': task_index, 'task_id': frozen['tasks'][task_index]['task_id'],
                  'source_output': spec['source_output'], 'manifest': str(manifest), 'arm': spec['arm'],
                  'candidate_count': spec['candidate_count'], 'remaining': spec['remaining'],
                  'quadrature_order': spec['quadrature_order'], 'verification_budget_after_next_query': 358,
                  'before_native_paid_episodes': 2, 'after_native_paid_episodes': 3,
                  'first_query': q1, 'first_raw_observation': raw, 'tau': paid['tau'],
                  'initialization_indices': paid['candidate_bank_indices'],
                  'restoration': restoration, 'initialization_replay': initialization,
                  'before_decision': baseline, 'after_decision': actual, 'comparison': contrast,
                  'parity_max_errors': checks, 'after_scores_recomputed': recomputed,
                  'after_scores_source': 'saved_actual_round2_decision',
                  'parity_scope': 'All saved marginal moments, outer weights, initial-to-q1 and updated-to-q2 cross-covariance slices; full after scores additionally checked on prespecified task0. Historical snapshots do not store full covariance or exact episode noise.',
                  'state_changes': changes, 'phase_cpu_seconds': phases,
                  'interpretation': 'Same-budget same-bank retrospective native posterior comparison. The first raw observation changes both its own point and other policies; query changes are not wholly attributable to cross-policy propagation. No counterfactual policy was executed.',
                  'cost': {key: attempt[key] for key in (
                      'saved_raw_observation_updates', 'query_score_calls', 'new_simulator_calls',
                      'new_verification_calls', 'new_target_truth_calls', 'new_source_labels',
                      'new_source_fits', 'new_deployments')}}
        result['cost'].update(cpu_seconds=time.process_time() - started,
                              wall_seconds=time.monotonic() - wall)
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        attempt.update(status='completed', result=str(result_path.relative_to(ROOT)))
        print(json.dumps({'task_index': task_index, **contrast, 'cpu_seconds': result['cost']['cpu_seconds']}), flush=True)
        return result
    except Exception:
        attempt['traceback'] = traceback.format_exc()
        raise
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.)
        attempt.update(cpu_seconds=time.process_time() - started, wall_seconds=time.monotonic() - wall,
                       maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       phase_cpu_seconds=phases)
        with (out / 'attempts.jsonl').open('a') as handle:
            handle.write(json.dumps(attempt, allow_nan=False) + '\n')


def summarize(output=None):
    """Summarize the 16 completed JSON records; no model calls or truth reads."""
    started, wall = time.process_time(), time.monotonic()
    out = ROOT / (output or OUTPUT)
    out.mkdir(parents=True, exist_ok=True)
    attempt = {'stage': 'summarize', 'status': 'failed',
               'started_at': datetime.now(timezone.utc).isoformat(),
               'query_score_calls': 0, 'saved_raw_observation_updates': 0,
               'new_simulator_calls': 0, 'new_verification_calls': 0,
               'new_target_truth_calls': 0, 'new_source_fits': 0}
    try:
        cases = [read(out / f'task{i:02d}.json') for i in range(16)]
        rows, parity_errors = [], []
        for index, case in enumerate(cases):
            assert case['task_index'] == index
            assert (case['candidate_count'], case['remaining'], case['quadrature_order']) == (104, 359, 3)
            c, change = case['comparison'], case['state_changes']
            after_gap = c['after_top_score'] - c['before_query_after_score']
            rows.append({'task_index': index, 'first_query': case['first_query'], **c,
                'after_query_after_gap_over_before_query': after_gap,
                'switch_strict_on_both_sides_at_1e_minus12': bool(c['query_changed'] and
                    c['after_query_before_gap_to_maximum'] > 1e-12 and after_gap > 1e-12),
                'unsampled_count': change['unsampled_count'],
                'unsampled_mean_changed_count': change['mean_changed_policy_count'],
                'unsampled_variance_changed_count': change['variance_changed_policy_count'],
                'unsampled_by_head': change['by_head'],
                'maximum_absolute_full_covariance_change': change['maximum_absolute_full_covariance_change']})
            pending = [case['parity_max_errors']]
            while pending:
                value = pending.pop()
                if isinstance(value, dict):
                    pending.extend(value.values())
                else:
                    parity_errors.append(float(value))
        recomputed = [c['task_index'] for c in cases if c['after_scores_recomputed']]
        assert recomputed == [0]
        result = {'task_count': 16, 'candidate_count': 104, 'remaining': 359, 'quadrature_order': 3,
            'query_changed_count': sum(r['query_changed'] for r in rows),
            'strict_non_tie_switch_count': sum(r['switch_strict_on_both_sides_at_1e_minus12'] for r in rows),
            'tie_tolerance': 1e-12,
            'non_tie_definition': 'Changed argmax and each selected query beats the other query by more than 1e-12 in its own state.',
            'query_distributions': {key: dict(sorted(Counter(r[key] for r in rows).items()))
                for key in ('first_query', 'before_query', 'after_query')},
            'query_transitions': [{'before_query': pair[0], 'after_query': pair[1], 'task_count': count}
                for pair, count in sorted(Counter((r['before_query'], r['after_query']) for r in rows).items())],
            'query_policy_counts': {side: {
                'unobserved_after_first_raw': sum(r[f'{side}_query_unobserved_after_first_raw'] for r in rows),
                'generated_beyond_original73': sum(r[f'{side}_query'] >= 73 for r in rows),
                'released_only_in_round2_index89plus': sum(r[f'{side}_query'] >= 89 for r in rows)}
                for side in ('before', 'after')},
            'unsampled_propagation': {
                'all_unsampled_mean_changed_tasks': sum(r['unsampled_mean_changed_count'] == r['unsampled_count'] for r in rows),
                'all_unsampled_variance_changed_tasks': sum(r['unsampled_variance_changed_count'] == r['unsampled_count'] for r in rows),
                'by_head': {head: {kind: {
                    'all_unsampled_changed_tasks': sum(r['unsampled_by_head'][head][kind]['changed_count'] == r['unsampled_count'] for r in rows),
                    'changed_policy_task_pairs': sum(r['unsampled_by_head'][head][kind]['changed_count'] for r in rows),
                    'maximum_absolute_change': max(r['unsampled_by_head'][head][kind]['max_absolute'] for r in rows)}
                    for kind in ('mean_delta', 'variance_delta')} for head in cases[0]['state_changes']['head_order']}},
            'maximum_parity_error': max(parity_errors), 'after_scores_recomputed_tasks': recomputed,
            'parity_scope': cases[0]['parity_scope'], 'interpretation': cases[0]['interpretation'],
            'rows': rows,
            'completed_case_cost': {key: sum(c['cost'][key] for c in cases) for key in
                ('cpu_seconds', 'query_score_calls', 'saved_raw_observation_updates',
                 'new_simulator_calls', 'new_verification_calls', 'new_target_truth_calls', 'new_source_fits')}}
        result['postprocessing_cpu_seconds'] = time.process_time() - started
        (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        attempt['status'] = 'completed'
        print(json.dumps({key: result[key] for key in ('task_count', 'query_changed_count',
            'strict_non_tie_switch_count', 'query_distributions', 'query_policy_counts',
            'maximum_parity_error', 'after_scores_recomputed_tasks', 'postprocessing_cpu_seconds')}), flush=True)
        return result
    except Exception:
        attempt['traceback'] = traceback.format_exc()
        raise
    finally:
        attempt.update(cpu_seconds=time.process_time() - started, wall_seconds=time.monotonic() - wall)
        with (out / 'postprocessing_attempts.jsonl').open('a') as handle:
            handle.write(json.dumps(attempt, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['case', 'summarize'])
    parser.add_argument('--task-index', type=int)
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.command == 'summarize':
        summarize(args.output)
    else:
        if args.task_index is None:
            parser.error('case requires --task-index')
        run_case(args.task_index, args.manifest, args.output)
