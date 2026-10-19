"""Saved-only response-novel candidate outcomes after complete verification."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = 'paper_artifacts/mainline_d1000_new_bank_v1'
DUAL, FROZEN = 'source_structural_dual_channel', 'source_structural_frozen_query'
TOLERANCE = 1e-12


def read(path):
    return json.loads((ROOT / path).read_text())


def coordinate_matches(point, rows):
    return [i for i, row in enumerate(rows) if max(abs(x-y) for x, y in zip(point, row)) <= TOLERANCE]


def analyze(output=DEFAULT):
    folder = ROOT / output
    started, wall = time.process_time(), time.monotonic()
    attempt = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'failed',
        'new_simulator_calls': 0, 'new_truth_calls': 0, 'new_integrations': 0,
        'new_source_fits': 0, 'new_bootstrap_draws': 0}
    try:
        result = read(folder / 'results.json')
        assert result['all_recommendations_frozen_before_verification'] and result['all_verification_finished_before_truth_scoring']
        summary, adaptation = read(folder / 'summary.json'), read(folder / 'query_adaptation_analysis.json')
        frozen, audit = read(folder / 'frozen_inputs.json'), read(folder / 'bank_preparation_audit.json')
        spec, original = frozen['plan'], frozen['dynamic_candidates']['original_count']
        previous = read(audit['previous_frozen_inputs'])
        old_coordinates, coordinates = read(previous['plan']['query_coordinates']), read(spec['query_coordinates'])
        assert len(result['rows']) == spec['task_count'] * sum(len(spec['policies_by_model'][m]) for m in spec['models'])
        mappings = []
        for i, point in enumerate(frozen['bank']):
            mean = coordinate_matches(coordinates['mean'][i], old_coordinates['mean'])
            risk = coordinate_matches(coordinates['risk'][i], old_coordinates['risk'])
            physical = [j for j, old_point in enumerate(previous['bank']) if point == old_point]
            mappings.append({'index': i, 'appended': i >= original, 'physical_match_old_indices': physical,
                'mean4_match_old_indices': mean, 'risk10_match_old_indices': risk,
                'joint14_match_old_indices': sorted(set(mean) & set(risk)),
                'physical_novel': not physical, 'response_novel': not mean,
                'risk10_novel': not risk, 'joint14_novel': not (set(mean) & set(risk))})

        def flags(index):
            return {k: mappings[index][k] for k in ('index', 'appended', 'physical_novel', 'response_novel', 'joint14_novel')}

        outcomes = []
        for row in result['rows']:
            task, arm, policy = (row[k] for k in ('task_index', 'arm', 'policy'))
            primary = row['action']['primary']
            unseen = primary not in row['observed_indices']
            path = [q for q in summary['queries'] if q['task_index'] == task and q['arm'] == arm]
            if policy == spec['policies'][0]:
                path = []
            outcomes.append({'task_index': task, 'arm': arm, 'policy': policy,
                'queries': [{'round': q['round'], **flags(q['query'])} for q in path],
                'final': flags(primary), 'unobserved': unseen, 'certified': row['verification']['certified'],
                **{k: row['scores'][k] for k in ('actual_safe_reward', 'conditional_safe_reward', 'true_feasible', 'false_certificate')},
                'response_novel_unsampled': mappings[primary]['response_novel'] and unseen})
        groups = []
        for arm in spec['models']:
            for policy in spec['policies_by_model'][arm]:
                rows = [r for r in outcomes if (r['arm'], r['policy']) == (arm, policy)]
                novel = [r for r in rows if r['response_novel_unsampled']]
                groups.append({'arm': arm, 'policy': policy, 'tasks': len(rows),
                    'response_novel_unsampled_recommendations': len(novel),
                    'certified_response_novel_unsampled_recommendations': sum(r['certified'] for r in novel),
                    'mean_actual_safe_reward_from_response_novel_unsampled': sum(r['actual_safe_reward'] for r in novel) / len(rows),
                    'mean_conditional_safe_reward_from_response_novel_unsampled': sum(r['conditional_safe_reward'] for r in novel) / len(rows)})
        online = spec['policies'][1]
        paired = {name: next(r for r in summary['paired_reward_differences']
            if r['lhs'] == [DUAL, online] and r['rhs'] == [arm, online])
            for name, arm in [('dual_minus_frozen', FROZEN), ('dual_minus_gp', spec['models'][1])]}
        tasks = [{'task_index': t['task_index'], 'second_query_changed': t['second_query_changed'],
            'dual_final': flags(t['dual']['primary']), 'frozen_final': flags(t['frozen']['primary']),
            'dual_certified': t['dual']['certified'], 'frozen_certified': t['frozen']['certified'],
            'actual_safe_reward_difference': t['dual']['actual_safe_reward'] - t['frozen']['actual_safe_reward']}
            for t in adaptation['tasks']]
        appended = mappings[original:]
        analysis = {'source_results': str(folder / 'results.json'), 'comparison_old_bank': audit['previous_frozen_inputs'],
            'equivalence_absolute_tolerance': TOLERANCE, 'all_old_index_matches': mappings,
            'appended_novel_counts': {key: sum(r[key] for r in appended)
                for key in ('physical_novel', 'response_novel', 'risk10_novel', 'joint14_novel')},
            'appended_physical_and_joint14_novel': sum(r['physical_novel'] and r['joint14_novel'] for r in appended),
            'outcomes': outcomes, 'response_novel_unsampled_reward_by_arm': groups,
            'primary_paired_comparisons': paired, 'task_query_changes': tasks,
            'bank_ceilings': summary['bank_ceilings'],
            'query_adaptation_controls': adaptation['controls'],
            'selection_prediction_reference': {'arm': spec['models'][0],
                'generated_prediction_summary': [r for r in summary['generated_prediction_summary'] if r['arm'] == spec['models'][0]],
                'full_dual_query_channel_agreement': summary['dual_channel']['full_query_channel_agreement'],
                'scope': 'Full-source predictions describe the dual query channel when saved channel agreement holds; they do not describe the independent point-only terminal channel.'},
            'interpretation': 'Response-novel means no old-bank mean4 match: these public three-group means and overall SD determine Inventory responses. Physical or joint14 novelty alone is insufficient. Novelty is descriptive, never a selection rule; near-equivalent regions need not be substantially different. Reward contributions divide by all registered tasks. Paired intervals and post-verification feasibility counts are reused unchanged.'}
        (folder / 'new_bank_transfer_analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False) + '\n')
        attempt['status'] = 'completed'
        return analysis
    except Exception:
        attempt['traceback'] = traceback.format_exc()
        raise
    finally:
        attempt.update(cpu_seconds=time.process_time() - started, wall_seconds=time.monotonic() - wall)
        with (folder / 'new_bank_transfer_analysis_attempts.jsonl').open('a') as handle:
            handle.write(json.dumps(attempt, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=DEFAULT)
    print(json.dumps(analyze(parser.parse_args().output)['response_novel_unsampled_reward_by_arm'], allow_nan=False))
