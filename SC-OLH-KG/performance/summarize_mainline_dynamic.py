"""Summarize a completed dynamic pilot using saved results and traces only."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = 'paper_artifacts/mainline_d1000_dynamic_v1'
TOLERANCE = 1e-12


def read(path):
    return json.loads(Path(path).read_text())


def changes(rows):
    rows = [r for r in rows if r['point_count']]
    if not rows:
        return {'units': 0}
    before = np.asarray([r['mse_before'] for r in rows])
    after = np.asarray([r['mse_after'] for r in rows])
    delta = after-before
    return {'units': len(rows), 'point_counts': [r['point_count'] for r in rows],
        'mean_mse_before': before.mean(axis=0).tolist(), 'mean_mse_after': after.mean(axis=0).tolist(),
        'mean_mse_change': delta.mean(axis=0).tolist(),
        'improved_unit_counts': (delta < -TOLERANCE).sum(axis=0).tolist(),
        'worsened_unit_counts': (delta > TOLERANCE).sum(axis=0).tolist()}


def matched_mse(rows, models, metric):
    by_key = {(r['task_index'], r['arm']): r for r in rows}
    paired = []
    for task in sorted({r['task_index'] for r in rows}):
        lhs, rhs = (by_key[task, model] for model in models)
        if lhs['indices'] == rhs['indices'] and lhs['point_count']:
            paired.append({'task_index': task,
                'source_minus_gp': (np.asarray(lhs[metric])-rhs[metric]).tolist()})
    return {'matched_point_set_units': len(paired), 'unit_differences': paired,
        'mean_source_minus_gp': np.mean([r['source_minus_gp'] for r in paired], axis=0).tolist() if paired else None}


def summarize(output=DEFAULT):
    folder = ROOT/output
    # The complete results file is written only after all actual verifiers and
    # post-freeze scoring. Do not open inputs/traces for an unfinished run.
    result = read(folder/'results.json')
    if not (result['all_recommendations_frozen_before_verification']
            and result['all_verification_finished_before_truth_scoring']):
        raise ValueError('Summary requires the completed verification/scoring stage')
    frozen = read(folder/'frozen_inputs.json')
    spec, diagnostic = frozen['plan'], result['dynamic_candidate_diagnostics']
    models, units = spec['models'], range(spec['task_count'])
    policies = {model: spec.get('policies_by_model', {}).get(model, spec['policies']) for model in models}
    if len(result['rows']) != spec['task_count']*sum(len(policies[model]) for model in models):
        raise ValueError('Completed results do not contain every registered arm')
    predictions, comparisons = [], []
    initial = diagnostic['initial_generated_predictions']
    for model in models:
        rows = [r for r in initial if r['arm'] == model]
        predictions.append({'arm': model, 'phase': 'initial_all_generated', 'units': len(rows),
            'point_counts': [r['point_count'] for r in rows],
            'mean_mse': np.mean([r['mse'] for r in rows], axis=0).tolist()})
    comparisons.append({'phase': 'initial_all_generated', **matched_mse(initial, models[:2], 'mse')})
    groups = [('initial_to_final_unobserved_generated', diagnostic['initial_to_final_prediction_changes']),
              ('round2_unreleased_before_observation', diagnostic['round2_unreleased_prediction_changes'])]
    for round_index in range(1, spec['rounds']+1):
        rows = [r for r in diagnostic['round_prediction_changes'] if r['round'] == round_index]
        groups.append((f'round{round_index}_all_generated_unsampled', rows))
        groups.append((f'round{round_index}_released_generated_unsampled', [
            {'task_index': r['task_index'], 'arm': r['arm'], **r['released_new_prediction_change']} for r in rows]))
    for phase, rows in groups:
        predictions.extend({'arm': model, 'phase': phase, **changes([r for r in rows if r['arm'] == model])}
                           for model in models)
        comparisons.append({'phase': phase, **matched_mse(rows, models[:2], 'mse_after')})
    paired_rewards = []
    scores = {(r['task_index'], r['arm'], r['policy']): r['scores'] for r in result['rows']}
    for comparison in result['paired_comparisons']:
        record = dict(comparison)
        for metric in ('actual_safe_reward', 'conditional_safe_reward'):
            delta = np.asarray([scores[(i, *comparison['lhs'])][metric]-scores[(i, *comparison['rhs'])][metric] for i in units])
            record[metric] = {'per_unit_difference': delta.tolist(), 'mean_difference': float(delta.mean()),
                'positive_units': int(np.sum(delta > TOLERANCE)), 'negative_units': int(np.sum(delta < -TOLERANCE))}
        paired_rewards.append(record)
    interventions = []
    for model in models:
        for round_index in range(1, spec['rounds']+1):
            rows = [r for r in diagnostic['newonly_terminal_interventions'] if r['arm'] == model and r['round'] == round_index]
            delta = np.asarray([r['conditional_safe_reward_difference'] for r in rows])
            interventions.append({'arm': model, 'round': round_index, 'units': len(rows),
                'changed_recommendations': sum(r['recommendation_changed'] for r in rows),
                'conditional_gains': int(np.sum(delta > TOLERANCE)), 'conditional_losses': int(np.sum(delta < -TOLERANCE)),
                'mean_conditional_difference': float(delta.mean()), 'per_unit_conditional_difference': delta.tolist()})
    queries, query_counts, cases = [], [], {}
    original = frozen['dynamic_candidates']['original_count']
    for model in models:
        for i in units:
            case = read(folder/f'task{i:02d}'/model/'case.json')
            cases[i, model] = case
            for trace in case['rounds']:
                query = trace['decision']['query']
                queries.append({'task_index': i, 'arm': model, 'round': trace['round'],
                                'query': query, 'is_generated': query >= original})
        rows = [r for r in queries if r['arm'] == model]
        query_counts.append({'arm': model, 'queries': len(rows),
            'generated_queries': sum(r['is_generated'] for r in rows),
            'original_queries': sum(not r['is_generated'] for r in rows)})
    attempts = [json.loads(line) for line in (folder/'attempts.jsonl').read_text().splitlines()]
    preparation = read(folder/'preparation_audit.json')
    stage_costs = []
    for stage in dict.fromkeys(a['stage'] for a in attempts):
        rows = [a for a in attempts if a['stage'] == stage]
        stage_costs.append({'stage': stage, 'attempt_status_counts': dict(Counter(a['status'] for a in rows)),
            'cpu_seconds': sum(a['cpu_seconds'] for a in rows),
            'wall_seconds_sum': sum(a['wall_seconds'] for a in rows),
            'maximum_rss_kib': max(a['maximum_rss_kib'] for a in rows)})
    cost = {**result['cost'], 'preparation_cpu_seconds': preparation['cpu_seconds'],
        'attempt_cpu_seconds': sum(a['cpu_seconds'] for a in attempts),
        'total_recorded_cpu_seconds': preparation['cpu_seconds']+sum(a['cpu_seconds'] for a in attempts),
        'attempts_by_stage': stage_costs}
    cost['case_attempts_by_model'] = []
    for index, model in enumerate(models):
        rows = [a for a in attempts if a['stage'] == 'case' and a['case_index'] % len(models) == index]
        cost['case_attempts_by_model'].append({'arm': model, 'attempts': len(rows),
            'cpu_seconds': sum(a['cpu_seconds'] for a in rows),
            'maximum_rss_kib': max(a['maximum_rss_kib'] for a in rows)})
    summary = {'source_results': str(folder/'results.json'), 'completed_at': result['completed_at'],
        'pairing_unit': spec['pairing_unit'], 'scope': spec['evaluation_scope'],
        'candidate_counts': diagnostic['counts'], 'arm_summary': result.get('arm_summary', result.get('four_arm_summary')),
        'paired_reward_differences': paired_rewards, 'head_order': diagnostic['head_order'],
        'numerical_zero_tolerance': TOLERANCE, 'generated_prediction_summary': predictions,
        'matched_source_minus_gp_mse': comparisons,
        'newonly_terminal_intervention_summary': interventions,
        'final_recommendations': diagnostic['final_recommendations'],
        'final_recommendation_counts': diagnostic['final_recommendation_counts'],
        'queries': queries, 'query_counts': query_counts,
        'bank_ceilings': diagnostic['post_verification_bank_ceilings'], 'cost': cost,
        'interpretation': 'MSE means weight the registered paired evaluation units equally. Source/GP MSE differences '
            'include only units with identical unsampled point sets. Conditional marginal-intervention '
            'rewards are diagnostics, not extra verified arms; bank best_quality omits verifier power.'}
    if len(models) >= 3:
        source, control = models[0], models[2]
        source_online, control_online = policies[source][-1], policies[control][-1]
        comparison = next(r for r in paired_rewards if
            r['lhs'] == [source, source_online] and r['rhs'] == [control, control_online])
        final = {(r['task_index'], r['arm'], r['policy']): r for r in result['rows']}
        summary['source_full_vs_no_cross'] = {'paired_reward_difference': comparison,
            'paths': [{'task_index': i,
                'source_full_queries': [q['query'] for q in queries if q['task_index'] == i and q['arm'] == source],
                'no_cross_queries': [q['query'] for q in queries if q['task_index'] == i and q['arm'] == control],
                'source_full_primary': final[i, source, source_online]['action']['primary'],
                'no_cross_primary': final[i, control, control_online]['action']['primary']} for i in units],
            'scope': 'Actual independent online paths from identical post-initialization marginals. '
                'The control updates only the queried policy; the comparison measures cross-policy '
                'dependence in acquisition and subsequent updates after shared initialization, '
                'not real-observation propagation alone or all source-prior benefits.'}
    dual = 'source_structural_dual_channel'
    if dual in models:
        online = policies[dual][-1]
        references = {'full': [models[0], policies[models[0]][-1]],
            'no_cross': [models[2], policies[models[2]][-1]],
            'gp': [models[1], policies[models[1]][-1]],
            'stop': [models[0], policies[models[0]][0]]}
        paired = {f'dual_vs_{name}': next(r for r in paired_rewards
            if r['lhs'] == [dual, online] and r['rhs'] == reference)
            for name, reference in references.items()}
        agreement, paths = [], []
        for i in units:
            path = {'task_index': i}
            for name, model in [('dual', dual), *[(name, value[0]) for name, value in references.items() if name != 'stop']]:
                path[f'{name}_queries'] = [r['decision']['query'] for r in cases[i, model]['rounds']]
                path[f'{name}_primary'] = final[i, model, policies[model][-1]]['action']['primary']
            path['stop_primary'] = final[(i, *references['stop'])]['action']['primary']
            paths.append(path)
            for full, split in zip(cases[i, models[0]]['rounds'], cases[i, dual]['rounds']):
                row = {'task_index': i, 'round': split['round'],
                    'same_query': full['decision']['query'] == split['decision']['query'],
                    'same_raw_observation': bool(np.array_equal(full['raw_observation'], split['raw_observation'])),
                    'query_scores_max_abs_difference': float(np.max(abs(
                        np.asarray(full['decision']['scores73'])-split['decision']['scores73'])))}
                for phase in ('before', 'after'):
                    key = f'selection_{phase}'
                    if key in split:
                        row[f'selection_{phase}_max_abs_difference'] = {field: float(np.max(abs(
                            np.asarray(split[key][field])-full[phase][field])))
                            for field in ('mean', 'point_cov', 'identity_weights')}
                agreement.append(row)
        summary['dual_channel'] = {**paired, 'paths': paths,
            'full_query_channel_agreement': agreement,
            'scope': 'Actual dual-channel arm: full posterior selects each query with the existing order3 '
                'acquisition; an independently updating point-only posterior selects the terminal primary. '
                'Both consume the same single paid raw episode per round. This is a declared heuristic, '
                'not a rederived KG objective for the split terminal rule. Comparisons are paired total-arm '
                'effects, not independent additive causal shares.'}
    (folder/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=DEFAULT, help='Completed artifact directory, relative to project root or absolute')
    print(json.dumps(summarize(parser.parse_args().output), indent=2, allow_nan=False))
