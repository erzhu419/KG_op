"""Compare the fixed role prior with saved baselines using cached truth only."""
import argparse

import numpy as np

from performance.diagnose_mainline_shared_neighbors import load_truth_cache
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.summarize_mainline_initialization import (
    complete_truth, distribution, point_prediction, snapshot_metrics,
)


MANIFEST = ROOT / 'performance/manifests/mainline_role_prior_v1.json'
BASELINE = ROOT / 'paper_artifacts/mainline_initialization_audit_v1/summary.json'


def observation_summary(record, snapshots):
    changes = []
    for step, observed in enumerate(record['pre_observation']):
        index = observed['bank_index']
        before, after = snapshots[step:step + 2]
        residual = observed['centered_raw'][1] - before['mean'][index, 2]
        change = after['mean'][index, 2] - before['mean'][index, 2]
        changes.append({
            'initialization_row': step, 'bank_index': index,
            'pre_g_mean': before['mean'][index, 2],
            'post_g_mean': after['mean'][index, 2],
            'pre_g_residual': residual,
            'fraction_raw_residual_followed_by_g_mean': (
                None if abs(residual) <= 1e-12 else change / residual),
            'g_noise_variance_multiplier': (
                after['noise'][index, 1] / before['noise'][index, 1]),
            'pre_log_predictive': observed['log_predictive'],
        })
    return changes, {
        'count': len(changes),
        'fraction_raw_residual_followed_by_g_mean': distribution([
            row['fraction_raw_residual_followed_by_g_mean'] for row in changes
            if row['fraction_raw_residual_followed_by_g_mean'] is not None]),
        'g_noise_variance_multiplier': distribution([
            row['g_noise_variance_multiplier'] for row in changes]),
        'prequential_g_log_score': float(sum(
            row['pre_log_predictive']['heads'][1] for row in changes)),
        'prequential_joint_log_score': float(sum(
            row['pre_log_predictive']['joint'] for row in changes)),
    }


def worst_noise(snapshot, points, cache, neighbors):
    known = [(i, cache.get(tuple(map(int, point)))) for i, point in enumerate(points)]
    known = [(i, truth) for i, truth in known if complete_truth(truth)]
    if not known:
        return {'known_truth_count': 0, 'missing_truth_count': len(points)}
    index, truth = max(known, key=lambda row: (
        np.sqrt(snapshot['noise'][row[0], 1]) / row[1]['constraint_sigma']))
    return {'known_truth_count': len(known), 'missing_truth_count': len(points) - len(known),
        'bank_index': index,
        'neighbor_index': neighbors.index(index) if index in neighbors else None,
        'truth': truth, **point_prediction(snapshot, index, truth)}


def summarize_case(plan, seed, cache, baselines, seed_truth):
    directory = ROOT / plan['output_dir'] / f'seed_{seed}'
    record = read(directory / 'trajectory.json')
    points = np.load(directory / 'bank.npz')['points']
    snapshots = [{**s, **{key: np.asarray(s[key]) for key in ('mean', 'variance', 'noise')}}
                 for s in record['snapshots']]
    neighbors = record['neighbor_bank_indices']
    new_indices = [neighbors[i] for i in record['new_neighbor_indices']]
    truths = [cache.get(tuple(map(int, points[index]))) for index in new_indices]
    incumbent = seed_truth['initial_truth']
    improving = [i for i in record['new_neighbor_indices']
        if (truth := cache.get(tuple(map(int, points[neighbors[i]])))) is not None
        and truth.get('true_feasible') is True
        and truth.get('objective', float('inf')) < incumbent['objective'] - 1e-12]
    metrics = [{'paid_prefix': s['paid_prefix'], **snapshot_metrics(s, new_indices, truths)}
               for s in snapshots]
    focused = [('initial0', record['initial_bank_indices'][0], None)]
    focused += [(f'neighbor_{i}', neighbors[i], i) for i in improving]
    focus = []
    for name, index, neighbor in focused:
        truth = cache.get(tuple(map(int, points[index])))
        focus.append({'name': name, 'bank_index': index, 'neighbor_index': neighbor,
            'truth': truth, 'snapshots': [point_prediction(s, index, truth) for s in snapshots]})
    changes, observations = observation_summary(record, snapshots)
    comparisons = []
    for baseline in [row for row in baselines['rows'] if row['seed'] == seed]:
        endpoints = []
        for prefix in (0, 10):
            before = next(row for row in baseline['new_neighbor_trajectory'] if row['paid_prefix'] == prefix)
            after = next(row for row in metrics if row['paid_prefix'] == prefix)
            available = bool(after['known_truth_count'] and before['known_truth_count'])
            endpoints.append({'paid_prefix': prefix, 'baseline': before, 'role_prior': after,
                'g_mse_change': (float(after['mse_f_g_noise_variance'][1])
                                 - before['mse_f_g_noise_variance'][1]) if available else None,
                'g_mean_bias_change': (float(after['mean_bias_f_g_noise_variance'][1])
                                      - before['mean_bias_f_g_noise_variance'][1]) if available else None})
        old_log_score = sum(row['pre_log_predictive']['heads'][1]
                            for row in baseline['observation_changes'])
        comparisons.append({'baseline_arm': baseline['arm'], 'endpoints': endpoints,
            'baseline_observation_summary': baseline['observation_summary'],
            'prequential_g_log_score_change': observations['prequential_g_log_score'] - old_log_score})
    row = {'seed': seed, 'new_neighbor_count': len(new_indices),
        'new_neighbor_indices': record['new_neighbor_indices'],
        'missing_neighbor_truth_indices': [i for i, truth in zip(record['new_neighbor_indices'], truths)
                                           if not complete_truth(truth)],
        'improving_neighbor_indices': improving,
        'new_neighbor_trajectory': metrics, 'focus_policies': focus,
        'observation_changes': changes, 'observation_summary': observations,
        'initial10_worst_noise_all_bank': worst_noise(snapshots[-1], points, cache, neighbors),
        'baseline_comparisons': comparisons,
        'trajectory_file': str(directory / 'trajectory.json')}
    return row, snapshots, new_indices, truths


def query_summary(plan, seed, cache, initial):
    path = ROOT / plan['output_dir'] / f'seed_{seed}' / 'query_scores.json'
    record = read(path)
    points = np.load(ROOT / record['policy_bank_file'])['points']
    parent = ROOT / plan['parent_output'] / f'seed_{seed}'
    neighbors = [tuple(map(int, point)) for point in np.load(parent / 'neighbors.npz')['points']]
    history = read(parent / 'current_structural_mixture/search.json')['history']
    initial_points = {tuple(row['point']) for row in history[:10]}
    stages = []
    for stage in record['stages']:
        observed = {tuple(row['point']) for row in history[:stage['paid_prefix']]}
        action_points = [tuple(map(int, points[index])) for index in stage['action_ids']]
        action_positions = {point: index for index, point in enumerate(action_points)}
        unseen = [point for point in neighbors if point not in observed and point in action_positions]
        predictions = {key: np.asarray(value) for key, value in stage['full_predictions'].items()}
        controls = {}
        for name in ('full', 'frozen10'):
            score = stage[name]
            selected = int(score['selected_id'])
            point = tuple(map(int, points[selected]))
            truth = cache.get(point)
            known = complete_truth(truth)
            controls[name] = {'selected_id': selected, 'truth': truth,
                'truth_available': known,
                'neighbor_index': neighbors.index(point) if point in neighbors else None,
                'new_neighbor': point in neighbors and point not in initial_points,
                'feasible_improving': (bool(truth['true_feasible']
                    and truth['objective'] < initial['objective'] - 1e-12) if known else None),
                'unchanged_from_baseline': selected == stage['baseline_selected_id'],
                'sampled_feasibility_mode': score['sampled_feasibility_mode'],
                'sampled_feasible_count': score['sampled_feasible_count'],
                'sampled_objective': score['sampled_objective'][score['selected_index']],
                'sampled_margin': score['sampled_margin'][score['selected_index']]}
            if name == 'full':
                index = score['selected_index']
                controls[name]['prediction'] = {
                    'f_mean': predictions['mean'][index, 0],
                    'g_mean_centered': predictions['mean'][index, 2],
                    'g_epistemic_variance': predictions['variance'][index, 2],
                    'g_sigma': np.sqrt(predictions['noise'][index, 1])}
        baseline_truth = cache.get(tuple(map(int, points[stage['baseline_selected_id']])))
        stages.append({'paid_prefix': stage['paid_prefix'], 'controls': controls,
            'baseline_selected_id': stage['baseline_selected_id'],
            'baseline_truth': baseline_truth,
            'full_unsampled_neighbor_predictions': snapshot_metrics(predictions,
                [action_positions[point] for point in unseen], [cache.get(point) for point in unseen]),
            'full_differs_from_frozen10': controls['full']['selected_id'] != controls['frozen10']['selected_id']})
    return {'seed': seed, 'available': True, 'query_stages': stages, 'query_file': str(path)}


def summarize(plan):
    output, parent = ROOT / plan['output_dir'], ROOT / plan['parent_output']
    for seed in plan['target_seeds']:
        for name in ('trajectory.json', 'query_scores.json'):
            path = output / f'seed_{seed}' / name
            if not path.exists() or read(path).get('status') != 'complete':
                raise RuntimeError(f'All registered role trajectories and query scores must finish before cached truth: {path}')
    with attempt(output, 'summarize-role-prior', plan['domain']) as cost:
        cost.update(new_truth_function_calls=0, new_source_fits=0, download_bytes=0)
        cache, baselines = load_truth_cache(plan, parent), read(BASELINE)
        parent_truth = read(parent / 'diagnosis.json')
        rows, retained, queries = [], [], []
        for seed in plan['target_seeds']:
            seed_truth = next(row for row in parent_truth['seeds'] if row['seed'] == seed)
            row, snapshots, indices, truths = summarize_case(plan, seed, cache, baselines, seed_truth)
            rows.append(row); retained.append((snapshots, indices, truths))
            queries.append(query_summary(plan, seed, cache, seed_truth['initial_truth']))
        trajectory = []
        truths = [truth for _, _, values in retained for truth in values]
        for prefix in (0, 10):
            snapshot = {key: np.concatenate([snapshots[prefix][key][indices]
                for snapshots, indices, _ in retained]) for key in ('mean', 'variance', 'noise')}
            trajectory.append({'paid_prefix': prefix,
                **snapshot_metrics(snapshot, list(range(len(truths))), truths)})
        changes = [change for row in rows for change in row['observation_changes']]
        aggregate = {'new_neighbor_endpoints': trajectory,
            'fraction_raw_residual_followed_by_g_mean': distribution([
                row['fraction_raw_residual_followed_by_g_mean'] for row in changes
                if row['fraction_raw_residual_followed_by_g_mean'] is not None]),
            'g_noise_variance_multiplier': distribution([row['g_noise_variance_multiplier'] for row in changes])}
        query_counts = []
        for prefixes in ((10, 11, 12), (11, 12)):
            stages = [stage for query in queries for stage in query['query_stages']
                      if stage['paid_prefix'] in prefixes]
            for name in ('full', 'frozen10'):
                selections = [stage['controls'][name] for stage in stages]
                query_counts.append({'control': name, 'paid_prefixes': list(prefixes),
                    'query_count': len(selections),
                    'known_truth_count': sum(row['truth_available'] for row in selections),
                    'safe_count': sum(row['truth_available'] and row['truth']['true_feasible']
                                      for row in selections),
                    'feasible_improving_count': sum(row['feasible_improving'] is True for row in selections),
                    'new_neighbor_count': sum(row['new_neighbor'] for row in selections),
                    'selected_neighbor_indices': [row['neighbor_index'] for row in selections]})
        result = {'status': 'complete', 'rows': rows, 'aggregate': aggregate,
            'query_replays': queries, 'query_counts': query_counts,
            'scope': 'Fixed prior on three saved observation streams from one opened geometry. Neighbor entries are not independent tasks; query replays are one-step scores on recorded histories, not deployments.',
            'missing_truth_policy': 'Compatible cached truth only; missing values remain missing and no truth function is called.',
            'baseline_file': str(BASELINE)}
    result['cost'] = cost
    save(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    args = parser.parse_args()
    result = summarize(read(args.manifest))
    print({'status': result['status'], 'cases': len(result['rows']),
           'cpu_seconds': result['cost']['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    main()
