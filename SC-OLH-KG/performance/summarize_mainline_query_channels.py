"""Summarize frozen query-channel interventions, reusing prior analytic truth.

These are one-step selections on recorded histories, not new optimization
trajectories. Truth is loaded only after every registered counterfactual is
complete; only a selected policy absent from the existing cache is evaluated.
"""
import argparse
from collections import defaultdict

import numpy as np
from scipy.stats import norm

from acquisition.decision_backends import _EPS
from performance.diagnose_mainline_shared_neighbors import load_truth_cache, truth_value
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, load_pickle, read, save


MANIFEST = ROOT / 'performance/manifests/mainline_query_channels_v1.json'
NATIVE = 'legacy_native_neighbors'
PREDICTION_FIELDS = (
    'objective_mean', 'objective_epistemic', 'constraint_mean',
    'constraint_epistemic', 'constraint_aleatoric', 'probability_violation',
    'sampled_objective', 'sampled_constraint', 'sampled_margin', 'total',
)


def completed_inputs(plan):
    """Prevent any truth-dependent work until all registered scores are frozen."""
    output = ROOT / plan['output_dir']
    records = {}
    for seed in plan['target_seeds']:
        for arm in plan['arms']:
            directory = output / f'seed_{seed}' / arm
            names = ['recovery.json', 'native_frozen10.json' if arm == NATIVE else 'channels.json']
            values = {}
            for name in names:
                path = directory / name
                if not path.exists():
                    raise RuntimeError(f'All query counterfactuals must finish before truth: {path}')
                value = read(path)
                if value['status'] != 'complete':
                    raise RuntimeError(f'Query counterfactual is incomplete: {path}')
                values[name] = value
            records[seed, arm] = values
    return records


def prediction_detail(score, index):
    total = np.asarray(score['total'])
    order = np.argsort(-total, kind='stable')
    ranks = np.empty(len(total), dtype=int)
    ranks[order] = np.arange(1, len(total) + 1)
    return {
        'score_rank': int(ranks[index]),
        'strictly_better_score_count': int(np.sum(total > total[index])),
        'tied_score_count': int(np.sum(total == total[index])),
        'prediction': {field: float(score[field][index]) for field in PREDICTION_FIELDS},
    }


def stage_controls(arm, audit, counterfactual, problem):
    if arm == NATIVE:
        return {'full': audit['score'], 'native_frozen10': counterfactual['score']}
    assert counterfactual['action_ids'] == audit['scored_ids']
    result = {}
    for name, control in counterfactual['controls'].items():
        if name == 'full':
            result[name] = audit['score']
            continue
        means = counterfactual['anchor' if name == 'frozen10' else
            'full' if name == 'no_cross_noise' else 'own']
        variance = counterfactual['anchor' if name == 'frozen10' else
            'own' if name == 'no_cross_all_fixed_correlation' else 'full']
        noise = counterfactual['anchor' if name == 'frozen10' else
            'full' if name == 'no_cross_mean' else 'own']
        mean = np.asarray(means['mean'])
        var, noise_g = np.asarray(variance['variance']), np.asarray(noise['noise'])[:, 1]
        margin = mean[:, 2] + norm.ppf(1 - problem.alpha) * np.sqrt(np.maximum(noise_g, _EPS))
        result[name] = dict(control,
            objective_mean=mean[:, 0], constraint_mean=mean[:, 2] + problem.tau,
            objective_epistemic=var[:, 0], constraint_epistemic=var[:, 2],
            constraint_aleatoric=noise_g,
            probability_violation=norm.cdf(margin / np.sqrt(np.maximum(var[:, 2], _EPS))))
    return result


def selection_detail(score, audit, points, neighbors, initial_points, initial, get_truth):
    index = int(np.argmax(score['total']))
    point_id = audit['scored_ids'][index]
    point = points[point_id]
    truth = get_truth(point)
    neighbor_index = neighbors.index(point) if point in neighbors else None
    return {
        'selected_index': index, 'selected_id': point_id,
        'neighbor_index': neighbor_index,
        'new_neighbor': neighbor_index is not None and point not in initial_points,
        'unchanged_from_full': point_id == audit['selected_id'],
        'route': audit['action_sources'][index],
        'truth': truth,
        'feasible_improving': bool(truth['true_feasible']
                                   and truth['objective'] < initial['objective'] - 1e-12),
        **prediction_detail(score, index),
    }


def neighbor_details(score, audit, points, neighbors, improving, neighbor_truth):
    positions = {points[point_id]: index for index, point_id in enumerate(audit['scored_ids'])}
    result = []
    for neighbor_index in improving:
        point = neighbors[neighbor_index]
        index = positions[point]
        result.append({
            'neighbor_index': neighbor_index, 'truth': neighbor_truth[neighbor_index],
            'selected': index == int(np.argmax(score['total'])),
            **prediction_detail(score, index),
        })
    return result


def prediction_errors(score, audit, points, neighbors, fixed, neighbor_truth, tau):
    positions = {points[point_id]: index for index, point_id in enumerate(audit['scored_ids'])}
    indices = [positions[neighbors[index]] for index in fixed]
    predicted = np.column_stack([np.asarray(score[field])[indices] for field in
        ('objective_mean', 'constraint_mean', 'constraint_aleatoric')])
    predicted[:, 1] -= tau
    actual = np.asarray([[neighbor_truth[index]['objective'],
        neighbor_truth[index]['constraint_mean_centered'],
        neighbor_truth[index]['constraint_sigma'] ** 2] for index in fixed])
    return {
        'mean_bias_f_g_noise_variance': np.mean(predicted - actual, axis=0),
        'mse_f_g_noise_variance': np.mean((predicted - actual) ** 2, axis=0),
    }


def aggregate(rows, prefixes):
    groups = defaultdict(list)
    for row in rows:
        for stage in row['query_stages']:
            if stage['paid_prefix'] in prefixes:
                for name, control in stage['controls'].items():
                    groups[row['arm'], name].append(control['selected'])
    result = []
    for (arm, control), selections in groups.items():
        result.append({
            'arm': arm, 'control': control, 'queries': len(selections),
            'safe_count': sum(row['truth']['true_feasible'] for row in selections),
            'feasible_improving_count': sum(row['feasible_improving'] for row in selections),
            'new_neighbor_count': sum(row['new_neighbor'] for row in selections),
            'unchanged_from_full_count': sum(row['unchanged_from_full'] for row in selections),
            'selected_neighbor_indices': [row['neighbor_index'] for row in selections],
        })
    return result


def summarize(plan):
    inputs = completed_inputs(plan)
    output, parent = ROOT / plan['output_dir'], ROOT / plan['parent_output']
    destination = output / 'summary.json'
    if destination.exists() and read(destination).get('status') == 'complete':
        return read(destination)
    with attempt(output, 'summarize-query-channels', plan['domain']) as cost:
        cost.update(new_truth_function_calls=0, new_source_fits=0, download_bytes=0)
        parent_plan = read(ROOT / plan['parent_manifest'])
        prior = read(parent / 'diagnosis.json')
        cache = load_truth_cache(parent_plan, parent)
        extra_path = output / 'truth_new.json'
        extra = {tuple(row['point']): row['truth'] for row in read(extra_path)} if extra_path.exists() else {}
        for point, value in extra.items():
            cache.setdefault(point, {}).update(value)
        rows = []
        try:
            for seed in plan['target_seeds']:
                parent_seed = parent / f'seed_{seed}'
                neighbors = [tuple(map(int, point)) for point in
                             np.load(parent_seed / 'neighbors.npz')['points']]
                stop = read(parent_seed / 'initialization_only/search.json')
                initial_points = {tuple(row['point']) for row in stop['history']}
                seed_truth = next(row for row in prior['seeds'] if row['seed'] == seed)
                initial, neighbor_truth = seed_truth['initial_truth'], seed_truth['neighbor_truth']
                improving = [index for index, value in enumerate(neighbor_truth)
                    if value['true_feasible'] and value['objective'] < initial['objective'] - 1e-12]
                problem = load_pickle(parent_seed / 'initialization_only/search_engine.pkl').problem
                get_truth = lambda point: truth_value(problem, point, cache, extra, cost)
                for arm in plan['arms']:
                    directory = parent_seed / arm
                    audit = read(directory / 'query_audit.json')
                    points = [tuple(map(int, point)) for point in
                              np.load(directory / 'query_policies.npz')['points']]
                    prior_path = next(row for row in prior['rows']
                                      if row['seed'] == seed and row['arm'] == arm)
                    fixed = prior_path['never_sampled_prediction_change']['fixed_neighbor_indices']
                    record = inputs[seed, arm]
                    frozen = record['native_frozen10.json' if arm == NATIVE else 'channels.json']
                    stages, snapshots = [], defaultdict(dict)
                    for stage, counterfactual in zip(audit['stages'], frozen['stages'], strict=True):
                        assert stage['paid_prefix'] == counterfactual['paid_prefix']
                        controls = stage_controls(arm, stage, counterfactual, problem)
                        results = {}
                        for name, score in controls.items():
                            selected = selection_detail(score, stage, points, neighbors,
                                initial_points, initial, get_truth)
                            results[name] = {
                                'selected': selected,
                                'improving_neighbors': neighbor_details(score, stage, points,
                                    neighbors, improving, neighbor_truth),
                                'sampled_feasible_count': score['sampled_feasible_count'],
                                'sampled_feasibility_mode': score['sampled_feasibility_mode'],
                            }
                            if stage['paid_prefix'] in (10, 12):
                                errors = prediction_errors(score, stage, points, neighbors, fixed, neighbor_truth,
                                                           problem.tau)
                                snapshots[name][str(stage['paid_prefix'])] = errors
                                if name == 'full':
                                    reference = prior_path['never_sampled_prediction_change']['snapshots'][str(stage['paid_prefix'])]
                                    np.testing.assert_allclose(errors['mse_f_g_noise_variance'],
                                        reference['mse_f_g_noise_variance'], rtol=1e-10, atol=1e-12)
                        stages.append({'query_index': stage['iteration'] + 1,
                            'paid_prefix': stage['paid_prefix'], 'candidate_count': len(stage['scored_ids']),
                            'controls': results})
                    propagation = {}
                    for name, values in snapshots.items():
                        propagation[name] = {'snapshots': values,
                            'mse_reduction_f_g_noise_variance': (
                                np.asarray(values['10']['mse_f_g_noise_variance'])
                                - np.asarray(values['12']['mse_f_g_noise_variance']))}
                    rows.append({'seed': seed, 'arm': arm, 'initial_truth': initial,
                        'all_raw_order_parity': record['recovery.json']['all_raw_order_parity'],
                        'initial10_controls_match_full': frozen['first_query_full_parity' if arm == NATIVE
                                                                else 'initial10_controls_match_full'],
                        'improving_neighbor_indices': improving, 'query_stages': stages,
                        'never_sampled_prediction_change': {
                            'fixed_neighbor_indices': fixed, 'count': len(fixed),
                            'full_parent_mse_parity': True, 'controls': propagation,
                            'scope': 'New shared neighbors never sampled in the full actual 13-row trajectory.'}})
            result = {'status': 'complete', 'rows': rows,
                'main_paid11_12': aggregate(rows, (11, 12)),
                'initial_paid10': aggregate(rows, (10,)),
                'barrier': {'recovery_files': 9, 'channel_files': 6, 'native_frozen_files': 3},
                'analytical_truth_new_unique_policies': len(extra),
                'interpretation': 'One-step counterfactual query selections on actual histories; no counterfactual continuation, certification or deployment reward.',
                'limitations': plan['limitations'],
                'prediction_error_order': ['objective_mean', 'constraint_mean_centered', 'constraint_episode_variance'],
                'rank_rule': 'Descending TS total with stable original candidate order; tie counts retained.'}
        finally:
            save(extra_path, [{'point': list(point), 'truth': value} for point, value in extra.items()])
    result['cost'] = cost
    save(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    args = parser.parse_args()
    result = summarize(read(args.manifest))
    print({'status': result['status'], 'cases': len(result['rows']),
           'new_truth_function_calls': result['cost']['new_truth_function_calls']}, flush=True)


if __name__ == '__main__':
    main()
