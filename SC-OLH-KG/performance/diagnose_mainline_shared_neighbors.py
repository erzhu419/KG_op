"""Post-verification coverage, recorded prediction changes and deployed benefit.

Only missing analytical truth is evaluated. Query posterior values come from
the original scoring calls; this module never restores or updates a model.
"""
import argparse

import numpy as np
from scipy.stats import norm

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, load_pickle, read, save


MANIFEST = ROOT / 'performance/manifests/mainline_shared_neighbors_v1.json'
FULL = 'current_structural_mixture'
LOCAL = 'current_structural_local_noise'
STOP = 'initialization_only'


def case_dir(output, seed, arm):
    return output / f'seed_{seed}' / arm


def load_truth_cache(plan, output):
    """Merge compatible nominal-geometry caches by physical policy coordinates."""
    cache = {}
    old = ROOT / 'paper_artifacts/mainline_candidate_pool_audit_v1'
    for point, row in zip(np.load(old / 'truth_policies.npz')['points'], read(old / 'truth.json')):
        cache[tuple(map(int, point))] = dict(row['truth'])
    parent = ROOT / 'paper_artifacts/mainline_local_noise_control_v1'
    paths = [parent / name for name in ('truth.json', 'diagnostic_truth.json', 'headroom_truth.json')]
    paths += [output / 'truth.json', output / 'truth_new.json']
    for path in paths:
        if path.exists():
            for row in read(path):
                cache.setdefault(tuple(row['point']), {}).update(row['truth'])
    return cache


def truth_value(problem, point, cache, extra, cost):
    key = tuple(map(int, point))
    value = dict(cache.get(key, {}))
    missing = []
    for field, function in (
        ('objective', lambda: float(problem.true_objective(key))),
        ('constraint_mean_centered', lambda: float(problem.true_constraint_mean(key)) - problem.tau),
        ('constraint_sigma', lambda: float(problem.true_sigma(key)[1])),
    ):
        if field not in value:
            value[field] = function()
            cost['new_truth_function_calls'] += 1
            missing.append(field)
    margin = value['constraint_mean_centered'] + norm.ppf(1 - problem.alpha) * value['constraint_sigma']
    reference = float(np.dot(problem.weights, problem.ref_point))
    value.update(chance_margin=float(margin), true_feasible=bool(margin <= 0),
                 quality=float(np.clip(1 - value['objective'] / reference, 0, 1)))
    cache[key] = value
    if missing:
        extra[key] = value
    return value


def coverage(points, initial, get_truth):
    values = [get_truth(point) for point in points]
    feasible = [i for i, value in enumerate(values) if value['true_feasible']]
    improving = [i for i in feasible if values[i]['objective'] < initial['objective'] - 1e-12]
    best = min(feasible, key=lambda i: values[i]['objective']) if feasible else None
    return {'count': len(points), 'feasible_count': len(feasible),
        'feasible_improving_count': len(improving), 'feasible_improving_indices': improving,
        'best_feasible_index': best,
        'best_feasible_objective': None if best is None else values[best]['objective'],
        'maximum_quality_gain': 0. if best is None else max(0., values[best]['quality'] - initial['quality'])}


def diagnose_path(directory, search, neighbors, initial_points, initial, get_truth):
    audit = read(directory / 'query_audit.json')
    points = [tuple(map(int, point)) for point in np.load(directory / 'query_policies.npz')['points']]
    shared = {points[i] for i in audit['shared_ids']}
    observed_all = {tuple(row['point']) for row in search['history']}
    unseen = [point for point in neighbors if point not in observed_all and point not in initial_points]
    fixed = [point for point in unseen if point in shared]
    snapshots, stages = {}, []
    for stage in audit['stages']:
        action = [points[i] for i in stage['action_ids']]
        positions = {point: i for i, point in enumerate(action)}
        scores = stage['score']
        total = np.asarray(scores['total'])
        ranks = np.empty(len(action), dtype=int)
        ranks[np.argsort(-total, kind='stable')] = np.arange(1, len(action) + 1)
        selected = points[stage['selected_id']]
        row = next(row for row in search['iterations'] if row['iteration'] == stage['iteration'])
        assert selected == tuple(search['history'][stage['paid_prefix']]['point']) == tuple(row['x_selected'])
        np.testing.assert_allclose(total[positions[selected]], row['score_selected'], rtol=1e-12, atol=1e-12)

        def detail(point):
            truth = get_truth(point)
            value = {'neighbor_index': neighbors.index(point) if point in neighbors else None,
                'in_action_pool': point in positions, 'selected': point == selected,
                'truth': truth, 'feasible_improving': bool(truth['true_feasible']
                    and truth['objective'] < initial['objective'] - 1e-12)}
            if point not in positions:
                return value
            index = positions[point]
            value.update(route=stage['action_sources'][index],
                score_rank=int(ranks[index]),
                strictly_better_score_count=int(np.sum(total > total[index])),
                tied_score_count=int(np.sum(total == total[index])),
                prediction={key: value[index] for key, value in scores.items()
                    if isinstance(value, list) and len(value) == len(action)
                    and all(isinstance(v, (float, int)) for v in value)})
            return value

        improvements = [point for point in neighbors if get_truth(point)['true_feasible']
            and get_truth(point)['objective'] < initial['objective'] - 1e-12]
        stages.append({'iteration': stage['iteration'], 'paid_prefix': stage['paid_prefix'],
            'selected_policy_score_parity': True, 'candidate_count': len(action),
            'shared_candidate_count': sum(point in positions for point in neighbors),
            'selected': detail(selected), 'improving_neighbors': [detail(point) for point in improvements],
            'sampled_feasible_count': scores['sampled_feasible_count'],
            'sampled_feasibility_mode': scores['sampled_feasibility_mode']})
        if stage['paid_prefix'] in (10, 12) and fixed:
            assert all(point in positions for point in fixed)
            idx = [positions[point] for point in fixed]
            predicted = np.column_stack([np.asarray(scores[key])[idx] for key in
                ('objective_mean', 'constraint_mean', 'constraint_aleatoric')])
            values = [get_truth(point) for point in fixed]
            actual = np.asarray([[v['objective'], v['constraint_mean_centered'], v['constraint_sigma']**2]
                                 for v in values])
            snapshots[str(stage['paid_prefix'])] = {'mean_bias_f_g_noise_variance': np.mean(predicted-actual, axis=0),
                'mse_f_g_noise_variance': np.mean((predicted-actual)**2, axis=0),
                'predictions_f_g_noise_variance': predicted}
    propagation = {'fixed_neighbor_indices': [neighbors.index(point) for point in fixed],
        'count': len(fixed), 'new_never_sampled_neighbor_count': len(unseen),
        'snapshots': snapshots, 'scope': 'New shared policies never sampled in this arm full 13-row history.'}
    if fixed:
        assert set(snapshots) == {'10', '12'}
        propagation['mse_reduction_f_g_noise_variance'] = (
            snapshots['10']['mse_f_g_noise_variance'] - snapshots['12']['mse_f_g_noise_variance'])
    return {'query_stages': stages, 'never_sampled_prediction_change': propagation}


def diagnose(plan):
    output = ROOT / plan['output_dir']
    names = [arm['name'] if isinstance(arm, dict) else arm for arm in plan['arms']]
    for seed in plan['target_seeds']:
        for name in names:
            if not (case_dir(output, seed, name) / 'verification.json').exists():
                raise RuntimeError('All 18 independent verifications must finish before diagnosis')
    results = read(output / 'results.json')
    destination = output / 'diagnosis.json'
    if destination.exists() and read(destination).get('status') == 'complete':
        return read(destination)
    with attempt(output, 'diagnose-shared-neighbors', plan['domain']) as cost:
        cost.update(new_truth_function_calls=0, new_source_fits=0, download_bytes=0)
        cache = load_truth_cache(plan, output)
        extra_path = output / 'truth_new.json'
        extra = {tuple(row['point']): row['truth'] for row in read(extra_path)} if extra_path.exists() else {}
        rows, seed_rows = [], []
        try:
            for seed in plan['target_seeds']:
                directory = output / f'seed_{seed}'
                neighbors = [tuple(map(int, point)) for point in np.load(directory / 'neighbors.npz')['points']]
                stop = read(case_dir(output, seed, STOP) / 'search.json')
                initial_points = {tuple(row['point']) for row in stop['history']}
                problem = load_pickle(case_dir(output, seed, STOP) / 'search_engine.pkl').problem
                get_truth = lambda point: truth_value(problem, point, cache, extra, cost)
                initial = get_truth(stop['history'][0]['point'])
                baseline = next(row['actual_safe_reward'] for row in results['rows']
                                if row['seed'] == seed and row['arm'] == STOP)
                new = [point for point in neighbors if point not in initial_points]
                seed_rows.append({'seed': seed, 'initial_truth': initial,
                    'all_shared': coverage(neighbors, initial, get_truth),
                    'new_shared': coverage(new, initial, get_truth),
                    'new_shared_neighbor_indices': [neighbors.index(point) for point in new],
                    'neighbor_truth': [get_truth(point) for point in neighbors]})
                for name in names:
                    verified = next(row for row in results['rows'] if row['seed'] == seed and row['arm'] == name)
                    value = {key: verified[key] for key in ('seed', 'arm', 'primary_truth', 'deployed_truth',
                        'certified', 'false_certificate', 'actual_safe_reward')}
                    value.update(deployed_quality_gain=verified['deployed_truth']['quality']-initial['quality'],
                        actual_safe_reward_gain=verified['actual_safe_reward']-baseline)
                    if name == LOCAL:
                        value['query_diagnostics_alias'] = FULL
                    elif name != STOP:
                        path = case_dir(output, seed, name)
                        value.update(diagnose_path(path, read(path / 'search.json'), neighbors,
                                                   initial_points, initial, get_truth))
                    rows.append(value)
            result = {'status': 'complete', 'seeds': seed_rows, 'rows': rows,
                'query_score_parity_count': sum(len(row.get('query_stages', [])) for row in rows),
                'interpretation': 'Recorded paid10 to paid12 predictions on each arm fixed never-sampled new neighbors; descriptive information propagation, not a causal attribution. Local-noise shares the full structural query history.',
                'noise_target': 'Constraint episode variance sigma_g squared; MSE is on variance, not standard deviation.',
                'analytical_truth_new_unique_policies': len(extra)}
        finally:
            save(extra_path, [{'point': list(point), 'truth': value} for point, value in extra.items()])
    result['cost'] = cost
    save(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    args = parser.parse_args()
    result = diagnose(read(args.manifest))
    print({'status': result['status'], 'query_score_parity_count': result['query_score_parity_count']}, flush=True)


if __name__ == '__main__':
    main()
