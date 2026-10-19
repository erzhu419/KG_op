"""Summarize frozen fresh-geometry decisions and recorded query predictions.

Each task owns its analytical truth cache. All verifiers must finish before
truth is opened; no posterior is restored and no simulator is called here.
"""
import argparse
import json

import numpy as np

from performance.diagnose_mainline_shared_neighbors import (
    coverage, diagnose_path, truth_value,
)
from performance.run_mainline_legacy_backend_bridge import (
    ROOT, attempt, load_pickle, read, save,
)

MANIFEST = ROOT / 'performance/manifests/mainline_role_fresh_v1.json'
STOP = 'initialization_only'
FULL = 'legacy_role_full'
POINT = 'legacy_role_point_only'
AFTER_INIT_POINT = 'legacy_role_after_init_point_only'
NATIVE = 'legacy_native_neighbors'
PREDICTION_ARMS = (FULL, POINT, 'same_source_gp', 'current_structural_mixture')
SCOPE = ('The point-only intervention starts at the first initialization raw '
         'and disables cross-policy updates in the replacement decision '
         'posterior. The original proposal frontend still learns from each '
         'arm\'s own paid raw; this does not disable frontend propagation.')
TERMINAL_METRICS_SCOPE = (
    'returned_policy_changed compares the returned point with the first initial '
    'point, even when certification failed. certified_deployment_changed also '
    'requires certification. Quality components require certification and true '
    'feasibility of the returned point. Initial quality is a numerical reference, '
    'not a certified or truly feasible baseline. Their sum equals actual_safe_reward; '
    'the quality-gain component is signed and can be negative. The historical '
    'deployed_quality_gain field is an ungated returned-point difference.')


def case_dir(output, seed, arm):
    return output / f'seed_{seed}' / arm


def prediction_arms(plan):
    """Include the initialization-matched control in the unsampled intersection."""
    names = [a['name'] if isinstance(a, dict) else a for a in plan['arms']]
    defaults = list(PREDICTION_ARMS)
    if AFTER_INIT_POINT in names:
        defaults.append(AFTER_INIT_POINT)
    return plan.get('prediction_arms', defaults)


def intervention_scope(plan):
    names = [a['name'] if isinstance(a, dict) else a for a in plan['arms']]
    if AFTER_INIT_POINT not in names:
        return SCOPE
    return (SCOPE + ' The after-init point-only control first conditions jointly '
            'on all ten initialization observations, identically to full. '
            'Subsequent observations update only their own physical policy in '
            'the decision posterior; unobserved marginals remain at this shared '
            'initialization posterior. Its original proposal frontend continues '
            'learning from its own paid raw.')


def terminal_metrics(initial_point, returned_point, initial, returned, certified):
    changed = tuple(returned_point) != tuple(initial_point)
    certified_feasible = bool(certified and returned['true_feasible'])
    return dict(
        initial_truth=initial,
        returned_policy_changed=changed,
        certified_deployment_changed=bool(certified and changed),
        certified_feasible_improving=bool(certified_feasible and
            returned['objective'] < initial['objective'] - 1e-12),
        certified_feasible_quality_gain=(returned['quality'] - initial['quality']
                                        if certified_feasible else 0.),
        certified_initial_quality_component=(initial['quality']
                                            if certified_feasible else 0.))


def paired_differences(rows, seeds, references=None):
    if references is None:
        references = (POINT, 'same_source_gp', STOP, NATIVE,
                      'current_structural_mixture')
    result = []
    for seed in seeds:
        group = {row['arm']: row for row in rows if row['seed'] == seed}
        full = group[FULL]
        for reference in references:
            other = group[reference]
            result.append(dict(seed=seed, arm=FULL, reference=reference,
                safe_reward_difference=full['actual_safe_reward']-other['actual_safe_reward'],
                verification_calls_difference=(full['new_verification_calls']
                                               - other['new_verification_calls']),
                primary_changed=full['primary'] != other['primary'],
                deployment_changed=full['deployed'] != other['deployed'],
                **{key + '_difference': full[key] - other[key] for key in (
                    'returned_policy_changed', 'certified_deployment_changed',
                    'certified_feasible_improving', 'certified_feasible_quality_gain',
                    'certified_initial_quality_component')}))
    return result


def shared_prediction_cohort(rows, seed_rows, arms=PREDICTION_ARMS):
    """Compare saved scores on policies never sampled by any new backend."""
    by_seed = []
    for seed_row in seed_rows:
        seed = seed_row['seed']
        group = {row['arm']: row for row in rows if row['seed'] == seed}
        common = set(seed_row['new_shared_neighbor_indices'])
        for arm in arms:
            common &= set(group[arm]['never_sampled_prediction_change']['fixed_neighbor_indices'])
        indices = sorted(common)
        actual = np.asarray([[seed_row['neighbor_truth'][i][name] for name in
            ('objective', 'constraint_mean_centered', 'constraint_sigma')] for i in indices])
        if indices:
            actual[:, 2] **= 2
        summaries = []
        for arm in arms:
            trace = group[arm]['never_sampled_prediction_change']
            position = {index: i for i, index in enumerate(trace['fixed_neighbor_indices'])}
            selected = [position[i] for i in indices]
            snapshots = {}
            for prefix in ('10', '12'):
                if not indices:
                    snapshots[prefix] = None
                    continue
                predicted = np.asarray(trace['snapshots'][prefix]['predictions_f_g_noise_variance'])[selected]
                snapshots[prefix] = dict(
                    mean_bias_f_g_noise_variance=np.mean(predicted-actual, axis=0),
                    mse_f_g_noise_variance=np.mean((predicted-actual)**2, axis=0),
                    sigma_ratio_median=float(np.median(np.sqrt(predicted[:, 2]/actual[:, 2]))),
                    sigma_ratio_maximum=float(np.max(np.sqrt(predicted[:, 2]/actual[:, 2]))))
            summaries.append(dict(arm=arm, snapshots=snapshots,
                mse_reduction_f_g_noise_variance=(None if not indices else
                    snapshots['10']['mse_f_g_noise_variance']-
                    snapshots['12']['mse_f_g_noise_variance'])))
        by_seed.append(dict(seed=seed, count=len(indices), neighbor_indices=indices,
                            arms=summaries))
    return by_seed


def summarize(plan):
    output = ROOT / plan['output_dir']
    names = [arm['name'] if isinstance(arm, dict) else arm for arm in plan['arms']]
    seeds = plan['target_seeds']
    references = [POINT, 'same_source_gp', STOP, NATIVE, 'current_structural_mixture']
    references += [arm for arm in names if arm != FULL and arm not in references]
    primary = plan.get('primary_reference', POINT)
    references = [primary] + [arm for arm in references if arm != primary]
    scope = intervention_scope(plan)
    for seed in seeds:
        for arm in names:
            if not (case_dir(output, seed, arm) / 'verification.json').is_file():
                raise RuntimeError('All independent verifications must finish before truth scoring')
    if (output / 'results.json').is_file() and (output / 'diagnosis.json').is_file():
        return read(output / 'results.json'), read(output / 'diagnosis.json')

    rows, diagnostics, seed_rows, truth_counts = [], [], [], []
    with attempt(output, 'summarize-fresh-role', plan['domain']) as cost:
        cost.update(new_truth_function_calls=0, new_target_initial_calls=0,
                    new_source_fits=0, download_bytes=0)
        for seed in seeds:
            seed_dir = output / f'seed_{seed}'
            stop = read(case_dir(output, seed, STOP) / 'search.json')
            problem = load_pickle(case_dir(output, seed, STOP) / 'search_engine.pkl').problem
            if float(problem.tau) != 0.:
                raise ValueError('Reused recorded-score diagnosis expects this protocol tau=0')
            initial_points = {tuple(row['point']) for row in stop['history']}
            neighbors = [tuple(map(int, point)) for point in np.load(seed_dir / 'neighbors.npz')['points']]
            truth_path = seed_dir / 'truth.json'
            cache = {tuple(row['point']): row['truth'] for row in read(truth_path)} if truth_path.exists() else {}
            extra = {}
            before = cost['new_truth_function_calls']
            get_truth = lambda point: truth_value(problem, point, cache, extra, cost)
            try:
                initial = get_truth(stop['history'][0]['point'])
                new = [point for point in neighbors if point not in initial_points]
                seed_rows.append(dict(seed=seed, initial_truth=initial,
                    all_shared=coverage(neighbors, initial, get_truth),
                    new_shared=coverage(new, initial, get_truth),
                    new_shared_neighbor_indices=[neighbors.index(point) for point in new],
                    neighbor_truth=[get_truth(point) for point in neighbors]))
                for arm in names:
                    directory = case_dir(output, seed, arm)
                    search, verified = read(directory / 'search.json'), read(directory / 'verification.json')
                    primary, deployed = get_truth(verified['primary']), get_truth(verified['deployed'])
                    certified = bool(verified['verification']['certified'])
                    row = dict(seed=seed, arm=arm, primary=verified['primary'], deployed=verified['deployed'],
                        primary_truth=primary, deployed_truth=deployed, certified=certified,
                        false_certificate=bool(certified and not deployed['true_feasible']),
                        actual_safe_reward=deployed['quality'] if certified and deployed['true_feasible'] else 0.,
                        new_initial_calls=search['new_initial_calls'],
                        new_search_calls=search['new_search_calls'],
                        new_verification_calls=verified['new_verification_calls'],
                        logical_total_calls=search['logical_search_calls']+verified['new_verification_calls'],
                        selected_shortlist_rank=verified['verification']['selected_shortlist_rank'])
                    row.update(terminal_metrics(stop['history'][0]['point'],
                        verified['deployed'], initial, deployed, certified))
                    rows.append(row)
                    diagnosis = {key: row[key] for key in ('seed', 'arm', 'certified',
                        'false_certificate', 'actual_safe_reward', 'primary_truth', 'deployed_truth',
                        'initial_truth', 'returned_policy_changed', 'certified_deployment_changed',
                        'certified_feasible_improving', 'certified_feasible_quality_gain',
                        'certified_initial_quality_component')}
                    diagnosis['deployed_quality_gain'] = deployed['quality']-initial['quality']
                    if arm != STOP:
                        diagnosis.update(diagnose_path(directory, search, neighbors,
                                                       initial_points, initial, get_truth))
                        stages = diagnosis['query_stages']
                        diagnosis['query_summary'] = dict(
                            count=len(stages),
                            true_feasible=sum(s['selected']['truth']['true_feasible'] for s in stages),
                            feasible_improving=sum(s['selected']['feasible_improving'] for s in stages),
                            new_neighbor_selected=sum(s['selected']['neighbor_index'] is not None
                                and neighbors[s['selected']['neighbor_index']] not in initial_points for s in stages),
                            new_neighbor_improving=sum(s['selected']['feasible_improving']
                                and s['selected']['neighbor_index'] is not None
                                and neighbors[s['selected']['neighbor_index']] not in initial_points for s in stages))
                    diagnostics.append(diagnosis)
            finally:
                save(truth_path, [dict(point=list(point), truth=value) for point, value in cache.items()])
            truth_counts.append(dict(seed=seed, unique_policies=len(cache),
                new_unique_policies=len(extra), new_truth_function_calls=cost['new_truth_function_calls']-before))

        arm_rows = []
        for arm in names:
            group = [row for row in rows if row['arm'] == arm]
            query = [row['query_summary'] for row in diagnostics
                     if row['arm'] == arm and 'query_summary' in row]
            arm_rows.append(dict(arm=arm, task_count=len(group),
                certified=sum(row['certified'] for row in group),
                false_certificates=sum(row['false_certificate'] for row in group),
                primary_true_feasible=sum(row['primary_truth']['true_feasible'] for row in group),
                mean_safe_reward=float(np.mean([row['actual_safe_reward'] for row in group])),
                **{key: sum(row[key] for row in group) for key in (
                    'returned_policy_changed', 'certified_deployment_changed',
                    'certified_feasible_improving')},
                **{'mean_' + key: float(np.mean([row[key] for row in group])) for key in (
                    'certified_feasible_quality_gain', 'certified_initial_quality_component')},
                verification_calls=sum(row['new_verification_calls'] for row in group),
                query_summary={key: sum(row[key] for row in query) for key in
                    ('count', 'true_feasible', 'feasible_improving', 'new_neighbor_selected', 'new_neighbor_improving')}))
        for row in diagnostics:
            baseline = next(other['actual_safe_reward'] for other in rows
                            if other['seed'] == row['seed'] and other['arm'] == STOP)
            row['actual_safe_reward_gain'] = row['actual_safe_reward']-baseline
        results = dict(status='complete', protocol=plan, rows=rows, arms=arm_rows,
            paired=paired_differences(rows, seeds, references), scope=scope,
            terminal_metrics_scope=TERMINAL_METRICS_SCOPE,
            pairing_unit=f'{len(seeds)} new task geometries with one paired observation stream each; descriptive pilot.')
        diagnosis = dict(status='complete', seeds=seed_rows, rows=diagnostics,
            shared_never_sampled_cohort=shared_prediction_cohort(
                diagnostics, seed_rows, prediction_arms(plan)),
            truth_by_seed=truth_counts, scope=scope,
            terminal_metrics_scope=TERMINAL_METRICS_SCOPE,
            query_score_parity_count=sum(len(row.get('query_stages', [])) for row in diagnostics),
            prediction_scope='Saved pre-query paid10 and paid12 scores; paid13 terminal predictions are not reconstructed.',
            noise_target='Constraint episode variance sigma_g squared.',
            interpretation=('Per-arm and common unsampled prediction changes are descriptive. '
                'The actual full-versus-point-only closed-loop comparison tests decision-posterior '
                'propagation with the same initial source marginals.' if primary == POINT else
                'Per-arm and common unsampled prediction changes are descriptive. The full-versus-'
                'after-init-point-only closed-loop comparison tests propagation after the shared '
                'initialization posterior; the source0 point-only comparison remains secondary.'))
    attempts = [json.loads(line) for path in output.rglob('attempts.jsonl')
                for line in path.read_text().splitlines() if line.strip()]
    results['recorded_physical_calls'] = {key: sum(row.get(key, 0) for row in attempts)
        for key in ('new_target_initial_calls', 'new_target_search_calls', 'new_verification_calls')}
    results['logical_target_calls'] = sum(row['logical_total_calls'] for row in rows)
    diagnosis['cost'] = cost
    save(output / 'results.json', results)
    save(output / 'diagnosis.json', diagnosis)
    return results, diagnosis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    args = parser.parse_args()
    results, diagnosis = summarize(read(args.manifest))
    print(dict(status=results['status'], cases=len(results['rows']),
               query_score_parity_count=diagnosis['query_score_parity_count']), flush=True)


if __name__ == '__main__':
    main()
