"""Score the registered fresh-geometry terminal 2x2 after every verifier ends."""
import argparse
from unittest.mock import patch

import numpy as np

from performance.diagnose_mainline_shared_neighbors import (
    coverage, diagnose_path, truth_value,
)
from performance.run_mainline_legacy_backend_bridge import (
    ROOT, attempt, load_pickle, read, save,
)
from performance.run_mainline_terminal_equal import case_dir, cases, require_all
from performance.summarize_mainline_role_fresh import (
    STOP, prediction_arms, shared_prediction_cohort,
)
from performance.summarize_mainline_terminal_cross import summarize as summarize_cross


MANIFEST = ROOT / 'performance/manifests/mainline_terminal_confirmation_cells_v1.json'


def prediction_by_arm(cohort, names):
    """Keep tasks as the primary unit and policy-weighted errors descriptive."""
    active = [row for row in cohort if row['count']]
    result = []
    for arm in names:
        values = [next(item for item in row['arms'] if item['arm'] == arm)
                  for row in active]
        snapshots = {}
        for prefix in ('10', '12'):
            snapshots[prefix] = {}
            for metric in ('mean_bias_f_g_noise_variance', 'mse_f_g_noise_variance'):
                data = np.asarray([item['snapshots'][prefix][metric] for item in values])
                snapshots[prefix][metric] = dict(
                    task_equal=None if not active else np.mean(data, axis=0),
                    policy_weighted=None if not active else np.average(
                        data, axis=0, weights=[row['count'] for row in active]))
        changes = np.asarray([item['mse_reduction_f_g_noise_variance'] for item in values])
        result.append(dict(arm=arm, task_count=len(active),
            expected_task_count=len(cohort), excluded_task_count=len(cohort)-len(active),
            policy_count=sum(row['count'] for row in active), snapshots=snapshots,
            mse_improved_tasks_f_g_noise_variance=(None if not active else
                np.sum(changes > 1e-12, axis=0)),
            mse_improvement_absolute_tolerance=1e-12,
            task_mse_reduction=[dict(seed=row['seed'], count=row['count'],
                f_g_noise_variance=value['mse_reduction_f_g_noise_variance'])
                for row, value in zip(active, values, strict=True)]))
    return result


def collect_truth(output, plan):
    # The same barrier covers all four treatment cells and all five references.
    require_all(output, plan, 'verification.json')
    source = ROOT / plan['source_output_dir']
    source_plan = read(ROOT / plan['source_manifest'])
    if source_plan['target_seeds'] != plan['target_seeds']:
        raise ValueError('Search and terminal summaries must cover the same registered tasks')
    destination = source / 'diagnosis.json'
    if destination.is_file():
        return read(destination)
    names = [arm['name'] if isinstance(arm, dict) else arm for arm in source_plan['arms']]
    failures = source_plan.get('execution_failures', [])
    failed_cases = {(item['seed'], item['arm']) for item in failures}
    rows, seed_rows, counts = [], [], []
    with attempt(output, 'score-fresh-terminal-truth', source_plan['domain']) as cost:
        cost.update(new_truth_function_calls=0, new_target_initial_calls=0,
                    new_source_fits=0, download_bytes=0)
        for seed in plan['target_seeds']:
            seed_dir = source / f'seed_{seed}'
            stop_dir = seed_dir / STOP
            stop = read(stop_dir / 'search.json')
            problem = load_pickle(stop_dir / 'search_engine.pkl').problem
            if float(problem.tau) != 0.:
                raise ValueError('Saved query-score diagnosis requires this protocol tau=0')
            initial_points = {tuple(row['point']) for row in stop['history']}
            neighbors = [tuple(map(int, point)) for point in
                         np.load(seed_dir / 'neighbors.npz')['points']]
            truth_path = seed_dir / 'truth.json'
            cache = ({tuple(row['point']): row['truth'] for row in read(truth_path)}
                     if truth_path.is_file() else {})
            extra = {}
            before = cost['new_truth_function_calls']
            get_truth = lambda point: truth_value(problem, point, cache, extra, cost)
            try:
                with patch.object(problem, 'simulate', side_effect=RuntimeError(
                        'Post-verification scoring cannot generate simulation observations')):
                    initial = get_truth(stop['history'][0]['point'])
                    new = [point for point in neighbors if point not in initial_points]
                    seed_rows.append(dict(seed=seed, initial_truth=initial,
                        all_shared=coverage(neighbors, initial, get_truth),
                        new_shared=coverage(new, initial, get_truth),
                        new_shared_neighbor_indices=[neighbors.index(point) for point in new],
                        neighbor_truth=[get_truth(point) for point in neighbors]))
                    for registered_seed, arm, protocol in cases(plan):
                        if registered_seed != seed:
                            continue
                        directory = case_dir(output, seed, arm, protocol)
                        item = read(directory / 'input.json')
                        verified = read(directory / 'verification.json')
                        for point in [item['initial'], item['primary'], verified['deployed']]:
                            get_truth(point)
                        for candidate in item['frozen_shortlist']:
                            get_truth(candidate['point'])
                    for arm in names:
                        if (seed, arm) in failed_cases:
                            continue
                        directory = seed_dir / arm
                        search = read(directory / 'search.json')
                        for observation in search['history']:
                            get_truth(observation['point'])
                        row = dict(seed=seed, arm=arm)
                        if arm != STOP:
                            row.update(diagnose_path(directory, search, neighbors,
                                initial_points, initial, get_truth))
                            stages = row['query_stages']
                            row['query_summary'] = dict(count=len(stages),
                                true_feasible=sum(stage['selected']['truth']['true_feasible']
                                                  for stage in stages),
                                feasible_improving=sum(stage['selected']['feasible_improving']
                                                      for stage in stages),
                                new_neighbor_selected=sum(
                                    stage['selected']['neighbor_index'] is not None and
                                    neighbors[stage['selected']['neighbor_index']] not in initial_points
                                    for stage in stages),
                                new_neighbor_improving=sum(
                                    stage['selected']['feasible_improving'] and
                                    stage['selected']['neighbor_index'] is not None and
                                    neighbors[stage['selected']['neighbor_index']] not in initial_points
                                    for stage in stages))
                        rows.append(row)
            finally:
                save(truth_path, [dict(point=list(point), truth=value)
                                  for point, value in cache.items()])
            counts.append(dict(seed=seed, unique_policies=len(cache),
                new_unique_policies=len(extra),
                new_truth_function_calls=cost['new_truth_function_calls'] - before))
        prediction_names = prediction_arms(source_plan)
        failed_prediction_seeds = {seed for seed, arm in failed_cases if arm in prediction_names}
        valid_cohort = shared_prediction_cohort(rows,
            [row for row in seed_rows if row['seed'] not in failed_prediction_seeds], prediction_names)
        by_seed = {row['seed']: row for row in valid_cohort}
        cohort = [dict(seed=seed, count=0, neighbor_indices=[], arms=[],
                       unavailable_reason='Registered five-arm intersection unavailable after search execution failure.')
                  if seed in failed_prediction_seeds else by_seed[seed] for seed in plan['target_seeds']]
        diagnosis = dict(status='complete', seeds=seed_rows, rows=rows,
            execution_failures=failures,
            shared_never_sampled_cohort=cohort,
            prediction_by_arm=prediction_by_arm(cohort, prediction_arms(source_plan)),
            truth_by_seed=counts,
            query_score_parity_count=sum(len(row.get('query_stages', [])) for row in rows),
            truth_scope=('All registered terminal shortlists and deployments, paid histories, '
                'and shared-neighbor policies, scored only after every new verifier finished. '
                'Each fresh task owns its own analytical truth cache.'),
            prediction_scope=('Saved paid10 and paid12 predictions on new shared policies '
                'never sampled by any of the five replacement backends. Native is outside '
                'this intersection. This is a descriptive prediction diagnostic, with task '
                'geometries as statistical units; terminal paid13 predictions are not reconstructed. '
                'A failed prediction arm makes that task whole registered intersection unavailable; '
                'the remaining arms do not redefine it.'),
            noise_target='Constraint episode variance sigma_g squared.')
    diagnosis['cost'] = cost
    save(destination, diagnosis)
    save(output / 'truth_collection.json', dict(status='complete', truth_by_seed=counts,
        query_score_parity_count=diagnosis['query_score_parity_count'], cost=cost))
    return diagnosis


def summarize(output, plan):
    diagnosis = collect_truth(output, plan)
    result = summarize_cross(output, plan)
    result['scope'] = ('Frozen 2x2 search-propagation by terminal-backend comparison on '
        'registered fresh task geometries, with five unchanged references. Initialization '
        'is shared within task; each search history is reused by its two terminal cells. '
        'All terminal decisions precede independent verification and target-truth scoring. '
        'Sources, model, representation, candidate frontend and budgets are unchanged.')
    result['statistical_unit'] = (f"{len(plan['target_seeds'])} fresh task geometries; "
        'one paired search-observation stream and one paired independent terminal stream per task.')
    result['prediction_diagnosis'] = str(ROOT / plan['source_output_dir'] / 'diagnosis.json')
    result['common_unobserved_prediction'] = diagnosis['prediction_by_arm']
    result['query_score_parity_count'] = diagnosis['query_score_parity_count']
    save(output / 'results.json', result)
    completion = read(output / 'completion.json')
    source_completion = ROOT / plan['source_output_dir'] / 'completion.json'
    completion['cost_scope'] = ('This terminal-stage ledger covers terminal preparation, '
        'verification and analytical truth scoring. Fresh initialization and search costs '
        'belong to the separate search-stage ledger; the parent completion combines both.')
    completion['new_physical_target_calls_in_terminal_stage'] = completion.pop(
        'new_physical_target_calls')
    completion['fresh_search_stage'] = dict(completion_file=str(source_completion),
        cost=read(source_completion)['cost'],
        unique_histories=completion.pop('reused_source_history_count'),
        unique_logical_search_calls=completion.pop('reused_logical_search_calls_unique_histories'),
        expanded_logical_search_calls=completion.pop('reused_logical_search_calls_expanded_cells'),
        physical_initial_and_search_calls=completion.pop('source_stage_physical_initial_and_search_calls'))
    save(output / 'completion.json', completion)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    args = parser.parse_args()
    plan = read(args.manifest)
    result = summarize(ROOT / plan['output_dir'], plan)
    print(dict(status=result['status'], cases=len(result['rows']),
               query_score_parity_count=result['query_score_parity_count']), flush=True)


if __name__ == '__main__':
    main()
