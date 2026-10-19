"""Summarize a fixed objective-pair replay using saved predictions and truth only."""
import argparse

import numpy as np
from scipy.special import ndtri

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = ROOT / 'performance/manifests/mainline_objective_pair_v1.json'
VARIANTS = ('original_role', 'head_specific_objective_pair')
FOCUS = {2026101672: (24, 36), 2026101675: (33,)}
METRICS = ('mse_f_g_noise_variance', 'bias_f_g_noise_variance',
           'objective_difference_mse', 'objective_difference_bias',
           'objective_direction_accuracy', 'risk_margin_mse', 'risk_margin_bias',
           'risk_feasibility_accuracy', 'feasible_objective_difference_mse',
           'feasible_objective_direction_accuracy')


def prediction_metrics(predicted, actual, baseline_prediction, baseline_truth, z):
    """All objective contrasts share the same initial physical policy."""
    if not len(actual):
        return dict(count=0, feasible_count=0, **{key: None for key in METRICS})
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    error = predicted - actual
    contrast = predicted[:, 0] - baseline_prediction
    true_contrast = actual[:, 0] - baseline_truth
    contrast_error = contrast - true_contrast
    predicted_margin = predicted[:, 1] + z * np.sqrt(predicted[:, 2])
    true_margin = actual[:, 1] + z * np.sqrt(actual[:, 2])
    feasible = true_margin <= 0
    direction = np.where(abs(contrast) <= 1e-12, 0, np.sign(contrast))
    true_direction = np.where(abs(true_contrast) <= 1e-12, 0, np.sign(true_contrast))
    correct = direction == true_direction
    return dict(count=len(actual), feasible_count=int(feasible.sum()),
        mse_f_g_noise_variance=np.mean(error ** 2, axis=0),
        bias_f_g_noise_variance=np.mean(error, axis=0),
        objective_difference_mse=float(np.mean(contrast_error ** 2)),
        objective_difference_bias=float(np.mean(contrast_error)),
        objective_direction_accuracy=float(np.mean(correct)),
        risk_margin_mse=float(np.mean((predicted_margin - true_margin) ** 2)),
        risk_margin_bias=float(np.mean(predicted_margin - true_margin)),
        risk_feasibility_accuracy=float(np.mean((predicted_margin <= 0) == feasible)),
        feasible_objective_difference_mse=(float(np.mean(contrast_error[feasible] ** 2))
                                           if feasible.any() else None),
        feasible_objective_direction_accuracy=(float(np.mean(correct[feasible]))
                                               if feasible.any() else None))


def difference(after, before):
    return {key: (None if after[key] is None or before[key] is None else
                  np.asarray(after[key]) - np.asarray(before[key])) for key in METRICS}


def average_metrics(rows, weighted=False):
    result = dict(task_count=len(rows), count=sum(row['count'] for row in rows),
                  feasible_count=sum(row['feasible_count'] for row in rows))
    for key in METRICS:
        present = [row for row in rows if row[key] is not None]
        weights = [row['feasible_count' if key.startswith('feasible_') else 'count']
                   if weighted else 1. for row in present]
        result[key] = (np.average([row[key] for row in present], weights=weights, axis=0)
                       if present else None)
        if key.startswith('feasible_'):
            result['feasible_metric_task_count'] = len(present)
    return result


def summarize(plan):
    output, parent = ROOT / plan['output_dir'], ROOT / plan['input_dir']
    seeds = plan['target_seeds']
    # The runner saves all projections before outcomes are used by this analysis.
    projections = {seed: read(output / f'seed_{seed}' / 'projection.json') for seed in seeds}
    if (output / 'summary.json').is_file():
        return read(output / 'summary.json')
    with attempt(output, 'summarize-objective-pair', 'FactorShockStatePolicyRZDT1') as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0,
                    new_truth_function_calls=0, download_bytes=0)
        task_rows, focus_rows, query_rows, changes = [], [], [], []
        z = float(ndtri(.95))
        for seed in seeds:
            directory = parent / f'seed_{seed}'
            projection = projections[seed]
            bank = np.load(output / f'seed_{seed}' / 'bank.npz')['points']
            neighbors = np.load(directory / 'neighbors.npz')['points']
            history = read(directory / 'legacy_role_full' / 'search.json')['history']
            seen = {tuple(row['point']) for row in history}
            initial = {tuple(row['point']) for row in history[:10]}
            indices = [i for i, point in enumerate(neighbors) if tuple(point) not in seen]
            truth = {tuple(row['point']): row['truth'] for row in read(directory / 'truth.json')}
            known = [i for i in indices if tuple(neighbors[i]) in truth]
            actual = np.asarray([[truth[tuple(neighbors[i])][key] for key in
                ('objective', 'constraint_mean_centered', 'constraint_sigma')] for i in known])
            if len(actual):
                actual[:, 2] **= 2
            baseline_index = projection['baseline_index']
            baseline = truth[tuple(bank[baseline_index])]
            positions = np.asarray(projection['neighbor_bank_indices'])
            snapshots = {}
            for variant in VARIANTS:
                state = projection['variants'][variant]
                snapshots[variant] = {}
                for p, prefix in enumerate(projection['prefixes']):
                    mean, noise = np.asarray(state['mean'][p]), np.asarray(state['noise'][p])
                    predicted = np.column_stack((mean[:, 0], mean[:, 2], noise[:, 1]))
                    metrics = prediction_metrics(predicted[positions[known]], actual,
                        mean[baseline_index, 0], baseline['objective'], z)
                    snapshots[variant][str(prefix)] = metrics
                    for neighbor_index in FOCUS.get(seed, ()):
                        bank_index = positions[neighbor_index]
                        target = truth[tuple(neighbors[neighbor_index])]
                        target_difference = target['objective'] - baseline['objective']
                        focus_rows.append(dict(seed=seed, variant=variant, paid_prefix=prefix,
                            neighbor_index=neighbor_index,
                            never_sampled=neighbor_index in indices,
                            observed_before_prefix=tuple(neighbors[neighbor_index]) in
                                {tuple(row['point']) for row in history[:prefix]},
                            true_objective=target['objective'], true_objective_difference=target_difference,
                            true_feasible=target['true_feasible'],
                            predicted_objective=float(mean[bank_index, 0]),
                            predicted_initial_objective=float(mean[baseline_index, 0]),
                            predicted_objective_difference=state['difference_mean'][p][neighbor_index],
                            objective_difference_variance=state['difference_variance'][p][neighbor_index],
                            probability_improving=state['probability_improving'][p][neighbor_index],
                            predicted_constraint_mean=float(mean[bank_index, 2]),
                            predicted_constraint_noise_variance=float(noise[bank_index, 1]),
                            predicted_risk_margin=float(mean[bank_index, 2] + z*np.sqrt(noise[bank_index, 1])),
                            true_risk_margin=target['chance_margin']))
                for begin, end in ((0, 10), (10, 12)):
                    changes.append(dict(seed=seed, variant=variant, begin=begin, end=end,
                        change_after_minus_before=difference(snapshots[variant][str(end)],
                                                            snapshots[variant][str(begin)])))
            paired = {str(prefix): difference(snapshots[VARIANTS[1]][str(prefix)],
                                               snapshots[VARIANTS[0]][str(prefix)])
                      for prefix in projection['prefixes']}
            task_rows.append(dict(seed=seed, initial_true_feasible=baseline['true_feasible'],
                new_neighbor_count=sum(tuple(point) not in initial for point in neighbors),
                never_sampled_neighbor_indices=indices, known_truth_neighbor_indices=known,
                unknown_truth_count=len(indices)-len(known), snapshots=snapshots,
                candidate_minus_original=paired))
            for stage in projection.get('query_rankings', []):
                for variant, choice in stage['variants'].items():
                    point = tuple(bank[choice['selected_bank_index']])
                    target = truth.get(point)
                    query_rows.append(dict(seed=seed, variant=variant,
                        paid_prefix=stage['paid_prefix'], candidate_count=stage['candidate_count'],
                        **choice, truth_known=target is not None,
                        selected_neighbor_index=next((i for i, p in enumerate(neighbors)
                                                      if tuple(p) == point), None),
                        true_feasible=None if target is None else target['true_feasible'],
                        true_objective_difference=(None if target is None else
                                                   target['objective']-baseline['objective']),
                        feasible_improving=(None if target is None else bool(target['true_feasible']
                            and target['objective'] < baseline['objective']-1e-12))))
        aggregates = {}
        for label, weighted in (('task_equal_weight', False), ('point_weighted', True)):
            snapshots = {variant: {str(prefix): average_metrics(
                [row['snapshots'][variant][str(prefix)] for row in task_rows], weighted)
                for prefix in (0, 10, 11, 12)} for variant in VARIANTS}
            aggregates[label] = dict(snapshots=snapshots,
                candidate_minus_original={str(prefix): difference(
                    snapshots[VARIANTS[1]][str(prefix)], snapshots[VARIANTS[0]][str(prefix)])
                    for prefix in (0, 10, 11, 12)},
                propagation_changes=[dict(variant=variant, begin=begin, end=end,
                    change_after_minus_before=difference(snapshots[variant][str(end)],
                                                        snapshots[variant][str(begin)]))
                    for variant in VARIANTS for begin, end in ((0, 10), (10, 12))])
        query_summary = []
        for variant in VARIANTS:
            group = [row for row in query_rows if row['variant'] == variant]
            query_summary.append(dict(variant=variant, choices=len(group),
                known_truth_count=sum(row['truth_known'] for row in group),
                unknown_truth_count=sum(not row['truth_known'] for row in group),
                known_true_feasible=sum(row['true_feasible'] is True for row in group),
                known_feasible_improving=sum(row['feasible_improving'] is True for row in group)))
        result = dict(status='complete', task_count=len(seeds),
            cohort_count=sum(len(row['never_sampled_neighbor_indices']) for row in task_rows),
            unknown_truth_count=sum(row['unknown_truth_count'] for row in task_rows),
            all_initials_true_feasible=all(row['initial_true_feasible'] for row in task_rows),
            tasks=task_rows, aggregates=aggregates, propagation_changes=changes,
            focal_predictions=focus_rows, query_mean_rule_choices=query_rows,
            query_mean_rule_summary=query_summary,
            scope=('Six original-role paid trajectories, frozen end-of-path never-sampled shared '
                   'neighbors. Cohort excludes all13 actual observations although projections '
                   'stop at paid12. Policies within a task are not independent replications. '
                   'Task-equal metrics are primary; point-weighted values are descriptive.'),
            metric_definitions=('f,g and constraint episode variance use cached truth. '
                'Objective differences subtract the same initial policy; negative is better. '
                'Risk margin is centered g plus normal .95 quantile times predicted sigma. '
                'Direction uses zero tolerance1e-12. Feasible-only averages omit tasks with '
                'no feasible cohort point and report their contributing task count.'),
            query_scope=('Selections use posterior means on saved query pools. They are one-step '
                'diagnostics, not original Thompson sampling or a new closed loop. Unknown truth '
                'remains unknown; counts are over cached outcomes and give no deployment credit.'),
            cost=cost)
    save(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    result = summarize(read(args.manifest))
    print(dict(status=result['status'], tasks=result['task_count'],
               cohort_count=result['cohort_count'], unknown_truth=result['unknown_truth_count']), flush=True)


if __name__ == '__main__':
    main()
