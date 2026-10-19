"""Summarize completed role-prior decisions without model or truth evaluation."""
import argparse

import numpy as np

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.summarize_mainline_role_fresh import (
    AFTER_INIT_POINT, FULL, POINT, prediction_arms,
)


MANIFEST = ROOT / 'performance/manifests/mainline_role_confirmation_v1.json'
METRICS = {
    'reward': 'actual_safe_reward',
    'quality': 'certified_feasible_quality_gain',
    'certification_reference': 'certified_initial_quality_component',
}


def win_tie_loss(values):
    values = np.asarray(values)
    return dict(wins=int(np.sum(values > 1e-12)),
                ties=int(np.sum(np.abs(values) <= 1e-12)),
                losses=int(np.sum(values < -1e-12)))


def paired_result(rows, seeds, arm, reference):
    differences = [dict(seed=s, **{key: rows[s, arm][field] - rows[s, reference][field]
                       for key, field in METRICS.items()}) for s in seeds]
    return dict(reference=reference,
        mean_difference={key: float(np.mean([r[key] for r in differences]))
                         for key in METRICS},
        win_tie_loss={key: win_tie_loss([r[key] for r in differences])
                      for key in METRICS},
        task_differences=differences)


def build_summary(plan, results, diagnosis, completion):
    """Pure aggregation of final artifacts, also usable for a read-only replay."""
    seeds = plan['target_seeds']
    names = [a['name'] if isinstance(a, dict) else a for a in plan['arms']]
    primary = plan.get('primary_reference', POINT)
    prediction_names = prediction_arms(plan)
    rows = {(r['seed'], r['arm']): r for r in results['rows']}
    if set(rows) != {(s, a) for s in seeds for a in names}:
        raise ValueError('Decision rows must cover exactly the registered task-arm pairs')
    if any(x['status'] != 'complete' for x in (results, diagnosis, completion)):
        raise ValueError('Final result, diagnosis and completion artifacts are required')
    # Detect a changed metric definition before interpreting a certification gain
    # as a policy-quality gain.
    errors = [abs(r['actual_safe_reward'] - r['certified_feasible_quality_gain']
                  - r['certified_initial_quality_component']) for r in rows.values()]
    if max(errors) > 1e-12:
        raise ValueError('Certified quality and initial-reference components do not sum to reward')

    paired = []
    for reference in [primary] + [a for a in names if a not in (FULL, primary)]:
        paired.append(paired_result(rows, seeds, FULL, reference))

    arms, improvements = [], []
    diagnostic_rows = {(r['seed'], r['arm']): r for r in diagnosis['rows']}
    for arm in names:
        group = [rows[s, arm] for s in seeds]
        queries = [diagnostic_rows[s, arm].get('query_summary', {}) for s in seeds]
        arms.append(dict(arm=arm, task_count=len(group),
            certified=sum(r['certified'] for r in group),
            certified_true_feasible=sum(r['certified'] and r['deployed_truth']['true_feasible']
                                        for r in group),
            false_certificates=sum(r['false_certificate'] for r in group),
            certified_feasible_improving=sum(r['certified_feasible_improving'] for r in group),
            **{'mean_' + key: float(np.mean([r[field] for r in group]))
               for key, field in METRICS.items()},
            logical_target_calls=sum(r['logical_total_calls'] for r in group),
            verification_calls=sum(r['new_verification_calls'] for r in group),
            query_summary={key: sum(q.get(key, 0) for q in queries) for key in
                ('count', 'true_feasible', 'feasible_improving',
                 'new_neighbor_selected', 'new_neighbor_improving')}))
        improvements.extend(dict(seed=r['seed'], arm=arm,
            initial_objective=r['initial_truth']['objective'],
            deployed_objective=r['deployed_truth']['objective'],
            quality_gain=r['certified_feasible_quality_gain'])
            for r in group if r['certified_feasible_improving'])

    cohort = diagnosis['shared_never_sampled_cohort']
    predictions = []
    for arm in prediction_names:
        active = [r for r in cohort if r['count']]
        values = [next(a for a in r['arms'] if a['arm'] == arm) for r in active]
        snapshots = {}
        for prefix in ('10', '12'):
            snapshots[prefix] = {}
            for metric in ('mean_bias_f_g_noise_variance', 'mse_f_g_noise_variance'):
                data = np.asarray([v['snapshots'][prefix][metric] for v in values])
                snapshots[prefix][metric] = dict(
                    task_equal=None if not active else np.mean(data, axis=0),
                    policy_weighted=None if not active else np.average(data, axis=0,
                        weights=[r['count'] for r in active]))
        changes = np.asarray([v['mse_reduction_f_g_noise_variance'] for v in values])
        predictions.append(dict(arm=arm, task_count=len(active),
            policy_count=sum(r['count'] for r in active), snapshots=snapshots,
            mse_improved_tasks_f_g_noise_variance=(None if not active else
                np.sum(changes > 1e-12, axis=0)),
            mse_improvement_absolute_tolerance=1e-12,
            task_mse_reduction=[dict(seed=r['seed'], count=r['count'],
                f_g_noise_variance=v['mse_reduction_f_g_noise_variance'])
                for r, v in zip(active, values, strict=True)]))

    coverage = [dict(seed=r['seed'], new_neighbor_count=r['new_shared']['count'],
        feasible_count=r['new_shared']['feasible_count'],
        feasible_improving_count=r['new_shared']['feasible_improving_count'],
        feasible_improving_neighbor_indices=[r['new_shared_neighbor_indices'][i]
            for i in r['new_shared']['feasible_improving_indices']],
        maximum_quality_gain=r['new_shared']['maximum_quality_gain'])
        for r in diagnosis['seeds']]
    summary = dict(status='complete', task_count=len(seeds),
        independent_unit=f'{len(seeds)} new task geometries; one paired observation stream per task.',
        primary_comparison=dict(arm=FULL, reference=primary,
            metric='quality', task_weighting='equal', result=paired[0]),
        metric_definitions=dict(
            reward='I(certified and true feasible) * returned quality',
            quality='I(certified and true feasible) * (returned quality - initial quality)',
            certification_reference='I(certified and true feasible) * initial quality',
            initial_reference='Initial quality is a numerical reference, not an assumed safe or certified policy.'),
        arms=arms, paired=paired, improving_deployments=improvements,
        shared_new_neighbor_coverage=coverage,
        tasks_with_improving_new_neighbors=sum(r['feasible_improving_count'] > 0 for r in coverage),
        common_unobserved_prediction=predictions,
        prediction_scope=('Fixed new shared neighbors never sampled in the full 13-row history of any '
            'of the four replacement backends: full, point-only, GP and structural. Native is outside '
            'this intersection. Saved prefix10 to prefix12 spans two adaptive observations; prefix13 '
            'predictions are not reconstructed. Tasks are primary units; policy-weighted errors are descriptive.'
            if len(prediction_names) == 4 else
            'Fixed new shared neighbors never sampled in the full 13-row history of any of these '
            'replacement backends: ' + ', '.join(prediction_names) + '. Native is outside this '
            'intersection. Saved prefix10 to prefix12 spans two adaptive observations; prefix13 '
            'predictions are not reconstructed. Tasks are primary units; policy-weighted errors are descriptive.'),
        intervention_scope=results['scope'],
        decomposition_maximum_error=max(errors), completion=completion)
    if AFTER_INIT_POINT in names:
        summary['initial_transfer_comparison'] = dict(
            arm=AFTER_INIT_POINT, reference=POINT, metric='quality',
            task_weighting='equal',
            result=paired_result(rows, seeds, AFTER_INIT_POINT, POINT))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    args = parser.parse_args()
    plan = read(args.manifest)
    output = ROOT / plan['output_dir']
    with attempt(output, 'decision-evidence-summary', plan['domain']) as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0,
                    new_truth_function_calls=0, download_bytes=0)
        result = build_summary(plan, read(output / 'results.json'),
                               read(output / 'diagnosis.json'), read(output / 'completion.json'))
        save(output / 'decision_summary.json', result)
    print(dict(status='complete', task_count=result['task_count'],
               primary=result['primary_comparison']['result']['mean_difference']), flush=True)


if __name__ == '__main__':
    main()
