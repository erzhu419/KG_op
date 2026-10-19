"""Aggregate already-scored frozen three-family decisions; never evaluate truth."""
import argparse
from collections import Counter

import numpy as np

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_terminal_equal import require_all
from performance.summarize_mainline_terminal_confirmation import prediction_by_arm
from performance.summarize_mainline_terminal_cross import (
    FULL, LOCAL, METRICS, NATIVE_SUFFIX, contrast, mean_known,
)

POINT = 'legacy_role_point_only'
MANIFEST = ROOT / 'performance/manifests/mainline_three_family_v1.json'


def paired(rows, families, terms, name):
    indexed = {((row['domain'], row['seed']), row['arm']): row for row in rows}
    tasks = list(dict.fromkeys((row['domain'], row['seed']) for row in rows))
    result = contrast(indexed, tasks, terms, name)
    for row in result['task_differences']:
        row['domain'], row['seed'] = row['seed']
    result['by_family'] = []
    for domain in families:
        family_rows = [row for row in rows if row['domain'] == domain]
        seeds = list(dict.fromkeys(row['seed'] for row in family_rows))
        values = contrast({(row['seed'], row['arm']): row for row in family_rows},
                          seeds, terms, name)
        result['by_family'].append(dict(domain=domain, **values))
    result['family_equal_mean_difference'] = {
        metric: mean_known([row['mean_difference'][metric]
                            for row in result['by_family']]) for metric in METRICS}
    return result


def arm_summary(rows, families, definitions):
    groups = []
    tasks = list(dict.fromkeys((row['domain'], row['seed']) for row in rows))
    indexed = {(row['domain'], row['seed'], row['arm']): row for row in rows}
    for arm, definition in definitions.items():
        group = [row for row in rows if row['arm'] == arm]
        means = {metric: mean_known([indexed.get((domain, seed, arm), {}).get(field)
                                    for domain, seed in tasks])
                 for metric, field in METRICS.items()}
        family_means = {metric: mean_known([
            mean_known([indexed.get((family, seed, arm), {}).get(field)
                        for family, seed in tasks if family == domain])
            for domain in families]) for metric, field in METRICS.items()}
        groups.append(dict(arm=arm, history_arm=definition['history_arm'],
            terminal_backend=definition['terminal_backend'], task_count=len(group),
            expected_task_count=len(tasks), execution_failure_count=len(tasks)-len(group),
            **{key: sum(row[key] is True for row in group) for key in (
                'certified', 'certified_true_feasible', 'certified_feasible_improving',
                'false_certificate', 'returned_policy_changed')},
            task_equal_means=means, family_equal_means=family_means,
            logical_target_calls=sum(row['logical_total_calls'] for row in group),
            verification_calls=sum(row['new_verification_calls'] for row in group),
            branches=dict(Counter(row['branch'] for row in group))))
    return groups


def prediction_summary(cohort, diagnoses):
    names = [row['arm'] for row in diagnoses[0]['prediction_by_arm']]
    aggregate = prediction_by_arm(cohort, names)
    for arm in aggregate:
        values = [next(row for row in diagnosis['prediction_by_arm']
                       if row['arm'] == arm['arm']) for diagnosis in diagnoses]
        for prefix in ('10', '12'):
            for metric in ('mean_bias_f_g_noise_variance', 'mse_f_g_noise_variance'):
                means = [row['snapshots'][prefix][metric]['task_equal'] for row in values]
                arm['snapshots'][prefix][metric]['family_equal'] = (
                    None if any(value is None for value in means) else np.mean(means, axis=0))
    return aggregate


def prediction_decision_rows(cohort, rows, positive, negative):
    """Expose paired prediction and actual decisions without treating policies as tasks."""
    indexed = {(row['domain'], row['seed'], row['arm']): row for row in rows}
    linked = []
    for task in cohort:
        domain, seed = task['domain'], task['seed']
        a, b = (indexed[domain, seed, arm] for arm in (positive, negative))
        arms = {row['arm']: row for row in task['arms']}
        difference = {}
        for prefix in ('10', '12'):
            difference[prefix] = (None if not task['count'] else
                np.asarray(arms[positive]['snapshots'][prefix]['mse_f_g_noise_variance']) -
                np.asarray(arms[negative]['snapshots'][prefix]['mse_f_g_noise_variance']))
        linked.append(dict(domain=domain, seed=seed, unsampled_policy_count=task['count'],
            mse_difference_f_g_noise_variance=difference,
            **{metric + '_difference': (None if a[field] is None or b[field] is None
                else a[field] - b[field]) for metric, field in METRICS.items()},
            same_deployment=a['deployed'] == b['deployed'],
            positive_certified=a['certified'], negative_certified=b['certified']))
    return linked


def numerical_events(plan):
    def counts(events):
        return dict(roundoff_status2=sum(event['integration_status'] == 2 for event in events),
                    accepted_interval_limit_status1=sum(event['integration_status'] == 1
                                                        for event in events))
    families = []
    for family in plan['families']:
        source = read(ROOT / family['search_manifest'])
        failed_cases = {(row['seed'], row['arm']) for row in source.get('execution_failures', [])}
        traces = [trace for seed in source['target_seeds'] for arm in source['arms']
            if (seed, arm['name']) not in failed_cases
            for trace in read(ROOT / source['output_dir'] / f'seed_{seed}' /
                arm['name'] / 'search.json').get('role_restoration_trace', [])]
        families.append(dict(domain=family['domain'], restoration_trace_count=len(traces),
            **counts([event for trace in traces for event in trace['roundoff_events']])))
    preflight = read(ROOT / plan['preflight_file'])
    preflight_families = [dict(domain=row['family'], **counts(row['roundoff_events']))
                         for row in preflight['results']]
    fields = ('roundoff_status2', 'accepted_interval_limit_status1')
    return dict(search=dict(by_family=families,
                    **{field: sum(row[field] for row in families) for field in fields}),
                preflight=dict(by_family=preflight_families,
                    **{field: sum(row[field] for row in preflight_families) for field in fields}),
                scope=('Recorded event occurrences in each unique search history restoration '
                    'trace, counted once per saved occurrence; paired terminal cells do not '
                    'duplicate search counts. Preflight replay occurrences remain separate.'))


def summarize(output, plan):
    # Enforce the global barrier before opening scored results or truth diagnostics.
    family_plans = [(family, read(ROOT / family['terminal_manifest']))
                    for family in plan['families']]
    for _, terminal_plan in family_plans:
        require_all(ROOT / terminal_plan['output_dir'], terminal_plan, 'verification.json')
    with attempt(output, 'aggregate-three-family-frozen-confirmation') as cost:
        cost.update(new_source_fits=0, new_target_initial_calls=0,
                    new_truth_function_calls=0, download_bytes=0)
        rows, cohorts, family_results, diagnoses = [], [], [], []
        query_rows = []
        for family, terminal_plan in family_plans:
            domain = family['domain']
            result = read(ROOT / terminal_plan['output_dir'] / 'results.json')
            diagnosis = read(ROOT / terminal_plan['source_output_dir'] / 'diagnosis.json')
            if result['status'] != 'complete' or diagnosis['status'] != 'complete':
                raise ValueError('All family summaries must complete before aggregation')
            family_results.append(dict(domain=domain, results=result))
            diagnoses.append(diagnosis)
            rows.extend(dict(domain=domain, **row) for row in result['rows'])
            cohorts.extend(dict(domain=domain, **row)
                           for row in diagnosis['shared_never_sampled_cohort'])
            query_rows.extend(dict(domain=domain, seed=row['seed'], arm=row['arm'],
                                   **row['query_summary'])
                              for row in diagnosis['rows'] if 'query_summary' in row)

        families = [family['domain'] for family, _ in family_plans]
        definitions = {arm['name']: arm for arm in family_plans[0][1]['arms']}
        comparisons = [('after_init_minus_source0_point_on_original_terminal',
                        {LOCAL: 1, POINT: -1})]
        first = family_results[0]['results']
        existing = first['paired'] + [first['interaction']] + first['reference_comparisons']
        comparisons.extend((item['name'], item['coefficients']) for item in existing)
        comparisons.extend((LOCAL + '_minus_' + reference, {LOCAL: 1, reference: -1})
            for reference in definitions if reference not in
            {FULL, LOCAL, POINT, FULL + NATIVE_SUFFIX, LOCAL + NATIVE_SUFFIX})
        contrasts = [paired(rows, families, terms, name) for name, terms in comparisons]
        prediction = prediction_summary(cohorts, diagnoses)
        # The shared helper reports seeds; restore explicit family labels in the pooled list.
        active = [row for row in cohorts if row['count']]
        for arm in prediction:
            for item, task in zip(arm['task_mse_reduction'], active, strict=True):
                item['domain'] = task['domain']
        result = dict(status='complete', protocol=plan, task_count=len(cohorts),
            family_count=len(families), rows=rows,
            execution_failures=[dict(failure, domain=family['domain'])
                for family, terminal_plan in family_plans
                for failure in terminal_plan.get('execution_failures', [])],
            arms=arm_summary(rows, families, definitions), paired=contrasts,
            primary=contrasts[0], by_family=[dict(domain=item['domain'],
                results_file=str(ROOT / terminal_plan['output_dir'] / 'results.json'),
                **{key: item['results'][key] for key in ('task_count', 'arms', 'paired',
                    'interaction', 'reference_comparisons', 'improving_deployments')})
                for item, (_, terminal_plan) in zip(family_results, family_plans, strict=True)],
            common_unobserved_prediction=prediction,
            prediction_to_decision=dict(
                after_init_minus_source0_point=prediction_decision_rows(cohorts, rows, LOCAL, POINT),
                full_minus_after_init=prediction_decision_rows(cohorts, rows, FULL, LOCAL)),
            query_by_task=query_rows,
            query_by_arm=[dict(arm=arm,
                task_count=sum(row['arm'] == arm for row in query_rows),
                expected_task_count=len(cohorts),
                execution_failure_count=len(cohorts)-sum(row['arm'] == arm for row in query_rows),
                **{key: sum(row[key] for row in query_rows
                if row['arm'] == arm) for key in ('count', 'true_feasible',
                'feasible_improving', 'new_neighbor_selected', 'new_neighbor_improving')})
                for arm in dict.fromkeys(row['arm'] for row in query_rows)],
            improving_deployments=[dict(domain=family['domain'], **row)
                for family in family_results for row in family['results']['improving_deployments']],
            truth_cache_complete=all(item['results']['truth_cache_complete'] for item in family_results),
            decomposition_maximum_error=max(item['results']['decomposition_maximum_error']
                                             for item in family_results),
            metric_definitions=first['metric_definitions'],
            numerical_events=numerical_events(plan),
            weighting=('Task-equal and family-equal separately reported; identical with four '
                'tasks per family. Win/tie/loss tolerance 1e-12. Policies are diagnostic units only.'),
            scope=('Frozen three-family confirmation. Primary: initialization propagation then '
                'point-only versus source0 point-only under the original terminal. '
                'Keep full propagation, native terminal 2x2, old native, same-source GP, '
                'current structural and STOP. Every terminal verification precedes all truth scoring.'),
            prediction_scope=diagnoses[0]['prediction_scope'],
            cost_scope='Aggregation ledger only; parent completion combines search, terminal and aggregation costs.')
        save(output / 'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    args = parser.parse_args()
    plan = read(args.manifest)
    result = summarize(ROOT / plan['output_dir'], plan)
    print(dict(status=result['status'], tasks=result['task_count'],
               primary=result['primary']['mean_difference']), flush=True)


if __name__ == '__main__':
    main()
