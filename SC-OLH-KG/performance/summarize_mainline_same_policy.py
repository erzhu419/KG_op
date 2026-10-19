"""Score the same-policy terminal challenge using saved truth only."""
import argparse

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_terminal_equal import cases
from performance.summarize_mainline_terminal_cross import FULL, LOCAL, NATIVE_SUFFIX, METRICS, mean_known
from performance.summarize_mainline_three_family import (
    POINT, arm_summary, paired, prediction_decision_rows,
)

MANIFEST = ROOT / 'performance/manifests/mainline_same_policy_challenge_v1.json'


def best_initial_fields(row, initial_points, best):
    truth = row['deployed_truth']
    feasible = row['certified_true_feasible']
    new = tuple(row['deployed']) not in initial_points
    known = truth is not None
    return dict(best_feasible_initial_truth=best,
        deployed_outside_initial10=new,
        certified_new_policy=bool(feasible and new) if feasible is not None else None,
        certified_new_policy_beats_best_initial=(False if not row['certified'] or feasible is False
            or not new else None if not known else truth['objective'] < best['objective'] - 1e-12),
        certified_quality_gain_vs_best_feasible_initial=(0. if not row['certified'] or feasible is False
            else None if not known else truth['quality'] - best['quality']))


def score_cell(baseline, item, saved, cache, initial_points, best):
    row = dict(baseline)
    terminal = saved['verification']
    actual = cache.get(tuple(saved['deployed']))
    certified = bool(terminal['certified'])
    feasible = None if actual is None else bool(actual['true_feasible'])
    initial = cache[tuple(item['initial'])]
    if not certified or feasible is False:
        reward = quality = reference = 0.
    elif feasible is True:
        reward, reference = actual['quality'], initial['quality']
        quality = reward - reference
    else:
        reward = quality = reference = None
    logical_verification = terminal['verification_budget']
    row.update(protocol='same_policy_challenge', deployed=saved['deployed'],
        verification_seed=saved['verification_seed'], initial_truth=initial,
        deployed_truth=actual, certified=certified,
        certified_true_feasible=False if not certified else feasible,
        false_certificate=False if not certified else None if feasible is None else not feasible,
        actual_safe_reward=reward, certified_feasible_quality_gain=quality,
        certified_initial_quality_component=reference,
        certified_feasible_improving=(False if not certified or feasible is False else
            None if actual is None else actual['objective'] < initial['objective'] - 1e-12),
        selected_shortlist_rank=terminal['selected_shortlist_rank'],
        branch=terminal['objective_comparison']['status'],
        returned_policy_changed=saved['deployed'] != item['initial'],
        new_verification_calls=saved['new_verification_calls'],
        logical_verification_calls=logical_verification,
        reused_verification_calls=logical_verification-saved['new_verification_calls'],
        logical_search_calls=item['search_evaluation_count'],
        logical_total_calls=item['search_evaluation_count']+logical_verification,
        baseline_branch=baseline['branch'],
        returned_policy_changed_vs_baseline=saved['deployed'] != baseline['deployed'],
        certified_deployment_changed_vs_baseline=(saved['deployed'] if certified else None) !=
            (baseline['deployed'] if baseline['certified'] else None))
    row.update(best_initial_fields(row, initial_points, best))
    return row


def change_summary(baseline, modified, families, expected_by_family):
    old = {(r['domain'], r['seed'], r['arm']): r for r in baseline}
    differences = []
    for row in modified:
        before = old[row['domain'], row['seed'], row['arm']]
        differences.append(dict(domain=row['domain'], seed=row['seed'], arm=row['arm'],
            baseline_branch=before['branch'], modified_branch=row['branch'],
            baseline_certified=before['certified'], modified_certified=row['certified'],
            returned_policy_changed=row['deployed'] != before['deployed'],
            certified_deployment_changed=row['certified_deployment_changed_vs_baseline'],
            new_verification_calls=row['new_verification_calls'],
            baseline_logical_verification_calls=before['new_verification_calls'],
            modified_logical_verification_calls=row['logical_verification_calls'],
            logical_verification_call_change=row['logical_verification_calls'] -
                before['new_verification_calls'],
            baseline_certified_new_policy=before['certified_new_policy_beats_best_initial'],
            modified_certified_new_policy=row['certified_new_policy_beats_best_initial'],
            **{metric: None if row[field] is None or before[field] is None else
                row[field]-before[field] for metric, field in METRICS.items()}))

    def aggregate(group, expected):
        completed_means = {metric: mean_known([r[metric] for r in group]) for metric in METRICS}
        return dict(cells=len(group), expected_task_count=expected,
            failed_task_count=expected-len(group),
            mean_difference=(completed_means if len(group) == expected else
                             {metric: None for metric in METRICS}),
            completed_tasks_mean_difference=completed_means,
            win_tie_loss={metric: dict(wins=sum(r[metric] is not None and r[metric] > 1e-12 for r in group),
                ties=sum(r[metric] is not None and abs(r[metric]) <= 1e-12 for r in group),
                losses=sum(r[metric] is not None and r[metric] < -1e-12 for r in group),
                unknown=expected-len(group)+sum(r[metric] is None for r in group)) for metric in METRICS},
            returned_policy_changes=sum(r['returned_policy_changed'] for r in group),
            certified_deployment_changes=sum(r['certified_deployment_changed'] for r in group),
            newly_certified=sum(r['modified_certified'] and not r['baseline_certified'] for r in group),
            certification_lost=sum(r['baseline_certified'] and not r['modified_certified'] for r in group),
            baseline_logical_verification_calls=sum(r['baseline_logical_verification_calls'] for r in group),
            modified_logical_verification_calls=sum(r['modified_logical_verification_calls'] for r in group),
            logical_verification_call_change=sum(r['logical_verification_call_change'] for r in group),
            marginal_new_physical_verification_calls=sum(r['new_verification_calls'] for r in group))

    return dict(rows=differences,
        by_arm=[dict(arm=arm, **aggregate([r for r in differences if r['arm'] == arm],
                                        sum(expected_by_family.values())),
            by_family=[dict(domain=family, **aggregate([r for r in differences if
                r['arm'] == arm and r['domain'] == family], expected_by_family[family]))
                for family in families])
            for arm in dict.fromkeys(r['arm'] for r in differences)],
        scope='Each arm pairs its completed original/modified cells; missing executions remain unknown in full-cohort means. Completed-task means are conditional. Crossed terminals are not independent tasks.')


def deployment_summary(rows, families, expected_by_family):
    def aggregate(group, expected):
        completed_mean = mean_known([
            r['certified_quality_gain_vs_best_feasible_initial'] for r in group])
        return dict(tasks=len(group), expected_task_count=expected,
            failed_task_count=expected-len(group),
            certified_new_policy=sum(r['certified_new_policy'] is True for r in group),
            certified_new_policy_beats_best_initial=sum(r['certified_new_policy_beats_best_initial']
                is True for r in group),
            mean_certified_quality_gain_vs_best_feasible_initial=(
                completed_mean if len(group) == expected else None),
            completed_tasks_mean_certified_quality_gain_vs_best_feasible_initial=completed_mean,
            unknown_certified_deployment_truth=sum(r['certified'] and r['deployed_truth'] is None
                for r in group))
    return [dict(arm=arm, **aggregate([r for r in rows if r['arm'] == arm],
                                    sum(expected_by_family.values())),
        by_family=[dict(domain=family, **aggregate([r for r in rows if r['arm'] == arm
            and r['domain'] == family], expected_by_family[family])) for family in families])
        for arm in dict.fromkeys(r['arm'] for r in rows)]


def summarize(output, plan):
    parent = read(ROOT / plan['source_manifest'])
    families = [family['domain'] for family in parent['families']]
    family_plans = [(family, read(ROOT / family['terminal_manifest']))
                    for family in parent['families']]
    # Detect incomplete terminal execution before any cached target truth is opened.
    paths = []
    for family, terminal_plan in family_plans:
        definitions = {row['name']: row for row in terminal_plan['arms']}
        paths.extend((family, terminal_plan, seed, definitions[arm],
                      output / family['domain'] / f'seed_{seed}' / arm / protocol)
                     for seed, arm, protocol in cases(terminal_plan))
    missing = [str(path / 'verification.json') for *_, path in paths
               if not (path / 'verification.json').is_file()]
    if missing:
        raise RuntimeError(f'All terminal verifications must finish before scoring: {missing}')
    with attempt(output, 'summarize-same-policy-challenge') as cost:
        cost.update(new_source_fits=0, new_target_initial_calls=0,
                    new_truth_function_calls=0, download_bytes=0)
        original = read(ROOT / parent['output_dir'] / 'results.json')
        failures = plan.get('execution_failures', [])
        expected_by_family = {family['domain']: len(terminal_plan['target_seeds'])
                              for family, terminal_plan in family_plans}
        baseline_index = {(r['domain'], r['seed'], r['arm']): r for r in original['rows']}
        task_data, cohorts = {}, []
        for family, terminal_plan in family_plans:
            diagnosis = read(ROOT / terminal_plan['source_output_dir'] / 'diagnosis.json')
            cohorts.extend(dict(domain=family['domain'], **row)
                           for row in diagnosis['shared_never_sampled_cohort'])
            for seed in terminal_plan['target_seeds']:
                directory = ROOT / terminal_plan['source_output_dir'] / f'seed_{seed}'
                cache = {tuple(r['point']): r['truth'] for r in read(directory / 'truth.json')}
                initial = {tuple(r['point']) for r in
                           read(directory / 'initialization.json')['history']}
                best_point = min((p for p in initial if cache[p]['true_feasible']),
                                 key=lambda p: cache[p]['objective'])
                task_data[family['domain'], seed] = cache, initial, cache[best_point]
        rows, baseline = [], []
        for family, _, seed, definition, path in paths:
            arm, domain = definition['name'], family['domain']
            before = dict(baseline_index[domain, seed, arm])
            cache, initial, best = task_data[domain, seed]
            before.update(best_initial_fields(before, initial, best))
            baseline.append(before)
            item, saved = read(path / 'input.json'), read(path / 'verification.json')
            row = score_cell(before, item, saved, cache, initial, best)
            row['input_changed'] = saved['input_changed']
            row['reused_raw_calls'] = saved['reused_raw_calls']
            rows.append(row)
        definitions = {arm['name']: arm for arm in family_plans[0][1]['arms']}
        contrasts = [paired(rows, families, comparison['coefficients'], comparison['name'])
                     for comparison in original['paired']]
        contrasts.extend(paired(rows, families, {FULL: 1, reference: -1},
            FULL + '_minus_' + reference) for reference in definitions
            if reference not in {FULL, LOCAL, FULL + NATIVE_SUFFIX, LOCAL + NATIVE_SUFFIX})
        contrasts[0]['source_comparison_name'] = contrasts[0]['name']
        contrasts[0]['name'] = 'after_init_minus_source0_point_under_same_policy_challenge'
        for family in contrasts[0]['by_family']:
            family['source_comparison_name'] = family['name']
            family['name'] = contrasts[0]['name']
        errors = [abs(row['actual_safe_reward'] - row['certified_feasible_quality_gain'] -
                      row['certified_initial_quality_component']) for row in rows
                  if row['actual_safe_reward'] is not None]
        if max(errors, default=0.) > 1e-12:
            raise ValueError('Reward decomposition differs from certification plus quality')
        arms = arm_summary(rows, families, definitions)
        baseline_arms = arm_summary(baseline, families, definitions)
        for summaries, arm_rows in ((arms, rows), (baseline_arms, baseline)):
            for summary in summaries:
                selected = [row for row in arm_rows if row['arm'] == summary['arm']]
                summary.update(logical_search_calls=sum(row['logical_search_calls'] for row in selected),
                    logical_verification_calls=sum(row.get('logical_verification_calls',
                        row['new_verification_calls']) for row in selected))
        result = dict(status='complete', protocol=plan, task_count=len(task_data),
            execution_status=('completed_with_declared_execution_failures' if failures
                              else 'complete'),
            registered_terminal_cells=plan['terminal_cells'],
            failed_terminal_cells=plan['terminal_cells']-len(rows),
            execution_failures=failures,
            family_count=len(families), terminal_cells=len(rows), rows=rows,
            arms=arms, baseline_arms=baseline_arms,
            primary=contrasts[0], baseline_primary=original['primary'], paired=contrasts,
            modified_minus_baseline=change_summary(baseline, rows, families, expected_by_family),
            prediction_to_decision=dict(
                after_init_minus_source0_point=prediction_decision_rows(cohorts, rows, LOCAL, POINT),
                full_minus_after_init=prediction_decision_rows(cohorts, rows, FULL, LOCAL)),
            best_initial_deployments=dict(baseline=deployment_summary(baseline, families, expected_by_family),
                modified=deployment_summary(rows, families, expected_by_family)),
            input_changed_cells=sum(r['input_changed'] for r in rows),
            returned_policy_changed_cells=sum(r['returned_policy_changed_vs_baseline'] for r in rows),
            certified_deployment_changed_cells=sum(r['certified_deployment_changed_vs_baseline'] for r in rows),
            certified=sum(r['certified'] for r in rows),
            certified_true_feasible=sum(r['certified_true_feasible'] is True for r in rows),
            false_certificates=sum(r['false_certificate'] is True for r in rows),
            truth_cache_complete=all(r['deployed_truth'] is not None for r in rows),
            decomposition_maximum_error=max(errors, default=0.),
            metric_definitions=dict(original['metric_definitions'],
                best_initial='Auxiliary: certified true-feasible quality minus best true-feasible initial10 quality; zero if uncertified. New policy means outside shared initial10.',
                physical_cost='new_verification_calls counts new physical observations; logical_total_calls includes all verification observations, reused or new.'),
            weighting=f'{len(task_data)} paired tasks across {len(families)} families; task-equal and family-equal means reported separately. Win/tie/loss tolerance 1e-12. Crossed terminal cells share trajectories.',
            scope=plan.get('scope', 'Development comparison on opened Stage47 tasks. All new verifiers completed before scoring. Frozen search, models and candidate lists; same-policy terminal-only intervention. Saved truth only; unknown truth remains unknown.') +
                (' Declared failed searches have no terminal result and are not zero-imputed. '
                 'Complete-cohort means involving them are unknown; completed-task means are conditional.'
                 if failures else ''))
        save(output / 'results.json', result)
    return result


def completion(output, result):
    from datetime import datetime, timezone
    import json
    attempts = [json.loads(line) for path in output.rglob('attempts.jsonl')
                for line in path.read_text().splitlines()]
    fields = ('cpu_seconds', 'wall_seconds', 'new_source_calls', 'new_target_initial_calls',
              'new_target_search_calls', 'new_verification_calls', 'new_source_fits',
              'new_truth_function_calls', 'source_model_restorations', 'download_bytes')
    totals = {field: sum(row.get(field, 0) for row in attempts) for field in fields}
    failed_logical = sum(row['logical_search_calls'] for row in result['execution_failures'])
    completed_logical = sum(r['logical_total_calls'] for r in result['rows'])
    completed = dict(status='complete', execution_status=result['execution_status'],
        registered_terminal_cells=result['registered_terminal_cells'],
        failed_terminal_cells=result['failed_terminal_cells'],
        execution_failures=result['execution_failures'],
        terminal_cells=len(result['rows']), cost=totals,
        attempt_count=len(attempts), failed_attempts=[r for r in attempts if r['status'] == 'failed'],
        new_physical_target_calls=sum(totals[key] for key in
            ('new_target_initial_calls', 'new_target_search_calls', 'new_verification_calls')),
        logical_target_calls=completed_logical+failed_logical,
        completed_cells_logical_target_calls=completed_logical,
        failed_cells_logical_search_calls=failed_logical,
        logical_verification_calls=sum(r['logical_verification_calls'] for r in result['rows']),
        reused_raw_calls_expanded_cells=sum(r['reused_raw_calls'] for r in result['rows']),
        input_changed_cells=result['input_changed_cells'],
        completed_utc=datetime.now(timezone.utc).isoformat())
    save(output / 'completion.json', completed)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    args = parser.parse_args()
    plan = read(args.manifest)
    output = ROOT / plan['output_dir']
    result = summarize(output, plan)
    finished = completion(output, result)
    plan['status'], plan['completed_utc'] = 'complete', finished['completed_utc']
    save(args.manifest, plan)
    print(dict(status='complete', tasks=result['task_count'], cells=result['terminal_cells'],
               primary=result['primary']['mean_difference'], cost=finished['cost']), flush=True)


if __name__ == '__main__':
    main()
