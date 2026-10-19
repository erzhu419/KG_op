"""Source-only mean-ranking margins from sealed forecasts and unused repeats.

This module neither runs an optimizer nor constructs a posterior. Selection here
is an objective-mean diagnostic on held policies, not a replay of acquisition.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import norm

from performance.run_mainline_legacy_backend_bridge import read
from performance.summarize_mainline_source_observed_ranking import (
    load_frozen_cases, scoring_labels,
)


AFTER = 'legacy_role_after_init_point_only'
POINT = 'legacy_role_point_only'
FULL = 'legacy_role_full'
NATIVE = 'legacy_native'
COMPARISONS = ((AFTER, POINT), (FULL, AFTER), (AFTER, NATIVE))
PAIR_GROUPS = ('observed_observed', 'observed_unobserved', 'unobserved_unobserved')
PAIR_METRICS = ('order_error', 'gap_weighted_order_error',
                'unobserved_relative_optimism_mean')
SELECTION_POOLS = ('all_held', 'observed', 'unobserved')


def pair_group(i, j, prefix):
    return ('observed_observed' if j < prefix else
            'observed_unobserved' if i < prefix else 'unobserved_unobserved')


def rank_pairs(mean, labels, prefix):
    """Keep pair facts; never manufacture posterior difference uncertainty."""
    rows = []
    for i, j in combinations(range(len(mean)), 2):
        predicted = float(mean[i] - mean[j])
        label = float(labels[i] - labels[j])
        rows.append(dict(i=i, j=j, group=pair_group(i, j, prefix),
            predicted_gap=predicted, unused_label_gap=label,
            untied_label=bool(label != 0),
            wrong_order=None if label == 0 else bool(np.sign(predicted) != np.sign(label)),
            predicted_tie=bool(predicted == 0)))
    # Exactly Stage61's within-case/current-prefix/group descriptive stratum.
    # It is defined before applying an arm's eligibility mask.
    for group in PAIR_GROUPS:
        group_rows = [row for row in rows if row['group'] == group]
        near = {(row['i'], row['j']) for row in sorted(group_rows,
            key=lambda row: abs(row['unused_label_gap']))[:int(np.ceil(len(group_rows) / 4))]}
        for row in group_rows:
            row['small_label_gap_quarter'] = (row['i'], row['j']) in near
    return rows


def pair_metrics(rows):
    untied = [row for row in rows if row['untied_label']]
    total_gap = sum(abs(row['unused_label_gap']) for row in untied)
    mixed = [row for row in rows if row['group'] == 'observed_unobserved']
    return dict(pairs=len(rows), untied_label_pairs=len(untied),
        wrong_order_pairs=sum(row['wrong_order'] for row in untied),
        predicted_ties=sum(row['predicted_tie'] for row in rows),
        order_error=(sum(row['wrong_order'] for row in untied) / len(untied) if untied else None),
        gap_weighted_order_error=(sum(abs(row['unused_label_gap']) * row['wrong_order']
            for row in untied) / total_gap if total_gap else None),
        unobserved_relative_optimism_mean=(float(np.mean([
            row['predicted_gap'] - row['unused_label_gap'] for row in mixed])) if mixed else None))


def selection(mean, labels, empirical_margin, mask, prefix, pool):
    """Objective ranking inside a frozen mask; an empty mask is not a failure."""
    indices = [i for i in range(len(mean)) if bool(mask[i]) and
               (pool == 'all_held' or (i < prefix) == (pool == 'observed'))]
    order = sorted(indices, key=lambda i: (float(mean[i]), i))
    result = dict(pool=pool, eligible_count=len(order), eligible_indices=indices,
        status='available' if order else 'no_eligible', selected=None, runner_up=None,
        predicted_lead=None, unused_label_lead_over_runner_up=None,
        selected_unused_label_excess=None, selected_empirically_feasible=None,
        selected_empirical_margin=None, selected_is_unobserved=None,
        label_best=None, label_best_is_unobserved=None,
        selected_empirical_feasible_excess=None)
    if not order:
        return result
    chosen = order[0]
    best = min(indices, key=lambda i: (float(labels[i]), i))
    feasible = [i for i in indices if empirical_margin[i] <= 0]
    result.update(selected=chosen, label_best=best,
        selected_unused_label_excess=float(labels[chosen] - labels[best]),
        selected_empirically_feasible=bool(empirical_margin[chosen] <= 0),
        selected_empirical_margin=float(empirical_margin[chosen]),
        selected_is_unobserved=chosen >= prefix, label_best_is_unobserved=best >= prefix,
        selected_empirical_feasible_excess=(float(labels[chosen] - min(labels[i] for i in feasible))
            if feasible and empirical_margin[chosen] <= 0 else None))
    if len(order) > 1:
        runner = order[1]
        result.update(runner_up=runner, predicted_lead=float(mean[runner] - mean[chosen]),
            unused_label_lead_over_runner_up=float(labels[runner] - labels[chosen]))
    return result


def analyze_snapshot(snapshot, labels, alpha, held, prefix):
    if snapshot.get('status') != 'complete':
        return dict(prefix=prefix, status=snapshot.get('status', 'missing'),
            failure=snapshot.get('error'), candidates=[], pairs=[], groups=[], selections=[])
    mean = np.asarray(snapshot['mean'], dtype=float)[:, 0]
    probability = np.asarray(snapshot['probability_violation'], dtype=float)
    empirical_margin = labels['mean'][:, 1] + norm.ppf(1-alpha) * np.sqrt(labels['variance'][:, 1])
    eligible = probability <= .5
    pairs = rank_pairs(mean, labels['mean'][:, 0], prefix)
    groups = []
    for group in PAIR_GROUPS:
        group_rows = [row for row in pairs if row['group'] == group]
        for scope in ('all', 'predicted_eligible', 'empirically_feasible'):
            selected = [row for row in group_rows if scope == 'all' or
                (eligible[row['i']] and eligible[row['j']] if scope == 'predicted_eligible' else
                 empirical_margin[row['i']] <= 0 and empirical_margin[row['j']] <= 0)]
            for subset in ('all', 'small_label_gap_quarter'):
                subset_rows = [row for row in selected if subset == 'all' or row[subset]]
                groups.append(dict(pair_group=group, scope=scope, subset=subset,
                    metrics=pair_metrics(subset_rows)))
    return dict(prefix=prefix, status='complete', failure=None, pairs=pairs, groups=groups,
        selections=[selection(mean, labels['mean'][:, 0], empirical_margin, eligible, prefix, pool)
                    for pool in SELECTION_POOLS],
        candidates=[dict(held_position=i, policy_id=int(held[i]), observed=i < prefix,
            objective_mean=float(mean[i]), probability_violation=float(probability[i]),
            predicted_eligible=bool(eligible[i]), unused_objective_mean=float(labels['mean'][i, 0]),
            unused_replicates=int(labels['replicate_counts'][i]),
            unused_objective_sample_variance=float(labels['variance'][i, 0]),
            empirical_margin=float(empirical_margin[i]), empirically_feasible=bool(empirical_margin[i] <= 0),
            saved_objective_epistemic=float(snapshot['variance'][i][0]),
            saved_objective_episode_noise=float(snapshot['noise'][i][0])) for i in range(len(mean))])


def paired_snapshot(left, right, left_name, right_name):
    """Symmetric common eligibility isolates objective mean ordering changes."""
    result = dict(left=left_name, right=right_name, prefix=left['prefix'],
        status='complete' if left['status'] == right['status'] == 'complete' else 'unavailable',
        eligibility_disagreements=None, common_eligible_indices=None, groups=[], selections=[])
    if result['status'] != 'complete':
        return result
    lc, rc = left['candidates'], right['candidates']
    common = np.asarray([a['predicted_eligible'] and b['predicted_eligible'] for a, b in zip(lc, rc)])
    result.update(common_eligible_indices=np.flatnonzero(common).tolist(),
        eligibility_disagreements=sum(a['predicted_eligible'] != b['predicted_eligible'] for a, b in zip(lc, rc)))
    for group in PAIR_GROUPS:
        for subset in ('all', 'small_label_gap_quarter'):
            endpoint = lambda row: common[row['i']] and common[row['j']] and row['group'] == group and (
                subset == 'all' or row[subset])
            lm, rm = [pair_metrics([row for row in snap['pairs'] if endpoint(row)]) for snap in (left, right)]
            result['groups'].append(dict(pair_group=group, subset=subset, left=lm, right=rm,
                delta={key: None if lm[key] is None or rm[key] is None else lm[key] - rm[key]
                       for key in PAIR_METRICS}))
    labels = np.asarray([row['unused_objective_mean'] for row in lc])
    margin = np.asarray([row['empirical_margin'] for row in lc])
    for pool in SELECTION_POOLS:
        a, b = [selection(np.asarray([row['objective_mean'] for row in candidates]),
            labels, margin, common, left['prefix'], pool) for candidates in (lc, rc)]
        chosen = a['selected'] is not None and b['selected'] is not None
        result['selections'].append(dict(pool=pool, left=a, right=b,
            changed=None if not chosen else a['selected'] != b['selected'],
            unused_label_objective_delta=(float(labels[a['selected']] - labels[b['selected']])
                if chosen else None),
            empirical_feasible_excess_delta=(a['selected_empirical_feasible_excess'] -
                b['selected_empirical_feasible_excess'] if a['selected_empirical_feasible_excess'] is not None
                and b['selected_empirical_feasible_excess'] is not None else None)))
    return result


def describe(values):
    """Overlapping source groups: descriptive paired counts, no task-level CI."""
    known = [float(value) for value in values if value is not None]
    return dict(registered_groups=len(values), available_groups=len(known),
        mean=float(np.mean(known)) if known and len(known) == len(values) else None,
        conditional_mean=float(np.mean(known)) if known else None,
        conditional_median=float(np.median(known)) if known else None,
        lower=sum(value < -1e-12 for value in known),
        tied=sum(abs(value) <= 1e-12 for value in known),
        higher=sum(value > 1e-12 for value in known))


def aggregate_cases(cases, arms):
    arm_rows, comparison_rows = [], []
    for arm in arms:
        for prefix in (10, 13):
            snapshots = [next(row for row in case['arms'] if row['arm'] == arm)['snapshots'][str(prefix)]
                         for case in cases]
            record = dict(arm=arm, prefix=prefix, registered_groups=len(cases),
                completed_groups=sum(s['status'] == 'complete' for s in snapshots), groups=[], selections=[])
            for group in PAIR_GROUPS:
                for scope in ('all', 'predicted_eligible', 'empirically_feasible'):
                    for subset in ('all', 'small_label_gap_quarter'):
                        rows = [next((row['metrics'] for row in snap['groups'] if
                            (row['pair_group'], row['scope'], row['subset']) == (group, scope, subset)), {})
                            for snap in snapshots]
                        record['groups'].append(dict(pair_group=group, scope=scope, subset=subset,
                            total_pairs=sum(row.get('pairs', 0) for row in rows),
                            metrics={key: describe([row.get(key) for row in rows]) for key in PAIR_METRICS}))
            for pool in SELECTION_POOLS:
                rows = [next((row for row in snap['selections'] if row['pool'] == pool), {}) for snap in snapshots]
                record['selections'].append(dict(pool=pool,
                    no_eligible=sum(row.get('status') == 'no_eligible' for row in rows),
                    metrics={key: describe([row.get(key) for row in rows]) for key in (
                        'selected_unused_label_excess', 'predicted_lead', 'unused_label_lead_over_runner_up',
                        'selected_empirically_feasible', 'selected_empirical_feasible_excess',
                        'selected_is_unobserved')}))
            arm_rows.append(record)
    for left, right in COMPARISONS:
        for prefix in (10, 13):
            selected = [next(row for row in case['comparisons'] if
                (row['left'], row['right'], row['prefix']) == (left, right, prefix)) for case in cases]
            record = dict(left=left, right=right, prefix=prefix, registered_groups=len(cases),
                completed_groups=sum(row['status'] == 'complete' for row in selected), groups=[], selections=[])
            for group in PAIR_GROUPS:
                for subset in ('all', 'small_label_gap_quarter'):
                    rows = [next((row for row in item['groups'] if
                        (row['pair_group'], row['subset']) == (group, subset)), {}) for item in selected]
                    record['groups'].append(dict(pair_group=group, subset=subset,
                        common_pairs=sum(row.get('left', {}).get('pairs', 0) for row in rows),
                        delta={key: describe([row.get('delta', {}).get(key) for row in rows]) for key in PAIR_METRICS}))
            for pool in SELECTION_POOLS:
                rows = [next((row for row in item['selections'] if row['pool'] == pool), {}) for item in selected]
                record['selections'].append(dict(pool=pool,
                    changed=sum(row.get('changed') is True for row in rows),
                    unused_label_objective_delta=describe([row.get('unused_label_objective_delta') for row in rows]),
                    empirical_feasible_excess_delta=describe([row.get('empirical_feasible_excess_delta') for row in rows])))
            comparison_rows.append(record)
    return dict(arms=arm_rows, comparisons=comparison_rows)


def analyze_source(root, plan):
    """Read all 24 groups on a scheduler worker; no models or RNG are called."""
    root = Path(root)
    source_file = plan['inputs']['source_manifest']
    source = read(root / source_file)
    output = root / source['output_dir']
    records = {(archive['heldout_target_family'], fold):
        read(output / archive['heldout_target_family'] / f'fold_{fold}' / 'forecasts.json')
        for archive in source['source_archives'] for fold in range(source['folds'])}
    load_frozen_cases(source, records)
    cases, flags = [], []
    for archive in source['source_archives']:
        heldout = archive['heldout_target_family']
        tasks = {task['name']: task for task in read(root / archive['path'])['tasks']}
        for fold in range(source['folds']):
            directory = output / heldout / f'fold_{fold}'
            frozen = records[heldout, fold]
            split, fit = read(directory / 'split.json'), read(directory / 'fit_diagnostics.json')
            flags.append(dict(archive_heldout_family=heldout, fold=fold,
                precision_certified=fit['precision_certified'], source_identity_heads=fit['source_identity_heads']))
            by_family = {case['family']: case for case in frozen['cases']}
            for family in archive['source_families']:
                case = by_family.get(family, {})
                held = case.get('held_policy_ids', split['held'])
                task = tasks[family]
                labels = scoring_labels(task, held)
                result = dict(archive_heldout_family=heldout, fold=fold, family=family,
                    d=len(task['X_integer'][0]), held_policy_ids=held, arms=[], comparisons=[])
                for arm in source['arms']:
                    prediction = case.get('predictions', {}).get(arm, {})
                    result['arms'].append(dict(arm=arm, status=prediction.get('status', 'failed'),
                        failures=prediction.get('failures', frozen.get('failure')),
                        saved_observed13_terminal=prediction.get('terminal13'),
                        snapshots={str(prefix): analyze_snapshot(prediction.get('snapshots', {}).get(str(prefix), {}),
                            labels, task['alpha'], held, prefix) for prefix in (10, 13)}))
                by_arm = {row['arm']: row for row in result['arms']}
                for left, right in COMPARISONS:
                    for prefix in (10, 13):
                        result['comparisons'].append(paired_snapshot(by_arm[left]['snapshots'][str(prefix)],
                            by_arm[right]['snapshots'][str(prefix)], left, right))
                cases.append(result)
    completion = read(output / 'completion.json')
    previous = read(root / plan['inputs']['source_contrasts'])
    result = dict(status='complete', registered_source_groups=len(cases), registered_arm_cases=len(cases)*len(source['arms']),
        completed_arm_cases=sum(arm['status'] == 'complete' for case in cases for arm in case['arms']),
        registered_snapshots=len(cases)*len(source['arms'])*2,
        completed_snapshots=sum(s['status'] == 'complete' for case in cases for arm in case['arms']
            for s in arm['snapshots'].values()), cases=cases, aggregates=aggregate_cases(cases, source['arms']),
        family_aggregates=[dict(family=family, **aggregate_cases([case for case in cases if case['family'] == family],
            source['arms'])) for family in sorted({case['family'] for case in cases})],
        source_fit_flags=flags,
        prior_paid_accounting=dict(file=str((output / 'completion.json').relative_to(root)),
            cost=completion['cost'], failed_attempts=completion['failed_attempts'],
            fold_failures=completion.get('fold_failures', []), low_ess_flags=completion.get('source_low_ess_flags', [])),
        reused_general_error_results=dict(file=plan['inputs']['source_contrasts'], status=previous['status'],
            completed_arm_cases=previous['completed_arm_cases'], completed_snapshots=previous['completed_snapshots'],
            scope='Stage61 general objective-difference MSE and raw comparisons retained at their existing path; not recalculated.'),
        definitions=dict(observation_partition='Current prefix: observed10/unobserved6 then observed13/unobserved3.',
            scoring='Original unused two repeats at final13 policies; all three at never-observed3, even when scoring prefix10.',
            eligible='Saved probability_violation <=0.5; paired comparisons use the symmetric intersection of the two arms.',
            empirical_feasibility='Unused g mean + normal quantile times sample standard deviation <=0; not a certificate.',
            near_gap='Stage61 lowest ceil(pair_count/4) absolute unused-label gaps within case/current-prefix/pair-group, before eligibility filtering.',
            relative_optimism='For observed i/unobserved j, (predicted f_i-f_j)-(unused-label f_i-f_j); positive means unobserved j relatively optimistic.',
            selection='Descriptive minimum objective mean within a saved eligibility mask, not actual optimizer acquisition or certified deployment.',
            statistical_unit='Source group first; 24 overlapping archive/fold/family groups, not 24 independent tasks.',
            ties='Exact zero label gaps are excluded from sign error; predicted ties on an untied label count as wrong. Paired changes <=1e-12 are descriptive ties.'),
        limits=['Only saved marginal uncertainty exists; cross-policy covariance, posterior difference variance and ordering confidence remain unknown.',
            'Two or three unused repeats give noisy source labels, not exact objective truth or safety certificates.',
            'd50 source holdouts do not establish d1000 adaptive-selection performance; the source histories are fixed.',
            'No correction or weight sweep is evaluated. Evidence may motivate a later unique change, but this audit does not validate one.'],
        new_work=dict(new_source_calls=0, new_target_calls=0, new_verification_calls=0, new_truth_function_calls=0,
            new_source_fits=0, posterior_restorations=0, raw_replays=0, new_rng_draws=0))
    result['coverage'] = source_coverage(result)
    return result


def source_coverage(result):
    return dict(source_groups=result['registered_source_groups'],
        arm_cases=result['registered_arm_cases'], snapshots=result['registered_snapshots'],
        completed_snapshots=result['completed_snapshots'], failed_arm_cases=[dict(
            archive_heldout_family=case['archive_heldout_family'], fold=case['fold'], family=case['family'],
            arm=arm['arm'], status=arm['status'], failures=arm['failures'])
            for case in result['cases'] for arm in case['arms'] if arm['status'] != 'complete'])


def compact_source(result):
    """All 24 signed paired cases retained; raw pairs and candidate arrays stay remote."""
    compact = {key: value for key, value in result.items() if key not in (
        'cases', 'family_aggregates', 'source_fit_flags', 'aggregates')}
    aliases = {AFTER: 'AFTER', POINT: 'POINT', FULL: 'FULL', NATIVE: 'NATIVE'}
    compact['source_fit_flags_file'] = 'source/analysis.json:source_fit_flags'
    compact['coverage'] = source_coverage(result)
    compact['table_columns'] = dict(
        arm_selection=['prefix', 'status', 'pool', 'eligible_count', 'selected', 'runner_up',
            'predicted_lead', 'unused_label_lead_over_runner_up', 'selected_unused_label_excess',
            'selected_empirically_feasible'],
        paired_group=['pair_group', 'all_common_pairs', 'order_error_delta',
            'gap_weighted_order_error_delta', 'unobserved_relative_optimism_delta',
            'near_common_pairs', 'near_order_error_delta'],
        paired_selection=['pool', 'left_selected', 'right_selected', 'changed',
            'unused_label_objective_delta', 'empirical_feasible_excess_delta',
            'left_predicted_lead', 'right_predicted_lead'])
    compact['aggregates'] = dict(arms=[], comparisons=result['aggregates']['comparisons'])
    for item in result['aggregates']['arms']:
        saved = {key: value for key, value in item.items() if key not in ('groups', 'selections')}
        saved['groups'] = [dict(pair_group=row['pair_group'], scope=row['scope'], subset=row['subset'],
            pairs=row['total_pairs'], metrics={key: dict(available_groups=metric['available_groups'],
                conditional_mean=metric['conditional_mean']) for key, metric in row['metrics'].items()})
            for row in item['groups'] if row['scope'] == 'predicted_eligible']
        saved['selections'] = item['selections']
        compact['aggregates']['arms'].append(saved)
    compact['cases'] = []
    for case in result['cases']:
        saved = {key: case[key] for key in ('archive_heldout_family', 'fold', 'family', 'd', 'held_policy_ids')}
        saved['arms'] = []
        for arm in case['arms']:
            rows = []
            for prefix, snap in arm['snapshots'].items():
                for choice in snap['selections']:
                    if choice['pool'] == 'observed':
                        continue  # Already fully reported by Stage58; retained in source/analysis.
                    rows.append([int(prefix), snap['status']] + [choice[key] for key in (
                        'pool', 'eligible_count', 'selected', 'runner_up', 'predicted_lead',
                        'unused_label_lead_over_runner_up', 'selected_unused_label_excess',
                        'selected_empirically_feasible')])
            saved['arms'].append(dict(arm=aliases[arm['arm']], status=arm['status'],
                failures=arm['failures'], snapshot_status={prefix: snap['status']
                    for prefix, snap in arm['snapshots'].items()}, selections=rows))
        saved['comparisons'] = []
        for comparison in case['comparisons']:
            paired = {key: value for key, value in comparison.items() if key not in ('groups', 'selections')}
            paired.update(left=aliases[comparison['left']], right=aliases[comparison['right']], groups=[], selections=[])
            for group in PAIR_GROUPS:
                rows = {row['subset']: row for row in comparison['groups'] if row['pair_group'] == group}
                if not rows:
                    continue
                regular, near = rows['all'], rows['small_label_gap_quarter']
                paired['groups'].append([group, regular['left']['pairs']] +
                    [regular['delta'][key] for key in PAIR_METRICS] +
                    [near['left']['pairs'], near['delta']['order_error']])
            paired['selections'] = [[row['pool'], row['left']['selected'], row['right']['selected'],
                row['changed'], row['unused_label_objective_delta'], row['empirical_feasible_excess_delta'],
                row['left']['predicted_lead'], row['right']['predicted_lead']] for row in comparison['selections']]
            saved['comparisons'].append(paired)
        compact['cases'].append(saved)
    return compact
