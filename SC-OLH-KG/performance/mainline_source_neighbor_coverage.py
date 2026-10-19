"""Score geometry-sealed source pairs without fitting, sampling or reselection."""
from pathlib import Path

import numpy as np
from scipy.stats import norm

from performance.run_mainline_legacy_backend_bridge import read
from performance.summarize_mainline_source_observed_ranking import load_frozen_cases, scoring_labels
from performance.mainline_objective_selection_source import COMPARISONS, PAIR_GROUPS, describe


SCOPES = ('all_fixed_pairs', 'both_predicted_eligible', 'both_empirically_feasible')
METRICS = ('wrong_order_fraction', 'gap_weighted_wrong_order', 'absolute_predicted_gap_mean',
           'mixed_unobserved_relative_optimism_mean')


def unused_repeats(task, held):
    """Same reserved labels as Stage58, including at the initial10 snapshot."""
    raw = np.asarray(task['Y_replicates'], dtype=float)[held]
    return [row[1:] if index < 13 else row for index, row in enumerate(raw)]


def difference_evidence(left, right):
    """Independent source draws: no arbitrary row-aligned CRN differences."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    gap = float(np.mean(left) - np.mean(right))
    variance = [float(np.var(values, ddof=1)) for values in (left, right)]
    mean_variance = variance[0] / len(left) + variance[1] / len(right)
    leave_one = ([float(np.mean(np.delete(left, i)) - np.mean(right)) for i in range(len(left))] +
                 [float(np.mean(left) - np.mean(np.delete(right, i))) for i in range(len(right))])
    signs = [int(np.sign(value)) for value in leave_one]
    return dict(unused_mean_difference=gap, unused_replicates=[len(left), len(right)],
        endpoint_sample_variances=variance,
        independent_episode_difference_variance_estimate=sum(variance),
        independent_mean_difference_standard_error=float(np.sqrt(mean_variance)),
        absolute_gap_over_label_standard_error=(abs(gap) / float(np.sqrt(mean_variance)) if mean_variance > 0 else None),
        crn_paired_difference_variance=None,
        leave_one_endpoint_repeat=dict(differences=leave_one, minimum=min(leave_one), maximum=max(leave_one),
            negative=signs.count(-1), zero=signs.count(0), positive=signs.count(1),
            all_same_sign_as_full=all(sign == int(np.sign(gap)) for sign in signs)))


def score_fixed_pair(pair, labels, repeats, alpha):
    i, j = pair['i'], pair['j']
    margins = labels['mean'][:, 1] + norm.ppf(1-alpha) * np.sqrt(labels['variance'][:, 1])
    evidence = difference_evidence(repeats[i][:, 0], repeats[j][:, 0])
    return dict(pair, **evidence,
        empirical_constraint_margins=[float(margins[i]), float(margins[j])],
        empirical_feasible=[bool(margins[i] <= 0), bool(margins[j] <= 0)],
        empirical_both_feasible=bool(margins[i] <= 0 and margins[j] <= 0))


def predict_fixed_pairs(snapshot, pairs, prefix):
    if snapshot.get('status') != 'complete':
        return dict(prefix=prefix, status=snapshot.get('status', 'missing'), failure=snapshot.get('error'), pairs=[])
    mean = np.asarray(snapshot['mean'], dtype=float)[:, 0]
    probability = np.asarray(snapshot['probability_violation'], dtype=float)
    predictions = []
    for index, pair in enumerate(pairs):
        i, j = pair['i'], pair['j']
        delta = float(mean[i] - mean[j])
        gap = pair['unused_mean_difference']
        group = pair['observed'][str(prefix)]['pair_group']
        predictions.append(dict(pair_index=index, pair_group=group, predicted_difference=delta,
            predicted_tie=bool(delta == 0), wrong_order=None if gap == 0 else bool(np.sign(delta) != np.sign(gap)),
            endpoint_probability_violation=[float(probability[i]), float(probability[j])],
            both_predicted_eligible=bool(probability[i] <= .5 and probability[j] <= .5),
            endpoint_eligible=[bool(probability[i] <= .5), bool(probability[j] <= .5)],
            both_currently_observed=bool(i < prefix and j < prefix),
            observed13_terminal_pair=bool(prefix == 13 and i < 13 and j < 13)))
    return dict(prefix=prefix, status='complete', failure=None, pairs=predictions)


def metrics(rows, pairs):
    untied = [row for row in rows if row['wrong_order'] is not None]
    denominator = sum(abs(pairs[row['pair_index']]['unused_mean_difference']) for row in untied)
    stable = [row for row in untied if pairs[row['pair_index']]['leave_one_endpoint_repeat']['all_same_sign_as_full']]
    mixed = [row for row in rows if row['pair_group'] == 'observed_unobserved']
    return dict(pairs=len(rows), untied_label_pairs=len(untied),
        wrong_order_pairs=sum(row['wrong_order'] for row in untied),
        wrong_order_fraction=(sum(row['wrong_order'] for row in untied) / len(untied) if untied else None),
        gap_weighted_wrong_order=(sum(abs(pairs[row['pair_index']]['unused_mean_difference']) * row['wrong_order']
            for row in untied) / denominator if denominator else None),
        absolute_predicted_gap_mean=float(np.mean([abs(row['predicted_difference']) for row in rows])) if rows else None,
        mixed_unobserved_relative_optimism_mean=(float(np.mean([row['predicted_difference'] -
            pairs[row['pair_index']]['unused_mean_difference'] for row in mixed])) if mixed else None),
        label_loo_unstable_pairs=sum(not pairs[row['pair_index']]['leave_one_endpoint_repeat']['all_same_sign_as_full'] for row in rows),
        stable_untied_label_pairs=len(stable), wrong_on_stable_label_pairs=sum(row['wrong_order'] for row in stable),
        both_empirically_feasible_pairs=sum(pairs[row['pair_index']]['empirical_both_feasible'] for row in rows))


def snapshot_groups(snapshot, pairs):
    if snapshot['status'] != 'complete':
        return []
    result = []
    for group in ('all',) + PAIR_GROUPS:
        for scope in SCOPES:
            rows = [row for row in snapshot['pairs'] if (group == 'all' or row['pair_group'] == group) and
                (scope == 'all_fixed_pairs' or
                 row['both_predicted_eligible'] if scope != 'both_empirically_feasible'
                 else pairs[row['pair_index']]['empirical_both_feasible'])]
            result.append(dict(pair_group=group, scope=scope, metrics=metrics(rows, pairs)))
    return result


def compare_snapshots(left, right, pairs, names):
    result = dict(left=names[0], right=names[1], prefix=left['prefix'],
        status='complete' if left['status'] == right['status'] == 'complete' else 'unavailable', groups=[])
    if result['status'] != 'complete':
        return result
    for group in ('all',) + PAIR_GROUPS:
        for scope in ('all_fixed_pairs', 'common_predicted_eligible', 'both_empirically_feasible'):
            indices = [index for index, (a, b) in enumerate(zip(left['pairs'], right['pairs']))
                if (group == 'all' or a['pair_group'] == group) and
                (scope == 'all_fixed_pairs' or
                 (a['both_predicted_eligible'] and b['both_predicted_eligible']) if scope != 'both_empirically_feasible'
                 else pairs[index]['empirical_both_feasible'])]
            a, b = [metrics([snap['pairs'][index] for index in indices], pairs) for snap in (left, right)]
            result['groups'].append(dict(pair_group=group, scope=scope, left=a, right=b,
                delta={key: None if a[key] is None or b[key] is None else a[key] - b[key] for key in METRICS}))
    return result


def aggregate_cases(cases, arms):
    result = dict(arms=[], comparisons=[])
    for arm in arms:
        for prefix in ('10', '13'):
            snapshots = [next(row for row in case['arms'] if row['arm'] == arm)['snapshots'][prefix] for case in cases]
            groups = []
            for group in ('all',) + PAIR_GROUPS:
                for scope in SCOPES:
                    rows = [next((row['metrics'] for row in snap['groups'] if
                        (row['pair_group'], row['scope']) == (group, scope)), {}) for snap in snapshots]
                    groups.append(dict(pair_group=group, scope=scope,
                        available_groups=sum(bool(row) for row in rows),
                        pair_totals={key: sum(row.get(key, 0) for row in rows) for key in (
                            'pairs', 'untied_label_pairs', 'wrong_order_pairs', 'label_loo_unstable_pairs',
                            'stable_untied_label_pairs', 'wrong_on_stable_label_pairs', 'both_empirically_feasible_pairs')},
                        metrics={key: describe([row.get(key) for row in rows]) for key in METRICS}))
            result['arms'].append(dict(arm=arm, prefix=int(prefix), registered_groups=len(cases),
                completed_groups=sum(snap['status'] == 'complete' for snap in snapshots), groups=groups))
    for names in COMPARISONS:
        for prefix in (10, 13):
            rows = [next(row for row in case['comparisons'] if
                (row['left'], row['right'], row['prefix']) == (*names, prefix)) for case in cases]
            groups = []
            for group in ('all',) + PAIR_GROUPS:
                for scope in ('all_fixed_pairs', 'common_predicted_eligible', 'both_empirically_feasible'):
                    selected = [next((item for item in row['groups'] if
                        (item['pair_group'], item['scope']) == (group, scope)), {}) for row in rows]
                    groups.append(dict(pair_group=group, scope=scope,
                        pairs=sum(row.get('left', {}).get('pairs', 0) for row in selected),
                        delta={key: describe([row.get('delta', {}).get(key) for row in selected]) for key in METRICS}))
            result['comparisons'].append(dict(left=names[0], right=names[1], prefix=prefix,
                registered_groups=len(cases), groups=groups))
    return result


def analyze_source_neighbors(root, plan):
    root = Path(root)
    fixed = read(root / plan['inputs']['fixed_pairs'])
    source = read(root / plan['inputs']['source_manifest'])
    directory = root / source['output_dir']
    records = {(archive['heldout_target_family'], fold): read(directory / archive['heldout_target_family'] /
        f'fold_{fold}' / 'forecasts.json') for archive in source['source_archives'] for fold in range(source['folds'])}
    load_frozen_cases(source, records)
    archives = {archive['heldout_target_family']: {task['name']: task for task in
        read(root / archive['path'])['tasks']} for archive in source['source_archives']}
    cases = []
    for group in fixed['groups']:
        heldout, fold, family = (group[key] for key in ('archive_heldout_family', 'fold', 'family'))
        frozen = records[heldout, fold]
        case = next((row for row in frozen['cases'] if row['family'] == family), {})
        held = group['held_policy_ids']
        if case:
            assert held == case['held_policy_ids']
        task = archives[heldout][family]
        labels, repeats = scoring_labels(task, held), unused_repeats(task, held)
        pairs = [score_fixed_pair(pair, labels, repeats, task['alpha']) for pair in group['pairs']]
        result = {key: group[key] for key in ('archive_heldout_family', 'fold', 'family', 'd', 'held_policy_ids')}
        result.update(pairs=pairs, arms=[], comparisons=[])
        for arm in source['arms']:
            prediction = case.get('predictions', {}).get(arm, {})
            snapshots = {str(prefix): predict_fixed_pairs(prediction.get('snapshots', {}).get(str(prefix), {}),
                pairs, prefix) for prefix in (10, 13)}
            for snapshot in snapshots.values():
                snapshot['groups'] = snapshot_groups(snapshot, pairs)
            result['arms'].append(dict(arm=arm, status=prediction.get('status', 'failed'),
                failures=prediction.get('failures', frozen.get('failure')), snapshots=snapshots))
        by_arm = {row['arm']: row for row in result['arms']}
        for names in COMPARISONS:
            for prefix in ('10', '13'):
                result['comparisons'].append(compare_snapshots(by_arm[names[0]]['snapshots'][prefix],
                    by_arm[names[1]]['snapshots'][prefix], pairs, names))
        cases.append(result)
    completion = read(directory / 'completion.json')
    flags = [dict(archive_heldout_family=heldout, fold=fold,
        diagnostics=read(directory / heldout / f'fold_{fold}' / 'fit_diagnostics.json')) for heldout, fold in records]
    return dict(status='complete', coverage=dict(source_groups=len(cases), arm_cases=len(cases)*len(source['arms']),
        snapshots=len(cases)*len(source['arms'])*2,
        completed_snapshots=sum(snap['status'] == 'complete' for case in cases for arm in case['arms']
            for snap in arm['snapshots'].values()), fixed_pairs=sum(len(case['pairs']) for case in cases),
        failed_arm_cases=[dict(archive_heldout_family=case['archive_heldout_family'], fold=case['fold'],
            family=case['family'], arm=arm['arm'], status=arm['status'], failures=arm['failures'])
            for case in cases for arm in case['arms'] if arm['status'] != 'complete']),
        fixed_pairs_file=plan['inputs']['fixed_pairs'], cases=cases, aggregates=aggregate_cases(cases, source['arms']),
        family_aggregates=[dict(family=family, **aggregate_cases([case for case in cases if case['family'] == family],
            source['arms'])) for family in sorted({case['family'] for case in cases})], source_fit_flags=flags,
        prior_paid_accounting=dict(source_completion=str((directory / 'completion.json').relative_to(root)),
            cost=completion['cost'], failed_attempts=completion['failed_attempts'],
            fold_failures=completion.get('fold_failures', []), low_ess_flags=completion.get('source_low_ess_flags', []),
            historical_accounts=plan.get('historical_accounts', plan['inputs'].get('historical_accounts', []))),
        definitions=dict(pair_selection='Read sealed geometry-only pairs verbatim; labels, margins, noise and eligibility do not select or replace pairs.',
            noise='Sequential independent source draws; episode difference variance is s_i^2+s_j^2 and label mean difference variance is s_i^2/n_i+s_j^2/n_j. No row-aligned CRN pairs exist.',
            noise_provenance=['representation/meta_prior.py:_record_source_data/_source_observation',
                'problems/rzdt.py:TestProblem.simulate', 'problems/single_objective.py:ScalarizedProblem.simulate'],
            labels='Same unused two repeats at held positions0..12 and all three at13..15 as Stage58, at both prefixes.',
            leave_one='Delete one repeat at one endpoint, leave the other endpoint intact. Record all n_i+n_j signs; no stability threshold filters pairs.',
            mixed_unobserved_relative_optimism='For observed i and unobserved j, (predicted f_i-f_j)-(unused-label f_i-f_j); positive means unobserved j is relatively optimistic.',
            empirical_feasibility='Unused constraint mean + normal quantile times sample standard deviation <=0, never a chance-feasibility certificate.',
            eligibility='Saved source terminal probability_violation <=0.5. Applying this criterion to unobserved points is diagnostic; the actual saved terminal13 considered observed13 only.',
            statistical_unit='Within-group pair metrics before descriptive means across24 overlapping archive/fold/family groups; pairs are not independent task replications.'),
        limits=['Two or three source scoring repeats are noisy labels. Leave-one sign stability is descriptive, not an objective-order certificate.',
            'Source marginal moments cannot reconstruct target acquisition sampled margins, cross-policy posterior covariance or sampling-temperature effects.',
            'No source or target means, thresholds, weights, search or terminal rules are changed; no minimal correction is preselected.'],
        new_work=dict(new_source_calls=0, new_target_calls=0, new_verification_calls=0, new_truth_function_calls=0,
            new_source_fits=0, posterior_restorations=0, raw_replays=0, new_rng_draws=0))


def compact_source_neighbors(result):
    compact = {key: value for key, value in result.items() if key not in ('cases', 'family_aggregates', 'source_fit_flags')}
    compact['source_fit_flags_location'] = 'full analysis:source_fit_flags'
    compact['case_table_columns'] = dict(pair=['i', 'j', 'unused_gap', 'label_standard_error',
        'loo_sign_stable', 'both_empirically_feasible'],
        paired=['pair_group', 'scope', 'pairs', 'order_error_delta', 'gap_weighted_error_delta',
            'mixed_unobserved_relative_optimism_delta'])
    compact['family_aggregates'] = [dict(family=family['family'], comparisons=[dict(
        left=item['left'], right=item['right'], prefix=item['prefix'], registered_groups=item['registered_groups'],
        groups=[dict(pair_group=group['pair_group'], scope=group['scope'], pairs=group['pairs'],
            delta={key: {field: metric[field] for field in ('available_groups', 'conditional_mean', 'lower', 'tied', 'higher')}
                for key, metric in group['delta'].items() if key != 'absolute_predicted_gap_mean'})
            for group in item['groups']]) for item in family['comparisons']]) for family in result['family_aggregates']]
    compact['cases'] = []
    for case in result['cases']:
        row = {key: case[key] for key in ('archive_heldout_family', 'fold', 'family', 'held_policy_ids')}
        row['pairs'] = [[pair['i'], pair['j'], pair['unused_mean_difference'],
            pair['independent_mean_difference_standard_error'],
            pair['leave_one_endpoint_repeat']['all_same_sign_as_full'], pair['empirical_both_feasible']] for pair in case['pairs']]
        row['arms'] = [dict(arm=arm['arm'], status=arm['status'], failures=arm['failures'],
            snapshot_status={prefix: snap['status'] for prefix, snap in arm['snapshots'].items()}) for arm in case['arms']]
        row['comparisons'] = [dict(left=item['left'], right=item['right'], prefix=item['prefix'], status=item['status'],
            groups=[[g['pair_group'], g['scope'], g['left']['pairs'], g['delta']['wrong_order_fraction'],
                g['delta']['gap_weighted_wrong_order'], g['delta']['mixed_unobserved_relative_optimism_mean']]
                for g in item['groups']]) for item in case['comparisons']]
        compact['cases'].append(row)
    return compact
