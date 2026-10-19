"""Score frozen source-only forecasts against unused archive repeats."""
from itertools import combinations

import numpy as np
from scipy.stats import norm

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = ROOT/'performance/manifests/mainline_source_observed_ranking_v1.json'
GROUPS = {'observed13': np.arange(13), 'unobserved3': np.arange(13, 16)}
HEADS = {'f': (0, 0), 'g': (2, 1)}
PREDICTION_FIELDS = tuple(head+'_'+suffix for head in HEADS for suffix in (
    'mse', 'bias', 'predicted_epistemic_mean', 'predicted_noise_mean', 'empirical_noise_mean',
    'noise_mse', 'noise_mean_ratio', 'mean_prediction_variance_with_label_noise')) + (
    'relative_f_mse', 'relative_f_direction_agreement', 'relative_f_pairs', 'relative_f_untied_label_pairs')
RANK_FIELDS = ('predicted_eligible_count', 'predicted_no_eligible', 'selected_empirical_margin',
    'selected_empirically_eligible', 'empirical_eligible_count', 'empirical_eligibility_agreement',
    'selected_objective_excess_vs_empirical_feasible_best', 'selected_empirical_feasible_regret',
    'selected_objective_excess_vs_empirical_unconstrained_best')
PROPAGATION_FIELDS = tuple('unobserved3_'+head+'_'+suffix for head in HEADS
                          for suffix in ('mse_change_13_minus_10', 'mean_rms_change_13_minus_10'))
METRICS = tuple(f'prefix{prefix}_{group}_{field}' for prefix in (10, 13)
                for group in GROUPS for field in PREDICTION_FIELDS) + RANK_FIELDS + PROPAGATION_FIELDS


def scoring_labels(task, held_policy_ids):
    """Separate labels from the first repeats used for the fixed 13-point history."""
    held = list(held_policy_ids)
    assert len(held) == len(set(held)) == 16
    raw = np.asarray(task['Y_replicates'], dtype=float)[held]
    assert raw.shape == (16, 3, 2)
    repeats = [values[1:] if i < 13 else values for i, values in enumerate(raw)]
    mean = np.asarray([values.mean(axis=0) for values in repeats])
    mean[:, 1] -= task['tau']
    variance = np.asarray([values.var(axis=0, ddof=1) for values in repeats])
    return dict(mean=mean, variance=variance,
                replicate_counts=np.asarray([len(values) for values in repeats]))


def relative_ordering(predicted, observed):
    pairs = list(combinations(range(len(predicted)), 2))
    p = np.asarray([predicted[i]-predicted[j] for i, j in pairs])
    y = np.asarray([observed[i]-observed[j] for i, j in pairs])
    untied = y != 0
    return dict(relative_f_mse=float(np.mean((p-y)**2)),
        relative_f_direction_agreement=(float(np.mean(np.sign(p[untied]) == np.sign(y[untied])))
                                        if untied.any() else None),
        relative_f_pairs=len(pairs), relative_f_untied_label_pairs=int(untied.sum()))


def prediction_metrics(snapshot, labels, indices):
    mean = np.asarray(snapshot['mean'], dtype=float)
    epistemic = np.asarray(snapshot['variance'], dtype=float)
    noise = np.asarray(snapshot['noise'], dtype=float)
    assert mean.shape == epistemic.shape == (16, 4) and noise.shape == (16, 2)
    result = {}
    for head, (column, observation_column) in HEADS.items():
        error = mean[indices, column]-labels['mean'][indices, observation_column]
        empirical_noise = labels['variance'][indices, observation_column]
        predicted_noise = noise[indices, observation_column]
        result.update({head+'_mse': float(np.mean(error**2)),
            head+'_bias': float(error.mean()),
            head+'_predicted_epistemic_mean': float(epistemic[indices, column].mean()),
            head+'_predicted_noise_mean': float(predicted_noise.mean()),
            head+'_empirical_noise_mean': float(empirical_noise.mean()),
            head+'_noise_mse': float(np.mean((predicted_noise-empirical_noise)**2)),
            head+'_noise_mean_ratio': (float(predicted_noise.mean()/empirical_noise.mean())
                                      if empirical_noise.mean() > 0 else None),
            head+'_mean_prediction_variance_with_label_noise': float(np.mean(
                epistemic[indices, column] + empirical_noise/labels['replicate_counts'][indices]))})
    result.update(relative_ordering(mean[indices, 0], labels['mean'][indices, 0]))
    return result


def observed_ranking_metrics(snapshot, terminal, labels, alpha):
    probability = np.asarray(snapshot['probability_violation'], dtype=float)[:13]
    objective = np.asarray(snapshot['mean'], dtype=float)[:13, 0]
    eligible = probability <= .5
    selected = int(terminal['selected_index'])
    assert 0 <= selected < 13
    assert terminal['eligible_indices'] == np.flatnonzero(eligible).tolist()
    if eligible.any():
        assert eligible[selected] and objective[selected] == objective[eligible].min()
    empirical_margin = (labels['mean'][:13, 1] +
                        norm.ppf(1-alpha)*np.sqrt(labels['variance'][:13, 1]))
    empirical_eligible = empirical_margin <= 0
    feasible_best = (float(labels['mean'][:13, 0][empirical_eligible].min())
                     if empirical_eligible.any() else None)
    excess = None if feasible_best is None else float(labels['mean'][selected, 0]-feasible_best)
    return dict(predicted_eligible_count=int(eligible.sum()),
        predicted_no_eligible=int(not eligible.any()),
        selected_held_position=selected,
        selected_empirical_margin=float(empirical_margin[selected]),
        selected_empirically_eligible=int(empirical_eligible[selected]),
        empirical_eligible_count=int(empirical_eligible.sum()),
        empirical_eligibility_agreement=float(np.mean(eligible == empirical_eligible)),
        selected_objective_excess_vs_empirical_feasible_best=excess,
        selected_empirical_feasible_regret=(excess if empirical_eligible[selected] else None),
        selected_objective_excess_vs_empirical_unconstrained_best=float(
            labels['mean'][selected, 0]-labels['mean'][:13, 0].min()))


def aggregate(rows, fields):
    """Keep the registered denominator; conditional summaries are explicitly separate."""
    metrics = {}
    for field in fields:
        values = [row['metrics'].get(field) for row in rows]
        known = [value for value in values if value is not None]
        metrics[field] = dict(registered_cases=len(rows), available_cases=len(known),
            mean=float(np.mean(known)) if len(known) == len(rows) and known else None,
            conditional_available_mean=float(np.mean(known)) if known else None)
    return dict(registered_cases=len(rows),
        completed_cases=sum(row['status'] == 'complete' for row in rows), metrics=metrics)


def load_frozen_cases(plan, case_records=None):
    """All 12 forecast or failure records must exist before any archive-label read."""
    expected = {(archive['heldout_target_family'], fold)
                for archive in plan['source_archives'] for fold in range(plan['folds'])}
    if case_records is None:
        output = ROOT/plan['output_dir']
        records = {key: read(output/key[0]/f'fold_{key[1]}'/'forecasts.json')
                   for key in sorted(expected)}
    else:
        records = dict(case_records)
    assert set(records) == expected
    for (archive, fold), record in records.items():
        assert record['archive_heldout_family'] == archive and record['fold'] == fold
        assert record['status'] in ('complete', 'complete_with_forecast_failures', 'fold_failed')
        expected_families = next(a['source_families'] for a in plan['source_archives']
                                 if a['heldout_target_family'] == archive)
        assert len({case['family'] for case in record['cases']}) == len(record['cases'])
        assert {case['family'] for case in record['cases']} <= set(expected_families)
        if record['status'] == 'fold_failed':
            assert record.get('failure')
        else:
            assert len(record['cases']) == len(expected_families)
        for case in record['cases']:
            assert set(case['predictions']) == set(plan['arms'])
            for arm in plan['arms']:
                prediction = case['predictions'][arm]
                assert prediction['status'] in ('complete', 'failed', 'partial_failure')
                if prediction['status'] == 'complete':
                    assert set(prediction['snapshots']) == {'10', '13'}
                    assert all(s['status'] == 'complete' for s in prediction['snapshots'].values())
                    assert prediction['terminal13'] is not None
                else:
                    assert prediction.get('failures')
                    assert all(s['status'] in ('complete', 'failed')
                               for s in prediction.get('snapshots', {}).values())
    return records


def score_case(case, task, arms):
    """Score one frozen family/fold; failed arms retain null metric slots."""
    identity = {key: case[key] for key in ('archive_heldout_family', 'fold', 'family')}
    labels = None if case.get('status') == 'fold_failed' else scoring_labels(task, case['held_policy_ids'])
    rows = []
    for arm in arms:
        prediction = case.get('predictions', {}).get(arm)
        if prediction is None:
            rows.append(dict(identity, arm=arm, status='failed',
                failures=case.get('failure'),
                metrics=dict.fromkeys(METRICS), terminal=None))
            continue
        metrics = dict.fromkeys(METRICS)
        snapshots = {prefix: snapshot for prefix, snapshot in prediction.get('snapshots', {}).items()
                     if snapshot['status'] == 'complete'}
        for prefix in (10, 13):
            if str(prefix) not in snapshots:
                continue
            for group, indices in GROUPS.items():
                measured = prediction_metrics(snapshots[str(prefix)], labels, indices)
                metrics.update({f'prefix{prefix}_{group}_{key}': value for key, value in measured.items()})
        terminal = None
        if '13' in snapshots and prediction.get('terminal13') is not None:
            terminal = observed_ranking_metrics(snapshots['13'], prediction['terminal13'], labels, task['alpha'])
            metrics.update({key: terminal[key] for key in RANK_FIELDS})
            terminal.update(frozen_ranking=prediction['terminal13'],
                            objective_means=np.asarray(snapshots['13']['mean'], dtype=float)[:13, 0].tolist())
        if set(snapshots) == {'10', '13'}:
            for head, (column, _) in HEADS.items():
                before, after = [np.asarray(snapshots[str(prefix)]['mean'], dtype=float)[13:, column]
                                 for prefix in (10, 13)]
                metrics['unobserved3_'+head+'_mse_change_13_minus_10'] = (
                    metrics[f'prefix13_unobserved3_{head}_mse']-metrics[f'prefix10_unobserved3_{head}_mse'])
                metrics['unobserved3_'+head+'_mean_rms_change_13_minus_10'] = float(
                    np.sqrt(np.mean((after-before)**2)))
        assert set(metrics) == set(METRICS)
        rows.append(dict(identity, arm=arm, status=prediction['status'],
            failures=prediction.get('failures', []), metrics=metrics, terminal=terminal,
            independent_label_replicate_counts=labels['replicate_counts'].tolist()))
    return rows


def comparison(rows, left, right):
    indexed = {(r['archive_heldout_family'], r['fold'], r['family'], r['arm']): r for r in rows}
    effects = []
    for row in rows:
        if row['arm'] != left:
            continue
        other = indexed[row['archive_heldout_family'], row['fold'], row['family'], right]
        terminal_pair = row['terminal'] is not None and other['terminal'] is not None
        ordering = None
        if terminal_pair:
            a, b = row['terminal'], other['terminal']
            eligible_a = set(a['frozen_ranking']['eligible_indices'])
            eligible_b = set(b['frozen_ranking']['eligible_indices'])
            fa, fb = np.asarray(a['objective_means']), np.asarray(b['objective_means'])
            ordering = dict(eligibility_disagreements=len(eligible_a ^ eligible_b),
                objective_pair_direction_changes=sum(
                    np.sign(fa[i]-fa[j]) != np.sign(fb[i]-fb[j]) for i, j in combinations(range(13), 2)),
                objective_pairs=78)
        effects.append(dict({key: row[key] for key in ('archive_heldout_family', 'fold', 'family')},
            status='complete' if row['status'] == other['status'] == 'complete' else 'failed',
            metrics={field: None if row['metrics'][field] is None or other['metrics'][field] is None
                     else row['metrics'][field]-other['metrics'][field] for field in METRICS},
            selected_policy_changed=(None if row['terminal'] is None or other['terminal'] is None
                else row['terminal']['selected_held_position'] != other['terminal']['selected_held_position']),
            predicted_ordering_difference=ordering))
    changes = [r['selected_policy_changed'] for r in effects if r['selected_policy_changed'] is not None]
    return dict(left=left, right=right, effects=effects, **aggregate(effects, METRICS),
        selected_policy_changes=dict(registered_cases=len(effects), available_cases=len(changes),
                                     changed=int(sum(changes))),
        by_family=[dict(family=family, **aggregate([r for r in effects if r['family'] == family], METRICS))
                   for family in dict.fromkeys(r['family'] for r in effects)],
        by_archive=[dict(archive_heldout_family=archive,
            **aggregate([r for r in effects if r['archive_heldout_family'] == archive], METRICS))
            for archive in dict.fromkeys(r['archive_heldout_family'] for r in effects)])


def score(plan, case_records=None):
    output = ROOT/plan['output_dir']
    with attempt(output, 'score-frozen-observed-ranking') as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0, new_truth_function_calls=0,
                    posterior_restorations=0, raw_replays=0, new_rng_draws=0, download_bytes=0)
        records = load_frozen_cases(plan, case_records)
        # No held score labels may be read before every available forecast/failure is sealed.
        archives = {archive['heldout_target_family']: read(ROOT/archive['path'])
                    for archive in plan['source_archives']}
        rows = []
        for archive in plan['source_archives']:
            name = archive['heldout_target_family']
            tasks = {task['name']: task for task in archives[name]['tasks']}
            for fold in range(plan['folds']):
                frozen = records[name, fold]
                cases = {case['family']: case for case in frozen['cases']}
                for family in archive['source_families']:
                    case = cases.get(family, dict(status='fold_failed', family=family,
                        failure=frozen.get('failure'), predictions={}))
                    case = dict(case, archive_heldout_family=name, fold=fold)
                    rows.extend(score_case(case, tasks[family], plan['arms']))
        expected = sum(len(a['source_families']) for a in plan['source_archives'])*plan['folds']
        assert len(rows) == expected*len(plan['arms'])
        complete = sum(row['status'] == 'complete' for row in rows)
        arms = []
        for arm in plan['arms']:
            selected = [row for row in rows if row['arm'] == arm]
            arms.append(dict(arm=arm, **aggregate(selected, METRICS),
                by_family=[dict(family=family,
                    **aggregate([r for r in selected if r['family'] == family], METRICS))
                    for family in dict.fromkeys(r['family'] for r in selected)],
                by_archive=[dict(archive_heldout_family=archive,
                    **aggregate([r for r in selected if r['archive_heldout_family'] == archive], METRICS))
                    for archive in dict.fromkeys(r['archive_heldout_family'] for r in selected)]))
        full, after, point, native = ('legacy_role_full', 'legacy_role_after_init_point_only',
                                      'legacy_role_point_only', 'legacy_native')
        contrasts = [comparison(rows, left, right) for left, right in
                     ((after, point), (full, after), (after, native), (full, native))]
        result = dict(status='complete' if complete == len(rows) else 'complete_with_forecast_failures',
            registered_source_cases=expected, registered_arm_cases=len(rows), completed_arm_cases=complete,
            rows=rows, arms=arms, comparisons=contrasts,
            failures=[{k: row[k] for k in ('archive_heldout_family', 'fold', 'family', 'arm', 'failures')}
                      for row in rows if row['status'] != 'complete'],
            source=plan['source_archives'], statistical_unit=plan['statistical_unit'],
            label_definition='Observed13 use raw repeats1,2; never-observed3 use0,1,2. '
                'g labels are centered by archived tau; noise labels are unbiased sample variances.',
            constraint_moment_semantics={
                'legacy_native': 'g mean is the centered native decision-risk input, including each '
                    'expert pilot constraint guard before mixing. Its epistemic variance includes '
                    'within-expert variance and between-expert variance of these guarded means. '
                    'Native g MSE/bias therefore score guarded decision-risk inputs against raw '
                    'held g labels; they are not unguarded latent-response calibration errors.',
                'role_arms': 'g mean/variance are the centered role latent-response moments. '
                    'Terminal eligibility uses the separately frozen decision probability.',
                'noise': 'Both arms report expected aleatoric raw-response variance; the pilot '
                    'mean guard is not added to the noise estimate.',
                'probability': 'Native probability remains the actual expert-mixture terminal '
                    'probability, not a Gaussian reconstructed from averaged moments. No guard '
                    'is removed and no decision input is changed for scoring.'},
            ranking_definition='Frozen probability-of-violation<=0.5 eligibility and observed13 '
                'objective rank/fallback; empirical margin is held gmean+z*sample_std. No certificate.',
            metric_direction='Lower MSE, regret and signed after-minus-before MSE change are better; '
                'higher pairwise direction and eligibility agreement are better. Empirical regret '
                'requires selected empirical eligibility; objective excess alone may be negative for '
                'an empirically ineligible selection.',
            missing='Every registered arm-case retained. Any missing/undefined metric keeps its '
                'full-group mean null; conditional_available_mean reports only its stated count.',
            limits=['Independent scoring repeats are noisy labels, not latent truth or chance certificates.',
                'These reused archives/folds provide source development evidence, not fresh source '
                'confirmation; scoring repeats are unused by this adaptation history, not globally unseen.',
                'Two/three-repeat noise and empirical feasibility estimates are imprecise.',
                'Native guarded g decision moments and role latent g moments have the distinct '
                'semantics recorded above; their MSE comparison does not isolate response calibration.',
                'Observed13 prediction at prefix10 includes three policies not yet observed at that prefix.',
                'Overlapping source archives, folds and families are not independent task replicates.',
                'Fixed-history d50 source diagnostics do not identify adaptive d1000 deployment gains.',
                'No alternative model, threshold or policy selector is selected by this scorer.'])
        save(output/'summary.json', result)
    return result


if __name__ == '__main__':
    score(read(MANIFEST))
