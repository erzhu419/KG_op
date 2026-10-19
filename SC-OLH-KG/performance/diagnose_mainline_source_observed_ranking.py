"""Exploratory source-only substitution of frozen objective and risk rankings."""
import argparse

import numpy as np

from acquisition.decision_backends import feasible_first_terminal_order
from performance import summarize_mainline_source_observed_ranking as scoring
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

CONTRASTS = [('legacy_role_after_init_point_only', 'legacy_native'),
             ('legacy_role_full', 'legacy_role_after_init_point_only')]
METRICS = ('choice_changed', 'objective_changes_right_choice', 'probability_changes_right_choice',
           'joint_only_choice_change', 'empirical_objective_change', 'empirical_margin_change',
           'empirical_eligibility_change', 'empirical_feasible_regret_change')


def crossed_choice(forecast_f, forecast_p, labels, task, held):
    snapshot = dict(mean=forecast_f['mean'], probability_violation=forecast_p['probability_violation'])
    objective = np.asarray(snapshot['mean'], dtype=float)[:13, 0]
    probability = np.asarray(snapshot['probability_violation'], dtype=float)[:13]
    order, mode = feasible_first_terminal_order(objective, probability, maximum_violation_probability=.5)
    terminal = dict(selected_index=int(order[0]), eligible_indices=np.flatnonzero(probability <= .5).tolist())
    metrics = scoring.observed_ranking_metrics(snapshot, terminal, labels, task['alpha'])
    return dict(metrics, selected_policy_id=int(held[int(order[0])]),
        selected_empirical_objective=float(labels['mean'][int(order[0]), 0]),
        selected_predicted_objective=float(objective[int(order[0])]),
        selected_probability_violation=float(probability[int(order[0])]),
        eligible_indices=terminal['eligible_indices'], ordered_indices=order.tolist(), mode=mode)


def effect(left, right):
    objective = left['selected_empirical_objective']-right['selected_empirical_objective']
    regret = [record['selected_empirical_feasible_regret'] for record in (left, right)]
    eligible = [record['selected_empirically_eligible'] for record in (left, right)]
    return dict(empirical_objective_change=objective,
        empirical_margin_change=left['selected_empirical_margin']-right['selected_empirical_margin'],
        empirical_eligibility_change=eligible[0]-eligible[1],
        empirical_feasible_regret_change=None if None in regret else regret[0]-regret[1],
        objective_direction='lower' if objective < 0 else 'higher' if objective > 0 else 'same',
        eligibility_transition=f'{eligible[1]}->{eligible[0]}',
        both_empirically_eligible=bool(all(eligible)))


def diagnose_case(case, task, left, right):
    result = dict({key: case[key] for key in ('archive_heldout_family', 'fold', 'family')},
        left=left, right=right, status='failed', metrics=dict.fromkeys(METRICS),
        grid=None, effects=None, choice_channels=None, failures=[])
    predictions = case.get('predictions', {})
    snapshots = {arm: predictions.get(arm, {}).get('snapshots', {}).get('13') for arm in (left, right)}
    for arm, snapshot in snapshots.items():
        if snapshot is None or snapshot['status'] != 'complete':
            result['failures'].append(dict(arm=arm, reason=predictions.get(arm, {}).get(
                'failures', case.get('failure', 'Final frozen snapshot missing'))))
    if result['failures']:
        return result
    labels = scoring.scoring_labels(task, case['held_policy_ids'])
    sides = {'left': left, 'right': right}
    grid = {f'{f}_f_{p}_p': crossed_choice(snapshots[sides[f]], snapshots[sides[p]],
        labels, task, case['held_policy_ids']) for f in sides for p in sides}
    for side, arm in sides.items():
        saved = predictions[arm].get('terminal13')
        if saved is not None:
            assert grid[f'{side}_f_{side}_p']['selected_held_position'] == saved['selected_index']
    base, final = grid['right_f_right_p'], grid['left_f_left_p']
    f_changed = grid['left_f_right_p']['selected_held_position'] != base['selected_held_position']
    p_changed = grid['right_f_left_p']['selected_held_position'] != base['selected_held_position']
    changed = final['selected_held_position'] != base['selected_held_position']
    effects = {key: effect(value, base) for key, value in grid.items()}
    channels = dict(changed_at_right_reference=[name for name, value in
        [('objective', f_changed), ('probability', p_changed)] if value],
        joint_only=bool(changed and not f_changed and not p_changed),
        changed_substitutions_but_same_final=bool(not changed and (f_changed or p_changed)))
    measured = dict(choice_changed=int(changed), objective_changes_right_choice=int(f_changed),
        probability_changes_right_choice=int(p_changed), joint_only_choice_change=int(channels['joint_only']),
        **{key: effects['left_f_left_p'][key] for key in METRICS[4:]})
    return dict(result, status='complete', metrics=measured, grid=grid, effects=effects, choice_channels=channels)


def run(plan):
    output = ROOT/plan['output_dir']
    with attempt(output, 'frozen-source-terminal-components') as cost:
        cost.update(new_source_fits=0, new_target_initial_calls=0, new_truth_function_calls=0,
                    posterior_restorations=0, raw_replays=0, new_rng_draws=0, download_bytes=0)
        records = scoring.load_frozen_cases(plan)
        summary = read(output/'summary.json')
        expected = sum(len(a['source_families']) for a in plan['source_archives'])*plan['folds']
        assert summary['registered_source_cases'] == expected
        rows = []
        for archive in plan['source_archives']:
            name = archive['heldout_target_family']
            tasks = {task['name']: task for task in read(ROOT/archive['path'])['tasks']}
            for fold in range(plan['folds']):
                frozen = records[name, fold]
                cases = {case['family']: case for case in frozen['cases']}
                for family in archive['source_families']:
                    case = dict(cases.get(family, dict(failure=frozen.get('failure'), predictions={})),
                                archive_heldout_family=name, fold=fold, family=family)
                    rows.extend(diagnose_case(case, tasks[family], left, right) for left, right in CONTRASTS)
        aggregates = []
        for left, right in CONTRASTS:
            matched = [row for row in rows if (row['left'], row['right']) == (left, right)]
            aggregates.append(dict(left=left, right=right, **scoring.aggregate(matched, METRICS),
                by_family=[dict(family=family, **scoring.aggregate(
                    [row for row in matched if row['family'] == family], METRICS))
                    for family in dict.fromkeys(row['family'] for row in matched)]))
        result = dict(status='complete' if all(row['status'] == 'complete' for row in rows)
            else 'complete_with_missing_forecasts', registered_contrast_cases=len(rows), rows=rows,
            aggregate=aggregates, score_status=summary['status'], statistical_unit=plan['statistical_unit'],
            substitutions='Each 2x2 combines one frozen objective vector and one saved violation-probability vector; observed13 only.',
            effect_reference='Every grid effect subtracts right_f_right_p. Negative objective change means lower held-repeat mean; eligibility is reported separately.',
            limits=['Exploratory static source substitution, not a coherent refitted posterior or new optimizer.',
                'Probability substitution combines risk mean, noise and uncertainty; it does not separate those three channels.',
                'Two-repeat empirical eligibility is noisy and is not true chance feasibility or an independent certificate.',
                'Source archive/fold cases overlap; no target labels, target precision changes or patch selection are used.'])
        save(output/'terminal_components.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(scoring.MANIFEST))
    run(read(parser.parse_args().manifest))
