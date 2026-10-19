"""Decompose saved constrained-TS choices without restoring any posterior."""
from collections import Counter
import json

import numpy as np
from scipy.stats import norm

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = ROOT / 'performance/manifests/mainline_role_choice_audit_v1.json'
MODES = {
    'actual': ('sampled_objective', 'sampled_margin'),
    'mean_objective': ('objective_mean', 'sampled_margin'),
    'mean_risk': ('sampled_objective', 'stochastic_margin_mean'),
    'both_means': ('objective_mean', 'stochastic_margin_mean'),
}


def score_choice(objective, margin):
    """The original constrained_ts lexicographic rule, including tie-break."""
    objective, margin = np.asarray(objective), np.asarray(margin)
    feasible = margin <= 0.
    if feasible.any():
        total = np.full(len(margin), -1e300)
        total[feasible] = -objective[feasible]
        mode = 'feasible_objective'
    else:
        total = -margin - np.finfo(float).eps * objective / max(np.max(abs(objective)), 1.)
        mode = 'minimum_violation'
    return int(np.argmax(total)), total, mode, int(feasible.sum())


def point_record(index, stage, points, neighbors, truth, initial, z):
    score = stage['score']
    point_id = stage['action_ids'][index]
    point = tuple(points[point_id])
    actual = truth.get(point)
    fields = ('objective_mean', 'objective_epistemic', 'sampled_objective',
              'constraint_mean', 'constraint_epistemic', 'constraint_aleatoric',
              'sampled_constraint', 'sampled_margin', 'stochastic_margin_mean')
    result = {key: score[key][index] for key in fields}
    sample_f = np.asarray(score['sampled_objective'])
    feasible = np.asarray(score['sampled_margin']) <= 0.
    initial_position = None if initial is None else stage['action_ids'].index(initial['id'])
    result.update(policy_id=point_id, neighbor_index=neighbors.get(point),
        previously_observed=point_id in stage['observed_ids'],
        actual_selected=point_id == stage['selected_id'], truth=actual,
        objective_draw_innovation=result['sampled_objective']-result['objective_mean'],
        constraint_draw_innovation=result['sampled_constraint']-result['constraint_mean'],
        objective_mean_difference_to_initial=(None if initial is None else result['objective_mean']-
            score['objective_mean'][initial_position]),
        sampled_feasible=bool(feasible[index]),
        sampled_objective_rank_within_feasible=(
            int(1 + np.sum(feasible & (sample_f < sample_f[index]))) if feasible[index] else None))
    if actual is not None:
        sigma = np.sqrt(max(result['constraint_aleatoric'], 1e-12))
        result.update(
            feasible_improving=None if initial is None else bool(actual['true_feasible'] and
                actual['objective'] < initial['truth']['objective']-1e-12),
            true_objective_difference_to_initial=None if initial is None else actual['objective']-initial['truth']['objective'],
            objective_mean_bias=result['objective_mean']-actual['objective'],
            risk_mean_bias=result['constraint_mean']-actual['constraint_mean_centered'],
            risk_noise_margin_bias=z*(sigma-actual['constraint_sigma']),
            sigma_ratio=sigma/actual['constraint_sigma'])
        # Detect incorrect units/centering in this diagnostic, not a model quality test.
        decomposed = (result['risk_mean_bias'] + result['risk_noise_margin_bias'] +
                      result['constraint_draw_innovation'])
        if not np.isclose(decomposed, result['sampled_margin']-actual['chance_margin'],
                          rtol=1e-10, atol=1e-10):
            raise ValueError('Saved risk margin cannot be decomposed in the declared units')
    return result


def summarize(plan):
    source = ROOT / plan['input_dir']
    parent = read(ROOT / plan['parent_manifest'])
    z = float(norm.ppf(1-parent['task']['alpha']))
    rows, focal = [], []
    max_score_error = 0.
    for seed in parent['target_seeds']:
        directory = source / f'seed_{seed}'
        truth = {tuple(row['point']): row['truth'] for row in read(directory/'truth.json')}
        with np.load(directory/'neighbors.npz') as data:
            neighbor_points = [tuple(p) for p in data['points']]
        neighbors = {point: i for i, point in enumerate(neighbor_points)}
        initial_point = tuple(read(directory/'initialization.json')['history'][0]['point'])
        for arm in parent['arms']:
            if not arm['online']:
                continue
            case = directory / arm['name']
            audit = read(case/'query_audit.json')
            with np.load(case/'query_policies.npz') as data:
                points = data['points'].copy()
            point_ids = {tuple(point): i for i, point in enumerate(points)}
            initial = dict(id=point_ids[initial_point], truth=truth[initial_point])
            for stage in audit['stages']:
                score = stage['score']
                choices = {}
                for name, (f_key, m_key) in MODES.items():
                    index, total, mode, count = score_choice(score[f_key], score[m_key])
                    if name == 'actual':
                        error = float(np.max(abs(total-np.asarray(score['total']))))
                        max_score_error = max(max_score_error, error)
                        if (index != stage['selected_index'] or
                            stage['action_ids'][index] != stage['selected_id'] or
                            mode != score['sampled_feasibility_mode'] or
                            count != score['sampled_feasible_count'] or error > 1e-12):
                            raise ValueError(f'Score reconstruction mismatch: {seed}/{arm["name"]}')
                    choices[name] = dict(mode=mode, feasible_count=count,
                        selected=point_record(index, stage, points, neighbors, truth, initial, z))
                row = dict(seed=seed, arm=arm['name'], paid_prefix=stage['paid_prefix'],
                    candidate_count=len(stage['action_ids']), choices=choices)
                rows.append(row)
                focus = plan['focus_neighbors'].get(str(seed))
                if focus:
                    details = []
                    for neighbor in dict.fromkeys([neighbors[initial_point], *focus]):
                        point_id = point_ids[neighbor_points[neighbor]]
                        if point_id not in stage['action_ids']:
                            details.append(dict(neighbor_index=neighbor, present=False))
                        else:
                            details.append(dict(present=True, **point_record(
                                stage['action_ids'].index(point_id), stage,
                                points, neighbors, truth, initial, z)))
                    focal.append(dict(seed=seed, arm=arm['name'], paid_prefix=stage['paid_prefix'],
                                      choices=choices, neighbors=details))
    counts = []
    for arm in (a['name'] for a in parent['arms'] if a['online']):
        group = [row for row in rows if row['arm'] == arm]
        for mode in MODES:
            choices = [row['choices'][mode] for row in group]
            known = [c['selected'] for c in choices if c['selected']['truth'] is not None]
            counts.append(dict(arm=arm, substitution=mode, recorded_states=len(group),
                changed_choice=sum(row['choices'][mode]['selected']['policy_id'] !=
                    row['choices']['actual']['selected']['policy_id'] for row in group),
                known_truth_count=len(known), unknown_truth_count=len(group)-len(known),
                known_true_feasible=sum(p['truth']['true_feasible'] for p in known),
                known_feasible_improving=sum(p['feasible_improving'] for p in known),
                selection_modes=dict(Counter(c['mode'] for c in choices))))
    return dict(status='complete', scope=plan['limits'], cases=rows, focal_cases=focal,
                descriptive_counts=counts,
                reconstruction=dict(count=len(rows), max_score_error=max_score_error))


def main():
    plan = read(MANIFEST)
    output = ROOT / plan['output_dir']
    with attempt(output, 'saved-choice-decomposition', 'FactorShockStatePolicyRZDT1') as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0,
                    new_truth_function_calls=0, download_bytes=0)
        result = summarize(plan)
        save(output/'diagnosis.json', result)
    attempts = [json.loads(line) for line in
                (output/'attempts.jsonl').read_text().splitlines()]
    save(output/'completion.json', dict(status='complete', reconstruction=result['reconstruction'],
        attempts=len(attempts), failed_attempts=[r for r in attempts if r['status'] != 'completed'],
        cost={key: sum(row.get(key, 0) for row in attempts) for key in (
            'cpu_seconds', 'wall_seconds', 'new_source_calls', 'new_target_initial_calls',
            'new_target_search_calls', 'new_verification_calls', 'new_source_fits',
            'new_truth_function_calls', 'download_bytes', 'source_model_restorations')},
        startup_issue=plan['startup_issue']))
    plan['status'] = 'complete'
    save(MANIFEST, plan)
    print(result['reconstruction'])
    for row in result['descriptive_counts']:
        if row['arm'] == 'legacy_role_full':
            print(row)


if __name__ == '__main__':
    main()
