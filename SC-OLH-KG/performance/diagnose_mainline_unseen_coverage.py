"""Explain saved certification coverage without restoring models or evaluating truth."""
from collections import Counter
import json

import numpy as np
from scipy.stats import norm

from performance.diagnose_mainline_role_choice import score_choice
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = ROOT/'performance/manifests/mainline_unseen_geometry_attribution_v1.json'
AFTER = 'legacy_role_after_init_point_only'
OLD = 'legacy_native_neighbors'
STRUCT = 'current_structural_mixture'
STAGE = 'saved-unseen-certification-coverage'
FOCUS = {2026103002: (OLD, 21), 2026103009: (OLD, 52), 2026103011: (STRUCT, 57)}


def labels(point, initial, neighbors):
    if point in initial:
        return f'initial:{initial.index(point)}'
    return f'neighbor:{neighbors[point]}' if point in neighbors else 'dynamic'


def focal_arm(directory, arm, target, initial, neighbors, truth, z):
    case = directory/arm
    history = read(case/'search.json')['history']
    points = [tuple(p) for p in np.load(case/'query_policies.npz')['points']]
    audit = read(case/'query_audit.json')
    stages, bundles = [], []
    for stage in audit['stages']:
        score = stage['score']
        actions = [points[i] for i in stage['action_ids']]
        index, total, mode, count = score_choice(score['sampled_objective'], score['sampled_margin'])
        # Detect misaligned saved arrays or policy IDs before interpreting rankings.
        assert np.max(abs(total-np.asarray(score['total']))) <= 1e-12
        assert index == stage['selected_index']
        assert stage['action_ids'][index] == stage['selected_id']
        assert actions[index] == tuple(history[stage['paid_prefix']]['point'])
        actual = dict(policy=labels(actions[index], initial, neighbors),
                      truth=truth.get(actions[index]))
        record = dict(paid_prefix=stage['paid_prefix'], candidate_count=len(actions),
            selected=actual, target_present=target in actions,
            selection_mode=mode, sampled_feasible_count=count)
        if target in actions:
            j = actions.index(target)
            margin = np.asarray(score['sampled_margin'])
            objective = np.asarray(score['sampled_objective'])
            means = np.asarray(score['objective_mean'])
            eligible = margin <= 0
            values = {k: score[k][j] for k in ('objective_mean', 'objective_epistemic',
                'sampled_objective', 'constraint_mean', 'constraint_epistemic',
                'constraint_aleatoric', 'sampled_constraint', 'stochastic_margin_mean', 'sampled_margin')}
            risk_bias = values['constraint_mean']-truth[target]['constraint_mean_centered']
            noise_bias = z*(np.sqrt(max(values['constraint_aleatoric'], 1e-12))-
                           truth[target]['constraint_sigma'])
            draw = values['sampled_constraint']-values['constraint_mean']
            assert abs(risk_bias+noise_bias+draw-values['sampled_margin']+
                       truth[target]['chance_margin']) < 1e-10
            reason = ('selected' if j == index else 'minimum_violation_competition'
                      if mode == 'minimum_violation' else 'risk_excluded'
                      if not eligible[j] else 'objective_competition')
            mean_index, _, _, _ = score_choice(means, margin)
            values.update(previously_observed=stage['action_ids'][j] in stage['observed_ids'],
                sampled_risk_eligible=bool(eligible[j]), selection_reason=reason,
                objective_mean_rank_all=int(1+np.sum(means < means[j])),
                sampled_objective_rank_all=int(1+np.sum(objective < objective[j])),
                sampled_objective_rank_within_eligible=(
                    int(1+np.sum(eligible & (objective < objective[j]))) if eligible[j] else None),
                objective_mean_rank_within_eligible=(
                    int(1+np.sum(eligible & (means < means[j]))) if eligible[j] else None),
                objective_draw=values['sampled_objective']-values['objective_mean'],
                risk_mean_bias=risk_bias, noise_margin_bias=noise_bias, risk_draw=draw,
                mean_objective_same_saved_risk_selects_target=mean_index == j)
            record['target'] = values
        stages.append(record)
        bundles.append(dict(prefix=stage['paid_prefix'], actions=actions, selected=actions[index]))
    return dict(arm=arm, queries=[dict(policy=labels(tuple(h['point']), initial, neighbors),
        truth=truth.get(tuple(h['point']))) for h in history[10:]], stages=stages), bundles, history


def diagnose(plan):
    source = ROOT/plan['source_output_dir']
    result = read(source/'results.json')
    indexed = {(r['seed'], r['arm']): r for r in result['rows']}
    prepared = read(ROOT/plan['prepared_file'])
    coverage, focal = [], []
    for replica in prepared['replicates']:
        for family in replica['families']:
            search_plan = read(ROOT/family['search_manifest'])
            z = float(norm.ppf(1-search_plan['task']['alpha']))
            for seed in search_plan['target_seeds']:
                directory = ROOT/search_plan['output_dir']/f'seed_{seed}'
                truth = {tuple(r['point']): r['truth'] for r in read(directory/'truth.json')}
                initial = [tuple(r['point']) for r in read(directory/'initialization.json')['history']]
                points = [tuple(p) for p in np.load(directory/'neighbors.npz')['points']]
                neighbors = {p: i for i, p in enumerate(points)}
                arm_rows = []
                for arm in (AFTER, OLD, STRUCT):
                    search = read(directory/arm/'search.json')
                    queries = list(dict.fromkeys(tuple(h['point']) for h in search['history'][10:]))
                    scored = indexed[seed, arm]
                    arm_rows.append(dict(arm=arm, certification=scored['certification'],
                        reward=scored['reward'], quality_reference_status=scored['quality_reference_status'],
                        deployed=labels(tuple(scored['deployed']), initial, neighbors),
                        unique_new_queries=sum(p not in initial for p in queries),
                        true_feasible_new_queries=sum(p not in initial and truth[p]['true_feasible']
                                                      for p in queries),
                        queries=[dict(policy=labels(tuple(h['point']), initial, neighbors),
                                      true_feasible=truth[tuple(h['point'])]['true_feasible'])
                                 for h in search['history'][10:]]))
                coverage.append(dict(seed=seed, domain=family['domain'],
                    initial_feasible=sum(truth[p]['true_feasible'] for p in initial),
                    shared_feasible=sum(truth[p]['true_feasible'] for p in neighbors), arms=arm_rows))
                if seed not in FOCUS:
                    continue
                comparator, neighbor = FOCUS[seed]
                target = points[neighbor]
                assert tuple(indexed[seed, comparator]['deployed']) == target
                records, bundles, histories = [], {}, {}
                for arm in (AFTER, comparator):
                    record, bundles[arm], histories[arm] = focal_arm(
                        directory, arm, target, initial, neighbors, truth, z)
                    records.append(record)
                paired, first = [], None
                for left, right in zip(bundles[AFTER], bundles[comparator]):
                    assert left['prefix'] == right['prefix']
                    prefix = left['prefix']
                    changed = left['selected'] != right['selected']
                    if changed and first is None:
                        first = prefix
                    paired.append(dict(paid_prefix=prefix, query_changed=changed,
                        first_divergence=changed and prefix == first,
                        same_paid_history=histories[AFTER][:prefix] == histories[comparator][:prefix],
                        same_physical_pool=set(left['actions']) == set(right['actions'])))
                focal.append(dict(seed=seed, domain=family['domain'], comparator=comparator,
                    target_policy=f'neighbor:{neighbor}', target_truth=truth[target],
                    arms=records, paired_states=paired))
    assert len(coverage) == 24
    comparisons = []
    for comparator in (OLD, STRUCT):
        pairs = [(indexed[r['seed'], AFTER]['certification'],
                  indexed[r['seed'], comparator]['certification']) for r in coverage]
        comparisons.append(dict(left=AFTER, right=comparator,
            paired_certification_counts={f'after_{a}_reference_{b}': n
                                         for (a, b), n in Counter(pairs).items()}))
    return dict(status='complete', registered_geometry_count=24, coverage=coverage,
        certification_comparisons=comparisons, focal_cases=focal,
        reconstructed_choices=sum(len(arm['stages']) for case in focal for arm in case['arms']),
        limits=[
            'All truth values are cached; no model restoration, fitting, simulation or new truth.',
            'Shared-neighbor counts and presence use physical policies; dynamic labels are origin '
            'categories and do not imply equal policies across arms.',
            'Observed risk and objective scores identify saved selection competition. Mean-objective '
            'choices hold the saved risk draw and pool fixed and are not new trajectories.',
            'After first differing query, history differences prevent attributing scores solely '
            'to the backend. Even matched first states compare entire backends, not a single prior parameter.',
            'No complete final paid13 role score arrays were saved; prequery12 cannot replace them.',
            'Target cases identify mechanisms, not source-data support for a new feasibility prior. '
            'No source refit or new prior/terminal selection is justified by these cases alone.',
        ])


def main():
    plan = read(MANIFEST)
    output = ROOT/plan['output_dir']
    with attempt(output, STAGE) as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0, new_truth_function_calls=0,
                    posterior_restorations=0, raw_replays=0, new_rng_draws=0, download_bytes=0)
        result = diagnose(plan)
        save(output/'coverage_diagnosis.json', result)
    attempts = [json.loads(line) for line in (output/'attempts.jsonl').read_text().splitlines()
                if json.loads(line)['stage'] == STAGE]
    fields = ('cpu_seconds', 'wall_seconds', 'new_source_calls', 'new_target_initial_calls',
              'new_target_search_calls', 'new_verification_calls', 'new_source_fits',
              'new_truth_function_calls', 'source_model_restorations', 'posterior_restorations',
              'raw_replays', 'new_rng_draws', 'download_bytes')
    save(output/'coverage_completion.json', dict(status='complete', attempts=len(attempts),
        cost={k:sum(r.get(k, 0) for r in attempts) for k in fields},
        failed_attempts=[r for r in attempts if r['status'] != 'completed']))
    print(json.dumps(dict(reconstructed_choices=result['reconstructed_choices'],
                         comparisons=result['certification_comparisons'])))


if __name__ == '__main__':
    main()
