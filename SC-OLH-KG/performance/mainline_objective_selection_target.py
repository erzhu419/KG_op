"""Fixed-risk objective ordering from saved scores, with no model or random calls."""
from collections import Counter
from pathlib import Path

import numpy as np

from performance.mainline_three_stream_query_attribution import ARMS, ARM_LABEL
from performance.run_mainline_legacy_backend_bridge import read

AFTER = 'legacy_role_after_init_point_only'


def policy_union(saved):
    """Actual exposure defines the focus set; a reported fallback is not deployment."""
    union, uncertified = {}, []
    for stream in saved['terminal']['streams']:
        for arm in stream['terminal']['arms']:
            base = dict(stream=stream['name'], arm=arm['arm'])
            for query in arm['adaptive_queries']:
                union.setdefault(query['policy_id'], []).append(dict(
                    **base, kind='query', prefix=query['history_index']))
            for name, protocol in arm['protocols'].items():
                if protocol['status'] != 'complete':
                    continue
                if protocol['certified']:
                    union.setdefault(protocol['deployed_policy_id'], []).append(dict(
                        **base, kind='certified_deployment', protocol=name))
                else:
                    uncertified.append(dict(**base, protocol=name,
                        policy_id=protocol['reported_policy_id']))
    return union, uncertified


def pair_record(candidate, reference):
    """Differences are candidate minus reference; lower objective is better."""
    if reference is None:
        return dict(status='reference_not_in_saved_pool')
    mean = candidate['objective_mean'] - reference['objective_mean']
    sampled = candidate['sampled_objective'] - reference['sampled_objective']
    ct, rt = candidate['truth'], reference['truth']
    true = None if ct is None or rt is None else ct['objective'] - rt['objective']
    fixed_eligible = candidate['risk_eligible'] and reference['risk_eligible']
    if candidate['bank_id'] == reference['bank_id']:
        ordering = 'same_policy'
    elif not fixed_eligible:
        ordering = 'outside_fixed_risk_set'
    elif mean < 0 and sampled > 0:
        ordering = 'draw_reverses_mean_preference'
    elif mean > 0 and sampled < 0:
        ordering = 'draw_reverses_mean_dispreference'
    elif mean == 0 or sampled == 0:
        ordering = 'mean_or_sampled_tie'
    elif mean < 0:
        ordering = 'mean_and_draw_prefer_candidate'
    else:
        ordering = 'mean_and_draw_disfavor_candidate'
    truth_order = ('unknown' if true is None else 'same_policy'
        if candidate['bank_id'] == reference['bank_id'] else 'true_objective_tie'
        if true == 0 else 'mean_wrong_order' if mean * true < 0 else
        'mean_tie' if mean == 0 else 'mean_correct_order')
    if true is None:
        transition = 'unknown'
    elif true == 0 or mean == 0 or sampled == 0:
        transition = 'mean_or_draw_tie'
    elif mean * true > 0:
        transition = 'both_correct' if sampled * true > 0 else 'mean_correct_draw_wrong'
    else:
        transition = 'mean_wrong_draw_correct' if sampled * true > 0 else 'both_wrong'
    return dict(status='available', reference_policy_id=reference['policy_id'],
        reference_bank_id=reference['bank_id'], both_risk_eligible=bool(fixed_eligible),
        mean_difference=float(mean), draw_difference=float(sampled-mean),
        sampled_difference=float(sampled), true_objective_difference=true,
        ordering=ordering, cached_truth_mean_order=truth_order,
        cached_truth_order_transition=transition,
        candidate_true_feasible=None if ct is None else ct['true_feasible'],
        reference_true_feasible=None if rt is None else rt['true_feasible'])


def analyze_state(stage, bank, history, initial, shared, registry, truth,
                  union_ids, expected_policy_id, after_policy_id):
    """Keep actual sampled risk eligibility fixed; inspect mean and saved draw ranks."""
    prefix = stage['paid_prefix']
    actions = [bank[i] for i in stage['action_ids']]
    observed = {tuple(row['point']) for row in history[:prefix]}
    score = stage['score']
    mean = np.asarray(score['objective_mean'])
    sampled = np.asarray(score['sampled_objective'])
    margin = np.asarray(score['sampled_margin'])
    eligible = margin <= 0
    actual_index = stage['selected_index']
    actual_point = actions[actual_index]
    if (stage['action_ids'][actual_index] != stage['selected_id'] or
            registry.get(actual_point) != expected_policy_id or
            actual_point != tuple(history[prefix]['point'])):
        raise ValueError('Saved score, Stage68 identity and observed query disagree')
    indices = np.flatnonzero(eligible)
    mean_index = int(indices[np.argmin(mean[indices])]) if len(indices) else None
    mode = 'feasible_objective' if len(indices) else 'minimum_violation'
    if mode != score['sampled_feasibility_mode']:
        raise ValueError('Saved risk eligibility and selection mode disagree')
    if len(indices) and int(indices[np.argmin(sampled[indices])]) != actual_index:
        raise ValueError('Saved sampled objective does not recover its eligible winner')

    def detail(j):
        point = actions[j]
        cached = truth.get(point)
        return dict(policy_id=registry.get(point), bank_id=stage['action_ids'][j],
            origin=('initial' if point in initial else 'shared_neighbor'
                    if point in shared else 'dynamic'),
            previously_observed=point in observed, risk_eligible=bool(eligible[j]),
            sampled_margin=float(margin[j]), actual_selected=j == actual_index,
            mean_selected=j == mean_index, objective_mean=float(mean[j]),
            sampled_objective=float(sampled[j]),
            objective_draw=float(sampled[j]-mean[j]),
            objective_epistemic=float(score['objective_epistemic'][j]),
            mean_rank=(int(1 + np.sum(eligible & (mean < mean[j]))) if eligible[j] else None),
            sampled_rank=(int(1 + np.sum(eligible & (sampled < sampled[j]))) if eligible[j] else None),
            truth=(None if cached is None else {k: cached[k]
                for k in ('objective', 'true_feasible', 'quality')}))

    details = [detail(j) for j in range(len(actions))]
    actual = details[actual_index]
    after = next((r for r in details if after_policy_id is not None and
                  r['policy_id'] == after_policy_id), None)
    indexed = {r['policy_id']: r for r in details if r['policy_id'] in union_ids}
    focused = []
    for identifier in union_ids:
        candidate = indexed.get(identifier)
        focused.append(dict(policy_id=identifier, present=False) if candidate is None else
            dict(present=True, **candidate, versus_actual=pair_record(candidate, actual),
                 versus_after_deployment=pair_record(candidate, after)))
    groups = []
    for origin in ('initial', 'shared_neighbor', 'dynamic'):
        for was_observed in (True, False):
            rows = [r for r in details if r['origin'] == origin and
                    r['previously_observed'] == was_observed]
            omitted = [r for r in rows if r['risk_eligible'] and not r['actual_selected']]
            pairs = [pair_record(r, actual) for r in omitted]
            better = [p for p in pairs if p['true_objective_difference'] is not None and
                p['true_objective_difference'] < 0 and p['candidate_true_feasible'] is True]
            better_causes = Counter('mean_misorders' if p['mean_difference'] > 0 else
                'mean_tie' if p['mean_difference'] == 0 else 'sampled_tie' if
                p['sampled_difference'] == 0 else 'draw_reverses_correct_mean' for p in better)
            groups.append(dict(origin=origin, previously_observed=was_observed,
                candidates=len(rows), eligible=sum(r['risk_eligible'] for r in rows),
                cached_truth=sum(r['truth'] is not None for r in rows),
                actual_selected=sum(r['actual_selected'] for r in rows),
                mean_selected=sum(r['mean_selected'] for r in rows),
                eligible_omitted=len(omitted),
                ordering=dict(Counter(p['ordering'] for p in pairs)),
                cached_truth_mean_order=dict(Counter(p['cached_truth_mean_order'] for p in pairs)),
                cached_truth_order_transition=dict(Counter(
                    p['cached_truth_order_transition'] for p in pairs)),
                cached_true_better_eligible_omissions=len(better),
                cached_better_feasible_omission_by_cause=dict(better_causes)))
    return dict(prefix=prefix, selection_mode=mode, actual=actual,
        mean_choice=details[mean_index] if mean_index is not None else None,
        mean_choice_versus_actual=(pair_record(details[mean_index], actual)
                                  if mean_index is not None else None),
        mean_choice_status='available' if mean_index is not None else 'not_applicable_no_eligible_policy',
        mean_choice_changed=None if mean_index is None else mean_index != actual_index,
        actual_repeated_query=actual_point in observed,
        counts=dict(candidates=len(actions), eligible=int(eligible.sum()),
            cached_truth=sum(r['truth'] is not None for r in details),
            union_present=len(indexed), union_absent=len(union_ids)-len(indexed)),
        groups=groups, union_candidates=focused)


def analyze_task(root, plan, seed):
    root = Path(root)
    prior = root / plan['attribution_output_dir'] / 'cases' / f'seed_{seed}'
    saved = read(prior / 'compact.json')
    registry = {tuple(r['point']): r['policy_id'] for r in read(prior / 'policy_vectors.json')}
    union, uncertified = policy_union(saved)
    contexts, truth, truth_sources = [], {}, []
    for descriptor in plan['streams']:
        cases = {r['arm']: r for r in read(root / descriptor['source_output_dir'] /
            'execution_status.json')['cases'] if r['seed'] == seed and r['arm'] in ARMS}
        directory = (root / cases[AFTER]['source_case']).parent
        cache = directory / 'truth.json'
        rows = read(cache)
        for row in rows:
            truth.setdefault(tuple(row['point']), row['truth'])
        truth_sources.append(dict(stream=descriptor['name'], file=str(cache.relative_to(root)),
                                  policies=len(rows)))
        contexts.append((descriptor, cases, directory))
    streams = []
    for descriptor, cases, directory in contexts:
        name = descriptor['name']
        old_query = next(s['query'] for s in saved['query']['streams'] if s['name'] == name)
        terminals = next(s['terminal'] for s in saved['terminal']['streams'] if s['name'] == name)
        after = next(a for a in terminals['arms'] if a['arm'] == AFTER)['protocols']['new']
        after_id = after['deployed_policy_id'] if after['certified'] else None
        initial = {tuple(r['point']) for r in read(directory / 'initialization.json')['history']}
        with np.load(directory / 'neighbors.npz') as data:
            shared = {tuple(map(int, point)) for point in data['points']}
        states = []
        for arm in ARMS:
            path = root / cases[arm]['source_case']
            with np.load(path / 'query_policies.npz') as data:
                bank = [tuple(map(int, point)) for point in data['points']]
            history = read(path / 'search.json')['history']
            for stage in read(path / 'query_audit.json')['stages']:
                actual = next(q for q in old_query['actual_queries']
                    if q['arm'] == ARM_LABEL[arm] and q['prefix'] == stage['paid_prefix'])
                state = analyze_state(stage, bank, history, initial, shared, registry, truth,
                    union, actual['physical_policy_id'], after_id)
                states.append(dict(arm=ARM_LABEL[arm], **state))
        streams.append(dict(name=name, states=states,
            first_divergences_reused=old_query['comparisons'],
            after_new_deployment=dict(policy_id=after_id, certified=after['certified']),
            recorded_outcomes=[{k: a[k] for k in ('arm', 'status', 'old', 'new', 'new_minus_old')}
                               for a in terminals['arms']],
            reconstruction=dict(recorded_choices=len(states), stage68_selection_locations_reused=True)))
    return dict(status='complete', seed=seed, domain=saved['query']['domain'], streams=streams,
        policy_union=[dict(policy_id=pid, exposures=exposures) for pid, exposures in union.items()],
        uncertified_reports=uncertified,
        coverage=dict(cached_physical_policies=len(truth), cache_sources=truth_sources,
            union_policies=len(union), union_cached_truth=sum(
                point in truth for point, pid in registry.items() if pid in union)),
        limits=['Actual risk eligibility remains fixed; no eligible point means no objective-only replacement.',
            'The actual query and certified deployment union is selected by realized exposure, not a full-pool truth oracle.',
            'Only existing truth from this geometry across its three streams is merged by exact physical policy.',
            'Cached objective ranks are descriptive; no candidate, query, safety or deployment is recomputed.',
            'Draw means the saved innovation, not a new draw or a pairwise posterior variance estimate.',
            'Ranks count strictly smaller objectives; the actual candidate order breaks equal-score ties.',
            'The joint ordering tie category includes a zero cached true-objective difference.',
            'First-divergence locations and component substitutions are copied from Stage68, not rerun.',
            'Mean replacement and pair differences are fixed-state diagnostics, not counterfactual trajectories.'])


def compact_task(result):
    """No vectors or raw observations are present; retain every state and union path."""
    return result


def aggregate_tasks(compact_tasks):
    rows = []
    names = list(dict.fromkeys(s['name'] for t in compact_tasks for s in t['streams']))
    for name in names:
        for arm in ARM_LABEL.values():
            states = [(task['seed'], state) for task in compact_tasks
                for stream in task['streams'] if stream['name'] == name
                for state in stream['states'] if state['arm'] == arm]
            eligible = [s for _, s in states if s['mean_choice'] is not None]
            strata = []
            for origin in ('initial', 'shared_neighbor', 'dynamic'):
                for was_observed in (True, False):
                    groups = [g for _, s in states for g in s['groups']
                        if g['origin'] == origin and g['previously_observed'] == was_observed]
                    counts = {key: sum(g[key] for g in groups) for key in (
                        'candidates', 'eligible', 'cached_truth', 'actual_selected', 'mean_selected',
                        'eligible_omitted', 'cached_true_better_eligible_omissions')}
                    for field in ('ordering', 'cached_truth_mean_order', 'cached_truth_order_transition',
                                  'cached_better_feasible_omission_by_cause'):
                        counter = Counter()
                        for group in groups:
                            counter.update(group[field])
                        counts[field] = dict(counter)
                    strata.append(dict(origin=origin, previously_observed=was_observed, **counts))
            winner_truth = Counter()
            for _, state in states:
                pair = state['mean_choice_versus_actual']
                if pair is None:
                    category = 'not_applicable_no_eligible_policy'
                elif pair['true_objective_difference'] is None:
                    category = 'unknown_truth'
                elif not (pair['candidate_true_feasible'] and pair['reference_true_feasible']):
                    category = 'true_infeasible_winner'
                elif pair['true_objective_difference'] > 0:
                    category = 'both_feasible_draw_repairs'
                elif pair['true_objective_difference'] < 0:
                    category = 'both_feasible_draw_harms'
                else:
                    category = 'both_feasible_true_tie'
                winner_truth[category] += 1
            rows.append(dict(stream=name, arm=arm, geometries=len({seed for seed, _ in states}),
                choices=len(states), eligible_choices=len(eligible),
                no_eligible_choices=len(states)-len(eligible),
                mean_choice_changed=sum(s['mean_choice_changed'] for s in eligible),
                tasks_with_mean_choice_change=sorted({seed for seed, s in states
                    if s['mean_choice_changed'] is True}),
                repeated_actual_queries=sum(s['actual_repeated_query'] for _, s in states),
                actual_origin=dict(Counter(s['actual']['origin'] for _, s in states)),
                mean_origin=dict(Counter(s['mean_choice']['origin'] for s in eligible)),
                candidate_exposures=sum(s['counts']['candidates'] for _, s in states),
                cached_truth_exposures=sum(s['counts']['cached_truth'] for _, s in states),
                fixed_state_winner_truth=dict(winner_truth),
                strata=strata))
    task_counts = []
    for task in compact_tasks:
        counts = []
        for stream in task['streams']:
            for arm in ARM_LABEL.values():
                selected = [s for s in stream['states'] if s['arm'] == arm]
                counts.append(dict(stream=stream['name'], arm=arm, states=len(selected),
                    no_eligible=sum(s['mean_choice'] is None for s in selected),
                    mean_choice_changed=sum(s['mean_choice_changed'] is True for s in selected),
                    repeated_queries=sum(s['actual_repeated_query'] for s in selected),
                    actual_policy_ids=[s['actual']['policy_id'] for s in selected],
                    mean_policy_ids=[None if s['mean_choice'] is None else
                        s['mean_choice']['policy_id'] for s in selected]))
        task_counts.append(dict(seed=task['seed'], domain=task['domain'], counts=counts,
            state_details=f'cases/seed_{task["seed"]}/compact.json'))
    choices = sum(r['choices'] for r in rows)
    return dict(geometries=len(compact_tasks), registered_choices=choices,
        coverage=dict(geometries=len(compact_tasks), states=choices,
            cached_truth_scope='Existing physical-policy cache union within each geometry only.'),
        stream_roles=rows, task_coverage=[dict(seed=t['seed'], **t['coverage']) for t in compact_tasks],
        task_counts=task_counts,
        interpretation='24 geometry units; repeated candidate exposures and three streams are descriptive counts, not independent samples.')
