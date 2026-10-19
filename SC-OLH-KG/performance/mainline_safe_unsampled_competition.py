"""Analyze sealed candidate arrays and outcomes; no I/O, models, truth calls or RNG."""
from collections import Counter
from itertools import combinations

import numpy as np

from performance.diagnose_mainline_role_choice import score_choice
from performance.mainline_confirmation_query_attribution import terminal_path
from performance.summarize_mainline_two_comparison_development import METRICS, difference, estimate


AFTER = 'legacy_role_after_init_point_only'
POINT = 'legacy_role_point_only'
FULL = 'legacy_role_full'
ARMS = (AFTER, POINT, FULL)
PREFIXES = (10, 11, 12)
CONTRASTS = (('AP', AFTER, POINT), ('FA', FULL, AFTER))
ORIGINS = ('initial', 'shared_neighbor', 'dynamic')


def key(point):
    return tuple(map(int, point))


def order(predicted, actual):
    if actual == 0:
        return 'true_tie'
    if predicted == 0:
        return 'predicted_tie'
    return 'correct' if predicted * actual > 0 else 'wrong'


def transition(left, right):
    if left == right == 'correct':
        return 'both_correct'
    if left == right == 'wrong':
        return 'both_wrong'
    if left == 'wrong' and right == 'correct':
        return 'left_wrong_right_correct'
    if left == 'correct' and right == 'wrong':
        return 'left_correct_right_wrong'
    return 'tie'


def pair_records(rows):
    """Every unordered truly feasible, currently unobserved pair; no label filtering."""
    safe = [row for row in rows if not row['observed'] and row['true_feasible'] is True]
    pairs = []
    for left, right in combinations(safe, 2):
        actual = left['truth']['objective'] - right['truth']['objective']
        mean = left['objective_mean'] - right['objective_mean']
        sampled = left['sampled_objective'] - right['sampled_objective']
        mo, so = order(mean, actual), order(sampled, actual)
        pairs.append(dict(left=left['policy_id'], right=right['policy_id'],
            origin_pair='+'.join(sorted((left['origin'], right['origin']))),
            both_eligible=left['eligible'] and right['eligible'],
            eligible_endpoints=int(left['eligible']) + int(right['eligible']),
            true_difference=float(actual), mean_difference=float(mean),
            draw_difference=float(sampled-mean), sampled_difference=float(sampled),
            mean_order=mo, sampled_order=so, mean_to_draw=transition(mo, so)))
    return pairs


def pair_summary(pairs):
    denominator = sum(row['true_difference'] != 0 for row in pairs)
    result = dict(pairs=len(pairs), strict_true_pairs=denominator,
        true_ties=len(pairs)-denominator,
        mean_order=dict(Counter(row['mean_order'] for row in pairs)),
        sampled_order=dict(Counter(row['sampled_order'] for row in pairs)),
        mean_to_draw=dict(Counter(row['mean_to_draw'] for row in pairs)))
    for name, field in (('mean', 'mean_order'), ('sampled', 'sampled_order')):
        result[name+'_misorder_rate'] = (sum(row[field] == 'wrong' for row in pairs)
                                         / denominator if denominator else None)
        result[name+'_tie_rate'] = (sum(row[field] == 'predicted_tie' for row in pairs)
                                    / denominator if denominator else None)
    return result


def pair_groups(pairs):
    return dict(all=pair_summary(pairs),
        both_eligible=pair_summary([row for row in pairs if row['both_eligible']]),
        by_origin={origin:pair_summary([row for row in pairs if row['origin_pair'] == origin])
                   for origin in sorted({row['origin_pair'] for row in pairs})})


def risk_counts(rows):
    return {f'{truth}_{eligibility}':sum(
        row['true_feasible'] is value and row['eligible'] == eligible for row in rows)
        for truth, value in (('true_feasible', True), ('true_infeasible', False), ('truth_unknown', None))
        for eligibility, eligible in (('eligible', True), ('excluded', False))}


def coverage(rows, baseline):
    unseen = [row for row in rows if not row['observed']]
    safe = [row for row in unseen if row['true_feasible'] is True]
    competing = [row for row in safe if row['eligible']]
    return dict(candidates=len(rows), observed=len(rows)-len(unseen), unobserved=len(unseen),
        truth_unknown=sum(row['truth'] is None for row in rows),
        true_feasible_unobserved=len(safe), sampled_eligible_unobserved=sum(row['eligible'] for row in unseen),
        safe_eligible_unobserved=len(competing), safe_competition_count=len(competing),
        safe_competition_size='zero' if not competing else 'one' if len(competing) == 1 else 'multiple',
        improving_safe_unobserved=(None if baseline is None else
            sum(row['truth']['objective'] < baseline['objective'] for row in safe)),
        risk_table=risk_counts(rows), unobserved_risk_table=risk_counts(unseen))


def _state(arm, prefix, bundle, stage, truth, identify, baseline, initial_unseen):
    points = [key(point) for point in bundle['points']]
    actions = [points[i] for i in stage['action_ids']]
    observed = {points[i] for i in stage['observed_ids']}
    scores = stage['score']
    chosen, total, mode, count = score_choice(scores['sampled_objective'], scores['sampled_margin'])
    score_error = float(np.max(np.abs(total-np.asarray(scores['total']))))
    selected_index = stage['selected_index']
    history = bundle['history']
    returned = len(history) > prefix
    history_matches = key(history[prefix]['point']) == actions[selected_index] if returned else None
    replay_valid = bool(score_error <= 1e-12 and chosen == selected_index and
        stage['action_ids'][chosen] == stage['selected_id'] and
        mode == scores['sampled_feasibility_mode'] and count == scores['sampled_feasible_count'] and
        history_matches is not False)
    rows = []
    for index, point in enumerate(actions):
        value = truth.get(point)
        rows.append(dict(**identify(point), bank_id=stage['action_ids'][index],
            observed=point in observed, initially_unsampled=(None if initial_unseen is None else point in initial_unseen),
            eligible=bool(scores['sampled_margin'][index] <= 0), selected=index == selected_index,
            true_feasible=None if value is None else bool(value['true_feasible']), truth=value,
            action_source=stage['action_sources'][index],
            **{field:float(scores[field][index]) for field in ('objective_mean', 'objective_epistemic',
                'sampled_objective', 'sampled_margin', 'constraint_mean', 'constraint_aleatoric',
                'constraint_epistemic', 'sampled_constraint')}))
    selected = rows[selected_index]
    safe_omissions = []
    eligible_rows = [candidate for candidate in rows if candidate['eligible']]
    for row in rows:
        row['mean_rank_within_eligible'] = (1+sum(other['objective_mean'] < row['objective_mean']
            for other in eligible_rows)) if row['eligible'] else None
        row['sampled_rank_within_eligible'] = (1+sum(other['sampled_objective'] < row['sampled_objective']
            for other in eligible_rows)) if row['eligible'] else None
        if row['observed'] or row['true_feasible'] is not True or row['selected']:
            continue
        actual = None if selected['truth'] is None else row['truth']['objective']-selected['truth']['objective']
        mean = row['objective_mean']-selected['objective_mean']
        sampled = row['sampled_objective']-selected['sampled_objective']
        reason = ('minimum_violation_competition' if mode == 'minimum_violation' else
                  'risk_excluded' if not row['eligible'] else 'objective_competition')
        cause = ('outside_objective_competition' if reason != 'objective_competition' else
                 'actual_truth_unknown' if actual is None else 'not_true_objective_improvement' if actual >= 0 else
                 'sampled_tie' if sampled == 0 else 'mean_misorder' if mean > 0 else
                 'mean_tie' if mean == 0 else 'draw_reverses_correct_mean')
        safe_omissions.append(dict(policy_id=row['policy_id'], origin=row['origin'],
            reason=reason, objective_cause=cause, actual_true_feasible=selected['true_feasible'],
            true_difference=actual, mean_difference=mean, draw_difference=sampled-mean,
            sampled_difference=sampled))
    safe = [row for row in rows if not row['observed'] and row['true_feasible'] is True and row['eligible']]
    if len(safe) < 2:
        reason = 'zero_safe_eligible_unobserved' if not safe else 'one_safe_eligible_unobserved'
    elif selected['observed']:
        reason = 'selected_previously_observed'
    elif selected['true_feasible'] is None:
        reason = 'selected_truth_unknown'
    elif not selected['true_feasible']:
        reason = 'selected_true_infeasible'
    elif not selected['eligible']:
        reason = 'selected_risk_excluded'
    else:
        reason = 'selected_in_safe_competition'
    regret = (selected['truth']['objective']-min(row['truth']['objective'] for row in safe)
              if reason == 'selected_in_safe_competition' and replay_valid else None)
    pairs = pair_records(rows)
    cohort = None if initial_unseen is None else dict(
        original_count=len(initial_unseen), present=sum(point in actions for point in initial_unseen),
        out_of_pool=sum(point not in actions for point in initial_unseen),
        observed=sum(point in observed for point in initial_unseen),
        present_unobserved=sum(point in actions and point not in observed for point in initial_unseen))
    return dict(arm=arm, paid_prefix=prefix, status='captured', search_status=bundle['status'],
        replay=dict(valid=replay_valid, max_total_error=score_error, mode=mode, sampled_feasible_count=count,
                    observation_returned=returned, returned_policy_matches=history_matches,
                    backend_conditioned=(True if history_matches and bundle['status'] == 'complete' else None)),
        selected=selected, counts=coverage(rows, baseline),
        observed_policy_ids=[identify(point)['policy_id'] for point in sorted(observed)],
        by_origin={origin:coverage([row for row in rows if row['origin'] == origin], baseline) for origin in ORIGINS},
        safe_omission_counts=dict(reason=dict(Counter(row['reason'] for row in safe_omissions)),
            objective_cause=dict(Counter(row['objective_cause'] for row in safe_omissions)),
            both_true_feasible_improvements=dict(Counter(row['objective_cause'] for row in safe_omissions
                if row['actual_true_feasible'] is True and row['true_difference'] is not None and row['true_difference'] < 0))),
        safe_omissions=safe_omissions,
        initial_unobserved_cohort=cohort, safe_competition=dict(status=reason, objective_regret=regret),
        pairs=pair_groups(pairs), safe_unobserved_pairs=pairs, candidates=rows)


def _common(left, right, lb, rb, initial_common):
    lrows = {row['policy_id']:row for row in left['candidates']}
    rrows = {row['policy_id']:row for row in right['candidates']}
    common = [pid for pid in lrows if pid in rrows]
    unseen = [pid for pid in common if not lrows[pid]['observed'] and not rrows[pid]['observed']]

    def eligibility(pid):
        a, b = lrows[pid]['eligible'], rrows[pid]['eligible']
        return 'both_eligible' if a and b else 'left_only_eligible' if a else 'right_only_eligible' if b else 'both_excluded'

    groups = []
    for name in ('both_eligible', 'left_only_eligible', 'right_only_eligible', 'both_excluded'):
        ids = [pid for pid in unseen if eligibility(pid) == name]
        safe = [pid for pid in ids if lrows[pid]['true_feasible'] is True]
        groups.append(dict(group=name, count=len(ids), true_feasible=len(safe),
            truth_unknown=sum(lrows[pid]['true_feasible'] is None for pid in ids),
            origins=dict(Counter(lrows[pid]['origin'] for pid in ids)),
            policy_ids=ids, safe_policy_ids=safe,
            left_pairs=pair_groups(pair_records([lrows[pid] for pid in ids])),
            right_pairs=pair_groups(pair_records([rrows[pid] for pid in ids]))))
    lpairs = pair_records([lrows[pid] for pid in unseen])
    rpairs = pair_records([rrows[pid] for pid in unseen])
    joined = []
    for a, b in zip(lpairs, rpairs, strict=True):
        joined.append(dict(left_policy_id=a['left'], right_policy_id=a['right'],
            origin_pair=a['origin_pair'], endpoint_eligibility=[eligibility(a['left']), eligibility(a['right'])],
            left=a, right=b, mean_transition=transition(a['mean_order'], b['mean_order']),
            sampled_transition=transition(a['sampled_order'], b['sampled_order'])))
    prefix = left['paid_prefix']
    same_history = (lb['history'][:prefix] == rb['history'][:prefix]
                    if min(len(lb['history']), len(rb['history'])) >= prefix else None)
    cohort = None if initial_common is None else dict(original_count=len(initial_common),
        common_present=sum(pid in lrows and pid in rrows for pid in initial_common),
        common_present_unobserved=sum(pid in unseen for pid in initial_common),
        left_observed=sum(pid in left['observed_policy_ids'] for pid in initial_common),
        right_observed=sum(pid in right['observed_policy_ids'] for pid in initial_common),
        left_out_of_pool=sum(pid not in lrows for pid in initial_common),
        right_out_of_pool=sum(pid not in rrows for pid in initial_common))
    return dict(status='available', same_paid_history=same_history,
        same_physical_pool=set(lrows) == set(rrows), common_pool_count=len(common),
        left_only_pool_count=len(lrows)-len(common), right_only_pool_count=len(rrows)-len(common),
        common_unobserved_count=len(unseen), common_unobserved_ids=unseen,
        query_changed=left['selected']['policy_id'] != right['selected']['policy_id'],
        both_observations_returned=left['replay']['observation_returned'] and right['replay']['observation_returned'],
        selected={left['arm']:left['selected']['policy_id'], right['arm']:right['selected']['policy_id']},
        eligibility_groups=groups, initial_common_unobserved_cohort=cohort,
        left_safe_pairs=pair_groups(lpairs), right_safe_pairs=pair_groups(rpairs),
        pair_order_transitions={field:dict(Counter(row[field] for row in joined))
                                for field in ('mean_transition', 'sampled_transition')},
        safe_common_pairs=joined, substitutions=[])


def _swaps(left, right):
    result = []
    for receiver, donor in ((left, right), (right, left)):
        theirs = {row['policy_id']:row for row in donor['candidates']}
        for component in ('risk', 'objective'):
            objective, margin = [], []
            for row in receiver['candidates']:
                other = theirs[row['policy_id']]
                objective.append((other if component == 'objective' else row)['sampled_objective'])
                margin.append((other if component == 'risk' else row)['sampled_margin'])
            index, _, mode, count = score_choice(objective, margin)
            chosen = receiver['candidates'][index]['policy_id']
            result.append(dict(receiver=receiver['arm'], donor=donor['arm'], component=component,
                selected_policy_id=chosen, mode=mode, eligible_count=count,
                donor_choice_recovered=chosen == donor['selected']['policy_id'],
                receiver_choice_changed=chosen != receiver['selected']['policy_id']))
    return result


def _terminal_effects(cases, outcomes, identify, truth, protocols=('old', 'new')):
    indexed, scored = {row['arm']:row for row in cases}, {row['arm']:row for row in outcomes}
    effects = []
    for contrast, left, right in CONTRASTS:
        for protocol in protocols:
            arms = {}
            for arm in (left, right):
                descriptor = indexed.get(arm, {}).get(protocol, {})
                record = descriptor.get('record') if descriptor.get('status') == 'complete' else None
                point = key(record['deployed']) if record is not None else None
                certified = bool(record['verification']['certified']) if record is not None else None
                arms[arm] = dict(status=descriptor.get('status', 'missing'), certified=certified,
                    reported_policy=None if point is None else identify(point),
                    deployed_policy=None if not certified else identify(point),
                    truth=truth.get(point), metrics=scored.get(arm, {}).get(protocol, dict.fromkeys(METRICS)))
            delta = difference(arms[left]['metrics'], arms[right]['metrics'])
            contributions = {}
            for metric in ('C', 'R', 'G', 'P', 'L'):
                values = {origin:0. for origin in (*ORIGINS, 'no_certified_deployment')}
                if delta[metric] is None:
                    values = dict.fromkeys(values)
                else:
                    for arm, sign in ((left, 1.), (right, -1.)):
                        deployed = arms[arm]['deployed_policy']
                        origin = 'no_certified_deployment' if deployed is None else deployed['origin']
                        values[origin] += sign*arms[arm]['metrics'][metric]
                contributions[metric] = values
            effects.append(dict(contrast=contrast, protocol=protocol, left=left, right=right,
                arms=arms, delta=delta, signed_origin_contributions=contributions))
    return effects


def analyze_task(seed, domain, bundles, truth, initial, shared, cases, outcomes, protocols=('old', 'new'),
                 include_substitutions=True):
    """Inputs are already loaded by the sealed scoring adapter; failed arms stay present.

    bundles[arm] contains points/audit/history/status. Partial history contains only
    returned raws; these do not prove a backend update completed. cases and outcomes
    are the existing load_sealed/score_pair rows for this geometry. No input mutates.
    """
    initial_points, shared_points = [key(row['point']) for row in initial], [key(p) for p in shared]
    initial_set, shared_set = set(initial_points), set(shared_points)
    registry = {}

    def identify(point):
        point = key(point)
        if point not in registry:
            registry[point] = f'P{len(registry):04d}'
        return dict(policy_id=registry[point], origin=('initial' if point in initial_set else
            'shared_neighbor' if point in shared_set else 'dynamic'))

    for point in [*initial_points, *shared_points]:
        identify(point)
    baseline = None
    if len(initial) == 10 and all(point in truth for point in initial_points):
        baseline = min((truth[p] for p in initial_points if truth[p]['true_feasible']),
                       key=lambda row:row['objective'], default=None)
    states, indexed, failures = [], {}, []
    for arm in ARMS:
        bundle = bundles.get(arm)
        stages = {} if not bundle or bundle.get('audit') is None else {
            stage['paid_prefix']:stage for stage in bundle['audit']['stages']}
        first = stages.get(10)
        initial_unseen = None if first is None else {
            key(bundle['points'][i]) for i in first['action_ids'] if i not in set(first['observed_ids'])}
        if not bundle or bundle['status'] != 'complete':
            failures.append(dict(arm=arm, status='missing' if not bundle else bundle['status']))
        for prefix in PREFIXES:
            if prefix not in stages:
                state = dict(arm=arm, paid_prefix=prefix, status='unavailable',
                    search_status='missing' if not bundle else bundle['status'])
            else:
                state = _state(arm, prefix, bundle, stages[prefix], truth, identify, baseline, initial_unseen)
            states.append(state)
            indexed[arm, prefix] = state
    comparisons = []
    for name, left, right in CONTRASTS:
        first_divergence, earlier_available, initial_common = None, True, None
        for prefix in PREFIXES:
            a, b = indexed[left, prefix], indexed[right, prefix]
            row = dict(contrast=name, left=left, right=right, paid_prefix=prefix)
            if a['status'] != 'captured' or b['status'] != 'captured':
                comparisons.append(dict(row, status='unavailable'))
                earlier_available = False
                continue
            result = _common(a, b, bundles[left], bundles[right], initial_common)
            if prefix == 10:
                initial_common = result['common_unobserved_ids']
                result['initial_common_unobserved_cohort'] = dict(original_count=len(initial_common),
                    common_present=len(initial_common), common_present_unobserved=len(initial_common),
                    left_observed=0, right_observed=0, left_out_of_pool=0, right_out_of_pool=0)
            first = result['query_changed'] and first_divergence is None
            result['first_recorded_divergence'] = first
            result['first_divergence'] = first if earlier_available else None
            if first:
                first_divergence = prefix
            causal = bool(result['first_divergence'] and result['same_paid_history'] and result['same_physical_pool'])
            result['scope'] = 'same_history_same_pool' if result['same_paid_history'] and result['same_physical_pool'] else 'diverged_trajectory_association'
            if include_substitutions and causal and a['replay']['valid'] and b['replay']['valid']:
                result['substitutions'] = _swaps(a, b)
            comparisons.append(dict(row, **result))
    chains = []
    for state in states:
        if state['status'] != 'captured':
            continue
        selected = state['selected']
        pid = selected['policy_id']
        point = next(point for point, identifier in registry.items() if identifier == pid)
        terminal_paths = []
        for case in cases:
            terminal_protocols = {}
            for protocol in protocols:
                descriptor = case[protocol]
                record = descriptor.get('record') if descriptor['status'] == 'complete' else None
                terminal_protocols[protocol] = terminal_path(point, descriptor.get('item'), record, descriptor['status'])
            terminal_paths.append(dict(arm=case['arm'], history_arm=case.get('history_arm'), protocols=terminal_protocols))
        exposures = []
        for arm in ARMS:
            rows = []
            for prefix in PREFIXES:
                other = indexed[arm, prefix]
                candidate = next((r for r in other.get('candidates', []) if r['policy_id'] == pid), None)
                rows.append(dict(paid_prefix=prefix, status=other['status'], present=candidate is not None,
                    observed=(pid in other['observed_policy_ids'] if other['status'] == 'captured' else None),
                    eligible=None if candidate is None else candidate['eligible'],
                    selected=None if candidate is None else candidate['selected']))
            history = (bundles.get(arm) or {}).get('history')
            exposures.append(dict(arm=arm, states=rows, returned_observation_indices=None if history is None else
                [i for i, record in enumerate(history) if key(record['point']) == point]))
        chains.append(dict(arm=state['arm'], paid_prefix=state['paid_prefix'], policy_id=pid,
            origin=selected['origin'], truth=selected['truth'], previously_observed=selected['observed'],
            eligible=selected['eligible'], observation_returned=state['replay']['observation_returned'],
            returned_policy_matches=state['replay']['returned_policy_matches'],
            backend_conditioned=state['replay']['backend_conditioned'], exposures=exposures, terminals=terminal_paths))
    terminal_effects = _terminal_effects(cases, outcomes, identify, truth, protocols=protocols)
    captured = [row for row in states if row['status'] == 'captured']
    invalid = sum(not row['replay']['valid'] for row in captured)
    return dict(seed=seed, domain=domain, status='invalid_replay' if invalid else 'partial' if failures or len(captured) < 9 else 'complete',
        coverage=dict(registered_states=9, captured_states=len(captured), unavailable_states=9-len(captured),
            invalid_replays=invalid, returned_query_observations=sum(row['replay']['observation_returned'] for row in captured),
            new_returned_policy_queries=sum(row['replay']['observation_returned'] and not row['selected']['observed'] for row in captured),
            registered_query_arms=3, quality_baseline_defined=baseline is not None,
            truth_policies=len(truth), failures=failures),
        states=states, comparisons=comparisons, query_chains=chains, terminal_effects=terminal_effects,
        policies=[dict(**identify(point), point=list(point)) for point in registry],
        limits=['Only actual saved pools and draws are evaluated; no counterfactual trajectory is generated.',
            'Safe-pair counts are within-geometry repeated diagnostics, not independent samples.',
            'Returned raw does not prove a partial-search backend update completed.',
            'Terminal metrics are per-arm outcomes, not additive gains for every query in a deployment chain.'])


def compact_task(raw):
    result = {key:value for key, value in raw.items() if key not in ('policies', 'states', 'comparisons')}
    result['states'] = [{key:value for key, value in row.items() if key not in ('candidates', 'safe_unobserved_pairs', 'safe_omissions')}
                        for row in raw['states']]
    result['comparisons'] = []
    for row in raw['comparisons']:
        reduced = {key:value for key, value in row.items() if key not in ('safe_common_pairs', 'common_unobserved_ids', 'eligibility_groups')}
        if 'eligibility_groups' in row:
            reduced['eligibility_groups'] = [{key:value for key, value in group.items()
                if key not in ('policy_ids', 'safe_policy_ids')} for group in row['eligibility_groups']]
        result['comparisons'].append(reduced)
    return result


def aggregate_tasks(tasks, protocols=('old', 'new')):
    """Average each geometry's pair rate; never pool pairs as independent evidence."""
    groups = []
    for arm in ARMS:
        for prefix in PREFIXES:
            states = [next(row for row in task['states'] if row['arm'] == arm and row['paid_prefix'] == prefix) for task in tasks]
            available = [row for row in states if row['status'] == 'captured']
            counts = Counter()
            risk, full_risk = Counter(), Counter()
            for row in available:
                counts.update({key:value for key, value in row['counts'].items()
                               if isinstance(value, int) and not isinstance(value, bool)})
                risk.update(row['counts']['unobserved_risk_table'])
                full_risk.update(row['counts']['risk_table'])
            pair_rates = {subset:{metric:estimate([None if row['status'] != 'captured' else
                row['pairs'][subset][metric] for row in states])
                for metric in ('mean_misorder_rate', 'sampled_misorder_rate', 'mean_tie_rate', 'sampled_tie_rate')}
                for subset in ('all', 'both_eligible')}
            groups.append(dict(arm=arm, paid_prefix=prefix, registered_geometries=len(tasks),
                available_geometries=len(available), counts=dict(counts), risk_table=dict(full_risk), unobserved_risk_table=dict(risk),
                selected_queries=dict(captured=len(available),
                    returned=sum(row['replay']['observation_returned'] for row in available),
                    previously_observed=sum(row['selected']['observed'] for row in available),
                    new_returned=sum(row['replay']['observation_returned'] and not row['selected']['observed'] for row in available)),
                safe_omission_counts={field:dict(sum((Counter(row['safe_omission_counts'][field]) for row in available), Counter()))
                    for field in ('reason', 'objective_cause', 'both_true_feasible_improvements')},
                known_initial_baseline_geometries=sum(row['counts']['improving_safe_unobserved'] is not None for row in available),
                by_origin={origin:dict(
                    candidates=sum(row['by_origin'][origin]['candidates'] for row in available),
                    unobserved=sum(row['by_origin'][origin]['unobserved'] for row in available),
                    true_feasible_unobserved=sum(row['by_origin'][origin]['true_feasible_unobserved'] for row in available),
                    safe_eligible_unobserved=sum(row['by_origin'][origin]['safe_eligible_unobserved'] for row in available))
                    for origin in ORIGINS},
                geometry_pair_rates_by_origin={origin:{metric:estimate([
                    None if row['status'] != 'captured' else row['pairs']['by_origin'].get(origin, {}).get(metric)
                    for row in states]) for metric in ('mean_misorder_rate', 'sampled_misorder_rate')}
                    for origin in sorted({origin for row in available for origin in row['pairs']['by_origin']})},
                safe_competition_sizes=dict(Counter(row['counts']['safe_competition_size'] for row in available)),
                safe_competition_dispositions=dict(Counter(row['safe_competition']['status'] for row in available)),
                geometry_pair_rates=pair_rates,
                selected_safe_competition_regret=estimate([None if row['status'] != 'captured' else
                    row['safe_competition']['objective_regret'] for row in states])))
    compared = []
    for contrast, _, _ in CONTRASTS:
        rows = [(task['seed'], row) for task in tasks for row in task['comparisons'] if row['contrast'] == contrast]
        available = [(seed, row) for seed, row in rows if row['status'] == 'available']
        changed = [(seed, row) for seed, row in available if row['query_changed']]
        first = [(seed, row) for seed, row in available if row['first_divergence']]
        prefix_groups = []
        for prefix in PREFIXES:
            selected = [row for _, row in available if row['paid_prefix'] == prefix]
            eligibility = []
            for name in ('both_eligible', 'left_only_eligible', 'right_only_eligible', 'both_excluded'):
                cells = [next(group for group in row['eligibility_groups'] if group['group'] == name) for row in selected]
                eligibility.append(dict(group=name, candidates=sum(cell['count'] for cell in cells),
                    true_feasible=sum(cell['true_feasible'] for cell in cells),
                    truth_unknown=sum(cell['truth_unknown'] for cell in cells),
                    geometries_with_safe_pair=sum(cell['true_feasible'] >= 2 for cell in cells)))
            prefix_groups.append(dict(paid_prefix=prefix, available_geometries=len(selected),
                same_history_same_pool_geometries=sum(row['same_paid_history'] is True and row['same_physical_pool'] for row in selected),
                eligibility_groups=eligibility,
                pair_order_transitions={field:dict(sum((Counter(row['pair_order_transitions'][field]) for row in selected), Counter()))
                    for field in ('mean_transition', 'sampled_transition')}))
        compared.append(dict(contrast=contrast, registered_states=3*len(tasks), available_states=len(available),
            changed_captured_choices=len(changed),
            changed_returned_queries=sum(row['both_observations_returned'] for _, row in changed),
            changed_task_seeds=sorted({seed for seed, row in changed}),
            prefix_groups=prefix_groups,
            first_divergences=[dict(seed=seed, paid_prefix=row['paid_prefix'], scope=row['scope'],
                substitutions=row['substitutions']) for seed, row in first],
            missing_first_divergence_tasks=sorted({seed for seed, row in rows if row['status'] == 'unavailable'})))
    effects = []
    for contrast, _, _ in CONTRASTS:
        for protocol in protocols:
            rows = [next(row for row in task['terminal_effects'] if row['contrast'] == contrast and row['protocol'] == protocol) for task in tasks]
            effects.append(dict(contrast=contrast, protocol=protocol,
                metrics={metric:estimate([row['delta'][metric] for row in rows]) for metric in METRICS},
                signed_origin_contributions={metric:{origin:estimate([
                    row['signed_origin_contributions'][metric][origin] for row in rows])
                    for origin in (*ORIGINS, 'no_certified_deployment')} for metric in ('C', 'R', 'G', 'P', 'L')}))
    return dict(coverage=dict(geometries=len(tasks), registered_states=9*len(tasks),
        captured_states=sum(task['coverage']['captured_states'] for task in tasks),
        invalid_replays=sum(task['coverage']['invalid_replays'] for task in tasks),
        task_statuses=dict(Counter(task['status'] for task in tasks))),
        state_groups=groups, comparisons=compared, terminal_effects=effects,
        per_task=[dict(seed=task['seed'], domain=task['domain'], status=task['status'], coverage=task['coverage'],
            terminal_effects=[dict(contrast=row['contrast'], protocol=row['protocol'], delta=row['delta'])
                             for row in task['terminal_effects']]) for task in tasks],
        statistical_unit='Registered task geometry. Pair rates are computed within each geometry before summaries.')
