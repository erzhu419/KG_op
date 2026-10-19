"""Connect three completed streams without recomputing predictions or choices.

The mechanism files already contain full candidate truth and saved rankings.
Discovery categories concern the first returned query; subsequent resampling is
an orthogonal observation, not an additional or counterfactual source of gain.
"""
from copy import deepcopy
from pathlib import Path

from performance.mainline_decision_path_attribution import METRICS, point_key, read
from performance.mainline_frozen_propagation_paths import AFTER, POINT, FULL, metric_difference


ROLES = (AFTER, POINT, FULL)
PREFIXES = (10, 11, 12)
CATEGORIES = ('multiple_safe_unobserved_competition', 'single_safe_unobserved_competition',
              'minimum_violation', 'initial', 'other', 'unknown',
              'unavailable_saved_path', 'no_true_certified_deployment')
SCORES = ('objective_mean', 'objective_epistemic', 'sampled_objective', 'sampled_margin',
          'constraint_mean', 'constraint_aleatoric', 'constraint_epistemic',
          'sampled_constraint', 'mean_rank_within_eligible', 'sampled_rank_within_eligible')
LIMITS = [
    'Reference stream is historical; average the two independent streams within geometry separately.',
    'First returned discovery categories are exclusive; resampling is orthogonal and not causal necessity.',
    'Only three role histories have complete saved competition paths; all nine terminal controls remain.',
    'Saved pair summaries and selection records are reused; no new pairs, substitutions or choices are generated.',
    'Only same-history same-pool first divergence describes a common-state propagation comparison.',
    'No certificate is not zero actual quality; missing baselines and incomplete truth remain unknown.',
]


def classify_first_query(state):
    """Classify a saved, returned new-policy query, never a terminal-selected subset."""
    replay, selected = state.get('replay', {}), state.get('selected', {})
    if (state.get('status') != 'captured' or replay.get('valid') is not True or
            replay.get('observation_returned') is not True or
            replay.get('returned_policy_matches') is not True):
        return 'unknown'
    if selected.get('observed') is not False:
        return 'other' if selected.get('observed') is True else 'unknown'
    if replay.get('mode') == 'minimum_violation':
        return 'minimum_violation'
    if replay.get('mode') != 'feasible_objective':
        return 'unknown'
    rows = state.get('candidates')
    complete = bool(rows) and all(row.get('truth') is not None and all(
        row['truth'].get(key) is not None for key in ('true_feasible', 'objective', 'quality')) for row in rows)
    if not complete or selected.get('true_feasible') is None or selected.get('eligible') is None:
        return 'unknown'
    if selected['true_feasible'] is not True or selected['eligible'] is not True:
        return 'other'
    count = state.get('counts', {}).get('safe_eligible_unobserved')
    if count is None:
        return 'unknown'
    return ('multiple_safe_unobserved_competition' if count >= 2 else
            'single_safe_unobserved_competition' if count == 1 else 'other')


def _identities(mechanism, attribution, registry):
    """Unify physical integer vectors across streams, retaining no vectors in output."""
    mapping, local_points = {}, {}
    for row in [*mechanism['policies'], *attribution['policies']]:
        point, local = point_key(row['point']), row['policy_id']
        if local in local_points and local_points[local] != point:
            raise ValueError('Saved mechanism and attribution physical identities disagree')
        local_points[local] = point
        if point not in registry:
            registry[point] = f'P{len(registry):04d}'
        mapping[local] = registry[point]
    return mapping


def _adapt(saved, kind):
    """The two existing attribution schemas; neither adapter performs attribution."""
    if kind == 'observed_terminal':
        attribution = saved['shortlists']
        comparisons = {row['arm']: row for row in saved['comparisons']['arms']}
        arms = []
        for row in attribution['arms']:
            comparison = comparisons[row['arm']]
            protocols = {}
            for protocol in ('old', 'new'):
                original = comparison['protocols'][protocol]
                protocols[protocol] = dict(status=original['status'], metrics=comparison[protocol],
                    deployed_policy_id=original.get('deployed_policy_id'),
                    reported_policy_id=original.get('reported_policy_id'),
                    no_deployment_reason=None if original.get('certified') else 'no_saved_certificate',
                    saved_costs=deepcopy(original.get('saved_costs', {})))
            arms.append(dict(row, protocols=protocols))
        return attribution, arms
    if kind == 'decision_path':
        return saved, saved['arms']
    raise ValueError('Expected observed_terminal or decision_path saved attribution')


def _query(state, mapping):
    selected, replay = state.get('selected', {}), state.get('replay', {})
    rows = state.get('candidates')
    complete = None if rows is None else bool(rows) and all(row.get('truth') is not None and all(
        row['truth'].get(key) is not None for key in ('true_feasible', 'objective', 'quality')) for row in rows)
    returned = replay.get('observation_returned')
    observed = selected.get('observed')
    action = ('not_returned' if returned is False else 'unknown' if returned is not True or
        replay.get('returned_policy_matches') is not True else
        'repeat' if observed is True else 'new' if observed is False else 'unknown')
    return dict(arm=state['arm'], paid_prefix=state['paid_prefix'], status=state['status'],
        selected_policy_id=mapping.get(selected.get('policy_id')),
        previously_observed=observed, observation_returned=returned,
        returned_policy_matches=replay.get('returned_policy_matches'), replay_valid=replay.get('valid'),
        mode=replay.get('mode'), query_action_kind=action,
        discovery_category=classify_first_query(state) if action == 'new' else None,
        pool_truth_complete=complete, counts=deepcopy(state.get('counts')),
        safe_competition=deepcopy(state.get('safe_competition')),
        saved_pair_summary=deepcopy(state.get('pairs')),
        selected_truth=deepcopy(selected.get('truth')),
        selected_saved_scores={key: selected.get(key) for key in SCORES})


def _discovery(history_arm, observed, states):
    if observed is None:
        return dict(discovery_category='unknown' if history_arm in ROLES else 'unavailable_saved_path', first_query_prefix=None,
                    returned_repeat_count=None, returned_adaptive_count=None)
    indices = observed['observation_indices']
    adaptive = [i for i in indices if i >= 10]
    first = min(adaptive) if adaptive else None
    state = states.get((history_arm, first))
    initial = any(i < 10 for i in indices)
    if initial:
        category = 'initial'
    elif history_arm not in ROLES:
        category = 'unavailable_saved_path'
    elif state is None or state.get('selected', {}).get('policy_id') != observed['policy_id']:
        category = 'unknown'
    else:
        category = classify_first_query(state)
    # Observation indices are actual returned raw positions, not captured choices.
    return dict(discovery_category=category, first_query_prefix=first,
        returned_repeat_count=len(adaptive) if initial else max(0, len(adaptive)-1),
        returned_adaptive_count=len(adaptive),
        initially_observed=initial, returned_query_prefixes=adaptive)


def _observed_path(observed, protocol):
    if observed is None:
        return dict(disposition='not_observed_in_this_history', counterfactual_certificate=None,
                    counterfactual_objective_comparison=None)
    return deepcopy(observed['protocols'][protocol])


def _one_stream(root, plan, descriptor, seed, registry):
    source = root / descriptor['source_output_dir']
    seal = read(source / 'execution_status.json')
    if seal['status'] != 'sealed':
        raise ValueError('All saved terminal results must already be sealed')
    mechanism_file = source / 'mechanism' / f'seed_{seed}' / 'analysis.json'
    attribution_file = root / descriptor['attribution_file_pattern'].format(seed=seed)
    mechanism = read(mechanism_file)
    attribution, arms = _adapt(read(attribution_file), descriptor['attribution_kind'])
    expected = {row['name'] for row in plan['terminal_arms']}
    if {row['arm'] for row in arms} != expected or len(arms) != len(expected):
        raise ValueError('Retain all registered terminal controls in the saved attribution')
    if mechanism['seed'] != seed or attribution['seed'] != seed:
        raise ValueError('Saved task identity differs from the registered geometry')
    mapping = _identities(mechanism, attribution, registry)
    states = {(row['arm'], row['paid_prefix']): row for row in mechanism['states']}
    queries = [_query(states.get((arm, prefix), dict(arm=arm, paid_prefix=prefix,
        status='unavailable')), mapping) for arm in ROLES for prefix in PREFIXES]
    by_arm = {row['arm']: row for row in arms}
    for query in queries:
        if query['observation_returned'] is True and query['returned_policy_matches'] is True:
            local = states[query['arm'], query['paid_prefix']]['selected']['policy_id']
            observed = next((row for row in by_arm[query['arm']]['observed'] if row['policy_id'] == local), None)
            query['discovery_category'] = _discovery(query['arm'], observed, states)['discovery_category']
    outcomes = []
    for cell in plan['terminal_arms']:
        row = by_arm[cell['name']]
        if row['history_arm'] != cell['history_arm']:
            raise ValueError('Saved history does not match the registered terminal control')
        observed = {value['policy_id']: value for value in row['observed']}
        for protocol in ('old', 'new'):
            saved = row['protocols'][protocol]
            metrics = {key: saved['metrics'].get(key) for key in METRICS}
            reported = observed.get(saved.get('reported_policy_id'))
            deployed = observed.get(saved.get('deployed_policy_id')) if metrics['C'] == 1 else None
            discovery = _discovery(row['history_arm'], deployed, states)
            if metrics['C'] == 0:
                discovery.update(discovery_category='no_true_certified_deployment',
                    first_query_prefix=None, returned_repeat_count=None, returned_adaptive_count=None)
            value = None if deployed is None else deployed.get('truth')
            outcomes.append(dict(arm=row['arm'], history_arm=row['history_arm'], protocol=protocol,
                status=saved['status'], metrics=metrics,
                reported_policy_id=mapping.get(saved.get('reported_policy_id')),
                deployed_policy_id=mapping.get(saved.get('deployed_policy_id')) if metrics['C'] == 1 else None,
                actual_deployed_quality=None if value is None else value.get('quality'),
                **discovery, terminal_path=_observed_path(deployed, protocol) if deployed is not None else None,
                reported_terminal_path=_observed_path(reported, protocol) if reported is not None else None,
                no_deployment_reason=saved.get('no_deployment_reason'),
                saved_costs=deepcopy(saved.get('saved_costs', saved.get('execution', {})))))
    paths = []
    for arm in ROLES:
        for observed in by_arm[arm]['observed']:
            if not any(i >= 10 for i in observed['observation_indices']):
                continue
            local = observed['policy_id']
            exposures = []
            for other in ROLES:
                entries = []
                for prefix in PREFIXES:
                    state = states.get((other, prefix), {})
                    candidate = next((r for r in state.get('candidates', []) if r['policy_id'] == local), None)
                    entries.append(dict(paid_prefix=prefix, status=state.get('status', 'unavailable'),
                        present=None if 'candidates' not in state else candidate is not None,
                        observed=None if candidate is None else candidate['observed'],
                        eligible=None if candidate is None else candidate['eligible'],
                        selected=None if candidate is None else candidate['selected'],
                        mode=state.get('replay', {}).get('mode'),
                        saved_scores=None if candidate is None else {key: candidate.get(key) for key in SCORES}))
                counterpart = next((r for r in by_arm[other]['observed'] if r['policy_id'] == local), None)
                exposures.append(dict(history_arm=other, states=entries,
                    returned_observation_indices=[] if counterpart is None else counterpart['observation_indices']))
            terminals = []
            for cell in arms:
                found = next((r for r in cell['observed'] if r['policy_id'] == local), None)
                for protocol in ('old', 'new'):
                    terminals.append(dict(arm=cell['arm'], history_arm=cell['history_arm'], protocol=protocol,
                        status=cell['protocols'][protocol]['status'], path=_observed_path(found, protocol)))
            paths.append(dict(history_arm=arm, policy_id=mapping[local], origin=observed['origin'],
                observation_indices=observed['observation_indices'], truth=deepcopy(observed.get('truth')),
                quality_gain_to_initial=observed.get('quality_gain_to_initial'),
                improving_safe=observed.get('improving_safe'),
                **_discovery(arm, observed, states), exposures=exposures, terminals=terminals))
    comparisons = []
    cells = {(row['arm'], row['protocol']): row for row in outcomes}
    for contrast, left, right in (('AP', AFTER, POINT), ('FA', FULL, AFTER)):
        for protocol in ('old', 'new'):
            a, b = cells[left, protocol], cells[right, protocol]
            comparisons.append(dict(contrast=contrast, left=left, right=right, protocol=protocol,
                **metric_difference(a['metrics'], b['metrics']),
                left_deployed_policy_id=a['deployed_policy_id'], right_deployed_policy_id=b['deployed_policy_id']))
    query_comparisons = []
    for row in mechanism['comparisons']:
        fields = ('contrast', 'left', 'right', 'paid_prefix', 'status', 'same_paid_history',
            'same_physical_pool', 'query_changed', 'first_divergence', 'scope')
        item = {key: row.get(key) for key in fields}
        item['common_state_first_divergence'] = (bool(row['first_divergence'] and
            row['same_paid_history'] and row['same_physical_pool']) if row.get('status') == 'available' else None)
        item['selected'] = {key: mapping[pid] for key, pid in row.get('selected', {}).items()}
        query_comparisons.append(item)
    return dict(name=descriptor['name'], batch=descriptor['batch'], seed=seed, domain=mechanism['domain'],
        status='complete' if mechanism['status'] == attribution['status'] == 'complete' else 'partial',
        baseline=deepcopy(attribution['baseline']), outcomes=outcomes, queries=queries,
        policy_paths=paths, comparisons=comparisons, query_comparisons=query_comparisons,
        coverage=dict(terminal_outcomes=len(outcomes), mechanism_histories=len(ROLES),
            mechanism_queries=len(queries), complete_truth_pools=sum(q['pool_truth_complete'] is True for q in queries),
            unknown_truth_pools=sum(q['pool_truth_complete'] is not True for q in queries)),
        sources=dict(mechanism_file=str(mechanism_file.relative_to(root)),
            attribution_file=str(attribution_file.relative_to(root)),
            historical_completion_file=str((source/'completion.json').relative_to(root))))


def analyze_task(root, plan, seed):
    """Read one geometry's reference stream and two independent repeats."""
    root, registry = Path(root), {}
    streams = [_one_stream(root, plan, descriptor, seed, registry) for descriptor in plan['streams']]
    if len(streams) != 3 or {s['name'] for s in streams} != {'reference', 'observation_1', 'observation_2'}:
        raise ValueError('The fixed three streams must remain separate and complete')
    if len({s['domain'] for s in streams}) != 1:
        raise ValueError('The three streams must share the same registered geometry family')
    return dict(seed=seed, domain=streams[0]['domain'],
        status='complete' if all(s['status'] == 'complete' for s in streams) else 'partial',
        streams=streams, physical_policies=len(registry), limits=LIMITS)
