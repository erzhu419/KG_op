"""Saved terminal comparison statistics and actual omissions; no new inference."""
from collections import Counter
from copy import deepcopy
from pathlib import Path

from performance import mainline_confirmation_terminal_attribution as prior
from performance.mainline_three_stream_query_attribution import register_policy


def sign(value):
    return 'unknown' if value is None else 'negative' if value < 0 else 'positive' if value > 0 else 'zero'


def comparison_diagnostics(record, true_difference=None):
    """Decompose the stored bound; never recompute a certificate or select a delta."""
    mean = record['sample_mean_difference']
    standard_error, critical = record['standard_error'], record['critical_value']
    selected = bool(record['challenger_dominates'])
    reason = ('challenger_dominates' if selected else
        'favorable_mean_blocked_by_uncertainty' if mean < 0 else
        'point_estimate_tie' if mean == 0 else 'point_estimate_favors_incumbent')
    return dict(point_estimate_direction=sign(mean), true_direction=sign(true_difference),
        sample_mean_difference=mean, sample_std_difference=record['sample_std_difference'],
        standard_error=standard_error, critical_value=critical,
        uncertainty_penalty=critical * standard_error,
        saved_upper_bound=record['one_sided_upper_confidence_bound'],
        delta=record['delta'], comparison_budget_per_policy=record['comparison_budget_per_policy'],
        degrees_of_freedom=record.get('degrees_of_freedom'),
        challenger_dominates=selected, disposition=reason,
        true_objective_difference=true_difference,
        point_estimate_error=None if true_difference is None else mean - true_difference,
        point_estimate_sign_matches_truth=None if true_difference is None else sign(mean) == sign(true_difference),
        true_better_objective_challenger_not_selected=None if true_difference is None else
            bool(true_difference < 0 and not selected),
        inference='Original mean + critical_value * standard_error; decision copied from the saved result.')


def comparison_slots(protocol, descriptor, saved, item):
    """Unavailable comparisons are branches, not failed objective tests."""
    if descriptor['status'] != 'complete':
        return [dict(block_index=block, status='unknown_after_' + descriptor['status'], executed=False,
            challenger_index=None, incumbent_index=None)
            for block in ((0,) if protocol == 'old' else (0, 1))]
    verification = saved['verification']
    if protocol == 'old':
        comparison = verification['objective_comparison']
        return [dict(block_index=0, executed=bool(comparison.get('simulation_calls', 0)),
            status='executed' if comparison.get('simulation_calls', 0) else comparison['status'],
            challenger_index=0, incumbent_index=item['objective_incumbent_position'] - 1)]
    comparisons = {row['block_index']: row for row in verification['objective_comparisons']}
    attempts = {row['candidate_index']: row['certified'] for row in verification['attempts']}
    first = verification['initial_stage_winner_index']
    slots = []
    for block, challenger, incumbent in ((0, 0, 1), (1, 2, first)):
        actual = comparisons.get(block)
        if actual is not None:
            slots.append(dict(block_index=block, executed=True, status='executed',
                challenger_index=actual['challenger_index'], incumbent_index=actual['incumbent_index']))
            continue
        if block == 0:
            reason = ('neither_primary_nor_incumbent_safety_certified' if not attempts.get(0) and not attempts.get(1)
                else 'challenger_not_safety_certified' if not attempts.get(0)
                else 'incumbent_not_safety_certified' if not attempts.get(1)
                else 'comparison_record_unavailable')
        else:
            reason = ('third_not_safety_certified' if not attempts.get(2)
                else 'third_only_without_first_stage_opponent' if first is None
                else 'comparison_record_unavailable')
        slots.append(dict(block_index=block, executed=False, status=reason,
            challenger_index=challenger, incumbent_index=incumbent))
    return slots


def analyze_pair(case, reference, items, saved, history, truth, policy_registry=None):
    """Join one frozen history's two terminals and existing cached point labels."""
    registry = {} if policy_registry is None else policy_registry
    result = prior.analyze_pair(case, reference, items, saved, history, truth)
    anchor = items.get('new') or items.get('old')
    if anchor is None:
        result.update(observed_pool=[], omitted_observed=[], comparison_slots={})
        return result
    points = [tuple(row['point']) for row in anchor['frozen_shortlist']]
    identifiers = [register_policy(registry, point) for point in points]
    label_ids = {candidate['policy']: identifier for candidate, identifier in zip(result['candidates'], identifiers)}
    for candidate, identifier in zip(result['candidates'], identifiers):
        candidate['policy_id'] = identifier
    observed = []
    for point in dict.fromkeys(tuple(row['point']) for row in history):
        indices = [i for i, row in enumerate(history) if tuple(row['point']) == point]
        actual = truth.get(point)
        index = points.index(point) if point in points else None
        observed.append(dict(policy_id=register_policy(registry, point), observed_at=indices,
            shortlist_index=index, shortlist_label=None if index is None else reference['shortlist'][index],
            cached_truth=actual,
            good_beyond_initial=None if actual is None or not reference['quality_baseline_defined'] else
                bool(actual['true_feasible'] and actual['quality'] > reference['best_initial_quality'])))
    result['observed_pool'] = observed
    result['omitted_observed'] = [row for row in observed if row['shortlist_index'] is None]
    result['comparison_slots'] = {}
    for protocol in ('old', 'new'):
        descriptor, record = case[protocol], saved.get(protocol)
        current = result['protocols'][protocol]
        current.update(deployed_policy_id=label_ids.get(current['deployed']),
            reported_policy_id=label_ids.get(current['reported_deployed']))
        current['saved_costs'] = {} if record is None else {key: record[key] for key in (
            'new_verification_calls', 'cumulative_new_verification_calls', 'logical_verification_calls',
            'reused_old_raw_calls', 'reused_new_raw_calls', 'failed_simulation_calls',
            'cumulative_new_simulation_invocations', 'historical_simulation_invocations') if key in record}
        slots = comparison_slots(protocol, descriptor, record, items.get(protocol) or anchor)
        for slot in slots:
            for role in ('challenger', 'incumbent'):
                index = slot[role + '_index']
                slot[role + '_policy_id'] = identifiers[index] if index is not None else None
                slot[role + '_label'] = reference['shortlist'][index] if index is not None else None
        result['comparison_slots'][protocol] = slots
        if descriptor['status'] != 'complete':
            continue
        verification = record['verification']
        actual_records = ({row['block_index']: row for row in verification['objective_comparisons']}
            if protocol == 'new' else {0: verification['objective_comparison']})
        for comparison in current['comparisons']:
            block = comparison['block_index']
            for role in ('challenger', 'incumbent', 'winner'):
                comparison[role + '_policy_id'] = label_ids[comparison[role]]
            comparison['statistics'] = comparison_diagnostics(actual_records[block],
                comparison['true_objective_difference'])
            slot = next(row for row in slots if row['block_index'] == block)
            comparison.update(challenger_index=slot['challenger_index'], incumbent_index=slot['incumbent_index'])
            comparison['cached_feasibility'] = {role: None if truth.get(points[slot[role + '_index']]) is None else
                truth[points[slot[role + '_index']]]['true_feasible'] for role in ('challenger', 'incumbent')}
    result['comparison_scope'] = 'Actual independent-block pairs only. Untested candidates have no counterfactual certificate.'
    return result


def analyze_task(root, plan, seed, policy_registry=None):
    """Read all nine pairs for one task; no raw replay or truth evaluation."""
    root = Path(root)
    directory = root / plan['cohort_dir']
    summary, seal = prior.read(directory / 'summary.json'), prior.read(directory / 'execution_status.json')
    if seal['status'] != 'sealed':
        raise ValueError('Terminal comparisons must already be globally sealed')
    cases = [row for row in seal['cases'] if row['seed'] == seed]
    references = {row['arm']: row for row in summary['rows'] if row['seed'] == seed}
    if not cases or {row['arm'] for row in cases} != set(references):
        raise ValueError('Retain every registered terminal pair and saved metric row')
    registry = {} if policy_registry is None else policy_registry
    histories, caches, arms = {}, {}, []
    for case in cases:
        items, saved = {}, {}
        for protocol in ('old', 'new'):
            descriptor = case[protocol]
            if descriptor.get('input_file'):
                items[protocol] = prior.read(root / descriptor['input_file'])
            key = 'verification_file' if descriptor['status'] == 'complete' else 'failure_file'
            if descriptor.get(key):
                saved[protocol] = prior.read(root / descriptor[key])
        anchor = items.get('new') or items.get('old')
        history, truth = [], {}
        if anchor is not None:
            source_case = anchor['source_case']
            if source_case not in histories:
                histories[source_case] = prior.read(root / source_case / 'search.json')['history']
            history = histories[source_case]
            cache_path = Path(source_case).parent / 'truth.json'
            if str(cache_path) not in caches:
                caches[str(cache_path)] = {tuple(row['point']): row['truth'] for row in prior.read(root / cache_path)}
            truth = caches[str(cache_path)]
        arms.append(analyze_pair(case, references[case['arm']], items, saved, history, truth, registry))
    return dict(seed=seed, domain=cases[0]['domain'], status='complete', arms=arms,
        quality_reference_status=next(iter(references.values()))['quality_reference_status'],
        source_output_dir=plan['cohort_dir'], new_work=dict(prior.ZERO),
        interpretation='Saved terminal directions, penalties and exposure. No new tests or modified thresholds.')


def compact_task(result):
    compact = prior.compact_task(result)
    indexed = {arm['arm']: arm for arm in result['arms']}
    for arm in compact['arms']:
        source = indexed[arm['arm']]
        arm['comparison_slots'] = source['comparison_slots']
        arm['observed_coverage'] = dict(observed_unique=len(source['observed_pool']),
            omitted_count=len(source['omitted_observed']),
            omitted_good=[row for row in source['omitted_observed'] if row['good_beyond_initial'] is True],
            omitted_unknown_truth=[row['policy_id'] for row in source['omitted_observed'] if row['cached_truth'] is None])
        for candidate, original in zip(arm['candidates'], source['candidates']):
            candidate['policy_id'] = original['policy_id']
        for protocol, value in arm['protocols'].items():
            original = source['protocols'][protocol]
            value.update({key: original[key] for key in ('deployed_policy_id', 'reported_policy_id', 'saved_costs')})
            value['comparisons'] = deepcopy(original['comparisons'])
    return compact


def aggregate_tasks(tasks):
    """Descriptive branch counts retain all tasks, arms, protocols and losses."""
    inherited = prior.aggregate_terminals(tasks)
    rows = []
    names = list(dict.fromkeys(arm['arm'] for task in tasks for arm in task['arms']))
    for arm in names:
        for protocol in ('old', 'new'):
            selected = [(task, next(row for row in task['arms'] if row['arm'] == arm)) for task in tasks]
            comparisons = [(task, comparison) for task, row in selected
                for comparison in row['protocols'].get(protocol, {}).get('comparisons', [])]
            rows.append(dict(arm=arm, protocol=protocol, registered_tasks=len(selected),
                terminal_status_counts=dict(Counter(row['protocols'].get(protocol, {}).get('status', 'inputs_unavailable')
                    for _, row in selected)),
                slot_status_counts=dict(Counter(str(slot['block_index']) + ':' + slot['status']
                    for _, row in selected for slot in row['comparison_slots'].get(protocol, []))),
                executed_comparisons=len(comparisons),
                comparison_disposition_counts=dict(Counter(c['statistics']['disposition'] for _, c in comparisons)),
                true_direction_counts=dict(Counter(c['statistics']['true_direction'] for _, c in comparisons)),
                point_estimate_direction_counts=dict(Counter(c['statistics']['point_estimate_direction'] for _, c in comparisons)),
                true_better_objective_challenger_not_selected=[dict(seed=task['seed'], domain=task['domain'], **comparison)
                    for task, comparison in comparisons
                    if comparison['statistics']['true_better_objective_challenger_not_selected'] is True],
                unknown_truth_comparisons=[dict(seed=task['seed'], block=comparison['block_index'])
                    for task, comparison in comparisons if comparison['statistics']['true_direction'] == 'unknown'],
                omitted_good_observed=[dict(seed=task['seed'], domain=task['domain'], policies=row['observed_coverage']['omitted_good'])
                    for task, row in selected if row['observed_coverage']['omitted_good']]))
    return dict(task_count=len(tasks), registered_pairs=sum(len(task['arms']) for task in tasks),
        terminals=inherited, comparison_groups=rows, new_work=dict(prior.ZERO),
        interpretation='Saved evidence only; comparison counts are exposure counts, not independent tasks or new certificates.')
