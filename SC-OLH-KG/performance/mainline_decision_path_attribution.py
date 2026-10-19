"""Trace frozen decisions using saved scores, outcomes and truth caches only.

Set maxima describe opportunities, not counterfactual certificates.  A missing
truth value or an empty true-safe set has no maximum; no deployment is not Q=0.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from performance.mainline_confirmation_query_attribution import terminal_path


ROLES = ('legacy_role_full', 'legacy_role_after_init_point_only', 'legacy_role_point_only')
LAYERS = ('search', 'shortlist', 'coverage', 'safety', 'comparison')
METRICS = ('A', 'C', 'false_certificate', 'R', 'G', 'P', 'L')


def read(path):
    return json.loads(Path(path).read_text())


def point_key(point):
    return tuple(map(int, point))


def quality_set(points, truth, registry, *, available=True, unavailable_reason=None):
    """Return an exact maximum only when membership and all cached truth are known."""
    points = list(dict.fromkeys(map(point_key, points)))
    known = [p for p in points if truth.get(p) is not None and
             truth[p].get('true_feasible') is not None and truth[p].get('quality') is not None]
    safe = [p for p in known if truth[p]['true_feasible']]
    status = (unavailable_reason or 'membership_unavailable') if not available else (
        'truth_incomplete' if len(known) != len(points) else
        'empty_set' if not points else 'no_true_safe_policy' if not safe else 'defined')
    maximum = max(truth[p]['quality'] for p in safe) if status == 'defined' else None
    return dict(status=status, count=len(points), truth_known=len(known),
        truth_unknown=len(points)-len(known), true_safe_count=len(safe),
        membership_complete=available, max_quality=maximum,
        best_policy_ids=[registry[p] for p in safe if truth[p]['quality'] == maximum],
        policy_ids=[registry[p] for p in points])


def opportunity_gaps(sets):
    """Adjacent differences telescope; downstream nulls are never zero-filled."""
    ordered = ('pool', 'observed', 'shortlist', 'attempted', 'safety_passed', 'deployed')
    gaps, reasons = {}, {}
    for name, left, right in zip(LAYERS, ordered, ordered[1:]):
        a, b = sets[left], sets[right]
        gaps[name] = None if a['max_quality'] is None or b['max_quality'] is None else a['max_quality']-b['max_quality']
        reasons[name] = None if gaps[name] is not None else dict(left=a['status'], right=b['status'])
    for name, layers in (('terminal_total', LAYERS[1:]), ('total', LAYERS)):
        gaps[name] = sum(gaps[layer] for layer in layers) if all(gaps[layer] is not None for layer in layers) else None
        reasons[name] = None if gaps[name] is not None else [layer for layer in layers if gaps[layer] is None]
    gaps['telescoping_residual'] = (None if gaps['total'] is None else
        gaps['total']-(sets['pool']['max_quality']-sets['deployed']['max_quality']))
    return gaps, reasons


def _initial_reference(initial, truth):
    points = [point_key(row['point']) for row in initial]
    if len(initial) != 10 or any(truth.get(p) is None or
            any(truth[p].get(k) is None for k in ('objective', 'quality', 'true_feasible')) for p in points):
        return dict(status='initial_reference_unavailable', truth=None)
    safe = [truth[p] for p in points if truth[p]['true_feasible']]
    return dict(status='defined' if safe else 'no_true_feasible_initial',
        truth=min(safe, key=lambda value: value['objective']) if safe else None)


def _no_deployment_reason(descriptor, metrics, paths):
    if descriptor['status'] != 'complete':
        return 'execution_' + descriptor['status']
    if metrics.get('C') == 1:
        return None
    if metrics.get('A') == 1:
        return 'false_certificate' if metrics.get('C') == 0 else 'certified_truth_unknown'
    attempted = [row for row in paths if row.get('safety_attempted')]
    if not attempted:
        return 'no_safety_verification_recorded'
    if not any(row.get('safety_passed') for row in attempted):
        return 'all_attempted_safety_certificates_failed'
    return 'safety_passed_without_certified_deployment'


def attribute_arm(case, history, frozen, truth, baseline, registry, origins, chains,
                  states, saved_metrics, recording, *, search_status='complete'):
    """Pure one-history/terminal attribution; preserves the actual C/G definition."""
    points = list(dict.fromkeys(point_key(row['point']) for row in history))
    record_rows = {row['observed_index']: row for row in recording.get('observed', [])}
    observed = []
    for index, point in enumerate(points):
        value, selection = truth.get(point), record_rows.get(index)
        indices = [i for i, row in enumerate(history) if point_key(row['point']) == point]
        gain = (None if value is None or value.get('quality') is None or baseline['truth'] is None else
            value['quality']-baseline['truth']['quality'])
        observed.append(dict(policy_id=registry[point], origin=origins.get(point, 'dynamic'),
            observation_indices=indices, first_observation_index=indices[0],
            first_adaptive_observation_index=next((i for i in indices if i >= 10), None),
            truth=value, quality_gain_to_initial=gain,
            improving_safe=None if gain is None or value.get('true_feasible') is None else bool(value['true_feasible'] and gain > 0),
            final_scores=None if selection is None else dict(
                objective=selection.get('objective_mean'),
                probability_violation=selection.get('objective_probability_violation'),
                terminal_order_rank=selection.get('posterior_rank'),
                eligible=selection.get('objective_eligible'), source='saved_terminal_recording'),
            selection_record=deepcopy(selection), quality_minus_deployed={},
            protocols={name: terminal_path(point, descriptor.get('item'),
                descriptor.get('record') if descriptor['status'] == 'complete' else None,
                descriptor['status']) for name, descriptor in ((p, case[p]) for p in ('old', 'new'))}))
    initial = [point_key(row['point']) for row in history[:10]]
    pool_points = list(points)
    for state in states:
        for row in state.get('candidates', []):
            point = next(p for p, pid in registry.items() if pid == row['policy_id'])
            pool_points.append(point)
    pool_available = (case.get('history_arm', case['arm']) in ROLES and
        len(states) == 3 and {row.get('paid_prefix') for row in states} == {10, 11, 12} and
        all(row.get('status') == 'captured' for row in states))
    sets = dict(pool=quality_set(pool_points, truth, registry, available=pool_available,
        unavailable_reason='full_pool_not_saved' if not states else 'full_pool_incomplete'),
        observed=quality_set(points, truth, registry, available=search_status == 'complete',
            unavailable_reason='search_history_incomplete'),
        initial=quality_set(initial, truth, registry, available=len(history) >= 10,
            unavailable_reason='initial_history_incomplete'))
    new_points = [p for p in points if p not in set(initial)]
    new_truth_unknown = sum(truth.get(p) is None or any(truth[p].get(k) is None
        for k in ('quality', 'true_feasible')) for p in new_points)
    improving_new = (None if baseline['truth'] is None else sum(truth.get(p) is not None and
        truth[p].get('true_feasible') is True and truth[p].get('quality') is not None and truth[p]['quality'] >
        baseline['truth']['quality'] for p in new_points))
    discovery_gain = (None if any(sets[k]['max_quality'] is None for k in ('observed', 'initial')) else
        sets['observed']['max_quality']-sets['initial']['max_quality'])
    result = dict(arm=case['arm'], history_arm=case.get('history_arm', case['arm']),
        terminal_backend=case.get('terminal_backend'), search_status=search_status,
        sets=sets, discovery=dict(quality_gain=discovery_gain,
            adaptive_query_count=max(0, len(history)-10), adaptive_new_policy_count=len(new_points),
            improving_safe_new_policy_count=improving_new if not new_truth_unknown else None,
            known_improving_safe_new_policy_count=improving_new,
            new_policy_truth_unknown=new_truth_unknown,
            query_chains=[row for row in chains if row['arm'] == case.get('history_arm', case['arm'])]),
        observed=observed, protocols={},
        recording=dict(status=recording.get('status', 'missing_record'), issues=recording.get('issues', [])),
        objective_challenger=deepcopy(frozen.get('terminal', {}).get('objective_challenger')))
    for protocol in ('old', 'new'):
        descriptor = case[protocol]
        item = descriptor.get('item')
        record = descriptor.get('record') if descriptor['status'] == 'complete' else None
        metrics = {k: saved_metrics.get(protocol, {}).get(k) for k in METRICS}
        shortlist = [] if item is None else [point_key(row['point']) for row in item['frozen_shortlist']]
        verification = None if record is None else record['verification']
        attempts = [] if verification is None else verification['attempts']
        attempted = [shortlist[row['candidate_index']] for row in attempts]
        passed = [shortlist[row['candidate_index']] for row in attempts if row['certified']]
        # V is actually certified AND truly feasible, with unknown truth retained
        # in the set so that an incomplete cache cannot create an optimistic max.
        safe_passed = [p for p in passed if truth.get(p) is None or truth[p].get('true_feasible') is not False]
        deployed_point = point_key(record['deployed']) if record is not None else None
        deployed = [deployed_point] if metrics['C'] == 1 else []
        protocol_sets = dict(shortlist=quality_set(shortlist, truth, registry, available=item is not None),
            attempted=quality_set(attempted, truth, registry, available=record is not None),
            safety_passed=quality_set(safe_passed, truth, registry, available=record is not None),
            deployed=quality_set(deployed, truth, registry,
                available=record is not None and metrics['C'] is not None,
                unavailable_reason='deployment_or_truth_unavailable'))
        if record is not None and metrics['C'] == 0:
            protocol_sets['deployed']['status'] = 'no_true_certified_deployment'
        all_sets = {**sets, **protocol_sets}
        gaps, reasons = opportunity_gaps(all_sets)
        paths = [row['protocols'][protocol] for row in result['observed']]
        nesting = [name for name, left, right in zip(LAYERS,
            ('pool', 'observed', 'shortlist', 'attempted', 'safety_passed'),
            ('observed', 'shortlist', 'attempted', 'safety_passed', 'deployed'))
            if not set(all_sets[right]['policy_ids']).issubset(all_sets[left]['policy_ids'])]
        if nesting:
            # Preserve the records and actual outcome, but do not label a
            # nonnested sequence as a telescoping opportunity decomposition.
            gaps = dict.fromkeys(gaps)
            reasons = {name: dict(non_nested_layers=nesting) for name in reasons}
        actual_quality = protocol_sets['deployed']['max_quality']
        for row in result['observed']:
            value = row['truth']
            row['quality_minus_deployed'][protocol] = (None if actual_quality is None or
                value is None or value.get('true_feasible') is not True or value.get('quality') is None
                else value['quality']-actual_quality)
        result['protocols'][protocol] = dict(status=descriptor['status'], metrics=metrics,
            sets=protocol_sets, opportunity_gaps=gaps, gap_unavailable_reasons=reasons,
            nesting_issues=nesting, deployed_policy_id=registry.get(deployed_point) if metrics['C'] == 1 else None,
            reported_policy_id=registry.get(deployed_point),
            no_deployment_reason=_no_deployment_reason(descriptor, metrics, paths),
            counts=dict(dispositions=dict(Counter(row.get('disposition', 'unknown') for row in paths)),
                improving_safe_dispositions=dict(Counter(row['protocols'][protocol]['disposition']
                    for row in result['observed'] if row['improving_safe'] is True)),
                better_than_deployed_dispositions=dict(Counter(row['protocols'][protocol]['disposition']
                    for row in result['observed'] if row['quality_minus_deployed'][protocol] is not None and
                    row['quality_minus_deployed'][protocol] > 1e-12))),
            execution=deepcopy(recording.get('protocols', {}).get(protocol, {})))
    return result


def analyze_task(root, plan, seed, *, stream, terminal_recording_file=None):
    """Read one sealed task-stream. Caller meters and schedules this read-only work."""
    root = Path(root)
    output = root / plan['output_dir']
    seal = read(output / 'execution_status.json')
    if seal['status'] != 'sealed':
        raise ValueError('Decision paths require the completed terminal seal')
    cases = [row for row in seal['cases'] if row['seed'] == seed]
    if len(cases) != len(plan['terminal_arms']):
        raise ValueError('Registered terminal controls are missing from the existing seal')
    mechanism_path = output / 'mechanism' / f'seed_{seed}' / 'analysis.json'
    analysis = read(mechanism_path) if mechanism_path.exists() else dict(policies=[], states=[], query_chains=[])
    registry = {point_key(row['point']): row['policy_id'] for row in analysis['policies']}
    origins = {point_key(row['point']): row['origin'] for row in analysis['policies']}
    directory = (root / cases[0]['source_case']).parent
    initial_path = directory / 'initialization.json'
    initial = read(initial_path)['history'] if initial_path.exists() else []
    truth_path = directory / 'truth.json'
    truth = {point_key(row['point']): row['truth'] for row in read(truth_path)} if truth_path.exists() else {}
    baseline = _initial_reference(initial, truth)
    summary = {row['arm']: row for row in read(output / 'summary.json')['rows'] if row['seed'] == seed}
    audit = read(root / terminal_recording_file) if terminal_recording_file else dict(cases=[])
    recordings = {row['arm']: row for row in audit['cases'] if row['seed'] == seed}
    result_arms, histories = [], {}
    for original in cases:
        case = deepcopy(original)
        cell = next(row for row in plan['terminal_arms'] if row['name'] == case['arm'])
        case['terminal_backend'] = cell['terminal_backend']
        case['history_arm'] = cell['history_arm']
        for protocol in ('old', 'new'):
            descriptor = case[protocol]
            path = descriptor.get('input_file')
            descriptor['item'] = read(root/path) if path and (root/path).exists() else None
            path = descriptor.get('verification_file')
            descriptor['record'] = read(root/path) if descriptor['status'] == 'complete' and path else None
        source = root / case['source_case']
        if case['source_case'] not in histories:
            if (source/'search.json').exists():
                search = read(source/'search.json')
                histories[case['source_case']] = (search, search['history'], 'complete')
            else:
                failures = sorted(source.glob('failed_search_raw_*.json'))
                raw = read(failures[-1]) if failures else []
                history = raw if case['history_arm'] == 'initialization_only' else [*initial, *raw]
                histories[case['source_case']] = ({}, history, 'partial' if failures else 'missing')
        search, history, search_status = histories[case['source_case']]
        native_file = source / 'native_terminal.json'
        frozen = read(native_file) if case['terminal_backend'] == 'native' and native_file.exists() else search
        for row in history:
            point = point_key(row['point'])
            if point not in registry:
                registry[point] = f'P{len(registry):04d}'
            origins.setdefault(point, 'initial' if point in {point_key(r['point']) for r in initial} else 'dynamic')
        # Complete terminal inputs are subsets of observed physical policies.
        # Keep any inconsistent saved identity visible, rather than dropping it.
        for protocol in ('old', 'new'):
            item = case[protocol]['item']
            for row in [] if item is None else item['frozen_shortlist']:
                point = point_key(row['point'])
                if point not in registry:
                    registry[point] = f'P{len(registry):04d}'
        states = [row for row in analysis['states'] if row['arm'] == case['history_arm']]
        result_arms.append(attribute_arm(case, history, frozen, truth, baseline, registry, origins,
            analysis['query_chains'], states, summary.get(case['arm'], {}), recordings.get(case['arm'], {}),
            search_status=search_status))
    return dict(status='complete', stream=stream, seed=seed, domain=cases[0]['domain'],
        baseline=baseline, arms=result_arms,
        coverage=dict(registered_arms=len(plan['terminal_arms']), arms=len(result_arms),
            terminal_cells=2*len(result_arms), cached_truth_policies=len(truth),
            mechanism_saved=mechanism_path.exists(),
            recording_file=None if terminal_recording_file is None else str(terminal_recording_file)),
        query_comparisons=analysis.get('comparisons', []),
        policies=[dict(policy_id=pid, point=list(point), origin=origins.get(point)) for point, pid in registry.items()],
        limits=['Pool opportunities apply only to the saved actual pool of the same search history.',
            'Set maxima use existing true-safe cached policies and are not attainable counterfactual deployments.',
            'Opportunity gaps telescope only on complete nested quality chains; no certificate is not zero quality.',
            'Query and policy paths share terminal outcomes; their quality gains must not be summed as independent contributions.'])


def compact_task(raw):
    """Retain all controls and terminal gaps without downloading physical vectors."""
    result = deepcopy({key: value for key, value in raw.items() if key not in ('policies', 'query_comparisons')})
    for arm in result['arms']:
        arm['discovery']['query_chains'] = [{k: row.get(k) for k in
            ('arm', 'paid_prefix', 'policy_id', 'origin', 'truth', 'previously_observed',
             'eligible', 'observation_returned', 'returned_policy_matches', 'backend_conditioned')}
            for row in arm['discovery']['query_chains']]
        for row in arm['observed']:
            row.pop('query_chains', None)
    return result
