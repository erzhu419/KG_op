"""Summarize existing same-search terminals; reference membership is frozen first."""
from collections import Counter
from copy import deepcopy

from performance.summarize_mainline_decision_path_attribution import measure
from performance.summarize_mainline_two_comparison_development import AFTER, FULL, POINT, METRICS


AFTER_NATIVE = AFTER + '_native_terminal'
FULL_NATIVE = FULL + '_native_terminal'
STAGES = ('shortlisted', 'attempted', 'certified', 'comparison_attempted', 'deployed')
ROLES = ('all_initial', 'empirical_incumbent', 'stop_primary', 'stop_safe_fallback',
         'stop_final_slot1', 'stop_final_slot2', 'stop_final_slot3')
EPS = 1e-12


def identity(row):
    return {key: row[key] for key in ('seed', 'domain', 'stream')}


def difference(left, right):
    return None if left is None or right is None else left-right


def _number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool)


def _signed_account(rows):
    """Descriptive task-stream accounting, never a 48-unit uncertainty estimate."""
    result = dict(task_streams=len(rows))
    for metric in ('C', 'G'):
        values = [row['metrics'][metric] for row in rows if row['metrics'][metric] is not None]
        positive, negative = [v for v in values if v > EPS], [v for v in values if v < -EPS]
        result[metric] = dict(defined=len(values), missing=len(rows)-len(values),
            positive=len(positive), negative=len(negative), zero=len(values)-len(positive)-len(negative),
            positive_sum=sum(positive), negative_sum=sum(negative),
            defined_sum=sum(values), all_registered_sum=sum(values) if len(values)==len(rows) else None)
    return result


def _references(task, saved):
    indexed = {row['reference_id']: row for row in saved.get('reference_rows', [])}
    result = []
    for reference in task.get('references', []):
        row = indexed.get(reference['reference_id'], {})
        result.append(dict(deepcopy(row), reference_id=reference['reference_id'],
            roles=deepcopy(reference.get('roles', [])), policy_id=reference.get('policy_id'),
            **{field: row.get(field) for field in STAGES}))
    return result


def _role_rows(cell, role):
    return [row for row in cell['references'] if role == 'all_initial' or role in row['roles']]


def _role_coverage(cells, role):
    selected = [_role_rows(cell, role) for cell in cells]
    known = sum(cell['analysis_status']=='complete' for cell in cells)
    result = dict(role=role, task_streams=len(cells), analysis_available_task_streams=known,
        task_streams_with_reference=sum(bool(rows) for rows in selected),
        reference_occurrences=sum(map(len, selected)), stages={})
    for stage in STAGES:
        values = [row[stage] for rows in selected for row in rows]
        result['stages'][stage] = dict(yes=sum(v is True for v in values),
            no=sum(v is False for v in values), unknown=sum(v is None for v in values))
    result['true_feasible_reference_occurrences'] = sum(
        row.get('truth', {}).get('true_feasible') is True for rows in selected for row in rows
        if isinstance(row.get('truth'), dict))
    result['better_than_actual_deployment'] = dict(
        known=sum(row.get('quality_minus_actual_deployed') is not None for rows in selected for row in rows),
        positive=sum(row.get('quality_minus_actual_deployed') is not None and
            row['quality_minus_actual_deployed'] > EPS for rows in selected for row in rows))
    return result


def _certificate_counts(cells):
    return dict(Counter('unknown' if cell['metrics']['A'] is None or cell['metrics']['C'] is None
        else 'no_certificate' if cell['metrics']['A']==0 else
        'true_certificate' if cell['metrics']['C']==1 else 'false_certificate' for cell in cells))


def _group(cells, streams):
    cost_keys = sorted({key for cell in cells for key, value in cell['inherited_scientific_costs'].items()
                        if _number(value)})
    return dict(arm=cells[0]['arm'], protocol=cells[0]['protocol'], registered_task_streams=len(cells),
        analysis_status_counts=dict(Counter(cell['analysis_status'] for cell in cells)),
        outcome_status_counts=dict(Counter(cell['status'] for cell in cells)),
        certificates=_certificate_counts(cells),
        metrics={metric:measure(cells, [cell['metrics'][metric] for cell in cells], streams) for metric in METRICS},
        reference_coverage=[_role_coverage(cells, role) for role in ROLES],
        saved_cell_costs={key:dict(known_cells=sum(_number(cell['inherited_scientific_costs'].get(key)) for cell in cells),
            defined_sum=sum(cell['inherited_scientific_costs'].get(key, 0) or 0 for cell in cells),
            all_registered_sum=sum(cell['inherited_scientific_costs'][key] for cell in cells)
                if all(_number(cell['inherited_scientific_costs'].get(key)) for cell in cells) else None)
            for key in cost_keys})


def _paired_references(left, right):
    a = {row['reference_id']: row for row in left['references']}
    b = {row['reference_id']: row for row in right['references']}
    return [dict(reference_id=key, policy_id=(a.get(key) or b[key]).get('policy_id'),
        roles=(a.get(key) or b[key]).get('roles', []),
        left={field:a.get(key, {}).get(field) for field in STAGES},
        right={field:b.get(key, {}).get(field) for field in STAGES},
        left_quality_minus_deployed=a.get(key, {}).get('quality_minus_actual_deployed'),
        right_quality_minus_deployed=b.get(key, {}).get('quality_minus_actual_deployed'))
        for key in dict.fromkeys([*a, *b])]


def _change_category(row, role, stage):
    refs = [ref for ref in row['reference_paths'] if role=='all_initial' or role in ref['roles']]
    if not refs or any(ref[side][stage] is None for ref in refs for side in ('left', 'right')):
        return 'unavailable'
    added = any(ref['left'][stage] and not ref['right'][stage] for ref in refs)
    removed = any(ref['right'][stage] and not ref['left'][stage] for ref in refs)
    return 'changed_both' if added and removed else 'native_added_only' if added else 'role_added_only' if removed else 'same_set'


def _contrast(cells, left, right, protocol, streams, *, same_search):
    indexed = {(cell['seed'], cell['stream'], cell['arm']):cell
               for cell in cells if cell['protocol']==protocol}
    rows = []
    for seed, stream, arm in indexed:
        if arm != left:
            continue
        a, b = indexed[seed, stream, left], indexed[seed, stream, right]
        history_agrees = (None if a['history_arm'] is None or b['history_arm'] is None else
                         a['history_arm']==b['history_arm'])
        if same_search and history_agrees is False:
            raise ValueError('Same-search terminal comparison requires the same history arm')
        cost_keys = sorted({key for source in (a, b) for key, value in source['inherited_scientific_costs'].items()
                            if _number(value)})
        rows.append(dict(identity(a), metrics={m:difference(a['metrics'][m], b['metrics'][m]) for m in METRICS},
            left_metrics=a['metrics'], right_metrics=b['metrics'],
            left_status=a['status'], right_status=b['status'],
            left_analysis_status=a['analysis_status'], right_analysis_status=b['analysis_status'],
            same_history_arm=history_agrees,
            left_deployed=a['scientific_certified_deployed'], right_deployed=b['scientific_certified_deployed'],
            left_policy_id=a['deployed_policy_id'], right_policy_id=b['deployed_policy_id'],
            saved_cost_delta={key:difference(a['inherited_scientific_costs'].get(key),
                                            b['inherited_scientific_costs'].get(key)) for key in cost_keys},
            reference_paths=_paired_references(a, b),
            terminal_structures=dict(left=a['terminal_structure'], right=b['terminal_structure'])))
    result = dict(left=left, right=right, protocol=protocol, same_search=same_search,
        causal_shortlist_effect_identified=False, rows=rows,
        metrics={metric:measure(rows, [row['metrics'][metric] for row in rows], streams) for metric in METRICS},
        by_family=[dict(domain=domain, metrics={metric:measure(selected,
            [row['metrics'][metric] for row in selected], streams) for metric in ('C', 'G')})
            for domain in dict.fromkeys(row['domain'] for row in rows)
            for selected in [[row for row in rows if row['domain']==domain]]])
    if same_search:
        result['reference_change_accounting'] = [dict(role=role, stage=stage,
            categories=[dict(category=category, **_signed_account(selected),
                cases=[dict(identity(row), delta_C=row['metrics']['C'], delta_G=row['metrics']['G']) for row in selected])
                for category in ('native_added_only', 'role_added_only', 'changed_both', 'same_set', 'unavailable')
                for selected in [[row for row in rows if _change_category(row, role, stage)==category]]])
            for role in ROLES for stage in STAGES]
    return result


def aggregate(tasks, plan, *, scientific_summary):
    streams = [row['name'] if isinstance(row, dict) else row for row in plan['streams']]
    arms = [row['name'] if isinstance(row, dict) else row for row in plan['terminal_arms']]
    expected = {(row['geometry_task_seed'], stream) for row in plan['tasks'] for stream in streams}
    if len(streams)!=2 or {(row['seed'], row['stream']) for row in tasks}!=expected or len(tasks)!=len(expected):
        raise ValueError('Both registered stream analyses, including failures, are required')
    scientific = {(row['seed'], stream['name'], row['arm']):row
                  for stream in scientific_summary['streams'] for row in stream['rows']}
    outcomes = {(row['seed'], row['stream'], row['arm'], row['protocol']):row
                for row in scientific_summary['outcomes']}
    expected_outcomes = {(seed, stream, arm, protocol) for seed, stream in expected
                         for arm in arms for protocol in ('old', 'new')}
    if set(outcomes)!=expected_outcomes or len(scientific_summary['outcomes'])!=len(expected_outcomes):
        raise ValueError('The saved scientific grid must retain all controls, protocols and failures')
    cells = []
    for task in tasks:
        indexed = {arm['arm']:arm for arm in task.get('arms', [])}
        if task['status']=='complete' and (set(indexed)!=set(arms) or len(task['arms'])!=len(arms)):
            raise ValueError('A complete analysis must retain every registered control')
        for arm in arms:
            source = indexed.get(arm, {})
            original = scientific[task['seed'], task['stream'], arm]
            for protocol in ('old', 'new'):
                saved = source.get('protocols', {}).get(protocol, {})
                outcome = outcomes[task['seed'], task['stream'], arm, protocol]
                cells.append(dict(identity(task), arm=arm, protocol=protocol,
                    history_arm=source.get('history_arm', original.get('history_arm')),
                    terminal_backend=source.get('terminal_backend'), analysis_status=task['status'],
                    status=original[protocol+'_status'], metrics=deepcopy(original[protocol]),
                    scientific_certified_deployed=original.get(protocol+'_deployed'),
                    scientific_reported_deployed=original.get(protocol+'_reported_deployed'),
                    deployed_policy_id=saved.get('deployed_policy_id'),
                    no_deployment_reason=saved.get('no_deployment_reason'),
                    references=_references(task, saved),
                    other_observed_policies=deepcopy(saved.get('other_observed_policies', [])),
                    terminal_structure=deepcopy(saved.get('terminal_structure', {})),
                    inherited_scientific_costs=deepcopy(outcome.get('costs', {}))))
    failures = [dict(identity(task), status=task['status'], error=deepcopy(task.get('error', task.get('failure'))))
                for task in tasks if task['status']!='complete']
    audit = _contrast(cells, AFTER_NATIVE, AFTER, 'old', streams, same_search=True)
    secondary = [_contrast(cells, left, right, protocol, streams, same_search=True)
        for left, right, protocol in ((AFTER_NATIVE, AFTER, 'new'), (FULL_NATIVE, FULL, 'old'), (FULL_NATIVE, FULL, 'new'))]
    return dict(status='complete_with_analysis_failures' if failures else 'complete',
        registered_geometries=len(plan['tasks']), registered_streams=len(streams),
        registered_task_streams=len(tasks), registered_controls=len(arms), registered_outcomes=len(cells),
        statistical_unit='Average both registered observation streams within geometry before descriptive intervals; reference and task-stream counts are descriptive only.',
        audit_primary=audit, audit_secondary=secondary,
        original_scientific_primary=_contrast(cells, AFTER, POINT, 'old', streams, same_search=False),
        groups=[_group([cell for cell in cells if cell['arm']==arm and cell['protocol']==protocol], streams)
                for arm in arms for protocol in ('old', 'new')],
        cells=cells, reference_snapshots=[dict(identity(task), references=deepcopy(task.get('references', [])),
            reference_source=task.get('reference_source'),
            reference_role_status=deepcopy(task.get('reference_role_status', {}))) for task in tasks],
        analysis_failures=failures, outcome_failures=[deepcopy(row) for row in outcomes.values() if row['status']!='complete'],
        false_certificates=[dict(identity(cell), arm=cell['arm'], protocol=cell['protocol'], metrics=cell['metrics'])
            for cell in cells if cell['metrics']['false_certificate'] is not None and cell['metrics']['false_certificate']>0],
        costs=dict(inherited_scientific_account=deepcopy(scientific_summary.get('costs', {})),
            historical_accounts=deepcopy(plan.get('historical_accounts', [])),
            reference_output_dir=plan.get('reference_output_dir'),
            scientific_output_dir=plan.get('scientific_output_dir'),
            note='Saved cell costs describe paid terminal paths; shared search costs and old/new prefix reuse must not be summed again. New analysis CPU is recorded separately by the runner.'),
        limits=[
            'All initial physical policies were registered before joining truth or outcomes. Existing empirical, STOP primary, STOP safe and STOP shortlist roles are descriptive labels; no label is selected using target truth.',
            'Physical references are deduplicated within each task. Role groups overlap and must not be summed.',
            'Native-minus-role is an actual same-search terminal outcome contrast, not the causal effect of changing only a shortlist. Membership, ordering, verification coverage, paid sample counts, comparison pairs and random paths can change together.',
            'Reference transitions and their signed gain contributions are observational accounting. A certificate from one terminal is never lent to another terminal.',
            'Missing stream metrics remain null; no-certificate paths retain their original C/G definition and are reported separately. All nine controls, failures, signed losses and historical costs remain included.',
            'No simulation, analytic truth, fitting, restored posterior or new candidate scoring is performed.'])


def _compact_measure(value):
    return {key:deepcopy(value[key]) for key in ('task_stream_description', 'geometry_statistics')}


def _compact_contrast(contrast, *, rows):
    result = {key:deepcopy(contrast[key]) for key in ('left', 'right', 'protocol', 'same_search', 'causal_shortlist_effect_identified')}
    result['metrics'] = {metric:_compact_measure(value) for metric, value in contrast['metrics'].items()}
    result['by_family'] = [dict(domain=family['domain'],
        metrics={metric:_compact_measure(value) for metric, value in family['metrics'].items()}) for family in contrast['by_family']]
    if 'reference_change_accounting' in contrast:
        result['reference_change_accounting'] = [dict(role=item['role'], stage=item['stage'],
            categories=[{key:deepcopy(value) for key, value in category.items() if key!='cases'}
                        for category in item['categories'] if category['task_streams']])
            for item in contrast['reference_change_accounting']]
    if rows:
        result['rows'] = []
        for row in contrast['rows']:
            saved = {key:deepcopy(value) for key, value in row.items()
                     if key not in ('terminal_structures', 'reference_paths')}
            saved['changed_reference_paths'] = [dict(reference_id=ref['reference_id'], policy_id=ref['policy_id'],
                roles=ref['roles'], changes={stage:[ref['right'][stage], ref['left'][stage]] for stage in STAGES
                    if ref['right'][stage]!=ref['left'][stage]},
                left_quality_minus_deployed=ref['left_quality_minus_deployed'],
                right_quality_minus_deployed=ref['right_quality_minus_deployed'])
                for ref in row['reference_paths'] if any(ref['right'][stage]!=ref['left'][stage] for stage in STAGES)]
            result['rows'].append(saved)
    return result


def compact_summary(result):
    compact = {key:deepcopy(value) for key, value in result.items()
        if key not in ('audit_primary', 'audit_secondary', 'original_scientific_primary', 'groups', 'cells', 'reference_snapshots', 'costs')}
    compact['audit_primary'] = _compact_contrast(result['audit_primary'], rows=True)
    compact['audit_secondary'] = [_compact_contrast(row, rows=False) for row in result['audit_secondary']]
    compact['original_scientific_primary'] = _compact_contrast(result['original_scientific_primary'], rows=False)
    compact['groups'] = [dict(arm=group['arm'], protocol=group['protocol'],
        analysis_status_counts=group['analysis_status_counts'], outcome_status_counts=group['outcome_status_counts'],
        certificates=group['certificates'], reference_coverage=group['reference_coverage'],
        saved_cell_costs=group['saved_cell_costs'],
        metrics={metric:_compact_measure(group['metrics'][metric]) for metric in ('C', 'G', 'false_certificate')})
        for group in result['groups']]
    compact['costs'] = {key:deepcopy(value) for key, value in result['costs'].items() if key!='inherited_scientific_account'}
    compact['detailed_summary_file'] = 'summary.json'
    return compact
