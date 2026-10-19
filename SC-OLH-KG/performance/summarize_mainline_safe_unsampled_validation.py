"""Report the registered AP-old comparison from already scored, sealed results.

The caller owns scoring, accounting and serialization. This module evaluates no
truth, observations, models or posteriors; it only summarizes saved metric rows.
"""
from copy import deepcopy

from performance.summarize_mainline_two_comparison_confirmation import (
    add_sensitivity, arm_names, leave_largest_positive,
)
from performance.summarize_mainline_two_comparison_development import (
    AFTER, FULL, POINT, METRICS, aggregates, difference, summarize_rows,
)


def contrast(rows, left, right):
    """Pair within geometry before computing any uncertainty or family summary."""
    indexed = {(row['seed'], row['arm']): row for row in rows}
    pairs = []
    for seed in sorted({row['seed'] for row in rows}):
        a, b = indexed[seed, left], indexed[seed, right]
        pairs.append(dict(seed=seed, domain=a['domain'],
            old_contrast=difference(a['old'], b['old']),
            new_contrast=difference(a['new'], b['new']),
            change=difference(a['new_minus_old'], b['new_minus_old'])))
    fields = ('old_contrast', 'new_contrast', 'change')

    def describe(selected):
        result = aggregates(selected, fields)
        for field in fields:
            for metric in METRICS:
                result[field][metric]['leave_largest_positive_geometry_out'] = (
                    leave_largest_positive(selected, field, metric))
        return result

    return dict(left=left, right=right, rows=pairs, **describe(pairs), by_family=[
        dict(domain=domain, **describe([row for row in pairs if row['domain'] == domain]))
        for domain in dict.fromkeys(row['domain'] for row in pairs)])


def _costs(plan, scored, completion, audit_costs):
    """Keep ledger totals and stage breakdowns separate, never sum them twice."""
    account = completion or {}
    work, terminal = account.get('new_work', {}), account.get('terminal_costs', {})
    outcomes = scored['outcomes']
    logical = {}
    for protocol in ('old', 'new'):
        selected = [row for row in outcomes if row['protocol'] == protocol]
        values = [row.get('costs', {}).get('logical_total_calls') for row in selected]
        known = [value for value in values if value is not None]
        logical[protocol] = dict(
            terminal_calls_including_paid_failed_prefixes=terminal.get(protocol + '_logical_calls'),
            registered_cells=len(selected), cells_with_known_logical_total=len(known),
            defined_cell_logical_target_call_sum=sum(known),
            all_registered_logical_target_calls=sum(known) if len(known) == len(selected) else None)
    return dict(
        finalized_account_available=completion is not None,
        physical=dict(total_target_calls=account.get('new_physical_target_calls'),
            initial_calls=work.get('new_target_initial_calls'),
            adaptive_calls=work.get('new_target_search_calls'),
            verification_calls=work.get('new_verification_calls'),
            old_verification_calls=terminal.get('old_physical_calls'),
            additional_new_protocol_calls=terminal.get('additional_new_protocol_physical_calls'),
            unique_paired_verification_calls=terminal.get('unique_paired_physical_calls'),
            recorded_failed_simulation_calls=work.get('failed_simulation_calls_recorded')),
        logical=logical, terminal_account=deepcopy(terminal),
        analytic_truth=dict(recorded_total_function_calls=work.get('new_truth_function_calls'),
            terminal_scoring_function_calls=scored.get('costs', {}).get('scoring_new_truth_function_calls'),
            mechanism_stage=deepcopy(audit_costs),
            accounting='Stage breakdowns are already included where metered in the finalized ledger; do not add them to recorded_total_function_calls again.'),
        recorded_cpu_seconds=account.get('recorded_cpu_seconds'),
        recorded_attempt_wall_seconds=account.get('recorded_attempt_wall_seconds'),
        failed_attempts=deepcopy(account.get('failed_attempts', [])),
        execution_failures=deepcopy(account.get('execution_failures', [])),
        historical_accounts=deepcopy(account.get('inherited_accounts', plan.get('historical_accounts', []))),
        completion_file=plan['output_dir'] + '/completion.json',
        legacy_summary_file=plan['output_dir'] + '/summary.json',
        interpretation='Physical calls count paid observations once; each protocol retains its logical consumption. Analytic truth calls and CPU are separate. Unknown totals are not zero; historical accounts are references, not additional current spending.')


def summarize(plan, terminal_summary, *, completion=None, audit_costs=None):
    """Return AP-old primary plus every registered arm/protocol and failure."""
    rows = terminal_summary['rows']
    arms = arm_names(plan)
    expected = {(task['geometry_task_seed'], arm) for task in plan['tasks'] for arm in arms}
    actual = {(row['seed'], row['arm']) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError('Saved rows must retain every registered geometry-arm cell, including failures')
    outcomes = terminal_summary['outcomes']
    expected_outcomes = {(seed, arm, protocol) for seed, arm in expected for protocol in ('old', 'new')}
    if len(outcomes) != len(expected_outcomes) or {
            (row['seed'], row['arm'], row['protocol']) for row in outcomes} != expected_outcomes:
        raise ValueError('Saved outcomes must retain both protocols for every registered cell')

    groups, existing = summarize_rows(rows, arms)
    add_sensitivity(groups, existing, rows)
    # The inherited terminal summary treats AP as secondary. Promote the frozen
    # old-terminal contrast explicitly; do not promote its new-terminal result.
    ap = contrast(rows, AFTER, POINT)
    fa = contrast(rows, FULL, AFTER)
    controls = [contrast(rows, AFTER, arm) for arm in arms if arm not in (AFTER, POINT, FULL)]
    indexed = {(row['seed'], row['arm']): row for row in rows}
    primary_rows = []
    for pair in ap['rows']:
        seed = pair['seed']
        a, b = indexed[seed, AFTER], indexed[seed, POINT]
        primary_rows.append(dict(seed=seed, domain=pair['domain'], metrics=pair['old_contrast'],
            candidate=a['old'], reference=b['old'],
            candidate_status=a['old_status'], reference_status=b['old_status'],
            quality_reference_status=a['quality_reference_status'],
            best_initial_quality=a['best_initial_quality'],
            candidate_deployed=a.get('old_deployed'), reference_deployed=b.get('old_deployed'),
            candidate_reported_deployed=a.get('old_reported_deployed'),
            reference_reported_deployed=b.get('old_reported_deployed')))
    primary = dict(left=AFTER, right=POINT, protocol='old', co_reported_endpoints=['C', 'G'],
        supporting_endpoints=['A', 'R', 'P', 'L', 'false_certificate'],
        definition=plan['analysis']['primary'], metrics=ap['old_contrast'], rows=primary_rows,
        by_family=[dict(domain=group['domain'], metrics=group['old_contrast']) for group in ap['by_family']])
    failures = [row for row in outcomes if row['status'] != 'complete']
    false_certificates = [dict(seed=row['seed'], domain=row['domain'], arm=row['arm'],
        protocol=protocol, reported_deployed=row.get(protocol + '_reported_deployed'),
        metrics=row[protocol]) for row in rows for protocol in ('old', 'new')
        if row[protocol]['false_certificate'] is not None and row[protocol]['false_certificate'] > 0]
    return dict(status='complete_with_missing_cells' if failures else 'complete',
        registered_geometry_count=len(plan['tasks']), registered_paired_cells=len(expected),
        registered_outcome_cells=len(expected_outcomes),
        completed_outcome_cells=sum(row['status'] == 'complete' for row in outcomes),
        scope=plan['scope'], statistical_unit=plan['statistical_unit'],
        metric_definitions=dict(terminal_summary['metric_definitions'], false_certificate='A-C'),
        primary=primary, secondary=dict(AFTER_minus_POINT=ap, FULL_minus_AFTER=fa,
            AFTER_minus_controls=controls), arms=groups, rows=deepcopy(rows),
        outcomes=deepcopy(outcomes), failures=deepcopy(failures), false_certificates=false_certificates,
        truth_by_task=deepcopy(terminal_summary.get('truth_by_task', [])),
        costs=_costs(plan, terminal_summary, completion, audit_costs),
        limits=[
            'Only the registered fresh geometries enter estimates; source groups and opened cohorts are not additional observations.',
            'C and signed G under the old default terminal are reported together. R retains lost-certificate losses even when G is zero.',
            'Undefined initial baselines and failed executions remain null. Full-cohort estimates are null if incomplete; defined-subset denominators remain explicit.',
            'Family summaries and largest-positive-geometry removal are descriptive; no task is removed from the registered primary estimate.',
            'One stream per known-family geometry does not establish within-geometry noise stability or unseen-family generalization.',
            'All other controls and the new terminal are secondary; no secondary result replaces the AP-old primary.'])


def compact_estimate(value):
    known = value['conditional_available']
    result = dict(registered=value['registered_geometries'], n=known['n'],
        all_registered_mean=value['all_registered_mean'],
        all_registered_ci95=value['all_registered_descriptive_95pct_t_interval'],
        defined_subset_mean=known['mean'], defined_subset_ci95=known['descriptive_95pct_t_interval'],
        positive=known['positive'], zero=known['zero'], negative=known['negative'],
        positive_sum=known['positive_sum'], negative_sum=known['negative_sum'],
        largest_positive_share=known['largest_positive_share'])
    if 'leave_largest_positive_geometry_out' in value:
        result['drop_largest_positive'] = value['leave_largest_positive_geometry_out']
    return result


def _compact_metrics(values, metrics=METRICS):
    return {metric: compact_estimate(values[metric]) for metric in metrics}


def compact_summary(result):
    """Keep all task effects/cell costs; leave detailed terminal diagnostics remote."""
    primary = result['primary']
    costs = deepcopy(result['costs'])
    costs['failed_attempt_count'] = len(costs.pop('failed_attempts'))
    arms = []
    for group in result['arms']:
        arm = dict(arm=group['arm'], registered=group['registered'], complete=group['complete'],
            **{field: _compact_metrics(group[field], ('C', 'R', 'G'))
                for field in ('old', 'new', 'new_minus_old')})
        arm['defined_subset_totals'] = {field: {metric: dict(
            n=group[field][metric]['conditional_available']['n'],
            sum=group[field][metric]['conditional_available']['positive_sum'] +
                group[field][metric]['conditional_available']['negative_sum'])
            for metric in ('A', 'P', 'L', 'false_certificate')} for field in ('old', 'new')}
        if group['arm'] == AFTER:
            arm.update({field: _compact_metrics(group[field], ('C', 'R', 'G'))
                for field in ('third_challenge', 'prechange')})
        arms.append(arm)

    def compact_contrast(group, *, family=True):
        result = dict(left=group['left'], right=group['right'], **{
            field: _compact_metrics(group[field], ('C', 'R', 'G'))
            for field in ('old_contrast', 'new_contrast', 'change')})
        if family:
            result['by_family'] = [dict(domain=item['domain'], **{
                field: _compact_metrics(item[field], ('C', 'R', 'G'))
                for field in ('old_contrast', 'new_contrast')}) for item in group['by_family']]
        return result

    tasks = []
    for row in primary['rows']:
        seed = row['seed']
        cells = []
        for source in result['rows']:
            if source['seed'] != seed:
                continue
            cells.append(dict(arm=source['arm'], old_status=source['old_status'],
                new_status=source['new_status'], old=source['old'], new=source['new'],
                new_minus_old=source['new_minus_old'], old_deployed=source.get('old_deployed'),
                new_deployed=source.get('new_deployed')))
        accounts = [dict(arm=outcome['arm'], protocol=outcome['protocol'], status=outcome['status'],
            costs=outcome.get('costs'), **({} if outcome['status'] == 'complete' else dict(
                files=outcome.get('files'), reason=outcome.get('reason'))))
            for outcome in result['outcomes'] if outcome['seed'] == seed]
        tasks.append(dict(row, arms=cells, cell_accounts=accounts))
    return dict(status=result['status'], registered_geometry_count=result['registered_geometry_count'],
        registered_outcome_cells=result['registered_outcome_cells'],
        completed_outcome_cells=result['completed_outcome_cells'], statistical_unit=result['statistical_unit'],
        primary=dict(left=primary['left'], right=primary['right'], protocol='old',
            co_reported_endpoints=primary['co_reported_endpoints'], metrics=_compact_metrics(primary['metrics']),
            by_family=[dict(domain=family['domain'], metrics=_compact_metrics(family['metrics']))
                for family in primary['by_family']]),
        secondary=dict(AFTER_minus_POINT=compact_contrast(result['secondary']['AFTER_minus_POINT']),
            FULL_minus_AFTER=compact_contrast(result['secondary']['FULL_minus_AFTER']),
            AFTER_minus_controls=[compact_contrast(group, family=False)
                for group in result['secondary']['AFTER_minus_controls']]),
        arms=arms, tasks=tasks, failures=result['failures'], false_certificates=result['false_certificates'],
        costs=costs, detailed_summary_file='mainline_summary.json', limits=result['limits'])
