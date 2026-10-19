"""Describe saved decision paths without scoring, fitting or restoring a model."""
from collections import Counter
from copy import deepcopy

from performance.summarize_mainline_two_comparison_development import (
    AFTER, FULL, POINT, METRICS, estimate,
)


LAYERS = ('search', 'shortlist', 'coverage', 'safety', 'comparison', 'terminal_total', 'total')
EPS = 1e-12


def _mean(values):
    return None if not values or any(value is None for value in values) else sum(values)/len(values)


def _difference(left, right):
    return None if left is None or right is None else left-right


def _identity(row):
    return {key: row[key] for key in ('seed', 'domain', 'stream')}


def _describe(rows, values):
    """Counts are task-stream descriptions, not 48 independent geometries."""
    known = [(row, value) for row, value in zip(rows, values) if value is not None]
    positive = [(row, value) for row, value in known if value > EPS]
    negative = [(row, value) for row, value in known if value < -EPS]
    largest = max(positive, key=lambda pair: pair[1]) if positive else None
    return dict(registered_task_streams=len(rows), defined_task_streams=len(known),
        undefined_task_streams=len(rows)-len(known), positive=len(positive), negative=len(negative),
        zero=len(known)-len(positive)-len(negative),
        positive_sum=sum(value for _, value in positive), negative_sum=sum(value for _, value in negative),
        largest_positive=None if largest is None else dict(_identity(largest[0]), value=largest[1]),
        undefined=[_identity(row) for row, value in zip(rows, values) if value is None])


def _geometry_rows(rows, values, streams):
    grouped = {}
    for row, value in zip(rows, values):
        grouped.setdefault(row['seed'], dict(domain=row['domain'], streams={}))["streams"][row['stream']] = value
    return [dict(seed=seed, domain=group['domain'],
        value=_mean([group['streams'].get(stream) for stream in streams]),
        stream_values={stream: group['streams'].get(stream) for stream in streams})
        for seed, group in sorted(grouped.items())]


def _estimate(rows):
    raw = estimate([row['value'] for row in rows])
    known = raw['conditional_available']
    positive = [row for row in rows if row['value'] is not None and row['value'] > EPS]
    largest = max(positive, key=lambda row: (row['value'], -row['seed'])) if positive else None
    remaining = [row['value'] for row in rows if largest is not None and row['seed'] != largest['seed']]
    defined_remaining = [value for value in remaining if value is not None]
    return dict(registered_geometries=len(rows), defined_geometries=known['n'],
        all_registered_mean=raw['all_registered_mean'],
        all_registered_ci95=raw['all_registered_descriptive_95pct_t_interval'],
        defined_subset_mean=known['mean'], defined_subset_ci95=known['descriptive_95pct_t_interval'],
        positive=known['positive'], zero=known['zero'], negative=known['negative'],
        positive_sum=known['positive_sum'], negative_sum=known['negative_sum'],
        largest_positive_share=known['largest_positive_share'],
        largest_positive=None if largest is None else dict(seed=largest['seed'], domain=largest['domain'], value=largest['value']),
        leave_largest_positive_out=None if largest is None else dict(excluded_seed=largest['seed'],
            all_registered_mean=_mean(remaining),
            defined_subset_mean=_mean(defined_remaining), defined_geometries=len(defined_remaining),
            primary_sample_changed=False),
        undefined_seeds=[row['seed'] for row in rows if row['value'] is None])


def measure(rows, values, streams):
    geometries = _geometry_rows(rows, values, streams)
    return dict(task_stream_description=_describe(rows, values),
        geometry_statistics=_estimate(geometries), geometry_rows=geometries)


def _set_summary(value):
    return {key: deepcopy(value.get(key)) for key in (
        'status', 'count', 'truth_known', 'truth_unknown', 'true_safe_count', 'max_quality', 'best_policy_ids')}


def _cell(task, arm, protocol):
    saved = arm.get('protocols', {}).get(protocol, {})
    return dict(seed=task['seed'], domain=task['domain'], stream=task['stream'],
        arm=arm['arm'], history_arm=arm.get('history_arm'), terminal_backend=arm.get('terminal_backend'),
        protocol=protocol, analysis_status=task['status'],
        status=saved.get('status', 'analysis_unavailable'), search_status=arm.get('search_status'),
        metrics={metric:saved.get('metrics', {}).get(metric) for metric in METRICS},
        opportunity_gaps={layer:saved.get('opportunity_gaps', {}).get(layer) for layer in LAYERS},
        sets={name:_set_summary(value) for name, value in {
            **arm.get('sets', {}), **saved.get('sets', {})}.items()},
        discovery=deepcopy(arm.get('discovery', {})),
        deployed_policy_id=saved.get('deployed_policy_id'),
        no_deployment_reason=saved.get('no_deployment_reason'),
        counts=deepcopy(saved.get('counts', {})),
        gap_unavailable_reasons=deepcopy(saved.get('gap_unavailable_reasons', {})),
        nesting_issues=deepcopy(saved.get('nesting_issues', [])))


def _certificate_categories(rows):
    counts = Counter()
    reasons = Counter()
    examples = []
    for row in rows:
        actual, true = row['metrics']['A'], row['metrics']['C']
        category = ('unavailable' if actual is None or true is None else
            'no_certificate' if not actual else 'true_certificate' if true else 'false_certificate')
        counts[category] += 1
        if category != 'true_certificate':
            reason = row.get('no_deployment_reason')
            reasons[str(reason)] += 1
            examples.append(dict(_identity(row), category=category, reason=deepcopy(reason)))
    return dict(counts=dict(counts), reasons=dict(reasons), cases=examples)


def _common_support(rows, streams):
    """Compare opportunities only where the same task-stream defines both."""
    search, terminal, difference = [], [], []
    for row in rows:
        s, t = (row['opportunity_gaps'][name] for name in ('search', 'terminal_total'))
        paired = all(row['opportunity_gaps'][layer] is not None
            for layer in ('search', 'shortlist', 'coverage', 'safety', 'comparison')) and t is not None
        search.append(s if paired else None)
        terminal.append(t if paired else None)
        difference.append(s-t if paired else None)
    return dict(search=measure(rows, search, streams), terminal=measure(rows, terminal, streams),
        search_minus_terminal=measure(rows, difference, streams),
        interpretation='Both quantities use the same defined task-streams. They are saved-set oracle opportunities, not attainable gains or causal loss percentages.')


def _group(rows, streams):
    result = dict(arm=rows[0]['arm'], protocol=rows[0]['protocol'], registered_task_streams=len(rows),
        outcome_status_counts=dict(Counter(row['status'] for row in rows)),
        analysis_status_counts=dict(Counter(row['analysis_status'] for row in rows)),
        certificate_categories=_certificate_categories(rows),
        policy_path_counts={field:dict(sum((Counter(row['counts'].get(field, {})) for row in rows), Counter()))
            for field in ('dispositions', 'improving_safe_dispositions', 'better_than_deployed_dispositions')},
        nesting_issue_cases=[dict(_identity(row), issues=row['nesting_issues']) for row in rows if row['nesting_issues']],
        metrics={metric:measure(rows, [row['metrics'][metric] for row in rows], streams) for metric in METRICS},
        opportunity_gaps={layer:measure(rows, [row['opportunity_gaps'][layer] for row in rows], streams) for layer in LAYERS},
        query_discovery_gain=measure(rows, [row['discovery'].get('quality_gain') for row in rows], streams),
        search_vs_terminal_common_support=_common_support(rows, streams))
    result['by_family'] = [dict(domain=domain,
        opportunity_gaps={layer:measure(selected, [row['opportunity_gaps'][layer] for row in selected], streams)
            for layer in LAYERS},
        search_vs_terminal_common_support=_common_support(selected, streams),
        certificate_categories=_certificate_categories(selected))
        for domain in dict.fromkeys(row['domain'] for row in rows)
        for selected in [[row for row in rows if row['domain']==domain]]]
    return result


def _contrast(cells, left, right, protocol, streams):
    indexed = {(row['seed'], row['stream'], row['arm']): row for row in cells if row['protocol']==protocol}
    rows = []
    for seed, stream, arm in indexed:
        if arm != left:
            continue
        a, b = indexed[seed, stream, left], indexed[seed, stream, right]
        rows.append(dict(_identity(a), metrics={metric:_difference(a['metrics'][metric], b['metrics'][metric])
            for metric in METRICS}, left_status=a['status'], right_status=b['status'],
            left_metrics=a['metrics'], right_metrics=b['metrics'],
            left_deployed=a['scientific_certified_deployed'], right_deployed=b['scientific_certified_deployed'],
            left_path_policy_id=a['deployed_policy_id'], right_path_policy_id=b['deployed_policy_id'],
            discovery_quality_gain_difference=_difference(a['discovery'].get('quality_gain'), b['discovery'].get('quality_gain'))))
    return dict(left=left, right=right, protocol=protocol, rows=rows,
        metrics={metric:measure(rows, [row['metrics'][metric] for row in rows], streams) for metric in METRICS},
        discovery_quality_gain_difference=measure(rows, [row['discovery_quality_gain_difference'] for row in rows], streams),
        by_family=[dict(domain=domain, metrics={metric:measure(selected,
            [row['metrics'][metric] for row in selected], streams) for metric in METRICS})
            for domain in dict.fromkeys(row['domain'] for row in rows)
            for selected in [[row for row in rows if row['domain']==domain]]])


def aggregate(tasks, plan, *, scientific_summary):
    """Aggregate 48 saved task analyses; failures retain all registered cells."""
    streams = [row['name'] if isinstance(row, dict) else row for row in plan['streams']]
    arms = [row['name'] if isinstance(row, dict) else row for row in plan['terminal_arms']]
    expected = {(row['geometry_task_seed'], stream) for row in plan['tasks'] for stream in streams}
    actual = {(row['seed'], row['stream']) for row in tasks}
    if len(streams) != 2 or actual != expected or len(tasks) != len(expected):
        raise ValueError('Every registered geometry requires both stream analyses, including failures')
    cells = []
    for task in tasks:
        indexed = {arm['arm']:arm for arm in task.get('arms', [])}
        if task['status'] == 'complete' and (set(indexed) != set(arms) or len(task['arms']) != len(arms)):
            raise ValueError('Complete task analysis must retain all registered controls')
        for arm in arms:
            source = indexed.get(arm, dict(arm=arm))
            cells.extend(_cell(task, source, protocol) for protocol in ('old', 'new'))
    # An attribution reader failure must not erase a completed scientific cell.
    # Scientific metrics/status/costs are copied from the already sealed summary.
    scientific = {(row['seed'], stream['name'], row['arm']):row
        for stream in scientific_summary['streams'] for row in stream['rows']}
    outcomes = {(row['seed'], row['stream'], row['arm'], row['protocol']):row
        for row in scientific_summary['outcomes']}
    expected_outcomes = {(seed, stream, arm, protocol) for seed, stream in expected
                         for arm in arms for protocol in ('old', 'new')}
    if set(outcomes) != expected_outcomes or len(scientific_summary['outcomes']) != len(expected_outcomes):
        raise ValueError('Saved scientific summary must retain every registered outcome')
    for cell in cells:
        key = cell['seed'], cell['stream'], cell['arm']
        source, outcome = scientific[key], outcomes[(*key, cell['protocol'])]
        cell['metrics'] = deepcopy(source[cell['protocol']])
        cell['status'] = source[cell['protocol']+'_status']
        cell['scientific_reported_deployed'] = source.get(cell['protocol']+'_reported_deployed')
        cell['scientific_certified_deployed'] = source.get(cell['protocol']+'_deployed')
        cell['inherited_scientific_costs'] = deepcopy(outcome.get('costs', {}))
    groups = [_group([row for row in cells if row['arm']==arm and row['protocol']==protocol], streams)
              for arm in arms for protocol in ('old', 'new')]
    primary = _contrast(cells, AFTER, POINT, 'old', streams)
    secondary = [_contrast(cells, AFTER, POINT, 'new', streams)]
    if FULL in arms:
        secondary.extend(_contrast(cells, FULL, AFTER, protocol, streams) for protocol in ('old', 'new'))
    failures = [dict(_identity(row), status=row['status'], failure=deepcopy(row.get('failure', row.get('error'))))
                for row in tasks if row['status'] != 'complete']
    outcome_failures = [row for row in cells if row['status'] not in ('complete', 'analysis_unavailable')]
    return dict(status='complete_with_analysis_failures' if failures else 'complete',
        statistical_unit='Two streams are averaged within each registered geometry before descriptive uncertainty summaries; task-stream counts are descriptive.',
        registered_geometries=len(plan['tasks']), registered_streams=len(streams),
        registered_task_streams=len(tasks), registered_controls=len(arms), registered_outcomes=len(cells),
        primary=primary, secondary=secondary, groups=groups, cells=cells,
        analysis_failures=failures, outcome_failures=outcome_failures,
        false_certificates=[row for row in cells if row['metrics']['false_certificate'] is not None and row['metrics']['false_certificate']>0],
        focal_cases=[deepcopy(task) for task in tasks if task['seed'] in (2026107017,2026107013)],
        costs={key:deepcopy(plan[key]) for key in ('source_manifest', 'source_completion',
            'reference_manifest', 'reference_output_dir', 'source_costs', 'inherited_accounts', 'historical_accounts') if key in plan},
        limits=[
            'Opportunity gaps compare maxima on actually saved sets. They are neither achievable gains nor causal fractions of deployment loss.',
            'A missing feasible maximum, incomplete truth coverage, or absence of a true-certified deployment leaves the relevant quality gap undefined. No certificate is classified separately; baseline quality is never substituted for deployment.',
            'Search-versus-terminal comparisons use common task-stream support, and a geometry average is undefined if either required stream is undefined.',
            'Full-pool search opportunities are available only for histories with their own saved pool. Other controls do not borrow a role backend pool.',
            'All nine controls, old and new terminals, positive and negative effects, failures and prior fees remain included. The old AFTER-minus-POINT C and signed G comparison is primary.',
            'This analysis evaluates no new truth, simulation, fitted model or restored posterior. New analysis CPU is recorded by the caller separately from inherited scientific costs.'])


def _compact_measure(value):
    return {key:deepcopy(value[key]) for key in ('task_stream_description', 'geometry_statistics')}


def _compact_group(group):
    result = {key:deepcopy(group[key]) for key in ('arm', 'protocol', 'registered_task_streams',
        'outcome_status_counts', 'analysis_status_counts', 'certificate_categories',
        'policy_path_counts', 'nesting_issue_cases')}
    for field in ('metrics', 'opportunity_gaps'):
        result[field] = {key:_compact_measure(value) for key, value in group[field].items()}
    result['query_discovery_gain'] = _compact_measure(group['query_discovery_gain'])
    result['search_vs_terminal_common_support'] = {key:_compact_measure(value) if isinstance(value, dict) else value
        for key, value in group['search_vs_terminal_common_support'].items()}
    result['by_family'] = [dict(domain=family['domain'],
        opportunity_gaps={key:_compact_measure(value) for key, value in family['opportunity_gaps'].items()},
        search_vs_terminal_common_support={key:_compact_measure(value) if isinstance(value, dict) else value
            for key, value in family['search_vs_terminal_common_support'].items()},
        certificate_categories=family['certificate_categories']) for family in group['by_family']]
    return result


def compact_summary(result):
    """Leave per-policy details server-side; keep every arm and signed contrast."""
    compact = {key:deepcopy(value) for key, value in result.items()
        if key not in ('groups', 'cells', 'focal_cases')}
    compact['groups'] = [_compact_group(group) for group in result['groups']]
    compact['AFTER_old_task_streams'] = [deepcopy(row) for row in result['cells']
        if row['arm']==AFTER and row['protocol']=='old']
    compact['focal_cells'] = [deepcopy(row) for row in result['cells'] if row['seed'] in (2026107017,2026107013)]
    return compact
