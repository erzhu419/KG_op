"""Deterministic precision planning from saved geometry-level paired effects."""
from math import sqrt
from statistics import mean, variance

from scipy.stats import t


AFTER = 'legacy_role_after_init_point_only'
POINT = 'legacy_role_point_only'
ENDPOINTS = ('C', 'G')


def _effect(unit, protocol_key, protocol):
    rows = unit.get('comparisons', [unit])
    return next(row for row in rows if row['left'] == AFTER and row['right'] == POINT
                and row[protocol_key] == protocol)


def _rows(effect):
    rows = [dict(seed=row['seed'], domain=row['domain'],
                 delta={metric: row['delta'][metric] for metric in ENDPOINTS})
            for row in effect['rows']]
    if len({row['seed'] for row in rows}) != len(rows):
        raise ValueError('Geometry effects must retain one row per registered geometry')
    return rows


def _interval(center, halfwidth):
    return None if center is None or halfwidth is None else [center - halfwidth, center + halfwidth]


def _halfwidth(sample_variance, n):
    return None if sample_variance is None or n < 2 else float(t.ppf(.975, n - 1)) * sqrt(sample_variance / n)


def metric_summary(rows, metric):
    available = [row for row in rows if row['delta'][metric] is not None]
    values = [row['delta'][metric] for row in available]
    center = mean(values) if values else None
    sample_variance = variance(values) if len(values) > 1 else None
    halfwidth = _halfwidth(sample_variance, len(values))
    complete = len(available) == len(rows)
    return dict(registered_geometries=len(rows), defined_geometries=len(values),
        missing_seeds=[row['seed'] for row in rows if row['delta'][metric] is None],
        all_registered_mean=center if complete else None,
        all_registered_descriptive_95pct_t_interval=_interval(center, halfwidth) if complete else None,
        conditional_available=dict(n=len(values), mean=center, sample_variance=sample_variance,
            sample_standard_deviation=None if sample_variance is None else sqrt(sample_variance),
            degrees_of_freedom=max(0, len(values) - 1), halfwidth=halfwidth,
            descriptive_95pct_t_interval=_interval(center, halfwidth)))


def project_precision(summary, candidates, target, *, coverage_adjusted):
    available, registered = summary['defined_geometries'], summary['registered_geometries']
    rate = available / registered
    saved = summary['conditional_available']
    projections = []
    for n in candidates:
        # Integer arithmetic avoids roundoff at exact historical coverage multiples.
        usable = (n * available) // registered if coverage_adjusted else n
        halfwidth = _halfwidth(saved['sample_variance'], usable)
        projections.append(dict(registered_geometries=n, planned_usable_geometry_effects=usable,
            historical_coverage_rate=rate, degrees_of_freedom=max(0, usable - 1),
            projected_95pct_halfwidth=halfwidth, planning_halfwidth_target=target,
            meets_precision_target=None if halfwidth is None else halfwidth <= target,
            projected_interval_at_saved_mean=_interval(saved['mean'], halfwidth)))
    return projections


def cohort_summary(name, rows, plan, *, stream_count, selection_eligible):
    metrics = {metric: metric_summary(rows, metric) for metric in ENDPOINTS}
    projections = {metric: {scenario: project_precision(metrics[metric], plan['candidate_geometries'],
        plan['halfwidth_targets'][metric], coverage_adjusted=scenario == 'historical_coverage')
        for scenario in ('historical_coverage', 'all_defined')} for metric in ENDPOINTS}
    positive = [row for row in rows if row['delta']['G'] is not None and row['delta']['G'] > 1e-12]
    sensitivity = dict(status='no_positive_gain', used_for_selection=False)
    if positive:
        largest = max(positive, key=lambda row: (row['delta']['G'], -row['seed']))
        remaining = [row for row in rows if row['seed'] != largest['seed']]
        sensitivity = dict(status='descriptive_only', used_for_selection=False,
            excluded_seed=largest['seed'], excluded_gain=largest['delta']['G'],
            original_registered_geometries=len(rows), remaining_registered_geometries=len(remaining),
            metrics={metric: metric_summary(remaining, metric) for metric in ENDPOINTS})
    return dict(name=name, stream_count_per_geometry=stream_count, selection_eligible=selection_eligible,
        registered_geometries=len(rows), rows=rows, metrics=metrics, projections=projections,
        family_metrics=[dict(domain=domain, metrics={metric: metric_summary(
            [row for row in rows if row['domain'] == domain], metric) for metric in ENDPOINTS})
            for domain in dict.fromkeys(row['domain'] for row in rows)],
        largest_positive_gain_sensitivity=sensitivity,
        planning_scope='Simple geometry-level sample SD; family summaries do not replace the estimand. '
            'Historical-coverage projections concern available baseline-defined effects, not an imputed full-cohort mean. '
            'The all-defined scenario assumes future baseline availability. No variance reduction is assumed for adding a second stream.')


def stream_difference_summary(first, second, averaged):
    indices = [{row['seed']: row for row in rows} for rows in (first, second, averaged)]
    if not (set(indices[0]) == set(indices[1]) == set(indices[2])):
        raise ValueError('Both streams and their mean must retain the same geometry set')
    paired = []
    for seed, a in indices[0].items():
        b, saved = indices[1][seed], indices[2][seed]
        if a['domain'] != b['domain'] or a['domain'] != saved['domain']:
            raise ValueError('Paired streams must use the same geometry family')
        values = {}
        for metric in ENDPOINTS:
            one, two = a['delta'][metric], b['delta'][metric]
            expected = None if one is None or two is None else (one + two) / 2
            actual = saved['delta'][metric]
            if (expected is None) != (actual is None) or (expected is not None and abs(expected - actual) > 1e-12):
                raise ValueError('The saved independent effect must be the within-geometry two-stream mean')
            values[metric] = dict(stream_effects=[one, two], saved_geometry_mean=actual,
                difference=None if expected is None else one - two)
        paired.append(dict(seed=seed, domain=a['domain'], metrics=values))
    metrics = {}
    for metric in ENDPOINTS:
        values = [row['metrics'][metric]['difference'] for row in paired
                  if row['metrics'][metric]['difference'] is not None]
        metrics[metric] = dict(registered_geometries=len(paired), defined_pairs=len(values),
            missing_seeds=[row['seed'] for row in paired if row['metrics'][metric]['difference'] is None],
            mean_stream_difference=mean(values) if values else None,
            mean_squared_stream_difference=mean(value * value for value in values) if values else None,
            half_mean_squared_difference_proxy=mean(value * value for value in values) / 2 if values else None,
            saved_geometry_mean_sample_variance=metric_summary(averaged, metric)['conditional_available']['sample_variance'])
    return dict(rows=paired, metrics=metrics,
        identification='With independent identically distributed repetitions conditional on geometry, half the mean squared '
            'stream difference estimates an average composite stream variance. Observations, adaptive query paths and terminal '
            'verification all vary together; two repetitions do not separate simulator noise, search noise and verification noise. '
            'No between-geometry variance is obtained by subtraction, and no sqrt(2) precision gain is assumed.')


def analyze_precision(stage83_readout, stage82_readout, plan):
    """Read only existing effects; select N using full registered cohorts' saved variance."""
    candidates = plan['candidate_geometries']
    if not candidates or candidates != sorted(set(candidates)) or any(n < 3 or n % 3 for n in candidates):
        raise ValueError('Candidate geometry counts must increase and balance the three families')
    if any(plan['halfwidth_targets'][metric] <= 0 for metric in ENDPOINTS):
        raise ValueError('Planning halfwidth targets must be positive')
    if stage83_readout['status'] != 'complete' or stage82_readout['status'] != 'complete':
        raise ValueError('Both saved scientific analyses must already be complete')
    old = _rows(_effect(stage83_readout['independent'], 'protocol', 'old'))
    streams = {unit['name']: unit for unit in stage83_readout['streams']}
    first, second = [_rows(_effect(streams[name], 'protocol', 'old')) for name in ('observation_1', 'observation_2')]
    paired = stream_difference_summary(first, second, old)
    reference = _rows(_effect(streams['reference'], 'protocol', 'old'))
    newer = _rows(_effect(stage82_readout['primary'], 'variant', 'original_old'))
    cohorts = [cohort_summary('old_geometries_two_independent_streams', old, plan, stream_count=2, selection_eligible=True),
               cohort_summary('new_geometries_one_stream', newer, plan, stream_count=1, selection_eligible=True)]
    selection = []
    for index, n in enumerate(candidates):
        requirements = [dict(cohort=cohort['name'], endpoint=metric,
            **cohort['projections'][metric]['historical_coverage'][index]) for cohort in cohorts for metric in ENDPOINTS]
        selection.append(dict(registered_geometries=n, geometries_per_family=n // 3,
            precision_planning_pass=all(row['meets_precision_target'] is True for row in requirements), requirements=requirements))
    selected = next((row['registered_geometries'] for row in selection if row['precision_planning_pass']), None)
    return dict(status='selected' if selected is not None else 'hold', selected_N=selected,
        selection_pass=selected is not None, candidate_geometries=candidates, halfwidth_targets=plan['halfwidth_targets'],
        selection_scenario='historical_coverage', selection_table=selection, cohorts=cohorts,
        reference_context=cohort_summary('old_reference_stream', reference, plan, stream_count=1, selection_eligible=False),
        within_geometry_stream_differences=paired,
        selection_rule='First registered candidate N meeting both C and G plug-in halfwidth targets separately in both cohorts. '
            'Keep historical coverage, all signed effects and all registered geometries. Reference results, mean signs and '
            'leave-largest-positive sensitivity do not select N. No extrapolation beyond the registered candidate grid.',
        limits=['Projected halfwidths are planning quantities, not utility thresholds or scientific acceptance criteria.',
            'Plug-in variances and historical baseline coverage may change on fresh geometries; projected precision is not guaranteed.',
            'The old two-stream cohort reused the Stage73 shared64 candidate bank, so its variance is conditional on that bank. '
            'Future streams will each generate their bank from their own STOP observations with the frozen generator; '
            'there is no observation-to-bank dependency across streams. The old variance does not guarantee precision under this change.',
            'The two cohorts and observation repetitions are not pooled as independent geometries.',
            'Undefined baselines stay undefined; conditional precision does not identify a full-cohort G mean.',
            'Future two-stream execution receives no assumed sqrt(2) gain from the single-stream cohort.'],
        new_work=dict(simulation=0, verification=0, truth_fields=0, candidate_scores=0, source_fit=0,
                      model_restore=0, posterior_restore=0, raw_replay=0, random_draws=0))
