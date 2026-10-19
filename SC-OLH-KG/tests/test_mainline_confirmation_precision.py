"""Precision choices use geometry variance, never outcome signs or winner removal."""
from copy import deepcopy
from math import sqrt
from statistics import variance

import pytest
from scipy.stats import t

from performance import mainline_confirmation_precision as precision


def plan():
    return dict(candidate_geometries=[24, 48, 72, 96, 120, 144, 192, 288, 384],
                halfwidth_targets=dict(C=.05, G=.0005))


def rows(values, *, offset=0, certificates=None):
    certificates = certificates or [0.] * len(values)
    return [dict(seed=offset + i + 1, domain=('Inventory', 'Queue', 'Factor')[i % 3],
                 delta=dict(C=certificates[i], G=value)) for i, value in enumerate(values)]


def comparison(records, protocol='old'):
    return dict(left=precision.AFTER, right=precision.POINT, protocol=protocol, rows=records)


def fixtures():
    values = [-.004, -.002, 0., .001, .002, .003]
    old = rows(values)
    first = rows([value + .0002 for value in values])
    second = rows([value - .0002 for value in values])
    old_readout = dict(status='complete', independent=dict(comparisons=[comparison(old)]),
        streams=[dict(name=name, comparisons=[comparison(records)]) for name, records in (
            ('reference', rows([.1] * 6)), ('observation_1', first), ('observation_2', second))])
    new_readout = dict(status='complete', primary=dict(left=precision.AFTER, right=precision.POINT,
        variant='original_old', rows=rows(values, offset=100)))
    return old_readout, new_readout


def test_geometry_means_are_the_units_and_two_stream_average_is_required():
    old, new = fixtures()
    result = precision.analyze_precision(old, new, plan())
    assert [row['registered_geometries'] for row in result['cohorts']] == [6, 6]
    assert result['within_geometry_stream_differences']['metrics']['G']['defined_pairs'] == 6
    assert result['cohorts'][0]['metrics']['G']['conditional_available']['sample_variance'] == pytest.approx(
        variance([-.004, -.002, 0., .001, .002, .003]))
    old['independent']['comparisons'][0]['rows'][0]['delta']['G'] += .01
    with pytest.raises(ValueError, match='within-geometry two-stream mean'):
        precision.analyze_precision(old, new, plan())


def test_missing_baseline_preserves_registered_units_and_conservative_coverage():
    old, new = fixtures()
    old['independent']['comparisons'][0]['rows'][1]['delta']['G'] = None
    for stream in old['streams']:
        if stream['name'] != 'reference':
            stream['comparisons'][0]['rows'][1]['delta']['G'] = None
    result = precision.analyze_precision(old, new, plan())
    cohort = result['cohorts'][0]
    metric = cohort['metrics']['G']
    assert metric['registered_geometries'] == 6 and metric['defined_geometries'] == 5
    assert metric['missing_seeds'] == [2]
    assert metric['all_registered_mean'] is None
    assert metric['all_registered_descriptive_95pct_t_interval'] is None
    conservative = cohort['projections']['G']['historical_coverage'][0]
    assert conservative['registered_geometries'] == 24
    assert conservative['planned_usable_geometry_effects'] == 20
    assert cohort['projections']['G']['all_defined'][0]['planned_usable_geometry_effects'] == 24
    assert result['cohorts'][1]['metrics']['G']['defined_geometries'] == 6
    assert result['within_geometry_stream_differences']['metrics']['G']['defined_pairs'] == 5


def test_first_common_precision_pass_ignores_mean_sign_and_reference_results():
    old, new = fixtures()
    baseline = precision.analyze_precision(old, new, plan())
    assert baseline['selection_pass'] and baseline['selected_N'] in plan()['candidate_geometries']
    selected = next(row for row in baseline['selection_table'] if row['registered_geometries'] == baseline['selected_N'])
    assert len(selected['requirements']) == 4 and all(row['meets_precision_target'] for row in selected['requirements'])
    assert all(not row['precision_planning_pass'] for row in baseline['selection_table']
               if row['registered_geometries'] < baseline['selected_N'])
    shifted_old, shifted_new = deepcopy(old), deepcopy(new)
    for unit in [shifted_old['independent'], *shifted_old['streams']]:
        for row in unit['comparisons'][0]['rows']:
            row['delta']['G'] -= 1.
    for row in shifted_new['primary']['rows']:
        row['delta']['G'] -= 1.
    shifted_old['streams'][0]['comparisons'][0]['rows'][0]['delta']['G'] = 1.e6
    shifted = precision.analyze_precision(shifted_old, shifted_new, plan())
    assert shifted['selected_N'] == baseline['selected_N']
    assert shifted['reference_context']['selection_eligible'] is False
    assert shifted['cohorts'][0]['metrics']['G']['all_registered_mean'] < 0.


def test_leave_largest_positive_is_sensitivity_only_and_grid_exhaustion_holds():
    old, new = fixtures()
    for unit in [old['independent'], *old['streams']]:
        for row in unit['comparisons'][0]['rows']:
            row['delta']['G'] = .1 if row['seed'] == 6 else 0.
    result = precision.analyze_precision(old, new, plan())
    assert result['status'] == 'hold' and result['selected_N'] is None and not result['selection_pass']
    sensitivity = result['cohorts'][0]['largest_positive_gain_sensitivity']
    assert sensitivity['excluded_seed'] == 6 and sensitivity['used_for_selection'] is False
    assert sensitivity['metrics']['G']['conditional_available']['sample_variance'] == 0.
    assert result['cohorts'][0]['metrics']['G']['conditional_available']['sample_variance'] > 0.
    assert result['candidate_geometries'][-1] == 384


def test_single_stream_cohort_gets_no_assumed_second_stream_variance_reduction():
    old, new = fixtures()
    result = precision.analyze_precision(old, new, plan())
    cohort = result['cohorts'][1]
    sample_variance = variance([-.004, -.002, 0., .001, .002, .003])
    projection = cohort['projections']['G']['historical_coverage'][0]
    expected = t.ppf(.975, 23) * sqrt(sample_variance / 24)
    assert projection['projected_95pct_halfwidth'] == pytest.approx(expected)
    assert cohort['stream_count_per_geometry'] == 1


def test_stream_difference_proxy_is_not_a_pure_simulator_noise_estimate():
    old, new = fixtures()
    result = precision.analyze_precision(old, new, plan())
    differences = result['within_geometry_stream_differences']
    metric = differences['metrics']['G']
    assert metric['mean_stream_difference'] == pytest.approx(.0004)
    assert metric['half_mean_squared_difference_proxy'] == pytest.approx(.0004 ** 2 / 2)
    assert 'do not separate simulator noise' in differences['identification']
    assert all(value == 0 for value in result['new_work'].values())


def test_incomplete_source_and_duplicate_geometries_do_not_enter_precision_planning():
    old, new = fixtures()
    new['status'] = 'complete_with_analysis_failures'
    with pytest.raises(ValueError, match='already be complete'):
        precision.analyze_precision(old, new, plan())
    new['status'] = 'complete'
    new['primary']['rows'][1]['seed'] = new['primary']['rows'][0]['seed']
    with pytest.raises(ValueError, match='one row per registered geometry'):
        precision.analyze_precision(old, new, plan())
