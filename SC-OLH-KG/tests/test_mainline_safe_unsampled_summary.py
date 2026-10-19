"""Tiny saved-row substitutes; no model, simulation or target truth is opened."""
from copy import deepcopy

import pytest

from performance.summarize_mainline_safe_unsampled_validation import (
    compact_summary, summarize,
)
from performance.summarize_mainline_two_comparison_development import (
    AFTER, FULL, POINT, decompose, metrics,
)


ARMS = [FULL, FULL + '_native_terminal', AFTER, AFTER + '_native_terminal',
        'legacy_native_neighbors', 'same_source_gp', POINT,
        'current_structural_mixture', 'initialization_only']


def fixture():
    plan = dict(tasks=[dict(geometry_task_seed=seed) for seed in (1, 2, 3)],
        terminal_arms=[dict(name=arm) for arm in ARMS], output_dir='fixture',
        analysis=dict(primary='AFTER-minus-POINT under the old terminal, C and signed G'),
        scope='Synthetic three-geometry rows', statistical_unit='Geometry', historical_accounts=['older.json'])
    rows, outcomes = [], []
    best = dict(quality=.5)
    for seed in (1, 2, 3):
        for arm in ARMS:
            # Old AP effects +.1, -.05, 0; the new terminal erases the positive.
            quality = .5 + ({1: .1, 2: -.05, 3: 0}[seed] if arm == AFTER else 0)
            old = metrics(True, dict(quality=quality, true_feasible=True), best)
            new = metrics(True, dict(quality=.5, true_feasible=True), best)
            rows.append(dict(seed=seed, domain='family_a' if seed < 3 else 'family_b', arm=arm,
                history_arm=arm, status='complete', old_status='complete', new_status='complete',
                quality_reference_status='defined', quality_baseline_defined=True,
                best_initial_quality=.5, old_deployed='initial:0', new_deployed='initial:0',
                old_reported_deployed='initial:0', new_reported_deployed='initial:0',
                **decompose(old, new, old)))
            for protocol, values in (('old', old), ('new', new)):
                outcomes.append(dict(seed=seed, domain=rows[-1]['domain'], arm=arm, protocol=protocol,
                    status='complete', metrics=values,
                    costs=dict(logical_total_calls=93, logical_terminal_calls=80,
                        new_verification_calls=80 if protocol == 'old' else 0), files={}))
    scored = dict(rows=rows, outcomes=outcomes, metric_definitions=dict(C='C', G='G'),
        costs=dict(scoring_new_truth_function_calls=9), truth_by_task=[])
    completion = dict(new_work=dict(new_target_initial_calls=30, new_target_search_calls=54,
        new_verification_calls=2160, new_truth_function_calls=39, failed_simulation_calls_recorded=0),
        new_physical_target_calls=2244, terminal_costs=dict(old_logical_calls=2160,
        new_logical_calls=2160, old_physical_calls=2160, additional_new_protocol_physical_calls=0,
        unique_paired_physical_calls=2160), recorded_cpu_seconds=12,
        failed_attempts=[], execution_failures=[], inherited_accounts=['older.json'])
    return plan, scored, completion


def test_old_ap_is_primary_and_geometry_is_the_unit():
    """Detect accidentally keeping the inherited new-minus-old primary or arm pseudoreplication."""
    plan, scored, completion = fixture()
    before = deepcopy(scored)
    result = summarize(plan, scored, completion=completion)
    primary = result['primary']
    assert (primary['left'], primary['right'], primary['protocol']) == (AFTER, POINT, 'old')
    g = primary['metrics']['G']
    assert g['registered_geometries'] == g['conditional_available']['n'] == 3
    assert g['all_registered_mean'] == pytest.approx(.05 / 3)
    assert g['conditional_available']['degrees_of_freedom'] == 2
    assert (g['conditional_available']['positive'], g['conditional_available']['zero'],
            g['conditional_available']['negative']) == (1, 1, 1)
    assert result['secondary']['AFTER_minus_POINT']['new_contrast']['G']['all_registered_mean'] == 0
    sensitivity = g['leave_largest_positive_geometry_out']
    assert sensitivity['excluded_seed'] == 1
    assert sensitivity['all_registered_leave_one_out_mean'] == pytest.approx(-.025)
    assert sensitivity['primary_sample_changed'] is False
    assert len(primary['rows']) == 3 and len(primary['by_family']) == 2
    assert scored == before


def test_undefined_baseline_failed_cell_and_lost_certificate_stay_distinct():
    """Detect null-to-zero imputation and a lost initial certificate hidden by zero G."""
    plan, scored, completion = fixture()
    for row in scored['rows']:
        if row['seed'] == 1:
            row.update(quality_reference_status='no_true_feasible_initial',
                quality_baseline_defined=False, best_initial_quality=None)
            old = metrics(True, dict(quality=.6, true_feasible=True), None)
            row.update(decompose(old, old, old))
        if row['seed'] == 2 and row['arm'] == AFTER:
            row.update(status='incomplete_pair', old_status='failed')
            row.update(decompose(metrics(failed=True), row['new'], row['B']))
        if row['seed'] == 3 and row['arm'] == AFTER:
            old = metrics(False, None, dict(quality=.5))
            row.update(decompose(old, row['new'], row['B']))
    for outcome in scored['outcomes']:
        if outcome['seed'] == 2 and outcome['arm'] == AFTER and outcome['protocol'] == 'old':
            outcome.update(status='failed', metrics=metrics(failed=True),
                failure=dict(reason='recorded failure'), costs=dict(new_verification_calls=7))
    result = summarize(plan, scored, completion=completion)
    g = result['primary']['metrics']['G']
    assert g['all_registered_mean'] is None
    assert g['conditional_available']['n'] == 1
    assert g['conditional_available']['mean'] == 0
    assert result['primary']['metrics']['C']['conditional_available']['n'] == 2
    last = result['primary']['rows'][-1]['metrics']
    assert last['C'] == -1 and last['R'] == -.5 and last['G'] == 0
    assert result['status'] == 'complete_with_missing_cells'
    assert result['failures'][0]['costs']['new_verification_calls'] == 7


def test_all_controls_compact_task_effects_and_costs_are_retained():
    """Detect dropping controls/task losses or adding physical and analytic ledger totals together."""
    plan, scored, completion = fixture()
    result = summarize(plan, scored, completion=completion,
        audit_costs=dict(new_truth_function_calls=30, cpu_seconds=2))
    compact = compact_summary(result)
    assert {row['arm'] for row in compact['arms']} == set(ARMS)
    assert {row['right'] for row in compact['secondary']['AFTER_minus_controls']} == (
        set(ARMS) - {AFTER, POINT, FULL})
    assert len(compact['tasks']) == 3
    assert len(compact['tasks'][0]['arms']) == 9
    assert len(compact['tasks'][0]['cell_accounts']) == 18
    assert set(compact['tasks'][0]['metrics']) == {'A', 'C', 'R', 'G', 'P', 'L', 'false_certificate'}
    assert compact['tasks'][1]['metrics']['L'] == pytest.approx(.05)
    costs = compact['costs']
    assert costs['physical']['total_target_calls'] == 2244
    assert costs['logical']['new']['terminal_calls_including_paid_failed_prefixes'] == 2160
    assert costs['analytic_truth']['recorded_total_function_calls'] == 39
    assert costs['analytic_truth']['mechanism_stage']['new_truth_function_calls'] == 30
    assert costs['historical_accounts'] == ['older.json']


def test_registered_failures_cannot_disappear_from_input_grid():
    """Detect a silently reduced denominator if an input drops an arm/geometry cell."""
    plan, scored, _ = fixture()
    scored['rows'].pop()
    with pytest.raises(ValueError, match='every registered geometry-arm'):
        summarize(plan, scored)
