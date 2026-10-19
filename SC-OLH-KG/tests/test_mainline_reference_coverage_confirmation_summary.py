"""Protect global sealing, fixed truth scope, missingness and paired accounting."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from performance import summarize_mainline_reference_coverage_confirmation as summary


def fixture(root, count=2):
    arms = [summary.AFTER, summary.POINT, summary.FULL]
    plan = dict(output_dir='cohort', tasks=[dict(geometry_task_seed=seed, domain='Inventory')
        for seed in range(1, count + 1)], terminal_arms=[dict(name=arm) for arm in arms],
        statistical_unit='paired geometry', analysis=dict(primary='AFTER joint C/G'),
        budget=dict(maximum_target_physical_calls=99999), historical_accounts=['history/completion.json'])
    cells, input_rows = [], []
    for task in plan['tasks']:
        seed = task['geometry_task_seed']
        initial = [dict(point=[i, 0]) for i in range(10)]
        directory = root / 'cohort' / f'search_{seed}'
        summary.save(directory / 'initialization.json', dict(history=initial))
        for arm in arms:
            source = f'cohort/search_{seed}/{arm}'
            input_rows.append(dict(seed=seed, source_case=source))
            summary.save(root / source / 'search.json', dict(history=initial + [dict(point=[10, 0])]))
            for variant in summary.VARIANTS:
                path = f'cohort/{seed}/{arm}/{variant}'
                item = dict(seed=seed, domain=task['domain'], arm=arm, source_case=source,
                    frozen_shortlist=[dict(point=[i, 0]) for i in (0, 1, 10)])
                record = dict(status='complete', deployed=[10, 0],
                    verification=dict(certified=True, verification_budget=1),
                    logical_verification_calls=1, reused_raw_calls=1,
                    new_verification_calls=0, cumulative_new_verification_calls=0,
                    cumulative_failed_simulation_calls=0)
                summary.save(root / path / 'input.json', item)
                summary.save(root / path / 'verification.json', record)
                cells.append(dict(seed=seed, domain=task['domain'], arm=arm, variant=variant,
                    status='complete', input_file=path + '/input.json', result_file=path + '/verification.json'))
    summary.save(root / 'cohort/frozen_inputs.json', dict(rows=input_rows))
    summary.save(root / 'cohort/execution_status.json', dict(status='sealed',
        registered_terminal_cells=len(cells), cells=cells, truth_access_allowed=True,
        physical_verification_calls=0, failed_simulation_calls=0))
    return plan


class AnalyticProblem:
    alpha, tau = .05, 0.
    weights, ref_point = np.array([1., 0.]), np.array([100., 0.])

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def true_objective(self, point):
        self.calls += 1
        if point[0] == self.fail_at:
            raise RuntimeError('retained analytic failure')
        return 30. - point[0]

    def true_constraint_mean(self, point):
        self.calls += 1
        return -1.

    def true_sigma(self, point):
        self.calls += 1
        return [0., 0.]

    def simulate(self, point, rng=None):
        raise AssertionError('Analytic scoring must never simulate')


def test_incomplete_global_grid_blocks_inventory_and_all_truth(tmp_path, monkeypatch):
    plan = fixture(tmp_path)
    path = tmp_path / 'cohort/execution_status.json'
    seal = summary.read(path)
    seal['cells'].pop()
    summary.save(path, seal)
    monkeypatch.setattr(summary, 'build_problem', lambda _: pytest.fail('Premature truth factory'))
    with pytest.raises(ValueError, match='complete registered four-cell grid'):
        summary.freeze_truth_inventory(plan, root=tmp_path)
    with pytest.raises(ValueError):
        summary.score_task(plan, 1, root=tmp_path)
    assert not (tmp_path / 'cohort/truth_inventory.json').exists()


def test_union_freezes_before_truth_and_caps_scope(tmp_path, monkeypatch):
    plan = fixture(tmp_path)
    monkeypatch.setattr(summary, 'build_problem', lambda _: pytest.fail('Freeze must not access truth'))
    inventory = summary.freeze_truth_inventory(plan, root=tmp_path)
    assert inventory['required_policies'] == 22
    assert inventory['analytic_field_call_upper_bound'] == 66
    assert inventory['truth_used'] is False
    assert summary.freeze_truth_inventory(plan, root=tmp_path) == inventory
    root2 = tmp_path / 'overflow'
    plan2 = fixture(root2, 1)
    initial_path = root2 / 'cohort/search_1/initialization.json'
    summary.save(initial_path, dict(history=[dict(point=[i, 0]) for i in range(29)]))
    with pytest.raises(ValueError, match='28-policy'):
        summary.freeze_truth_inventory(plan2, root=root2)


def test_truth_needs_inventory_even_after_terminal_seal(tmp_path, monkeypatch):
    plan = fixture(tmp_path, 1)
    monkeypatch.setattr(summary, 'build_problem', lambda _: pytest.fail('Truth union is not frozen'))
    with pytest.raises(ValueError, match='union must be frozen'):
        summary.score_task(plan, 1, root=tmp_path)


def test_truth_uses_only_union_three_fields_and_completed_work_is_reused(tmp_path, monkeypatch):
    plan = fixture(tmp_path, 1)
    summary.freeze_truth_inventory(plan, root=tmp_path)
    problem = AnalyticProblem()
    monkeypatch.setattr(summary, 'build_problem', lambda _: problem)
    result = summary.score_task(plan, 1, root=tmp_path)
    assert result['new_truth_function_calls'] == problem.calls == 33
    assert len(result['outcomes']) == 12
    assert result['outcomes'][0]['reported_deployed'].endswith(':history:10')
    assert result['outcomes'][0]['metrics']['G'] == pytest.approx(.01)
    assert result['new_simulation_calls'] == result['posterior_restorations'] == 0
    assert summary.score_task(plan, 1, root=tmp_path) == result
    assert problem.calls == 33


def test_truth_failure_is_retained_without_retry_and_missing_means_stay_null(tmp_path, monkeypatch):
    plan = fixture(tmp_path, 1)
    summary.freeze_truth_inventory(plan, root=tmp_path)
    problem = AnalyticProblem(fail_at=3)
    monkeypatch.setattr(summary, 'build_problem', lambda _: problem)
    result = summary.score_task(plan, 1, root=tmp_path)
    assert result['status'] == 'complete_with_truth_failure'
    assert result['failure']['automatic_retry'] is False
    assert all(row['metrics']['G'] is None for row in result['outcomes'])
    before = problem.calls
    assert summary.score_task(plan, 1, root=tmp_path) == result
    assert problem.calls == before
    report = summary.collect(plan, root=tmp_path)
    assert report['status'] == 'complete_with_retained_failures'
    assert report['primary']['joint']['G']['all_registered_mean'] is None


def test_interrupted_score_is_not_silently_restarted(tmp_path, monkeypatch):
    plan = fixture(tmp_path, 1)
    summary.freeze_truth_inventory(plan, root=tmp_path)
    directory = tmp_path / 'cohort/truth/seed_1'
    directory.mkdir(parents=True)
    (directory / 'attempts.jsonl').write_text('{}\n')
    monkeypatch.setattr(summary, 'build_problem', lambda _: pytest.fail('No automatic truth retry'))
    with pytest.raises(RuntimeError, match='explicit review'):
        summary.score_task(plan, 1, root=tmp_path)


def synthetic_outcomes(plan):
    outcomes = []
    for task in plan['tasks']:
        for arm in summary.arm_names(plan):
            for variant in summary.VARIANTS:
                values = dict.fromkeys(summary.METRICS, 0.)
                values.update(A=1, C=1, G=0., R=.5)
                outcomes.append(dict(seed=task['geometry_task_seed'], domain=task['domain'],
                    arm=arm, variant=variant, status='complete', metrics=values,
                    quality_reference_status='defined', shortlist=['a', 'b', 'c']))
    return outcomes


def test_factorial_and_after_point_are_paired_and_keep_negative_contributions(tmp_path):
    plan = fixture(tmp_path, 3)
    outcomes = synthetic_outcomes(plan)
    effects = {1: (.01, .02, .10), 2: (.02, .01, -.03), 3: (0., 0., 0.)}
    for row in outcomes:
        if row['arm'] == summary.AFTER:
            reference, coverage, joint = effects[row['seed']]
            row['metrics']['G'] = dict(original_old=0., reference_old=reference,
                original_two_comparison=coverage, reference_two_comparison=joint)[row['variant']]
    rows, arms, secondary = summary.summarize_outcomes(plan, outcomes)
    group = next(row for row in arms if row['arm'] == summary.AFTER)
    assert group['interaction']['G']['all_registered_mean'] == pytest.approx((.07 - .06) / 3)
    stat = group['joint']['G']
    assert stat['all_registered_mean'] == pytest.approx(.07 / 3)
    assert stat['conditional_available']['positive_sum'] == .10
    assert stat['conditional_available']['negative_sum'] == -.03
    assert stat['conditional_available']['zero'] == 1
    assert stat['leave_largest_positive_geometry_out']['excluded_seed'] == 1
    assert stat['leave_largest_positive_geometry_out']['all_registered_leave_one_out_mean'] == -.015
    ap = next(row for row in secondary if (row['left'], row['right']) == (summary.AFTER, summary.POINT))
    assert ap['reference_two_comparison']['G']['all_registered_mean'] == stat['all_registered_mean']
    assert len(ap['rows']) == len(plan['tasks']) == 3


def test_failed_cells_false_certificates_and_undefined_baselines_are_not_zero_filled(tmp_path):
    plan = fixture(tmp_path)
    outcomes = synthetic_outcomes(plan)
    for row in outcomes:
        if row['arm'] == summary.AFTER and row['seed'] == 1 and row['variant'] == 'reference_two_comparison':
            row['status'], row['metrics'] = 'failed', summary.metrics(failed=True)
        if row['arm'] == summary.POINT and row['seed'] == 2:
            row['metrics'] = summary.metrics(True, dict(true_feasible=False, quality=.5), dict(quality=.4))
    _, arms, secondary = summary.summarize_outcomes(plan, outcomes)
    after = next(row for row in arms if row['arm'] == summary.AFTER)
    assert after['joint']['G']['all_registered_mean'] is None
    assert after['joint']['G']['conditional_available']['n'] == 1
    assert after['outcome_counts']['reference_two_comparison']['failed'] == 1
    point = next(row for row in arms if row['arm'] == summary.POINT)
    assert point['original_old']['false_certificate']['all_registered_mean'] == .5


def test_collect_requires_every_scoring_outcome_without_implicit_truth(tmp_path, monkeypatch):
    plan = fixture(tmp_path)
    summary.freeze_truth_inventory(plan, root=tmp_path)
    monkeypatch.setattr(summary, 'build_problem', lambda _: pytest.fail('Collect never fills missing truth'))
    with pytest.raises(ValueError, match='Every geometry'):
        summary.collect(plan, root=tmp_path)


def test_accounting_charges_shared_search_and_cumulative_terminal_once(tmp_path):
    plan = fixture(tmp_path, 1)
    seal = summary.load_sealed(plan, root=tmp_path)
    seal['physical_verification_calls'], seal['failed_simulation_calls'] = 4, 1
    rows = [dict(stage='initial', status='completed', new_target_initial_calls=10, cpu_seconds=1.),
        dict(stage='online', status='completed', new_target_search_calls=3, cpu_seconds=2.),
        dict(stage='terminal', status='failed', new_verification_calls=4, cpu_seconds=3.)]
    (tmp_path / 'cohort/attempts.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    summary.save(tmp_path / 'cohort/cluster_control/validation/attempt1/checks.json',
                 dict(status='passed', cpu_seconds=99.))
    result = summary.accounting(plan, seal, root=tmp_path)
    assert result['successful_target_physical_calls'] == 17
    assert result['target_invocations_including_recorded_terminal_failures'] == 18
    assert result['attempt_cpu_seconds'] == 6
    assert len(result['failed_attempts']) == 1
    assert result['search_model_and_engine_restorations'] is None
    assert result['checks'][0]['cpu_seconds'] == 99.
    assert result['historical_accounts'] == plan['historical_accounts']
    seal['physical_verification_calls'] = 5
    with pytest.raises(ValueError, match='cumulative raw'):
        summary.accounting(plan, seal, root=tmp_path)


def test_batch_invokes_importable_module_and_only_registered_slice(tmp_path, monkeypatch):
    plan = fixture(tmp_path, 3)
    summary.freeze_truth_inventory(plan, root=tmp_path)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(summary.subprocess, 'run', run)
    result = summary.score_batch(plan, 'manifest.json', 1, 3, 2, root=tmp_path)
    assert result['status'] == 'complete'
    assert {command[-1] for command in calls} == {'2', '3'}
    assert all(command[3] == 'performance.summarize_mainline_reference_coverage_confirmation' for command in calls)


def test_interrupted_terminal_raw_and_unclosed_search_costs_remain_lower_bounds(tmp_path):
    plan = fixture(tmp_path, 1)
    plan['execution_failures'] = [dict(seed=1, arm=summary.FULL,
        status='search_subprocess_failed_before_ledger', call_accounting_complete=False,
        new_initial_calls=None, new_search_calls=None)]
    seal = summary.load_sealed(plan, root=tmp_path)
    cell = seal['cells'][0]
    cell['status'] = 'failed'
    record = dict(status='failed', reason='Terminal subprocess interrupted; no automatic replay',
        subprocess_job=dict(returncode=-9), cumulative_new_verification_calls=4,
        cumulative_failed_simulation_calls=1, logical_verification_calls=None, reused_raw_calls=None)
    summary.save(tmp_path / cell['result_file'], record)
    seal['physical_verification_calls'], seal['failed_simulation_calls'] = 4, 1
    result = summary.accounting(plan, seal, root=tmp_path)
    assert result['unique_terminal_physical_calls'] == result['unclosed_attempt_physical_calls'] == 4
    assert result['known_recorded_target_invocations_including_failures'] == 5
    assert result['successful_target_physical_calls'] is None
    assert result['unclosed_attempt_cpu_seconds'] is None
    assert result['physical_spending_interpretation'] == 'known_recorded_lower_bound'
    assert result['terminal_logical_and_raw_ownership_by_variant'][cell['variant']]['logical_verification_calls'] is None
    assert result['search_histories_with_unclosed_accounting'] == plan['execution_failures']
