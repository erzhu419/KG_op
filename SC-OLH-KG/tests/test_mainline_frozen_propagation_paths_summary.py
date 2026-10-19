"""Keep signed deployment gains, all paired units, zero-work boundaries and fees."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from performance import analyze_mainline_frozen_propagation_paths as runner


def fixture_plan(count=3):
    names = [runner.AFTER, runner.POINT, runner.FULL, runner.AFTER + '_native_terminal',
        runner.FULL + '_native_terminal', 'legacy_native_neighbors', 'same_source_gp',
        'current_structural_mixture', 'initialization_only']
    return dict(output_dir='analysis', reference_manifest='source_manifest.json',
        reference_output_dir='source', tasks=[dict(geometry_task_seed=i + 1, domain='family') for i in range(count)],
        terminal_arms=[dict(name=arm) for arm in names], terminal_variants=list(runner.VARIANTS),
        historical_accounts=['old/completion.json'], statistical_unit='fixed paired geometry', limits=['cached truth only'])


def values(gain=0., certificate=1):
    return dict(A=certificate, C=certificate, false_certificate=0, R=.5 + gain if certificate else 0.,
                G=gain, P=max(gain, 0.), L=max(-gain, 0.))


def task_result(plan, task):
    return dict(status='complete', seed=task['geometry_task_seed'], domain=task['domain'],
        outcomes=[dict(seed=task['geometry_task_seed'], domain=task['domain'], arm=arm,
            variant=variant, status='complete', metrics=values())
            for arm in runner.arm_names(plan) for variant in runner.VARIANTS],
        policies=[], histories=[], cells=[], comparisons=[], query_comparisons=[])


def write_source(root, plan):
    source = {key: deepcopy(plan[key]) for key in ('tasks', 'terminal_arms', 'terminal_variants')}
    source['status'] = 'complete'
    runner.save(root / 'source_manifest.json', source)
    runner.save(root / 'source/completion.json', dict(status='complete'))
    cells = []
    for task in plan['tasks']:
        result = task_result(plan, task)
        cells.extend(result['outcomes'])
        runner.save(root / 'source/truth' / f"seed_{task['geometry_task_seed']}" / 'scored.json',
            dict(result, failure=None))
    runner.save(root / 'source/execution_status.json', dict(status='sealed', cells=cells, truth_access_allowed=True))


def test_gain_loss_decomposition_is_signed_and_does_not_claim_discovery():
    row = runner.decomposition(values(.2), values(-.1))
    assert row['delta']['G'] == pytest.approx(.3)
    assert row['positive_gain_component'] == .2
    assert row['avoided_loss_component'] == .1
    assert 'not a causal' in row['interpretation']
    reverse = runner.decomposition(values(-.1), values(.2))
    assert reverse['positive_gain_component'] == -.2
    assert reverse['avoided_loss_component'] == -.1
    broken = values(.2)
    broken['P'] = 0.
    with pytest.raises(ValueError, match='positive gain minus'):
        runner.decomposition(broken, values())
    missing = runner.decomposition(dict.fromkeys(runner.METRICS), values())
    assert missing['delta']['G'] is missing['positive_gain_component'] is missing['avoided_loss_component'] is None


def test_complete_grid_gate_precedes_analysis_and_does_not_fill_missing_scores(tmp_path):
    plan = fixture_plan(2)
    write_source(tmp_path, plan)
    admitted = runner.require_source(plan, root=tmp_path)
    assert admitted['registered_terminal_outcomes'] == 2 * 9 * 4
    assert all(admitted[key] == 0 for key in runner.ZERO)
    path = tmp_path / 'source/truth/seed_2/scored.json'
    result = runner.read(path)
    result['outcomes'].pop()
    runner.save(path, result)
    with pytest.raises(ValueError, match='all four cells'):
        runner.require_source(plan, root=tmp_path)
    assert len(runner.read(path)['outcomes']) == 35


def test_failed_geometry_keeps_all_controls_and_primary_denominator():
    plan = fixture_plan()
    tasks = [task_result(plan, task) for task in plan['tasks'][:2]]
    for row in tasks[0]['outcomes']:
        if row['arm'] == runner.AFTER:
            row['metrics'] = values(.2)
        elif row['arm'] == runner.POINT:
            row['metrics'] = values(-.1)
    for row in tasks[1]['outcomes']:
        if row['arm'] in (runner.AFTER, runner.POINT):
            row['metrics'] = values(0., 0)
    tasks.append(runner.failed_task(plan, plan['tasks'][2], 'retained analysis failure'))
    result = runner.summarize(tasks, plan)
    assert result['registered_terminal_outcomes'] == 3 * 9 * 4
    assert len(result['controls']) == 36
    assert result['primary']['delta']['G']['all_registered_mean'] is None
    assert result['primary']['delta']['G']['conditional_available']['mean'] == pytest.approx(.15)
    assert result['primary']['positive_gain_component']['conditional_available']['mean'] == .1
    assert result['primary']['avoided_loss_component']['conditional_available']['mean'] == .05
    assert result['after_no_certificate_seeds'] == [2]
    assert result['status'] == 'complete_with_analysis_failures'
    assert len(result['primary']['rows']) == 3


def test_registration_requires_passed_substitutes_and_preserves_source(tmp_path):
    plan = fixture_plan(1)
    write_source(tmp_path, plan)
    before = (tmp_path / 'source_manifest.json').read_text()
    runner.save(tmp_path / 'checks.json', dict(status='failed'))
    with pytest.raises(ValueError, match='must pass'):
        runner.register(plan, 'analysis_manifest.json', 'checks.json', root=tmp_path)
    assert not (tmp_path / 'analysis/registration/result.json').exists()
    runner.save(tmp_path / 'checks.json', dict(status='passed'))
    result = runner.register(plan, 'analysis_manifest.json', 'checks.json', root=tmp_path)
    assert result['status'] == 'registered'
    assert (tmp_path / 'source_manifest.json').read_text() == before
    assert runner.register(plan, 'analysis_manifest.json', 'checks.json', root=tmp_path) == result


def test_worker_failure_is_retained_and_never_automatically_replayed(tmp_path, monkeypatch):
    plan = fixture_plan(1)
    runner.save(tmp_path / 'plan.json', plan)
    calls = []
    def fail(*args):
        calls.append(args)
        raise RuntimeError('saved query recording unavailable')
    monkeypatch.setitem(sys.modules, 'performance.mainline_frozen_propagation_paths', SimpleNamespace(analyze_task=fail))
    first = runner.geometry_worker(('plan.json', 1, str(tmp_path)))
    second = runner.geometry_worker(('plan.json', 1, str(tmp_path)))
    assert first['status'] == second['status'] == 'failed'
    assert second['reused'] is True and len(calls) == 1
    result = runner.read(tmp_path / 'analysis/cases/seed_1/result.json')
    assert len(result['outcomes']) == 36
    assert all(row['metrics']['G'] is None for row in result['outcomes'])
    ledger = [json.loads(line) for line in (tmp_path / 'analysis/cases/seed_1/attempts.jsonl').read_text().splitlines()]
    assert len(ledger) == 1 and ledger[0]['status'] == 'failed'
    assert all(ledger[0][key] == 0 for key in runner.ZERO)


def test_collect_blocks_missing_geometry_and_never_implicitly_analyzes(tmp_path, monkeypatch):
    plan = fixture_plan(2)
    runner.save(tmp_path / 'analysis/cases/seed_1/result.json', task_result(plan, plan['tasks'][0]))
    monkeypatch.setattr(runner, 'geometry_worker', lambda *_: pytest.fail('Collect must never fill missing analysis'))
    with pytest.raises(ValueError, match='Every registered geometry'):
        runner.collect(plan, 'plan.json', root=tmp_path)


def test_collect_keeps_science_zero_costs_separate_and_compact_output_without_vectors(tmp_path):
    plan = fixture_plan(1)
    task = task_result(plan, plan['tasks'][0])
    task['policies'] = [dict(policy_id='p1', point=[5] * 1000)]
    runner.save(tmp_path / 'analysis/cases/seed_1/result.json', task)
    ledger = tmp_path / 'analysis/cases/seed_1/attempts.jsonl'
    ledger.write_text(json.dumps(dict(stage='analysis', status='completed', cpu_seconds=2., **runner.ZERO)) + '\n')
    runner.save(tmp_path / 'analysis/dispatch/shard_0_1/result.json', dict(coordinator_cpu_seconds=3.))
    runner.save(tmp_path / 'analysis/cluster_control/validation/stage82_attempt1/checks.json',
                dict(status='passed', cpu_seconds=7.))
    result = runner.collect(plan, 'plan.json', root=tmp_path)
    assert result['status'] == 'complete'
    assert result['recorded_analysis_cpu_seconds'] >= 2.
    assert result['recorded_coordinator_cpu_seconds'] == 3.
    assert result['validations'][0]['cpu_seconds'] == 7.
    assert result['historical_accounts'] == ['old/completion.json']
    assert all(value == 0 for value in result['new_work'].values())
    compact = (tmp_path / 'analysis/readout.json').read_text()
    assert '"point"' not in compact
    assert runner.collect(plan, 'plan.json', root=tmp_path) == result


def test_compact_paths_keep_uncertified_quality_and_unknown_competition_unknown():
    plan = fixture_plan(1)
    task = task_result(plan, plan['tasks'][0])
    outcome = next(row for row in task['outcomes'] if (row['arm'], row['variant']) == (runner.AFTER, 'original_old'))
    outcome.update(metrics=values(0., 0), reported_policy_id='P010', deployed_policy_id=None,
        actual_deployed_quality=None, newly_queried_deployment=False, deployment_query_indices=[])
    task['policies'] = [dict(policy_id='P010', point=[7] * 1000, initial_indices=[],
        origin='queried_noninitial', observed_locations=[dict(history_arm=runner.AFTER, observation_index=10)],
        truth=dict(quality=.9, true_feasible=True), truth_status='cached')]
    task['histories'] = [dict(history_arm=runner.AFTER, status='complete',
        queries=[dict(paid_prefix=10, selected_policy_id='P010', pool_truth=dict(truth_complete=False,
            unknown_truth_count=100, max_true_safe_quality=None), selection_replay=dict(status='matched'))],
        policy_paths=[dict(policy_id='P010', states=[dict(paid_prefix=10, initialization_propagation_delta=None)])],
        observed_truth=dict(status='truth_incomplete', max_quality=None))]
    task['cells'] = [dict(arm=runner.AFTER, variant='original_old', status='complete',
        reported_policy_id='P010', deployed_policy_id=None, shortlist_policy_ids=['P010'],
        quality_sets={}, original_selection_record_status='complete', selection_issues=[], observed=[],
        safety_attempts=[dict(candidate_index=0, certified=False, upper_margin=.02, verification_budget=80,
            point=[7] * 1000)], actual_objective_comparisons=[])]
    task['query_comparisons'] = [dict(left=runner.AFTER, right=runner.POINT,
        states=[dict(paid_prefix=10, same_paid_history=True, same_physical_pool=True,
                     common_state_first_divergence=True)])]
    result = runner.compact_task(task)
    assert result['terminals'][0]['actual_deployed_quality'] is None
    assert result['terminals'][0]['reported_policy_id'] == 'P010'
    assert result['terminals'][0]['deployed_policy_id'] is None
    assert result['histories'][0]['queries'][0]['pool_truth']['max_true_safe_quality'] is None
    assert result['histories'][0]['focal_policy_paths'][0]['states'][0]['initialization_propagation_delta'] is None
    assert '"point"' not in json.dumps(result)
