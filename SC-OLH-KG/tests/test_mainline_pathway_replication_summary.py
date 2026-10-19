"""Preserve signed control evidence and geometry pairing in saved-path replication."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from performance import run_mainline_pathway_replication as runner


MULTI, SINGLE, MINIMUM = runner.CATEGORIES[:3]


def fixture_plan(count=3):
    arms = [runner.AFTER, runner.POINT, runner.FULL, runner.AFTER + '_native_terminal',
        runner.FULL + '_native_terminal', 'legacy_native_neighbors', 'same_source_gp',
        'current_structural_mixture', 'initialization_only']
    return dict(output_dir='analysis', tasks=[dict(geometry_task_seed=i + 1, domain='family') for i in range(count)],
        terminal_arms=[dict(name=arm, history_arm=arm) for arm in arms],
        streams=[dict(name=name, batch=batch, source_manifest=name + '.json', source_output_dir=output,
            attribution_file_pattern='paths/' + name + '/seed_{seed}.json') for name, batch, output in (
                ('reference', 'reference', 'source'), ('observation_1', 'independent', 'repeats/stream_1'),
                ('observation_2', 'independent', 'repeats/stream_2'))],
        completed_source_batches=['source', 'repeats'], historical_accounts=['old/completion.json'],
        statistical_unit='two-stream geometry average, reference separate', limits=['saved records only'])


def values(gain=0., certificate=1):
    return dict(A=certificate, C=certificate, false_certificate=0, R=.5 + gain if certificate else 0.,
                G=gain, P=max(gain, 0.), L=max(-gain, 0.))


def outcome(gain=0., category='initial', certificate=1):
    return dict(status='complete', metrics=values(gain, certificate), discovery_category=category,
        deployed_policy_id='P001' if certificate else None, reported_policy_id='P001',
        actual_deployed_quality=.5 + gain if certificate else None,
        first_query_prefix=10 if category not in ('initial', 'no_true_certified_deployment') else None,
        returned_repeat_count=0)


def task_result(plan, task):
    return dict(status='complete', seed=task['geometry_task_seed'], domain=task['domain'],
        streams=[dict(name=descriptor['name'], batch=descriptor['batch'], status='complete',
            outcomes=[dict(outcome(), arm=arm, protocol=protocol) for arm in runner.arm_names(plan) for protocol in runner.PROTOCOLS],
            queries=[], policy_paths=[], comparisons=[]) for descriptor in plan['streams']])


def set_outcome(task, stream, arm, gain, category):
    rows = next(row for row in task['streams'] if row['name'] == stream)['outcomes']
    for row in rows:
        if row['arm'] == arm:
            row.update(outcome(gain, category))


def write_source(root, plan):
    for directory in plan['completed_source_batches']:
        runner.save(root / directory / 'completion.json', dict(status='complete'))
    for descriptor in plan['streams']:
        source = dict(status='registered_not_started', terminal_arms=deepcopy(plan['terminal_arms']),
            tasks=[dict(task, geometry=dict(shift=[.1, .2])) for task in plan['tasks']])
        runner.save(root / descriptor['source_manifest'], source)
        cases = [dict(seed=task['geometry_task_seed'], arm=arm, old=dict(status='complete'), new=dict(status='complete'))
                 for task in plan['tasks'] for arm in runner.arm_names(plan)]
        directory = root / descriptor['source_output_dir']
        runner.save(directory / 'execution_status.json', dict(status='sealed', cases=cases))
        runner.save(directory / 'mechanism/inventory.json', dict(status='sealed',
            tasks=[dict(seed=task['geometry_task_seed']) for task in plan['tasks']]))
        for task in plan['tasks']:
            seed = task['geometry_task_seed']
            folder = directory / 'mechanism' / f'seed_{seed}'
            runner.save(folder / 'result.json', dict(status='complete', cached_inventory_policies=80, distinct_inventory_policies=80))
            runner.save(folder / 'analysis.json', dict(status='complete'))
            runner.save(root / descriptor['attribution_file_pattern'].format(seed=seed), dict(status='complete'))
    runner.save(root / 'repeats/mechanism/inventory.json', dict(status='sealed', registered_streams=2,
        registered_task_streams=2 * len(plan['tasks']), registered_terminal_cells=36 * len(plan['tasks'])))


def test_signed_categories_include_point_gains_after_losses_and_avoided_losses():
    row = runner.category_decomposition(outcome(-.1, MINIMUM), outcome(.2, MULTI))
    assert row['delta']['G'] == pytest.approx(-.3)
    assert row['categories'][MULTI]['positive_gain_component'] == -.2
    assert row['categories'][MINIMUM]['avoided_loss_component'] == -.1
    assert sum(value['net_gain'] for value in row['categories'].values()) == pytest.approx(-.3)
    reverse = runner.category_decomposition(outcome(.2, MULTI), outcome(-.1, SINGLE))
    assert reverse['categories'][MULTI]['positive_gain_component'] == .2
    assert reverse['categories'][SINGLE]['avoided_loss_component'] == .1
    assert reverse['categories'][SINGLE]['positive_gain_component'] == 0.
    after, point = outcome(-.1, MINIMUM), outcome(.2, MULTI)
    after['returned_repeat_count'] = 2
    repeated = runner.category_decomposition(after, point)
    assert repeated['requery']['requery']['avoided_loss_component'] == -.1
    assert repeated['requery']['no_requery']['positive_gain_component'] == -.2
    assert sum(row['net_gain'] for row in repeated['requery'].values()) == pytest.approx(-.3)


def test_missing_baseline_is_unknown_in_every_category_even_without_certificate():
    missing = outcome(0., 'no_true_certified_deployment', 0)
    missing['metrics'].update(G=None, P=None, L=None)
    row = runner.category_decomposition(missing, outcome(.1, MULTI))
    assert row['delta']['C'] == -1 and row['delta']['G'] is None
    assert all(value is None for category in row['categories'].values() for value in category.values())
    assert missing['actual_deployed_quality'] is None


def test_new_stream_mean_is_paired_before_statistics_and_not_positive_part_of_mean():
    plan = fixture_plan(2)
    tasks = [task_result(plan, task) for task in plan['tasks']]
    set_outcome(tasks[0], 'reference', runner.AFTER, 100., MULTI)
    set_outcome(tasks[0], 'observation_1', runner.AFTER, .4, MULTI)
    set_outcome(tasks[0], 'observation_2', runner.AFTER, -.2, MINIMUM)
    result = runner.summarize(tasks, plan)
    primary = result['primary']
    assert primary['delta']['G']['registered_geometries'] == 2
    assert primary['delta']['G']['all_registered_mean'] == pytest.approx(.05)
    assert primary['rows'][0]['left']['P'] == .2
    assert primary['rows'][0]['left']['L'] == .1
    assert primary['rows'][0]['categories'][MULTI]['net_gain'] == .2
    assert primary['rows'][0]['categories'][MINIMUM]['net_gain'] == -.1
    assert len(result['independent']['controls']) == 18
    assert result['registered_terminal_outcomes'] == 2 * 3 * 18
    assert result['streams'][0]['comparisons'][0]['delta']['G']['all_registered_mean'] == 50.


def test_one_stream_missing_keeps_full_geometry_denominator_and_controls():
    plan = fixture_plan(2)
    tasks = [task_result(plan, task) for task in plan['tasks']]
    set_outcome(tasks[0], 'observation_1', runner.AFTER, .2, MULTI)
    set_outcome(tasks[0], 'observation_2', runner.AFTER, .4, MULTI)
    missing = next(row for row in tasks[1]['streams'] if row['name'] == 'observation_2')
    for row in missing['outcomes']:
        row['metrics'].update(G=None, P=None, L=None)
    result = runner.summarize(tasks, plan)
    assert result['primary']['delta']['G']['all_registered_mean'] is None
    assert result['primary']['delta']['G']['conditional_available']['mean'] == pytest.approx(.3)
    assert result['primary']['delta']['G']['available_geometries'] == 1
    assert result['primary']['delta']['C']['available_geometries'] == 2
    assert all(row is None for row in result['primary']['rows'][1]['categories'][MULTI].values())
    assert result['status'] == 'complete'


def test_same_category_after_gain_does_not_imply_positive_ap_contribution():
    plan = fixture_plan(1)
    task = task_result(plan, plan['tasks'][0])
    for stream in ('observation_1', 'observation_2'):
        set_outcome(task, stream, runner.AFTER, .2, MULTI)
        set_outcome(task, stream, runner.POINT, .3, MULTI)
    result = runner.summarize([task], plan)
    repeated = result['replication'][0]
    assert repeated['category_agreement'][MULTI]['left_own_positive_both']['true'] == 1
    assert repeated['category_agreement'][MULTI]['signed_contribution_positive_both']['true'] == 0
    assert repeated['total_signed_gain_positive_both']['true'] == 0
    assert repeated['rows'][0]['categories'][MULTI]['stream_components'][0]['net_gain'] == pytest.approx(-.1)
    for stream in task['streams']:
        if stream['name'] != 'reference':
            for row in stream['outcomes']:
                if row['arm'] == runner.AFTER:
                    row.update(outcome(0., 'no_true_certified_deployment', 0), returned_repeat_count=None)
    no_certificate = runner.summarize([task], plan)['replication'][0]['requery_agreement']['left_own_positive_both']
    assert no_certificate == dict(registered_geometries=1, defined_geometries=1, true=0, false=1, unknown=0)


def test_source_gate_uses_completed_seals_not_stale_manifest_and_checks_geometry(tmp_path):
    plan = fixture_plan(1)
    write_source(tmp_path, plan)
    result = runner.require_source(plan, root=tmp_path)
    assert result['registered_terminal_outcomes'] == 54
    assert all(result[key] == 0 for key in runner.ZERO)
    source = runner.read(tmp_path / 'observation_2.json')
    source['tasks'][0]['geometry']['shift'][0] = .9
    runner.save(tmp_path / 'observation_2.json', source)
    with pytest.raises(ValueError, match='identical geometry'):
        runner.require_source(plan, root=tmp_path)


def test_source_gate_blocks_incomplete_pool_and_registration_needs_passed_checks(tmp_path):
    plan = fixture_plan(1)
    write_source(tmp_path, plan)
    runner.save(tmp_path / 'checks.json', dict(status='failed'))
    with pytest.raises(ValueError, match='must pass'):
        runner.register(plan, 'plan.json', 'checks.json', root=tmp_path)
    runner.save(tmp_path / 'checks.json', dict(status='passed'))
    path = tmp_path / 'repeats/stream_2/mechanism/seed_1/result.json'
    runner.save(path, dict(status='complete', cached_inventory_policies=79, distinct_inventory_policies=80))
    with pytest.raises(ValueError, match='complete full-pool'):
        runner.register(plan, 'plan.json', 'checks.json', root=tmp_path)
    assert not (tmp_path / 'analysis/registration/result.json').exists()


def test_worker_failure_keeps_54_cells_and_never_retries(tmp_path, monkeypatch):
    plan = fixture_plan(1)
    runner.save(tmp_path / 'plan.json', plan)
    calls = []
    def fail(*args):
        calls.append(args)
        raise RuntimeError('missing saved attribution')
    monkeypatch.setitem(sys.modules, 'performance.mainline_pathway_replication', SimpleNamespace(analyze_task=fail))
    first = runner.geometry_worker(('plan.json', 1, str(tmp_path)))
    second = runner.geometry_worker(('plan.json', 1, str(tmp_path)))
    assert first['status'] == second['status'] == 'failed'
    assert second['reused'] and len(calls) == 1
    task = runner.read(tmp_path / 'analysis/cases/seed_1/result.json')
    result = runner.summarize([task], plan)
    assert result['registered_terminal_outcomes'] == 54
    assert len(result['analysis_failures']) == 3
    assert result['primary']['delta']['G']['available_geometries'] == 0
    assert result['status'] == 'complete_with_analysis_failures'


def test_collect_is_read_only_and_keeps_attempt_validation_and_history_separate(tmp_path, monkeypatch):
    plan = fixture_plan(1)
    monkeypatch.setattr(runner, 'geometry_worker', lambda *_: pytest.fail('Collect cannot analyze missing cases'))
    with pytest.raises(ValueError, match='Every registered geometry'):
        runner.collect(plan, 'plan.json', root=tmp_path)
    runner.save(tmp_path / 'analysis/cases/seed_1/result.json', task_result(plan, plan['tasks'][0]))
    ledger = tmp_path / 'analysis/cases/seed_1/attempts.jsonl'
    ledger.write_text(json.dumps(dict(status='completed', cpu_seconds=2., **runner.ZERO)) + '\n')
    runner.save(tmp_path / 'analysis/dispatch/shard_0_1/result.json', dict(coordinator_cpu_seconds=3.))
    runner.save(tmp_path / 'analysis/cluster_control/validation/stage83_attempt1/checks.json', dict(status='passed', cpu_seconds=7.))
    runner.save(tmp_path / 'analysis/cluster_control/stage83_preflight.json', dict(metadata_cpu_seconds=.5))
    result = runner.collect(plan, 'plan.json', root=tmp_path)
    assert result['registered_task_streams'] == 3
    assert result['recorded_analysis_cpu_seconds'] >= 2.
    assert result['recorded_coordinator_cpu_seconds'] == 3.
    assert result['validations'][0]['cpu_seconds'] == 7.
    assert result['preparation'][0]['metadata_cpu_seconds'] == .5
    assert result['historical_accounts'] == ['old/completion.json']
    assert all(value == 0 for value in result['new_work'].values())
    assert runner.collect(plan, 'plan.json', root=tmp_path) == result


def test_readout_keeps_both_arm_paths_but_no_full_control_paths_or_coordinates():
    plan = fixture_plan(1)
    task = task_result(plan, plan['tasks'][0])
    stream = task['streams'][0]
    stream['policy_paths'] = [dict(history_arm=runner.POINT, policy_id='P001', point=[1] * 1000,
        discovery_category=MULTI, returned_repeat_count=2, improving_safe=True,
        terminals=[dict(arm=arm, protocol='old', status='complete', path=dict(disposition='safety_certificate_failed',
            safety_passed=False, counterfactual_certificate=None, point=[1] * 1000))
            for arm in (runner.AFTER, runner.POINT, 'same_source_gp')])]
    compact = runner.compact_task(task)
    path = compact['streams'][0]['policy_paths'][0]
    assert path['history_arm'] == runner.POINT and path['returned_repeat_count'] == 2
    assert len(path['terminals']) == 2
    assert path['terminals'][0]['path']['counterfactual_certificate'] is None
    assert '"point"' not in json.dumps(compact)
