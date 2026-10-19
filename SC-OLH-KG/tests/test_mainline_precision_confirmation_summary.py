"""Tiny old-only substitutes detect broken barriers, pairing and physical accounting."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from performance import summarize_mainline_precision_confirmation as runner
from performance import run_mainline_two_comparison_development as development


def plan_fixture(count=2):
    arms = [runner.AFTER, runner.POINT, runner.FULL, runner.AFTER+'_native_terminal',
        runner.FULL+'_native_terminal', 'legacy_native_neighbors', 'same_source_gp',
        'current_structural_mixture', 'initialization_only']
    return dict(output_dir='confirmation', frozen={'d': 1},
        tasks=[dict(geometry_task_seed=i+1, domain='family') for i in range(count)],
        streams=[dict(name='observation_'+str(i), manifest=f'child_{i}.json') for i in (1, 2)],
        terminal_arms=[dict(name=arm, history_arm=arm.removesuffix('_native_terminal')) for arm in arms],
        statistical_unit='mean of two streams within each geometry', historical_accounts=['previous/completion.json'])


def metric(gain=0., certificate=1):
    return dict(A=certificate, C=certificate, false_certificate=0, R=.5+gain if certificate else 0.,
        G=gain, P=max(gain, 0.), L=max(-gain, 0.))


def task_fixture(plan, task):
    return dict(seed=task['geometry_task_seed'], domain=task['domain'], status='complete',
        streams=[dict(name=stream['name'], status='complete', coverage={}, queries=[], policy_paths=[],
            outcomes=[dict(seed=task['geometry_task_seed'], domain=task['domain'], arm=cell['name'],
                history_arm=cell['history_arm'], protocol='old', status='complete', metrics=metric(),
                discovery_category='initial', reported_policy_id='P0', deployed_policy_id='P0',
                actual_deployed_quality=.5, returned_repeat_count=0, first_query_prefix=None,
                verification_budget=80) for cell in plan['terminal_arms']]) for stream in plan['streams']])


def selected(task, stream, arm=runner.AFTER):
    return next(row for row in task['streams'][stream]['outcomes'] if row['arm'] == arm)


def test_two_streams_are_paired_before_statistics_and_all_controls_retained():
    plan = plan_fixture()
    tasks = [task_fixture(plan, task) for task in plan['tasks']]
    selected(tasks[0], 0).update(metrics=metric(.4), discovery_category=runner.CATEGORIES[0])
    selected(tasks[0], 1).update(metrics=metric(-.2), discovery_category=runner.CATEGORIES[2])
    result = runner.summarize_loaded(plan, tasks)
    primary = result['primary']
    assert primary['delta']['G']['registered_geometries'] == 2
    assert primary['delta']['G']['all_registered_mean'] == pytest.approx(.05)
    assert primary['rows'][0]['left']['P'] == .2 and primary['rows'][0]['left']['L'] == .1
    assert primary['rows'][0]['categories'][runner.CATEGORIES[0]]['net_gain'] == .2
    assert primary['rows'][0]['categories'][runner.CATEGORIES[2]]['net_gain'] == -.1
    assert len(result['independent']['controls']) == 9
    assert len(result['independent']['comparisons']) == 2
    assert all(row['protocol'] == 'old' for row in result['independent']['controls'])
    assert primary['delta']['G']['leave_largest_positive_geometry_out'] is not None


def test_point_gains_after_losses_missing_baselines_and_scoring_failures_are_not_dropped():
    plan = plan_fixture()
    tasks = [task_fixture(plan, task) for task in plan['tasks']]
    for stream in (0, 1):
        selected(tasks[0], stream).update(metrics=metric(-.1), discovery_category=runner.CATEGORIES[2])
        selected(tasks[0], stream, runner.POINT).update(metrics=metric(.2), discovery_category=runner.CATEGORIES[0])
        selected(tasks[1], stream).update(metrics=dict(metric(0., 0), G=None, P=None, L=None),
            discovery_category='no_true_certified_deployment', actual_deployed_quality=None)
    result = runner.summarize_loaded(plan, tasks)
    primary = result['primary']
    assert primary['delta']['G']['all_registered_mean'] is None
    assert primary['delta']['G']['available_geometries'] == 1
    assert primary['rows'][0]['categories'][runner.CATEGORIES[0]]['net_gain'] == -.2
    assert primary['rows'][0]['categories'][runner.CATEGORIES[2]]['net_gain'] == -.1
    assert primary['delta']['C']['registered_geometries'] == 2
    tasks[1]['streams'][0] = runner.failed_stream(plan, plan['tasks'][1], plan['streams'][0], 'truth unavailable')
    tasks[1]['status'] = 'scoring_failed'
    failed = runner.summarize_loaded(plan, tasks)
    assert len(failed['scoring_failures']) == 1 and failed['terminal_failures'] == []
    assert len(failed['independent']['controls']) == 9
    assert failed['primary']['delta']['C']['all_registered_mean'] is None


INITIAL = [dict(point=[i], raw=[5., -1.]) for i in range(10)]


def bundle(points, first):
    bank = [(i,) for i in range(14)]
    actions = [(0,), *points]
    history, stages = deepcopy(INITIAL), []
    for step, choice in enumerate((first, 2, first)):
        objective = [9., 3., 3., 3.]
        objective[choice] = 1.
        margin = [1., -1., -1., -1.]
        chosen, total, mode, count = runner.competition.score_choice(objective, margin)
        stages.append(dict(paid_prefix=10+step, action_ids=[bank.index(point) for point in actions],
            observed_ids=[bank.index(tuple(row['point'])) for row in history],
            selected_index=chosen, selected_id=bank.index(actions[chosen]), action_sources=['fixture']*4,
            score=dict(objective_mean=objective, sampled_objective=objective, sampled_margin=margin,
                total=total.tolist(), sampled_feasibility_mode=mode, sampled_feasible_count=count,
                objective_epistemic=[.1]*4, constraint_mean=[-1.]*4, constraint_epistemic=[.1]*4,
                constraint_aleatoric=[.1]*4, sampled_constraint=margin, probability_violation=[.01]*4)))
        history.append(dict(point=list(actions[chosen]), raw=[1., -1.]))
    return dict(points=bank, history=history, status='complete',
        audit=dict(shared_ids=[bank.index(point) for point in points], stages=stages))


def source_fixture(root, monkeypatch):
    plan = plan_fixture(1)
    cases, materials = [], {}
    runner.save(root/'source.json', dict(execution_failures=[]))
    for number, stream in enumerate(plan['streams']):
        child = dict(output_dir='confirmation/'+stream['name'], frozen={'d': 1}, marker=number)
        runner.save(root/stream['manifest'], child)
        points = [(10+number,), (11+number,), (12+number,)]
        bundles = {arm: bundle(points, 3 if arm == runner.POINT else 1)
                   for arm in (runner.AFTER, runner.POINT, runner.FULL)}
        materials[number] = (root/child['output_dir'], {}, deepcopy(INITIAL), points, bundles)
        for cell in plan['terminal_arms']:
            directory = Path('confirmation')/stream['name']/cell['name']
            item = dict(frozen_shortlist=[dict(point=list(point)) for point in points])
            verification = dict(certified=True, verification_budget=80, max_verification_budget=352,
                attempts=[dict(candidate_index=0, certified=True, upper_margin=-1.)], objective_comparisons=[])
            runner.save(root/directory/'input.json', item)
            runner.save(root/directory/'verification.json', dict(deployed=list(points[0]), verification=verification))
            runner.save(root/directory/'search.json', dict(history=deepcopy(INITIAL)+[
                dict(point=[0], raw=[5., -1.]), dict(point=[0], raw=[5., -1.]), dict(point=list(points[0]), raw=[1., -1.])]))
            cases.append(dict(stream=stream['name'], seed=1, domain='family', arm=cell['name'],
                history_arm=cell['history_arm'], source_case=str(directory), search_manifest='source.json',
                terminal_manifest='unused.json', old=dict(status='complete', input_file=str(directory/'input.json'),
                    verification_file=str(directory/'verification.json'))))
    runner.save(root/plan['output_dir']/'execution_status.json', dict(status='sealed', truth_access_allowed=True, cases=cases))
    monkeypatch.setattr(runner.audit, 'task_material', lambda child, task, selected_cases: (*materials[child['marker']], selected_cases))
    return plan


class TruthProblem:
    tau, alpha, weights, ref_point = 0., .05, np.array([1.]), np.array([10.])

    def simulate(self, *args, **kwargs):
        raise AssertionError('Post-seal scoring must not simulate')

    def true_objective(self, point):
        return 5. if point[0] < 10 else 1.+(point[0]-10)/10

    def true_constraint_mean(self, point):
        return -1.

    def true_sigma(self, point):
        return [0., 0.]


def test_global_seal_blocks_truth_until_both_streams_last_old_cell_is_sealed(tmp_path, monkeypatch):
    plan = source_fixture(tmp_path, monkeypatch)
    path = tmp_path/plan['output_dir']/'execution_status.json'
    saved = runner.read(path)
    saved['cases'].pop()
    runner.save(path, saved)
    monkeypatch.setattr(runner.audit, 'score_points', lambda *args: pytest.fail('Truth before the complete global seal'))
    with pytest.raises(ValueError, match='globally sealed'):
        runner.freeze(plan, root=tmp_path)
    assert not (tmp_path/plan['output_dir']/'mechanism/inventory.json').exists()


def test_old_only_full_pool_cache_reuse_repeat_paths_and_no_counterfactual_swaps(tmp_path, monkeypatch):
    plan = source_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(development, 'build_problem', lambda item: TruthProblem())
    monkeypatch.setattr(runner.competition, '_swaps', lambda *args: pytest.fail('Unregistered score substitution'))
    inventory = runner.freeze(plan, root=tmp_path)
    assert inventory['registered_terminal_cells'] == 18
    assert inventory['known_distinct_policy_task_units'] == 14  # Entire pools, not selected-only policies.
    assert not (tmp_path/plan['output_dir']/'cases').exists()
    index = runner.read(tmp_path/inventory['tasks'][0]['index_file'])
    assert [len(row['cases']) for row in index['streams']] == [9, 9]
    result = runner.score_task(plan, 1, root=tmp_path)
    assert result['status'] == 'complete', result
    assert [row['new_truth_function_calls'] for row in result['streams']] == [39, 3]
    assert result['cached_policies'] == 14 and result['maximum_analytic_function_calls'] == 42
    for stream in result['streams']:
        assert len(stream['outcomes']) == 9 and len(stream['queries']) == 9
        assert all(row['protocol'] == 'old' for row in stream['outcomes'])
        repeat = next(row for row in stream['queries'] if row['arm'] == runner.AFTER and row['paid_prefix'] == 12)
        assert repeat['query_action_kind'] == 'repeat'
        assert repeat['discovery_category'] == runner.CATEGORIES[0]
        assert all(row.get('substitutions', []) == [] for row in stream['mechanism']['comparisons'])
    monkeypatch.setattr(runner.audit, 'score_points', lambda *args: pytest.fail('Completed work was repeated'))
    assert runner.score_task(plan, 1, root=tmp_path) == result


def write_ledger(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_fees_count_physical_ledgers_once_and_scoring_failure_does_not_erase_target_calls(tmp_path, monkeypatch):
    plan = source_fixture(tmp_path, monkeypatch)
    task = task_fixture(plan, plan['tasks'][0])
    task['streams'][0] = runner.failed_stream(plan, plan['tasks'][0], plan['streams'][0], 'truth failed')
    task['status'] = 'scoring_failed'
    output = tmp_path/plan['output_dir']
    write_ledger(output/'execution/attempts.jsonl', [dict(stage='search', status='completed',
        new_target_initial_calls=20, new_target_search_calls=36, cpu_seconds=3.),
        dict(stage='verify', status='completed', new_verification_calls=1440, cpu_seconds=4.)])
    write_ledger(output/'cases/attempts.jsonl', [dict(stage='score-sealed-complete-old-pool', status='failed',
        new_truth_function_calls=3, cpu_seconds=1.)])
    accounts = runner.accounting(plan, [task], root=tmp_path)
    assert accounts['physical_target_calls'] == 1496
    assert accounts['known_recorded']['cpu_seconds'] == 8.
    assert accounts['known_recorded']['new_truth_function_calls'] == 3
    assert accounts['posterior_restorations'] is None
    assert len(accounts['failed_attempts']) == 1
    source = runner.read(tmp_path/'source.json')
    source['execution_failures'] = [dict(seed=1, arm=runner.AFTER, new_search_calls=None, call_accounting_complete=False)]
    runner.save(tmp_path/'source.json', source)
    accounts = runner.accounting(plan, [task], root=tmp_path)
    assert accounts['physical_target_calls'] is None
    assert accounts['known_recorded_physical_target_calls'] == 1496


def test_collect_separates_scoring_from_terminal_failure_and_finishes_both_children(tmp_path, monkeypatch):
    plan = source_fixture(tmp_path, monkeypatch)
    runner.freeze(plan, root=tmp_path)
    task = task_fixture(plan, plan['tasks'][0])
    task['status'] = 'scoring_failed'
    task['streams'][0] = runner.failed_stream(plan, plan['tasks'][0], plan['streams'][0], 'retained truth failure')
    runner.save(tmp_path/plan['output_dir']/'cases/seed_1/result.json', task)
    runner.save(tmp_path/'plan.json', plan)
    result = runner.collect(plan, 'plan.json', root=tmp_path)
    assert result['status'] == 'complete_with_retained_failures'
    assert result['terminal_failures'] == [] and len(result['scoring_failures']) == 1
    assert runner.read(tmp_path/'child_1.json')['status'] == 'complete_with_retained_failures'
    assert runner.read(tmp_path/'child_2.json')['status'] == 'complete'
    summary = runner.read(tmp_path/result['headline_file'])
    assert len(summary['independent']['controls']) == 9
    assert len(summary['independent']['comparisons'][0]['rows']) == 1
    assert 'paths' not in summary and 'candidates' not in json.dumps(summary)
    assert result['costs']['physical_target_calls'] == 0  # No simulation in this substitute.


def test_uninstrumented_nonrole_initial_repeat_count_stays_unknown():
    point = (0,)
    raw = dict(policies=[dict(point=[0], policy_id='local0')], states=[])
    item = dict(frozen_shortlist=[dict(point=[0])])
    record = dict(deployed=[0], verification=dict(certified=True, verification_budget=80))
    case = dict(seed=1, domain='family', arm='same_source_gp', history_arm='same_source_gp',
        old=dict(status='complete', item=item, record=record))
    truth = dict(quality=.5, true_feasible=True)
    outcomes, _, _ = runner.attach_paths(raw, [case], {}, INITIAL, {point: truth}, {point:'P0'}, truth, 'defined')
    assert outcomes[0]['discovery_category'] == 'initial'
    assert outcomes[0]['returned_repeat_count'] is None
