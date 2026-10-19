"""Target admission and post-3168-cell routing with metadata substitutes only."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest


spec = importlib.util.spec_from_file_location('source_alignment_cluster_controller',
    Path(__file__).resolve().parents[2] / 'scripts/run_mainline_source_alignment_science_cluster.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def plan():
    return dict(name='mainline_source_alignment_targets_v1', output_dir='fixture',
        tasks=[dict(seed=i) for i in range(144)], streams=['stream1', 'stream2'],
        search_arms=list(range(9)), terminal_arms=list(range(11)), terminal_protocols=['old'],
        budget=dict(registered_terminal_cells=3168, maximum_verification_calls_per_cell=352),
        source_fitting_started=True, execution_started=True, scientific_execution_started=True,
        target_execution_preparation=dict(status='validated'), target_execution_authorized=True)


@pytest.mark.parametrize('preparation,authorized', [
    ({}, False), ({'status': 'pending'}, True), ({'status': 'validated'}, False),
    ({'status': 'complete'}, True),
])
def test_completed_source_fits_never_admit_target_execution(tmp_path, monkeypatch,
                                                          preparation, authorized):
    value = plan()
    value.update(target_execution_preparation=preparation,
                 target_execution_authorized=authorized)
    monkeypatch.setattr(controller.base, 'PROJECT', tmp_path)
    monkeypatch.setattr(controller.base, 'read', Mock(return_value=value))
    with pytest.raises(RuntimeError, match='Explicit validated target admission'):
        controller.Controller()
    assert list(tmp_path.iterdir()) == []


def test_target_admission_selects_distinct_state_and_frozen_352_budget(tmp_path, monkeypatch):
    value = plan()
    monkeypatch.setattr(controller.base, 'PROJECT', tmp_path)
    monkeypatch.setattr(controller.base, 'read', Mock(return_value=value))
    admitted = controller.Controller()
    assert admitted.state_path == tmp_path / 'fixture/cluster_control/source_alignment_science_state.json'
    assert admitted.state['node_pool'] == [f'node{i:03d}' for i in range(1, 7)]
    assert admitted.state['automatic_scientific_retries'] == 0
    assert admitted.state['target_execution_authorized'] is True
    assert admitted.state['jobs'] == {}
    assert list(tmp_path.iterdir()) == []
    value['budget']['maximum_verification_calls_per_cell'] = 353
    with pytest.raises(AssertionError):
        controller.Controller()


def fixture_controller(*, seal=None, inventory=None):
    value = controller.Controller.__new__(controller.Controller)
    value.state = dict(status='running', stages={}, jobs={}, attempts=[])
    value.plan = dict(output_dir='fixture')
    value.manifest = 'registered-targets.json'
    value.run_stage = Mock()
    value.job = Mock()
    value.update = lambda **fields: value.state.update(fields)
    seal = seal if seal is not None else dict(status='sealed', registered_terminal_cells=3168,
        truth_access_allowed=True, truth_used=False)
    inventory = inventory if inventory is not None else dict(status='sealed',
        registered_terminal_cells=3168, registered_query_states=4320,
        query_arms=list(controller.QUERY_ARMS), inventory_complete=False)
    value.metadata = Mock(side_effect=lambda paths, kind='full': {
        paths[0]: inventory if kind == 'alignment_inventory' else seal})
    return value


@pytest.mark.parametrize('change', [
    dict(registered_terminal_cells=2592), dict(registered_terminal_cells=3167),
    dict(truth_access_allowed=False), dict(truth_used=True), dict(status='running'),
])
def test_no_truth_dispatch_until_all_3168_cells_are_sealed(change):
    seal = dict(status='sealed', registered_terminal_cells=3168,
                truth_access_allowed=True, truth_used=False)
    seal.update(change)
    value = fixture_controller(seal=seal)
    with pytest.raises(RuntimeError, match='All 3168'):
        value.run()
    value.job.assert_not_called()
    assert [c.args[0] for c in value.run_stage.call_args_list] == list(controller.STAGES)
    assert value.state['status'] == 'blocked'


@pytest.mark.parametrize('change', [
    dict(query_arms=list(controller.QUERY_ARMS[:3])),
    dict(query_arms=list(controller.QUERY_ARMS[:4]) + [controller.QUERY_ARMS[0]]),
    dict(registered_query_states=2592), dict(registered_terminal_cells=3167),
])
def test_old_three_role_or_incomplete_inventory_never_dispatches_scoring(change):
    inventory = dict(status='sealed', registered_terminal_cells=3168,
        registered_query_states=4320, query_arms=list(controller.QUERY_ARMS))
    inventory.update(change)
    value = fixture_controller(inventory=inventory)
    with pytest.raises(RuntimeError, match='all five roles and 4320'):
        value.run()
    assert [call.args[0] for call in value.job.call_args_list] == ['truth/freeze']


def test_sealed_failures_remain_in_inventory_and_144_geometry_scoring_dispatch():
    value = fixture_controller()
    value.run()
    assert [call.args[0] for call in value.run_stage.call_args_list] == list(controller.STAGES)
    jobs = value.job.call_args_list
    assert [call.args[0] for call in jobs] == ['truth/freeze', 'truth/tasks', 'truth/collect']
    assert jobs[1].args[1] == 144
    assert all(controller.SUMMARY in call.args[2] for call in jobs)
    assert 'score-batch' in jobs[1].args[2]
    assert value.state['query_inventory']['inventory_complete'] is False
    assert value.state['status'] == 'complete'
    value.job.reset_mock()
    value.run_stage.reset_mock()
    value.run()
    value.job.assert_not_called()
    value.run_stage.assert_not_called()


@pytest.mark.parametrize('stage,previous_stage', [('online', 'neighbors'), ('terminal', 'native')])
def test_stage_workers_wait_for_preceding_global_barrier(stage, previous_stage):
    value = fixture_controller()
    value.run_stage = controller.Controller.run_stage.__get__(value)
    value.state['stages'][previous_stage] = dict(status='running')
    with pytest.raises(RuntimeError, match='preceding global stage'):
        value.run_stage(stage)
    value.job.assert_not_called()


def test_stage_routes_new_runtime_and_retains_documented_cell_failures():
    value = fixture_controller()
    value.run_stage = controller.Controller.run_stage.__get__(value)
    value.state['stages']['native'] = dict(status='complete')
    failure = dict(arm='source_permuted_role_point_only', seed=19, returncode=1)
    completion = dict(status='complete', truth_used=False, jobs=3168, failed_jobs=[failure])
    value.metadata = lambda paths, kind='full': {paths[0]:
        dict(jobs=3168) if kind == 'descriptor' else deepcopy(completion)}
    value.run_stage('terminal')
    jobs = value.job.call_args_list
    assert [c.args[0] for c in jobs] == ['terminal/describe', 'terminal/workers', 'terminal/finish']
    assert jobs[1].args[1] == 3168
    assert all(controller.DISPATCH in call.args[2] for call in jobs)
    assert value.state['stages']['terminal']['completion']['failed_jobs'] == [failure]
    value.job.reset_mock()
    value.run_stage('terminal')
    value.job.assert_not_called()


def test_inherited_cpu_dispatch_preserves_existing_ids_and_wrapper_fees():
    value = fixture_controller()
    value.plan['name'] = 'mainline_source_alignment_targets_v1'
    value.job = controller.Controller.job.__get__(value)
    captured = []

    def capture(label, command):
        captured.append((label, command))
        return json.dumps(dict(plan=[dict(start=0, end=1)])) + '\nsubmitted t123 cpu=1\n'

    value.capture = capture
    value.wait = Mock(return_value=[dict(id='t123', status='done', exit_code=0)])
    paid = dict(status='complete', child_returncode=0, total_cpu_seconds=7.25)
    value.metadata = lambda paths, kind='full': {p: deepcopy(paid) for p in paths}
    for _ in range(2):
        entry = value.job('initial/workers', 1, ['python', '-m', controller.DISPATCH, 'worker'])
    assert len(captured) == 1
    command = captured[0][1]
    assert command[2] == 'submit-cpu-batch'
    assert command[command.index('--nodes') + 1] == 'node001,node002,node003,node004,node005,node006'
    shell = command[command.index('--cmd-template') + 1]
    assert all(name + '=1' in shell for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'))
    assert 'source_alignment_targets' in command[command.index('--signature') + 1]
    assert entry['task_ids'] == ['t123']
    assert next(iter(entry['records'].values()))['total_cpu_seconds'] == 7.25


def test_inherited_wrapper_failure_stops_without_automatic_resubmission():
    value = fixture_controller()
    value.state['jobs']['initial/workers'] = dict(record_paths=['failed.json'], manual_dispatch={})
    value.wrapper_records = Mock(return_value={
        'failed.json': dict(status='child_failed', child_returncode=1)})
    value.capture = Mock()
    value.task_states = Mock()
    with pytest.raises(RuntimeError, match='Retained wrapper child failure'):
        value.wait(['t123'], 'initial/workers')
    value.capture.assert_not_called()
    value.task_states.assert_not_called()
