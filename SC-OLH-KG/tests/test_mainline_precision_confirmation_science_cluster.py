"""Protect the two-stream barriers and single dispatch of existing stage jobs."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


spec = importlib.util.spec_from_file_location('stage85_controller',
    Path(__file__).resolve().parents[2] / 'scripts/run_mainline_precision_confirmation_science_cluster.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def fixture_controller(seal):
    value = controller.Controller.__new__(controller.Controller)
    value.state = dict(status='running', stages={}, jobs={})
    value.plan = dict(output_dir='fixture')
    value.manifest = 'fixture.json'
    value.run_stage = Mock()
    value.job = Mock()
    value.update = lambda **fields: value.state.update(fields)
    value.metadata = lambda paths, kind='full': {paths[0]: (
        dict(status='passed') if paths[0].endswith('checks.json') else seal)}
    return value


@pytest.mark.parametrize('seal', [
    dict(status='sealed', registered_terminal_cells=1296, truth_access_allowed=True, truth_used=False),
    dict(status='sealed', registered_terminal_cells=2591, truth_access_allowed=True, truth_used=False),
    dict(status='sealed', registered_terminal_cells=2592, truth_access_allowed=False, truth_used=False),
    dict(status='sealed', registered_terminal_cells=2592, truth_access_allowed=True, truth_used=True),
])
def test_truth_requires_both_streams_and_last_terminal_cell(seal):
    value = fixture_controller(seal)
    with pytest.raises(RuntimeError, match='All 2592'):
        value.run()
    assert [call.args[0] for call in value.job.call_args_list] == ['validation/stage85_attempt1']


def test_terminal_workers_wait_for_global_native_input_barrier():
    value = fixture_controller({})
    value.run_stage = controller.Controller.run_stage.__get__(value)
    value.state['stages']['native'] = dict(status='running')
    with pytest.raises(RuntimeError, match='preceding global stage'):
        value.run_stage('terminal')
    value.job.assert_not_called()


def test_completed_cohort_never_submits_science_again():
    value = fixture_controller({})
    value.state['status'] = 'complete'
    value.run()
    value.job.assert_not_called()
    value.run_stage.assert_not_called()


def test_manual_dispatch_targets_submitted_ids_once():
    value = fixture_controller({})
    value.state['jobs']['initial/workers'] = dict(record_paths=['a.json', 'b.json'])
    value.capture = Mock()
    value.wrapper_records = Mock(return_value={name: dict(status='complete', child_returncode=0)
                                               for name in ['a.json', 'b.json']})
    value.task_states = Mock(return_value=[dict(id=task, status='done', exit_code=0)
                                          for task in ['t100', 't101']])
    for _ in range(2):
        value.wait(['t100', 't101'], 'initial/workers')
    value.capture.assert_called_once()
    command = value.capture.call_args.args[1]
    assert command[2:] == ['dispatch', '--task-id', 't100', '--task-id', 't101']
    account = value.state['jobs']['initial/workers']['manual_dispatch']
    assert account['task_ids'] == ['t100', 't101']
    assert account['new_submissions'] == account['automatic_scientific_retries'] == 0
    assert value.task_states.call_count == 2


def test_sealed_inventory_leads_to_144_geometry_scoring_items():
    value = fixture_controller(dict(status='sealed', registered_terminal_cells=2592,
                                    truth_access_allowed=True, truth_used=False))
    value.run()
    assert [call.args[0] for call in value.job.call_args_list] == [
        'validation/stage85_attempt1', 'truth/freeze', 'truth/tasks', 'truth/collect']
    score = value.job.call_args_list[2]
    assert score.args[1] == 144
    assert value.state['status'] == 'complete'


def waiting_fixture(record):
    value = fixture_controller({})
    value.poll_seconds = 1
    value.state['jobs']['initial/workers'] = dict(record_paths=['a.json'], manual_dispatch={})
    value.wrapper_records = Mock(return_value={'a.json': record})
    value.task_states = Mock(return_value=[dict(id='t100', status='running', exit_code=0,
                                               last_block_reason='terminal diagnosis pending')])
    return value


def test_wrapper_completion_advances_without_rewriting_late_scheduler_status():
    value = waiting_fixture(dict(status='complete', child_returncode=0))
    rows = value.wait(['t100'], 'initial/workers')
    assert rows[0]['status'] == 'running'
    assert value.state['jobs']['initial/workers']['scheduler_recovery_pending'] == ['t100']


@pytest.mark.parametrize('record', [dict(status='missing'), dict(status='started'), dict(status='writing')])
def test_wrapper_completion_missing_or_unfinished_waits(record, monkeypatch):
    value = waiting_fixture(record)
    class WaitObserved(Exception):
        pass
    monkeypatch.setattr(controller.time, 'sleep', Mock(side_effect=WaitObserved))
    with pytest.raises(WaitObserved):
        value.wait(['t100'], 'initial/workers')
    assert 'completion_basis' not in value.state['jobs']['initial/workers']


def test_wrapper_completion_child_failure_blocks_without_retry():
    value = waiting_fixture(dict(status='child_failed', child_returncode=1))
    with pytest.raises(RuntimeError, match='Retained wrapper child failure'):
        value.wait(['t100'], 'initial/workers')
    assert 'completion_basis' not in value.state['jobs']['initial/workers']
