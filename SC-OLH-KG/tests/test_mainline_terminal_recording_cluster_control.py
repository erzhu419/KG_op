"""Two-stream ordering and resume tests without scheduler or scientific calls."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location('recording_controller',
    Path(__file__).resolve().parents[2] / 'scripts/run_mainline_terminal_recording_cluster.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def fixture_controller():
    value = controller.Controller.__new__(controller.Controller)
    value.manifest = 'outer.json'
    value.plan = dict(output_dir='outer')
    value.streams = [dict(name=f'observation_{i}', manifest=f'child{i}.json',
                         plan=dict(output_dir=f'child{i}')) for i in (1, 2)]
    value.state = dict(status='running', stages={}, jobs={})
    value.update = lambda **fields: value.state.update(fields)
    value.mark_started = Mock()
    return value


def test_next_stage_blocked_when_second_child_has_not_finished():
    value = fixture_controller()
    value.state['stages']['neighbors'] = dict(children={
        'observation_1': dict(status='complete'), 'observation_2': dict(status='running')})
    value.job = Mock()
    with pytest.raises(RuntimeError, match='both streams'):
        value.run_stage('online')
    value.job.assert_not_called()


def test_resume_skips_finished_child_and_preserves_its_failure_account():
    value = fixture_controller()
    retained = dict(status='complete', completion=dict(failed_jobs=[dict(seed=7)]))
    value.state['stages']['initial'] = dict(children={'observation_1': retained})
    value.job = Mock()
    value.metadata = lambda paths, kind: {paths[0]: dict(jobs=24) if kind == 'descriptor'
        else dict(status='complete', jobs=24, truth_used=False, failed_jobs=[])}
    value.run_stage('initial')
    assert [call.args[0] for call in value.job.call_args_list] == [
        'initial/observation_2/describe', 'initial/observation_2/workers',
        'initial/observation_2/finish']
    assert value.state['stages']['initial']['status'] == 'complete'
    assert value.state['stages']['initial']['children']['observation_1'] is retained
    assert retained['completion']['failed_jobs'] == [dict(seed=7)]


def test_second_finish_interruption_does_not_seal_stage_or_start_audit():
    value = fixture_controller()
    def invoke(label, *_):
        if label == 'initial/observation_2/finish':
            raise RuntimeError('pending finish')
    value.job = Mock(side_effect=invoke)
    value.metadata = lambda paths, kind: {paths[0]: dict(jobs=24) if kind == 'descriptor'
        else dict(status='complete', jobs=24, truth_used=False, failed_jobs=[])}
    with pytest.raises(RuntimeError, match='pending finish'):
        value.run()
    state = value.state['stages']['initial']
    assert state.get('status') != 'complete'
    assert state['children']['observation_1']['status'] == 'complete'
    assert not any(call.args[0].startswith(('neighbors', 'audit'))
                   for call in value.job.call_args_list)


def test_only_one_terminal_seal_cannot_enable_inventory_or_truth():
    value = fixture_controller()
    value.run_stage = Mock()
    value.job = Mock()
    value.metadata = lambda paths, kind: {
        path: dict(status='sealed', registered_terminal_cells=432 if i == 0 else 431,
                   truth_used=False) for i, path in enumerate(paths)}
    with pytest.raises(RuntimeError, match='864'):
        value.run()
    value.job.assert_not_called()


def test_existing_submitted_job_is_waited_without_second_submission():
    value = fixture_controller()
    value.state['jobs']['initial/observation_2/workers'] = dict(
        task_ids=['t_saved'], record_paths=['saved.json'])
    value.wait = Mock(return_value=[dict(id='t_saved', status='done', exit_code=0)])
    value.metadata = lambda paths: {'saved.json': dict(status='complete', child_returncode=0)}
    value.capture = Mock()
    value.job('initial/observation_2/workers', 24, ['never_resubmit'])
    value.capture.assert_not_called()
    value.wait.assert_called_once_with(['t_saved'], 'initial/observation_2/workers')
    assert value.state['jobs']['initial/observation_2/workers']['status'] == 'complete'


def test_interrupted_submission_requires_reconciliation_without_resubmit():
    value = fixture_controller()
    value.state['jobs']['initial/observation_2/workers'] = dict(submission_started='saved')
    value.capture = Mock()
    with pytest.raises(RuntimeError, match='interrupted submission'):
        value.job('initial/observation_2/workers', 24, ['never_resubmit'])
    value.capture.assert_not_called()
