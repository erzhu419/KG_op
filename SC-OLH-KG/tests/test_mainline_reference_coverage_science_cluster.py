"""Catch early truth admission and duplicate execution at the new controller."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


spec = importlib.util.spec_from_file_location('stage81_controller',
    Path(__file__).resolve().parents[2] / 'scripts/run_mainline_reference_coverage_science_cluster.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def fixture_controller(seal):
    value = controller.Controller.__new__(controller.Controller)
    value.state = dict(status='running', stages={})
    value.plan = dict(output_dir='fixture')
    value.manifest = 'fixture.json'
    value.run_stage = Mock()
    value.job = Mock()
    value.update = lambda **fields: value.state.update(fields)
    value.metadata = lambda paths, kind='full': {paths[0]: (
        dict(status='passed') if paths[0].endswith('checks.json') else seal)}
    return value


@pytest.mark.parametrize('seal', [
    dict(status='sealed', registered_terminal_cells=863, truth_access_allowed=True, truth_used=False),
    dict(status='sealed', registered_terminal_cells=864, truth_access_allowed=False, truth_used=False),
    dict(status='sealed', registered_terminal_cells=864, truth_access_allowed=True, truth_used=True),
])
def test_truth_jobs_are_not_submitted_before_complete_unopened_seal(seal):
    value = fixture_controller(seal)
    with pytest.raises(RuntimeError, match='All 864'):
        value.run()
    assert [call.args[0] for call in value.job.call_args_list] == ['validation/stage81_attempt1']


def test_terminal_workers_cannot_precede_global_native_input_barrier():
    value = fixture_controller({})
    value.run_stage = controller.Controller.run_stage.__get__(value)
    value.state['stages']['native'] = dict(status='running')
    with pytest.raises(RuntimeError, match='preceding global stage'):
        value.run_stage('terminal')
    value.job.assert_not_called()


def test_completed_scientific_cohort_does_not_submit_again():
    value = fixture_controller({})
    value.state['status'] = 'complete'
    value.run()
    value.job.assert_not_called()
    value.run_stage.assert_not_called()
