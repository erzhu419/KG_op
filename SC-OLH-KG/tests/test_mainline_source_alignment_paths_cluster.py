"""Stage94 may dispatch saved-record analysis only, after its specific checks."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock
import pytest

spec = importlib.util.spec_from_file_location('stage94_controller',
    Path(__file__).resolve().parents[2] / 'scripts/run_mainline_source_alignment_paths_cluster.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def make_controller(checks='passed'):
    value = controller.Controller.__new__(controller.Controller)
    value.state = dict(status='running', stages={}, jobs={})
    value.plan = dict(output_dir='fixture')
    value.manifest = 'fixture.json'
    value.job = Mock()
    value.update = lambda **fields: value.state.update(fields)
    value.metadata = lambda paths: {paths[0]: dict(status=checks)}
    return value


def test_checks_then_144_saved_items_then_collect_only():
    value = make_controller()
    value.run()
    calls = value.job.call_args_list
    assert [(c.args[0], c.args[1]) for c in calls] == [
        ('validation/stage94_attempt1', 1), ('audit/tasks', 144), ('audit/collect', 1)]
    assert calls[1].args[2] == controller.base.module(controller.AUDIT, 'batch',
        '--manifest', 'fixture.json', '--start', '{start}', '--end', '{end}', '--workers', '{workers}')
    assert value.state['status'] == 'complete'


def test_failed_substitutes_block_analysis():
    value = make_controller('failed')
    with pytest.raises(RuntimeError, match='substitute checks'):
        value.run()
    assert len(value.job.call_args_list) == 1
    assert value.state['status'] == 'blocked'


def test_completed_analysis_is_not_resubmitted():
    value = make_controller()
    value.state['status'] = 'complete'
    value.run()
    value.job.assert_not_called()
