"""Check that Stage86 admits only the registered saved-record analysis stages."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


spec = importlib.util.spec_from_file_location('stage86_controller',
    Path(__file__).resolve().parents[2] / 'scripts/run_mainline_precision_propagation_paths_cluster.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def fixture_controller(checks='passed'):
    value = controller.Controller.__new__(controller.Controller)
    value.state = dict(status='running', stages={}, jobs={})
    value.plan = dict(output_dir='fixture')
    value.manifest = 'fixture.json'
    value.job = Mock()
    value.update = lambda **fields: value.state.update(fields)
    value.metadata = lambda paths: {paths[0]: dict(status=checks)}
    return value


def test_passed_validation_runs_144_saved_analysis_items_then_collect():
    value = fixture_controller()
    value.run()
    calls = value.job.call_args_list
    assert [(call.args[0], call.args[1]) for call in calls] == [
        ('validation/stage86_attempt1', 1), ('audit/tasks', 144), ('audit/collect', 1)]
    assert calls[1].args[2] == controller.base.module(controller.AUDIT, 'batch',
        '--manifest', 'fixture.json', '--start', '{start}', '--end', '{end}', '--workers', '{workers}')
    assert calls[2].args[2] == controller.base.module(controller.AUDIT, 'collect', '--manifest', 'fixture.json')
    assert value.state['status'] == 'complete'


def test_failed_substitutes_block_all_analysis():
    value = fixture_controller('failed')
    with pytest.raises(RuntimeError, match='substitute checks'):
        value.run()
    assert [call.args[0] for call in value.job.call_args_list] == ['validation/stage86_attempt1']
    assert value.state['status'] == 'blocked'


def test_completed_analysis_never_submits_again():
    value = fixture_controller()
    value.state['status'] = 'complete'
    value.run()
    value.job.assert_not_called()
