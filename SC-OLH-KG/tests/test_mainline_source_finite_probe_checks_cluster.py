"""Check the scheduler entry selects substitutes only and retains failures."""
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/run_mainline_source_finite_probe_checks_cluster.py'
spec = importlib.util.spec_from_file_location('finite_probe_checks_cluster', SCRIPT)
cluster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cluster)


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.setattr(cluster.base, 'PROJECT', tmp_path)
    plan = {'name': 'mainline_source_finite_numerical_validation_v1',
            'output_dir': 'output', 'execution_authorized': False}
    (tmp_path / 'manifest.json').write_text(json.dumps(plan))
    return cluster.Controller('manifest.json')


def test_only_explicit_substitutes_and_completed_checks_are_not_repeated(controller, monkeypatch):
    calls = []
    monkeypatch.setattr(controller, 'job', lambda *args: calls.append(args))
    monkeypatch.setattr(controller, 'metadata', lambda paths:
                        {paths[0]: {'status': 'passed', 'returncode': 0}})
    original = (cluster.base.PROJECT / 'manifest.json').read_text()
    controller.run()
    controller.run()
    assert len(calls) == 1 and calls[0][1] == 1
    command = calls[0][2]
    assert command[command.index('-m') + 1] == 'performance.run_mainline_source_alignment_validation'
    assert [command[i + 1] for i, value in enumerate(command) if value == '--test'] == list(cluster.TESTS)
    assert controller.state['status'] == 'complete'
    assert controller.state['node_pool'] == [f'node{i:03d}' for i in range(1, 7)]
    assert controller.state['workers'] == 1 and controller.state['vram'] == 0
    assert (cluster.base.PROJECT / 'manifest.json').read_text() == original


@pytest.mark.parametrize('child_failure', [False, True])
def test_failure_stops_without_automatic_retry(controller, monkeypatch, child_failure):
    calls = []
    def job(*args):
        calls.append(args)
        if child_failure:
            raise RuntimeError('Retained child failure')
    monkeypatch.setattr(controller, 'job', job)
    monkeypatch.setattr(controller, 'metadata', lambda paths:
                        {paths[0]: {'status': 'failed', 'returncode': 1}})
    with pytest.raises(RuntimeError):
        controller.run()
    assert len(calls) == 1
    assert controller.state['status'] == 'blocked'
    assert controller.state['automatic_scientific_retries'] == 0
