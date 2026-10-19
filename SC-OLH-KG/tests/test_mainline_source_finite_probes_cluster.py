"""Check the source-only scheduler barriers without launching any jobs."""
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/run_mainline_source_finite_probes_cluster.py'
spec = importlib.util.spec_from_file_location('finite_probes_cluster', SCRIPT)
cluster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cluster)


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.setattr(cluster.base, 'PROJECT', tmp_path)
    plan = {'name': 'mainline_source_finite_numerical_validation_v1',
            'output_dir': 'output', 'execution_authorized': True,
            'cases': [{'case_id': str(i)} for i in range(12)]}
    (tmp_path / 'manifest.json').write_text(json.dumps(plan))
    return cluster.Controller('manifest.json')


def test_only_new_checks_then_twelve_fixed_cases_then_collect(controller, monkeypatch):
    jobs = []
    monkeypatch.setattr(controller, 'job', lambda *args: jobs.append(args))
    monkeypatch.setattr(controller, 'metadata', lambda paths:
                        {paths[0]: {'status': 'passed', 'returncode': 0}})
    original = (cluster.base.PROJECT / 'manifest.json').read_text()
    controller.run()
    controller.run()
    assert [(job[0], job[1]) for job in jobs] == [
        ('validation/stage100_attempt1', 1), ('probes/batch', 12), ('probes/collect', 1)]
    checks = jobs[0][2]
    assert [checks[i + 1] for i, value in enumerate(checks) if value == '--test'] == list(cluster.TESTS)
    assert '--workers' in jobs[1][2] and '{workers}' in jobs[1][2]
    assert jobs[2][2][jobs[2][2].index('--action') + 1] == 'collect'
    assert controller.state['node_pool'] == [f'node{i:03d}' for i in range(1, 7)]
    assert controller.state['vram'] == 0
    assert (cluster.base.PROJECT / 'manifest.json').read_text() == original


@pytest.mark.parametrize('failure,expected_jobs', [
    ('checks', 1), ('authorization', 1), ('batch', 2),
])
def test_unmet_barrier_never_dispatches_dependent_work(controller, monkeypatch, failure, expected_jobs):
    jobs = []
    def job(*args):
        jobs.append(args)
        if failure == 'batch' and args[0] == 'probes/batch':
            raise RuntimeError('Retained failed shard wrapper')
    monkeypatch.setattr(controller, 'job', job)
    monkeypatch.setattr(controller, 'metadata', lambda paths:
                        {paths[0]: {'status': 'failed' if failure == 'checks' else 'passed',
                                    'returncode': 1 if failure == 'checks' else 0}})
    if failure == 'authorization':
        controller.plan['execution_authorized'] = False
    with pytest.raises(RuntimeError):
        controller.run()
    assert len(jobs) == expected_jobs
    assert controller.state['status'] == 'blocked'
    assert controller.state['automatic_scientific_retries'] == 0
