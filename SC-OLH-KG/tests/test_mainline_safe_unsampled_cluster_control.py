"""Deterministic gates against repeated paid work and premature truth access."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from performance import run_mainline_safe_unsampled_cluster_job as job


spec = importlib.util.spec_from_file_location('stage73_controller',
    Path(__file__).resolve().parents[2]/'scripts/run_mainline_safe_unsampled_cluster.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def test_duplicate_scheduler_launch_does_not_repeat_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(job,'ROOT',tmp_path)
    monkeypatch.setattr(job.socket,'gethostname',lambda:'node001')
    invoke=Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(job.subprocess,'run',invoke)
    record=tmp_path/'stage/shard.json'
    job.execute(record,['fixture'])
    original=record.read_text()
    job.execute(record,['fixture'])
    invoke.assert_called_once()
    assert record.read_text()==original
    assert json.loads(record.with_suffix('.duplicates.jsonl').read_text())['status']=='duplicate_not_executed'


def test_child_failure_is_retained_and_never_replayed(tmp_path, monkeypatch):
    monkeypatch.setattr(job,'ROOT',tmp_path)
    monkeypatch.setattr(job.socket,'gethostname',lambda:'node001')
    invoke=Mock(return_value=SimpleNamespace(returncode=7))
    monkeypatch.setattr(job.subprocess,'run',invoke)
    record=tmp_path/'stage/shard.json'
    assert job.execute(record,['fixture']) is None
    value=json.loads(record.read_text())
    assert value['status']=='child_failed' and value['child_returncode']==7
    job.execute(record,['fixture'])
    invoke.assert_called_once()


def test_online_cannot_pass_incomplete_neighbors_barrier():
    value=controller.Controller.__new__(controller.Controller)
    value.state=dict(stages={'neighbors':dict(status='running')})
    value.job=Mock()
    with pytest.raises(RuntimeError,match='preceding global stage'):
        value.run_stage('online')
    value.job.assert_not_called()


def test_inventory_cannot_start_before_all_432_terminal_outcomes_are_sealed():
    value=controller.Controller.__new__(controller.Controller)
    value.state=dict(status='running',stages={})
    value.plan=dict(output_dir='fixture')
    value.run_stage=Mock()
    value.job=Mock()
    value.update=lambda **fields:value.state.update(fields)
    value.metadata=lambda paths,kind:{paths[0]:dict(status='sealed',registered_terminal_cells=431,truth_used=False)}
    with pytest.raises(RuntimeError,match='All 432'):
        value.run()
    value.job.assert_not_called()


def test_interrupted_submission_is_not_submitted_again():
    value=controller.Controller.__new__(controller.Controller)
    value.state=dict(jobs={'initial/workers':dict(submission_started='fixture')})
    value.capture=Mock()
    with pytest.raises(RuntimeError,match='interrupted submission'):
        value.job('initial/workers',24,['fixture'])
    value.capture.assert_not_called()
