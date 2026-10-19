"""Zero-simulation checks of scheduler boundaries and real CLI parsing."""
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import pytest

from performance import mainline_cluster_dispatch as cluster


def test_complete_manifest_rejected_without_loading_models(tmp_path):
    path = tmp_path / 'plan.json'
    cluster.save(path, {'status': 'complete'})
    with pytest.raises(ValueError, match='cannot be restarted'):
        cluster.describe(path, 'initial')


def test_local_worker_rejected_before_reading_descriptor():
    with patch.object(cluster.socket, 'gethostname', return_value='local-machine'):
        with pytest.raises(RuntimeError, match='only on node001-node006'):
            cluster.worker('missing.json', 0, 1, 1)


def test_online_requires_completed_neighbor_stage(tmp_path):
    plan = dict(status='registered', output_dir=str(tmp_path))
    with patch.object(cluster, 'active_plan', return_value=plan), \
         patch.object(cluster.run, 'prepare', return_value={}), \
         patch.object(cluster.run.replication, 'search_jobs', side_effect=AssertionError('must not launch')):
        with pytest.raises(FileNotFoundError):
            cluster.describe(tmp_path / 'plan.json', 'online')


def test_finish_rejects_missing_or_overlapping_shards(tmp_path):
    plan = dict(status='registered', output_dir=str(tmp_path))
    descriptor = tmp_path / 'jobs.json'
    cluster.save(descriptor, dict(manifest='unused.json', stage='neighbors', jobs=[dict(label='a'), dict(label='b')]))
    cluster.save(tmp_path / 'cluster/neighbors/shard_0_1/result.json', dict(labels=['a'], failed=[]))
    with patch.object(cluster, 'active_plan', return_value=plan):
        with pytest.raises(AssertionError, match='Unfinished or overlapping'):
            cluster.finish(descriptor)
        cluster.save(tmp_path / 'cluster/neighbors/shard_1_2/result.json', dict(labels=['a'], failed=[]))
        with pytest.raises(AssertionError, match='Unfinished or overlapping'):
            cluster.finish(descriptor)
        cluster.save(tmp_path / 'cluster/neighbors/shard_1_2/result.json', dict(labels=['b'], failed=[]))
        assert cluster.finish(descriptor)['jobs'] == 2


def scheduler_modules():
    directory = cluster.ROOT.parents[1] / 'scheduleurm/skill'
    if not directory.exists():
        pytest.skip('Scheduler checkout is only needed for CLI integration checks')
    sys.path.insert(0, str(directory))
    from scheduler_commands.cli import build_parser, SchedulerCliDeps
    from scheduler_cpu.batch_plan import cpu_batch_plan
    return build_parser, SchedulerCliDeps, cpu_batch_plan


def test_submit_parses_with_actual_scheduler_cli():
    build_parser, deps_class, _ = scheduler_modules()
    values = {field.name: (lambda *a, **k: None) if field.name.startswith('cmd_') else 1
              for field in fields(deps_class)}
    values['node_names'] = list(cluster.DEFAULT_CPU_NODES)
    parser = build_parser(deps_class(**values))
    args = SimpleNamespace(descriptor=str(cluster.ROOT / 'jobs.json'),
        remote_python='/remote/python', remote_cwd='/remote/SC-OLH-KG',
        scheduler='/scheduler.py', ram_mb=16384, dry_run=True)
    descriptor = dict(manifest='plan.json', stage='online', jobs=[{}] * 144)
    with patch.object(cluster, 'active_plan', return_value=dict(name='test')):
        command = cluster.submit_command(args, descriptor)
    parsed = parser.parse_args(command[2:])
    assert parsed.items == 144 and parsed.skip_launch_staging and parsed.env_spec == 'none'
    assert parsed.nodes.split(',') == list(cluster.DEFAULT_CPU_NODES)
    assert parsed.result_dir_template is None and parsed.local_result_dir_template is None
    assert '--start' in parsed.cmd_template and '{workers}' in parsed.cmd_template


def test_explicit_six_node_pool_has_no_local_fallback():
    _, _, cpu_batch_plan = scheduler_modules()
    nodes = list(cluster.DEFAULT_CPU_NODES)
    infos = {n: dict(cpu=96, physical_cores=96, os='linux') for n in nodes}
    full = {n: dict(alive=True, free_cpu=0, total_cpu=96) for n in nodes}
    assert cpu_batch_plan(144, nodes, node_states=full, node_info_by_name=infos) == []
    available = {n: dict(alive=True, free_cpu=96, total_cpu=96) for n in nodes}
    plan = cpu_batch_plan(144, nodes, node_states=available, node_info_by_name=infos)
    assert {p['node'] for p in plan} == set(nodes)
    assert sum(p['items'] for p in plan) == 144
