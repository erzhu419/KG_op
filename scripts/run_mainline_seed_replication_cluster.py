#!/usr/bin/env python3
"""Control Stage67's two frozen streams; computation runs only through scheduler."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1] / 'SC-OLH-KG'
SCHEDULER = PROJECT.parent.parent / 'scheduleurm/skill/scheduler.py'
REMOTE = '/home/zhengliang01/scheduleurm_work/KG_op_mainline_cluster/SC-OLH-KG'
PYTHON = '/home/zhengliang01/scheduleurm_work/conda_envs/scomp-mainline-py310-20261017/bin/python'
NODES = [f'node{i:03d}' for i in range(1, 7)]
STAGES = ('initial', 'neighbors', 'online', 'native', 'old', 'new')
DEFAULT = 'performance/manifests/mainline_terminal_two_comparison_seed_replication_v1.json'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def command(module, *arguments):
    return shlex.join(['env', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1',
        'MKL_NUM_THREADS=1', 'PYTHONPATH=.', PYTHON, '-u', '-m', module, *arguments])


class Controller:
    def __init__(self, manifest, score_module, poll_seconds=30):
        self.manifest, self.score_module = manifest, score_module
        self.plan = read(PROJECT / manifest)
        if len(self.plan['streams']) != 2:
            raise ValueError('Stage67 requires its two registered streams')
        self.output = PROJECT / self.plan['output_dir'] / 'cluster_control'
        self.state_path = self.output / 'state.json'
        self.state = read(self.state_path) if self.state_path.exists() else dict(
            manifest=manifest, status='registered', stages={}, attempts=[],
            node_pool=NODES, remote_cwd=REMOTE, remote_python=PYTHON,
            local_scientific_computation=False)
        self.poll_seconds = poll_seconds

    def update(self, **fields):
        self.state.update(fields, updated_utc=datetime.now(timezone.utc).isoformat())
        save(self.state_path, self.state)

    def capture(self, label, argv):
        result = subprocess.run(argv, cwd=PROJECT, text=True, capture_output=True)
        record = dict(label=label, command=argv, returncode=result.returncode,
                      stdout=result.stdout, stderr=result.stderr,
                      utc=datetime.now(timezone.utc).isoformat())
        self.state['attempts'].append(record)
        self.update()
        if result.returncode:
            raise RuntimeError(f'{label} failed; recorded output retained in {self.state_path}')
        return result.stdout

    def remote(self, label, shell_command):
        sys.path.insert(0, str(SCHEDULER.parent))
        import scheduler
        rc, stdout, stderr = scheduler.run_on('node002', shell_command, timeout=60, check=False)
        self.state['attempts'].append(dict(label=label, returncode=rc, stderr=stderr,
            remote_command=shell_command, utc=datetime.now(timezone.utc).isoformat()))
        self.update()
        if rc:
            raise RuntimeError(f'{label} failed; preserve current state and inspect its record')
        return stdout

    def task_states(self, ids):
        queue = read(Path.home() / '.claude/scheduler/queue.json')['tasks']
        rows = queue.values() if isinstance(queue, dict) else queue
        selected = {r['id']: r for r in rows if r['id'] in ids}
        if set(selected) != set(ids):
            raise RuntimeError(f'Scheduler task missing from live queue: {set(ids)-set(selected)}')
        return [{key: selected[i].get(key) for key in (
            'id', 'status', 'node', 'cpu_cores', 'est_vram_mb', 'exit_code',
            'started_at', 'finished_at', 'last_block_reason', 'log_path')} for i in ids]

    def wait(self, ids, label):
        if not ids:
            return []
        while True:
            rows = self.task_states(ids)
            self.update(waiting_for=label, latest_tasks=rows)
            counts = {s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})}
            print(json.dumps(dict(stage=label, scheduler_status=counts)), flush=True)
            if any(r['status'] in ('failed', 'cancelled', 'lost') or
                   (r['status'] == 'done' and r['exit_code'] not in (None, 0)) for r in rows):
                raise RuntimeError(f'Scheduler failure in {label}; no automatic resubmission')
            if all(r['status'] == 'done' for r in rows):
                return rows
            time.sleep(self.poll_seconds)

    def submit_one(self, label, shell_command):
        args = [sys.executable, str(SCHEDULER), 'submit', '--description', label,
            '--cmd', shell_command, '--cwd', REMOTE,
            '--signature', f"KG_op/{self.plan['name']}/{label}",
            '--cpu', '1', '--vram', '0', '--ram-mb', '32768',
            '--project', 'KG_op', '--skip-launch-staging', '--env-spec', 'none']
        for node in NODES:
            args += ['--allowed-node', node]
        output = self.capture(label, args)
        ids = re.findall(r'^submitted (t\d+)\b', output, re.MULTILINE)
        if len(ids) != 1:
            raise RuntimeError(f'Could not identify submitted task for {label}; inspect before resuming')
        return ids

    def dispatch(self, label):
        self.capture(label, [sys.executable, str(SCHEDULER), 'dispatch'])

    def run_stage(self, stage):
        state = self.state['stages'].setdefault(stage, dict(streams={}))
        if state.get('status') == 'complete':
            return
        for stream in self.plan['streams']:
            key = stream['name']
            entry = state['streams'].setdefault(key, {})
            child = read(PROJECT / stream['manifest'])
            descriptor = str(Path(child['output_dir']) / 'cluster' / stage / 'jobs.json')
            if 'task_ids' in entry:
                continue
            self.remote(f'{stage}/{key}/describe', 'cd ' + shlex.quote(REMOTE) + ' && ' +
                command('performance.mainline_cluster_dispatch', 'describe', '--manifest',
                        stream['manifest'], '--stage', stage))
            contents = self.remote(f'{stage}/{key}/descriptor',
                'cat ' + shlex.quote(REMOTE + '/' + descriptor))
            target = PROJECT / descriptor
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents)
            jobs = read(target)['jobs']
            entry.update(descriptor=descriptor, registered_jobs=len(jobs))
            if jobs:
                output = self.capture(f'{stage}/{key}/submit', [sys.executable, '-m',
                    'performance.mainline_cluster_dispatch', 'submit', '--descriptor', descriptor,
                    '--remote-cwd', REMOTE, '--remote-python', PYTHON, '--ram-mb', '32768'])
                entry['task_ids'] = re.findall(r'^submitted (t\d+)\b', output, re.MULTILINE)
                if not entry['task_ids']:
                    raise RuntimeError('Submission returned no task IDs; inspect before resuming')
            else:
                entry['task_ids'] = []
            self.update(status='running', active_stage=stage)
        worker_ids = [i for e in state['streams'].values() for i in e['task_ids']]
        if not state.get('workers_dispatched'):
            self.dispatch(stage + '/workers_dispatch')
            state['workers_dispatched'] = True
            self.update()
        state['worker_tasks'] = self.wait(worker_ids, stage + '/workers')
        for key, entry in state['streams'].items():
            if 'finish_ids' not in entry:
                entry['finish_ids'] = self.submit_one(stage + '/' + key + '/finish',
                    command('performance.mainline_cluster_dispatch', 'finish', '--descriptor', entry['descriptor']))
                self.update()
        finish_ids = [i for e in state['streams'].values() for i in e['finish_ids']]
        if not state.get('finish_dispatched'):
            self.dispatch(stage + '/finish_dispatch')
            state['finish_dispatched'] = True
            self.update()
        state['finish_tasks'] = self.wait(finish_ids, stage + '/finish')
        state['status'] = 'complete'
        self.update()
        print(json.dumps(dict(stage=stage, status='complete')), flush=True)

    def run(self):
        if self.state['status'] == 'complete':
            print('Stage67 controller already complete', flush=True)
            return
        try:
            for stage in STAGES:
                self.run_stage(stage)
            if 'score_ids' not in self.state:
                self.state['score_ids'] = self.submit_one('score', command(self.score_module,
                    'score', '--manifest', self.manifest))
                self.update()
                self.dispatch('score_dispatch')
            rows = self.wait(self.state['score_ids'], 'score')
            self.update(status='complete', score_tasks=rows, waiting_for=None)
        except Exception as error:
            self.update(status='blocked', error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=DEFAULT)
    parser.add_argument('--score-module', default='performance.summarize_mainline_confirmation_seed_replication')
    args = parser.parse_args()
    Controller(args.manifest, args.score_module).run()


if __name__ == '__main__':
    main()
