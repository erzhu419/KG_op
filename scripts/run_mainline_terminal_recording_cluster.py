#!/usr/bin/env python3
"""Resume the two registered streams through shared scheduler stage barriers."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shlex
import sys


spec = importlib.util.spec_from_file_location(
    'safe_unsampled_controller', Path(__file__).with_name('run_mainline_safe_unsampled_cluster.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

MANIFEST = 'performance/manifests/mainline_terminal_recording_seed_replication_v1.json'
AUDIT = 'performance.run_mainline_terminal_recording_replication'


class Controller(base.Controller):
    stage_label = 'Stage76'
    signature_stage = 'stage76'

    def __init__(self, manifest=MANIFEST, poll_seconds=30):
        self.manifest = manifest
        self.plan = base.read(base.PROJECT / manifest)
        assert self.plan['name'] == 'mainline_terminal_recording_seed_replication_v1'
        assert len(self.plan['tasks']) == 24 and len(self.plan['streams']) == 2
        self.streams = [dict(stream, plan=base.read(base.PROJECT / stream['manifest']))
                        for stream in self.plan['streams']]
        self.output = base.PROJECT / self.plan['output_dir'] / 'cluster_control'
        self.state_path = self.output / 'stage76_state.json'
        self.state = base.read(self.state_path) if self.state_path.exists() else dict(
            manifest=manifest, status='not_started', stages={}, jobs={}, attempts=[],
            node_pool=base.NODES, remote_cwd=base.REMOTE, remote_python=base.PYTHON,
            local_scientific_computation=False)
        self.poll_seconds = poll_seconds

    def run_stage(self, stage):
        state = self.state['stages'].setdefault(stage, {})
        if state.get('status') == 'complete':
            return
        index = base.STAGES.index(stage)
        if index and self.state['stages'].get(base.STAGES[index - 1], {}).get('status') != 'complete':
            raise RuntimeError('The preceding global stage is not complete for both streams')
        children = state.setdefault('children', {})
        for stream in self.streams:
            child = children.setdefault(stream['name'], {})
            if child.get('status') == 'complete':
                continue
            label = stage + '/' + stream['name']
            descriptor = str(Path(stream['plan']['output_dir']) / 'cluster' / stage / 'jobs.json')
            self.job(label + '/describe', 1, base.module('performance.mainline_cluster_dispatch',
                'describe', '--manifest', stream['manifest'], '--stage', stage))
            detail = self.metadata([descriptor], 'descriptor')[descriptor]
            child['registered_jobs'] = detail['jobs']
            self.job(label + '/workers', detail['jobs'], base.module(
                'performance.mainline_cluster_dispatch', 'worker', '--descriptor', descriptor,
                '--start', '{start}', '--end', '{end}', '--workers', '{workers}'))
            self.job(label + '/finish', 1, base.module('performance.mainline_cluster_dispatch',
                'finish', '--descriptor', descriptor))
            completion = str(Path(stream['plan']['output_dir']) / 'cluster' / stage / 'completion.json')
            result = self.metadata([completion], 'stage')[completion]
            if result['status'] != 'complete' or result['truth_used']:
                raise RuntimeError('Child completion did not satisfy the frozen pre-truth barrier')
            child.update(status='complete', completion=result)
            self.update()
            print(json.dumps(dict(stage=stage, stream=stream['name'], status='complete',
                jobs=result['jobs'], failed_jobs=len(result['failed_jobs']))), flush=True)
        if any(children[s['name']].get('status') != 'complete' for s in self.streams):
            raise RuntimeError('Both streams must finish before the next global stage')
        state['status'] = 'complete'
        self.update()

    def require_terminal_seals(self):
        paths = [str(Path(s['plan']['output_dir']) / 'execution_status.json')
                 for s in self.streams]
        seals = self.metadata(paths, 'seal')
        if any(seals[p].get('status') != 'sealed'
               or seals[p].get('registered_terminal_cells') != 432
               or seals[p].get('truth_used') for p in paths):
            raise RuntimeError('Both streams and all 864 terminal outcomes must be sealed before truth')
        self.state['terminal_seals'] = seals
        self.update()

    def mark_started(self):
        if self.state.get('remote_start_recorded'):
            return
        sys.path.insert(0, str(base.SCHEDULER.parent))
        import scheduler
        source = (
            'import json; from pathlib import Path; from datetime import datetime, timezone\n'
            'path=Path(' + repr(base.REMOTE) + ')/' + repr(self.manifest) + '\n'
            "plan=json.loads(path.read_text())\n"
            "assert plan['execution_preparation']['status']=='validated'\n"
            "if plan['status'] not in ('complete','complete_with_failures'):\n"
            " plan['execution_started']=True; plan['status']='search_running'\n"
            " plan.setdefault('execution_started_utc',datetime.now(timezone.utc).isoformat())\n"
            " path.write_text(json.dumps(plan,indent=2)+'\\n')\n"
            "print(json.dumps({'status':plan['status'],'execution_started':plan['execution_started']}))\n")
        rc, stdout, stderr = scheduler.run_on('node002', shlex.join([base.PYTHON, '-c', source]),
                                             timeout=45, check=False)
        self.state['attempts'].append(dict(label='metadata/mark_started', returncode=rc,
            stdout=stdout, stderr=stderr, payload_bytes=len(stdout.encode()),
            utc=datetime.now(timezone.utc).isoformat()))
        self.update()
        if rc:
            raise RuntimeError('Execution admission metadata failed; no scientific job submitted')
        self.update(remote_start_recorded=True)

    def run(self):
        if self.state['status'] == 'complete':
            print('Stage76 controller already complete', flush=True)
            return
        try:
            self.mark_started()
            for stage in base.STAGES:
                self.run_stage(stage)
            self.require_terminal_seals()
            self.job('audit/freeze', 1, base.module(AUDIT, 'freeze', '--manifest', self.manifest))
            self.job('audit/tasks', 24, base.module(AUDIT, 'batch', '--manifest', self.manifest,
                '--start', '{start}', '--end', '{end}', '--workers', '{workers}'))
            self.job('audit/collect', 1, base.module(AUDIT, 'collect', '--manifest', self.manifest))
            self.update(status='complete', waiting_for=None)
        except Exception as error:
            self.update(status='blocked', error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    Controller(args.manifest, args.poll_seconds).run()


if __name__ == '__main__':
    main()
