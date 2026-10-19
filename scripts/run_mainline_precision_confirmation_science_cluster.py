#!/usr/bin/env python3
"""Execute the admitted old-only two-stream cohort through scheduled global stages."""
import argparse
import importlib.util
import resource
import time
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'safe_unsampled_controller', Path(__file__).with_name('run_mainline_safe_unsampled_cluster.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

MANIFEST = 'performance/manifests/mainline_precision_confirmation_v1.json'
DISPATCH = 'performance.run_mainline_precision_confirmation_science'
SUMMARY = 'performance.summarize_mainline_precision_confirmation'
STAGES = ('initial', 'neighbors', 'online', 'native', 'terminal')


def wrappers_complete(records):
    for row in records.values():
        if row.get('status') in ('child_failed', 'launch_failed') or row.get('child_returncode') not in (None, 0):
            raise RuntimeError('Retained wrapper child failure; no automatic resubmission')
    return bool(records) and all(row.get('status') == 'complete' and row.get('child_returncode') == 0
                                 for row in records.values())


class Controller(base.Controller):
    stage_label = 'Stage85'
    signature_stage = 'stage85'

    def __init__(self, manifest=MANIFEST, poll_seconds=30):
        self.manifest = manifest
        self.plan = base.read(base.PROJECT / manifest)
        assert self.plan['name'] == 'mainline_precision_confirmation_v1'
        assert len(self.plan['tasks']) == 144 and len(self.plan['terminal_arms']) == 9
        assert len(self.plan['streams']) == 2
        assert self.plan['terminal_protocols'] == ['old']
        assert self.plan['budget']['registered_terminal_cells'] == 2592
        self.output = base.PROJECT / self.plan['output_dir'] / 'cluster_control'
        self.state_path = self.output / 'stage85_state.json'
        self.state = base.read(self.state_path) if self.state_path.exists() else dict(
            manifest=manifest, status='not_started', stages={}, jobs={}, attempts=[],
            node_pool=base.NODES, remote_cwd=base.REMOTE, remote_python=base.PYTHON,
            local_scientific_computation=False, automatic_scientific_retries=0, scientific_execution_authorized=True)
        self.poll_seconds = poll_seconds

    def wrapper_records(self, paths):
        """Project this stage's existing completion records without changing queue state."""
        base.sys.path.insert(0, str(base.SCHEDULER.parent))
        import scheduler
        script = ("import json,time; from pathlib import Path; start=time.process_time(); root=Path(" +
            repr(base.REMOTE) + "); paths=" + repr(paths) + "; records={}\n"
            "for name in paths:\n"
            " path=root/name\n"
            " if not path.exists(): records[name]={'status':'missing'}; continue\n"
            " try: row=json.loads(path.read_text())\n"
            " except json.JSONDecodeError: records[name]={'status':'writing'}; continue\n"
            " records[name]={k:row.get(k) for k in ['status','child_returncode']}\n"
            "print(json.dumps(dict(records=records,cpu_seconds=time.process_time()-start)))\n")
        rc, stdout, stderr = scheduler.run_on('node002', base.shlex.join([base.PYTHON, '-c', script]),
                                              timeout=45, check=False)
        record = dict(label='metadata/wrapper-completion', paths=paths, returncode=rc,
                      payload_bytes=len(stdout.encode()), stderr=stderr)
        self.state['attempts'].append(record)
        self.update()
        if rc:
            raise RuntimeError('Completion metadata read failed; no task is resubmitted')
        result = base.json.loads(stdout)
        record['remote_cpu_seconds'] = result['cpu_seconds']
        return result['records']

    def wait(self, ids, label):
        """Dispatch these already submitted IDs once, as requested by the operator."""
        entry = self.state['jobs'][label]
        if 'manual_dispatch' not in entry:
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu, wall = usage.ru_utime + usage.ru_stime, time.monotonic()
            command = [base.sys.executable, str(base.SCHEDULER), 'dispatch']
            for task_id in ids:
                command.extend(['--task-id', task_id])
            try:
                self.capture(label + '/manual-dispatch', command)
            finally:
                usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                entry['manual_dispatch'] = dict(task_ids=list(ids),
                    cpu_seconds=usage.ru_utime + usage.ru_stime - cpu,
                    wall_seconds=time.monotonic() - wall,
                    new_submissions=0, automatic_scientific_retries=0)
                self.update()
        previous, last_print = None, 0.
        while True:
            records = self.wrapper_records(entry['record_paths'])
            complete = wrappers_complete(records)
            rows = self.task_states(ids)
            self.update(waiting_for=label, latest_tasks=rows)
            if any(row['status'] in ('failed', 'cancelled', 'lost') or
                   (row['status'] == 'done' and row['exit_code'] not in (None, 0)) for row in rows):
                raise RuntimeError(f'Scheduler infrastructure failure in {label}; no scientific retry')
            if complete:
                entry['completion_basis'] = 'existing successful wrapper records'
                entry['scheduler_recovery_pending'] = [row['id'] for row in rows if row['status'] != 'done']
                self.update()
                return rows
            if all(row['status'] == 'done' for row in rows):
                raise RuntimeError('Finished scheduler tasks have missing or incomplete wrapper records')
            counts = {status: sum(row['status'] == status for row in rows)
                      for status in sorted({row['status'] for row in rows})}
            if counts != previous or time.monotonic() - last_print >= 60:
                print(base.json.dumps(dict(stage=label, scheduler_status=counts)), flush=True)
                previous, last_print = counts, time.monotonic()
            time.sleep(self.poll_seconds)

    def run_stage(self, stage):
        state = self.state['stages'].setdefault(stage, {})
        if state.get('status') == 'complete':
            return
        index = STAGES.index(stage)
        if index and self.state['stages'].get(STAGES[index - 1], {}).get('status') != 'complete':
            raise RuntimeError('The preceding global stage is not complete')
        descriptor = str(Path(self.plan['output_dir']) / 'cluster' / stage / 'jobs.json')
        self.job(stage + '/describe', 1, base.module(DISPATCH, 'describe',
            '--manifest', self.manifest, '--stage', stage))
        detail = self.metadata([descriptor], 'descriptor')[descriptor]
        state['registered_jobs'] = detail['jobs']
        self.job(stage + '/workers', detail['jobs'], base.module(DISPATCH, 'worker',
            '--descriptor', descriptor, '--start', '{start}', '--end', '{end}', '--workers', '{workers}'))
        self.job(stage + '/finish', 1, base.module(DISPATCH, 'finish', '--descriptor', descriptor))
        completion = str(Path(self.plan['output_dir']) / 'cluster' / stage / 'completion.json')
        result = self.metadata([completion], 'stage')[completion]
        if result['status'] != 'complete' or result['truth_used']:
            raise RuntimeError('Stage completion did not satisfy the frozen pre-truth barrier')
        state.update(status='complete', completion=result)
        self.update()
        print(base.json.dumps(dict(stage=stage, status='complete', jobs=result['jobs'],
            failed_jobs=len(result['failed_jobs']))), flush=True)

    def run(self):
        if self.state['status'] == 'complete':
            print('Stage85 already complete; registered scientific cohort is not repeated', flush=True)
            return
        try:
            self.job('validation/stage85_attempt1', 1,
                base.module('performance.run_mainline_precision_confirmation_science_validation',
                            '--manifest', self.manifest))
            checks_path = str(Path(self.plan['output_dir']) /
                              'cluster_control/validation/stage85_attempt1/checks.json')
            checks = self.metadata([checks_path])[checks_path]
            if checks['status'] != 'passed':
                raise RuntimeError('Changed-wiring substitutes must pass before sampling')
            self.state['validation'] = checks
            self.update()
            for stage in STAGES:
                self.run_stage(stage)
            seal_path = str(Path(self.plan['output_dir']) / 'execution_status.json')
            seal = self.metadata([seal_path], 'seal')[seal_path]
            if (seal['status'] != 'sealed' or seal['registered_terminal_cells'] != 2592 or
                    seal['truth_access_allowed'] is not True or seal['truth_used']):
                raise RuntimeError('All 2592 terminal outcomes or failures must be sealed before truth')
            self.state['terminal_seal'] = seal
            self.update()
            self.job('truth/freeze', 1, base.module(SUMMARY, 'freeze', '--manifest', self.manifest))
            inventory_path = str(Path(self.plan['output_dir']) / 'mechanism/inventory.json')
            inventory = self.metadata([inventory_path], 'seal')[inventory_path]
            if inventory['status'] != 'sealed' or inventory['registered_terminal_cells'] != 2592:
                raise RuntimeError('Post-seal physical policy inventory is incomplete')
            self.job('truth/tasks', 144, base.module(SUMMARY, 'score-batch',
                '--manifest', self.manifest, '--start', '{start}', '--end', '{end}', '--workers', '{workers}'))
            self.job('truth/collect', 1, base.module(SUMMARY, 'collect', '--manifest', self.manifest))
            self.update(status='complete', waiting_for=None, error=None)
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
