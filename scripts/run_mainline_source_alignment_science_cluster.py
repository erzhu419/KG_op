#!/usr/bin/env python3
"""Run explicitly admitted source-alignment targets through frozen global stages."""
import argparse
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('source_alignment_previous_controller',
    Path(__file__).with_name('run_mainline_precision_confirmation_science_cluster.py'))
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
base = previous.base

MANIFEST = 'performance/manifests/mainline_source_alignment_targets_v1.json'
DISPATCH = 'performance.run_mainline_source_alignment_science'
SUMMARY = 'performance.summarize_mainline_source_alignment'
STAGES = ('initial', 'neighbors', 'online', 'native', 'terminal')
QUERY_ARMS = ('legacy_role_after_init_point_only', 'legacy_role_point_only',
    'legacy_role_full', 'source_permuted_role_after_init_point_only',
    'source_permuted_role_point_only')


class Controller(previous.Controller):
    stage_label = 'Source alignment target confirmation'
    signature_stage = 'source_alignment_targets'

    def __init__(self, manifest=MANIFEST, poll_seconds=30):
        self.manifest = manifest
        self.plan = base.read(base.PROJECT / manifest)
        # Source-fit completion and generic execution_started are not admission.
        if (self.plan.get('target_execution_preparation', {}).get('status') != 'validated'
                or self.plan.get('target_execution_authorized') is not True):
            raise RuntimeError('Explicit validated target admission is required before scheduling')
        assert self.plan['name'] == 'mainline_source_alignment_targets_v1'
        assert len(self.plan['tasks']) == 144 and len(self.plan['streams']) == 2
        assert len(self.plan['search_arms']) == 9 and len(self.plan['terminal_arms']) == 11
        assert self.plan['terminal_protocols'] == ['old']
        assert self.plan['budget']['registered_terminal_cells'] == 3168
        assert self.plan['budget']['maximum_verification_calls_per_cell'] == 352
        self.output = base.PROJECT / self.plan['output_dir'] / 'cluster_control'
        self.state_path = self.output / 'source_alignment_science_state.json'
        self.state = base.read(self.state_path) if self.state_path.exists() else dict(
            manifest=manifest, status='not_started', stages={}, jobs={}, attempts=[],
            node_pool=base.NODES, remote_cwd=base.REMOTE, remote_python=base.PYTHON,
            local_scientific_computation=False, automatic_scientific_retries=0,
            target_execution_authorized=True)
        self.poll_seconds = poll_seconds

    def metadata(self, paths, kind='full'):
        if kind != 'alignment_inventory':
            return super().metadata(paths, kind)
        # Project the inventory header remotely; do not download its policy bank.
        base.sys.path.insert(0, str(base.SCHEDULER.parent))
        import scheduler
        source = ("import json,time; from pathlib import Path; start=time.process_time(); root=Path(" +
            repr(base.REMOTE) + "); paths=" + repr(paths) + "; records={}\n"
            "for name in paths:\n"
            " row=json.loads((root/name).read_text())\n"
            " records[name]={k:row.get(k) for k in "
            "['status','registered_terminal_cells','registered_query_states','query_arms']}\n"
            "print(json.dumps(dict(records=records,cpu_seconds=time.process_time()-start)))\n")
        rc, stdout, stderr = scheduler.run_on('node002',
            base.shlex.join([base.PYTHON, '-c', source]), timeout=45, check=False)
        record = dict(label='metadata/alignment-inventory', paths=paths, returncode=rc,
            payload_bytes=len(stdout.encode()), stderr=stderr)
        self.state['attempts'].append(record)
        if rc:
            self.update()
            raise RuntimeError('Inventory metadata read failed; no task is resubmitted')
        result = base.json.loads(stdout)
        record['remote_cpu_seconds'] = result['cpu_seconds']
        self.update()
        return result['records']

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
            print('Source alignment targets already complete; cohort is not repeated', flush=True)
            return
        try:
            for stage in STAGES:
                self.run_stage(stage)
            seal_path = str(Path(self.plan['output_dir']) / 'execution_status.json')
            seal = self.metadata([seal_path], 'seal')[seal_path]
            if (seal['status'] != 'sealed' or seal['registered_terminal_cells'] != 3168 or
                    seal['truth_access_allowed'] is not True or seal['truth_used']):
                raise RuntimeError('All 3168 terminal outcomes or failures must be sealed before truth')
            self.state['terminal_seal'] = seal
            self.update()
            self.job('truth/freeze', 1, base.module(SUMMARY, 'freeze', '--manifest', self.manifest))
            inventory_path = str(Path(self.plan['output_dir']) / 'mechanism/inventory.json')
            inventory = self.metadata([inventory_path], 'alignment_inventory')[inventory_path]
            if (inventory['status'] != 'sealed' or inventory['registered_terminal_cells'] != 3168 or
                    inventory['registered_query_states'] != 4320 or
                    len(inventory['query_arms']) != 5 or set(inventory['query_arms']) != set(QUERY_ARMS)):
                raise RuntimeError('Post-seal inventory must retain all five roles and 4320 registered query states')
            self.state['query_inventory'] = inventory
            self.update()
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
