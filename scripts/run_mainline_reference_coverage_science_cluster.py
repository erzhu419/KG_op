#!/usr/bin/env python3
"""Execute the admitted four-terminal cohort through scheduled global stages."""
import argparse
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'safe_unsampled_controller', Path(__file__).with_name('run_mainline_safe_unsampled_cluster.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

MANIFEST = 'performance/manifests/mainline_reference_coverage_confirmation_v1.json'
DISPATCH = 'performance.mainline_reference_coverage_dispatch'
SUMMARY = 'performance.summarize_mainline_reference_coverage_confirmation'
STAGES = ('initial', 'neighbors', 'online', 'native', 'terminal')


class Controller(base.Controller):
    stage_label = 'Stage81'
    signature_stage = 'stage81'

    def __init__(self, manifest=MANIFEST, poll_seconds=30):
        self.manifest = manifest
        self.plan = base.read(base.PROJECT / manifest)
        assert self.plan['name'] == 'mainline_reference_coverage_confirmation_v1'
        assert len(self.plan['tasks']) == 24 and len(self.plan['terminal_arms']) == 9
        assert self.plan['budget']['registered_terminal_cells'] == 864
        self.output = base.PROJECT / self.plan['output_dir'] / 'cluster_control'
        self.state_path = self.output / 'stage81_state.json'
        self.state = base.read(self.state_path) if self.state_path.exists() else dict(
            manifest=manifest, status='not_started', stages={}, jobs={}, attempts=[],
            node_pool=base.NODES, remote_cwd=base.REMOTE, remote_python=base.PYTHON,
            local_scientific_computation=False, automatic_scientific_retries=0)
        self.poll_seconds = poll_seconds

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
            print('Stage81 already complete; registered scientific cohort is not repeated', flush=True)
            return
        try:
            self.job('validation/stage81_attempt1', 1,
                base.module('performance.run_mainline_reference_coverage_science_validation',
                            '--manifest', self.manifest))
            checks_path = str(Path(self.plan['output_dir']) /
                              'cluster_control/validation/stage81_attempt1/checks.json')
            checks = self.metadata([checks_path])[checks_path]
            if checks['status'] != 'passed':
                raise RuntimeError('Changed-wiring substitutes must pass before sampling')
            self.state['validation'] = checks
            self.update()
            for stage in STAGES:
                self.run_stage(stage)
            seal_path = str(Path(self.plan['output_dir']) / 'execution_status.json')
            seal = self.metadata([seal_path], 'seal')[seal_path]
            if (seal['status'] != 'sealed' or seal['registered_terminal_cells'] != 864 or
                    seal['truth_access_allowed'] is not True or seal['truth_used']):
                raise RuntimeError('All 864 terminal outcomes or failures must be sealed before truth')
            self.state['terminal_seal'] = seal
            self.update()
            self.job('truth/freeze', 1, base.module(SUMMARY, 'freeze-truth', '--manifest', self.manifest))
            inventory_path = str(Path(self.plan['output_dir']) / 'truth_inventory.json')
            inventory = self.metadata([inventory_path], 'seal')[inventory_path]
            if inventory['status'] != 'frozen' or inventory['registered_terminal_cells'] != 864:
                raise RuntimeError('Post-seal physical policy inventory is incomplete')
            self.job('truth/tasks', 24, base.module(SUMMARY, 'score-batch',
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
