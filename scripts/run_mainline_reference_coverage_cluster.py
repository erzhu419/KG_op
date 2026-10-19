#!/usr/bin/env python3
"""Run Stage79's substitute checks and protocol registration, without science."""
import argparse
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'safe_unsampled_controller', Path(__file__).with_name('run_mainline_safe_unsampled_cluster.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

MANIFEST = 'performance/manifests/mainline_reference_coverage_preparation_v1.json'


class Controller(base.Controller):
    stage_label = 'Stage79'
    signature_stage = 'stage79'

    def __init__(self, manifest=MANIFEST, poll_seconds=30, validation_attempt='attempt1'):
        self.manifest = manifest
        self.plan = base.read(base.PROJECT / manifest)
        assert self.plan['name'] == 'mainline_reference_coverage_preparation_v1'
        self.output = base.PROJECT / self.plan['output_dir'] / 'cluster_control'
        self.state_path = self.output / 'stage79_state.json'
        self.state = base.read(self.state_path) if self.state_path.exists() else dict(
            manifest=manifest, status='not_started', stages={}, jobs={}, attempts=[],
            node_pool=base.NODES, remote_cwd=base.REMOTE, remote_python=base.PYTHON,
            local_scientific_computation=False, independent_scientific_execution=False)
        self.poll_seconds = poll_seconds
        self.validation_attempt = validation_attempt

    def run(self):
        if self.state['status'] == 'complete':
            print('Stage79 preparation already complete; science has not started', flush=True)
            return
        try:
            validation = str(Path(self.plan['output_dir']) /
                             f'cluster_control/run_stage79_validation_{self.validation_attempt}.py')
            self.job(f'validation/{self.validation_attempt}', 1, [base.PYTHON, '-u', validation])
            self.update(status='complete', waiting_for=None, error=None,
                        independent_scientific_execution=False)
        except Exception as error:
            self.update(status='blocked', error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--validation-attempt', choices=('attempt1', 'attempt2', 'attempt3'), default='attempt1')
    args = parser.parse_args()
    Controller(args.manifest, args.poll_seconds, args.validation_attempt).run()


if __name__ == '__main__':
    main()
