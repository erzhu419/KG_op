#!/usr/bin/env python3
"""Run Stage82's saved propagation-path audit through the existing scheduler."""
import argparse
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'safe_unsampled_controller', Path(__file__).with_name('run_mainline_safe_unsampled_cluster.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

MANIFEST = 'performance/manifests/mainline_frozen_propagation_paths_v1.json'
AUDIT = 'performance.analyze_mainline_frozen_propagation_paths'


class Controller(base.Controller):
    stage_label = 'Stage82'
    signature_stage = 'stage82'

    def __init__(self, manifest=MANIFEST, poll_seconds=30):
        self.manifest = manifest
        self.plan = base.read(base.PROJECT / manifest)
        assert self.plan['name'] == 'mainline_frozen_propagation_paths_v1'
        assert len(self.plan['tasks']) == self.plan['expected']['geometries'] == 24
        self.output = base.PROJECT / self.plan['output_dir'] / 'cluster_control'
        self.state_path = self.output / 'stage82_state.json'
        self.state = base.read(self.state_path) if self.state_path.exists() else dict(
            manifest=manifest, status='not_started', stages={}, jobs={}, attempts=[],
            node_pool=base.NODES, remote_cwd=base.REMOTE, remote_python=base.PYTHON,
            local_scientific_computation=False, automatic_scientific_retries=0)
        self.poll_seconds = poll_seconds

    def run(self):
        if self.state['status'] == 'complete':
            print('Stage82 audit already complete; saved analysis is not repeated', flush=True)
            return
        try:
            self.job('validation/stage82_attempt1', 1,
                base.module('performance.run_mainline_frozen_propagation_paths_validation',
                            '--manifest', self.manifest))
            checks_path = str(Path(self.plan['output_dir']) /
                              'cluster_control/validation/stage82_attempt1/checks.json')
            checks = self.metadata([checks_path])[checks_path]
            if checks['status'] != 'passed':
                raise RuntimeError('Saved-array substitute checks must pass before the audit')
            self.state['validation'] = checks
            self.update()
            self.job('audit/tasks', 24, base.module(AUDIT, 'batch', '--manifest', self.manifest,
                '--start', '{start}', '--end', '{end}', '--workers', '{workers}'))
            self.job('audit/collect', 1, base.module(AUDIT, 'collect', '--manifest', self.manifest))
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
