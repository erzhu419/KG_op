#!/usr/bin/env python3
"""Run Stage94's frozen source-alignment saved-path audit via scheduleurm."""
import argparse
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'precision_paths_controller', Path(__file__).with_name('run_mainline_precision_propagation_paths_cluster.py'))
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
base = previous.base
MANIFEST = 'performance/manifests/mainline_source_alignment_paths_v1.json'
AUDIT = 'performance.analyze_mainline_source_alignment_paths'


class Controller(previous.Controller):
    stage_label = 'Stage94'
    signature_stage = 'stage94'

    def __init__(self, manifest=MANIFEST, poll_seconds=45):
        self.manifest = manifest
        self.plan = base.read(base.PROJECT / manifest)
        assert self.plan['name'] == 'mainline_source_alignment_paths_v1'
        assert len(self.plan['tasks']) == 144 and len(self.plan['streams']) == 2
        assert len(self.plan['terminal_arms']) == 11 and self.plan['terminal_protocols'] == ['old']
        self.output = base.PROJECT / self.plan['output_dir'] / 'cluster_control'
        self.state_path = self.output / 'stage94_state.json'
        self.state = base.read(self.state_path) if self.state_path.exists() else dict(
            manifest=manifest, status='not_started', stages={}, jobs={}, attempts=[],
            node_pool=base.NODES, remote_cwd=base.REMOTE, remote_python=base.PYTHON,
            local_scientific_computation=False, automatic_scientific_retries=0,
            scientific_execution_authorized=False, saved_record_analysis_only=True)
        self.poll_seconds = poll_seconds

    def run(self):
        if self.state['status'] == 'complete':
            print('Stage94 saved-path audit already complete; no resubmission', flush=True)
            return
        try:
            self.job('validation/stage94_attempt1', 1,
                base.module('performance.run_mainline_source_alignment_paths_validation',
                            '--manifest', self.manifest))
            path = str(Path(self.plan['output_dir']) / 'cluster_control/validation/stage94_attempt1/checks.json')
            checks = self.metadata([path])[path]
            if checks['status'] != 'passed':
                raise RuntimeError('Saved-array substitute checks must pass before the audit')
            self.state['validation'] = checks
            self.update()
            self.job('audit/tasks', 144, base.module(AUDIT, 'batch', '--manifest', self.manifest,
                '--start', '{start}', '--end', '{end}', '--workers', '{workers}'))
            self.job('audit/collect', 1, base.module(AUDIT, 'collect', '--manifest', self.manifest))
            self.update(status='complete', waiting_for=None, error=None)
        except Exception as error:
            self.update(status='blocked', error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--poll-seconds', type=int, default=45)
    args = parser.parse_args()
    Controller(args.manifest, args.poll_seconds).run()


if __name__ == '__main__':
    main()
