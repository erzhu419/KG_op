#!/usr/bin/env python3
"""Schedule the twelve registered finite source probes once, then seal them."""
import argparse
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('finite_probes_previous_controller',
    Path(__file__).with_name('run_mainline_precision_propagation_paths_cluster.py'))
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
base = previous.base
MANIFEST = 'performance/manifests/mainline_source_finite_numerical_validation_v1.json'
RUNNER = 'performance.run_mainline_source_finite_probes'
TESTS = ('tests/test_mainline_source_finite_probes.py',
         'tests/test_mainline_source_finite_probes_cluster.py')


class Controller(previous.Controller):
    stage_label = 'Stage100 registered finite source probes'
    signature_stage = 'stage100'

    def __init__(self, manifest=MANIFEST, poll_seconds=45):
        self.manifest = manifest
        self.plan = base.read(base.PROJECT / manifest)
        assert self.plan['name'] == 'mainline_source_finite_numerical_validation_v1'
        assert len(self.plan['cases']) == 12
        self.output = base.PROJECT / self.plan['output_dir'] / 'cluster_control'
        self.state_path = self.output / 'stage100_state.json'
        self.state = base.read(self.state_path) if self.state_path.exists() else dict(
            manifest=manifest, status='not_started', stages={}, jobs={}, attempts=[],
            node_pool=base.NODES, remote_cwd=base.REMOTE, remote_python=base.PYTHON,
            local_scientific_computation=False, automatic_scientific_retries=0,
            source_only=True, registered_cases=12, vram=0)
        self.poll_seconds = poll_seconds

    def run(self):
        if self.state['status'] == 'complete':
            print('Stage100 registered probes already sealed; no resubmission', flush=True)
            return
        try:
            attempt = str(Path(self.plan['output_dir']) /
                          'cluster_control/validation/stage100_attempt1')
            command = base.module('performance.run_mainline_source_alignment_validation',
                                  '--attempt', attempt)
            for test in TESTS:
                command.extend(['--test', test])
            self.job('validation/stage100_attempt1', 1, command)
            path = attempt + '/checks.json'
            checks = self.metadata([path])[path]
            if checks['status'] != 'passed' or checks['returncode'] != 0:
                raise RuntimeError('New dispatch/collector substitutes failed; source probes are not submitted')
            self.update(checks=checks)
            if self.plan.get('execution_authorized') is not True:
                raise RuntimeError('Explicit source-probe execution authorization is required')
            self.job('probes/batch', 12, base.module(RUNNER, '--action', 'batch',
                '--manifest', self.manifest, '--start', '{start}', '--end', '{end}',
                '--workers', '{workers}'))
            # Collector refuses uncovered cases, and retains sealed missing/failed outcomes.
            self.job('probes/collect', 1, base.module(RUNNER, '--action', 'collect',
                '--manifest', self.manifest))
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
