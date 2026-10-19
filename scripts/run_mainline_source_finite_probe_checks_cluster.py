#!/usr/bin/env python3
"""Run only the finite source-probe substitutes as one scheduled CPU item."""
import argparse
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('finite_probe_checks_previous',
    Path(__file__).with_name('run_mainline_precision_propagation_paths_cluster.py'))
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
base = previous.base
MANIFEST = 'performance/manifests/mainline_source_finite_numerical_validation_v1.json'
TESTS = ('tests/test_mainline_source_finite_numerical_probe.py',
         'tests/test_mainline_source_finite_probe_checks_cluster.py')


class Controller(previous.Controller):
    # Same submit-cpu-batch / single dispatch / saved wrapper gate as Stage97.
    stage_label = 'Stage99 finite source-probe substitutes only'
    signature_stage = 'stage99'

    def __init__(self, manifest=MANIFEST, poll_seconds=45):
        self.manifest = manifest
        self.plan = base.read(base.PROJECT / manifest)
        assert self.plan['name'] == 'mainline_source_finite_numerical_validation_v1'
        self.output = base.PROJECT / self.plan['output_dir'] / 'cluster_control'
        self.state_path = self.output / 'stage99_state.json'
        self.state = base.read(self.state_path) if self.state_path.exists() else dict(
            manifest=manifest, status='not_started', stages={}, jobs={}, attempts=[],
            node_pool=base.NODES, remote_cwd=base.REMOTE, remote_python=base.PYTHON,
            local_scientific_computation=False, automatic_scientific_retries=0,
            substitutes_only=True, scheduled_items=1, workers=1, vram=0)
        self.poll_seconds = poll_seconds

    def run(self):
        if self.state['status'] == 'complete':
            print('Stage99 substitutes already complete; no resubmission', flush=True)
            return
        try:
            attempt = str(Path(self.plan['output_dir']) /
                          'cluster_control/validation/stage99_attempt1')
            command = base.module('performance.run_mainline_source_alignment_validation',
                                  '--attempt', attempt)
            for test in TESTS:
                command.extend(['--test', test])
            self.job('validation/stage99_attempt1', 1, command)
            path = attempt + '/checks.json'
            checks = self.metadata([path])[path]
            if checks['status'] != 'passed' or checks['returncode'] != 0:
                raise RuntimeError('Finite source-probe substitutes failed; no automatic retry')
            self.update(status='complete', checks=checks, waiting_for=None, error=None)
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
