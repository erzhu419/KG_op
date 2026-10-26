#!/usr/bin/env python3
"""Compare sealed backends after substitutes in one scheduled CPU item."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import resource
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1] / 'SC-OLH-KG'
OUTPUT = 'paper_artifacts/mainline_frozen_backend_comparison_v1'
AUDIT = 'performance.analyze_mainline_frozen_backend_comparison'
TESTS = ('tests/test_mainline_frozen_backend_comparison.py',)


def run_worker():
    """The outer existing wrapper retains failures without scheduler retries."""
    assert socket.gethostname().split('.')[0] in {f'node{i:03d}' for i in range(1, 7)}
    relative = Path(OUTPUT) / 'cluster_control/validation/stage103_attempt1'
    sequence = ROOT / relative.parent / 'stage103_attempt1_sequence.json'
    sequence.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='started', started_utc=datetime.now(timezone.utc).isoformat(),
                  phases=[], new_simulator_calls=0, new_scores=0, new_fits=0,
                  new_posterior_restorations=0, automatic_retries=0)
    with sequence.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    validation = [sys.executable, '-m', 'pytest', '-p', 'no:capture', '-q', *TESTS]
    audit = [sys.executable, '-u', '-m', AUDIT,
             '--checks', str(relative / 'checks.json')]
    (ROOT / relative).mkdir(exist_ok=False)
    for label, command in [('substitute_checks', validation), ('saved_record_audit', audit)]:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu, wall = usage.ru_utime + usage.ru_stime, time.monotonic()
        error = None
        stdout, stderr = '', ''
        try:
            result = subprocess.run(command, cwd=ROOT, text=True,
                                    capture_output=label == 'substitute_checks')
            returncode = result.returncode
            if label == 'substitute_checks':
                stdout, stderr = result.stdout, result.stderr
        except OSError as exc:
            returncode, error = 127, repr(exc)
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        report['phases'].append(dict(phase=label, command=command, returncode=returncode,
            launch_error=error, child_cpu_seconds=usage.ru_utime + usage.ru_stime - cpu,
            wall_seconds=time.monotonic() - wall))
        if label == 'substitute_checks':
            checks = dict(status='passed' if returncode == 0 else 'failed',
                returncode=returncode, test_files=list(TESTS), command=command,
                test_counts={name: int(count) for count, name in re.findall(
                    r'(\d+) (passed|failed|skipped|error|errors)\b', stdout)},
                cpu_seconds=report['phases'][-1]['child_cpu_seconds'],
                wall_seconds=report['phases'][-1]['wall_seconds'],
                completed_utc=datetime.now(timezone.utc).isoformat(),
                automatic_retries=0, new_simulator_calls=0, new_scores=0,
                new_fits=0, new_posterior_restorations=0)
            (ROOT / relative / 'pytest.log').write_text(stdout + stderr)
            (ROOT / relative / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
            print(stdout + stderr, end='', flush=True)
        report['status'] = 'failed' if returncode else (
            'complete' if label == 'saved_record_audit' else 'checks_passed')
        report['updated_utc'] = datetime.now(timezone.utc).isoformat()
        sequence.write_text(json.dumps(report, indent=2) + '\n')
        if returncode:
            return returncode
    return 0


def run_controller(poll_seconds):
    spec = importlib.util.spec_from_file_location('saved_audit_previous_controller',
        Path(__file__).with_name('run_mainline_precision_propagation_paths_cluster.py'))
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    base = previous.base

    class Controller(previous.Controller):
        stage_label = 'Stage103 frozen AFTER versus saved backend comparison'
        signature_stage = 'stage103'

        def __init__(self):
            self.plan = dict(name='mainline_frozen_backend_comparison_v1', output_dir=OUTPUT)
            self.output = base.PROJECT / self.plan['output_dir'] / 'cluster_control'
            self.state_path = self.output / 'stage103_state.json'
            self.state = base.read(self.state_path) if self.state_path.exists() else dict(
                analysis_module=AUDIT, status='not_started', stages={}, jobs={}, attempts=[],
                node_pool=base.NODES, remote_cwd=base.REMOTE, remote_python=base.PYTHON,
                local_scientific_computation=False, automatic_scientific_retries=0,
                saved_record_analysis_only=True, scheduled_items=1, workers=1, vram=0)
            self.poll_seconds = poll_seconds

        def run(self):
            if self.state['status'] == 'complete':
                print('Stage103 audit already complete; no resubmission', flush=True)
                return
            try:
                command = [base.PYTHON, '-u',
                    str(Path(base.REMOTE).parent / 'scripts' / Path(__file__).name),
                    '--worker']
                self.job('audit/checks_and_saved_records', 1, command)
                self.update(status='complete', waiting_for=None, error=None)
            except Exception as error:
                self.update(status='blocked', error=str(error))
                raise

    Controller().run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=45)
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(run_worker())
    run_controller(args.poll_seconds)


if __name__ == '__main__':
    main()
