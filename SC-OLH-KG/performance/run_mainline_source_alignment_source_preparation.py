"""Run Stage90 substitutes, then write source-fit metadata only on success."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TESTS = ('tests/test_mainline_source_alignment_archive.py',
         'tests/test_mainline_source_alignment_sources.py')
MANIFEST = 'performance/manifests/mainline_source_alignment_control_v1.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    attempt = Path(args.attempt)
    phase_path = ROOT / attempt.parent / (attempt.name + '_sequence.json')
    validation = [sys.executable, '-u', '-m',
        'performance.run_mainline_source_alignment_validation', '--attempt', str(attempt)]
    for test in TESTS:
        validation.extend(['--test', test])
    preparation = [sys.executable, '-u', '-m',
        'performance.prepare_mainline_source_alignment_sources',
        '--action', 'prepare', '--manifest', MANIFEST]
    report = dict(status='started', stages=[], new_scientific_calls=0, new_fits=0,
                  new_posterior_restorations=0, scientific_execution_started=False)
    phase_path.parent.mkdir(parents=True, exist_ok=True)
    with phase_path.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    for label, command in [('substitute_checks', validation),
                           ('metadata_preparation', preparation)]:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu, wall = usage.ru_utime + usage.ru_stime, time.monotonic()
        error = None
        try:
            returncode = subprocess.run(command, cwd=ROOT).returncode
        except OSError as exc:
            returncode, error = 127, repr(exc)
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        report['stages'].append(dict(stage=label, command=command,
            returncode=returncode, launch_error=error,
            cpu_seconds=usage.ru_utime + usage.ru_stime - cpu,
            wall_seconds=time.monotonic() - wall))
        report['status'] = 'failed' if returncode else (
            'complete' if label == 'metadata_preparation' else 'checks_passed')
        phase_path.write_text(json.dumps(report, indent=2) + '\n')
        if returncode:
            raise SystemExit(returncode)
    print('Stage90 substitute checks and metadata preparation complete', flush=True)


if __name__ == '__main__':
    main()
