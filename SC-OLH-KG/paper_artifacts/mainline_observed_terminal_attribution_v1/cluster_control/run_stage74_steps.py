"""Run Stage74's targeted checks, saved-record batch and collection once."""
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


OUTPUT = Path('paper_artifacts/mainline_observed_terminal_attribution_v1')
MANIFEST = 'performance/manifests/mainline_observed_terminal_attribution_v1.json'
MODULE = 'performance.analyze_mainline_observed_terminal_attribution'
TESTS = ['tests/test_mainline_observed_shortlist_attribution.py',
         'tests/test_mainline_saved_terminal_comparison.py']


def child_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def run(label, command):
    cpu, wall = child_cpu(), time.monotonic()
    result = subprocess.run(command, check=False)
    record = dict(label=label, command=command, returncode=result.returncode,
                  status='complete' if result.returncode == 0 else 'failed',
                  cpu_seconds=child_cpu()-cpu, wall_seconds=time.monotonic()-wall)
    path = OUTPUT/('validation/pytest_attempt1.json' if label == 'tests'
                   else f'cluster_control/{label}_process.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(record, handle, indent=2)
        handle.write('\n')
    print(json.dumps(record), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


run('tests', [sys.executable, '-m', 'pytest', '-p', 'no:capture', '-q', *TESTS])
run('analysis', [sys.executable, '-u', '-m', MODULE, '--manifest', MANIFEST,
                 '--start', '0', '--end', '24', '--workers', '1'])
run('collect', [sys.executable, '-u', '-m', MODULE, '--manifest', MANIFEST, '--collect'])
