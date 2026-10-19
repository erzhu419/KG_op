"""Validate four-cell execution substitutes, then register the independent cohort."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import resource
import subprocess
import sys
import time


ROOT = Path.cwd()
OUT = ROOT / 'paper_artifacts/mainline_reference_coverage_confirmation_v1'
MANIFEST = 'performance/manifests/mainline_reference_coverage_confirmation_v1.json'
CHECKS = OUT / 'cluster_control/validation/attempt1/checks.json'
TESTS = ['tests/test_mainline_reference_coverage_confirmation_registration.py',
         'tests/test_mainline_reference_coverage_cohort.py',
         'tests/test_mainline_reference_coverage_execution.py']


def child_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def main():
    CHECKS.parent.mkdir(parents=True, exist_ok=True)
    assert not CHECKS.exists(), 'Preserve the existing validation attempt'
    command = [sys.executable, '-m', 'pytest', '-p', 'no:capture', '-q', *TESTS]
    cpu, wall = child_cpu(), time.monotonic()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    print(result.stdout, end='', flush=True)
    print(result.stderr, end='', file=sys.stderr, flush=True)
    counts = {name: int(count) for count, name in re.findall(
        r'(\d+) (passed|failed|skipped|error|errors)\b', result.stdout)}
    report = dict(status='passed' if result.returncode == 0 else 'failed',
        command=command, test_files=TESTS, test_counts=counts, returncode=result.returncode,
        cpu_seconds=child_cpu() - cpu, wall_seconds=time.monotonic() - wall,
        checked_utc=datetime.now(timezone.utc).isoformat(),
        new_scientific_calls=0, new_truth_calls=0, new_fits=0,
        new_posterior_restorations=0, independent_scientific_execution=False)
    with CHECKS.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    (CHECKS.parent / 'pytest.log').write_text(result.stdout + result.stderr)
    print(json.dumps(report), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    subprocess.run([sys.executable, '-u', '-m',
        'performance.register_mainline_reference_coverage_confirmation',
        '--manifest', MANIFEST, '--checks', str(CHECKS.relative_to(ROOT))],
        cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
