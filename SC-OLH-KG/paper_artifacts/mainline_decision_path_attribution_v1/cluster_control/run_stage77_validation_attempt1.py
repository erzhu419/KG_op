"""Check the saved-file attribution wiring, then register its zero-call audit."""
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


ROOT = Path.cwd()
OUT = ROOT / 'paper_artifacts/mainline_decision_path_attribution_v1'
MANIFEST = 'performance/manifests/mainline_decision_path_attribution_v1.json'
CHECKS = OUT / 'cluster_control/validation/attempt1/checks.json'
TESTS = ['tests/test_mainline_decision_path_attribution.py',
         'tests/test_mainline_decision_path_summary.py']


def child_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def main():
    CHECKS.parent.mkdir(parents=True, exist_ok=True)
    assert not CHECKS.exists(), 'Preserve the existing validation attempt'
    command = [sys.executable, '-m', 'pytest', '-p', 'no:capture', '-q', *TESTS]
    cpu, wall = child_cpu(), time.monotonic()
    result = subprocess.run(command, cwd=ROOT)
    report = dict(status='complete' if result.returncode == 0 else 'failed',
        command=command, test_files=TESTS, expected_instances=9, returncode=result.returncode,
        cpu_seconds=child_cpu() - cpu, wall_seconds=time.monotonic() - wall,
        checked_utc=datetime.now(timezone.utc).isoformat(),
        new_scientific_calls=0, new_truth_calls=0, new_fits=0, new_posterior_restorations=0)
    with CHECKS.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(report), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    subprocess.run([sys.executable, '-u', '-m', 'performance.run_mainline_decision_path_audit',
        'register', '--manifest', MANIFEST], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
