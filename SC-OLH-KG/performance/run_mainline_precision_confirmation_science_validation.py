"""Record changed-wiring substitute tests before the registered science starts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

from performance.run_mainline_legacy_backend_bridge import ROOT, read


TESTS = [
    'tests/test_mainline_precision_confirmation_science.py',
    'tests/test_mainline_precision_confirmation_summary.py',
    'tests/test_mainline_precision_confirmation_science_cluster.py',
    'tests/test_mainline_safe_unsampled_competition.py',
]
SCIENCE = 'performance.run_mainline_precision_confirmation_science'



def child_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    plan = read(ROOT / args.manifest)
    output = ROOT / plan['output_dir'] / 'cluster_control/validation/stage85_attempt1'
    checks = output / 'checks.json'
    assert not checks.exists(), 'Preserve the existing validation attempt'
    output.mkdir(parents=True, exist_ok=True)
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
        new_scientific_calls=0, new_truth_calls=0, new_fits=0, new_posterior_restorations=0)
    with checks.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    (output / 'pytest.log').write_text(result.stdout + result.stderr)
    print(json.dumps(report), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    admission = subprocess.run([sys.executable, '-u', '-m', SCIENCE, 'admit',
        '--manifest', args.manifest, '--checks', str(checks.relative_to(ROOT))], cwd=ROOT)
    raise SystemExit(admission.returncode)


if __name__ == '__main__':
    main()
