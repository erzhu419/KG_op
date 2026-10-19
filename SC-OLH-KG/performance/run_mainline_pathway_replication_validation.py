"""Check Stage83 saved-array wiring, then register the frozen zero-call audit."""
import argparse
from datetime import datetime, timezone
import json
import re
import subprocess
import sys
import time

from performance.run_mainline_legacy_backend_bridge import ROOT, read
from performance.run_mainline_reference_coverage_science_validation import child_cpu


TESTS = [
    'tests/test_mainline_pathway_replication.py',
    'tests/test_mainline_pathway_replication_summary.py',
]
AUDIT = 'performance.run_mainline_pathway_replication'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    plan = read(ROOT / args.manifest)
    output = ROOT / plan['output_dir'] / 'cluster_control/validation/stage83_attempt2'
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
    registration = subprocess.run([sys.executable, '-u', '-m', AUDIT, 'register',
        '--manifest', args.manifest, '--checks', str(checks.relative_to(ROOT))], cwd=ROOT)
    raise SystemExit(registration.returncode)


if __name__ == '__main__':
    main()
