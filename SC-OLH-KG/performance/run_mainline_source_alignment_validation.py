"""Run explicit source-alignment substitute checks; never launch science."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def child_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True,
                        help='New project-relative attempt directory; existing attempts are preserved')
    parser.add_argument('--test', action='append', required=True,
                        help='Explicit pytest test file or node ID; may be repeated')
    args = parser.parse_args()
    output = ROOT / args.attempt
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-m', 'pytest', '-p', 'no:capture', '-q', *args.test]
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    started = datetime.now(timezone.utc).isoformat()
    (output / 'started.json').write_text(json.dumps(dict(
        started_utc=started, command=command, tests=args.test,
        scientific_execution_authorized=False), indent=2) + '\n')
    cpu, self_cpu, wall = child_cpu(), time.process_time(), time.monotonic()
    error = None
    try:
        completed = subprocess.run(command, cwd=ROOT, env=environment,
                                   text=True, capture_output=True)
        stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
    except OSError as exc:
        stdout, stderr, returncode = '', str(exc), 127
        error = repr(exc)
    counts = {name: int(count) for count, name in re.findall(
        r'(\d+) (passed|failed|skipped|error|errors)\b', stdout)}
    report = dict(status='passed' if returncode == 0 else 'failed',
        command=command, test_files=args.test, test_counts=counts, returncode=returncode,
        launch_error=error, cpu_seconds=child_cpu() - cpu,
        coordinator_cpu_seconds=time.process_time() - self_cpu,
        wall_seconds=time.monotonic() - wall, started_utc=started,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Explicit substitute checks only; no scientific runner or registration invoked',
        new_scientific_calls=0, new_truth_calls=0, new_fits=0,
        new_posterior_restorations=0, registered_permutations_materialized_by_checks=False,
        registered_geometries_materialized_by_checks=False, automatic_retries=0)
    (output / 'pytest.log').write_text(stdout + stderr)
    (output / 'checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(stdout, end='', flush=True)
    print(stderr, end='', file=sys.stderr, flush=True)
    print(json.dumps(report), flush=True)
    raise SystemExit(returncode)


if __name__ == '__main__':
    main()
