"""Install cached test wheels, then run Stage72 registration attempt 3."""
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'paper_artifacts/mainline_safe_unsampled_validation_registration_v1'


def usage(kind):
    value = resource.getrusage(kind)
    return dict(user_seconds=value.ru_utime, system_seconds=value.ru_stime)


def main():
    node = socket.gethostname().split('.')[0]
    assert node in {f'node{i:03d}' for i in range(1, 7)}, node
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checks_file = OUTPUT / 'checks_attempt3.json'
    assert not checks_file.exists(), 'Preserve the first registration check and any failure'
    started = time.monotonic()
    checks = dict(status='running', node=node, attempt=3,
        prior_failed_checks=['paper_artifacts/mainline_safe_unsampled_validation_registration_v1/checks.json',
            'paper_artifacts/mainline_safe_unsampled_validation_registration_v1/checks_attempt2.json'],
        started_utc=datetime.now(timezone.utc).isoformat(), steps=[],
        scope='Deterministic fixtures and registration metadata only; no scientific cohort execution')
    commands = [
        ('install_pytest', [sys.executable, '-m', 'pip', '--disable-pip-version-check',
            'install', '--no-index', '--find-links', str(OUTPUT/'cluster_control/pytest_wheels'),
            'pytest==8.3.5']),
        ('import_pytest', [sys.executable, '-c',
            'import sys, pytest; print(sys.executable); print(pytest.__version__)']),
        ('pytest', [sys.executable, '-m', 'pytest', '-q',
            'tests/test_mainline_safe_unsampled_registration.py',
            'tests/test_mainline_failed_query_audit.py']),
        ('registration', [sys.executable, '-m',
            'performance.register_mainline_safe_unsampled_validation']),
    ]
    code = 0
    for name, command in commands:
        before = usage(resource.RUSAGE_CHILDREN)
        step_started = time.monotonic()
        log_file = OUTPUT / f'{name}_attempt3.log'
        with log_file.open('x') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        after = usage(resource.RUSAGE_CHILDREN)
        cpu = {key: after[key] - before[key] for key in before}
        checks['steps'].append(dict(name=name, command=command, returncode=result.returncode,
            wall_seconds=time.monotonic() - step_started,
            cpu_seconds=sum(cpu.values()), **cpu,
            log=str(log_file.relative_to(ROOT))))
        print(json.dumps(checks['steps'][-1]), flush=True)
        if result.returncode:
            code = result.returncode
            break
    own, children = usage(resource.RUSAGE_SELF), usage(resource.RUSAGE_CHILDREN)
    checks.update(status='complete' if code == 0 else 'failed', returncode=code,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.monotonic() - started, wrapper_cpu=own,
        child_cpu=children, total_job_cpu_seconds=sum(own.values()) + sum(children.values()),
        cpu_scope='Wrapper plus completed child process CPU at final bookkeeping snapshot; scheduler overhead excluded')
    with checks_file.open('x') as target:
        json.dump(checks, target, indent=2)
        target.write('\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
