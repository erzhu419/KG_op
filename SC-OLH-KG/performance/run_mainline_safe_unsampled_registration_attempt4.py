"""Run only the Stage72 fixtures, retaining failures without automatic retries."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'paper_artifacts/mainline_safe_unsampled_validation_registration_v1'


def main():
    node = socket.gethostname().split('.')[0]
    assert node in {f'node{i:03d}' for i in range(1, 7)}, node
    checks_file = OUTPUT/'checks_attempt4.json'
    assert not checks_file.exists(), 'Preserve this check attempt'
    started = time.monotonic()
    checks = dict(status='running', attempt=4, node=node, steps=[],
        started_utc=datetime.now(timezone.utc).isoformat(),
        scope='Fixtures and metadata registration only; no scientific cohort',
        pytest_plugin_autoload=False, faulthandler=True)
    env = dict(os.environ, PYTHONFAULTHANDLER='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')

    def run(name, command):
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        step_started = time.monotonic()
        log_file = OUTPUT/f'{name}_attempt4.log'
        with log_file.open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_user, cpu_system = after.ru_utime-before.ru_utime, after.ru_stime-before.ru_stime
        checks['steps'].append(dict(name=name, command=command, returncode=result.returncode,
            wall_seconds=time.monotonic()-step_started, user_seconds=cpu_user,
            system_seconds=cpu_system, cpu_seconds=cpu_user+cpu_system,
            log=str(log_file.relative_to(ROOT))))
        print(json.dumps(checks['steps'][-1]), flush=True)
        return result.returncode

    code = run('pytest', [sys.executable, '-X', 'faulthandler', '-m', 'pytest', '-q',
        'tests/test_mainline_safe_unsampled_registration.py',
        'tests/test_mainline_failed_query_audit.py'])
    if code:
        imports = "import importlib\nfor name in ['numpy','scipy','performance.register_mainline_safe_unsampled_validation','performance.run_mainline_role_fresh']:\n print('IMPORT '+name, flush=True)\n importlib.import_module(name)\n print('OK '+name, flush=True)\n"
        run('import_diagnostic', [sys.executable, '-X', 'faulthandler', '-c', imports])
    else:
        code = run('registration', [sys.executable, '-X', 'faulthandler', '-m',
            'performance.register_mainline_safe_unsampled_validation'])
    own, child = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    checks.update(status='complete' if code == 0 else 'failed', child_returncode=code,
        completed_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic()-started,
        wrapper_cpu_seconds=own.ru_utime+own.ru_stime,
        child_cpu_seconds=child.ru_utime+child.ru_stime,
        total_job_cpu_seconds=own.ru_utime+own.ru_stime+child.ru_utime+child.ru_stime,
        cpu_scope='Wrapper and child CPU at final bookkeeping snapshot; scheduler overhead excluded',
        outer_exit_policy='Return zero after saving any child failure; no scheduler automatic retry')
    with checks_file.open('x') as target:
        json.dump(checks, target, indent=2)
        target.write('\n')
    print('Done: retained Stage72 check status='+checks['status'], flush=True)


if __name__ == '__main__':
    main()
