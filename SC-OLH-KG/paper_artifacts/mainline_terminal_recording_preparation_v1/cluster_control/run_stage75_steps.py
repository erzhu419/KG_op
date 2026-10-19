"""Run passive-recording fixtures, then register metadata without experiments."""
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


OUTPUT = Path('paper_artifacts/mainline_terminal_recording_preparation_v1')


def child_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def run(name, command):
    before, wall = child_cpu(), time.monotonic()
    result = subprocess.run(command, check=False)
    record = dict(name=name, command=command, returncode=result.returncode,
                  status='complete' if result.returncode == 0 else 'failed',
                  cpu_seconds=child_cpu()-before, wall_seconds=time.monotonic()-wall)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/f'{name}.json').open('x') as handle:
        json.dump(record, handle, indent=2)
        handle.write('\n')
    print(json.dumps(record), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


run('checks', [sys.executable, '-m', 'pytest', '-p', 'no:capture', '-q',
    'tests/test_mainline_terminal_recording.py',
    'tests/test_mainline_terminal_recording_seed_replication.py'])
run('registration_process', [sys.executable, '-u', '-m',
    'performance.register_mainline_terminal_recording_seed_replication'])
