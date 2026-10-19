"""Record one Stage73 scheduler command; never repeat its paid subprocess."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import socket
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def cpu():
    own, child = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(own_cpu_seconds=own.ru_utime+own.ru_stime,
        child_cpu_seconds=child.ru_utime+child.ru_stime,
        total_cpu_seconds=own.ru_utime+own.ru_stime+child.ru_utime+child.ru_stime)


def execute(record_path, command):
    node = socket.gethostname().split('.')[0]
    assert node in {f'node{i:03d}' for i in range(1, 7)}, node
    record_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    record = dict(status='started', node=node, command=command,
        started_utc=datetime.now(timezone.utc).isoformat(), subprocess_started=False)
    try:
        with record_path.open('x') as handle:
            json.dump(record, handle, indent=2)
            handle.write('\n')
    except FileExistsError:
        duplicate = dict(status='duplicate_not_executed', node=node,
            utc=datetime.now(timezone.utc).isoformat(), record=str(record_path), **cpu())
        with record_path.with_suffix('.duplicates.jsonl').open('a') as handle:
            handle.write(json.dumps(duplicate)+'\n')
        print('Done: existing command record retained; subprocess not repeated', flush=True)
        return
    log_path = record_path.with_suffix('.log')
    try:
        with log_path.open('x') as handle:
            record['subprocess_started'] = True
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        record.update(status='complete' if result.returncode == 0 else 'child_failed',
            child_returncode=result.returncode)
    except Exception as error:
        record.update(status='launch_failed', child_returncode=None,
            error=f'{type(error).__name__}: {error}')
    record.update(completed_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.monotonic()-start, log_file=str(log_path.relative_to(ROOT)),
        **cpu(), outer_exit_policy='Saved child failure is not retried by scheduler')
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    print('Done: '+json.dumps(dict(status=record['status'], child_returncode=record.get('child_returncode'),
        record=str(record_path.relative_to(ROOT)), total_cpu_seconds=record['total_cpu_seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A recorded subprocess command is required')
    execute(ROOT/args.record, command)


if __name__ == '__main__':
    main()
