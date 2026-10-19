"""Scheduler shard for the 24 registered post-seal audit tasks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import socket
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    for name in ('start', 'end', 'workers'):
        parser.add_argument('--'+name, type=int, required=True)
    args = parser.parse_args()
    node = socket.gethostname().split('.')[0]
    assert node in {f'node{i:03d}' for i in range(1, 7)}, node
    plan = json.loads((ROOT/args.manifest).read_text())
    tasks = plan['tasks'][args.start:args.end]
    assert 0 <= args.start < args.end <= len(plan['tasks']) == 24 and args.workers > 0
    output = ROOT/plan['output_dir']/'cluster/audit'/f'shard_{args.start}_{args.end}'
    output.mkdir(parents=True, exist_ok=True)
    assert not (output/'result.json').exists(), 'Preserve the completed audit shard'

    def invoke(task):
        seed = task['geometry_task_seed']
        command = [sys.executable, '-u', '-m', 'performance.run_mainline_safe_unsampled_audit',
            'task', '--manifest', args.manifest, '--seed', str(seed)]
        log_file = output/f'seed_{seed}.log'
        with log_file.open('x') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        failure = ROOT/plan['output_dir']/'mechanism'/f'seed_{seed}'/'failure.json'
        if result.returncode and not failure.exists():
            save(failure, dict(status='failed',seed=seed,domain=task['domain'],
                reason='subprocess_exit',returncode=result.returncode,
                log=str(log_file.relative_to(ROOT)),cpu_seconds=None,
                cpu_source='Measured only at scheduler shard wrapper; task CPU is not assigned zero',
                cpu_account=str(Path(plan['output_dir'])/'cluster_control/jobs/audit/tasks'/
                    f'{args.start}_{args.end}.json')))
        return dict(seed=seed, returncode=result.returncode, log_file=str(log_file.relative_to(ROOT)))

    outcomes = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        for future in as_completed([pool.submit(invoke, task) for task in tasks]):
            outcomes.append(future.result())
            save(output/'progress.json', dict(completed=len(outcomes), registered_jobs=len(tasks)))
    save(output/'result.json', dict(status='finished', node=node, start=args.start, end=args.end,
        workers=min(args.workers, len(tasks)), outcomes=outcomes,
        failed=[r for r in outcomes if r['returncode']]))
    print('Done: audit shard '+json.dumps(dict(completed=len(outcomes),
        failed=sum(bool(r['returncode']) for r in outcomes))), flush=True)


if __name__ == '__main__':
    main()
