"""Scheduler shards for the frozen mainline; scientific stages remain unchanged.

For each stage: describe, upload its small descriptor, submit, wait, finish.
Run finish on the shared server checkout. No truth is read by these commands;
after finish new, run the existing summary and finalize commands serially.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys

from performance import run_mainline_two_comparison_confirmation as run
from performance.run_mainline_legacy_backend_bridge import ROOT, read, save

sys.path.insert(0, str(ROOT.parent / 'scripts'))
from scheduler_node_policy import DEFAULT_CPU_NODES, parse_cpu_nodes

STAGES = ('initial', 'neighbors', 'online', 'native', 'old', 'new')
MODULE = 'performance.mainline_cluster_dispatch'


def active_plan(path):
    plan = read(path)
    if plan['status'].startswith('complete'):
        raise ValueError('Completed scientific manifest cannot be restarted')
    return plan


def stage_path(plan, stage):
    return ROOT / plan['output_dir'] / 'cluster' / stage


def describe(manifest, stage):
    plan = active_plan(manifest)
    target = stage_path(plan, stage) / 'jobs.json'
    if target.exists():
        return read(target)
    prepared, jobs = run.prepare(plan), []
    if stage in ('initial', 'online'):
        if stage == 'online':
            assert read(stage_path(plan, 'neighbors') / 'completion.json')['status'] == 'complete'
        jobs = run.replication.search_jobs(prepared, stage == 'initial')
        for job in jobs:
            job['command'] = run.command('search', '--child-manifest', job['search_manifest'],
                                         '--seed', job['seed'], '--arm', job['arm'])
    elif stage == 'neighbors':
        assert read(stage_path(plan, 'initial') / 'completion.json')['status'] == 'complete'
        for replica, family in run.families(plan):
            source = read(ROOT / family['search_manifest'])
            failed = {(r['seed'], r['arm']) for r in source.get('execution_failures', [])}
            for seed in source['target_seeds']:
                if (seed, run.STOP) not in failed:
                    jobs.append(dict(label=f"neighbors_{seed}", seed=seed,
                        command=run.command('neighbors', '--child-manifest', family['search_manifest'], '--seed', seed)))
    elif stage == 'native':
        run.search_barrier(plan)
        for replica, family in run.families(plan):
            source = read(ROOT / family['search_manifest'])
            failed = {(r['seed'], r['arm']) for r in source.get('execution_failures', [])}
            for seed in source['target_seeds']:
                for arm in run.ROLES:
                    directory = ROOT / source['output_dir'] / f'seed_{seed}' / arm
                    if (seed, arm) in failed or any((directory / f).exists() for f in
                            ('native_terminal.json', 'native_terminal_failure.json')):
                        continue
                    jobs.append(dict(label=f'native_{seed}_{arm}', seed=seed, arm=arm, **family,
                        command=run.command('native', '--child-manifest', family['search_manifest'],
                                            '--seed', seed, '--arm', arm)))
    else:
        rows = run.frozen_rows(plan)
        if stage == 'new':
            inventory = read(ROOT / plan['output_dir'] / 'block_inventory.json')
            available = {(r['seed'], r['arm']) for r in inventory['rows']}
            rows = [r for r in rows if (r['seed'], r['arm']) in available]
        for row in rows:
            if row['status'] == 'frozen':
                jobs.append(dict(label=f"{stage}_{row['seed']}_{row['arm']}",
                    seed=row['seed'], arm=row['arm'], command=run.command(stage,
                    '--manifest', Path(manifest).resolve().relative_to(ROOT),
                    '--seed', row['seed'], '--arm', row['arm'])))
    descriptor = dict(stage=stage, manifest=str(Path(manifest).resolve().relative_to(ROOT)),
        jobs=jobs, created_utc=datetime.now(timezone.utc).isoformat(), truth_used=False)
    save(target, descriptor)
    if stage == 'initial':
        plan.update(status='search_running', execution_started=True)
        save(manifest, plan)
    return descriptor


def worker(descriptor_path, start, end, workers):
    node = socket.gethostname().split('.')[0]
    if node not in DEFAULT_CPU_NODES:
        raise RuntimeError(f'Mainline scientific shards run only on node001-node006, got {node}')
    descriptor = read(descriptor_path)
    plan = active_plan(ROOT / descriptor['manifest'])
    jobs = descriptor['jobs'][start:end]
    assert 0 <= start < end <= len(descriptor['jobs']) and workers > 0
    for job in jobs:
        job['command'][0] = sys.executable
    output = stage_path(plan, descriptor['stage']) / f'shard_{start}_{end}'
    failed = run.replication.run_stage(output, descriptor['stage'], jobs, min(workers, len(jobs)))
    save(output / 'result.json', dict(status='finished', start=start, end=end,
        node=node, workers=min(workers, len(jobs)), failed=failed,
        labels=[j['label'] for j in jobs], truth_used=False))
    return failed


def finish(descriptor_path):
    """One coordinator records failures and crosses the existing global barrier."""
    descriptor = read(descriptor_path)
    manifest = ROOT / descriptor['manifest']
    plan, stage = active_plan(manifest), descriptor['stage']
    output = stage_path(plan, stage)
    if (output / 'completion.json').exists():
        return read(output / 'completion.json')
    results = [read(p) for p in sorted(output.glob('shard_*/result.json'))]
    labels = [label for r in results for label in r['labels']]
    assert sorted(labels) == sorted(j['label'] for j in descriptor['jobs']), 'Unfinished or overlapping shards'
    failures = [j for r in results for j in r['failed']]
    if stage in ('initial', 'online'):
        for job in failures:
            record = run.retain_search_failure(job)
            current = plan.setdefault('execution_failures', [])
            if not any((r['seed'], r['arm']) == (record['seed'], record['arm']) for r in current):
                current.append(record)
        plan.update(status='search_running', execution_started=True)
        save(manifest, plan)
        if stage == 'online':
            rows = run.search_barrier(plan)
            save(ROOT / plan['output_dir'] / 'search_status.json', dict(status='sealed', rows=rows,
                registered_histories=len(rows), completed_histories=sum(r['status'] == 'complete' for r in rows),
                execution_failures=plan.get('execution_failures', []), truth_used=False))
    elif stage == 'neighbors':
        if failures:
            raise RuntimeError('Neighbor preparation failed; online search remains blocked')
    elif stage == 'native':
        for _, family in run.families(plan):
            source, terminal = read(ROOT / family['search_manifest']), read(ROOT / family['terminal_manifest'])
            for seed in terminal['target_seeds']:
                for cell in terminal['arms']:
                    if cell['terminal_backend'] != 'native' or cell['history_arm'] not in run.ROLES:
                        continue
                    failure = ROOT / source['output_dir'] / f'seed_{seed}' / cell['history_arm'] / 'native_terminal_failure.json'
                    if failure.exists():
                        run.append_failure(family['terminal_manifest'], dict(read(failure), arm=cell['name'],
                            domain=family['domain'], failure_file=str(failure.relative_to(ROOT))))
            save(ROOT / source['output_dir'] / 'completion.json',
                 dict(status='search_complete', **run.costs(ROOT / source['output_dir'])))
        run.freeze_inputs(plan)
    elif stage == 'old':
        run.prepare_inventory(plan)
    else:
        run.seal_execution(plan)
    result = dict(status='complete', stage=stage, jobs=len(labels), failed_jobs=failures,
                  truth_used=False, completed_utc=datetime.now(timezone.utc).isoformat())
    save(output / 'completion.json', result)
    return result


def submit_command(args, descriptor):
    plan = active_plan(ROOT / descriptor['manifest'])
    nodes = parse_cpu_nodes(','.join(DEFAULT_CPU_NODES))
    relative = Path(args.descriptor).resolve().relative_to(ROOT)
    command = shlex.join(['env', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1',
        'MKL_NUM_THREADS=1', 'PYTHONPATH=.', args.remote_python, '-u', '-m', MODULE,
        'worker', '--descriptor', str(relative), '--start', '{start}', '--end', '{end}',
        '--workers', '{workers}'])
    result = [sys.executable, str(args.scheduler), 'submit-cpu-batch',
        '--items', str(len(descriptor['jobs'])), '--cmd-template', command,
        '--cwd', args.remote_cwd, '--nodes', ','.join(nodes),
        '--signature', f"KG_op/{plan['name']}/{descriptor['stage']}/{{start}}-{{end}}",
        '--description', f"SC-OLH-KG {plan['name']} {descriptor['stage']} {{node}} [{{start}},{{end}})",
        '--ram-mb', str(args.ram_mb), '--project', 'KG_op', '--skip-launch-staging',
        '--env-spec', 'none', '--json']
    if args.dry_run:
        result.append('--dry-run')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('describe'); p.add_argument('--manifest', required=True); p.add_argument('--stage', choices=STAGES, required=True)
    p = sub.add_parser('worker'); p.add_argument('--descriptor', required=True)
    for name in ('start', 'end', 'workers'):
        p.add_argument('--' + name, type=int, required=True)
    p = sub.add_parser('finish'); p.add_argument('--descriptor', required=True)
    p = sub.add_parser('submit'); p.add_argument('--descriptor', required=True)
    p.add_argument('--scheduler', default=str(Path.home() / 'mine_code/scheduleurm/skill/scheduler.py'))
    p.add_argument('--remote-cwd', required=True); p.add_argument('--remote-python', required=True)
    p.add_argument('--ram-mb', type=int, required=True, help='Per scheduler shard memory reservation')
    p.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.action == 'describe':
        result = describe(args.manifest, args.stage)
        print(json.dumps(dict(stage=args.stage, jobs=len(result['jobs']))))
    elif args.action == 'worker':
        worker(args.descriptor, args.start, args.end, args.workers)
    elif args.action == 'finish':
        print(json.dumps(finish(args.descriptor)))
    else:
        descriptor = read(args.descriptor)
        if not descriptor['jobs']:
            print('No pending jobs; run finish directly')
            return
        command = submit_command(args, descriptor)
        completed = subprocess.run(command, text=True, capture_output=True)
        plan = active_plan(ROOT / descriptor['manifest'])
        save(stage_path(plan, descriptor['stage']) / ('scheduler_plan.json' if args.dry_run else 'scheduler_submission.json'),
            dict(command=command, returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr))
        print(completed.stdout, end=''); print(completed.stderr, end='', file=sys.stderr)
        raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
