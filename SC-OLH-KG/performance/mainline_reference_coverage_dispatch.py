"""Five scheduler stages for the registered four-terminal confirmation.

Search implementations and terminal rules are reused unchanged. Every arm gets
one subprocess for its four terminals; target truth is absent from this module.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys

from performance import mainline_reference_coverage_cohort as cohort
from performance import mainline_reference_coverage_execution as execution
from performance import run_mainline_two_comparison_confirmation as run
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_two_comparison_development import build_problem

sys.path.insert(0, str(ROOT.parent / 'scripts'))
from scheduler_node_policy import DEFAULT_CPU_NODES

STAGES = ('initial', 'neighbors', 'online', 'native', 'terminal')
MODULE = 'performance.mainline_reference_coverage_dispatch'


def active_plan(path):
    plan = read(path)
    if plan['status'].startswith('complete'):
        raise ValueError('Completed scientific manifest cannot be restarted')
    return plan


def stage_path(plan, stage):
    return ROOT / plan['output_dir'] / 'cluster' / stage


def require_node():
    node = socket.gethostname().split('.')[0]
    if node not in DEFAULT_CPU_NODES:
        raise RuntimeError('Scientific execution requires node001-node006: ' + node)
    return node


def _completed(plan, stage):
    path = stage_path(plan, stage) / 'completion.json'
    if not path.exists() or read(path).get('status') != 'complete':
        raise ValueError('Required global scheduler stage is unfinished: ' + stage)


def _command(action, *args):
    return [sys.executable, '-u', '-m', MODULE, action, *map(str, args)]


def describe(manifest, stage):
    plan = active_plan(manifest)
    target = stage_path(plan, stage) / 'jobs.json'
    if target.exists():
        return read(target)
    prepared, jobs = run.prepare(plan), []
    relative_manifest = str(Path(manifest).resolve().relative_to(ROOT))
    if stage in ('initial', 'online'):
        if stage == 'online':
            _completed(plan, 'neighbors')
            cohort.require_initial_references(plan, root=ROOT)
        jobs = run.replication.search_jobs(prepared, stage == 'initial')
        for job in jobs:
            if stage == 'online':
                cohort.require_initial_references(plan, seed=job['seed'], root=ROOT)
            job['command'] = run.command('search', '--child-manifest', job['search_manifest'],
                                         '--seed', job['seed'], '--arm', job['arm'])
    elif stage == 'neighbors':
        _completed(plan, 'initial')
        references = cohort.require_initial_references(plan, root=ROOT)
        admitted = {row['seed'] for row in references['rows'] if row['status'] == 'frozen'}
        for _, family in run.families(plan):
            source = read(ROOT / family['search_manifest'])
            for seed in source['target_seeds']:
                directory = ROOT / source['output_dir'] / f'seed_{seed}'
                if seed not in admitted or all((directory / name).exists()
                        for name in ('neighbors.json', 'neighbors.npz')):
                    continue
                jobs.append(dict(label=f'neighbors_{seed}', seed=seed,
                    command=run.command('neighbors', '--child-manifest', family['search_manifest'], '--seed', seed)))
    elif stage == 'native':
        _completed(plan, 'online')
        run.search_barrier(plan)
        for replica, family in run.families(plan):
            source = read(ROOT / family['search_manifest'])
            failed = {(row['seed'], row['arm']) for row in source.get('execution_failures', [])}
            for seed in source['target_seeds']:
                for arm in run.ROLES:
                    directory = ROOT / source['output_dir'] / f'seed_{seed}' / arm
                    if (seed, arm) in failed or any((directory / name).exists()
                            for name in ('native_terminal.json', 'native_terminal_failure.json')):
                        continue
                    jobs.append(dict(label=f'native_{seed}_{arm}', seed=seed, arm=arm,
                        replicate=replica['replicate'], **family,
                        command=run.command('native', '--child-manifest', family['search_manifest'],
                                            '--seed', seed, '--arm', arm)))
    elif stage == 'terminal':
        _completed(plan, 'native')
        for row in cohort.require_frozen_inputs(plan, root=ROOT)['rows']:
            if row['status'] != 'frozen' or (ROOT / row['execution_dir'] / 'execution.json').exists():
                continue
            jobs.append(dict(label=f"terminal_{row['seed']}_{row['arm']}",
                seed=row['seed'], arm=row['arm'],
                command=_command('terminal', '--manifest', relative_manifest,
                                 '--seed', row['seed'], '--arm', row['arm'])))
    else:
        raise ValueError('Unknown frozen stage: ' + stage)
    descriptor = dict(stage=stage, manifest=relative_manifest, jobs=jobs,
        created_utc=datetime.now(timezone.utc).isoformat(), truth_used=False)
    save(target, descriptor)
    if stage == 'initial':
        plan.update(status='search_running', execution_started=True)
        save(manifest, plan)
    return descriptor


def worker(descriptor_path, start, end, workers):
    node = require_node()
    descriptor = read(descriptor_path)
    plan = active_plan(ROOT / descriptor['manifest'])
    if not (0 <= start < end <= len(descriptor['jobs']) and workers > 0):
        raise ValueError('Invalid scheduler shard interval')
    if descriptor['stage'] == 'online':
        _completed(plan, 'neighbors')
        for job in descriptor['jobs'][start:end]:
            cohort.require_initial_references(plan, seed=job['seed'], root=ROOT)
    elif descriptor['stage'] == 'terminal':
        cohort.require_frozen_inputs(plan, root=ROOT)
    output = stage_path(plan, descriptor['stage']) / f'shard_{start}_{end}'
    if (output / 'result.json').exists():
        return read(output / 'result.json')['failed']
    jobs = descriptor['jobs'][start:end]
    for job in jobs:
        job['command'][0] = sys.executable
    failed = run.replication.run_stage(output, descriptor['stage'], jobs, min(workers, len(jobs)))
    save(output / 'result.json', dict(status='finished', start=start, end=end, node=node,
        workers=min(workers, len(jobs)), failed=failed, labels=[job['label'] for job in jobs],
        truth_used=False))
    return failed


def retain_search_failure(job):
    """Keep startup failures too; they have no simulator ledger to replay."""
    source = read(ROOT / job['search_manifest'])
    directory = ROOT / source['output_dir'] / f"seed_{job['seed']}" / job['arm']
    ledger = directory / 'attempts.jsonl'
    if ledger.exists() and ledger.read_text().strip():
        return run.retain_search_failure(job)
    record = dict(seed=job['seed'], arm=job['arm'], domain=source['domain'],
        replicate=job['replicate'], status='search_subprocess_failed_before_ledger',
        reason=f"Subprocess exit {job.get('returncode')}; no simulator attempt ledger",
        attempts_file=None, new_initial_calls=None, new_search_calls=None,
        logical_search_calls=None, call_accounting_complete=False,
        accounting_note='No completed attempt ledger; unrecorded invocations cannot be inferred as zero.',
        failed_history_arm=job['arm'], subprocess_job=job, truth_used=False)
    affected = ([arm['name'] for arm in source['arms']] if job['arm'] == run.STOP else [job['arm']])
    for arm in affected:
        run.append_failure(job['search_manifest'], dict(record, arm=arm))
    for cell in read(ROOT / job['terminal_manifest'])['arms']:
        if cell['history_arm'] in affected:
            run.append_failure(job['terminal_manifest'], dict(record, arm=cell['name'], history_arm=cell['history_arm']))
    return record


def _retain_reference_dependencies(plan, reference):
    missing = {row['seed']: row for row in reference['rows'] if row['status'] != 'frozen'}
    for _, family in run.families(plan):
        source = read(ROOT / family['search_manifest'])
        for seed in source['target_seeds']:
            if seed not in missing:
                continue
            for arm in source['arms']:
                if arm['name'] == run.STOP:
                    continue
                record = dict(seed=seed, arm=arm['name'], status='initial_reference_dependency_failed',
                    initial_reference=missing[seed], new_initial_calls=0, new_search_calls=0,
                    logical_search_calls=10, truth_used=False)
                run.append_failure(family['search_manifest'], record)
                for cell in read(ROOT / family['terminal_manifest'])['arms']:
                    if cell['history_arm'] == arm['name']:
                        run.append_failure(family['terminal_manifest'], dict(record, arm=cell['name'], history_arm=arm['name']))


def _retain_terminal_failure(plan, job):
    """Seal a crashed worker's paid prefixes without replaying any simulator."""
    row = next(row for row in cohort.require_frozen_inputs(plan, root=ROOT)['rows']
               if (row['seed'], row['arm']) == (job['seed'], job['arm']))
    directory = ROOT / row['execution_dir']
    if (directory / 'execution.json').exists():
        return
    blocks = [read(path) for path in sorted((directory / 'raw_blocks').glob('*.json'))]
    results = {}
    for name in cohort.VARIANTS:
        cell = directory / name
        complete, failure = cell / 'verification.json', cell / 'failure.json'
        if not complete.exists() and not failure.exists():
            save(failure, dict(status='failed', variant=name, seed=row['seed'], arm=row['arm'],
                stream=row['stream'], reason='Terminal subprocess interrupted; no automatic replay',
                subprocess_job=job, truth_used=False, logical_verification_calls=None,
                reused_raw_calls=None, **execution._charges(blocks, name)))
        results[name] = read(complete if complete.exists() else failure)
    physical = sum(len(block['raw']) for block in blocks)
    failed = sum(len(block['failed_calls']) for block in blocks)
    save(directory / 'execution.json', dict(status='complete_with_failures',
        seed=row['seed'], arm=row['arm'], stream=row['stream'], variants=results,
        physical_verification_calls=physical, failed_simulation_calls=failed,
        physical_simulation_invocations=physical + failed,
        logical_verification_calls=None, reused_raw_calls=None,
        accounting_note='Interrupted cell logical replay counts are unknown; every saved paid raw and failed invocation is retained.',
        truth_used=False))


def finish(descriptor_path):
    descriptor = read(descriptor_path)
    manifest = ROOT / descriptor['manifest']
    plan, stage = active_plan(manifest), descriptor['stage']
    output = stage_path(plan, stage)
    if (output / 'completion.json').exists():
        return read(output / 'completion.json')
    results = [read(path) for path in sorted(output.glob('shard_*/result.json'))]
    labels = [label for result in results for label in result['labels']]
    if sorted(labels) != sorted(job['label'] for job in descriptor['jobs']):
        raise ValueError('Unfinished or overlapping scheduler shards')
    failures = [job for result in results for job in result['failed']]
    # Meter coordinator work separately; subprocess ledgers own scientific fees.
    with attempt(output, 'finish-reference-coverage-' + stage) as cost:
        cost.update(new_source_fits=0, new_truth_function_calls=0, download_bytes=0)
        if stage in ('initial', 'online'):
            for job in failures:
                record = retain_search_failure(job)
                current = plan.setdefault('execution_failures', [])
                if not any((row['seed'], row['arm']) == (record['seed'], record['arm']) for row in current):
                    current.append(record)
            plan.update(status='search_running', execution_started=True)
            save(manifest, plan)
            if stage == 'initial':
                _retain_reference_dependencies(plan, cohort.freeze_initial_references(plan, root=ROOT))
            else:
                run.search_barrier(plan)
        elif stage == 'neighbors':
            if failures:
                raise RuntimeError('Neighbor preparation failed; online search remains blocked')
        elif stage == 'native':
            for job in failures:
                source = read(ROOT / job['search_manifest'])
                directory = ROOT / source['output_dir'] / f"seed_{job['seed']}" / job['arm']
                if not any((directory / name).exists() for name in ('native_terminal.json', 'native_terminal_failure.json')):
                    save(directory / 'native_terminal_failure.json', dict(status='native_terminal_failed',
                        seed=job['seed'], history_arm=job['arm'], subprocess_job=job, truth_used=False))
            cohort.freeze_cohort_inputs(plan, root=ROOT)
        elif stage == 'terminal':
            for job in failures:
                _retain_terminal_failure(plan, job)
            cohort.seal_cohort_results(plan, root=ROOT)
        else:
            raise ValueError('Unknown stage: ' + stage)
    result = dict(status='complete', stage=stage, jobs=len(labels), failed_jobs=failures,
        truth_used=False, completed_utc=datetime.now(timezone.utc).isoformat())
    save(output / 'completion.json', result)
    return result


def terminal(plan, seed, arm):
    require_node()
    rows = cohort.require_frozen_inputs(plan, root=ROOT)['rows']
    matches = [row for row in rows if (row['seed'], row['arm']) == (seed, arm)]
    if len(matches) != 1 or matches[0]['status'] != 'frozen':
        raise ValueError('No admitted frozen arm for terminal execution')
    row = matches[0]
    directory = ROOT / row['execution_dir']
    if (directory / 'execution.json').exists():
        return read(directory / 'execution.json')
    variants = {name: read(ROOT / row['variants'][name]['input_file']) for name in cohort.VARIANTS}
    return execution.execute_four_variants(directory, variants, stream=row['stream'],
        problem_factory=build_problem, resume=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('describe'); p.add_argument('--manifest', required=True)
    p.add_argument('--stage', choices=STAGES, required=True)
    p = sub.add_parser('worker'); p.add_argument('--descriptor', required=True)
    for name in ('start', 'end', 'workers'):
        p.add_argument('--' + name, type=int, required=True)
    p = sub.add_parser('finish'); p.add_argument('--descriptor', required=True)
    p = sub.add_parser('terminal'); p.add_argument('--manifest', required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--arm', required=True)
    args = parser.parse_args()
    if args.action == 'describe':
        result = describe(args.manifest, args.stage)
        print(json.dumps(dict(stage=args.stage, jobs=len(result['jobs']))))
    elif args.action == 'worker':
        worker(args.descriptor, args.start, args.end, args.workers)
    elif args.action == 'finish':
        print(json.dumps(finish(args.descriptor)))
    else:
        print(terminal(active_plan(args.manifest), args.seed, args.arm)['status'], flush=True)


if __name__ == '__main__':
    main()
