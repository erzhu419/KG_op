"""Joint two-stream seal and full-pool audit using the frozen Stage73 machinery."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import socket
import subprocess
import sys

from performance import run_mainline_safe_unsampled_audit as audit
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = 'performance/manifests/mainline_terminal_recording_seed_replication_v1.json'


def children(plan):
    return [(row, read(ROOT/row['manifest'])) for row in plan['streams']]


def inventory_path(plan):
    return ROOT/plan['output_dir']/'mechanism/inventory.json'


def require_inventory(plan):
    path = inventory_path(plan)
    if not path.exists():
        raise RuntimeError('Both streams must seal all terminal outcomes and full inventories before truth')
    value = read(path)
    assert value['status'] == 'sealed' and value['registered_streams'] == 2
    assert value['registered_terminal_cells'] == 864 and value['registered_task_streams'] == 48
    return value


def freeze(plan):
    if inventory_path(plan).exists():
        return require_inventory(plan)
    streams = children(plan)
    assert len(streams) == 2
    seals = []
    # Neither child's candidate inventory nor any truth cache opens until both
    # complete terminal file sets, including retained failures, are validated.
    with attempt(ROOT/plan['output_dir']/'mechanism/global_seal', 'seal-both-terminal-streams') as cost:
        cost.update(audit.ZERO)
        for descriptor, child in streams:
            seal, _ = audit.terminal.load_sealed(child)
            assert seal['registered_pairs'] == 216 and seal['registered_terminal_cells'] == 432
            seals.append(dict(stream=descriptor['name'], manifest=descriptor['manifest'],
                registered_pairs=seal['registered_pairs'],
                registered_terminal_cells=seal['registered_terminal_cells'],
                completed_old=seal['completed_old'], completed_new=seal['completed_new']))
    inventories = []
    for descriptor, child in streams:
        saved = audit.freeze(child)
        inventories.append(dict(stream=descriptor['name'],
            inventory_file=str((audit.directory(child)/'inventory.json').relative_to(ROOT)),
            distinct_policy_task_units=saved['distinct_policy_task_units'],
            maximum_analytic_function_calls=saved['maximum_analytic_function_calls']))
    result = dict(status='sealed', registered_streams=2, registered_task_streams=48,
        registered_terminal_cells=864, seals=seals, inventories=inventories, truth_used=False,
        maximum_analytic_function_calls=sum(row['maximum_analytic_function_calls'] for row in inventories),
        sealed_utc=datetime.now(timezone.utc).isoformat())
    save(inventory_path(plan), result)
    return result


def task_directory(plan, task):
    prepared = read(ROOT/plan['output_dir']/'prepared.json')
    family = next(f for f in prepared['replicates'][0]['families'] if f['domain'] == task['domain'])
    source = read(ROOT/family['search_manifest'])
    return ROOT/source['output_dir']/f"seed_{task['geometry_task_seed']}"


def reuse_task_truth(outer, child, seed, previous):
    """Reuse deterministic labels only, after the joint inventory barrier."""
    require_inventory(outer)
    task = next(t for t in child['tasks'] if t['geometry_task_seed'] == seed)
    target = task_directory(child, task)
    with attempt(ROOT/child['output_dir']/'truth_cache_reuse'/f'seed_{seed}',
                 'reuse-sealed-full-pool-truth', task['domain']) as cost:
        cost.update(audit.ZERO, seed=seed)
        if not (target/'initialization.json').exists():
            return dict(status='initialization_unavailable', seed=seed)
        initial = [r['point'] for r in read(target/'initialization.json')['history']]
        cache = audit.cache_read(target/'truth.json')
        rows = []
        for prior in previous:
            old_task = next(t for t in prior['tasks'] if t['geometry_task_seed'] == seed)
            assert all(task[k] == old_task[k] for k in ('domain', 'geometry', 'optimizer_seed'))
            source = task_directory(prior, old_task)
            if not (source/'truth.json').exists():
                continue
            if not (source/'initialization.json').exists():
                continue
            assert initial == [r['point'] for r in read(source/'initialization.json')['history']]
            added = 0
            for point, values in audit.cache_read(source/'truth.json').items():
                current = cache.setdefault(point, {})
                for field, value in values.items():
                    if field not in current:
                        current[field] = value
                        added += 1
            rows.append(dict(source_file=str((source/'truth.json').relative_to(ROOT)), added_fields=added))
        audit.cache_save(target/'truth.json', cache)
        result = dict(status='reused', seed=seed, sources=rows, cached_policies=len(cache),
            new_truth_function_calls=0, target_file=str((target/'truth.json').relative_to(ROOT)))
        save(ROOT/child['output_dir']/'truth_cache_reuse'/f'seed_{seed}'/'result.json', result)
        return result


def task_score(plan, seed):
    require_inventory(plan)
    previous = [read(ROOT/plan['reference_manifest'])]
    outcomes = []
    for descriptor, child in children(plan):
        output = audit.directory(child)/f'seed_{seed}'
        result_file, failure_file = output/'result.json', output/'failure.json'
        if not result_file.exists() and not failure_file.exists():
            try:
                reuse_task_truth(plan, child, seed, previous)
                audit.task_score(child, seed)
            except Exception as error:
                if not failure_file.exists():
                    save(failure_file, dict(status='failed', seed=seed,
                        reason=repr(error), cpu_seconds=None,
                        accounting='Retained at the geometry worker; no automatic retry.'))
        record = read(result_file if result_file.exists() else failure_file)
        outcomes.append(dict(stream=descriptor['name'], seed=seed, status=record['status']))
        previous.append(child)
    return dict(status='finished', outcomes=outcomes)


def batch(plan, manifest, start, end, workers):
    require_inventory(plan)
    assert 0 <= start < end <= len(plan['tasks']) == 24 and workers > 0
    output = ROOT/plan['output_dir']/'cluster/audit'/f'shard_{start}_{end}'
    assert not (output/'result.json').exists(), 'Preserve the completed geometry shard'
    output.mkdir(parents=True, exist_ok=True)
    streams = children(plan)

    def invoke(task):
        seed = task['geometry_task_seed']
        logfile = output/f'seed_{seed}.log'
        with logfile.open('x') as handle:
            process = subprocess.run([sys.executable, '-u', '-m',
                'performance.run_mainline_terminal_recording_replication', 'task',
                '--manifest', str(manifest), '--seed', str(seed)], cwd=ROOT,
                stdout=handle, stderr=subprocess.STDOUT)
        if process.returncode:
            for _, child in streams:
                folder = audit.directory(child)/f'seed_{seed}'
                if not (folder/'result.json').exists() and not (folder/'failure.json').exists():
                    save(folder/'failure.json', dict(status='failed', seed=seed,
                        reason='geometry_audit_subprocess_exit', returncode=process.returncode,
                        log=str(logfile.relative_to(ROOT)), cpu_seconds=None,
                        accounting='CPU belongs to scheduler shard wrapper; not zero or attributed twice.'))
        return dict(seed=seed, returncode=process.returncode, log=str(logfile.relative_to(ROOT)))

    results = []
    with ThreadPoolExecutor(max_workers=min(workers, end-start)) as pool:
        for future in as_completed([pool.submit(invoke, task) for task in plan['tasks'][start:end]]):
            results.append(future.result())
            save(output/'progress.json', dict(completed=len(results), registered=end-start))
    result = dict(status='finished', outcomes=results,
        failures=[row for row in results if row['returncode']])
    save(output/'result.json', result)
    return result


def collect(plan):
    require_inventory(plan)
    streams = children(plan)
    for _, child in streams:
        for task in child['tasks']:
            folder = audit.directory(child)/f"seed_{task['geometry_task_seed']}"
            assert (folder/'result.json').exists() or (folder/'failure.json').exists(), 'Unfinished two-stream full-pool audit'
    for descriptor, child in streams:
        audit.collect(child, ROOT/descriptor['manifest'])
    from performance.mainline_terminal_recording_audit import audit_stream, compact_audit
    records = []
    for descriptor, child in streams:
        folder = ROOT/plan['output_dir']/'terminal_recording'/descriptor['name']
        if not (folder/'summary.json').exists():
            with attempt(folder, 'audit-saved-terminal-recording') as cost:
                cost.update(audit.ZERO)
                detail = audit_stream(ROOT, child)
                save(folder/'details.json', detail)
                save(folder/'summary.json', compact_audit(detail))
        records.append(dict(stream=descriptor['name'], **read(folder/'summary.json')))
    save(ROOT/plan['output_dir']/'terminal_recording/summary.json', dict(streams=records))
    from performance.summarize_mainline_terminal_recording_replication import summarize
    return summarize(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'task', 'batch', 'collect'))
    parser.add_argument('--manifest', default=MANIFEST)
    for key in ('seed', 'start', 'end', 'workers'):
        parser.add_argument('--'+key, type=int)
    args = parser.parse_args()
    assert socket.gethostname().split('.')[0] in {f'node{i:03d}' for i in range(1, 7)}
    plan = read(ROOT/args.manifest)
    if args.stage == 'freeze':
        result = freeze(plan)
    elif args.stage == 'task':
        result = task_score(plan, args.seed)
    elif args.stage == 'batch':
        result = batch(plan, args.manifest, args.start, args.end, args.workers)
    else:
        result = collect(plan)
        account = read(ROOT/plan['output_dir']/'completion.json')
        plan.update(status=result['status'], completed_utc=account['completed_utc'],
            summary_file=str(Path(plan['output_dir'])/'summary.json'),
            compact_summary_file=str(Path(plan['output_dir'])/'compact_summary.json'),
            completion_file=str(Path(plan['output_dir'])/'completion.json'))
        save(ROOT/args.manifest, plan)
    print(result.get('status', 'complete'), flush=True)


if __name__ == '__main__':
    main()
