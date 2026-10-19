"""Two-stream, old-only science stages with one global terminal truth barrier."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys

from performance import mainline_reference_coverage_dispatch as previous_dispatch
from performance import run_mainline_two_comparison_confirmation as run
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_same_policy import transform
from performance.run_mainline_terminal_confirmation import input_record


MANIFEST = 'performance/manifests/mainline_precision_confirmation_v1.json'
MODULE = 'performance.run_mainline_precision_confirmation_science'
EXECUTION_PROTOCOL = 'docs/mainline_precision_confirmation_execution_protocol_20261018.md'
STAGES = ('initial', 'neighbors', 'online', 'native', 'terminal')
NODES = {f'node{i:03d}' for i in range(1, 7)}
ZERO = dict(new_target_initial_calls=0, new_target_search_calls=0,
    new_verification_calls=0, new_source_calls=0, new_source_fits=0,
    new_truth_function_calls=0, download_bytes=0)


def require_node():
    node = socket.gethostname().split('.')[0]
    if node not in NODES:
        raise RuntimeError('Scientific execution requires node001-node006: ' + node)
    return node


def children(plan):
    return [(descriptor, read(ROOT / descriptor['manifest'])) for descriptor in plan['streams']]


def families(plan):
    """Read the already materialized family recipes; never regenerate registration."""
    for descriptor, child in children(plan):
        prepared = read(ROOT / child['output_dir'] / 'prepared.json')
        for replica in prepared['replicates']:
            for family in replica['families']:
                yield descriptor, child, replica, family


def stage_path(plan, stage):
    return ROOT / plan['output_dir'] / 'cluster' / stage


def _contract(plan):
    n = len(plan['tasks'])
    assert plan['selected_N'] == n and len(plan['streams']) == 2
    assert len(plan['search_arms']) == 7 and len(plan['terminal_arms']) == 9
    assert plan['terminal_protocols'] == ['old'] and plan['old_protocol'] == run.PROTOCOL
    assert plan['frozen']['d'] == 1000 and plan['frozen']['search_calls'] == 13
    assert plan['frozen']['candidate_budgets'] == [80, 128, 128]
    assert plan['budget']['maximum_verification_calls_per_cell'] == 352
    assert plan['budget']['registered_terminal_cells'] == 18 * n
    stream_seeds = []
    for _, child in children(plan):
        assert child['tasks'] == plan['tasks']
        assert child['search_arms'] == plan['search_arms'] and child['terminal_arms'] == plan['terminal_arms']
        assert child['terminal_protocols'] == ['old'] and child['frozen'] == plan['frozen']
        assert all(t['optimizer_seed'] == t['geometry_task_seed'] for t in child['tasks'])
        stream_seeds.extend(v for row in child['replicates'][0]['seed_by_task'].values() for v in row.values())
    assert len(set(stream_seeds)) == len(stream_seeds) == 4 * n


def admit(manifest, checks):
    require_node()
    assert read(ROOT / checks)['status'] == 'passed', 'Substitute checks must pass before admission'
    plan = read(ROOT / manifest)
    _contract(plan)
    assert plan['status'] == 'registered_not_started' and not plan['execution_started']
    for _, child, _, family in families(plan):
        source, terminal = read(ROOT / family['search_manifest']), read(ROOT / family['terminal_manifest'])
        assert 'fixed_neighbors_file_by_task' not in source and 'fixed_initialization_file_by_task' not in source
        assert terminal['protocols'] == {run.PROTOCOL: [80, 128, 128]}
        mapping = child['replicates'][0]['seed_by_task']
        for seed in source['target_seeds']:
            assert source['observation_seed_by_task'][str(seed)] == mapping[str(seed)]['observation_seed']
            assert terminal['verification_seed_by_task'][str(seed)] == mapping[str(seed)]['verification_seed']
    with attempt(ROOT / plan['output_dir'] / 'admission', 'admit-old-only-two-stream-execution') as cost:
        cost.update(ZERO)
        for descriptor, child in children(plan):
            child['execution_preparation'].update(status='validated', validation_file=str(checks),
                entrypoint=MODULE, independent_scientific_execution=True,
                remaining='Admitted for the frozen old-only two-stream stages and post-global-seal complete-pool audit.')
            child['execution_protocol_document'] = EXECUTION_PROTOCOL
            save(ROOT / descriptor['manifest'], child)
        plan['execution_preparation'].update(status='validated', validation_file=str(checks),
            entrypoint=MODULE, independent_scientific_execution=True,
            remaining='Admitted for the frozen old-only two-stream stages and post-global-seal complete-pool audit.')
        plan['execution_protocol_document'] = EXECUTION_PROTOCOL
        save(ROOT / manifest, plan)
    return dict(status='validated', registered_geometries=len(plan['tasks']),
        registered_terminal_cells=plan['budget']['registered_terminal_cells'],
        execution_started=False, truth_used=False)


def active_plan(path):
    plan = read(path)
    if plan['status'].startswith('complete'):
        raise ValueError('Completed scientific manifest cannot be restarted')
    if plan['execution_preparation']['status'] != 'validated':
        raise ValueError('Old-only two-stream execution has not been admitted')
    return plan


def _completed(plan, stage):
    path = stage_path(plan, stage) / 'completion.json'
    if not path.exists() or read(path).get('status') != 'complete':
        raise ValueError('Required global scheduler stage is unfinished: ' + stage)


def _command(action, *args):
    return [sys.executable, '-u', '-m', MODULE, action, *map(str, args)]


def require_frozen_inputs(plan):
    path = ROOT / plan['output_dir'] / 'frozen_inputs.json'
    if not path.exists():
        raise RuntimeError('Both streams must freeze every old-terminal input before verification')
    frozen = read(path)
    assert frozen['status'] == 'sealed' and frozen['registered_terminal_cells'] == 18 * len(plan['tasks'])
    assert len(frozen['rows']) == frozen['registered_terminal_cells']
    return frozen


def require_seal(plan):
    path = ROOT / plan['output_dir'] / 'execution_status.json'
    if not path.exists():
        raise RuntimeError('Both streams and every old-terminal outcome must be sealed before truth')
    seal = read(path)
    assert seal['status'] == 'sealed' and seal['truth_access_allowed'] is True
    assert seal['registered_streams'] == 2 and seal['registered_terminal_cells'] == 18 * len(plan['tasks'])
    assert len(seal['cases']) == seal['registered_terminal_cells']
    assert all(row['old']['status'] in ('complete', 'failed', 'missing') for row in seal['cases'])
    return seal


def describe(manifest, stage):
    require_node()
    plan = active_plan(ROOT / manifest)
    if stage not in STAGES:
        raise ValueError('Unknown scientific stage: ' + stage)
    index = STAGES.index(stage)
    if index:
        _completed(plan, STAGES[index - 1])
    target = stage_path(plan, stage) / 'jobs.json'
    if target.exists():
        return read(target)
    jobs = []
    relative = str((ROOT / manifest).resolve().relative_to(ROOT))
    if stage in ('initial', 'online'):
        for descriptor, child in children(plan):
            prepared = read(ROOT / child['output_dir'] / 'prepared.json')
            for job in run.replication.search_jobs(prepared, stage == 'initial'):
                job.update(stream=descriptor['name'], child_manifest=descriptor['manifest'])
                job['label'] = descriptor['name'] + '_' + job['label']
                job['command'] = run.command('search', '--child-manifest', job['search_manifest'],
                    '--seed', job['seed'], '--arm', job['arm'])
                jobs.append(job)
    elif stage in ('neighbors', 'native'):
        for descriptor, _, replica, family in families(plan):
            source = read(ROOT / family['search_manifest'])
            failed = {(r['seed'], r['arm']) for r in source.get('execution_failures', [])}
            for seed in source['target_seeds']:
                directory = ROOT / source['output_dir'] / f'seed_{seed}'
                if stage == 'neighbors':
                    if (seed, run.STOP) in failed or (directory / 'neighbors_failure.json').exists():
                        continue
                    if all((directory / name).exists() for name in ('neighbors.json', 'neighbors.npz')):
                        continue
                    arms = [None]
                else:
                    arms = [arm for arm in run.ROLES if (seed, arm) not in failed and not any(
                        (directory / arm / name).exists() for name in ('native_terminal.json', 'native_terminal_failure.json'))]
                for arm in arms:
                    arguments = ['--child-manifest', family['search_manifest'], '--seed', seed]
                    if arm is not None:
                        arguments.extend(['--arm', arm])
                    jobs.append(dict(label=f"{descriptor['name']}_{stage}_{seed}_{arm or 'bank'}",
                        stream=descriptor['name'], child_manifest=descriptor['manifest'],
                        replicate=replica['replicate'], seed=seed, arm=arm, **family,
                        command=run.command(stage, *arguments)))
    else:
        for row in require_frozen_inputs(plan)['rows']:
            if row['status'] != 'frozen':
                continue
            folder = (ROOT / row['old_input']).parent
            if any((folder / name).exists() for name in ('verification.json', 'failure.json')):
                continue
            jobs.append(dict(label=f"{row['stream']}_terminal_{row['seed']}_{row['arm']}",
                stream=row['stream'], seed=row['seed'], arm=row['arm'], old_input=row['old_input'],
                command=_command('terminal', '--manifest', relative, '--stream', row['stream'],
                    '--seed', row['seed'], '--arm', row['arm'])))
    result = dict(stage=stage, manifest=relative, jobs=jobs, truth_used=False,
        created_utc=datetime.now(timezone.utc).isoformat())
    save(target, result)
    if stage == 'initial':
        for descriptor, child in children(plan):
            child.update(status='search_running', execution_started=True)
            save(ROOT / descriptor['manifest'], child)
        plan.update(status='search_running', execution_started=True)
        save(ROOT / manifest, plan)
    return result


def _job_resolved(job, stage):
    if stage == 'terminal':
        directory = (ROOT / job['old_input']).parent
        return any((directory / name).exists() for name in ('verification.json', 'failure.json'))
    source = read(ROOT / job['search_manifest'])
    path = ROOT / source['output_dir'] / f"seed_{job['seed']}"
    if stage == 'neighbors':
        return (path / 'neighbors_failure.json').exists() or all((path / name).exists()
            for name in ('neighbors.json', 'neighbors.npz'))
    path = path / job['arm']
    if stage == 'native':
        return any((path / name).exists() for name in ('native_terminal.json', 'native_terminal_failure.json'))
    return (path / 'search.json').exists() or any((r['seed'], r['arm']) ==
        (job['seed'], job['arm']) for r in source.get('execution_failures', []))


def worker(descriptor_path, start, end, workers):
    node = require_node()
    descriptor = read(ROOT / descriptor_path)
    plan, stage = active_plan(ROOT / descriptor['manifest']), descriptor['stage']
    assert 0 <= start < end <= len(descriptor['jobs']) and workers > 0
    index = STAGES.index(stage)
    if index:
        _completed(plan, STAGES[index - 1])
    if stage == 'terminal':
        require_frozen_inputs(plan)
    output = stage_path(plan, stage) / f'shard_{start}_{end}'
    if (output / 'result.json').exists():
        return read(output / 'result.json')
    assigned = descriptor['jobs'][start:end]
    pending = [job for job in assigned if not _job_resolved(job, stage)]
    for job in pending:
        job['command'][0] = sys.executable
    failed = run.replication.run_stage(output, stage, pending, min(workers, len(pending))) if pending else []
    result = dict(status='finished', start=start, end=end, node=node, failed=failed,
        workers=min(workers, len(pending)), labels=[job['label'] for job in assigned],
        executed_jobs=len(pending), reused_jobs=len(assigned) - len(pending), truth_used=False)
    save(output / 'result.json', result)
    return result


def _initial_barrier(plan):
    rows, physical = [], {}
    for descriptor, _, _, family in families(plan):
        source = read(ROOT / family['search_manifest'])
        failed = {(r['seed'], r['arm']): r for r in source.get('execution_failures', [])}
        for seed in source['target_seeds']:
            path = ROOT / source['output_dir'] / f'seed_{seed}' / run.STOP / 'search.json'
            assert path.exists() or (seed, run.STOP) in failed, 'Unfinished initialization'
            if path.exists():
                history = read(path)['history']
                assert len(history) == 10
                points = [row['point'] for row in history]
                if seed in physical:
                    assert physical[seed] == points, 'Same-geometry source-only initial physical policies differ across streams'
                physical[seed] = points
            rows.append(dict(stream=descriptor['name'], seed=seed, domain=family['domain'],
                status='complete' if path.exists() else 'failed',
                search_file=str(path.relative_to(ROOT)) if path.exists() else None))
    assert len(rows) == 2 * len(plan['tasks'])
    save(ROOT / plan['output_dir'] / 'initial_status.json', dict(status='sealed', rows=rows,
        registered_initial_histories=len(rows), truth_used=False))


def _retain_neighbor_failure(job):
    source = read(ROOT / job['search_manifest'])
    path = ROOT / source['output_dir'] / f"seed_{job['seed']}"
    record = dict(seed=job['seed'], stream=job['stream'], domain=source['domain'],
        status='neighbor_preparation_failed', subprocess_job=job, truth_used=False,
        reason='Neighbor subprocess failed; retain STOP and skip dependent online histories without retry.')
    save(path / 'neighbors_failure.json', record)
    affected = {arm['name'] for arm in source['arms'] if arm['name'] != run.STOP}
    for arm in affected:
        run.append_failure(job['search_manifest'], dict(record, arm=arm,
            status='neighbor_dependency_failed', new_search_calls=0))
    for cell in read(ROOT / job['terminal_manifest'])['arms']:
        if cell['history_arm'] in affected:
            run.append_failure(job['terminal_manifest'], dict(record, arm=cell['name'],
                history_arm=cell['history_arm'], status='neighbor_dependency_failed'))
    return record


def freeze_inputs(plan):
    """Reuse the original input constructor and transform; never construct a new terminal."""
    all_rows = []
    for descriptor, child in children(plan):
        run.search_barrier(child)
        rows = []
        child_families = [f for d, _, _, f in families(plan) if d['name'] == descriptor['name']]
        for family in child_families:
            source, terminal = read(ROOT / family['search_manifest']), read(ROOT / family['terminal_manifest'])
            inherited = terminal['inherited_replay_check_file']
            assert read(ROOT / inherited)['status'] == 'passed'
            failed = {(r['seed'], r['arm']): r for r in terminal.get('execution_failures', [])}
            for seed in source['target_seeds']:
                for cell in terminal['arms']:
                    arm, history_arm = cell['name'], cell['history_arm']
                    directory = ROOT / source['output_dir'] / f'seed_{seed}' / history_arm
                    row = dict(stream=descriptor['name'], child_manifest=descriptor['manifest'],
                        seed=seed, arm=arm, history_arm=history_arm,
                        source_case=str(directory.relative_to(ROOT)), **family)
                    failure = failed.get((seed, arm))
                    if failure is None:
                        try:
                            search = read(directory / 'search.json')
                            frozen = read(directory / 'native_terminal.json')['terminal'] if cell['terminal_backend'] == 'native' else search['terminal']
                            item = transform(input_record(search, frozen, source, cell, seed,
                                terminal['verification_seed_by_task'][str(seed)], row['source_case'], tau=0.))
                            path = run.verifier.case_dir(ROOT / terminal['output_dir'], seed, arm, run.PROTOCOL) / 'input.json'
                            if path.exists():
                                assert read(path) == item, 'Frozen old-terminal input changed'
                            else:
                                save(path, item)
                        except Exception as error:
                            failure = dict(status='input_freeze_failed', seed=seed, arm=arm,
                                history_arm=history_arm, reason=repr(error), truth_used=False)
                            run.append_failure(family['terminal_manifest'], failure)
                    rows.append(dict(row, status='missing', failure=failure, old_input=None) if failure is not None
                        else dict(row, status='frozen', old_input=str(path.relative_to(ROOT))))
            save(ROOT / terminal['output_dir'] / 'prepared.json', dict(status='complete',
                protocol=run.PROTOCOL, case_count=sum(r['status'] == 'frozen' for r in rows
                    if r['terminal_manifest'] == family['terminal_manifest']), truth_used=False))
            save(ROOT / terminal['output_dir'] / 'replay_check.json', dict(status='passed', inherited=inherited,
                new_raw_replay=False, scope='Unchanged input_record and same_policy transform; substitutes admit two-stream old-only wiring.'))
        assert len(rows) == 9 * len(child['tasks'])
        save(ROOT / child['output_dir'] / 'frozen_inputs.json', dict(status='sealed', rows=rows,
            registered_pairs=len(rows), registered_terminal_cells=len(rows), truth_used=False))
        all_rows.extend(rows)
    assert len(all_rows) == 18 * len(plan['tasks'])
    assert len({(r['stream'], r['seed'], r['arm']) for r in all_rows}) == len(all_rows)
    result = dict(status='sealed', registered_streams=2, registered_terminal_cells=len(all_rows),
        rows=all_rows, frozen_cells=sum(r['status'] == 'frozen' for r in all_rows),
        missing_cells=sum(r['status'] == 'missing' for r in all_rows), truth_used=False)
    save(ROOT / plan['output_dir'] / 'frozen_inputs.json', result)
    return result


def _retain_terminal_failure(plan, job):
    row = next(r for r in require_frozen_inputs(plan)['rows'] if
        (r['stream'], r['seed'], r['arm']) == (job['stream'], job['seed'], job['arm']))
    directory = (ROOT / row['old_input']).parent
    if any((directory / name).exists() for name in ('verification.json', 'failure.json')):
        return
    ledger = directory / 'attempts.jsonl'
    attempts = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
    raws = sorted(directory.glob('failed_verification_raw_*.json'))
    save(directory / 'failure.json', dict(status='failed', seed=row['seed'], arm=row['arm'],
        domain=row['domain'], reason='Old-terminal subprocess failed without a final record',
        subprocess_job=job, new_verification_calls=sum(r.get('new_verification_calls', 0) for r in attempts) if attempts else None,
        attempts_file=str(ledger.relative_to(ROOT)) if ledger.exists() else None,
        failed_raw_file=str(raws[-1].relative_to(ROOT)) if raws else None,
        call_accounting_complete=bool(attempts), truth_used=False))


def seal_results(plan):
    """An unfinished last cell blocks both streams, including the first stream's truth."""
    cases = []
    for row in require_frozen_inputs(plan)['rows']:
        case = {key: row[key] for key in ('stream', 'seed', 'domain', 'arm', 'history_arm',
            'source_case', 'search_manifest', 'terminal_manifest')}
        if row['status'] != 'frozen':
            old = dict(status='missing', input_file=None, failure=row['failure'])
        else:
            directory = (ROOT / row['old_input']).parent
            result, failure = directory / 'verification.json', directory / 'failure.json'
            assert result.exists() or failure.exists(), f"Unfinished old terminal {row['stream']}/{row['seed']}/{row['arm']}"
            if result.exists():
                saved = read(result)
                assert saved['verification']['max_verification_budget'] == 352
                assert saved['verification']['verification_budget'] == saved['new_verification_calls'] <= 352
            old = dict(status='complete' if result.exists() else 'failed', input_file=row['old_input'],
                **{('verification_file' if result.exists() else 'failure_file'): str((result if result.exists() else failure).relative_to(ROOT))})
        cases.append(dict(case, old=old))
    for descriptor, child in children(plan):
        selected = [row for row in cases if row['stream'] == descriptor['name']]
        save(ROOT / child['output_dir'] / 'execution_status.json', dict(status='sealed', cases=selected,
            registered_terminal_cells=len(selected), registered_pairs=len(selected),
            completed_old=sum(r['old']['status'] == 'complete' for r in selected),
            truth_access_allowed=False, global_truth_gate=str(Path(plan['output_dir']) / 'execution_status.json'),
            truth_used=False))
    result = dict(status='sealed', registered_streams=2, registered_task_streams=2 * len(plan['tasks']),
        registered_terminal_cells=len(cases), cases=cases,
        completed_old=sum(r['old']['status'] == 'complete' for r in cases),
        failed_old=sum(r['old']['status'] == 'failed' for r in cases),
        missing_old=sum(r['old']['status'] == 'missing' for r in cases),
        costs=run.costs(ROOT / plan['output_dir']),
        accounting_note='Snapshot of unique attempt ledgers; final collection includes this coordinator ledger after it closes. Uninstrumented restorations and failed simulator invocations remain unknown.',
        truth_access_allowed=True, truth_used=False, sealed_utc=datetime.now(timezone.utc).isoformat())
    save(ROOT / plan['output_dir'] / 'execution_status.json', result)
    return result


def finish(descriptor_path):
    require_node()
    descriptor = read(ROOT / descriptor_path)
    manifest = ROOT / descriptor['manifest']
    plan, stage = active_plan(manifest), descriptor['stage']
    output = stage_path(plan, stage)
    if (output / 'completion.json').exists():
        return read(output / 'completion.json')
    shards = [read(path) for path in sorted(output.glob('shard_*/result.json'))]
    labels = [label for shard in shards for label in shard['labels']]
    assert sorted(labels) == sorted(job['label'] for job in descriptor['jobs']), 'Unfinished or overlapping scheduler shards'
    failures = [job for shard in shards for job in shard['failed']]
    with attempt(output, 'finish-precision-confirmation-' + stage) as cost:
        cost.update(ZERO)
        if stage in ('initial', 'online'):
            for job in failures:
                source = read(ROOT / job['search_manifest'])
                if not (ROOT / source['output_dir'] / f"seed_{job['seed']}" / job['arm'] / 'search.json').exists():
                    record = previous_dispatch.retain_search_failure(job)
                    plan.setdefault('execution_failures', []).append(dict(record, stream=job['stream']))
            save(manifest, plan)
            if stage == 'initial':
                _initial_barrier(plan)
            else:
                rows = [dict(row, stream=d['name']) for d, child in children(plan) for row in run.search_barrier(child)]
                save(ROOT / plan['output_dir'] / 'search_status.json', dict(status='sealed', rows=rows,
                    registered_histories=len(rows), truth_used=False))
        elif stage == 'neighbors':
            for job in failures:
                if not _job_resolved(job, stage):
                    _retain_neighbor_failure(job)
            for _, _, _, family in families(plan):
                source = read(ROOT / family['search_manifest'])
                failed = {(r['seed'], r['arm']) for r in source.get('execution_failures', [])}
                for seed in source['target_seeds']:
                    directory = ROOT / source['output_dir'] / f'seed_{seed}'
                    assert (seed, run.STOP) in failed or (directory / 'neighbors_failure.json').exists() or all(
                        (directory / name).exists() for name in ('neighbors.json', 'neighbors.npz')), 'Unfinished neighbor bank'
        elif stage == 'native':
            for job in failures:
                source = read(ROOT / job['search_manifest'])
                path = ROOT / source['output_dir'] / f"seed_{job['seed']}" / job['arm']
                if not any((path / name).exists() for name in ('native_terminal.json', 'native_terminal_failure.json')):
                    save(path / 'native_terminal_failure.json', dict(status='native_terminal_failed',
                        seed=job['seed'], history_arm=job['arm'], subprocess_job=job, truth_used=False))
            for _, _, _, family in families(plan):
                source, terminal = read(ROOT / family['search_manifest']), read(ROOT / family['terminal_manifest'])
                for seed in source['target_seeds']:
                    for cell in terminal['arms']:
                        if cell['terminal_backend'] != 'native':
                            continue
                        path = ROOT / source['output_dir'] / f'seed_{seed}' / cell['history_arm'] / 'native_terminal_failure.json'
                        if path.exists():
                            run.append_failure(family['terminal_manifest'], dict(read(path), arm=cell['name'],
                                failure_file=str(path.relative_to(ROOT))))
            freeze_inputs(plan)
        elif stage == 'terminal':
            for job in failures:
                _retain_terminal_failure(plan, job)
            seal_results(plan)
        else:
            raise ValueError('Unknown scientific stage: ' + stage)
    result = dict(status='complete', stage=stage, jobs=len(labels), failed_jobs=failures,
        truth_used=False, completed_utc=datetime.now(timezone.utc).isoformat())
    save(output / 'completion.json', result)
    return result


def terminal(plan, stream, seed, arm):
    require_node()
    row = next(r for r in require_frozen_inputs(plan)['rows'] if
        (r['stream'], r['seed'], r['arm']) == (stream, seed, arm))
    assert row['status'] == 'frozen'
    folder = (ROOT / row['old_input']).parent
    for name in ('verification.json', 'failure.json'):
        if (folder / name).exists():
            return read(folder / name)
    child = next(child for descriptor, child in children(plan) if descriptor['name'] == stream)
    return run.verify_old(child, row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('admit'); p.add_argument('--manifest', default=MANIFEST); p.add_argument('--checks', required=True)
    p = sub.add_parser('describe'); p.add_argument('--manifest', default=MANIFEST); p.add_argument('--stage', choices=STAGES, required=True)
    p = sub.add_parser('worker'); p.add_argument('--descriptor', required=True)
    for name in ('start', 'end', 'workers'):
        p.add_argument('--' + name, type=int, required=True)
    p = sub.add_parser('finish'); p.add_argument('--descriptor', required=True)
    p = sub.add_parser('terminal'); p.add_argument('--manifest', default=MANIFEST)
    p.add_argument('--stream', required=True); p.add_argument('--seed', type=int, required=True); p.add_argument('--arm', required=True)
    args = parser.parse_args()
    if args.action == 'admit':
        result = admit(args.manifest, args.checks)
    elif args.action == 'describe':
        descriptor = describe(args.manifest, args.stage)
        result = dict(stage=args.stage, jobs=len(descriptor['jobs']))
    elif args.action == 'worker':
        result = worker(args.descriptor, args.start, args.end, args.workers)
    elif args.action == 'finish':
        result = finish(args.descriptor)
    else:
        result = terminal(active_plan(ROOT / args.manifest), args.stream, args.seed, args.arm)
    print(json.dumps({key: value for key, value in result.items() if key not in ('cases', 'rows', 'verification')}), flush=True)


if __name__ == '__main__':
    main()
