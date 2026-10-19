"""Fresh-geometry paired terminals with global search, input and truth barriers."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys

from performance import run_mainline_observation_seed_replication as replication
from performance import run_mainline_role_fresh as fresh
from performance import run_mainline_terminal_equal as verifier
from performance import run_mainline_two_comparison_development as candidate
from performance.mainline_modified_terminal import prepare as prepare_old_inputs, PROTOCOL
from performance.prepare_mainline_two_comparison_terminal import candidate_input
from performance.register_mainline_two_comparison_development import block_inventory
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_terminal_confirmation import native_terminal, costs, ROLES

MANIFEST = ROOT / 'performance/manifests/mainline_terminal_two_comparison_confirmation_v1.json'
MODULE = 'performance.run_mainline_two_comparison_confirmation'
STOP = 'initialization_only'


def prepare(plan):
    return replication.materialize(plan)


def families(plan):
    return [(replica, family) for replica in prepare(plan)['replicates']
            for family in replica['families']]


def command(stage, *args):
    return [sys.executable, '-u', '-m', MODULE, stage, *map(str, args)]


def append_failure(path, record):
    child = read(ROOT / path)
    failed = child.setdefault('execution_failures', [])
    if not any((r['seed'], r['arm']) == (record['seed'], record['arm']) for r in failed):
        failed.append(record)
        save(ROOT / path, child)


def retain_search_failure(job):
    """Retain any actual history failure and every terminal depending on it."""
    source = read(ROOT / job['search_manifest'])
    directory = ROOT / source['output_dir'] / f"seed_{job['seed']}" / job['arm']
    raw_files = sorted(directory.glob('failed_search_raw_*.json'))
    ledger = [json.loads(line) for line in (directory / 'attempts.jsonl').read_text().splitlines()]
    paid = sum(r.get('new_target_initial_calls', 0) + r.get('new_target_search_calls', 0)
               for r in ledger)
    state = directory / 'failed_search_engine.pkl'
    record = dict(replicate=job['replicate'], domain=source['domain'], seed=job['seed'],
        arm=job['arm'], status='search_failed', logical_search_calls=paid + (0 if job['arm'] == STOP else 10),
        new_search_calls=sum(r.get('new_target_search_calls', 0) for r in ledger),
        new_initial_calls=sum(r.get('new_target_initial_calls', 0) for r in ledger),
        reason=ledger[-1].get('error', f"Subprocess exit {job.get('returncode')}"),
        failed_raw_file=str(raw_files[-1].relative_to(ROOT)) if raw_files else None,
        failed_state_file=str(state.relative_to(ROOT)) if state.exists() else None,
        attempts_file=str((directory / 'attempts.jsonl').relative_to(ROOT)))
    affected = [a['name'] for a in source['arms']] if job['arm'] == STOP else [job['arm']]
    for arm in affected:
        append_failure(job['search_manifest'], dict(record, arm=arm,
            status='search_failed' if arm == job['arm'] else 'initialization_dependency_failed',
            failed_history_arm=job['arm']))
    terminal = read(ROOT / job['terminal_manifest'])
    for cell in terminal['arms']:
        if cell['history_arm'] in affected:
            append_failure(job['terminal_manifest'], dict(record, arm=cell['name'],
                history_arm=cell['history_arm'], failed_history_arm=job['arm']))
    return record


def search_barrier(plan):
    rows = []
    for _, family in families(plan):
        source = read(ROOT / family['search_manifest'])
        failures = {(r['seed'], r['arm']): r for r in source.get('execution_failures', [])}
        for seed in source['target_seeds']:
            for arm in source['arms']:
                name = arm['name']
                path = ROOT / source['output_dir'] / f'seed_{seed}' / name / 'search.json'
                key = (seed, name)
                assert path.exists() or key in failures, f'Unfinished search {seed}/{name}'
                rows.append(dict(seed=seed, arm=name, domain=family['domain'],
                    status='complete' if path.exists() else 'failed',
                    search_file=str(path.relative_to(ROOT)) if path.exists() else None,
                    failure=failures.get(key)))
    assert len(rows) == len(plan['tasks']) * len(plan['search_arms'])
    return rows


def run_search(plan, workers, manifest_path=MANIFEST):
    """Initialize every task, freeze every neighbor bank, then search each history."""
    prepared, output = prepare(plan), ROOT / plan['output_dir']
    plan.update(status='search_running', execution_started=True)
    save(manifest_path, plan)

    def launch(initial):
        jobs = replication.search_jobs(prepared, initial)
        for job in jobs:
            job['command'] = command('search', '--child-manifest', job['search_manifest'],
                                     '--seed', job['seed'], '--arm', job['arm'])
        failed = replication.run_stage(output, 'initialization' if initial else 'online-search', jobs, workers)
        for job in failed:
            record = retain_search_failure(job)
            plan.setdefault('execution_failures', []).append(record)
        save(manifest_path, plan)

    launch(True)
    jobs = []
    for replica, family in families(plan):
        source = read(ROOT / family['search_manifest'])
        failed = {(r['seed'], r['arm']) for r in source.get('execution_failures', [])}
        for seed in source['target_seeds']:
            if (seed, STOP) in failed:
                continue
            assert (ROOT / source['output_dir'] / f'seed_{seed}' / STOP / 'search.json').exists()
            jobs.append(dict(label=f"r{replica['replicate']}_neighbors_{seed}",
                command=command('neighbors', '--child-manifest', family['search_manifest'], '--seed', seed)))
    if replication.run_stage(output, 'fixed-neighbors', jobs, workers):
        raise RuntimeError('Neighbor preparation failed; no online search started')
    for _, family in families(plan):
        source = read(ROOT / family['search_manifest'])
        failed = {(r['seed'], r['arm']) for r in source.get('execution_failures', [])}
        for seed in source['target_seeds']:
            if (seed, STOP) not in failed:
                for name in ('neighbors.json', 'neighbors.npz'):
                    assert (ROOT / source['output_dir'] / f'seed_{seed}' / name).exists()
    launch(False)
    rows = search_barrier(plan)
    seal = dict(status='sealed', rows=rows, registered_histories=len(rows),
        completed_histories=sum(r['status'] == 'complete' for r in rows),
        execution_failures=plan.get('execution_failures', []), truth_used=False)
    save(output / 'search_status.json', seal)
    return seal


def freeze_native(source, seed, arm):
    directory = ROOT / source['output_dir'] / f'seed_{seed}' / arm
    failure = directory / 'native_terminal_failure.json'
    if failure.exists():
        raise RuntimeError('Recorded native freeze failure is not retried')
    try:
        native_terminal(source, seed, arm)
    except BaseException as error:
        save(failure, dict(status='native_terminal_failed', seed=seed, history_arm=arm,
            reason=repr(error), source_case=str(directory.relative_to(ROOT)), truth_used=False))
        raise


def freeze_inputs(plan):
    """Freeze both protocols together, with one descriptor for every registered pair."""
    search_barrier(plan)
    output, rows = ROOT / plan['output_dir'], []
    children = families(plan)
    # This existing helper meters itself; keep it outside the parent attempt.
    for _, family in children:
        prepare_old_inputs(read(ROOT / family['search_manifest']), read(ROOT / family['terminal_manifest']))
    with attempt(output, 'freeze-paired-terminal-inputs') as cost:
        cost.update(new_source_fits=0, new_truth_function_calls=0, download_bytes=0)
        for _, family in children:
            source, terminal = read(ROOT / family['search_manifest']), read(ROOT / family['terminal_manifest'])
            failures = {(r['seed'], r['arm']): r for r in terminal.get('execution_failures', [])}
            for seed in terminal['target_seeds']:
                for cell in terminal['arms']:
                    arm = cell['name']
                    old = verifier.case_dir(ROOT / terminal['output_dir'], seed, arm, PROTOCOL) / 'input.json'
                    new = output / 'candidate_inputs' / f'seed_{seed}' / arm / 'input.json'
                    row = dict(seed=seed, arm=arm, domain=family['domain'], history_arm=cell['history_arm'],
                        source_case=str(Path(source['output_dir']) / f'seed_{seed}' / cell['history_arm']),
                        search_manifest=family['search_manifest'], terminal_manifest=family['terminal_manifest'])
                    if (seed, arm) in failures:
                        rows.append(dict(row, status='missing', failure=failures[seed, arm],
                                         old_input=None, new_input=None))
                        continue
                    item = candidate_input(read(old))
                    if new.exists():
                        assert read(new) == item, 'Frozen candidate input changed'
                    else:
                        save(new, item)
                    rows.append(dict(row, status='frozen', old_input=str(old.relative_to(ROOT)),
                                     new_input=str(new.relative_to(ROOT))))
        assert len(rows) == len(plan['tasks']) * len(plan['terminal_arms'])
        assert len({(r['seed'], r['arm']) for r in rows}) == len(rows)
        result = dict(status='sealed', registered_pairs=len(rows), rows=rows,
            frozen_pairs=sum(r['status'] == 'frozen' for r in rows),
            missing_pairs=sum(r['status'] == 'missing' for r in rows), truth_used=False,
            frozen_utc=datetime.now(timezone.utc).isoformat())
        save(output / 'frozen_inputs.json', result)
    return result


def freeze(plan, workers):
    search_barrier(plan)
    jobs = []
    for replica, family in families(plan):
        source = read(ROOT / family['search_manifest'])
        failed = {(r['seed'], r['arm']) for r in source.get('execution_failures', [])}
        for seed in source['target_seeds']:
            for arm in ROLES:
                directory = ROOT / source['output_dir'] / f'seed_{seed}' / arm
                if (seed, arm) in failed or (directory / 'native_terminal.json').exists():
                    continue
                if (directory / 'native_terminal_failure.json').exists():
                    continue
                jobs.append(dict(label=f"r{replica['replicate']}_native_{seed}_{arm}",
                    seed=seed, arm=arm, **family,
                    command=command('native', '--child-manifest', family['search_manifest'],
                                    '--seed', seed, '--arm', arm)))
    replication.run_stage(ROOT / plan['output_dir'], 'native-terminals', jobs, workers)
    for _, family in families(plan):
        source = read(ROOT / family['search_manifest'])
        terminal = read(ROOT / family['terminal_manifest'])
        for seed in source['target_seeds']:
            for cell in terminal['arms']:
                if cell['terminal_backend'] != 'native' or cell['history_arm'] not in ROLES:
                    continue
                failure = ROOT / source['output_dir'] / f'seed_{seed}' / cell['history_arm'] / 'native_terminal_failure.json'
                if failure.exists():
                    append_failure(family['terminal_manifest'], dict(read(failure), arm=cell['name'],
                        domain=family['domain'], failure_file=str(failure.relative_to(ROOT))))
        save(ROOT / source['output_dir'] / 'completion.json', dict(status='search_complete', **costs(ROOT / source['output_dir'])))
    return freeze_inputs(plan)


def frozen_rows(plan):
    frozen = read(ROOT / plan['output_dir'] / 'frozen_inputs.json')
    assert frozen['status'] == 'sealed'
    assert len(frozen['rows']) == len(plan['tasks']) * len(plan['terminal_arms'])
    return frozen['rows']


def verify_old(plan, row):
    assert row in frozen_rows(plan), 'All paired inputs must be frozen before terminal calls'
    assert row['status'] == 'frozen'
    directory = (ROOT / row['old_input']).parent
    complete, failure = directory / 'verification.json', directory / 'failure.json'
    if complete.exists():
        return read(complete)
    if failure.exists():
        raise RuntimeError('Recorded old verifier failure is not retried')
    terminal = read(ROOT / row['terminal_manifest'])
    try:
        verifier.verify(ROOT / terminal['output_dir'], terminal, row['seed'], row['arm'], PROTOCOL)
        return read(complete)
    except BaseException as error:
        ledger = directory / 'attempts.jsonl'
        attempts = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        raw_files = sorted(directory.glob('failed_verification_raw_*.json'))
        save(failure, dict(status='failed', seed=row['seed'], arm=row['arm'], domain=row['domain'],
            reason=repr(error), new_verification_calls=sum(r.get('new_verification_calls', 0) for r in attempts),
            failed_raw_file=str(raw_files[-1].relative_to(ROOT)) if raw_files else None,
            attempts_file=str(ledger.relative_to(ROOT)), truth_used=False))
        raise


def prepare_inventory(plan):
    """Only completed old terminals provide cache blocks; failed cells remain missing."""
    output, rows, missing = ROOT / plan['output_dir'], [], []
    with attempt(output, 'index-paired-confirmation-blocks') as cost:
        cost.update(new_source_fits=0, new_truth_function_calls=0, download_bytes=0)
        for row in frozen_rows(plan):
            if row['status'] != 'frozen':
                missing.append(row)
                continue
            directory = (ROOT / row['old_input']).parent
            old = directory / 'verification.json'
            if not old.exists():
                failure = directory / 'failure.json'
                assert failure.exists(), f'Unfinished old terminal {row["seed"]}/{row["arm"]}'
                missing.append(dict(row, status='old_verification_failed', failure_file=str(failure.relative_to(ROOT))))
                continue
            rows.append(block_inventory(dict(seed=row['seed'], domain=row['domain'], arm=row['arm'],
                input_file=row['new_input'], original_input_file=row['old_input'],
                original_verification_file=str(old.relative_to(ROOT)), quality_baseline_defined=None)))
        inventory = dict(status='complete', rows=rows, missing_cells=missing,
            registered_cells=len(rows) + len(missing), available_cells=len(rows), truth_used=False)
        save(output / 'block_inventory.json', inventory)
        child = dict(output_dir=str(Path(plan['output_dir']) / 'new_terminal'),
            block_inventory=str((output / 'block_inventory.json').relative_to(ROOT)),
            available_candidate_cells=len(rows), registered_cells_per_protocol=len(rows) + len(missing),
            inherited_failures=missing, budget=dict(
                maximum_new_physical_verification_calls=sum(r['new_physical_calls_upper_bound'] for r in rows),
                maximum_candidate_terminal_logical_calls=352 * len(rows)))
        save(output / 'new_terminal_plan.json', child)
    return child, inventory


def verify_new(plan, row):
    # The inventory is written only after every old result/failure has been recorded.
    child = read(ROOT / plan['output_dir'] / 'new_terminal_plan.json')
    inventory = read(ROOT / child['block_inventory'])
    matches = [r for r in inventory['rows'] if (r['seed'], r['arm']) == (row['seed'], row['arm'])]
    assert len(matches) == 1 and row in frozen_rows(plan)
    return candidate.verify_case(child, matches[0])


def seal_execution(plan):
    output, cases = ROOT / plan['output_dir'], []
    child = read(output / 'new_terminal_plan.json')
    for row in frozen_rows(plan):
        pair = {k: row[k] for k in ('seed', 'arm', 'domain', 'history_arm', 'source_case')}
        if row['status'] != 'frozen':
            pair.update(old=dict(status='missing', input_file=None, failure=row['failure']),
                        new=dict(status='missing', input_file=None, failure=row['failure']))
        else:
            old_dir = (ROOT / row['old_input']).parent
            old_result, old_failure = old_dir / 'verification.json', old_dir / 'failure.json'
            if old_result.exists():
                pair['old'] = dict(status='complete', input_file=row['old_input'],
                                   verification_file=str(old_result.relative_to(ROOT)))
                new_dir = candidate.case_dir(child, row)
                result, failure = new_dir / 'verification.json', new_dir / 'failure.json'
                assert result.exists() or failure.exists(), f'Unfinished new terminal {row["seed"]}/{row["arm"]}'
                pair['new'] = dict(status='complete' if result.exists() else 'failed', input_file=row['new_input'],
                    **{('verification_file' if result.exists() else 'failure_file'):
                       str((result if result.exists() else failure).relative_to(ROOT))})
            else:
                assert old_failure.exists(), f'Unfinished old terminal {row["seed"]}/{row["arm"]}'
                pair['old'] = dict(status='failed', input_file=row['old_input'],
                                   failure_file=str(old_failure.relative_to(ROOT)))
                pair['new'] = dict(status='missing', input_file=row['new_input'],
                    reason='old_verification_failed', failure_file=str(old_failure.relative_to(ROOT)))
        cases.append(pair)
    accounting = costs(output)
    physical = sum(accounting['cost'][key] for key in
                   ('new_target_initial_calls', 'new_target_search_calls', 'new_verification_calls'))
    assert physical <= plan['budget']['maximum_target_physical_calls']
    result = dict(status='sealed', cases=cases, registered_pairs=len(cases),
        registered_terminal_cells=2 * len(cases),
        completed_old=sum(r['old']['status'] == 'complete' for r in cases),
        completed_new=sum(r['new']['status'] == 'complete' for r in cases),
        costs=accounting, new_physical_target_calls=physical,
        truth_access_allowed=True, truth_used=False,
        sealed_utc=datetime.now(timezone.utc).isoformat())
    save(output / 'execution_status.json', result)
    return result


def run_terminal(plan, workers, manifest_path=MANIFEST):
    rows, output = frozen_rows(plan), ROOT / plan['output_dir']
    save(output / 'execution_status.json', dict(status='running', truth_access_allowed=False))
    jobs = []
    for row in rows:
        if row['status'] != 'frozen':
            continue
        directory = (ROOT / row['old_input']).parent
        if not any((directory / name).exists() for name in ('verification.json', 'failure.json')):
            jobs.append(dict(label=f"old_{row['seed']}_{row['arm']}",
                command=command('old', '--manifest', manifest_path, '--seed', row['seed'], '--arm', row['arm'])))
    replication.run_stage(output, 'old-terminals', jobs, workers)
    child, inventory = prepare_inventory(plan)
    jobs = []
    for row in inventory['rows']:
        directory = candidate.case_dir(child, row)
        if not any((directory / name).exists() for name in ('verification.json', 'failure.json')):
            jobs.append(dict(label=f"new_{row['seed']}_{row['arm']}",
                command=command('new', '--manifest', manifest_path, '--seed', row['seed'], '--arm', row['arm'])))
    replication.run_stage(output, 'new-terminals', jobs, workers)
    return seal_execution(plan)


def finalize_accounting(plan, manifest_path=MANIFEST):
    """Finish only after global terminal sealing and the independent truth summary."""
    output = ROOT / plan['output_dir']
    ledger = output / 'finalization' / 'attempts.jsonl'
    with attempt(ledger.parent, 'finalize-fresh-paired-confirmation') as cost:
        cost.update(new_source_fits=0, new_target_initial_calls=0,
            new_truth_function_calls=0, new_rng_draws=0, download_bytes=0)
        seal, summary = read(output / 'execution_status.json'), read(output / 'summary.json')
        assert seal['status'] == 'sealed'
        assert summary['registered_paired_cells'] == seal['registered_pairs']
        assert summary['registered_outcome_cells'] == seal['registered_terminal_cells']
        assert summary['completed_old_cells'] == seal['completed_old']
        assert summary['completed_new_cells'] == seal['completed_new']
        old_paid = new_paid = old_logical = new_logical = reused_old = reused_new = old_tail = 0
        failed_old_paid = failed_new_paid = 0
        for pair in seal['cases']:
            old, new = pair['old'], pair['new']
            if old['status'] in ('complete', 'failed'):
                saved = read(ROOT / old['verification_file' if old['status'] == 'complete' else 'failure_file'])
                paid = saved['new_verification_calls']
                old_paid += paid
                old_logical += saved['verification']['verification_budget'] if old['status'] == 'complete' else paid
                if old['status'] == 'complete':
                    old_tail += sum(max(r['verification_budget'] - (80 if r['candidate_index'] == 0 else 120), 0)
                                    for r in saved['verification']['attempts'])
                else:
                    failed_old_paid += paid
            if new['status'] in ('complete', 'failed'):
                saved = read(ROOT / new['verification_file' if new['status'] == 'complete' else 'failure_file'])
                new_paid += saved['cumulative_new_verification_calls']
                new_logical += saved['logical_verification_calls']
                reused_old += saved['reused_old_raw_calls']
                reused_new += saved['reused_new_raw_calls']
                if new['status'] == 'failed':
                    failed_new_paid += saved['cumulative_new_verification_calls']
        rows = []
        for path in sorted(output.rglob('attempts.jsonl')):
            rows.extend(dict(json.loads(line), ledger=str(path.relative_to(ROOT)))
                        for line in path.read_text().splitlines() if line.strip())
        assert sum(r.get('new_verification_calls', 0) for r in rows) == old_paid + new_paid
        logical_bound = plan['budget']['maximum_verification_logical_calls_per_protocol']
        assert old_logical <= logical_bound and new_logical <= logical_bound
    rows.append(dict(json.loads(ledger.read_text().splitlines()[-1]), ledger=str(ledger.relative_to(ROOT))))
    fields = ('new_source_calls', 'new_target_initial_calls', 'new_target_search_calls',
              'new_verification_calls', 'new_source_fits', 'new_truth_function_calls', 'download_bytes')
    new_work = {field: sum(r.get(field, 0) for r in rows) for field in fields}
    physical = sum(new_work[k] for k in ('new_target_initial_calls', 'new_target_search_calls', 'new_verification_calls'))
    assert physical <= plan['budget']['maximum_target_physical_calls']
    new_work.update(new_rng_draws=None, randomized_new_observations=bool(physical),
        source_model_restorations=None, search_engine_restorations=None,
        failed_simulation_calls_recorded=sum(r.get('failed_simulation_calls', 0) for r in rows))
    parent = read(ROOT / plan['parent_manifest'])
    inherited = list(dict.fromkeys(plan['historical_accounts'] +
        [str(Path(parent['output_dir']) / 'completion.json')]))
    result = dict(status=summary['status'], completed_utc=datetime.now(timezone.utc).isoformat(),
        registered_geometries=len(plan['tasks']), registered_pairs=seal['registered_pairs'],
        registered_terminal_cells=seal['registered_terminal_cells'],
        completed_old=seal['completed_old'], completed_new=seal['completed_new'],
        recorded_cpu_seconds=sum(r.get('cpu_seconds') or 0 for r in rows),
        recorded_attempt_wall_seconds=sum(r.get('wall_seconds') or 0 for r in rows),
        new_work=new_work, new_physical_target_calls=physical,
        terminal_costs=dict(old_logical_calls=old_logical, new_logical_calls=new_logical,
            old_physical_calls=old_paid, additional_new_protocol_physical_calls=new_paid,
            unique_paired_physical_calls=old_paid + new_paid,
            old_raw_consumed_by_new_last_attempt=reused_old,
            new_stage_raw_replayed_last_attempt=reused_new,
            complete_old_safety_tail_calls_beyond_candidate_prefix=old_tail,
            old_paid_raw_not_consumed_by_new_last_attempt=old_paid - reused_old,
            paid_rows_in_failed_old_cells=failed_old_paid, paid_rows_in_failed_new_cells=failed_new_paid),
        failed_attempts=[r for r in rows if r['status'] == 'failed'], attempts=rows,
        execution_failures=plan.get('execution_failures', []), inherited_accounts=inherited,
        source_manifest=plan['source_manifest'],
        accounting_note='Each local attempt ledger is summed once, including registration, failed harnesses, '
            'checks, shared search, native freezes, both terminals and scoring. Shared initial/search calls '
            'are charged once physically. All old safety tails and failed paid prefixes remain charged. '
            'Historical accounts are linked without adding their nested costs. Reused models are unchanged; '
            'search source/engine loading counts are uninstrumented, not zero. Recorded failed simulator '
            'invocations cover the new-protocol adapter; older paths retain failed raw/count/CPU without an '
            'exact failed-invocation counter. CPU excludes pre-meter imports, shell inspection, documentation '
            'and final serialization. Summed attempt wall time is not cohort elapsed time.')
    save(output / 'completion.json', result)
    plan.update(status=summary['status'], completed_utc=result['completed_utc'])
    save(manifest_path, plan)
    return result


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'External termination signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run_search', 'search', 'neighbors',
                                         'freeze', 'terminal', 'native', 'old', 'new', 'finalize'))
    parser.add_argument('--manifest', default=str(MANIFEST))
    parser.add_argument('--child-manifest')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--arm')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if args.stage in ('prepare', 'run_search', 'freeze', 'terminal', 'old', 'new', 'finalize'):
        plan = read(args.manifest)
        if args.stage == 'prepare':
            result = prepare(plan)
        elif args.stage == 'run_search':
            result = run_search(plan, args.workers, Path(args.manifest))
        elif args.stage == 'freeze':
            result = freeze(plan, args.workers)
        elif args.stage == 'terminal':
            result = run_terminal(plan, args.workers, Path(args.manifest))
        elif args.stage == 'finalize':
            result = finalize_accounting(plan, Path(args.manifest))
        else:
            rows = [r for r in frozen_rows(plan) if (r['seed'], r['arm']) == (args.seed, args.arm)]
            assert len(rows) == 1
            result = (verify_old if args.stage == 'old' else verify_new)(plan, rows[0])
        print(result.get('status', 'complete'), flush=True)
    else:
        source = read(ROOT / args.child_manifest)
        assert args.seed in source['target_seeds']
        source['target_seeds'] = [args.seed]
        if args.stage == 'search':
            assert args.arm in [r['name'] for r in source['arms']]
            fresh.search(ROOT / source['output_dir'], source, args.seed, args.arm)
        elif args.stage == 'native':
            assert args.arm in ROLES
            freeze_native(source, args.seed, args.arm)
        else:
            fresh.prepare(ROOT / source['output_dir'], source)


if __name__ == '__main__':
    main()
