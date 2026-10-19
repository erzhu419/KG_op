"""Metadata barriers for the registered four-terminal confirmation cohort.

These helpers read saved searches and verifier outcomes only. They never restore
an engine, score a candidate, invoke a simulator, or open target truth.
"""
from copy import deepcopy
from pathlib import Path

from performance.mainline_reference_coverage import (
    VARIANTS, freeze_initial_order, make_variants,
)
from performance.run_mainline_legacy_backend_bridge import ROOT, read, save
from performance.run_mainline_same_policy import transform
from performance.run_mainline_terminal_confirmation import input_record

STOP = 'initialization_only'


def _output(plan, root):
    return Path(root) / plan['output_dir']


def _families(plan, root):
    # The existing materialize(plan) writes this metadata without loading models.
    prepared = read(_output(plan, root) / 'prepared.json')
    for replica in prepared['replicates']:
        for family in replica['families']:
            yield replica, family, read(Path(root) / family['search_manifest']), read(
                Path(root) / family['terminal_manifest'])


def _failure(source, seed, arm):
    return next((row for row in source.get('execution_failures', [])
                 if (row['seed'], row['arm']) == (seed, arm)), None)


def _save_frozen(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen metadata changed: ' + str(path))
    else:
        save(path, value)


def _require(plan, root, name, field, expected):
    path = _output(plan, root) / name
    if not path.exists():
        raise ValueError('Required global barrier is absent: ' + name)
    record = read(path)
    if record.get('status') != 'sealed' or record.get(field) != expected:
        raise ValueError('Required global barrier is incomplete: ' + name)
    return record


def require_initial_references(plan, *, seed=None, root=ROOT):
    """Call before online job admission, and for its particular geometry."""
    result = _require(plan, root, 'initial_references.json',
                      'registered_geometries', len(plan['tasks']))
    if seed is not None:
        matches = [row for row in result['rows'] if row['seed'] == seed]
        if len(matches) != 1 or matches[0]['status'] != 'frozen':
            raise ValueError('Online search has no frozen initial reference order')
    return result


def freeze_initial_references(plan, *, root=ROOT):
    """Freeze all prefix-ten orders before any online history has started."""
    output, root = _output(plan, root), Path(root)
    if (output / 'initial_references.json').exists():
        return require_initial_references(plan, root=root)
    families = list(_families(plan, root))
    for _, _, source, _ in families:
        for seed in source['target_seeds']:
            for arm in source['arms']:
                directory = root / source['output_dir'] / f'seed_{seed}' / arm['name']
                if arm['name'] != STOP and any((directory / name).exists() for name in
                                               ('search.json', 'attempts.jsonl')):
                    raise ValueError('Initial reference order must precede online search')
    rows = []
    for replica, _, source, terminal in families:
        cell = next(cell for cell in terminal['arms'] if cell['name'] == STOP)
        for seed in source['target_seeds']:
            directory = root / source['output_dir'] / f'seed_{seed}' / STOP
            path, failure = directory / 'search.json', _failure(source, seed, STOP)
            row = dict(seed=seed, domain=source['domain'], replicate=replica['replicate'],
                       stream=f"replicate_{replica['replicate']}", order_file=None)
            if not path.exists():
                if failure is None:
                    raise ValueError(f'Unfinished STOP initialization: {seed}')
                rows.append(dict(row, status='missing', failure=failure))
                continue
            search = read(path)
            try:
                item = transform(input_record(search, search['terminal'], source, cell, seed,
                    terminal['verification_seed_by_task'][str(seed)],
                    str(directory.relative_to(root)), tau=0.))
                reference = freeze_initial_order(item, stream=row['stream'],
                                                 input_file=str(path.relative_to(root)))
            except (AssertionError, KeyError, TypeError, ValueError) as error:
                failure = dict(status='initial_reference_admission_failed', seed=seed,
                    error=repr(error), search_file=str(path.relative_to(root)), truth_used=False)
                target = output / 'reference_orders' / f'seed_{seed}.failure.json'
                _save_frozen(target, failure)
                rows.append(dict(row, status='missing', failure=failure))
                continue
            target = output / 'reference_orders' / f'seed_{seed}.json'
            _save_frozen(target, reference)
            rows.append(dict(row, status='frozen', order_file=str(target.relative_to(root))))
    expected = {task['geometry_task_seed'] for task in plan['tasks']}
    if len(rows) != len(expected) or {row['seed'] for row in rows} != expected:
        raise ValueError('Initial reference barrier does not cover every geometry')
    result = dict(status='sealed', registered_geometries=len(rows), rows=rows,
        frozen_references=sum(row['status'] == 'frozen' for row in rows), truth_used=False,
        new_target_calls=0, new_candidate_scores=0, posterior_restorations=0)
    _save_frozen(output / 'initial_references.json', result)
    return result


def _search_barrier(plan, root):
    rows = []
    for _, _, source, _ in _families(plan, root):
        for seed in source['target_seeds']:
            for arm in source['arms']:
                name = arm['name']
                path = Path(root) / source['output_dir'] / f'seed_{seed}' / name / 'search.json'
                failure = _failure(source, seed, name)
                if not path.exists() and failure is None:
                    raise ValueError(f'Unfinished search history: {seed}/{name}')
                rows.append(dict(seed=seed, arm=name, status='complete' if path.exists() else 'failed',
                                 search_file=str(path.relative_to(root)) if path.exists() else None,
                                 failure=failure))
    expected = {(task['geometry_task_seed'], arm) for task in plan['tasks']
                for arm in plan['search_arms']}
    if len(rows) != len(expected) or {(r['seed'], r['arm']) for r in rows} != expected:
        raise ValueError('Search barrier does not retain all registered histories')
    result = dict(status='sealed', registered_histories=len(rows), rows=rows, truth_used=False)
    _save_frozen(_output(plan, root) / 'search_status.json', result)
    return {(row['seed'], row['arm']): row for row in rows}


def require_frozen_inputs(plan, *, root=ROOT):
    expected = len(plan['tasks']) * len(plan['terminal_arms']) * len(VARIANTS)
    return _require(plan, root, 'frozen_inputs.json', 'registered_terminal_cells', expected)


def freeze_cohort_inputs(plan, *, root=ROOT):
    """Keep four entries for every arm, including retained search failures."""
    output, root = _output(plan, root), Path(root)
    if (output / 'frozen_inputs.json').exists():
        return require_frozen_inputs(plan, root=root)
    orders = {row['seed']: row for row in require_initial_references(plan, root=root)['rows']}
    histories = _search_barrier(plan, root)
    rows = []
    for replica, _, source, terminal in _families(plan, root):
        for seed in terminal['target_seeds']:
            for cell in terminal['arms']:
                arm, history_arm = cell['name'], cell['history_arm']
                directory = root / source['output_dir'] / f'seed_{seed}' / history_arm
                history = histories[seed, history_arm]
                row = dict(seed=seed, arm=arm, history_arm=history_arm, domain=source['domain'],
                    stream=f"replicate_{replica['replicate']}", source_case=str(directory.relative_to(root)),
                    execution_dir=str((output / 'terminal_results' / f'seed_{seed}' / arm).relative_to(root)))
                failure = history['failure'] if history['status'] == 'failed' else None
                if orders[seed]['status'] != 'frozen':
                    failure = dict(reason='initial_reference_unavailable', reference=orders[seed])
                if failure is None and cell['terminal_backend'] == 'native':
                    native = directory / 'native_terminal.json'
                    native_failure = directory / 'native_terminal_failure.json'
                    if not native.exists():
                        if not native_failure.exists():
                            raise ValueError(f'Unfinished native input: {seed}/{arm}')
                        failure = read(native_failure)
                if failure is not None:
                    rows.append(dict(row, status='missing', variants={name: dict(status='missing',
                        input_file=None, failure=deepcopy(failure)) for name in VARIANTS}))
                    continue
                search = read(directory / 'search.json')
                frozen = (read(directory / 'native_terminal.json')['terminal']
                          if cell['terminal_backend'] == 'native' else search['terminal'])
                try:
                    item = transform(input_record(search, frozen, source, cell, seed,
                        terminal['verification_seed_by_task'][str(seed)],
                        str(directory.relative_to(root)), tau=0.))
                    variants = make_variants(item, read(root / orders[seed]['order_file']))
                except (AssertionError, KeyError, TypeError, ValueError) as error:
                    failure = dict(status='terminal_input_admission_failed', error=repr(error),
                                   source_case=row['source_case'], truth_used=False)
                    rows.append(dict(row, status='missing', variants={name: dict(status='missing',
                        input_file=None, failure=deepcopy(failure)) for name in VARIANTS}))
                    continue
                inputs = {}
                for name in VARIANTS:
                    target = output / 'terminal_inputs' / f'seed_{seed}' / arm / name / 'input.json'
                    _save_frozen(target, variants[name])
                    inputs[name] = dict(status='frozen', input_file=str(target.relative_to(root)))
                rows.append(dict(row, status='frozen', variants=inputs))
    expected = {(task['geometry_task_seed'], cell['name']) for task in plan['tasks']
                for cell in plan['terminal_arms']}
    if len(rows) != len(expected) or {(r['seed'], r['arm']) for r in rows} != expected:
        raise ValueError('Terminal input barrier lost a registered arm')
    result = dict(status='sealed', registered_arm_inputs=len(rows),
        registered_terminal_cells=len(rows) * len(VARIANTS), rows=rows, truth_used=False,
        new_target_calls=0, new_candidate_scores=0, posterior_restorations=0)
    _save_frozen(output / 'frozen_inputs.json', result)
    return result


def seal_cohort_results(plan, *, root=ROOT):
    """Permit truth only after every registered cell has an outcome or failure."""
    output, root = _output(plan, root), Path(root)
    execution_path = output / 'execution_status.json'
    if execution_path.exists() and read(execution_path).get('status') == 'sealed':
        return require_truth_seal(plan, root=root)
    rows, cells = require_frozen_inputs(plan, root=root)['rows'], []
    physical = failed_invocations = 0
    for row in rows:
        directory = root / row['execution_dir']
        if row['status'] == 'frozen':
            execution = directory / 'execution.json'
            if not execution.exists():
                raise ValueError(f'Unfinished four-terminal execution: {row["seed"]}/{row["arm"]}')
            account = read(execution)
            physical += account['physical_verification_calls']
            failed_invocations += account['failed_simulation_calls']
        for name in VARIANTS:
            descriptor = row['variants'][name]
            cell = {key: row[key] for key in ('seed', 'arm', 'history_arm', 'domain', 'stream')}
            cell.update(variant=name, input_file=descriptor['input_file'], C=None, G=None)
            if descriptor['status'] != 'frozen':
                cells.append(dict(cell, status='missing', failure=descriptor['failure']))
                continue
            complete, failure = directory / name / 'verification.json', directory / name / 'failure.json'
            path = complete if complete.exists() else failure
            if not path.exists():
                raise ValueError(f'Unfinished terminal cell: {row["seed"]}/{row["arm"]}/{name}')
            result = read(path)
            if result['status'] not in ('complete', 'failed'):
                raise ValueError('Unsealed terminal outcome')
            cells.append(dict(cell, status=result['status'], result_file=str(path.relative_to(root)),
                historical_failure_file=str(failure.relative_to(root)) if failure.exists() else None))
    result = dict(status='sealed', registered_terminal_cells=len(cells), cells=cells,
        completed_cells=sum(row['status'] == 'complete' for row in cells),
        failed_cells=sum(row['status'] == 'failed' for row in cells),
        missing_cells=sum(row['status'] == 'missing' for row in cells),
        physical_verification_calls=physical, failed_simulation_calls=failed_invocations,
        truth_access_allowed=True, truth_used=False,
        historical_accounts=deepcopy(plan.get('historical_accounts', [])))
    save(execution_path, result)
    return result


def require_truth_seal(plan, *, root=ROOT):
    expected = len(plan['tasks']) * len(plan['terminal_arms']) * len(VARIANTS)
    result = _require(plan, root, 'execution_status.json', 'registered_terminal_cells', expected)
    if result.get('truth_access_allowed') is not True:
        raise ValueError('Global terminal seal has not admitted target truth')
    return result
