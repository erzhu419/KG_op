"""Recorded four-cell terminal execution on one frozen arm/search history.

The caller supplies the admitted simulator (or a deterministic substitute).
Only raw replications are shared: every terminal recomputes its certificates.
"""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core import terminal_two_comparison as tournament
from core import terminal_verification as original
from performance import mainline_reference_coverage as wiring
from performance.run_mainline_legacy_backend_bridge import attempt, read, save


def _charges(blocks, variant):
    rows = [row for block in blocks for row in block['raw']]
    failures = [row for block in blocks for row in block['failed_calls']]
    paid = sum(row['paid_by_variant'] == variant for row in rows)
    failed = sum(row['variant'] == variant for row in failures)
    return dict(cumulative_new_verification_calls=paid,
        cumulative_failed_simulation_calls=failed,
        cumulative_new_simulation_invocations=paid + failed)


def execute_four_variants(directory, variants, *, stream, problem_factory=None, resume=False):
    """Execute the fixed four cells, retaining failures and same-block paid prefixes.

    ``problem_factory(item)`` is called lazily when a missing raw row is needed.
    Omit it for replay-only operation. A failed variant requires explicit resume;
    completed variants are returned unchanged, and other cells still run after a
    failure. The directory is exclusive to this arm, geometry and observation
    stream. Run parallel arms in separate worker processes and directories:
    the scoped verifier hooks are shared within a Python process.
    """
    if tuple(variants) != wiring.VARIANTS:
        raise ValueError('Expected all four variants in registered execution order')
    directory = Path(directory)
    first = variants[wiring.VARIANTS[0]]
    fields = ('seed', 'arm', 'domain', 'task', 'geometry', 'verification_seed',
              'search_evaluation_count', 'tau')
    context = {key: deepcopy(first[key]) for key in fields}
    for name, item in variants.items():
        if item['variant_name'] != name or any(item[key] != value for key, value in context.items()):
            raise ValueError('Four variants must share exactly one frozen arm/search history')
    frozen = dict(stream=stream, variants=variants)
    input_path = directory / 'inputs.json'
    if input_path.exists():
        if read(input_path) != frozen:
            raise ValueError('Saved arm/history, stream or frozen terminal inputs changed')
    else:
        save(input_path, frozen)
    context['stream'] = stream
    same_third = (variants['original_old']['frozen_shortlist'][2]['point'] ==
                  variants['reference_old']['frozen_shortlist'][2]['point'])
    physical_bound = 368 if same_third else 512
    blocks = [read(path) for path in sorted((directory / 'raw_blocks').glob('*.json'))]
    if any(block['context'] != context for block in blocks):
        raise ValueError('Raw block belongs to another arm/history or observation stream')
    if not resume:
        for name in variants:
            cell = directory / name
            owned = _charges(blocks, name)['cumulative_new_simulation_invocations']
            if owned and not (cell / 'verification.json').exists() and not (cell / 'failure.json').exists():
                raise RuntimeError('Interrupted paid prefix requires explicit resume; no resampling')
    results = {}
    for name, item in variants.items():
        cell = directory / name
        if (cell / 'verification.json').exists():
            results[name] = read(cell / 'verification.json')
        elif (cell / 'failure.json').exists() and not resume:
            results[name] = read(cell / 'failure.json')
        else:
            try:
                results[name] = _execute_variant(directory, cell, name, item, context,
                    blocks, problem_factory, physical_bound, resume=resume)
            except Exception:
                if not (cell / 'failure.json').exists():
                    raise
                results[name] = read(cell / 'failure.json')
    total_rows = sum(len(block['raw']) for block in blocks)
    total_failures = sum(len(block['failed_calls']) for block in blocks)
    result = dict(status='complete' if all(row['status'] == 'complete' for row in results.values())
        else 'complete_with_failures', seed=first['seed'], arm=first['arm'], stream=stream,
        variants=results, truth_used=False, new_source_fits=0, posterior_restorations=0,
        physical_verification_calls=total_rows, failed_simulation_calls=total_failures,
        physical_simulation_invocations=total_rows + total_failures,
        physical_successful_call_upper_bound=physical_bound,
        logical_verification_calls=sum(row['logical_verification_calls'] for row in results.values()),
        reused_raw_calls=sum(row['reused_raw_calls'] for row in results.values()),
        accounting_scope='Logical/reused fields describe the last attempt per cell. '
            'Physical totals include every successful and failed invocation across all attempts; '
            'attempts.jsonl preserves earlier logical replays and CPU.',
        variant_execution_order=list(wiring.VARIANTS))
    save(directory / 'execution.json', result)
    return result


def _execute_variant(directory, cell, name, item, context, blocks, factory, physical_bound, *, resume):
    used, actual = [], None
    points = [list(row['point']) for row in item['frozen_shortlist']]
    proxy_fields = dict(alpha=item['task']['alpha'], tau=item['tau'],
                        simulation_noise_model='iid_gaussian')
    with attempt(cell, 'verify-reference-coverage', item['domain'], item['arm']) as cost:
        cost.update(variant=name, resume=bool(resume), new_source_fits=0,
            source_model_restorations=0, posterior_restorations=0, engine_restorations=0,
            new_truth_function_calls=0, new_target_initial_calls=0, new_target_search_calls=0,
            download_bytes=0, failed_simulation_calls=0, new_simulation_invocations=0,
            reused_raw_calls=0, reused_previous_variant_raw_calls=0,
            reused_same_variant_raw_calls=0)

        def accounting():
            return dict(logical_verification_calls=sum(row['count'] for row in used),
                reused_raw_calls=cost['reused_raw_calls'],
                reused_previous_variant_raw_calls=cost['reused_previous_variant_raw_calls'],
                reused_same_variant_raw_calls=cost['reused_same_variant_raw_calls'],
                new_verification_calls=cost['new_verification_calls'],
                failed_simulation_calls=cost['failed_simulation_calls'],
                new_simulation_invocations=cost['new_simulation_invocations'],
                **_charges(blocks, name))

        def block_run(kind, index, indices, expected, operation):
            nonlocal actual
            identity = dict(kind=kind, index=index, ordered_candidate_indices=indices,
                            ordered_points=[points[i] for i in indices])
            matches = [block for block in blocks if block['identity'] == identity]
            if len(matches) > 1:
                raise ValueError('Duplicate saved raw identity')
            if matches:
                block = matches[0]
            else:
                block = dict(name=f'block_{len(blocks):04d}', context=deepcopy(context),
                             identity=identity, raw=[], failed_calls=[])
                blocks.append(block)
            path = directory / 'raw_blocks' / (block['name'] + '.json')
            entry = dict(identity, raw_file=str(path), expected_count=expected,
                stored_rows_before=len(block['raw']), count=0, status='partial',
                reused_raw_calls=0, new_verification_calls=0)
            used.append(entry)

            def simulate(point, rng=None):
                nonlocal actual
                cursor = entry['count']
                width = len(indices)
                row_identity = dict(replication=cursor // width, side=cursor % width,
                                    point=identity['ordered_points'][cursor % width])
                if cursor >= expected or list(point) != row_identity['point']:
                    raise ValueError('Verifier changed the recorded point/order or budget')
                if cursor < len(block['raw']):
                    row = block['raw'][cursor]
                    if any(row[key] != value for key, value in row_identity.items()):
                        raise ValueError('Saved replication, pair side or physical point changed')
                    value = row['raw']
                    cost['reused_raw_calls'] += 1
                    field = ('reused_same_variant_raw_calls' if row['paid_by_variant'] == name
                             else 'reused_previous_variant_raw_calls')
                    cost[field] += 1
                    entry['reused_raw_calls'] += 1
                else:
                    if factory is None:
                        raise RuntimeError('Missing recorded raw requires an explicitly admitted problem_factory')
                    if sum(len(saved['raw']) for saved in blocks) >= physical_bound:
                        raise RuntimeError('Four-cell registered physical successful-call bound exceeded')
                    if actual is None:
                        actual = factory(item)
                    cost['new_simulation_invocations'] += 1
                    try:
                        value = np.asarray(actual.simulate(point, rng), dtype=float).tolist()
                    except Exception as error:
                        block['failed_calls'].append(dict(row_identity, variant=name, error=repr(error)))
                        cost['failed_simulation_calls'] += 1
                        save(path, block)
                        raise
                    block['raw'].append(dict(row_identity, raw=value, paid_by_variant=name))
                    cost['new_verification_calls'] += 1
                    entry['new_verification_calls'] += 1
                    # Save the first side before a later side of a CRN pair can fail.
                    save(path, block)
                entry['count'] += 1
                return np.asarray(value, dtype=float)

            returned = operation(SimpleNamespace(**proxy_fields, simulate=simulate))
            if entry['count'] != expected:
                raise ValueError('Verifier did not consume its complete registered block')
            entry['status'] = 'complete'
            return returned

        original_safety = original.verify_frozen_policy
        original_objective = original.verify_paired_objective_dominance

        def safety(problem, point, **kwargs):
            index = kwargs['candidate_index']
            if kwargs['seed'] != item['verification_seed'] or list(point) != points[index]:
                raise ValueError('Safety slot or verification stream changed')
            return block_run('safety', index, [index], kwargs['verification_budget'],
                             lambda proxy: original_safety(proxy, point, **kwargs))

        def objective(problem, challenger, incumbent, **kwargs):
            if kwargs['seed'] != item['verification_seed']:
                raise ValueError('Objective verification stream changed')
            indices = [points.index(list(challenger)), points.index(list(incumbent))]
            return block_run('objective', kwargs.get('comparison_block_index', 0), indices,
                2 * kwargs['comparison_budget'],
                lambda proxy: original_objective(proxy, challenger, incumbent, **kwargs))

        try:
            with ExitStack() as stack:
                for module in (original, tournament):
                    stack.enter_context(patch.object(module, 'verify_frozen_policy', safety))
                    stack.enter_context(patch.object(module, 'verify_paired_objective_dominance', objective))
                deployed, verification = wiring.run_variant(SimpleNamespace(**proxy_fields), item)
            charges = accounting()
            if not (verification['verification_budget'] == charges['logical_verification_calls'] ==
                    charges['reused_raw_calls'] + charges['new_verification_calls'] <= 352):
                raise ValueError('Logical/reused/new verification accounting or 352 cap failed')
            if verification['total_evaluation_count'] > (362 if item['arm'] == 'initialization_only' else 365):
                raise ValueError('Frozen total online budget exceeded')
            result = dict(status='complete', variant=name, seed=item['seed'], arm=item['arm'],
                stream=context['stream'], protocol=item['protocol'], deployed=list(deployed),
                verification_seed=item['verification_seed'], verification=verification,
                blocks_used=used, truth_used=False, **charges)
            save(cell / 'verification.json', result)
            return result
        except Exception as error:
            result = dict(status='failed', variant=name, seed=item['seed'], arm=item['arm'],
                stream=context['stream'], protocol=item['protocol'], error=repr(error),
                blocks_used=used, truth_used=False, **accounting())
            failure_path = cell / 'failure.json'
            if failure_path.exists():
                history = sorted(cell.glob('failure_attempt_*.json'))
                save(cell / f'failure_attempt_{len(history) + 1:03d}.json', read(failure_path))
            save(failure_path, result)
            # The existing attempt context records the exception and its costs;
            # execute_four_variants then retains it and continues the other cells.
            raise
