"""Execute the frozen terminal development comparison with same-cell raw reuse."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core import terminal_two_comparison as tournament
from performance import benchmark_lodo_meta_prior as benchmark
from performance.prepare_mainline_two_comparison_terminal import candidate_input, run_verifier
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = ROOT / 'performance/manifests/mainline_terminal_two_comparison_development_v1.json'


def case_dir(plan, row):
    return ROOT / plan['output_dir'] / 'cases' / f"seed_{row['seed']}" / row['arm']


def build_problem(item):
    """Construct only the simulator, and only when an unrecorded row is needed."""
    task, geometry = item['task'], dict(item['geometry'])
    if item['domain'] == 'FactorShockStatePolicyRZDT1':
        geometry['shared_shock_scale'] = task['shared_shock_scale']
    problem = benchmark.build_scalarized_problem(item['domain'], task['d'], task['L'],
        task['sigma'], task['alpha'], tuple(task['weights']), problem_kwargs=geometry)
    assert problem.d == 1000 and problem.tau == item['tau']
    np.testing.assert_array_equal(problem.base.task_geometry_shift, geometry['task_geometry_shift'])
    assert problem.base.task_geometry_radius_scale == geometry['task_geometry_radius_scale']
    return problem


def verify_case(plan, row, *, resume=False):
    """Recompute certificates from exact blocks; resume only an explicitly saved prefix."""
    directory = case_dir(plan, row)
    complete, failure = directory / 'verification.json', directory / 'failure.json'
    if complete.exists():
        return read(complete)
    if not resume and (failure.exists() or list((directory / 'new_blocks').glob('*.json'))):
        raise RuntimeError('Saved incomplete verification requires explicit resume; no resampling')
    seal = ROOT / plan['output_dir'] / 'execution_status.json'
    if seal.exists():
        save(seal, dict(status='running', truth_access_allowed=False))
    used, blocks, item, actual = [], {}, None, None
    with attempt(directory, 'verify-two-comparison-development', row['domain'], row['arm']) as cost:
        cost.update(seed=row['seed'], resume=bool(resume), new_source_fits=0,
            new_target_initial_calls=0, new_truth_function_calls=0, engine_restorations=0,
            posterior_restorations=0, download_bytes=0, new_rng_draws=0,
            failed_simulation_calls=0, new_simulation_invocations=0,
            reused_old_raw_calls=0, reused_new_raw_calls=0)

        def accounting():
            fresh = sum(len(block['raw']) for block in blocks.values())
            failed = sum(block.get('failed_simulation_calls', 0) for block in blocks.values())
            return dict(logical_verification_calls=sum(block['count'] for block in used),
                reused_old_raw_calls=cost['reused_old_raw_calls'],
                reused_new_raw_calls=cost['reused_new_raw_calls'],
                new_verification_calls=cost['new_verification_calls'],
                cumulative_new_verification_calls=fresh,
                failed_simulation_calls=failed,
                cumulative_new_simulation_invocations=fresh + failed,
                old_verification_calls=row['old_raw_rows'],
                historical_verification_calls=row['old_raw_rows'] + fresh,
                historical_simulation_invocations=row['old_raw_rows'] + fresh + failed,
                randomized_new_observations=bool(fresh),
                new_rng_draws=None if fresh or failed else 0)

        try:
            item = read(ROOT / row['candidate_input'])
            assert item == candidate_input(read(ROOT / row['original_input'])), 'Frozen input changed'
            assert (item['seed'], item['arm'], item['verification_seed']) == (
                row['seed'], row['arm'], row['verification_seed'])
            for path in sorted((directory / 'new_blocks').glob('*.json')):
                blocks[path.stem] = read(path)
            old_raw = read(ROOT / row['old_raw_file'])
            assert len(old_raw) == row['old_raw_rows']
            points = [record['point'] for record in item['frozen_shortlist']]
            problem = SimpleNamespace(alpha=item['task']['alpha'], tau=item['tau'],
                                      simulation_noise_model='iid_gaussian')

            def block_run(kind, index, indices, expected, operation):
                nonlocal actual
                name = f'{kind}_{index}'
                path = directory / 'new_blocks' / f'{name}.json'
                ordered_points = [points[i] for i in indices]
                identity = dict(seed=item['seed'], arm=item['arm'], domain=item['domain'],
                    verification_seed=item['verification_seed'], kind=kind,
                    ordered_candidate_indices=indices, ordered_points=ordered_points,
                    expected_count=expected)
                index_key = 'candidate_index' if kind == 'safety' else 'comparison_block_index'
                identity[index_key] = index
                matches = [b for b in row['available_blocks']
                           if b['kind'] == kind and b[index_key] == index]
                assert len(matches) <= 1
                old = matches[0] if matches else None
                cached = None
                if old is not None:
                    assert old['reusable_prefix_count'] == expected
                    assert old['stored_count'] >= expected
                    if kind == 'objective':
                        assert indices == [old['challenger_index'], old['incumbent_index']]
                    values = old_raw[old['raw_start']:old['raw_start'] + expected]
                    assert len(values) == expected
                    assert all(v['point'] == ordered_points[i % len(indices)]
                               for i, v in enumerate(values)), 'Old raw point/order mismatch'
                    raw = [v['raw'] for v in values]
                else:
                    cached = blocks.get(name)
                    if cached is None:
                        cached = dict(identity, status='partial', raw=[], failed_simulation_calls=0)
                        blocks[name] = cached
                    assert all(cached[k] == v for k, v in identity.items()), 'Saved block identity changed'
                    assert len(cached['raw']) <= expected
                    raw = cached['raw']
                cursor = 0
                entry = dict(kind=kind, index=index, ordered_candidate_indices=indices,
                    count=0, expected_count=expected, status='partial',
                    origin='old' if old is not None else 'this_stage',
                    raw_file=row['old_raw_file'] if old is not None else str(path.relative_to(ROOT)),
                    raw_start=old['raw_start'] if old is not None else 0,
                    reused_old_raw_calls=0, reused_new_raw_calls=0, new_verification_calls=0)
                used.append(entry)

                def simulate(point, rng=None):
                    nonlocal cursor, actual
                    assert cursor < expected and list(point) == ordered_points[cursor % len(indices)]
                    if cursor < len(raw):
                        value = raw[cursor]
                        field = 'reused_old_raw_calls' if old is not None else 'reused_new_raw_calls'
                        cost[field] += 1
                        entry[field] += 1
                    else:
                        assert old is None
                        assert sum(len(b['raw']) for b in blocks.values()) < row['new_physical_calls_upper_bound']
                        if actual is None:
                            actual = build_problem(item)
                        cost['new_rng_draws'] = None
                        cost['new_simulation_invocations'] += 1
                        try:
                            returned = actual.simulate(point, rng)
                        except BaseException:
                            cached['failed_simulation_calls'] += 1
                            cost['failed_simulation_calls'] += 1
                            save(path, cached)
                            raise
                        cost['new_verification_calls'] += 1
                        entry['new_verification_calls'] += 1
                        value = np.asarray(returned, dtype=float).tolist()
                        raw.append(value)
                        # Persist before the next side of a CRN pair can fail.
                        save(path, cached)
                    cursor += 1
                    entry['count'] = cursor
                    return np.asarray(value, dtype=float)

                proxy = SimpleNamespace(**vars(problem), simulate=simulate)
                try:
                    result = operation(proxy)
                    assert cursor == expected
                    entry['status'] = 'complete'
                    return result
                finally:
                    if cached is not None:
                        cached['status'] = 'complete' if len(raw) == expected else 'partial'
                        save(path, cached)

            original_safety = tournament.verify_frozen_policy
            original_objective = tournament.verify_paired_objective_dominance

            def safety(problem_arg, point, **kwargs):
                index = kwargs['candidate_index']
                assert kwargs['seed'] == item['verification_seed'] and list(point) == points[index]
                return block_run('safety', index, [index], kwargs['verification_budget'],
                    lambda proxy: original_safety(proxy, point, **kwargs))

            def objective(problem_arg, challenger, incumbent, **kwargs):
                assert kwargs['seed'] == item['verification_seed']
                indices = [points.index(list(challenger)), points.index(list(incumbent))]
                return block_run('objective', kwargs['comparison_block_index'], indices,
                    2 * kwargs['comparison_budget'],
                    lambda proxy: original_objective(proxy, challenger, incumbent, **kwargs))

            with patch.object(tournament, 'verify_frozen_policy', safety), \
                 patch.object(tournament, 'verify_paired_objective_dominance', objective):
                deployed, verification = run_verifier(problem, item)
            charges = accounting()
            assert verification['verification_budget'] == charges['logical_verification_calls']
            assert charges['logical_verification_calls'] == (charges['reused_old_raw_calls'] +
                charges['reused_new_raw_calls'] + charges['new_verification_calls'])
            assert charges['logical_verification_calls'] <= 352
            assert charges['cumulative_new_verification_calls'] <= row['new_physical_calls_upper_bound']
            assert verification['total_evaluation_count'] <= (362 if row['arm'] == 'initialization_only' else 365)
            result = dict(status='complete', seed=row['seed'], arm=row['arm'], domain=row['domain'],
                protocol=item['protocol'], verification_seed=item['verification_seed'],
                candidate_input=row['candidate_input'], deployed=list(deployed),
                verification=verification, blocks_used=used, truth_used=False,
                completed_utc=datetime.now(timezone.utc).isoformat(), **charges)
            save(complete, result)
            return result
        except BaseException as error:
            save(failure, dict(status='failed', seed=row['seed'], arm=row['arm'], domain=row['domain'],
                error=repr(error), candidate_input=row['candidate_input'], blocks_used=used,
                truth_used=False, failed_utc=datetime.now(timezone.utc).isoformat(), **accounting()))
            raise


def seal_execution(plan, rows):
    """Seal every registered available result or failure before any truth join."""
    cases, totals = [], dict.fromkeys(('logical_verification_calls', 'reused_old_raw_calls',
        'reused_new_raw_calls', 'cumulative_new_verification_calls', 'old_verification_calls',
        'historical_verification_calls', 'failed_simulation_calls',
        'cumulative_new_simulation_invocations', 'historical_simulation_invocations'), 0)
    for row in rows:
        directory = case_dir(plan, row)
        complete = directory / 'verification.json'
        path = complete if complete.exists() else directory / 'failure.json'
        record = read(path)  # A missing cell prevents sealing, including interrupted runs.
        assert record['status'] in ('complete', 'failed')
        field = 'verification_file' if record['status'] == 'complete' else 'failure_file'
        cases.append(dict(seed=row['seed'], arm=row['arm'], domain=row['domain'],
                          status=record['status'], **{field: str(path.relative_to(ROOT))}))
        for key in totals:
            totals[key] += record[key]
    assert len(cases) == plan['available_candidate_cells']
    assert totals['cumulative_new_verification_calls'] <= plan['budget']['maximum_new_physical_verification_calls']
    assert totals['logical_verification_calls'] <= plan['budget']['maximum_candidate_terminal_logical_calls']
    result = dict(status='sealed', cases=cases, available_cases=len(cases),
        registered_cases=plan['registered_cells_per_protocol'],
        inherited_failures=plan['inherited_failures'], costs=totals,
        completed_cases=sum(r['status'] == 'complete' for r in cases),
        failed_cases=sum(r['status'] == 'failed' for r in cases), truth_access_allowed=True,
        sealed_utc=datetime.now(timezone.utc).isoformat())
    save(ROOT / plan['output_dir'] / 'execution_status.json', result)
    return result


def run(plan):
    rows = read(ROOT / plan['block_inventory'])['rows']
    assert sum(row['new_physical_calls_upper_bound'] for row in rows) <= plan['budget']['maximum_new_physical_verification_calls']
    save(ROOT / plan['output_dir'] / 'execution_status.json', dict(status='running', truth_access_allowed=False))
    for position, row in enumerate(rows, 1):
        try:
            verify_case(plan, row)
        except Exception as error:
            print(f"FAILED {row['seed']} {row['arm']}: {error}", flush=True)
        if position % 9 == 0 or position == len(rows):
            print(f"Finished {position}/{len(rows)} terminal cells; last seed {row['seed']}", flush=True)
    return seal_execution(plan, rows)


def finalize_accounting(plan):
    """Sum local attempt ledgers once; inherited accounts remain linked, not recharged."""
    output = ROOT / plan['output_dir']
    ledger = output / 'finalization' / 'attempts.jsonl'
    with attempt(ledger.parent, 'finalize-terminal-development-accounting') as cost:
        cost.update(new_source_fits=0, new_target_initial_calls=0,
            new_truth_function_calls=0, engine_restorations=0, posterior_restorations=0,
            new_rng_draws=0, download_bytes=0)
        seal, summary = read(output / 'execution_status.json'), read(output / 'summary.json')
        assert seal['status'] == 'sealed'
        assert summary['available_candidate_cells'] == seal['available_cases']
        assert summary['completed_candidate_cells'] == seal['completed_cases']
        assert summary['new_failed_cells'] == seal['failed_cases']
        registration = read(ROOT / plan['registration_completion'])
        history = read(ROOT / plan['inherited_accounts'][0])
        rows = []
        for path in sorted(output.rglob('attempts.jsonl')):
            rows.extend(dict(json.loads(line), ledger=str(path.relative_to(ROOT)))
                        for line in path.read_text().splitlines() if line.strip())
        new = sum(r.get('new_verification_calls', 0) for r in rows)
        failed = sum(r.get('failed_simulation_calls', 0) for r in rows)
        assert new == seal['costs']['cumulative_new_verification_calls']
        assert failed == seal['costs']['failed_simulation_calls']
        assert new <= plan['budget']['maximum_new_physical_verification_calls']
    # The context has now appended this finalization attempt; include it exactly once.
    rows.append(dict(json.loads(ledger.read_text().splitlines()[-1]), ledger=str(ledger.relative_to(ROOT))))
    zeros = ('new_source_calls', 'new_target_initial_calls', 'new_target_search_calls',
        'new_truth_function_calls', 'new_source_fits', 'source_model_restorations',
        'engine_restorations', 'posterior_restorations', 'download_bytes')
    new_work = {key: sum(r.get(key, 0) for r in rows) for key in zeros}
    new_work.update(new_verification_calls=new, failed_simulation_calls=failed,
        new_simulation_invocations=new + failed,
        new_rng_draws=None if new or failed else 0,
        randomized_new_observations=bool(new))
    cpu = sum(r.get('cpu_seconds') or 0 for r in rows)
    wall = sum(r.get('wall_seconds') or 0 for r in rows)
    old_terminal = history['cost']['new_verification_calls']
    old_target = history['new_physical_target_calls']
    result = dict(status='complete_with_missing_cells' if plan['inherited_failures'] or seal['failed_cases']
        else 'complete', finished_utc=datetime.now(timezone.utc).isoformat(),
        registered_cases=seal['registered_cases'], completed_candidate_cases=seal['completed_cases'],
        failed_candidate_cases=seal['failed_cases'], recorded_cpu_seconds=cpu,
        recorded_wall_seconds=wall,
        included_registration_cpu_seconds=registration['recorded_cpu_seconds'],
        post_registration_recorded_cpu_seconds=cpu - registration['recorded_cpu_seconds'],
        new_work=new_work, execution_costs=seal['costs'],
        terminal_raw_replay_calls_across_attempts=sum(r.get('reused_old_raw_calls', 0) +
            r.get('reused_new_raw_calls', 0) for r in rows),
        historical_terminal_successful_calls=old_terminal + new,
        historical_target_successful_calls=old_target + new,
        historical_target_invocations_including_new_failures=old_target + new + failed,
        attempts=rows, attempt_statuses=dict(Counter(r['status'] for r in rows)),
        failed_attempts=[r for r in rows if r['status'] != 'completed'],
        inherited_failures=plan['inherited_failures'],
        inherited_accounts=plan['inherited_accounts'], registration_completion=plan['registration_completion'],
        preparation_completion=str((ROOT / read(ROOT / plan['parent_manifest'])['output_dir'] /
                                    'completion.json').relative_to(ROOT)),
        accounting_note='Each local attempt ledger is summed once, including registration and synthetic checks. '
            'Registration CPU is included, not added again. Inherited CPU is linked only. Synthetic fixture '
            'calls are not physical observations. Historical target calls include shared initial/search once; '
            'raw reuse affects logical consumption and CPU, not new physical cost. Failed simulator invocations '
            'are reported separately from returned observations. RNG draw counts are unknown for actual new '
            'observations. CPU excludes pre-meter imports, shell inspection, documentation and final serialization.')
    save(output / 'completion.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'verify', 'resume'])
    parser.add_argument('--manifest', default=str(MANIFEST))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--arm')
    args = parser.parse_args()
    plan = read(args.manifest)
    if args.action == 'run':
        result = run(plan)
    else:
        rows = read(ROOT / plan['block_inventory'])['rows']
        selected = [r for r in rows if r['seed'] == args.seed and r['arm'] == args.arm]
        if len(selected) != 1:
            parser.error('verify/resume requires exactly one registered --seed and --arm')
        result = verify_case(plan, selected[0], resume=args.action == 'resume')
    print(result['status'], flush=True)
