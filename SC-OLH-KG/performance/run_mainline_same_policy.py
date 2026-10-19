"""Frozen distinct-challenger terminal experiment, reusing exact paid blocks."""
import argparse
import copy
from datetime import datetime, timezone
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core import terminal_verification as terminal
from performance import benchmark_lodo_meta_prior as benchmark
from performance import run_mainline_terminal_equal as verifier
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_terminal_equal import run_verifier

MANIFEST = ROOT / 'performance/manifests/mainline_same_policy_challenge_v1.json'


def cases(plan):
    source = read(ROOT / plan['source_manifest'])
    for family in source['families']:
        cells = read(ROOT / family['terminal_manifest'])
        for seed, arm, protocol in verifier.cases(cells):
            tail = f"seed_{seed}/{arm}/{protocol}"
            yield (ROOT / cells['output_dir'] / tail,
                   ROOT / plan['output_dir'] / family['domain'] / tail)


def transform(item):
    result = copy.deepcopy(item)
    rows = result['frozen_shortlist']
    incumbent = result['objective_incumbent_position'] - 1
    changed = rows[0]['point'] == rows[incumbent]['point']
    if changed:
        assert incumbent == 0 and len(rows) == 3
        challenger = min(range(1, 3), key=lambda i: rows[i]['posterior_rank'])
        order = [challenger, 0, 3 - challenger]
        rows = [rows[i] for i in order]
        for position, row in enumerate(rows, 1):
            row['source_shortlist_role'] = row['shortlist_role']
            row['shortlist_position'] = position
            row['shortlist_role'] = ('distinct_posterior_challenger',
                                    'initial_objective_incumbent', 'original_remaining')[position - 1]
        result.update(frozen_shortlist=rows, primary=rows[0]['point'],
                      objective_incumbent_position=2)
    result.update(protocol='same_policy_distinct_challenger', input_changed=changed)
    if result.get('terminal_recording') is not None:
        result['terminal_recording']['stages'].append(dict(
            phase='same_policy_transform', changed=changed,
            before=copy.deepcopy(item['frozen_shortlist']),
            after=copy.deepcopy(result['frozen_shortlist']),
            incumbent_position_before=item['objective_incumbent_position'],
            incumbent_position_after=result['objective_incumbent_position']))
    return result


def context(item):
    # Direct values, not digests. Search count and row labels only affect reporting.
    return (item['domain'], json.dumps(item['task'], sort_keys=True),
            json.dumps(item['geometry'], sort_keys=True), item['verification_seed'],
            item['tau'], item['method'], item['mean_delta_fraction'],
            item['familywise_delta'], len(item['frozen_shortlist']))


def safety_key(item, index):
    return context(item) + ('safety', index,
        tuple(item['frozen_shortlist'][index]['point']), item['candidate_budgets'][index])


def objective_key(item):
    rows = item['frozen_shortlist']
    return context(item) + ('objective', tuple(rows[0]['point']),
        tuple(rows[item['objective_incumbent_position'] - 1]['point']),
        item['objective_comparison_budget'], item['objective_comparison_delta'])


def ordered_key(item):
    return context(item) + (tuple(tuple(r['point']) for r in item['frozen_shortlist']),
        tuple(item['candidate_budgets']), item['objective_incumbent_position'],
        item['objective_comparison_budget'], item['objective_comparison_delta'])


def prepare(plan):
    output = ROOT / plan['output_dir']
    with attempt(output, 'freeze-same-policy-inputs') as cost:
        cost.update(new_source_fits=0, new_truth_function_calls=0, download_bytes=0)
        originals = [(old, new, read(old / 'input.json')) for old, new in cases(plan)]
        matches = {}
        for old, _, item in originals:
            matches.setdefault(ordered_key(item), str(old.relative_to(ROOT)))
        coverage = []
        for old, new, item in originals:
            changed = transform(item)
            path = new / 'input.json'
            if path.exists():
                assert read(path) == changed, 'Frozen input changed'
            else:
                save(path, changed)
            coverage.append(dict(domain=item['domain'], seed=item['seed'], arm=item['arm'],
                input_changed=changed['input_changed'], baseline_directory=str(old.relative_to(ROOT)),
                directory=str(new.relative_to(ROOT)),
                complete_old_input_match=matches.get(ordered_key(changed))))
        save(output / 'prepared.json', dict(status='complete', cases=coverage,
            case_count=len(coverage), changed=sum(r['input_changed'] for r in coverage),
            complete_old_input_matches=sum(r['complete_old_input_match'] is not None for r in coverage),
            truth_used=False, frozen_utc=datetime.now(timezone.utc).isoformat()))


def load_blocks(plan):
    """Split only the actual logged safety blocks and paired objective tails."""
    output = ROOT / plan['output_dir']
    pool = {}
    with attempt(output, 'index-existing-paid-terminal-blocks') as cost:
        cost.update(new_source_fits=0, new_truth_function_calls=0, download_bytes=0)
        for old, _ in cases(plan):
            item = read(old / 'input.json')
            result = read(old / 'verification.json')['verification']
            raw = read(old / 'paid_verification_raw.json')
            cursor = 0

            def add(key, count, points):
                nonlocal cursor
                chunk = raw[cursor:cursor + count]
                assert len(chunk) == count
                assert all(row['point'] == points[i % len(points)] for i, row in enumerate(chunk))
                values = [row['raw'] for row in chunk]
                if key in pool:
                    assert pool[key]['raw'] == values, 'Same stream has inconsistent existing raw'
                else:
                    pool[key] = dict(raw=values, status='complete', origin_stage=plan.get('source_stage_label','stage47'),
                        origin=str((old / 'paid_verification_raw.json').relative_to(ROOT)),
                        start=cursor, count=count)
                cursor += count

            for row in result['attempts']:
                index = row['candidate_index']
                assert row['verification_budget'] == item['candidate_budgets'][index]
                add(safety_key(item, index), row['verification_budget'],
                    [item['frozen_shortlist'][index]['point']])
            count = result['objective_comparison_budget']
            if count:
                assert count == 2 * item['objective_comparison_budget']
                add(objective_key(item), count, [item['frozen_shortlist'][0]['point'],
                    item['frozen_shortlist'][item['objective_incumbent_position'] - 1]['point']])
            assert cursor == len(raw) == result['verification_budget']
        for path in sorted((output / 'new_blocks').glob('*.json')):
            block = read(path)
            key = tuple(tuple(v) if isinstance(v, list) else v for v in block['key'])
            pool[key] = dict(raw=block['raw'], status=block['status'], origin_stage=plan.get('stage_label','stage49'),
                origin=str(path.relative_to(ROOT)), count=len(block['raw']))
        save(output / 'block_index.json', dict(existing_safety_blocks=sum(k[9] == 'safety' for k in pool),
            existing_objective_blocks=sum(k[9] == 'objective' for k in pool), truth_used=False))
    return pool


def verify_case(plan, directory, pool, baseline):
    if (directory / 'verification.json').exists():
        return
    output = ROOT / plan['output_dir']
    item = read(directory / 'input.json')
    with attempt(directory, 'verify-frozen-distinct-challenger', item['domain'], item['arm']) as cost:
        cost.update(new_source_fits=0, new_truth_function_calls=0, download_bytes=0,
                    reused_raw_calls=0, new_target_initial_calls=0)
        problem = SimpleNamespace(alpha=item['task']['alpha'], tau=item['tau'],
                                  simulation_noise_model='iid_gaussian')
        actual = None
        used = []

        def block_run(key, points, expected, operation):
            nonlocal actual
            cached = pool.get(key)
            if cached is None:
                path = output / 'new_blocks' / f"{len(list((output / 'new_blocks').glob('*.json'))):04d}.json"
                cached = dict(raw=[], status='partial', origin_stage=plan.get('stage_label','stage49'),
                              origin=str(path.relative_to(ROOT)), count=0)
                pool[key] = cached
            cursor, newly_paid = 0, False

            def simulate(point, rng=None):
                nonlocal cursor, newly_paid, actual
                assert list(point) == points[cursor % len(points)]
                if cursor < len(cached['raw']):
                    value = cached['raw'][cursor]
                    cost['reused_raw_calls'] += 1
                else:
                    assert cached['status'] == 'partial', 'Completed paid block unexpectedly exhausted'
                    if actual is None:
                        task, kwargs = item['task'], dict(item['geometry'])
                        if item['domain'] == 'FactorShockStatePolicyRZDT1':
                            kwargs['shared_shock_scale'] = task['shared_shock_scale']
                        actual = benchmark.build_scalarized_problem(item['domain'], task['d'], task['L'],
                            task['sigma'], task['alpha'], tuple(task['weights']), problem_kwargs=kwargs)
                        assert actual.d == 1000 and actual.tau == item['tau']
                    value = np.asarray(actual.simulate(point, rng), dtype=float).tolist()
                    cached['raw'].append(value)
                    newly_paid = True
                    cost['new_verification_calls'] += 1
                cursor += 1
                return np.asarray(value, dtype=float)

            proxy = SimpleNamespace(**vars(problem), simulate=simulate)
            try:
                result = operation(proxy)
                assert cursor == len(cached['raw']) == expected
                cached['status'] = 'complete'
                return result
            finally:
                if cached['origin_stage'] == plan.get('stage_label','stage49') and (newly_paid or cached['status'] == 'partial'):
                    save(ROOT / cached['origin'], dict(key=key, raw=cached['raw'], status=cached['status']))
                used.append(dict(kind=key[9], origin=cached['origin'], start=cached.get('start', 0),
                                 count=cursor, origin_stage=cached['origin_stage']))

        original_safety = terminal.verify_frozen_policy
        original_objective = terminal.verify_paired_objective_dominance

        def safety(problem_arg, point, **kwargs):
            index = kwargs['candidate_index']
            return block_run(safety_key(item, index), [list(point)], kwargs['verification_budget'],
                lambda proxy: original_safety(proxy, point, **kwargs))

        def objective(problem_arg, challenger, incumbent, **kwargs):
            return block_run(objective_key(item), [list(challenger), list(incumbent)],
                2 * kwargs['comparison_budget'],
                lambda proxy: original_objective(proxy, challenger, incumbent, **kwargs))

        try:
            with patch.object(terminal, 'verify_frozen_policy', safety), \
                 patch.object(terminal, 'verify_paired_objective_dominance', objective):
                deployed, result = run_verifier(problem, item)
            assert result['verification_budget'] == cost['new_verification_calls'] + cost['reused_raw_calls']
            assert result['max_verification_budget'] == 352
            assert result['total_evaluation_count'] <= (362 if item['arm'] == 'initialization_only' else 365)
            if not item['input_changed']:
                original = read(baseline / 'verification.json')
                assert list(deployed) == original['deployed'] and result == original['verification'], \
                    'New raw-block adapter changed an unchanged terminal'
            save(directory / 'verification.json', dict(seed=item['seed'], arm=item['arm'],
                verification_seed=item['verification_seed'], input_changed=item['input_changed'],
                primary=item['primary'], deployed=list(map(int, deployed)), verification=result,
                new_verification_calls=cost['new_verification_calls'], reused_raw_calls=cost['reused_raw_calls'],
                logical_verification_calls=result['verification_budget'], truth_used=False,
                completed_utc=datetime.now(timezone.utc).isoformat()))
        finally:
            save(directory / 'verification_blocks.json', used)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'verify'])
    args = parser.parse_args()
    plan = read(MANIFEST)
    if args.stage == 'prepare':
        prepare(plan)
    else:
        prepared = read(ROOT / plan['output_dir'] / 'prepared.json')
        assert prepared['case_count'] == 108 and prepared['status'] == 'complete'
        pool = load_blocks(plan)
        for n, (baseline, directory) in enumerate(cases(plan), 1):
            verify_case(plan, directory, pool, baseline)
            print(f'verified {n}/108 {directory.parent.name}', flush=True)
        save(ROOT / plan['output_dir'] / 'verification_complete.json', dict(status='complete', cases=108,
            completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
