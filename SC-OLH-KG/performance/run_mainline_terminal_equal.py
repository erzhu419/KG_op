"""Frozen-search comparison of original and equal terminal safety budgets."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.terminal_verification import verify_frozen_shortlist
from performance import benchmark_lodo_meta_prior as benchmark
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = ROOT / 'performance/manifests/mainline_terminal_equal_budget_v1.json'


def cases(plan):
    names = [arm['name'] if isinstance(arm, dict) else arm for arm in plan['arms']]
    failed = {(row['seed'], row['arm']) for row in plan.get('execution_failures', [])}
    return [(seed, arm, protocol) for seed in plan['target_seeds']
            for arm in names for protocol in plan['protocols'] if (seed, arm) not in failed]


def case_dir(output, seed, arm, protocol):
    return output / f'seed_{seed}' / arm / protocol


def require_all(output, plan, filename):
    missing = [str(case_dir(output, *case) / filename) for case in cases(plan)
               if not (case_dir(output, *case) / filename).is_file()]
    if missing:
        raise RuntimeError(f'All frozen cases must have {filename}: {missing}')


def prepare(output, plan):
    source_plan = read(ROOT / plan['source_manifest'])
    source = ROOT / plan['source_output_dir']
    with attempt(output, 'prepare-terminal-inputs', source_plan['domain']) as cost:
        cost.update(new_source_fits=0, download_bytes=0, new_truth_function_calls=0)
        for seed, arm, protocol in cases(plan):
            old = source / f'seed_{seed}' / arm
            search, saved = read(old / 'search.json'), read(old / 'verification.json')
            terminal = saved['verification']
            shortlist = terminal['frozen_shortlist']
            assert [r['point'] for r in shortlist] == [r['point'] for r in search['terminal']['frozen_shortlist']]
            assert terminal['candidate_verification_budgets'] == plan['protocols']['original']
            assert len(shortlist) == 3 and terminal['objective_incumbent_guard_enabled']
            budgets = plan['protocols'][protocol]
            guard = terminal['objective_comparison_budget_per_policy']
            assert sum(budgets) + 2 * guard == 352
            first = terminal['attempts'][0]
            item = dict(seed=seed, arm=arm, protocol=protocol,
                domain=source_plan['domain'], task=source_plan['task'],
                geometry=source_plan['geometry_by_seed'][str(seed)]['problem_kwargs'],
                verification_seed=int(plan['verification_seed_by_task'][str(seed)]),
                primary=shortlist[0]['point'], initial=search['history'][0]['point'],
                frozen_shortlist=shortlist, candidate_budgets=budgets,
                search_evaluation_count=search['logical_search_calls'],
                familywise_delta=terminal['familywise_delta'],
                method=first['method_mode'], mean_delta_fraction=terminal['mean_delta_fraction'],
                shortlist_mode=terminal['shortlist_mode'],
                objective_incumbent_position=terminal['objective_incumbent_position'],
                objective_comparison_budget=guard,
                objective_comparison_delta=terminal['objective_comparison']['delta'],
                tau=first['tau'], source_case=str(old.relative_to(ROOT)), truth_used=False)
            path = case_dir(output, seed, arm, protocol) / 'input.json'
            if path.exists():
                assert read(path) == item, 'Prepared terminal input changed'
            else:
                save(path, item)
        save(output / 'prepared.json', dict(case_count=len(cases(plan)), truth_used=False,
            new_simulation_calls=0, frozen_utc=datetime.now(timezone.utc).isoformat()))


def run_verifier(problem, item, *, seed=None):
    return verify_frozen_shortlist(problem, item['frozen_shortlist'],
        seed=item['verification_seed'] if seed is None else seed,
        **{key: item[key] for key in (
            'search_evaluation_count', 'candidate_budgets', 'familywise_delta',
            'method', 'mean_delta_fraction', 'shortlist_mode',
            'objective_incumbent_position', 'objective_comparison_budget',
            'objective_comparison_delta')})


def replay(output, plan):
    """Detect differences between the native and shared verifier on paid raws."""
    require_all(output, plan, 'input.json')
    rows = []
    with attempt(output, 'replay-native-verifier') as cost:
        cost.update(new_source_fits=0, download_bytes=0, new_truth_function_calls=0,
                    replayed_raw_observations=0)
        for seed, arm, protocol in cases(plan):
            if protocol != 'original':
                continue
            item = read(case_dir(output, seed, arm, protocol) / 'input.json')
            source = ROOT / item['source_case']
            saved, raw = read(source / 'verification.json'), read(source / 'paid_verification_raw.json')
            position = 0

            def simulate(point, rng=None):
                nonlocal position
                row = raw[position]
                assert list(point) == row['point'], 'Verifier requested a different policy/order'
                position += 1
                return np.asarray(row['raw'], dtype=float)

            problem = SimpleNamespace(alpha=item['task']['alpha'], tau=item['tau'],
                                      simulation_noise_model='iid_gaussian', simulate=simulate)
            deployed, result = run_verifier(problem, item, seed=seed)
            expected = saved['verification']
            assert list(deployed) == saved['deployed'] and position == len(raw)
            for key in ('certified', 'selected_shortlist_rank', 'verification_budget',
                        'safety_verification_budget', 'objective_comparison_budget',
                        'candidate_verification_budgets', 'familywise_delta',
                        'per_candidate_delta', 'attempts', 'objective_comparison'):
                assert result[key] == expected[key], f'Native/shared verifier mismatch: {key}'
            rows.append(dict(seed=seed, arm=arm, passed=True, replayed_raw_count=position,
                             branch=result['objective_comparison']['status']))
            cost['replayed_raw_observations'] += position
        save(output / 'replay_check.json', dict(status='passed', cases=len(rows), rows=rows,
            new_simulation_calls=0, new_truth_function_calls=0))


def verify(output, plan, seed, arm, protocol, *, runner=None):
    directory = case_dir(output, seed, arm, protocol)
    observed = []
    with attempt(directory, 'verify', arm=arm) as cost:
        cost.update(seed=seed, protocol=protocol, new_source_fits=0, download_bytes=0,
                    new_truth_function_calls=0)
        if (directory / 'verification.json').is_file():
            cost['reused'] = True
            return
        require_all(output, plan, 'input.json')
        assert read(output / 'replay_check.json')['status'] == 'passed'
        for path in directory.glob('failed_verification_raw_*.json'):
            if read(path):
                raise RuntimeError('Recover saved paid verification before repeating a partially consumed stage')
        item = read(directory / 'input.json')
        task = item['task']
        kwargs = dict(item['geometry'])
        if item['domain'] == 'FactorShockStatePolicyRZDT1':
            kwargs['shared_shock_scale'] = task['shared_shock_scale']
        problem = benchmark.build_scalarized_problem(item['domain'], task['d'], task['L'],
            task['sigma'], task['alpha'], tuple(task['weights']), problem_kwargs=kwargs)
        assert problem.d == 1000 and problem.tau == item['tau']
        np.testing.assert_array_equal(problem.base.task_geometry_shift, kwargs['task_geometry_shift'])
        assert problem.base.task_geometry_radius_scale == kwargs['task_geometry_radius_scale']
        original = problem.simulate

        def simulate(point, rng=None):
            value = original(point, rng)
            cost['new_verification_calls'] += 1
            observed.append(dict(point=list(map(int, point)), raw=np.asarray(value).tolist()))
            return value

        try:
            with patch.object(problem, 'simulate', simulate):
                deployed, result = (run_verifier if runner is None else runner)(problem, item)
            assert result['verification_budget'] == len(observed) == cost['new_verification_calls']
            assert result['max_verification_budget'] == 352
            assert len(observed) + item['search_evaluation_count'] <= (362 if arm == 'initialization_only' else 365)
            save(directory / 'paid_verification_raw.json', observed)
            save(directory / 'verification.json', dict(seed=seed, arm=arm, protocol=protocol,
                verification_seed=item['verification_seed'], primary=item['primary'],
                deployed=list(map(int, deployed)), verification=result,
                new_verification_calls=len(observed), truth_used=False,
                completed_utc=datetime.now(timezone.utc).isoformat()))
        except BaseException:
            save(directory / f'failed_verification_raw_{time.time_ns()}.json', observed)
            raise


def summarize(output, plan):
    # Truth may only be opened once every registered paid verifier has finished.
    require_all(output, plan, 'verification.json')
    rows, caches = [], {}
    with attempt(output, 'summarize-terminal-comparison') as cost:
        cost.update(new_source_fits=0, download_bytes=0, new_truth_function_calls=0)
        for seed in plan['target_seeds']:
            path = ROOT / plan['source_output_dir'] / f'seed_{seed}' / 'truth.json'
            caches[seed] = {tuple(row['point']): row['truth'] for row in read(path)}
        for seed, arm, protocol in cases(plan):
            directory = case_dir(output, seed, arm, protocol)
            item, saved = read(directory / 'input.json'), read(directory / 'verification.json')
            result = saved['verification']
            truth = caches[seed].get(tuple(saved['deployed']))
            initial = caches[seed][tuple(item['initial'])]
            certified = result['certified']
            rows.append(dict(seed=seed, arm=arm, protocol=protocol,
                verification_seed=saved['verification_seed'], deployed=saved['deployed'],
                deployed_truth=truth, certified=certified,
                false_certificate=None if truth is None else bool(certified and not truth['true_feasible']),
                actual_safe_reward=None if truth is None else (truth['quality'] if certified and truth['true_feasible'] else 0.),
                returned_policy_changed=saved['deployed'] != item['initial'],
                certified_deployment_changed=bool(certified and saved['deployed'] != item['initial']),
                deployed_quality_gain=None if truth is None else truth['quality']-initial['quality'],
                deployed_feasible_improving=None if truth is None else bool(certified and truth['true_feasible'] and truth['objective'] < initial['objective']-1e-12),
                selected_shortlist_rank=result['selected_shortlist_rank'],
                branch=result['objective_comparison']['status'],
                new_verification_calls=saved['new_verification_calls'],
                logical_total_calls=item['search_evaluation_count']+saved['new_verification_calls']))
        groups, paired = [], []
        names = list(dict.fromkeys(row['arm'] for row in rows))
        for arm in names:
            for protocol in plan['protocols']:
                group = [row for row in rows if row['arm'] == arm and row['protocol'] == protocol]
                missing = sum(row['deployed_truth'] is None for row in group)
                groups.append(dict(arm=arm, protocol=protocol, task_count=len(group),
                    certified=sum(row['certified'] for row in group),
                    false_certificates=sum(row['false_certificate'] is True for row in group),
                    truth_cache_missing=missing,
                    mean_safe_reward=None if missing else float(np.mean([row['actual_safe_reward'] for row in group])),
                    feasible_improving_deployments=sum(row['deployed_feasible_improving'] is True for row in group),
                    returned_policy_changes=sum(row['returned_policy_changed'] for row in group),
                    certified_deployment_changes=sum(row['certified_deployment_changed'] for row in group),
                    verification_calls=sum(row['new_verification_calls'] for row in group),
                    branches=dict(Counter(row['branch'] for row in group))))
            for seed in plan['target_seeds']:
                pair = {row['protocol']: row for row in rows if row['seed'] == seed and row['arm'] == arm}
                original, equal = pair['original'], pair['equal']
                known = all(row['actual_safe_reward'] is not None for row in pair.values())
                paired.append(dict(seed=seed, arm=arm,
                    safe_reward_difference=None if not known else equal['actual_safe_reward']-original['actual_safe_reward'],
                    verification_calls_difference=equal['new_verification_calls']-original['new_verification_calls'],
                    returned_policy_changed=equal['deployed'] != original['deployed'],
                    certified_deployment_changed=(
                        (equal['deployed'] if equal['certified'] else None) !=
                        (original['deployed'] if original['certified'] else None))))
        save(output / 'results.json', dict(status='complete', rows=rows, arms=groups, paired=paired,
            truth_cache_complete=all(row['deployed_truth'] is not None for row in rows),
            scope='Paired new terminal streams on 18 fixed stage29 searches and shortlists; same three task geometries, no new searches. Safety familywise delta and independent objective delta are separate guarantees.'))
    attempts = [json.loads(line) for path in output.rglob('attempts.jsonl') for line in path.read_text().splitlines()]
    fields = ('cpu_seconds', 'wall_seconds', 'new_source_calls', 'new_target_search_calls',
              'new_verification_calls', 'new_source_fits', 'download_bytes', 'new_truth_function_calls')
    totals = {key: sum(row.get(key, 0) for row in attempts) for key in fields}
    save(output / 'completion.json', dict(status='complete', recommendations=len(rows),
        cost=totals, attempt_count=len(attempts), failed_attempts=[row for row in attempts if row['status'] == 'failed'],
        new_physical_target_calls=totals['new_verification_calls'],
        logical_target_calls=sum(row['logical_total_calls'] for row in rows),
        completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'replay', 'verify', 'summarize'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--arm')
    parser.add_argument('--protocol', choices=['original', 'equal'])
    args = parser.parse_args()
    plan = read(MANIFEST)
    output = ROOT / plan['output_dir']
    if args.stage == 'verify':
        assert (args.seed, args.arm, args.protocol) in cases(plan)
        verify(output, plan, args.seed, args.arm, args.protocol)
    else:
        {'prepare': prepare, 'replay': replay, 'summarize': summarize}[args.stage](output, plan)
    print(args.stage, args.seed, args.arm, args.protocol, 'complete', flush=True)
