"""Frozen common-neighborhood pilot with unchanged sources and V69 verifier."""
import argparse
from datetime import datetime, timezone
import json
import time
from unittest.mock import patch

import numpy as np

from performance.mainline_legacy_backend import build_backend
from performance.mainline_shared_neighbors import make_shared_neighbors, shared_neighbor_candidates
from performance.run_mainline_legacy_backend_bridge import (
    ROOT, attempt, count_simulations, load_pickle, read, save, save_engine,
)
from performance.run_mainline_local_noise_control import (
    FULL, LOCAL, STOP, NEW, case_dir, cases, require_all, search as terminal_noise_search,
    verify, score as score_recommendations,
)

MANIFEST = ROOT/'performance/manifests/mainline_shared_neighbors_v1.json'
NATIVE_NEIGHBORS = 'legacy_native_neighbors'


def prepare(output, plan):
    for seed in plan['target_seeds']:
        directory = output/f'seed_{seed}'
        with attempt(directory, 'prepare-neighbors', plan['domain']) as cost:
            cost['seed'] = seed
            if (directory/'neighbors.json').exists():
                cost['reused'] = True
                continue
            old_seed = plan['initialization_seed_by_run'][str(seed)]
            parent = case_dir(ROOT/plan['parent_output'], old_seed, STOP)
            engine = load_pickle(parent/'search_engine.pkl')
            with patch.object(engine.problem, 'simulate', side_effect=AssertionError('Reuse initial raw')):
                neighbors = make_shared_neighbors(engine, seed, plan['neighbor_count'])
            assert neighbors['metadata']['radius'] == plan['neighbor_radius']
            assert neighbors['metadata']['safe_margin_scale'] == plan['neighbor_safe_margin_scale']
            np.savez_compressed(directory/'neighbors.npz', points=np.asarray(neighbors['points'], dtype=np.int16))
            save(directory/'neighbors.json', {**neighbors['metadata'], 'initialization_seed': old_seed,
                 'initialization_file': str((parent/'search.json').relative_to(ROOT)),
                 'frozen_utc': datetime.now(timezone.utc).isoformat()})


def search(output, plan, seed, arm):
    if arm == LOCAL:
        # The full path and query scores are physically shared; only terminal
        # noise and the subsequent independent verification differ.
        return terminal_noise_search(output, plan, seed, arm)
    directory = case_dir(output, seed, arm)
    observed = []
    with attempt(directory, 'search', plan['domain'], arm) as cost:
        cost.update(seed=seed, new_target_initial_calls=0, reused_initial_raw_rows=10)
        if (directory/'search.json').exists():
            cost['reused'] = True
            return
        # All three supplies must be frozen before any new target search.
        for registered in plan['target_seeds']:
            read(output/f'seed_{registered}'/'neighbors.json')
        old_seed = plan['initialization_seed_by_run'][str(seed)]
        initial = read(case_dir(ROOT/plan['parent_output'], old_seed, STOP)/'search.json')['history']
        prior = load_pickle(ROOT/plan['frozen_assets']['native_source_prior_file'])
        bundle = load_pickle(ROOT/plan['frozen_assets']['source_models_file']) if arm in NEW else None
        backend = 'legacy_native_v69' if arm == NATIVE_NEIGHBORS else arm
        engine = build_backend(plan['domain'], backend, bundle, source_prior=prior,
                               seed=seed, N=13, initial_rows=initial)
        points = (np.load(output/f'seed_{seed}'/'neighbors.npz')['points']
                  if arm in plan['augmented_arms'] else [])
        try:
            with shared_neighbor_candidates(engine, points) as audit:
                with count_simulations(engine, cost, 'new_target_search_calls', observed):
                    engine.run_search()
            # Keep paid raw even if subsequent audit or engine saving fails.
            save(directory/'paid_search_raw.json', observed)
        except BaseException:
            save(directory/f'failed_search_raw_{time.time_ns()}.json', observed)
            raise
        assert cost['new_target_search_calls'] == (0 if arm == STOP else 3)
        history = [{'point': list(point), 'raw': np.asarray(raw).tolist()} for point, raw in engine.history]
        assert history[:10] == initial
        assert [s['paid_prefix'] for s in audit['stages']] == ([] if arm == STOP else [10, 11, 12])
        if arm != STOP:
            for stage, original in zip(audit['stages'], engine.iteration_log, strict=True):
                point = list(audit['points'][stage['selected_id']])
                assert point == original['x_selected'] == history[stage['paid_prefix']]['point']
                selected = stage['selected_index']
                assert selected == int(np.argmax(stage['score']['total']))
                assert float(stage['score']['total'][selected]) == original['score_selected']
                assert set(audit['shared_ids']).issubset(stage['action_ids'])
            np.savez_compressed(directory/'query_policies.npz', points=np.asarray(audit.pop('points'), dtype=np.int16))
            save(directory/'query_audit.json', audit)
        result = {'seed': seed, 'arm': arm, 'backend_arm': backend, 'family': plan['domain'],
            **engine.frozen_terminal, 'history': history, 'iterations': engine.iteration_log,
            'backend_trace': getattr(engine, 'backend_trace', []), 'logical_search_calls': len(history),
            'new_initial_calls': 0, 'new_search_calls': cost['new_target_search_calls'],
            'coupled_query_calls_reused': 0, 'initial_design_seed': 80,
            'initialization_seed': old_seed, 'shared_neighbor_count': len(points), 'truth_used': False,
            'frozen_utc': datetime.now(timezone.utc).isoformat()}
        save_engine(directory/'search_engine.pkl', engine)
        save(directory/'search.json', result)


def score(output, plan):
    require_all(output, plan, 'verification.json')
    if not (output/'truth.json').exists():
        from performance.diagnose_mainline_shared_neighbors import load_truth_cache
        cache = load_truth_cache(plan, output)
        requested = set()
        for seed, arm in cases(plan):
            verified = read(case_dir(output, seed, arm)/'verification.json')
            requested.update(tuple(verified[key]) for key in ('primary', 'deployed'))
        cached = [{'point': list(point), 'truth': cache[point]} for point in sorted(requested)
                  if point in cache and 'constraint_sigma' in cache[point]]
        save(output/'truth.json', cached)
    score_recommendations(output, plan)


def complete(output, plan):
    require_all(output, plan, 'verification.json')
    results, diagnostic = read(output/'results.json'), read(output/'diagnosis.json')
    assert diagnostic['status'] == 'complete'
    attempts = [json.loads(line) for path in sorted(output.rglob('*attempts.jsonl'))
                for line in path.read_text().splitlines() if line.strip()]
    fields = ('cpu_seconds', 'wall_seconds', 'new_source_calls', 'new_target_initial_calls',
              'new_target_search_calls', 'new_verification_calls', 'new_source_fits', 'download_bytes')
    costs = {key: sum(row.get(key, 0) for row in attempts) for key in fields}
    costs['truth_function_calls'] = sum(row.get('truth_function_calls', 0)
        + row.get('new_truth_function_calls', 0) for row in attempts)
    save(output/'completion.json', {'status': 'complete', 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'recommendations': len(results['rows']), 'attempt_count': len(attempts),
        'failed_attempts': [row for row in attempts if row['status'] == 'failed'], 'cost': costs,
        'new_physical_target_calls': sum(costs[k] for k in (
            'new_target_initial_calls', 'new_target_search_calls', 'new_verification_calls')),
        'logical_target_calls': sum(row['logical_total_calls'] for row in results['rows']),
        'reused_historical_initial_raw_rows': 30,
        'wall_time_definition': 'Sum of attempt wall seconds, with concurrent processes; not elapsed wall time.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'search', 'verify', 'score', 'complete'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--arm')
    args = parser.parse_args()
    plan = read(MANIFEST)
    output = ROOT/plan['output_dir']
    if args.stage in ('search', 'verify'):
        if (args.seed, args.arm) not in cases(plan):
            parser.error('Use a registered seed and arm')
        (search if args.stage == 'search' else verify)(output, plan, args.seed, args.arm)
    else:
        {'prepare': prepare, 'score': score, 'complete': complete}[args.stage](output, plan)
    print(args.stage, args.seed, args.arm, 'complete', flush=True)


if __name__ == '__main__':
    main()
