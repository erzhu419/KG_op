"""Fresh-stream terminal-noise control with frozen sources and native V69 search."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from scipy.stats import norm

from performance.mainline_legacy_backend import build_backend
from performance.mainline_legacy_native import SearchFrozen
from performance.run_mainline_legacy_backend_bridge import (
    ROOT, attempt, count_simulations, load_pickle, read, save, save_engine,
)

MANIFEST = ROOT/'performance/manifests/mainline_local_noise_control_v1.json'
LOCAL = 'current_structural_local_noise'
FULL = 'current_structural_mixture'
STOP = 'initialization_only'
NEW = {'same_source_gp', FULL, LOCAL}


def case_dir(output, seed, arm):
    return output/f'seed_{seed}'/arm


def cases(plan):
    return [(seed, arm['name']) for seed in plan['target_seeds'] for arm in plan['arms']]


def require_all(output, plan, filename):
    missing = [str(case_dir(output, seed, arm)/filename) for seed, arm in cases(plan)
               if not (case_dir(output, seed, arm)/filename).is_file()]
    if missing:
        raise RuntimeError(f'All {len(cases(plan))} prior stages must complete: {missing}')


def load_engine(output, plan, seed, arm):
    engine = load_pickle(case_dir(output, seed, arm)/'search_engine.pkl')
    if arm in NEW:
        engine.backend_bundle = load_pickle(ROOT/plan['frozen_assets']['source_models_file'])
    return engine


def search(output, plan, seed, arm):
    directory = case_dir(output, seed, arm)
    observations = []
    with attempt(directory, 'search', plan['domain'], arm) as cost:
        cost.update(seed=seed, new_target_initial_calls=0, coupled_query_calls_reused=0)
        if (directory/'search.json').exists():
            cost['reused'] = True
            return
        if arm == LOCAL:
            base = read(case_dir(output, seed, FULL)/'search.json')
            engine = load_engine(output, plan, seed, FULL)
            engine.backend_arm = LOCAL
            engine.frozen_terminal = None
            pool = list(engine._last_terminal_pool) or list(engine.observations)
            with patch.object(engine.problem, 'simulate', side_effect=AssertionError('Paired search is reused')):
                primary, _ = engine._decision_backend_terminal_recommendation(pool)
                try:
                    engine._terminal_verification_protocol(primary)
                except SearchFrozen:
                    pass
            if engine.frozen_terminal is None:
                raise AssertionError('Local terminal shortlist did not freeze')
            cost['coupled_query_calls_reused'] = 3
        else:
            prior = load_pickle(ROOT/plan['frozen_assets']['native_source_prior_file'])
            bundle = load_pickle(ROOT/plan['frozen_assets']['source_models_file']) if arm in NEW else None
            initial = [] if arm == STOP else read(case_dir(output, seed, STOP)/'search.json')['history']
            engine = build_backend(plan['domain'], arm, bundle, source_prior=prior,
                                   seed=seed, N=plan['anchor']['search_calls'], initial_rows=initial)
            field = 'new_target_initial_calls' if arm == STOP else 'new_target_search_calls'
            try:
                with count_simulations(engine, cost, field, observations):
                    engine.run_search()
            except BaseException:
                save(directory/f'failed_search_raw_{time.time_ns()}.json', observations)
                raise
            expected = 10 if arm == STOP else 3
            if cost[field] != expected:
                raise AssertionError('Physical initialization/search calls differ from protocol')
        history = [{'point': list(point), 'raw': np.asarray(raw).tolist()} for point, raw in engine.history]
        if arm != STOP:
            shared = read(case_dir(output, seed, STOP)/'search.json')['history']
            if history[:10] != shared:
                raise AssertionError('Fresh initialization differs between arms')
        if arm == LOCAL and (history != base['history'] or engine.backend_trace != base['backend_trace']):
            raise AssertionError('Local-noise control changed paired search history')
        result = {'seed': seed, 'arm': arm, 'family': plan['domain'], **engine.frozen_terminal,
            'history': history, 'iterations': engine.iteration_log,
            'backend_trace': getattr(engine, 'backend_trace', []),
            'logical_search_calls': len(history), 'new_initial_calls': cost['new_target_initial_calls'],
            'new_search_calls': cost['new_target_search_calls'],
            'coupled_query_calls_reused': cost['coupled_query_calls_reused'],
            'initial_design_seed': 80, 'truth_used': False,
            'frozen_utc': datetime.now(timezone.utc).isoformat()}
        save_engine(directory/'search_engine.pkl', engine)
        save(directory/'search.json', result)


def verify(output, plan, seed, arm):
    directory = case_dir(output, seed, arm)
    observations = []
    with attempt(directory, 'verify', plan['domain'], arm) as cost:
        cost['seed'] = seed
        if (directory/'verification.json').exists():
            cost['reused'] = True
            return
        require_all(output, plan, 'search.json')
        search_result = read(directory/'search.json')
        engine = load_engine(output, plan, seed, arm)
        engine.defer_terminal = False
        try:
            with count_simulations(engine, cost, 'new_verification_calls', observations):
                deployed, verified = engine._terminal_verification_protocol(tuple(search_result['primary']))
        except BaseException:
            save(directory/f'failed_verification_raw_{time.time_ns()}.json', observations)
            raise
        if [r['point'] for r in verified['frozen_shortlist']] != [r['point'] for r in search_result['terminal']['frozen_shortlist']]:
            raise AssertionError('Shortlist changed after freeze')
        if cost['new_verification_calls'] != verified['verification_budget']:
            raise AssertionError('Physical and native verifier ledgers disagree')
        if len(engine.history)+cost['new_verification_calls'] > (362 if arm == STOP else 365):
            raise AssertionError('Logical budget exceeded')
        save(directory/'verification.json', {'seed': seed, 'arm': arm,
            'primary': search_result['primary'], 'deployed': list(map(int, deployed)),
            'verification': verified, 'new_verification_calls': cost['new_verification_calls'],
            'truth_used': False, 'completed_utc': datetime.now(timezone.utc).isoformat()})


def score(output, plan):
    with attempt(output, 'score', plan['domain']) as cost:
        cost['truth_function_calls'] = 0
        if (output/'results.json').exists():
            cost['reused'] = True
            return
        require_all(output, plan, 'verification.json')
        truth_path = output/'truth.json'
        cached = read(truth_path) if truth_path.exists() else []
        truth = {tuple(r['point']): r['truth'] for r in cached}
        engine = load_engine(output, plan, plan['target_seeds'][0], STOP)
        problem = engine.problem
        reference = float(np.dot(problem.weights, problem.ref_point))

        def get_truth(point):
            key = tuple(point)
            if key not in truth:
                f = float(problem.true_objective(key)); cost['truth_function_calls'] += 1
                g = float(problem.true_constraint_mean(key))-problem.tau; cost['truth_function_calls'] += 1
                sigma = float(problem.true_sigma(key)[1]); cost['truth_function_calls'] += 1
                margin = g+norm.ppf(1-problem.alpha)*sigma
                truth[key] = {'objective': f, 'constraint_mean_centered': g, 'constraint_sigma': sigma,
                    'chance_margin': float(margin), 'true_feasible': bool(margin <= 0),
                    'quality': float(np.clip(1-f/reference, 0, 1))}
                save(truth_path, [{'point': list(k), 'truth': v} for k, v in truth.items()])
            return truth[key]

        rows = []
        for seed, arm in cases(plan):
            directory = case_dir(output, seed, arm)
            s, v = read(directory/'search.json'), read(directory/'verification.json')
            primary, deployed = get_truth(v['primary']), get_truth(v['deployed'])
            certified = bool(v['verification']['certified'])
            rows.append({'seed': seed, 'arm': arm, 'primary': v['primary'], 'deployed': v['deployed'],
                'primary_truth': primary, 'deployed_truth': deployed, 'certified': certified,
                'false_certificate': bool(certified and not deployed['true_feasible']),
                'actual_safe_reward': deployed['quality'] if certified and deployed['true_feasible'] else 0.,
                'new_initial_calls': s['new_initial_calls'], 'new_search_calls': s['new_search_calls'],
                'new_verification_calls': v['new_verification_calls'],
                'logical_total_calls': s['logical_search_calls']+v['new_verification_calls'],
                'selected_shortlist_rank': v['verification']['selected_shortlist_rank']})
        arms = []
        for spec in plan['arms']:
            group = [r for r in rows if r['arm'] == spec['name']]
            arms.append({'arm': spec['name'], 'certified': sum(r['certified'] for r in group),
                'primary_true_feasible': sum(r['primary_truth']['true_feasible'] for r in group),
                'mean_safe_reward': np.mean([r['actual_safe_reward'] for r in group]),
                'verification_calls': sum(r['new_verification_calls'] for r in group)})
        paired = []
        for seed in plan['target_seeds']:
            group = {r['arm']: r for r in rows if r['seed'] == seed}
            a, b = group[FULL], group[LOCAL]
            paired.append({'seed': seed, 'local_minus_full_reward': b['actual_safe_reward']-a['actual_safe_reward'],
                'local_minus_full_verification_calls': b['new_verification_calls']-a['new_verification_calls'],
                'primary_changed': a['primary'] != b['primary'], 'deployment_changed': a['deployed'] != b['deployed']})
        save(output/'results.json', {'protocol': plan, 'rows': rows, 'arms': arms, 'paired': paired,
            'unique_policies_truth_scored': len(truth), 'completed_physical_calls': {
                k: sum(r[k] for r in rows) for k in ['new_initial_calls', 'new_search_calls', 'new_verification_calls']}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('search', 'verify', 'score'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--arm')
    args = parser.parse_args()
    plan = read(MANIFEST)
    output = ROOT/plan['output_dir']
    if args.stage != 'score' and (args.seed, args.arm) not in cases(plan):
        parser.error('Use a registered seed and arm')
    if args.stage == 'search': search(output, plan, args.seed, args.arm)
    elif args.stage == 'verify': verify(output, plan, args.seed, args.arm)
    else: score(output, plan)
    print(args.stage, args.seed, args.arm, 'complete', flush=True)


if __name__ == '__main__':
    main()
