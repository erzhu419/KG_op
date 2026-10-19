"""Recover paid V69 query pools and TS scores, then score their frozen union.

Replay stops immediately after supplying the third saved query observation:
terminal ordering and verification already exist and are not rerun here.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
from scipy.stats import norm

from acquisition.decision_backends import score_decision_backend
from performance.mainline_legacy_backend import build_backend
from performance.run_mainline_legacy_backend_bridge import (
    ROOT, attempt, load_pickle, read, save,
)

MANIFEST = ROOT/'performance/manifests/mainline_candidate_pool_audit_v1.json'


class QueriesRecovered(Exception):
    pass


def case_dir(output, seed, arm):
    return output/f'seed_{seed}'/arm


def cases(plan):
    return [(seed, arm) for seed in plan['target_seeds'] for arm in plan['arms']]


def recover(plan, seed, arm):
    output = ROOT/plan['output_dir']
    directory = case_dir(output, seed, arm)
    with attempt(directory, 'recover', plan['domain'], arm) as cost:
        cost.update(seed=seed, new_truth_function_calls=0, new_source_fits=0,
                    download_bytes=0)
        if (directory/'recovery.json').exists():
            cost['reused'] = True
            return
        saved = read(case_dir(ROOT/plan['parent_output'], seed, arm)/'search.json')
        prior = load_pickle(ROOT/plan['frozen_assets']['native_source_prior_file'])
        bundle = (None if arm == 'legacy_native_v69' else
                  load_pickle(ROOT/plan['frozen_assets']['source_models_file']))
        engine = build_backend(plan['domain'], arm, bundle, source_prior=prior,
                               seed=seed, N=13, initial_rows=saved['history'])
        bank, index, stages = [], {}, []

        def ids(points):
            result = []
            for point in points:
                key = tuple(map(int, point))
                if key not in index:
                    index[key] = len(bank)
                    bank.append(key)
                result.append(index[key])
            return result

        history_ids = ids(row['point'] for row in saved['history'])
        active = {}
        generate = engine._generate_candidates
        backend = (score_decision_backend if arm == 'legacy_native_v69'
                   else engine._score_backend)
        supply = engine._simulate_and_store

        def capture_generate(iteration):
            candidates, sources = generate(iteration)
            active.clear()
            active.update(iteration=int(iteration), paid_prefix=len(engine.history),
                          generated_ids=ids(candidates),
                          generated_sources=[sources.get(tuple(x), 'unknown') for x in candidates])
            return candidates, sources

        def capture_score(name, candidates, *args, **kwargs):
            result = backend(name, candidates, *args, **kwargs)
            active.update(scored_ids=ids(candidates), score=result)
            return result

        def capture_selected(candidates, *, selected=None, prefix='', sources=None):
            if prefix != 'candidate':
                return {}
            position = len(stages)
            old = saved['iterations'][position]
            action_ids = ids(candidates)
            selected_index = candidates.index(selected)
            scores = active['score']
            selected_id = ids([selected])[0]
            assert action_ids == active['scored_ids'], 'Scored pool changed before selection'
            assert old['selection_policy'] == 'acquisition' and not old['decision_backend_forced_override']
            assert selected_index == int(np.argmax(scores['total']))
            assert list(selected) == old['x_selected'] == saved['history'][10+position]['point']
            assert active['iteration'] == old['iteration'] and active['paid_prefix'] == old['stage']
            assert len(candidates) == old['n_candidates']
            assert len(engine._last_terminal_pool) == old['terminal_pool_size']
            assert sources[tuple(selected)] == old['candidate_source_selected']
            checked = {'total': old['score_selected']}
            checked.update({field: old[f'decision_{field}_selected']
                            for field in scores if f'decision_{field}_selected' in old})
            for field, expected in checked.items():
                np.testing.assert_allclose(np.asarray(scores[field])[selected_index], expected,
                                           rtol=1e-10, atol=1e-12, err_msg=field)
            active.update(action_ids=action_ids,
                          action_sources=[sources.get(tuple(x), 'unknown') for x in candidates],
                          terminal_union_ids=ids(engine._last_terminal_pool),
                          observed_ids=ids(point for point, _ in engine.history),
                          selected_index=selected_index, selected_id=selected_id,
                          parity=True, parity_score_fields=list(checked))
            stages.append(dict(active))
            print(f'recovered {seed} {arm} query {position+1}/3', flush=True)
            return {}

        def replay_raw(point):
            value = supply(point)
            if len(engine.history) == 13:
                raise QueriesRecovered()
            return value

        with ExitStack() as stack:
            for name in ('simulate', 'true_objective', 'true_constraint_mean', 'true_sigma'):
                stack.enter_context(patch.object(engine.problem, name,
                    side_effect=AssertionError(f'{name} is outside raw-only recovery')))
            stack.enter_context(patch.object(engine, '_generate_candidates', capture_generate))
            stack.enter_context(patch.object(engine, '_truth_pool_diagnostics', capture_selected))
            stack.enter_context(patch.object(engine, '_simulate_and_store', replay_raw))
            if arm == 'legacy_native_v69':
                stack.enter_context(patch('algorithms.single_olhkg.score_decision_backend', capture_score))
            else:
                stack.enter_context(patch.object(engine, '_score_backend', capture_score))
            try:
                engine.run_search()
            except QueriesRecovered:
                pass
        assert len(stages) == 3 and engine.replayed_initial_calls == 13
        assert engine.actual_search_calls == 0
        for (point, raw), row in zip(engine.history, saved['history']):
            assert list(point) == row['point']
            np.testing.assert_array_equal(raw, row['raw'])
        np.savez_compressed(directory/'policies.npz', points=np.asarray(bank, dtype=np.int16))
        save(directory/'recovery.json', {
            'status': 'complete', 'seed': seed, 'arm': arm, 'history_ids': history_ids,
            'bank_size': len(bank), 'stages': stages, 'raw_rows_reused': 13,
            'selected_policy_score_route_parity': True,
            'scope': 'Three actual queries; no final posterior update, terminal ordering or verification replay.',
            'frozen_utc': datetime.now(timezone.utc).isoformat(),
        })


def freeze(plan):
    output = ROOT/plan['output_dir']
    rows = []
    for seed, arm in cases(plan):
        directory = case_dir(output, seed, arm)
        recovered = read(directory/'recovery.json')
        assert recovered['status'] == 'complete' and recovered['selected_policy_score_route_parity']
        assert len(recovered['stages']) == 3
        rows.append({'seed': seed, 'arm': arm, 'bank_size': recovered['bank_size'],
                     'parity_stages': len(recovered['stages'])})
    barrier = {'status': 'frozen', 'cases': rows, 'parity_stages': sum(r['parity_stages'] for r in rows),
               'new_simulator_calls': 0, 'truth_before_freeze': False,
               'frozen_utc': datetime.now(timezone.utc).isoformat()}
    assert barrier['parity_stages'] == 27
    save(output/'recovery_barrier.json', barrier)


def score_truth(plan):
    output = ROOT/plan['output_dir']
    barrier = read(output/'recovery_barrier.json')
    assert barrier['status'] == 'frozen' and barrier['parity_stages'] == 27
    with attempt(output, 'truth', plan['domain']) as cost:
        cost.update(new_truth_function_calls=0, new_source_fits=0, download_bytes=0,
                    reused_complete_policies=0, newly_completed_policies=0)
        if (output/'truth_summary.json').exists():
            cost['reused'] = True
            return
        bank = list(dict.fromkeys(tuple(map(int, x)) for seed, arm in cases(plan)
                    for x in np.load(case_dir(output, seed, arm)/'policies.npz')['points']))
        cached = {}
        for path in plan['truth_access']['cache_files']:
            data = read(ROOT/path)
            if isinstance(data, dict):
                # The earlier seed80 diagnosis stores coordinates separately.
                points = read((ROOT/path).parent/'inputs.json')['bank']
                data = [{'point': point, 'truth': {
                    'objective': row['f'], 'constraint_mean_centered': row['g_centered'],
                    'constraint_sigma': row['episode_sigma'][1],
                }} for point, row in zip(points, data['rows'], strict=True)]
            for row in data:
                cached.setdefault(tuple(row['point']), {}).update(row['truth'])
        # Only the existing problem is loaded, with no posterior restoration.
        engine = load_pickle(case_dir(ROOT/plan['parent_output'], *cases(plan)[0])/'search_engine.pkl')
        problem = engine.problem
        records = []
        for point in bank:
            truth = dict(cached.get(point, {}))
            missing = []
            for field, function in (
                ('objective', lambda: float(problem.true_objective(point))),
                ('constraint_mean_centered', lambda: float(problem.true_constraint_mean(point))-problem.tau),
                ('constraint_sigma', lambda: float(problem.true_sigma(point)[1])),
            ):
                if field not in truth:
                    truth[field] = function()
                    cost['new_truth_function_calls'] += 1
                    missing.append(field)
            cost['newly_completed_policies' if missing else 'reused_complete_policies'] += 1
            margin = truth['constraint_mean_centered'] + norm.ppf(1-problem.alpha)*truth['constraint_sigma']
            truth.update(chance_margin=float(margin), true_feasible=bool(margin <= 0),
                         quality=float(np.clip(1-truth['objective']/1.5, 0, 1)))
            records.append({'truth': truth, 'new_fields': missing})
        np.savez_compressed(output/'truth_policies.npz', points=np.asarray(bank, dtype=np.int16))
        save(output/'truth.json', records)
        save(output/'truth_summary.json', dict(cost, unique_policies=len(bank), status='complete'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['recover', 'freeze', 'truth'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--arm')
    args = parser.parse_args()
    plan = read(MANIFEST)
    if args.stage == 'recover':
        if (args.seed, args.arm) not in cases(plan):
            parser.error('recover requires a registered --seed and --arm')
        recover(plan, args.seed, args.arm)
    elif args.stage == 'freeze':
        freeze(plan)
    else:
        score_truth(plan)


if __name__ == '__main__':
    main()
