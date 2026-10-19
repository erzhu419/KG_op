"""Check the post-initialization control on paid histories, without new outcomes."""
import argparse

import numpy as np

from performance.diagnose_mainline_initialization import query_parity
from performance.mainline_role_after_init import after_init_point_only_moment_state
from performance.mainline_role_prior import role_moment_state
from performance.run_mainline_legacy_backend_bridge import (
    ROOT, attempt, load_pickle, read, save, save_engine,
)

MANIFEST = ROOT / 'performance/manifests/mainline_role_after_init_v1.json'


def variance(state):
    return np.diag(state['cov'].reshape(4*len(state['mean']), -1)).reshape(-1, 4)


def check(plan, seed):
    parent_plan = read(ROOT / plan['preflight']['parent_manifest'])
    parent = ROOT / parent_plan['output_dir']
    output = ROOT / plan['output_dir'] / 'preflight' / f'seed_{seed}'
    with attempt(output, 'after-init-interface', plan['domain']) as cost:
        cost.update(seed=seed, new_target_initial_calls=0, new_source_fits=0,
                    new_truth_function_calls=0, download_bytes=0)
        directory = parent / f'seed_{seed}' / 'legacy_role_full'
        rows = read(directory / 'search.json')['history']
        engine = load_pickle(directory / 'search_engine.pkl')
        bundle = load_pickle(ROOT / plan['frozen_assets']['source_models_file'])
        neighbors = [tuple(map(int, p)) for p in np.load(parent/f'seed_{seed}'/'neighbors.npz')['points']]
        observed = {tuple(r['point']) for r in rows}
        late = next(p for p in reversed(neighbors) if p not in observed)
        bank = list(dict.fromkeys([tuple(r['point']) for r in rows]+neighbors))
        bank.remove(late)
        errors = {}

        def equal(label, actual, expected):
            np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10, err_msg=label)
            errors[label] = float(np.max(np.abs(np.asarray(actual)-expected)))

        cache = {}
        base = after_init_point_only_moment_state(bundle, engine, bank, rows[:10], cache=cache)
        expected = role_moment_state(bundle, engine, bank, rows[:10])
        for key in ('mean', 'cov', 'noise', 'weights'):
            equal('initial10_'+key, base[key], expected[key])
        saved_scores = query_parity(parent, seed, 'legacy_role_full', bank,
                                   dict(mean=base['mean'], variance=variance(base), noise=base['noise']))
        records = []
        previous = base
        for prefix in (11, 12, 13):
            state = after_init_point_only_moment_state(bundle, engine, bank, rows[:prefix], cache=cache)
            touched = {tuple(r['point']) for r in rows[10:prefix]}
            untouched = [i for i,p in enumerate(bank) if p not in touched]
            for key in ('mean', 'noise', 'policy_weights'):
                equal(f'{prefix}_untouched_'+key, state[key][untouched], base[key][untouched])
            equal(f'{prefix}_untouched_cov', state['cov'][untouched][:, :, untouched, :],
                  base['cov'][untouched][:, :, untouched, :])
            current_point = tuple(rows[prefix-1]['point'])
            other = [i for i,p in enumerate(bank) if p != current_point]
            for key in ('mean', 'noise', 'policy_weights'):
                equal(f'{prefix}_other_'+key, state[key][other], previous[key][other])
            records.append(dict(prefix=prefix, cost=state['cost']))
            previous = state
        for point in sorted({tuple(r['point']) for r in rows[10:]}):
            index = bank.index(point)
            own = [r for r in rows[10:] if tuple(r['point']) == point]
            own_bank = list(dict.fromkeys([tuple(r['point']) for r in rows[:10]]+[point]))
            full = role_moment_state(bundle, engine, own_bank, rows[:10]+own)
            position = own_bank.index(point)
            for key in ('mean', 'noise'):
                equal('own_'+str(index)+'_'+key, state[key][index], full[key][position])
            equal('own_'+str(index)+'_variance', variance(state)[index], variance(full)[position])
            equal('own_'+str(index)+'_weights', state['policy_weights'][index], full['weights'])
            assert cache['own'][point]['model'].paid_episodes == len(own)
        repeat = after_init_point_only_moment_state(bundle, engine, bank, rows, cache=cache)
        assert repeat['cost']['own_raw_updates'] == 0
        for key in ('mean', 'cov', 'noise', 'policy_weights'):
            equal('repeat_'+key, repeat[key], state[key])
        enlarged = [late] + list(reversed(bank))
        expanded = after_init_point_only_moment_state(bundle, engine, enlarged, rows, cache=cache)
        expected = role_moment_state(bundle, engine, enlarged, rows[:10])
        positions = [enlarged.index(p) for p in bank]
        for key in ('mean', 'noise', 'policy_weights'):
            equal('reorder_'+key, expanded[key][positions], state[key])
        for key in ('mean', 'noise'):
            equal('late_'+key, expanded[key][0], expected[key][0])
        equal('late_variance', variance(expanded)[0], variance(expected)[0])
        equal('late_weights', expanded['policy_weights'][0], expected['weights'])
        assert expanded['cost']['own_raw_updates'] == 0
        engine.backend_arm = 'legacy_role_after_init_point_only'
        engine.backend_bundle = bundle
        engine._point_only_cache = cache
        engine._moment_cache = {}
        before, before_noise = engine._posterior_views(enlarged)
        checkpoint = output / 'restoration_check.pkl'
        save_engine(checkpoint, engine)
        restored = load_pickle(checkpoint)
        restored.backend_bundle = bundle
        after, after_noise = restored._posterior_views(enlarged)
        for head in (0, 1):
            equal(f'restored_head_{head}', after[head].posterior_mean_many(enlarged),
                  before[head].posterior_mean_many(enlarged))
            equal(f'restored_noise_{head}', after_noise.predict_variance_many(head, enlarged, restored.problem),
                  before_noise.predict_variance_many(head, enlarged, engine.problem))
        checkpoint.unlink()
        assert np.max(np.abs(state['mean']-base['mean'])) > 1e-10
        save(output/'result.json', dict(status='pass', seed=seed, errors=errors,
            maximum_error=max(errors.values()), saved_score_parity=saved_scores,
            prefix_costs=records, own_update_count=sum(r['cost']['own_raw_updates'] for r in records),
            initial_node_counts=[[len(f.log_weights) for f in pair]
                                 for pair in cache['initial']['components']],
            new_simulator_calls=0, new_truth_calls=0))
        print(seed, 'pass', 'max_error', max(errors.values()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    plan = read(MANIFEST)
    if args.seed is not None:
        assert args.seed in plan['preflight']['saved_seeds']
        check(plan, args.seed)
    else:
        output = ROOT/plan['output_dir']/'preflight'
        rows = [read(output/f'seed_{seed}'/'result.json') for seed in plan['preflight']['saved_seeds']]
        assert all(r['status']=='pass' and r['own_update_count']==3 for r in rows)
        save(output/'results.json', dict(status='pass', rows=rows,
            maximum_error=max(r['maximum_error'] for r in rows), new_simulator_calls=0, new_truth_calls=0))
        plan['status'] = 'preflight_passed'
        save(MANIFEST, plan)
        print('All saved-history preflights passed', flush=True)


if __name__ == '__main__':
    main()
