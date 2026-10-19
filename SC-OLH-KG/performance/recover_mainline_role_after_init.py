"""Recover a completed paid search's audit/terminal computation using saved raw."""
import argparse
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np

from performance.mainline_role_backend import build_fresh_backend
from performance.mainline_shared_neighbors import shared_neighbor_candidates
from performance.run_mainline_legacy_backend_bridge import (
    ROOT, attempt, load_pickle, read, save, save_engine,
)


def recover(plan, seed, arm):
    output = ROOT/plan['output_dir']
    directory = output/f'seed_{seed}'/arm
    assert not (directory/'search.json').exists()
    failed = load_pickle(directory/'failed_search_engine.pkl')
    history = [dict(point=list(p), raw=np.asarray(y).tolist()) for p,y in failed.history]
    raw_files = list(directory.glob('failed_search_raw_*.json'))
    assert len(raw_files) == 1 and len(history) == 13 and len(failed.iteration_log) == 3
    paid = read(raw_files[0])
    assert paid == history[10:]
    assert history[:10] == read(output/f'seed_{seed}'/'initialization_only'/'search.json')['history']
    with attempt(directory, 'recover-paid-search-computation', plan['domain'], arm) as cost:
        cost.update(seed=seed, new_target_initial_calls=0, new_source_fits=0,
                    new_truth_function_calls=0, download_bytes=0, replayed_raw_rows=13)
        # A fresh engine avoids the partially updated mixture at the failure.
        engine = build_fresh_backend(plan, seed, arm, history)
        neighbors = np.load(output/f'seed_{seed}'/'neighbors.npz')['points']
        with patch.object(engine.problem, 'simulate', side_effect=AssertionError('Recovery cannot simulate')):
            with shared_neighbor_candidates(engine, neighbors) as audit:
                engine.run_search()
        replayed = [dict(point=list(p), raw=np.asarray(y).tolist()) for p,y in engine.history]
        assert replayed == history and engine.actual_search_calls == 0
        assert engine.replayed_initial_calls == 13
        errors = []
        for old, new in zip(failed.iteration_log, engine.iteration_log, strict=True):
            for key in ('x_selected', 'n_candidates', 'candidate_source_selected'):
                assert new[key] == old[key], key
            for key in ('score_selected', 'decision_objective_mean_selected',
                        'decision_objective_epistemic_selected', 'decision_constraint_mean_selected',
                        'decision_constraint_epistemic_selected', 'decision_constraint_aleatoric_selected'):
                np.testing.assert_allclose(new[key], old[key], rtol=1e-8, atol=1e-10, err_msg=key)
                errors.append(abs(new[key]-old[key]))
        assert [s['paid_prefix'] for s in audit['stages']] == [10,11,12]
        points = np.asarray(audit.pop('points'), dtype=np.int16)
        np.savez_compressed(directory/'query_policies.npz', points=points)
        save(directory/'query_audit.json', audit)
        save(directory/'paid_search_raw.json', paid)
        save_engine(directory/'search_engine.pkl', engine)
        save(directory/'search.json', dict(seed=seed, arm=arm, family=plan['domain'],
            **engine.frozen_terminal, geometry=plan['geometry_by_seed'][str(seed)]['problem_kwargs'],
            history=history, iterations=engine.iteration_log, backend_trace=engine.backend_trace,
            role_restoration_trace=[*getattr(failed, 'role_restoration_trace', []),
                                    *getattr(engine, 'role_restoration_trace', [])],
            logical_search_calls=13, new_initial_calls=0, new_search_calls=len(paid),
            initial_design_seed=80, truth_used=False,
            frozen_utc=datetime.now(timezone.utc).isoformat(),
            recovery=dict(saved_paid_raw=str(raw_files[0].relative_to(ROOT)),
                replayed_raw_rows=13, new_physical_calls=0, selected_score_maximum_error=max(errors),
                original_failure_retained=True)))
        print(seed, 'recovered', 'new physical calls0', 'selected score error', max(errors), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--arm', default='legacy_role_after_init_point_only',
                        choices=['legacy_role_after_init_point_only', 'legacy_role_full'])
    args = parser.parse_args()
    plan = read(ROOT/'performance/manifests/mainline_role_after_init_v1.json')
    assert args.seed in plan['target_seeds']
    recover(plan, args.seed, args.arm)
