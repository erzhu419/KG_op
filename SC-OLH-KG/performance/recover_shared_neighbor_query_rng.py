"""Recover acquisition RNG states from the frozen shared-neighbor raw paths.

No simulator, truth function, source fit, final update or terminal verification
is executed. Native frozen10 scores use the same actual pool and random draw,
changing only the posterior models supplied to the original decision backend.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import pickle
import time
from unittest.mock import patch

import numpy as np

from acquisition.decision_backends import score_decision_backend
from performance.mainline_legacy_backend import build_backend
from performance.mainline_shared_neighbors import shared_neighbor_candidates
from performance.run_mainline_legacy_backend_bridge import (
    ROOT, _NativePickler, attempt, load_pickle, read, save,
)


MANIFEST = ROOT / 'performance/manifests/mainline_query_channels_v1.json'
NATIVE = 'legacy_native_neighbors'


class QueriesRecovered(Exception):
    pass


def forbidden_model_call(*args, **kwargs):
    raise AssertionError('Simulation and truth calls are outside raw-only recovery')


def assert_score_equal(actual, expected):
    """Detect a replay/model snapshot mismatch before using recovered draws."""
    assert actual.keys() == expected.keys(), 'Decision score fields changed'
    for field, old in expected.items():
        value = actual[field]
        if old is None or isinstance(old, (str, bool)):
            assert value == old, field
        else:
            np.testing.assert_allclose(value, old, rtol=1e-10, atol=1e-12,
                                       err_msg=field)


def registered_cases(plan):
    return [(seed, arm) for seed in plan['target_seeds'] for arm in plan['arms']]


def recover(plan, seed, arm):
    parent = ROOT / plan['parent_output']
    source = parent / f'seed_{seed}' / arm
    directory = ROOT / plan['output_dir'] / f'seed_{seed}' / arm
    with attempt(directory, 'recover-query-rng', plan['domain'], arm) as cost:
        cost.update(seed=seed, new_target_calls=0, new_truth_function_calls=0,
                    new_source_fits=0, download_bytes=0)
        if (directory / 'recovery.json').exists():
            cost['reused'] = True
            return
        saved = read(source / 'search.json')
        expected_audit = read(source / 'query_audit.json')
        expected_points = np.load(source / 'query_policies.npz')['points']
        prior = load_pickle(ROOT / plan['frozen_assets']['native_source_prior_file'])
        bundle = (None if arm == NATIVE else
                  load_pickle(ROOT / plan['frozen_assets']['source_models_file']))
        backend_arm = 'legacy_native_v69' if arm == NATIVE else arm
        engine = build_backend(plan['domain'], backend_arm, bundle,
                               source_prior=prior, seed=seed, N=13,
                               initial_rows=saved['history'])
        neighbors = np.load(parent / f'seed_{seed}' / 'neighbors.npz')['points']
        backend = score_decision_backend if arm == NATIVE else engine._score_backend
        supply = engine._simulate_and_store
        rng_records, frozen_records = [], []
        frozen_models = None

        def capture_score(name, candidates, obj, con, variance, problem, **kwargs):
            nonlocal frozen_models
            position = len(rng_records)
            old = expected_audit['stages'][position]
            assert len(engine.history) == old['paid_prefix']
            np.testing.assert_array_equal(
                np.asarray(candidates), expected_points[old['scored_ids']])
            rng_state = deepcopy(kwargs['rng'].bit_generator.state)
            frozen_rng = deepcopy(kwargs['rng']) if arm == NATIVE else None
            if arm == NATIVE and frozen_models is None:
                started = time.process_time()
                # Reuse the native engine serializer for its delegated problem
                # adapters; ordinary deepcopy probes recurse through __getattr__.
                buffer = BytesIO()
                _NativePickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(
                    (obj, con, variance, kwargs['task_ensemble']))
                frozen_models = pickle.loads(buffer.getvalue())
                cost['native_snapshot_cpu_seconds'] = time.process_time() - started
            actual = backend(name, candidates, obj, con, variance, problem, **kwargs)
            assert_score_equal(actual, old['score'])
            record = {
                'query_index': position + 1, 'iteration': old['iteration'],
                'paid_prefix': old['paid_prefix'], 'rng_state': rng_state,
                'full_score_parity': True, 'score_field_count': len(actual),
            }
            rng_records.append(record)
            if arm == NATIVE:
                fixed_obj, fixed_con, fixed_variance, fixed_ensemble = frozen_models
                frozen_kwargs = dict(kwargs, rng=frozen_rng, task_ensemble=fixed_ensemble)
                frozen_score = score_decision_backend(
                    name, candidates, fixed_obj, fixed_con, fixed_variance,
                    problem, **frozen_kwargs)
                if position == 0:
                    assert_score_equal(frozen_score, actual)
                selected_index = int(np.argmax(frozen_score['total']))
                frozen_records.append({
                    'query_index': position + 1, 'paid_prefix': old['paid_prefix'],
                    'selected_index': selected_index,
                    'selected_id': old['scored_ids'][selected_index],
                    'score': frozen_score,
                })
            print(f'parity {seed} {arm} query {position + 1}/3', flush=True)
            return actual

        def replay_raw(point):
            value = supply(point)
            if len(engine.history) == 13:
                raise QueriesRecovered()
            return value

        with ExitStack() as stack:
            for name in ('simulate', 'true_objective', 'true_constraint_mean', 'true_sigma'):
                stack.enter_context(patch.object(engine.problem, name, forbidden_model_call))
            stack.enter_context(patch.object(engine, '_simulate_and_store', replay_raw))
            if arm == NATIVE:
                stack.enter_context(patch(
                    'performance.mainline_shared_neighbors.score_decision_backend', capture_score))
            else:
                stack.enter_context(patch.object(engine, '_score_backend', capture_score))
            with shared_neighbor_candidates(engine, neighbors) as audit:
                try:
                    engine.run_search()
                except QueriesRecovered:
                    pass

        assert len(rng_records) == len(audit['stages']) == 3
        assert engine.replayed_initial_calls == 13 and engine.actual_search_calls == 0
        np.testing.assert_array_equal(audit['points'], expected_points)
        assert audit['shared_ids'] == expected_audit['shared_ids']
        for actual, expected, record in zip(audit['stages'], expected_audit['stages'],
                                            rng_records, strict=True):
            for field, value in expected.items():
                if field != 'score':
                    assert actual[field] == value, field
            selected_point = list(audit['points'][actual['selected_id']])
            assert selected_point == saved['history'][actual['paid_prefix']]['point']
            assert actual['selected_index'] == int(np.argmax(actual['score']['total']))
            record.update(candidate_order_route_selection_parity=True,
                          candidate_count=len(actual['action_ids']),
                          selected_index=actual['selected_index'])
        for (point, raw), row in zip(engine.history, saved['history'], strict=True):
            assert list(point) == row['point']
            np.testing.assert_array_equal(raw, row['raw'])
        if arm == NATIVE:
            save(directory / 'native_frozen10.json', {
                'status': 'complete', 'seed': seed, 'arm': arm,
                'source_audit': str((source / 'query_audit.json').relative_to(ROOT)),
                'scope': 'Initial10 posterior models; actual pool, observed set and scorer kwargs; same pre-call RNG.',
                'first_query_full_parity': True, 'stages': frozen_records,
            })
        save(directory / 'recovery.json', {
            'status': 'complete', 'seed': seed, 'arm': arm,
            'source_audit': str((source / 'query_audit.json').relative_to(ROOT)),
            'stages': rng_records, 'raw_rows_reused': 13,
            'all_raw_order_parity': True,
            'scope': 'Actual three queries; stop at third raw supply before posterior update or terminal work.',
            'frozen_utc': datetime.now(timezone.utc).isoformat(),
        })
        print(f'recovered {seed} {arm}: three pools, scores, selections and RNG states', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--arm', required=True)
    args = parser.parse_args()
    plan = read(args.manifest)
    if (args.seed, args.arm) not in registered_cases(plan):
        parser.error('seed and arm must be registered in the manifest')
    recover(plan, args.seed, args.arm)


if __name__ == '__main__':
    main()
