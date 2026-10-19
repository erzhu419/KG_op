"""Observation-stream routing with synthetic responses only; no source restore."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from algorithms.single_olhkg import (
    SIMULATION_STREAM_TAG, SingleOLHKGAlgorithm, SingleOLHKGConfig,
)
from performance.mainline_legacy_native import LegacyNativeAlgorithm
from performance.mainline_role_backend import (
    MODEL_ARMS, STOP, build_fresh_backend,
)


class StreamStub:
    d = 2

    def __init__(self):
        self.calls = []

    @staticmethod
    def normalize(point):
        return np.asarray(point, dtype=float)

    def simulate(self, point, rng):
        self.calls.append(tuple(point))
        return rng.normal(size=2)


def engine(observation_seed=None, *, initial_rows=None):
    config = SingleOLHKGConfig(
        seed=2026102111, observation_seed=observation_seed,
        use_state_coupling=False, use_state_basis=False, lambda_coupling=0,
    )
    if initial_rows is None:
        return SingleOLHKGAlgorithm(StreamStub(), config)
    return LegacyNativeAlgorithm(StreamStub(), config, initial_rows=initial_rows)


def expected_raw(seed, index):
    return np.random.default_rng(np.random.SeedSequence([
        seed, index, SIMULATION_STREAM_TAG,
    ])).normal(size=2)


def test_default_observation_stream_preserves_historical_seed_and_indices():
    algorithm = engine()
    for index in range(13):
        np.testing.assert_array_equal(
            algorithm._simulate_and_store((index, 0)),
            expected_raw(algorithm.config.seed, index),
        )
    assert algorithm.config.observation_seed is None
    assert len(algorithm.history) == 13


def test_observation_seeds_change_responses_without_changing_optimizer_streams():
    first, second = engine(2026102211), engine(2026102311)
    assert first.config.seed == second.config.seed == 2026102111
    np.testing.assert_array_equal(first.rng.normal(size=16), second.rng.normal(size=16))
    np.testing.assert_array_equal(first.rec_rng.normal(size=16), second.rec_rng.normal(size=16))
    for index in range(13):
        raws = [algorithm._simulate_and_store((index, 0))
                for algorithm in (first, second)]
        for algorithm, raw in zip((first, second), raws):
            np.testing.assert_array_equal(
                raw, expected_raw(algorithm.config.observation_seed, index))
        assert not np.array_equal(*raws)
    # Observation draws neither consume the main RNG nor change proposal namespaces.
    np.testing.assert_array_equal(first.rng.normal(size=16), second.rng.normal(size=16))
    for iteration in (10, 11, 12):
        np.testing.assert_array_equal(
            first._proposal_rng(iteration, 'task_expert').normal(size=16),
            second._proposal_rng(iteration, 'task_expert').normal(size=16),
        )


def test_shared_synthetic_initialization_continues_at_observation_index_ten():
    observation_seed = 2026102211
    rows = [{'point': [index, 0], 'raw': expected_raw(observation_seed, index).tolist()}
            for index in range(10)]
    first, second = (engine(observation_seed, initial_rows=rows) for _ in range(2))
    for algorithm in (first, second):
        for index in range(13):
            np.testing.assert_array_equal(
                algorithm._simulate_and_store((index, 0)),
                expected_raw(observation_seed, index),
            )
        assert algorithm.problem.calls == [(10, 0), (11, 0), (12, 0)]
        assert algorithm.replayed_initial_calls == 10
        assert algorithm.actual_search_calls == 3
        assert len(algorithm.history) == 13
    np.testing.assert_array_equal([raw for _, raw in first.history],
                                  [raw for _, raw in second.history])


@pytest.mark.parametrize('arm', sorted(MODEL_ARMS | {STOP, 'legacy_native_neighbors'}))
@pytest.mark.parametrize('observation_seed', [None, 2026102211, 2026102311])
def test_fresh_backend_routes_observation_seed_before_initialization(arm, observation_seed):
    task_seed = 2026102111
    plan = {
        'domain': 'stub_domain',
        'geometry_by_seed': {str(task_seed): {'problem_kwargs': {'geometry': 1}}},
        'frozen_assets': {'native_source_prior_file': 'unused_prior.pkl',
                          'source_models_file': 'unused_models.pkl'},
    }
    if observation_seed is not None:
        plan['observation_seed_by_task'] = {str(task_seed): observation_seed}
    config = SingleOLHKGConfig(seed=task_seed)
    result = SimpleNamespace(config=config, history=[])
    rows = [{'point': [0, 0], 'raw': [0.1, -0.2]}]
    with patch('performance.mainline_role_backend.load_pickle', return_value=object()), \
         patch('performance.mainline_role_backend.build_native', return_value=result) as native:
        returned = build_fresh_backend(plan, task_seed, arm, rows)
    assert returned is result
    assert returned.config.seed == task_seed
    assert returned.config.observation_seed == observation_seed
    assert returned.history == []
    assert native.call_args.kwargs['seed'] == task_seed
    assert native.call_args.kwargs['initial_rows'] is rows
    assert native.call_args.kwargs['N'] == (10 if arm == STOP else 13)
    assert native.call_args.kwargs['problem_kwargs'] == {'geometry': 1}
