"""Frozen public neighbors from the original target-feedback-only generator.

The supplement is constructed once from the shared ten paid initialization
rows. Temporary hooks append it to the native pools and retain the actual
decision scores. Neighbor generation uses a separate RNG; native proposal and
acquisition RNG objects are not replaced.
"""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import numpy as np

from acquisition.decision_backends import score_decision_backend


NEIGHBOR_STREAM_TAG = 0x4E454947


def make_shared_neighbors(initial_engine, seed, count=64):
    """Use only the saved STOP initialization, without fitting or truth calls."""
    if len(initial_engine.history) != 10:
        raise ValueError('Shared neighbors require exactly the ten initialization rows')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), NEIGHBOR_STREAM_TAG]))
    points = [tuple(map(int, point)) for point in
              initial_engine._observed_neighbor_candidates(n=count, rng=rng)]
    observed = set(initial_engine.observations)
    return {
        'points': points,
        'metadata': {
            'generator': 'SingleOLHKGAlgorithm._observed_neighbor_candidates',
            'seed': int(seed), 'stream_tag': NEIGHBOR_STREAM_TAG,
            'requested_count': int(count), 'count': len(points),
            'new_policy_count': sum(point not in observed for point in points),
            'initialization_rows': len(initial_engine.history),
            'radius': float(initial_engine.config.observed_neighbor_radius),
            'safe_margin_scale': float(initial_engine.config.observed_neighbor_safe_margin_scale),
            'truth_used': False,
        },
    }


@contextmanager
def shared_neighbor_candidates(engine, points):
    """Append the same frozen supplement and capture native decision records.

    ``audit['points']`` is the unique policy bank; all *_ids fields index it.
    Each stage contains the scorer's full per-candidate output before its paid
    observation. The first stage is therefore the initial posterior snapshot.
    Exit this context before pickling the engine so no temporary hooks remain.
    """
    supplement = list(dict.fromkeys(tuple(map(int, point)) for point in points))
    audit = {'points': [], 'stages': []}
    index = {}

    def ids(rows):
        result = []
        for row in rows:
            point = tuple(map(int, row))
            if point not in index:
                index[point] = len(audit['points'])
                audit['points'].append(point)
            result.append(index[point])
        return result

    audit['shared_ids'] = ids(supplement)
    active = {}
    generate = engine._generate_candidates
    recommend = engine._recommendation_pool
    diagnose = engine._truth_pool_diagnostics
    native = engine.backend_arm in {'legacy_native_v69', 'initialization_only'}
    backend = score_decision_backend if native else engine._score_backend

    def generate_with_neighbors(iteration):
        candidates, sources = generate(iteration)
        original_count = len(candidates)
        for point in supplement:
            if point not in sources:
                candidates.append(point)
                sources[point] = 'shared_observed_neighbor'
        active.clear()
        active.update(iteration=int(iteration), paid_prefix=len(engine.history),
                      generated_ids=ids(candidates),
                      native_candidate_count=original_count,
                      added_candidate_count=len(candidates)-original_count)
        return candidates, sources

    def recommend_with_neighbors():
        return list(dict.fromkeys([*recommend(), *supplement]))

    def capture_score(name, candidates, *args, **kwargs):
        result = backend(name, candidates, *args, **kwargs)
        active.update(scored_ids=ids(candidates), score=dict(result))
        return result

    def capture_selected(candidates, *, selected=None, prefix='', sources=None):
        if prefix == 'candidate':
            action_ids = ids(candidates)
            if action_ids != active['scored_ids']:
                raise AssertionError('Candidate pool changed after backend scoring')
            active.update(action_ids=action_ids,
                          action_sources=[sources.get(tuple(point), 'unknown') for point in candidates],
                          terminal_union_ids=ids(engine._last_terminal_pool),
                          observed_ids=ids(point for point, _ in engine.history),
                          selected_index=candidates.index(selected),
                          selected_id=ids([selected])[0])
            audit['stages'].append(dict(active))
        return diagnose(candidates, selected=selected, prefix=prefix, sources=sources)

    with ExitStack() as stack:
        stack.enter_context(patch.object(engine, '_generate_candidates', generate_with_neighbors))
        stack.enter_context(patch.object(engine, '_recommendation_pool', recommend_with_neighbors))
        stack.enter_context(patch.object(engine, '_truth_pool_diagnostics', capture_selected))
        if native:
            stack.enter_context(patch('algorithms.single_olhkg.score_decision_backend', capture_score))
        else:
            stack.enter_context(patch.object(engine, '_score_backend', capture_score))
        yield audit
