"""Passive copies of terminal values that the algorithm has already computed.

This module never evaluates a policy, queries a model, sorts a pool, or draws a
random number. Tests may replace capture with a no-op to compare decisions.
"""
from copy import deepcopy

import numpy as np


def _copy(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key:_copy(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy(item) for item in value]
    return deepcopy(value)


def capture(engine, phase, **saved_values):
    """Append one snapshot; start replaces the preceding terminal invocation."""
    if phase == 'start':
        observed, index, sequence = [], {}, []
        for position, (point, _) in enumerate(engine.history):
            point = tuple(map(int, point))
            if point not in index:
                index[point] = len(observed)
                observed.append(dict(point=list(point), observation_indices=[], replicate_count=0))
            local = index[point]
            observed[local]['observation_indices'].append(position)
            observed[local]['replicate_count'] += 1
            sequence.append(local)
        engine._terminal_recording = dict(status='partial', paid_prefix=len(engine.history),
            backend_arm=getattr(engine, 'backend_arm', None),
            posterior_context=getattr(engine, '_decision_context', 0),
            observed=observed, history_observed_indices=sequence, stages=[],
            scope='Copies of already computed terminal values; no model evaluation or random draw.')
    payload = getattr(engine, '_terminal_recording', None)
    if payload is None:
        return
    record = dict(phase=phase, **_copy(saved_values))
    if 'components' in record:
        record['available_components'] = list(record['components'])
    payload['stages'].append(record)
    if phase == 'frozen_shortlist':
        payload['status'] = 'frozen'


def snapshot(engine):
    """Return the completed or partial record without sharing mutable objects."""
    return deepcopy(getattr(engine, '_terminal_recording', None))
