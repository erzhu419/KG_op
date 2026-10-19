"""Whole-policy source response permutation for the registered two-family control.

Only an index record is persisted. The derived archive is an in-memory fitting
input with no new fingerprint; reconstruct it from the original archive and
the saved indices, rather than serializing it as an independently hashed source.
"""
from dataclasses import replace

import numpy as np


POLICY_COUNT = 64
REPEAT_COUNT = 3


def _statistics(values, nominal_sigma):
    """Original replicated-source formulas, including native sigma shrinkage."""
    mean = np.mean(values, axis=1)
    variance = np.maximum(np.var(values, axis=1, ddof=1), 1e-12)
    centered_g = values[:, :, 1] - mean[:, 1, None]
    # LearnedMetaPrior._source_observation uses two nominal-variance df. This
    # is deliberately distinct from sqrt(the unbiased replicate variance).
    sigma = np.sqrt(np.maximum((np.sum(centered_g ** 2, axis=1)
        + 2.0 * max(float(nominal_sigma), 1e-8) ** 2) / 4.0, 1e-12))
    return dict(Y_mean=mean, replicate_variance=variance,
                mean_observation_variance=variance / REPEAT_COUNT,
                constraint_sigma=np.maximum(sigma, 1e-8))


def validate_source_archive(archive, *, nominal_sigma=.04):
    """Reject a changed source contract before drawing or fitting a control."""
    archive.validate()
    if (len(archive.tasks) != 2 or len(set(archive.source_domains)) != 2
            or archive.observation_mode != 'replicated'):
        raise ValueError('Source alignment requires two distinct replicated source families')
    if not np.isfinite(nominal_sigma) or float(nominal_sigma) <= 0:
        raise ValueError('Source nominal sigma must be positive and finite')
    for task in archive.tasks:
        if task.n_profiles != POLICY_COUNT or len(task.origins) != POLICY_COUNT:
            raise ValueError('Source alignment requires exactly 64 policies per family')
        if any(np.shape(values) != (REPEAT_COUNT, 2) for values in task.Y_replicates):
            raise ValueError('Each source policy must retain its paired 3 by 2 response block')
        values = np.asarray(task.Y_replicates, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Source response blocks contain nonfinite observations')
        for field, expected in _statistics(values, nominal_sigma).items():
            actual = np.asarray(getattr(task, field))
            if actual.shape != expected.shape or not np.allclose(
                    actual, expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f'{task.name} {field} differs from the frozen replicated-source formula')
    first, second = archive.tasks
    if (not np.array_equal(first.X_integer, second.X_integer)
            or not np.array_equal(first.X, second.X)):
        raise ValueError('Source families must share the same physical-policy order')
    if len(set(map(tuple, first.X_integer))) != POLICY_COUNT:
        raise ValueError('The shared source design must contain 64 distinct physical policies')
    return archive


def _indices(permutation):
    indices = np.asarray(permutation)
    if (indices.shape != (POLICY_COUNT,) or indices.dtype.kind not in 'iu'
            or not np.array_equal(np.sort(indices), np.arange(POLICY_COUNT))):
        raise ValueError('Expected one permutation of the 64 physical-policy indices')
    return indices


def draw_source_permutation(seed):
    """Future one-time materialization; never screen or redraw fixed points."""
    return np.random.Generator(np.random.PCG64(int(seed))).permutation(POLICY_COUNT)


def permutation_record(archive, permutation, seed, *, nominal_sigma=.04):
    """Small provenance record; the original digest identifies only its input."""
    indices = _indices(permutation)
    return dict(permutation=indices.tolist(), permutation_seed=int(seed),
        source_domains=list(archive.source_domains), source_seed=int(archive.source_seed),
        observation_mode=archive.observation_mode,
        original_archive_fingerprint=archive.fingerprint,
        derived_archive_fingerprint=None, policies_per_domain=POLICY_COUNT,
        repeats_per_policy=REPEAT_COUNT, nominal_sigma=float(nominal_sigma),
        assignment='response at X[i] is the original same-family block at permutation[i]',
        same_permutation_across_source_families=True,
        fixed_points=[int(i) for i in np.flatnonzero(indices == np.arange(POLICY_COUNT))],
        formulas=dict(Y_mean='mean of the three paired response rows',
            replicate_variance='max(var(ddof=1),1e-12)',
            mean_observation_variance='replicate_variance/3',
            constraint_sigma='sqrt(max((sum((g-mean_g)^2)+2*nominal_sigma^2)/4,1e-12))'),
        source_only=True, new_source_simulations=0)


def permute_response_blocks(archive, permutation, *, nominal_sigma=.04):
    """Move complete response blocks and recompute their original summaries.

    X, family, origins and task metadata stay at their original positions.
    Both families use the same indices. Every mutable array is copied so a
    future source fit cannot modify the original archive through this input.
    """
    validate_source_archive(archive, nominal_sigma=nominal_sigma)
    indices = _indices(permutation)
    tasks = []
    for task in archive.tasks:
        blocks = tuple(np.array(task.Y_replicates[i], dtype=float, copy=True) for i in indices)
        tasks.append(replace(task, X=task.X.copy(), X_integer=task.X_integer.copy(),
            Y_replicates=blocks, **_statistics(np.asarray(blocks), nominal_sigma)))
    # Do not inherit an identifier that would falsely describe changed labels.
    # FrozenTransferArchive.save/load is intentionally not the storage API here.
    return replace(archive, tasks=tuple(tasks), fingerprint='')
