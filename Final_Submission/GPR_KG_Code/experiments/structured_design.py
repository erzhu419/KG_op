"""Shared structured finite-grid initial designs for RZDT experiments."""

from __future__ import annotations

import numpy as np


def boundary_seed_solutions(problem) -> list[tuple[int, ...]]:
    """Return problem-independent center and coordinate-axis box seeds."""
    lo, hi = problem.int_bounds()
    lo = np.asarray(lo, dtype=int)
    hi = np.asarray(hi, dtype=int)
    center = np.round((lo + hi) / 2.0).astype(int)
    seeds = {tuple(lo), tuple(hi), tuple(center)}
    for j in range(problem.d):
        x_hi = lo.copy()
        x_hi[j] = hi[j]
        seeds.add(tuple(int(v) for v in x_hi))
        x_mid = lo.copy()
        x_mid[j] = center[j]
        seeds.add(tuple(int(v) for v in x_mid))
    return sorted(seeds)


def structured_initial_samples(problem, n0: int) -> list[tuple[int, ...]]:
    """Build the common initial design used for fair baseline comparison.

    The deterministic part uses only integer-box bounds. Remaining samples are
    filled with the problem's random integer sampler, so callers control
    reproducibility by setting the NumPy seed before invoking this function.
    """
    samples: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for x in boundary_seed_solutions(problem):
        if x not in seen:
            samples.append(x)
            seen.add(x)
        if len(samples) >= n0:
            return samples

    while len(samples) < n0:
        x = tuple(int(v) for v in problem.sample_random())
        if x not in seen:
            samples.append(x)
            seen.add(x)
    return samples


def common_random_initial_samples(
        problem, n0: int, seed: int) -> list[tuple[int, ...]]:
    """Build a reproducible common random finite-grid initial design."""
    rng = np.random.RandomState(seed)
    lo, hi = problem.int_bounds()
    lo = np.asarray(lo, dtype=int)
    hi = np.asarray(hi, dtype=int)
    samples: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(samples) < n0:
        x = tuple(int(rng.randint(lo[j], hi[j] + 1))
                  for j in range(problem.d))
        if x not in seen:
            samples.append(x)
            seen.add(x)
    return samples
