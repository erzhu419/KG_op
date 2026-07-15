"""Shared experimental designs used by audited comparison runners."""

from __future__ import annotations

import hashlib
import json
import math

from scipy.stats import qmc


COMMON_SOBOL_SEED_OFFSET = 7919


def integer_design_fingerprint(points):
    """Hash ordered integer points without optimizer-specific metadata."""

    payload = [list(map(int, point)) for point in points]
    return hashlib.sha256(json.dumps(
        payload,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def common_sobol_integer_design(
    problem,
    n,
    seed,
    *,
    seed_offset=COMMON_SOBOL_SEED_OFFSET,
):
    """Return a deterministic deduplicated Sobol design in problem space.

    The design is independent of an optimizer and its source archive. Transfer
    baselines and SC-OLH-KG therefore receive byte-identical target initial
    points when they share ``problem``, ``n``, and ``seed``.
    """

    n = int(n)
    if n < 0:
        raise ValueError("Sobol design size must be nonnegative")
    if n == 0:
        return []
    if not hasattr(problem, "continuous_to_int"):
        raise TypeError("problem must expose continuous_to_int")

    requested = max(2, 4 * n)
    for _ in range(12):
        exponent = int(math.ceil(math.log2(requested)))
        profiles = qmc.Sobol(
            d=int(problem.d),
            scramble=True,
            seed=int(seed) + int(seed_offset),
        ).random_base2(exponent)
        points = []
        seen = set()
        for profile in profiles:
            point = tuple(map(int, problem.continuous_to_int(profile)))
            if point in seen:
                continue
            seen.add(point)
            points.append(point)
            if len(points) == n:
                return points
        requested *= 2
    raise RuntimeError(
        "integer search space cannot supply the requested unique Sobol design"
    )
