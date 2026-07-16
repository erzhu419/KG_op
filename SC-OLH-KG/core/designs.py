"""Shared experimental designs used by audited comparison runners."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from scipy.stats import qmc


COMMON_SOBOL_SEED_OFFSET = 7919


def integer_design_fingerprint(points):
    """Hash ordered integer points without optimizer-specific metadata."""

    payload = [list(map(int, point)) for point in points]
    return hashlib.sha256(json.dumps(
        payload,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def load_frozen_source_informed_design(
    path,
    *,
    heldout,
    seed,
    n0,
    dimension,
):
    """Load one source-only warm start under the frozen LODO contract."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    supported_kinds = {
        "frozen_source_informed_rank_spanning",
        "frozen_source_informed_risk_coordinate_atlas",
    }
    if payload.get("design_kind") not in supported_kinds:
        raise ValueError("unexpected source-informed design contract")
    if payload.get("heldout_target_domain") != str(heldout):
        raise ValueError("source-informed design heldout domain mismatch")
    if int(payload.get("dimension", -1)) != int(dimension):
        raise ValueError("source-informed design dimension mismatch")
    if int(payload.get("n0", -1)) != int(n0):
        raise ValueError("source-informed design n0 mismatch")
    if payload.get("source_archive_oracle_aided") is not False:
        raise ValueError("source-informed design must use an oracle-free archive")
    if payload.get("target_labels_used") is not False:
        raise ValueError("source-informed design must not use target labels")
    if payload.get("target_oracle_used") is not False:
        raise ValueError("source-informed design must not use target oracle")
    source_fingerprint = str(payload.get("source_archive_fingerprint") or "")
    if not source_fingerprint:
        raise ValueError("source-informed design is missing archive fingerprint")

    design = (payload.get("designs") or {}).get(str(int(seed)))
    if not design:
        raise ValueError(f"source-informed design is missing seed {int(seed)}")
    points = tuple(tuple(map(int, point)) for point in design.get("points", ()))
    if len(points) != int(n0) or len(set(points)) != int(n0):
        raise ValueError("source-informed design must contain n0 unique points")
    if any(len(point) != int(dimension) for point in points):
        raise ValueError("source-informed design point dimension mismatch")
    fingerprint = integer_design_fingerprint(points)
    if fingerprint != design.get("fingerprint"):
        raise ValueError("source-informed design fingerprint mismatch")
    return points, {
        "design_kind": str(payload["design_kind"]),
        "proposal_mode": str(payload.get("proposal_mode", "rank_spanning")),
        "structural_prior_profile": str(
            payload.get("structural_prior_profile", "inherit")),
        "source_dimension": int(payload.get("source_dimension", dimension)),
        "target_dimension": int(payload.get("dimension", dimension)),
        "fingerprint": fingerprint,
        "source_archive_fingerprint": source_fingerprint,
        "source_archive_oracle_aided": False,
        "target_labels_used": False,
        "target_oracle_used": False,
    }


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
