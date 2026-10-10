"""Explicit source-scored designs for ordered policy profiles.

This module is intentionally independent of the historical ``LearnedMetaPrior``
class.  It exposes the complete finite algorithm used by the next experiment
track: a dimension-stable cosine coordinate, task-wise outcome ranks, optional
outcome-free descriptor weighting, and Gonzalez farthest-first selection in one
declared augmented metric.

The source archive contains ordinary replicated simulator observations.  A
target descriptor may weight source tasks, but no target outcome is accepted by
the API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import norm


def _finite_vector(values, *, name, minimum_size=1):
    vector = np.asarray(values, dtype=float).reshape(-1)
    if len(vector) < int(minimum_size):
        raise ValueError(f"{name} must contain at least {minimum_size} values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def regular_profile_nodes(dimension):
    """Cell-midpoint nodes on ``[0, 1]`` for a profile discretization."""

    dimension = int(dimension)
    if dimension < 1:
        raise ValueError("profile dimension must be positive")
    return (np.arange(dimension, dtype=float) + 0.5) / float(dimension)


def profile_quadrature_weights(nodes):
    """Voronoi-cell quadrature weights for regular or irregular nodes."""

    nodes = _finite_vector(nodes, name="profile nodes")
    if np.any(nodes < 0.0) or np.any(nodes > 1.0):
        raise ValueError("profile nodes must lie in [0, 1]")
    if len(nodes) > 1 and np.any(np.diff(nodes) <= 0.0):
        raise ValueError("profile nodes must be strictly increasing")
    if len(nodes) == 1:
        return np.ones(1, dtype=float)
    edges = np.empty(len(nodes) + 1, dtype=float)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges[1:-1] = 0.5 * (nodes[:-1] + nodes[1:])
    weights = np.diff(edges)
    if np.any(weights <= 0.0):
        raise RuntimeError("profile quadrature produced nonpositive weights")
    return weights / float(np.sum(weights))


def profile_voronoi_edges(nodes):
    """Voronoi cell edges on ``[0, 1]`` for ordered profile nodes."""

    nodes = _finite_vector(nodes, name="profile nodes")
    if np.any(nodes < 0.0) or np.any(nodes > 1.0):
        raise ValueError("profile nodes must lie in [0, 1]")
    if len(nodes) > 1 and np.any(np.diff(nodes) <= 0.0):
        raise ValueError("profile nodes must be strictly increasing")
    edges = np.empty(len(nodes) + 1, dtype=float)
    edges[0] = 0.0
    edges[-1] = 1.0
    if len(nodes) > 1:
        edges[1:-1] = 0.5 * (nodes[:-1] + nodes[1:])
    return edges


def profile_cosine_coordinate(
    profile,
    *,
    nodes=None,
    max_frequency=8,
    frequency_penalty=0.25,
    include_diagonal_quadratic=True,
):
    """Return a fixed-size weighted cosine coordinate.

    For nodes ``t_i`` and Voronoi cells ``I_i``, the linear block is

    ``c_0 = sum_i h_i |I_i|`` and
    ``c_k = sqrt(2) sum_i h_i int_{I_i} cos(pi k t) dt``.

    Thus the implementation computes the exact continuous cosine coefficient
    of the Voronoi piecewise-constant profile reconstruction, rather than a
    midpoint approximation to the basis itself.

    Coefficient ``k`` is divided by ``1 + frequency_penalty * k``.  When
    requested, the complete diagonal expansion ``(c_0^2, ..., c_K^2)`` is
    appended.  No cross-products or hidden feature selection are performed.
    """

    profile = _finite_vector(profile, name="profile")
    max_frequency = int(max_frequency)
    if max_frequency < 0:
        raise ValueError("max_frequency must be nonnegative")
    frequency_penalty = float(frequency_penalty)
    if frequency_penalty < 0.0 or not np.isfinite(frequency_penalty):
        raise ValueError("frequency_penalty must be finite and nonnegative")
    if nodes is None:
        nodes = regular_profile_nodes(len(profile))
    else:
        nodes = _finite_vector(nodes, name="profile nodes")
    if len(nodes) != len(profile):
        raise ValueError("profile and node dimensions differ")
    edges = profile_voronoi_edges(nodes)
    frequencies = np.arange(max_frequency + 1, dtype=float)
    basis_integrals = np.empty(
        (max_frequency + 1, len(nodes)), dtype=float)
    basis_integrals[0] = np.diff(edges)
    if max_frequency > 0:
        positive = frequencies[1:, None]
        basis_integrals[1:] = (
            np.sqrt(2.0)
            * (
                np.sin(np.pi * positive * edges[None, 1:])
                - np.sin(np.pi * positive * edges[None, :-1])
            )
            / (np.pi * positive)
        )
    coefficients = basis_integrals @ profile
    coefficients /= 1.0 + frequency_penalty * frequencies
    if include_diagonal_quadratic:
        return np.concatenate([coefficients, coefficients ** 2])
    return coefficients


def resample_profile(profile, target_nodes, *, source_nodes=None):
    """Linearly map one ordered profile to a target discretization."""

    profile = _finite_vector(profile, name="profile")
    target_nodes = _finite_vector(target_nodes, name="target nodes")
    if np.any(target_nodes < 0.0) or np.any(target_nodes > 1.0):
        raise ValueError("target nodes must lie in [0, 1]")
    if source_nodes is None:
        source_nodes = regular_profile_nodes(len(profile))
    else:
        source_nodes = _finite_vector(source_nodes, name="source nodes")
    if len(source_nodes) != len(profile):
        raise ValueError("profile and source node dimensions differ")
    if len(source_nodes) > 1 and np.any(np.diff(source_nodes) <= 0.0):
        raise ValueError("source nodes must be strictly increasing")
    return np.interp(
        target_nodes,
        source_nodes,
        profile,
        left=float(profile[0]),
        right=float(profile[-1]),
    )


def percentile_ranks(values):
    """Deterministic average-tie percentile ranks in ``[0, 1]``."""

    values = _finite_vector(values, name="rank values")
    count = len(values)
    if count == 1:
        return np.zeros(1, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(count, dtype=float)
    start = 0
    while start < count:
        stop = start + 1
        while stop < count and values[order[stop]] == values[order[start]]:
            stop += 1
        average = 0.5 * float(start + stop - 1)
        ranks[order[start:stop]] = average / float(count - 1)
        start = stop
    return ranks


def farthest_first_indices(coordinates, count, *, initial_index=0):
    """Gonzalez farthest-first centers with deterministic tie breaking."""

    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or len(coordinates) < 1:
        raise ValueError("coordinates must be a nonempty matrix")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must be finite")
    count = int(count)
    if count < 1 or count > len(coordinates):
        raise ValueError("center count must lie between one and library size")
    initial_index = int(initial_index)
    if initial_index < 0 or initial_index >= len(coordinates):
        raise ValueError("initial center does not belong to the library")
    selected = [initial_index]
    minimum_distance = np.linalg.norm(
        coordinates - coordinates[initial_index][None, :], axis=1)
    minimum_distance[initial_index] = -np.inf
    while len(selected) < count:
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = np.linalg.norm(
            coordinates - coordinates[next_index][None, :], axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[np.asarray(selected, dtype=int)] = -np.inf
    return tuple(selected)


def covering_radius(coordinates, center_indices):
    """Maximum distance from a finite library to its nearest center."""

    coordinates = np.asarray(coordinates, dtype=float)
    centers = np.asarray(tuple(int(index) for index in center_indices), dtype=int)
    if coordinates.ndim != 2 or len(coordinates) < 1:
        raise ValueError("coordinates must be a nonempty matrix")
    if len(centers) < 1 or np.any(centers < 0) or np.any(centers >= len(coordinates)):
        raise ValueError("center indices must be nonempty and valid")
    distances = np.linalg.norm(
        coordinates[:, None, :] - coordinates[centers][None, :, :], axis=2)
    return float(np.max(np.min(distances, axis=1)))


def gonzalez_witness_certificate(coordinates, center_indices, *, tolerance=1e-10):
    """Return a finite, directly checkable farthest-first 2-approx certificate.

    For a nonzero covering radius, the selected ``k`` centers plus one farthest
    library point form ``k + 1`` witnesses separated by at least the achieved
    radius.  This is exactly the finite certificate consumed by the Lean
    factor-two theorem.  A zero-radius cover is recorded as a trivial optimum.
    """

    coordinates = np.asarray(coordinates, dtype=float)
    centers = tuple(int(index) for index in center_indices)
    tolerance = float(tolerance)
    if coordinates.ndim != 2 or len(coordinates) < 1:
        raise ValueError("coordinates must be a nonempty matrix")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must be finite")
    if (
        not centers
        or len(set(centers)) != len(centers)
        or min(centers) < 0
        or max(centers) >= len(coordinates)
    ):
        raise ValueError("center indices must be unique and valid")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("certificate tolerance must be finite and nonnegative")

    distance_to_centers = np.linalg.norm(
        coordinates[:, None, :] - coordinates[np.asarray(centers)][None, :, :],
        axis=2,
    )
    nearest = np.min(distance_to_centers, axis=1)
    witness_index = int(np.argmax(nearest))
    radius = float(nearest[witness_index])
    if radius <= tolerance:
        return {
            "contract_id": "gonzalez_farthest_first_witness_v1",
            "status": "zero_radius_optimum",
            "valid": True,
            "center_count": int(len(centers)),
            "witness_count": int(len(centers)),
            "farthest_witness_index": witness_index,
            "covering_radius": radius,
            "minimum_witness_pair_distance": 0.0,
            "minimum_insertion_radius": 0.0,
            "tolerance": tolerance,
        }

    insertion_radii = []
    for order in range(1, len(centers)):
        previous = np.asarray(centers[:order], dtype=int)
        insertion_radii.append(float(np.min(np.linalg.norm(
            coordinates[previous] - coordinates[centers[order]][None, :],
            axis=1,
        ))))
    witnesses = centers + (witness_index,)
    distinct = len(set(witnesses)) == len(witnesses)
    witness_coordinates = coordinates[np.asarray(witnesses, dtype=int)]
    pairwise = np.linalg.norm(
        witness_coordinates[:, None, :] - witness_coordinates[None, :, :],
        axis=2,
    )
    upper = pairwise[np.triu_indices(len(witnesses), k=1)]
    minimum_pairwise = float(np.min(upper))
    minimum_insertion = float(min(insertion_radii + [radius]))
    valid = bool(
        distinct
        and len(witnesses) == len(centers) + 1
        and minimum_pairwise + tolerance >= radius
        and minimum_insertion + tolerance >= radius
    )
    return {
        "contract_id": "gonzalez_farthest_first_witness_v1",
        "status": "certified" if valid else "invalid",
        "valid": valid,
        "center_count": int(len(centers)),
        "witness_count": int(len(witnesses)),
        "farthest_witness_index": witness_index,
        "covering_radius": radius,
        "minimum_witness_pair_distance": minimum_pairwise,
        "minimum_insertion_radius": minimum_insertion,
        "tolerance": tolerance,
    }


@dataclass(frozen=True)
class SourceProfileRecord:
    """Replicated outcomes for one shared profile in one source task."""

    task_id: str
    profile_id: str
    profile: tuple[float, ...]
    objective_samples: tuple[float, ...]
    constraint_samples: tuple[float, ...]
    alpha: float = 0.05
    tau: float = 0.0
    descriptor: tuple[float, ...] = ()
    nodes: tuple[float, ...] = ()


@dataclass(frozen=True)
class ProfileAtlasConfig:
    """Complete specification of the source-scored atlas."""

    n0: int = 10
    max_frequency: int = 8
    frequency_penalty: float = 0.25
    include_diagonal_quadratic: bool = True
    safety_metric_weight: float = 1.0
    objective_metric_weight: float = 1.0
    first_center_safety_weight: float = 0.5
    descriptor_temperature: float = 1.0
    variance_floor: float = 1e-6

    def validate(self):
        if int(self.n0) < 1:
            raise ValueError("n0 must be positive")
        if int(self.max_frequency) < 0:
            raise ValueError("max_frequency must be nonnegative")
        for name in (
            "frequency_penalty",
            "safety_metric_weight",
            "objective_metric_weight",
            "descriptor_temperature",
            "variance_floor",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and nonnegative")
        if float(self.descriptor_temperature) <= 0.0:
            raise ValueError("descriptor_temperature must be positive")
        weight = float(self.first_center_safety_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("first_center_safety_weight must lie in [0, 1]")
        return self


@dataclass(frozen=True)
class AtlasMember:
    profile_id: str
    profile: tuple[float, ...]
    nodes: tuple[float, ...]
    safety_rank: float
    objective_rank: float
    robust_source_feasible: bool
    selected_order: int


@dataclass(frozen=True)
class AtlasSelection:
    members: tuple[AtlasMember, ...]
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def target_profiles(self, dimension, *, target_nodes=None):
        if target_nodes is None:
            target_nodes = regular_profile_nodes(int(dimension))
        else:
            target_nodes = _finite_vector(target_nodes, name="target nodes")
            if len(target_nodes) != int(dimension):
                raise ValueError("target node count differs from dimension")
        return tuple(
            resample_profile(
                member.profile,
                target_nodes,
                source_nodes=(member.nodes or None),
            )
            for member in self.members
        )

    def target_integer_points(self, dimension, levels, *, target_nodes=None):
        levels = int(levels)
        if levels < 1:
            raise ValueError("levels must be positive")
        return tuple(
            tuple(np.clip(np.rint(profile * levels), 0, levels).astype(int))
            for profile in self.target_profiles(
                dimension, target_nodes=target_nodes)
        )


def _task_weights(task_descriptors, target_descriptor, temperature):
    task_ids = sorted(task_descriptors)
    if target_descriptor is None:
        return {task_id: 1.0 / len(task_ids) for task_id in task_ids}
    target = _finite_vector(target_descriptor, name="target descriptor")
    matrix = np.vstack([
        _finite_vector(task_descriptors[task_id], name="source descriptor")
        for task_id in task_ids
    ])
    if matrix.shape[1] != len(target):
        raise ValueError("source and target descriptor dimensions differ")
    scale = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    squared = np.sum(((matrix - target[None, :]) / scale[None, :]) ** 2, axis=1)
    logits = -0.5 * squared / float(temperature) ** 2
    logits -= float(np.max(logits))
    values = np.exp(logits)
    values /= float(np.sum(values))
    return {task_id: float(value) for task_id, value in zip(task_ids, values)}


class SourceScoredProfileAtlas:
    """Fit and select one source-scored finite profile atlas."""

    contract_id = "source_scored_profile_atlas_v2"

    def __init__(self, config=None):
        self.config = (config or ProfileAtlasConfig()).validate()
        self.selection = None

    def fit(self, records: Iterable[SourceProfileRecord], *, target_descriptor=None):
        records = tuple(records)
        if not records:
            raise ValueError("source atlas requires at least one record")
        by_task = {}
        by_profile = {}
        task_descriptors = {}
        task_profile_ids = {}
        for record in records:
            if not isinstance(record, SourceProfileRecord):
                raise TypeError("source atlas records must be SourceProfileRecord")
            task_id = str(record.task_id)
            profile_id = str(record.profile_id)
            if not task_id or not profile_id:
                raise ValueError("task_id and profile_id must be nonempty")
            profile = _finite_vector(record.profile, name="source profile")
            if np.any(profile < 0.0) or np.any(profile > 1.0):
                raise ValueError("source profiles must lie in [0, 1]")
            objective = _finite_vector(
                record.objective_samples, name="objective samples")
            constraint = _finite_vector(
                record.constraint_samples, name="constraint samples")
            if len(objective) != len(constraint):
                raise ValueError("objective and constraint replication counts differ")
            if not 0.0 < float(record.alpha) < 1.0:
                raise ValueError("source alpha must lie in (0, 1)")
            nodes = (
                regular_profile_nodes(len(profile))
                if not record.nodes
                else _finite_vector(record.nodes, name="source nodes")
            )
            if len(nodes) != len(profile):
                raise ValueError("source profile and node dimensions differ")
            descriptor = tuple(float(value) for value in record.descriptor)
            existing_descriptor = task_descriptors.setdefault(task_id, descriptor)
            if existing_descriptor != descriptor:
                raise ValueError("a source task has inconsistent descriptors")
            key = (task_id, profile_id)
            if key in by_task:
                raise ValueError("duplicate task/profile source record")
            variance = (
                float(np.var(constraint, ddof=1))
                if len(constraint) > 1 else 0.0
            )
            sigma = float(np.sqrt(max(variance, self.config.variance_floor)))
            margin = float(
                np.mean(constraint)
                + norm.ppf(1.0 - float(record.alpha)) * sigma
                - float(record.tau)
            )
            row = {
                "profile": profile.copy(),
                "nodes": nodes.copy(),
                "objective": float(np.mean(objective)),
                "margin": margin,
                "replications": int(len(objective)),
            }
            by_task[key] = row
            by_profile.setdefault(profile_id, []).append((task_id, row))
            task_profile_ids.setdefault(task_id, set()).add(profile_id)

        task_ids = sorted(task_profile_ids)
        common_profile_ids = set.intersection(*(
            task_profile_ids[task_id] for task_id in task_ids
        ))
        if not common_profile_ids:
            raise ValueError("source tasks share no common profile library")
        if any(task_profile_ids[task_id] != common_profile_ids for task_id in task_ids):
            raise ValueError(
                "ProfileAtlasV2 requires the same profile library in every source task")
        profile_ids = sorted(common_profile_ids)
        if int(self.config.n0) > len(profile_ids):
            raise ValueError("n0 exceeds the shared source profile library")

        source_weights = _task_weights(
            task_descriptors,
            None if target_descriptor is None else tuple(target_descriptor),
            self.config.descriptor_temperature,
        )
        task_ranks = {}
        for task_id in task_ids:
            objective = np.asarray([
                by_task[(task_id, profile_id)]["objective"]
                for profile_id in profile_ids
            ])
            margin = np.asarray([
                by_task[(task_id, profile_id)]["margin"]
                for profile_id in profile_ids
            ])
            task_ranks[task_id] = {
                "objective": percentile_ranks(objective),
                "safety": percentile_ranks(margin),
                "margin": margin,
            }
        safety_rank = np.asarray([
            sum(
                source_weights[task_id]
                * task_ranks[task_id]["safety"][index]
                for task_id in task_ids
            )
            for index in range(len(profile_ids))
        ])
        objective_rank = np.asarray([
            sum(
                source_weights[task_id]
                * task_ranks[task_id]["objective"][index]
                for task_id in task_ids
            )
            for index in range(len(profile_ids))
        ])
        robust = np.asarray([
            all(
                task_ranks[task_id]["margin"][index] <= 0.0
                for task_id in task_ids
            )
            for index in range(len(profile_ids))
        ], dtype=bool)

        profiles = []
        nodes = []
        profile_coordinates = []
        for profile_id in profile_ids:
            reference = by_task[(task_ids[0], profile_id)]
            for task_id in task_ids[1:]:
                other = by_task[(task_id, profile_id)]
                if len(other["profile"]) != len(reference["profile"]) or not np.allclose(
                    other["profile"], reference["profile"], atol=1e-12, rtol=0.0
                ):
                    raise ValueError("shared profile values differ across source tasks")
                if not np.allclose(other["nodes"], reference["nodes"], atol=1e-12, rtol=0.0):
                    raise ValueError("shared profile nodes differ across source tasks")
            profiles.append(reference["profile"])
            nodes.append(reference["nodes"])
            profile_coordinates.append(profile_cosine_coordinate(
                reference["profile"],
                nodes=reference["nodes"],
                max_frequency=self.config.max_frequency,
                frequency_penalty=self.config.frequency_penalty,
                include_diagonal_quadratic=self.config.include_diagonal_quadratic,
            ))
        profile_coordinates = np.vstack(profile_coordinates)
        scale = np.std(profile_coordinates, axis=0)
        scale = np.where(scale > 1e-10, scale, 1.0)
        standardized = (profile_coordinates - np.mean(
            profile_coordinates, axis=0, keepdims=True)) / scale[None, :]
        augmented = np.column_stack([
            standardized,
            np.sqrt(float(self.config.safety_metric_weight)) * safety_rank,
            np.sqrt(float(self.config.objective_metric_weight)) * objective_rank,
        ])
        weight = float(self.config.first_center_safety_weight)
        balanced = weight * safety_rank + (1.0 - weight) * objective_rank
        initial_index = min(
            range(len(profile_ids)),
            key=lambda index: (
                float(balanced[index]),
                float(safety_rank[index]),
                float(objective_rank[index]),
                profile_ids[index],
            ),
        )
        selected = farthest_first_indices(
            augmented, self.config.n0, initial_index=initial_index)
        gonzalez_certificate = gonzalez_witness_certificate(
            augmented, selected)
        if not gonzalez_certificate["valid"]:
            raise RuntimeError("farthest-first witness certificate failed")
        members = tuple(
            AtlasMember(
                profile_id=profile_ids[index],
                profile=tuple(float(value) for value in profiles[index]),
                nodes=tuple(float(value) for value in nodes[index]),
                safety_rank=float(safety_rank[index]),
                objective_rank=float(objective_rank[index]),
                robust_source_feasible=bool(robust[index]),
                selected_order=order,
            )
            for order, index in enumerate(selected, start=1)
        )
        record_digest = hashlib.sha256(json.dumps([
            {
                "task_id": record.task_id,
                "profile_id": record.profile_id,
                "profile": list(record.profile),
                "objective_samples": list(record.objective_samples),
                "constraint_samples": list(record.constraint_samples),
                "alpha": record.alpha,
                "tau": record.tau,
                "descriptor": list(record.descriptor),
                "nodes": list(record.nodes),
            }
            for record in records
        ], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        diagnostics = {
            "contract_id": self.contract_id,
            "source_only": True,
            "target_outcomes_used": False,
            "target_oracle_used": False,
            "source_task_count": int(len(task_ids)),
            "shared_library_size": int(len(profile_ids)),
            "source_replications_min": int(min(
                row["replications"] for row in by_task.values())),
            "source_replications_max": int(max(
                row["replications"] for row in by_task.values())),
            "source_task_weights": dict(source_weights),
            "descriptor_conditioned": bool(target_descriptor is not None),
            "selected_profile_ids": [member.profile_id for member in members],
            "selected_safety_ranks": [member.safety_rank for member in members],
            "selected_objective_ranks": [member.objective_rank for member in members],
            "robust_source_feasible_selected": int(sum(
                member.robust_source_feasible for member in members)),
            "coordinate_dimension": int(augmented.shape[1]),
            "covering_radius": covering_radius(augmented, selected),
            "gonzalez_witness_certificate": gonzalez_certificate,
            "first_center_rule": "minimum_weighted_source_rank",
            "remaining_center_rule": "gonzalez_farthest_first",
            "tie_breaking": "stable_profile_id_then_lowest_index",
            "source_archive_sha256": record_digest,
            "config": {
                name: getattr(self.config, name)
                for name in self.config.__dataclass_fields__
            },
        }
        self.selection = AtlasSelection(members, diagnostics)
        return self

    def selected(self):
        if self.selection is None:
            raise RuntimeError("source atlas has not been fit")
        return self.selection


def generic_dct_maximin(
    profiles: Sequence[Sequence[float]],
    count,
    *,
    nodes=None,
    max_frequency=8,
    frequency_penalty=0.25,
    include_diagonal_quadratic=True,
):
    """Outcome-free structured control on the same finite profile library."""

    rows = tuple(_finite_vector(profile, name="profile") for profile in profiles)
    if not rows:
        raise ValueError("generic DCT design requires a nonempty library")
    if nodes is None:
        node_rows = tuple(regular_profile_nodes(len(profile)) for profile in rows)
    else:
        shared_nodes = _finite_vector(nodes, name="profile nodes")
        node_rows = tuple(shared_nodes for _ in rows)
    coordinates = np.vstack([
        profile_cosine_coordinate(
            profile,
            nodes=node_row,
            max_frequency=max_frequency,
            frequency_penalty=frequency_penalty,
            include_diagonal_quadratic=include_diagonal_quadratic,
        )
        for profile, node_row in zip(rows, node_rows)
    ])
    distances = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    initial_index = int(np.argmin(np.mean(distances, axis=1)))
    selected = farthest_first_indices(
        coordinates, int(count), initial_index=initial_index)
    gonzalez_certificate = gonzalez_witness_certificate(coordinates, selected)
    if not gonzalez_certificate["valid"]:
        raise RuntimeError("generic DCT farthest-first certificate failed")
    return selected, {
        "contract_id": "generic_dct_maximin_v1",
        "source_outcomes_used": False,
        "target_outcomes_used": False,
        "target_oracle_used": False,
        "first_center_rule": "finite_library_medoid",
        "remaining_center_rule": "gonzalez_farthest_first",
        "selected_indices": list(selected),
        "covering_radius": covering_radius(coordinates, selected),
        "gonzalez_witness_certificate": gonzalez_certificate,
        "coordinate_dimension": int(coordinates.shape[1]),
    }
