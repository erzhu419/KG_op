"""Randomized ordered-profile tasks for benchmark-overfitting audits.

The generator deliberately includes aligned, shifted, and misspecified task
families.  Nominal dimension and latent complexity are separate parameters.
Every task exposes ordinary simulation calls and analytic truth only for
post-run audit.  Algorithms may use the declared grid/schema but never the
hidden basis, centers, radius, or task seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import norm

from core.profile_atlas import (
    SourceProfileRecord,
    regular_profile_nodes,
    resample_profile,
)


PROFILE_STRESS_REGIMES = {
    "aligned_low_frequency": {
        "basis": "cosine",
        "default_rank": 8,
        "transfer": "aligned",
        "grid": "regular",
        "permuted": False,
    },
    "growing_effective_rank": {
        "basis": "cosine",
        "default_rank": 24,
        "transfer": "aligned",
        "grid": "regular",
        "permuted": False,
    },
    "frequency_support_shift": {
        "basis": "cosine",
        "default_rank": 8,
        "transfer": "frequency_shift",
        "grid": "regular",
        "permuted": False,
    },
    "coordinate_permutation": {
        "basis": "cosine",
        "default_rank": 8,
        "transfer": "aligned",
        "grid": "regular",
        "permuted": True,
    },
    "irregular_grid": {
        "basis": "cosine",
        "default_rank": 8,
        "transfer": "aligned",
        "grid": "irregular",
        "permuted": False,
    },
    "piecewise_smooth": {
        "basis": "piecewise",
        "default_rank": 12,
        "transfer": "aligned",
        "grid": "regular",
        "permuted": False,
    },
    "sparse_high_frequency": {
        "basis": "sparse",
        "default_rank": 16,
        "transfer": "aligned",
        "grid": "regular",
        "permuted": False,
    },
    "misspecified_target": {
        "basis": "cosine",
        "default_rank": 24,
        "transfer": "independent_target",
        "grid": "regular",
        "permuted": False,
    },
}


CALIBRATION_LIBRARY_SIZE = 64
CALIBRATION_LIBRARY_DIMENSION = 128
CALIBRATION_LIBRARY_SEED_OFFSET = 37
CALIBRATION_LIBRARY_MAXIMUM_FREQUENCY = 32


@dataclass(frozen=True)
class StructuralProfile:
    profile_id: str
    values: tuple[float, ...]
    nodes: tuple[float, ...]
    family: str


def _normalized_irregular_nodes(dimension, rng):
    widths = rng.lognormal(mean=0.0, sigma=0.65, size=int(dimension))
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    edges /= float(edges[-1])
    return 0.5 * (edges[:-1] + edges[1:])


def _profile_weights(nodes):
    nodes = np.asarray(nodes, dtype=float)
    if len(nodes) == 1:
        return np.ones(1, dtype=float)
    edges = np.empty(len(nodes) + 1, dtype=float)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges[1:-1] = 0.5 * (nodes[:-1] + nodes[1:])
    return np.diff(edges)


def _cosine_profile(nodes, coefficients, *, start_frequency=1):
    values = np.full(len(nodes), 0.5, dtype=float)
    for offset, coefficient in enumerate(coefficients):
        frequency = int(start_frequency + offset)
        values += float(coefficient) * np.cos(np.pi * frequency * nodes)
    return np.clip(values, 0.0, 1.0)


def generate_structural_profile_library(
    count=64,
    *,
    dimension=128,
    seed=20260808,
    maximum_frequency=32,
):
    """Generate a fixed broad library without task outcomes."""

    count = int(count)
    dimension = int(dimension)
    maximum_frequency = int(maximum_frequency)
    if count < 10 or dimension < 8 or maximum_frequency < 4:
        raise ValueError("profile library specification is too small")
    rng = np.random.default_rng(int(seed))
    nodes = regular_profile_nodes(dimension)
    rows = []

    def add(values, family):
        values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
        key = tuple(np.rint(values * 1e8).astype(np.int64))
        if any(item[0] == key for item in rows):
            return
        rows.append((key, values.copy(), str(family)))

    for level in np.linspace(0.05, 0.95, 10):
        add(np.full(dimension, level), "constant")
    for start, stop in (
        (0.05, 0.95), (0.95, 0.05), (0.20, 0.80), (0.80, 0.20),
        (0.35, 0.65), (0.65, 0.35),
    ):
        add(np.linspace(start, stop, dimension), "ramp")

    family_cycle = ("low_frequency", "piecewise", "high_frequency")
    cursor = 0
    while len(rows) < count:
        family = family_cycle[cursor % len(family_cycle)]
        cursor += 1
        if family == "low_frequency":
            rank = int(rng.integers(2, 9))
            coefficients = rng.normal(0.0, 0.12, size=rank) / np.sqrt(
                np.arange(1, rank + 1))
            baseline = float(rng.uniform(0.25, 0.75))
            values = _cosine_profile(nodes, coefficients)
            values = np.clip(values + baseline - 0.5, 0.0, 1.0)
        elif family == "piecewise":
            blocks = int(rng.integers(3, 13))
            levels = rng.uniform(0.08, 0.92, size=blocks)
            values = levels[np.minimum(
                (nodes * blocks).astype(int), blocks - 1)]
            smoothing = max(1, dimension // 64)
            if smoothing > 1:
                kernel = np.ones(smoothing) / float(smoothing)
                values = np.convolve(values, kernel, mode="same")
        else:
            frequency = int(rng.integers(9, maximum_frequency + 1))
            values = (
                float(rng.uniform(0.30, 0.70))
                + float(rng.uniform(0.08, 0.25))
                * np.cos(np.pi * frequency * nodes + rng.uniform(0.0, 2.0 * np.pi))
            )
        add(values, family)
    return tuple(
        StructuralProfile(
            profile_id=f"profile_{index:04d}",
            values=tuple(float(value) for value in values),
            nodes=tuple(float(value) for value in nodes),
            family=family,
        )
        for index, (_, values, family) in enumerate(rows[:count])
    )


def generate_calibration_profile_library(family_seed):
    """Return the outcome-free library used to calibrate task safe mass.

    Keeping this construction public and deterministic lets post-run auditors
    use the same fixed reference set without depending on the source-design
    library or any target observations.
    """

    return generate_structural_profile_library(
        CALIBRATION_LIBRARY_SIZE,
        dimension=CALIBRATION_LIBRARY_DIMENSION,
        seed=int(family_seed) + CALIBRATION_LIBRARY_SEED_OFFSET,
        maximum_frequency=CALIBRATION_LIBRARY_MAXIMUM_FREQUENCY,
    )


class RandomizedOrderedProfileProblem:
    """Chance-constrained profile problem with controlled latent complexity."""

    simulation_noise_model = "iid_gaussian"
    verification_distribution_scope = "task_specific_iid_gaussian_replications"

    def __init__(
        self,
        *,
        regime,
        role,
        task_seed,
        family_seed=20260808,
        d=1000,
        L=100,
        alpha=0.05,
        active_rank=None,
        safe_mass=0.08,
        calibration_library=None,
    ):
        regime = str(regime)
        role = str(role)
        if regime not in PROFILE_STRESS_REGIMES:
            raise ValueError(f"unknown randomized profile regime: {regime}")
        if role not in {"source", "target"}:
            raise ValueError("task role must be source or target")
        self.regime = regime
        self.role = role
        self.task_seed = int(task_seed)
        self.family_seed = int(family_seed)
        self.d = int(d)
        self.L = int(L)
        self.alpha = float(alpha)
        self.safe_mass = float(safe_mass)
        self.tau = 0.0
        if (
            self.d < 8 or self.L < 2
            or not 0.0 < self.alpha < 1.0
            or not 0.0 < self.safe_mass < 0.5
        ):
            raise ValueError("invalid randomized profile task dimensions")
        spec = PROFILE_STRESS_REGIMES[regime]
        self.active_rank = int(active_rank or spec["default_rank"])
        if self.active_rank < 1 or self.active_rank > self.d:
            raise ValueError("active rank must lie in [1, d]")
        grid_rng = np.random.default_rng(np.random.SeedSequence([
            self.family_seed, self.task_seed, 1701,
        ]))
        self.nodes = (
            _normalized_irregular_nodes(self.d, grid_rng)
            if spec["grid"] == "irregular"
            else regular_profile_nodes(self.d)
        )
        self.weights = _profile_weights(self.nodes)
        permutation_rng = np.random.default_rng(np.random.SeedSequence([
            self.family_seed, self.task_seed, 1709,
        ]))
        self.semantic_to_raw = (
            permutation_rng.permutation(self.d)
            if spec["permuted"] and self.role == "target"
            else np.arange(self.d, dtype=int)
        )
        self._basis = self._build_basis(spec)
        self.effective_rank = int(self._basis.shape[0])
        self._safe_profile, self._objective_profile = self._build_centers(spec)
        self._safe_center = self._features_semantic(self._safe_profile)
        self._objective_center = self._features_semantic(self._objective_profile)
        calibration_library = (
            generate_calibration_profile_library(self.family_seed)
            if calibration_library is None else tuple(calibration_library)
        )
        distances = []
        objective_distances = []
        for item in calibration_library:
            semantic = resample_profile(
                item.values,
                self.nodes,
                source_nodes=item.nodes,
            )
            features = self._features_semantic(semantic)
            distances.append(self._distance(features, self._safe_center))
            objective_distances.append(self._distance(
                features, self._objective_center))
        self.safe_radius = max(float(np.quantile(
            distances, self.safe_mass)), 1e-4)
        self.objective_scale = max(float(np.median(objective_distances)), 1e-4)
        self.problem_name = (
            f"RandomizedProfile[{self.regime}:{self.role}:"
            f"rank{self.effective_rank}:task{self.task_seed}]"
        )

    def _build_basis(self, spec):
        if spec["basis"] == "piecewise":
            block = np.minimum(
                (self.nodes * self.active_rank).astype(int),
                self.active_rank - 1,
            )
            basis = np.zeros((self.active_rank, self.d), dtype=float)
            for index in range(self.active_rank):
                mask = block == index
                mass = float(np.sum(self.weights[mask]))
                basis[index, mask] = 1.0 / max(mass, 1e-12)
            return basis
        if spec["basis"] == "sparse":
            rng = np.random.default_rng(np.random.SeedSequence([
                self.family_seed, 1741,
            ]))
            indices = np.sort(rng.choice(
                self.d, size=self.active_rank, replace=False))
            basis = np.zeros((self.active_rank, self.d), dtype=float)
            for row, index in enumerate(indices):
                basis[row, index] = 1.0 / max(float(self.weights[index]), 1e-12)
            self.sparse_semantic_indices = tuple(int(index) for index in indices)
            return basis
        transfer = spec["transfer"]
        if transfer == "frequency_shift" and self.role == "target":
            frequencies = np.arange(9, 9 + self.active_rank, dtype=float)
        elif transfer == "independent_target" and self.role == "target":
            frequencies = np.arange(17, 17 + self.active_rank, dtype=float)
        else:
            frequencies = np.arange(self.active_rank, dtype=float)
        basis = np.cos(np.pi * frequencies[:, None] * self.nodes[None, :])
        if len(basis) > 1 and frequencies[0] == 0.0:
            basis[1:] *= np.sqrt(2.0)
        else:
            basis *= np.sqrt(2.0)
        return basis

    def _build_centers(self, spec):
        family_rng = np.random.default_rng(np.random.SeedSequence([
            self.family_seed, 1777,
        ]))
        task_rng = np.random.default_rng(np.random.SeedSequence([
            self.family_seed, self.task_seed, 1783,
        ]))
        if spec["basis"] == "piecewise":
            levels = family_rng.uniform(0.25, 0.75, size=self.active_rank)
            block = np.minimum(
                (self.nodes * self.active_rank).astype(int),
                self.active_rank - 1,
            )
            safe = levels[block]
        elif spec["basis"] == "sparse":
            safe = np.full(self.d, 0.52, dtype=float)
            safe[list(self.sparse_semantic_indices)] = family_rng.uniform(
                0.20, 0.80, size=self.active_rank)
        else:
            coefficients = family_rng.normal(0.0, 0.10, size=8) / np.sqrt(
                np.arange(1, 9))
            safe = _cosine_profile(self.nodes, coefficients)
        transfer = spec["transfer"]
        if transfer == "frequency_shift" and self.role == "target":
            shifted = task_rng.normal(0.0, 0.10, size=min(self.active_rank, 12))
            safe = np.clip(
                safe + _cosine_profile(
                    self.nodes, shifted, start_frequency=9) - 0.5,
                0.0,
                1.0,
            )
        elif transfer == "independent_target" and self.role == "target":
            independent = task_rng.normal(0.0, 0.16, size=min(
                self.active_rank, 24))
            safe = _cosine_profile(
                self.nodes, independent, start_frequency=17)
        else:
            safe = np.clip(
                safe
                + task_rng.normal(0.0, 0.015)
                + 0.012 * np.cos(
                    np.pi * (1 + self.task_seed % 5) * self.nodes),
                0.0,
                1.0,
            )
        objective = np.clip(
            safe - 0.10 + 0.05 * np.cos(2.0 * np.pi * self.nodes),
            0.0,
            1.0,
        )
        return safe, objective

    def _features_semantic(self, semantic_profile):
        semantic_profile = np.asarray(semantic_profile, dtype=float).reshape(-1)
        if len(semantic_profile) != self.d:
            raise ValueError("semantic profile dimension mismatch")
        return self._basis @ (self.weights * semantic_profile)

    @staticmethod
    def _distance(first, second):
        difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
        return float(np.sqrt(np.mean(difference ** 2)))

    def int_bounds(self):
        return np.zeros(self.d, dtype=int), np.full(self.d, self.L, dtype=int)

    def normalize(self, x):
        values = np.asarray(x, dtype=float).reshape(-1)
        if len(values) != self.d:
            raise ValueError(f"expected {self.d} policy entries")
        return np.clip(values / float(self.L), 0.0, 1.0)

    def continuous_to_int(self, x_norm):
        values = np.asarray(x_norm, dtype=float).reshape(-1)
        if len(values) != self.d:
            raise ValueError(f"expected {self.d} normalized entries")
        return tuple(np.clip(np.rint(values * self.L), 0, self.L).astype(int))

    def sample_random(self, rng=None):
        rng = rng or np.random.default_rng()
        return tuple(int(value) for value in rng.integers(
            0, self.L + 1, size=self.d))

    def semantic_profile(self, x):
        raw = self.normalize(x)
        return raw[self.semantic_to_raw]

    def encode_semantic_profile(self, semantic_profile):
        semantic = np.clip(
            np.asarray(semantic_profile, dtype=float).reshape(-1), 0.0, 1.0)
        if len(semantic) != self.d:
            raise ValueError("semantic profile dimension mismatch")
        raw = np.empty(self.d, dtype=float)
        raw[self.semantic_to_raw] = semantic
        return self.continuous_to_int(raw)

    def point_from_structural_profile(self, profile, *, schema_mode="declared"):
        if not isinstance(profile, StructuralProfile):
            raise TypeError("expected a StructuralProfile")
        mode = str(schema_mode)
        if mode == "declared":
            semantic = resample_profile(
                profile.values, self.nodes, source_nodes=profile.nodes)
            return self.encode_semantic_profile(semantic)
        if mode == "schema_blind":
            raw = resample_profile(
                profile.values,
                regular_profile_nodes(self.d),
                source_nodes=profile.nodes,
            )
            return self.continuous_to_int(raw)
        raise ValueError("schema_mode must be declared or schema_blind")

    def latent_features(self, x):
        return self._features_semantic(self.semantic_profile(x))

    def true_chance_margin(self, x):
        return float(
            self._distance(self.latent_features(x), self._safe_center)
            - self.safe_radius
        )

    def true_sigma(self, x):
        features = self.latent_features(x)
        modulation = float(np.mean(np.abs(features)))
        constraint_sigma = float(np.clip(
            0.012 + 0.030 * modulation, 0.012, 0.065))
        return np.asarray([0.006, constraint_sigma], dtype=float)

    def true_constraint_mean(self, x):
        sigma = float(self.true_sigma(x)[1])
        return float(
            self.true_chance_margin(x)
            - norm.ppf(1.0 - self.alpha) * sigma
        )

    def true_objective(self, x):
        return float(
            self._distance(self.latent_features(x), self._objective_center)
            / self.objective_scale
        )

    def true_outputs(self, x):
        return np.asarray([
            self.true_objective(x), self.true_constraint_mean(x)], dtype=float)

    def true_feasibility_probability(self, x):
        sigma = float(self.true_sigma(x)[1])
        return float(norm.cdf(
            (self.tau - self.true_constraint_mean(x)) / sigma))

    def is_truly_feasible(self, x):
        return bool(self.true_chance_margin(x) <= 0.0)

    def simulate(self, x, rng=None):
        rng = rng or np.random.default_rng()
        return self.true_outputs(x) + rng.normal(0.0, self.true_sigma(x))

    def observable_descriptor(self):
        spacing = np.diff(np.concatenate([[0.0], self.nodes, [1.0]]))
        return (
            float(np.log1p(self.d)),
            float(self.active_rank),
            float(np.std(spacing) / max(float(np.mean(spacing)), 1e-12)),
            float(self.regime == "piecewise_smooth"),
            float(self.regime == "sparse_high_frequency"),
        )

    def information_contract(self):
        return {
            "problem": self.problem_name,
            "regime": self.regime,
            "role": self.role,
            "nominal_dimension": self.d,
            "effective_rank": self.effective_rank,
            "safe_mass_calibration_quantile": self.safe_mass,
            "chance_alpha": self.alpha,
            "grid_declared": True,
            "channel_permutation_declared": bool(
                PROFILE_STRESS_REGIMES[self.regime]["permuted"]),
            "hidden_basis_used_by_algorithm": False,
            "hidden_centers_used_by_algorithm": False,
            "hidden_radius_used_by_algorithm": False,
            "task_seed_used_by_algorithm": False,
            "target_oracle_available_to_algorithm": False,
        }


def source_profile_records(
    source_tasks: Sequence[RandomizedOrderedProfileProblem],
    library: Sequence[StructuralProfile],
    *,
    replications=3,
    seed=20260808,
):
    """Evaluate a shared library on source tasks using ordinary simulations."""

    replications = int(replications)
    if replications < 1:
        raise ValueError("source replications must be positive")
    rows = []
    for task_index, task in enumerate(source_tasks):
        if task.role != "source":
            raise ValueError("source archive received a non-source task")
        for profile_index, profile in enumerate(library):
            point = task.point_from_structural_profile(profile)
            values = []
            for replication in range(replications):
                rng = np.random.default_rng(np.random.SeedSequence([
                    int(seed), task_index, profile_index, replication, 1877,
                ]))
                values.append(task.simulate(point, rng))
            values = np.vstack(values)
            rows.append(SourceProfileRecord(
                task_id=task.problem_name,
                profile_id=profile.profile_id,
                profile=profile.values,
                objective_samples=tuple(float(value) for value in values[:, 0]),
                constraint_samples=tuple(float(value) for value in values[:, 1]),
                alpha=task.alpha,
                tau=task.tau,
                descriptor=task.observable_descriptor(),
                nodes=profile.nodes,
            ))
    return tuple(rows)
