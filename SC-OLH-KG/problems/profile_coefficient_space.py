"""Target-only low-frequency coordinates for ordered policy profiles."""

from __future__ import annotations

import math

import numpy as np


class CosineCoefficientProfileProblem:
    """Expose a raw profile simulator through fixed cosine coefficients.

    This is a deliberately source-free functional-search baseline.  The
    coefficient box and inverse map are fixed before target outcomes are seen;
    only the target simulator observations are used by the optimizer.
    """

    contract_id = "target_only_cosine_coefficient_space_v1"
    simulation_noise_model = "delegated_to_target_problem"

    def __init__(
        self,
        target,
        *,
        coefficient_count=8,
        coefficient_scale=0.25,
        level_bounds=(0.05, 0.95),
        schema_mode="declared",
        lattice_level=100,
        nominal_sigma=0.04,
    ):
        self.target = target
        self.coefficient_count = int(coefficient_count)
        self.coefficient_scale = float(coefficient_scale)
        self.level_bounds = tuple(float(value) for value in level_bounds)
        self.schema_mode = str(schema_mode)
        self.L = int(lattice_level)
        self.d = self.coefficient_count + 1
        self.alpha = float(target.alpha)
        self.tau = float(target.tau)
        self.sigma_level = float(nominal_sigma)
        if self.coefficient_count < 1:
            raise ValueError("coefficient_count must be positive")
        if self.coefficient_scale <= 0.0:
            raise ValueError("coefficient_scale must be positive")
        if (
            len(self.level_bounds) != 2
            or not 0.0 <= self.level_bounds[0] < self.level_bounds[1] <= 1.0
        ):
            raise ValueError("level_bounds must be an increasing subset of [0, 1]")
        if self.schema_mode not in {"declared", "schema_blind"}:
            raise ValueError("schema_mode must be declared or schema_blind")
        if self.L < 2:
            raise ValueError("lattice_level must be at least two")
        if not hasattr(target, "nodes"):
            raise TypeError("target problem must expose ordered profile nodes")
        self.problem_name = (
            f"CosineCoefficient[{target.problem_name}:K{self.coefficient_count}]"
        )

    def int_bounds(self):
        return (
            np.zeros(self.d, dtype=int),
            np.full(self.d, self.L, dtype=int),
        )

    def normalize(self, x):
        values = np.asarray(x, dtype=float).reshape(-1)
        if len(values) != self.d:
            raise ValueError(f"expected {self.d} coefficient entries")
        return np.clip(values / float(self.L), 0.0, 1.0)

    def continuous_to_int(self, x_norm):
        values = np.asarray(x_norm, dtype=float).reshape(-1)
        if len(values) != self.d:
            raise ValueError(f"expected {self.d} normalized entries")
        return tuple(np.clip(
            np.rint(values * self.L), 0, self.L,
        ).astype(int))

    def semantic_profile(self, x):
        unit = self.normalize(x)
        lower, upper = self.level_bounds
        profile = np.full(
            len(self.target.nodes),
            lower + (upper - lower) * float(unit[0]),
            dtype=float,
        )
        for frequency in range(1, self.coefficient_count + 1):
            amplitude = (
                self.coefficient_scale
                * (2.0 * float(unit[frequency]) - 1.0)
                / math.sqrt(float(frequency))
            )
            profile += (
                math.sqrt(2.0)
                * amplitude
                * np.cos(np.pi * frequency * self.target.nodes)
            )
        return np.clip(profile, 0.0, 1.0)

    def raw_point(self, x):
        semantic = self.semantic_profile(x)
        if self.schema_mode == "declared":
            return tuple(self.target.encode_semantic_profile(semantic))
        return tuple(self.target.continuous_to_int(semantic))

    def simulate(self, x, rng=None):
        return self.target.simulate(self.raw_point(x), rng)

    def information_contract(self):
        return {
            "contract_id": self.contract_id,
            "source_outcomes_used": False,
            "source_archive_used": False,
            "target_outcomes_used_to_define_coordinate": False,
            "target_oracle_used": False,
            "schema_mode": self.schema_mode,
            "coefficient_count": self.coefficient_count,
            "coefficient_scale": self.coefficient_scale,
            "level_bounds": list(self.level_bounds),
            "frequency_amplitude_decay": "inverse_sqrt_frequency",
            "raw_profile_dimension": int(self.target.d),
            "coefficient_dimension": int(self.d),
            "nominal_aleatoric_sigma": float(self.sigma_level),
        }
