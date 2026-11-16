"""Core data contracts and strict oracle-isolation interfaces for `v2_or`.

All benchmark families in `v2_or` normalize constraint outputs as `Y_g - tau` so
that chance-constraint feasibility is universally:

    P(Y_g_centered(x) <= 0) >= 1 - alpha.

Crucially, `nominal_exposure(x)` MUST depend exclusively on public exogenous
forecasts and nominal physical dynamics—never on hidden stochastic shocks or
private task parameters. This invariance is enforced by automated unit tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NominalTrajectoryExposure:
    """Dimension-invariant state-risk exposure extracted from a nominal rollout.

    Attributes
    ----------
    psi : np.ndarray
        Combined public trajectory feature vector in R^{p_psi}, invariant to the
        raw policy dimension `d` so policies from horizons `d_s != d_t` map to
        the exact same functional coordinate space.
    A_local : np.ndarray
        Nonnegative local/idiosyncratic trajectory exposure amplitudes in R^{p_A}
        entering the HVD term `A_local^T diag(lambda) A_local`.
    N_shared : np.ndarray
        Nonnegative regime/shared-shock occupancy exposures in R^{p_N} entering
        the HVD terms `N_shared^T B N_shared + N_shared^T omega`.
    feature_names : tuple[str, ...]
        Human-readable names for `psi` coordinates.
    """

    psi: np.ndarray
    A_local: np.ndarray
    N_shared: np.ndarray
    feature_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        psi = np.asarray(self.psi, dtype=float).reshape(-1)
        a_loc = np.asarray(self.A_local, dtype=float).reshape(-1)
        n_shr = np.asarray(self.N_shared, dtype=float).reshape(-1)
        if not (np.all(np.isfinite(psi)) and np.all(np.isfinite(a_loc)) and np.all(np.isfinite(n_shr))):
            raise ValueError("NominalTrajectoryExposure contains non-finite values")
        object.__setattr__(self, "psi", psi)
        object.__setattr__(self, "A_local", a_loc)
        object.__setattr__(self, "N_shared", n_shr)
        if not self.feature_names:
            object.__setattr__(
                self,
                "feature_names",
                tuple(f"psi_{i}" for i in range(psi.size)),
            )


@dataclass(frozen=True)
class SufficientStatsRecord:
    """Decoupled within-policy sufficient statistics immune to mean-shift leakage.

    Given `m` replications at policy `x`, the within-policy sum of squared
    deviations `ssq_con = sum_{r=1}^m (Y_{g,r} - mean_con)^2` satisfies
    `E[ssq_con] = (m - 1) * v_g(x)` with ZERO dependence on any source-target
    constraint mean shift `Delta_g(x)`.
    """

    x_norm: np.ndarray
    n_reps: int
    mean_obj: float
    ssq_obj: float
    mean_con: float
    ssq_con: float
    violation_count: int
    exposure: NominalTrajectoryExposure

    @classmethod
    def from_samples(
        cls,
        x_norm: np.ndarray,
        samples: np.ndarray,
        exposure: NominalTrajectoryExposure,
    ) -> "SufficientStatsRecord":
        arr = np.asarray(samples, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 1:
            raise ValueError(f"Expected (reps, 2) simulation array, got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Simulation samples contain non-finite entries")
        m = int(arr.shape[0])
        obj = arr[:, 0]
        con = arr[:, 1]
        mean_obj = float(np.mean(obj))
        mean_con = float(np.mean(con))
        ssq_obj = float(np.sum((obj - mean_obj) ** 2)) if m > 1 else 0.0
        ssq_con = float(np.sum((con - mean_con) ** 2)) if m > 1 else 0.0
        violations = int(np.sum(con > 0.0))
        return cls(
            x_norm=np.clip(np.asarray(x_norm, dtype=float).reshape(-1), 0.0, 1.0),
            n_reps=m,
            mean_obj=mean_obj,
            ssq_obj=ssq_obj,
            mean_con=mean_con,
            ssq_con=ssq_con,
            violation_count=violations,
            exposure=exposure,
        )

    def merge(self, other: "SufficientStatsRecord") -> "SufficientStatsRecord":
        """Exact Chan-Golub-LeVeque parallel variance merge for repeated queries."""
        n1, n2 = self.n_reps, other.n_reps
        n_tot = n1 + n2
        delta_obj = other.mean_obj - self.mean_obj
        mean_obj = self.mean_obj + delta_obj * (n2 / n_tot)
        ssq_obj = self.ssq_obj + other.ssq_obj + (delta_obj ** 2) * (n1 * n2 / n_tot)

        delta_con = other.mean_con - self.mean_con
        mean_con = self.mean_con + delta_con * (n2 / n_tot)
        ssq_con = self.ssq_con + other.ssq_con + (delta_con ** 2) * (n1 * n2 / n_tot)

        return SufficientStatsRecord(
            x_norm=self.x_norm,
            n_reps=n_tot,
            mean_obj=float(mean_obj),
            ssq_obj=float(ssq_obj),
            mean_con=float(mean_con),
            ssq_con=float(ssq_con),
            violation_count=int(self.violation_count + other.violation_count),
            exposure=self.exposure,
        )


@dataclass(frozen=True)
class LegacyTrajectoryRecord:
    """One policy evaluation logged during an offline legacy optimizer run."""

    task_id: str
    solver_name: str
    step_index: int
    policy_dim: int
    x_norm: tuple[float, ...]
    n_reps: int
    mean_obj: float
    ssq_obj: float
    mean_con: float
    ssq_con: float
    violation_count: int
    psi: tuple[float, ...]
    A_local: tuple[float, ...]
    N_shared: tuple[float, ...]


class BaseDynamicORProblem(ABC):
    """Abstract base interface for genuine dynamic stochastic OR benchmarks."""

    family_name: str = "abstract_or_family"
    task_id: str = "task_0"
    d: int = 1
    alpha: float = 0.05
    is_gaussian_constraint_noise: bool = False

    def __init__(self) -> None:
        self.search_calls: int = 0
        self.verification_calls: int = 0
        self.audit_calls: int = 0

    def clip_policy(self, x_norm: np.ndarray) -> np.ndarray:
        arr = np.asarray(x_norm, dtype=float).reshape(-1)
        if arr.size != self.d:
            raise ValueError(f"Expected policy of dimension {self.d}, got {arr.size}")
        return np.clip(arr, 0.0, 1.0)

    @abstractmethod
    def nominal_exposure(self, x_norm: np.ndarray) -> NominalTrajectoryExposure:
        """Compute public nominal rollout features with ZERO private law access."""

    @abstractmethod
    def _simulate_raw(
        self,
        x_norm: np.ndarray,
        reps: int,
        rng: np.random.Generator,
        split: str = "search",
    ) -> np.ndarray:
        """Draw `reps` independent simulation replications returning `[obj, con_centered]`."""

    def simulate_replications(
        self,
        x_norm: np.ndarray,
        reps: int,
        rng: np.random.Generator,
        *,
        split: str = "search",
    ) -> np.ndarray:
        """Metered simulation call enforcing budget accounting across splits."""
        m = int(reps)
        if m < 1:
            raise ValueError("reps must be >= 1")
        x_clean = self.clip_policy(x_norm)
        out = np.asarray(self._simulate_raw(x_clean, m, rng, split=split), dtype=float)
        if out.shape != (m, 2) or not np.all(np.isfinite(out)):
            raise RuntimeError(f"Invalid simulation output shape/finiteness: {out.shape}")
        if split == "search":
            self.search_calls += m
        elif split == "verification":
            self.verification_calls += m
        elif split == "audit":
            self.audit_calls += m
        else:
            raise ValueError(f"Unknown simulation split {split!r}")
        return out

    def domain_heuristic_anchor(self) -> np.ndarray:
        """Public operational baseline policy (e.g., nominal base-stock / fixed green)."""
        return np.full(self.d, 0.5, dtype=float)

    def audit_ground_truth(
        self,
        x_norm: np.ndarray,
        *,
        n_eval: int = 4000,
        seed: int = 20261116,
    ) -> dict[str, Any]:
        """Post-hoc truth evaluation on an independent audit stream (never seen by learners)."""
        rng = np.random.default_rng(seed)
        samples = self.simulate_replications(x_norm, n_eval, rng, split="audit")
        obj = samples[:, 0]
        con = samples[:, 1]
        viol_prob = float(np.mean(con > 0.0))
        return {
            "true_objective": float(np.mean(obj)),
            "true_objective_se": float(np.std(obj, ddof=1) / np.sqrt(n_eval)),
            "true_constraint_mean": float(np.mean(con)),
            "true_constraint_std": float(np.std(con, ddof=1)),
            "true_violation_prob": viol_prob,
            "true_quantile_1_minus_alpha": float(np.quantile(con, 1.0 - self.alpha)),
            "is_truly_feasible": bool(viol_prob <= self.alpha),
            "n_audit_samples": int(n_eval),
        }
