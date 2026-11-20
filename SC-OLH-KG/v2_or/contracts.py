"""Core data contracts, channel-preserving policy adapters, and strict oracle isolation.

All benchmark families in `v2_or` normalize constraint outputs as `Y_g - tau` so
that chance-constraint feasibility is universally:

    P(Y_g_centered(x) <= 0) >= 1 - alpha.

Crucially:
1. `nominal_exposure(x)` MUST depend exclusively on public exogenous forecasts and
   nominal deterministic physical equations—never on hidden stochastic shocks or
   private task parameters (`psi in R^12` across all four families).
2. Terminal certification explicitly separates three decision states:
   - `"certified_deployment"`: at least one frozen shortlist slot passed its finite-sample
     safety gate (`deployed_policy` is that certified policy).
   - `"abstention"`: no shortlist slot passed certification (`deployed_policy` is `None`;
     the organization retains its pre-existing status-quo anchor `domain_heuristic_anchor()`).
   - `"fallback_deployment"`: only when an explicitly declared pre-certified fallback is used.
3. Post-hoc Monte Carlo auditing reports sample estimates and exact Clopper-Pearson
   confidence intervals `[audit_viol_ci95_lo, audit_viol_ci95_hi]` rather than treating
   finite Monte Carlo draws as exact deterministic truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import beta as beta_dist


def clopper_pearson_interval(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson confidence interval for a Binomial proportion."""
    alpha_tail = (1.0 - float(conf)) / 2.0
    k_int, n_int = int(k), int(n)
    if n_int <= 0:
        return 0.0, 1.0
    lo = 0.0 if k_int <= 0 else float(beta_dist.ppf(alpha_tail, k_int, n_int - k_int + 1))
    hi = 1.0 if k_int >= n_int else float(beta_dist.ppf(1.0 - alpha_tail, k_int + 1, n_int - k_int))
    return lo, hi


def warp_multichannel_policy(
    x_src: np.ndarray,
    target_d: int,
    num_channels: int = 1,
) -> np.ndarray:
    """Interpolate a source policy to `target_d` separately within each control channel.

    For interleaved multi-channel policies (e.g., `[reg_0, exp_0, reg_1, exp_1, ...]`
    with `num_channels = 2`), interpolating channel-by-channel preserves exact channel
    semantics: `[0.2, 0.8] * 3 -> [0.2, 0.8] * 4`.
    """
    arr = np.asarray(x_src, dtype=float).reshape(-1)
    d_tgt = int(target_d)
    if arr.size == d_tgt:
        return np.clip(arr.copy(), 0.0, 1.0)
    c_cnt = max(int(num_channels), 1)
    if c_cnt == 1 or arr.size < c_cnt or d_tgt < c_cnt:
        if arr.size == 1:
            return np.full(d_tgt, float(np.clip(arr[0], 0.0, 1.0)), dtype=float)
        u_s = np.linspace(0.0, 1.0, arr.size)
        u_t = np.linspace(0.0, 1.0, d_tgt)
        fn = interp1d(u_s, arr, kind="linear", fill_value="extrapolate")
        return np.clip(fn(u_t), 0.0, 1.0)

    out = np.empty(d_tgt, dtype=float)
    for c in range(c_cnt):
        ch_src = arr[c::c_cnt]
        n_tgt_c = out[c::c_cnt].size
        if n_tgt_c == 0:
            continue
        if ch_src.size == 1:
            out[c::c_cnt] = float(ch_src[0])
        elif n_tgt_c == 1:
            out[c::c_cnt] = float(np.mean(ch_src))
        else:
            u_s = np.linspace(0.0, 1.0, ch_src.size)
            u_t = np.linspace(0.0, 1.0, n_tgt_c)
            fn = interp1d(u_s, ch_src, kind="linear", fill_value="extrapolate")
            out[c::c_cnt] = fn(u_t)
    return np.clip(out, 0.0, 1.0)


@dataclass(frozen=True)
class NominalTrajectoryExposure:
    """Dimension-invariant state-risk exposure (`psi in R^12`) from a public nominal rollout."""

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
    """Decoupled within-policy sufficient statistics invariant to common mean shifts.

    Given `m >= 2` replications at policy `x`, the within-policy sum of squared
    deviations `ssq_con = sum_{r=1}^m (Y_{g,r} - mean_con)^2` satisfies
    `E[ssq_con] = (m - 1) * v_g(x)` identically for all common mean shifts `Delta_g(x)`.
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
    """One policy evaluation logged during an offline adaptive solver run."""

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
    """Abstract base interface for dynamic stochastic OR simulation benchmarks."""

    family_name: str = "abstract_or_family"
    task_id: str = "task_0"
    d: int = 1
    num_control_channels: int = 1
    alpha: float = 0.05
    is_gaussian_constraint_noise: bool = False

    def __init__(self) -> None:
        self.search_calls: int = 0
        self.verification_calls: int = 0
        self.audit_calls: int = 0
        self.call_limits: dict[str, int] | None = None
        self.attempted_calls: dict[str, int] = {}

    def clip_policy(self, x_norm: np.ndarray) -> np.ndarray:
        arr = np.asarray(x_norm, dtype=float).reshape(-1)
        if arr.size != self.d:
            raise ValueError(f"Expected policy of dimension {self.d}, got {arr.size}")
        return np.clip(arr, 0.0, 1.0)

    def warp_source_policy(self, x_src: np.ndarray) -> np.ndarray:
        """Channel-preserving policy dimension adapter from `d_s` to `self.d`."""
        return warp_multichannel_policy(x_src, self.d, num_channels=self.num_control_channels)

    @abstractmethod
    def nominal_exposure(self, x_norm: np.ndarray) -> NominalTrajectoryExposure:
        """Compute public nominal rollout features (`psi in R^12`) with ZERO private law access."""

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
        if self.call_limits is not None:
            attempted = self.attempted_calls.get(split, 0) + m
            if attempted > self.call_limits.get(split, 0):
                raise RuntimeError(f"Usage {attempted} exceeds {split} cap {self.call_limits.get(split, 0)}")
            self.attempted_calls[split] = attempted
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
        """Public status-quo operational baseline policy."""
        return np.full(self.d, 0.5, dtype=float)

    def audit_ground_truth(
        self,
        x_norm: np.ndarray,
        *,
        n_eval: int = 4000,
        seed: int = 20261116,
        split: str = "audit",
    ) -> dict[str, Any]:
        """Post-hoc Monte Carlo audit on an independent stream with Clopper-Pearson CIs."""
        rng = np.random.default_rng(seed)
        samples = self.simulate_replications(x_norm, n_eval, rng, split=split)
        obj = samples[:, 0]
        con = samples[:, 1]
        k_viol = int(np.sum(con > 0.0))
        viol_prob = float(k_viol / float(n_eval))
        ci_lo, ci_hi = clopper_pearson_interval(k_viol, int(n_eval), conf=0.95)
        if ci_hi <= self.alpha:
            safety_class = "confidently_safe"
        elif ci_lo > self.alpha:
            safety_class = "confidently_unsafe"
        else:
            safety_class = "boundary_ambiguous"

        return {
            "mc_objective_mean": float(np.mean(obj)),
            "mc_objective_se": float(np.std(obj, ddof=1) / np.sqrt(n_eval)),
            "mc_constraint_mean": float(np.mean(con)),
            "mc_constraint_std": float(np.std(con, ddof=1)),
            "mc_violation_count": k_viol,
            "mc_violation_prob": viol_prob,
            "mc_viol_ci95_lo": ci_lo,
            "mc_viol_ci95_hi": ci_hi,
            "safety_audit_class": safety_class,
            # Backwards-compatible aliases
            "true_objective": float(np.mean(obj)),
            "true_objective_se": float(np.std(obj, ddof=1) / np.sqrt(n_eval)),
            "true_constraint_mean": float(np.mean(con)),
            "true_constraint_std": float(np.std(con, ddof=1)),
            "true_violation_prob": viol_prob,
            "true_quantile_1_minus_alpha": float(np.quantile(con, 1.0 - self.alpha)),
            "is_truly_feasible": bool(viol_prob <= self.alpha),
            "n_audit_samples": int(n_eval),
            "audit_split": str(split),
        }
