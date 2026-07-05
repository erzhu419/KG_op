"""Live ingolstadt21 traffic wrapper for scalarized SC-OLH-KG.

This module intentionally delegates the simulator to the original paper code
under `Final_Submission/GPR_KG_Code`.  It gives the new single-objective
OLH-KG implementation the same `(objective, constraint)` interface used by the
synthetic benchmarks while preserving the original SUMO evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
from scipy.stats import norm


REPO_ROOT = Path(__file__).resolve().parents[2]
GPR_KG_CODE = REPO_ROOT / "Final_Submission" / "GPR_KG_Code"
if str(GPR_KG_CODE) not in sys.path:
    sys.path.insert(0, str(GPR_KG_CODE))

from experiments.ingolstadt21.ingolstadt21_problem import Ingolstadt21Problem  # noqa: E402


class Ingolstadt21ScalarizedTrafficProblem:
    """Scalarized traffic objective with the original emission chance constraint.

    The algorithm observes two outputs:

    * objective: `w1 * f1 + w2 * f2`
    * constraint: original `f3`

    Expensive "true" metrics are only Monte Carlo certification estimates and
    should be treated as diagnostics.  Paper-grade feasibility is still done by
    `experiments.ingolstadt21.validate_oos_feasibility` with fresh seeds.
    """

    problem_name = "Ingolstadt21Traffic"

    def __init__(
        self,
        weights=(0.5, 0.5),
        seed: int = 0,
        true_replications: int = 5,
        sigma_replications: int = 8,
        historical_anchor_policy: str = "historical",
    ):
        policy = str(historical_anchor_policy).strip().lower()
        if policy not in {"historical", "none", "strict_none", "only"}:
            raise ValueError(
                "historical_anchor_policy must be one of: "
                "historical, none, strict_none, only"
            )
        self.base = Ingolstadt21Problem()
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / max(float(np.sum(self.weights)), 1e-12)
        self.d = int(self.base.d)
        self.L = None
        self.alpha = float(self.base.alpha)
        self.tau = float(self.base.tau)
        self.ref_point = getattr(self.base, "ref_point", None)
        self.problem_name = "Ingolstadt21Traffic_scalar"
        self.sigma_level = 0.03
        self.variance_features = (0,)
        self.recommended_partition_features = self.variance_features
        self.true_replications = max(1, int(true_replications))
        self.sigma_replications = max(2, int(sigma_replications))
        self.historical_anchor_policy = policy
        self._rng = np.random.default_rng(int(seed))
        self._historical_anchor_cache = None

    def int_bounds(self):
        return self.base.int_bounds()

    def normalize(self, x):
        return self.base.normalize(x)

    def continuous_to_int(self, x_norm):
        return self.base.continuous_to_int(np.asarray(x_norm, dtype=float))

    def sample_random(self, rng=None):
        rng = rng or self._rng
        lo, hi = self.int_bounds()
        return tuple(
            int(rng.integers(int(lo[k]), int(hi[k]) + 1))
            for k in range(self.d)
        )

    def _simulate_vector(self, x, seed: int | None = None):
        if seed is None:
            seed = int(self._rng.integers(0, 2**31 - 1))
        return self.base._sim.simulate(
            var_map=self.base.var_map,
            x=np.asarray(x, dtype=float),
            route_file=self.base._route_files[0],
            T0=self.base.T0,
            A0=self.base.A0,
            E0=self.base.E0,
            seed=int(seed),
        ).astype(float)

    def simulate(self, x, rng=None):
        rng = rng or self._rng
        seed = int(rng.integers(0, 2**31 - 1))
        y = self._simulate_vector(x, seed=seed)
        obj = float(self.weights[0] * y[0] + self.weights[1] * y[1])
        return np.array([obj, float(y[2])], dtype=float)

    def true_vector_objectives(self, x):
        rows = np.array([
            self._simulate_vector(x, seed=int(self._rng.integers(0, 2**31 - 1)))
            for _ in range(self.true_replications)
        ], dtype=float)
        return rows.mean(axis=0)

    def true_objective(self, x):
        f1, f2, _ = self.true_vector_objectives(x)
        return float(self.weights[0] * f1 + self.weights[1] * f2)

    def true_constraint_mean(self, x):
        return float(self.true_vector_objectives(x)[2])

    def true_sigma(self, x):
        rows = np.array([
            self._simulate_vector(x, seed=int(self._rng.integers(0, 2**31 - 1)))
            for _ in range(self.sigma_replications)
        ], dtype=float)
        sig = rows.std(axis=0, ddof=1)
        obj_sig = np.sqrt((self.weights[0] * sig[0]) ** 2 + (self.weights[1] * sig[1]) ** 2)
        return np.array([float(obj_sig), float(sig[2])], dtype=float)

    def is_truly_feasible(self, x):
        vals = [
            self._simulate_vector(x, seed=int(self._rng.integers(0, 2**31 - 1)))[2]
            for _ in range(self.true_replications)
        ]
        return float(np.mean(np.asarray(vals) <= self.tau)) >= 1.0 - self.alpha

    def true_best_feasible(self):
        # The traffic domain has no finite exhaustive oracle.  Fresh-seed OOS
        # validation supplies the paper-grade comparison after optimization.
        return None, float("inf")

    def initial_samples(self, n=5, rng=None):
        rng = rng or self._rng
        rows = []
        default = tuple(int(v) for v in np.asarray(self.base.default_x, dtype=int))
        if self.historical_anchor_policy == "only":
            rows.extend(self._historical_anchor_candidates(max_count=int(n)))
        else:
            rows.append(default)
            if self.historical_anchor_policy == "historical":
                rows.extend(self._historical_anchor_candidates(max_count=max(0, int(n) - 1)))
        lo, hi = self.int_bounds()
        lo = np.asarray(lo, dtype=int)
        hi = np.asarray(hi, dtype=int)
        center = np.round((lo + hi) / 2.0).astype(int)
        if self.historical_anchor_policy not in {"only", "strict_none"}:
            rows.extend([tuple(lo), tuple(center), tuple(hi)])
        while len(rows) < int(n):
            rows.append(self.sample_random(rng))
        return _unique(rows)[: int(n)]

    def structured_candidates(self, n=10, rng=None):
        rng = rng or self._rng
        lo, hi = self.int_bounds()
        lo = np.asarray(lo, dtype=int)
        hi = np.asarray(hi, dtype=int)
        default = np.asarray(self.base.default_x, dtype=int)
        if self.historical_anchor_policy == "only":
            anchors = self._historical_anchor_candidates(max_count=max(1, int(n)))
            if anchors:
                return _unique(anchors)[: int(n)]
            return [tuple(default)]
        if self.historical_anchor_policy == "strict_none":
            return []

        rows = [tuple(default)]
        anchors = self._historical_anchor_candidates(max_count=max(4, int(n)))
        if self.historical_anchor_policy == "historical":
            rows.extend(anchors[: max(0, int(n) - len(rows))])
        for frac in (0.75, 0.9, 1.0, 1.1, 1.25):
            x = np.clip(np.round(default * frac), lo, hi).astype(int)
            rows.append(tuple(x))
        if self.historical_anchor_policy == "historical":
            for anchor in anchors[:4]:
                x0 = np.asarray(anchor, dtype=int)
                for scale in (0.04, 0.08):
                    noise = rng.normal(0.0, scale, size=self.d)
                    span = np.maximum(hi - lo, 1)
                    x = np.clip(np.round(x0 + noise * span), lo, hi).astype(int)
                    rows.append(tuple(x))
        for _ in range(max(0, int(n) - len(rows))):
            anchor = default.copy()
            j = int(rng.integers(0, self.d))
            anchor[j] = int(rng.integers(lo[j], hi[j] + 1))
            rows.append(tuple(anchor))
        return _unique(rows)[: int(n)]

    def _coerce_candidate(self, row):
        try:
            x = tuple(int(float(v)) for v in row)
        except (TypeError, ValueError):
            return None
        if len(x) != self.d:
            return None
        lo, hi = self.int_bounds()
        lo = np.asarray(lo, dtype=int)
        hi = np.asarray(hi, dtype=int)
        arr = np.asarray(x, dtype=int)
        arr = np.clip(arr, lo, hi)
        return tuple(int(v) for v in arr)

    def _historical_anchor_candidates(self, max_count=16):
        if self.historical_anchor_policy in {"none", "strict_none"}:
            return []
        if self._historical_anchor_cache is not None:
            return self._historical_anchor_cache[: int(max_count)]
        root = GPR_KG_CODE / "results" / "ingolstadt21"
        rows = []

        def add(row):
            x = self._coerce_candidate(row)
            if x is not None:
                rows.append(x)

        for audit_path in sorted(root.glob("oos_feasibility_validation_R*_audit.json")):
            try:
                data = json.loads(audit_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            best = data.get("best_feasible_probability_candidate") or {}
            if best.get("x") is not None:
                add(best["x"])

        preferred = []
        preferred.extend(sorted(root.glob("GPR_KG_binary_bin_seed*/summary.json")))
        preferred.extend(sorted(root.glob("GPR_KG_nV_binary_bin_seed*/summary.json")))
        preferred.extend(sorted(root.glob("GPR_KG_*seed*/summary.json")))
        seen_paths = set()
        for summary_path in preferred:
            if summary_path in seen_paths:
                continue
            seen_paths.add(summary_path)
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for row in data.get("final_pareto_set") or []:
                add(row)

        self._historical_anchor_cache = _unique(rows)
        return self._historical_anchor_cache[: int(max_count)]

    def historical_anchor_candidates(self, max_count=16):
        return list(self._historical_anchor_candidates(max_count=max_count))

    def recommendation_refinement_candidates(self):
        lo, hi = self.int_bounds()
        lo = np.asarray(lo, dtype=int)
        hi = np.asarray(hi, dtype=int)
        rows = []
        if self.historical_anchor_policy in {"historical", "only"}:
            rows.extend(self._historical_anchor_candidates(max_count=24))
        if self.historical_anchor_policy == "none":
            default = np.asarray(self.base.default_x, dtype=int)
            center = np.round((lo + hi) / 2.0).astype(int)
            rows.extend([tuple(default), tuple(center), tuple(lo), tuple(hi)])
        if self.historical_anchor_policy == "strict_none":
            return []
        for anchor in list(rows)[:8]:
            x0 = np.asarray(anchor, dtype=int)
            for frac in (0.95, 1.05):
                rows.append(tuple(np.clip(np.round(x0 * frac), lo, hi).astype(int)))
        return _unique(rows)

    def state_anchor_points(self, n=10, rng=None):
        """Traffic state/meta anchors used only by SC candidate generation.

        These anchors live in a low-dimensional policy-state space:
        target network green intensity, heterogeneity/spread, and a coarse
        temporal pattern.  They deliberately do not reuse historical
        solutions or deterministic final-recommendation refinement points.
        """
        rng = rng or self._rng
        n = max(0, int(n))
        templates = [
            # Low-risk meta anchors are intentionally specified in state space
            # rather than as raw low/center/high signal vectors.  This keeps the
            # strict no-history ablation from inheriting the old deterministic
            # "low green ratio" refinement shortcut while still letting SC
            # explore the safe traffic regime.
            {"mean": 0.08, "spread": 0.025, "pattern": "balanced"},
            {"mean": 0.14, "spread": 0.035, "pattern": "corridor"},
            {"mean": 0.20, "spread": 0.045, "pattern": "balanced"},
            {"mean": 0.26, "spread": 0.060, "pattern": "alternating"},
            {"mean": 0.32, "spread": 0.075, "pattern": "balanced"},
            {"mean": 0.38, "spread": 0.120, "pattern": "corridor"},
            {"mean": 0.44, "spread": 0.180, "pattern": "alternating"},
            {"mean": 0.50, "spread": 0.100, "pattern": "clustered"},
            {"mean": 0.58, "spread": 0.150, "pattern": "corridor"},
            {"mean": 0.66, "spread": 0.200, "pattern": "alternating"},
        ]
        anchors = []
        for i in range(n):
            base = dict(templates[i % len(templates)])
            if i >= len(templates):
                if rng.random() < 0.45:
                    base["mean"] = float(rng.uniform(0.06, 0.34))
                    base["spread"] = float(rng.uniform(0.02, 0.10))
                else:
                    base["mean"] = float(rng.uniform(0.28, 0.72))
                    base["spread"] = float(rng.uniform(0.06, 0.22))
                base["pattern"] = str(rng.choice([
                    "balanced", "corridor", "alternating", "clustered"
                ]))
            base["phase"] = float(rng.uniform(0.0, 2.0 * np.pi))
            anchors.append(base)
        return anchors

    def inverse_state_anchor(self, anchor, rng=None, n=1):
        """Invert a traffic meta anchor into raw integer signal parameters."""
        rng = rng or self._rng
        n = max(1, int(n))
        lo, hi = self.int_bounds()
        lo = np.asarray(lo, dtype=int)
        hi = np.asarray(hi, dtype=int)
        span = np.maximum(hi - lo, 1)
        d = int(self.d)
        target_mean = float(np.clip(anchor.get("mean", 0.5), 0.05, 0.95))
        target_spread = float(np.clip(anchor.get("spread", 0.12), 0.01, 0.35))
        pattern = str(anchor.get("pattern", "balanced"))
        phase = float(anchor.get("phase", 0.0))
        idx = np.arange(d, dtype=float)
        rows = []
        for _ in range(n):
            if pattern == "corridor":
                wave = np.sin(2.0 * np.pi * idx / max(d, 1) + phase)
                block = np.where((idx.astype(int) % 4) < 2, -0.6, 0.8)
                z = target_mean + target_spread * (0.55 * wave + 0.45 * block)
            elif pattern == "alternating":
                wave = np.where((idx.astype(int) % 2) == 0, -1.0, 1.0)
                z = target_mean + target_spread * wave
            elif pattern == "clustered":
                groups = np.floor(idx / max(1.0, d / 6.0))
                levels = rng.normal(0.0, target_spread, size=7)
                z = target_mean + levels[np.clip(groups.astype(int), 0, 6)]
            else:
                z = rng.normal(target_mean, target_spread, size=d)
            z = np.asarray(z, dtype=float)
            z += rng.normal(0.0, 0.025, size=d)
            # Match the requested meta mean without collapsing the spread.
            z += target_mean - float(np.mean(z))
            z = np.clip(z, 0.0, 1.0)
            x = np.clip(np.round(lo + z * span), lo, hi).astype(int)
            rows.append(tuple(int(v) for v in x))
        return _unique(rows)

    def risk_class(self, x):
        z = self.normalize(x)
        emission_pressure = float(np.mean(z))
        if emission_pressure < 0.4:
            return 0
        if emission_pressure < 0.65:
            return 1
        return 2

    def hvd_features(self, x):
        z = self.normalize(x)
        if len(z) == 0:
            z = np.array([0.0])
        stats = np.array([
            float(np.mean(z)),
            float(np.std(z)),
            float(np.min(z)),
            float(np.max(z)),
            float(np.linalg.norm(z - 0.5) / np.sqrt(len(z))),
        ])
        return np.concatenate([[1.0], stats, z[: min(12, len(z))]])

    def cumulative_risk_features(self, x, output_index=1):
        z = self.normalize(x)
        if len(z) == 0:
            z = np.array([0.0])
        low = np.maximum(0.0, 0.35 - z)
        high = np.maximum(0.0, z - 0.75)
        return np.array([
            1.0,
            float(np.mean(z)),
            float(np.sum(low) / len(z)),
            float(np.sum(high) / len(z)),
            float(np.std(z)),
        ], dtype=float)

    def cumulative_risk_feature_names(self, output_index=1):
        return ["floor", "mean_green_norm", "low_green_exposure", "high_green_exposure", "dispersion"]

    def hvd_residual_variance_cap(self, output_index=0):
        return 0.20 if int(output_index) == 1 else 0.10

    def recommendation_random_pool_size(self):
        return 128


def chance_margin_from_oos(mean_f3, std_f3, tau, alpha):
    return float(mean_f3 + norm.ppf(1 - alpha) * max(float(std_f3), 1e-12) - tau)


def _unique(rows: Iterable[Iterable[int]]):
    seen = set()
    out = []
    for row in rows:
        item = tuple(int(v) for v in row)
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
