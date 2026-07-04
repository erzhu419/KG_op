"""Deterministic state-policy coupling features for synthetic experiments."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


class SyntheticPolicyStateEncoder:
    """Map a policy/design vector to a deterministic occupancy proxy.

    Real traffic trajectory encoding is intentionally deferred.  For synthetic
    RZDT-like problems this encoder creates a low-dimensional proxy for
    policy-induced occupancy/risk regimes from normalized decision variables.
    """

    def __init__(self, problem, lengthscale=0.35):
        self.problem = problem
        self.lengthscale = float(lengthscale)
        self.feature_dim = 8

    def occupancy(self, x):
        z = np.asarray(self.problem.normalize(x), dtype=float)
        center = np.full_like(z, 0.5)
        u0 = float(z[0]) if len(z) else 0.0
        tail = z[1:] if len(z) > 1 else np.array([0.0])
        return np.array([
            float(np.mean(z)),
            float(np.std(z)),
            float(np.min(z)),
            float(np.max(z)),
            float(np.linalg.norm(z - center) / np.sqrt(max(1, len(z)))),
            float(np.sin(np.pi * u0)),
            float(np.cos(np.pi * u0)),
            float(np.mean(tail)),
        ], dtype=float)

    def features(self, x):
        return self.occupancy(x)

    def distance(self, x, y):
        dx = self.occupancy(x) - self.occupancy(y)
        return float(np.linalg.norm(dx))

    def kernel(self, x, y):
        dist = self.distance(x, y)
        return float(np.exp(-0.5 * (dist / max(self.lengthscale, 1e-8)) ** 2))

    def propagation_scores(self, candidates, observed):
        """Occupancy coverage score for state-coupled exploration.

        The early prototype rewarded similarity to already observed states,
        which made the coupling term largely redundant.  SC exploration should
        instead favor candidates whose occupancy proxy is under-covered by the
        current sample set.
        """
        observed = self._observed_x(observed)
        if not candidates:
            return np.zeros(0, dtype=float)
        if not observed:
            return np.ones(len(candidates), dtype=float)
        cand = np.vstack([self.occupancy(x) for x in candidates])
        obs = np.vstack([self.occupancy(tuple(o)) for o in observed])
        diff = cand[:, None, :] - obs[None, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        scale = max(self.lengthscale, 1e-8) ** 2
        similarity = np.exp(-0.5 * dist2 / scale)
        max_similarity = np.max(similarity, axis=1)
        scores = 1.0 - np.clip(max_similarity, 0.0, 1.0)
        hi = float(np.max(scores))
        lo = float(np.min(scores))
        if hi - lo <= 1e-14:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def coupling_scores(self, candidates, observed):
        """State-coupling score combining promising states and coverage."""
        coverage = self.propagation_scores(candidates, observed)
        history = self._observed_history(observed)
        if len(candidates) == 0 or not history:
            return coverage
        z = 1.6448536269514722
        sigma = float(getattr(self.problem, "sigma_level", 0.04))
        feasible = []
        for x, y in history:
            margin = float(y[1]) + z * sigma - float(getattr(self.problem, "tau", 0.0))
            if margin <= 0.0:
                feasible.append((x, y))
        if not feasible:
            return coverage

        obs_x = [x for x, _ in feasible]
        obs_y = np.array([float(y[0]) for _, y in feasible], dtype=float)
        scale_y = max(float(np.std(obs_y)), 1e-8)
        weights = np.exp(-(obs_y - float(np.min(obs_y))) / scale_y)
        cand = np.vstack([self.occupancy(x) for x in candidates])
        obs = np.vstack([self.occupancy(x) for x in obs_x])
        dist2 = np.sum((cand[:, None, :] - obs[None, :, :]) ** 2, axis=2)
        sim = np.exp(-0.5 * dist2 / max(self.lengthscale, 1e-8) ** 2)
        promising = np.max(sim * weights[None, :], axis=1)
        promising = self._normalize01(promising)
        return self._normalize01(0.75 * promising + 0.25 * coverage)

    def state_space_candidates(
        self,
        n_anchors=10,
        inverse_pool_size=500,
        inverse_neighbors=1,
        rng=None,
        observed=None,
    ):
        """Generate raw candidates by searching through state/meta anchors.

        This is the first real SC candidate-generation path: propose anchors in
        a lower-dimensional state/meta space, then invert each anchor back to
        one or more raw decision vectors.  Problems can provide exact synthetic
        inverses; otherwise we approximate the inverse by nearest-neighbor
        matching in occupancy space over a random raw pool.
        """
        rng = rng or np.random.default_rng()
        n_anchors = max(0, int(n_anchors))
        inverse_neighbors = max(1, int(inverse_neighbors))
        if n_anchors <= 0:
            return []

        if (
            hasattr(self.problem, "state_anchor_points")
            and hasattr(self.problem, "inverse_state_anchor")
        ):
            rows = []
            anchors = self.problem.state_anchor_points(n=n_anchors, rng=rng)
            for anchor in anchors:
                rows.extend(self.problem.inverse_state_anchor(
                    anchor,
                    rng=rng,
                    n=inverse_neighbors,
                ))
            return self._unique(rows)

        pool = self._raw_inverse_pool(inverse_pool_size, rng, observed)
        if not pool:
            return []
        target = rng.random((n_anchors, self.feature_dim))
        occ = np.vstack([self.occupancy(x) for x in pool])
        chosen = []
        for rho in target:
            dist = np.linalg.norm(occ - rho[None, :], axis=1)
            for idx in np.argsort(dist)[:inverse_neighbors]:
                chosen.append(pool[int(idx)])
        return self._unique(chosen)

    def _raw_inverse_pool(self, n, rng, observed=None):
        pool = []
        pool.extend(self._observed_x(observed or []))
        if hasattr(self.problem, "structured_candidates"):
            pool.extend(self.problem.structured_candidates(
                n=max(5, int(n) // 10),
                rng=rng,
            ))
        for _ in range(max(0, int(n))):
            pool.append(self.problem.sample_random(rng))
        return self._unique(pool)

    @staticmethod
    def _observed_x(observed):
        rows = []
        for item in observed or []:
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and not np.isscalar(item[1])
            ):
                rows.append(tuple(int(v) for v in item[0]))
            else:
                rows.append(tuple(int(v) for v in item))
        return rows

    @staticmethod
    def _observed_history(observed):
        rows = []
        for item in observed or []:
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and not np.isscalar(item[1])
            ):
                rows.append((tuple(int(v) for v in item[0]), np.asarray(item[1], dtype=float)))
        return rows

    @staticmethod
    def _normalize01(values):
        arr = np.asarray(values, dtype=float)
        if len(arr) == 0:
            return arr
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi - lo <= 1e-14:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    @staticmethod
    def _unique(candidates):
        seen = set()
        rows = []
        for x in candidates:
            x_tuple = tuple(int(v) for v in x)
            if x_tuple not in seen:
                seen.add(x_tuple)
                rows.append(x_tuple)
        return rows


class SelfSupervisedTrajectoryEncoder:
    """Lightweight self-supervised trajectory representation.

    This is intentionally dependency-light: it gives the experimental pipeline
    a real masked-reconstruction / contrastive / transformer-style encoder
    interface without making PyTorch training a hard dependency for unit tests
    and paper sweeps.  The learned map is a standardized low-rank projection of
    sequence summaries; `transformer` mode changes the sequence pooling step to
    deterministic attention-style weights over trajectory tokens.
    """

    NUMERIC_FIELDS = ("occupancy", "queue", "wait", "flow", "demand_shock")

    def __init__(self, latent_dim=8, mode="masked", ridge=1e-6):
        self.latent_dim = int(latent_dim)
        self.mode = str(mode)
        self.ridge = float(ridge)
        self.feature_dim = int(latent_dim)
        self.policy_features: dict[str, np.ndarray] = {}
        self.policy_summaries: dict[str, np.ndarray] = {}
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.class_centers_: dict[str, np.ndarray] = {}
        self.diagnostics_: dict[str, float | int | str] = {}

    def fit(self, trajectory_logs):
        return self.fit_masked_prediction(trajectory_logs)

    def fit_masked_prediction(self, trajectory_logs):
        grouped = self._group_records(trajectory_logs)
        summaries = {
            policy_id: self._sequence_summary(rows)
            for policy_id, rows in grouped.items()
        }
        self._fit_projection(summaries)
        baseline, recon = self._reconstruction_errors(summaries)
        self.diagnostics_ = {
            "mode": "masked",
            "n_policies": int(len(summaries)),
            "latent_dim": int(self.latent_dim),
            "masked_baseline_mse": float(baseline),
            "masked_reconstruction_mse": float(recon),
        }
        return self

    def fit_contrastive(self, trajectory_logs, policy_ids=None, risk_labels=None):
        grouped = self._group_records(trajectory_logs)
        summaries = {
            policy_id: self._sequence_summary(rows)
            for policy_id, rows in grouped.items()
        }
        self._fit_projection(summaries)
        if risk_labels is None:
            risk_labels = {
                policy_id: self._auto_risk_label(rows)
                for policy_id, rows in grouped.items()
            }
        if policy_ids is None:
            policy_ids = list(summaries)
        centers: dict[str, list[np.ndarray]] = defaultdict(list)
        for policy_id in policy_ids:
            key = str(policy_id)
            if key in summaries and key in risk_labels:
                centers[str(risk_labels[key])].append(self.features(key))
        self.class_centers_ = {
            label: np.mean(np.vstack(vals), axis=0)
            for label, vals in centers.items()
            if vals
        }
        self.diagnostics_ = {
            "mode": "contrastive",
            "n_policies": int(len(summaries)),
            "n_classes": int(len(self.class_centers_)),
            "latent_dim": int(self.latent_dim),
            "contrastive_separation": float(self._contrastive_separation(risk_labels)),
        }
        return self

    def fit_transformer_pooling(self, trajectory_logs):
        old_mode = self.mode
        self.mode = "transformer"
        try:
            return self.fit_masked_prediction(trajectory_logs)
        finally:
            self.mode = old_mode if old_mode else "transformer"

    def features(self, policy_id):
        return np.asarray(self.policy_features[str(policy_id)], dtype=float)

    def diagnostics(self):
        return dict(self.diagnostics_)

    def _group_records(self, records):
        grouped = defaultdict(list)
        for row in records:
            if "policy_id" not in row:
                raise ValueError("trajectory row missing 'policy_id'")
            grouped[str(row["policy_id"])].append(row)
        if not grouped:
            raise ValueError("at least one trajectory record is required")
        for rows in grouped.values():
            rows.sort(key=lambda r: self._float(r.get("time", len(rows)), 0.0))
        return grouped

    def _sequence_summary(self, rows):
        tokens = np.vstack([self._row_vector(row) for row in rows])
        if self.mode == "transformer":
            pooled = self._attention_pool(tokens)
        else:
            pooled = np.mean(tokens, axis=0)
        std = np.std(tokens, axis=0)
        final = tokens[-1] if len(tokens) else np.zeros(tokens.shape[1], dtype=float)
        delta = final - tokens[0] if len(tokens) else np.zeros(tokens.shape[1], dtype=float)
        return np.concatenate([
            pooled,
            std,
            final,
            delta,
            np.array([float(len(rows))], dtype=float),
        ])

    def _row_vector(self, row):
        numeric = [self._float(row.get(field, 0.0), 0.0) for field in self.NUMERIC_FIELDS]
        state = self._stable_bucket(row.get("state", ""), 997) / 997.0
        action = self._stable_bucket(row.get("action", ""), 991) / 991.0
        return np.asarray([state, action] + numeric, dtype=float)

    def _attention_pool(self, tokens):
        if len(tokens) == 0:
            return np.zeros(7, dtype=float)
        risk = tokens[:, 3] + tokens[:, 4] + 0.5 * np.maximum(tokens[:, 6], 0.0)
        risk = risk - float(np.max(risk))
        weights = np.exp(risk)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        return weights @ tokens

    def _fit_projection(self, summaries):
        keys = sorted(summaries)
        X = np.vstack([summaries[key] for key in keys])
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0) + max(self.ridge, 1e-12)
        Z = (X - self.mean_) / self.scale_
        try:
            _, svals, vt = np.linalg.svd(Z, full_matrices=False)
        except np.linalg.LinAlgError:
            svals = np.zeros(min(Z.shape), dtype=float)
            vt = np.zeros((min(Z.shape), Z.shape[1]), dtype=float)
        n_comp = min(max(1, self.latent_dim), vt.shape[0])
        components = vt[:n_comp]
        if n_comp < self.latent_dim:
            pad = np.zeros((self.latent_dim - n_comp, Z.shape[1]), dtype=float)
            components = np.vstack([components, pad])
        self.components_ = components
        self.singular_values_ = svals
        self.policy_summaries = summaries
        self.policy_features = {
            key: self._project(summary)
            for key, summary in summaries.items()
        }

    def _project(self, summary):
        if self.mean_ is None or self.scale_ is None or self.components_ is None:
            raise RuntimeError("encoder has not been fitted")
        z = (np.asarray(summary, dtype=float) - self.mean_) / self.scale_
        return np.asarray(self.components_ @ z, dtype=float)

    def _reconstruction_errors(self, summaries):
        if self.mean_ is None or self.scale_ is None or self.components_ is None:
            return 0.0, 0.0
        X = np.vstack([summaries[key] for key in sorted(summaries)])
        Z = (X - self.mean_) / self.scale_
        recon = (Z @ self.components_.T) @ self.components_
        baseline = float(np.mean(Z ** 2))
        err = float(np.mean((Z - recon) ** 2))
        return baseline, err

    def _contrastive_separation(self, risk_labels):
        if not self.class_centers_:
            return 0.0
        within = []
        between = []
        for policy_id, feat in self.policy_features.items():
            label = str(risk_labels.get(policy_id, ""))
            if label in self.class_centers_:
                within.append(float(np.linalg.norm(feat - self.class_centers_[label])))
        centers = list(self.class_centers_.values())
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                between.append(float(np.linalg.norm(centers[i] - centers[j])))
        return (float(np.mean(between)) if between else 0.0) / (
            float(np.mean(within)) + 1e-12 if within else 1.0
        )

    @staticmethod
    def _auto_risk_label(rows):
        vals = [SelfSupervisedTrajectoryEncoder._float(r.get("queue", 0.0), 0.0)
                + SelfSupervisedTrajectoryEncoder._float(r.get("wait", 0.0), 0.0)
                for r in rows]
        score = float(np.mean(vals)) if vals else 0.0
        if score < 3.0:
            return "low"
        if score < 8.0:
            return "medium"
        return "high"

    @staticmethod
    def _stable_bucket(value, modulo):
        text = str(value)
        total = 0
        for idx, ch in enumerate(text):
            total += (idx + 1) * ord(ch)
        return int(total % int(modulo))

    @staticmethod
    def _float(value, default):
        try:
            val = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(val):
            return float(default)
        return val


class TransformerTrajectoryEncoder(SelfSupervisedTrajectoryEncoder):
    """Self-supervised encoder with deterministic attention-style pooling."""

    def __init__(self, latent_dim=8, ridge=1e-6):
        super().__init__(latent_dim=latent_dim, mode="transformer", ridge=ridge)


class SelfSupervisedPolicyStateEncoder(SyntheticPolicyStateEncoder):
    """Synthetic policy-state encoder learned from unlabeled policy samples."""

    def __init__(
        self,
        problem,
        latent_dim=8,
        mode="masked",
        fit_pool_size=512,
        lengthscale=0.35,
        rng=None,
    ):
        self.problem = problem
        self.lengthscale = float(lengthscale)
        self.feature_dim = int(latent_dim)
        self.mode = str(mode)
        self.fit_pool_size = int(fit_pool_size)
        self.raw_encoder = SyntheticPolicyStateEncoder(problem, lengthscale=lengthscale)
        self.rng = rng or np.random.default_rng(12345)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self._fit_projection()

    def occupancy(self, x):
        raw = self._expanded_raw(x)
        if self.mean_ is None or self.scale_ is None or self.components_ is None:
            return raw[: self.feature_dim]
        z = (raw - self.mean_) / self.scale_
        feat = self.components_ @ z
        if len(feat) < self.feature_dim:
            feat = np.pad(feat, (0, self.feature_dim - len(feat)))
        return np.asarray(feat[: self.feature_dim], dtype=float)

    def state_space_candidates(self, *args, **kwargs):
        return self.raw_encoder.state_space_candidates(*args, **kwargs)

    def _fit_projection(self):
        pool = self._fit_pool()
        X = np.vstack([self._expanded_raw(x) for x in pool])
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0) + 1e-8
        Z = (X - self.mean_) / self.scale_
        try:
            _, _, vt = np.linalg.svd(Z, full_matrices=False)
        except np.linalg.LinAlgError:
            vt = np.zeros((min(Z.shape), Z.shape[1]), dtype=float)
        n_comp = min(max(1, self.feature_dim), vt.shape[0])
        components = vt[:n_comp]
        if n_comp < self.feature_dim:
            components = np.vstack([
                components,
                np.zeros((self.feature_dim - n_comp, Z.shape[1]), dtype=float),
            ])
        self.components_ = components

    def _fit_pool(self):
        rows = []
        if hasattr(self.problem, "structured_candidates"):
            rows.extend(self.problem.structured_candidates(
                n=max(10, self.fit_pool_size // 5),
                rng=self.rng,
            ))
        rows.extend(self.raw_encoder._raw_inverse_pool(
            max(10, self.fit_pool_size),
            self.rng,
        ))
        if not rows:
            rows = [self.problem.sample_random(self.rng) for _ in range(max(2, self.fit_pool_size))]
        return self._unique(rows)

    def _expanded_raw(self, x):
        raw = self.raw_encoder.occupancy(x)
        if self.mode == "transformer":
            risk = raw[1] + raw[4] + np.maximum(raw[5], 0.0)
            weights = np.exp(raw - float(np.max(raw)))
            weights = weights / max(float(np.sum(weights)), 1e-12)
            attended = weights * risk
            return np.concatenate([raw, raw ** 2, np.sin(np.pi * raw), attended])
        return np.concatenate([raw, raw ** 2, np.sin(np.pi * raw)])


class StateCoupledFeatureMap:
    """Feature map for the mean belief model.

    It combines normalized design variables and occupancy features.  The final
    GPR basis adds an intercept outside this object.
    """

    def __init__(self, problem, encoder=None, state_scale=0.2):
        self.problem = problem
        self.encoder = encoder or SyntheticPolicyStateEncoder(problem)
        self.state_scale = float(state_scale)
        d = int(problem.d)
        rho_d = int(self.encoder.feature_dim)
        self.feature_dim = 2 * d + rho_d

    def features(self, x):
        z = np.asarray(self.problem.normalize(x), dtype=float)
        rho = self.state_scale * self.encoder.occupancy(x)
        return np.concatenate([z, z ** 2, rho])


class TrafficTrajectoryEncoder:
    """Aggregate fresh-seed traffic trajectories into occupancy/risk features.

    Expected row fields are CSV-friendly: `policy_id`, `seed`, `time`, `state`,
    `action`, `occupancy`, `queue`, `wait`, `flow`, and `demand_shock`.
    Extra columns are ignored.
    """

    REQUIRED_FIELDS = ("policy_id", "state", "action")

    def __init__(self, records=None):
        self.policy_features: dict[str, np.ndarray] = {}
        self.policy_occupancy: dict[str, dict[str, float]] = {}
        self.policy_exposures: dict[str, dict[str, np.ndarray]] = {}
        if records is not None:
            self.fit_records(records)

    @classmethod
    def from_csv(cls, path):
        path = Path(path)
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return cls(rows)

    @staticmethod
    def missing_data_status(path):
        path = Path(path) if path else None
        if path is None or not path.exists():
            return {
                "status": "missing_data",
                "reason": "fresh-seed traffic trajectory log not found",
                "path": None if path is None else str(path),
            }
        return {"status": "available", "path": str(path)}

    def fit_records(self, records):
        grouped = defaultdict(list)
        for row in records:
            for field in self.REQUIRED_FIELDS:
                if field not in row:
                    raise ValueError(f"traffic trajectory row missing {field!r}")
            grouped[str(row["policy_id"])].append(row)
        self.policy_features.clear()
        self.policy_occupancy.clear()
        self.policy_exposures.clear()
        for policy_id, rows in grouped.items():
            self._fit_policy(policy_id, rows)
        return self

    def _fit_policy(self, policy_id, rows):
        occ = defaultdict(float)
        queue = []
        wait = []
        flow = []
        shock = []
        for row in rows:
            key = f"{row.get('state', '')}|{row.get('action', '')}"
            occ[key] += self._float(row.get("occupancy", 1.0), 1.0)
            queue.append(self._float(row.get("queue", 0.0), 0.0))
            wait.append(self._float(row.get("wait", 0.0), 0.0))
            flow.append(self._float(row.get("flow", 0.0), 0.0))
            shock.append(self._float(row.get("demand_shock", 0.0), 0.0))
        total_occ = max(float(sum(occ.values())), 1e-12)
        occ_norm = {key: float(value / total_occ) for key, value in occ.items()}
        probs = np.asarray(list(occ_norm.values()), dtype=float)
        entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))
        queue_arr = np.asarray(queue, dtype=float)
        wait_arr = np.asarray(wait, dtype=float)
        flow_arr = np.asarray(flow, dtype=float)
        shock_arr = np.asarray(shock, dtype=float)
        local_exposure = np.array([
            float(np.mean(queue_arr)) if len(queue_arr) else 0.0,
            float(np.mean(wait_arr)) if len(wait_arr) else 0.0,
            float(np.mean(np.maximum(flow_arr, 0.0))) if len(flow_arr) else 0.0,
        ], dtype=float)
        shared_exposure = np.array([
            float(np.mean(shock_arr)) if len(shock_arr) else 0.0,
            float(np.std(shock_arr)) if len(shock_arr) else 0.0,
        ], dtype=float)
        self.policy_occupancy[str(policy_id)] = occ_norm
        self.policy_exposures[str(policy_id)] = {
            "local": local_exposure,
            "shared": shared_exposure,
        }
        self.policy_features[str(policy_id)] = np.array([
            total_occ,
            entropy,
            local_exposure[0],
            local_exposure[1],
            local_exposure[2],
            shared_exposure[0],
            shared_exposure[1],
            float(len(rows)),
        ], dtype=float)

    def features(self, policy_id):
        return np.asarray(self.policy_features[str(policy_id)], dtype=float)

    def occupancy(self, policy_id):
        return dict(self.policy_occupancy[str(policy_id)])

    def risk_exposure(self, policy_id):
        return np.asarray(self.policy_exposures[str(policy_id)]["local"], dtype=float)

    def shared_shock_exposure(self, policy_id):
        return np.asarray(self.policy_exposures[str(policy_id)]["shared"], dtype=float)

    @staticmethod
    def _float(value, default):
        try:
            val = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(val):
            return float(default)
        return val
