"""Dependency-light self-supervised policy/trajectory encoders."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .manifold import PCAManifoldEncoder, _unique


class MaskedTrajectoryEncoder(PCAManifoldEncoder):
    """Masked-reconstruction encoder for policies or trajectory CSV rows."""

    NUMERIC_FIELDS = ("occupancy", "queue", "wait", "flow", "demand_shock")

    def __init__(
        self,
        problem=None,
        latent_dim=8,
        fit_pool_size=512,
        rng=None,
        records_or_policy_pool=None,
        **kwargs,
    ):
        self.policy_features: dict[str, np.ndarray] = {}
        self.policy_summaries: dict[str, np.ndarray] = {}
        self.policy_x: dict[str, tuple[int, ...]] = {}
        self.raw_to_latent_beta_: np.ndarray | None = None
        self.raw_mean_: np.ndarray | None = None
        self.raw_scale_: np.ndarray | None = None
        self.pretext_mode = "masked"
        auto_fit = bool(kwargs.pop("auto_fit", True))
        if problem is None:
            self.problem = None
            self.latent_dim = int(latent_dim)
            self.feature_dim = int(latent_dim)
            self.fit_pool_size = int(fit_pool_size)
            self.lengthscale = float(kwargs.get("lengthscale", 0.35))
            self.rng = rng or np.random.default_rng(12345)
            self.mean_ = None
            self.scale_ = None
            self.components_ = None
            self.singular_values_ = None
            self.train_raw_ = None
            self.train_x_ = []
            self.train_features_ = None
            self.policy_x = {}
            self.raw_to_latent_beta_ = None
            self.raw_mean_ = None
            self.raw_scale_ = None
            self.diagnostics_ = {"encoder": "ssl_masked", "status": "unfit"}
            if auto_fit and records_or_policy_pool is not None:
                self.fit(records_or_policy_pool)
        else:
            super().__init__(
                problem,
                latent_dim=latent_dim,
                fit_pool_size=fit_pool_size,
                rng=rng,
                auto_fit=False,
                **kwargs,
            )
            if auto_fit:
                self.fit(records_or_policy_pool)

    def fit(self, records_or_policy_pool=None):
        if records_or_policy_pool is not None and not isinstance(records_or_policy_pool, (list, tuple)):
            records_or_policy_pool = list(records_or_policy_pool)
        if self._looks_like_records(records_or_policy_pool):
            return self._fit_records(records_or_policy_pool)
        out = super().fit(records_or_policy_pool)
        baseline, recon = self._projection_reconstruction_error(self.train_raw_)
        self.diagnostics_.update({
            "encoder": "ssl_masked",
            "pretext": "masked_reconstruction",
            "masked_baseline_mse": float(baseline),
            "masked_reconstruction_mse": float(recon),
        })
        return out

    def features(self, x_or_policy_id):
        return self.occupancy(x_or_policy_id)

    def occupancy(self, x_or_policy_id):
        key = str(x_or_policy_id)
        if key in self.policy_features:
            return np.asarray(self.policy_features[key], dtype=float)
        if self.raw_to_latent_beta_ is not None and self.problem is not None:
            raw = self._raw_feature(x_or_policy_id)
            z = (raw - self.raw_mean_) / self.raw_scale_
            aug = np.concatenate([[1.0], z])
            feat = aug @ self.raw_to_latent_beta_
            if len(feat) < self.feature_dim:
                feat = np.pad(feat, (0, self.feature_dim - len(feat)))
            return np.asarray(feat[: self.feature_dim], dtype=float)
        if (
            self.problem is not None
            and self.mean_ is not None
            and len(self.mean_) != len(self._raw_feature(x_or_policy_id))
        ):
            return np.zeros(self.feature_dim, dtype=float)
        return super().occupancy(x_or_policy_id)

    def _raw_feature(self, x):
        base = super()._raw_feature(x)
        masked_context = np.concatenate([
            base,
            base ** 2,
            np.sin(np.pi * base),
            np.cos(np.pi * base),
        ])
        return masked_context

    def _fit_records(self, records):
        grouped = self._group_records(records)
        summaries = {
            policy_id: self._sequence_summary(rows)
            for policy_id, rows in grouped.items()
        }
        self._fit_summary_projection(summaries)
        self._fit_raw_to_latent_projection(grouped)
        baseline, recon = self._projection_reconstruction_error(
            np.vstack([summaries[key] for key in sorted(summaries)])
        )
        self.diagnostics_ = {
            "encoder": "ssl_masked",
            "status": "fit_records",
            "pretext": "masked_reconstruction",
            "n_policies": int(len(summaries)),
            "n_policies_with_x": int(len(self.policy_x)),
            "latent_dim": int(self.latent_dim),
            "masked_baseline_mse": float(baseline),
            "masked_reconstruction_mse": float(recon),
        }
        return self

    def _fit_summary_projection(self, summaries):
        keys = sorted(summaries)
        X = np.vstack([summaries[key] for key in keys])
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0) + 1e-8
        Z = (X - self.mean_) / self.scale_
        try:
            _, svals, vt = np.linalg.svd(Z, full_matrices=False)
        except np.linalg.LinAlgError:
            svals = np.zeros(min(Z.shape), dtype=float)
            vt = np.zeros((min(Z.shape), Z.shape[1]), dtype=float)
        n_comp = min(max(1, self.latent_dim), vt.shape[0])
        components = vt[:n_comp]
        if n_comp < self.latent_dim:
            components = np.vstack([
                components,
                np.zeros((self.latent_dim - n_comp, Z.shape[1]), dtype=float),
            ])
        self.components_ = components
        self.singular_values_ = svals
        self.policy_summaries = summaries
        self.policy_features = {
            key: self._project_summary(summary)
            for key, summary in summaries.items()
        }

    def _fit_raw_to_latent_projection(self, grouped):
        self.policy_x = {}
        self.raw_to_latent_beta_ = None
        self.raw_mean_ = None
        self.raw_scale_ = None
        if self.problem is None:
            return
        raw_rows = []
        target_rows = []
        for policy_id, rows in grouped.items():
            x = self._policy_x_from_rows(rows)
            if x is None or policy_id not in self.policy_features:
                continue
            self.policy_x[str(policy_id)] = x
            raw_rows.append(self._raw_feature(x))
            target_rows.append(np.asarray(self.policy_features[policy_id], dtype=float))
        if not raw_rows:
            return
        R = np.vstack(raw_rows)
        Y = np.vstack(target_rows)
        self.raw_mean_ = np.mean(R, axis=0)
        self.raw_scale_ = np.std(R, axis=0) + 1e-8
        Z = (R - self.raw_mean_) / self.raw_scale_
        A = np.column_stack([np.ones(len(Z), dtype=float), Z])
        reg = 1e-3 * np.eye(A.shape[1])
        reg[0, 0] = 0.0
        try:
            self.raw_to_latent_beta_ = np.linalg.solve(A.T @ A + reg, A.T @ Y)
        except np.linalg.LinAlgError:
            self.raw_to_latent_beta_ = np.linalg.lstsq(A.T @ A + reg, A.T @ Y, rcond=None)[0]

    def _policy_x_from_rows(self, rows):
        for row in rows:
            if "x" not in row:
                continue
            x = self._parse_x(row.get("x"))
            if x is not None:
                return x
        return None

    def _project_summary(self, summary):
        z = (np.asarray(summary, dtype=float) - self.mean_) / self.scale_
        feat = self.components_ @ z
        if len(feat) < self.feature_dim:
            feat = np.pad(feat, (0, self.feature_dim - len(feat)))
        return np.asarray(feat[: self.feature_dim], dtype=float)

    def _projection_reconstruction_error(self, X):
        if X is None or self.mean_ is None or self.scale_ is None or self.components_ is None:
            return 0.0, 0.0
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Z = (X - self.mean_) / self.scale_
        recon = (Z @ self.components_.T) @ self.components_
        return float(np.mean(Z ** 2)), float(np.mean((Z - recon) ** 2))

    @staticmethod
    def _looks_like_records(value):
        if value is None:
            return False
        rows = list(value)
        return bool(rows) and isinstance(rows[0], dict) and "policy_id" in rows[0]

    def _group_records(self, records):
        grouped = defaultdict(list)
        for row in records:
            if "policy_id" not in row:
                raise ValueError("trajectory row missing 'policy_id'")
            grouped[str(row["policy_id"])].append(row)
        if not grouped:
            raise ValueError("at least one trajectory record is required")
        for rows in grouped.values():
            rows.sort(key=lambda r: self._float(r.get("time", 0.0), 0.0))
        return grouped

    def _sequence_summary(self, rows):
        tokens = np.vstack([self._row_vector(row) for row in rows])
        pooled = np.mean(tokens, axis=0)
        std = np.std(tokens, axis=0)
        final = tokens[-1]
        delta = final - tokens[0]
        return np.concatenate([pooled, std, final, delta, [float(len(rows))]])

    def _row_vector(self, row):
        numeric = [self._float(row.get(field, 0.0), 0.0) for field in self.NUMERIC_FIELDS]
        state = self._stable_bucket(row.get("state", ""), 997) / 997.0
        action = self._stable_bucket(row.get("action", ""), 991) / 991.0
        return np.asarray([state, action] + numeric, dtype=float)

    @staticmethod
    def _stable_bucket(value, modulo):
        total = 0
        for idx, ch in enumerate(str(value)):
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


class ContrastivePolicyEncoder(MaskedTrajectoryEncoder):
    """Risk-regime contrastive encoder using deterministic labels."""

    def __init__(self, *args, **kwargs):
        self.class_centers_: dict[str, np.ndarray] = {}
        super().__init__(*args, **kwargs)

    def fit(self, records_or_policy_pool=None):
        if records_or_policy_pool is not None and not isinstance(records_or_policy_pool, (list, tuple)):
            records_or_policy_pool = list(records_or_policy_pool)
        if self._looks_like_records(records_or_policy_pool):
            super()._fit_records(records_or_policy_pool)
            labels = self._record_labels(records_or_policy_pool)
        else:
            super().fit(records_or_policy_pool)
            labels = {
                str(x): str(self.problem.risk_class(x)) if hasattr(self.problem, "risk_class") else "0"
                for x in self.train_x_
            }
            self.policy_features = {
                str(x): self.occupancy(x)
                for x in self.train_x_
            }
        self._fit_centers(labels)
        self.diagnostics_.update({
            "encoder": "ssl_contrastive",
            "pretext": "contrastive_risk",
            "n_classes": int(len(self.class_centers_)),
            "contrastive_separation": float(self._contrastive_separation(labels)),
        })
        return self

    def _raw_feature(self, x):
        base = super()._raw_feature(x)
        if self.problem is not None and hasattr(self.problem, "risk_class"):
            label = float(self.problem.risk_class(x))
        else:
            label = 0.0
        return np.concatenate([base, [label / 10.0], base * (1.0 + label / 10.0)])

    def _fit_centers(self, labels):
        centers = defaultdict(list)
        for key, feat in self.policy_features.items():
            if key in labels:
                centers[str(labels[key])].append(np.asarray(feat, dtype=float))
        self.class_centers_ = {
            label: np.mean(np.vstack(vals), axis=0)
            for label, vals in centers.items()
            if vals
        }

    def _record_labels(self, records):
        grouped = self._group_records(records)
        labels = {}
        for policy_id, rows in grouped.items():
            risk = [
                self._float(row.get("queue", 0.0), 0.0)
                + self._float(row.get("wait", 0.0), 0.0)
                for row in rows
            ]
            score = float(np.mean(risk)) if risk else 0.0
            labels[str(policy_id)] = "low" if score < 3.0 else ("medium" if score < 8.0 else "high")
        return labels

    def _contrastive_separation(self, labels):
        if not self.class_centers_:
            return 0.0
        within = []
        between = []
        for key, feat in self.policy_features.items():
            label = str(labels.get(key, ""))
            if label in self.class_centers_:
                within.append(float(np.linalg.norm(feat - self.class_centers_[label])))
        vals = list(self.class_centers_.values())
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                between.append(float(np.linalg.norm(vals[i] - vals[j])))
        return (float(np.mean(between)) if between else 0.0) / (
            float(np.mean(within)) + 1e-12 if within else 1.0
        )


class NextRiskEncoder(MaskedTrajectoryEncoder):
    """Encoder with a next-risk/exposure prediction pretext."""

    def fit(self, records_or_policy_pool=None):
        out = super().fit(records_or_policy_pool)
        self.diagnostics_.update({
            "encoder": "ssl_next_risk",
            "pretext": "next_risk_prediction",
            "next_risk_proxy_mse": float(self._next_risk_proxy_mse()),
        })
        return out

    def _raw_feature(self, x):
        base = super()._raw_feature(x)
        if self.problem is not None and hasattr(self.problem, "true_sigma"):
            try:
                risk = float(self.problem.true_sigma(x)[1])
            except Exception:
                risk = float(np.std(base))
        else:
            risk = float(np.std(base))
        return np.concatenate([base, np.roll(base, -1) - base, [risk]])

    def _next_risk_proxy_mse(self):
        if self.train_raw_ is None or len(self.train_raw_) <= 1:
            return 0.0
        target = self.train_raw_[:, -1]
        pred = np.full_like(target, float(np.mean(target)))
        return float(np.mean((target - pred) ** 2))


class SmallTransformerEncoder(MaskedTrajectoryEncoder):
    """Small optional transformer-style encoder with deterministic fallback."""

    def __init__(self, *args, **kwargs):
        try:
            import torch  # noqa: F401
            self.torch_status = "torch_available"
        except Exception:
            self.torch_status = "torch_unavailable"
        super().__init__(*args, **kwargs)

    def fit(self, records_or_policy_pool=None):
        out = super().fit(records_or_policy_pool)
        self.diagnostics_.update({
            "encoder": "ssl_transformer",
            "pretext": "attention_sequence_pooling",
            "torch_status": self.torch_status,
            "used_fallback": bool(self.torch_status != "torch_available"),
        })
        return out

    def _raw_feature(self, x):
        base = super()._raw_feature(x)
        risk = base - float(np.max(base))
        weights = np.exp(np.clip(risk, -30.0, 30.0))
        weights = weights / max(float(np.sum(weights)), 1e-12)
        attended = weights * (base + np.roll(base, 1))
        return np.concatenate([base, attended, np.sin(np.pi * attended)])

    def _sequence_summary(self, rows):
        tokens = np.vstack([self._row_vector(row) for row in rows])
        risk = tokens[:, 3] + tokens[:, 4] + 0.5 * np.maximum(tokens[:, 6], 0.0)
        risk = risk - float(np.max(risk))
        weights = np.exp(np.clip(risk, -30.0, 30.0))
        weights = weights / max(float(np.sum(weights)), 1e-12)
        pooled = weights @ tokens
        std = np.std(tokens, axis=0)
        final = tokens[-1]
        delta = final - tokens[0]
        return np.concatenate([pooled, std, final, delta, [float(len(rows))]])


class HybridSSLPolicyEncoder(MaskedTrajectoryEncoder):
    """Contextual/contrastive policy encoder for state-coupled BO.

    This is the lightweight roadmap bridge between self-supervised features,
    contextual BO, and latent-space BO: the pretext feature keeps the generic
    masked-reconstruction basis, but augments it with policy-induced state
    moments and regime indicators.  It is still deterministic and dependency
    light, so it can be swept at 10k dimensions before deciding whether a
    heavier transformer is worth the cost.
    """

    def fit(self, records_or_policy_pool=None):
        out = super().fit(records_or_policy_pool)
        self.diagnostics_.update({
            "encoder": "ssl_hybrid",
            "pretext": "masked_contextual_contrastive",
            "contextual_coupling": True,
            "context_weight": 3.0,
        })
        return out

    def _raw_feature(self, x):
        base = super()._raw_feature(x)
        state = self._state_context(x)
        label = self._risk_label_value(x)
        state_block = np.concatenate([
            3.0 * state,
            2.0 * state ** 2,
            np.sin(np.pi * state),
            np.cos(np.pi * state),
        ])
        base_summary = np.asarray([
            float(np.mean(base)),
            float(np.std(base)),
            float(np.min(base)),
            float(np.max(base)),
            *np.quantile(base, [0.1, 0.25, 0.5, 0.75, 0.9]),
        ], dtype=float)
        interactions = np.concatenate([
            state * (1.0 + label),
            state * base_summary[: len(state)],
            0.35 * base[: min(32, len(base))],
            [label],
        ])
        return np.concatenate([state_block, interactions, base_summary, 0.20 * base])

    def _state_context(self, x):
        if self.problem is None:
            return np.zeros(8, dtype=float)
        target = self.problem.base if hasattr(self.problem, "base") else self.problem
        if not hasattr(target, "policy_state"):
            z = np.asarray(self.problem.normalize(x), dtype=float)
            if len(z) == 0:
                return np.zeros(8, dtype=float)
            tail = z[1:] if len(z) > 1 else z
            return np.asarray([
                float(z[0]),
                float(np.mean(tail)),
                float(np.std(tail)),
                float(np.mean(z)),
                float(np.std(z)),
                float(np.min(z)),
                float(np.max(z)),
                float(np.linalg.norm(z - 0.5) / np.sqrt(len(z))),
            ], dtype=float)
        try:
            u, q, spread = target.policy_state(x)
        except Exception:
            return np.zeros(8, dtype=float)
        q_ref = float(getattr(target, "reference_q", getattr(target, "q_star", 0.70)))
        return np.asarray([
            float(u),
            float(q),
            float(spread),
            float(abs(q - q_ref)),
            float(u * q),
            float(q * spread),
            float(np.sin(np.pi * u)),
            float(np.cos(np.pi * u)),
        ], dtype=float)

    def _risk_label_value(self, x):
        if self.problem is None:
            return 0.0
        target = self.problem.base if hasattr(self.problem, "base") else self.problem
        if not hasattr(target, "risk_class"):
            return 0.0
        try:
            label = float(target.risk_class(x))
        except Exception:
            return 0.0
        return float(np.clip(label / 50.0, 0.0, 4.0))
