"""Lightweight manifold representations for policy-state coupling.

The classes in this module intentionally avoid heavy dependencies.  They
learn a low-dimensional representation of policy-induced state summaries and
expose the same small surface used by `SyntheticPolicyStateEncoder`:
`occupancy/features`, `coupling_scores`, and state-space candidate inversion.
"""

from __future__ import annotations

import numpy as np


def _unique(candidates):
    seen = set()
    rows = []
    for x in candidates:
        x_tuple = tuple(int(v) for v in x)
        if x_tuple not in seen:
            seen.add(x_tuple)
            rows.append(x_tuple)
    return rows


class PCAManifoldEncoder:
    """PCA/whitened low-dimensional state-policy manifold."""

    def __init__(
        self,
        problem,
        latent_dim=8,
        fit_pool_size=512,
        lengthscale=0.35,
        rng=None,
        auto_fit=True,
    ):
        self.problem = problem
        self.latent_dim = int(latent_dim)
        self.feature_dim = int(latent_dim)
        self.fit_pool_size = int(fit_pool_size)
        self.lengthscale = float(lengthscale)
        self.rng = rng or np.random.default_rng(12345)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.train_raw_: np.ndarray | None = None
        self.train_x_: list[tuple[int, ...]] = []
        self.train_features_: np.ndarray | None = None
        self.diagnostics_: dict[str, float | int | str | bool] = {
            "encoder": "pca_manifold",
            "status": "unfit",
        }
        if auto_fit:
            self.fit()

    def fit(self, records_or_policy_pool=None):
        rows = self._policy_pool(records_or_policy_pool)
        if not rows:
            rows = [self.problem.sample_random(self.rng) for _ in range(max(2, self.fit_pool_size))]
        rows = _unique(rows)
        X = np.vstack([self._raw_feature(x) for x in rows])
        self.train_x_ = rows
        self.train_raw_ = X
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
        fallback = False
        if n_comp < self.latent_dim:
            fallback = True
            components = np.vstack([
                components,
                np.zeros((self.latent_dim - n_comp, Z.shape[1]), dtype=float),
            ])
        self.components_ = components
        self.singular_values_ = svals
        self.train_features_ = np.vstack([self.occupancy(x) for x in rows])
        explained = float(np.sum(svals[:n_comp] ** 2) / max(np.sum(svals ** 2), 1e-12))
        self.diagnostics_ = {
            "encoder": "pca_manifold",
            "status": "fit",
            "n_train": int(len(rows)),
            "raw_dim": int(X.shape[1]),
            "latent_dim": int(self.latent_dim),
            "explained_energy": explained,
            "fallback_small_sample": bool(fallback),
        }
        return self

    def features(self, x_or_policy_id):
        return self.occupancy(x_or_policy_id)

    def occupancy(self, x):
        raw = self._raw_feature(x)
        if self.mean_ is None or self.scale_ is None or self.components_ is None:
            return raw[: self.feature_dim]
        z = (raw - self.mean_) / self.scale_
        feat = self.components_ @ z
        if len(feat) < self.feature_dim:
            feat = np.pad(feat, (0, self.feature_dim - len(feat)))
        return np.asarray(feat[: self.feature_dim], dtype=float)

    def diagnostics(self):
        return dict(self.diagnostics_)

    def distance(self, x, y):
        return float(np.linalg.norm(self.occupancy(x) - self.occupancy(y)))

    def kernel(self, x, y):
        dist = self.distance(x, y)
        return float(np.exp(-0.5 * (dist / max(self.lengthscale, 1e-8)) ** 2))

    def propagation_scores(self, candidates, observed):
        observed = self._observed_x(observed)
        if not candidates:
            return np.zeros(0, dtype=float)
        if not observed:
            return np.ones(len(candidates), dtype=float)
        cand = np.vstack([self.occupancy(x) for x in candidates])
        obs = np.vstack([self.occupancy(x) for x in observed])
        dist2 = np.sum((cand[:, None, :] - obs[None, :, :]) ** 2, axis=2)
        sim = np.exp(-0.5 * dist2 / max(self.lengthscale, 1e-8) ** 2)
        return self._normalize01(1.0 - np.clip(np.max(sim, axis=1), 0.0, 1.0))

    def coupling_scores(self, candidates, observed):
        coverage = self.propagation_scores(candidates, observed)
        history = self._observed_history(observed)
        if len(candidates) == 0 or not history:
            return coverage
        z_alpha = 1.6448536269514722
        sigma = float(getattr(self.problem, "sigma_level", 0.04))
        feasible = []
        for x, y in history:
            margin = float(y[1]) + z_alpha * sigma - float(getattr(self.problem, "tau", 0.0))
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
        promising = self._normalize01(np.max(sim * weights[None, :], axis=1))
        return self._normalize01(0.70 * promising + 0.30 * coverage)

    def state_space_candidates(
        self,
        n_anchors=10,
        inverse_pool_size=500,
        inverse_neighbors=1,
        rng=None,
        observed=None,
    ):
        return self.inverse_candidates(
            n_anchors=n_anchors,
            inverse_pool_size=inverse_pool_size,
            inverse_neighbors=inverse_neighbors,
            rng=rng,
            observed=observed,
        )

    def inverse_candidates(
        self,
        n_anchors=10,
        inverse_pool_size=500,
        inverse_neighbors=1,
        rng=None,
        observed=None,
    ):
        rng = rng or self.rng
        n_anchors = max(0, int(n_anchors))
        inverse_neighbors = max(1, int(inverse_neighbors))
        if n_anchors <= 0:
            return []

        anchor_rows = self._problem_state_inverse_candidates(
            n_anchors=n_anchors,
            inverse_neighbors=inverse_neighbors,
            rng=rng,
        )
        if anchor_rows:
            self.diagnostics_["last_inverse_mode"] = "problem_state_anchor"
            self.diagnostics_["last_inverse_count"] = int(len(anchor_rows))
            return anchor_rows

        pool = self._raw_inverse_pool(inverse_pool_size, rng, observed)
        if not pool:
            return []
        feats = np.vstack([self.occupancy(x) for x in pool])
        if self.train_features_ is not None and len(self.train_features_):
            lo = np.percentile(self.train_features_, 5, axis=0)
            hi = np.percentile(self.train_features_, 95, axis=0)
        else:
            lo = np.min(feats, axis=0)
            hi = np.max(feats, axis=0)
        anchors = rng.uniform(lo, hi, size=(n_anchors, feats.shape[1]))
        chosen = []
        for anchor in anchors:
            dist = np.linalg.norm(feats - anchor[None, :], axis=1)
            for idx in np.argsort(dist)[:inverse_neighbors]:
                chosen.append(pool[int(idx)])
        chosen = _unique(chosen)
        self.diagnostics_["last_inverse_mode"] = "latent_nearest_neighbor"
        self.diagnostics_["last_inverse_count"] = int(len(chosen))
        return chosen

    def _problem_state_inverse_candidates(self, n_anchors, inverse_neighbors, rng):
        if not (
            hasattr(self.problem, "state_anchor_points")
            and hasattr(self.problem, "inverse_state_anchor")
        ):
            return []
        try:
            anchors = self.problem.state_anchor_points(n=n_anchors, rng=rng)
        except Exception:
            return []
        rows = []
        for anchor in anchors or []:
            try:
                rows.extend(self.problem.inverse_state_anchor(
                    anchor,
                    rng=rng,
                    n=inverse_neighbors,
                ))
            except Exception:
                continue
        return _unique(rows)

    def _policy_pool(self, records_or_policy_pool=None):
        if records_or_policy_pool is not None:
            rows = []
            for item in records_or_policy_pool:
                if isinstance(item, dict):
                    if "x" in item:
                        rows.append(self._parse_x(item["x"]))
                else:
                    rows.append(tuple(int(v) for v in item))
            return _unique([x for x in rows if x is not None])
        rows = []
        if hasattr(self.problem, "structured_candidates"):
            rows.extend(self.problem.structured_candidates(
                n=max(10, self.fit_pool_size // 5),
                rng=self.rng,
            ))
        if hasattr(self.problem, "all_axis_solutions") and self.problem.d <= 20:
            axis = list(self.problem.all_axis_solutions())
            if axis:
                idx = np.linspace(0, len(axis) - 1, min(len(axis), self.fit_pool_size // 2))
                rows.extend(axis[int(round(i))] for i in idx)
        rows.extend(self._raw_inverse_pool(max(10, self.fit_pool_size), self.rng))
        return _unique(rows)[: max(2, self.fit_pool_size)]

    def _raw_inverse_pool(self, n, rng, observed=None):
        pool = []
        pool.extend(self._observed_x(observed or []))
        if hasattr(self.problem, "structured_candidates"):
            pool.extend(self.problem.structured_candidates(n=max(5, int(n) // 10), rng=rng))
        for _ in range(max(0, int(n))):
            pool.append(self.problem.sample_random(rng))
        return _unique(pool)

    def _raw_feature(self, x):
        z = np.asarray(self.problem.normalize(x), dtype=float)
        if len(z) == 0:
            z = np.array([0.0], dtype=float)
        tail = z[1:] if len(z) > 1 else z
        quantiles = np.quantile(z, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        hist, _ = np.histogram(z, bins=np.linspace(0.0, 1.0, 9), density=False)
        hist = hist.astype(float) / max(float(len(z)), 1.0)
        head = np.zeros(8, dtype=float)
        head[: min(8, len(z))] = z[: min(8, len(z))]
        moments = np.array([
            float(np.mean(z)),
            float(np.std(z)),
            float(np.mean(tail)),
            float(np.std(tail)),
            float(np.min(z)),
            float(np.max(z)),
            float(np.linalg.norm(z - 0.5) / np.sqrt(len(z))),
            float(np.sin(np.pi * z[0])),
            float(np.cos(np.pi * z[0])),
        ], dtype=float)
        state = self._problem_state_summary(x)
        return np.concatenate([moments, quantiles, hist, head, state])

    def _problem_state_summary(self, x):
        target = self.problem
        if not hasattr(target, "policy_state") and hasattr(target, "base"):
            target = target.base
        if not hasattr(target, "policy_state"):
            return np.zeros(8, dtype=float)
        try:
            state = np.asarray(target.policy_state(x), dtype=float)
        except Exception:
            return np.zeros(8, dtype=float)
        if state.ndim != 1 or len(state) == 0 or not np.all(np.isfinite(state)):
            return np.zeros(8, dtype=float)
        u = float(state[0]) if len(state) > 0 else 0.0
        q = float(state[1]) if len(state) > 1 else 0.0
        spread = float(state[2]) if len(state) > 2 else 0.0
        q_star = float(getattr(target, "q_star", 0.72))
        return np.asarray([
            u,
            q,
            spread,
            u ** 2,
            q ** 2,
            spread ** 2,
            abs(q - q_star),
            np.sin(np.pi * u),
        ], dtype=float)

    @staticmethod
    def _parse_x(value):
        if isinstance(value, str):
            vals = [int(float(v)) for v in value.replace(",", " ").split() if v.strip()]
            return tuple(vals) if vals else None
        try:
            return tuple(int(v) for v in value)
        except TypeError:
            return None

    @staticmethod
    def _observed_x(observed):
        rows = []
        for item in observed or []:
            if isinstance(item, tuple) and len(item) == 2 and not np.isscalar(item[1]):
                rows.append(tuple(int(v) for v in item[0]))
            else:
                rows.append(tuple(int(v) for v in item))
        return rows

    @staticmethod
    def _observed_history(observed):
        rows = []
        for item in observed or []:
            if isinstance(item, tuple) and len(item) == 2 and not np.isscalar(item[1]):
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


class KernelManifoldEncoder(PCAManifoldEncoder):
    """RBF-kernel manifold encoder with PCA fallback for tiny samples."""

    def __init__(self, *args, gamma=None, **kwargs):
        self.gamma = gamma
        self.alpha_: np.ndarray | None = None
        self.eigvals_: np.ndarray | None = None
        self.train_z_: np.ndarray | None = None
        self.gamma_: float | None = None
        self.kernel_col_mean_: np.ndarray | None = None
        self.kernel_total_mean_: float = 0.0
        self._fallback_pca: PCAManifoldEncoder | None = None
        auto_fit = bool(kwargs.pop("auto_fit", True))
        super().__init__(*args, auto_fit=False, **kwargs)
        if auto_fit:
            self.fit()

    def fit(self, records_or_policy_pool=None):
        rows = self._policy_pool(records_or_policy_pool)
        rows = _unique(rows)
        if len(rows) < max(4, self.latent_dim + 1):
            self._fallback_pca = PCAManifoldEncoder(
                self.problem,
                latent_dim=self.latent_dim,
                fit_pool_size=self.fit_pool_size,
                lengthscale=self.lengthscale,
                rng=self.rng,
                auto_fit=False,
            ).fit(rows)
            self.train_x_ = self._fallback_pca.train_x_
            self.train_features_ = self._fallback_pca.train_features_
            self.diagnostics_ = dict(self._fallback_pca.diagnostics())
            self.diagnostics_.update({
                "encoder": "kernel_manifold",
                "status": "fit_pca_fallback",
                "fallback_small_sample": True,
            })
            return self

        X = np.vstack([self._raw_feature(x) for x in rows])
        self.train_x_ = rows
        self.train_raw_ = X
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0) + 1e-8
        Z = (X - self.mean_) / self.scale_
        self.train_z_ = Z
        gamma = self._gamma(Z)
        self.gamma_ = float(gamma)
        dist2 = self._sqdist(Z, Z)
        K = np.exp(-gamma * dist2)
        self.kernel_col_mean_ = np.mean(K, axis=0)
        self.kernel_total_mean_ = float(np.mean(K))
        Kc = K - self.kernel_col_mean_[None, :] - self.kernel_col_mean_[:, None] + self.kernel_total_mean_
        try:
            eigvals, eigvecs = np.linalg.eigh(0.5 * (Kc + Kc.T))
        except np.linalg.LinAlgError:
            return self._fit_kernel_fallback(rows)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        n_comp = min(max(1, self.latent_dim), eigvecs.shape[1])
        vals = eigvals[:n_comp]
        vecs = eigvecs[:, :n_comp]
        scale = np.sqrt(np.maximum(vals, 1e-12))
        self.alpha_ = vecs / scale[None, :]
        self.eigvals_ = vals
        self.train_features_ = np.vstack([self.occupancy(x) for x in rows])
        energy = float(np.sum(vals) / max(np.sum(eigvals), 1e-12))
        self.diagnostics_ = {
            "encoder": "kernel_manifold",
            "status": "fit",
            "n_train": int(len(rows)),
            "raw_dim": int(X.shape[1]),
            "latent_dim": int(self.latent_dim),
            "kernel_gamma": float(gamma),
            "explained_energy": energy,
            "fallback_small_sample": False,
        }
        return self

    def occupancy(self, x):
        if self._fallback_pca is not None:
            return self._fallback_pca.occupancy(x)
        if (
            self.mean_ is None
            or self.scale_ is None
            or self.train_z_ is None
            or self.alpha_ is None
            or self.kernel_col_mean_ is None
        ):
            return super().occupancy(x)
        z = (self._raw_feature(x) - self.mean_) / self.scale_
        gamma = float(self.gamma_ if self.gamma_ is not None else self._gamma(self.train_z_))
        k = np.exp(-gamma * np.sum((self.train_z_ - z[None, :]) ** 2, axis=1))
        row_mean = float(np.mean(k))
        kc = k - row_mean - self.kernel_col_mean_ + self.kernel_total_mean_
        feat = kc @ self.alpha_
        if len(feat) < self.feature_dim:
            feat = np.pad(feat, (0, self.feature_dim - len(feat)))
        return np.asarray(feat[: self.feature_dim], dtype=float)

    def _fit_kernel_fallback(self, rows):
        self._fallback_pca = PCAManifoldEncoder(
            self.problem,
            latent_dim=self.latent_dim,
            fit_pool_size=self.fit_pool_size,
            lengthscale=self.lengthscale,
            rng=self.rng,
            auto_fit=False,
        ).fit(rows)
        self.diagnostics_ = dict(self._fallback_pca.diagnostics())
        self.diagnostics_.update({"encoder": "kernel_manifold", "status": "fit_pca_fallback"})
        return self

    def _gamma(self, Z):
        if self.gamma is not None:
            return max(float(self.gamma), 1e-12)
        if len(Z) <= 1:
            return 1.0
        dist2 = self._sqdist(Z, Z)
        vals = dist2[np.triu_indices_from(dist2, k=1)]
        med = float(np.median(vals[vals > 1e-12])) if np.any(vals > 1e-12) else 1.0
        return float(1.0 / max(med, 1e-12))

    @staticmethod
    def _sqdist(A, B):
        aa = np.sum(A ** 2, axis=1)[:, None]
        bb = np.sum(B ** 2, axis=1)[None, :]
        return np.maximum(aa + bb - 2.0 * A @ B.T, 0.0)


class GraphLaplacianEncoder(PCAManifoldEncoder):
    """Diffusion-map / graph-Laplacian state-policy representation.

    This is a dependency-light implementation of the graph-structured coupling
    route from the roadmap.  It builds a kNN graph over policy-state summaries,
    extracts the leading non-trivial diffusion coordinates, and uses RBF
    Nyström interpolation for new policies.  Candidate inversion is inherited
    from the manifold encoder, so it can still use exact synthetic state anchors
    when the problem provides them.
    """

    def __init__(
        self,
        *args,
        n_neighbors=12,
        diffusion_time=1.0,
        gamma=None,
        **kwargs,
    ):
        self.n_neighbors = int(n_neighbors)
        self.diffusion_time = float(diffusion_time)
        self.gamma = gamma
        self.train_z_: np.ndarray | None = None
        self.gamma_: float | None = None
        self.eigvals_: np.ndarray | None = None
        self.train_degrees_: np.ndarray | None = None
        self._fallback_pca: PCAManifoldEncoder | None = None
        auto_fit = bool(kwargs.pop("auto_fit", True))
        super().__init__(*args, auto_fit=False, **kwargs)
        if auto_fit:
            self.fit()

    def fit(self, records_or_policy_pool=None):
        rows = _unique(self._policy_pool(records_or_policy_pool))
        min_rows = max(6, self.latent_dim + 2)
        if len(rows) < min_rows:
            self._fallback_pca = PCAManifoldEncoder(
                self.problem,
                latent_dim=self.latent_dim,
                fit_pool_size=self.fit_pool_size,
                lengthscale=self.lengthscale,
                rng=self.rng,
                auto_fit=False,
            ).fit(rows)
            self.train_x_ = self._fallback_pca.train_x_
            self.train_features_ = self._fallback_pca.train_features_
            self.diagnostics_ = dict(self._fallback_pca.diagnostics())
            self.diagnostics_.update({
                "encoder": "graph_laplacian",
                "status": "fit_pca_fallback",
                "fallback_small_sample": True,
            })
            return self

        X = np.vstack([self._raw_feature(x) for x in rows])
        self.train_x_ = rows
        self.train_raw_ = X
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0) + 1e-8
        Z = (X - self.mean_) / self.scale_
        self.train_z_ = Z
        gamma = self._gamma(Z)
        self.gamma_ = float(gamma)
        dist2 = KernelManifoldEncoder._sqdist(Z, Z)
        W = self._knn_affinity(dist2, gamma)
        deg = np.sum(W, axis=1)
        self.train_degrees_ = np.maximum(deg, 1e-12)
        D_inv_sqrt = 1.0 / np.sqrt(self.train_degrees_)
        S = D_inv_sqrt[:, None] * W * D_inv_sqrt[None, :]
        try:
            eigvals, eigvecs = np.linalg.eigh(0.5 * (S + S.T))
        except np.linalg.LinAlgError:
            return self._fit_graph_fallback(rows)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        start = 1 if len(eigvals) > 1 else 0
        stop = min(start + max(1, self.latent_dim), eigvecs.shape[1])
        vals = np.maximum(eigvals[start:stop], 0.0)
        vecs = eigvecs[:, start:stop]
        if vecs.shape[1] < self.latent_dim:
            vecs = np.hstack([
                vecs,
                np.zeros((vecs.shape[0], self.latent_dim - vecs.shape[1]), dtype=float),
            ])
            vals = np.pad(vals, (0, self.latent_dim - len(vals)))
        scale = np.power(np.maximum(vals[: self.latent_dim], 1e-12), self.diffusion_time)
        self.eigvals_ = vals[: self.latent_dim]
        self.train_features_ = np.asarray(vecs[:, : self.latent_dim] * scale[None, :], dtype=float)
        energy = float(np.sum(np.maximum(vals, 0.0)) / max(np.sum(np.maximum(eigvals[start:], 0.0)), 1e-12))
        self.diagnostics_ = {
            "encoder": "graph_laplacian",
            "status": "fit",
            "n_train": int(len(rows)),
            "raw_dim": int(X.shape[1]),
            "latent_dim": int(self.latent_dim),
            "n_neighbors": int(min(max(1, self.n_neighbors), len(rows) - 1)),
            "kernel_gamma": float(gamma),
            "diffusion_time": float(self.diffusion_time),
            "explained_energy": energy,
            "fallback_small_sample": False,
        }
        return self

    def occupancy(self, x):
        if self._fallback_pca is not None:
            return self._fallback_pca.occupancy(x)
        if (
            self.mean_ is None
            or self.scale_ is None
            or self.train_z_ is None
            or self.train_features_ is None
        ):
            return super().occupancy(x)
        z = (self._raw_feature(x) - self.mean_) / self.scale_
        gamma = float(self.gamma_ if self.gamma_ is not None else self._gamma(self.train_z_))
        k = np.exp(-gamma * np.sum((self.train_z_ - z[None, :]) ** 2, axis=1))
        if np.all(k <= 1e-300):
            return np.zeros(self.feature_dim, dtype=float)
        weights = k / max(float(np.sum(k)), 1e-12)
        feat = weights @ self.train_features_
        if len(feat) < self.feature_dim:
            feat = np.pad(feat, (0, self.feature_dim - len(feat)))
        return np.asarray(feat[: self.feature_dim], dtype=float)

    def _fit_graph_fallback(self, rows):
        self._fallback_pca = PCAManifoldEncoder(
            self.problem,
            latent_dim=self.latent_dim,
            fit_pool_size=self.fit_pool_size,
            lengthscale=self.lengthscale,
            rng=self.rng,
            auto_fit=False,
        ).fit(rows)
        self.train_x_ = self._fallback_pca.train_x_
        self.train_features_ = self._fallback_pca.train_features_
        self.diagnostics_ = dict(self._fallback_pca.diagnostics())
        self.diagnostics_.update({"encoder": "graph_laplacian", "status": "fit_pca_fallback"})
        return self

    def _knn_affinity(self, dist2, gamma):
        n = dist2.shape[0]
        W = np.zeros_like(dist2, dtype=float)
        k = min(max(1, self.n_neighbors), max(1, n - 1))
        for i in range(n):
            order = np.argsort(dist2[i])
            neigh = [idx for idx in order if idx != i][:k]
            for j in neigh:
                W[i, j] = float(np.exp(-gamma * dist2[i, j]))
        W = np.maximum(W, W.T)
        np.fill_diagonal(W, 0.0)
        return W

    def _gamma(self, Z):
        if self.gamma is not None:
            return max(float(self.gamma), 1e-12)
        if len(Z) <= 1:
            return 1.0
        dist2 = KernelManifoldEncoder._sqdist(Z, Z)
        vals = dist2[np.triu_indices_from(dist2, k=1)]
        med = float(np.median(vals[vals > 1e-12])) if np.any(vals > 1e-12) else 1.0
        return float(1.0 / max(med, 1e-12))


class ManifoldRiskDecomposer:
    """Split a variance estimate into tangent/normal/shared/residual blocks."""

    def __init__(self, encoder, tangent_dim=None, residual_fraction=0.05):
        self.encoder = encoder
        self.tangent_dim = tangent_dim
        self.residual_fraction = float(residual_fraction)

    def decompose(self, x, total_variance=None):
        feat = np.asarray(self.encoder.occupancy(x), dtype=float)
        if feat.ndim != 1 or len(feat) == 0:
            feat = np.zeros(1, dtype=float)
        total = float(total_variance) if total_variance is not None else float(np.sum(feat ** 2))
        total = max(total, 1e-12)
        tdim = self.tangent_dim
        if tdim is None:
            tdim = max(1, len(feat) // 2)
        tdim = int(np.clip(tdim, 1, len(feat)))
        tangent_energy = float(np.sum(feat[:tdim] ** 2))
        normal_energy = float(np.sum(feat[tdim:] ** 2))
        shared_energy = float(np.mean(np.abs(feat)) ** 2)
        residual_energy = max(float(self.residual_fraction), 0.0)
        energies = np.array([
            max(tangent_energy, 0.0),
            max(normal_energy, 0.0),
            max(shared_energy, 0.0),
            residual_energy,
        ], dtype=float)
        if float(np.sum(energies)) <= 1e-14:
            energies = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        weights = energies / float(np.sum(energies))
        tangent, normal, shared, residual = weights * total
        return {
            "tangent": float(tangent),
            "normal": float(normal),
            "shared": float(shared),
            "residual": float(residual),
            "total": float(total),
            "latent_dim": int(len(feat)),
            "tangent_dim": int(tdim),
        }
