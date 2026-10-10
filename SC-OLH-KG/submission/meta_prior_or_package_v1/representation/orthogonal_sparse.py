"""Low-frequency orthogonal sparse state-coupled representations.

This module implements the first concrete step of the LF-OS-SC-HVD-KG
direction: build a small library of state-coupled risk functions, measure their
graph frequency on the policy-induced ``psi=(A,N)`` cloud, retain only a few
low-frequency components, orthogonalize them, and expose the discarded energy
as a conservative residual floor for chance certification.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from core.cumulative_risk import get_risk_exposure, vech_quadratic_features


def _unique(candidates):
    seen = set()
    rows = []
    for x in candidates:
        x_tuple = tuple(int(v) for v in x)
        if x_tuple not in seen:
            seen.add(x_tuple)
            rows.append(x_tuple)
    return rows


@dataclass
class OrthogonalSparseDiagnostics:
    encoder: str
    status: str
    n_train: int
    library_dim: int
    active_dim: int
    low_frequency_components: int
    max_offdiag_gram: float
    gram_condition_number: float
    residual_floor_mean: float
    min_low_frequency_ratio: float
    mean_low_frequency_ratio: float


class LowFrequencyOrthogonalSparsePolicyEncoder:
    """State-coupled low-frequency orthogonal sparse basis.

    The encoder is deliberately dependency-light.  It is not a full OAK GP; it
    supplies the low-frequency, whitened active coordinates that the current
    GPR/HVD stack can consume immediately.  Sparse evidence updates can be added
    on top of this object without changing its public API.
    """

    def __init__(
        self,
        problem,
        latent_dim=8,
        fit_pool_size=512,
        max_library_size=30,
        low_frequency_components=8,
        max_active=None,
        n_neighbors=12,
        lengthscale=0.35,
        residual_floor_scale=0.05,
        ridge=1e-8,
        use_problem_state_anchor=True,
        rng=None,
        records_or_policy_pool=None,
        auto_fit=True,
    ):
        self.problem = problem
        self.latent_dim = int(latent_dim)
        self.feature_dim = int(latent_dim)
        self.fit_pool_size = int(fit_pool_size)
        self.max_library_size = int(max_library_size)
        self.low_frequency_components = int(low_frequency_components)
        self.max_active = int(max_active) if max_active is not None else int(latent_dim)
        self.n_neighbors = int(n_neighbors)
        self.lengthscale = float(lengthscale)
        self.residual_floor_scale = float(residual_floor_scale)
        self.ridge = float(ridge)
        self.use_problem_state_anchor = bool(use_problem_state_anchor)
        self.rng = rng or np.random.default_rng(12345)
        self.records_or_policy_pool = records_or_policy_pool

        self.train_x_: list[tuple[int, ...]] = []
        self.psi_mean_: np.ndarray | None = None
        self.psi_scale_: np.ndarray | None = None
        self.basis_mean_: np.ndarray | None = None
        self.basis_scale_: np.ndarray | None = None
        self.active_idx_: np.ndarray | None = None
        self.low_ratio_: np.ndarray | None = None
        self.whitening_: np.ndarray | None = None
        self.train_features_: np.ndarray | None = None
        self._feature_cache: OrderedDict[tuple[int, ...], np.ndarray] = OrderedDict()
        self._residual_floor_cache: OrderedDict[tuple[int, ...], float] = OrderedDict()
        self._cache_limit = max(128, min(1024, 2 * int(self.fit_pool_size)))
        self._inverse_pool_rows_: list[tuple[int, ...]] = []
        self._inverse_pool_features_: np.ndarray | None = None
        self._inverse_pool_target_size_: int = 0
        self.library_names_: list[str] = []
        self.diagnostics_: dict = {
            "encoder": "lf_os",
            "status": "unfit",
        }
        if auto_fit:
            self.fit(records_or_policy_pool)

    def fit(self, records_or_policy_pool=None):
        rows = self._policy_pool(records_or_policy_pool)
        if not rows:
            rows = [
                self.problem.sample_random(self.rng)
                for _ in range(max(2, self.fit_pool_size))
            ]
        rows = _unique(rows)[: max(2, self.fit_pool_size)]
        self.train_x_ = rows

        psi = np.vstack([self._psi_coordinates(x) for x in rows])
        self.psi_mean_ = np.mean(psi, axis=0)
        self.psi_scale_ = np.std(psi, axis=0) + 1e-8
        psi_z = (psi - self.psi_mean_) / self.psi_scale_

        basis, names = self._basis_library_from_psi(psi)
        self.library_names_ = names
        self.basis_mean_ = np.mean(basis, axis=0)
        self.basis_scale_ = np.std(basis, axis=0) + 1e-8
        basis_z = (basis - self.basis_mean_) / self.basis_scale_

        low_ratio = self._low_frequency_ratio(psi_z, basis_z)
        self.low_ratio_ = low_ratio
        order = np.argsort(-low_ratio)
        n_active = min(
            max(1, self.max_active),
            max(1, self.feature_dim),
            max(1, len(order)),
        )
        active = np.sort(order[:n_active])
        self.active_idx_ = active

        active_basis = basis_z[:, active] * low_ratio[active][None, :]
        active_basis = active_basis - np.mean(active_basis, axis=0, keepdims=True)
        cov = active_basis.T @ active_basis / max(float(len(active_basis)), 1.0)
        cov = 0.5 * (cov + cov.T)
        try:
            _, svals, vt = np.linalg.svd(active_basis, full_matrices=False)
        except np.linalg.LinAlgError:
            svals = np.ones(active_basis.shape[1], dtype=float)
            vt = np.eye(active_basis.shape[1], dtype=float)
        tol = max(self.ridge, 1e-10) * max(active_basis.shape)
        rank = int(np.sum(svals > tol))
        rank = max(1, min(rank, active_basis.shape[1], self.feature_dim))
        scale = np.sqrt(max(float(len(active_basis)), 1.0)) / np.maximum(
            svals[:rank],
            self.ridge,
        )
        self.whitening_ = vt[:rank].T * scale[None, :]
        train_active = active_basis @ self.whitening_
        if train_active.shape[1] < self.feature_dim:
            train_active = np.hstack([
                train_active,
                np.zeros(
                    (train_active.shape[0], self.feature_dim - train_active.shape[1]),
                    dtype=float,
                ),
            ])
        self.train_features_ = train_active[:, : self.feature_dim]

        gram = self.train_features_.T @ self.train_features_ / max(float(len(rows)), 1.0)
        off = gram - np.diag(np.diag(gram))
        diag = np.diag(gram)
        pos_diag = diag[diag > 1e-12]
        cond = float(np.max(pos_diag) / np.min(pos_diag)) if len(pos_diag) else 0.0
        floor_vals = np.array([self.residual_floor(x) for x in rows], dtype=float)
        diag_obj = OrthogonalSparseDiagnostics(
            encoder="lf_os",
            status="fit",
            n_train=int(len(rows)),
            library_dim=int(basis_z.shape[1]),
            active_dim=int(rank),
            low_frequency_components=int(min(
                self.low_frequency_components,
                len(rows),
            )),
            max_offdiag_gram=float(np.max(np.abs(off))) if off.size else 0.0,
            gram_condition_number=cond,
            residual_floor_mean=float(np.mean(floor_vals)) if len(floor_vals) else 0.0,
            min_low_frequency_ratio=float(np.min(low_ratio)) if len(low_ratio) else 0.0,
            mean_low_frequency_ratio=float(np.mean(low_ratio)) if len(low_ratio) else 0.0,
        )
        self.diagnostics_ = dict(diag_obj.__dict__)
        self.diagnostics_["active_names"] = [
            self.library_names_[int(i)] for i in active
        ]
        self._feature_cache.clear()
        self._residual_floor_cache.clear()
        self._inverse_pool_rows_ = []
        self._inverse_pool_features_ = None
        self._inverse_pool_target_size_ = 0
        return self

    def features(self, x):
        return self.occupancy(x)

    def features_many(self, X):
        return self.occupancy_many(X)

    def occupancy(self, x):
        key = self._as_tuple(x)
        cached = self._cache_get(self._feature_cache, key)
        if cached is not None:
            return cached.copy()
        if (
            self.basis_mean_ is None
            or self.basis_scale_ is None
            or self.active_idx_ is None
            or self.low_ratio_ is None
            or self.whitening_ is None
        ):
            raw = self._psi_coordinates(x)
            if len(raw) < self.feature_dim:
                raw = np.pad(raw, (0, self.feature_dim - len(raw)))
            out = np.asarray(raw[: self.feature_dim], dtype=float)
            self._cache_put(self._feature_cache, key, out)
            return out.copy()
        basis, _ = self._basis_library_from_psi(self._psi_coordinates(x)[None, :])
        z = (basis[0] - self.basis_mean_) / self.basis_scale_
        active = z[self.active_idx_] * self.low_ratio_[self.active_idx_]
        active = active @ self.whitening_
        if len(active) < self.feature_dim:
            active = np.pad(active, (0, self.feature_dim - len(active)))
        out = np.asarray(active[: self.feature_dim], dtype=float)
        self._cache_put(self._feature_cache, key, out)
        return out.copy()

    def occupancy_many(self, X):
        rows = [self._as_tuple(x) for x in X]
        if not rows:
            return np.empty((0, self.feature_dim), dtype=float)
        out = np.empty((len(rows), self.feature_dim), dtype=float)
        missing_pos = []
        missing_rows = []
        for pos, row in enumerate(rows):
            cached = self._cache_get(self._feature_cache, row)
            if cached is None:
                missing_pos.append(pos)
                missing_rows.append(row)
            else:
                out[pos] = cached
        if missing_rows:
            feats = self._occupancy_many_uncached(missing_rows)
            for pos, row, feat in zip(missing_pos, missing_rows, feats):
                out[pos] = feat
                self._cache_put(self._feature_cache, row, feat)
        return out

    def _occupancy_many_uncached(self, rows):
        if (
            self.basis_mean_ is None
            or self.basis_scale_ is None
            or self.active_idx_ is None
            or self.low_ratio_ is None
            or self.whitening_ is None
        ):
            psi = np.vstack([self._psi_coordinates(x) for x in rows])
            if psi.shape[1] < self.feature_dim:
                psi = np.hstack([
                    psi,
                    np.zeros((len(psi), self.feature_dim - psi.shape[1]), dtype=float),
                ])
            return np.asarray(psi[:, : self.feature_dim], dtype=float)
        psi = np.vstack([self._psi_coordinates(x) for x in rows])
        basis, _ = self._basis_library_from_psi(psi)
        z = (basis - self.basis_mean_[None, :]) / self.basis_scale_[None, :]
        active = z[:, self.active_idx_] * self.low_ratio_[self.active_idx_][None, :]
        feats = active @ self.whitening_
        if feats.shape[1] < self.feature_dim:
            feats = np.hstack([
                feats,
                np.zeros((len(feats), self.feature_dim - feats.shape[1]), dtype=float),
            ])
        return np.asarray(feats[:, : self.feature_dim], dtype=float)

    def residual_floor(self, x, output_index=None):
        """Conservative floor for pruned/high-frequency components."""
        del output_index
        key = self._as_tuple(x)
        cached = self._cache_get(self._residual_floor_cache, key)
        if cached is not None:
            return float(cached)
        if self.basis_mean_ is None or self.basis_scale_ is None:
            return 0.0
        basis, _ = self._basis_library_from_psi(self._psi_coordinates(x)[None, :])
        z = (basis[0] - self.basis_mean_) / self.basis_scale_
        active = set([] if self.active_idx_ is None else map(int, self.active_idx_))
        weights = np.ones_like(z)
        if self.low_ratio_ is not None and len(self.low_ratio_) == len(z):
            weights = np.maximum(1.0 - self.low_ratio_, 0.0)
        mask = np.array([j not in active for j in range(len(z))], dtype=bool)
        if not np.any(mask):
            return 0.0
        energy = np.mean(weights[mask] * z[mask] ** 2)
        out = float(max(self.residual_floor_scale * energy, 0.0))
        self._cache_put(self._residual_floor_cache, key, out)
        return out

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
        return self.propagation_scores(candidates, observed)

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
            n_anchors,
            inverse_neighbors,
            rng,
        )
        if anchor_rows:
            self.diagnostics_["last_inverse_mode"] = "problem_state_anchor"
            self.diagnostics_["last_inverse_count"] = int(len(anchor_rows))
            return anchor_rows
        pool, feats = self._raw_inverse_pool_with_features(
            inverse_pool_size,
            rng,
            observed,
        )
        if not pool:
            return []
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
        self.diagnostics_["last_inverse_mode"] = "lf_os_nearest_neighbor"
        self.diagnostics_["last_inverse_count"] = int(len(chosen))
        return chosen

    def _psi_coordinates(self, x):
        exposure = get_risk_exposure(self.problem, x)
        if exposure is not None:
            psi = np.concatenate([exposure.A, exposure.N]).astype(float)
            if len(psi):
                return psi
        z = np.asarray(self.problem.normalize(x), dtype=float)
        if len(z) == 0:
            return np.zeros(1, dtype=float)
        tail = z[1:] if len(z) > 1 else z
        return np.asarray([
            float(np.mean(z)),
            float(np.std(z)),
            float(np.min(z)),
            float(np.max(z)),
            float(np.mean(tail)),
            float(np.std(tail)),
            float(np.linalg.norm(z - 0.5) / np.sqrt(max(1, len(z)))),
            float(np.sin(np.pi * z[0])),
            float(np.cos(np.pi * z[0])),
        ], dtype=float)

    def _basis_library_from_psi(self, psi):
        P = np.asarray(psi, dtype=float)
        if P.ndim == 1:
            P = P[None, :]
        cols = []
        names = []
        p_dim = P.shape[1]
        for j in range(p_dim):
            cols.append(P[:, j])
            names.append(f"psi{j}")
        for j in range(p_dim):
            cols.append(P[:, j] ** 2)
            names.append(f"psi{j}^2")
        for j in range(p_dim):
            cols.append(np.abs(P[:, j]))
            names.append(f"|psi{j}|")
        if p_dim:
            quad = np.vstack([vech_quadratic_features(row) for row in P])
            q_names = []
            for i in range(p_dim):
                for j in range(i, p_dim):
                    q_names.append(f"psi{i}*psi{j}" if i == j else f"2*psi{i}*psi{j}")
            for j in range(quad.shape[1]):
                cols.append(quad[:, j])
                names.append(q_names[j])
        if not cols:
            return np.ones((len(P), 1), dtype=float), ["constant"]
        B = np.vstack(cols).T
        keep = min(max(1, self.max_library_size), B.shape[1])
        return np.asarray(B[:, :keep], dtype=float), names[:keep]

    def _low_frequency_ratio(self, psi_z, basis_z):
        n = len(psi_z)
        if n <= 2:
            return np.ones(basis_z.shape[1], dtype=float)
        dist2 = self._sqdist(psi_z, psi_z)
        vals = dist2[np.triu_indices_from(dist2, k=1)]
        med = float(np.median(vals[vals > 1e-12])) if np.any(vals > 1e-12) else 1.0
        gamma = 1.0 / max(med, 1e-12)
        W = np.exp(-gamma * dist2)
        np.fill_diagonal(W, 0.0)
        k = min(max(1, self.n_neighbors), n - 1)
        if k < n - 1:
            keep = np.zeros_like(W, dtype=bool)
            for i in range(n):
                idx = np.argsort(dist2[i])[: k + 1]
                keep[i, idx] = True
            keep = np.logical_or(keep, keep.T)
            W = np.where(keep, W, 0.0)
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        try:
            eigvals, eigvecs = np.linalg.eigh(0.5 * (L + L.T))
        except np.linalg.LinAlgError:
            return np.ones(basis_z.shape[1], dtype=float)
        order = np.argsort(eigvals)
        eigvecs = eigvecs[:, order]
        Lmax = min(max(1, self.low_frequency_components), eigvecs.shape[1])
        U = eigvecs[:, :Lmax]
        total = np.sum(basis_z ** 2, axis=0) + 1e-12
        low = np.sum((U.T @ basis_z) ** 2, axis=0)
        return np.clip(low / total, 0.0, 1.0)

    def _problem_state_inverse_candidates(self, n_anchors, inverse_neighbors, rng):
        if not self.use_problem_state_anchor:
            return []
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
                if isinstance(item, dict) and "x" in item:
                    parsed = self._parse_x(item["x"])
                    if parsed is not None:
                        rows.append(parsed)
                elif not isinstance(item, dict):
                    rows.append(tuple(int(v) for v in item))
            return _unique(rows)
        rows = []
        if hasattr(self.problem, "structured_candidates"):
            rows.extend(self.problem.structured_candidates(
                n=max(10, self.fit_pool_size // 5),
                rng=self.rng,
            ))
        if hasattr(self.problem, "all_axis_solutions") and self.problem.d <= 200:
            axis = list(self.problem.all_axis_solutions())
            if axis:
                idx = np.linspace(0, len(axis) - 1, min(len(axis), self.fit_pool_size // 2))
                rows.extend(axis[int(round(i))] for i in idx)
        rows.extend(self._raw_inverse_pool_uncached(max(10, self.fit_pool_size), self.rng))
        return _unique(rows)[: max(2, self.fit_pool_size)]

    def _raw_inverse_pool(self, n, rng, observed=None):
        pool, _ = self._raw_inverse_pool_with_features(n, rng, observed)
        return pool

    def _raw_inverse_pool_with_features(self, n, rng, observed=None):
        base_rows, base_feats = self._cached_inverse_pool(max(0, int(n)), rng)
        rows = list(base_rows)
        feats = np.asarray(base_feats, dtype=float)
        observed_rows = [
            row for row in self._observed_x(observed or [])
            if row not in set(rows)
        ]
        if observed_rows:
            obs_feats = self.occupancy_many(observed_rows)
            rows.extend(observed_rows)
            feats = (
                obs_feats if len(feats) == 0
                else np.vstack([feats, obs_feats])
            )
        return _unique(rows), feats[: len(_unique(rows))]

    def _cached_inverse_pool(self, n, rng):
        n = max(0, int(n))
        if n == 0:
            return [], np.empty((0, self.feature_dim), dtype=float)
        if (
            self._inverse_pool_features_ is not None
            and self._inverse_pool_target_size_ >= n
            and len(self._inverse_pool_rows_) >= n
        ):
            rows = self._inverse_pool_rows_[:n]
            return rows, self._inverse_pool_features_[: len(rows)]
        rows = self._raw_inverse_pool_uncached(n, rng)
        feats = self.occupancy_many(rows) if rows else np.empty((0, self.feature_dim), dtype=float)
        self._inverse_pool_rows_ = list(rows)
        self._inverse_pool_features_ = np.asarray(feats, dtype=float)
        self._inverse_pool_target_size_ = int(n)
        return self._inverse_pool_rows_, self._inverse_pool_features_

    def _raw_inverse_pool_uncached(self, n, rng):
        pool = []
        if hasattr(self.problem, "structured_candidates"):
            pool.extend(self.problem.structured_candidates(n=max(5, int(n) // 10), rng=rng))
        for _ in range(max(0, int(n))):
            pool.append(self.problem.sample_random(rng))
        return _unique(pool)

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
    def _as_tuple(x):
        return tuple(int(v) for v in np.asarray(x, dtype=int))

    @staticmethod
    def _cache_get(cache, key):
        try:
            value = cache.pop(key)
        except KeyError:
            return None
        cache[key] = value
        return value

    def _cache_put(self, cache, key, value):
        cache[key] = np.asarray(value, dtype=float) if not np.isscalar(value) else float(value)
        while len(cache) > self._cache_limit:
            cache.popitem(last=False)

    @staticmethod
    def _sqdist(A, B):
        aa = np.sum(A ** 2, axis=1)[:, None]
        bb = np.sum(B ** 2, axis=1)[None, :]
        return np.maximum(aa + bb - 2.0 * A @ B.T, 0.0)

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
