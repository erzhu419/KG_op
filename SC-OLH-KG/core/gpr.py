"""Parametric GPR belief model with cached/vectorized feature helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


@dataclass
class BasisConfig:
    """Configuration for the default quadratic basis."""

    normalize: bool = True


class ParametricGPR:
    """Parametric GPR with solution-specific deviations.

    The model is the modular equivalent of the original repo's `ParametricGPR`:

        f(x) = phi(x)^T beta + zeta(x)

    `beta` is a low-dimensional parametric surface and `zeta(x)` is activated
    when a solution is visited.  The implementation adds vectorized feature
    builders so KG can evaluate a candidate set without rebuilding the same
    augmented features repeatedly.
    """

    def __init__(
        self,
        d: int,
        lambda_i: float = 0.1,
        prior_var: float = 100.0,
        normalize_func: Optional[Callable[[ArrayLike], np.ndarray]] = None,
        basis_map=None,
        basis_config: BasisConfig | None = None,
    ):
        self.d = int(d)
        self.lambda_i = float(lambda_i)
        self.normalize_func = normalize_func
        self.basis_map = basis_map
        self.basis_config = basis_config or BasisConfig()

        self.p = self._infer_basis_dim()
        self.a = np.zeros(self.p, dtype=float)
        self.C = float(prior_var) * np.eye(self.p)

        self.sampled_set: list[tuple[int, ...]] = []
        self.sol_to_idx: dict[tuple[int, ...], int] = {}

    def _infer_basis_dim(self) -> int:
        if self.basis_map is not None:
            return 1 + int(self.basis_map.feature_dim)
        return 2 * self.d + 1

    def _normalize(self, x: ArrayLike) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if self.normalize_func is not None and self.basis_config.normalize:
            return np.asarray(self.normalize_func(x_arr), dtype=float)
        return x_arr

    def basis(self, x: ArrayLike) -> np.ndarray:
        """Return the parametric basis vector including intercept."""
        if self.basis_map is not None:
            feats = np.asarray(self.basis_map.features(x), dtype=float)
            return np.concatenate([[1.0], feats])
        z = self._normalize(x)
        return np.concatenate([[1.0], z, z ** 2])

    def basis_matrix(self, X: list[ArrayLike] | np.ndarray) -> np.ndarray:
        """Vectorized basis matrix for a candidate list."""
        if len(X) == 0:
            return np.empty((0, self.p), dtype=float)
        if self.basis_map is not None:
            feats = np.vstack([self.basis_map.features(x) for x in X])
            return np.column_stack([np.ones(len(feats)), feats])
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if self.normalize_func is not None and self.basis_config.normalize:
            Z = np.vstack([self.normalize_func(row) for row in X_arr])
        else:
            Z = X_arr
        return np.column_stack([np.ones(len(Z)), Z, Z ** 2])

    def augmented_feature(self, x: ArrayLike) -> np.ndarray:
        """Return augmented feature `(phi(x), e_x)`."""
        x_tuple = tuple(int(v) for v in np.asarray(x, dtype=int))
        phi = self.basis(x)
        e = np.zeros(len(self.sampled_set), dtype=float)
        idx = self.sol_to_idx.get(x_tuple)
        if idx is not None:
            e[idx] = 1.0
        return np.concatenate([phi, e])

    def augmented_feature_matrix(
        self,
        X: list[ArrayLike] | np.ndarray,
    ) -> np.ndarray:
        """Vectorized augmented feature matrix.

        For unvisited candidates the deviation block is zero, matching the
        current GPR-KG behavior before dimension augmentation.
        """
        Phi = self.basis_matrix(X)
        n = len(Phi)
        if len(self.sampled_set) == 0:
            return Phi
        E = np.zeros((n, len(self.sampled_set)), dtype=float)
        for row, x in enumerate(X):
            x_tuple = tuple(int(v) for v in np.asarray(x, dtype=int))
            idx = self.sol_to_idx.get(x_tuple)
            if idx is not None:
                E[row, idx] = 1.0
        return np.hstack([Phi, E])

    def posterior_mean(self, x: ArrayLike) -> float:
        return float(self.augmented_feature(x) @ self.a)

    def posterior_mean_many(self, X: list[ArrayLike] | np.ndarray) -> np.ndarray:
        A = self.augmented_feature_matrix(X)
        return A @ self.a

    def posterior_var(self, x: ArrayLike) -> float:
        x_tuple = tuple(int(v) for v in np.asarray(x, dtype=int))
        e = self.augmented_feature(x)
        var = float(e @ self.C @ e)
        if x_tuple not in self.sol_to_idx:
            var += self.lambda_i
        return max(var, 1e-12)

    def posterior_var_many(self, X: list[ArrayLike] | np.ndarray) -> np.ndarray:
        """Vectorized posterior variance for candidate pools."""
        if len(X) == 0:
            return np.zeros(0, dtype=float)
        A = self.augmented_feature_matrix(X)
        var = np.einsum("ij,jk,ik->i", A, self.C, A)
        unseen = np.array([
            tuple(int(v) for v in np.asarray(x, dtype=int)) not in self.sol_to_idx
            for x in X
        ], dtype=bool)
        var = np.asarray(var, dtype=float)
        var[unseen] += self.lambda_i
        return np.maximum(var, 1e-12)

    def dimension_augment(self, x: ArrayLike) -> None:
        x_tuple = tuple(int(v) for v in np.asarray(x, dtype=int))
        if x_tuple in self.sol_to_idx:
            return
        idx = len(self.sampled_set)
        self.sampled_set.append(x_tuple)
        self.sol_to_idx[x_tuple] = idx

        self.a = np.concatenate([self.a, [0.0]])
        n = len(self.a)
        C_new = np.zeros((n, n), dtype=float)
        C_new[: n - 1, : n - 1] = self.C
        C_new[n - 1, n - 1] = self.lambda_i
        self.C = C_new

    def set_parametric_prior(
        self,
        beta_mean: np.ndarray,
        lambda_i: float,
        prior_var: float,
    ) -> None:
        """Reset to a data-driven parametric prior."""
        beta_mean = np.asarray(beta_mean, dtype=float)
        if len(beta_mean) != self.p:
            raise ValueError(f"beta length {len(beta_mean)} != basis dim {self.p}")
        self.lambda_i = max(float(lambda_i), 1e-12)
        self.a = beta_mean.copy()
        self.C = max(float(prior_var), 1e-12) * np.eye(self.p)
        self.sampled_set = []
        self.sol_to_idx = {}

    def update(self, x: ArrayLike, y: float, sigma2_hat: float) -> None:
        """Rank-one Kalman update using the plug-in observation variance."""
        self.dimension_augment(x)
        e = self.augmented_feature(x)
        Ce = self.C @ e
        denom = max(float(sigma2_hat) + float(e @ Ce), 1e-15)
        gain = Ce / denom
        innovation = float(y) - float(e @ self.a)
        self.a = self.a + gain * innovation
        self.C = self.C - np.outer(gain, Ce)
        self.C = 0.5 * (self.C + self.C.T)
