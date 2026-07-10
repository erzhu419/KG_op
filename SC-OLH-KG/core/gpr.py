"""Parametric GPR belief model with cached/vectorized feature helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np


ArrayLike = Union[np.ndarray, list, tuple]


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
        numeric_backend: str = "numpy",
        numeric_backend_device: str = "auto",
        torch_dtype: str = "float64",
        torch_min_rows: int = 128,
    ):
        self.d = int(d)
        self.lambda_i = float(lambda_i)
        self.normalize_func = normalize_func
        self.basis_map = basis_map
        self.basis_config = basis_config or BasisConfig()
        self.numeric_backend = str(numeric_backend or "numpy").lower()
        self.numeric_backend_device = str(numeric_backend_device or "auto")
        self.torch_dtype = str(torch_dtype or "float64").lower()
        self.torch_min_rows = max(1, int(torch_min_rows))
        self._last_torch_import_error = None

        self.p = self._infer_basis_dim()
        self.a = np.zeros(self.p, dtype=float)
        self.C = float(prior_var) * np.eye(self.p)

        self.sampled_set: list[tuple[int, ...]] = []
        self.sol_to_idx: dict[tuple[int, ...], int] = {}
        self._state_version = 0
        self._torch_cache = {}
        if self.numeric_backend in ("torch", "torch_cuda", "cuda"):
            if self._import_torch() is None:
                raise RuntimeError(
                    "numeric backend requested torch, but torch is not importable: "
                    f"{self._last_torch_import_error}"
                )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_torch_cache"] = {}
        return state

    def _invalidate_backend_cache(self):
        self._state_version += 1
        self._torch_cache = {}

    def _import_torch(self):
        try:
            import torch  # noqa: WPS433
        except Exception as exc:
            self._last_torch_import_error = repr(exc)
            return None
        self._last_torch_import_error = None
        return torch

    def _torch_dtype_obj(self, torch):
        if self.torch_dtype in ("float32", "single"):
            return torch.float32
        return torch.float64

    def _resolve_torch_device(self, torch):
        requested = self.numeric_backend_device
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("numeric backend requested CUDA, but torch.cuda is unavailable")
        return device

    def _torch_context(self, rows=0, force=False):
        """Return `(torch, device, dtype)` when the optional torch backend is active."""
        backend = self.numeric_backend
        if backend in ("", "numpy", "np"):
            return None
        if not force and int(rows) > 0 and int(rows) < self.torch_min_rows:
            return None
        torch = self._import_torch()
        if torch is None:
            if backend in ("torch", "torch_cuda", "cuda"):
                raise RuntimeError(
                    "numeric backend requested torch, but torch is not importable: "
                    f"{self._last_torch_import_error}"
                )
            return None
        if backend in ("auto", "cuda_auto") and not torch.cuda.is_available():
            return None
        if backend in ("torch_cuda", "cuda") and not torch.cuda.is_available():
            raise RuntimeError("numeric backend requested CUDA, but torch.cuda is unavailable")
        device = self._resolve_torch_device(torch)
        if backend in ("auto", "cuda_auto") and device.type != "cuda":
            return None
        return torch, device, self._torch_dtype_obj(torch)

    def backend_status(self):
        torch = self._import_torch()
        effective = "numpy"
        device = None
        cuda_available = False
        if torch is not None:
            cuda_available = bool(torch.cuda.is_available())
            try:
                ctx = self._torch_context(rows=self.torch_min_rows, force=True)
            except RuntimeError:
                ctx = None
            if ctx is not None and self.numeric_backend not in ("numpy", "np", ""):
                effective = "torch"
                device = str(ctx[1])
        return {
            "requested_backend": self.numeric_backend,
            "effective_backend": effective,
            "device": device,
            "torch_available": bool(torch is not None),
            "cuda_available": bool(cuda_available),
            "torch_dtype": self.torch_dtype,
            "torch_min_rows": int(self.torch_min_rows),
        }

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
            if hasattr(self.basis_map, "features_many"):
                feats = np.asarray(self.basis_map.features_many(X), dtype=float)
            else:
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
        ctx = self._torch_context(rows=len(A))
        if ctx is not None:
            torch, device, dtype = ctx
            with torch.no_grad():
                A_t = torch.as_tensor(A, dtype=dtype, device=device)
                a_t, _ = self.torch_state(rows=len(A), force=True)
                return (A_t @ a_t).detach().cpu().numpy()
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
        ctx = self._torch_context(rows=len(A))
        if ctx is not None:
            torch, device, dtype = ctx
            with torch.no_grad():
                A_t = torch.as_tensor(A, dtype=dtype, device=device)
                _, C_t = self.torch_state(rows=len(A), force=True)
                var_t = torch.sum((A_t @ C_t) * A_t, dim=1)
                var = var_t.detach().cpu().numpy()
        else:
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
        self._invalidate_backend_cache()

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
        self._invalidate_backend_cache()

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
        self._invalidate_backend_cache()

    def torch_state(self, rows=0, force=False):
        ctx = self._torch_context(rows=rows, force=force)
        if ctx is None:
            return None
        torch, device, dtype = ctx
        key = (self._state_version, str(device), str(dtype))
        cached = self._torch_cache.get(key)
        if cached is not None:
            return cached
        with torch.no_grad():
            a_t = torch.as_tensor(self.a, dtype=dtype, device=device)
            C_t = torch.as_tensor(self.C, dtype=dtype, device=device)
        self._torch_cache = {key: (a_t, C_t)}
        return a_t, C_t

    def augmented_feature_matrix_torch(
        self,
        X: list[ArrayLike] | np.ndarray,
        rows=0,
        force=False,
    ):
        ctx = self._torch_context(rows=rows or len(X), force=force)
        if ctx is None:
            return None
        torch, device, dtype = ctx
        A = self.augmented_feature_matrix(X)
        return torch.as_tensor(A, dtype=dtype, device=device)
