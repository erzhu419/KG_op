"""Parametric GPR belief model with cached/vectorized feature helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from representation.adaptive_sparsity import AdaptiveSpikeSlabPosterior


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
        self._adaptive_sparsity = None
        self._adaptive_records = []
        self._adaptive_spec = None
        self._covariance_projection_count = 0
        self._min_quadratic_variance_seen = float("inf")
        self._max_abs_posterior_mean_seen = 0.0
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
        value = float(self.augmented_feature(x) @ self.a)
        if not np.isfinite(value):
            raise FloatingPointError("non-finite GPR posterior mean")
        self._max_abs_posterior_mean_seen = max(
            float(getattr(self, "_max_abs_posterior_mean_seen", 0.0)),
            abs(value),
        )
        return value

    def posterior_mean_many(self, X: list[ArrayLike] | np.ndarray) -> np.ndarray:
        A = self.augmented_feature_matrix(X)
        ctx = self._torch_context(rows=len(A))
        if ctx is not None:
            torch, device, dtype = ctx
            with torch.no_grad():
                A_t = torch.as_tensor(A, dtype=dtype, device=device)
                a_t, _ = self.torch_state(rows=len(A), force=True)
                values = (A_t @ a_t).detach().cpu().numpy()
        else:
            values = A @ self.a
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("non-finite GPR posterior means")
        if len(values):
            self._max_abs_posterior_mean_seen = max(
                float(getattr(self, "_max_abs_posterior_mean_seen", 0.0)),
                float(np.max(np.abs(values))),
            )
        return values

    def _project_covariance_psd(self) -> None:
        """Repair a covariance only after a negative quadratic form is detected."""

        covariance = np.asarray(self.C, dtype=float)
        if not np.all(np.isfinite(covariance)):
            raise FloatingPointError("non-finite GPR covariance")
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if not np.all(np.isfinite(eigenvalues)):
            raise FloatingPointError("non-finite GPR covariance eigenvalues")
        self.C = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        self.C = 0.5 * (self.C + self.C.T)
        self._covariance_projection_count = int(
            getattr(self, "_covariance_projection_count", 0)
        ) + 1

    @staticmethod
    def _negative_variance_tolerance(values) -> float:
        arr = np.asarray(values, dtype=float)
        scale = max(1.0, float(np.max(np.abs(arr))) if arr.size else 1.0)
        return 1e-10 * scale

    def _record_quadratic_variance(self, value: float) -> None:
        self._min_quadratic_variance_seen = min(
            float(getattr(self, "_min_quadratic_variance_seen", float("inf"))),
            float(value),
        )

    def posterior_var(self, x: ArrayLike) -> float:
        x_tuple = tuple(int(v) for v in np.asarray(x, dtype=int))
        e = self.augmented_feature(x)
        var = float(e @ self.C @ e)
        self._record_quadratic_variance(var)
        if var < -self._negative_variance_tolerance([var]):
            self._project_covariance_psd()
            var = float(e @ self.C @ e)
            self._record_quadratic_variance(var)
        var = max(var, 0.0)
        if x_tuple not in self.sol_to_idx:
            var += self.lambda_i
        var += self.adaptive_model_uncertainty(x)
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
        var = np.asarray(var, dtype=float)
        if not np.all(np.isfinite(var)):
            raise FloatingPointError("non-finite GPR posterior variances")
        if len(var):
            self._record_quadratic_variance(float(np.min(var)))
        if len(var) and float(np.min(var)) < -self._negative_variance_tolerance(var):
            self._project_covariance_psd()
            var = np.einsum("ij,jk,ik->i", A, self.C, A)
            self._record_quadratic_variance(float(np.min(var)))
        var = np.maximum(np.asarray(var, dtype=float), 0.0)
        unseen = np.array([
            tuple(int(v) for v in np.asarray(x, dtype=int)) not in self.sol_to_idx
            for x in X
        ], dtype=bool)
        var[unseen] += self.lambda_i
        var += self.adaptive_model_uncertainty_many(X)
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
        prior_var: float | np.ndarray,
    ) -> None:
        """Reset to a data-driven parametric prior.

        ``prior_var`` may be a scalar, a diagonal covariance vector, or a full
        covariance matrix.  The matrix path is used by the adaptive
        spike-and-slab posterior after transforming its standardized basis back
        to the raw GPR feature coordinates.
        """
        beta_mean = np.asarray(beta_mean, dtype=float)
        if len(beta_mean) != self.p:
            raise ValueError(f"beta length {len(beta_mean)} != basis dim {self.p}")
        if not np.all(np.isfinite(beta_mean)):
            raise ValueError("parametric prior mean must be finite")
        self.lambda_i = max(float(lambda_i), 1e-12)
        self.a = beta_mean.copy()
        prior = np.asarray(prior_var, dtype=float)
        if prior.ndim == 0:
            prior_diag = np.full(self.p, max(float(prior), 1e-12), dtype=float)
            covariance = np.diag(prior_diag)
        elif prior.ndim == 1:
            prior_diag = prior.reshape(-1)
            if len(prior_diag) != self.p:
                raise ValueError(
                    f"prior variance length {len(prior_diag)} != basis dim {self.p}")
            if not np.all(np.isfinite(prior_diag)):
                raise ValueError("prior variance must be finite")
            prior_diag = np.maximum(prior_diag, 1e-12)
            covariance = np.diag(prior_diag)
        elif prior.ndim == 2:
            if prior.shape != (self.p, self.p):
                raise ValueError(
                    f"prior covariance shape {prior.shape} != ({self.p}, {self.p})")
            if not np.all(np.isfinite(prior)):
                raise ValueError("prior covariance must be finite")
            covariance = 0.5 * (prior + prior.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            covariance = (
                eigenvectors * np.maximum(eigenvalues, 1e-12)
            ) @ eigenvectors.T
        else:
            raise ValueError("prior variance must be scalar, vector, or matrix")
        self.C = covariance
        self.sampled_set = []
        self.sol_to_idx = {}
        self._adaptive_sparsity = None
        self._adaptive_records = []
        self._adaptive_spec = None
        self._covariance_projection_count = 0
        self._min_quadratic_variance_seen = float("inf")
        self._max_abs_posterior_mean_seen = 0.0
        self._invalidate_backend_cache()

    def update(self, x: ArrayLike, y: float, sigma2_hat: float) -> None:
        """Rank-one Kalman update using the plug-in observation variance."""
        if self._adaptive_sparsity is not None:
            self._adaptive_records.append({
                "x": tuple(int(v) for v in np.asarray(x, dtype=int)),
                "y": float(y),
                "sigma2": max(float(sigma2_hat), 1e-12),
            })
            self._refit_adaptive_sparsity()
            return
        y = float(y)
        observation_variance = float(sigma2_hat)
        if not np.isfinite(y):
            raise ValueError("GPR observation must be finite")
        if not np.isfinite(observation_variance):
            raise ValueError("GPR observation variance must be finite")
        observation_variance = max(observation_variance, 1e-12)
        self.dimension_augment(x)
        e = self.augmented_feature(x)
        Ce = self.C @ e
        predictive_variance = float(e @ Ce)
        self._record_quadratic_variance(predictive_variance)
        if predictive_variance < -self._negative_variance_tolerance(
                [predictive_variance]):
            self._project_covariance_psd()
            Ce = self.C @ e
            predictive_variance = float(e @ Ce)
            self._record_quadratic_variance(predictive_variance)
        predictive_variance = max(predictive_variance, 0.0)
        denom = observation_variance + predictive_variance
        gain = Ce / denom
        innovation = y - float(e @ self.a)
        updated_mean = self.a + gain * innovation
        updated_covariance = self.C - np.outer(Ce, Ce) / denom
        updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
        if not np.all(np.isfinite(updated_mean)):
            raise FloatingPointError("non-finite GPR mean after rank-one update")
        if not np.all(np.isfinite(updated_covariance)):
            raise FloatingPointError("non-finite GPR covariance after rank-one update")
        self.a = updated_mean
        self.C = updated_covariance
        if float(np.min(np.diag(self.C))) < -self._negative_variance_tolerance(
                np.diag(self.C)):
            self._project_covariance_psd()
        self._invalidate_backend_cache()

    def numerical_diagnostics(self):
        covariance = 0.5 * (np.asarray(self.C) + np.asarray(self.C).T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        minimum_seen = float(getattr(
            self, "_min_quadratic_variance_seen", float("inf")))
        return {
            "finite_state": bool(
                np.all(np.isfinite(self.a)) and np.all(np.isfinite(self.C))
            ),
            "basis_dim": int(self.p),
            "state_dim": int(len(self.a)),
            "covariance_projection_count": int(getattr(
                self, "_covariance_projection_count", 0)),
            "covariance_min_eigenvalue": float(np.min(eigenvalues)),
            "covariance_max_eigenvalue": float(np.max(eigenvalues)),
            "minimum_quadratic_variance_seen": (
                None if not np.isfinite(minimum_seen) else minimum_seen
            ),
            "max_abs_coefficient": float(np.max(np.abs(self.a))),
            "max_abs_posterior_mean_seen": float(getattr(
                self, "_max_abs_posterior_mean_seen", 0.0)),
        }

    def enable_adaptive_sparsity(
        self,
        spec,
        X,
        y,
        noise_variance,
        *,
        deviation_variance,
    ):
        """Enable a target-updated sparse posterior over the fixed basis.

        Every subsequent ``update`` refits from ``_adaptive_records``.  Exact KG
        clones therefore update their own PIPs under each fantasy observation,
        while the live model remains unchanged.
        """

        if self.basis_map is None:
            raise ValueError("adaptive sparsity requires an explicit fixed basis")
        rows = list(X)
        values = np.asarray(y, dtype=float).reshape(-1)
        noise = np.asarray(noise_variance, dtype=float).reshape(-1)
        if len(noise) == 1 and len(rows) > 1:
            noise = np.full(len(rows), float(noise[0]), dtype=float)
        if len(rows) != len(values) or len(rows) != len(noise):
            raise ValueError("adaptive initial observations must align")
        self.lambda_i = max(float(deviation_variance), 1e-12)
        self._adaptive_spec = dict(spec)
        self._adaptive_sparsity = AdaptiveSpikeSlabPosterior(
            self._adaptive_spec["source_pip"],
            self._adaptive_spec["source_slab_scale"],
            min_pip=self._adaptive_spec.get("min_pip", 0.05),
            max_pip=self._adaptive_spec.get("max_pip", 0.95),
            spike_ratio=self._adaptive_spec.get("spike_ratio", 0.05),
            damping=self._adaptive_spec.get("damping", 0.5),
            max_iter=self._adaptive_spec.get("max_iter", 40),
            tolerance=self._adaptive_spec.get("tolerance", 1e-5),
            residual_floor_scale=self._adaptive_spec.get(
                "residual_floor_scale", 0.05),
            multiplicity_correction=self._adaptive_spec.get(
                "multiplicity_correction", 1.0),
            max_effective_fraction=self._adaptive_spec.get(
                "max_effective_fraction", 0.35),
            always_active_count=self._adaptive_spec.get(
                "always_active_count", 0),
            allowed_mask=self._adaptive_spec.get("allowed_mask"),
        )
        self._adaptive_records = [
            {
                "x": tuple(int(v) for v in np.asarray(x, dtype=int)),
                "y": float(value),
                "sigma2": max(float(sigma2), 1e-12),
            }
            for x, value, sigma2 in zip(rows, values, noise)
        ]
        self._refit_adaptive_sparsity()

    def _refit_adaptive_sparsity(self):
        if self._adaptive_sparsity is None or not self._adaptive_records:
            return
        X = [row["x"] for row in self._adaptive_records]
        dictionary_dim = min(
            int(self._adaptive_spec.get("dictionary_dim", self.p - 1)),
            self.p - 1,
        )
        features = self.basis_matrix(X)[:, 1:1 + dictionary_dim]
        response = np.asarray([row["y"] for row in self._adaptive_records])
        noise = np.asarray([row["sigma2"] for row in self._adaptive_records])
        self._adaptive_sparsity.fit(
            features,
            response,
            noise,
            X,
            deviation_variance=self.lambda_i,
        )
        result = self._adaptive_sparsity.result_
        self.sampled_set = list(result.sampled_set)
        self.sol_to_idx = dict(result.sol_to_idx)
        n_sampled = len(self.sampled_set)
        full_dim = self.p + n_sampled
        active_parametric_dim = 1 + dictionary_dim
        source_indices = list(range(active_parametric_dim)) + list(range(
            active_parametric_dim,
            active_parametric_dim + n_sampled,
        ))
        target_indices = list(range(active_parametric_dim)) + list(range(
            self.p,
            self.p + n_sampled,
        ))
        self.a = np.zeros(full_dim, dtype=float)
        self.a[target_indices] = np.asarray(result.mean, dtype=float)[source_indices]
        self.C = 1e-12 * np.eye(full_dim, dtype=float)
        self.C[np.ix_(target_indices, target_indices)] = np.asarray(
            result.covariance, dtype=float)[np.ix_(source_indices, source_indices)]
        basis_map = self.basis_map
        if hasattr(basis_map, "record_adaptive_sparsity_diagnostics"):
            basis_map.record_adaptive_sparsity_diagnostics(result.diagnostics)
        self._invalidate_backend_cache()

    def adaptive_sparsity_enabled(self):
        return self._adaptive_sparsity is not None

    def adaptive_model_uncertainty(self, x):
        if self._adaptive_sparsity is None:
            return 0.0
        dictionary_dim = min(
            int(self._adaptive_spec.get("dictionary_dim", self.p - 1)),
            self.p - 1,
        )
        features = self.basis_matrix([x])[:, 1:1 + dictionary_dim]
        return float(self._adaptive_sparsity.mask_uncertainty(features)[0])

    def adaptive_model_uncertainty_many(self, X):
        if self._adaptive_sparsity is None or len(X) == 0:
            return np.zeros(len(X), dtype=float)
        dictionary_dim = min(
            int(self._adaptive_spec.get("dictionary_dim", self.p - 1)),
            self.p - 1,
        )
        features = self.basis_matrix(X)[:, 1:1 + dictionary_dim]
        return np.asarray(
            self._adaptive_sparsity.mask_uncertainty(features), dtype=float)

    def adaptive_sparsity_diagnostics(self):
        if self._adaptive_sparsity is None:
            return {"status": "disabled"}
        return self._adaptive_sparsity.diagnostics()

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
