"""Parametric GPR belief model with cached/vectorized feature helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from representation.adaptive_sparsity import (
    AdaptiveGroupRidgePosterior,
    AdaptiveSpikeSlabPosterior,
)


ArrayLike = Union[np.ndarray, list, tuple]


def normalize_mixture_weights(weights):
    """Normalize nonnegative mixture mass without inventing support."""

    values = np.asarray(weights, dtype=float).reshape(-1)
    if not len(values):
        raise ValueError("mixture weights cannot be empty")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("mixture weights must be finite and nonnegative")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("mixture weights need positive mass")
    return values / total


def posterior_mixture_weights(prior_weights, log_evidence, temperature=1.0):
    """Bayes-update finite mixture mass while preserving unsupported atoms.

    Zero prior mass remains exactly zero. Tiny positive mass is handled in the
    log domain instead of being raised to an arbitrary numerical floor.
    """

    prior = normalize_mixture_weights(prior_weights)
    evidence = np.asarray(log_evidence, dtype=float).reshape(-1)
    if len(prior) != len(evidence):
        raise ValueError("mixture prior and log evidence must align")
    if not np.all(np.isfinite(evidence)):
        raise ValueError("mixture log evidence must be finite")
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("mixture evidence temperature must be positive")
    log_weight = np.full(len(prior), -np.inf, dtype=float)
    supported = prior > 0.0
    log_weight[supported] = (
        np.log(prior[supported]) + evidence[supported] / temperature
    )
    normalizer = float(np.max(log_weight))
    if not np.isfinite(normalizer):
        raise FloatingPointError("mixture posterior has no supported atom")
    posterior = np.exp(log_weight - normalizer)
    posterior /= float(np.sum(posterior))
    return prior, posterior


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
        self._finite_mixture_components = []
        self._finite_mixture_weights = None
        self._finite_mixture_component_names = []
        self._finite_mixture_sequential = False
        self._finite_mixture_update_count = 0
        self._finite_mixture_hierarchical_misspecification = False
        self._finite_mixture_cross_validated_structure = False
        self._finite_mixture_preserve_group_masses = False
        self._finite_mixture_group_labels = []
        self._finite_mixture_group_masses = {}
        self._finite_mixture_component_priors = []
        self._finite_mixture_prior_weights = None
        self._finite_mixture_target_history = []
        self._finite_mixture_misspecification_prior_df = 4.0
        self._finite_mixture_misspecification_max_scale = 100.0
        self._finite_mixture_misspecification_mode = (
            "hierarchical_predictive_scale")
        self._finite_mixture_misspecification_ridge = 1.0
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
        self._finite_mixture_components = []
        self._finite_mixture_weights = None
        self._finite_mixture_component_names = []
        self._finite_mixture_sequential = False
        self._finite_mixture_update_count = 0
        self._finite_mixture_hierarchical_misspecification = False
        self._finite_mixture_cross_validated_structure = False
        self._finite_mixture_preserve_group_masses = False
        self._finite_mixture_group_labels = []
        self._finite_mixture_group_masses = {}
        self._finite_mixture_component_priors = []
        self._finite_mixture_prior_weights = None
        self._finite_mixture_target_history = []
        self._finite_mixture_misspecification_prior_df = 4.0
        self._finite_mixture_misspecification_max_scale = 100.0
        self._finite_mixture_misspecification_mode = (
            "hierarchical_predictive_scale")
        self._finite_mixture_misspecification_ridge = 1.0
        self._covariance_projection_count = 0
        self._min_quadratic_variance_seen = float("inf")
        self._max_abs_posterior_mean_seen = 0.0
        self._invalidate_backend_cache()

    @staticmethod
    def _gaussian_log_density(residual, covariance):
        """Stable multivariate Gaussian log density at ``residual``."""

        residual = np.asarray(residual, dtype=float).reshape(-1)
        covariance = np.asarray(covariance, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        jitter = max(
            1e-12,
            1e-10 * float(np.trace(covariance)) / max(len(residual), 1),
        )
        identity = np.eye(len(residual), dtype=float)
        for _ in range(8):
            try:
                chol = np.linalg.cholesky(covariance + jitter * identity)
                solved = np.linalg.solve(chol, residual)
                log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
                return float(-0.5 * (
                    solved @ solved
                    + log_det
                    + len(residual) * np.log(2.0 * np.pi)
                ))
            except np.linalg.LinAlgError:
                jitter *= 10.0
        raise np.linalg.LinAlgError(
            "finite-mixture predictive covariance is not positive definite"
        )

    @staticmethod
    def _mahalanobis_square(residual, covariance):
        """Return a stable nonnegative Mahalanobis square."""

        residual = np.asarray(residual, dtype=float).reshape(-1)
        covariance = np.asarray(covariance, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        jitter = max(
            1e-12,
            1e-10 * float(np.trace(covariance)) / max(len(residual), 1),
        )
        identity = np.eye(len(residual), dtype=float)
        for _ in range(8):
            try:
                chol = np.linalg.cholesky(covariance + jitter * identity)
                solved = np.linalg.solve(
                    chol.T, np.linalg.solve(chol, residual))
                return max(float(residual @ solved), 0.0)
            except np.linalg.LinAlgError:
                jitter *= 10.0
        return max(float(residual @ np.linalg.pinv(covariance) @ residual), 0.0)

    @staticmethod
    def _gaussian_loo_log_score(residual, covariance):
        """Exact leave-one-out Gaussian predictive log score.

        If ``Q = covariance^-1`` and ``alpha = Q residual``, the conditional
        predictive variance of observation ``i`` given all other observations
        is ``1 / Q_ii`` and its conditional residual is
        ``alpha_i / Q_ii``. The score therefore evaluates every charged target
        response out of fold without repeatedly fitting ``n`` models.
        """

        residual = np.asarray(residual, dtype=float).reshape(-1)
        covariance = np.asarray(covariance, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        if covariance.shape != (len(residual), len(residual)):
            raise ValueError("LOO covariance must align with residuals")
        if len(residual) == 0:
            return 0.0, {
                "loo_count": 0,
                "loo_mean_log_score": 0.0,
                "loo_median_abs_standardized_residual": 0.0,
            }
        jitter = max(
            1e-12,
            1e-10 * float(np.trace(covariance)) / max(len(residual), 1),
        )
        identity = np.eye(len(residual), dtype=float)
        precision = None
        for _ in range(8):
            try:
                chol = np.linalg.cholesky(covariance + jitter * identity)
                precision = np.linalg.solve(
                    chol.T, np.linalg.solve(chol, identity))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        if precision is None:
            precision = np.linalg.pinv(covariance)
        precision = 0.5 * (precision + precision.T)
        diagonal = np.maximum(np.diag(precision), 1e-12)
        alpha = precision @ residual
        conditional_variance = 1.0 / diagonal
        conditional_residual = alpha / diagonal
        standardized = conditional_residual / np.sqrt(
            conditional_variance)
        terms = -0.5 * (
            np.log(2.0 * np.pi * conditional_variance)
            + standardized ** 2
        )
        if not np.all(np.isfinite(terms)):
            raise FloatingPointError("LOO predictive score is non-finite")
        return float(np.sum(terms)), {
            "loo_count": int(len(residual)),
            "loo_mean_log_score": float(np.mean(terms)),
            "loo_median_abs_standardized_residual": float(np.median(
                np.abs(standardized))),
            "loo_minimum_conditional_variance": float(np.min(
                conditional_variance)),
            "loo_maximum_conditional_variance": float(np.max(
                conditional_variance)),
        }

    @staticmethod
    def _residual_rank_mixture_diagnostics(names, weights):
        marker = "|target_residual_rank="
        mass = {}
        source_mass = {}
        null_mass = {}
        for name, weight in zip(names, np.asarray(weights, dtype=float)):
            name = str(name)
            if marker not in name:
                continue
            suffix = name.rsplit(marker, 1)[1]
            try:
                rank = int(suffix.split("|", 1)[0])
            except ValueError:
                continue
            value = max(float(weight), 0.0)
            mass[rank] = mass.get(rank, 0.0) + value
            destination = (
                null_mass if name.startswith("target:null") else source_mass)
            destination[rank] = destination.get(rank, 0.0) + value
        if not mass:
            return {
                "target_residual_rank_posterior_active": False,
                "target_residual_rank_posterior_mass": {},
            }
        structured_total = float(sum(mass.values()))
        source_total = float(sum(source_mass.values()))
        null_total = float(sum(null_mass.values()))
        conditional = {
            str(rank): float(value / max(structured_total, 1e-300))
            for rank, value in sorted(mass.items())
        }
        absolute = {
            str(rank): float(value)
            for rank, value in sorted(mass.items())
        }
        return {
            "target_residual_rank_posterior_active": True,
            "target_residual_rank_posterior_mass": absolute,
            "target_residual_rank_conditional_mass": conditional,
            "target_residual_rank_conditional_source_mass": {
                str(rank): float(value / max(source_total, 1e-300))
                for rank, value in sorted(source_mass.items())
            },
            "target_residual_rank_conditional_null_mass": {
                str(rank): float(value / max(null_total, 1e-300))
                for rank, value in sorted(null_mass.items())
            },
            "target_residual_rank_structured_mass": structured_total,
            "target_residual_rank_structured_source_mass": source_total,
            "target_residual_rank_structured_null_mass": null_total,
            "target_residual_rank_selected": int(max(
                mass, key=lambda rank: mass[rank])),
            "target_residual_rank_target_labels_used_for_update": True,
            "target_residual_rank_target_oracle_used": False,
        }

    @staticmethod
    def _role_assignment_mixture_diagnostics(names, weights):
        marker = "|role_assignment="
        mass = {}
        source_mass = {}
        null_mass = {}
        for name, weight in zip(names, np.asarray(weights, dtype=float)):
            name = str(name)
            if marker not in name:
                continue
            assignment = name.rsplit(marker, 1)[1].split("|", 1)[0]
            value = max(float(weight), 0.0)
            mass[assignment] = mass.get(assignment, 0.0) + value
            destination = (
                null_mass if name.startswith("target:null") else source_mass)
            destination[assignment] = (
                destination.get(assignment, 0.0) + value)
        if not mass:
            return {
                "target_role_assignment_posterior_active": False,
                "target_role_assignment_posterior_mass": {},
            }
        total = float(sum(mass.values()))
        source_total = float(sum(source_mass.values()))
        null_total = float(sum(null_mass.values()))
        return {
            "target_role_assignment_posterior_active": True,
            "target_role_assignment_posterior_mass": {
                key: float(value)
                for key, value in sorted(mass.items())
            },
            "target_role_assignment_conditional_mass": {
                key: float(value / max(total, 1e-300))
                for key, value in sorted(mass.items())
            },
            "target_role_assignment_conditional_source_mass": {
                key: float(value / max(source_total, 1e-300))
                for key, value in sorted(source_mass.items())
            },
            "target_role_assignment_conditional_null_mass": {
                key: float(value / max(null_total, 1e-300))
                for key, value in sorted(null_mass.items())
            },
            "target_role_assignment_structured_mass": total,
            "target_role_assignment_structured_source_mass": source_total,
            "target_role_assignment_structured_null_mass": null_total,
            "target_role_assignment_selected": str(max(
                mass, key=lambda assignment: mass[assignment])),
            "target_role_assignment_target_labels_used_for_update": True,
            "target_role_assignment_target_oracle_used": False,
            "target_role_assignment_permutation_equivariant": True,
        }

    def set_hierarchical_misspecification_posterior(
        self,
        component_models,
        component_priors,
        prior_weights,
        samples,
        targets,
        observation_variances,
        diagnostics=None,
        *,
        prior_df=4.0,
        max_scale=100.0,
        misspecification_mode="hierarchical_predictive_scale",
        misspecification_ridge=1.0,
        group_labels=None,
        group_masses=None,
    ) -> None:
        """Fit an online source-mixture posterior with a latent scale law.

        The unconditioned source laws are retained.  Every charged target
        observation updates the sufficient statistic of each source scale,
        after which all components are refit from their original laws.  This
        avoids repeatedly multiplying an already-inflated covariance and also
        prevents ordinary posterior contraction from silently erasing the
        learned misspecification guard.
        """

        components = list(component_models)
        priors = [dict(value) for value in component_priors]
        weight = np.asarray(prior_weights, dtype=float).reshape(-1)
        rows = [tuple(int(v) for v in np.asarray(x, dtype=int)) for x in samples]
        values = np.asarray(targets, dtype=float).reshape(-1)
        noise = np.asarray(observation_variances, dtype=float).reshape(-1)
        if len(noise) == 1 and len(rows) > 1:
            noise = np.full(len(rows), float(noise[0]), dtype=float)
        if (
            not components
            or len(components) != len(priors)
            or len(components) != len(weight)
        ):
            raise ValueError("hierarchical mixture components must align")
        if len(rows) != len(values) or len(rows) != len(noise):
            raise ValueError("hierarchical target observations must align")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(noise)):
            raise ValueError("hierarchical target observations must be finite")
        if np.any(weight < 0.0) or not np.all(np.isfinite(weight)):
            raise ValueError("hierarchical mixture weights must be nonnegative")
        if float(np.sum(weight)) <= 0.0:
            raise ValueError("hierarchical mixture weights need positive mass")

        labels = (
            [str(value) for value in group_labels]
            if group_labels is not None else [])
        preserve_group_masses = bool(labels)
        if preserve_group_masses:
            if len(labels) != len(weight) or group_masses is None:
                raise ValueError(
                    "hierarchical group labels and masses must align")
            weight, fixed_masses = self.group_mass_preserving_weights(
                weight,
                np.zeros(len(weight), dtype=float),
                labels,
                group_masses,
                temperature=1.0,
            )
        else:
            fixed_masses = {}

        self._finite_mixture_components = components
        self._finite_mixture_component_priors = priors
        self._finite_mixture_prior_weights = normalize_mixture_weights(weight)
        self._finite_mixture_component_names = [
            str(prior.get("name", f"component:{index}"))
            for index, prior in enumerate(priors)
        ]
        self._finite_mixture_target_history = [
            {
                "x": row,
                "y": float(value),
                "observation_variance": max(float(sigma2), 1e-12),
            }
            for row, value, sigma2 in zip(rows, values, noise)
        ]
        self._finite_mixture_hierarchical_misspecification = True
        self._finite_mixture_preserve_group_masses = preserve_group_masses
        self._finite_mixture_group_labels = labels
        self._finite_mixture_group_masses = dict(fixed_masses)
        self._finite_mixture_sequential = True
        self._finite_mixture_update_count = int(
            (diagnostics or {}).get("online_mixture_update_count", 0))
        self._finite_mixture_misspecification_prior_df = max(
            float(prior_df), 1e-8)
        self._finite_mixture_misspecification_max_scale = max(
            float(max_scale), 1.0)
        mode = str(
            misspecification_mode or "hierarchical_predictive_scale"
        ).strip().lower()
        if mode not in {
            "none",
            "predictive_scale",
            "predictive_scale_directional",
            "hierarchical_predictive_scale",
        }:
            raise ValueError(
                "sequential misspecification mode must be none, "
                "predictive_scale, predictive_scale_directional, or "
                "hierarchical_predictive_scale")
        self._finite_mixture_misspecification_mode = mode
        self._finite_mixture_misspecification_ridge = max(
            float(misspecification_ridge), 1e-10)
        self.source_parametric_prior_diagnostics = dict(diagnostics or {})
        if preserve_group_masses:
            self.source_parametric_prior_diagnostics.update({
                "assignment_group_masses_fixed": True,
                "assignment_group_labels": labels,
                "assignment_group_masses": dict(fixed_masses),
                "target_oracle_used_for_group_masses": False,
            })
        self._refit_hierarchical_finite_mixture()

    def _refit_hierarchical_finite_mixture(self) -> None:
        """Refit every component and its source scale from charged data."""

        components = list(self._finite_mixture_components)
        priors = list(self._finite_mixture_component_priors)
        prior_weight = np.asarray(
            self._finite_mixture_prior_weights, dtype=float).reshape(-1)
        history = list(self._finite_mixture_target_history)
        if (
            not components
            or len(components) != len(priors)
            or len(components) != len(prior_weight)
        ):
            raise RuntimeError("hierarchical finite-mixture state is incomplete")

        rows = [entry["x"] for entry in history]
        target = np.asarray([entry["y"] for entry in history], dtype=float)
        observation_variance = np.asarray([
            entry["observation_variance"] for entry in history
        ], dtype=float)
        prior_df = max(
            float(self._finite_mixture_misspecification_prior_df), 1e-8)
        max_scale = max(
            float(self._finite_mixture_misspecification_max_scale), 1.0)
        misspecification_mode = str(getattr(
            self,
            "_finite_mixture_misspecification_mode",
            "hierarchical_predictive_scale",
        )).strip().lower()
        misspecification_ridge = max(float(getattr(
            self, "_finite_mixture_misspecification_ridge", 1.0)), 1e-10)
        diagnostics = dict(getattr(
            self, "source_parametric_prior_diagnostics", {}) or {})
        temperature = max(
            float(diagnostics.get("evidence_temperature", 1.0)), 1e-6)
        component_diagnostics = []
        log_evidence = []

        for index, (component, prior) in enumerate(zip(components, priors)):
            mean = np.asarray(prior["mean"], dtype=float).reshape(-1)
            covariance = np.asarray(prior["covariance"], dtype=float)
            covariance = 0.5 * (covariance + covariance.T)
            deviation = max(
                float(prior.get("deviation_variance", 1e-6)), 1e-12)
            name = str(prior.get("name", f"component:{index}"))
            is_source = not name.startswith("target:")
            if rows:
                phi = np.asarray(component.basis_matrix(rows), dtype=float)
                residual = target - phi @ mean
                base_predictive = phi @ covariance @ phi.T
                base_predictive = 0.5 * (
                    base_predictive + base_predictive.T)
                base_predictive += np.diag(
                    deviation + observation_variance)
                mahalanobis = self._mahalanobis_square(
                    residual, base_predictive)
            else:
                phi = np.zeros((0, len(mean)), dtype=float)
                residual = np.zeros(0, dtype=float)
                mahalanobis = 0.0
            scale = 1.0
            if (
                is_source
                and rows
                and misspecification_mode != "none"
            ):
                scale = float(np.clip(
                    (prior_df + mahalanobis) / (prior_df + len(rows)),
                    1.0,
                    max_scale,
                ))
            scaled_covariance = scale * covariance
            scaled_deviation = scale * deviation
            directional_mass = 0.0
            directional_energy = 0.0
            if (
                is_source
                and rows
                and misspecification_mode == "predictive_scale_directional"
            ):
                gram = (
                    phi.T @ phi
                    + misspecification_ridge
                    * np.eye(phi.shape[1], dtype=float)
                )
                raw_direction = np.linalg.solve(
                    gram, phi.T @ residual)
                direction_norm = float(np.linalg.norm(raw_direction))
                if direction_norm > 1e-12:
                    direction = raw_direction / direction_norm
                    directional_energy = float(np.mean(
                        (phi @ direction) ** 2))
                    reference_variance = float(np.mean(
                        np.diag(base_predictive)))
                    empirical_error = float(np.mean(residual ** 2))
                    excess = max(
                        empirical_error - reference_variance, 0.0)
                    directional_mass = min(
                        excess,
                        max(max_scale - 1.0, 0.0)
                        * max(reference_variance, 1e-12),
                    )
                    if (
                        directional_energy > 1e-12
                        and directional_mass > 0.0
                    ):
                        scaled_covariance += (
                            directional_mass / directional_energy
                        ) * np.outer(direction, direction)
            scaled_covariance = 0.5 * (
                scaled_covariance + scaled_covariance.T)
            eigenvalues, eigenvectors = np.linalg.eigh(scaled_covariance)
            scaled_covariance = (
                eigenvectors * np.maximum(eigenvalues, 1e-12)
            ) @ eigenvectors.T
            if rows:
                predictive = phi @ scaled_covariance @ phi.T
                predictive = 0.5 * (predictive + predictive.T)
                predictive += np.diag(
                    scaled_deviation + observation_variance)
                evidence = self._gaussian_log_density(residual, predictive)
            else:
                evidence = 0.0
            log_evidence.append(float(evidence))

            component.set_parametric_prior(
                mean, scaled_deviation, scaled_covariance)
            for entry in history:
                component.update(
                    entry["x"],
                    entry["y"],
                    entry["observation_variance"],
                )
            base_diagnostics = dict(prior.get("diagnostics", {}))
            base_diagnostics.update({
                "name": name,
                "source_mean_misspecification_mode": (
                    misspecification_mode if is_source else "none"
                ),
                "source_mean_misspecification_applied": bool(
                    is_source and misspecification_mode != "none"),
                "source_mean_misspecification_scale": float(scale),
                "source_mean_misspecification_mahalanobis": float(mahalanobis),
                "source_mean_misspecification_prior_df": float(prior_df),
                "source_mean_misspecification_directional_mass": float(
                    directional_mass),
                "source_mean_misspecification_directional_energy": float(
                    directional_energy),
                "source_mean_misspecification_ridge": float(
                    misspecification_ridge),
                "source_mean_misspecification_target_count": int(len(rows)),
                "source_mean_prior_covariance_trace_before": float(
                    np.trace(covariance)),
                "source_mean_prior_covariance_trace_after": float(
                    np.trace(scaled_covariance)),
                "source_mean_residual_floor_before": float(deviation),
                "source_mean_residual_floor_after": float(scaled_deviation),
                "misspecification_uncertainty_can_only_increase": bool(
                    is_source),
                "target_oracle_used_for_misspecification": False,
            })
            component_diagnostics.append(base_diagnostics)

        log_evidence = np.asarray(log_evidence, dtype=float)
        if bool(getattr(
            self, "_finite_mixture_preserve_group_masses", False
        )):
            posterior_weight, fixed_group_masses = (
                self.group_mass_preserving_weights(
                    prior_weight,
                    log_evidence,
                    self._finite_mixture_group_labels,
                    self._finite_mixture_group_masses,
                    temperature,
                )
            )
        else:
            prior_weight, posterior_weight = posterior_mixture_weights(
                prior_weight, log_evidence, temperature)
            fixed_group_masses = None
        names = list(self._finite_mixture_component_names)
        trajectory = list(diagnostics.get(
            "source_mean_misspecification_scale_trajectory", []))
        trajectory.append({
            "target_observation_count": int(len(rows)),
            "online_mixture_update_count": int(
                self._finite_mixture_update_count),
            "component_scales": {
                str(item["name"]): float(
                    item["source_mean_misspecification_scale"])
                for item in component_diagnostics
            },
        })
        single_aggregate = bool(
            diagnostics.get("single_aggregate_hyperlaw", False)
            and len(components) == 1
            and names == ["source:aggregate"]
        )
        diagnostics.update({
            "adaptation_mode": (
                "sequential_single_aggregate_hyperlaw"
                if single_aggregate
                else "sequential_target_evidence_mixture"
            ),
            "component_names": names,
            "component_prior_weights": prior_weight.tolist(),
            "component_log_evidence": log_evidence.tolist(),
            "component_posterior_weights": posterior_weight.tolist(),
            "selected_component": str(names[int(np.argmax(posterior_weight))]),
            "target_only_posterior_weight": float(sum(
                mass for name, mass in zip(names, posterior_weight)
                if str(name).startswith("target:null")
            )),
            "source_posterior_weight": float(sum(
                mass for name, mass in zip(names, posterior_weight)
                if not str(name).startswith("target:null")
            )),
            "target_observation_count": int(len(rows)),
            "online_mixture_update_count": int(
                self._finite_mixture_update_count),
            "posterior_target_data_used": bool(rows),
            "target_oracle_used": False,
            "source_mean_misspecification_mode": (
                misspecification_mode),
            "source_mean_misspecification_online": bool(
                misspecification_mode != "none"),
            "source_mean_misspecification_refit_from_frozen_law": True,
            "source_mean_misspecification_scale_trajectory": trajectory,
            "component_deviation_diagnostics": component_diagnostics,
        })
        if single_aggregate:
            aggregate_diagnostics = component_diagnostics[0]
            diagnostics.update({
                "single_aggregate_hyperlaw": True,
                "single_aggregate_component_count": 1,
                "source_domain_identity_marginalized": True,
                "source_components_retained_in_target_posterior": False,
                "target_null_component_retained": False,
                "source_mean_misspecification_applied": bool(
                    aggregate_diagnostics[
                        "source_mean_misspecification_applied"]),
                "source_mean_misspecification_scale": float(
                    aggregate_diagnostics[
                        "source_mean_misspecification_scale"]),
                "source_mean_prior_covariance_trace_before": float(
                    aggregate_diagnostics[
                        "source_mean_prior_covariance_trace_before"]),
                "source_mean_prior_covariance_trace_after": float(
                    aggregate_diagnostics[
                        "source_mean_prior_covariance_trace_after"]),
                "source_mean_residual_floor_before": float(
                    aggregate_diagnostics[
                        "source_mean_residual_floor_before"]),
                "source_mean_residual_floor_after": float(
                    aggregate_diagnostics[
                        "source_mean_residual_floor_after"]),
                "source_mean_misspecification_directional_mass": float(
                    aggregate_diagnostics[
                        "source_mean_misspecification_directional_mass"]),
                "misspecification_uncertainty_can_only_increase": True,
                "target_oracle_used_for_misspecification": False,
            })
        if fixed_group_masses is not None:
            diagnostics.update({
                "adaptation_mode": (
                    "sequential_assignment_prior_conditional_hierarchical_"
                    "expert_mixture"),
                "assignment_group_masses_fixed": True,
                "assignment_group_masses": dict(fixed_group_masses),
                "target_oracle_used_for_group_masses": False,
                "target_role_assignment_conditional_expert_uses_target_labels": (
                    bool(rows)),
            })
        self.set_moment_matched_posterior(
            components,
            posterior_weight,
            diagnostics=diagnostics,
            sequential_updates=True,
        )

    def set_cross_validated_structure_posterior(
        self,
        component_models,
        component_priors,
        prior_weights,
        samples,
        targets,
        observation_variances,
        diagnostics=None,
    ) -> None:
        """Fit a finite structure posterior from exact LOO predictions.

        The component laws and structure prior remain frozen. Every charged
        target response is predicted from all other charged responses, after
        which all component posteriors are refit on the complete history for
        downstream prediction. Online updates repeat this same operation from
        the frozen laws rather than multiplying incremental pseudo-evidence.
        """

        components = list(component_models)
        priors = [dict(value) for value in component_priors]
        weight = np.asarray(prior_weights, dtype=float).reshape(-1)
        rows = [tuple(int(v) for v in np.asarray(x, dtype=int)) for x in samples]
        values = np.asarray(targets, dtype=float).reshape(-1)
        noise = np.asarray(observation_variances, dtype=float).reshape(-1)
        if len(noise) == 1 and len(rows) > 1:
            noise = np.full(len(rows), float(noise[0]), dtype=float)
        if (
            not components
            or len(components) != len(priors)
            or len(components) != len(weight)
        ):
            raise ValueError("cross-validated mixture components must align")
        if len(rows) != len(values) or len(rows) != len(noise):
            raise ValueError("cross-validated target observations must align")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(noise)):
            raise ValueError(
                "cross-validated target observations must be finite")
        if np.any(weight < 0.0) or not np.all(np.isfinite(weight)):
            raise ValueError(
                "cross-validated mixture weights must be nonnegative")
        if float(np.sum(weight)) <= 0.0:
            raise ValueError(
                "cross-validated mixture weights need positive mass")

        self._finite_mixture_components = components
        self._finite_mixture_component_priors = priors
        self._finite_mixture_prior_weights = normalize_mixture_weights(weight)
        self._finite_mixture_component_names = [
            str(prior.get("name", f"component:{index}"))
            for index, prior in enumerate(priors)
        ]
        self._finite_mixture_target_history = [
            {
                "x": row,
                "y": float(value),
                "observation_variance": max(float(sigma2), 1e-12),
            }
            for row, value, sigma2 in zip(rows, values, noise)
        ]
        self._finite_mixture_cross_validated_structure = True
        self._finite_mixture_hierarchical_misspecification = False
        self._finite_mixture_sequential = True
        self._finite_mixture_update_count = int(
            (diagnostics or {}).get("online_mixture_update_count", 0))
        self.source_parametric_prior_diagnostics = dict(diagnostics or {})
        self._refit_cross_validated_finite_mixture()

    def _refit_cross_validated_finite_mixture(self) -> None:
        """Recompute LOO structure evidence from the frozen component laws."""

        components = list(self._finite_mixture_components)
        priors = list(self._finite_mixture_component_priors)
        prior_weight = np.asarray(
            self._finite_mixture_prior_weights, dtype=float).reshape(-1)
        history = list(self._finite_mixture_target_history)
        if (
            not components
            or len(components) != len(priors)
            or len(components) != len(prior_weight)
        ):
            raise RuntimeError(
                "cross-validated finite-mixture state is incomplete")

        rows = [entry["x"] for entry in history]
        target = np.asarray([entry["y"] for entry in history], dtype=float)
        observation_variance = np.asarray([
            entry["observation_variance"] for entry in history
        ], dtype=float)
        diagnostics = dict(getattr(
            self, "source_parametric_prior_diagnostics", {}) or {})
        temperature = max(
            float(diagnostics.get("evidence_temperature", 1.0)), 1e-6)
        log_score = []
        score_diagnostics = []

        for index, (component, prior) in enumerate(zip(components, priors)):
            mean = np.asarray(prior["mean"], dtype=float).reshape(-1)
            covariance = np.asarray(prior["covariance"], dtype=float)
            covariance = 0.5 * (covariance + covariance.T)
            deviation = max(
                float(prior.get("deviation_variance", 1e-6)), 1e-12)
            if rows:
                phi = np.asarray(component.basis_matrix(rows), dtype=float)
                residual = target - phi @ mean
                predictive = phi @ covariance @ phi.T
                predictive = 0.5 * (predictive + predictive.T)
                predictive += np.diag(deviation + observation_variance)
                score, score_info = self._gaussian_loo_log_score(
                    residual, predictive)
            else:
                score = 0.0
                score_info = {
                    "loo_count": 0,
                    "loo_mean_log_score": 0.0,
                    "loo_median_abs_standardized_residual": 0.0,
                }
            log_score.append(float(score))
            score_diagnostics.append({
                "name": str(prior.get("name", f"component:{index}")),
                **score_info,
                "target_oracle_used_for_structure_score": False,
            })

            component.set_parametric_prior(
                mean, deviation, covariance)
            for entry in history:
                component.update(
                    entry["x"],
                    entry["y"],
                    entry["observation_variance"],
                )

        log_score = np.asarray(log_score, dtype=float)
        prior_weight, posterior_weight = posterior_mixture_weights(
            prior_weight, log_score, temperature)
        names = list(self._finite_mixture_component_names)
        diagnostics.update({
            "adaptation_mode": (
                "sequential_cross_validated_target_evidence_mixture"),
            "structure_score_mode": "loo_predictive",
            "structure_score_cross_fitted": True,
            "component_names": names,
            "component_prior_weights": prior_weight.tolist(),
            "component_log_evidence": log_score.tolist(),
            "component_loo_predictive_diagnostics": score_diagnostics,
            "component_posterior_weights": posterior_weight.tolist(),
            "selected_component": str(names[int(
                np.argmax(posterior_weight))]),
            "target_only_posterior_weight": float(sum(
                mass for name, mass in zip(names, posterior_weight)
                if str(name).startswith("target:null")
            )),
            "source_posterior_weight": float(sum(
                mass for name, mass in zip(names, posterior_weight)
                if not str(name).startswith("target:null")
            )),
            "target_observation_count": int(len(rows)),
            "online_mixture_update_count": int(
                self._finite_mixture_update_count),
            "posterior_target_data_used": bool(rows),
            "target_oracle_used": False,
            "target_oracle_used_for_structure_score": False,
        })
        self.set_moment_matched_posterior(
            components,
            posterior_weight,
            diagnostics=diagnostics,
            sequential_updates=True,
        )

    def set_moment_matched_posterior(
        self,
        component_models,
        weights,
        diagnostics=None,
        *,
        sequential_updates=False,
    ) -> None:
        """Project a finite Gaussian-mixture posterior to two moments.

        Every component must have been conditioned on the same observations.
        The covariance retains both within-component uncertainty and
        between-component disagreement, so source-model ambiguity can only
        increase uncertainty rather than disappear during projection.
        """

        components = list(component_models)
        weight = np.asarray(weights, dtype=float).reshape(-1)
        if not components or len(components) != len(weight):
            raise ValueError("mixture components and weights must align")
        if not np.all(np.isfinite(weight)) or np.any(weight < 0.0):
            raise ValueError("mixture weights must be finite and nonnegative")
        total = float(np.sum(weight))
        if total <= 0.0:
            raise ValueError("mixture weights must have positive mass")
        weight = weight / total

        reference = components[0]
        reference_samples = list(reference.sampled_set)
        state_dim = len(reference.a)
        for component in components:
            if component.p != self.p or len(component.a) != state_dim:
                raise ValueError("mixture component GPR dimensions disagree")
            if list(component.sampled_set) != reference_samples:
                raise ValueError(
                    "mixture components must condition on identical samples"
                )
            if not np.all(np.isfinite(component.a)) or not np.all(
                np.isfinite(component.C)
            ):
                raise FloatingPointError("mixture component state is non-finite")

        mean = np.sum([
            mass * np.asarray(component.a, dtype=float)
            for mass, component in zip(weight, components)
        ], axis=0)
        covariance = np.zeros((state_dim, state_dim), dtype=float)
        for mass, component in zip(weight, components):
            delta = np.asarray(component.a, dtype=float) - mean
            covariance += mass * (
                np.asarray(component.C, dtype=float) + np.outer(delta, delta)
            )
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (
            eigenvectors * np.maximum(eigenvalues, 1e-12)
        ) @ eigenvectors.T

        self.a = np.asarray(mean, dtype=float)
        self.C = np.asarray(covariance, dtype=float)
        self.lambda_i = max(float(np.sum([
            mass * float(component.lambda_i)
            for mass, component in zip(weight, components)
        ])), 1e-12)
        self.sampled_set = reference_samples
        self.sol_to_idx = {
            sample: index for index, sample in enumerate(self.sampled_set)
        }
        self._adaptive_sparsity = None
        self._adaptive_records = []
        self._adaptive_spec = None
        self._finite_mixture_components = (
            components if bool(sequential_updates) else [])
        self._finite_mixture_weights = (
            np.asarray(weight, dtype=float).copy()
            if bool(sequential_updates) else None
        )
        names = list((diagnostics or {}).get("component_names", []))
        self._finite_mixture_component_names = (
            [str(value) for value in names]
            if len(names) == len(components)
            else [f"component:{index}" for index in range(len(components))]
        ) if bool(sequential_updates) else []
        self._finite_mixture_sequential = bool(sequential_updates)
        self._finite_mixture_update_count = int(
            (diagnostics or {}).get("online_mixture_update_count", 0)
        ) if bool(sequential_updates) else 0
        self._covariance_projection_count = 0
        self._min_quadratic_variance_seen = float("inf")
        self._max_abs_posterior_mean_seen = 0.0
        self.source_parametric_prior_diagnostics = dict(diagnostics or {})
        single_aggregate = bool(
            (diagnostics or {}).get("single_aggregate_hyperlaw", False)
            and len(components) == 1
        )
        self.source_parametric_prior_diagnostics.update({
            "posterior_projection": (
                "single_gaussian_identity_projection"
                if single_aggregate
                else "finite_mixture_moment_match"
            ),
            "posterior_component_count": int(len(components)),
            "posterior_weight_sum": float(np.sum(weight)),
            "posterior_effective_component_count": float(
                1.0 / np.sum(weight ** 2)),
            "between_component_covariance_trace": float(
                np.trace(covariance)
                - np.sum([
                    mass * np.trace(component.C)
                    for mass, component in zip(weight, components)
                ])
            ),
        })
        diagnostic_names = (
            self._finite_mixture_component_names
            if self._finite_mixture_component_names else [
                str(value) for value in names
            ]
        )
        self.source_parametric_prior_diagnostics.update(
            self._residual_rank_mixture_diagnostics(
                diagnostic_names, weight))
        role_diagnostics = self._role_assignment_mixture_diagnostics(
            diagnostic_names, weight)
        if bool((diagnostics or {}).get(
            "assignment_group_masses_fixed", False
        )):
            role_diagnostics.update({
                "target_role_assignment_target_labels_used_for_update": False,
                "target_role_assignment_target_labels_used_for_prior": bool(
                    (diagnostics or {}).get(
                        "target_labels_used_for_group_masses", False)),
                "target_role_assignment_target_labels_used_for_online_update": (
                    False),
                "target_role_assignment_conditional_expert_uses_target_labels": (
                    bool((diagnostics or {}).get(
                        "posterior_target_data_used", False))
                ),
                "target_role_assignment_update_scope": (
                    "charged_pilot_assignment_prior_then_frozen_"
                    "conditional_expert_only"
                    if bool((diagnostics or {}).get(
                        "target_labels_used_for_group_masses", False))
                    else "frozen_assignment_marginal_conditional_expert_only"),
            })
        self.source_parametric_prior_diagnostics.update(role_diagnostics)
        self._invalidate_backend_cache()

    @staticmethod
    def group_mass_preserving_weights(
        weights,
        log_evidence,
        group_labels,
        group_masses,
        temperature=1.0,
    ):
        """Update component conditionals while keeping group masses fixed."""

        weight = np.asarray(weights, dtype=float).reshape(-1)
        evidence = np.asarray(log_evidence, dtype=float).reshape(-1)
        labels = [str(value) for value in group_labels]
        if len(weight) == 0 or len(weight) != len(evidence):
            raise ValueError("grouped mixture weights and evidence must align")
        if len(labels) != len(weight):
            raise ValueError("group labels must align with mixture components")
        if np.any(weight < 0.0) or not np.all(np.isfinite(weight)):
            raise ValueError("grouped mixture weights must be nonnegative")
        if not np.all(np.isfinite(evidence)):
            raise ValueError("grouped mixture evidence must be finite")
        masses = {
            str(key): max(float(value), 0.0)
            for key, value in dict(group_masses).items()
        }
        groups = sorted(set(labels))
        if set(masses) != set(groups):
            raise ValueError("fixed group masses must cover every group")
        total_mass = float(sum(masses.values()))
        if total_mass <= 0.0:
            raise ValueError("fixed group masses need positive total mass")
        masses = {key: value / total_mass for key, value in masses.items()}
        posterior = np.zeros(len(weight), dtype=float)
        for group in groups:
            indices = np.asarray([
                index for index, label in enumerate(labels) if label == group
            ], dtype=int)
            conditional = weight[indices]
            if float(np.sum(conditional)) <= 0.0:
                conditional = np.ones(len(indices), dtype=float)
            conditional /= float(np.sum(conditional))
            _, updated = posterior_mixture_weights(
                conditional,
                evidence[indices],
                max(float(temperature), 1e-6),
            )
            posterior[indices] = float(masses[group]) * updated
        posterior /= float(np.sum(posterior))
        return posterior, masses

    def set_group_mass_preserving_posterior(
        self,
        component_models,
        weights,
        group_labels,
        group_masses,
        diagnostics=None,
    ) -> None:
        """Install a hierarchical mixture with frozen top-level masses."""

        components = list(component_models)
        labels = [str(value) for value in group_labels]
        weight = np.asarray(weights, dtype=float).reshape(-1)
        if len(components) != len(weight) or len(labels) != len(weight):
            raise ValueError("grouped posterior components must align")
        zero_evidence = np.zeros(len(weight), dtype=float)
        normalized, masses = self.group_mass_preserving_weights(
            weight,
            zero_evidence,
            labels,
            group_masses,
            temperature=1.0,
        )
        payload = dict(diagnostics or {})
        payload.update({
            "assignment_group_masses_fixed": True,
            "assignment_group_labels": labels,
            "assignment_group_masses": dict(masses),
            "target_oracle_used_for_group_masses": False,
        })
        self.set_moment_matched_posterior(
            components,
            normalized,
            diagnostics=payload,
            sequential_updates=True,
        )
        self._finite_mixture_preserve_group_masses = True
        self._finite_mixture_group_labels = labels
        self._finite_mixture_group_masses = dict(masses)

    def _update_finite_mixture(
        self,
        x: ArrayLike,
        y: float,
        observation_variance: float,
    ) -> None:
        """Apply one exact finite-mixture Bayes update before moment matching."""

        if bool(getattr(
            self, "_finite_mixture_cross_validated_structure", False
        )):
            self._finite_mixture_target_history.append({
                "x": tuple(int(v) for v in np.asarray(x, dtype=int)),
                "y": float(y),
                "observation_variance": max(
                    float(observation_variance), 1e-12),
            })
            self._finite_mixture_update_count = int(getattr(
                self, "_finite_mixture_update_count", 0)) + 1
            self._refit_cross_validated_finite_mixture()
            return

        if bool(getattr(
            self, "_finite_mixture_hierarchical_misspecification", False
        )):
            self._finite_mixture_target_history.append({
                "x": tuple(int(v) for v in np.asarray(x, dtype=int)),
                "y": float(y),
                "observation_variance": max(
                    float(observation_variance), 1e-12),
            })
            self._finite_mixture_update_count = int(getattr(
                self, "_finite_mixture_update_count", 0)) + 1
            self._refit_hierarchical_finite_mixture()
            return

        components = list(getattr(self, "_finite_mixture_components", []))
        weight = np.asarray(
            getattr(self, "_finite_mixture_weights", None),
            dtype=float,
        ).reshape(-1)
        if not components or len(components) != len(weight):
            raise RuntimeError("sequential mixture state is incomplete")
        if np.any(weight < 0.0) or not np.all(np.isfinite(weight)):
            raise FloatingPointError("sequential mixture weights are invalid")
        total = float(np.sum(weight))
        if total <= 0.0:
            raise FloatingPointError("sequential mixture has zero mass")
        weight /= total

        predictive_mean = np.asarray([
            component.posterior_mean(x) for component in components
        ], dtype=float)
        predictive_variance = np.asarray([
            component.posterior_var(x) + observation_variance
            for component in components
        ], dtype=float)
        predictive_variance = np.maximum(predictive_variance, 1e-12)
        log_predictive = -0.5 * (
            np.log(2.0 * np.pi * predictive_variance)
            + (float(y) - predictive_mean) ** 2 / predictive_variance
        )
        diagnostics = dict(getattr(
            self, "source_parametric_prior_diagnostics", {}) or {})
        temperature = max(
            float(diagnostics.get("evidence_temperature", 1.0)), 1e-6)
        weight_before = weight.copy()
        if bool(getattr(
            self, "_finite_mixture_preserve_group_masses", False
        )):
            posterior_weight, fixed_group_masses = (
                self.group_mass_preserving_weights(
                    weight,
                    log_predictive,
                    self._finite_mixture_group_labels,
                    self._finite_mixture_group_masses,
                    temperature,
                )
            )
        else:
            weight_before, posterior_weight = posterior_mixture_weights(
                weight, log_predictive, temperature)
            fixed_group_masses = None

        for component in components:
            component.update(x, float(y), observation_variance)

        update_count = int(getattr(
            self, "_finite_mixture_update_count", 0)) + 1
        component_names = list(getattr(
            self, "_finite_mixture_component_names", []))
        diagnostics.update({
            "adaptation_mode": "sequential_target_evidence_mixture",
            "component_names": component_names,
            "component_posterior_weights_before": weight_before.tolist(),
            "component_log_predictive": log_predictive.tolist(),
            "component_posterior_weights": posterior_weight.tolist(),
            "selected_component": str(component_names[int(
                np.argmax(posterior_weight))]),
            "target_only_posterior_weight": float(sum(
                mass for name, mass in zip(component_names, posterior_weight)
                if str(name).startswith("target:null")
            )),
            "source_posterior_weight": float(sum(
                mass for name, mass in zip(component_names, posterior_weight)
                if not str(name).startswith("target:null")
            )),
            "target_observation_count": int(
                diagnostics.get("target_observation_count", 0)) + 1,
            "online_mixture_update_count": int(update_count),
            "posterior_target_data_used": True,
            "target_oracle_used": False,
        })
        if fixed_group_masses is not None:
            diagnostics.update({
                "adaptation_mode": (
                    "sequential_assignment_prior_conditional_expert_mixture"),
                "assignment_group_masses_fixed": True,
                "assignment_group_masses": dict(fixed_group_masses),
                "target_oracle_used_for_group_masses": False,
            })
        self.set_moment_matched_posterior(
            components,
            posterior_weight,
            diagnostics=diagnostics,
            sequential_updates=True,
        )

    def update(self, x: ArrayLike, y: float, sigma2_hat: float) -> None:
        """Rank-one Kalman update using the plug-in observation variance."""
        if bool(getattr(self, "_finite_mixture_sequential", False)):
            y = float(y)
            observation_variance = max(float(sigma2_hat), 1e-12)
            if not np.isfinite(y):
                raise ValueError("GPR observation must be finite")
            if not np.isfinite(observation_variance):
                raise ValueError("GPR observation variance must be finite")
            self._update_finite_mixture(
                x, y, observation_variance)
            return
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
        diagnostics = {
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
        source_prior = getattr(
            self, "source_parametric_prior_diagnostics", None)
        if source_prior is not None:
            diagnostics["source_parametric_prior"] = dict(source_prior)
        if (
            self.basis_map is not None
            and hasattr(self.basis_map, "posterior_coefficient_diagnostics")
        ):
            basis_posterior = (
                self.basis_map.posterior_coefficient_diagnostics(
                    self.a[:self.p], self.C[:self.p, :self.p])
            )
            if basis_posterior is not None:
                diagnostics["basis_posterior"] = basis_posterior
        return diagnostics

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
        method = str(self._adaptive_spec.get(
            "method", "variational_spike_slab_bma"))
        if method == "nested_loo_group_ridge":
            self._adaptive_sparsity = AdaptiveGroupRidgePosterior(
                self._adaptive_spec["group_ids"],
                penalty_grid=self._adaptive_spec.get(
                    "penalty_grid", (1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0)),
                initial_feature_penalty=self._adaptive_spec.get(
                    "initial_feature_penalty"),
                coordinate_passes=self._adaptive_spec.get(
                    "coordinate_passes", 2),
                safety_weight=self._adaptive_spec.get("safety_weight", 2.0),
                residual_floor_scale=self._adaptive_spec.get(
                    "residual_floor_scale", 0.05),
            )
        else:
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
                shared_shrinkage_groups=self._adaptive_spec.get(
                    "shared_shrinkage_groups"),
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
