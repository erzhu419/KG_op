"""Source-invariant low-frequency basis learning in risk coordinates.

The learner in this module is deliberately separated from HVD priors and
candidate proposals.  It answers one narrow question: can source domains
identify a compact set of smooth, response-relevant functions of
``psi=(A,N)`` that remains frozen on a held-out target?
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class SourceDomainBatch:
    domain: str
    psi: np.ndarray
    signals: np.ndarray
    sample_weight: np.ndarray | None = None
    signal_weight: np.ndarray | None = None


class TransferableSpectralBasis:
    """Learn a frozen low-frequency orthogonal basis from source domains.

    Candidate functions are fixed and domain-agnostic.  Source observations
    only choose which functions are both smooth on the ``psi`` graph and
    consistently relevant to objective/constraint signals.  The selected
    functions are then whitened under the pooled source distribution.
    """

    def __init__(
        self,
        active_dim=6,
        max_library_size=64,
        low_frequency_components=8,
        n_neighbors=10,
        relevance_floor=0.05,
        ridge=1e-8,
        pilot_cv_size=10,
        pilot_cv_repeats=3,
        pilot_cv_weight=1.0,
    ):
        self.active_dim = int(active_dim)
        self.max_library_size = int(max_library_size)
        self.low_frequency_components = int(low_frequency_components)
        self.n_neighbors = int(n_neighbors)
        self.relevance_floor = float(relevance_floor)
        self.ridge = float(ridge)
        self.pilot_cv_size = int(pilot_cv_size)
        self.pilot_cv_repeats = int(pilot_cv_repeats)
        self.pilot_cv_weight = float(pilot_cv_weight)

        self.psi_mean_: np.ndarray | None = None
        self.psi_scale_: np.ndarray | None = None
        self.library_mean_: np.ndarray | None = None
        self.library_scale_: np.ndarray | None = None
        self.library_names_: list[str] = []
        self.selected_idx_: np.ndarray | None = None
        self.whitening_: np.ndarray | None = None
        self.feature_dim = 0
        self.source_domains_: list[str] = []
        self.diagnostics_: dict = {"status": "unfit"}

    def fit(self, batches):
        batches = [self._validate_batch(batch) for batch in batches]
        if not batches:
            raise ValueError("at least one source-domain batch is required")
        psi_dim = batches[0].psi.shape[1]
        if any(batch.psi.shape[1] != psi_dim for batch in batches):
            raise ValueError("all source domains must use the same psi dimension")

        pooled_psi = np.vstack([batch.psi for batch in batches])
        self.psi_mean_ = np.mean(pooled_psi, axis=0)
        self.psi_scale_ = np.std(pooled_psi, axis=0)
        self.psi_scale_ = np.where(self.psi_scale_ < 1e-8, 1.0, self.psi_scale_)

        pooled_z = self._scale_psi(pooled_psi)
        pooled_library, names = self._library(pooled_z)
        self.library_names_ = names
        self.library_mean_ = np.mean(pooled_library, axis=0)
        self.library_scale_ = np.std(pooled_library, axis=0)
        self.library_scale_ = np.where(self.library_scale_ < 1e-8, 1.0, self.library_scale_)

        low_rows = []
        relevance_rows = []
        sign_rows = []
        standardized_libraries = []
        pooled_weights = []
        for batch in batches:
            library = self._standardized_library(batch.psi)
            standardized_libraries.append(library)
            pooled_weights.append(self._weights(batch))
            low_rows.append(self._low_frequency_ratio(self._scale_psi(batch.psi), library))
            relevance, signs = self._signal_relevance(
                library,
                batch.signals,
                self._weights(batch),
                batch.signal_weight,
            )
            relevance_rows.append(relevance)
            sign_rows.append(signs)

        low = np.vstack(low_rows)
        relevance = np.vstack(relevance_rows)
        signs = np.stack(sign_rows, axis=0)
        low_stable = np.quantile(low, 0.25, axis=0)
        relevance_mean = np.mean(relevance, axis=0)
        relevance_std = np.std(relevance, axis=0)
        relevance_stability = relevance_mean / (relevance_mean + relevance_std + 1e-12)
        prevalence = np.mean(relevance >= self.relevance_floor, axis=0)
        sign_consistency = self._sign_consistency(signs, relevance)
        score = (
            low_stable
            * relevance_mean
            * (0.5 + 0.5 * relevance_stability)
            * (0.5 + 0.5 * prevalence)
            * (0.75 + 0.25 * sign_consistency)
        )

        n_select = min(max(1, self.active_dim), len(score))
        selected_idx = []
        selected_cv_loss = []
        remaining = list(range(len(score)))
        for _ in range(n_select):
            candidates = []
            for feature_idx in remaining:
                subset = selected_idx + [feature_idx]
                cv_loss = self._pilot_cv_loss(
                    batches,
                    standardized_libraries,
                    subset,
                )
                structural = max(float(score[feature_idx]), 1e-12)
                value = structural / (
                    1.0 + max(self.pilot_cv_weight, 0.0) * cv_loss)
                candidates.append((-value, cv_loss, feature_idx))
            _, cv_loss, chosen = min(candidates)
            selected_idx.append(int(chosen))
            selected_cv_loss.append(float(cv_loss))
            remaining.remove(chosen)
        self.selected_idx_ = np.asarray(selected_idx, dtype=int)

        selected = np.vstack(standardized_libraries)[:, self.selected_idx_]
        weights = np.concatenate(pooled_weights)
        weights = np.clip(weights, 1e-8, np.inf)
        weights = weights / float(np.sum(weights))
        covariance = selected.T @ (selected * weights[:, None])
        covariance = 0.5 * (covariance + covariance.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            eigenvalues = np.ones(n_select, dtype=float)
            eigenvectors = np.eye(n_select, dtype=float)
        eigenvalues = np.maximum(eigenvalues, self.ridge)
        self.whitening_ = (
            eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]
        ) @ eigenvectors.T
        transformed = selected @ self.whitening_
        gram = transformed.T @ (transformed * weights[:, None])
        offdiag = gram - np.diag(np.diag(gram))

        self.feature_dim = int(transformed.shape[1])
        self.source_domains_ = sorted(batch.domain for batch in batches)
        selected_names = [self.library_names_[int(i)] for i in self.selected_idx_]
        self.diagnostics_ = {
            "status": "fit",
            "source_domains": list(self.source_domains_),
            "n_source_domains": int(len(batches)),
            "n_source_records": int(len(pooled_psi)),
            "psi_dim": int(psi_dim),
            "library_dim": int(len(self.library_names_)),
            "active_dim": int(self.feature_dim),
            "selected_names": selected_names,
            "selected_scores": [float(score[i]) for i in self.selected_idx_],
            "selected_low_frequency": [float(low_stable[i]) for i in self.selected_idx_],
            "selected_relevance": [float(relevance_mean[i]) for i in self.selected_idx_],
            "selected_prevalence": [float(prevalence[i]) for i in self.selected_idx_],
            "selected_sign_consistency": [
                float(sign_consistency[i]) for i in self.selected_idx_
            ],
            "selected_pilot_cv_loss": selected_cv_loss,
            "pilot_cv_size": int(self.pilot_cv_size),
            "pilot_cv_repeats": int(self.pilot_cv_repeats),
            "pilot_cv_weight": float(self.pilot_cv_weight),
            "max_offdiag_gram": float(np.max(np.abs(offdiag))) if offdiag.size else 0.0,
            "max_diag_error": float(np.max(np.abs(np.diag(gram) - 1.0))),
            "fingerprint": self.fingerprint(),
        }
        return self

    def transform(self, psi):
        if self.selected_idx_ is None or self.whitening_ is None:
            raise RuntimeError("TransferableSpectralBasis must be fit before use")
        arr = np.asarray(psi, dtype=float)
        one = arr.ndim == 1
        if one:
            arr = arr[None, :]
        library = self._standardized_library(arr)
        out = library[:, self.selected_idx_] @ self.whitening_
        return out[0].copy() if one else np.asarray(out, dtype=float)

    def diagnostics(self):
        return dict(self.diagnostics_)

    def fingerprint(self):
        digest = hashlib.sha256()
        for value in (
            self.psi_mean_,
            self.psi_scale_,
            self.library_mean_,
            self.library_scale_,
            self.selected_idx_,
            self.whitening_,
        ):
            if value is not None:
                digest.update(np.asarray(value).tobytes())
        return digest.hexdigest()[:16]

    @staticmethod
    def _validate_batch(batch):
        if not isinstance(batch, SourceDomainBatch):
            batch = SourceDomainBatch(**batch)
        psi = np.asarray(batch.psi, dtype=float)
        signals = np.asarray(batch.signals, dtype=float)
        if psi.ndim != 2 or len(psi) < 3:
            raise ValueError("each source batch needs at least three 2-D psi rows")
        if signals.ndim == 1:
            signals = signals[:, None]
        if signals.ndim != 2 or len(signals) != len(psi):
            raise ValueError("signals must have one row per psi row")
        if not np.all(np.isfinite(psi)) or not np.all(np.isfinite(signals)):
            raise ValueError("source batches may not contain NaN or infinity")
        weights = None
        if batch.sample_weight is not None:
            weights = np.asarray(batch.sample_weight, dtype=float).reshape(-1)
            if len(weights) != len(psi):
                raise ValueError("sample_weight must have one value per row")
        signal_weight = None
        if batch.signal_weight is not None:
            signal_weight = np.asarray(batch.signal_weight, dtype=float).reshape(-1)
            if len(signal_weight) != signals.shape[1]:
                raise ValueError("signal_weight must have one value per signal")
            if not np.all(np.isfinite(signal_weight)) or np.any(signal_weight < 0.0):
                raise ValueError("signal_weight must be finite and nonnegative")
        return SourceDomainBatch(
            str(batch.domain),
            psi,
            signals,
            weights,
            signal_weight,
        )

    @staticmethod
    def _weights(batch):
        if batch.sample_weight is None:
            return np.ones(len(batch.psi), dtype=float)
        return np.clip(np.asarray(batch.sample_weight, dtype=float), 1e-8, np.inf)

    def _scale_psi(self, psi):
        if self.psi_mean_ is None or self.psi_scale_ is None:
            raise RuntimeError("psi scaler is not fit")
        return (np.asarray(psi, dtype=float) - self.psi_mean_) / self.psi_scale_

    def _standardized_library(self, psi):
        if self.library_mean_ is None or self.library_scale_ is None:
            raise RuntimeError("library scaler is not fit")
        library, _ = self._library(self._scale_psi(psi))
        return (library - self.library_mean_) / self.library_scale_

    def _library(self, psi_z):
        z = np.asarray(psi_z, dtype=float)
        if z.ndim == 1:
            z = z[None, :]
        smooth = np.tanh(0.5 * z)
        cols = []
        names = []

        def add(values, name):
            if len(cols) < max(1, self.max_library_size):
                cols.append(np.asarray(values, dtype=float))
                names.append(name)

        for j in range(smooth.shape[1]):
            add(smooth[:, j], f"psi{j}")
        for j in range(smooth.shape[1]):
            add(smooth[:, j] ** 2, f"psi{j}^2")
        for j in range(smooth.shape[1]):
            add(np.sin(np.pi * smooth[:, j]), f"sin(pi*psi{j})")
            add(np.cos(np.pi * smooth[:, j]), f"cos(pi*psi{j})")
        for i in range(smooth.shape[1]):
            for j in range(i + 1, smooth.shape[1]):
                add(smooth[:, i] * smooth[:, j], f"psi{i}*psi{j}")
        if not cols:
            return np.ones((len(z), 1), dtype=float), ["constant"]
        return np.vstack(cols).T, names

    def _low_frequency_ratio(self, psi_z, library):
        n = len(psi_z)
        if n <= 3:
            return np.ones(library.shape[1], dtype=float)
        dist2 = self._sqdist(psi_z, psi_z)
        positive = dist2[np.triu_indices_from(dist2, k=1)]
        positive = positive[positive > 1e-12]
        scale = float(np.median(positive)) if len(positive) else 1.0
        W = np.exp(-dist2 / max(scale, 1e-12))
        np.fill_diagonal(W, 0.0)
        k = min(max(1, self.n_neighbors), n - 1)
        if k < n - 1:
            keep = np.zeros_like(W, dtype=bool)
            for row in range(n):
                keep[row, np.argsort(dist2[row])[1:k + 1]] = True
            W = np.where(np.logical_or(keep, keep.T), W, 0.0)
        degree = np.sum(W, axis=1)
        inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
        normalized = np.eye(n) - inv_sqrt[:, None] * W * inv_sqrt[None, :]
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (normalized + normalized.T))
        except np.linalg.LinAlgError:
            return np.ones(library.shape[1], dtype=float)
        order = np.argsort(eigenvalues)
        count = min(max(1, self.low_frequency_components), n)
        low_vectors = eigenvectors[:, order[:count]]
        total = np.sum(library ** 2, axis=0) + 1e-12
        low = np.sum((low_vectors.T @ library) ** 2, axis=0)
        return np.clip(low / total, 0.0, 1.0)

    @staticmethod
    def _signal_relevance(library, signals, weights, signal_weight=None):
        weights = np.asarray(weights, dtype=float)
        weights = weights / float(np.sum(weights))
        signal = np.asarray(signals, dtype=float)
        signal = signal - np.sum(signal * weights[:, None], axis=0, keepdims=True)
        signal_scale = np.sqrt(np.sum(signal ** 2 * weights[:, None], axis=0))
        signal_scale = np.where(signal_scale < 1e-10, 1.0, signal_scale)
        signal = signal / signal_scale
        feature = library - np.sum(library * weights[:, None], axis=0, keepdims=True)
        feature_scale = np.sqrt(np.sum(feature ** 2 * weights[:, None], axis=0))
        feature_scale = np.where(feature_scale < 1e-10, 1.0, feature_scale)
        feature = feature / feature_scale
        corr = feature.T @ (signal * weights[:, None])
        if signal_weight is None:
            signal_weight = np.ones(signal.shape[1], dtype=float)
        signal_weight = np.asarray(signal_weight, dtype=float).reshape(1, -1)
        relevance = np.max(np.abs(corr) * signal_weight, axis=1)
        return relevance, np.sign(corr)

    @staticmethod
    def _sign_consistency(signs, relevance):
        # Pick each feature's most consistently relevant signal, then measure
        # orientation agreement across domains.  Sign only breaks ties; a
        # smooth feature remains admissible when domains reverse its effect.
        n_domains, n_features, n_signals = signs.shape
        out = np.full(n_features, 0.5, dtype=float)
        for feature in range(n_features):
            best_signal = 0
            best_agreement = -1.0
            for signal in range(n_signals):
                vals = signs[:, feature, signal]
                nonzero = vals[vals != 0.0]
                if len(nonzero) == 0:
                    agreement = 0.5
                else:
                    positive = float(np.mean(nonzero > 0.0))
                    agreement = max(positive, 1.0 - positive)
                if agreement > best_agreement:
                    best_agreement = agreement
                    best_signal = signal
            del best_signal
            out[feature] = best_agreement
        return out

    def _pilot_cv_loss(self, batches, libraries, subset):
        """Few-shot source validation matching held-out target adaptation."""

        losses = []
        subset = np.asarray(subset, dtype=int)
        for batch, library in zip(batches, libraries):
            n_rows = len(library)
            if n_rows < 6:
                continue
            pilot = min(max(5, self.pilot_cv_size), max(5, n_rows // 2))
            pilot = min(pilot, n_rows - 2)
            domain_seed = int.from_bytes(
                hashlib.sha256(batch.domain.encode("utf-8")).digest()[:8],
                "little",
            )
            for repeat in range(max(1, self.pilot_cv_repeats)):
                rng = np.random.default_rng(domain_seed + 104729 * repeat)
                order = rng.permutation(n_rows)
                train = order[:pilot]
                test = order[pilot:]
                X_train = library[train][:, subset]
                X_test = library[test][:, subset]
                Y_train = batch.signals[train]
                Y_test = batch.signals[test]
                prediction = self._ridge_predict_many(
                    X_train,
                    Y_train,
                    X_test,
                )
                test_weight = self._weights(batch)[test]
                test_weight = test_weight / max(float(np.sum(test_weight)), 1e-12)
                signal_weight = (
                    np.asarray(batch.signal_weight, dtype=float)
                    if batch.signal_weight is not None
                    else np.ones(Y_test.shape[1], dtype=float)
                )
                per_signal = []
                for signal_idx in range(Y_test.shape[1]):
                    truth = Y_test[:, signal_idx]
                    pred = prediction[:, signal_idx]
                    center = float(np.sum(truth * test_weight))
                    variance = float(np.sum(
                        (truth - center) ** 2 * test_weight))
                    mse = float(np.sum(
                        (truth - pred) ** 2 * test_weight)) / max(variance, 1e-8)
                    if signal_idx == 1:
                        infeasible = truth > 0.0
                        feasible = ~infeasible
                        false_feasible = (
                            float(np.mean(pred[infeasible] <= 0.0))
                            if np.any(infeasible)
                            else 0.0
                        )
                        false_infeasible = (
                            float(np.mean(pred[feasible] > 0.0))
                            if np.any(feasible)
                            else 0.0
                        )
                        mse += 3.0 * false_feasible + 0.25 * false_infeasible
                    per_signal.append(float(mse))
                losses.append(float(np.average(
                    np.asarray(per_signal),
                    weights=np.maximum(signal_weight, 1e-8),
                )))
        return float(np.mean(losses)) if losses else float("inf")

    def _ridge_predict_many(self, train_x, train_y, test_x):
        train_x = np.asarray(train_x, dtype=float)
        test_x = np.asarray(test_x, dtype=float)
        train_y = np.asarray(train_y, dtype=float)
        x_mean = np.mean(train_x, axis=0)
        x_scale = np.std(train_x, axis=0)
        x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
        X = (train_x - x_mean) / x_scale
        X_test = (test_x - x_mean) / x_scale
        X = np.column_stack([np.ones(len(X)), X])
        X_test = np.column_stack([np.ones(len(X_test)), X_test])
        y_mean = np.mean(train_y, axis=0)
        y_scale = np.std(train_y, axis=0)
        y_scale = np.where(y_scale < 1e-8, 1.0, y_scale)
        Y = (train_y - y_mean) / y_scale
        penalty = max(self.ridge, 1e-4) * np.eye(X.shape[1])
        penalty[0, 0] = 0.0
        try:
            beta = np.linalg.solve(X.T @ X + penalty, X.T @ Y)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(
                X.T @ X + penalty,
                X.T @ Y,
                rcond=None,
            )[0]
        return (X_test @ beta) * y_scale + y_mean

    @staticmethod
    def _sqdist(A, B):
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        aa = np.sum(A ** 2, axis=1)[:, None]
        bb = np.sum(B ** 2, axis=1)[None, :]
        return np.maximum(aa + bb - 2.0 * A @ B.T, 0.0)
