"""Transferable orthogonal additive groups in cumulative-risk coordinates."""

from __future__ import annotations

import hashlib

import numpy as np

from representation.transferable_spectral import (
    SourceDomainBatch,
    TransferableSpectralBasis,
)


class TransferableAdditiveGroupBank:
    """Learn a source-only finite prior over low-order ANOVA groups.

    Each retained group contributes one low-frequency function.  Main effects
    and two-way interactions are ranked across source domains, then whitened in
    that source-determined order.  A held-out target may select groups, but it
    cannot change their functions, ordering, or prior weights.
    """

    def __init__(
        self,
        max_groups=8,
        max_library_size=64,
        low_frequency_components=8,
        n_neighbors=10,
        relevance_floor=0.05,
        temperature=0.5,
        ridge=1e-8,
        base_basis=None,
        strong_heredity=False,
        max_interactions=2,
    ):
        self.max_groups = max(1, int(max_groups))
        self.max_library_size = max(1, int(max_library_size))
        self.low_frequency_components = max(
            1, int(low_frequency_components))
        self.n_neighbors = max(1, int(n_neighbors))
        self.relevance_floor = max(float(relevance_floor), 0.0)
        self.temperature = max(float(temperature), 1e-8)
        self.ridge = max(float(ridge), 1e-12)
        self.base_basis = base_basis
        self.strong_heredity = bool(strong_heredity)
        self.max_interactions = max(0, int(max_interactions))

        self.psi_mean_: np.ndarray | None = None
        self.psi_scale_: np.ndarray | None = None
        self.library_mean_: np.ndarray | None = None
        self.library_scale_: np.ndarray | None = None
        self.selected_idx_: np.ndarray | None = None
        self.whitening_: np.ndarray | None = None
        self.base_projection_: np.ndarray | None = None
        self.group_names_: list[str] = []
        self.function_names_: list[str] = []
        self.source_weights_: np.ndarray | None = None
        self.clip_low_: np.ndarray | None = None
        self.clip_high_: np.ndarray | None = None
        self.feature_dim = 0
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
        pooled_library, function_names, group_names = self._library(pooled_psi)
        self.library_mean_ = np.mean(pooled_library, axis=0)
        self.library_scale_ = np.std(pooled_library, axis=0)
        self.library_scale_ = np.where(
            self.library_scale_ < 1e-8, 1.0, self.library_scale_)

        helper = TransferableSpectralBasis(
            low_frequency_components=self.low_frequency_components,
            n_neighbors=self.n_neighbors,
            relevance_floor=self.relevance_floor,
        )
        libraries = []
        low_rows = []
        relevance_rows = []
        sign_rows = []
        pooled_weights = []
        for batch in batches:
            library = self._standardized_library(batch.psi)
            libraries.append(library)
            weights = helper._weights(batch)
            pooled_weights.append(weights)
            low_rows.append(helper._low_frequency_ratio(
                self._scale_psi(batch.psi), library))
            relevance, signs = helper._signal_relevance(
                library,
                batch.signals,
                weights,
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
        relevance_stability = relevance_mean / (
            relevance_mean + relevance_std + 1e-12)
        prevalence = np.mean(relevance >= self.relevance_floor, axis=0)
        sign_consistency = helper._sign_consistency(signs, relevance)
        function_score = (
            low_stable
            * relevance_mean
            * (0.5 + 0.5 * relevance_stability)
            * (0.5 + 0.5 * prevalence)
            * (0.75 + 0.25 * sign_consistency)
        )

        best_by_group = {}
        for index, group in enumerate(group_names):
            candidate = (float(function_score[index]), -int(index), int(index))
            if group not in best_by_group or candidate > best_by_group[group]:
                best_by_group[group] = candidate
        ranked_candidates = sorted(
            (
                (score, -neg_index, index, group)
                for group, (score, neg_index, index) in best_by_group.items()
            ),
            key=lambda row: (-row[0], row[1], row[3]),
        )
        if self.strong_heredity:
            ranked_candidates = self._strong_heredity_order(ranked_candidates)
        pooled_standardized = np.vstack(libraries)
        selection_weights = np.concatenate(pooled_weights)
        selection_weights = np.clip(selection_weights, 1e-8, np.inf)
        selection_weights /= max(float(np.sum(selection_weights)), 1e-12)
        ranked = []
        selected_columns = []
        for row in ranked_candidates:
            candidate = pooled_standardized[:, int(row[2])]
            total_energy = float(np.sum(selection_weights * candidate ** 2))
            if total_energy <= 1e-12:
                continue
            if selected_columns:
                design = pooled_standardized[:, selected_columns]
                gram = design.T @ (design * selection_weights[:, None])
                cross = design.T @ (candidate * selection_weights)
                try:
                    coefficient = np.linalg.solve(gram, cross)
                except np.linalg.LinAlgError:
                    coefficient = np.linalg.lstsq(
                        gram, cross, rcond=None)[0]
                residual = candidate - design @ coefficient
                independence = float(np.sum(
                    selection_weights * residual ** 2) / total_energy)
                if independence <= 1e-5:
                    continue
            ranked.append(row)
            selected_columns.append(int(row[2]))
            if len(ranked) >= self.max_groups:
                break
        self.selected_idx_ = np.asarray([row[2] for row in ranked], dtype=int)
        self.group_names_ = [row[3] for row in ranked]
        self.function_names_ = [
            function_names[int(index)] for index in self.selected_idx_
        ]

        selected_raw = np.vstack(libraries)[:, self.selected_idx_]
        weights = selection_weights
        max_cross_gram = 0.0
        if self.base_basis is not None:
            base = np.asarray(
                self.base_basis.transform(pooled_psi), dtype=float)
            if base.ndim == 1:
                base = base[:, None]
            sqrt_weight = np.sqrt(weights)[:, None]
            self.base_projection_ = np.linalg.lstsq(
                base * sqrt_weight,
                selected_raw * sqrt_weight,
                rcond=1e-10,
            )[0]
            selected = selected_raw - base @ self.base_projection_
        else:
            self.base_projection_ = None
            selected = selected_raw
        retained = []
        for position in range(selected.shape[1]):
            candidate = selected[:, position]
            raw_energy = float(np.sum(
                weights * selected_raw[:, position] ** 2))
            residual_energy = float(np.sum(weights * candidate ** 2))
            if residual_energy <= 1e-5 * max(raw_energy, 1e-12):
                continue
            if retained:
                design = selected[:, retained]
                sqrt_weight = np.sqrt(weights)[:, None]
                coefficient = np.linalg.lstsq(
                    design * sqrt_weight,
                    candidate[:, None] * sqrt_weight,
                    rcond=1e-10,
                )[0][:, 0]
                residual = candidate - design @ coefficient
                independence = float(np.sum(weights * residual ** 2))
                if independence <= 1e-5 * residual_energy:
                    continue
            retained.append(position)
        if not retained:
            main_positions = [
                position for position, row in enumerate(ranked)
                if str(row[3]).startswith("main:")
            ]
            pool = main_positions or list(range(selected.shape[1]))
            retained = [max(
                pool,
                key=lambda position: float(np.sum(
                    selected[:, position] ** 2 * weights)),
            )]
        if self.strong_heredity:
            retained_names = {ranked[position][3] for position in retained}
            retained = [
                position for position in retained
                if self._parents_present(ranked[position][3], retained_names)
            ]
            if not retained:
                retained = [next(
                    position for position, row in enumerate(ranked)
                    if str(row[3]).startswith("main:")
                )]
        selected = selected[:, retained]
        selected_raw = selected_raw[:, retained]
        ranked = [ranked[position] for position in retained]
        self.selected_idx_ = self.selected_idx_[retained]
        self.group_names_ = [self.group_names_[position] for position in retained]
        self.function_names_ = [
            self.function_names_[position] for position in retained
        ]
        if self.base_projection_ is not None:
            self.base_projection_ = self.base_projection_[:, retained]
        covariance = selected.T @ (selected * weights[:, None])
        covariance = 0.5 * (covariance + covariance.T)
        try:
            lower = np.linalg.cholesky(covariance)
            self.whitening_ = np.linalg.solve(
                lower.T, np.eye(len(lower), dtype=float))
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            stable = np.maximum(eigenvalues, min(self.ridge, 1e-8))
            self.whitening_ = (
                eigenvectors * (1.0 / np.sqrt(stable))[None, :]
            ) @ eigenvectors.T
        transformed = selected @ self.whitening_
        if self.base_basis is not None:
            cross_gram = base.T @ (transformed * weights[:, None])
            max_cross_gram = (
                float(np.max(np.abs(cross_gram)))
                if cross_gram.size else 0.0
            )
        self.clip_low_ = np.quantile(transformed, 0.01, axis=0)
        self.clip_high_ = np.quantile(transformed, 0.99, axis=0)
        gram = transformed.T @ (transformed * weights[:, None])
        offdiag = gram - np.diag(np.diag(gram))

        group_scores = np.asarray([row[0] for row in ranked], dtype=float)
        reference = float(np.max(group_scores)) if len(group_scores) else 0.0
        logits = (group_scores - reference) / self.temperature
        source_weights = np.exp(np.clip(logits, -50.0, 0.0))
        source_weights /= max(float(np.sum(source_weights)), 1e-12)
        self.source_weights_ = source_weights
        self.feature_dim = int(len(self.selected_idx_))
        self.diagnostics_ = {
            "status": "fit",
            "method": "source_low_frequency_orthogonal_anova_groups",
            "strong_heredity": bool(self.strong_heredity),
            "max_interactions": int(self.max_interactions),
            "target_data_used": False,
            "source_domains": sorted(batch.domain for batch in batches),
            "n_source_records": int(len(pooled_psi)),
            "library_dim": int(len(function_names)),
            "feature_dim": int(self.feature_dim),
            "group_names": list(self.group_names_),
            "function_names": list(self.function_names_),
            "source_scores": [float(value) for value in group_scores],
            "source_weights": [float(value) for value in source_weights],
            "source_clip_low": self.clip_low_.tolist(),
            "source_clip_high": self.clip_high_.tolist(),
            "max_offdiag_gram": (
                float(np.max(np.abs(offdiag))) if offdiag.size else 0.0
            ),
            "max_stage1_cross_gram": float(max_cross_gram),
            "max_diag_error": float(np.max(np.abs(np.diag(gram) - 1.0))),
            "fingerprint": self.fingerprint(),
        }
        return self

    def _strong_heredity_order(self, ranked_candidates):
        mains = [
            row for row in ranked_candidates
            if str(row[3]).startswith("main:")
        ]
        interactions = [
            row for row in ranked_candidates
            if str(row[3]).startswith("interaction:")
        ]
        reserve = min(self.max_interactions, max(self.max_groups - 2, 0))
        main_limit = max(1, self.max_groups - reserve)
        selected_mains = mains[:main_limit]
        selected_names = {row[3] for row in selected_mains}
        selected_interactions = []
        for row in interactions:
            if len(selected_interactions) >= reserve:
                break
            if self._parents_present(row[3], selected_names):
                selected_interactions.append(row)
        ordered = selected_mains + selected_interactions
        if len(ordered) < self.max_groups:
            used = {row[3] for row in ordered}
            ordered.extend(
                row for row in mains[main_limit:]
                if row[3] not in used
            )
        return ordered[:self.max_groups]

    @staticmethod
    def _parents_present(group, selected_names):
        text = str(group)
        if not text.startswith("interaction:"):
            return True
        _, left, right = text.split(":", 2)
        return (
            f"main:{left}" in selected_names
            and f"main:{right}" in selected_names
        )

    def transform(self, psi):
        if self.selected_idx_ is None or self.whitening_ is None:
            raise RuntimeError("additive group bank must be fit before use")
        arr = np.asarray(psi, dtype=float)
        one = arr.ndim == 1
        if one:
            arr = arr[None, :]
        out = self._transform_unclipped(arr)
        out = np.clip(out, self.clip_low_, self.clip_high_)
        return out[0].copy() if one else np.asarray(out, dtype=float)

    def support_saturation(self, psi, index):
        arr = np.asarray(psi, dtype=float)
        one = arr.ndim == 1
        if one:
            arr = arr[None, :]
        raw = self._transform_unclipped(arr)[:, int(index)]
        saturated = np.logical_or(
            raw < float(self.clip_low_[int(index)]),
            raw > float(self.clip_high_[int(index)]),
        )
        return bool(saturated[0]) if one else saturated

    def transform_groups(self, psi, indices):
        values = np.asarray(self.transform(psi), dtype=float)
        indices = np.asarray(indices, dtype=int)
        if values.ndim == 1:
            return values[indices]
        return values[:, indices]

    def source_weight(self, index):
        if self.source_weights_ is None:
            return 0.0
        return float(self.source_weights_[int(index)])

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
            self.base_projection_,
            self.clip_low_,
            self.clip_high_,
        ):
            if value is not None:
                digest.update(np.asarray(value).tobytes())
        return digest.hexdigest()[:16]

    def _transform_unclipped(self, psi):
        library = self._standardized_library(psi)
        selected = library[:, self.selected_idx_]
        if self.base_projection_ is not None:
            base = np.asarray(self.base_basis.transform(psi), dtype=float)
            if base.ndim == 1:
                base = base[None, :]
            selected = selected - base @ self.base_projection_
        return selected @ self.whitening_

    def _scale_psi(self, psi):
        return (
            np.asarray(psi, dtype=float) - self.psi_mean_
        ) / self.psi_scale_

    def _standardized_library(self, psi):
        library, _, _ = self._library(psi)
        return (library - self.library_mean_) / self.library_scale_

    def _library(self, psi):
        z = self._scale_psi(psi) if self.psi_mean_ is not None else np.asarray(
            psi, dtype=float)
        if z.ndim == 1:
            z = z[None, :]
        smooth = np.tanh(0.5 * z)
        columns = []
        functions = []
        groups = []

        def add(values, function, group):
            if len(columns) < self.max_library_size:
                columns.append(np.asarray(values, dtype=float))
                functions.append(function)
                groups.append(group)

        for index in range(smooth.shape[1]):
            group = f"main:{index}"
            add(smooth[:, index], f"psi{index}", group)
            add(smooth[:, index] ** 2, f"psi{index}^2", group)
            add(np.sin(np.pi * smooth[:, index]), f"sin(pi*psi{index})", group)
            add(np.cos(np.pi * smooth[:, index]), f"cos(pi*psi{index})", group)
        for left in range(smooth.shape[1]):
            for right in range(left + 1, smooth.shape[1]):
                add(
                    smooth[:, left] * smooth[:, right],
                    f"psi{left}*psi{right}",
                    f"interaction:{left}:{right}",
                )
        if not columns:
            return (
                np.ones((len(z), 1), dtype=float),
                ["constant"],
                ["constant"],
            )
        return np.vstack(columns).T, functions, groups

    @staticmethod
    def _validate_batch(batch):
        if not isinstance(batch, SourceDomainBatch):
            batch = SourceDomainBatch(**batch)
        return TransferableSpectralBasis._validate_batch(batch)
