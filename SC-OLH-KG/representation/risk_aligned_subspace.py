"""Boundary-conditioned alignment of transferable risk subspaces.

The transferable object in this module is a collection of projectors rather
than individually named latent axes.  Source domains are first aligned by
chance-margin prototypes.  A generalized eigensystem then balances boundary
relevance against conditional domain discrepancy.  Features are computed from
projectors and a canonical boundary direction, so rotations inside a retained
subspace do not change the representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from representation.transferable_spectral import SourceDomainBatch
from representation.transferable_spectral import TransferableSpectralBasis


@dataclass(frozen=True)
class TargetRiskAlignment:
    """Target-pilot orthogonal adapter and its audit diagnostics."""

    matrix: np.ndarray
    accepted: bool
    diagnostics: dict
    boundary_axis: np.ndarray | None = None
    expert_weights: np.ndarray | None = None


class BoundaryAlignedRiskSubspaces:
    """Learn source-invariant risk subspaces conditioned on chance margin.

    ``signals[:, boundary_signal_index]`` must be a dimensionless signed
    chance margin, with nonpositive values denoting feasibility.  Only source
    batches are used by :meth:`fit`.  A held-out target may estimate an
    orthogonal adapter from its ordinary pilot observations via
    :meth:`fit_target_adapter`; no target oracle pool is read.
    """

    def __init__(
        self,
        active_dim=4,
        subspace_dim=2,
        boundary_signal_index=1,
        boundary_edges=(-1.0, -0.25, 0.25, 1.0),
        boundary_temperature=1.0,
        boundary_relevance=1.0,
        boundary_between=0.5,
        domain_penalty=0.5,
        ridge=1e-6,
        procrustes_iterations=8,
        source_adapter_ridge=0.25,
        apply_source_procrustes=False,
        target_adapter_ridge=5.0,
        target_min_gain=0.02,
        target_min_bins=3,
        target_max_rotation=2.0,
    ):
        self.active_dim = max(1, int(active_dim))
        self.subspace_dim = max(1, int(subspace_dim))
        self.boundary_signal_index = int(boundary_signal_index)
        self.boundary_edges = tuple(sorted(float(v) for v in boundary_edges))
        self.boundary_temperature = max(float(boundary_temperature), 1e-8)
        self.boundary_relevance = max(float(boundary_relevance), 0.0)
        self.boundary_between = max(float(boundary_between), 0.0)
        self.domain_penalty = max(float(domain_penalty), 0.0)
        self.ridge = max(float(ridge), 1e-12)
        self.procrustes_iterations = max(1, int(procrustes_iterations))
        self.source_adapter_ridge = max(float(source_adapter_ridge), 0.0)
        self.apply_source_procrustes = bool(apply_source_procrustes)
        self.target_adapter_ridge = max(float(target_adapter_ridge), 0.0)
        self.target_min_gain = max(float(target_min_gain), 0.0)
        self.target_min_bins = max(2, int(target_min_bins))
        self.target_max_rotation = max(float(target_max_rotation), 0.0)

        self.psi_mean_: np.ndarray | None = None
        self.psi_scale_: np.ndarray | None = None
        self.source_adapters_: dict[str, np.ndarray] = {}
        self.source_procrustes_adapters_: dict[str, np.ndarray] = {}
        self.reference_prototypes_: np.ndarray | None = None
        self.reference_counts_: np.ndarray | None = None
        self.subspace_bases_: list[np.ndarray] = []
        self.projectors_: list[np.ndarray] = []
        self.boundary_axes_: list[np.ndarray] = []
        self.expert_domains_: list[str] = []
        self.expert_coefficients_: np.ndarray | None = None
        self.risk_correction_coefficients_: np.ndarray | None = None
        self.expert_prior_weights_: np.ndarray | None = None
        self.source_expert_weights_: dict[str, np.ndarray] = {}
        self.expert_cross_loss_: np.ndarray | None = None
        self.feature_dim = 0
        self.source_residual_guard_ = 0.0
        self.diagnostics_: dict = {"status": "unfit"}

    @property
    def n_bins(self):
        return len(self.boundary_edges) + 1

    def fit(self, batches):
        batches = [TransferableSpectralBasis._validate_batch(batch) for batch in batches]
        if not batches:
            raise ValueError("at least one source-domain batch is required")
        psi_dim = int(batches[0].psi.shape[1])
        if any(batch.psi.shape[1] != psi_dim for batch in batches):
            raise ValueError("all source domains must use the same psi dimension")
        if not 0 <= self.boundary_signal_index < batches[0].signals.shape[1]:
            raise ValueError("boundary_signal_index is outside source signals")

        pooled = np.vstack([batch.psi for batch in batches])
        self.psi_mean_ = np.mean(pooled, axis=0)
        self.psi_scale_ = np.std(pooled, axis=0)
        self.psi_scale_ = np.where(self.psi_scale_ < 1e-8, 1.0, self.psi_scale_)

        scaled = {
            batch.domain: self._scale(batch.psi)
            for batch in batches
        }
        margins = {
            batch.domain: np.asarray(
                batch.signals[:, self.boundary_signal_index], dtype=float)
            for batch in batches
        }
        self._fit_source_boundary_experts(batches, scaled, margins)
        prototypes = {}
        counts = {}
        pooled_z = np.vstack([scaled[batch.domain] for batch in batches])
        pooled_m = np.concatenate([margins[batch.domain] for batch in batches])
        reference, reference_counts = self._prototypes(pooled_z, pooled_m)
        raw_reference = reference.copy()
        for batch in batches:
            proto, count = self._prototypes(
                scaled[batch.domain], margins[batch.domain], fallback=reference)
            prototypes[batch.domain] = proto
            counts[batch.domain] = count

        before = self._prototype_discrepancy(
            prototypes, counts, reference, adapters=None)
        adapters = {
            batch.domain: np.eye(psi_dim, dtype=float)
            for batch in batches
        }
        for _ in range(self.procrustes_iterations):
            for batch in batches:
                domain = batch.domain
                adapters[domain] = self._orthogonal_fit(
                    prototypes[domain],
                    reference,
                    counts[domain],
                    ridge=self.source_adapter_ridge,
                )
            numerator = np.zeros_like(reference)
            denominator = np.zeros(self.n_bins, dtype=float)
            for batch in batches:
                domain = batch.domain
                aligned = prototypes[domain] @ adapters[domain]
                w = counts[domain]
                numerator += aligned * w[:, None]
                denominator += w
            updated = reference.copy()
            observed = denominator > 0.0
            updated[observed] = numerator[observed] / denominator[observed, None]
            if np.linalg.norm(updated - reference) <= 1e-8:
                reference = updated
                break
            reference = updated

        self.source_procrustes_adapters_ = adapters
        if self.apply_source_procrustes:
            self.source_adapters_ = adapters
            effective_reference = reference
        else:
            self.source_adapters_ = {
                batch.domain: np.eye(psi_dim, dtype=float)
                for batch in batches
            }
            effective_reference = raw_reference
        self.reference_prototypes_ = effective_reference
        self.reference_counts_ = reference_counts
        after = self._prototype_discrepancy(
            prototypes, counts, reference, adapters=adapters)

        aligned_rows = []
        expert_feature_rows = []
        margin_rows = []
        weight_rows = []
        domain_rows = []
        for batch in batches:
            aligned_rows.append(scaled[batch.domain] @ adapters[batch.domain])
            expert_feature_rows.append(self._expert_features_from_scaled(
                scaled[batch.domain],
                self.source_expert_weights_.get(
                    batch.domain, self.expert_prior_weights_),
            ))
            margin_rows.append(margins[batch.domain])
            weight_rows.append(TransferableSpectralBasis._weights(batch))
            domain_rows.extend([batch.domain] * len(batch.psi))
        Z = np.vstack(aligned_rows)
        margin = np.concatenate(margin_rows)
        weights = np.concatenate(weight_rows)
        weights = np.clip(weights, 1e-8, np.inf)
        weights /= max(float(np.sum(weights)), 1e-12)
        center = np.sum(Z * weights[:, None], axis=0)
        centered = Z - center

        total_scatter = centered.T @ (centered * weights[:, None])
        boundary_signals = self._boundary_signal_library(margin, weights)
        cross = centered.T @ (boundary_signals * weights[:, None])
        relevance_scatter = cross @ cross.T
        between_scatter = self._between_boundary_scatter(
            Z, margin, weights, center)
        domain_scatter = self._conditional_domain_scatter(
            Z, margin, weights, domain_rows, effective_reference)

        objective = (
            self.boundary_relevance * self._trace_normalize(relevance_scatter)
            + self.boundary_between * self._trace_normalize(between_scatter)
            - self.domain_penalty * self._trace_normalize(domain_scatter)
        )
        metric = self._trace_normalize(total_scatter) + self.ridge * np.eye(psi_dim)
        basis, eigenvalues = self._generalized_subspace(objective, metric)
        retained = min(self.active_dim, psi_dim, basis.shape[1])
        basis = basis[:, :retained]

        self.subspace_bases_ = []
        self.projectors_ = []
        self.boundary_axes_ = []
        margin_centered = margin - float(np.sum(weights * margin))
        margin_direction = centered.T @ (weights * margin_centered)
        for start in range(0, retained, self.subspace_dim):
            subspace = basis[:, start:min(start + self.subspace_dim, retained)]
            subspace, _ = np.linalg.qr(subspace)
            projector = subspace @ subspace.T
            axis = projector @ margin_direction
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1e-10:
                axis = subspace[:, 0].copy()
            else:
                axis /= axis_norm
            if float(axis @ margin_direction) < 0.0:
                axis = -axis
            self.subspace_bases_.append(subspace)
            self.projectors_.append(0.5 * (projector + projector.T))
            self.boundary_axes_.append(axis)

        self.feature_dim = 4 + 3 * len(self.projectors_)
        transformed = np.column_stack([
            np.vstack(expert_feature_rows),
            self._features_from_aligned(Z),
        ])
        self.source_residual_guard_ = self._source_residual_guard(
            transformed, margin, weights)
        conditional_trace = float(np.trace(domain_scatter))
        boundary_trace = float(np.trace(relevance_scatter + between_scatter))
        self.diagnostics_ = {
            "status": "fit",
            "method": "boundary_conditioned_invariant_projector_subspaces",
            "target_data_used": False,
            "source_domains": sorted(self.source_adapters_),
            "n_source_domains": int(len(batches)),
            "n_source_records": int(len(Z)),
            "psi_dim": int(psi_dim),
            "active_dim": int(retained),
            "subspace_dim": int(self.subspace_dim),
            "n_subspaces": int(len(self.projectors_)),
            "feature_dim": int(self.feature_dim),
            "expert_domains": list(self.expert_domains_),
            "expert_prior_weights": self.expert_prior_weights_.tolist(),
            "expert_cross_loss": self.expert_cross_loss_.tolist(),
            "boundary_edges": list(self.boundary_edges),
            "prototype_discrepancy_before": float(before),
            "prototype_discrepancy_after": float(after),
            "prototype_alignment_gain": float(
                (before - after) / max(before, 1e-12)),
            "source_procrustes_applied": bool(self.apply_source_procrustes),
            "boundary_scatter_trace": boundary_trace,
            "conditional_domain_scatter_trace": conditional_trace,
            "selected_eigenvalues": [float(v) for v in eigenvalues[:retained]],
            "projector_idempotence_error": float(max(
                (np.linalg.norm(P @ P - P) for P in self.projectors_),
                default=0.0,
            )),
            "source_residual_guard": float(self.source_residual_guard_),
            "fingerprint": self.fingerprint(),
        }
        return self

    def transform(self, psi, domain=None, adapter=None):
        arr = np.asarray(psi, dtype=float)
        one = arr.ndim == 1
        if one:
            arr = arr[None, :]
        scaled = self._scale(arr)
        matrix = self._adapter_matrix(domain=domain, adapter=adapter)
        aligned = scaled @ matrix
        boundary_axis = (
            adapter.boundary_axis
            if isinstance(adapter, TargetRiskAlignment)
            else None
        )
        expert_weights = (
            adapter.expert_weights
            if isinstance(adapter, TargetRiskAlignment)
            and adapter.expert_weights is not None
            else self.source_expert_weights_.get(
                str(domain), self.expert_prior_weights_)
        )
        out = np.column_stack([
            self._expert_features_from_scaled(scaled, expert_weights),
            self._features_from_aligned(
                aligned, boundary_axis=boundary_axis),
        ])
        return out[0].copy() if one else np.asarray(out, dtype=float)

    def transform_compact(self, psi, domain=None, adapter=None):
        """Four semantic risk coordinates suitable for a tiny target pilot."""
        values = np.asarray(
            self.transform(psi, domain=domain, adapter=adapter), dtype=float)
        return values[..., :4].copy()

    def aligned_coordinates(self, psi, domain=None, adapter=None):
        if self.psi_mean_ is None or self.psi_scale_ is None:
            raise RuntimeError("risk subspace alignment must be fit before use")
        arr = np.asarray(psi, dtype=float)
        one = arr.ndim == 1
        if one:
            arr = arr[None, :]
        Z = self._scale(arr)
        matrix = self._adapter_matrix(domain=domain, adapter=adapter)
        out = Z @ matrix
        return out[0].copy() if one else np.asarray(out, dtype=float)

    def transform_batches(self, batches):
        out = []
        for batch in batches:
            batch = TransferableSpectralBasis._validate_batch(batch)
            out.append(SourceDomainBatch(
                domain=batch.domain,
                psi=self.transform(batch.psi, domain=batch.domain),
                signals=batch.signals,
                sample_weight=batch.sample_weight,
                signal_weight=batch.signal_weight,
            ))
        return out

    def fit_target_adapter(self, psi, margins):
        if self.reference_prototypes_ is None:
            raise RuntimeError("risk subspace alignment must be fit before target adaptation")
        arr = np.asarray(psi, dtype=float)
        margin = np.asarray(margins, dtype=float).reshape(-1)
        if arr.ndim != 2 or len(arr) != len(margin):
            raise ValueError("target psi and margins must have matching rows")
        p = int(arr.shape[1])
        identity = np.eye(p, dtype=float)
        if len(arr) < 5 or not np.all(np.isfinite(arr)) or not np.all(np.isfinite(margin)):
            return TargetRiskAlignment(identity, False, {
                "status": "fallback_identity",
                "reason": "insufficient_or_nonfinite_target_pilot",
                "n_observations": int(len(arr)),
                "target_oracle_used": False,
            }, None, None)
        Z = self._scale(arr)
        risk_correction = np.asarray(
            self._target_risk_correction(Z), dtype=float).reshape(-1)
        bins = self._bin_index(margin)
        target = self.reference_prototypes_[bins]
        weights = 1.0 + np.exp(
            -0.5 * (margin / self.boundary_temperature) ** 2)
        candidate = self._orthogonal_fit(
            Z, target, weights, ridge=self.target_adapter_ridge)
        identity_loss = self._weighted_alignment_loss(Z, target, weights, identity)
        aligned_loss = self._weighted_alignment_loss(Z, target, weights, candidate)
        full_gain = (identity_loss - aligned_loss) / max(identity_loss, 1e-12)

        loo_gains = []
        if len(Z) >= 6:
            for heldout in range(len(Z)):
                keep = np.arange(len(Z)) != heldout
                trial = self._orthogonal_fit(
                    Z[keep], target[keep], weights[keep],
                    ridge=self.target_adapter_ridge,
                )
                base = float(np.sum((Z[heldout] - target[heldout]) ** 2))
                loss = float(np.sum((Z[heldout] @ trial - target[heldout]) ** 2))
                loo_gains.append((base - loss) / max(base, 1e-12))
        median_loo_gain = float(np.median(loo_gains)) if loo_gains else 0.0
        rotation_distance = float(
            np.linalg.norm(candidate - identity) / np.sqrt(max(p, 1)))
        unique_bins = int(len(np.unique(bins)))
        accepted = bool(
            unique_bins >= self.target_min_bins
            and full_gain >= self.target_min_gain
            and median_loo_gain >= self.target_min_gain
            and rotation_distance <= self.target_max_rotation
        )
        matrix = candidate if accepted else identity
        reasons = []
        if unique_bins < self.target_min_bins:
            reasons.append("insufficient_boundary_bins")
        if full_gain < self.target_min_gain:
            reasons.append("insufficient_alignment_gain")
        if median_loo_gain < self.target_min_gain:
            reasons.append("insufficient_loo_alignment_gain")
        if rotation_distance > self.target_max_rotation:
            reasons.append("unstable_large_rotation")
        boundary_axis, boundary_diagnostics = self._fit_target_boundary_axis(
            Z @ matrix, margin, weights)
        expert_weights, expert_diagnostics = self._fit_target_expert_mixture(
            Z, margin, weights)
        diagnostics = {
            "status": (
                "accepted"
                if accepted or boundary_axis is not None
                or expert_weights is not None
                else "fallback_identity"
            ),
            "n_observations": int(len(Z)),
            "risk_correction": {
                "method": "source_lodo_low_frequency_ensemble",
                "mean": float(np.mean(risk_correction)),
                "median": float(np.median(risk_correction)),
                "max_abs": float(np.max(np.abs(risk_correction))),
                "nominal_margin_used": True,
                "target_variance_oracle_used": False,
            },
            "n_boundary_bins": unique_bins,
            "identity_loss": float(identity_loss),
            "aligned_loss": float(aligned_loss),
            "alignment_gain": float(full_gain),
            "median_loo_gain": float(median_loo_gain),
            "rotation_distance": rotation_distance,
            "rejection_reasons": reasons,
            "boundary_axis": boundary_diagnostics,
            "expert_mixture": expert_diagnostics,
            "target_oracle_used": False,
        }
        return TargetRiskAlignment(
            matrix,
            bool(
                accepted
                or boundary_axis is not None
                or expert_weights is not None
            ),
            diagnostics,
            boundary_axis,
            expert_weights,
        )

    def diagnostics(self):
        return dict(self.diagnostics_)

    def fingerprint(self):
        digest = hashlib.sha256()
        for value in (
            self.psi_mean_,
            self.psi_scale_,
            self.reference_prototypes_,
            *[self.source_adapters_[key] for key in sorted(self.source_adapters_)],
            *self.projectors_,
            *self.boundary_axes_,
            self.expert_coefficients_,
            self.risk_correction_coefficients_,
            self.expert_prior_weights_,
        ):
            if value is not None:
                digest.update(np.asarray(value, dtype=float).tobytes())
        return digest.hexdigest()[:16]

    def _adapter_matrix(self, domain=None, adapter=None):
        p = int(len(self.psi_mean_))
        if adapter is not None:
            if isinstance(adapter, TargetRiskAlignment):
                matrix = adapter.matrix
            else:
                matrix = adapter
            matrix = np.asarray(matrix, dtype=float)
            if matrix.shape != (p, p):
                raise ValueError("risk alignment adapter has the wrong shape")
            return matrix
        if domain is not None and str(domain) in self.source_adapters_:
            return self.source_adapters_[str(domain)]
        return np.eye(p, dtype=float)

    def _scale(self, psi):
        return (np.asarray(psi, dtype=float) - self.psi_mean_) / self.psi_scale_

    def _features_from_aligned(self, aligned, boundary_axis=None):
        arr = np.asarray(aligned, dtype=float)
        one = arr.ndim == 1
        if one:
            arr = arr[None, :]
        columns = []
        for projector, axis in zip(self.projectors_, self.boundary_axes_):
            signed = arr @ axis
            projected = arr @ projector
            radius_sq = np.sum(projected * arr, axis=1)
            orth_sq = np.maximum(radius_sq - signed ** 2, 0.0)
            columns.extend([signed, np.sqrt(orth_sq), np.maximum(radius_sq, 0.0)])
        out = np.column_stack(columns) if columns else np.zeros((len(arr), 0))
        if boundary_axis is not None and out.shape[1] > 0:
            axis = np.asarray(boundary_axis, dtype=float).reshape(-1)
            if len(axis) == arr.shape[1] and np.all(np.isfinite(axis)):
                norm = float(np.linalg.norm(axis))
                if norm > 1e-12:
                    out[:, 0] = arr @ (axis / norm)
        return out[0].copy() if one else np.asarray(out, dtype=float)

    def _fit_source_boundary_experts(self, batches, scaled, margins):
        """Fit one frozen low-frequency chance-margin expert per source.

        Every source-domain feature row later uses a leave-that-domain-out
        ensemble.  Consequently the aligned coordinate cannot obtain a good
        source training score merely by replaying an expert on its own domain.
        """
        self.expert_domains_ = [str(batch.domain) for batch in batches]
        coefficients = []
        correction_coefficients = []
        for batch in batches:
            domain = str(batch.domain)
            sample_weight = TransferableSpectralBasis._weights(batch)
            coefficients.append(self._fit_boundary_expert(
                scaled[domain], margins[domain], sample_weight))
            correction = (
                np.asarray(batch.signals[:, 4], dtype=float)
                if batch.signals.shape[1] > 4
                else np.zeros(len(batch.psi), dtype=float)
            )
            correction_coefficients.append(self._fit_expert_response(
                scaled[domain], correction, sample_weight))
        self.expert_coefficients_ = np.vstack(coefficients)
        self.risk_correction_coefficients_ = np.vstack(
            correction_coefficients)

        cross_loss = np.zeros(len(batches), dtype=float)
        for expert_index in range(len(batches)):
            losses = []
            for target_index, batch in enumerate(batches):
                if target_index == expert_index and len(batches) > 1:
                    continue
                domain = str(batch.domain)
                prediction = self._expert_predictions(
                    scaled[domain])[:, expert_index]
                response = np.tanh(
                    margins[domain] / self.boundary_temperature)
                sample_weight = TransferableSpectralBasis._weights(batch)
                sample_weight = np.clip(sample_weight, 1e-8, np.inf)
                losses.append(float(np.average(
                    (response - prediction) ** 2,
                    weights=sample_weight,
                )))
            cross_loss[expert_index] = (
                float(np.mean(losses)) if losses else 1.0)
        self.expert_cross_loss_ = cross_loss
        temperature = max(float(np.median(cross_loss)), 0.05)
        logits = -(cross_loss - float(np.min(cross_loss))) / temperature
        raw = np.exp(np.clip(logits, -30.0, 0.0))
        raw += 0.10
        self.expert_prior_weights_ = raw / float(np.sum(raw))
        self.source_expert_weights_ = {}
        for index, domain in enumerate(self.expert_domains_):
            leave_one_out = self.expert_prior_weights_.copy()
            if len(leave_one_out) > 1:
                leave_one_out[index] = 0.0
                total = float(np.sum(leave_one_out))
                leave_one_out = (
                    leave_one_out / total
                    if total > 1e-12
                    else np.full(
                        len(leave_one_out),
                        1.0 / len(leave_one_out),
                    )
                )
            self.source_expert_weights_[domain] = leave_one_out

    def _fit_boundary_expert(self, Z, margin, weights):
        Z = np.asarray(Z, dtype=float)
        margin = np.asarray(margin, dtype=float).reshape(-1)
        weights = np.clip(
            np.asarray(weights, dtype=float).reshape(-1), 1e-8, np.inf)
        response = np.tanh(margin / self.boundary_temperature)
        return self._fit_expert_response(Z, response, weights)

    def _fit_expert_response(self, Z, response, weights):
        Z = np.asarray(Z, dtype=float)
        response = np.asarray(response, dtype=float).reshape(-1)
        weights = np.clip(
            np.asarray(weights, dtype=float).reshape(-1), 1e-8, np.inf)
        design = self._expert_design(Z)
        gram = design.T @ (design * weights[:, None])
        penalty = max(self.target_adapter_ridge, 0.5) * np.eye(
            design.shape[1], dtype=float)
        penalty[0, 0] = self.ridge
        rhs = design.T @ (weights * response)
        try:
            return np.linalg.solve(gram + penalty, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]

    @staticmethod
    def _expert_design(scaled):
        arr = np.asarray(scaled, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        smooth = np.tanh(0.5 * arr)
        return np.column_stack([
            np.ones(len(arr), dtype=float),
            smooth,
            smooth ** 2,
            np.sin(np.pi * smooth),
        ])

    def _expert_predictions(self, scaled):
        arr = np.asarray(scaled, dtype=float)
        one = arr.ndim == 1
        if one:
            arr = arr[None, :]
        if self.expert_coefficients_ is None:
            out = np.zeros((len(arr), 0), dtype=float)
        else:
            design = self._expert_design(arr)
            out = np.clip(
                design @ self.expert_coefficients_.T,
                -2.0,
                2.0,
            )
        return out[0].copy() if one else out

    def _risk_correction_predictions(self, scaled):
        arr = np.asarray(scaled, dtype=float)
        one = arr.ndim == 1
        if one:
            arr = arr[None, :]
        if self.risk_correction_coefficients_ is None:
            out = np.zeros((len(arr), len(self.expert_domains_)), dtype=float)
        else:
            out = np.clip(
                self._expert_design(arr)
                @ self.risk_correction_coefficients_.T,
                -2.0,
                2.0,
            )
        return out[0].copy() if one else out

    def _target_risk_correction(self, scaled):
        predictions = np.asarray(
            self._risk_correction_predictions(scaled), dtype=float)
        one = predictions.ndim == 1
        if one:
            predictions = predictions[None, :]
        if predictions.shape[1] == 0:
            out = np.zeros(len(predictions), dtype=float)
        else:
            out = predictions @ self.expert_prior_weights_
        return float(out[0]) if one else out

    def _expert_features_from_scaled(self, scaled, weights):
        predictions = np.asarray(
            self._expert_predictions(scaled), dtype=float)
        one = predictions.ndim == 1
        if one:
            predictions = predictions[None, :]
        if predictions.shape[1] == 0:
            out = np.zeros((len(predictions), 4), dtype=float)
        else:
            weight = np.asarray(weights, dtype=float).reshape(-1)
            if len(weight) != predictions.shape[1] or not np.all(np.isfinite(weight)):
                weight = np.full(
                    predictions.shape[1], 1.0 / predictions.shape[1])
            weight = np.maximum(weight, 0.0)
            weight /= max(float(np.sum(weight)), 1e-12)
            score = predictions @ weight
            disagreement = np.sqrt(np.maximum(
                np.sum(
                    (predictions - score[:, None]) ** 2 * weight[None, :],
                    axis=1,
                ),
                0.0,
            ))
            correction_predictions = np.asarray(
                self._risk_correction_predictions(scaled), dtype=float)
            if correction_predictions.ndim == 1:
                correction_predictions = correction_predictions[None, :]
            correction = correction_predictions @ weight
            out = np.column_stack([
                score,
                np.abs(score),
                disagreement,
                correction,
            ])
        return out[0].copy() if one else out

    @staticmethod
    def _project_simplex(values):
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) == 1:
            return np.ones(1, dtype=float)
        order = np.sort(values)[::-1]
        cumulative = np.cumsum(order) - 1.0
        indices = np.arange(1, len(values) + 1, dtype=float)
        support = np.where(order - cumulative / indices > 0.0)[0]
        rho = int(support[-1]) if len(support) else 0
        threshold = cumulative[rho] / float(rho + 1)
        projected = np.maximum(values - threshold, 0.0)
        return projected / max(float(np.sum(projected)), 1e-12)

    def _fit_target_expert_mixture(self, Z, margin, weights):
        expert = np.asarray(self._expert_predictions(Z), dtype=float)
        if expert.ndim != 2 or expert.shape[1] < 2:
            return None, {
                "status": "fallback_source_ensemble",
                "reason": "insufficient_source_experts",
            }
        response = np.tanh(
            np.asarray(margin, dtype=float) / self.boundary_temperature)
        weights = np.clip(np.asarray(weights, dtype=float), 1e-8, np.inf)
        prior = np.asarray(self.expert_prior_weights_, dtype=float)
        n_experts = int(expert.shape[1])

        def fit_mixture(X, y, w):
            gram = X.T @ (X * w[:, None])
            rhs = X.T @ (w * y)
            for left in range(len(X)):
                for right in range(left + 1, len(X)):
                    pair_weight = float(np.sqrt(w[left] * w[right]))
                    difference = X[left] - X[right]
                    target_difference = float(y[left] - y[right])
                    gram += 0.5 * pair_weight * np.outer(
                        difference, difference)
                    rhs += 0.5 * pair_weight * difference * target_difference
            ridge = max(self.target_adapter_ridge, self.ridge)
            try:
                raw = np.linalg.solve(
                    gram + ridge * np.eye(n_experts),
                    rhs + ridge * prior,
                )
            except np.linalg.LinAlgError:
                raw = np.linalg.lstsq(
                    gram + ridge * np.eye(n_experts),
                    rhs + ridge * prior,
                    rcond=None,
                )[0]
            return self._project_simplex(raw)

        def calibrate(score, y, w):
            score = np.asarray(score, dtype=float)
            total = max(float(np.sum(w)), 1e-12)
            score_mean = float(np.sum(score * w) / total)
            y_mean = float(np.sum(y * w) / total)
            centered = score - score_mean
            slope = float(np.sum(w * centered * (y - y_mean))) / max(
                float(np.sum(w * centered ** 2)) + self.ridge,
                self.ridge,
            )
            return slope, score_mean, y_mean

        full = fit_mixture(expert, response, weights)
        candidate_errors = []
        baseline_errors = []
        candidate_predictions = []
        baseline_predictions = []
        stability = []
        for heldout in range(len(expert)):
            keep = np.arange(len(expert)) != heldout
            trial = fit_mixture(
                expert[keep], response[keep], weights[keep])
            stability.append(float(trial @ full) / max(
                float(np.linalg.norm(trial) * np.linalg.norm(full)), 1e-12))
            trial_score = expert[keep] @ trial
            slope, center, y_center = calibrate(
                trial_score, response[keep], weights[keep])
            candidate = float(
                y_center + slope * (expert[heldout] @ trial - center))
            prior_score = expert[keep] @ prior
            base_slope, base_center, base_y_center = calibrate(
                prior_score, response[keep], weights[keep])
            baseline = float(
                base_y_center
                + base_slope * (expert[heldout] @ prior - base_center))
            truth = float(response[heldout])
            candidate_predictions.append(candidate)
            baseline_predictions.append(baseline)
            candidate_errors.append((truth - candidate) ** 2)
            baseline_errors.append((truth - baseline) ** 2)
        candidate_error = float(np.mean(candidate_errors))
        baseline_error = float(np.mean(baseline_errors))
        gain = (baseline_error - candidate_error) / max(baseline_error, 1e-8)
        infeasible = response > 0.0
        candidate_false_feasible = (
            float(np.mean(np.asarray(candidate_predictions)[infeasible] <= 0.0))
            if np.any(infeasible) else 0.0)
        baseline_false_feasible = (
            float(np.mean(np.asarray(baseline_predictions)[infeasible] <= 0.0))
            if np.any(infeasible) else 0.0)
        median_stability = float(np.median(stability)) if stability else 0.0
        accepted = bool(
            gain >= self.target_min_gain
            and median_stability >= 0.60
            and candidate_false_feasible <= baseline_false_feasible + 0.05
        )
        return (full if accepted else None), {
            "status": "accepted" if accepted else "fallback_source_ensemble",
            "loo_prediction_gain": float(gain),
            "loo_source_ensemble_error": baseline_error,
            "loo_adapted_ensemble_error": candidate_error,
            "median_weight_cosine": median_stability,
            "source_false_feasible_rate": baseline_false_feasible,
            "adapted_false_feasible_rate": candidate_false_feasible,
            "effective_unknowns": int(max(n_experts - 1, 1)),
            "prior_weights": prior.tolist(),
            "posterior_weights": full.tolist(),
            "target_oracle_used": False,
        }

    def _fit_target_boundary_axis(self, Z, margin, weights):
        Z = np.asarray(Z, dtype=float)
        margin = np.asarray(margin, dtype=float).reshape(-1)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        p = int(Z.shape[1])
        retained = np.column_stack(self.subspace_bases_)
        retained, _ = np.linalg.qr(retained)
        q = int(retained.shape[1])
        latent = Z @ retained
        projector = retained @ retained.T
        prior_axis = np.sum(np.vstack(self.boundary_axes_), axis=0)
        prior_norm = float(np.linalg.norm(prior_axis))
        if prior_norm <= 1e-12:
            prior_axis = self.boundary_axes_[0].copy()
        else:
            prior_axis /= prior_norm
        prior_latent = retained.T @ prior_axis
        prior_latent_norm = float(np.linalg.norm(prior_latent))
        if prior_latent_norm <= 1e-12:
            prior_latent = np.zeros(q, dtype=float)
            prior_latent[0] = 1.0
        else:
            prior_latent /= prior_latent_norm

        robust_scale = max(
            0.7413 * float(np.quantile(margin, 0.75) - np.quantile(margin, 0.25)),
            0.5,
        )
        response = np.tanh(margin / robust_scale)

        def fit_axis(X, y, w):
            total = max(float(np.sum(w)), 1e-12)
            x_mean = np.sum(X * w[:, None], axis=0) / total
            y_mean = float(np.sum(y * w) / total)
            centered = X - x_mean
            centered_response = y - y_mean
            gram = centered.T @ (centered * w[:, None])
            rhs = centered.T @ (w * centered_response)
            for left in range(len(X)):
                for right in range(left + 1, len(X)):
                    pair_weight = float(np.sqrt(w[left] * w[right]))
                    difference = X[left] - X[right]
                    target_difference = float(y[left] - y[right])
                    gram += 0.5 * pair_weight * np.outer(difference, difference)
                    rhs += 0.5 * pair_weight * difference * target_difference
            penalty = max(self.target_adapter_ridge, self.ridge)
            gram += penalty * np.eye(q, dtype=float)
            rhs += penalty * prior_latent
            try:
                coefficient = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                coefficient = np.linalg.lstsq(gram, rhs, rcond=None)[0]
            return coefficient, x_mean, y_mean

        def fit_source_calibration(X, y, w):
            score = X @ prior_latent
            total = max(float(np.sum(w)), 1e-12)
            score_mean = float(np.sum(score * w) / total)
            y_mean = float(np.sum(y * w) / total)
            centered_score = score - score_mean
            numerator = float(np.sum(w * centered_score * (y - y_mean)))
            denominator = float(np.sum(w * centered_score ** 2)) + self.ridge
            slope = numerator / max(denominator, self.ridge)
            return slope, score_mean, y_mean

        coefficient, _, _ = fit_axis(latent, response, weights)
        coefficient_norm = float(np.linalg.norm(coefficient))
        if coefficient_norm <= 1e-10:
            return None, {
                "status": "fallback_source_axis",
                "reason": "degenerate_target_direction",
            }
        axis = retained @ (coefficient / coefficient_norm)
        if float(axis @ prior_axis) < 0.0:
            axis = -axis

        baseline_errors = []
        candidate_errors = []
        direction_stability = []
        for heldout in range(len(latent)):
            keep = np.arange(len(latent)) != heldout
            trial, trial_mean, trial_y_mean = fit_axis(
                latent[keep], response[keep], weights[keep])
            trial_norm = float(np.linalg.norm(trial))
            if trial_norm <= 1e-10:
                continue
            trial_axis = retained @ (trial / trial_norm)
            direction_stability.append(abs(float(trial_axis @ axis)))
            prediction = float(
                trial_y_mean + (latent[heldout] - trial_mean) @ trial)
            slope, source_mean, source_y_mean = fit_source_calibration(
                latent[keep], response[keep], weights[keep])
            source_prediction = float(
                source_y_mean
                + slope * (latent[heldout] @ prior_latent - source_mean)
            )
            candidate_errors.append(
                (float(response[heldout]) - prediction) ** 2)
            baseline_errors.append(
                (float(response[heldout]) - source_prediction) ** 2)
        candidate_error = (
            float(np.mean(candidate_errors)) if candidate_errors else np.inf)
        baseline_error = (
            float(np.mean(baseline_errors)) if baseline_errors else np.inf)
        prediction_gain = (
            (baseline_error - candidate_error) / max(baseline_error, 1e-8)
            if np.isfinite(baseline_error) and np.isfinite(candidate_error)
            else -np.inf
        )
        median_stability = (
            float(np.median(direction_stability))
            if direction_stability else 0.0)
        accepted = bool(
            prediction_gain >= self.target_min_gain
            and median_stability >= 0.60
        )
        return (axis if accepted else None), {
            "status": "accepted" if accepted else "fallback_source_axis",
            "loo_prediction_gain": float(prediction_gain),
            "loo_source_axis_error": float(baseline_error),
            "loo_candidate_axis_error": float(candidate_error),
            "median_direction_cosine": median_stability,
            "outside_subspace_fraction": float(
                np.linalg.norm((np.eye(p) - projector) @ axis) ** 2),
            "effective_unknowns": int(q),
            "response": "tanh_scaled_margin_plus_pairwise_order",
            "response_scale": float(robust_scale),
            "target_oracle_used": False,
        }

    def _bin_index(self, margins):
        return np.digitize(
            np.asarray(margins, dtype=float), self.boundary_edges, right=False)

    def _prototypes(self, Z, margins, fallback=None):
        Z = np.asarray(Z, dtype=float)
        bins = self._bin_index(margins)
        if fallback is None:
            fallback = np.repeat(
                np.mean(Z, axis=0, keepdims=True), self.n_bins, axis=0)
        proto = np.asarray(fallback, dtype=float).copy()
        counts = np.zeros(self.n_bins, dtype=float)
        for index in range(self.n_bins):
            mask = bins == index
            if np.any(mask):
                proto[index] = np.mean(Z[mask], axis=0)
                counts[index] = float(np.sum(mask))
        return proto, counts

    @staticmethod
    def _orthogonal_fit(source, target, weights, ridge=0.0):
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        p = int(source.shape[1])
        cross = source.T @ (target * weights[:, None])
        cross += max(float(ridge), 0.0) * np.eye(p)
        try:
            left, _, right_t = np.linalg.svd(cross, full_matrices=False)
            return left @ right_t
        except np.linalg.LinAlgError:
            return np.eye(p, dtype=float)

    @staticmethod
    def _weighted_alignment_loss(source, target, weights, matrix):
        residual = np.asarray(source) @ np.asarray(matrix) - np.asarray(target)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        return float(np.average(np.sum(residual ** 2, axis=1), weights=weights))

    @staticmethod
    def _prototype_discrepancy(prototypes, counts, reference, adapters=None):
        numerator = 0.0
        denominator = 0.0
        for domain, proto in prototypes.items():
            matrix = (
                adapters[domain] if adapters is not None
                else np.eye(proto.shape[1], dtype=float)
            )
            residual = proto @ matrix - reference
            weight = counts[domain]
            numerator += float(np.sum(weight * np.sum(residual ** 2, axis=1)))
            denominator += float(np.sum(weight))
        return numerator / max(denominator, 1e-12)

    def _boundary_signal_library(self, margin, weights):
        temp = self.boundary_temperature
        raw = np.column_stack([
            margin,
            np.tanh(margin / temp),
            np.exp(-0.5 * (margin / temp) ** 2),
            np.maximum(margin, 0.0),
        ])
        mean = np.sum(raw * weights[:, None], axis=0)
        centered = raw - mean
        scale = np.sqrt(np.sum(centered ** 2 * weights[:, None], axis=0))
        scale = np.where(scale < 1e-8, 1.0, scale)
        return centered / scale

    def _between_boundary_scatter(self, Z, margin, weights, center):
        out = np.zeros((Z.shape[1], Z.shape[1]), dtype=float)
        bins = self._bin_index(margin)
        for index in range(self.n_bins):
            mask = bins == index
            if not np.any(mask):
                continue
            total = float(np.sum(weights[mask]))
            mean = np.sum(Z[mask] * weights[mask, None], axis=0) / max(total, 1e-12)
            delta = mean - center
            out += total * np.outer(delta, delta)
        return out

    def _conditional_domain_scatter(self, Z, margin, weights, domains, reference):
        out = np.zeros((Z.shape[1], Z.shape[1]), dtype=float)
        bins = self._bin_index(margin)
        domains = np.asarray(domains, dtype=object)
        for domain in sorted(set(domains.tolist())):
            domain_mask = domains == domain
            for index in range(self.n_bins):
                mask = domain_mask & (bins == index)
                if not np.any(mask):
                    continue
                total = float(np.sum(weights[mask]))
                mean = np.sum(Z[mask] * weights[mask, None], axis=0) / max(total, 1e-12)
                delta = mean - reference[index]
                out += total * np.outer(delta, delta)
        return out

    @staticmethod
    def _trace_normalize(matrix):
        matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
        trace = float(np.trace(matrix))
        return matrix / max(abs(trace), 1e-12)

    @staticmethod
    def _generalized_subspace(objective, metric):
        metric = 0.5 * (metric + metric.T)
        values, vectors = np.linalg.eigh(metric)
        values = np.maximum(values, 1e-12)
        inverse_sqrt = (vectors * (1.0 / np.sqrt(values))[None, :]) @ vectors.T
        reduced = inverse_sqrt @ (0.5 * (objective + objective.T)) @ inverse_sqrt
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (reduced + reduced.T))
        order = np.argsort(-eigenvalues, kind="stable")
        basis = inverse_sqrt @ eigenvectors[:, order]
        basis, _ = np.linalg.qr(basis)
        return basis, eigenvalues[order]

    def _source_residual_guard(self, features, margin, weights):
        X = np.column_stack([np.ones(len(features)), features])
        reg = self.ridge * np.eye(X.shape[1], dtype=float)
        reg[0, 0] = 0.0
        gram = X.T @ (X * weights[:, None]) + reg
        rhs = X.T @ (weights * margin)
        try:
            beta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        residual = np.abs(margin - X @ beta)
        boundary = 1.0 + np.exp(
            -0.5 * (margin / self.boundary_temperature) ** 2)
        return self._weighted_quantile(residual, weights * boundary, 0.90)

    @staticmethod
    def _weighted_quantile(values, weights, quantile):
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        order = np.argsort(values, kind="stable")
        values = values[order]
        weights = np.maximum(weights[order], 0.0)
        cumulative = np.cumsum(weights)
        if len(values) == 0 or cumulative[-1] <= 0.0:
            return 0.0
        index = int(np.searchsorted(
            cumulative, float(quantile) * cumulative[-1], side="left"))
        return float(values[min(index, len(values) - 1)])
