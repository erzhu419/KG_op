"""Learned admissible meta-prior for LODO SC-OLH-KG experiments.

This module intentionally uses a small dependency-free model.  The goal is to
replace target-specific anchors/refinement/risk coordinates with a frozen
source-trained structural prior:

* a dimension-invariant policy descriptor,
* whitened continuous local exposures A,
* soft shared-shock regime exposures N,
* a source-fitted cumulative-HVD beta prior,
* and a meta-anchor proposal distribution in psi=(A,N) space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from core.admissibility import (
    domain_tuned_audit,
    lodo_meta_prior_audit,
    strict_universal_audit,
)
from core.candidates import unique_candidates
from core.cumulative_risk import (
    RiskExposure,
    cumulative_feature_names,
    cumulative_feature_vector,
)
from representation.transferable_spectral import (
    SourceDomainBatch,
    TransferableSpectralBasis,
)


HIDDEN_TARGET_STRUCTURAL_METHODS = {
    "all_axis_solutions",
    "cumulative_risk_features",
    "cumulative_risk_feature_names",
    "cumulative_risk_parameters",
    "cumulative_risk_provider_status",
    "gpr_basis_map",
    "hvd_features",
    "initial_samples",
    "inverse_state_anchor",
    "recommendation_random_pool_size",
    "recommendation_refinement_candidates",
    "risk_class",
    "risk_exposures",
    "state_anchor_points",
    "structured_candidates",
    "surrogate_basis_map",
    "true_cumulative_risk_decomposition",
}


def _as_tuple(x):
    return tuple(int(v) for v in x)


def _softmax_negdist(d2, temperature):
    d2 = np.asarray(d2, dtype=float)
    temp = max(float(temperature), 1e-8)
    logits = -d2 / temp
    logits -= float(np.max(logits))
    w = np.exp(logits)
    total = float(np.sum(w))
    if total <= 1e-12:
        return np.full(len(d2), 1.0 / max(len(d2), 1), dtype=float)
    return w / total


def _project_psd_features(beta, n_local, n_shared):
    """Project cumulative beta coefficients onto the admissible nonnegative cone."""

    beta = np.asarray(beta, dtype=float).reshape(-1).copy()
    expected = 1 + n_local + n_shared * (n_shared + 1) // 2 + n_shared
    if len(beta) != expected:
        return np.maximum(beta, 0.0)
    beta[0] = max(float(beta[0]), 1e-10)
    beta[1:1 + n_local] = np.maximum(beta[1:1 + n_local], 0.0)
    start = 1 + n_local
    end = start + n_shared * (n_shared + 1) // 2
    B = np.zeros((n_shared, n_shared), dtype=float)
    pos = start
    for i in range(n_shared):
        for j in range(i, n_shared):
            B[i, j] = B[j, i] = float(beta[pos])
            pos += 1
    try:
        vals, vecs = np.linalg.eigh(0.5 * (B + B.T))
        vals = np.maximum(vals, 0.0)
        B = (vecs * vals) @ vecs.T
    except np.linalg.LinAlgError:
        B = np.maximum(B, 0.0)
    pos = start
    for i in range(n_shared):
        for j in range(i, n_shared):
            beta[pos] = float(B[i, j])
            pos += 1
    beta[end:] = np.maximum(beta[end:], 0.0)
    return beta


@dataclass
class SourceRecord:
    domain: str
    x: tuple[int, ...]
    y: np.ndarray
    descriptor: np.ndarray
    profile: np.ndarray | None
    tau: float
    alpha: float
    sigma_level: float
    constraint_sigma: float | None = None
    origin: str = "random"
    sample_weight: float = 1.0


class LearnedMetaPrior:
    """Frozen source-trained prior used by held-out target adapters."""

    VALID_COMPONENT_STAGES = {"legacy_all", "coordinate", "spectral"}

    def __init__(
        self,
        local_dim=3,
        shared_dim=3,
        anchor_count=24,
        kmeans_iters=25,
        soft_temperature=0.75,
        ridge=1e-4,
        boundary_weight=1.0,
        boundary_temperature=1.0,
        variance_weight=0.5,
        feasible_penalty=6.0,
        feasible_bonus=0.15,
        elite_fraction=0.40,
        boundary_fraction=0.35,
        teacher_records_per_domain=0,
        teacher_weight=3.0,
        teacher_pool_size=2048,
        teacher_elite_fraction=0.50,
        teacher_boundary_fraction=0.35,
        anchor_sampling_temperature=0.0,
        hvd_noise_floor_scale=0.0,
        universal_shape_count=0,
        component_stage="legacy_all",
        spectral_active_dim=6,
        spectral_max_library_size=64,
        spectral_low_frequency_components=8,
        spectral_graph_neighbors=10,
        spectral_relevance_floor=0.05,
        spectral_gate_boundary_weight=2.0,
        spectral_gate_dangerous_weight=3.0,
        spectral_gate_selection_tolerance=0.02,
        spectral_gate_calibration_quantile=0.90,
        coordinate_mode="pca",
        coordinate_relevance_floor=0.05,
        seed=123,
    ):
        self.local_dim = int(local_dim)
        self.shared_dim = int(shared_dim)
        self.anchor_count = int(anchor_count)
        self.kmeans_iters = int(kmeans_iters)
        self.soft_temperature = float(soft_temperature)
        self.ridge = float(ridge)
        self.boundary_weight = float(boundary_weight)
        self.boundary_temperature = float(boundary_temperature)
        self.variance_weight = float(variance_weight)
        self.feasible_penalty = float(feasible_penalty)
        self.feasible_bonus = float(feasible_bonus)
        self.elite_fraction = float(elite_fraction)
        self.boundary_fraction = float(boundary_fraction)
        self.teacher_records_per_domain = int(teacher_records_per_domain)
        self.teacher_weight = float(teacher_weight)
        self.teacher_pool_size = int(teacher_pool_size)
        self.teacher_elite_fraction = float(teacher_elite_fraction)
        self.teacher_boundary_fraction = float(teacher_boundary_fraction)
        self.anchor_sampling_temperature = float(anchor_sampling_temperature)
        self.hvd_noise_floor_scale = float(hvd_noise_floor_scale)
        self.universal_shape_count = int(universal_shape_count)
        self.component_stage = str(component_stage)
        if self.component_stage not in self.VALID_COMPONENT_STAGES:
            raise ValueError(
                f"component_stage must be one of {sorted(self.VALID_COMPONENT_STAGES)}"
            )
        self.spectral_active_dim = int(spectral_active_dim)
        self.spectral_max_library_size = int(spectral_max_library_size)
        self.spectral_low_frequency_components = int(
            spectral_low_frequency_components)
        self.spectral_graph_neighbors = int(spectral_graph_neighbors)
        self.spectral_relevance_floor = float(spectral_relevance_floor)
        self.spectral_gate_boundary_weight = float(spectral_gate_boundary_weight)
        self.spectral_gate_dangerous_weight = float(spectral_gate_dangerous_weight)
        self.spectral_gate_selection_tolerance = float(
            spectral_gate_selection_tolerance)
        self.spectral_gate_calibration_quantile = float(
            spectral_gate_calibration_quantile)
        self.coordinate_mode = str(coordinate_mode)
        if self.coordinate_mode not in {"pca", "stable_supervised"}:
            raise ValueError("coordinate_mode must be 'pca' or 'stable_supervised'")
        self.coordinate_relevance_floor = float(coordinate_relevance_floor)
        self.seed = int(seed)
        self.feature_mean = None
        self.feature_scale = None
        self.pca_components = None
        self.cluster_centers = None
        self.anchor_psi = np.empty((0, self.local_dim + self.shared_dim))
        self.anchor_scores = np.empty(0, dtype=float)
        self.anchor_meta = []
        self.profile_templates = []
        self.beta_prior = {}
        self.mean_prior = {}
        self.mean_prior_sigma = {}
        self.spectral_basis: TransferableSpectralBasis | None = None
        self.source_domains = []
        self.n_records = 0
        self.fit_status = "unfit"
        self.training_diagnostics = {}
        self.coordinate_diagnostics = {"mode": self.coordinate_mode}

    def component_enabled(self, name):
        name = str(name)
        if name == "coordinate":
            return True
        if self.component_stage == "legacy_all":
            return name in {"hvd", "mean", "proposal"}
        if self.component_stage == "spectral":
            return name == "spectral"
        return False

    @staticmethod
    def descriptor(problem, x):
        """Dimension-invariant policy descriptor from observable bounds only."""

        z = np.asarray(problem.normalize(x), dtype=float).reshape(-1)
        if len(z) == 0:
            z = np.zeros(1, dtype=float)
        qs = np.quantile(z, [0.10, 0.25, 0.50, 0.75, 0.90])
        center_norm = float(np.linalg.norm(z - 0.5) / np.sqrt(max(len(z), 1)))
        diffs = np.diff(z) if len(z) > 1 else np.array([0.0])
        segs = np.array_split(z, 4)
        seg_stats = []
        for seg in segs:
            if len(seg) == 0:
                seg_stats.extend([0.0, 0.0])
            else:
                seg_stats.extend([float(np.mean(seg)), float(np.std(seg))])
        hist, _ = np.histogram(z, bins=np.linspace(0.0, 1.0, 6))
        hist = hist.astype(float) / max(float(len(z)), 1.0)
        out = np.concatenate([
            np.array([
                float(np.mean(z)),
                float(np.std(z)),
                float(np.min(z)),
                float(np.max(z)),
                center_norm,
                float(np.mean(np.abs(diffs))),
                float(z[0]),
                float(z[-1]),
                float(np.mean(z[1:])) if len(z) > 1 else float(z[0]),
                float(np.std(z[1:])) if len(z) > 1 else 0.0,
            ]),
            qs,
            np.asarray(seg_stats, dtype=float),
            hist,
        ])
        return np.asarray(out, dtype=float)

    def _scaled_descriptor(self, descriptor):
        desc = np.asarray(descriptor, dtype=float)
        return (desc - self.feature_mean) / self.feature_scale

    def _fit_scaler_pca(self, descriptors, records=None):
        X = np.vstack(descriptors)
        self.feature_mean = np.mean(X, axis=0)
        self.feature_scale = np.std(X, axis=0)
        self.feature_scale = np.where(self.feature_scale < 1e-8, 1.0, self.feature_scale)
        Z = (X - self.feature_mean) / self.feature_scale
        if self.coordinate_mode == "stable_supervised":
            if records is None or len(records) != len(Z):
                raise ValueError(
                    "stable_supervised coordinates require aligned source records")
            return self._fit_stable_descriptor_projection(Z, records)
        try:
            _, _, vt = np.linalg.svd(Z, full_matrices=False)
        except np.linalg.LinAlgError:
            vt = np.eye(Z.shape[1], dtype=float)
        k = min(self.local_dim, vt.shape[0])
        comp = np.zeros((self.local_dim, Z.shape[1]), dtype=float)
        comp[:k, :] = vt[:k, :]
        if k < self.local_dim:
            comp[k:, :self.local_dim - k] = np.eye(self.local_dim - k)
        self.pca_components = comp
        A = Z @ self.pca_components.T
        a_scale = np.std(A, axis=0)
        a_scale = np.where(a_scale < 1e-8, 1.0, a_scale)
        self.pca_components = self.pca_components / a_scale[:, None]
        self.coordinate_diagnostics = {
            "mode": "pca",
            "selected_names": [],
        }
        return Z

    @staticmethod
    def _descriptor_names():
        return [
            "mean", "std", "min", "max", "center_norm", "mean_abs_diff",
            "first", "last", "tail_mean", "tail_std",
            "q10", "q25", "q50", "q75", "q90",
            "segment0_mean", "segment0_std", "segment1_mean", "segment1_std",
            "segment2_mean", "segment2_std", "segment3_mean", "segment3_std",
            "hist0", "hist1", "hist2", "hist3", "hist4",
        ]

    def _fit_stable_descriptor_projection(self, Z, records):
        by_domain = {}
        for index, rec in enumerate(records):
            by_domain.setdefault(rec.domain, []).append(index)
        relevance_rows = []
        sign_rows = []
        for indices in by_domain.values():
            idx = np.asarray(indices, dtype=int)
            Xd = Z[idx]
            signals = np.column_stack([
                np.asarray([float(records[i].y[0]) for i in idx], dtype=float),
                np.asarray([
                    (float(records[i].y[1]) - float(records[i].tau))
                    / self._source_margin_scale(records[i])
                    for i in idx
                ], dtype=float),
            ])
            Xd = Xd - np.mean(Xd, axis=0, keepdims=True)
            x_scale = np.sqrt(np.mean(Xd ** 2, axis=0))
            x_scale = np.where(x_scale < 1e-10, 1.0, x_scale)
            Xd = Xd / x_scale
            signals = signals - np.mean(signals, axis=0, keepdims=True)
            y_scale = np.sqrt(np.mean(signals ** 2, axis=0))
            y_scale = np.where(y_scale < 1e-10, 1.0, y_scale)
            signals = signals / y_scale
            corr = Xd.T @ signals / max(float(len(Xd)), 1.0)
            relevance_rows.append(np.max(np.abs(corr), axis=1))
            sign_rows.append(np.sign(corr))
        relevance = np.vstack(relevance_rows)
        signs = np.stack(sign_rows, axis=0)
        relevance_mean = np.mean(relevance, axis=0)
        relevance_std = np.std(relevance, axis=0)
        stability = relevance_mean / (relevance_mean + relevance_std + 1e-12)
        prevalence = np.mean(relevance >= self.coordinate_relevance_floor, axis=0)
        sign_consistency = np.empty(Z.shape[1], dtype=float)
        for feature in range(Z.shape[1]):
            agreements = []
            for signal in range(signs.shape[2]):
                values = signs[:, feature, signal]
                values = values[values != 0.0]
                positive = float(np.mean(values > 0.0)) if len(values) else 0.5
                agreements.append(max(positive, 1.0 - positive))
            sign_consistency[feature] = max(agreements)
        score = relevance_mean * (
            0.5 + 0.5 * stability
        ) * (
            0.5 + 0.5 * prevalence
        ) * (
            0.75 + 0.25 * sign_consistency
        )
        order = np.argsort(-score, kind="stable")
        selected = []
        corr_z = np.corrcoef(Z, rowvar=False)
        corr_z = np.nan_to_num(corr_z, nan=0.0)
        for feature in order:
            if any(abs(float(corr_z[int(feature), old])) > 0.97 for old in selected):
                continue
            selected.append(int(feature))
            if len(selected) >= self.local_dim:
                break
        for feature in order:
            if len(selected) >= self.local_dim:
                break
            if int(feature) not in selected:
                selected.append(int(feature))
        components = np.zeros((self.local_dim, Z.shape[1]), dtype=float)
        for row, feature in enumerate(selected[: self.local_dim]):
            components[row, feature] = 1.0
        self.pca_components = components
        names = self._descriptor_names()
        self.coordinate_diagnostics = {
            "mode": "stable_supervised",
            "selected_indices": selected[: self.local_dim],
            "selected_names": [
                names[index] if index < len(names) else f"descriptor{index}"
                for index in selected[: self.local_dim]
            ],
            "selected_scores": [float(score[index]) for index in selected[: self.local_dim]],
            "selected_prevalence": [
                float(prevalence[index]) for index in selected[: self.local_dim]
            ],
            "selected_sign_consistency": [
                float(sign_consistency[index]) for index in selected[: self.local_dim]
            ],
        }
        return Z

    def _fit_kmeans(self, Z):
        rng = np.random.default_rng(self.seed + 17)
        n = len(Z)
        k = max(1, min(self.shared_dim, n))
        if n == 0:
            self.cluster_centers = np.zeros((self.shared_dim, self.local_dim), dtype=float)
            return
        # Cluster in the local-exposure coordinates so A and N share a geometry
        # without using target-specific labels.
        A = Z @ self.pca_components.T
        first = int(rng.integers(0, n))
        centers = [A[first]]
        while len(centers) < k:
            C = np.vstack(centers)
            d2 = np.min(np.sum((A[:, None, :] - C[None, :, :]) ** 2, axis=2), axis=1)
            if float(np.sum(d2)) <= 1e-12:
                centers.append(A[int(rng.integers(0, n))])
            else:
                probs = d2 / float(np.sum(d2))
                centers.append(A[int(rng.choice(n, p=probs))])
        centers = np.vstack(centers)
        for _ in range(max(1, self.kmeans_iters)):
            d2 = np.sum((A[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(d2, axis=1)
            new_centers = centers.copy()
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    new_centers[j] = np.mean(A[mask], axis=0)
            if np.linalg.norm(new_centers - centers) <= 1e-8:
                break
            centers = new_centers
        if k < self.shared_dim:
            pad = np.repeat(centers[-1:], self.shared_dim - k, axis=0)
            centers = np.vstack([centers, pad])
        self.cluster_centers = centers[: self.shared_dim]

    def exposure_from_descriptor(self, descriptor):
        if (
            self.feature_mean is None
            or self.feature_scale is None
            or self.pca_components is None
            or self.cluster_centers is None
        ):
            raise RuntimeError("LearnedMetaPrior must be fit before use")
        z = self._scaled_descriptor(descriptor)
        A = z @ self.pca_components.T
        d2 = np.sum((A[None, :] - self.cluster_centers) ** 2, axis=1)
        N = _softmax_negdist(d2, self.soft_temperature)
        return np.asarray(A, dtype=float), np.asarray(N, dtype=float)

    def risk_exposure_from_descriptor(self, descriptor):
        A, N = self.exposure_from_descriptor(descriptor)
        return RiskExposure(
            A,
            N,
            local_names=tuple(f"meta_A{j}" for j in range(self.local_dim)),
            shared_names=tuple(f"meta_N{j}" for j in range(self.shared_dim)),
            meta={"provider": "LearnedMetaPrior", "source_domains": list(self.source_domains)},
        )

    def risk_coordinate_from_descriptor(self, descriptor):
        exposure = self.risk_exposure_from_descriptor(descriptor)
        return np.concatenate([exposure.A, exposure.N]).astype(float)

    def risk_exposure(self, problem, x, output_index=1):
        del output_index
        return self.risk_exposure_from_descriptor(self.descriptor(problem, x))

    def risk_coordinate(self, problem, x):
        exposure = self.risk_exposure(problem, x)
        return np.concatenate([exposure.A, exposure.N]).astype(float)

    def cumulative_features(self, problem, x, output_index=1):
        del output_index
        return cumulative_feature_vector(self.risk_exposure(problem, x))

    def cumulative_feature_names(self):
        return cumulative_feature_names(
            RiskExposure(
                np.zeros(self.local_dim),
                np.zeros(self.shared_dim),
                local_names=tuple(f"meta_A{j}" for j in range(self.local_dim)),
                shared_names=tuple(f"meta_N{j}" for j in range(self.shared_dim)),
            )
        )

    def hvd_features(self, problem, x):
        desc = self._scaled_descriptor(self.descriptor(problem, x))
        exposure = self.risk_exposure(problem, x)
        A = exposure.A
        N = exposure.N
        return np.concatenate([
            np.array([1.0], dtype=float),
            desc,
            A,
            N,
            A ** 2,
            N ** 2,
        ])

    def risk_class(self, problem, x):
        # Called after risk_exposure in most hot paths; recompute deliberately
        # for simple statelessness.
        return int(np.argmax(self.risk_exposure(problem, x).N))

    def _record_source_data(self, source_problems, n_records_per_domain, rng):
        records = []
        for domain_name, problem in source_problems:
            for _ in range(max(1, int(n_records_per_domain))):
                x = _as_tuple(problem.sample_random(rng))
                y = np.asarray(problem.simulate(x, rng), dtype=float)
                sigma = (
                    float(problem.true_sigma(x)[1])
                    if hasattr(problem, "true_sigma")
                    else float(getattr(problem, "sigma_level", 0.04))
                )
                records.append(SourceRecord(
                    domain=str(domain_name),
                    x=x,
                    y=y,
                    descriptor=self.descriptor(problem, x),
                    profile=np.asarray(problem.normalize(x), dtype=float),
                    tau=float(getattr(problem, "tau", 0.0)),
                    alpha=float(getattr(problem, "alpha", 0.05)),
                    sigma_level=float(getattr(problem, "sigma_level", 0.04)),
                    constraint_sigma=sigma,
                    origin="random",
                    sample_weight=1.0,
                ))
            records.extend(self._record_teacher_source_data(
                str(domain_name),
                problem,
                rng,
            ))
        return records

    def _candidate_pool_from_teacher_hooks(self, problem, rng):
        rows = []
        n_pool = max(1, int(self.teacher_pool_size))
        try:
            rows.extend(problem.initial_samples(n=min(64, n_pool), rng=rng))
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            rows.extend(problem.structured_candidates(n=min(128, n_pool), rng=rng))
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            for anchor in problem.state_anchor_points(n=min(64, n_pool), rng=rng):
                rows.extend(problem.inverse_state_anchor(anchor, rng=rng, n=2))
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            rows.extend(problem.recommendation_refinement_candidates())
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            rows.extend(problem.all_axis_solutions())
        except (AttributeError, TypeError, ValueError):
            pass
        rows = unique_candidates(rows)
        if len(rows) > max(n_pool * 4, 64):
            order = rng.permutation(len(rows))[: max(n_pool * 4, 64)]
            rows = [rows[int(i)] for i in order]
        return rows

    def _record_teacher_source_data(self, domain_name, problem, rng):
        n_teacher = max(0, int(self.teacher_records_per_domain))
        if n_teacher <= 0:
            return []
        rows = self._candidate_pool_from_teacher_hooks(problem, rng)
        if not rows:
            return []
        scored = []
        z_alpha = norm.ppf(1.0 - float(getattr(problem, "alpha", 0.05)))
        for x in rows:
            x = _as_tuple(x)
            if hasattr(problem, "true_outputs"):
                y = np.asarray(problem.true_outputs(x), dtype=float)
            else:
                y = np.asarray(problem.simulate(x, rng), dtype=float)
            if hasattr(problem, "true_sigma"):
                sigma_con = float(problem.true_sigma(x)[1])
            else:
                sigma_con = float(getattr(problem, "sigma_level", 0.04))
            tau = float(getattr(problem, "tau", 0.0))
            margin = float(y[1]) + z_alpha * sigma_con - tau
            scale = max(abs(tau), sigma_con, 1e-6)
            scaled = margin / scale
            feasible = margin <= 0.0
            score = (
                float(y[0])
                + self.feasible_penalty * max(scaled, 0.0)
                + self.boundary_weight * abs(scaled)
                - self.feasible_bonus * float(feasible)
            )
            scored.append({
                "x": x,
                "y": y,
                "sigma_con": sigma_con,
                "margin": margin,
                "scaled": scaled,
                "feasible": feasible,
                "score": float(score),
            })
        selected = []
        seen = set()
        n_keep = min(n_teacher, len(scored))
        n_elite = int(np.ceil(
            n_keep * np.clip(self.teacher_elite_fraction, 0.0, 1.0)))
        n_boundary = int(np.ceil(
            n_keep * np.clip(self.teacher_boundary_fraction, 0.0, 1.0)))

        def add(items, limit):
            for row in items:
                if len(selected) >= n_keep or limit <= 0:
                    break
                if row["x"] in seen:
                    continue
                seen.add(row["x"])
                selected.append(row)
                limit -= 1

        feasible_rows = [row for row in scored if row["feasible"]]
        feasible_rows.sort(key=lambda row: (row["y"][0], abs(row["scaled"])))
        add(feasible_rows, n_elite)
        boundary_rows = sorted(
            scored,
            key=lambda row: (abs(row["scaled"]), max(row["scaled"], 0.0), row["y"][0]),
        )
        add(boundary_rows, n_boundary)
        add(sorted(scored, key=lambda row: row["score"]), n_keep)

        if len(selected) < n_keep:
            chosen_desc = [
                self.descriptor(problem, row["x"])
                for row in selected
            ]
            while len(selected) < n_keep:
                if not chosen_desc:
                    add(sorted(scored, key=lambda row: row["score"]), 1)
                    chosen_desc = [
                        self.descriptor(problem, row["x"])
                        for row in selected
                    ]
                    continue
                D = np.vstack(chosen_desc)
                diverse = []
                for row in scored:
                    if row["x"] in seen:
                        continue
                    desc = self.descriptor(problem, row["x"])
                    dist = float(np.min(np.linalg.norm(D - desc[None, :], axis=1)))
                    diverse.append((dist, row))
                if not diverse:
                    break
                diverse.sort(key=lambda item: (-item[0], item[1]["score"]))
                add([diverse[0][1]], 1)
                chosen_desc.append(self.descriptor(problem, diverse[0][1]["x"]))

        records = []
        for row in selected:
            records.append(SourceRecord(
                domain=domain_name,
                x=row["x"],
                y=np.asarray(row["y"], dtype=float),
                descriptor=self.descriptor(problem, row["x"]),
                profile=np.asarray(problem.normalize(row["x"]), dtype=float),
                tau=float(getattr(problem, "tau", 0.0)),
                alpha=float(getattr(problem, "alpha", 0.05)),
                sigma_level=float(getattr(problem, "sigma_level", 0.04)),
                constraint_sigma=float(row["sigma_con"]),
                origin="source_domain_tuned_teacher",
                sample_weight=max(float(self.teacher_weight), 1e-8),
            ))
        return records

    @staticmethod
    def _source_margin(rec):
        z = norm.ppf(1.0 - float(rec.alpha))
        sigma = (
            float(rec.constraint_sigma)
            if rec.constraint_sigma is not None
            else float(rec.sigma_level)
        )
        return float(rec.y[1]) + z * sigma - float(rec.tau)

    @staticmethod
    def _source_margin_scale(rec):
        sigma = (
            float(rec.constraint_sigma)
            if rec.constraint_sigma is not None
            else float(rec.sigma_level)
        )
        return max(abs(float(rec.tau)), sigma, 1e-6)

    def _boundary_sample_weight(self, rec):
        margin = self._source_margin(rec)
        scaled = margin / self._source_margin_scale(rec)
        temp = max(float(self.boundary_temperature), 1e-6)
        boundary = np.exp(-0.5 * (scaled / temp) ** 2)
        violation = max(float(scaled), 0.0)
        base = 1.0 + self.boundary_weight * boundary + 0.25 * violation
        return float(base * max(float(rec.sample_weight), 1e-8))

    def _record_training_diagnostics(self, records, weights):
        margins = np.asarray([self._source_margin(rec) for rec in records], dtype=float)
        scaled = np.asarray([
            self._source_margin(rec) / self._source_margin_scale(rec)
            for rec in records
        ], dtype=float)
        weights = np.asarray(weights, dtype=float)
        feasible = margins <= 0.0
        positive = np.maximum(margins, 0.0)
        near_boundary = np.abs(scaled) <= 1.0
        if np.any(near_boundary):
            slack_source = float(np.quantile(positive[near_boundary], 0.75))
        else:
            slack_source = float(np.quantile(positive, 0.75)) if len(positive) else 0.0
        self.training_diagnostics = {
            "source_feasible_rate": float(np.mean(feasible)) if len(feasible) else None,
            "source_margin_mean": float(np.mean(margins)) if len(margins) else None,
            "source_margin_median": float(np.median(margins)) if len(margins) else None,
            "source_scaled_margin_median_abs": (
                float(np.median(np.abs(scaled))) if len(scaled) else None
            ),
            "source_recommendation_slack": max(slack_source, 0.0),
            "boundary_weight_mean": float(np.mean(weights)) if len(weights) else None,
            "boundary_weight_max": float(np.max(weights)) if len(weights) else None,
            "boundary_temperature": float(self.boundary_temperature),
            "boundary_weight": float(self.boundary_weight),
            "variance_weight": float(self.variance_weight),
            "feasible_penalty": float(self.feasible_penalty),
            "feasible_bonus": float(self.feasible_bonus),
            "teacher_records_per_domain": int(self.teacher_records_per_domain),
            "teacher_weight": float(self.teacher_weight),
            "teacher_pool_size": int(self.teacher_pool_size),
            "hvd_noise_floor_scale": float(self.hvd_noise_floor_scale),
            "teacher_record_count": int(sum(
                1 for rec in records
                if rec.origin == "source_domain_tuned_teacher"
            )),
            "record_origins": {
                origin: int(sum(1 for rec in records if rec.origin == origin))
                for origin in sorted({rec.origin for rec in records})
            },
        }

    def fit_from_source_problems(self, source_problems, n_records_per_domain=128, rng=None):
        rng = rng or np.random.default_rng(self.seed)
        source_problems = list(source_problems)
        if not source_problems:
            raise ValueError("at least one source domain is required")
        records = self._record_source_data(source_problems, n_records_per_domain, rng)
        if not records:
            raise ValueError("source training produced no records")
        self.source_domains = sorted({rec.domain for rec in records})
        self.n_records = int(len(records))
        descriptors = [rec.descriptor for rec in records]
        Z = self._fit_scaler_pca(descriptors, records=records)
        self._fit_kmeans(Z)
        self.fit_status = "fitting"
        weights = [self._boundary_sample_weight(rec) for rec in records]
        self._record_training_diagnostics(records, weights)
        if self.component_enabled("spectral"):
            self._fit_spectral_basis(records)
        if self.component_enabled("hvd") or self.component_enabled("mean"):
            self._fit_hvd_beta_priors(records)
        if self.component_enabled("proposal"):
            self._fit_anchor_distribution(records)
        self.training_diagnostics["component_stage"] = self.component_stage
        self.training_diagnostics["enabled_components"] = [
            name for name in ("coordinate", "spectral", "hvd", "mean", "proposal")
            if self.component_enabled(name)
        ]
        self.fit_status = "fit"
        return self

    def _fit_spectral_basis(self, records):
        by_domain = {}
        for rec in records:
            by_domain.setdefault(rec.domain, []).append(rec)
        batches = []
        for domain, domain_records in sorted(by_domain.items()):
            psi = np.vstack([
                self.risk_coordinate_from_descriptor(rec.descriptor)
                for rec in domain_records
            ])
            objective = np.asarray([
                float(rec.y[0]) for rec in domain_records
            ], dtype=float)
            constraint = np.asarray([
                self._source_margin(rec) / self._source_margin_scale(rec)
                for rec in domain_records
            ], dtype=float)
            objective_scale = max(float(np.std(objective)), 1e-8)
            objective_z = (objective - float(np.median(objective))) / objective_scale
            temperature = max(float(self.boundary_temperature), 1e-6)
            boundary_sign = np.tanh(constraint / temperature)
            decision = (
                objective_z
                + self.feasible_penalty * np.maximum(constraint, 0.0)
                + 0.25 * np.abs(constraint)
            )
            signals = np.column_stack([
                objective,
                constraint,
                boundary_sign,
                decision,
            ])
            sample_weight = np.asarray([
                self._boundary_sample_weight(rec) for rec in domain_records
            ], dtype=float)
            batches.append(SourceDomainBatch(
                domain=str(domain),
                psi=psi,
                signals=signals,
                sample_weight=sample_weight,
                signal_weight=np.asarray([0.35, 1.0, 1.25, 0.75]),
            ))
        self.spectral_basis = TransferableSpectralBasis(
            active_dim=self.spectral_active_dim,
            max_library_size=self.spectral_max_library_size,
            low_frequency_components=self.spectral_low_frequency_components,
            n_neighbors=self.spectral_graph_neighbors,
            relevance_floor=self.spectral_relevance_floor,
            ridge=self.ridge,
        ).fit(batches)

    def spectral_features_from_descriptor(self, descriptor):
        if self.spectral_basis is None:
            raise RuntimeError("source-invariant spectral basis is not enabled")
        psi = self.risk_coordinate_from_descriptor(descriptor)
        return self.spectral_basis.transform(psi)

    def spectral_features(self, problem, x):
        return self.spectral_features_from_descriptor(self.descriptor(problem, x))

    def coordinate_basis_features(self, problem, x):
        desc = self._scaled_descriptor(self.descriptor(problem, x))
        psi = self.risk_coordinate(problem, x)
        exposure = self.risk_exposure(problem, x)
        cumulative = cumulative_feature_vector(exposure)
        return np.concatenate([desc, psi, psi ** 2, cumulative[1:]])

    def _fit_hvd_beta_priors(self, records):
        by_domain = {}
        for rec in records:
            by_domain.setdefault(rec.domain, []).append(rec)
        beta_by_output = {0: [], 1: []}
        mean_by_output = {0: [], 1: []}
        sigma_by_output = {0: [], 1: []}
        for domain_records in by_domain.values():
            X_desc = np.vstack([
                np.concatenate([[1.0], self._scaled_descriptor(rec.descriptor)])
                for rec in domain_records
            ])
            X_meta = np.vstack([
                np.concatenate([[
                    1.0],
                    self._mean_prior_features_from_descriptor(rec.descriptor),
                ])
                for rec in domain_records
            ])
            F = np.vstack([
                cumulative_feature_vector(
                    self.risk_exposure_from_descriptor(rec.descriptor)
                )
                for rec in domain_records
            ])
            weights = np.asarray([
                self._boundary_sample_weight(rec) for rec in domain_records
            ], dtype=float)
            weights = np.clip(weights, 1e-4, 1e4)
            sqrt_w = np.sqrt(weights)
            reg_mean = self.ridge * np.eye(X_desc.shape[1], dtype=float)
            reg_mean[0, 0] = 0.0
            for out_idx in (0, 1):
                y = np.asarray([float(rec.y[out_idx]) for rec in domain_records], dtype=float)
                Xw = X_desc * sqrt_w[:, None]
                yw = y * sqrt_w
                try:
                    beta_mean = np.linalg.solve(
                        Xw.T @ Xw + reg_mean,
                        Xw.T @ yw,
                    )
                except np.linalg.LinAlgError:
                    beta_mean = np.linalg.lstsq(
                        Xw.T @ Xw + reg_mean,
                        Xw.T @ yw,
                        rcond=None,
                    )[0]
                reg_meta = self.ridge * np.eye(X_meta.shape[1], dtype=float)
                reg_meta[0, 0] = 0.0
                Xmw = X_meta * sqrt_w[:, None]
                try:
                    beta_meta = np.linalg.solve(
                        Xmw.T @ Xmw + reg_meta,
                        Xmw.T @ yw,
                    )
                except np.linalg.LinAlgError:
                    beta_meta = np.linalg.lstsq(
                        Xmw.T @ Xmw + reg_meta,
                        Xmw.T @ yw,
                        rcond=None,
                    )[0]
                resid_meta = y - X_meta @ beta_meta
                mean_by_output[out_idx].append(beta_meta)
                sigma_by_output[out_idx].append(float(
                    np.sqrt(np.mean(resid_meta ** 2)) if len(resid_meta) else 0.0
                ))
                resid2 = np.maximum((y - X_desc @ beta_mean) ** 2, 1e-10)
                if self.hvd_noise_floor_scale > 0.0:
                    sigmas = []
                    for rec in domain_records:
                        if out_idx == 1 and rec.constraint_sigma is not None:
                            sigmas.append(float(rec.constraint_sigma))
                        else:
                            sigmas.append(float(rec.sigma_level))
                    noise_floor = (
                        float(self.hvd_noise_floor_scale)
                        * np.asarray(sigmas, dtype=float)
                    ) ** 2
                    resid2 = np.maximum(resid2, noise_floor)
                resid_scale = float(np.median(resid2)) if len(resid2) else 0.0
                if resid_scale <= 1e-12:
                    resid_scale = float(np.mean(resid2) + 1e-10)
                var_weights = weights * (
                    1.0 + self.variance_weight * np.minimum(resid2 / resid_scale, 10.0)
                )
                sqrt_vw = np.sqrt(np.clip(var_weights, 1e-4, 1e4))
                Fw = F * sqrt_vw[:, None]
                rw = resid2 * sqrt_vw
                reg_var = self.ridge * np.eye(F.shape[1], dtype=float)
                try:
                    beta = np.linalg.solve(Fw.T @ Fw + reg_var, Fw.T @ rw)
                except np.linalg.LinAlgError:
                    beta = np.linalg.lstsq(Fw.T @ Fw + reg_var, Fw.T @ rw, rcond=None)[0]
                beta = _project_psd_features(beta, self.local_dim, self.shared_dim)
                beta_by_output[out_idx].append(beta)
        for out_idx, rows in beta_by_output.items():
            if rows:
                self.beta_prior[out_idx] = np.mean(np.vstack(rows), axis=0)
        for out_idx, rows in mean_by_output.items():
            if rows:
                self.mean_prior[out_idx] = np.mean(np.vstack(rows), axis=0)
                sigmas = np.asarray(sigma_by_output.get(out_idx, []), dtype=float)
                self.mean_prior_sigma[out_idx] = float(
                    np.median(sigmas) if len(sigmas) else 0.0)

    def _mean_prior_features_from_descriptor(self, descriptor):
        desc = self._scaled_descriptor(descriptor)
        psi = self.risk_coordinate_from_descriptor(descriptor)
        exposure = self.risk_exposure_from_descriptor(descriptor)
        cumulative = cumulative_feature_vector(exposure)
        return np.concatenate([
            desc,
            psi,
            psi ** 2,
            cumulative[1:],
        ])

    def mean_prior_features(self, problem, x):
        return self._mean_prior_features_from_descriptor(self.descriptor(problem, x))

    def source_mean_prior_predict(self, problem, x, output_index=1):
        if not self.component_enabled("mean"):
            return None
        beta = self.mean_prior.get(int(output_index))
        if beta is None:
            return None
        phi = np.concatenate([[1.0], self.mean_prior_features(problem, x)])
        if len(phi) != len(beta):
            return None
        return float(phi @ beta)

    def source_mean_prior_predict_many(self, problem, xs, output_index=1):
        if not self.component_enabled("mean"):
            return None
        beta = self.mean_prior.get(int(output_index))
        if beta is None:
            return None
        Phi = np.vstack([
            np.concatenate([[1.0], self.mean_prior_features(problem, x)])
            for x in xs
        ])
        if Phi.shape[1] != len(beta):
            return None
        return Phi @ beta

    def source_mean_prior_sigma(self, output_index=1):
        if not self.component_enabled("mean"):
            return 0.0
        return float(max(
            self.mean_prior_sigma.get(int(output_index), 0.0) or 0.0,
            1e-8,
        ))

    def _fit_anchor_distribution(self, records):
        scored = []
        for rec in records:
            psi = self.risk_coordinate_from_descriptor(rec.descriptor)
            # Source-only observed feasibility heuristic.  No held-out target
            # truth enters this score.
            margin = self._source_margin(rec)
            scaled_margin = margin / self._source_margin_scale(rec)
            violation = max(scaled_margin, 0.0)
            boundary_distance = abs(scaled_margin)
            feasible = margin <= 0.0
            score = (
                float(rec.y[0])
                + self.feasible_penalty * violation
                + self.boundary_weight * boundary_distance
                - self.feasible_bonus * float(feasible)
            )
            scored.append({
                "score": float(score),
                "objective": float(rec.y[0]),
                "margin": float(margin),
                "scaled_margin": float(scaled_margin),
                "psi": np.asarray(psi, dtype=float),
                "profile": (
                    None
                    if rec.profile is None
                    else np.asarray(rec.profile, dtype=float).reshape(-1)
                ),
                "domain": rec.domain,
                "feasible": bool(feasible),
                "anchor_type": "calibrated_score",
            })
        n_keep = max(1, min(self.anchor_count, len(scored)))
        n_elite = int(np.ceil(n_keep * np.clip(self.elite_fraction, 0.0, 1.0)))
        n_boundary = int(np.ceil(n_keep * np.clip(self.boundary_fraction, 0.0, 1.0)))
        selected = []
        seen = set()

        def add_rows(rows, limit, anchor_type):
            for row in rows:
                if len(selected) >= n_keep or limit <= 0:
                    break
                key = tuple(np.round(row["psi"], 8))
                if key in seen:
                    continue
                seen.add(key)
                item = dict(row)
                item["anchor_type"] = anchor_type
                selected.append(item)
                limit -= 1

        feasible_rows = [row for row in scored if row["feasible"]]
        feasible_rows.sort(key=lambda row: (row["objective"], abs(row["scaled_margin"])))
        add_rows(feasible_rows, n_elite, "source_feasible_elite")

        boundary_rows = sorted(
            scored,
            key=lambda row: (abs(row["scaled_margin"]), max(row["scaled_margin"], 0.0)),
        )
        add_rows(boundary_rows, n_boundary, "source_chance_boundary")

        calibrated_rows = sorted(scored, key=lambda row: row["score"])
        add_rows(calibrated_rows, n_keep, "calibrated_score")

        # Fill leftovers with far-apart psi points to keep the frozen proposal
        # from collapsing into one source-domain basin.
        while len(selected) < n_keep:
            if not selected:
                add_rows(calibrated_rows, 1, "calibrated_score")
                continue
            chosen = np.vstack([row["psi"] for row in selected])
            diverse_rows = []
            for row in scored:
                key = tuple(np.round(row["psi"], 8))
                if key in seen:
                    continue
                d = float(np.min(np.linalg.norm(chosen - row["psi"][None, :], axis=1)))
                item = dict(row)
                item["diversity_distance"] = d
                diverse_rows.append(item)
            if not diverse_rows:
                break
            diverse_rows.sort(key=lambda row: (-row["diversity_distance"], row["score"]))
            add_rows(diverse_rows, 1, "diverse_source_psi")

        self.anchor_scores = np.asarray([row["score"] for row in selected], dtype=float)
        self.anchor_psi = np.vstack([row["psi"] for row in selected])
        self.anchor_meta = [
            {
                "domain": row["domain"],
                "margin": float(row["margin"]),
                "scaled_margin": float(row["scaled_margin"]),
                "objective": float(row["objective"]),
                "feasible": bool(row["feasible"]),
                "anchor_type": row["anchor_type"],
            }
            for row in selected
        ]
        self.profile_templates = []
        profile_seen = set()
        for row in selected:
            profile = row.get("profile")
            if profile is None or len(profile) == 0:
                continue
            profile = np.clip(np.asarray(profile, dtype=float).reshape(-1), 0.0, 1.0)
            key = tuple(np.round(profile, 3))
            if key in profile_seen:
                continue
            profile_seen.add(key)
            self.profile_templates.append({
                "profile": profile,
                "domain": row["domain"],
                "anchor_type": row["anchor_type"],
                "feasible": bool(row["feasible"]),
                "score": float(row["score"]),
                "margin": float(row["margin"]),
                "objective": float(row["objective"]),
            })
        if selected:
            margins = np.asarray([row["margin"] for row in selected], dtype=float)
            self.training_diagnostics.update({
                "anchor_feasible_rate": float(np.mean(margins <= 0.0)),
                "anchor_margin_median": float(np.median(margins)),
                "anchor_margin_abs_median": float(np.median(np.abs(margins))),
                "anchor_types": {
                    anchor_type: int(sum(
                        1 for row in selected if row["anchor_type"] == anchor_type
                    ))
                    for anchor_type in sorted({row["anchor_type"] for row in selected})
                },
                "profile_template_count": int(len(self.profile_templates)),
            })

    def state_anchor_points(self, n=10, rng=None):
        if not self.component_enabled("proposal"):
            return []
        rng = rng or np.random.default_rng(self.seed)
        if len(self.anchor_psi) == 0:
            return []
        n_take = max(0, int(n))
        if (
            self.anchor_sampling_temperature > 0.0
            and len(self.anchor_scores) == len(self.anchor_psi)
        ):
            scores = np.asarray(self.anchor_scores, dtype=float)
            scale = float(np.std(scores))
            scale = max(scale, 1e-8)
            logits = -scores / (scale * self.anchor_sampling_temperature)
            logits -= float(np.max(logits))
            probs = np.exp(logits)
            probs = probs / max(float(np.sum(probs)), 1e-12)
            order = rng.choice(
                len(self.anchor_psi),
                size=min(n_take, len(self.anchor_psi)),
                replace=False,
                p=probs,
            )
        else:
            order = rng.permutation(len(self.anchor_psi))[:n_take]
        anchors = []
        for pos in order:
            psi = np.asarray(self.anchor_psi[int(pos)], dtype=float)
            anchors.append({
                "psi": psi.tolist(),
                "A": psi[: self.local_dim].tolist(),
                "N": psi[self.local_dim:].tolist(),
                "source_score": float(self.anchor_scores[int(pos)]),
                "source_meta": (
                    self.anchor_meta[int(pos)]
                    if int(pos) < len(self.anchor_meta)
                    else {}
                ),
                "coordinate": "learned_meta_psi=(A,N)",
            })
        return anchors

    def _continuous_to_tuple(self, problem, z):
        z = np.asarray(z, dtype=float).reshape(-1)
        z = np.clip(z, 0.0, 1.0)
        try:
            return _as_tuple(problem.continuous_to_int(z))
        except (AttributeError, TypeError, ValueError):
            L = int(getattr(problem, "L", 100))
            return tuple(int(np.clip(round(v * L), 0, L)) for v in z)

    def universal_shape_candidates(self, problem, n=0, rng=None):
        """Admissible low-complexity policy shapes shared by all domains.

        These candidates use only bounds and dimension, not held-out target
        objectives, constraints, anchors, or risk coordinates.  They act like a
        weak smoothness/low-complexity prior: constants, one-control-plus-tail,
        piecewise thirds, and monotone ramps.
        """
        if not self.component_enabled("proposal"):
            return []
        n_take = max(0, int(n))
        if n_take <= 0:
            return []
        d = max(1, int(getattr(problem, "d", 1)))
        rows = []

        def add(z):
            rows.append(self._continuous_to_tuple(problem, z))

        levels = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
        for value in levels:
            add(np.full(d, value, dtype=float))

        constant_end = len(rows)

        head_levels = [0.25, 0.35, 0.50, 0.65, 0.15, 0.80]
        tail_levels = [0.75, 0.65, 0.85, 0.55, 0.45, 0.30]
        for head in head_levels:
            for tail in tail_levels:
                z = np.full(d, tail, dtype=float)
                z[0] = head
                add(z)

        head_tail_end = len(rows)

        third_levels = [0.25, 0.40, 0.55, 0.70]
        third_templates = [
            (a, b, c)
            for a in third_levels
            for b in third_levels
            for c in third_levels
        ]
        for a, b, c in third_templates:
            z = np.empty(d, dtype=float)
            for j in range(d):
                z[j] = (a, b, c)[min(2, int(3 * j / max(d, 1)))]
            add(z)

        thirds_end = len(rows)

        for lo, hi in [(0.20, 0.80), (0.30, 0.70), (0.40, 0.60)]:
            add(np.linspace(lo, hi, d))
            add(np.linspace(hi, lo, d))

        blocks = [
            rows[:constant_end],
            rows[constant_end:head_tail_end],
            rows[head_tail_end:thirds_end],
            rows[thirds_end:],
        ]
        balanced = []
        max_block = max((len(block) for block in blocks), default=0)
        for pos in range(max_block):
            for block in blocks:
                if pos < len(block):
                    balanced.append(block[pos])
        rows = unique_candidates(balanced)
        if len(rows) <= n_take:
            return rows
        # Keep early hand-balanced shapes and randomly thin only the excess.
        head = rows[: min(len(rows), max(min(n_take, 8), n_take // 3))]
        tail = rows[len(head):]
        rng = rng or np.random.default_rng(self.seed)
        need = max(0, n_take - len(head))
        if need > 0 and tail:
            order = rng.permutation(len(tail))[:need]
            head.extend(tail[int(i)] for i in order)
        return unique_candidates(head)[:n_take]

    def profile_template_candidates(self, problem, n=0, rng=None):
        """Replay source-learned normalized policy profiles on a held-out target.

        Templates are selected only from source-domain records during
        `fit_from_source_problems`.  At test time we resample each normalized
        profile to the target dimension and convert through target bounds.  This
        is a LODO meta-prior: it transfers policy shape, not target objective or
        feasibility labels.
        """
        if not self.component_enabled("proposal"):
            return []
        n_take = max(0, int(n))
        if n_take <= 0 or not self.profile_templates:
            return []
        rng = rng or np.random.default_rng(self.seed)
        d = max(1, int(getattr(problem, "d", 1)))
        order = list(range(len(self.profile_templates)))
        # Keep source-safe/low-score templates early, randomize ties to avoid a
        # single source domain dominating all held-out proposals.
        order.sort(key=lambda i: (
            0 if self.profile_templates[i].get("feasible", False) else 1,
            float(self.profile_templates[i].get("score", 0.0)),
            float(abs(self.profile_templates[i].get("margin", 0.0))),
        ))
        if len(order) > n_take:
            head = order[: max(1, min(len(order), n_take // 2))]
            tail = order[len(head):]
            need = n_take - len(head)
            if need > 0 and tail:
                pick = rng.permutation(len(tail))[:need]
                head.extend(tail[int(i)] for i in pick)
            order = head
        rows = []
        for idx in order[:n_take]:
            profile = np.asarray(
                self.profile_templates[int(idx)]["profile"],
                dtype=float,
            ).reshape(-1)
            if len(profile) == 0:
                continue
            if len(profile) == d:
                z = profile.copy()
            else:
                xp = np.linspace(0.0, 1.0, len(profile))
                xnew = np.linspace(0.0, 1.0, d)
                z = np.interp(xnew, xp, profile)
            rows.append(self._continuous_to_tuple(problem, np.clip(z, 0.0, 1.0)))
        return unique_candidates(rows)[:n_take]

    def inverse_state_anchor(self, problem, anchor, rng=None, n=1, pool_size=512):
        if not self.component_enabled("proposal"):
            return []
        rng = rng or np.random.default_rng(self.seed)
        n = max(1, int(n))
        target = None
        if isinstance(anchor, dict):
            if anchor.get("psi") is not None:
                target = np.asarray(anchor["psi"], dtype=float)
            elif anchor.get("A") is not None and anchor.get("N") is not None:
                target = np.concatenate([
                    np.asarray(anchor["A"], dtype=float),
                    np.asarray(anchor["N"], dtype=float),
                ])
        if target is None:
            return []
        rows = [problem.sample_random(rng) for _ in range(max(pool_size, 8 * n))]
        scored = []
        for x in unique_candidates(rows):
            psi = self.risk_coordinate(problem, x)
            scored.append((float(np.linalg.norm(psi - target)), _as_tuple(x)))
        scored.sort(key=lambda item: item[0])
        return [row for _, row in scored[:n]]

    def proposal_candidates(self, problem, n=32, rng=None, pool_size=1024):
        if not self.component_enabled("proposal"):
            rng = rng or np.random.default_rng(self.seed)
            return unique_candidates([
                problem.sample_random(rng) for _ in range(max(0, int(n)))
            ])
        rng = rng or np.random.default_rng(self.seed)
        n_target = max(0, int(n))
        rows = []
        n_profiles = min(
            len(self.profile_templates),
            n_target,
            max(min(n_target, 3), int(round(0.35 * n_target))),
        )
        rows.extend(self.profile_template_candidates(
            problem,
            n=n_profiles,
            rng=rng,
        ))
        n_universal = min(
            max(0, int(self.universal_shape_count)),
            max(0, n_target - len(rows)),
            max(min(max(0, n_target - len(rows)), 4), int(round(0.35 * n_target))),
        )
        rows.extend(self.universal_shape_candidates(
            problem,
            n=n_universal,
            rng=rng,
        ))
        n_anchor = max(1, n_target - len(rows))
        for anchor in self.state_anchor_points(n=n_anchor, rng=rng):
            rows.extend(self.inverse_state_anchor(
                problem,
                anchor,
                rng=rng,
                n=1,
                pool_size=max(64, int(pool_size) // max(1, n_anchor)),
            ))
        while len(rows) < n_target:
            rows.append(problem.sample_random(rng))
        return unique_candidates(rows)[:n_target]

    def cumulative_hvd_prior_beta(self, output_index=1, feature_dim=None):
        if not self.component_enabled("hvd"):
            return None
        beta = self.beta_prior.get(int(output_index))
        if beta is None:
            return None
        beta = np.asarray(beta, dtype=float)
        if feature_dim is not None and len(beta) != int(feature_dim):
            return None
        return beta.copy()

    def source_calibrated_recommendation_slack(self):
        if not self.component_enabled("proposal"):
            return 0.0
        return float(max(
            self.training_diagnostics.get("source_recommendation_slack", 0.0) or 0.0,
            0.0,
        ))

    def diagnostics(self):
        return {
            "status": self.fit_status,
            "component_stage": self.component_stage,
            "enabled_components": [
                name for name in (
                    "coordinate", "spectral", "hvd", "mean", "proposal"
                )
                if self.component_enabled(name)
            ],
            "source_domains": list(self.source_domains),
            "n_records": int(self.n_records),
            "local_dim": int(self.local_dim),
            "shared_dim": int(self.shared_dim),
            "n_anchors": int(len(self.anchor_psi)),
            "n_profile_templates": int(len(self.profile_templates)),
            "universal_shape_count": int(self.universal_shape_count),
            "has_beta_prior": {
                str(key): value is not None
                for key, value in self.beta_prior.items()
            },
            "has_mean_prior": {
                str(key): value is not None
                for key, value in self.mean_prior.items()
            },
            "spectral_basis": (
                None
                if self.spectral_basis is None
                else self.spectral_basis.diagnostics()
            ),
            "coordinate": dict(self.coordinate_diagnostics),
            "training": dict(self.training_diagnostics),
        }


class MetaPriorSurrogateBasis:
    """Admissible target-observation calibration basis induced by frozen psi."""

    def __init__(self, meta_prior: LearnedMetaPrior, problem):
        self.meta_prior = meta_prior
        self.problem = problem
        descriptor_dim = len(meta_prior.feature_mean)
        psi_dim = meta_prior.local_dim + meta_prior.shared_dim
        cumulative_dim = (
            1
            + meta_prior.local_dim
            + meta_prior.shared_dim * (meta_prior.shared_dim + 1) // 2
            + meta_prior.shared_dim
        )
        self.feature_dim = (
            int(meta_prior.spectral_basis.feature_dim)
            if meta_prior.component_enabled("spectral")
            else descriptor_dim + 2 * psi_dim + cumulative_dim - 1
        )

    def features(self, x):
        if self.meta_prior.component_enabled("spectral"):
            return self.meta_prior.spectral_features(self.problem, x)
        return self.meta_prior.coordinate_basis_features(self.problem, x)

    def features_many(self, X):
        if len(X) == 0:
            return np.empty((0, self.feature_dim), dtype=float)
        return np.vstack([self.features(x) for x in X])


class PilotGatedMetaPriorBasis:
    """Choose a frozen source basis with a decision-aware pilot gate.

    The gate never reads target oracle values.  Stage 1 learns only the
    constraint/risk-boundary representation: it scores leave-one-out
    chance-margin calibration, ordering near the boundary, and false-feasible
    errors.  The objective keeps the Stage-0 coordinate basis until a separate
    transferable objective module is validated.  Plain LOO NMSE and objective
    spectral scores remain in diagnostics, but do not control Stage-1 behavior.
    """

    adaptive_meta_basis = True

    def __init__(self, meta_prior: LearnedMetaPrior, problem, output_index, ridge=1.0):
        self.meta_prior = meta_prior
        self.problem = problem
        self.output_index = int(output_index)
        self.ridge = float(ridge)
        self.identity_dim = min(
            max(1, int(meta_prior.spectral_active_dim)),
            meta_prior.local_dim + meta_prior.shared_dim,
        )
        descriptor_dim = len(meta_prior.feature_mean)
        psi_dim = meta_prior.local_dim + meta_prior.shared_dim
        cumulative_dim = (
            1
            + meta_prior.local_dim
            + meta_prior.shared_dim * (meta_prior.shared_dim + 1) // 2
            + meta_prior.shared_dim
        )
        self.feature_dim = max(
            descriptor_dim + 2 * psi_dim + cumulative_dim - 1,
            self.identity_dim,
            int(meta_prior.spectral_basis.feature_dim),
        )
        self.selected_basis = "coordinate"
        self.gate_diagnostics = {
            "status": "unfit",
            "selected_basis": self.selected_basis,
            "output_index": self.output_index,
        }

    def _variant_features(self, x, variant):
        if variant == "coordinate":
            return self.meta_prior.coordinate_basis_features(self.problem, x)
        if variant == "source_spectral":
            return self.meta_prior.spectral_features(self.problem, x)
        psi = self.meta_prior.risk_coordinate(self.problem, x)
        return np.asarray(psi[: self.identity_dim], dtype=float)

    def features(self, x):
        values = np.asarray(
            self._variant_features(x, self.selected_basis), dtype=float).reshape(-1)
        out = np.zeros(self.feature_dim, dtype=float)
        out[: min(len(values), self.feature_dim)] = values[: self.feature_dim]
        return out

    def features_many(self, X):
        if len(X) == 0:
            return np.empty((0, self.feature_dim), dtype=float)
        return np.vstack([self.features(x) for x in X])

    def fit_from_observations(self, observations, output_index=None):
        output_index = self.output_index if output_index is None else int(output_index)
        xs = list(observations)
        if len(xs) < 5:
            self.gate_diagnostics = {
                "status": "insufficient_pilot",
                "selected_basis": self.selected_basis,
                "output_index": output_index,
                "n_observations": int(len(xs)),
            }
            return self.selected_basis
        observed = np.vstack([
            np.mean(np.asarray(observations[x], dtype=float), axis=0)
            for x in xs
        ])
        target = np.asarray(observed[:, output_index], dtype=float)
        scores = {}
        nmse_scores = {}
        component_scores = {}
        for variant in ("coordinate", "fixed_psi", "source_spectral"):
            matrix = np.vstack([self._variant_features(x, variant) for x in xs])
            prediction = self._ridge_loo_predictions(matrix, target)
            nmse_scores[variant] = self._normalized_mse(target, prediction)
            if output_index == 1:
                components = self._constraint_decision_score(target, prediction)
            else:
                components = self._objective_decision_score(
                    target,
                    prediction,
                    observed[:, 1],
                )
            component_scores[variant] = components
            scores[variant] = float(components["total"])
        # Stage 1 is an incremental learned-prior ablation: it may replace the
        # Stage-0 coordinate basis only when target pilot evidence supports the
        # frozen source-spectral basis.  The compact fixed-psi model remains an
        # audit diagnostic, but is not a behavior-changing fallback.
        eligible = (
            ("coordinate", "source_spectral")
            if output_index == 1
            else ("coordinate",)
        )
        tie_order = {"coordinate": 0, "source_spectral": 1}
        selected = min(
            eligible, key=lambda name: (scores[name], tie_order[name]))
        baseline_score = float(scores["coordinate"])
        tolerance = self.meta_prior.spectral_gate_selection_tolerance * max(
            abs(baseline_score), 1.0)
        if (
            selected != "coordinate"
            and float(scores[selected]) >= baseline_score - tolerance
        ):
            selected = "coordinate"
        if output_index == 1 and selected == "source_spectral":
            baseline_components = component_scores["coordinate"]
            selected_components = component_scores[selected]
            raw_false_feasible_worse = (
                selected_components["raw_false_feasible_rate"]
                > baseline_components["raw_false_feasible_rate"] + 0.05
            )
            dangerous_limit = max(
                1.25 * baseline_components["dangerous_underprediction"],
                baseline_components["dangerous_underprediction"] + 0.05,
            )
            if (
                raw_false_feasible_worse
                or selected_components["dangerous_underprediction"] > dangerous_limit
            ):
                selected = "coordinate"
        self.selected_basis = selected
        self.gate_diagnostics = {
            "status": "fit",
            "selected_basis": self.selected_basis,
            "output_index": output_index,
            "n_observations": int(len(xs)),
            "selection_metric": "decision_aware_loo",
            "gate_scope": "constraint_boundary_only",
            "decision_score": {name: float(value) for name, value in scores.items()},
            "decision_components": component_scores,
            "loo_nmse": {name: float(value) for name, value in nmse_scores.items()},
            "selection_tolerance": float(tolerance),
            "eligible_bases": list(eligible),
            "pilot_constraint_guard": (
                float(component_scores[self.selected_basis]["calibration_guard"])
                if output_index == 1
                else 0.0
            ),
        }
        return self.selected_basis

    def diagnostics(self):
        return dict(self.gate_diagnostics)

    def certification_guard(self):
        if self.output_index != 1:
            return 0.0
        return max(float(self.gate_diagnostics.get("pilot_constraint_guard", 0.0)), 0.0)

    def _ridge_loo_score(self, features, target):
        prediction = self._ridge_loo_predictions(features, target)
        return self._normalized_mse(target, prediction)

    def _ridge_loo_predictions(self, features, target):
        predictions = []
        for heldout in range(len(features)):
            train = np.arange(len(features)) != heldout
            prediction = self._ridge_predict(
                features[train],
                target[train],
                features[heldout:heldout + 1],
            )[0]
            predictions.append(float(prediction))
        return np.asarray(predictions, dtype=float)

    @staticmethod
    def _normalized_mse(target, prediction, weights=None):
        target = np.asarray(target, dtype=float)
        prediction = np.asarray(prediction, dtype=float)
        if weights is None:
            weights = np.ones(len(target), dtype=float)
        weights = np.clip(np.asarray(weights, dtype=float), 1e-8, np.inf)
        weights = weights / float(np.sum(weights))
        center = float(np.sum(target * weights))
        scale = float(np.sum((target - center) ** 2 * weights))
        error = float(np.sum((target - prediction) ** 2 * weights))
        return error / max(scale, 1e-10)

    @staticmethod
    def _pairwise_order_loss(target, prediction, weights, scale):
        target = np.asarray(target, dtype=float)
        prediction = np.asarray(prediction, dtype=float)
        weights = np.asarray(weights, dtype=float)
        losses = []
        pair_weights = []
        scale = max(float(scale), 1e-8)
        for i in range(len(target)):
            for j in range(i + 1, len(target)):
                truth_order = np.tanh((target[i] - target[j]) / scale)
                pred_order = np.tanh((prediction[i] - prediction[j]) / scale)
                losses.append(float((truth_order - pred_order) ** 2))
                pair_weights.append(float(np.sqrt(weights[i] * weights[j])))
        if not losses:
            return 0.0
        pair_weights = np.clip(np.asarray(pair_weights), 1e-8, np.inf)
        return float(np.average(np.asarray(losses), weights=pair_weights))

    def _chance_margin(self, constraint_mean):
        alpha = float(getattr(self.problem, "alpha", 0.05))
        z_alpha = float(norm.ppf(1.0 - alpha))
        sigma = max(float(getattr(self.problem, "sigma_level", 0.04)), 1e-8)
        tau = float(getattr(self.problem, "tau", 0.0))
        return np.asarray(constraint_mean, dtype=float) + z_alpha * sigma - tau

    def _frontier_weights(self, margins):
        margins = np.asarray(margins, dtype=float)
        sigma = max(float(getattr(self.problem, "sigma_level", 0.04)), 1e-8)
        robust = 0.7413 * float(
            np.quantile(margins, 0.75) - np.quantile(margins, 0.25))
        scale = max(sigma, robust, 1e-8)
        boundary = np.exp(-0.5 * (margins / scale) ** 2)
        feasible = margins <= 0.0
        if np.any(feasible):
            frontier = boundary
        else:
            frontier = np.exp(-(margins - float(np.min(margins))) / scale)
        weight = (
            1.0
            + self.meta_prior.spectral_gate_boundary_weight
            * np.maximum(boundary, frontier)
            + feasible.astype(float)
        )
        return np.asarray(weight, dtype=float), float(scale)

    def _constraint_decision_score(self, target, prediction):
        truth_margin = self._chance_margin(target)
        pred_margin = self._chance_margin(prediction)
        weights, scale = self._frontier_weights(truth_margin)
        margin_nmse = self._normalized_mse(
            truth_margin, pred_margin, weights=weights)
        boundary_mse = float(np.average(
            ((truth_margin - pred_margin) / scale) ** 2,
            weights=weights,
        ))
        rank_loss = self._pairwise_order_loss(
            truth_margin, pred_margin, weights, scale)
        probability = 1.0 / (
            1.0 + np.exp(np.clip(pred_margin / scale, -40.0, 40.0)))
        feasible = truth_margin <= 0.0
        brier = float(np.average(
            (probability - feasible.astype(float)) ** 2,
            weights=weights,
        ))
        residual = truth_margin - pred_margin
        infeasible = ~feasible
        raw_false_feasible = (
            float(np.mean(pred_margin[infeasible] <= 0.0))
            if np.any(infeasible)
            else 0.0
        )
        calibrated = np.empty_like(pred_margin)
        quantile = float(np.clip(
            self.meta_prior.spectral_gate_calibration_quantile, 0.5, 0.99))
        for heldout in range(len(residual)):
            keep = np.arange(len(residual)) != heldout
            correction = (
                float(np.quantile(residual[keep], quantile))
                if np.any(keep)
                else 0.0
            )
            calibrated[heldout] = pred_margin[heldout] + max(correction, 0.0)
        false_feasible = (
            float(np.mean(calibrated[infeasible] <= 0.0))
            if np.any(infeasible)
            else 0.0
        )
        false_infeasible = (
            float(np.mean(calibrated[feasible] > 0.0))
            if np.any(feasible)
            else 0.0
        )
        dangerous_underprediction = float(np.average(
            (np.maximum(residual, 0.0) / scale) ** 2,
            weights=weights,
        ))
        total = (
            0.15 * margin_nmse
            + 0.20 * boundary_mse
            + 0.25 * rank_loss
            + 0.20 * brier
            + self.meta_prior.spectral_gate_dangerous_weight
            * false_feasible
            + 0.50 * self.meta_prior.spectral_gate_dangerous_weight
            * raw_false_feasible
            + 0.20 * false_infeasible
            + 0.35 * dangerous_underprediction
        )
        return {
            "total": float(total),
            "margin_nmse": float(margin_nmse),
            "boundary_mse": float(boundary_mse),
            "rank_loss": float(rank_loss),
            "brier": float(brier),
            "false_feasible_rate": float(false_feasible),
            "raw_false_feasible_rate": float(raw_false_feasible),
            "false_infeasible_rate": float(false_infeasible),
            "dangerous_underprediction": float(dangerous_underprediction),
            "margin_scale": float(scale),
            "calibration_guard": float(max(
                np.quantile(residual, quantile), 0.0)),
        }

    def _objective_decision_score(self, target, prediction, constraint_mean):
        margins = self._chance_margin(constraint_mean)
        weights, margin_scale = self._frontier_weights(margins)
        objective_scale = max(float(np.std(target)), 1e-8)
        weighted_nmse = self._normalized_mse(target, prediction, weights=weights)
        rank_loss = self._pairwise_order_loss(
            target, prediction, weights, objective_scale)
        feasible = np.where(margins <= 0.0)[0]
        if len(feasible) == 0:
            count = min(max(2, len(target) // 3), len(target))
            relevant = np.argsort(margins)[:count]
        else:
            relevant = feasible
        chosen = int(relevant[int(np.argmin(prediction[relevant]))])
        best = float(np.min(target[relevant]))
        selection_regret = max(float(target[chosen]) - best, 0.0) / objective_scale
        total = 0.50 * weighted_nmse + 0.35 * rank_loss + 0.15 * selection_regret
        return {
            "total": float(total),
            "weighted_nmse": float(weighted_nmse),
            "rank_loss": float(rank_loss),
            "selection_regret": float(selection_regret),
            "margin_scale": float(margin_scale),
            "n_observed_feasible": int(len(feasible)),
        }

    def _ridge_predict(self, train_x, train_y, test_x):
        mean = np.mean(train_x, axis=0)
        scale = np.std(train_x, axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        X = (train_x - mean) / scale
        X_test = (test_x - mean) / scale
        X = np.column_stack([np.ones(len(X)), X])
        X_test = np.column_stack([np.ones(len(X_test)), X_test])
        y_mean = float(np.mean(train_y))
        y_scale = max(float(np.std(train_y)), 1e-8)
        y = (np.asarray(train_y, dtype=float) - y_mean) / y_scale
        penalty = self.ridge * np.eye(X.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        try:
            beta = np.linalg.solve(X.T @ X + penalty, X.T @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(X.T @ X + penalty, X.T @ y, rcond=None)[0]
        return (X_test @ beta) * y_scale + y_mean


class AdmissibleProblemAdapter:
    """Hide target-specific structural hooks while preserving simulation API."""

    def __init__(self, base_problem, variant="strict_universal"):
        self.base = base_problem
        self.problem_name = f"{base_problem.problem_name}_{variant}"
        self.d = base_problem.d
        self.L = base_problem.L
        self.alpha = base_problem.alpha
        self.tau = base_problem.tau
        self.sigma_level = base_problem.sigma_level
        self.ref_point = getattr(base_problem, "ref_point", None)
        self.variant = str(variant)

    def __getattr__(self, name):
        if name in HIDDEN_TARGET_STRUCTURAL_METHODS:
            raise AttributeError(name)
        return getattr(self.base, name)

    def admissibility_audit(self):
        if self.variant == "domain_tuned_upper_bound":
            return domain_tuned_audit().to_dict()
        return strict_universal_audit().to_dict()

    def int_bounds(self):
        return self.base.int_bounds()

    def normalize(self, x):
        return self.base.normalize(x)

    def continuous_to_int(self, x_norm):
        return self.base.continuous_to_int(x_norm)

    def sample_random(self, rng=None):
        return self.base.sample_random(rng)

    def simulate(self, x, rng=None):
        return self.base.simulate(x, rng)


class MetaPriorProblemAdapter(AdmissibleProblemAdapter):
    """Held-out target adapter using only a frozen source-trained meta-prior."""

    def __init__(
        self,
        base_problem,
        meta_prior: LearnedMetaPrior,
        proposal_pool_size=1024,
        refinement_count=128,
    ):
        super().__init__(base_problem, variant="lodo_meta")
        self.meta_prior = meta_prior
        self.problem_name = f"{base_problem.problem_name}_lodo_meta"
        self.proposal_pool_size = int(proposal_pool_size)
        self.refinement_count = int(refinement_count)
        self.prefer_direct_gpr_basis = self.meta_prior.component_stage in {
            "coordinate", "spectral"
        }
        self._gpr_basis_maps = {}

    def admissibility_audit(self):
        out = lodo_meta_prior_audit().to_dict()
        out["meta_prior"] = self.meta_prior.diagnostics()
        return out

    def cumulative_risk_provider_status(self):
        return {
            "status": "available",
            "provider": "LearnedMetaPrior",
            "coordinate": "frozen_source_trained_psi=(A,N)",
            "source_domains": list(self.meta_prior.source_domains),
        }

    def risk_exposures(self, x, output_index=1):
        return self.meta_prior.risk_exposure(self, x, output_index=output_index)

    def risk_class(self, x):
        return self.meta_prior.risk_class(self, x)

    def cumulative_risk_features(self, x, output_index=1):
        return self.meta_prior.cumulative_features(self, x, output_index=output_index)

    def cumulative_risk_feature_names(self, output_index=1):
        del output_index
        return self.meta_prior.cumulative_feature_names()

    def cumulative_hvd_prior_beta(self, output_index=1, feature_dim=None):
        return self.meta_prior.cumulative_hvd_prior_beta(
            output_index=output_index,
            feature_dim=feature_dim,
        )

    def source_calibrated_recommendation_slack(self):
        return self.meta_prior.source_calibrated_recommendation_slack()

    def surrogate_basis_map(self):
        # Recommendation and certification calibration model the constraint
        # boundary, so they must use the same pilot-gated representation as
        # the constraint GPR.  Falling back to an unconditional spectral map
        # here would silently bypass a coordinate gate decision.
        constraint_basis = self._gpr_basis_maps.get(1)
        if constraint_basis is not None:
            return constraint_basis
        return MetaPriorSurrogateBasis(self.meta_prior, self)

    def gpr_basis_map(self, output_index=0):
        output_index = int(output_index)
        if not self.prefer_direct_gpr_basis:
            return None
        if output_index not in self._gpr_basis_maps:
            if self.meta_prior.component_enabled("spectral"):
                basis = PilotGatedMetaPriorBasis(
                    self.meta_prior,
                    self,
                    output_index=output_index,
                )
            else:
                basis = MetaPriorSurrogateBasis(self.meta_prior, self)
            self._gpr_basis_maps[output_index] = basis
        return self._gpr_basis_maps[output_index]

    def meta_basis_diagnostics(self):
        return {
            str(index): (
                basis.diagnostics()
                if hasattr(basis, "diagnostics")
                else {
                    "status": "fixed",
                    "selected_basis": "coordinate",
                    "output_index": int(index),
                }
            )
            for index, basis in self._gpr_basis_maps.items()
        }

    def pilot_constraint_guard(self):
        basis = self._gpr_basis_maps.get(1)
        if basis is None or not hasattr(basis, "certification_guard"):
            return 0.0
        return max(float(basis.certification_guard()), 0.0)

    def source_mean_prior_predict_many(self, xs, output_index=1):
        return self.meta_prior.source_mean_prior_predict_many(
            self,
            xs,
            output_index=output_index,
        )

    def source_mean_prior_sigma(self, output_index=1):
        return self.meta_prior.source_mean_prior_sigma(output_index=output_index)

    def hvd_features(self, x):
        return self.meta_prior.hvd_features(self, x)

    def hvd_residual_variance_cap(self, output_index=0):
        del output_index
        return float(8.0 * max(float(self.sigma_level), 1e-8) ** 2)

    def initial_samples(self, n=5, rng=None):
        return self.meta_prior.proposal_candidates(
            self,
            n=n,
            rng=rng,
            pool_size=self.proposal_pool_size,
        )

    def state_anchor_points(self, n=10, rng=None):
        return self.meta_prior.state_anchor_points(n=n, rng=rng)

    def inverse_state_anchor(self, anchor, rng=None, n=1):
        return self.meta_prior.inverse_state_anchor(
            self,
            anchor,
            rng=rng,
            n=n,
            pool_size=self.proposal_pool_size,
        )

    def recommendation_refinement_candidates(self):
        return self.meta_prior.proposal_candidates(
            self,
            n=self.refinement_count,
            rng=np.random.default_rng(self.meta_prior.seed + 7919),
            pool_size=max(self.proposal_pool_size, 4 * self.refinement_count),
        )
