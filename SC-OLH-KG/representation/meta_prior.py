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
    tau: float
    alpha: float
    sigma_level: float


class LearnedMetaPrior:
    """Frozen source-trained prior used by held-out target adapters."""

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
        self.seed = int(seed)
        self.feature_mean = None
        self.feature_scale = None
        self.pca_components = None
        self.cluster_centers = None
        self.anchor_psi = np.empty((0, self.local_dim + self.shared_dim))
        self.anchor_scores = np.empty(0, dtype=float)
        self.anchor_meta = []
        self.beta_prior = {}
        self.source_domains = []
        self.n_records = 0
        self.fit_status = "unfit"
        self.training_diagnostics = {}

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

    def _fit_scaler_pca(self, descriptors):
        X = np.vstack(descriptors)
        self.feature_mean = np.mean(X, axis=0)
        self.feature_scale = np.std(X, axis=0)
        self.feature_scale = np.where(self.feature_scale < 1e-8, 1.0, self.feature_scale)
        Z = (X - self.feature_mean) / self.feature_scale
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
                records.append(SourceRecord(
                    domain=str(domain_name),
                    x=x,
                    y=y,
                    descriptor=self.descriptor(problem, x),
                    tau=float(getattr(problem, "tau", 0.0)),
                    alpha=float(getattr(problem, "alpha", 0.05)),
                    sigma_level=float(getattr(problem, "sigma_level", 0.04)),
                ))
        return records

    @staticmethod
    def _source_margin(rec):
        z = norm.ppf(1.0 - float(rec.alpha))
        return float(rec.y[1]) + z * float(rec.sigma_level) - float(rec.tau)

    @staticmethod
    def _source_margin_scale(rec):
        return max(abs(float(rec.tau)), float(rec.sigma_level), 1e-6)

    def _boundary_sample_weight(self, rec):
        margin = self._source_margin(rec)
        scaled = margin / self._source_margin_scale(rec)
        temp = max(float(self.boundary_temperature), 1e-6)
        boundary = np.exp(-0.5 * (scaled / temp) ** 2)
        violation = max(float(scaled), 0.0)
        return float(1.0 + self.boundary_weight * boundary + 0.25 * violation)

    def _record_training_diagnostics(self, records, weights):
        margins = np.asarray([self._source_margin(rec) for rec in records], dtype=float)
        scaled = np.asarray([
            self._source_margin(rec) / self._source_margin_scale(rec)
            for rec in records
        ], dtype=float)
        weights = np.asarray(weights, dtype=float)
        feasible = margins <= 0.0
        self.training_diagnostics = {
            "source_feasible_rate": float(np.mean(feasible)) if len(feasible) else None,
            "source_margin_mean": float(np.mean(margins)) if len(margins) else None,
            "source_margin_median": float(np.median(margins)) if len(margins) else None,
            "source_scaled_margin_median_abs": (
                float(np.median(np.abs(scaled))) if len(scaled) else None
            ),
            "boundary_weight_mean": float(np.mean(weights)) if len(weights) else None,
            "boundary_weight_max": float(np.max(weights)) if len(weights) else None,
            "boundary_temperature": float(self.boundary_temperature),
            "boundary_weight": float(self.boundary_weight),
            "variance_weight": float(self.variance_weight),
            "feasible_penalty": float(self.feasible_penalty),
            "feasible_bonus": float(self.feasible_bonus),
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
        Z = self._fit_scaler_pca(descriptors)
        self._fit_kmeans(Z)
        self.fit_status = "fitting"
        weights = [self._boundary_sample_weight(rec) for rec in records]
        self._record_training_diagnostics(records, weights)
        self._fit_hvd_beta_priors(records)
        self._fit_anchor_distribution(records)
        self.fit_status = "fit"
        return self

    def _fit_hvd_beta_priors(self, records):
        by_domain = {}
        for rec in records:
            by_domain.setdefault(rec.domain, []).append(rec)
        beta_by_output = {0: [], 1: []}
        for domain_records in by_domain.values():
            X_desc = np.vstack([
                np.concatenate([[1.0], self._scaled_descriptor(rec.descriptor)])
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
                resid2 = np.maximum((y - X_desc @ beta_mean) ** 2, 1e-10)
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
            })

    def state_anchor_points(self, n=10, rng=None):
        rng = rng or np.random.default_rng(self.seed)
        if len(self.anchor_psi) == 0:
            return []
        order = rng.permutation(len(self.anchor_psi))
        anchors = []
        for pos in order[: max(0, int(n))]:
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

    def inverse_state_anchor(self, problem, anchor, rng=None, n=1, pool_size=512):
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
        rng = rng or np.random.default_rng(self.seed)
        rows = []
        for anchor in self.state_anchor_points(n=max(1, int(n)), rng=rng):
            rows.extend(self.inverse_state_anchor(
                problem,
                anchor,
                rng=rng,
                n=1,
                pool_size=max(64, int(pool_size) // max(1, int(n))),
            ))
        while len(rows) < int(n):
            rows.append(problem.sample_random(rng))
        return unique_candidates(rows)[: max(0, int(n))]

    def cumulative_hvd_prior_beta(self, output_index=1, feature_dim=None):
        beta = self.beta_prior.get(int(output_index))
        if beta is None:
            return None
        beta = np.asarray(beta, dtype=float)
        if feature_dim is not None and len(beta) != int(feature_dim):
            return None
        return beta.copy()

    def diagnostics(self):
        return {
            "status": self.fit_status,
            "source_domains": list(self.source_domains),
            "n_records": int(self.n_records),
            "local_dim": int(self.local_dim),
            "shared_dim": int(self.shared_dim),
            "n_anchors": int(len(self.anchor_psi)),
            "has_beta_prior": {
                str(key): value is not None
                for key, value in self.beta_prior.items()
            },
            "training": dict(self.training_diagnostics),
        }

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
