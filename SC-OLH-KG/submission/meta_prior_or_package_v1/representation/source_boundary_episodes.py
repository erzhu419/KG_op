"""Source-only boundary episodes for frozen alignment admission.

The target gate must not learn whether a representation is useful from target
oracle values.  This module estimates that relation on source domains: each
episode fits on a small, two-sided chance-boundary pilot and is scored on a
disjoint source evaluation set.  At target time only pilot summaries and LOO
predictions are compared with this frozen episode bank.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class AlignmentAdmissionDecision:
    accepted: bool
    diagnostics: dict


class SourceBoundaryEpisodePrior:
    """A conservative local prior for admitting frozen risk coordinates."""

    def __init__(
        self,
        pilot_size=10,
        pilot_sizes=None,
        evaluation_size=24,
        episodes_per_domain=8,
        ridge=0.1,
        min_neighbor_domains=2,
        min_global_win_rate=(2.0 / 3.0),
        min_local_win_rate=0.60,
        max_false_feasible_delta=0.05,
        support_multiplier=1.5,
        seed=123,
    ):
        self.pilot_size = max(6, int(pilot_size))
        if pilot_sizes is None:
            pilot_sizes = (self.pilot_size,)
        self.pilot_sizes = tuple(sorted({
            max(6, int(value)) for value in pilot_sizes
        }))
        self.evaluation_size = max(8, int(evaluation_size))
        self.episodes_per_domain = max(1, int(episodes_per_domain))
        self.ridge = max(float(ridge), 1e-10)
        self.min_neighbor_domains = max(1, int(min_neighbor_domains))
        self.min_global_win_rate = float(np.clip(
            min_global_win_rate, 0.5, 1.0))
        self.min_local_win_rate = float(np.clip(
            min_local_win_rate, 0.5, 1.0))
        self.max_false_feasible_delta = max(
            float(max_false_feasible_delta), 0.0)
        self.support_multiplier = max(float(support_multiplier), 1.0)
        self.seed = int(seed)

        self.descriptor_mean_: np.ndarray | None = None
        self.descriptor_scale_: np.ndarray | None = None
        self.descriptors_: np.ndarray | None = None
        self.gains_: np.ndarray | None = None
        self.false_feasible_deltas_: np.ndarray | None = None
        self.domains_: np.ndarray | None = None
        self.support_radius_ = 0.0
        self.episode_rows_: list[dict] = []
        self.diagnostics_: dict = {"status": "unfit"}

    def fit(self, domains, margins, baseline_features, aligned_features):
        domains = np.asarray(domains, dtype=str).reshape(-1)
        margins = np.asarray(margins, dtype=float).reshape(-1)
        baseline = np.asarray(baseline_features, dtype=float)
        aligned = np.asarray(aligned_features, dtype=float)
        n = len(domains)
        if (
            len(margins) != n
            or baseline.ndim != 2
            or aligned.ndim != 2
            or len(baseline) != n
            or len(aligned) != n
        ):
            raise ValueError("source episode arrays must have matching rows")
        if not (
            np.all(np.isfinite(margins))
            and np.all(np.isfinite(baseline))
            and np.all(np.isfinite(aligned))
        ):
            raise ValueError("source episode arrays must be finite")

        rows = []
        skipped = {}
        for domain in sorted(set(domains.tolist())):
            indices = np.where(domains == domain)[0]
            feasible = indices[margins[indices] <= 0.0]
            infeasible = indices[margins[indices] > 0.0]
            minimum_side = 3
            if len(feasible) < minimum_side or len(infeasible) < minimum_side:
                skipped[domain] = {
                    "reason": "insufficient_two_sided_source_records",
                    "n_feasible": int(len(feasible)),
                    "n_infeasible": int(len(infeasible)),
                }
                continue
            domain_seed = self._stable_seed(domain)
            rng = np.random.default_rng(domain_seed)
            for episode in range(self.episodes_per_domain):
                split = self._episode_split(
                    feasible, infeasible, margins, episode, rng)
                if split is None:
                    continue
                pilot, evaluation = split
                base_loo = self._ridge_loo(
                    baseline[pilot], margins[pilot])
                aligned_loo = self._ridge_loo(
                    aligned[pilot], margins[pilot])
                base_pilot_score = self._decision_score(
                    margins[pilot], base_loo)
                aligned_pilot_score = self._decision_score(
                    margins[pilot], aligned_loo)
                base_eval_prediction = self._ridge_predict(
                    baseline[pilot], margins[pilot], baseline[evaluation])
                aligned_eval_prediction = self._ridge_predict(
                    aligned[pilot], margins[pilot], aligned[evaluation])
                base_eval_score = self._decision_score(
                    margins[evaluation], base_eval_prediction)
                aligned_eval_score = self._decision_score(
                    margins[evaluation], aligned_eval_prediction)
                descriptor = self._episode_descriptor(
                    margins[pilot], base_pilot_score, aligned_pilot_score)
                rows.append({
                    "domain": str(domain),
                    "episode": int(episode),
                    "pilot_indices": [int(value) for value in pilot],
                    "evaluation_indices": [int(value) for value in evaluation],
                    "pilot_evaluation_disjoint": bool(
                        set(pilot).isdisjoint(set(evaluation))),
                    "n_pilot_feasible": int(np.sum(margins[pilot] <= 0.0)),
                    "n_pilot_infeasible": int(np.sum(margins[pilot] > 0.0)),
                    "descriptor": descriptor.tolist(),
                    "pilot_gain": float(
                        base_pilot_score["total"]
                        - aligned_pilot_score["total"]),
                    "evaluation_gain": float(
                        base_eval_score["total"]
                        - aligned_eval_score["total"]),
                    "baseline_false_feasible_rate": float(
                        base_eval_score["false_feasible_rate"]),
                    "aligned_false_feasible_rate": float(
                        aligned_eval_score["false_feasible_rate"]),
                    "false_feasible_delta": float(
                        aligned_eval_score["false_feasible_rate"]
                        - base_eval_score["false_feasible_rate"]),
                })

        self.episode_rows_ = rows
        if not rows:
            self.diagnostics_ = {
                "status": "insufficient_source_boundary_episodes",
                "skipped_domains": skipped,
                "target_data_used": False,
                "target_oracle_used": False,
            }
            return self

        self.descriptors_ = np.vstack([row["descriptor"] for row in rows])
        self.gains_ = np.asarray([
            row["evaluation_gain"] for row in rows], dtype=float)
        self.false_feasible_deltas_ = np.asarray([
            row["false_feasible_delta"] for row in rows], dtype=float)
        self.domains_ = np.asarray([row["domain"] for row in rows], dtype=str)
        self.descriptor_mean_ = np.mean(self.descriptors_, axis=0)
        self.descriptor_scale_ = np.std(self.descriptors_, axis=0)
        self.descriptor_scale_ = np.where(
            self.descriptor_scale_ < 1e-8, 1.0, self.descriptor_scale_)
        standardized = self._standardize(self.descriptors_)
        cross_domain_distance = []
        for index in range(len(standardized)):
            other = self.domains_ != self.domains_[index]
            if np.any(other):
                cross_domain_distance.append(float(np.min(np.linalg.norm(
                    standardized[other] - standardized[index], axis=1))))
        self.support_radius_ = (
            float(np.quantile(cross_domain_distance, 0.90))
            * self.support_multiplier
            if cross_domain_distance else 0.0
        )
        by_domain = {}
        for domain in sorted(set(self.domains_.tolist())):
            mask = self.domains_ == domain
            by_domain[domain] = {
                "n_episodes": int(np.sum(mask)),
                "median_evaluation_gain": float(np.median(self.gains_[mask])),
                "win_rate": float(np.mean(self.gains_[mask] > 0.0)),
                "median_false_feasible_delta": float(np.median(
                    self.false_feasible_deltas_[mask])),
            }
        self.diagnostics_ = {
            "status": "fit",
            "method": "source_only_disjoint_boundary_episode_admission",
            "n_episodes": int(len(rows)),
            "n_domains": int(len(set(self.domains_.tolist()))),
            "pilot_size": int(self.pilot_size),
            "pilot_sizes": [int(value) for value in self.pilot_sizes],
            "evaluation_size": int(self.evaluation_size),
            "episodes_per_domain": int(self.episodes_per_domain),
            "median_evaluation_gain": float(np.median(self.gains_)),
            "evaluation_win_rate": float(np.mean(self.gains_ > 0.0)),
            "median_false_feasible_delta": float(np.median(
                self.false_feasible_deltas_)),
            "support_radius": float(self.support_radius_),
            "all_splits_disjoint": bool(all(
                row["pilot_evaluation_disjoint"] for row in rows)),
            "by_domain": by_domain,
            "skipped_domains": skipped,
            "target_data_used": False,
            "target_oracle_used": False,
            "fingerprint": self.fingerprint(),
        }
        return self

    def admit(self, margins, baseline_loo, aligned_loo):
        margins = np.asarray(margins, dtype=float).reshape(-1)
        baseline_loo = np.asarray(baseline_loo, dtype=float).reshape(-1)
        aligned_loo = np.asarray(aligned_loo, dtype=float).reshape(-1)
        reasons = []
        if self.diagnostics_.get("status") != "fit":
            reasons.append("source_episode_prior_unavailable")
            return AlignmentAdmissionDecision(False, {
                "status": "rejected",
                "rejection_reasons": reasons,
                "source_prior_status": self.diagnostics_.get("status"),
                "target_oracle_used": False,
            })
        if not (
            len(margins) == len(baseline_loo) == len(aligned_loo)
            and len(margins) >= 6
            and np.all(np.isfinite(margins))
            and np.all(np.isfinite(baseline_loo))
            and np.all(np.isfinite(aligned_loo))
        ):
            reasons.append("invalid_target_pilot_summary")
            return AlignmentAdmissionDecision(False, {
                "status": "rejected",
                "rejection_reasons": reasons,
                "target_oracle_used": False,
            })

        base_score = self._decision_score(margins, baseline_loo)
        aligned_score = self._decision_score(margins, aligned_loo)
        descriptor = self._episode_descriptor(
            margins, base_score, aligned_score)
        standardized = self._standardize(descriptor.reshape(1, -1))[0]
        source = self._standardize(self.descriptors_)
        distance = np.linalg.norm(source - standardized, axis=1)
        order = np.argsort(distance, kind="stable")
        neighbor_count = min(max(6, 2 * self.min_neighbor_domains), len(order))
        neighbors = order[:neighbor_count]
        local_distance = distance[neighbors]
        bandwidth = max(float(np.median(local_distance)), 0.25)
        weight = np.exp(-0.5 * (local_distance / bandwidth) ** 2)
        weight /= max(float(np.sum(weight)), 1e-12)
        local_gain = self.gains_[neighbors]
        local_false_delta = self.false_feasible_deltas_[neighbors]
        predicted_gain = float(np.sum(weight * local_gain))
        lower_gain = float(self._weighted_quantile(
            local_gain, weight, 0.25))
        local_win_rate = float(np.sum(weight * (local_gain > 0.0)))
        upper_false_delta = float(self._weighted_quantile(
            local_false_delta, weight, 0.75))
        neighbor_domains = sorted(set(self.domains_[neighbors].tolist()))
        nearest_distance = float(local_distance[0])
        pilot_gain = float(base_score["total"] - aligned_score["total"])
        pilot_false_delta = float(
            aligned_score["false_feasible_rate"]
            - base_score["false_feasible_rate"])

        global_win_rate = float(self.diagnostics_.get(
            "evaluation_win_rate", 0.0))
        if global_win_rate + 1e-12 < self.min_global_win_rate:
            reasons.append("insufficient_global_source_win_rate")
        if nearest_distance > self.support_radius_:
            reasons.append("outside_source_episode_support")
        if len(neighbor_domains) < self.min_neighbor_domains:
            reasons.append("insufficient_neighbor_domains")
        if lower_gain <= 0.0:
            reasons.append("nonpositive_source_gain_lower_quartile")
        if local_win_rate < self.min_local_win_rate:
            reasons.append("insufficient_source_local_win_rate")
        if upper_false_delta > self.max_false_feasible_delta:
            reasons.append("source_false_feasible_risk")
        if pilot_false_delta > self.max_false_feasible_delta:
            reasons.append("target_pilot_false_feasible_risk")
        accepted = not reasons
        return AlignmentAdmissionDecision(accepted, {
            "status": "accepted" if accepted else "rejected",
            "rejection_reasons": reasons,
            "descriptor": descriptor.tolist(),
            "nearest_source_distance": nearest_distance,
            "source_support_radius": float(self.support_radius_),
            "neighbor_domains": neighbor_domains,
            "neighbor_count": int(len(neighbors)),
            "predicted_evaluation_gain": predicted_gain,
            "source_gain_lower_quartile": lower_gain,
            "source_local_win_rate": local_win_rate,
            "source_global_win_rate": global_win_rate,
            "minimum_source_global_win_rate": float(
                self.min_global_win_rate),
            "source_false_feasible_delta_upper_quartile": upper_false_delta,
            "target_pilot_gain": pilot_gain,
            "target_pilot_false_feasible_delta": pilot_false_delta,
            "target_data_used": True,
            "target_oracle_used": False,
        })

    def diagnostics(self):
        return dict(self.diagnostics_)

    def fingerprint(self):
        payload = {
            "config": {
                "pilot_size": self.pilot_size,
                "pilot_sizes": self.pilot_sizes,
                "evaluation_size": self.evaluation_size,
                "episodes_per_domain": self.episodes_per_domain,
                "ridge": self.ridge,
                "seed": self.seed,
            },
            "episodes": [
                {
                    "domain": row["domain"],
                    "episode": row["episode"],
                    "descriptor": row["descriptor"],
                    "evaluation_gain": row["evaluation_gain"],
                    "false_feasible_delta": row["false_feasible_delta"],
                }
                for row in self.episode_rows_
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

    def _episode_split(self, feasible, infeasible, margins, episode, rng):
        pilot_size = self.pilot_sizes[episode % len(self.pilot_sizes)]
        ratio_index = (episode // len(self.pilot_sizes)) % 3
        if ratio_index == 0:
            n_pilot_feasible = 1
        elif ratio_index == 1:
            n_pilot_feasible = max(1, int(round(0.10 * pilot_size)))
        else:
            n_pilot_feasible = max(2, int(round(0.20 * pilot_size)))
        n_pilot_feasible = int(np.clip(
            n_pilot_feasible, 1, pilot_size - 2))
        n_pilot_infeasible = pilot_size - n_pilot_feasible
        n_eval_feasible = max(2, self.evaluation_size // 2)
        n_eval_infeasible = self.evaluation_size - n_eval_feasible
        if (
            len(feasible) < n_pilot_feasible + n_eval_feasible
            or len(infeasible) < n_pilot_infeasible + n_eval_infeasible
        ):
            n_eval_feasible = min(
                n_eval_feasible, len(feasible) - n_pilot_feasible)
            n_eval_infeasible = min(
                n_eval_infeasible, len(infeasible) - n_pilot_infeasible)
        if n_eval_feasible < 2 or n_eval_infeasible < 2:
            return None

        pilot_feasible = self._boundary_sample(
            feasible, n_pilot_feasible, margins, rng)
        pilot_infeasible = self._boundary_sample(
            infeasible, n_pilot_infeasible, margins, rng)
        pilot = np.concatenate([pilot_feasible, pilot_infeasible])
        remaining_feasible = np.setdiff1d(
            feasible, pilot_feasible, assume_unique=False)
        remaining_infeasible = np.setdiff1d(
            infeasible, pilot_infeasible, assume_unique=False)
        evaluation = np.concatenate([
            self._boundary_sample(
                remaining_feasible, n_eval_feasible, margins, rng),
            self._boundary_sample(
                remaining_infeasible, n_eval_infeasible, margins, rng),
        ])
        rng.shuffle(pilot)
        rng.shuffle(evaluation)
        return pilot.astype(int), evaluation.astype(int)

    @staticmethod
    def _boundary_sample(indices, count, margins, rng):
        indices = np.asarray(indices, dtype=int)
        if count >= len(indices):
            return indices.copy()
        distance = np.abs(np.asarray(margins, dtype=float)[indices])
        scale = max(float(np.median(distance)), 1e-6)
        probability = np.exp(-distance / scale) + 0.10
        probability /= float(np.sum(probability))
        return np.asarray(rng.choice(
            indices, size=int(count), replace=False, p=probability), dtype=int)

    def _ridge_loo(self, features, target):
        features = np.asarray(features, dtype=float)
        target = np.asarray(target, dtype=float)
        out = np.empty(len(target), dtype=float)
        for heldout in range(len(target)):
            keep = np.arange(len(target)) != heldout
            out[heldout] = self._ridge_predict(
                features[keep], target[keep], features[heldout:heldout + 1])[0]
        return out

    def _ridge_predict(self, train_x, train_y, test_x):
        train_x = np.asarray(train_x, dtype=float)
        train_y = np.asarray(train_y, dtype=float).reshape(-1)
        test_x = np.asarray(test_x, dtype=float)
        mean = np.mean(train_x, axis=0)
        scale = np.std(train_x, axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        X = (train_x - mean) / scale
        X_test = (test_x - mean) / scale
        X = np.column_stack([np.ones(len(X)), X])
        X_test = np.column_stack([np.ones(len(X_test)), X_test])
        penalty = self.ridge * np.eye(X.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        try:
            beta = np.linalg.solve(X.T @ X + penalty, X.T @ train_y)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(
                X.T @ X + penalty, X.T @ train_y, rcond=None)[0]
        return np.asarray(X_test @ beta, dtype=float)

    @staticmethod
    def _decision_score(truth_margin, predicted_margin):
        truth = np.asarray(truth_margin, dtype=float).reshape(-1)
        prediction = np.asarray(predicted_margin, dtype=float).reshape(-1)
        robust = 0.7413 * float(
            np.quantile(truth, 0.75) - np.quantile(truth, 0.25))
        scale = max(robust, 0.25, 1e-8)
        weight = 1.0 + 2.0 * np.exp(-0.5 * (truth / scale) ** 2)
        boundary_mse = float(np.average(
            ((truth - prediction) / scale) ** 2, weights=weight))
        feasible = truth <= 0.0
        infeasible = ~feasible
        predicted_feasible = prediction <= 0.0
        false_feasible = (
            float(np.mean(predicted_feasible[infeasible]))
            if np.any(infeasible) else 0.0)
        false_infeasible = (
            float(np.mean(~predicted_feasible[feasible]))
            if np.any(feasible) else 0.0)
        dangerous = float(np.average(
            (np.maximum(truth - prediction, 0.0) / scale) ** 2,
            weights=weight,
        ))
        total = (
            0.30 * boundary_mse
            + 3.0 * false_feasible
            + 0.25 * false_infeasible
            + 0.35 * dangerous
        )
        return {
            "total": float(total),
            "boundary_mse": boundary_mse,
            "false_feasible_rate": false_feasible,
            "false_infeasible_rate": false_infeasible,
            "dangerous_underprediction": dangerous,
        }

    @staticmethod
    def _episode_descriptor(margins, baseline_score, aligned_score):
        margins = np.asarray(margins, dtype=float).reshape(-1)
        robust = 0.7413 * float(
            np.quantile(margins, 0.75) - np.quantile(margins, 0.25))
        scale = max(robust, 0.25, 1e-8)
        normalized = np.clip(margins / scale, -8.0, 8.0)
        base_total = float(baseline_score["total"])
        aligned_total = float(aligned_score["total"])
        relative_gain = (base_total - aligned_total) / max(abs(base_total), 1.0)
        false_delta = float(
            aligned_score["false_feasible_rate"]
            - baseline_score["false_feasible_rate"])
        return np.asarray([
            float(np.mean(margins <= 0.0)),
            float(np.median(normalized)),
            float(np.quantile(normalized, 0.75) - np.quantile(normalized, 0.25)),
            float(np.min(normalized)),
            float(np.max(normalized)),
            float(relative_gain),
            float(false_delta),
            float(aligned_score["dangerous_underprediction"]
                  - baseline_score["dangerous_underprediction"]),
        ], dtype=float)

    def _standardize(self, values):
        values = np.asarray(values, dtype=float)
        return (values - self.descriptor_mean_) / self.descriptor_scale_

    @staticmethod
    def _weighted_quantile(values, weights, quantile):
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        order = np.argsort(values, kind="stable")
        values = values[order]
        weights = weights[order]
        cumulative = np.cumsum(weights)
        cutoff = float(np.clip(quantile, 0.0, 1.0)) * float(cumulative[-1])
        index = int(np.searchsorted(cumulative, cutoff, side="left"))
        return float(values[min(index, len(values) - 1)])

    def _stable_seed(self, domain):
        digest = hashlib.sha256(str(domain).encode("utf-8")).digest()
        offset = int.from_bytes(digest[:8], "little")
        return int((self.seed + offset) % (2**63 - 1))
