"""Source-trained equivariant alignment of observable channel roles."""

from __future__ import annotations

import copy
from collections import defaultdict
from itertools import permutations
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

from representation.observable_exposure import (
    MAX_OBSERVABLE_CHANNELS,
    ObservableStateExposure,
    as_observable_state_exposure,
    get_observable_state_exposure,
    partially_aligned_observable_state_descriptor,
    role_aligned_observable_state_descriptor,
)


def _safe_correlation(left, right):
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(left @ right / denominator)


def _profile_response_library(profiles):
    """Dimension-equivariant low-frequency interventions for policy rows."""

    rows = [np.asarray(value, dtype=float).reshape(-1) for value in profiles]
    if not rows or any(len(value) != len(rows[0]) for value in rows):
        raise ValueError("policy profiles must have one stable dimension")
    d = len(rows[0])
    if d <= 0:
        raise ValueError("policy profiles cannot be empty")
    matrix = np.vstack(rows)
    positions = (np.arange(d, dtype=float) + 0.5) / float(d)
    atoms = [
        np.ones(d, dtype=float),
        2.0 * positions - 1.0,
    ]
    for frequency in range(1, 4):
        atoms.extend([
            np.cos(np.pi * frequency * positions),
            np.sin(np.pi * frequency * positions),
        ])
    atoms = np.vstack(atoms)
    atoms /= np.maximum(
        np.sqrt(np.mean(atoms ** 2, axis=1, keepdims=True)), 1e-12)
    projected = matrix @ atoms.T / float(d)
    projected = np.column_stack([
        projected,
        np.std(matrix, axis=1),
        np.mean(np.abs(np.diff(matrix, axis=1)), axis=1)
        if d > 1 else np.zeros(len(matrix), dtype=float),
    ])
    center = np.mean(projected, axis=0)
    scale = np.std(projected, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (projected - center) / scale


def _intervention_response_signatures(exposures, profiles):
    """Identify channel roles by observable responses to policy interventions."""

    rows = [as_observable_state_exposure(value) for value in exposures]
    if not rows or any(value is None for value in rows):
        raise ValueError("response signatures require observable exposures")
    count = len(rows[0].channel_means)
    if any(len(value.channel_means) != count for value in rows):
        raise ValueError("a domain's observable channel count must be stable")
    intervention = _profile_response_library(profiles)
    if len(intervention) != len(rows):
        raise ValueError("policy profiles and observable exposures must align")
    design = np.column_stack([np.ones(len(intervention)), intervention])
    gram = design.T @ design
    ridge = 1e-3 * max(float(np.trace(gram)) / len(gram), 1e-8)
    penalty = ridge * np.eye(len(gram), dtype=float)
    penalty[0, 0] = 0.0
    solver = np.linalg.pinv(gram + penalty) @ design.T
    means = np.vstack([value.channel_means for value in rows])
    signatures = []
    for channel in range(count):
        mean_response = solver @ means[:, channel]
        mean_slope = mean_response[1:]
        mean_norm = float(np.linalg.norm(mean_slope))
        signatures.append(np.asarray([
            *(mean_slope / max(mean_norm, 1e-12)),
            np.log1p(mean_norm),
        ], dtype=float))
    result = np.vstack(signatures)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(
            "observable intervention-response signatures are non-finite")
    return result


def _channel_signatures(exposures, profiles=None, mode="distribution"):
    mode = str(mode or "distribution").strip().lower().replace("-", "_")
    if mode == "intervention_response":
        if profiles is None:
            raise ValueError(
                "intervention-response signatures require policy profiles")
        return _intervention_response_signatures(exposures, profiles)
    if mode != "distribution":
        raise ValueError(
            "channel signature mode must be distribution or "
            "intervention_response")
    rows = [as_observable_state_exposure(value) for value in exposures]
    if not rows or any(value is None for value in rows):
        raise ValueError("channel signatures require observable exposures")
    count = len(rows[0].channel_means)
    if count <= 0 or count > MAX_OBSERVABLE_CHANNELS:
        raise ValueError("observable channel count is outside the supported range")
    if any(len(value.channel_means) != count for value in rows):
        raise ValueError("a domain's observable channel count must be stable")
    means = np.vstack([value.channel_means for value in rows])
    scales = np.vstack([value.channel_scales for value in rows])
    global_mean = np.mean(means, axis=1)
    global_scale = np.mean(scales, axis=1)
    signatures = []
    for channel in range(count):
        channel_mean = means[:, channel]
        channel_scale = scales[:, channel]
        signatures.append(np.asarray([
            *np.quantile(channel_mean, [0.10, 0.50, 0.90]),
            float(np.std(channel_mean)),
            *np.quantile(channel_scale, [0.10, 0.50, 0.90]),
            float(np.std(channel_scale)),
            float(np.mean(channel_mean - global_mean)),
            float(np.mean(channel_scale - global_scale)),
            _safe_correlation(channel_mean, global_mean),
            _safe_correlation(channel_scale, global_scale),
        ], dtype=float))
    result = np.vstack(signatures)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("observable channel signatures are non-finite")
    return result


class EquivariantChannelRoleAligner:
    """Match domain channels to source-learned canonical roles.

    Role prototypes use only observable channel distributions. Source chance
    margins orient the otherwise arbitrary role numbering, but target matching
    uses an unlabeled deterministic policy pool and never reads target outcomes.
    """

    def __init__(
        self,
        target_pool_size=64,
        max_iter=25,
        seed=123,
        partial_transport=False,
        signature_mode="distribution",
        barycentric_transport=False,
        transport_temperature_grid=(0.05, 0.10, 0.25, 0.50, 1.0, 2.0),
    ):
        self.target_pool_size = max(int(target_pool_size), 16)
        self.max_iter = max(int(max_iter), 1)
        self.seed = int(seed)
        self.partial_transport = bool(partial_transport)
        self.signature_mode = str(
            signature_mode or "distribution"
        ).strip().lower().replace("-", "_")
        if self.signature_mode not in {
            "distribution", "intervention_response"
        }:
            raise ValueError(
                "channel signature mode must be distribution or "
                "intervention_response")
        self.barycentric_transport = bool(barycentric_transport)
        self.transport_temperature_grid = tuple(sorted(set(
            max(float(value), 1e-6)
            for value in transport_temperature_grid
        )))
        if not self.transport_temperature_grid:
            raise ValueError("transport temperature grid must be non-empty")
        self.n_roles = 0
        self.signature_mean = None
        self.signature_scale = None
        self.prototypes = None
        self.source_assignments = {}
        self.source_transport_weights = {}
        self.role_relevance = None
        self.role_relevance_fisher_mean = None
        self.role_relevance_fisher_variance = None
        self.role_relevance_effective_count = None
        self.role_relevance_source_sample_sizes = None
        self.source_assignment_temperature = 1.0
        self.transport_temperature = None
        self.transport_selection_diagnostics = {"status": "disabled"}
        self.source_diagnostics = {"status": "unfit"}
        self._target_cache = {}

    @staticmethod
    def _assignment(signatures, prototypes):
        cost = np.sum(
            (signatures[:, None, :] - prototypes[None, :, :]) ** 2,
            axis=2,
        )
        channels, roles = linear_sum_assignment(cost)
        assignment = np.full(len(signatures), -1, dtype=int)
        assignment[channels] = roles
        return assignment, float(np.sum(cost[channels, roles]))

    @staticmethod
    def _assignment_loss_table(signatures, prototypes, assignments=None):
        """Enumerate injection losses in a channel-permutation-equivariant way."""

        signatures = np.asarray(signatures, dtype=float)
        prototypes = np.asarray(prototypes, dtype=float)
        if signatures.ndim != 2 or prototypes.ndim != 2:
            raise ValueError("role signatures and prototypes must be matrices")
        if signatures.shape[1] != prototypes.shape[1]:
            raise ValueError("role signatures and prototypes must align")
        channel_count = int(len(signatures))
        role_count = int(len(prototypes))
        if channel_count <= 0 or channel_count > role_count:
            raise ValueError("role assignment requires 0 < channels <= roles")
        if assignments is None:
            assignments = permutations(range(role_count), channel_count)
        cost = np.sum(
            (signatures[:, None, :] - prototypes[None, :, :]) ** 2,
            axis=2,
        )
        rows = []
        for assignment in assignments:
            assignment = tuple(int(value) for value in assignment)
            if (
                len(assignment) != channel_count
                or len(set(assignment)) != channel_count
                or min(assignment) < 0
                or max(assignment) >= role_count
            ):
                raise ValueError("role assignment must be an injection")
            loss = float(np.sum(cost[
                np.arange(channel_count, dtype=int),
                np.asarray(assignment, dtype=int),
            ]))
            rows.append((assignment, loss))
        if not rows:
            raise ValueError("role assignment table cannot be empty")
        return rows

    @staticmethod
    def _partial_assignment(
        signatures, prototypes, temperature, *, barycentric=False,
    ):
        """Entropic partial matching with explicit missing-role mass.

        Dummy rows complete a rectangular channel-role problem to a square
        transport problem. Sinkhorn scaling then gives each observed channel
        unit mass and caps every role at unit observed mass. Removing the dummy
        rows preserves a continuous missing-role mask.
        """

        signatures = np.asarray(signatures, dtype=float)
        prototypes = np.asarray(prototypes, dtype=float)
        if signatures.ndim != 2 or prototypes.ndim != 2:
            raise ValueError("partial role matching requires matrices")
        if signatures.shape[1] != prototypes.shape[1]:
            raise ValueError("channel signatures and role prototypes disagree")
        n_channels = int(len(signatures))
        n_roles = int(len(prototypes))
        if n_channels <= 0 or n_channels > n_roles:
            raise ValueError(
                "partial role matching requires 1 <= channels <= roles")
        cost = np.sum(
            (signatures[:, None, :] - prototypes[None, :, :]) ** 2,
            axis=2,
        )
        if n_channels == n_roles:
            channels, roles = linear_sum_assignment(cost)
            observed = np.zeros((n_channels, n_roles), dtype=float)
            observed[channels, roles] = 1.0
            return observed, {
                "expected_matching_cost": float(
                    np.sum(observed * cost) / n_channels),
                "normalized_assignment_entropy": 0.0,
                "observed_role_mass": np.ones(n_roles).tolist(),
                "missing_role_mass": 0.0,
                "solver_status": "square_hard_assignment",
                "optimizer_success": True,
                "optimizer_candidate_feasible": True,
                "optimizer_row_error": 0.0,
                "optimizer_column_excess": 0.0,
                "transport_geometry": (
                    "barycentric_response" if barycentric else "pairwise_cost"),
            }
        temperature = max(float(temperature), 1e-8)
        size = n_channels * n_roles
        initial = np.full(size, 1.0 / n_roles, dtype=float)

        def objective(flat):
            values = np.maximum(np.asarray(flat, dtype=float), 1e-15)
            if barycentric:
                weights = values.reshape(n_channels, n_roles)
                residual = weights @ prototypes - signatures
                fit_loss = float(np.sum(residual ** 2))
            else:
                fit_loss = float(cost.reshape(-1) @ values)
            return float(
                fit_loss
                + temperature * np.sum(values * np.log(values)))

        def gradient(flat):
            values = np.maximum(np.asarray(flat, dtype=float), 1e-15)
            if barycentric:
                weights = values.reshape(n_channels, n_roles)
                residual = weights @ prototypes - signatures
                fit_gradient = 2.0 * residual @ prototypes.T
                fit_gradient = fit_gradient.reshape(-1)
            else:
                fit_gradient = cost.reshape(-1)
            return fit_gradient + temperature * (np.log(values) + 1.0)

        constraints = [
            {
                "type": "eq",
                "fun": lambda flat, channel=channel: (
                    np.sum(np.asarray(flat).reshape(
                        n_channels, n_roles)[channel]) - 1.0),
            }
            for channel in range(n_channels)
        ]
        constraints.extend({
            "type": "ineq",
            "fun": lambda flat, role=role: (
                1.0 - np.sum(np.asarray(flat).reshape(
                    n_channels, n_roles)[:, role])),
        } for role in range(n_roles))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Values in x were outside bounds during a minimize step",
                category=RuntimeWarning,
            )
            fitted = minimize(
                objective,
                initial,
                jac=gradient,
                method="SLSQP",
                bounds=[(1e-12, 1.0)] * size,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-12, "disp": False},
            )
        candidate = np.asarray(fitted.x, dtype=float).reshape(
            n_channels, n_roles)
        candidate_finite = bool(np.all(np.isfinite(candidate)))
        candidate_nonnegative = bool(
            candidate_finite and float(np.min(candidate)) >= -1e-8)
        candidate_row_error = (
            float(np.max(np.abs(np.sum(candidate, axis=1) - 1.0)))
            if candidate_finite else float("inf")
        )
        candidate_column_excess = (
            max(float(np.max(np.sum(candidate, axis=0)) - 1.0), 0.0)
            if candidate_finite else float("inf")
        )
        candidate_feasible = bool(
            candidate_nonnegative
            and candidate_row_error <= 1e-6
            and candidate_column_excess <= 1e-6
        )
        if not fitted.success and not candidate_feasible:
            # A deterministic rectangular assignment is always feasible and
            # is the zero-temperature limit of the same objective.
            channels, roles = linear_sum_assignment(cost)
            observed = np.zeros((n_channels, n_roles), dtype=float)
            observed[channels, roles] = 1.0
            solver_status = "hard_assignment_fallback"
        else:
            observed = np.maximum(
                candidate,
                0.0,
            )
            observed /= np.maximum(
                np.sum(observed, axis=1, keepdims=True), 1e-300)
            solver_status = "entropic_partial_transport"
        role_mass = np.sum(observed, axis=0)
        if float(np.max(role_mass)) > 1.0 + 1e-6:
            raise FloatingPointError(
                "partial role matching violated role capacity")
        expected_cost = float(
            np.sum((observed @ prototypes - signatures) ** 2) / n_channels
            if barycentric
            else np.sum(observed * cost) / n_channels
        )
        entropy = float(-np.sum(
            observed * np.log(np.maximum(observed, 1e-300))
        ) / max(n_channels * np.log(max(n_roles, 2)), 1e-12))
        return observed, {
            "expected_matching_cost": expected_cost,
            "normalized_assignment_entropy": float(np.clip(
                entropy, 0.0, 1.0)),
            "observed_role_mass": role_mass.tolist(),
            "missing_role_mass": float(n_roles - np.sum(role_mass)),
            "solver_status": solver_status,
            "optimizer_success": bool(fitted.success),
            "optimizer_candidate_feasible": candidate_feasible,
            "optimizer_row_error": candidate_row_error,
            "optimizer_column_excess": candidate_column_excess,
            "transport_geometry": (
                "barycentric_response" if barycentric else "pairwise_cost"),
        }

    @staticmethod
    def _drop_channel(exposure, channel):
        exposure = as_observable_state_exposure(exposure)
        keep = [
            index for index in range(len(exposure.channel_means))
            if index != int(channel)
        ]
        return ObservableStateExposure(
            exposure.channel_means[keep],
            exposure.channel_scales[keep],
            exposure.occupancy,
            exposure.dynamics,
            channel_names=tuple(exposure.channel_names[index] for index in keep),
            meta=dict(exposure.meta),
        )

    @staticmethod
    def _rank_correlation(left, right):
        left = np.asarray(left, dtype=float).reshape(-1)
        right = np.asarray(right, dtype=float).reshape(-1)
        if len(left) < 2 or len(left) != len(right):
            return 0.0
        left_rank = np.empty(len(left), dtype=float)
        right_rank = np.empty(len(right), dtype=float)
        left_rank[np.argsort(left, kind="stable")] = np.arange(len(left))
        right_rank[np.argsort(right, kind="stable")] = np.arange(len(right))
        return _safe_correlation(left_rank, right_rank)

    def _select_transport_temperature(
        self,
        standardized_signatures,
        grouped_exposures,
        grouped_margins,
        grouped_profiles,
    ):
        """Select transport smoothing by source-domain dropout replay."""

        candidate_scores = []
        domains = sorted(standardized_signatures)
        for temperature in self.transport_temperature_grid:
            full_descriptors = {}
            for domain in domains:
                weights, _ = self._partial_assignment(
                    standardized_signatures[domain],
                    self.prototypes,
                    temperature,
                    barycentric=self.barycentric_transport,
                )
                full_descriptors[domain] = np.vstack([
                    partially_aligned_observable_state_descriptor(
                        exposure, weights, n_roles=self.n_roles)
                    for exposure in grouped_exposures[domain]
                ])
            losses = []
            ranks = []
            for heldout in domains:
                train_domains = [value for value in domains if value != heldout]
                if not train_domains:
                    continue
                train_x = np.vstack([
                    full_descriptors[value] for value in train_domains])
                train_y = np.concatenate([
                    np.asarray(grouped_margins[value], dtype=float)
                    for value in train_domains
                ])
                center = np.mean(train_x, axis=0)
                scale = np.std(train_x, axis=0)
                scale = np.where(scale < 1e-8, 1.0, scale)
                design = np.column_stack([
                    np.ones(len(train_x)), (train_x - center) / scale])
                penalty = np.eye(design.shape[1], dtype=float)
                penalty[0, 0] = 0.0
                coefficients = np.linalg.solve(
                    design.T @ design + penalty,
                    design.T @ train_y,
                )
                heldout_exposures = grouped_exposures[heldout]
                channel_count = len(as_observable_state_exposure(
                    heldout_exposures[0]).channel_means)
                for channel in range(channel_count):
                    dropped = [
                        self._drop_channel(exposure, channel)
                        for exposure in heldout_exposures
                    ]
                    dropped_signature = (
                        _channel_signatures(
                            dropped,
                            grouped_profiles[heldout],
                            mode=self.signature_mode,
                        ) - self.signature_mean
                    ) / self.signature_scale
                    weights, _ = self._partial_assignment(
                        dropped_signature,
                        self.prototypes,
                        temperature,
                        barycentric=self.barycentric_transport,
                    )
                    test_x = np.vstack([
                        partially_aligned_observable_state_descriptor(
                            exposure, weights, n_roles=self.n_roles)
                        for exposure in dropped
                    ])
                    prediction = np.column_stack([
                        np.ones(len(test_x)), (test_x - center) / scale
                    ]) @ coefficients
                    target = np.asarray(
                        grouped_margins[heldout], dtype=float)
                    denominator = max(float(np.var(target)), 1e-8)
                    losses.append(float(np.mean(
                        (prediction - target) ** 2) / denominator))
                    ranks.append(self._rank_correlation(prediction, target))
            nmse = float(np.mean(losses)) if losses else float("inf")
            rank_loss = float(np.mean([
                0.5 * (1.0 - value) for value in ranks
            ])) if ranks else 1.0
            candidate_scores.append({
                "temperature": float(temperature),
                "dropout_nmse": nmse,
                "dropout_rank_loss": rank_loss,
                "score": float(nmse + 0.25 * rank_loss),
            })
        selected = min(candidate_scores, key=lambda item: (
            item["score"], item["temperature"]))
        self.transport_temperature = float(selected["temperature"])
        self.transport_selection_diagnostics = {
            "status": "source_domain_dropout_selected",
            "selected_temperature": self.transport_temperature,
            "candidate_scores": candidate_scores,
            "source_domain_count": int(len(domains)),
            "selection_target": "source_chance_margin",
            "channel_dropout_augmentation": True,
            "signature_mode": self.signature_mode,
            "transport_geometry": (
                "barycentric_response"
                if self.barycentric_transport else "pairwise_cost"),
            "target_data_used": False,
            "target_oracle_used": False,
        }

    def fit(
        self,
        exposures,
        domains,
        boundary_margins=None,
        profiles=None,
        source_problems=None,
    ):
        exposures = list(exposures)
        domains = np.asarray(domains, dtype=object).reshape(-1)
        if len(exposures) != len(domains) or not exposures:
            raise ValueError("role-alignment exposures and domains must align")
        margins = (
            np.zeros(len(exposures), dtype=float)
            if boundary_margins is None
            else np.asarray(boundary_margins, dtype=float).reshape(-1)
        )
        if len(margins) != len(exposures):
            raise ValueError("role-alignment margins must align")
        profile_rows = None if profiles is None else list(profiles)
        if self.signature_mode == "intervention_response":
            if profile_rows is None or len(profile_rows) != len(exposures):
                raise ValueError(
                    "intervention-response role alignment requires one "
                    "policy profile per exposure")

        grouped_exposures = defaultdict(list)
        grouped_margins = defaultdict(list)
        grouped_profiles = defaultdict(list)
        for index, (exposure, domain, margin) in enumerate(zip(
            exposures, domains, margins
        )):
            key = str(domain)
            grouped_exposures[key].append(exposure)
            grouped_margins[key].append(float(margin))
            if profile_rows is not None:
                grouped_profiles[key].append(profile_rows[index])
        signature_exposures = grouped_exposures
        signature_profiles = grouped_profiles
        source_problem_map = {
            str(domain): problem
            for domain, problem in (source_problems or [])
        }
        if self.signature_mode == "intervention_response" and source_problem_map:
            signature_exposures = defaultdict(list)
            signature_profiles = defaultdict(list)
            for domain in sorted(grouped_exposures):
                if domain not in source_problem_map:
                    raise ValueError(
                        f"missing source problem for role domain {domain!r}")
                problem = source_problem_map[domain]
                for x in self._target_policy_pool(problem):
                    signature_exposures[domain].append(
                        get_observable_state_exposure(problem, x))
                    signature_profiles[domain].append(problem.normalize(x))
        raw_signatures = {
            domain: _channel_signatures(
                values,
                signature_profiles[domain]
                if self.signature_mode == "intervention_response" else None,
                mode=self.signature_mode,
            )
            for domain, values in signature_exposures.items()
        }
        all_rows = np.vstack(list(raw_signatures.values()))
        self.signature_mean = np.mean(all_rows, axis=0)
        self.signature_scale = np.std(all_rows, axis=0)
        self.signature_scale = np.where(
            self.signature_scale < 1e-8, 1.0, self.signature_scale)
        signatures = {
            domain: (values - self.signature_mean) / self.signature_scale
            for domain, values in raw_signatures.items()
        }
        self.n_roles = min(
            max(len(values) for values in signatures.values()),
            MAX_OBSERVABLE_CHANNELS,
        )
        anchor_domain = min(
            (
                domain for domain, values in signatures.items()
                if len(values) == self.n_roles
            ),
            key=str,
        )
        anchor = signatures[anchor_domain]
        order = np.lexsort(tuple(
            anchor[:, index] for index in range(anchor.shape[1] - 1, -1, -1)
        ))
        prototypes = anchor[order].copy()

        assignments = {}
        losses = {}
        for _ in range(self.max_iter):
            buckets = [[] for _ in range(self.n_roles)]
            assignments = {}
            losses = {}
            for domain in sorted(signatures):
                assignment, loss = self._assignment(
                    signatures[domain], prototypes)
                assignments[domain] = assignment
                losses[domain] = loss
                for channel, role in enumerate(assignment):
                    if role >= 0:
                        buckets[int(role)].append(signatures[domain][channel])
            updated = prototypes.copy()
            for role, values in enumerate(buckets):
                if values:
                    updated[role] = np.mean(values, axis=0)
            if float(np.max(np.abs(updated - prototypes))) <= 1e-10:
                prototypes = updated
                break
            prototypes = updated

        relevance_samples = [[] for _ in range(self.n_roles)]
        relevance_sample_sizes = [[] for _ in range(self.n_roles)]
        for domain, assignment in assignments.items():
            domain_exposures = grouped_exposures[domain]
            domain_margins = np.asarray(grouped_margins[domain], dtype=float)
            means = np.vstack([
                as_observable_state_exposure(value).channel_means
                for value in domain_exposures
            ])
            for channel, role in enumerate(assignment):
                if role >= 0:
                    relevance_samples[int(role)].append(_safe_correlation(
                        means[:, channel], domain_margins))
                    relevance_sample_sizes[int(role)].append(
                        int(len(domain_margins)))
        relevance = np.asarray([
            float(np.mean(values)) if values else 0.0
            for values in relevance_samples
        ], dtype=float)
        role_order = np.lexsort((
            np.arange(self.n_roles),
            -relevance,
            -np.abs(relevance),
        ))
        inverse = np.empty(self.n_roles, dtype=int)
        inverse[role_order] = np.arange(self.n_roles, dtype=int)
        self.prototypes = prototypes[role_order]
        self.role_relevance = relevance[role_order]
        ordered_samples = [
            relevance_samples[int(role)] for role in role_order
        ]
        ordered_sample_sizes = [
            relevance_sample_sizes[int(role)] for role in role_order
        ]
        fisher_samples = [
            np.arctanh(np.clip(np.asarray(values, dtype=float), -0.995, 0.995))
            if values else np.zeros(0, dtype=float)
            for values in ordered_samples
        ]
        self.role_relevance_fisher_mean = np.asarray([
            float(np.mean(values)) if len(values) else 0.0
            for values in fisher_samples
        ], dtype=float)
        self.role_relevance_fisher_variance = np.asarray([
            (
                float(np.var(values, ddof=1))
                if len(values) > 1 else 0.0
            ) + float(np.mean([
                1.0 / max(int(count) - 3, 1)
                for count in sample_sizes
            ]))
            if sample_sizes else 1.0
            for values, sample_sizes in zip(
                fisher_samples, ordered_sample_sizes)
        ], dtype=float)
        self.role_relevance_effective_count = np.asarray([
            int(len(values)) for values in ordered_samples
        ], dtype=int)
        self.role_relevance_source_sample_sizes = [
            [int(value) for value in values]
            for values in ordered_sample_sizes
        ]
        self.source_assignments = {
            domain: inverse[assignment]
            for domain, assignment in assignments.items()
        }
        source_gap = {}
        positive_gaps = []
        signature_dim = max(int(self.prototypes.shape[1]), 1)
        for domain, values in signatures.items():
            table = sorted(
                self._assignment_loss_table(values, self.prototypes),
                key=lambda item: item[1],
            )
            normalized = [float(loss) / signature_dim for _, loss in table]
            gap = (
                normalized[1] - normalized[0]
                if len(normalized) > 1 else 1.0
            )
            gap = max(float(gap), 0.0)
            source_gap[domain] = gap
            if gap > 1e-8:
                positive_gaps.append(gap)
        self.source_assignment_temperature = max(
            float(np.median(positive_gaps)) if positive_gaps else 1.0,
            1e-8,
        )
        if self.partial_transport:
            self._select_transport_temperature(
                signatures,
                grouped_exposures,
                grouped_margins,
                grouped_profiles,
            )
            self.source_transport_weights = {
                domain: self._partial_assignment(
                    values,
                    self.prototypes,
                    self.transport_temperature,
                    barycentric=self.barycentric_transport,
                )[0]
                for domain, values in signatures.items()
            }
        self.source_diagnostics = {
            "status": "fit",
            "n_roles": int(self.n_roles),
            "signature_dim": int(self.prototypes.shape[1]),
            "source_domains": sorted(signatures),
            "source_channel_counts": {
                domain: int(len(values))
                for domain, values in signatures.items()
            },
            "source_assignments": {
                domain: assignment.tolist()
                for domain, assignment in self.source_assignments.items()
            },
            "source_matching_loss": {
                domain: float(losses[domain]) for domain in sorted(losses)
            },
            "source_assignment_normalized_gaps": {
                domain: float(source_gap[domain])
                for domain in sorted(source_gap)
            },
            "source_assignment_temperature": float(
                self.source_assignment_temperature),
            "source_assignment_temperature_rule": (
                "median_positive_best_second_gap"),
            "source_signature_pool": (
                "deterministic_unlabeled_intervention_pool"
                if self.signature_mode == "intervention_response"
                and source_problem_map
                else "source_archive"
            ),
            "role_relevance": self.role_relevance.tolist(),
            "role_relevance_fisher_mean": (
                self.role_relevance_fisher_mean.tolist()),
            "role_relevance_fisher_variance": (
                self.role_relevance_fisher_variance.tolist()),
            "role_relevance_effective_count": (
                self.role_relevance_effective_count.tolist()),
            "role_relevance_source_sample_sizes": copy.deepcopy(
                self.role_relevance_source_sample_sizes),
            "role_orientation_uses_source_margins": True,
            "partial_transport": bool(self.partial_transport),
            "signature_mode": self.signature_mode,
            "barycentric_transport": bool(self.barycentric_transport),
            "transport_selection": dict(
                self.transport_selection_diagnostics),
            "target_labels_used": False,
            "target_oracle_used": False,
        }
        self._target_cache = {}
        return self

    def source_descriptor(self, domain, exposure):
        if self.prototypes is None:
            raise RuntimeError("channel-role aligner is not fit")
        key = str(domain)
        if key not in self.source_assignments:
            raise KeyError(f"unknown source role domain {key!r}")
        return role_aligned_observable_state_descriptor(
            exposure,
            self.source_assignments[key],
            n_roles=self.n_roles,
        )

    def source_transport_descriptor(self, domain, exposure):
        if not self.partial_transport or self.transport_temperature is None:
            raise RuntimeError("partial channel-role transport is not fit")
        key = str(domain)
        if key not in self.source_transport_weights:
            raise KeyError(f"unknown source transport domain {key!r}")
        return partially_aligned_observable_state_descriptor(
            exposure,
            self.source_transport_weights[key],
            n_roles=self.n_roles,
        )

    def _target_policy_pool(self, problem):
        d = max(int(getattr(problem, "d", 1)), 1)
        positions = (np.arange(d, dtype=float) + 0.5) / float(d)
        profiles = []
        for level in np.linspace(0.05, 0.95, 10):
            profiles.append(np.full(d, float(level), dtype=float))
        for left, right in (
            (0.15, 0.85), (0.30, 0.70), (0.45, 0.55),
            (0.85, 0.15), (0.70, 0.30), (0.55, 0.45),
        ):
            profiles.append(np.linspace(left, right, d))
        for first in (0.15, 0.50, 0.85):
            for second in (0.15, 0.50, 0.85):
                for third in (0.15, 0.50, 0.85):
                    row = np.empty(d, dtype=float)
                    for index in range(d):
                        row[index] = (first, second, third)[min(
                            2, int(3 * index / d))]
                    profiles.append(row)
        rng = np.random.default_rng(self.seed + 1_000_003 + 17 * d)
        while len(profiles) < self.target_pool_size:
            row = np.full(d, float(rng.uniform(0.10, 0.90)), dtype=float)
            for frequency in range(1, 5):
                row += float(rng.normal(0.0, 0.18 / frequency)) * np.cos(
                    np.pi * frequency * positions)
            profiles.append(np.clip(row, 0.0, 1.0))
        return [
            tuple(int(value) for value in problem.continuous_to_int(profile))
            for profile in profiles[:self.target_pool_size]
        ]

    def target_policy_pool(self, problem):
        """Return the deterministic unlabeled pool used for target matching."""

        return list(self._target_policy_pool(problem))

    def target_assignment(self, problem):
        if self.prototypes is None:
            raise RuntimeError("channel-role aligner is not fit")
        key = (
            str(getattr(problem, "problem_name", type(problem).__name__)),
            int(getattr(problem, "d", 1)),
        )
        if key not in self._target_cache:
            exposures = [
                get_observable_state_exposure(problem, x)
                for x in self._target_policy_pool(problem)
            ]
            if any(value is None for value in exposures):
                raise ValueError(
                    "target role alignment requires observable exposures")
            target_points = self._target_policy_pool(problem)
            signature = _channel_signatures(
                exposures,
                [problem.normalize(x) for x in target_points],
                mode=self.signature_mode,
            )
            standardized = (
                signature - self.signature_mean
            ) / self.signature_scale
            assignment, loss = self._assignment(
                standardized, self.prototypes)
            target_match = {
                "assignment": assignment,
                "matching_loss": float(loss),
                "channel_count": int(len(signature)),
                "unlabeled_policy_count": int(len(exposures)),
                "target_labels_used": False,
                "target_oracle_used": False,
            }
            if self.partial_transport:
                weights, transport = self._partial_assignment(
                    standardized,
                    self.prototypes,
                    self.transport_temperature,
                    barycentric=self.barycentric_transport,
                )
                target_match.update({
                    "transport_weights": weights,
                    "transport_temperature": float(
                        self.transport_temperature),
                    **transport,
                })
            self._target_cache[key] = target_match
        return self._target_cache[key]["assignment"].copy()

    def target_assignment_prior(
        self,
        problem,
        assignments,
        *,
        temperature_scale=1.0,
    ):
        """Return a source-calibrated, target-unlabeled assignment prior."""

        if self.prototypes is None:
            raise RuntimeError("channel-role aligner is not fit")
        points = self._target_policy_pool(problem)
        exposures = [
            get_observable_state_exposure(problem, x) for x in points
        ]
        if any(value is None for value in exposures):
            raise ValueError(
                "target assignment prior requires observable exposures")
        signature = _channel_signatures(
            exposures,
            [problem.normalize(x) for x in points],
            mode=self.signature_mode,
        )
        standardized = (
            signature - self.signature_mean
        ) / self.signature_scale
        table = self._assignment_loss_table(
            standardized, self.prototypes, assignments)
        signature_dim = max(int(self.prototypes.shape[1]), 1)
        normalized_loss = np.asarray([
            float(loss) / signature_dim for _, loss in table
        ], dtype=float)
        temperature = max(
            float(self.source_assignment_temperature)
            * max(float(temperature_scale), 1e-8),
            1e-8,
        )
        log_weight = -(
            normalized_loss - float(np.min(normalized_loss))
        ) / temperature
        log_weight -= float(np.max(log_weight))
        weight = np.exp(log_weight)
        weight /= float(np.sum(weight))
        hard_assignment = tuple(int(value) for value in self.target_assignment(
            problem))
        labels = [tuple(int(value) for value in assignment) for assignment, _ in table]
        diagnostics = {
            "status": "fit",
            "mode": "source_geometry",
            "source_calibrated_temperature": float(
                self.source_assignment_temperature),
            "temperature_scale": float(temperature_scale),
            "effective_temperature": float(temperature),
            "normalized_losses": normalized_loss.tolist(),
            "weights": weight.tolist(),
            "effective_assignment_count": float(1.0 / np.sum(weight ** 2)),
            "hard_assignment": "-".join(
                str(value) for value in hard_assignment),
            "maximum_prior_assignment": "-".join(
                str(value) for value in labels[int(np.argmax(weight))]),
            "maximum_prior_matches_hard_assignment": bool(
                labels[int(np.argmax(weight))] == hard_assignment),
            "target_labels_used": False,
            "target_oracle_used": False,
            "permutation_equivariant": True,
        }
        return weight, diagnostics

    @staticmethod
    def boundary_assignment_update(
        assignments,
        prior_weights,
        target_fisher_mean,
        target_fisher_variance,
        role_fisher_mean,
        role_fisher_variance,
        *,
        likelihood_temperature=1.0,
    ):
        """Update assignment mass from noisy channel/role boundary effects.

        Each assignment maps target channels to canonical source roles.  The
        likelihood compares Fisher-transformed channel/margin correlations to
        the source-trained role distribution.  Relabeling target channels and
        relabeling the assignment list induces the same posterior mass.
        """

        assignment_rows = [
            tuple(int(value) for value in assignment)
            for assignment in assignments
        ]
        prior = np.asarray(prior_weights, dtype=float).reshape(-1)
        target_mean = np.asarray(
            target_fisher_mean, dtype=float).reshape(-1)
        target_variance = np.asarray(
            target_fisher_variance, dtype=float).reshape(-1)
        role_mean = np.asarray(role_fisher_mean, dtype=float).reshape(-1)
        role_variance = np.asarray(
            role_fisher_variance, dtype=float).reshape(-1)
        if not assignment_rows or len(prior) != len(assignment_rows):
            raise ValueError("boundary assignment prior must align")
        if len(target_mean) == 0 or len(target_mean) != len(target_variance):
            raise ValueError("target boundary summaries must align")
        if len(role_mean) == 0 or len(role_mean) != len(role_variance):
            raise ValueError("source role boundary summaries must align")
        if not np.all(np.isfinite(np.concatenate([
            prior, target_mean, target_variance, role_mean, role_variance,
        ]))):
            raise ValueError("boundary assignment summaries must be finite")
        if np.any(prior < 0.0) or float(np.sum(prior)) <= 0.0:
            raise ValueError("boundary assignment prior needs positive mass")
        if np.any(target_variance <= 0.0) or np.any(role_variance <= 0.0):
            raise ValueError("boundary assignment variances must be positive")
        if any(
            len(assignment) != len(target_mean)
            or min(assignment) < 0
            or max(assignment) >= len(role_mean)
            or len(set(assignment)) != len(assignment)
            for assignment in assignment_rows
        ):
            raise ValueError("boundary role assignments must be injections")
        temperature = max(float(likelihood_temperature), 1e-8)
        log_likelihood = []
        for assignment in assignment_rows:
            role_index = np.asarray(assignment, dtype=int)
            variance = np.maximum(
                target_variance + role_variance[role_index], 1e-12)
            residual = target_mean - role_mean[role_index]
            log_likelihood.append(float(-0.5 * np.sum(
                residual ** 2 / variance
                + np.log(2.0 * np.pi * variance)
            )))
        log_likelihood = np.asarray(log_likelihood, dtype=float)
        log_weight = np.full(len(prior), -np.inf, dtype=float)
        supported = prior > 0.0
        normalized_prior = prior / float(np.sum(prior))
        log_weight[supported] = (
            np.log(normalized_prior[supported])
            + log_likelihood[supported] / temperature
        )
        normalizer = float(np.max(log_weight))
        if not np.isfinite(normalizer):
            raise FloatingPointError(
                "boundary assignment posterior has no supported atom")
        posterior = np.exp(log_weight - normalizer)
        posterior /= float(np.sum(posterior))
        return posterior, log_likelihood

    def target_boundary_assignment_posterior(
        self,
        problem,
        assignments,
        samples,
        targets,
        observation_variances,
        *,
        geometry_prior_weights=None,
        likelihood_temperature=1.0,
    ):
        """Adapt role mass from charged target chance-margin observations.

        The source archive fixes the role correlation law.  The target uses
        only already charged observations and observable exposures; no truth
        pool, latent simulator state, or target oracle enters this update.
        """

        assignments = tuple(
            tuple(int(value) for value in assignment)
            for assignment in assignments
        )
        samples = [tuple(int(v) for v in np.asarray(x, dtype=int))
                   for x in samples]
        target = np.asarray(targets, dtype=float).reshape(-1)
        noise = np.asarray(observation_variances, dtype=float).reshape(-1)
        if len(noise) == 1 and len(samples) > 1:
            noise = np.full(len(samples), float(noise[0]), dtype=float)
        if len(samples) != len(target) or len(samples) != len(noise):
            raise ValueError("target boundary observations must align")
        if geometry_prior_weights is None:
            prior = np.full(
                len(assignments), 1.0 / max(len(assignments), 1), dtype=float)
        else:
            prior = np.asarray(
                geometry_prior_weights, dtype=float).reshape(-1)
        if len(prior) != len(assignments):
            raise ValueError("geometry assignment prior must align")
        prior = prior / float(np.sum(prior))
        base = {
            "status": "insufficient_target_boundary_evidence",
            "mode": "source_geometry_boundary",
            "geometry_prior_weights": prior.tolist(),
            "posterior_weights": prior.tolist(),
            "target_observation_count": int(len(samples)),
            "target_labels_used": bool(len(samples)),
            "target_oracle_used": False,
            "permutation_equivariant": True,
        }
        if len(samples) < 4 or not len(assignments):
            return prior, base
        exposures = [
            get_observable_state_exposure(problem, x) for x in samples
        ]
        if any(value is None for value in exposures):
            raise ValueError(
                "boundary role adaptation requires observable exposures")
        channel_means = np.vstack([
            as_observable_state_exposure(value).channel_means
            for value in exposures
        ])
        channel_count = int(channel_means.shape[1])
        if any(len(assignment) != channel_count for assignment in assignments):
            raise ValueError(
                "boundary assignments and target channels must align")
        margin = float(getattr(problem, "tau", 0.0)) - target
        target_correlation = np.asarray([
            _safe_correlation(channel_means[:, channel], margin)
            for channel in range(channel_count)
        ], dtype=float)
        target_fisher = np.arctanh(np.clip(
            target_correlation, -0.995, 0.995))
        empirical_variance = max(float(np.var(margin)), 1e-12)
        average_noise = max(float(np.mean(np.maximum(noise, 0.0))), 0.0)
        noise_fraction = float(np.clip(
            average_noise / max(empirical_variance, average_noise, 1e-12),
            0.0,
            1.0,
        ))
        target_fisher_variance = np.full(
            channel_count,
            (1.0 + noise_fraction) / max(len(samples) - 3, 1),
            dtype=float,
        )
        role_fisher_mean = np.asarray(
            self.role_relevance_fisher_mean, dtype=float).reshape(-1)
        role_fisher_variance = np.asarray(
            self.role_relevance_fisher_variance, dtype=float).reshape(-1)
        posterior, log_likelihood = self.boundary_assignment_update(
            assignments,
            prior,
            target_fisher,
            target_fisher_variance,
            role_fisher_mean,
            role_fisher_variance,
            likelihood_temperature=likelihood_temperature,
        )
        labels = ["-".join(str(value) for value in assignment)
                  for assignment in assignments]
        diagnostics = {
            **base,
            "status": "fit",
            "posterior_weights": posterior.tolist(),
            "assignment_log_likelihood": log_likelihood.tolist(),
            "likelihood_temperature": float(likelihood_temperature),
            "target_channel_correlations": target_correlation.tolist(),
            "target_channel_fisher_mean": target_fisher.tolist(),
            "target_channel_fisher_variance": (
                target_fisher_variance.tolist()),
            "source_role_fisher_mean": role_fisher_mean.tolist(),
            "source_role_fisher_variance": role_fisher_variance.tolist(),
            "target_margin_empirical_variance": float(empirical_variance),
            "target_observation_noise_mean": float(average_noise),
            "target_observation_noise_fraction": float(noise_fraction),
            "effective_assignment_count_before": float(
                1.0 / np.sum(prior ** 2)),
            "effective_assignment_count_after": float(
                1.0 / np.sum(posterior ** 2)),
            "maximum_prior_assignment": labels[int(np.argmax(prior))],
            "maximum_posterior_assignment": labels[int(np.argmax(posterior))],
            "target_labels_used": True,
            "target_oracle_used": False,
        }
        return posterior, diagnostics

    def target_transport_weights(self, problem):
        if not self.partial_transport:
            raise RuntimeError("partial channel-role transport is disabled")
        self.target_assignment(problem)
        key = (
            str(getattr(problem, "problem_name", type(problem).__name__)),
            int(getattr(problem, "d", 1)),
        )
        return np.asarray(
            self._target_cache[key]["transport_weights"], dtype=float
        ).copy()

    def target_epistemic_calibration(self, problem):
        """Return an outcome-free trust factor for the target role match.

        Channel signatures are standardized with source-only moments.  The
        average squared matching residual should therefore be order one when
        the target channels are supported by the source role atlas.  Excess
        residual is converted to a Gaussian Bayes-factor penalty.  A target
        with a different number or meaning of channels can consequently fall
        back to the non-transfer component instead of being force-matched to a
        source role.
        """

        self.target_assignment(problem)
        key = (
            str(getattr(problem, "problem_name", type(problem).__name__)),
            int(getattr(problem, "d", 1)),
        )
        match = self._target_cache[key]
        signature_dim = int(self.prototypes.shape[1])
        matched_channels = max(int(match["channel_count"]), 1)
        degrees_of_freedom = max(signature_dim * matched_channels, 1)
        raw_matching_loss = (
            float(match["expected_matching_cost"])
            if self.partial_transport
            else float(match["matching_loss"])
        )
        normalized_loss = max(
            raw_matching_loss / float(signature_dim), 0.0)
        excess_loss = max(normalized_loss - 1.0, 0.0)
        log_trust = -0.5 * excess_loss
        trust = float(np.exp(max(log_trust, -745.0)))
        assignment_entropy = float(
            match.get("normalized_assignment_entropy", 0.0))
        cardinality_gap = max(
            (self.n_roles - int(match["channel_count"]))
            / max(self.n_roles, 1),
            0.0,
        )
        epistemic_scale = float(min(
            100.0,
            1.0 + excess_loss + assignment_entropy + cardinality_gap,
        ))
        result = {
            "status": "calibrated",
            "matching_loss": float(match["matching_loss"]),
            "matching_degrees_of_freedom": int(degrees_of_freedom),
            "normalized_matching_loss": float(normalized_loss),
            "excess_normalized_matching_loss": float(excess_loss),
            "source_role_trust": float(np.clip(trust, 0.0, 1.0)),
            "partial_transport": bool(self.partial_transport),
            "transport_temperature": (
                None if self.transport_temperature is None
                else float(self.transport_temperature)),
            "normalized_assignment_entropy": assignment_entropy,
            "channel_cardinality_gap": float(cardinality_gap),
            "epistemic_covariance_scale": epistemic_scale,
            "calibration_law": "exp(-0.5*max(loss_per_coordinate-1,0))",
            "source_signature_standardized": True,
            "target_labels_used": False,
            "target_oracle_used": False,
        }
        match["epistemic_calibration"] = dict(result)
        return result

    def target_support_diagnostics(self, problem):
        """Audit whether target role cardinality was observed in source data."""

        self.target_assignment(problem)
        key = (
            str(getattr(problem, "problem_name", type(problem).__name__)),
            int(getattr(problem, "d", 1)),
        )
        match = self._target_cache[key]
        source_counts = sorted(set(int(value) for value in (
            self.source_diagnostics.get("source_channel_counts", {})
        ).values()))
        target_count = int(match["channel_count"])
        supported = target_count in source_counts
        result = {
            "status": "supported" if supported else "unsupported_cardinality",
            "target_channel_count": target_count,
            "source_channel_count_support": source_counts,
            "channel_cardinality_supported": bool(supported),
            "selection_uses_target_labels": False,
            "selection_uses_target_oracle": False,
        }
        match["support_diagnostics"] = dict(result)
        return result

    def descriptor(self, problem, exposure):
        return role_aligned_observable_state_descriptor(
            exposure,
            self.target_assignment(problem),
            n_roles=self.n_roles,
        )

    def transport_descriptor(self, problem, exposure):
        return partially_aligned_observable_state_descriptor(
            exposure,
            self.target_transport_weights(problem),
            n_roles=self.n_roles,
        )

    def diagnostics(self):
        return {
            **self.source_diagnostics,
            "target_pool_size": int(self.target_pool_size),
            "target_matches": {
                str(key): {
                    **value,
                    "assignment": value["assignment"].tolist(),
                    **({
                        "transport_weights": np.asarray(
                            value["transport_weights"], dtype=float).tolist(),
                    } if "transport_weights" in value else {}),
                }
                for key, value in self._target_cache.items()
            },
        }
