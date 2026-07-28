"""Observable state/trajectory exposures shared across transfer domains.

The chance-constraint mean and cumulative variance need not be functions of
the same low-dimensional coordinates.  They can, however, start from the same
observable simulator record.  This module defines that record without reading
objectives, constraints, hidden optima, or a target cumulative-risk provider.

Synthetic problems expose control/state groups in lieu of a full trajectory.
Traffic and other simulators can populate the same contract from occupancy or
trajectory logs.  The canonical descriptor has a fixed dimension even when a
domain has a different number of observable channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


MAX_OBSERVABLE_CHANNELS = 4
OCCUPANCY_BINS = 8
LOW_FREQUENCY_COMPONENTS = 6


@dataclass
class ObservableStateExposure:
    """Target-observable state/action summary used before either model head."""

    channel_means: np.ndarray
    channel_scales: np.ndarray
    occupancy: np.ndarray
    dynamics: np.ndarray
    channel_names: tuple[str, ...] = field(default_factory=tuple)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.channel_means = np.asarray(
            self.channel_means, dtype=float).reshape(-1)
        self.channel_scales = np.asarray(
            self.channel_scales, dtype=float).reshape(-1)
        self.occupancy = np.asarray(self.occupancy, dtype=float).reshape(-1)
        self.dynamics = np.asarray(self.dynamics, dtype=float).reshape(-1)
        if len(self.channel_means) != len(self.channel_scales):
            raise ValueError("observable channel means/scales must align")
        if not self.channel_names:
            self.channel_names = tuple(
                f"channel_{index}" for index in range(len(self.channel_means)))
        self.channel_names = tuple(str(value) for value in self.channel_names)
        if len(self.channel_names) != len(self.channel_means):
            raise ValueError("observable channel names must align")
        values = np.concatenate([
            self.channel_means,
            self.channel_scales,
            self.occupancy,
            self.dynamics,
        ])
        if not np.all(np.isfinite(values)):
            raise ValueError("observable state exposure must be finite")


def as_observable_state_exposure(value):
    if value is None:
        return None
    if isinstance(value, ObservableStateExposure):
        return value
    if isinstance(value, dict):
        required = {
            "channel_means", "channel_scales", "occupancy", "dynamics"
        }
        if required.issubset(value):
            return ObservableStateExposure(
                value["channel_means"],
                value["channel_scales"],
                value["occupancy"],
                value["dynamics"],
                channel_names=tuple(value.get("channel_names", ())),
                meta=dict(value.get("meta", {})),
            )
    return None


def grouped_policy_state_exposure(
    profile,
    groups,
    *,
    channel_names=(),
    provider="grouped_policy_state",
):
    """Build an observable exposure from declared control/state groups.

    Group membership is part of the simulator's action schema.  It is not a
    target outcome or target-specific safe-region formula.  Every statistic
    below is therefore available before the first target evaluation.
    """

    z = np.asarray(profile, dtype=float).reshape(-1)
    if len(z) == 0:
        z = np.zeros(1, dtype=float)
    if not np.all(np.isfinite(z)):
        raise ValueError("observable policy profile must be finite")
    z = np.clip(z, 0.0, 1.0)
    group_rows = []
    for group in groups:
        row = np.asarray(group, dtype=float).reshape(-1)
        row = row[np.isfinite(row)]
        if len(row):
            group_rows.append(np.clip(row, 0.0, 1.0))
    if not group_rows:
        group_rows = [z]
    means = np.asarray([float(np.mean(row)) for row in group_rows])
    scales = np.asarray([float(np.std(row)) for row in group_rows])

    histogram, _ = np.histogram(
        z, bins=np.linspace(0.0, 1.0, OCCUPANCY_BINS + 1))
    histogram = histogram.astype(float) / max(float(len(z)), 1.0)
    quantiles = np.quantile(z, [0.10, 0.25, 0.50, 0.75, 0.90])
    occupancy = np.concatenate([histogram, quantiles])

    differences = np.diff(z) if len(z) > 1 else np.zeros(1, dtype=float)
    centered = z - float(np.mean(z))
    positions = np.arange(len(z), dtype=float) + 0.5
    low_frequency = np.zeros(LOW_FREQUENCY_COMPONENTS, dtype=float)
    normalization = np.sqrt(2.0 / max(len(z), 1))
    for frequency in range(1, LOW_FREQUENCY_COMPONENTS + 1):
        low_frequency[frequency - 1] = normalization * float(np.sum(
            centered
            * np.cos(np.pi * frequency * positions / max(len(z), 1))
        ))
    dynamics = np.concatenate([
        np.asarray([
            float(np.mean(z)),
            float(np.std(z)),
            float(np.min(z)),
            float(np.max(z)),
            float(np.mean(np.abs(differences))),
            float(np.std(differences)),
        ]),
        low_frequency,
    ])
    names = tuple(channel_names)
    if not names:
        names = tuple(f"channel_{index}" for index in range(len(means)))
    return ObservableStateExposure(
        means,
        scales,
        occupancy,
        dynamics,
        channel_names=names,
        meta={
            "provider": str(provider),
            "schema": "observable_state_trajectory_v1",
            "target_outcomes_used": False,
            "target_risk_provider_used": False,
        },
    )


def get_observable_state_exposure(problem, x):
    """Read the declared observable exposure without structural fallbacks."""

    if problem is None or not hasattr(problem, "observable_state_exposure"):
        return None
    try:
        value = problem.observable_state_exposure(x)
    except AttributeError:
        return None
    return as_observable_state_exposure(value)


def canonical_observable_state_descriptor(exposure, mode="ordered"):
    """Return an ordered or permutation-invariant observable descriptor."""

    mode = str(mode or "ordered").strip().lower().replace("-", "_")
    if mode in {"set", "set_invariant", "permutation_invariant", "invariant"}:
        return canonical_set_invariant_observable_state_descriptor(exposure)
    if mode != "ordered":
        raise ValueError(
            "observable descriptor mode must be ordered or set_invariant")

    exposure = as_observable_state_exposure(exposure)
    if exposure is None:
        raise ValueError("canonical descriptor requires observable exposure")
    means = np.zeros(MAX_OBSERVABLE_CHANNELS, dtype=float)
    scales = np.zeros(MAX_OBSERVABLE_CHANNELS, dtype=float)
    mask = np.zeros(MAX_OBSERVABLE_CHANNELS, dtype=float)
    count = min(len(exposure.channel_means), MAX_OBSERVABLE_CHANNELS)
    means[:count] = exposure.channel_means[:count]
    scales[:count] = exposure.channel_scales[:count]
    mask[:count] = 1.0

    pair_products = []
    pair_differences = []
    for left in range(MAX_OBSERVABLE_CHANNELS):
        for right in range(left, MAX_OBSERVABLE_CHANNELS):
            pair_products.append(float(means[left] * means[right]))
            if left < right:
                pair_differences.append(float(means[left] - means[right]))
    descriptor = np.concatenate([
        means,
        scales,
        mask,
        exposure.occupancy,
        exposure.dynamics,
        np.asarray(pair_products, dtype=float),
        np.asarray(pair_differences, dtype=float),
    ])
    if not np.all(np.isfinite(descriptor)):
        raise FloatingPointError("observable state descriptor is non-finite")
    return descriptor


def canonical_set_invariant_observable_state_descriptor(exposure):
    """Fixed-dimensional channel-set descriptor.

    Channel identities are deliberately removed.  Symmetric moments preserve
    set-level information, while sorting by the observable channel mean keeps
    low/medium/high exposure roles without depending on a domain's channel
    enumeration.  Global occupancy and dynamics are already channel-order
    invariant and are retained verbatim.
    """

    exposure = as_observable_state_exposure(exposure)
    if exposure is None:
        raise ValueError("set-invariant descriptor requires observable exposure")
    means = np.asarray(exposure.channel_means, dtype=float).reshape(-1)
    scales = np.asarray(exposure.channel_scales, dtype=float).reshape(-1)
    if len(means) == 0:
        means = np.zeros(1, dtype=float)
        scales = np.zeros(1, dtype=float)

    def summary(values):
        values = np.asarray(values, dtype=float).reshape(-1)
        quantiles = np.quantile(values, [0.0, 0.25, 0.50, 0.75, 1.0])
        return np.concatenate([
            quantiles,
            np.asarray([
                float(np.mean(values)),
                float(np.std(values)),
                float(np.mean(values ** 2)),
            ]),
        ])

    order = np.lexsort((scales, means))
    sorted_means = np.zeros(MAX_OBSERVABLE_CHANNELS, dtype=float)
    sorted_scales = np.zeros(MAX_OBSERVABLE_CHANNELS, dtype=float)
    mask = np.zeros(MAX_OBSERVABLE_CHANNELS, dtype=float)
    count = min(len(order), MAX_OBSERVABLE_CHANNELS)
    selected = order[:count]
    sorted_means[:count] = means[selected]
    sorted_scales[:count] = scales[selected]
    mask[:count] = 1.0

    centered_means = means - float(np.mean(means))
    centered_scales = scales - float(np.mean(scales))
    covariance = float(np.mean(centered_means * centered_scales))
    pair_differences = np.asarray([
        abs(float(means[left] - means[right]))
        for left in range(len(means))
        for right in range(left + 1, len(means))
    ], dtype=float)
    if len(pair_differences) == 0:
        pair_differences = np.zeros(1, dtype=float)
    descriptor = np.concatenate([
        np.asarray([float(min(len(means), MAX_OBSERVABLE_CHANNELS))
                    / MAX_OBSERVABLE_CHANNELS]),
        summary(means),
        summary(scales),
        np.asarray([
            float(np.mean(means * scales)),
            covariance,
            float(np.mean(pair_differences)),
            float(np.max(pair_differences)),
        ]),
        sorted_means,
        sorted_scales,
        mask,
        exposure.occupancy,
        exposure.dynamics,
    ])
    if not np.all(np.isfinite(descriptor)):
        raise FloatingPointError(
            "set-invariant observable state descriptor is non-finite")
    return descriptor


def role_aligned_observable_state_descriptor(
    exposure,
    channel_to_role,
    *,
    n_roles=MAX_OBSERVABLE_CHANNELS,
):
    """Place observable channels into learned canonical roles.

    ``channel_to_role`` is equivariant: permuting the input channels and the
    corresponding assignment leaves the descriptor unchanged. Missing roles
    remain explicit through the mask rather than being imputed from outcomes.
    """

    exposure = as_observable_state_exposure(exposure)
    if exposure is None:
        raise ValueError("role-aligned descriptor requires observable exposure")
    assignment = np.asarray(channel_to_role, dtype=int).reshape(-1)
    if len(assignment) != len(exposure.channel_means):
        raise ValueError("channel-role assignment must align with exposure")
    n_roles = min(max(int(n_roles), 1), MAX_OBSERVABLE_CHANNELS)
    means = np.zeros(MAX_OBSERVABLE_CHANNELS, dtype=float)
    scales = np.zeros(MAX_OBSERVABLE_CHANNELS, dtype=float)
    mask = np.zeros(MAX_OBSERVABLE_CHANNELS, dtype=float)
    for channel, role in enumerate(assignment):
        if role < 0:
            continue
        if role >= n_roles:
            raise ValueError("channel role exceeds the fitted role count")
        if mask[role] > 0.0:
            raise ValueError("channel-role assignment must be one-to-one")
        means[role] = float(exposure.channel_means[channel])
        scales[role] = float(exposure.channel_scales[channel])
        mask[role] = 1.0

    pair_products = []
    pair_differences = []
    for left in range(MAX_OBSERVABLE_CHANNELS):
        for right in range(left, MAX_OBSERVABLE_CHANNELS):
            pair_products.append(float(means[left] * means[right]))
            if left < right:
                pair_differences.append(float(means[left] - means[right]))
    descriptor = np.concatenate([
        means,
        scales,
        mask,
        exposure.occupancy,
        exposure.dynamics,
        np.asarray(pair_products, dtype=float),
        np.asarray(pair_differences, dtype=float),
    ])
    if not np.all(np.isfinite(descriptor)):
        raise FloatingPointError(
            "role-aligned observable state descriptor is non-finite")
    return descriptor


def partially_aligned_observable_state_descriptor(
    exposure,
    channel_role_weights,
    *,
    n_roles=MAX_OBSERVABLE_CHANNELS,
):
    """Pool channels into canonical roles with a partial soft assignment.

    Every channel has one unit of transport mass, while a canonical role may
    receive any mass in ``[0, 1]``. This preserves the missing-role signal when
    a target has fewer observable channels than the source atlas. Simultaneously
    permuting channels and rows of ``channel_role_weights`` leaves the returned
    descriptor unchanged.
    """

    exposure = as_observable_state_exposure(exposure)
    if exposure is None:
        raise ValueError(
            "partially aligned descriptor requires observable exposure")
    weights = np.asarray(channel_role_weights, dtype=float)
    n_channels = len(exposure.channel_means)
    n_roles = min(max(int(n_roles), 1), MAX_OBSERVABLE_CHANNELS)
    if weights.shape != (n_channels, n_roles):
        raise ValueError(
            "partial channel-role weights must align with channels and roles")
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < -1e-12)
        or not np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-8)
    ):
        raise ValueError(
            "partial channel-role weights must be finite row probabilities")
    weights = np.maximum(weights, 0.0)
    weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-15)

    role_mass = np.sum(weights, axis=0)
    denominator = np.maximum(role_mass, 1e-12)
    role_means = (
        weights.T @ np.asarray(exposure.channel_means, dtype=float)
    ) / denominator
    role_scales = (
        weights.T @ np.asarray(exposure.channel_scales, dtype=float)
    ) / denominator
    absent = role_mass <= 1e-12
    role_means[absent] = 0.0
    role_scales[absent] = 0.0

    means = np.zeros(MAX_OBSERVABLE_CHANNELS, dtype=float)
    scales = np.zeros(MAX_OBSERVABLE_CHANNELS, dtype=float)
    mask = np.zeros(MAX_OBSERVABLE_CHANNELS, dtype=float)
    means[:n_roles] = role_means
    scales[:n_roles] = role_scales
    # The capacity-constrained matcher keeps role mass at most one. Retaining
    # the continuous mass here records uncertainty/missingness without adding
    # a new descriptor dimension.
    mask[:n_roles] = np.clip(role_mass, 0.0, 1.0)

    pair_products = []
    pair_differences = []
    for left in range(MAX_OBSERVABLE_CHANNELS):
        for right in range(left, MAX_OBSERVABLE_CHANNELS):
            pair_products.append(float(means[left] * means[right]))
            if left < right:
                pair_differences.append(float(means[left] - means[right]))
    descriptor = np.concatenate([
        means,
        scales,
        mask,
        exposure.occupancy,
        exposure.dynamics,
        np.asarray(pair_products, dtype=float),
        np.asarray(pair_differences, dtype=float),
    ])
    if not np.all(np.isfinite(descriptor)):
        raise FloatingPointError(
            "partially aligned observable state descriptor is non-finite")
    return descriptor


def observable_state_descriptor_names(mode="ordered"):
    mode = str(mode or "ordered").strip().lower().replace("-", "_")
    if mode in {"role", "role_aligned", "equivariant_roles"}:
        names = []
        names.extend(
            f"canonical_role_{index}_mean"
            for index in range(MAX_OBSERVABLE_CHANNELS))
        names.extend(
            f"canonical_role_{index}_scale"
            for index in range(MAX_OBSERVABLE_CHANNELS))
        names.extend(
            f"canonical_role_{index}_present"
            for index in range(MAX_OBSERVABLE_CHANNELS))
        names.extend(f"occupancy_bin_{index}" for index in range(OCCUPANCY_BINS))
        names.extend(("q10", "q25", "median", "q75", "q90"))
        names.extend((
            "global_mean", "global_scale", "global_minimum", "global_maximum",
            "mean_absolute_transition", "transition_scale",
        ))
        names.extend(
            f"low_frequency_{index}"
            for index in range(1, LOW_FREQUENCY_COMPONENTS + 1))
        for left in range(MAX_OBSERVABLE_CHANNELS):
            for right in range(left, MAX_OBSERVABLE_CHANNELS):
                names.append(f"canonical_role_product_{left}_{right}")
        for left in range(MAX_OBSERVABLE_CHANNELS):
            for right in range(left + 1, MAX_OBSERVABLE_CHANNELS):
                names.append(f"canonical_role_difference_{left}_{right}")
        return tuple(names)
    if mode in {"set", "set_invariant", "permutation_invariant", "invariant"}:
        names = ["channel_count_fraction"]
        for label in ("mean", "scale"):
            names.extend(
                f"channel_{label}_{stat}"
                for stat in (
                    "minimum", "q25", "median", "q75", "maximum",
                    "average", "std", "second_moment",
                )
            )
        names.extend((
            "channel_mean_scale_product",
            "channel_mean_scale_covariance",
            "channel_mean_pair_difference_average",
            "channel_mean_pair_difference_maximum",
        ))
        names.extend(
            f"ordered_by_value_channel_{index}_mean"
            for index in range(MAX_OBSERVABLE_CHANNELS))
        names.extend(
            f"ordered_by_value_channel_{index}_scale"
            for index in range(MAX_OBSERVABLE_CHANNELS))
        names.extend(
            f"ordered_by_value_channel_{index}_present"
            for index in range(MAX_OBSERVABLE_CHANNELS))
        names.extend(f"occupancy_bin_{index}" for index in range(OCCUPANCY_BINS))
        names.extend(("q10", "q25", "median", "q75", "q90"))
        names.extend((
            "global_mean", "global_scale", "global_minimum", "global_maximum",
            "mean_absolute_transition", "transition_scale",
        ))
        names.extend(
            f"low_frequency_{index}"
            for index in range(1, LOW_FREQUENCY_COMPONENTS + 1))
        return tuple(names)
    if mode != "ordered":
        raise ValueError(
            "observable descriptor mode must be ordered, set_invariant, "
            "or role_aligned")
    names = []
    names.extend(
        f"channel_{index}_mean" for index in range(MAX_OBSERVABLE_CHANNELS))
    names.extend(
        f"channel_{index}_scale" for index in range(MAX_OBSERVABLE_CHANNELS))
    names.extend(
        f"channel_{index}_present" for index in range(MAX_OBSERVABLE_CHANNELS))
    names.extend(f"occupancy_bin_{index}" for index in range(OCCUPANCY_BINS))
    names.extend(("q10", "q25", "median", "q75", "q90"))
    names.extend((
        "global_mean", "global_scale", "global_minimum", "global_maximum",
        "mean_absolute_transition", "transition_scale",
    ))
    names.extend(
        f"low_frequency_{index}" for index in range(1, LOW_FREQUENCY_COMPONENTS + 1))
    for left in range(MAX_OBSERVABLE_CHANNELS):
        for right in range(left, MAX_OBSERVABLE_CHANNELS):
            names.append(f"channel_product_{left}_{right}")
    for left in range(MAX_OBSERVABLE_CHANNELS):
        for right in range(left + 1, MAX_OBSERVABLE_CHANNELS):
            names.append(f"channel_difference_{left}_{right}")
    return tuple(names)
