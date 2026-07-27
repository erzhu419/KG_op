"""Frozen source archives and explicit transfer-learning information contracts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm


def dimension_equivariant_profile_features(
    profiles,
    *,
    max_frequency=8,
    frequency_penalty=0.10,
):
    """Map policies of any raw dimension to one fixed ordered coordinate.

    The map is the frozen low-frequency coordinate used by the cross-dimension
    proposal: mean, standard deviation, cosine coefficients, and their upper
    triangular quadratic products.  It is deterministic and label-free.
    """

    values = np.asarray(profiles, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("profiles must be a nonempty finite matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("profiles contain non-finite values")
    max_frequency = max(0, int(max_frequency))
    frequency_penalty = max(0.0, float(frequency_penalty))
    positions = (
        np.arange(values.shape[1], dtype=float) + 0.5
    ) / float(values.shape[1])
    centered = values - np.mean(values, axis=1, keepdims=True)
    columns = [
        np.mean(values, axis=1),
        np.std(values, axis=1),
    ]
    for frequency in range(1, max_frequency + 1):
        columns.append(2.0 * np.mean(
            centered
            * np.cos(np.pi * frequency * positions)[None, :],
            axis=1,
        ))
    linear = np.column_stack(columns)
    if max_frequency:
        weights = np.concatenate([
            np.ones(2, dtype=float),
            1.0 / (
                1.0
                + frequency_penalty
                * np.arange(max_frequency, dtype=float)
            ),
        ])
        linear = linear * weights[None, :]
    upper = np.triu_indices(linear.shape[1])
    quadratic = (
        linear[:, upper[0]] * linear[:, upper[1]]
    )
    coordinate = np.concatenate([linear, quadratic], axis=1)
    scale = np.maximum(
        np.linalg.norm(coordinate, axis=1, keepdims=True),
        1.0,
    )
    return coordinate / scale


def resample_normalized_profiles(profiles, target_dimension):
    """Interpolate normalized policy curves onto a target policy grid."""

    values = np.asarray(profiles, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("profiles must be a nonempty finite matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("profiles contain non-finite values")
    target_dimension = max(1, int(target_dimension))
    if values.shape[1] == target_dimension:
        return values.copy()
    if values.shape[1] == 1:
        return np.repeat(values, target_dimension, axis=1)
    source_grid = np.linspace(0.0, 1.0, values.shape[1])
    target_grid = np.linspace(0.0, 1.0, target_dimension)
    return np.vstack([
        np.interp(target_grid, source_grid, row)
        for row in values
    ])


@dataclass(frozen=True)
class TransferTaskArchive:
    """Ordinary replicated observations from one source domain."""

    name: str
    X: np.ndarray
    X_integer: np.ndarray
    Y_mean: np.ndarray
    Y_replicates: tuple[np.ndarray, ...]
    replicate_variance: np.ndarray
    mean_observation_variance: np.ndarray
    constraint_sigma: np.ndarray
    tau: float
    alpha: float
    origins: tuple[str, ...]

    @property
    def n_profiles(self):
        return int(len(self.X))

    @property
    def simulator_calls(self):
        return int(sum(len(values) for values in self.Y_replicates))

    def chance_margin(self):
        z_alpha = float(norm.ppf(1.0 - self.alpha))
        return (
            self.Y_mean[:, 1]
            + z_alpha * self.constraint_sigma
            - self.tau
        )


@dataclass(frozen=True)
class FrozenTransferArchive:
    """The exact source data shared by every transfer baseline."""

    tasks: tuple[TransferTaskArchive, ...]
    source_seed: int
    observation_mode: str
    fingerprint: str

    @property
    def source_domains(self):
        return tuple(task.name for task in self.tasks)

    @property
    def profiles_per_domain(self):
        return {
            task.name: task.n_profiles for task in self.tasks
        }

    @property
    def simulator_calls(self):
        return int(sum(task.simulator_calls for task in self.tasks))

    def information_contract(self):
        replicate_counts = sorted({
            len(values)
            for task in self.tasks
            for values in task.Y_replicates
        })
        return {
            "n_source_domains": int(len(self.tasks)),
            "source_domains": list(self.source_domains),
            "source_profiles_per_domain": self.profiles_per_domain,
            "source_observation_replicates": replicate_counts,
            "source_simulator_calls": int(self.simulator_calls),
            "source_seed": int(self.source_seed),
            "source_observation_mode": self.observation_mode,
            "archive_fingerprint": self.fingerprint,
            "source_oracle_aided": False,
            "source_true_outputs_used": False,
            "source_true_sigma_used": False,
        }

    def validate(self, *, expected_domains=None, expected_dimension=None):
        if not self.tasks:
            raise ValueError("a frozen transfer archive needs source tasks")
        dimensions = {int(task.X.shape[1]) for task in self.tasks}
        if len(dimensions) != 1:
            raise ValueError("all source tasks must share one input dimension")
        if expected_dimension is not None and dimensions != {
            int(expected_dimension)
        }:
            raise ValueError(
                "source archive dimension does not match the target problem"
            )
        if expected_domains is not None and set(self.source_domains) != set(
            map(str, expected_domains)
        ):
            raise ValueError("source archive domains do not match the LODO split")
        for task in self.tasks:
            n = task.n_profiles
            if task.X_integer.shape != task.X.shape:
                raise ValueError("normalized and integer source designs differ")
            if task.Y_mean.shape != (n, 2):
                raise ValueError("source means must have shape (profiles, 2)")
            if len(task.Y_replicates) != n:
                raise ValueError("source replicate rows do not match profiles")
            if task.replicate_variance.shape != (n, 2):
                raise ValueError("source replicate variance has invalid shape")
            if task.mean_observation_variance.shape != (n, 2):
                raise ValueError("source mean variance has invalid shape")
            if not np.all(np.isfinite(task.X)) or not np.all(
                np.isfinite(task.Y_mean)
            ):
                raise ValueError("source archive contains non-finite values")
        return self

    def to_payload(self):
        return {
            "schema_version": 1,
            "source_seed": int(self.source_seed),
            "observation_mode": str(self.observation_mode),
            "fingerprint": str(self.fingerprint),
            "tasks": [
                {
                    "name": task.name,
                    "X": task.X.tolist(),
                    "X_integer": task.X_integer.tolist(),
                    "Y_mean": task.Y_mean.tolist(),
                    "Y_replicates": [row.tolist() for row in task.Y_replicates],
                    "replicate_variance": task.replicate_variance.tolist(),
                    "mean_observation_variance": (
                        task.mean_observation_variance.tolist()
                    ),
                    "constraint_sigma": task.constraint_sigma.tolist(),
                    "tau": float(task.tau),
                    "alpha": float(task.alpha),
                    "origins": list(task.origins),
                }
                for task in self.tasks
            ],
        }

    @classmethod
    def from_payload(cls, payload):
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported frozen transfer archive schema")
        tasks = tuple(TransferTaskArchive(
            name=str(row["name"]),
            X=np.asarray(row["X"], dtype=float),
            X_integer=np.asarray(row["X_integer"], dtype=int),
            Y_mean=np.asarray(row["Y_mean"], dtype=float),
            Y_replicates=tuple(
                np.asarray(values, dtype=float)
                for values in row["Y_replicates"]
            ),
            replicate_variance=np.asarray(
                row["replicate_variance"], dtype=float),
            mean_observation_variance=np.asarray(
                row["mean_observation_variance"], dtype=float),
            constraint_sigma=np.asarray(row["constraint_sigma"], dtype=float),
            tau=float(row["tau"]),
            alpha=float(row["alpha"]),
            origins=tuple(map(str, row["origins"])),
        ) for row in payload["tasks"])
        archive = cls(
            tasks=tasks,
            source_seed=int(payload["source_seed"]),
            observation_mode=str(payload["observation_mode"]),
            fingerprint=str(payload["fingerprint"]),
        ).validate()
        if _archive_fingerprint(
            archive.tasks,
            archive.source_seed,
            archive.observation_mode,
        ) != archive.fingerprint:
            raise ValueError("frozen transfer archive fingerprint mismatch")
        return archive

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_payload(), separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        return cls.from_payload(json.loads(Path(path).read_text(
            encoding="utf-8")))


def _record_replicates(record):
    values = getattr(record, "replicates", None)
    if values is None:
        values = np.asarray(record.y, dtype=float).reshape(1, -1)
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("source replicates must have shape (replicates, 2)")
    return values.copy()


def _task_from_records(name, records):
    records = list(records)
    if not records:
        raise ValueError(f"source task {name!r} has no records")
    X = np.vstack([
        np.asarray(record.profile, dtype=float).reshape(-1)
        for record in records
    ])
    X_integer = np.vstack([
        np.asarray(record.x, dtype=int).reshape(-1)
        for record in records
    ])
    Y_mean = np.vstack([
        np.asarray(record.y, dtype=float).reshape(-1)
        for record in records
    ])
    replicates = tuple(_record_replicates(record) for record in records)
    nominal = np.asarray([
        max(float(record.sigma_level), 1e-8) ** 2
        for record in records
    ])
    replicate_variance = []
    mean_variance = []
    for values, nominal_variance in zip(replicates, nominal):
        if len(values) >= 2:
            sample = np.var(values, axis=0, ddof=1)
        else:
            sample = np.full(2, nominal_variance, dtype=float)
        sample = np.maximum(sample, 1e-12)
        replicate_variance.append(sample)
        mean_variance.append(sample / max(len(values), 1))
    constraint_sigma = np.asarray([
        (
            float(record.constraint_sigma)
            if record.constraint_sigma is not None
            else np.sqrt(replicate_variance[index][1])
        )
        for index, record in enumerate(records)
    ])
    tau_values = {float(record.tau) for record in records}
    alpha_values = {float(record.alpha) for record in records}
    if len(tau_values) != 1 or len(alpha_values) != 1:
        raise ValueError("one source task must have fixed tau and alpha")
    return TransferTaskArchive(
        name=str(name),
        X=X,
        X_integer=X_integer,
        Y_mean=Y_mean,
        Y_replicates=replicates,
        replicate_variance=np.vstack(replicate_variance),
        mean_observation_variance=np.vstack(mean_variance),
        constraint_sigma=np.maximum(constraint_sigma, 1e-8),
        tau=tau_values.pop(),
        alpha=alpha_values.pop(),
        origins=tuple(str(record.origin) for record in records),
    )


def _archive_fingerprint(tasks, source_seed, observation_mode):
    digest_payload = {
        "source_seed": int(source_seed),
        "observation_mode": str(observation_mode),
        "tasks": [
            {
                "name": task.name,
                "X": np.round(task.X, 14).tolist(),
                "Y_replicates": [
                    np.round(values, 14).tolist()
                    for values in task.Y_replicates
                ],
                "tau": task.tau,
                "alpha": task.alpha,
                "origins": list(task.origins),
            }
            for task in tasks
        ],
    }
    return hashlib.sha256(json.dumps(
        digest_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def frozen_archive_from_meta_prior(prior, source_seed=0):
    """Export the exact archive consumed by ``LearnedMetaPrior.fit``."""

    records = list(getattr(prior, "source_records_", ()))
    if not records:
        raise ValueError("meta-prior does not expose a fitted source archive")
    domains = sorted({str(record.domain) for record in records})
    tasks = tuple(_task_from_records(
        domain,
        [record for record in records if str(record.domain) == domain],
    ) for domain in domains)
    fingerprint = _archive_fingerprint(
        tasks,
        source_seed,
        prior.source_observation_mode,
    )
    return FrozenTransferArchive(
        tasks=tasks,
        source_seed=int(source_seed),
        observation_mode=str(prior.source_observation_mode),
        fingerprint=fingerprint,
    ).validate()
