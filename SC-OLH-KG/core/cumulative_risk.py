"""State-coupled cumulative risk coordinates.

The paper-facing SC-OLH-KG path uses one shared coordinate object
``psi(x) = (A(x), N(x))``.  ``A`` are local/idiosyncratic risk exposures and
``N`` are shared-shock exposures.  HVD, certification, state candidates, and
exact KG all consume this same representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RiskExposure:
    """Local/shared risk exposure returned by a cumulative-risk provider."""

    A: np.ndarray
    N: np.ndarray
    local_names: tuple[str, ...] = field(default_factory=tuple)
    shared_names: tuple[str, ...] = field(default_factory=tuple)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.A = np.asarray(self.A, dtype=float).reshape(-1)
        self.N = np.asarray(self.N, dtype=float).reshape(-1)
        if not self.local_names:
            self.local_names = tuple(f"A{i}" for i in range(len(self.A)))
        if not self.shared_names:
            self.shared_names = tuple(f"N{i}" for i in range(len(self.N)))
        self.local_names = tuple(str(v) for v in self.local_names)
        self.shared_names = tuple(str(v) for v in self.shared_names)
        if len(self.local_names) != len(self.A):
            self.local_names = tuple(f"A{i}" for i in range(len(self.A)))
        if len(self.shared_names) != len(self.N):
            self.shared_names = tuple(f"N{i}" for i in range(len(self.N)))

    def __iter__(self):
        """Backwards-compatible unpacking: ``A, N = exposure``."""
        yield self.A
        yield self.N

    @property
    def n_local(self):
        return int(len(self.A))

    @property
    def n_shared(self):
        return int(len(self.N))


@dataclass
class CumulativeRiskParameters:
    """Parameters for floor + A^T Lambda A + N^T B N + N^T omega."""

    Lambda: np.ndarray
    B: np.ndarray
    omega: np.ndarray
    floor: float = 0.0

    def __post_init__(self):
        self.Lambda = np.maximum(np.asarray(self.Lambda, dtype=float).reshape(-1), 0.0)
        self.B = project_psd(np.asarray(self.B, dtype=float))
        self.omega = np.maximum(np.asarray(self.omega, dtype=float).reshape(-1), 0.0)
        self.floor = max(float(self.floor), 0.0)


def as_risk_exposure(value) -> RiskExposure | None:
    """Coerce legacy ``(A, N)`` tuples or ``RiskExposure`` instances."""

    if value is None:
        return None
    if isinstance(value, RiskExposure):
        return value
    if isinstance(value, dict) and "A" in value and "N" in value:
        return RiskExposure(
            value["A"],
            value["N"],
            local_names=tuple(value.get("local_names", ())),
            shared_names=tuple(value.get("shared_names", ())),
            meta=dict(value.get("meta", {})),
        )
    try:
        A, N = value
    except (TypeError, ValueError):
        return None
    return RiskExposure(A, N)


def get_risk_exposure(problem, x, output_index=1) -> RiskExposure | None:
    """Return ``RiskExposure`` from a problem/provider if available."""

    if problem is None:
        return None
    if hasattr(problem, "risk_exposures"):
        try:
            value = problem.risk_exposures(x, output_index=output_index)
        except TypeError:
            try:
                value = problem.risk_exposures(x)
            except AttributeError:
                return None
        except AttributeError:
            return None
        return as_risk_exposure(value)
    provider = getattr(problem, "cumulative_risk_provider", None)
    if provider is not None and hasattr(provider, "risk_exposures"):
        try:
            return as_risk_exposure(provider.risk_exposures(x, output_index=output_index))
        except TypeError:
            return as_risk_exposure(provider.risk_exposures(x))
    return None


def vech_quadratic_features(N):
    """Features for ``N^T B N`` using diag terms and doubled cross terms."""

    N = np.asarray(N, dtype=float).reshape(-1)
    out = []
    for i in range(len(N)):
        for j in range(i, len(N)):
            value = N[i] * N[j] if i == j else 2.0 * N[i] * N[j]
            out.append(float(value))
    return np.asarray(out, dtype=float)


def vech_names(shared_names):
    names = tuple(str(v) for v in shared_names)
    out = []
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names[i:], start=i):
            if i == j:
                out.append(f"{name_i}^2")
            else:
                out.append(f"2*{name_i}*{name_j}")
    return out


def cumulative_feature_vector(exposure: RiskExposure):
    exposure = as_risk_exposure(exposure)
    if exposure is None:
        return None
    return np.concatenate([
        np.array([1.0], dtype=float),
        np.asarray(exposure.A, dtype=float) ** 2,
        vech_quadratic_features(exposure.N),
        np.asarray(exposure.N, dtype=float),
    ])


def cumulative_feature_names(exposure: RiskExposure):
    exposure = as_risk_exposure(exposure)
    if exposure is None:
        return None
    return (
        ["floor"]
        + [f"{name}^2" for name in exposure.local_names]
        + vech_names(exposure.shared_names)
        + list(exposure.shared_names)
    )


def cumulative_layout(exposure: RiskExposure):
    exposure = as_risk_exposure(exposure)
    if exposure is None:
        return None
    n_a = exposure.n_local
    n_n = exposure.n_shared
    n_b = n_n * (n_n + 1) // 2
    return {
        "n_local": n_a,
        "n_shared": n_n,
        "floor": 0,
        "lambda": slice(1, 1 + n_a),
        "B": slice(1 + n_a, 1 + n_a + n_b),
        "omega": slice(1 + n_a + n_b, 1 + n_a + n_b + n_n),
        "feature_dim": 1 + n_a + n_b + n_n,
    }


def project_psd(B):
    B = np.asarray(B, dtype=float)
    if B.size == 0:
        return B.reshape(0, 0)
    B = 0.5 * (B + B.T)
    try:
        vals, vecs = np.linalg.eigh(B)
        vals = np.maximum(vals, 0.0)
        out = (vecs * vals) @ vecs.T
    except np.linalg.LinAlgError:
        out = np.maximum(B, 0.0)
    return 0.5 * (out + out.T)


def params_to_beta(params: CumulativeRiskParameters):
    lam = np.asarray(params.Lambda, dtype=float).reshape(-1)
    B = np.asarray(params.B, dtype=float)
    omega = np.asarray(params.omega, dtype=float).reshape(-1)
    vech = []
    for i in range(len(omega)):
        for j in range(i, len(omega)):
            vech.append(float(B[i, j]))
    return np.concatenate([[float(params.floor)], lam, np.asarray(vech), omega])


def beta_to_params(beta, exposure: RiskExposure) -> CumulativeRiskParameters:
    exposure = as_risk_exposure(exposure)
    layout = cumulative_layout(exposure)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    if layout is None or len(beta) < layout["feature_dim"]:
        raise ValueError("beta is incompatible with risk-exposure layout")
    floor = max(float(beta[layout["floor"]]), 0.0)
    lam = np.maximum(beta[layout["lambda"]], 0.0)
    n = int(layout["n_shared"])
    b_vals = beta[layout["B"]]
    B = np.zeros((n, n), dtype=float)
    pos = 0
    for i in range(n):
        for j in range(i, n):
            B[i, j] = B[j, i] = float(b_vals[pos])
            pos += 1
    omega = np.maximum(beta[layout["omega"]], 0.0)
    return CumulativeRiskParameters(lam, project_psd(B), omega, floor=floor)


def project_cumulative_beta(beta, exposure: RiskExposure):
    params = beta_to_params(beta, exposure)
    return params_to_beta(params), params


def decompose_cumulative_risk(exposure: RiskExposure, params: CumulativeRiskParameters):
    exposure = as_risk_exposure(exposure)
    params = CumulativeRiskParameters(
        params.Lambda,
        params.B,
        params.omega,
        floor=params.floor,
    )
    if exposure.n_local != len(params.Lambda):
        raise ValueError("local exposure and Lambda dimensions disagree")
    if exposure.n_shared != len(params.omega) or params.B.shape != (
        exposure.n_shared,
        exposure.n_shared,
    ):
        raise ValueError("shared exposure and shock parameter dimensions disagree")
    independent = float(np.sum(params.Lambda * exposure.A ** 2))
    shared = float(exposure.N @ params.B @ exposure.N)
    linear = float(exposure.N @ params.omega)
    floor = float(params.floor)
    total = float(max(floor + independent + shared + linear, 0.0))
    return {
        "A": exposure.A.tolist(),
        "N": exposure.N.tolist(),
        "Lambda": params.Lambda.tolist(),
        "B": params.B.tolist(),
        "omega": params.omega.tolist(),
        "floor": floor,
        "independent": independent,
        "shared": shared,
        "linear": linear,
        "total": total,
    }


class CumulativeRiskFeatureProvider:
    """Mixin for interpretable cumulative risk providers."""

    def risk_coordinate(self, x, output_index=1):
        exposure = get_risk_exposure(self, x, output_index=output_index)
        if exposure is None:
            return None
        return np.concatenate([exposure.A, exposure.N]).astype(float)

    def cumulative_risk_features(self, x, output_index=1):
        exposure = get_risk_exposure(self, x, output_index=output_index)
        return cumulative_feature_vector(exposure)

    def cumulative_risk_feature_names(self, output_index=1):
        del output_index
        if hasattr(self, "_reference_risk_exposure"):
            exposure = self._reference_risk_exposure()
        else:
            exposure = RiskExposure([], [])
        return cumulative_feature_names(exposure)

    def true_cumulative_risk_decomposition(self, x, output_index=1):
        if not hasattr(self, "cumulative_risk_parameters"):
            return None
        params = self.cumulative_risk_parameters(output_index=output_index)
        if params is None:
            return None
        if isinstance(params, dict):
            params = CumulativeRiskParameters(
                params["Lambda"],
                params["B"],
                params["omega"],
                floor=params.get("floor", 0.0),
            )
        exposure = get_risk_exposure(self, x, output_index=output_index)
        if exposure is None:
            return None
        return decompose_cumulative_risk(exposure, params)

    def cumulative_risk_provider_status(self):
        return {
            "status": "available",
            "provider": type(self).__name__,
            "coordinate": "psi=(A,N)",
        }

    def state_anchor_points(self, n=10, rng=None):
        """Default SC anchors in the provider's psi=(A,N) coordinates."""
        rng = rng or np.random.default_rng()
        n = max(0, int(n))
        if n <= 0 or not hasattr(self, "sample_random"):
            return []
        pool = []
        if hasattr(self, "structured_candidates"):
            try:
                pool.extend(self.structured_candidates(n=max(8, n), rng=rng))
            except TypeError:
                pool.extend(self.structured_candidates(max(8, n)))
        for _ in range(max(16, 8 * n)):
            pool.append(self.sample_random(rng))
        pool = _unique_tuples(pool)
        if not pool:
            return []
        psi = []
        kept = []
        for x in pool:
            coord = self.risk_coordinate(x)
            if coord is None or len(coord) == 0 or not np.all(np.isfinite(coord)):
                continue
            kept.append(tuple(int(v) for v in x))
            psi.append(coord)
        if not kept:
            return []
        Z = np.vstack(psi)
        chosen = [int(np.argmin(np.linalg.norm(Z, axis=1)))]
        while len(chosen) < min(n, len(kept)):
            dist = np.min(
                np.linalg.norm(Z[:, None, :] - Z[chosen][None, :, :], axis=2),
                axis=1,
            )
            dist[chosen] = -1.0
            chosen.append(int(np.argmax(dist)))
        anchors = []
        for idx in chosen[:n]:
            exposure = get_risk_exposure(self, kept[idx])
            anchors.append({
                "psi": Z[idx].tolist(),
                "A": exposure.A.tolist() if exposure is not None else [],
                "N": exposure.N.tolist() if exposure is not None else [],
                "candidate": list(map(int, kept[idx])),
                "coordinate": "psi=(A,N)",
            })
        return anchors

    def inverse_state_anchor(self, anchor, rng=None, n=1):
        """Invert a psi anchor by nearest-neighbor search in raw policy space."""
        rng = rng or np.random.default_rng()
        n = max(1, int(n))
        rows = []
        if isinstance(anchor, dict) and anchor.get("candidate") is not None:
            rows.append(tuple(int(v) for v in anchor["candidate"]))
        target = None
        if isinstance(anchor, dict):
            if anchor.get("psi") is not None:
                target = np.asarray(anchor["psi"], dtype=float)
            elif anchor.get("A") is not None and anchor.get("N") is not None:
                target = np.concatenate([
                    np.asarray(anchor["A"], dtype=float).reshape(-1),
                    np.asarray(anchor["N"], dtype=float).reshape(-1),
                ])
        if target is None or not hasattr(self, "sample_random"):
            return _unique_tuples(rows)
        pool = list(rows)
        if hasattr(self, "structured_candidates"):
            try:
                pool.extend(self.structured_candidates(n=max(8, 4 * n), rng=rng))
            except TypeError:
                pool.extend(self.structured_candidates(max(8, 4 * n)))
        for _ in range(max(32, 64 * n)):
            pool.append(self.sample_random(rng))
        candidates = []
        distances = []
        for x in _unique_tuples(pool):
            coord = self.risk_coordinate(x)
            if coord is None or len(coord) != len(target):
                continue
            candidates.append(tuple(int(v) for v in x))
            distances.append(float(np.linalg.norm(coord - target)))
        if candidates:
            order = np.argsort(np.asarray(distances, dtype=float))
            rows.extend(candidates[int(idx)] for idx in order[:n])
        return _unique_tuples(rows)[:n]


def _unique_tuples(rows):
    seen = set()
    out = []
    for row in rows:
        item = tuple(int(v) for v in row)
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
