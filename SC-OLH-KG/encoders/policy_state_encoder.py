"""Deterministic state-policy coupling features for synthetic experiments."""

from __future__ import annotations

import numpy as np


class SyntheticPolicyStateEncoder:
    """Map a policy/design vector to a deterministic occupancy proxy.

    Real traffic trajectory encoding is intentionally deferred.  For synthetic
    RZDT-like problems this encoder creates a low-dimensional proxy for
    policy-induced occupancy/risk regimes from normalized decision variables.
    """

    def __init__(self, problem, lengthscale=0.35):
        self.problem = problem
        self.lengthscale = float(lengthscale)
        self.feature_dim = 8

    def occupancy(self, x):
        z = np.asarray(self.problem.normalize(x), dtype=float)
        center = np.full_like(z, 0.5)
        u0 = float(z[0]) if len(z) else 0.0
        tail = z[1:] if len(z) > 1 else np.array([0.0])
        return np.array([
            float(np.mean(z)),
            float(np.std(z)),
            float(np.min(z)),
            float(np.max(z)),
            float(np.linalg.norm(z - center) / np.sqrt(max(1, len(z)))),
            float(np.sin(np.pi * u0)),
            float(np.cos(np.pi * u0)),
            float(np.mean(tail)),
        ], dtype=float)

    def features(self, x):
        return self.occupancy(x)

    def distance(self, x, y):
        dx = self.occupancy(x) - self.occupancy(y)
        return float(np.linalg.norm(dx))

    def kernel(self, x, y):
        dist = self.distance(x, y)
        return float(np.exp(-0.5 * (dist / max(self.lengthscale, 1e-8)) ** 2))

    def propagation_scores(self, candidates, observed):
        """How much one sample could propagate through occupancy similarity."""
        if not candidates:
            return np.zeros(0, dtype=float)
        if not observed:
            return np.ones(len(candidates), dtype=float)
        obs = [tuple(o) for o in observed]
        scores = []
        for x in candidates:
            vals = [self.kernel(x, y) for y in obs]
            scores.append(float(np.mean(vals) + np.max(vals)))
        scores = np.asarray(scores, dtype=float)
        if np.max(scores) > 0:
            scores = scores / np.max(scores)
        return scores


class StateCoupledFeatureMap:
    """Feature map for the mean belief model.

    It combines normalized design variables and occupancy features.  The final
    GPR basis adds an intercept outside this object.
    """

    def __init__(self, problem, encoder=None, state_scale=0.2):
        self.problem = problem
        self.encoder = encoder or SyntheticPolicyStateEncoder(problem)
        self.state_scale = float(state_scale)
        d = int(problem.d)
        rho_d = int(self.encoder.feature_dim)
        self.feature_dim = 2 * d + rho_d

    def features(self, x):
        z = np.asarray(self.problem.normalize(x), dtype=float)
        rho = self.state_scale * self.encoder.occupancy(x)
        return np.concatenate([z, z ** 2, rho])
