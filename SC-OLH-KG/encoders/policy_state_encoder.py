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
        """Occupancy coverage score for state-coupled exploration.

        The early prototype rewarded similarity to already observed states,
        which made the coupling term largely redundant.  SC exploration should
        instead favor candidates whose occupancy proxy is under-covered by the
        current sample set.
        """
        observed = self._observed_x(observed)
        if not candidates:
            return np.zeros(0, dtype=float)
        if not observed:
            return np.ones(len(candidates), dtype=float)
        cand = np.vstack([self.occupancy(x) for x in candidates])
        obs = np.vstack([self.occupancy(tuple(o)) for o in observed])
        diff = cand[:, None, :] - obs[None, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        scale = max(self.lengthscale, 1e-8) ** 2
        similarity = np.exp(-0.5 * dist2 / scale)
        max_similarity = np.max(similarity, axis=1)
        scores = 1.0 - np.clip(max_similarity, 0.0, 1.0)
        hi = float(np.max(scores))
        lo = float(np.min(scores))
        if hi - lo <= 1e-14:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def coupling_scores(self, candidates, observed):
        """State-coupling score combining promising states and coverage."""
        coverage = self.propagation_scores(candidates, observed)
        history = self._observed_history(observed)
        if len(candidates) == 0 or not history:
            return coverage
        z = 1.6448536269514722
        sigma = float(getattr(self.problem, "sigma_level", 0.04))
        feasible = []
        for x, y in history:
            margin = float(y[1]) + z * sigma - float(getattr(self.problem, "tau", 0.0))
            if margin <= 0.0:
                feasible.append((x, y))
        if not feasible:
            return coverage

        obs_x = [x for x, _ in feasible]
        obs_y = np.array([float(y[0]) for _, y in feasible], dtype=float)
        scale_y = max(float(np.std(obs_y)), 1e-8)
        weights = np.exp(-(obs_y - float(np.min(obs_y))) / scale_y)
        cand = np.vstack([self.occupancy(x) for x in candidates])
        obs = np.vstack([self.occupancy(x) for x in obs_x])
        dist2 = np.sum((cand[:, None, :] - obs[None, :, :]) ** 2, axis=2)
        sim = np.exp(-0.5 * dist2 / max(self.lengthscale, 1e-8) ** 2)
        promising = np.max(sim * weights[None, :], axis=1)
        promising = self._normalize01(promising)
        return self._normalize01(0.75 * promising + 0.25 * coverage)

    def state_space_candidates(
        self,
        n_anchors=10,
        inverse_pool_size=500,
        inverse_neighbors=1,
        rng=None,
        observed=None,
    ):
        """Generate raw candidates by searching through state/meta anchors.

        This is the first real SC candidate-generation path: propose anchors in
        a lower-dimensional state/meta space, then invert each anchor back to
        one or more raw decision vectors.  Problems can provide exact synthetic
        inverses; otherwise we approximate the inverse by nearest-neighbor
        matching in occupancy space over a random raw pool.
        """
        rng = rng or np.random.default_rng()
        n_anchors = max(0, int(n_anchors))
        inverse_neighbors = max(1, int(inverse_neighbors))
        if n_anchors <= 0:
            return []

        if (
            hasattr(self.problem, "state_anchor_points")
            and hasattr(self.problem, "inverse_state_anchor")
        ):
            rows = []
            anchors = self.problem.state_anchor_points(n=n_anchors, rng=rng)
            for anchor in anchors:
                rows.extend(self.problem.inverse_state_anchor(
                    anchor,
                    rng=rng,
                    n=inverse_neighbors,
                ))
            return self._unique(rows)

        pool = self._raw_inverse_pool(inverse_pool_size, rng, observed)
        if not pool:
            return []
        target = rng.random((n_anchors, self.feature_dim))
        occ = np.vstack([self.occupancy(x) for x in pool])
        chosen = []
        for rho in target:
            dist = np.linalg.norm(occ - rho[None, :], axis=1)
            for idx in np.argsort(dist)[:inverse_neighbors]:
                chosen.append(pool[int(idx)])
        return self._unique(chosen)

    def _raw_inverse_pool(self, n, rng, observed=None):
        pool = []
        pool.extend(self._observed_x(observed or []))
        if hasattr(self.problem, "structured_candidates"):
            pool.extend(self.problem.structured_candidates(
                n=max(5, int(n) // 10),
                rng=rng,
            ))
        for _ in range(max(0, int(n))):
            pool.append(self.problem.sample_random(rng))
        return self._unique(pool)

    @staticmethod
    def _observed_x(observed):
        rows = []
        for item in observed or []:
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and not np.isscalar(item[1])
            ):
                rows.append(tuple(int(v) for v in item[0]))
            else:
                rows.append(tuple(int(v) for v in item))
        return rows

    @staticmethod
    def _observed_history(observed):
        rows = []
        for item in observed or []:
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and not np.isscalar(item[1])
            ):
                rows.append((tuple(int(v) for v in item[0]), np.asarray(item[1], dtype=float)))
        return rows

    @staticmethod
    def _normalize01(values):
        arr = np.asarray(values, dtype=float)
        if len(arr) == 0:
            return arr
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi - lo <= 1e-14:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    @staticmethod
    def _unique(candidates):
        seen = set()
        rows = []
        for x in candidates:
            x_tuple = tuple(int(v) for v in x)
            if x_tuple not in seen:
                seen.add(x_tuple)
                rows.append(x_tuple)
        return rows


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
