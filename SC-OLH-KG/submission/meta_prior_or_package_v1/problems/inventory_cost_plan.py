"""Public inventory dynamics with unknown Gaussian cumulative cost risk.

Demand is a known planning schedule. Policies observe inventory before ordering;
uncertainty enters costs, not demand. Computing trajectories therefore uses no
unknown target outcomes. This is separate from the legacy synthetic Inventory
ellipses and is not a learned trajectory encoder.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import digamma, polygamma

from core.cumulative_risk import CumulativeRiskParameters, RiskExposure


@dataclass(frozen=True)
class InventoryTrajectory:
    stock_before: np.ndarray
    orders: np.ndarray
    lost_sales: np.ndarray
    stock_after: np.ndarray
    local_weights: np.ndarray
    cost_features: np.ndarray

    def risk_exposures(self):
        return RiskExposure(
            np.linalg.norm(self.local_weights, axis=0),
            self.local_weights[:, [0, 2]].sum(axis=0),
            local_names=("order_cost", "holding_cost", "lost_sales_cost"),
            shared_names=("order_price", "lost_sales_price"))

    def coordinates(self, capacity):
        exposure = self.risk_exposures()
        # Eight public state/cost coordinates; the last three retain average
        # holding, ordering frequency and terminal stock, respectively.
        return np.r_[exposure.A, exposure.N, self.cost_features[[1, 3]],
                     self.stock_after[-1] / capacity]


@dataclass(frozen=True)
class InventoryPlan:
    demand: tuple = (4., 9., 6., 12., 7., 14., 5., 11., 8., 13., 6., 10.)
    capacity: float = 20.
    initial_stock: float = 8.
    cost_budget: float = 1.

    def __post_init__(self):
        if (not len(self.demand) or min(self.demand) < 0 or sum(self.demand) <= 0
                or self.capacity <= 0 or not 0 <= self.initial_stock <= self.capacity):
            raise ValueError("inventory planning requires positive demand/capacity and valid stock")

    def trajectory(self, policy):
        """Four inventory bins select an order-up-to level in [0, capacity]."""
        policy = np.asarray(policy, dtype=float)
        if policy.shape != (4,) or not np.isfinite(policy).all() or np.any((policy < 0) | (policy > 1)):
            raise ValueError("policy must contain four order-up-to fractions in [0,1]")
        stock = self.initial_stock
        before, orders, lost, after = [], [], [], []
        for demand in self.demand:
            before.append(stock)
            index = min(int(4 * stock / self.capacity), 3)
            order = max(policy[index] * self.capacity - stock, 0.)
            available = stock + order
            orders.append(order)
            lost.append(max(demand - available, 0.))
            stock = max(available - demand, 0.)
            after.append(stock)
        before, orders, lost, after = map(np.asarray, (before, orders, lost, after))
        local = np.column_stack([orders / (self.capacity * len(self.demand)),
                                 after / (self.capacity * len(self.demand)),
                                 lost / sum(self.demand)])
        features = np.r_[local[:, 0].sum(), local[:, 1].sum(), local[:, 2].sum(),
                         np.mean(orders > 0)]
        return InventoryTrajectory(before, orders, lost, after, local, features)

    def coordinates_many(self, policies):
        return np.array([self.trajectory(x).coordinates(self.capacity) for x in policies])


@dataclass(frozen=True)
class InventoryCostLaw:
    """Private simulator parameters; learners receive batch labels only."""
    objective_prices: np.ndarray
    constraint_prices: np.ndarray
    risk: CumulativeRiskParameters

    def __post_init__(self):
        prices = np.asarray(self.objective_prices)
        if prices.shape != (4,) or np.any((prices < 0) | (prices > 1)):
            raise ValueError("objective prices must respect the declared [0,1] bounds")


def sample_cost_law(seed, constraint_price_range=(.6, 1.4)):
    """Declared synthetic task family, without a hidden optimum or basin.

    ``constraint_price_range`` is part of the public task contract.  The
    default preserves the original Inventory experiments; a narrower range is
    used by the later feasible-deployment contract so that feasibility is not
    accidentally determined by an almost-always-infeasible constraint law.
    """
    low, high = map(float, constraint_price_range)
    if not 0 <= low <= high:
        raise ValueError("constraint price range must satisfy 0 <= low <= high")
    rng = np.random.default_rng(seed)
    shared_scale = rng.uniform(.12, .35, 2)
    correlation = rng.uniform(-.35, .35)
    B = np.diag(shared_scale ** 2)
    B[0, 1] = B[1, 0] = correlation * np.prod(shared_scale)
    return InventoryCostLaw(rng.uniform(.35, 1., 4), rng.uniform(low, high, 4),
        CumulativeRiskParameters(rng.uniform(.2, 1., 3), B,
                                 rng.uniform(.01, .04, 2), floor=.0025))


class InventoryCostSimulator:
    """Paid episode observations; oracle moments are for diagnostics only."""
    objective_weights = np.array([1.5, 2., 5., 1.5])
    constraint_weights = np.array([1.2, .8, 2., .2])
    objective_sd = .12

    def __init__(self, plan, law):
        self.plan, self._law = plan, law
        self.episode_calls = 0

    def _means(self, trajectory):
        # Each feature lies in [0,1], so objective mean is inside [-5,5].
        f = trajectory.cost_features @ (self.objective_weights * self._law.objective_prices) - 5.
        g = trajectory.cost_features @ (self.constraint_weights * self._law.constraint_prices) - self.plan.cost_budget
        return float(f), float(g)

    def oracle_moments(self, policy):
        trajectory = self.plan.trajectory(policy)
        exposure, risk = trajectory.risk_exposures(), self._law.risk
        variance = (risk.floor + exposure.A ** 2 @ risk.Lambda
                    + exposure.N @ risk.B @ exposure.N + exposure.N @ risk.omega)
        return np.r_[self._means(trajectory), np.log(variance)]

    def episodes(self, policy, reps, rng):
        trajectory = self.plan.trajectory(policy)
        exposure, risk = trajectory.risk_exposures(), self._law.risk
        local = rng.normal(size=(reps, len(self.plan.demand), 3)) * np.sqrt(risk.Lambda)
        local_cost = (local * trajectory.local_weights).sum(axis=(1, 2))
        shared = rng.multivariate_normal(np.zeros(2), risk.B, reps) @ exposure.N
        # Gaussian volume-dependent cost noise: this is not a Poisson process.
        volume_weights = trajectory.local_weights[:, [0, 2]]
        volume = (rng.normal(size=(reps, len(self.plan.demand), 2))
                  * np.sqrt(volume_weights * risk.omega)).sum(axis=(1, 2))
        f, g = self._means(trajectory)
        cost = g + local_cost + shared + volume + np.sqrt(risk.floor) * rng.normal(size=reps)
        objective = f + self.objective_sd * rng.normal(size=reps)
        self.episode_calls += reps
        return np.column_stack([objective, cost])

    def observe(self, policy, reps, rng):
        if reps < 2:
            raise ValueError("risk labels require at least two independent episodes")
        outcomes = self.episodes(policy, reps, rng)
        sample_variance = float(np.var(outcomes[:, 1], ddof=1))
        df = reps - 1
        bias = float(digamma(df / 2) + np.log(2 / df))
        response = np.r_[outcomes.mean(axis=0), np.log(sample_variance) - bias]
        noise = np.array([self.objective_sd ** 2 / reps, sample_variance / reps,
                          float(polygamma(1, df / 2))])
        return response, noise
