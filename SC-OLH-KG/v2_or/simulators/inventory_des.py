"""Family 1: Multi-Period Capacitated Stochastic Inventory & Dual-Sourcing DES.

Genuine period-by-period dynamic state transitions over horizon `H` (`d = 2 * H`):
- State at period `h`: `(on_hand_stock I_h, pipeline_order P_h)`
- Policy `x in [0, 1]^{2H}`:
    * `x[2*h]`   -> regular order-up-to target fraction `S_reg_h` (1-period lead time,
                    subject to stochastic supplier yield)
    * `x[2*h+1]` -> emergency expediting trigger fraction `S_exp_h` (0-period lead time,
                    higher unit cost, capped by expedited capacity)
- Public nominal exposure `nominal_exposure(x)` rolls out the exact inventory equations
  under the PUBLIC deterministic forecast `demand_forecast` and nominal yield, with ZERO
  access to the private task shock/cost law (`InventoryEnvironmentLaw`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from v2_or.contracts import BaseDynamicORProblem, NominalTrajectoryExposure


@dataclass(frozen=True)
class InventoryEnvironmentLaw:
    """Private task-specific stochastic demand, yield, and cost law."""

    regime_drift: np.ndarray          # shape (3,): off-peak, shoulder, peak demand bias
    regime_shock_cov: np.ndarray      # shape (3, 3): correlated macro demand shock cov
    idiosyncratic_cv: float           # period-level demand noise CV
    yield_alpha: float                # Beta(alpha, beta) random supplier yield
    yield_beta: float
    reg_order_cost: float
    exp_order_cost: float
    holding_cost: float
    shortage_penalty: float
    setup_cost: float
    sla_max_shortage_rate: float      # chance-constraint threshold tau


def sample_inventory_tasks(
    n_tasks: int,
    *,
    seed: int,
    horizons: tuple[int, ...] = (48, 96),
    shift_scale: float = 1.0,
) -> list["DynamicInventoryNetworkDES"]:
    """Sample a family of related but genuinely distinct inventory environments."""
    rng = np.random.default_rng(seed)
    tasks: list[DynamicInventoryNetworkDES] = []
    for idx in range(int(n_tasks)):
        H = int(horizons[idx % len(horizons)])
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        t_grid = np.linspace(0.0, 4.0 * np.pi, H, endpoint=False)
        promo_center = float(rng.uniform(0.25, 0.80))
        u_grid = np.linspace(0.0, 1.0, H, endpoint=False)
        promo_bump = 0.45 * np.exp(-0.5 * ((u_grid - promo_center) / 0.08) ** 2)
        base_level = float(rng.uniform(8.5, 12.5))
        forecast = base_level * (
            1.0 + 0.32 * np.sin(t_grid + phase) + 0.14 * np.cos(2.0 * t_grid) + promo_bump
        )
        forecast = np.maximum(forecast, 2.0)

        drift = shift_scale * rng.uniform(-0.06, 0.14, size=3)
        stds = rng.uniform(0.06, 0.18, size=3) * np.sqrt(max(shift_scale, 0.25))
        corr = float(rng.uniform(0.10, 0.55))
        cov = np.outer(stds, stds) * corr
        np.fill_diagonal(cov, stds ** 2)

        law = InventoryEnvironmentLaw(
            regime_drift=drift,
            regime_shock_cov=cov,
            idiosyncratic_cv=float(rng.uniform(0.10, 0.24)),
            yield_alpha=float(rng.uniform(10.0, 22.0)),
            yield_beta=float(rng.uniform(1.0, 2.8)),
            reg_order_cost=float(rng.uniform(0.85, 1.25)),
            exp_order_cost=float(rng.uniform(2.10, 3.10)),
            holding_cost=float(rng.uniform(0.22, 0.45)),
            shortage_penalty=float(rng.uniform(3.80, 6.20)),
            setup_cost=float(rng.uniform(0.25, 0.60)),
            sla_max_shortage_rate=float(rng.uniform(0.045, 0.075)),
        )
        tasks.append(
            DynamicInventoryNetworkDES(
                task_id=f"inv_H{H}_{idx:03d}",
                demand_forecast=forecast,
                capacity=float(2.4 * np.max(forecast)),
                initial_stock=float(1.15 * forecast[0]),
                law=law,
            )
        )
    return tasks


class DynamicInventoryNetworkDES(BaseDynamicORProblem):
    """Multi-period stochastic dual-sourcing inventory discrete-event simulator."""

    family_name = "DynamicInventoryDES"
    is_gaussian_constraint_noise = False

    def __init__(
        self,
        *,
        task_id: str,
        demand_forecast: np.ndarray,
        capacity: float,
        initial_stock: float,
        law: InventoryEnvironmentLaw,
        alpha: float = 0.05,
    ) -> None:
        super().__init__()
        self.task_id = str(task_id)
        self.demand_forecast = np.asarray(demand_forecast, dtype=float).reshape(-1)
        self.H = int(self.demand_forecast.size)
        if self.H < 8 or np.any(self.demand_forecast <= 0.0):
            raise ValueError("demand_forecast must have >= 8 positive periods")
        self.d = 2 * self.H
        self.capacity = float(capacity)
        self.initial_stock = float(np.clip(initial_stock, 0.0, self.capacity))
        self._law = law
        self.alpha = float(alpha)

        # Public regime labels derived strictly from public forecast quantiles
        q33, q67 = np.quantile(self.demand_forecast, [1.0 / 3.0, 2.0 / 3.0])
        self._regime_idx = np.where(
            self.demand_forecast <= q33,
            0,
            np.where(self.demand_forecast <= q67, 1, 2),
        ).astype(int)
        self._forecast_norm = self.demand_forecast / float(np.mean(self.demand_forecast))

    def domain_heuristic_anchor(self) -> np.ndarray:
        """Conservative practitioner safety-stock rule (safe buffer, high holding cost)."""
        s_reg = np.clip(1.85 * self.demand_forecast / self.capacity, 0.35, 0.92)
        s_exp = np.full(self.H, 0.72, dtype=float)
        out = np.empty(self.d, dtype=float)
        out[0::2] = s_reg
        out[1::2] = s_exp
        return out

    def nominal_exposure(self, x_norm: np.ndarray) -> NominalTrajectoryExposure:
        """Roll out deterministic inventory balance under PUBLIC forecast only."""
        x = self.clip_policy(x_norm)
        s_reg = (0.10 + 0.85 * x[0::2]) * self.capacity
        s_exp = 0.70 * x[1::2] * s_reg
        exp_cap = 0.35 * self.capacity
        nominal_yield = 0.90

        stock = self.initial_stock
        pipeline = 0.0
        stocks = np.empty(self.H, dtype=float)
        reg_orders = np.empty(self.H, dtype=float)
        exp_orders = np.empty(self.H, dtype=float)
        shortages = np.empty(self.H, dtype=float)
        buffers = np.empty(self.H, dtype=float)

        for h in range(self.H):
            arrived = nominal_yield * pipeline
            avail_pre = min(self.capacity, stock + arrived)
            exp_q = min(max(s_exp[h] - avail_pre, 0.0), exp_cap)
            avail = min(self.capacity, avail_pre + exp_q)
            dem = float(self.demand_forecast[h])
            short = max(dem - avail, 0.0)
            stock_after = max(avail - dem, 0.0)
            reg_q = max(min(s_reg[h] - stock_after, self.capacity - stock_after), 0.0)

            stocks[h] = stock_after
            reg_orders[h] = reg_q
            exp_orders[h] = exp_q
            shortages[h] = short
            buffers[h] = max(avail - dem, 0.0) / max(dem, 1e-6)
            stock = stock_after
            pipeline = reg_q

        total_dem = float(np.sum(self.demand_forecast))
        lean_exposure = np.exp(-1.5 * buffers)

        # Dimension-invariant shared regime exposures N_shared in R^3
        n_shared = np.array(
            [
                float(np.mean(lean_exposure[self._regime_idx == k] * self._forecast_norm[self._regime_idx == k]))
                if np.any(self._regime_idx == k)
                else 0.0
                for k in range(3)
            ],
            dtype=float,
        )
        # Dimension-invariant local exposures A_local in R^4
        a_local = np.array(
            [
                float(np.sqrt(np.mean(lean_exposure ** 2))),
                float(np.sqrt(np.mean((reg_orders / self.capacity) ** 2))),
                float(np.sqrt(np.mean((exp_orders / self.capacity) ** 2))),
                float(np.sqrt(np.mean(np.diff(s_reg / self.capacity, prepend=s_reg[0] / self.capacity) ** 2))),
            ],
            dtype=float,
        )
        psi = np.array(
            [
                float(np.sum(reg_orders) / total_dem),
                float(np.sum(exp_orders) / total_dem),
                float(np.mean(stocks) / self.capacity),
                float(np.sum(shortages) / total_dem),
                float(np.mean(reg_orders > 0.02 * self.capacity)),
                float(np.mean(buffers)),
                float(np.quantile(lean_exposure, 0.85)),
                a_local[0],
                a_local[1],
                n_shared[0],
                n_shared[1],
                n_shared[2],
            ],
            dtype=float,
        )
        names = (
            "nom_reg_ratio",
            "nom_exp_ratio",
            "nom_stock_frac",
            "nom_shortage_frac",
            "nom_order_freq",
            "nom_mean_buffer",
            "nom_p85_lean",
            "A_lean_rms",
            "A_reg_rms",
            "N_offpeak_lean",
            "N_shoulder_lean",
            "N_peak_lean",
        )
        return NominalTrajectoryExposure(psi=psi, A_local=a_local, N_shared=n_shared, feature_names=names)

    def _simulate_raw(
        self,
        x_norm: np.ndarray,
        reps: int,
        rng: np.random.Generator,
        split: str = "search",
    ) -> np.ndarray:
        del split  # i.i.d. Monte Carlo generator controlled by caller's split-specific rng
        law = self._law
        s_reg = (0.10 + 0.85 * x_norm[0::2]) * self.capacity
        s_exp = 0.70 * x_norm[1::2] * s_reg
        exp_cap = 0.35 * self.capacity

        # Shared macro regime shocks per replication: (reps, 3)
        macro_shocks = rng.multivariate_normal(law.regime_drift, law.regime_shock_cov, size=reps)
        idio_noise = rng.normal(0.0, law.idiosyncratic_cv, size=(reps, self.H))
        yield_draws = rng.beta(law.yield_alpha, law.yield_beta, size=(reps, self.H))

        stock = np.full(reps, self.initial_stock, dtype=float)
        pipeline = np.zeros(reps, dtype=float)

        tot_reg = np.zeros(reps, dtype=float)
        tot_exp = np.zeros(reps, dtype=float)
        tot_hold = np.zeros(reps, dtype=float)
        tot_short = np.zeros(reps, dtype=float)
        tot_peak_short = np.zeros(reps, dtype=float)
        tot_setups = np.zeros(reps, dtype=float)
        tot_demand = np.zeros(reps, dtype=float)

        for h in range(self.H):
            reg_k = self._regime_idx[h]
            arrived = yield_draws[:, h] * pipeline
            avail_pre = np.minimum(self.capacity, stock + arrived)
            exp_q = np.minimum(np.maximum(s_exp[h] - avail_pre, 0.0), exp_cap)
            avail = np.minimum(self.capacity, avail_pre + exp_q)

            multiplier = np.maximum(0.05, 1.0 + macro_shocks[:, reg_k] + idio_noise[:, h])
            demand = self.demand_forecast[h] * multiplier
            short = np.maximum(demand - avail, 0.0)
            stock_after = np.maximum(avail - demand, 0.0)
            reg_q = np.maximum(np.minimum(s_reg[h] - stock_after, self.capacity - stock_after), 0.0)

            tot_reg += reg_q
            tot_exp += exp_q
            tot_hold += stock_after + 0.5 * np.maximum(stock_after - 0.6 * self.capacity, 0.0)
            tot_short += short
            if reg_k == 2:
                tot_peak_short += short
            tot_setups += (reg_q > 0.02 * self.capacity).astype(float) + 0.5 * (exp_q > 1e-6).astype(float)
            tot_demand += demand

            stock = stock_after
            pipeline = reg_q

        norm_dem = float(np.sum(self.demand_forecast))
        obj = (
            law.reg_order_cost * (tot_reg / norm_dem)
            + law.exp_order_cost * (tot_exp / norm_dem)
            + law.holding_cost * (tot_hold / (self.H * self.capacity)) * 4.0
            + law.shortage_penalty * (tot_short / np.maximum(tot_demand, 1e-6))
            + law.setup_cost * (tot_setups / self.H)
        )
        shortage_rate = (tot_short + 0.5 * tot_peak_short) / np.maximum(tot_demand, 1e-6)
        con_centered = shortage_rate - law.sla_max_shortage_rate
        return np.column_stack([obj, con_centered])
