"""Family 2: Nonstationary Multi-Server Stochastic Queueing & Surge Routing DES.

Genuine slot-by-slot discrete-time / discrete-event queue state transitions over
horizon `H` (`d = 2 * H`):
- State at slot `h`: backlog queue length `Q_h >= 0`
- Policy `x in [0, 1]^{2H}`:
    * `x[2*h]`   -> primary server staffing capacity `c_h` in `[c_min, c_max]`
    * `x[2*h+1]` -> surge overflow diversion threshold `theta_div_h`: backlog above
                    `theta_div_h` is diverted to costly surge/cloud capacity
- Stochastic physics:
    * Overdispersed Gamma-Poisson (Negative Binomial) arrivals with correlated diurnal
      burst multipliers across Low / Normal / Rush-Hour regimes;
    * Random server efficiency shocks (correlated fatigue/slowdown under heavy utilization);
    * Nonlinear heavy-traffic heteroscedastic variance explosion when `rho_h -> 1`.
- Public nominal exposure `nominal_exposure(x)` integrates a deterministic fluid queue
  strictly under the PUBLIC arrival forecast `arrival_forecast`, with ZERO access to
  private task burstiness, slowdown, or cost parameters (`QueueEnvironmentLaw`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from v2_or.contracts import BaseDynamicORProblem, NominalTrajectoryExposure


@dataclass(frozen=True)
class QueueEnvironmentLaw:
    """Private task-specific stochastic arrival burstiness, service slowdown, and SLA law."""

    regime_burst_mean: np.ndarray     # shape (3,): log-mean arrival shift per regime
    regime_burst_cov: np.ndarray      # shape (3, 3): shared surge shock covariance
    arrival_overdispersion: float     # Gamma-Poisson dispersion k (smaller = burstier)
    service_efficiency_mean: float    # mean server rate multiplier
    service_slowdown_std: float       # random server slowdown std under congestion
    abandonment_rate: float           # per-slot queue abandonment fraction
    staffing_cost: float
    surge_diversion_cost: float
    waiting_cost: float
    sla_queue_threshold_frac: float   # backlog / c_max threshold defining congested slot
    sla_max_breach_fraction: float    # chance-constraint budget tau


def sample_queueing_tasks(
    n_tasks: int,
    *,
    seed: int,
    horizons: tuple[int, ...] = (48, 96),
    shift_scale: float = 1.0,
) -> list["DynamicQueueingStaffingDES"]:
    """Sample a family of nonstationary queueing/staffing tasks across horizons."""
    rng = np.random.default_rng(seed)
    tasks: list[DynamicQueueingStaffingDES] = []
    for idx in range(int(n_tasks)):
        H = int(horizons[idx % len(horizons)])
        u = np.linspace(0.0, 1.0, H, endpoint=False)
        morning_peak = float(rng.uniform(0.20, 0.35))
        evening_peak = float(rng.uniform(0.62, 0.80))
        base_rate = float(rng.uniform(18.0, 28.0))
        profile = (
            0.55
            + 0.70 * np.exp(-0.5 * ((u - morning_peak) / 0.08) ** 2)
            + 0.85 * np.exp(-0.5 * ((u - evening_peak) / 0.10) ** 2)
            + 0.10 * np.sin(2.0 * np.pi * u)
        )
        arrival_forecast = base_rate * profile

        burst_mean = shift_scale * rng.uniform(-0.04, 0.10, size=3)
        stds = rng.uniform(0.05, 0.15, size=3) * np.sqrt(max(shift_scale, 0.25))
        corr = float(rng.uniform(0.15, 0.60))
        cov = np.outer(stds, stds) * corr
        np.fill_diagonal(cov, stds ** 2)

        law = QueueEnvironmentLaw(
            regime_burst_mean=burst_mean,
            regime_burst_cov=cov,
            arrival_overdispersion=float(rng.uniform(6.0, 16.0)),
            service_efficiency_mean=float(rng.uniform(0.92, 1.03)),
            service_slowdown_std=float(rng.uniform(0.05, 0.12)),
            abandonment_rate=float(rng.uniform(0.03, 0.08)),
            staffing_cost=float(rng.uniform(1.35, 1.95)),
            surge_diversion_cost=float(rng.uniform(2.60, 3.80)),
            waiting_cost=float(rng.uniform(1.60, 2.60)),
            sla_queue_threshold_frac=float(rng.uniform(0.16, 0.22)),
            sla_max_breach_fraction=float(rng.uniform(0.065, 0.105)),
        )
        tasks.append(
            DynamicQueueingStaffingDES(
                task_id=f"queue_H{H}_{idx:03d}",
                arrival_forecast=arrival_forecast,
                c_min=float(0.45 * np.min(arrival_forecast)),
                c_max=float(1.45 * np.max(arrival_forecast)),
                initial_queue=float(0.20 * arrival_forecast[0]),
                law=law,
            )
        )
    return tasks


class DynamicQueueingStaffingDES(BaseDynamicORProblem):
    """Nonstationary multi-server stochastic queueing & surge-diversion DES."""

    family_name = "DynamicQueueingDES"
    num_control_channels = 2
    is_gaussian_constraint_noise = False

    def __init__(
        self,
        *,
        task_id: str,
        arrival_forecast: np.ndarray,
        c_min: float,
        c_max: float,
        initial_queue: float,
        law: QueueEnvironmentLaw,
        alpha: float = 0.05,
    ) -> None:
        super().__init__()
        self.task_id = str(task_id)
        self.arrival_forecast = np.asarray(arrival_forecast, dtype=float).reshape(-1)
        self.H = int(self.arrival_forecast.size)
        if self.H < 8 or np.any(self.arrival_forecast <= 0.0):
            raise ValueError("arrival_forecast must have >= 8 positive slots")
        self.d = 2 * self.H
        self.c_min = float(c_min)
        self.c_max = float(c_max)
        self.initial_queue = max(float(initial_queue), 0.0)
        self._law = law
        self.alpha = float(alpha)

        q33, q67 = np.quantile(self.arrival_forecast, [1.0 / 3.0, 2.0 / 3.0])
        self._regime_idx = np.where(
            self.arrival_forecast <= q33,
            0,
            np.where(self.arrival_forecast <= q67, 1, 2),
        ).astype(int)

    def domain_heuristic_anchor(self) -> np.ndarray:
        """Conservative peak-sized safety staffing rule (high SLA safety, high payroll cost)."""
        target_cap = 1.22 * self.arrival_forecast + 2.4 * np.sqrt(self.arrival_forecast)
        x_staff = np.clip((target_cap - self.c_min) / max(self.c_max - self.c_min, 1e-6), 0.35, 0.92)
        x_div = np.full(self.H, 0.35, dtype=float)
        out = np.empty(self.d, dtype=float)
        out[0::2] = x_staff
        out[1::2] = x_div
        return out

    def nominal_exposure(self, x_norm: np.ndarray) -> NominalTrajectoryExposure:
        """Deterministic fluid queue rollout under PUBLIC `arrival_forecast` only."""
        x = self.clip_policy(x_norm)
        staff_cap = self.c_min + x[0::2] * (self.c_max - self.c_min)
        div_thresh = (0.28 + 0.72 * x[1::2]) * self.c_max

        q = self.initial_queue
        queues = np.empty(self.H, dtype=float)
        diverted = np.empty(self.H, dtype=float)
        util = np.empty(self.H, dtype=float)
        kingman_proxy = np.empty(self.H, dtype=float)

        for h in range(self.H):
            lam = float(self.arrival_forecast[h])
            pre_q = 0.95 * q + lam
            div = max(pre_q - div_thresh[h], 0.0)
            post_div = pre_q - div
            q_next = max(post_div - staff_cap[h], 0.0)

            rho = lam / max(staff_cap[h], 1e-6)
            # Smooth heavy-traffic amplification proxy near rho -> 1, capped by surge threshold
            congestion_amp = min(1.0 / max(1.12 - min(rho, 1.08), 0.04), 18.0) * (
                div_thresh[h] / self.c_max
            )

            queues[h] = q_next
            diverted[h] = div
            util[h] = rho
            kingman_proxy[h] = congestion_amp
            q = q_next

        tot_arr = float(np.sum(self.arrival_forecast))
        n_shared = np.array(
            [
                float(np.mean(kingman_proxy[self._regime_idx == k]))
                if np.any(self._regime_idx == k)
                else 0.0
                for k in range(3)
            ],
            dtype=float,
        ) / 10.0

        a_local = np.array(
            [
                float(np.sqrt(np.mean((kingman_proxy / 10.0) ** 2))),
                float(np.sqrt(np.mean((queues / self.c_max) ** 2))),
                float(np.sqrt(np.mean((diverted / self.c_max) ** 2))),
                float(np.sqrt(np.mean(np.diff(staff_cap / self.c_max, prepend=staff_cap[0] / self.c_max) ** 2))),
            ],
            dtype=float,
        )
        psi = np.array(
            [
                float(np.mean(staff_cap) / self.c_max),
                float(np.sum(diverted) / tot_arr),
                float(np.mean(queues) / self.c_max),
                float(np.max(queues) / self.c_max),
                float(np.mean(util)),
                float(np.quantile(util, 0.90)),
                float(np.mean(div_thresh) / self.c_max),
                a_local[0],
                a_local[1],
                n_shared[0],
                n_shared[1],
                n_shared[2],
            ],
            dtype=float,
        )
        names = (
            "nom_staff_frac",
            "nom_divert_ratio",
            "nom_mean_queue",
            "nom_peak_queue",
            "nom_mean_util",
            "nom_p90_util",
            "nom_div_thresh",
            "A_kingman_rms",
            "A_queue_rms",
            "N_low_cong",
            "N_normal_cong",
            "N_rush_cong",
        )
        return NominalTrajectoryExposure(psi=psi, A_local=a_local, N_shared=n_shared, feature_names=names)

    def _simulate_raw(
        self,
        x_norm: np.ndarray,
        reps: int,
        rng: np.random.Generator,
        split: str = "search",
    ) -> np.ndarray:
        del split
        law = self._law
        staff_cap = self.c_min + x_norm[0::2] * (self.c_max - self.c_min)
        div_thresh = (0.28 + 0.72 * x_norm[1::2]) * self.c_max
        sla_q_cap = law.sla_queue_threshold_frac * self.c_max

        regime_shocks = np.exp(
            rng.multivariate_normal(law.regime_burst_mean, law.regime_burst_cov, size=reps)
        )
        disp_k = law.arrival_overdispersion
        gamma_factors = rng.gamma(shape=disp_k, scale=1.0 / disp_k, size=(reps, self.H))
        slowdown_shocks = rng.normal(0.0, law.service_slowdown_std, size=(reps, self.H))

        q = np.full(reps, self.initial_queue, dtype=float)
        tot_staff = float(np.sum(staff_cap))
        ramp_penalty = float(np.sum(np.abs(np.diff(staff_cap))))
        tot_divert = np.zeros(reps, dtype=float)
        tot_wait = np.zeros(reps, dtype=float)
        tot_breach_slots = np.zeros(reps, dtype=float)
        tot_arrivals = np.zeros(reps, dtype=float)

        for h in range(self.H):
            reg_k = self._regime_idx[h]
            mean_arr = self.arrival_forecast[h] * regime_shocks[:, reg_k] * gamma_factors[:, h]
            arrivals = rng.poisson(np.maximum(mean_arr, 0.05)).astype(float)

            retained_q = np.maximum(0.0, (1.0 - law.abandonment_rate) * q)
            abandoned = q - retained_q
            pre_div = retained_q + arrivals
            div = np.maximum(pre_div - div_thresh[h], 0.0)
            post_div = pre_div - div

            # Congestion-coupled service efficiency drop under heavy queue pressure
            pressure = np.clip(post_div / max(self.c_max, 1e-6), 0.0, 2.0)
            eff = np.clip(
                law.service_efficiency_mean - 0.08 * pressure + slowdown_shocks[:, h] * (0.5 + pressure),
                0.45,
                1.25,
            )
            served = np.minimum(post_div, staff_cap[h] * eff)
            q_next = np.maximum(post_div - served, 0.0)

            tot_divert += div
            tot_wait += q_next + 1.5 * abandoned
            tot_breach_slots += (q_next > sla_q_cap).astype(float)
            tot_arrivals += arrivals
            q = q_next

        norm_arr = float(np.sum(self.arrival_forecast))
        obj = (
            law.staffing_cost * (tot_staff / norm_arr)
            + 0.15 * (ramp_penalty / norm_arr)
            + law.surge_diversion_cost * (tot_divert / np.maximum(tot_arrivals, 1.0))
            + law.waiting_cost * (tot_wait / (self.H * self.c_max)) * 3.0
        )
        breach_frac = tot_breach_slots / float(self.H)
        con_centered = breach_frac - law.sla_max_breach_fraction
        return np.column_stack([obj, con_centered])
