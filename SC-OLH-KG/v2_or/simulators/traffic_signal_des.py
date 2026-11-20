"""Family 4: Urban Traffic Signal-Timing Network DES (`Ingolstadt21` Topology + Live SUMO).

Provides two tightly aligned interfaces sharing the exact 12-dim nominal signal-queue
trajectory exposure `nominal_exposure(x)`:
1. `IngolstadtSignalNetworkDES`: Multi-junction store-and-forward + spillback-blocking
   stochastic queue & HBEFA3 stop-idle-cruise CO2 network DES built directly on the
   RESCO `ingolstadt21` signal phase structure (`decision_space.json`, up to 21 TLS /
   `d = 44` main phases, or arterial subnetworks `d_s = 22` for cross-dimension transfer).
2. `IngolstadtLiveSUMOBenchmark`: Direct microscopic `libsumo` wrapper around
   the RESCO `ingolstadt21` scenario for SUMO 1.22 validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from v2_or.contracts import BaseDynamicORProblem, NominalTrajectoryExposure


DEFAULT_INGOLSTADT_DECISION_SPACE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "ingolstadt21"
    / "decision_space.json"
)


def _load_ingolstadt_phase_spec(
    max_tls: int | None = None,
    spec_path: Path | str = DEFAULT_INGOLSTADT_DECISION_SPACE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load `(lo, hi, defaults, tls_idx_per_var, fixed_lost_per_tls)` from RESCO spec."""
    data = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    tls_list = data["tls_list"]
    if max_tls is not None:
        tls_list = tls_list[: int(max_tls)]
    allowed_ids = {entry["id"]: idx for idx, entry in enumerate(tls_list)}

    lo_list, hi_list, def_list, var_tls = [], [], [], []
    for k, (tls_id, _ph_idx) in enumerate(data["var_map"]):
        if tls_id in allowed_ids:
            lo_list.append(float(data["bounds"][k][0]))
            hi_list.append(float(data["bounds"][k][1]))
            def_list.append(float(data["defaults"][k]))
            var_tls.append(int(allowed_ids[tls_id]))

    n_tls = len(tls_list)
    fixed_overhead = np.zeros(n_tls, dtype=float)
    var_tls_arr = np.asarray(var_tls, dtype=int)
    def_arr = np.asarray(def_list, dtype=float)
    for j, entry in enumerate(tls_list):
        opt_sum = float(np.sum(def_arr[var_tls_arr == j]))
        # Auxiliary turn phases + yellow/all-red intergreens preserved from RESCO cycle
        fixed_overhead[j] = max(float(entry["cycle"]) - opt_sum, 10.0)

    return (
        np.asarray(lo_list, dtype=float),
        np.asarray(hi_list, dtype=float),
        def_arr,
        var_tls_arr,
        fixed_overhead,
    )


@dataclass(frozen=True)
class TrafficEnvironmentLaw:
    """Private stochastic OD surge, weather saturation drop, and emission physics law."""

    wave_drift: np.ndarray            # shape (3,): early, peak-core, shoulder OD surge
    wave_cov: np.ndarray              # shape (3, 3): shared network-wide demand shock cov
    platoon_dispersion: float         # Gamma-Poisson arrival burstiness
    weather_sat_mean: float           # saturation flow factor (e.g., <1 in rain/incident)
    weather_sat_std: float            # cycle-level discharge variability
    spillback_sensitivity: float      # upstream capacity throttling under downstream queue
    stop_co2_factor: float            # HBEFA3 acceleration CO2 penalty per stop/cycle
    idle_co2_factor: float            # HBEFA3 idling CO2 rate per queued vehicle
    tau_co2_ratio: float              # chance-constraint CO2 ceiling relative to baseline


def sample_traffic_tasks(
    n_tasks: int,
    *,
    seed: int,
    tls_counts: tuple[int, ...] = (11, 21),
    shift_scale: float = 1.0,
) -> list["IngolstadtSignalNetworkDES"]:
    """Sample distinct urban traffic demand/weather regimes on Ingolstadt subnetworks."""
    rng = np.random.default_rng(seed)
    tasks: list[IngolstadtSignalNetworkDES] = []
    for idx in range(int(n_tasks)):
        n_tls = int(tls_counts[idx % len(tls_counts)])
        lo, hi, defaults, var_tls, fixed_ov = _load_ingolstadt_phase_spec(max_tls=n_tls)
        d_vars = len(defaults)
        H_cycles = 40  # ~1 hour of 90s peak cycles

        # Public asymmetric arterial vs. side-street commuter demand:
        # primary corridor phases (even indices) carry heavy PM commuter volume (v/c ~ 0.78-0.90),
        # while cross-streets (odd indices) carry lighter volume (v/c ~ 0.34-0.48), creating
        # substantial delay & CO2 savings when reallocating green time away from 50/50 fixed defaults!
        sat_rate = 0.52  # veh / sec of effective green (~1872 veh/h)
        nom_cap_per_var = sat_rate * defaults
        vc_target = np.empty(d_vars, dtype=float)
        vc_target[0::2] = rng.uniform(0.70, 0.83, size=(d_vars + 1) // 2)
        vc_target[1::2] = rng.uniform(0.32, 0.45, size=d_vars // 2)
        base_demand_var = nom_cap_per_var * vc_target

        u = np.linspace(0.0, 1.0, H_cycles, endpoint=False)
        peak_u = float(rng.uniform(0.32, 0.68))
        temporal_wave = 0.82 + 0.34 * np.exp(-0.5 * ((u - peak_u) / 0.18) ** 2)
        public_arrival_matrix = np.outer(temporal_wave, base_demand_var)  # (H_cycles, d_vars)

        drift = shift_scale * rng.uniform(-0.03, 0.06, size=3)
        stds = rng.uniform(0.03, 0.07, size=3) * np.sqrt(max(shift_scale, 0.25))
        corr = float(rng.uniform(0.20, 0.60))
        cov = np.outer(stds, stds) * corr
        np.fill_diagonal(cov, stds ** 2)

        law = TrafficEnvironmentLaw(
            wave_drift=drift,
            wave_cov=cov,
            platoon_dispersion=float(rng.uniform(10.0, 22.0)),
            weather_sat_mean=float(rng.uniform(0.94, 1.02)),
            weather_sat_std=float(rng.uniform(0.03, 0.06)),
            spillback_sensitivity=float(rng.uniform(0.22, 0.40)),
            stop_co2_factor=float(rng.uniform(0.26, 0.34)),
            idle_co2_factor=float(rng.uniform(0.44, 0.52)),
            tau_co2_ratio=float(rng.uniform(1.07, 1.14)),
        )
        tasks.append(
            IngolstadtSignalNetworkDES(
                task_id=f"traffic_tls{n_tls}_d{d_vars}_{idx:03d}",
                lo=lo,
                hi=hi,
                defaults=defaults,
                var_tls=var_tls,
                fixed_overhead=fixed_ov,
                public_arrivals=public_arrival_matrix,
                law=law,
            )
        )
    return tasks


class IngolstadtSignalNetworkDES(BaseDynamicORProblem):
    """Stylized multi-intersection signal-timing queue, spillback & CO2 network DES."""

    family_name = "IngolstadtTrafficDES"
    num_control_channels = 2
    is_gaussian_constraint_noise = False

    def __init__(
        self,
        *,
        task_id: str,
        lo: np.ndarray,
        hi: np.ndarray,
        defaults: np.ndarray,
        var_tls: np.ndarray,
        fixed_overhead: np.ndarray,
        public_arrivals: np.ndarray,
        law: TrafficEnvironmentLaw,
        alpha: float = 0.05,
    ) -> None:
        super().__init__()
        self.task_id = str(task_id)
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        self.defaults = np.asarray(defaults, dtype=float)
        self.var_tls = np.asarray(var_tls, dtype=int)
        self.fixed_overhead = np.asarray(fixed_overhead, dtype=float)
        self.n_tls = int(self.fixed_overhead.size)
        self.d = int(self.defaults.size)
        self.public_arrivals = np.asarray(public_arrivals, dtype=float)
        self.H = int(self.public_arrivals.shape[0])
        self._law = law
        self.alpha = float(alpha)
        self._sat_flow = 0.52  # veh / sec of green
        self._storage_cap = 38.0  # veh per approach before spillback

        # Corridor coupling matrix along neighboring Ingolstadt intersections
        routing = np.zeros((self.d, self.d), dtype=float)
        for i in range(self.d):
            j_next = (i + 2) % self.d
            j_turn = (i + 3) % self.d
            routing[i, j_next] = 0.24
            routing[i, j_turn] = 0.08
        self._routing = routing

        tot_h = np.sum(self.public_arrivals, axis=1)
        q33, q67 = np.quantile(tot_h, [1.0 / 3.0, 2.0 / 3.0])
        self._regime_idx = np.where(tot_h <= q33, 0, np.where(tot_h <= q67, 1, 2)).astype(int)

        # Calibrate public baseline normalization constants (T0, E0) from nominal fixed-time plan
        x0 = self.domain_heuristic_anchor()
        nom_t0, nom_e0 = self._deterministic_metrics(x0)
        self._T0 = max(nom_t0, 1e-3)
        self._E0 = max(nom_e0, 1e-3)

    def _greens_and_cycles(self, x_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        greens = self.lo + x_norm * (self.hi - self.lo)
        cycles = self.fixed_overhead.copy()
        np.add.at(cycles, self.var_tls, greens)
        var_cycles = cycles[self.var_tls]
        return greens, var_cycles

    def domain_heuristic_anchor(self) -> np.ndarray:
        """Published RESCO fixed-time signal plan (`defaults`) normalized to `[0, 1]^d`."""
        return np.clip((self.defaults - self.lo) / np.maximum(self.hi - self.lo, 1e-6), 0.0, 1.0)

    def warp_source_policy(self, x_src: np.ndarray) -> np.ndarray:
        """Subnetwork-prefix + arterial/side-street parity-preserving policy adapter."""
        arr = np.asarray(x_src, dtype=float).reshape(-1)
        if arr.size == self.d:
            return np.clip(arr.copy(), 0.0, 1.0)
        anc = self.domain_heuristic_anchor()
        if arr.size > self.d:
            return np.clip(arr[: self.d].copy(), 0.0, 1.0)
        out = anc.copy()
        k = arr.size
        out[:k] = arr
        # Propagate average arterial (even) and cross-street (odd) relative adjustment to downstream TLS
        delta_even = float(np.mean(arr[0:k:2] - anc[0:k:2])) if k >= 1 else 0.0
        delta_odd = float(np.mean(arr[1:k:2] - anc[1:k:2])) if k >= 2 else delta_even
        for j in range(k, self.d):
            out[j] = anc[j] + (delta_even if (j % 2 == 0) else delta_odd)
        return np.clip(out, 0.0, 1.0)

    def _deterministic_metrics(self, x_norm: np.ndarray) -> tuple[float, float]:
        greens, var_cycles = self._greens_and_cycles(x_norm)
        nom_cap = self._sat_flow * greens * (90.0 / var_cycles)
        q = np.zeros(self.d, dtype=float)
        tot_delay = 0.0
        tot_co2 = 0.0
        for h in range(self.H):
            ext_in = self.public_arrivals[h] * 0.68
            avail = q + ext_in
            spill_factor = np.clip(1.0 - 0.35 * np.maximum(q / self._storage_cap - 0.65, 0.0), 0.45, 1.0)
            served = np.minimum(avail, nom_cap * spill_factor)
            q_rem = np.maximum(avail - served, 0.0)
            red_frac = np.clip(1.0 - greens / var_cycles, 0.10, 0.92)
            stops = served * red_frac * (90.0 / np.maximum(var_cycles, 35.0)) + 0.65 * q_rem
            tot_delay += float(np.sum(q_rem * (0.5 * var_cycles / 90.0) + 0.25 * served * red_frac))
            tot_co2 += float(np.sum(0.30 * stops + 0.48 * q_rem * (var_cycles / 90.0) + 0.18 * served))
            q = q_rem + served @ self._routing
        return tot_delay, tot_co2

    def nominal_exposure(self, x_norm: np.ndarray) -> NominalTrajectoryExposure:
        """Public Webster/store-and-forward trajectory features with ZERO private law access."""
        x = self.clip_policy(x_norm)
        greens, var_cycles = self._greens_and_cycles(x)
        green_splits = greens / var_cycles
        nom_cap = self._sat_flow * greens * (90.0 / var_cycles)

        q = np.zeros(self.d, dtype=float)
        cycle_q_mean = np.empty(self.H, dtype=float)
        cycle_spill = np.empty(self.H, dtype=float)
        cycle_stops = np.empty(self.H, dtype=float)
        tls_delay_acc = np.zeros(self.n_tls, dtype=float)

        for h in range(self.H):
            ext_in = self.public_arrivals[h] * 0.68
            avail = q + ext_in
            occ_ratio = q / self._storage_cap
            spill_factor = np.clip(1.0 - 0.35 * np.maximum(occ_ratio - 0.65, 0.0), 0.45, 1.0)
            served = np.minimum(avail, nom_cap * spill_factor)
            q_rem = np.maximum(avail - served, 0.0)
            red_frac = np.clip(1.0 - green_splits, 0.10, 0.92)
            stops = served * red_frac * (90.0 / np.maximum(var_cycles, 35.0)) + 0.65 * q_rem

            cycle_q_mean[h] = float(np.mean(q_rem) / self._storage_cap)
            cycle_spill[h] = float(np.mean(np.maximum(occ_ratio - 0.55, 0.0)))
            cycle_stops[h] = float(np.mean(stops) / max(np.mean(nom_cap), 1e-6))
            np.add.at(tls_delay_acc, self.var_tls, q_rem)
            q = q_rem + served @ self._routing

        mean_vc = np.mean(self.public_arrivals, axis=0) / np.maximum(nom_cap, 1e-6)
        oversat = np.maximum(mean_vc - 0.80, 0.0)
        equity_disp = float(np.std(tls_delay_acc) / (np.mean(tls_delay_acc) + 1e-6))

        n_shared = np.array(
            [
                float(np.mean(cycle_q_mean[self._regime_idx == k] + cycle_spill[self._regime_idx == k]))
                if np.any(self._regime_idx == k)
                else 0.0
                for k in range(3)
            ],
            dtype=float,
        )
        a_local = np.array(
            [
                float(np.sqrt(np.mean(cycle_q_mean ** 2))),
                float(np.sqrt(np.mean(oversat ** 2))),
                float(np.sqrt(np.mean(cycle_stops ** 2))),
                float(np.sqrt(np.mean(((var_cycles - 90.0) / 90.0) ** 2))),
            ],
            dtype=float,
        )
        psi = np.array(
            [
                float(np.mean(green_splits)),
                float(np.mean(var_cycles) / 90.0),
                float(np.std(var_cycles) / 90.0),
                float(np.mean(cycle_q_mean)),
                float(np.max(cycle_q_mean)),
                float(np.mean(cycle_stops)),
                equity_disp,
                a_local[0],
                a_local[1],
                n_shared[0],
                n_shared[1],
                n_shared[2],
            ],
            dtype=float,
        )
        names = (
            "nom_green_split",
            "nom_cycle_ratio",
            "nom_cycle_std",
            "nom_mean_queue",
            "nom_peak_queue",
            "nom_stop_rate",
            "nom_tls_inequity",
            "A_queue_rms",
            "A_oversat_rms",
            "N_early_cong",
            "N_mid_cong",
            "N_peak_cong",
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
        greens, var_cycles = self._greens_and_cycles(x_norm)
        green_splits = greens / var_cycles
        red_frac = np.clip(1.0 - green_splits, 0.10, 0.92)
        base_cap = self._sat_flow * greens * (90.0 / var_cycles)

        wave_mult = np.exp(rng.multivariate_normal(law.wave_drift, law.wave_cov, size=reps))
        k_disp = law.platoon_dispersion

        q = np.zeros((reps, self.d), dtype=float)
        tot_delay = np.zeros(reps, dtype=float)
        tot_co2 = np.zeros(reps, dtype=float)
        tls_delay = np.zeros((reps, self.n_tls), dtype=float)

        for h in range(self.H):
            reg_k = self._regime_idx[h]
            gamma_burst = rng.gamma(shape=k_disp, scale=1.0 / k_disp, size=(reps, self.d))
            lam = (self.public_arrivals[h] * 0.68)[None, :] * wave_mult[:, reg_k : reg_k + 1] * gamma_burst
            ext_arr = rng.poisson(np.maximum(lam, 0.05)).astype(float)

            avail = q + ext_arr
            occ_ratio = q / self._storage_cap
            spill_factor = np.clip(
                1.0 - law.spillback_sensitivity * np.maximum(occ_ratio - 0.60, 0.0),
                0.35,
                1.0,
            )
            sat_shock = np.clip(
                rng.normal(law.weather_sat_mean, law.weather_sat_std, size=(reps, self.d)),
                0.55,
                1.20,
            )
            eff_cap = base_cap[None, :] * spill_factor * sat_shock
            served = np.minimum(avail, eff_cap)
            q_rem = np.maximum(avail - served, 0.0)

            stops = (
                served * red_frac[None, :] * (90.0 / np.maximum(var_cycles[None, :], 35.0))
                + 0.70 * q_rem
            )
            step_delay = q_rem * (0.5 * var_cycles[None, :] / 90.0) + 0.25 * served * red_frac[None, :]
            step_co2 = (
                law.stop_co2_factor * stops
                + law.idle_co2_factor * q_rem * (var_cycles[None, :] / 90.0)
                + 0.18 * served
            )
            tot_delay += np.sum(step_delay, axis=1)
            tot_co2 += np.sum(step_co2, axis=1)
            for j in range(self.n_tls):
                mask = self.var_tls == j
                if np.any(mask):
                    tls_delay[:, j] += np.sum(step_delay[:, mask], axis=1)

            q = q_rem + served @ self._routing

        delay_ratio = tot_delay / self._T0
        # Normalized spatial queue inequity proxy (Atkinson-style dispersion across intersections)
        mean_tls = np.maximum(np.mean(tls_delay, axis=1), 1e-6)
        inequity_ratio = 1.0 + 0.35 * (np.std(tls_delay, axis=1) / mean_tls)
        obj = 0.65 * delay_ratio + 0.35 * inequity_ratio
        co2_ratio = tot_co2 / (self._E0 * (law.idle_co2_factor / 0.48))
        con_centered = co2_ratio - law.tau_co2_ratio
        return np.column_stack([obj, con_centered])


class IngolstadtLiveSUMOBenchmark(BaseDynamicORProblem):
    """Live microscopic `libsumo` RESCO `ingolstadt21` benchmark for cluster validation."""

    family_name = "IngolstadtLiveSUMO"
    num_control_channels = 2
    is_gaussian_constraint_noise = False

    def __init__(
        self,
        *,
        task_id: str = "ingolstadt21_live_sumo",
        weights: tuple[float, float] = (0.5, 0.5),
        tau_emission: float = 1.0,
        alpha: float = 0.05,
        baseline_override: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.task_id = str(task_id)
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / np.sum(self.weights)
        self.tau_emission = float(tau_emission)
        self.alpha = float(alpha)
        self.baseline_override = dict(baseline_override) if baseline_override is not None else None

        lo, hi, defaults, var_tls, fixed_ov = _load_ingolstadt_phase_spec(max_tls=None)
        self.lo = lo
        self.hi = hi
        self.defaults = defaults
        self.d = int(defaults.size)

        # Share the exact oracle-free nominal traffic corridor exposure operator!
        u = np.linspace(0.0, 1.0, 40, endpoint=False)
        wave = 0.85 + 0.32 * np.exp(-0.5 * ((u - 0.50) / 0.20) ** 2)
        pub_arr = np.outer(wave, 0.52 * defaults * 0.74)
        dummy_law = TrafficEnvironmentLaw(
            wave_drift=np.zeros(3),
            wave_cov=np.eye(3) * 0.01,
            platoon_dispersion=12.0,
            weather_sat_mean=1.0,
            weather_sat_std=0.05,
            spillback_sensitivity=0.35,
            stop_co2_factor=0.30,
            idle_co2_factor=0.48,
            tau_co2_ratio=tau_emission,
        )
        self._nominal_helper = IngolstadtSignalNetworkDES(
            task_id=task_id + "_nom",
            lo=lo,
            hi=hi,
            defaults=defaults,
            var_tls=var_tls,
            fixed_overhead=fixed_ov,
            public_arrivals=pub_arr,
            law=dummy_law,
            alpha=alpha,
        )
        self._live_problem: Any = None

    def _get_live_problem(self) -> Any:
        if self._live_problem is None:
            import sys
            sc_dir = Path(__file__).resolve().parents[2]
            if str(sc_dir) not in sys.path:
                sys.path.insert(0, str(sc_dir))
            from data.ingolstadt21.ingolstadt21_problem import Ingolstadt21Problem
            prob = Ingolstadt21Problem(tau=self.tau_emission, alpha=self.alpha)
            if self.baseline_override is not None:
                if "T0" in self.baseline_override:
                    prob.T0 = float(self.baseline_override["T0"])
                if "A0" in self.baseline_override:
                    prob.A0 = float(self.baseline_override["A0"])
                if "E0" in self.baseline_override:
                    prob.E0 = float(self.baseline_override["E0"])
            self._live_problem = prob
        return self._live_problem

    def domain_heuristic_anchor(self) -> np.ndarray:
        return self._nominal_helper.domain_heuristic_anchor()

    def warp_source_policy(self, x_src: np.ndarray) -> np.ndarray:
        return self._nominal_helper.warp_source_policy(x_src)

    def nominal_exposure(self, x_norm: np.ndarray) -> NominalTrajectoryExposure:
        return self._nominal_helper.nominal_exposure(x_norm)

    def _simulate_raw(
        self,
        x_norm: np.ndarray,
        reps: int,
        rng: np.random.Generator,
        split: str = "search",
    ) -> np.ndarray:
        del split
        prob = self._get_live_problem()
        x_sec = self.lo + x_norm * (self.hi - self.lo)
        rows = np.empty((int(reps), 2), dtype=float)
        for r in range(int(reps)):
            sim_seed = int(rng.integers(1, 2**31 - 1))
            y = prob._sim.simulate(
                var_map=prob.var_map,
                x=x_sec,
                route_file=prob._route_files[0],
                T0=prob.T0,
                A0=prob.A0,
                E0=prob.E0,
                seed=sim_seed,
            )
            obj = float(self.weights[0] * y[0] + self.weights[1] * y[1])
            con_centered = float(y[2] - self.tau_emission)
            rows[r, 0] = obj
            rows[r, 1] = con_centered
        return rows
