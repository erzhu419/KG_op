"""Family 3: Real-Data European Grid Battery Reserve & Dispatch (`OPSDStorage`).

Self-contained implementation on the pinned Open Power System Data archive
(`10.25832/time_series/2020-10-06`, `SC-OLH-KG/data/external/opsd_time_series_extended_v2.npz`),
with ZERO imports from legacy `SC-OLH-KG` representation/algorithm modules.

Contract & physical specification:
- Single-channel hourly reserve / state-of-charge target trajectory `x in [0, 1]^d`
  where `d` is the number of hours (e.g., `d_s in {168, 240}` -> `d_t = 336` hours = 14 days);
- Realized stochastic disturbance is normalized day-ahead load forecast error
  `(load_actual - load_forecast) / median_search_forecast`, with optional Winsorization
  threshold `clamp_limit` (`None` preserves raw tails) and logged `tail_clamped_fraction`;
- Temporal block partition separates non-overlapping 48-hour blocks across `[year-1, year+2)`
  with an explicit `buffer_hours` isolation gap (`default = 24` hours between consecutive blocks)
  into `search`, `verification` (frozen empirical distribution certified by the binomial gate),
  and `audit` (held-out out-of-sample temporal block distribution).
- Zero-leakage `nominal_exposure(x)` (`psi in R^12`) rolls out battery SOC dynamics purely under
  search-split public day-ahead load forecasts & price profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from data.opsd import OPSDMarketSeries, load_opsd_market
from v2_or.contracts import BaseDynamicORProblem, NominalTrajectoryExposure


DEFAULT_OPSD_NPZ = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "external"
    / "opsd_time_series_extended_v2.npz"
)


@dataclass(frozen=True)
class StorageAssetPhysics:
    """Declared grid-scale battery storage operating & reliability constants."""

    energy_capacity: float = 0.42
    power_capacity: float = 0.35
    efficiency: float = 0.92
    reserve_floor: float = 0.10
    reserve_span: float = 0.85
    reserve_cost: float = 0.045
    cycle_cost: float = 0.030
    unserved_cost: float = 8.0
    spill_cost: float = 0.08
    maximum_unserved_fraction: float = 0.11


class OPSDGridStorageBenchmark(BaseDynamicORProblem):
    """Real-data European electricity market storage reserve & reliability problem."""

    family_name = "OPSDGridStorage"
    num_control_channels = 1
    is_gaussian_constraint_noise = False

    def __init__(
        self,
        *,
        market: str,
        d: int = 168,
        year: int = 2018,
        alpha: float = 0.05,
        buffer_hours: int = 24,
        clamp_limit: float | None = 0.20,
        physics: StorageAssetPhysics | None = None,
        data_path: Path | str = DEFAULT_OPSD_NPZ,
    ) -> None:
        super().__init__()
        self.market = str(market)
        self.d = int(d)
        self.year = int(year)
        self.alpha = float(alpha)
        self.buffer_hours = max(int(buffer_hours), 0)
        self.clamp_limit = None if clamp_limit is None else float(clamp_limit)
        self.physics = physics or StorageAssetPhysics()
        self.data_path = Path(data_path)
        self.task_id = f"opsd_{self.market}_d{self.d}"

        series: OPSDMarketSeries = load_opsd_market(
            self.data_path, self.market, include_outcomes=True
        )
        self._block_len = 48
        span_start = f"{self.year - 1:04d}-01-01T00"
        span_stop = f"{self.year + 2:04d}-01-01T00"
        raw_blocks = series.valid_window_starts(self._block_len, span_start, span_stop)
        disjoint_blocks: list[int] = []
        last_hour = -10**12
        min_stride = self._block_len + self.buffer_hours
        for st_idx in raw_blocks:
            h_val = int(series.timestamp_hour[int(st_idx)])
            if h_val >= last_hour + min_stride:
                disjoint_blocks.append(int(st_idx))
                last_hour = h_val
        blk_arr = np.asarray(disjoint_blocks, dtype=np.int64)
        self._starts = {
            "search": blk_arr[0::3],
            "verification": blk_arr[1::3],
            "audit": blk_arr[2::3],
        }
        for split, starts in self._starts.items():
            if len(starts) < 20:
                raise ValueError(f"Market {self.market} split {split} has <20 disjoint blocks ({len(starts)})")

        # Mark strictly search-split hourly indices for public profile calibration (zero leakage)
        s_mask = np.zeros(series.timestamp_hour.size, dtype=bool)
        for st in self._starts["search"]:
            s_mask[int(st) : int(st) + self._block_len] = True
        load_scale = float(np.median(series.load_forecast[s_mask]))
        if not np.isfinite(load_scale) or load_scale <= 0.0:
            raise ValueError(f"Invalid load_scale for {self.market}")

        raw_rel_error = np.asarray(series.load_actual - series.load_forecast, dtype=float) / load_scale
        all_eval_mask = np.zeros(series.timestamp_hour.size, dtype=bool)
        for starts in self._starts.values():
            for st in starts:
                all_eval_mask[int(st) : int(st) + self._block_len] = True
        if self.clamp_limit is not None:
            self.tail_clamped_fraction = float(np.mean(np.abs(raw_rel_error[all_eval_mask]) > self.clamp_limit))
            self._net_error = np.clip(raw_rel_error, -self.clamp_limit, self.clamp_limit)
        else:
            self.tail_clamped_fraction = 0.0
            self._net_error = raw_rel_error.copy()

        # Public day-ahead exogenous profiles fit strictly on search-split forecasts & prices
        price_scale = max(float(np.median(np.abs(series.price[s_mask]))), 1.0)
        self._norm_price = np.clip(np.asarray(series.price, dtype=float) / price_scale, -2.0, 6.0)
        forecast = np.asarray(series.load_forecast, dtype=float) / load_scale
        lo_f, hi_f = np.quantile(forecast[s_mask], [0.05, 0.95])
        f_level = np.clip((forecast - lo_f) / max(float(hi_f - lo_f), 1e-8), 0.0, 1.0)
        f_ramp_raw = np.abs(np.diff(forecast, prepend=forecast[0]))
        ramp_scale = max(float(np.quantile(f_ramp_raw[s_mask], 0.95)), 1e-8)
        f_ramp = np.clip(f_ramp_raw / ramp_scale, 0.0, 3.0)

        # Tile mean 48-hour diurnal search profile cleanly to horizon `d`
        blk_rows = self._starts["search"][:, None] + np.arange(self._block_len, dtype=np.int64)[None, :]
        n_tiles = (self.d + self._block_len - 1) // self._block_len
        self._forecast_level = np.tile(np.mean(f_level[blk_rows], axis=0), n_tiles)[: self.d]
        self._forecast_ramp = np.tile(np.mean(f_ramp[blk_rows], axis=0), n_tiles)[: self.d]
        p_mean48 = np.mean(self._norm_price[blk_rows], axis=0)
        lo_p, hi_p = np.quantile(p_mean48, [0.05, 0.95])
        p_prof48 = np.clip((p_mean48 - lo_p) / max(float(hi_p - lo_p), 1e-8), 0.0, 1.0)
        self._price_profile = np.tile(p_prof48, n_tiles)[: self.d]

        q33, q67 = np.quantile(self._forecast_level, [1.0 / 3.0, 2.0 / 3.0])
        self._regime_idx = np.where(
            self._forecast_level <= q33,
            0,
            np.where(self._forecast_level <= q67, 1, 2),
        ).astype(int)

    def domain_heuristic_anchor(self) -> np.ndarray:
        """Operational rule: track forecast load level + ramp buffer, dip on peak prices."""
        raw = (
            0.48
            + 0.32 * self._forecast_level
            + 0.15 * np.clip(self._forecast_ramp, 0.0, 1.0)
            - 0.10 * self._price_profile
        )
        return np.clip(raw, 0.10, 0.92)

    def nominal_exposure(self, x_norm: np.ndarray) -> NominalTrajectoryExposure:
        """Roll out deterministic battery SOC trajectory on PUBLIC day-ahead profiles."""
        x = self.clip_policy(x_norm)
        phys = self.physics
        target_soc = phys.reserve_floor + phys.reserve_span * x
        cap = float(phys.energy_capacity)
        pwr = float(phys.power_capacity)
        eff = float(phys.efficiency)

        soc = float(np.clip(target_soc[0] * cap, 0.0, cap))
        soc_trace = np.empty(self.d, dtype=float)
        charge_trace = np.empty(self.d, dtype=float)
        release_trace = np.empty(self.d, dtype=float)

        for h in range(self.d):
            desired = float(target_soc[h] * cap)
            delta = desired - soc
            if delta > 0.0:
                ch = min(delta / eff, pwr)
                soc = min(cap, soc + eff * ch)
                charge_trace[h] = ch
                release_trace[h] = 0.0
            elif delta < 0.0:
                rel = min(-delta * eff, pwr, soc * eff)
                soc = max(0.0, soc - rel / eff)
                charge_trace[h] = 0.0
                release_trace[h] = rel
            else:
                charge_trace[h] = 0.0
                release_trace[h] = 0.0
            soc_trace[h] = soc / cap

        deficit = np.maximum(1.0 - soc_trace, 0.0)
        headroom_low = np.maximum(0.28 - soc_trace, 0.0)
        ramp = np.abs(np.diff(soc_trace, prepend=soc_trace[0]))
        block = max(1, min(24, self.d))
        sustained_deficit = np.convolve(deficit, np.ones(block) / float(block), mode="same")

        n_shared = np.array(
            [
                float(np.mean(deficit[self._regime_idx == 0])),
                float(np.mean(deficit[self._regime_idx == 1] * (0.5 + self._forecast_ramp[self._regime_idx == 1]))),
                float(np.mean(deficit[self._regime_idx == 2] * (0.5 + self._forecast_level[self._regime_idx == 2]))),
            ],
            dtype=float,
        )
        a_local = np.array(
            [
                float(np.sqrt(np.mean(deficit ** 2))),
                float(np.sqrt(np.mean(headroom_low ** 2))),
                float(np.sqrt(np.mean(ramp ** 2))),
                float(np.sqrt(np.mean(sustained_deficit ** 2))),
            ],
            dtype=float,
        )
        psi = np.array(
            [
                float(np.mean(soc_trace)),
                float(np.mean(charge_trace * self._price_profile) / max(pwr, 1e-6)),
                float(np.mean(charge_trace + release_trace) / max(pwr, 1e-6)),
                float(np.quantile(deficit, 0.85)),
                float(np.mean(deficit * self._forecast_level)),
                float(np.mean(deficit * np.clip(self._forecast_ramp, 0.0, 1.5))),
                float(np.min(soc_trace)),
                a_local[0],
                a_local[1],
                n_shared[0],
                n_shared[1],
                n_shared[2],
            ],
            dtype=float,
        )
        names = (
            "nom_mean_soc",
            "nom_priced_charge",
            "nom_throughput",
            "nom_p85_deficit",
            "nom_peak_deficit",
            "nom_ramp_deficit",
            "nom_min_soc",
            "A_deficit_rms",
            "A_low_headroom_rms",
            "N_offpeak_def",
            "N_shoulder_def",
            "N_peak_def",
        )
        return NominalTrajectoryExposure(psi=psi, A_local=a_local, N_shared=n_shared, feature_names=names)

    def _evaluate_hour_index_matrix(self, x_norm: np.ndarray, hour_idx_mat: np.ndarray) -> np.ndarray:
        """Roll out battery physics over `(cnt, d)` matrix of historical hourly indices."""
        phys = self.physics
        target = phys.reserve_floor + phys.reserve_span * x_norm
        cap = float(phys.energy_capacity)
        pwr = float(phys.power_capacity)
        eff = float(phys.efficiency)
        cnt = int(hour_idx_mat.shape[0])

        soc = np.full(cnt, float(np.clip(target[0] * cap, 0.0, cap)), dtype=float)
        grid_charge = np.zeros(cnt, dtype=float)
        throughput = np.zeros(cnt, dtype=float)
        unserved = np.zeros(cnt, dtype=float)
        spill = np.zeros(cnt, dtype=float)
        pos_error = np.zeros(cnt, dtype=float)

        for h in range(self.d):
            h_col = hour_idx_mat[:, h]
            desired = float(target[h] * cap)
            delta = desired - soc
            ch_mask = delta > 0.0
            if np.any(ch_mask):
                ch = np.minimum(delta[ch_mask] / eff, pwr)
                soc[ch_mask] = np.minimum(cap, soc[ch_mask] + eff * ch)
                prices = self._norm_price[h_col[ch_mask]]
                grid_charge[ch_mask] += np.maximum(prices, 0.0) * ch
                throughput[ch_mask] += ch
            rel_mask = delta < 0.0
            if np.any(rel_mask):
                rel = np.minimum.reduce([
                    -delta[rel_mask] * eff,
                    np.full(int(np.sum(rel_mask)), pwr, dtype=float),
                    soc[rel_mask] * eff,
                ])
                soc[rel_mask] = np.maximum(0.0, soc[rel_mask] - rel / eff)
                throughput[rel_mask] += rel

            shock = self._net_error[h_col]
            sh_mask = shock >= 0.0
            if np.any(sh_mask):
                pos = shock[sh_mask]
                pos_error[sh_mask] += pos
                dis = np.minimum.reduce([
                    pos,
                    np.full(int(np.sum(sh_mask)), pwr, dtype=float),
                    soc[sh_mask] * eff,
                ])
                soc[sh_mask] = np.maximum(0.0, soc[sh_mask] - dis / eff)
                throughput[sh_mask] += dis
                unserved[sh_mask] += pos - dis
            sur_mask = ~sh_mask
            if np.any(sur_mask):
                sur = -shock[sur_mask]
                ch_s = np.minimum.reduce([
                    sur,
                    np.full(int(np.sum(sur_mask)), pwr, dtype=float),
                    np.maximum(cap - soc[sur_mask], 0.0) / eff,
                ])
                soc[sur_mask] = np.minimum(cap, soc[sur_mask] + eff * ch_s)
                throughput[sur_mask] += ch_s
                spill[sur_mask] += sur - ch_s

        horizon = float(self.d)
        reserve = float(np.mean(target))
        obj = (
            phys.reserve_cost * reserve
            + grid_charge / horizon
            + phys.cycle_cost * throughput / horizon
            + phys.unserved_cost * unserved / horizon
            + phys.spill_cost * spill / horizon
        )
        denom = np.maximum(pos_error, 1e-4 * horizon)
        unserved_frac = unserved / denom
        con = unserved_frac - phys.maximum_unserved_fraction
        return np.column_stack([obj, con])

    def _sample_split_hour_matrix(self, reps: int, rng: np.random.Generator, split: str) -> np.ndarray:
        blocks_pool = self._starts[split]
        n_blocks = (self.d + self._block_len - 1) // self._block_len
        chosen_starts = blocks_pool[rng.integers(0, len(blocks_pool), size=(int(reps), n_blocks))]
        offsets = np.arange(self._block_len, dtype=np.int64)[None, None, :]
        full_idx = (chosen_starts[:, :, None] + offsets).reshape(int(reps), n_blocks * self._block_len)
        return full_idx[:, : self.d]

    def _simulate_raw(
        self,
        x_norm: np.ndarray,
        reps: int,
        rng: np.random.Generator,
        split: str = "search",
    ) -> np.ndarray:
        idx_mat = self._sample_split_hour_matrix(int(reps), rng, split)
        return self._evaluate_hour_index_matrix(x_norm, idx_mat)

    def audit_ground_truth(
        self,
        x_norm: np.ndarray,
        *,
        n_eval: int = 4000,
        seed: int = 20261116,
        split: str = "audit",
    ) -> dict:
        """High-precision Monte Carlo audit on `split` (`"verification"` IID law or `"audit"` OOD blocks)."""
        from v2_or.contracts import clopper_pearson_interval

        x_clean = self.clip_policy(x_norm)
        rng = np.random.default_rng(int(seed))
        total_eval = max(int(n_eval), 2000)
        batches = []
        rem = total_eval
        while rem > 0:
            b_sz = min(rem, 512)
            idx_mat = self._sample_split_hour_matrix(b_sz, rng, str(split))
            batches.append(self._evaluate_hour_index_matrix(x_clean, idx_mat))
            rem -= b_sz
        samples = np.vstack(batches)
        self.audit_calls += int(total_eval)
        obj = samples[:, 0]
        con = samples[:, 1]
        k_viol = int(np.sum(con > 0.0))
        viol_prob = float(k_viol / float(total_eval))
        ci_lo, ci_hi = clopper_pearson_interval(k_viol, int(total_eval), conf=0.95)
        if ci_hi <= self.alpha:
            safety_class = "confidently_safe"
        elif ci_lo > self.alpha:
            safety_class = "confidently_unsafe"
        else:
            safety_class = "boundary_ambiguous"
        return {
            "mc_objective_mean": float(np.mean(obj)),
            "mc_objective_se": float(np.std(obj, ddof=1) / np.sqrt(total_eval)),
            "mc_constraint_mean": float(np.mean(con)),
            "mc_constraint_std": float(np.std(con, ddof=1)),
            "mc_violation_count": k_viol,
            "mc_violation_prob": viol_prob,
            "mc_viol_ci95_lo": ci_lo,
            "mc_viol_ci95_hi": ci_hi,
            "safety_audit_class": safety_class,
            "true_objective": float(np.mean(obj)),
            "true_objective_se": float(np.std(obj, ddof=1) / np.sqrt(total_eval)),
            "true_constraint_mean": float(np.mean(con)),
            "true_constraint_std": float(np.std(con, ddof=1)),
            "true_violation_prob": viol_prob,
            "true_quantile_1_minus_alpha": float(np.quantile(con, 1.0 - self.alpha)),
            "is_truly_feasible": bool(viol_prob <= self.alpha),
            "n_audit_samples": int(total_eval),
            "audit_split": str(split),
            "buffer_hours": int(self.buffer_hours),
            "clamp_limit": self.clamp_limit,
            "tail_clamped_fraction": float(self.tail_clamped_fraction),
        }


def discover_valid_opsd_markets(
    *,
    d: int = 168,
    year: int = 2018,
    candidates: Sequence[str] = (
        "DK_1",
        "DK_2",
        "GB_GBN",
        "IT_NORD",
        "IT_CNOR",
        "IT_CSUD",
        "IT_SUD",
        "IT_SARD",
        "IT_SICI",
        "NO_1",
        "NO_2",
        "NO_3",
        "NO_4",
        "NO_5",
        "SE_1",
        "SE_2",
        "SE_3",
        "SE_4",
        "IE_sem",
        "AT",
    ),
    data_path: Path | str = DEFAULT_OPSD_NPZ,
) -> list[str]:
    """Return markets with complete search/audit/verification splits for horizon `d`."""
    valid: list[str] = []
    for mkt in candidates:
        try:
            _ = load_opsd_market(data_path, mkt, include_outcomes=False)
            prob = OPSDGridStorageBenchmark(market=mkt, d=d, year=year, data_path=data_path)
            if len(prob._starts["search"]) >= 48:
                valid.append(mkt)
        except Exception:
            continue
    return valid
