#!/usr/bin/env python3
"""Run a small, frozen Elexon development diagnostic with the existing engine."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import requests

from core.profile_atlas import regular_profile_nodes
from data.opsd import OPSDMarketSeries
from performance.inspect_independent_energy_data import indexed, instant, retrieve, settlement_dates, write_csv
from problems.energy_forecast_policy import OPSDForecastIndexedStorageProblem
from problems.energy_reliability import StoragePhysics
from problems.randomized_profiles import generate_structural_profile_library

PROTOCOL = ROOT / "performance/manifests/independent_energy_development_preflight_20260930.json"
OUTPUT = ROOT / "paper_artifacts/independent_data_feasibility_20260930/development"


def load_period(session, interface, period, prefix_hours, price_request_days, out, label):
    start, stop = map(instant, period)
    first = start - timedelta(hours=prefix_hours)
    times = {"from": first.isoformat(), "to": stop.isoformat()}
    queries = {
        "forecast": (interface["forecast_endpoint"], times),
        "actual": (interface["actual_endpoint"], settlement_dates(first, stop)),
        "price": (interface["price_endpoint"],
                  {**times, "dataProviders": interface["price_provider"], "format": "json"}),
    }
    records, request_log = {}, {}
    for kind, (endpoint, params) in queries.items():
        if kind == "price":
            # The real MID interface rejects ranges longer than seven days.
            # Trim inclusive endpoints before joining the two small responses.
            rows, parts, left = [], [], first
            while left < stop:
                right = min(stop, left + timedelta(days=price_request_days))
                piece, part = retrieve(session, interface["api_base"], endpoint,
                                       {**params, "from": left.isoformat(), "to": right.isoformat()},
                                       out / f"{label}_price_{len(parts)}.json")
                rows.extend(indexed(piece, left, right).values())
                parts.append(part)
                left = right
            meta = {"parts": parts, "response_bytes": sum(p["response_bytes"] for p in parts)}
        else:
            rows, meta = retrieve(session, interface["api_base"], endpoint, params,
                                  out / f"{label}_{kind}.json")
        records[kind] = indexed(rows, first, stop)
        request_log[kind] = {**meta, "returned_rows": len(rows), "retained_rows": len(records[kind])}
    count = int((stop - first).total_seconds() / 1800)
    expected = {first + timedelta(minutes=30 * i) for i in range(count)}
    for kind, rows in records.items():
        if set(rows) != expected:
            raise ValueError(f"{label}/{kind}: incomplete half-hour coverage")
    half_hours = []
    for stamp in sorted(expected):
        forecast, actual, price = (records[k][stamp] for k in ("forecast", "actual", "price"))
        published = instant(forecast["publishTime"])
        if published >= stamp or forecast["boundary"] != "N":
            raise ValueError(f"{label}: late forecast or inconsistent demand boundary")
        if price["dataProvider"] != interface["price_provider"]:
            raise ValueError(f"{label}: unexpected price provider")
        values = [float(forecast[interface["forecast_field"]]),
                  float(actual[interface["actual_field"]]), float(price["price"])]
        if not np.all(np.isfinite(values)) or min(values[:2]) <= 0:
            raise ValueError(f"{label}: nonfinite values or nonpositive demand")
        half_hours.append({"start_time_utc": stamp.isoformat(),
                           "forecast_publish_time_utc": published.isoformat(),
                           "forecast_MW": values[0], "actual_MW": values[1], "price_GBP_per_MWh": values[2]})
    hours = []
    for i in range(0, count, 2):
        pair = half_hours[i:i + 2]
        if max(instant(r["forecast_publish_time_utc"]) for r in pair) >= instant(pair[0]["start_time_utc"]):
            raise ValueError(f"{label}: complete hourly forecast was not available before the hour")
        hours.append({"start_time_utc": pair[0]["start_time_utc"],
                      **{k: sum(r[k] for r in pair) / 2
                         for k in ("forecast_MW", "actual_MW", "price_GBP_per_MWh")}})
    write_csv(out / f"{label}_half_hours.csv", half_hours)
    write_csv(out / f"{label}_hours.csv", hours)
    print(f"{label}: {count} matched half-hours; {len(hours)} complete hours including forecast prefix", flush=True)
    return hours, request_log


class _DevelopmentStorage(OPSDForecastIndexedStorageProblem):
    """Data-only preview adapter; inherits V5 normalization and physical engine.

    One extra hour ensures each evaluated hour's ramp uses consecutive
    forecasts, including the first hour of the development sample.
    """

    def __init__(self, rows, protocol):
        self.d, self.L = protocol["policy_dimension"], 100
        self.horizon = protocol["simulation_horizon_hours"]
        self.nodes = regular_profile_nodes(self.d)
        self.physics = StoragePhysics()
        self.initial_soc_fraction = protocol["initial_soc_fraction"]
        self.outcome_access = True
        self._periods = {"search": tuple(v[:13] for v in protocol["calibration_period"])}
        self.series = OPSDMarketSeries(
            market="GB_ND", timestamp_hour=np.array([int(instant(r["start_time_utc"]).timestamp()) // 3600 for r in rows]),
            load_actual=np.array([r["actual_MW"] for r in rows]),
            load_forecast=np.array([r["forecast_MW"] for r in rows]),
            price=np.array([r["price_GBP_per_MWh"] for r in rows]), solar=None, wind=None,
            metadata={"dataset": "Elexon Insights NDF/INDO/APXMIDP", "role": "development preflight"})
        self._prepare_observable_state()
        self.starts = self.series.valid_window_starts(
            self.horizon, *(v[:13] for v in protocol["development_period"]))
        if len(self.starts) < 32:
            raise ValueError("development sample has fewer than 32 complete seven-day windows")


def summarize_storage(protocol, calibration, development, request_logs, out):
    problem = _DevelopmentStorage(calibration + development, protocol)
    library = generate_structural_profile_library(
        protocol["library"]["profiles"], dimension=protocol["library"]["nodes"],
        seed=protocol["library"]["seed"], maximum_frequency=protocol["library"]["maximum_frequency"])
    policies = [(p.profile_id, problem.continuous_to_int(np.interp(problem.nodes, p.nodes, p.values))) for p in library]
    policies += [(f"constant_{v:.2f}", problem.continuous_to_int(np.full(problem.d, v)))
                 for v in protocol["reference_profiles"]]
    profiles = []
    for profile_id, point in policies:
        values, diagnostics = problem._evaluate_start_batch(point, problem.starts, return_diagnostics=True)
        exchange = float(np.max(diagnostics["maximum_hourly_energy_exchange"]))
        if exchange > problem.physics.power_capacity + 1e-12:
            raise ValueError("preview exceeded the declared shared gross hourly power budget")
        fraction = float(np.mean(values[:, 1] <= 0.0))
        profiles.append({"profile_id": profile_id, "window_success_fraction": fraction,
                         "empirically_feasible": fraction >= 1 - protocol["chance_failure_probability"],
                         "median_service_cost": float(np.median(values[:, 0])),
                         "max_unserved_fraction": float(np.max(diagnostics["unserved_fraction"])),
                         "max_hourly_exchange": exchange})
    feasible = sum(r["empirically_feasible"] for r in profiles)
    summary = {"protocol_id": protocol["protocol_id"], "status": "development_preflight_complete",
               "physics": asdict(problem.physics), "normalization_load_scale_MW": problem._load_scale,
               "policy_count": len(profiles), "feasible_policy_count": feasible,
               "development_window_starts": len(problem.starts),
               "diagnostic_window_evaluations": len(profiles) * len(problem.starts),
               "optimizer_calls": 0, "terminal_verifier_calls": 0, "confirmation_year_accessed": False,
               "certification_comparison_gate": "proceed" if 0 < feasible < len(profiles) else "stop",
               "request_logs": request_logs, "profiles": profiles,
               "interpretation": "finite overlapping development windows; no independent confirmation or method-effect estimate"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("policy_count", "feasible_policy_count", "development_window_starts", "diagnostic_window_evaluations", "certification_comparison_gate")}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    interface = json.loads((args.protocol.parent / protocol["interface_protocol"]).read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    calibration, calibration_log = load_period(session, interface, protocol["calibration_period"],
                                                protocol["forecast_prefix_hours"], protocol["market_index_request_days"],
                                                args.out, "calibration")
    development, development_log = load_period(session, interface, protocol["development_period"],
                                                protocol["forecast_prefix_hours"], protocol["market_index_request_days"],
                                                args.out, "development")
    summarize_storage(protocol, calibration, development,
                      {"calibration": calibration_log, "development": development_log}, args.out)


if __name__ == "__main__":
    main()
