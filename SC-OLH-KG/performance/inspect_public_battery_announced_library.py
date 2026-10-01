#!/usr/bin/env python3
"""Decode the frozen library and account for saved pilots; no controller run."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import requests

from performance.inspect_independent_energy_data import indexed, instant
from performance.inspect_public_battery_controller import calibration, policy_pool, read_csv
from problems.public_battery_dispatch import positive_integral
from problems.public_battery_pending_availability import pending_pulse_power

PROTOCOL = ROOT / "performance/manifests/public_battery_announced_library_objective_preflight_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_announced_library_objective_preflight_v1_20261001"


def read(path):
    return json.loads(Path(path).read_text())


def write(out, name, value):
    (out / name).write_text(json.dumps(value, indent=2) + "\n")


def decode_library(protocol):
    library_protocol = read(ROOT / protocol["library_protocol"])
    basis = read(ROOT / library_protocol["basis_controller_protocol"])
    ids, nodes, points = policy_pool(basis)
    components = {name: point for (name, _, library), point in zip(ids, points) if library}
    roster = read_csv(ROOT / protocol["library"]["roster_csv"])
    expected = [(i, j) for i in range(10) for j in range(10)]
    expected += [(i, 10 + (i - 10 + 27) % 54) for i in range(10, 64)]
    actual = [(int(r["unit_1_component"].split("_")[-1]),
               int(r["unit_2_component"].split("_")[-1])) for r in roster]
    if len(components) != 64 or actual != expected:
        raise ValueError("original joint-library mapping changed")
    channels = np.array([[components[r["unit_1_component"]],
                          components[r["unit_2_component"]]] for r in roster])
    for name, fractions in (("constant_pair_0003_0007", [.35, .75]),
                            ("constant_pair_0004_0007", [.45, .75])):
        position = next(i for i, r in enumerate(roster) if r["profile_id"] == name)
        if not np.array_equal(channels[position], np.repeat(np.array(fractions)[:, None], 1000, axis=1)):
            raise ValueError(f"decoded constant does not match saved screen: {name}")
    metadata = {"library_recipe": basis["library"], "policy_integer_levels": basis["policy_integer_levels"],
                "joint_candidates": len(roster), "constant_pairs": 100, "functional_pairs": 54,
                "coordinates": 2000, "nodes_per_ordered_channel": len(nodes),
                "constant_continuity_pass": True, "ordered_roster_pass": True,
                "extra_shared_references": basis["reference_profiles"], "references_in_library": False,
                "components": [{"id": name, "family": family, "minimum": float(points[i].min()),
                                "maximum": float(points[i].max())}
                               for i, (name, family, library) in enumerate(ids) if library]}
    return roster, nodes, channels, metadata


def raw_rows(path):
    payload = read(path)["payload"]
    return payload if isinstance(payload, list) else payload["data"]


def boundary_records(protocol, data, out):
    interface = read(ROOT / "performance/manifests/independent_energy_data_preflight_20260930.json")
    log_path = out / "boundary_request_log.json"
    logs = read(log_path) if log_path.exists() else []
    session = requests.Session()
    records, coverage = {}, []
    for case in data["development"]:
        day, stop = case["start"][:10], instant(case["stop"])
        stamps = [stop + timedelta(minutes=30 * i) for i in range(4)]
        folder = (ROOT / case["csv"]).parent
        raw = {"forecast": raw_rows(folder / "development_forecast.json"),
               "price": raw_rows(folder / "development_price_0.json") + raw_rows(folder / "development_price_1.json")}
        selected, cached_counts, reused_counts = {}, {}, {}
        for kind in ("forecast", "price"):
            selected[kind] = indexed(raw[kind], stop, stop + timedelta(hours=2))
            cached_counts[kind] = len(selected[kind])
            missing = [t for t in stamps if t not in selected[kind]]
            reused_counts[kind] = 0
            if not missing:
                continue
            path = out / f"{day}_boundary_{kind}.json"
            cache_path = path if path.exists() else OUTPUT / path.name
            if cache_path.exists():
                rows = raw_rows(cache_path)
                reused_counts[kind] = len(rows)
            else:
                if len(logs) >= protocol["input_boundary"]["maximum_new_public_data_requests"]:
                    raise ValueError("frozen boundary-request budget exhausted")
                # These historical endpoints include the requested final startTime.
                params = {"from": min(missing).isoformat(), "to": max(missing).isoformat()}
                if kind == "price":
                    params.update(dataProviders=interface["price_provider"], format="json")
                endpoint = interface[f"{kind}_endpoint"]
                entry = {"sample": day, "kind": kind, "endpoint": endpoint, "params": params,
                         "requested_missing_slots": len(missing), "status": "started"}
                logs.append(entry)
                write(out, "boundary_request_log.json", logs)
                response = session.get(interface["api_base"] + endpoint, params=params, timeout=30)
                entry.update(url=response.url, status_code=response.status_code,
                             response_bytes=len(response.content), retrieved_at=datetime.now(timezone.utc).isoformat())
                write(out, "boundary_request_log.json", logs)
                response.raise_for_status()
                payload = response.json()
                returned = payload if isinstance(payload, list) else payload["data"]
                fetched = indexed(returned, min(missing), max(missing) + timedelta(minutes=30))
                rows = [fetched[t] for t in missing if t in fetched]
                entry.update(status="complete", returned_rows=len(returned), retained_rows=len(rows))
                write(out, path.name, {"url": response.url, "retrieved_at": entry["retrieved_at"], "payload": rows})
                write(out, "boundary_request_log.json", logs)
            selected[kind].update(indexed(rows, stop, stop + timedelta(hours=2)))
            if set(selected[kind]) != set(stamps):
                raise ValueError(f"incomplete historical {kind} boundary: {day}")
        merged = []
        for stamp in stamps:
            forecast, price = selected["forecast"][stamp], selected["price"][stamp]
            if forecast["boundary"] != "N" or price["dataProvider"] != interface["price_provider"]:
                raise ValueError("boundary demand/provider differs from original data")
            merged.append({"start_time_utc": stamp.isoformat(),
                           "forecast_publish_time_utc": instant(forecast["publishTime"]).isoformat(),
                           "forecast_MW": float(forecast[interface["forecast_field"]]),
                           "price_GBP_per_MWh": float(price["price"])})
        records[day] = merged
        coverage.append({"sample": day, "required_boundary_slots": 4, "original_raw_cached_slots": cached_counts,
                         "reused_preflight_boundary_slots": reused_counts,
                         "boundary_complete": True, "start": stamps[0].isoformat(), "last": stamps[-1].isoformat()})
    write(out, "boundary_half_hours.json", records)
    return records, coverage, logs


def forecast_stress(rows, delivery, decision, fit):
    previous = rows[delivery - timedelta(minutes=30)]
    current = rows[delivery]
    published = max(instant(r["forecast_publish_time_utc"]) for r in (previous, current))
    if published > decision:
        raise ValueError("forecast publication exceeds nomination decision")
    level = (float(current["forecast_MW"]) - fit["forecast_q05_MW"]) / (fit["forecast_q95_MW"] - fit["forecast_q05_MW"])
    ramp = abs(float(current["forecast_MW"]) - float(previous["forecast_MW"])) / fit["absolute_ramp_q95_MW"]
    return float(max(np.clip(level, 0, 1), np.clip(ramp, 0, 1))), (decision - published).total_seconds() / 60


def target_fractions(nodes, channels, stress):
    return np.array([np.interp(stress, nodes, values) for values in channels])


def inspect_visibility(data, boundary, nodes, channels):
    fit, reports = calibration(data), []
    for case in data["development"]:
        day, start = case["start"][:10], instant(case["start"])
        rows = {instant(r["start_time_utc"]): r for r in read_csv(ROOT / case["csv"]) + boundary[day]}
        leads, stresses, delivery_stamps = [], [], set()
        for window in range(0, 4321, 60):
            execution_start = start + timedelta(minutes=window)
            for minute in range(0, 10111, 30):
                decision = execution_start + timedelta(minutes=minute)
                delivery = decision + timedelta(minutes=60)
                stress, lead = forecast_stress(rows, delivery, decision, fit)
                stresses.append(stress)
                leads.append(lead)
                delivery_stamps.add(delivery)
            for offset in range(0, 10200, 30):
                price = float(rows[execution_start + timedelta(minutes=offset)]["price_GBP_per_MWh"])
                if not np.isfinite(price):
                    raise ValueError("execution settlement price is nonfinite")
        unique = sorted(set(stresses))
        targets = np.array([target_fractions(nodes, pair, unique) for pair in channels])
        if not np.all(np.isfinite(targets)) or np.min(targets) < 0 or np.max(targets) > 1:
            raise ValueError("decoded forecast target outside [0,1]")
        reports.append({"sample": day, "windows": 73, "decisions_per_window": 338,
                        "decision_visibility_checks": len(leads), "minimum_publication_lead_minutes": min(leads),
                        "unique_delivery_slots": len(delivery_stamps), "stress_range": [min(stresses), max(stresses)],
                        "target_range": [float(targets.min()), float(targets.max())],
                        "last_execution_settlement": (start + timedelta(minutes=14490)).isoformat(),
                        "last_eligible_PN_delivery": max(delivery_stamps).isoformat(),
                        "unused_post_horizon_targets_constructed": 0, "execution_prices_complete": True})
    return {"fit": fit, "fit_year": 2022, "development_refit": False,
            "ordered_channels": True, "samples": reports, "all_visibility_checks_pass": True}


def settlement_cash(first, last, hour_start, rows):
    imported = positive_integral(-first, -last, 1 / 60).reshape(2, 30, 2).sum(axis=1)
    exported = positive_integral(first, last, 1 / 60).reshape(2, 30, 2).sum(axis=1)
    prices = np.array([float(rows[hour_start + timedelta(minutes=30 * i)]["price_GBP_per_MWh"]) for i in range(2)])
    return float(np.sum(prices[:, None] * (imported - exported))), imported, exported


def terminal_adjustment(initial, final, final_price, efficiency):
    value = efficiency * final_price
    return value, value * float(np.sum(np.asarray(initial) - np.asarray(final)))


def pilot_economics(protocol, data):
    folder = ROOT / protocol["pilot_outputs"]
    runs, hours = read(folder / "runs.json"), read(folder / "hourly_dispatch.json")
    rows = {instant(r["start_time_utc"]): r for r in read_csv(ROOT / data["development"][0]["csv"])}
    asset, results, signatures = data["asset"], [], {}
    for run in runs:
        key = (run["profile_id"], run["dispatch_rule"], run["period"], run["window_start_minute"])
        saved = sorted([r for r in hours if tuple(r[k] for k in
                       ("profile_id", "dispatch_rule", "period", "window_start_minute")) == key], key=lambda r: r["minute"])
        if not run["physical_complete"] or run["completed_minutes"] != 10200:
            raise ValueError("preflight pilot is not a complete 170-hour execution")
        if [r["minute"] for r in saved] != list(range(0, 10200, 60)):
            raise ValueError("saved pilot hour grid incomplete")
        start = instant(run["period"] + "T00:00:00Z") + timedelta(minutes=run["window_start_minute"])
        initial = np.asarray(saved[0]["initial_SOC_MWh"])
        soc, imports, exports, cash, errors = initial.copy(), np.zeros(2), np.zeros(2), 0., []
        for hour in saved:
            errors.append(float(np.max(np.abs(soc - np.asarray(hour["initial_SOC_MWh"])))))
            if hour["delivery_prefix_minutes"] != 60 or not hour["delivery_accounting_complete"]:
                raise ValueError("saved pilot contains a partial delivery hour")
            first, last = pending_pulse_power(hour["locked_PN_MW"], hour["admitted_peak_MW"])
            value, imported, exported = settlement_cash(first, last, start + timedelta(minutes=hour["minute"]), rows)
            cash += value
            imports += imported.sum(axis=0)
            exports += exported.sum(axis=0)
            soc += asset["charge_efficiency"] * imported.sum(axis=0) - exported.sum(axis=0) / asset["discharge_efficiency"]
        errors.extend([float(np.max(np.abs(imports - run["gross_import_MWh"]))),
                       float(np.max(np.abs(exports - run["gross_export_MWh"]))),
                       float(np.max(np.abs(soc - run["final_or_last_safe_SOC_MWh"])))])
        if max(errors) > data["failure"]["numerical_energy_tolerance_MWh"]:
            raise ValueError(f"saved pilot gross-flow/inventory discrepancy: {key}: {max(errors)}")
        final_clock = start + timedelta(minutes=run["completed_minutes"] - 30)
        final_price = float(rows[final_clock]["price_GBP_per_MWh"])
        value, adjustment = terminal_adjustment(initial, run["final_or_last_safe_SOC_MWh"], final_price, asset["discharge_efficiency"])
        signature = [(h["minute"], h["locked_PN_MW"], h["admitted_peak_MW"]) for h in saved]
        cost = cash + adjustment
        if run["profile_id"] in signatures:
            prior_signature, prior_cost = signatures[run["profile_id"]]
            if signature != prior_signature or cost != prior_cost:
                raise ValueError("saved identical rule paths do not yield identical economics")
        signatures[run["profile_id"]] = signature, cost
        results.append({"profile_id": run["profile_id"], "dispatch_rule": run["dispatch_rule"],
                        "period": run["period"], "window_start_minute": run["window_start_minute"],
                        "saved_hours": len(saved), "gross_import_MWh": imports.tolist(), "gross_export_MWh": exports.tolist(),
                        "maximum_accounting_error_MWh": max(errors), "metered_cash_GBP": cash,
                        "last_execution_settlement": final_clock.isoformat(), "last_execution_price_GBP_per_MWh": final_price,
                        "terminal_value_GBP_per_MWh": value, "inventory_adjustment_GBP": adjustment, "cost_GBP": cost})
    return {"saved_hour_records": len(hours), "runs": results, "identical_rule_paths_same_cost": True,
            "failed_windows_full_cost": None, "costs_for_other_screen_windows_constructed": 0}


def inspect(protocol, out):
    out.mkdir(parents=True, exist_ok=True)
    write(out, "effective_protocol.json", protocol)
    data = read(ROOT / protocol["cached_data_protocol"])
    roster, nodes, channels, metadata = decode_library(protocol)
    write(out, "library_metadata.json", metadata)
    boundary, coverage, logs = boundary_records(protocol, data, out)
    new_slots = len({(entry["sample"], row["startTime"]) for entry in logs if entry["status"] == "complete"
                     for row in raw_rows(out / f"{entry['sample']}_boundary_{entry['kind']}.json")})
    if new_slots > protocol["input_boundary"]["maximum_retained_new_half_hour_slots"]:
        raise ValueError("frozen retained boundary-slot budget exceeded")
    visibility = inspect_visibility(data, boundary, nodes, channels)
    write(out, "forecast_target_visibility.json", visibility)
    write(out, "input_coverage.json", {"samples": coverage, "all_execution_prices_complete": True,
                                      "required_boundary_slots": 12, "new_unique_half_hour_slots": new_slots,
                                      "new_public_data_requests": len(logs), "original_CSVs_modified": False})
    economics = pilot_economics(protocol, data)
    write(out, "pilot_economics.json", economics)
    summary = {"protocol_id": protocol["protocol_id"], "status": "library_objective_preflight_pass",
               "joint_library_candidates": len(roster), "forecast_decision_visibility_checks": 73 * 338 * 3,
               "public_boundary_requests": len(logs), "public_boundary_response_bytes": sum(r["response_bytes"] for r in logs),
               "new_unique_half_hour_slots": new_slots, "accounted_saved_pilots": len(economics["runs"]),
               "accounted_saved_hours": economics["saved_hour_records"],
               **protocol["accounting_constraints"], "library_matrix_launch": False, "source_comparison_gate": "HOLD",
               "next": "freeze bounded dynamic-target integration and components before any library rollout"}
    write(out, "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(inspect(read(args.protocol), args.output), indent=2))
