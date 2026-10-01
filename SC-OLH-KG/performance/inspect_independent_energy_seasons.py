#!/usr/bin/env python3
"""Check three frozen seasons, reusing the completed January calibration."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import requests
from performance.inspect_independent_energy_development import load_period, summarize_storage


def main():
    manifest = ROOT / "performance/manifests/independent_energy_seasonal_preflight_20260930.json"
    protocol = json.loads(manifest.read_text())
    base = json.loads((manifest.parent / protocol["base_protocol"]).read_text())
    interface = json.loads((manifest.parent / base["interface_protocol"]).read_text())
    previous = ROOT / protocol["existing_case"]
    january = json.loads((previous / "summary.json").read_text())
    with (previous / "calibration_hours.csv").open() as handle:
        calibration = [{k: v if k == "start_time_utc" else float(v) for k, v in row.items()}
                       for row in csv.DictReader(handle)]
    summaries = [january]
    out = previous.parent / "seasonal"
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    for period in protocol["additional_development_periods"]:
        case = out / period[0][:10]
        case.mkdir(exist_ok=True)
        effective = {**base, "protocol_id": protocol["protocol_id"], "development_period": period}
        (case / "effective_protocol.json").write_text(json.dumps(effective, indent=2) + "\n")
        development, log = load_period(session, interface, period, base["forecast_prefix_hours"],
                                       base["market_index_request_days"], case, "development")
        summaries.append(summarize_storage(effective, calibration, development,
                         {"calibration": january["request_logs"]["calibration"], "development": log}, case))
    report = {"protocol_id": protocol["protocol_id"], "status": "seasonal_preflight_complete",
              "calibration_load_scale_MW": january["normalization_load_scale_MW"],
              "seasons": [{"start": period[0], "policy_count": s["policy_count"],
                           "feasible_policy_count": s["feasible_policy_count"],
                           "minimum_window_success_fraction": min(r["window_success_fraction"] for r in s["profiles"]),
                           "maximum_unserved_fraction": max(r["max_unserved_fraction"] for r in s["profiles"]),
                           "development_window_starts": s["development_window_starts"]}
                          for period, s in zip([base["development_period"]] + protocol["additional_development_periods"], summaries)],
              "diagnostic_window_evaluations_including_previous_january": sum(s["diagnostic_window_evaluations"] for s in summaries),
              "certification_comparison_gate": "stop" if all(s["certification_comparison_gate"] == "stop" for s in summaries) else "proceed",
              "confirmation_year_accessed": False, "optimizer_calls": 0, "terminal_verifier_calls": 0,
              "scope": "four fixed development samples; single national market; no annual-coverage claim"}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
