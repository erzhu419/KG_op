#!/usr/bin/env python3
"""Check three frozen Elexon sample days, retaining only small API responses.

This does not run an optimizer, fit source profiles, or read a confirmation year.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "performance/manifests/independent_energy_data_preflight_20260930.json"
DEFAULT_OUTPUT = ROOT / "paper_artifacts/independent_data_feasibility_20260930"


def instant(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def settlement_dates(start, stop):
    """Cover a UTC interval with British local settlement days."""
    local = ZoneInfo("Europe/London")
    return {"settlementDateFrom": start.astimezone(local).date().isoformat(),
            "settlementDateTo": (stop - timedelta(seconds=1)).astimezone(local).date().isoformat()}


def retrieve(session, base, endpoint, params, path):
    # Cached small responses make offline inspection possible without re-fetching.
    if path.exists():
        saved = json.loads(path.read_text())
    else:
        response = session.get(base + endpoint, params=params, timeout=30)
        response.raise_for_status()
        saved = {"url": response.url, "response_bytes": len(response.content),
                 "retrieved_at": datetime.now(timezone.utc).isoformat(),
                 "payload": response.json()}
        path.write_text(json.dumps(saved, indent=2) + "\n")
    data = saved["payload"]
    rows = data if isinstance(data, list) else data["data"]
    return rows, {k: v for k, v in saved.items() if k != "payload"}


def indexed(rows, start, stop):
    selected = {}
    for row in rows:
        stamp = instant(row["startTime"])
        if start <= stamp < stop:
            if stamp in selected:
                raise ValueError(f"duplicate settlement interval: {stamp}")
            selected[stamp] = row
    return selected


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect(protocol, out):
    samples = out / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    reports, half_hours, hours = [], [], []
    for day in protocol["sample_dates"]:
        start = instant(day + "T00:00:00Z")
        stop = start + timedelta(days=1)
        times = {"from": start.isoformat(), "to": stop.isoformat()}
        queries = {
            "forecast": (protocol["forecast_endpoint"], times),
            "actual": (protocol["actual_endpoint"], settlement_dates(start, stop)),
            "price": (protocol["price_endpoint"],
                      {**times, "dataProviders": protocol["price_provider"], "format": "json"}),
        }
        records, requests_log = {}, {}
        for kind, (endpoint, params) in queries.items():
            rows, metadata = retrieve(session, protocol["api_base"], endpoint, params,
                                      samples / f"{day}_{kind}.json")
            records[kind] = indexed(rows, start, stop)
            requests_log[kind] = {**metadata, "returned_rows": len(rows),
                                  "retained_rows": len(records[kind])}
        expected = {start + timedelta(minutes=30 * i)
                    for i in range(protocol["expected_intervals_per_sample"])}
        for kind, rows in records.items():
            if set(rows) != expected:
                raise ValueError(f"{day}/{kind}: missing or unexpected half-hour intervals")
        day_rows, leads = [], []
        for stamp in sorted(expected):
            forecast, actual, price = (records[k][stamp] for k in ("forecast", "actual", "price"))
            published = instant(forecast["publishTime"])
            if published >= stamp or forecast["boundary"] != "N":
                raise ValueError(f"{day}: forecast is late or has a different demand boundary")
            if price["dataProvider"] != protocol["price_provider"]:
                raise ValueError(f"{day}: unexpected market-index provider")
            values = [float(forecast[protocol["forecast_field"]]),
                      float(actual[protocol["actual_field"]]), float(price["price"])]
            if not all(math.isfinite(v) for v in values) or min(values[:2]) <= 0:
                raise ValueError(f"{day}: missing/nonfinite values or nonpositive demand")
            leads.append((stamp - published).total_seconds() / 3600)
            row = {"start_time_utc": stamp.isoformat(), "forecast_publish_time_utc": published.isoformat(),
                   "forecast_MW": values[0], "actual_MW": values[1], "price_GBP_per_MWh": values[2]}
            day_rows.append(row)
        half_hours.extend(day_rows)
        for i in range(0, len(day_rows), 2):
            pair = day_rows[i:i + 2]
            hours.append({"start_time_utc": pair[0]["start_time_utc"],
                          "forecast_available_by_utc": max(r["forecast_publish_time_utc"] for r in pair),
                          **{k: sum(r[k] for r in pair) / 2
                             for k in ("forecast_MW", "actual_MW", "price_GBP_per_MWh")}})
        reports.append({"date": day, "matched_half_hours": len(day_rows),
                        "complete_UTC_hours": len(day_rows) // 2,
                        "forecast_lead_hours_min": min(leads), "forecast_lead_hours_max": max(leads),
                        "forecasts_published_before_delivery": True, "requests": requests_log})
        print(f"{day}: {len(day_rows)} matched half-hours; forecast leads {min(leads):.2f}–{max(leads):.2f} hours",
              flush=True)
    write_csv(out / "matched_half_hours.csv", half_hours)
    write_csv(out / "matched_hours.csv", hours)
    summary = {"protocol_id": protocol["protocol_id"], "status": "sample_alignment_pass",
               "samples": reports, "response_bytes": sum(r["response_bytes"] for d in reports
                                                           for r in d["requests"].values()),
               "full_year_coverage_checked": False, "confirmation_year_accessed": False,
               "algorithm_evaluations": 0, "historical_policy_archives_recovered": 0,
               "scientific_scope": "independent-provider counterfactual storage candidate; no deployment evidence"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = inspect(json.loads(args.protocol.read_text()), args.out)
    print(json.dumps({k: summary[k] for k in ("status", "response_bytes", "algorithm_evaluations")}))


if __name__ == "__main__":
    main()
