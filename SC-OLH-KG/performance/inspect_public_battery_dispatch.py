#!/usr/bin/env python3
"""Reconstruct a small saved BOA workload, preserving receipt time and revisions.

No network, optimizer or public-API latency inference. Gaps remain unobserved.
The idle-gap energy envelope is a diagnostic, not a plant operating history.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "performance/manifests/public_battery_dispatch_preflight_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_application_spec_20260930"


def instant(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass(frozen=True)
class Segment:
    start: datetime
    stop: datetime
    received: datetime
    number: int
    first_MW: float
    last_MW: float

    def power(self, time):
        weight = (time - self.start) / (self.stop - self.start)
        return self.first_MW + weight * (self.last_MW - self.first_MW)


def decode(rows, bm_unit):
    if any(r["bmUnit"] != bm_unit for r in rows):
        raise ValueError("workload contains a different BM unit")
    return [Segment(instant(r["timeFrom"]), instant(r["timeTo"]),
                    instant(r["acceptanceTime"]), int(r["acceptanceNumber"]),
                    float(r["levelFrom"]), float(r["levelTo"])) for r in rows]


def known_segment(segments, target_time, observed_at):
    """A received instruction can include future points; future receipts cannot."""
    covering = [s for s in segments if s.received <= observed_at
                and s.start <= target_time < s.stop]
    return max(covering, key=lambda s: (s.received, s.number), default=None)


def energy_pieces(first, last, hours):
    """Chronological export/import integrals, splitting a ramp at zero."""
    if first * last < 0:
        fraction = -first / (last - first)
        return energy_pieces(first, 0, hours * fraction) + energy_pieces(0, last, hours * (1 - fraction))
    signed = (first + last) * hours / 2
    return [(max(signed, 0), max(-signed, 0))]


def reconstruct(segments, start, stop):
    # These offline ledger boundaries are not a controller decision clock.
    boundaries = {start, stop}
    for s in segments:
        for t in (s.start, s.stop, s.received):
            if start < t < stop:
                boundaries.add(t)
    times = sorted(boundaries)
    ledger = []
    for left, right in zip(times, times[1:]):
        middle = left + (right - left) / 2
        selected = known_segment(segments, middle, middle)
        covering = [s for s in segments if s.received <= middle and s.start <= middle < s.stop]
        first, last = (selected.power(left), selected.power(right)) if selected else (None, None)
        exported, imported = 0.0, 0.0
        if selected:
            pieces = energy_pieces(first, last, (right - left).total_seconds() / 3600)
            exported, imported = map(sum, zip(*pieces))
        ledger.append({"from_utc": left.isoformat(), "to_utc": right.isoformat(),
                       "active_BOA": selected is not None,
                       "acceptance_number": selected.number if selected else None,
                       "received_utc": selected.received.isoformat() if selected else None,
                       "from_MW": first, "to_MW": last,
                       "export_MWh": exported, "import_MWh": imported,
                       "overlapping_segments": len(covering),
                       "naive_sum_from_MW": sum(s.power(left) for s in covering) if covering else None,
                       "naive_sum_to_MW": sum(s.power(right) for s in covering) if covering else None})
    return ledger


def summarize(ledger, asset):
    covered, overlap, cumulative, minimum, maximum = 0.0, 0.0, 0.0, 0.0, 0.0
    peak, naive_peak = 0.0, 0.0
    for row in ledger:
        hours = (instant(row["to_utc"]) - instant(row["from_utc"])).total_seconds() / 3600
        if not row["active_BOA"]:
            continue
        covered += hours
        overlap += hours if row["overlapping_segments"] > 1 else 0
        peak = max(peak, abs(row["from_MW"]), abs(row["to_MW"]))
        naive_peak = max(naive_peak, abs(row["naive_sum_from_MW"]), abs(row["naive_sum_to_MW"]))
        for exported, imported in energy_pieces(row["from_MW"], row["to_MW"], hours):
            cumulative += asset["charge_efficiency"] * imported - exported / asset["discharge_efficiency"]
            minimum, maximum = min(minimum, cumulative), max(maximum, cumulative)
    lower, upper = -minimum, asset["energy_MWh"] - maximum
    common = asset["energy_MWh"] / 2
    return {"active_BOA_hours": covered, "overlapping_BOA_hours": overlap,
            "peak_requested_MW": peak, "naive_sum_peak_MW": naive_peak,
            "gross_export_MWh": sum(r["export_MWh"] for r in ledger),
            "gross_import_MWh": sum(r["import_MWh"] for r in ledger),
            "idle_gap_envelope": {"minimum_capacity_MWh": maximum - minimum,
                "initial_inventory_lower_MWh": lower, "initial_inventory_upper_MWh": upper,
                "initial_inventory_interval_nonempty": lower <= upper,
                "common_initial_inventory_MWh": common,
                "common_initial_inventory_within_energy_bounds": lower <= common <= upper,
                "terminal_inventory_change_MWh": cumulative},
            "power_within_reference_limit": peak <= asset["power_MW"]}


def inspect(protocol, out):
    saved = json.loads((out / protocol["data"]["input"]).read_text())
    rows = saved["payload"]["data"]
    segments = decode(rows, protocol["asset"]["workload_bm_unit"])
    start, stop = map(instant, protocol["data"]["evaluation_interval_utc"])
    relevant = [s for s in segments if s.start < stop and s.stop > start]
    ledger = reconstruct(relevant, start, stop)
    with (out / "dispatch_20240415.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    availability = []
    for name in protocol["data"]["source_availability_probes"]:
        probe = json.loads((out / name).read_text())["payload"]["data"]
        availability.append({"file": name, "segments": len(probe),
                             "acceptances": len({r["acceptanceNumber"] for r in probe})})
    result = {"protocol_id": protocol["protocol_id"], "saved_response_bytes": saved["response_bytes"],
              "input_segments": len(rows), "evaluated_segments": len(relevant),
              "evaluated_acceptances": len({s.number for s in relevant}),
              "ledger_intervals": len(ledger), "source_availability": availability,
              **summarize(ledger, protocol["asset"]),
              "scope": protocol["diagnostic"]["energy_envelope"],
              "optimizer_calls": 0, "terminal_verifier_calls": 0, "confirmation_year_access": False}
    (out / "dispatch_preflight_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
