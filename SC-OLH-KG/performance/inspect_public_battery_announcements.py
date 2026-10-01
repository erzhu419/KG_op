#!/usr/bin/env python3
"""Build and audit the frozen advance-announced simulation interface."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.public_battery_announcements import (
    announce_requests, delivery_clock_roster, released_announcements,
)

PROTOCOL = ROOT / "performance/manifests/public_battery_announced_task_contract_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_announced_task_contract_v1_20261001"


def inspect(protocol, out):
    contract = protocol["contract"]
    marks = json.loads((ROOT / protocol["cached_request_marks"]).read_text())
    signal = contract["announcement_clocks"]
    clocks = list(range(signal["start_minute"], signal["stop_exclusive_minute"], signal["step_minutes"]))
    if ([m["minute"] for m in marks] != clocks
            or len(marks) != contract["original_signal_clocks"]
            or sum(m["direction"] is not None for m in marks) != contract["expected_active_requests"]):
        raise ValueError("cached request population differs from the frozen announcement contract")
    ledger = announce_requests(marks, contract)
    delivery = delivery_clock_roster(ledger, contract)
    lead = contract["announcement_lead_minutes"]
    visibility_probes = 0
    margins = []
    for original, announced in zip(marks, ledger):
        if (announced["release_minute"] != original["minute"]
                or announced["delivery_minute"] != original["minute"] + lead
                or announced["direction"] != original["direction"]
                or announced["source"] != original["source"]
                or announced["peak_MW"] != (None if original["direction"] is None
                                             else contract["absolute_peak_MW"][original["direction"]])):
            raise ValueError("announcement transformation changed an original request")
        cutoff = announced["delivery_minute"] - 90
        visible = released_announcements(ledger, cutoff)
        visibility_probes += 1
        if announced not in visible or any(r["release_minute"] > cutoff for r in visible):
            raise ValueError("request is missing at its critical cutoff or an unreleased request is exposed")
        margins.append(cutoff - announced["release_minute"])
    requested = contract["delivery_clocks"]
    expected_delivery = list(range(requested["start_minute"], requested["stop_exclusive_minute"],
                                   requested["step_minutes"]))
    scheduled = [row for row in delivery if not row["bootstrap"]]
    warmup = [row for row in delivery if row["bootstrap"]]
    active = [row for row in scheduled if row["direction"] is not None]
    if ([row["delivery_minute"] for row in scheduled] != expected_delivery
            or len(active) != contract["expected_active_requests"]
            or [row["delivery_minute"] for row in warmup] != [0, 60]
            or any(row["release_minute"] is not None or row["direction"] is not None
                   or row["peak_MW"] is not None or row["source"] is not None for row in warmup)):
        raise ValueError("delivery roster changes service obligations or invents warmup announcements")
    summary = {"protocol_id": protocol["protocol_id"], "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "causal_announcement_interface_pass",
               "announcement_records": len(ledger), "delivery_clock_records": len(delivery),
               "active_service_obligations": len(active),
               "active_request_directions": dict(Counter(row["direction"] for row in active)),
               "inactive_announced_slots": len(scheduled) - len(active), "warmup_slots": len(warmup),
               "selected_sources_received_by_release": len(active),
               "critical_cutoff_visibility_probes": visibility_probes,
               "minimum_notice_margin_before_critical_cutoff_minutes": min(margins),
               "announcement_lead_minutes": lead,
               "total_execution_horizon_minutes": contract["total_execution_horizon_minutes"],
               "controller_interface": "released_announcements only; complete delivery roster is dispatcher data",
               "decision": "freeze_bounded_announced_nomination_component_preflight",
               **protocol["accounting_constraints"], "library_matrix_launch": False,
               "source_comparison_gate": protocol["source_comparison_gate"],
               "limitations": protocol["limitations"]}
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("announcements.json", ledger), ("delivery_clocks.json", delivery), ("summary.json", summary)):
        (out / name).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
