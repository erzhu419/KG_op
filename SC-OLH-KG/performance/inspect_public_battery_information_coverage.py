#!/usr/bin/env python3
"""Describe adjacent calls and cached cues at frozen nomination deadlines."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "performance/manifests/public_battery_nomination_information_coverage_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_nomination_information_coverage_v1_20261001"


def coverage(rows):
    eligible = [r for r in rows if not r["bootstrap_deadline"]]
    present = [r for r in eligible if r["deadline_cue"]["direction"] is not None]
    late = [r for r in rows if r["late_cue"]["direction"] is not None]
    return {"pairs": len(rows), "bootstrap_deadlines": len(rows) - len(eligible),
            "auditable_deadlines": len(eligible), "deadline_cue_present": len(present),
            "deadline_cue_absent": len(eligible) - len(present),
            "deadline_cue_matches_first": sum(r["deadline_matches_first"] is True for r in present),
            "deadline_cue_matches_second": sum(r["deadline_matches_second"] is True for r in present),
            "final_first_source_already_received": sum(
                r["final_first_source_already_received"] is True for r in eligible),
            "final_first_source_not_yet_received": sum(
                r["final_first_source_already_received"] is False for r in eligible),
            "late_cue_present": len(late), "late_cue_absent": len(rows) - len(late),
            "late_cue_matches_second": sum(r["late_matches_second"] is True for r in late)}


def audit(marks, cues):
    by_minute = {c["minute"]: c for c in cues}
    rows = []
    for first, second in zip(marks, marks[1:]):
        if (second["minute"] - first["minute"] != 60
                or first["direction"] is None or second["direction"] is None):
            continue
        deadline, late_clock = second["minute"] - 90, second["minute"] - 30
        bootstrap = deadline < 0
        # Missing cached rows are data errors; a present row with no sign is an
        # absent signal. Bootstrap is separate and never invents a cue.
        cue = None if bootstrap else by_minute[deadline]
        late = by_minute[late_clock]
        sign = None if cue is None else cue["direction"]
        received = first["source"]["received_minute"]
        rows.append({"first_call_minute": first["minute"], "second_call_minute": second["minute"],
                     "sign_pair": first["direction"] + "_" + second["direction"],
                     "first_direction": first["direction"], "second_direction": second["direction"],
                     "tail_PN_nomination_minute": deadline,
                     "tail_PN_delivery_minute": second["minute"] - 30,
                     "bootstrap_deadline": bootstrap, "deadline_cue": cue,
                     "deadline_matches_first": None if sign is None else sign == first["direction"],
                     "deadline_matches_second": None if sign is None else sign == second["direction"],
                     "final_first_source_received_minute": received,
                     "final_first_source_already_received": None if bootstrap else received <= deadline,
                     "late_cue": late, "late_PN_delivery_minute": second["minute"] + 30,
                     "late_matches_second": None if late["direction"] is None
                     else late["direction"] == second["direction"]})
    groups = {a + "_" + b: coverage([r for r in rows if r["sign_pair"] == a + "_" + b])
              for a in ("export", "import") for b in ("export", "import")}
    return {"total": coverage(rows), "by_sign_pair": groups}, rows


def inspect(protocol, out):
    inputs, scope = protocol["cached_inputs"], protocol["scope"]
    marks = json.loads((ROOT / inputs["request_marks"]).read_text())
    cues = json.loads((ROOT / inputs["nomination_cues"]).read_text())
    active = [m for m in marks if m["direction"] is not None]
    if ([m["minute"] for m in marks] != list(range(0, scope["request_clocks"] * 60, 60))
            or [c["minute"] for c in cues] != list(range(0, scope["nomination_clocks"] * 30, 30))
            or len(active) != scope["expected_active_requests"]):
        raise ValueError("cached clock roster differs from the frozen audit population")
    totals, rows = audit(marks, cues)
    touched = {r[k] for r in rows for k in ("first_call_minute", "second_call_minute")}
    summary = {"protocol_id": protocol["protocol_id"],
               "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "cached_nomination_information_coverage_complete",
               "request_clocks": len(marks), "nomination_clocks": len(cues),
               "active_requests": len(active), "active_requests_in_adjacent_pairs": len(touched),
               "active_requests_without_active_neighbor": len(active) - len(touched),
               "active_request_directions": dict(Counter(m["direction"] for m in active)),
               **totals, "decision": "review_task_information_contract_before_controller_search",
               **protocol["accounting_constraints"], "library_matrix_launch": False,
               "source_comparison_gate": protocol["source_comparison_gate"],
               "interpretation": scope["interpretation"], "limitations": protocol["limitations"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "adjacent_pairs.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
