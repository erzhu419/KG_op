#!/usr/bin/env python3
"""Eight frozen two-call paths with stipulated advance request announcements."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.public_battery_dispatch import positive_integral
from problems.public_battery_pending_availability import pending_pulse_power
from problems.public_battery_visibility import trajectory

PROTOCOL = ROOT / "performance/manifests/public_battery_announced_pair_preflight_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_announced_pair_preflight_v1_20261001"


def pair_power(context, pair, roster):
    """Keep the first locked slot; replace only its tail, then use zero PN."""
    directions = pair.split("_")
    locked = np.vstack((context["first_hour_locked_PN_MW"][0], roster["first_tail_PN_MW"][pair]))
    first_hour = pending_pulse_power(locked, roster["absolute_peak_MW"][directions[0]])
    second_hour = pending_pulse_power(roster["second_hour_locked_PN_MW"],
                                      roster["absolute_peak_MW"][directions[1]])
    return tuple(np.concatenate((a, b)) for a, b in zip(first_hour, second_hour))


def inspect(protocol, out):
    basis = json.loads((ROOT / protocol["basis_protocol"]).read_text())
    pilot = json.loads((ROOT / basis["basis_protocol"]).read_text())
    absolute = json.loads((ROOT / pilot["basis_protocol"]).read_text())
    causal = json.loads((ROOT / absolute["basis_previous_pilot"]).read_text())
    base = json.loads((ROOT / causal["retained_controller_protocol"]).read_text())
    retained = json.loads((ROOT / protocol["retained_asset_protocol"]).read_text())["retained_asset"]
    asset = {"power_MW": retained["shared_gross_power_MW"],
             **{k: retained[k] for k in ("unit_energy_MWh", "charge_efficiency", "discharge_efficiency")}}
    tolerance = base["failure"]["numerical_energy_tolerance_MWh"]
    roster, information = protocol["component_roster"], protocol["information_intervention"]
    contexts = basis["component_probes"]["contexts"]
    lead = base["control"]["nomination_lead_minutes"]
    first_clock, second_clock = roster["first_call_minute"], roster["second_call_minute"]
    if (roster["absolute_peak_MW"] != absolute["contract"]["absolute_request_peak_MW"]
            or roster["tail_delivery_minute"] - lead != information["critical_nomination_minute"]
            or first_clock - information["first_call_announcement_minute"] != information["announcement_lead_minutes"]
            or second_clock - information["second_call_announcement_minute"] != information["announcement_lead_minutes"]
            or max(information["first_call_announcement_minute"], information["second_call_announcement_minute"])
            > information["critical_nomination_minute"]
            or [t + lead for t in roster["second_hour_nomination_minutes"]] != [second_clock, second_clock + 30]
            or roster["duration_minutes"] != 120
            or second_clock - first_clock != 60
            or roster["stop_exclusive_minute"] - first_clock != 120):
        raise ValueError("advance-notice path does not retain the frozen pulse and nomination timing")
    rows = []
    for context in contexts:
        if (context["first_call_minute"] != first_clock or context["second_call_minute"] != second_clock):
            raise ValueError("saved context clocks differ from the frozen two-call roster")
        initial = np.array(context["first_call_initial_SOC_MWh"])
        for pair in roster["sign_pairs"]:
            first, last = pair_power(context, pair, roster)
            metrics = trajectory(first, last, initial, roster["duration_minutes"], asset)
            outgoing = positive_integral(first, last, 1 / 60)
            incoming = positive_integral(-first, -last, 1 / 60)
            delta = asset["charge_efficiency"] * incoming - outgoing / asset["discharge_efficiency"]
            balance = np.array(metrics["final_SOC_MWh"]) - initial - delta.sum(axis=0)
            if np.max(np.abs(balance)) > tolerance:
                raise ValueError("two-call assembly does not conserve per-unit inventory")
            safe = (max(metrics["unit_bound_violation_MWh"]) <= tolerance
                    and metrics["power_bound_violation_MW"] <= tolerance)
            row = {"profile_id": context["profile_id"], "sign_pair": pair,
                   "initial_SOC_MWh": initial.tolist(), "target_fractions": context["target_fractions"],
                   "first_tail_PN_MW": roster["first_tail_PN_MW"][pair],
                   "second_call_initial_SOC_MWh": (initial + delta[:60].sum(axis=0)).tolist(),
                   "physics": metrics, "physical_feasible": bool(safe),
                   "full_requested_pulses_executed": 2,
                   "both_full_pulses_physically_deliverable": bool(safe),
                   "gross_export_MWh": outgoing.sum(axis=0).tolist(),
                   "gross_import_MWh": incoming.sum(axis=0).tolist(),
                   "maximum_shared_gross_power_MW": float(max(np.abs(first).sum(axis=1).max(),
                                                               np.abs(last).sum(axis=1).max())),
                   "energy_balance_error_MWh": balance.tolist()}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("profile_id", "sign_pair", "physical_feasible", "physics")}), flush=True)
    if len(rows) != roster["planned_component_trajectory_evaluations"]:
        raise ValueError("evaluated path count differs from the frozen eight-path roster")
    safe_count = sum(r["physical_feasible"] for r in rows)
    all_safe = safe_count == len(rows)
    summary = {"protocol_id": protocol["protocol_id"], "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "advance_notice_local_obstruction_removed" if all_safe else "advance_notice_component_failure",
               "component_trajectory_evaluations": len(rows), "physically_safe_components": safe_count,
               "full_request_pulses_evaluated": 2 * len(rows),
               "minimum_SOC_MWh": min(min(r["physics"]["minimum_SOC_MWh"]) for r in rows),
               "minimum_capacity_headroom_MWh": min(min(np.array(asset["unit_energy_MWh"])
                   - r["physics"]["maximum_SOC_MWh"]) for r in rows),
               "maximum_shared_gross_power_MW": max(r["maximum_shared_gross_power_MW"] for r in rows),
               "physical_tolerance": tolerance,
               "delivery_scope": "exact full pulses by construction, including PN ramps and both mandatory tails; no admission clipping or rejected requests",
               "decision": "assess_announced_task_justification_before_any_rollout" if all_safe
               else "retain_component_failures_and_diagnose",
               **protocol["accounting_constraints"], "source_comparison_gate": protocol["source_comparison_gate"],
               "library_matrix_launch": False, "limitations": protocol["limitations"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "components.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
