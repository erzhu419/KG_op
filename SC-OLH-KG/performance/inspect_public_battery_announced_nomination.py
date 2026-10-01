#!/usr/bin/env python3
"""Eight frozen announcement-aware nominations and at most eight local paths."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.public_battery_announced_nomination import nominate_announced_pair
from problems.public_battery_dispatch import positive_integral
from problems.public_battery_pending_availability import pending_pulse_power
from problems.public_battery_visibility import trajectory

PROTOCOL = ROOT / "performance/manifests/public_battery_announced_nomination_preflight_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_announced_nomination_preflight_v1_20261001"


def read(relative):
    return json.loads((ROOT / relative).read_text())


def fixture_announcements(pair, roster, peaks):
    directions = pair.split("_")
    return [{"release_minute": release, "delivery_minute": delivery,
             "direction": direction, "peak_MW": peaks[direction], "source": None}
            for direction, release, delivery in zip(directions, roster["announcement_release_minutes"],
                                                    (roster["first_call_minute"], roster["second_call_minute"]))]


def inspect(protocol, out):
    contract = read(protocol["basis_protocol"])["contract"]
    targets = read(protocol["saved_target_protocol"])
    pilot = read(targets["basis_protocol"])
    absolute = read(pilot["basis_protocol"])
    causal = read(absolute["basis_previous_pilot"])
    base = read(causal["retained_controller_protocol"])
    retained = read(protocol["retained_asset_protocol"])["retained_asset"]
    asset = {"power_MW": retained["shared_gross_power_MW"],
             **{k: retained[k] for k in ("unit_energy_MWh", "charge_efficiency", "discharge_efficiency")}}
    tolerance = base["failure"]["numerical_energy_tolerance_MWh"]
    diagnosis = {r["profile_id"]: r["earlier_tail_nomination"]
                 for r in read(protocol["saved_nomination_contexts"])["rows"]}
    fractions = {c["profile_id"]: c["target_fractions"] for c in targets["component_probes"]["contexts"]}
    roster = protocol["component_roster"]
    minute = roster["nomination_minute"]
    if (minute - roster["original_nomination_minute"] != roster["clock_translation_minutes"]
            or roster["clock_translation_minutes"] != contract["announcement_lead_minutes"]
            or [roster["first_call_minute"], roster["second_call_minute"]] != [minute + 30, minute + 90]
            or roster["stop_exclusive_minute"] - minute != roster["duration_minutes"]
            or roster["duration_minutes"] != 150
            or [t + base["control"]["nomination_lead_minutes"] for t in roster["second_hour_nomination_minutes"]]
            != [roster["second_call_minute"], roster["second_call_minute"] + 30]):
        raise ValueError("automatic nomination fixture changes the frozen clock geometry")
    rows, path_count = [], 0
    out.mkdir(parents=True, exist_ok=True)
    for profile in roster["profiles"]:
        saved = diagnosis[profile]
        if saved["minute"] != roster["original_nomination_minute"]:
            raise ValueError("saved nomination does not match the frozen state origin")
        initial = np.array(saved["SOC_MWh"])
        current, following = np.array(saved["locked_PN_MW"])
        for pair in roster["sign_pairs"]:
            announcements = fixture_announcements(pair, roster, contract["absolute_peak_MW"])
            command, info = nominate_announced_pair(initial, current, following, fractions[profile],
                                                    minute, announcements, asset)
            if info["scalar_stock_projection_evaluations"] > 300:
                raise ValueError("nomination exceeds the frozen scalar-projection budget")
            row = {"profile_id": profile, "sign_pair": pair, "nomination_minute": minute,
                   "initial_SOC_MWh": initial.tolist(), "locked_PN_MW": saved["locked_PN_MW"],
                   "target_fractions": fractions[profile], "fixture_announcements": announcements,
                   "nomination_MW": None if command is None else command.tolist(),
                   **info, "physics": None, "probe_pass": False}
            if command is not None:
                prefix = np.repeat(current[None], 30, axis=0)
                first_hour = pending_pulse_power(np.vstack((following, command)), announcements[0]["peak_MW"])
                second_hour = pending_pulse_power(roster["second_hour_locked_PN_MW"], announcements[1]["peak_MW"])
                first, last = (np.concatenate((prefix, a, b)) for a, b in zip(first_hour, second_hour))
                metrics = trajectory(first, last, initial, roster["duration_minutes"], asset)
                path_count += 1
                incoming = positive_integral(-first, -last, 1 / 60)
                outgoing = positive_integral(first, last, 1 / 60)
                delta = asset["charge_efficiency"] * incoming - outgoing / asset["discharge_efficiency"]
                second_stock = initial + delta[:90].sum(axis=0)
                projection_error = second_stock - info["projected_second_call_SOC_MWh"]
                balance = np.array(metrics["final_SOC_MWh"]) - initial - delta.sum(axis=0)
                band = np.array(info["second_stock_band_MWh"])
                matches = bool(np.max(np.abs(projection_error)) <= tolerance)
                ready = bool(np.all(second_stock >= band[:, 0] - tolerance)
                             and np.all(second_stock <= band[:, 1] + tolerance))
                safe = bool(max(metrics["unit_bound_violation_MWh"]) <= tolerance
                            and metrics["power_bound_violation_MW"] <= tolerance)
                row.update(physics=metrics, physical_feasible=safe,
                           both_full_pulses_physically_deliverable=safe, full_requested_pulses_tested=2,
                           second_call_initial_SOC_MWh=second_stock.tolist(),
                           projection_error_MWh=projection_error.tolist(), projection_matches_execution=matches,
                           second_call_stock_in_required_band=ready, energy_balance_error_MWh=balance.tolist(),
                           maximum_shared_gross_power_MW=float(max(np.abs(first).sum(axis=1).max(),
                                                                 np.abs(last).sum(axis=1).max())),
                           probe_pass=bool(matches and ready and safe and np.max(np.abs(balance)) <= tolerance
                                           and info["nominated_gross_power_MW"] <= asset["power_MW"] + tolerance))
            rows.append(row)
            (out / "components.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps({k: row[k] for k in ("profile_id", "sign_pair", "nomination_MW", "probe_pass", "physics")}), flush=True)
    if (len(rows) != roster["planned_nomination_probes"]
            or path_count > roster["maximum_component_trajectory_evaluations"]):
        raise ValueError("component accounting differs from the frozen probe roster")
    passed = sum(row["probe_pass"] for row in rows)
    evaluated = [row for row in rows if row["physics"] is not None]
    summary = {"protocol_id": protocol["protocol_id"], "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "announced_nomination_component_pass" if passed == len(rows) else "announced_nomination_component_failure",
               "nomination_probes": len(rows), "passed_probes": passed,
               "component_trajectory_evaluations": path_count,
               "scalar_stock_projection_evaluations": sum(r["scalar_stock_projection_evaluations"] for r in rows),
               "maximum_scalar_stock_evaluations_per_probe": max(r["scalar_stock_projection_evaluations"] for r in rows),
               "full_requested_pulses_tested": 2 * path_count,
               "minimum_SOC_MWh": min(min(r["physics"]["minimum_SOC_MWh"]) for r in evaluated) if evaluated else None,
               "minimum_capacity_headroom_MWh": float(min(min(np.array(asset["unit_energy_MWh"])
                   - r["physics"]["maximum_SOC_MWh"]) for r in evaluated)) if evaluated else None,
               "maximum_shared_gross_power_MW": max(r["maximum_shared_gross_power_MW"] for r in evaluated) if evaluated else None,
               "maximum_projection_error_MWh": max(max(abs(e) for e in r["projection_error_MWh"])
                                                    for r in evaluated) if evaluated else None,
               "physical_tolerance": tolerance,
               "decision": "specify_complete_announced_policy_and_freeze_bounded_pilot" if passed == len(rows)
               else "retain_failed_probes_and_diagnose_before_controller_changes",
               **protocol["accounting_constraints"], "source_comparison_gate": protocol["source_comparison_gate"],
               "library_matrix_launch": False, "limitations": protocol["limitations"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
