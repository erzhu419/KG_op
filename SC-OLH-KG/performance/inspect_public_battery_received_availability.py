#!/usr/bin/env python3
"""Four received-BOA fixtures, then 292 probes of saved April V5 states."""
import argparse
from collections import Counter
from datetime import timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from performance.inspect_public_battery_dispatch import Segment, instant
from performance.inspect_public_battery_partition import cached_unit
from problems.public_battery_pending_availability import declare_pending, evaluate_pending
from problems.public_battery_site import receipt_plans

PROTOCOL = ROOT / "performance/manifests/public_battery_received_availability_preflight_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_received_availability_preflight_v1_20261001"
TRACE_KEYS = ("SOC_trace_MWh", "power_first_MW", "power_last_MW")


def received_reference(locked, plans, index):
    baseline = np.repeat(np.asarray(locked, float), 30, axis=0)
    active = plans["active"][index, :60]
    return tuple(np.where(active, plans[key][index, :60], baseline)
                 for key in ("first_MW", "last_MW"))


def fixture_reference(spec, fixture):
    # Only a relative clock for synthetic fixtures, not historical plant data.
    start = instant("2024-04-15T00:00:00Z")
    units = [[], []]
    for number, item in enumerate([spec["known_BOA"]] +
                                  ([fixture["future_BOA"]] if "future_BOA" in fixture else []), 1):
        at = lambda key: start + timedelta(minutes=item[key])
        units[item["unit"] - 1].append(Segment(at("start_minute"), at("stop_minute"),
                                              at("received_minute"), number,
                                              item["power_MW"], item["power_MW"]))
    return received_reference(spec["locked_baseline_MW"], receipt_plans(units, start, 90), 0)


def compact_result(declaration, result):
    fields = {k: v for k, v in result.items() if k not in TRACE_KEYS + ("physics",)}
    fields.update(declaration_reason=declaration["reason"],
                  available_capacity_MW=declaration["available_capacity_MW"],
                  guaranteed_peak_MW=declaration["guaranteed_peak_MW"],
                  envelope_path_evaluations=declaration["envelope_path_evaluations"])
    return {**{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fields.items()},
            **result["physics"]}


def component_fixtures(spec, asset):
    rows, traces = [], []
    for fixture in spec["cases"]:
        reference = fixture_reference(spec, fixture)
        declaration = declare_pending(fixture["initial_SOC_MWh"], spec["locked_baseline_MW"],
                                      spec["service_capacity_MW"], spec["direction"], asset,
                                      reference_power=reference)
        proposal = None if fixture["new_request"] is None else fixture["new_request"]["peak_MW"]
        result = evaluate_pending(declaration, proposal, spec["service_capacity_MW"], asset)
        rows.append({"fixture": fixture["id"], **compact_result(declaration, result)})
        traces.append({"fixture": fixture["id"], **{k: result[k].tolist() for k in TRACE_KEYS},
                       "reference_first_MW": reference[0].tolist(), "reference_last_MW": reference[1].tolist()})
    # These contract checks stop the historical diagnosis on a fixture discrepancy.
    absent, zero, future, unsafe = rows
    np.testing.assert_allclose(absent["gross_export_MWh"], [20., 0.], atol=1e-8)
    np.testing.assert_allclose(absent["gross_import_MWh"], [0., 13.], atol=1e-8)
    np.testing.assert_allclose(zero["gross_export_MWh"], [20 * 29 / 60, 0.], atol=1e-8)
    np.testing.assert_allclose(zero["gross_import_MWh"], [0., 13 * 29 / 60], atol=1e-8)
    np.testing.assert_allclose(zero["admitted_peak_MW"], [0., 0.], atol=1e-8)
    np.testing.assert_allclose(absent["guaranteed_peak_MW"], [36., 36.], atol=1e-8)
    if {k: v for k, v in absent.items() if k != "fixture"} != {k: v for k, v in future.items() if k != "fixture"}:
        raise ValueError("a later BOA receipt altered the current declaration or execution")
    if not all(r["physical_feasible"] and r["service_success"] for r in rows[:3]) or not unsafe["known_commitment_failure"]:
        raise ValueError("received-commitment fixture accounting disagrees with the frozen contract")
    return rows, traces


def inspect(protocol, out):
    original = json.loads((ROOT / protocol["retained_asset_protocol"]).read_text())["retained_asset"]
    asset = {"power_MW": original["shared_gross_power_MW"],
             **{k: original[k] for k in ("unit_energy_MWh", "charge_efficiency", "discharge_efficiency")}}
    fixtures, traces = component_fixtures(protocol["component_fixtures"], asset)
    spec = protocol["retained_state_diagnosis"]
    history = json.loads((ROOT / protocol["historical_basis"]).read_text())
    base = json.loads((ROOT / history["basis_controller_protocol"]).read_text())
    prior = ROOT / "paper_artifacts/public_battery_unit_targets_failure_visibility_v1_20260930"
    contexts = json.loads((prior / "probes.json").read_text())
    expected = {(name, minute) for name in spec["profiles"] for minute in range(0, 4321, 60)}
    found = [(p["row"]["profile_id"], p["row"]["window_start_minute"]) for p in contexts]
    if set(found) != expected or len(contexts) != spec["contexts"]:
        raise ValueError("retained context roster differs from the frozen 146-state diagnosis")
    case = next(c for c in base["development"] if c["start"][:10] == spec["period"])
    start, stop = instant(case["start"]), instant(case["stop"])
    units = [cached_unit(base, case, u) for u in (0, 1)]
    plans = receipt_plans(units, start, int((stop - start).total_seconds() / 60))
    tolerance = base["failure"]["numerical_energy_tolerance_MWh"]
    rows = []
    for probe in contexts:
        saved, context = probe["row"], probe["submission_context"]
        decision = saved["window_start_minute"] + context["nomination_decision_minute"]
        if context["period"] != spec["period"] or decision != saved["decision_absolute_minute"]:
            raise ValueError("retained context decision clock disagrees with its provenance")
        reference = received_reference(context["pending_MW"], plans, decision // 30)
        for direction in spec["directions"]:
            declaration = declare_pending(context["decision_SOC_MWh"], context["pending_MW"],
                                          spec["service_probe_MW"], direction, asset, tolerance,
                                          reference_power=reference)
            # Detect a wrong receipt/state linkage; do not repeat the V5 rollout.
            for key, value in probe["reconstruction"]["known_pending_hour"].items():
                np.testing.assert_allclose(declaration["baseline_physics"][key], value, atol=tolerance, rtol=0.)
            peak = declaration["anchor_MW"] + (1 if direction == "export" else -1) * np.array(spec["service_probe_MW"])
            result = evaluate_pending(declaration, peak, spec["service_probe_MW"], asset, tolerance)
            if not declaration["known_commitment_failure"] and not result["physical_feasible"]:
                raise ValueError("admission made a previously safe mandatory reference physically unsafe")
            rows.append({**{k: saved[k] for k in ("period", "profile_id", "window_start_minute",
                                                  "decision_absolute_minute", "failure_absolute_minute")},
                         "initial_SOC_MWh": context["decision_SOC_MWh"],
                         "locked_baseline_MW": context["pending_MW"], "direction": direction,
                         "received_BOA_unit_minutes": np.sum(plans["active"][decision // 30, :60], axis=0).tolist(),
                         **compact_result(declaration, result)})
    groups = []
    for name in spec["profiles"]:
        for direction in spec["directions"]:
            group = [r for r in rows if r["profile_id"] == name and r["direction"] == direction]
            groups.append({"profile_id": name, "direction": direction, "probes": len(group),
                           "physically_feasible_probes": sum(r["physical_feasible"] for r in group),
                           "full_capacity_service_successes": sum(r["service_success"] for r in group),
                           "declaration_reasons": dict(Counter(r["declaration_reason"] for r in group)),
                           "available_capacity_range_MW": [np.min([r["available_capacity_MW"] for r in group], axis=0).tolist(),
                                                           np.max([r["available_capacity_MW"] for r in group], axis=0).tolist()],
                           "total_unmet_energy_MWh": sum(r["total_unmet_energy_MWh"] for r in group)})
    summary = {"protocol_id": protocol["protocol_id"], "status": "received_availability_diagnostic_complete",
               "fixture_replays": len(fixtures), "physically_feasible_fixtures": sum(r["physical_feasible"] for r in fixtures),
               "fixture_service_successes": sum(r["service_success"] for r in fixtures), "fixtures": fixtures,
               "retained_states": len(contexts), "historical_capacity_probes": len(rows),
               "known_safe_mandatory_references": len(contexts) - sum(r["known_commitment_failure"] for r in rows[::2]),
               "physically_feasible_historical_probes": sum(r["physical_feasible"] for r in rows),
               "full_capacity_service_successes": sum(r["service_success"] for r in rows),
               "declaration_reasons": dict(Counter(r["declaration_reason"] for r in rows)), "groups": groups,
               "distinct_absolute_nomination_minutes": len({r["decision_absolute_minute"] for r in rows}),
               "distinct_absolute_failure_minutes": len({r["failure_absolute_minute"] for r in rows}),
               "envelope_path_evaluations": sum(r["envelope_path_evaluations"] for r in fixtures + rows),
               "pulse_replays": len(fixtures) + len(rows), "source_comparison_gate": protocol["source_comparison_gate"],
               **protocol["accounting"], "full_experiment_launch": False,
               "decision": "freeze_a_bounded_causal_service_simulation_before_any_controller_rollout",
               "limitations": "four synthetic component fixtures and correlated saved development states; 49 MW is an analyst stress probe; common scaling is conservative; capacity shortfall is not universal online impossibility or identified historical BM service performance"}
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("summary.json", summary), ("probes.json", rows), ("component_traces.json", traces)):
        (out / name).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("fixtures", "groups")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
