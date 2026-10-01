#!/usr/bin/env python3
"""Interpret stored preflight results without re-running any simulation."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from problems.randomized_profiles import generate_structural_profile_library


def main():
    manifest = ROOT / "performance/manifests/independent_energy_seasonal_preflight_20260930.json"
    protocol = json.loads(manifest.read_text())
    base = json.loads((manifest.parent / protocol["base_protocol"]).read_text())
    previous = ROOT / protocol["existing_case"]
    library = base["library"]
    ids = {p.profile_id for p in generate_structural_profile_library(library["profiles"],
           dimension=library["nodes"], seed=library["seed"], maximum_frequency=library["maximum_frequency"])}
    folders = [previous] + [previous.parent / "seasonal" / p[0][:10]
                           for p in protocol["additional_development_periods"]]
    seasons, response_bytes, diagnostic_calls = [], 0, 0
    for folder in folders:
        s = json.loads((folder / "summary.json").read_text())
        pool = [r for r in s["profiles"] if r["profile_id"] in ids]
        if {r["profile_id"] for r in pool} != ids:
            raise ValueError("stored results do not cover the frozen 64-profile candidate library")
        seasons.append({"case": str(folder.relative_to(ROOT)), "library_profiles": len(pool),
                        "library_feasible_profiles": sum(r["empirically_feasible"] for r in pool),
                        "library_minimum_window_success_fraction": min(r["window_success_fraction"] for r in pool),
                        "development_window_starts": s["development_window_starts"],
                        "infeasible_reference_ids": [r["profile_id"] for r in s["profiles"]
                                                     if r["profile_id"] not in ids and not r["empirically_feasible"]],
                        "maximum_hourly_exchange": max(r["max_hourly_exchange"] for r in s["profiles"])})
        response_bytes += sum(r["response_bytes"] for r in s["request_logs"]["development"].values())
        diagnostic_calls += s["diagnostic_window_evaluations"]
    january = json.loads((previous / "summary.json").read_text())
    response_bytes += sum(r["response_bytes"] for r in january["request_logs"]["calibration"].values())
    uniform = all(s["library_minimum_window_success_fraction"] == 1.0 for s in seasons)
    report = {"status": "data_available_initializer_certification_contrast_inadequate" if uniform else "initializer_contrast_requires_review",
              "seasons": seasons, "accepted_preflight_response_bytes": response_bytes,
              "diagnostic_window_evaluations": diagnostic_calls,
              "physical_pool_gate": "active through additional zero-response reference",
              "full_comparison_decision": "defer; all frozen library candidates succeed in every sampled window" if uniform else "review the nonuniform candidate-library outcomes before registration",
              "implied_hypothetical_asset": {"energy_MWh": january["normalization_load_scale_MW"] * january["physics"]["energy_capacity"],
                                             "power_MW": january["normalization_load_scale_MW"] * january["physics"]["power_capacity"]},
              "confirmation_year_accessed": False, "algorithm_effect_estimated": False,
              "next_step": "establish public operational task and asset parameters before registering a new comparison",
              "limitations": "short source calibration; four overlapping-window development samples; one national market; no annual coverage or deployment evidence"}
    path = previous.parent / "feasibility_decision.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "accepted_preflight_response_bytes", "diagnostic_window_evaluations", "full_comparison_decision")}))


if __name__ == "__main__":
    main()
