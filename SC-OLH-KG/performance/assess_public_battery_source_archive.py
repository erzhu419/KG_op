#!/usr/bin/env python3
"""Assess saved population support before registering a historical source pilot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.inspect_public_battery_announced_library import decode_library, read
from performance.inspect_public_battery_announced_population import IDENTITY, key, lines

PROTOCOL = ROOT / "performance/manifests/public_battery_source_archive_assessment_v1_20261002.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_source_archive_assessment_v1_20261002"


def assess(rows, profile_ids, references, periods, starts, rules, required):
    expected = {(day, minute, profile, rule) for day in periods for minute in starts
                for profile in profile_ids + references for rule in rules}
    cells = {key(row): row for row in rows}
    if len(cells) != len(rows) or set(cells) != expected:
        raise ValueError("complete unique original population required; a local migration snapshot is insufficient")
    for row in rows:
        cost = row.get("cost_GBP")
        if (row["window_success"] and (cost is None or not np.isfinite(cost))
                or not row["window_success"] and cost is not None):
            raise ValueError("successful cost is missing or a failed window has a fabricated full cost")
    groups, reliable, full_support = [], [], []
    for profile in profile_ids:
        profile_groups = []
        for day in periods:
            for rule in rules:
                chosen = [cells[(day, minute, profile, rule)] for minute in starts]
                success = sum(r["window_success"] for r in chosen)
                group = dict(profile_id=profile, period=day, dispatch_rule=rule,
                             successful_windows=success, assessed_windows=len(starts),
                             meets_development_criterion=success >= required,
                             full_population_mean_cost_GBP=(float(np.mean([r["cost_GBP"] for r in chosen]))
                                                            if success == len(starts) else None))
                groups.append(group)
                profile_groups.append(group)
        if all(g["meets_development_criterion"] for g in profile_groups):
            reliable.append(profile)
        if all(g["successful_windows"] == len(starts) for g in profile_groups):
            full_support.append(profile)
    common, comparisons = [], []
    for day in periods:
        for rule in rules:
            shared = [minute for minute in starts if reliable and
                      all(cells[(day, minute, profile, rule)]["window_success"] for profile in reliable)]
            common.append(dict(period=day, dispatch_rule=rule, window_start_minutes=shared,
                               common_successful_windows=len(shared), candidates=len(reliable)))
            means = {profile: float(np.mean([cells[(day, minute, profile, rule)]["cost_GBP"] for minute in shared]))
                     for profile in reliable} if shared else {}
            comparisons.append(dict(period=day, dispatch_rule=rule, common_successful_windows=len(shared),
                                    mean_cost_GBP_by_candidate=means,
                                    interpretation="same successful window set; descriptive development comparison"))
    differences = []
    for day in periods:
        for minute in starts:
            for profile in profile_ids + references:
                a, b = [cells[(day, minute, profile, rule)] for rule in rules]
                shared_success = a["window_success"] and b["window_success"]
                differences.append(dict(success_differs=a["window_success"] != b["window_success"],
                    shared_success=shared_success,
                    cost_difference_GBP=abs(a["cost_GBP"] - b["cost_GBP"]) if shared_success else None))
    return dict(assessed_unique_cells=len(cells), original_joint_candidates=len(profile_ids),
        reliable_joint_candidates=reliable, reliable_constant_pairs=[p for p in reliable if p.startswith("constant_pair")],
        reliable_functional_pairs=[p for p in reliable if p.startswith("functional_pair")],
        full_cost_support_candidates=full_support, groups=groups, common_success_windows=common,
        common_cost_comparisons=comparisons,
        dispatch_rule_comparison=dict(paired_cells=len(differences),
            service_outcome_disagreements=sum(r["success_differs"] for r in differences),
            jointly_successful_pairs=sum(r["shared_success"] for r in differences),
            maximum_successful_cost_difference_GBP=max((r["cost_difference_GBP"] for r in differences
                                                       if r["shared_success"]), default=None)),
        all_library_profiles_have_full_cost_support=len(full_support) == len(profile_ids),
        source_pilot_registration_ready=bool(reliable) and len(reliable) < len(profile_ids),
        source_library_pruned=False, target_outcomes_allowed_in_source_selection=False,
        original_objective_atlas_interface_ready=len(full_support) == len(profile_ids),
        source_comparison_gate="HOLD", confirmation_year_access=False,
        new_controller_window_evaluations=0, algorithm_optimizer_calls=0, terminal_verifier_calls=0,
        limitations="Development support only; failed full-window costs remain undefined. Reliable candidates are not a source search domain. No source benefit or independent confirmation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--population", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    protocol = read(args.protocol)
    population = read(ROOT / protocol["population_protocol"])
    roster, _, _, _ = decode_library(read(ROOT / population["library_protocol"]))
    population_dir = args.population or ROOT / protocol["population_outputs"]
    result = assess(lines(population_dir / "cells.jsonl"), [r["profile_id"] for r in roster],
                    [f"constant_{v:.2f}" for v in population["population"]["extra_shared_references"]],
                    population["population"]["development_samples"], population["population"]["window_start_minutes"],
                    population["population"]["dispatch_rules"], population["gate"]["required_successes_per_sample"])
    result.update(protocol_id=protocol["protocol_id"], assessed_at_utc=datetime.now(timezone.utc).isoformat(),
                  population_journal_location=str(population_dir / "cells.jsonl"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("assessed_unique_cells", "reliable_constant_pairs", "reliable_functional_pairs",
          "full_cost_support_candidates", "dispatch_rule_comparison", "original_objective_atlas_interface_ready")}))


if __name__ == "__main__":
    main()
