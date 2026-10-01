#!/usr/bin/env python3
"""Retrospective coverage of frozen source designs using saved development counts."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from performance.inspect_public_battery_announced_library import read

PROTOCOL=ROOT/"performance/manifests/public_battery_frozen_design_coverage_v1_20261002.json"
OUTPUT=ROOT/"paper_artifacts/public_battery_frozen_design_coverage_v1_20261002"


def main():
    protocol=read(PROTOCOL)
    source=read(ROOT/protocol["source_designs"])
    target=read(ROOT/protocol["target_assessment"])
    if source["status"]!="source_first_center_designs_frozen" or target["assessed_unique_cells"]!=68766:
        raise ValueError("complete source designs and saved development population required")
    lookup={(r["profile_id"],r["period"],r["dispatch_rule"]):r for r in target["groups"]}
    rows=[]
    for arm,ids in source["designs"].items():
        if len(ids)!=10 or len(set(ids))!=10: raise ValueError("frozen ten-point design changed")
        for period in protocol["periods"]:
            for rule in protocol["dispatch_rules"]:
                groups=[lookup[(profile,period,rule)] for profile in ids]
                safe=[r["profile_id"] for r in groups if r["successful_windows"]>=70]
                rows.append(dict(arm=arm,period=period,dispatch_rule=rule,initial_profile_ids=ids,
                    initial_success_counts={r["profile_id"]:r["successful_windows"] for r in groups},
                    development_feasible_initial_candidates=safe,initial_feasible_coverage=bool(safe),
                    first_center=ids[0],first_center_successful_windows=groups[0]["successful_windows"],
                    first_center_full_population_mean_cost_GBP=groups[0]["full_population_mean_cost_GBP"]))
    primary=[r for r in rows if r["dispatch_rule"]==protocol["primary_rule"]]
    contrast={arm:dict(covered_development_samples=sum(r["initial_feasible_coverage"] for r in primary if r["arm"]==arm),
                      assessed_development_samples=len(protocol["periods"])) for arm in source["designs"]}
    result=dict(protocol_id=protocol["protocol_id"],status="saved_development_coverage_complete",rows=rows,
        primary_rule_coverage=contrast,source_archive_controller_evaluations_paid_once=44968,
        new_controller_window_evaluations=0,algorithm_optimizer_calls=0,terminal_verifier_calls=0,
        source_or_target_designs_modified=False,independent_confirmation=False,confirmation_year_access=False,
        source_comparison_gate="HOLD",limitations=protocol["limitations"],
        next_action="register_fixed_budget_target_search_and_verification_with_explicit_undefined_cost_handling")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(coverage=contrast,new_controller_window_evaluations=0,source_comparison_gate="HOLD")))


if __name__=="__main__":main()
