"""Compare one preselected saved release run with its frozen Stage85 behavior.

JSON reads and scalar comparisons only: no simulation, model restoration,
engine checkpoint, large search.json, posterior update or statistical refit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time


TOLERANCE = 1e-12
ARM = "legacy_role_after_init_point_only"
SEMANTIC_ROW_KEYS = ("point", "shortlist_position", "shortlist_role", "source_shortlist_role")
INPUT_KEYS = ("seed", "arm", "domain", "protocol", "verification_seed", "primary",
    "candidate_budgets", "search_evaluation_count", "familywise_delta", "method",
    "mean_delta_fraction", "shortlist_mode", "objective_incumbent_position",
    "objective_comparison_budget", "objective_comparison_delta", "tau")


def read(path):
    return json.loads(Path(path).read_text())


def _summary(value):
    if isinstance(value, list) and len(value) <= 10 and all(not isinstance(v, (list, dict)) for v in value):
        return value
    if isinstance(value, (list, dict)):
        return dict(type=type(value).__name__, length=len(value))
    return value


def compare(old_case, old_input, new_output, case_json):
    started, wall = time.process_time(), time.perf_counter()
    report = dict(schema="mainline-frozen-behavior-parity-v1", status="failed",
        automatic_retries=0, new_simulation_calls=0, posterior_restorations=0,
        frozen_behavior_matches=False, complete_record_metadata_matches=False,
        float_tolerance=dict(absolute=TOLERANCE, relative=TOLERANCE),
        checks={}, differences=[], metadata_differences=[],
        mismatch_count=0, metadata_mismatch_count=0, float_values_compared=0,
        maximum_absolute_float_difference=0., truth_used=False,
        inputs=dict(old_case=str(old_case), old_input=str(old_input),
                    new_output=str(new_output), case_json=str(case_json)))

    def difference(group, path, expected, actual, metadata):
        field = "metadata_mismatch_count" if metadata else "mismatch_count"
        details = "metadata_differences" if metadata else "differences"
        report[field] += 1
        if not metadata:
            report["checks"][group] = False
        if len(report[details]) < 20:
            report[details].append(dict(check=group, path=path,
                expected=_summary(expected), actual=_summary(actual)))

    def equal(group, expected, actual, path="", *, metadata=False):
        report["checks"].setdefault(group, True)
        if isinstance(expected, dict) and isinstance(actual, dict):
            if set(expected) != set(actual):
                difference(group, path + ".keys", sorted(expected), sorted(actual), metadata)
            for key in expected.keys() & actual.keys():
                equal(group, expected[key], actual[key], path + "." + key, metadata=metadata)
        elif isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                difference(group, path + ".length", len(expected), len(actual), metadata)
            for index, (left, right) in enumerate(zip(expected, actual)):
                equal(group, left, right, path + f"[{index}]", metadata=metadata)
        elif isinstance(expected, bool) or isinstance(actual, bool):
            if type(expected) is not type(actual) or expected != actual:
                difference(group, path, expected, actual, metadata)
        elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            policy_coordinate = (".point" in path or ".primary" in path
                or path.startswith("deployed_policy") or path.startswith("actual_TS_selections"))
            if policy_coordinate or isinstance(expected, int) and isinstance(actual, int):
                matched = expected == actual
            else:
                report["float_values_compared"] += 1
                matched = math.isclose(expected, actual, rel_tol=TOLERANCE, abs_tol=TOLERANCE)
                if math.isfinite(expected) and math.isfinite(actual):
                    report["maximum_absolute_float_difference"] = max(
                        report["maximum_absolute_float_difference"], abs(expected-actual))
            if not matched:
                difference(group, path, expected, actual, metadata)
        elif type(expected) is not type(actual) or expected != actual:
            difference(group, path, expected, actual, metadata)

    def shortlist(group, expected, actual):
        equal(group, len(expected), len(actual), "shortlist.length")
        for index, (left, right) in enumerate(zip(expected, actual)):
            # After the production transform, the verifier uses ordered points
            # and the incumbent position. Roles are additionally kept exact.
            equal(group, {key: left[key] for key in SEMANTIC_ROW_KEYS if key in left},
                {key: right[key] for key in SEMANTIC_ROW_KEYS if key in right}, f"shortlist[{index}]")
            equal(group, {key: value for key, value in left.items() if key not in SEMANTIC_ROW_KEYS},
                {key: value for key, value in right.items() if key not in SEMANTIC_ROW_KEYS},
                f"shortlist[{index}].record_metadata", metadata=True)

    try:
        old_case, old_input, new_output, case_json = map(Path, (old_case, old_input, new_output, case_json))
        case = read(case_json)
        initial = read(old_case.parent / "initialization.json")["history"]
        portable_initial = read(case_json.parent / case["initial_rows_file"])
        if isinstance(portable_initial, dict):
            portable_initial = portable_initial["history"]
        queries = read(old_case / "paid_search_raw.json")
        frozen = read(old_input)
        old_verified = read(old_input.parent / "verification.json")
        new_history = read(new_output / "returned_search_history.json")
        new_frozen = read(new_output / "frozen_terminal_input.json")
        new_verified = read(new_output / "verification.json")
        receipt = read(new_output / "receipt.json")
        with (new_output / "raw_calls.jsonl").open() as source:
            raw = [json.loads(line) for line in source if line.strip()]
        search_raw = [row for row in raw if row["phase"] == "search"]
        verification_raw = [row for row in raw if row["phase"] == "verification"]
        equal("registration", ["InventorySupplyChain", 2026120001, "observation_1", ARM],
            [case["domain"], case["geometry_task_seed"], case["stream"], frozen["arm"]])
        equal("registration", case["problem_kwargs"], frozen["geometry"], "geometry")
        equal("registration", case["verification_seed"], frozen["verification_seed"], "verification_seed")
        equal("registration", 2026130001, case["observation_seed"], "observation_seed")
        equal("registration", 2026140001, case["verification_seed"], "verification_seed_registered")
        equal("completed_attempt", "complete", receipt["status"], "status")
        equal("completed_attempt", 0, receipt["automatic_retries"], "automatic_retries")
        equal("initial10", 10, len(initial), "old_initial.length")
        equal("initial10", True, all(len(row["point"]) == 1000 and len(row["raw"]) == 2
                                    for row in initial), "initial_d1000_paired_raw_shape")
        equal("initial10", initial, portable_initial, "portable_initial")
        equal("initial10", initial, new_history[:10], "replayed_initial")
        equal("queries", 13, len(new_history), "new_history.length")
        equal("queries", 3, len(queries), "old_queries.length")
        equal("queries", True, all(len(row["point"]) == 1000 and len(row["raw"]) == 2
                                  for row in queries), "query_d1000_paired_raw_shape")
        equal("queries", queries, new_history[10:], "old_vs_new_queries")
        equal("queries", queries, [{"point": row["point"], "raw": row["raw"]}
                                    for row in search_raw], "returned_raw_queries")
        equal("queries", ["complete"]*len(raw), [row["status"] for row in raw], "raw_call_statuses")
        audit = read(new_output / "query_audit.json")
        equal("queries", [10, 11, 12], [row["paid_prefix"] for row in audit["decisions"]], "decision_prefixes")
        equal("queries", [row["point"] for row in queries],
            [row["selected"] for row in audit["decisions"]], "actual_TS_selections")
        equal("frozen_input", {key: frozen[key] for key in INPUT_KEYS},
            {key: new_frozen[key] for key in INPUT_KEYS}, "terminal_inputs")
        shortlist("frozen_input", frozen["frozen_shortlist"], new_frozen["frozen_shortlist"])
        shortlist("frozen_input", frozen["frozen_shortlist"], old_verified["verification"]["frozen_shortlist"])
        shortlist("frozen_input", new_frozen["frozen_shortlist"], new_verified["verification"]["frozen_shortlist"])
        equal("deployment", old_verified["deployed"], new_verified["deployed"], "deployed_policy")
        required = ("status", "certified", "selected_shortlist_rank", "recommendation_changed",
                    "attempt_count", "attempts", "objective_comparison", "verification_budget",
                    "safety_verification_budget", "objective_comparison_budget", "total_evaluation_count")
        for payload in (old_verified["verification"], new_verified["verification"]):
            if not set(required).issubset(payload):
                raise KeyError(f"Missing required branch/claim fields: {sorted(set(required)-payload.keys())}")
        equal("verification", {key: value for key, value in old_verified["verification"].items()
                               if key not in ("frozen_shortlist", "primary_policy_fingerprint", "deployed_policy_fingerprint")},
            {key: value for key, value in new_verified["verification"].items()
             if key not in ("frozen_shortlist", "primary_policy_fingerprint", "deployed_policy_fingerprint")},
            "certification_and_objective_comparison")
        calls = old_verified["new_verification_calls"]
        equal("actual_calls", [3, 3, 3, 10], [len(queries), len(search_raw),
            receipt["new_target_search_calls"], receipt["reused_initial_calls"]], "search_and_reuse")
        equal("actual_calls", [calls]*4, [len(verification_raw), receipt["new_verification_calls"],
            old_verified["verification"]["verification_budget"],
            new_verified["verification"]["verification_budget"]], "verification_calls")
        equal("actual_calls", 3 + calls, receipt["new_physical_target_calls"], "new_physical_target_calls")
        equal("actual_calls", 13 + calls, receipt["logical_total_calls"], "logical_total_calls")
        equal("actual_calls", True, calls <= 352, "old_terminal_cap")
        report["frozen_behavior_matches"] = report["mismatch_count"] == 0
        report["complete_record_metadata_matches"] = report["metadata_mismatch_count"] == 0
        report["status"] = ("passed" if report["frozen_behavior_matches"] and report["complete_record_metadata_matches"]
            else "behavior_matches_metadata_differs" if report["frozen_behavior_matches"] else "failed")
        report["observed_calls"] = dict(new_search=len(search_raw), new_verification=len(verification_raw),
            old_verification=calls, initial_reused=receipt["reused_initial_calls"])
        report["metadata_scope"] = "Remaining frozen-row records are compared and reported; differing metadata prevents automatic full parity PASS even when all scientific branches match."
    except (OSError, ValueError, TypeError, KeyError) as error:
        report["error"] = repr(error)
    finally:
        report.update(cpu_seconds=time.process_time()-started, wall_seconds=time.perf_counter()-wall,
            completed_utc=datetime.now(timezone.utc).isoformat())
    report["exit_code"] = 0 if report["status"] == "passed" else 1
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-case", type=Path, required=True, help="Frozen AFTER source_case directory")
    parser.add_argument("--old-input", type=Path, required=True, help="Frozen old terminal input.json")
    parser.add_argument("--new-output", type=Path, required=True, help="Completed run_small output directory")
    parser.add_argument("--case-json", type=Path, required=True, help="Portable registered case.json")
    parser.add_argument("--output", type=Path, required=True, help="New compact parity JSON, never overwritten")
    args = parser.parse_args(argv)
    result = compare(args.old_case, args.old_input, args.new_output, args.case_json)
    with args.output.open("x") as destination:
        destination.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
