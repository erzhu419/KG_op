#!/usr/bin/env python3
"""Read paid independent-panel receipts or summarize operating risk and call contrasts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import time


FAMILIES = ("Inventory_CrossHorizon", "Queueing_CrossHorizon", "Traffic_Ingolstadt")
METHODS = ("GP-FTA-KG", "GP-FTA-TS", "PublicPsi-TargetGP-TS")
INCUMBENT_ROLES = {"empirical_feasible_incumbent", "safe_initial_incumbent"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def audit_class(lower: float, upper: float) -> str:
    if upper <= 0.05:
        return "confidently_safe"
    if lower > 0.05:
        return "confidently_unsafe"
    return "boundary_ambiguous"


def binomial_tail(k: int, n: int) -> float:
    return math.fsum(math.comb(n, j) * 0.05 ** j * 0.95 ** (n - j) for j in range(k + 1))


def verify_terminal(row: dict, certs: list[dict]) -> list[int]:
    passing = []
    for index, entry in enumerate(certs):
        n = int(entry["n_samples"])
        count = int(entry["observed_violations"])
        delta = float(entry["delta_allocated"])
        critical = int(entry["critical_k"])
        assert int(entry["slot_index"]) == index and entry["method"] == "exact_binomial"
        assert 0 <= count <= n and -1 <= critical <= n
        assert binomial_tail(critical, n) <= delta
        if critical < n:
            assert binomial_tail(critical + 1, n) > delta
        probability = binomial_tail(count, n)
        assert math.isclose(probability, float(entry["p_value_or_margin"]), rel_tol=1e-12)
        assert (probability <= delta) == (count <= critical) == (str(entry["certified"]) == "True")
        if count <= critical:
            passing.append(index)
    assert certs and sum(int(c["n_samples"]) for c in certs) == int(row["verification_calls_used"])
    assert int(row["search_calls_used"]) + int(row["verification_calls_used"]) == int(row["total_target_calls"])

    roles = row.get("shortlist_roles", [entry["role"] for entry in certs])
    guard = len(roles) >= 2 and roles[1] in INCUMBENT_ROLES
    if guard and 0 in passing and 1 in passing:
        selected = min((0, 1), key=lambda slot: float(certs[slot]["sample_mean_obj"]))
    else:
        selected = passing[0] if passing else None
    recorded = row["selected_slot"]
    assert selected == (None if recorded in (None, "") else int(recorded))
    assert (selected is not None) == (str(row["certified"]) == "True")
    assert row["decision_state"] == ("abstention" if selected is None else "certified_deployment")
    if selected is not None:
        assert row["selected_role"] == roles[selected]
    expected_tests = len(roles) if selected is None else max(selected + 1, 2 if guard else 1)
    assert len(certs) == expected_tests
    if "shortlist_policies" in row:
        policies = row["shortlist_policies"]
        assert row["primary_policy"] == policies[0]
        assert row["deployed_policy"] == (None if selected is None else policies[selected])
    return passing


def collect(campaign: Path) -> dict:
    """Extract compact statistics at the sealed campaign directory on the cluster."""
    start = time.process_time()
    summary = read_json(campaign / "compact_summary.json")
    bindings = {key: summary[key] for key in ("manifest_sha256", "execution_code_sha256")}
    anchors = []
    for path in sorted((campaign / "stages/audit_anchors/items").glob("*.json")):
        row = read_json(path)
        family, index = row["job_id"].removeprefix("anchor_").rsplit("_t", 1)
        audit = row["anchor_audit"]
        assert row["truth_used"] and all(row[key] == value for key, value in bindings.items())
        assert row["audit_calls_used"] == audit["n_audit_samples"] == 2000
        assert audit["mc_violation_prob"] == audit["mc_violation_count"] / audit["n_audit_samples"]
        assert audit["safety_audit_class"] == audit_class(audit["mc_viol_ci95_lo"], audit["mc_viol_ci95_hi"])
        anchors.append({
            "family": family, "task": int(index), "seed": row["audit_seed"], "n": audit["n_audit_samples"],
            "violations": audit["mc_violation_count"], "mean_loss": audit["mc_objective_mean"],
            "violation_probability": audit["mc_violation_prob"],
            "ci95": [audit["mc_viol_ci95_lo"], audit["mc_viol_ci95_hi"]],
            "audit_class": audit["safety_audit_class"],
        })
    assert {(row["family"], row["task"]) for row in anchors} == {
        (family, idx) for family in FAMILIES for idx in range(24)
    }

    groups = {(family, method): Counter() for family in FAMILIES for method in METHODS}
    n_slots = 0
    for path in sorted((campaign / "stages/search_and_verify/items").glob("*.json")):
        row = read_json(path)
        assert not row["truth_used"] and all(row[key] == value for key, value in bindings.items())
        assert row["search_calls_used"] == 136 and row["actual_online_steps"] == 60
        certs = row["slot_certificates"]
        verify_terminal(row, certs)
        n_slots += len(certs)
        group = groups[row["family"], row["method_name"]]
        visits = row["n_replication_queries"]
        group.update({
            "runs": 1, "online_queries": row["actual_online_steps"], "revisits": visits,
            "runs_with_revisit": int(visits > 0),
        })
    assert all(group["runs"] == 48 for group in groups.values())
    assert n_slots == 885
    return {
        **bindings,
        "anchors": anchors,
        "visits": {family: {method: dict(groups[family, method]) for method in METHODS} for family in FAMILIES},
        "terminal_slot_rows_checked": n_slots,
        "worker_cpu_seconds": time.process_time() - start,
        "simulator_calls": 0,
    }


def summarize(campaign: Path, receipts: dict) -> dict:
    """Join the compact server readout to the already exported ledgers."""
    from scipy.stats import t as student_t

    summary = read_json(campaign / "compact_summary.json")
    assert all(receipts[k] == summary[k] for k in ("manifest_sha256", "execution_code_sha256"))
    rows = read_csv(campaign / "per_run_ledger.csv")
    by_job = defaultdict(list)
    certs = read_csv(campaign / "per_slot_certificates.csv")
    for row in certs:
        by_job[row["job_id"]].append(row)
    assert len(certs) == receipts["terminal_slot_rows_checked"]
    assert len(by_job) == len(rows) == len({row["job_id"] for row in rows}) == 432

    anchors = {(row["family"], row["task"]): row for row in receipts["anchors"]}
    groups = {(family, method): [] for family in FAMILIES for method in METHODS}
    for row in rows:
        cert = str(row["certified"]) == "True"
        records = sorted(by_job[row["job_id"]], key=lambda c: int(c["slot_index"]))
        verify_terminal(row, records)
        anchor = anchors[row["family"], int(row["target_task_idx"])]
        assert float(row["anchor_mc_obj"]) == anchor["mean_loss"] and int(row["audit_seed"]) == anchor["seed"]
        class_key = row["safety_audit_class"]
        if cert:
            assert class_key == audit_class(float(row["deployed_mc_viol_ci95_lo"]), float(row["deployed_mc_viol_ci95_hi"]))
        else:
            assert class_key == "abstained_not_deployed" and float(row["status_quo_loss"]) == anchor["mean_loss"]
        groups[row["family"], row["method_name"]].append({
            "task": int(row["target_task_idx"]), "stream": int(row["rep_seed"]),
            "calls": int(row["total_target_calls"]), "search": int(row["search_calls_used"]),
            "selection": int(row["selected_slot"]) if cert else None,
            "risk": class_key if cert else anchor["audit_class"],
            "anchor_probability": anchor["violation_probability"],
        })

    families = {}
    mult = float(student_t.ppf(0.975, 23))
    for family in FAMILIES:
        audit_rows = [row for (fam, _), row in anchors.items() if fam == family]
        min_risk_anchor = min(audit_rows, key=lambda row: row["violation_probability"])
        families[family] = {
            "anchor_classes": dict(Counter(row["audit_class"] for row in audit_rows)),
            "anchor_risk_range": [min_risk_anchor["violation_probability"], max(row["violation_probability"] for row in audit_rows)],
            "minimum_risk_anchor": min_risk_anchor,
            "arms": {},
        }
        all_calls = {}
        for method in METHODS:
            selected = groups[family, method]
            operated = Counter(row["risk"] for row in selected)
            abstained = [row for row in selected if row["selection"] is None]
            fallback_probs = [row["anchor_probability"] for row in abstained]
            assert len(selected) == 48 and {row["search"] for row in selected} == {136}
            all_calls[method] = {(row["task"], row["stream"]): row["calls"] for row in selected}
            families[family]["arms"][method] = {
                "certified": len(selected) - len(abstained), "retained_anchor": len(abstained),
                "operating_policy_classes": dict(operated),
                "retained_anchor_classes": dict(Counter(row["risk"] for row in abstained)),
                "retained_anchor_risk_range": [min(fallback_probs), max(fallback_probs)] if fallback_probs else None,
                "selected_outside_primary": sum(row["selection"] in (1, 2) for row in selected),
                **receipts["visits"][family][method],
            }
        kg_calls = all_calls[METHODS[0]]
        for method in METHODS[1:]:
            control = all_calls[method]
            assert set(kg_calls) == set(control) and len(control) == 48
            differences = [statistics.mean(control[key] - kg_calls[key] for key in sorted(control) if key[0] == task)
                           for task in range(24)]
            mean = statistics.mean(differences)
            halfwidth = mult * statistics.stdev(differences) / math.sqrt(len(differences))
            families[family]["arms"][method]["paired_call_difference_vs_kg"] = {
                "control_minus_kg_mean": mean, "ci95": [mean - halfwidth, mean + halfwidth],
            }
    return {"run_rows_checked": len(rows), "terminal_slot_rows_checked": len(certs), "families": families}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--stage-readout", type=Path, help="Join a compact server readout with local sealed CSVs.")
    args = parser.parse_args()
    if args.stage_readout is None:
        print(json.dumps(collect(args.campaign), separators=(",", ":")))
    else:
        print(json.dumps(summarize(args.campaign, read_json(args.stage_readout)), indent=2))


if __name__ == "__main__":
    main()
