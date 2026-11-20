#!/usr/bin/env python3
"""Inventory sealed U3 artifacts and replay one saved terminal decision per cohort."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time


CAMPAIGNS = ("v2_or_u3_confirm_v1", "v2_or_u3_messy_archive_v1", "v2_or_u3_live_sumo_v1")
FROZEN_CODE = "5965482dba305513d62307c362558e304c2c5943930648738f5cc71f6e10b89a"
INCUMBENT_ROLES = {"empirical_feasible_incumbent", "safe_initial_incumbent"}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_bindings(root: Path, campaign: str) -> tuple[Path, dict]:
    output = root / "paper_artifacts" / campaign
    summary = read_json(output / "compact_summary.json")
    manifest = read_json(root / "v2_or" / "manifests" / f"{campaign}.json")
    require(manifest.pop("sealed_complete") is True, "Campaign is not sealed")
    require(manifest.pop("sealed_execution_code_sha256") == FROZEN_CODE, "Wrong sealed runtime")
    manifest.pop("sealed_utc")
    preseal = (json.dumps(manifest, indent=2) + "\n").encode()
    require(hashlib.sha256(preseal).hexdigest() == summary["manifest_sha256"], "Wrong historical manifest")
    require(summary["execution_code_sha256"] == FROZEN_CODE, "Wrong summary runtime")
    return output, summary


def count_json_paths(path: Path) -> int:
    with os.scandir(path) as entries:
        return sum(entry.name.endswith(".json") and entry.is_file() for entry in entries)


def inventory(output: Path, summary: dict) -> dict:
    stages = output / "stages"
    records = {}
    for stage in ("source_archive", "search_and_verify", "audit_and_summarize"):
        directory = stages / stage
        completion = read_json(directory / "completion.json")
        require(completion["status"] == "complete", f"Incomplete phase: {stage}")
        require(completion["execution_code_sha256"] == FROZEN_CODE, f"Phase runtime mismatch: {stage}")
        require(completion["manifest_sha256"] == summary["manifest_sha256"], f"Phase manifest mismatch: {stage}")
        found = count_json_paths(directory / "items")
        require(found == completion["n_jobs"], f"Item count mismatch: {stage}")
        records[stage] = found
    require(records["search_and_verify"] == records["audit_and_summarize"] == summary["total_evaluated_cells"],
            "Target stage count disagrees with sealed ledger")

    archives = {}
    for family, row in summary["prior_diagnostics"].items():
        directory = output / "source_bundles" / family
        files = [directory / "archive.json", *(directory / f"prior_{name}.pkl" for name in row["variants"])]
        archives[family] = {
            "source_calls": row["source_simulation_calls"],
            "source_records": row["n_logged_evaluations"],
            "file_bytes": {path.name: path.stat().st_size for path in files},
        }
    return {"manifest_sha256": summary["manifest_sha256"], "items": records, "source_bundles": archives}


def binomial_lower_tail(count: int, replications: int) -> float:
    return math.fsum(math.comb(replications, j) * 0.05 ** j * 0.95 ** (replications - j)
                     for j in range(count + 1))


def replay(output: Path, summary: dict, job: str) -> dict:
    require(Path(job).name == job and job.startswith("sv_"), "Use a sealed search job ID")
    stages = output / "stages"
    row = read_json(stages / "search_and_verify" / "items" / f"{job}.json")
    audited = read_json(stages / "audit_and_summarize" / "items" / f"aud_{job}.json")
    for item in (row, audited):
        require(item["job_id"] == job, "Wrong saved job ID")
        require(item["execution_code_sha256"] == FROZEN_CODE, "Wrong saved runtime")
        require(item["manifest_sha256"] == summary["manifest_sha256"], "Wrong saved manifest")
    require(row["truth_used"] is False and audited["truth_used"] is True, "Wrong stage identity")

    roles, policies = row["shortlist_roles"], row["shortlist_policies"]
    budgets, deltas = row["slot_budgets_configured"], row["slot_deltas_configured"]
    certs = row["slot_certificates"]
    require(1 <= len(certs) <= len(roles) == len(policies) == len(budgets) == len(deltas) <= 3,
            "Inconsistent saved shortlist or tests")
    require(row["primary_policy"] == policies[0], "Primary policy differs from the frozen shortlist")
    passing = []
    for slot, cert in enumerate(certs):
        require(cert["slot_index"] == slot and cert["role"] == roles[slot], "Wrong slot tested")
        require(cert["method"] == "exact_binomial", "Saved gate is outside U3 binomial replay")
        n, count, delta = cert["n_samples"], cert["observed_violations"], cert["delta_allocated"]
        require(n == budgets[slot] and delta == deltas[slot] and 0 <= count <= n, "Wrong verification budget")
        probability = binomial_lower_tail(count, n)
        require(math.isclose(probability, cert["p_value_or_margin"], rel_tol=1e-12), "Saved gate value differs")
        require((probability <= delta) == cert["certified"] == (count <= cert["critical_k"]),
                "Saved gate decision differs")
        if cert["certified"]:
            passing.append(slot)

    guard = len(roles) >= 2 and roles[1] in INCUMBENT_ROLES
    both_pass = guard and 0 in passing and 1 in passing
    if both_pass:
        selected = min((0, 1), key=lambda slot: certs[slot]["sample_mean_obj"])
    else:
        selected = passing[0] if passing else None
    expected_tests = len(roles) if selected is None else max(selected + 1, 2 if guard else 1)
    require(len(certs) == expected_tests, "Gate stop differs")
    require(row["selected_slot"] == audited["selected_slot"] == selected, "Selected slot differs")
    require(row["certified"] == audited["certified"] == (selected is not None), "Decision differs")
    selected_policy = None if selected is None else policies[selected]
    require(row["deployed_policy"] == selected_policy, "Deployed vector differs")
    verification_calls = sum(cert["n_samples"] for cert in certs)
    require(row["verification_calls_used"] == audited["verification_calls_used"] == verification_calls,
            "Verification call count differs")
    require(row["search_calls_used"] + verification_calls == row["total_target_calls"], "Target call count differs")
    audit_deployed = audited.get("deployed_audit")
    require((audit_deployed is not None) == (selected is not None), "Audit deployment differs")
    cost = (audit_deployed or audited["anchor_audit"])["mc_objective_mean"]
    require(cost == audited["status_quo_loss"], "Saved status-quo endpoint differs")
    return {
        "job_id": job,
        "policy_dim": len(policies[0]),
        "tested_slots": len(certs),
        "passing_slots": passing,
        "incumbent_guard": guard,
        "selected_slot": selected,
        "verification_calls": verification_calls,
        "total_target_calls": row["total_target_calls"],
        "audit_calls_used": audited["audit_calls_used"],
        "safety_audit_class": audited["safety_audit_class"],
        "status_quo_loss": cost,
    }


def aggregate(root: Path, output: Path, summary: dict) -> dict:
    """Rebuild the sealed ledgers and statistical summary in temporary storage."""
    sys.path.insert(0, str(root))
    from v2_or.dispatch import campaign_runner

    start = time.process_time()
    require(campaign_runner.compute_v2_or_code_sha256() == FROZEN_CODE, "Aggregation needs the frozen U3 runtime")
    manifest = read_json(root / "v2_or" / "manifests" / f"{summary['campaign_name']}.json")
    audit = output / "stages" / "audit_and_summarize"
    descriptor = read_json(audit / "jobs.json")
    require(descriptor["manifest_sha256"] == summary["manifest_sha256"], "Wrong audit descriptor")
    require(descriptor["execution_code_sha256"] == FROZEN_CODE, "Wrong audit descriptor runtime")
    rows = [read_json(audit / "items" / f"{job['job_id']}.json") for job in descriptor["jobs"]]
    require(len(rows) == summary["total_evaluated_cells"], "Wrong number of saved audit receipts")
    with TemporaryDirectory(prefix="gp-fta-u3-aggregate-") as scratch:
        scratch_path = Path(scratch)
        rebuilt_hashes = campaign_runner._export_ledgers(scratch_path, rows)
        for field, value in rebuilt_hashes.items():
            require(value == summary[field], f"Rebuilt ledger fingerprint differs: {field}")
        (scratch_path / "stages").mkdir()
        (scratch_path / "stages" / "source_archive").symlink_to(output / "stages" / "source_archive")
        rebuilt = campaign_runner._compile_statistical_summary(manifest, descriptor, scratch_path, rows, rebuilt_hashes)
    fields = sorted(key for key in set(rebuilt) | set(summary) if key != "generated_utc")
    changed = [key for key in fields if json.dumps(rebuilt.get(key), sort_keys=True) !=
               json.dumps(summary.get(key), sort_keys=True)]
    require(not changed, f"Rebuilt summary fields differ: {changed}")
    return {"receipt_rows": len(rows), "ledger_fingerprints_match": True, "summary_fields_match": len(fields),
            "aggregation_cpu_seconds": time.process_time() - start}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sc-root", type=Path, required=True, help="Project SC-OLH-KG directory")
    parser.add_argument("--campaign", required=True, choices=CAMPAIGNS)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--job", action="append", default=[], help="Read one selected saved search job")
    operation.add_argument("--aggregate", action="store_true", help="Rebuild the ledgers from all saved audit receipts")
    args = parser.parse_args()
    output, summary = check_bindings(args.sc_root, args.campaign)
    if args.job:
        report = {"replays": [replay(output, summary, job) for job in args.job]}
    elif args.aggregate:
        report = aggregate(args.sc_root.resolve(), output.resolve(), summary)
    else:
        report = inventory(output, summary)
    print(json.dumps({"campaign": args.campaign, **report}, separators=(",", ":")))


if __name__ == "__main__":
    main()
