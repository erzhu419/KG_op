#!/usr/bin/env python3
"""Aggregate deterministic-atlas audits without overstating global coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_DOMAINS = {
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(paths, *, expected_domains=EXPECTED_DOMAINS):
    rows = []
    observed = set()
    for path in map(Path, paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        domain = str(payload["heldout_target_domain"])
        if domain in observed:
            raise ValueError(f"duplicate proposal audit for {domain}")
        observed.add(domain)
        geometric = payload["geometric_atlas_theorem_audit"]
        lipschitz = geometric["lipschitz_audit"]
        rows.append({
            "domain": domain,
            "audit_sha256": _sha256(path),
            "source_archive_fingerprint": str(
                payload["source_archive_fingerprint"]),
            "n0": int(payload["n0"]),
            "deterministic_atlas": bool(
                payload["unique_design_fingerprint_count"] == 1),
            "source_support_atlas_cover_radius": float(
                geometric["source_support_atlas_cover_radius"]),
            "source_support_shift": float(
                geometric["best_safe_center"]["source_support_shift"]),
            "finite_library_safe_radius": float(
                geometric["best_safe_center"][
                    "finite_library_safe_radius"]),
            "finite_library_coverage_slack": float(
                geometric["best_safe_center"]["coverage_slack"]),
            "finite_library_conditions_hold": bool(
                geometric["finite_library_theorem_conditions_hold"]),
            "observed_atlas_contains_feasible": bool(
                geometric["observed_atlas_contains_feasible"]),
            "global_lipschitz_upper_bound_certified": bool(
                lipschitz["global_lipschitz_upper_bound_certified"]),
            "global_lipschitz_condition_status": str(
                lipschitz["global_condition_status"]),
            "target_truth_used_post_run_only": bool(
                geometric["target_truth_used_post_run_only"]),
            "target_truth_used_for_proposal_or_selection": bool(
                geometric["target_truth_used_for_proposal_or_selection"]),
            "finite_sample_rank_conditions_hold": bool(
                payload["finite_sample_rank_theorem_audit"][
                    "theorem_conditions_hold"]),
        })
    expected_domains = set(map(str, expected_domains))
    if observed != expected_domains:
        raise ValueError(
            "proposal audit domains differ: "
            f"observed={sorted(observed)} expected={sorted(expected_domains)}"
        )
    finite_pass = all(
        row["deterministic_atlas"]
        and row["n0"] == 10
        and row["finite_library_conditions_hold"]
        and row["observed_atlas_contains_feasible"]
        and row["target_truth_used_post_run_only"]
        and not row["target_truth_used_for_proposal_or_selection"]
        for row in rows
    )
    if not finite_pass:
        raise ValueError("one or more finite-library atlas audits failed")
    global_count = sum(
        row["global_lipschitz_upper_bound_certified"] for row in rows)
    return {
        "schema_version": 1,
        "status": (
            "complete"
            if global_count == len(rows)
            else "complete_with_conditional_global_bound"
        ),
        "contract_id": "source_target_geometric_atlas_coverage_v1",
        "deployed_proposal_contract": "deterministic_finite_atlas",
        "domain_count": len(rows),
        "finite_library_condition_pass_count": len(rows),
        "observed_feasible_atlas_count": sum(
            row["observed_atlas_contains_feasible"] for row in rows),
        "global_lipschitz_certified_count": int(global_count),
        "global_theorem_claim_mode": (
            "empirically_instantiated"
            if global_count == len(rows)
            else "conditional_theorem_only"
        ),
        "unconditional_global_coverage_claim_allowed": bool(
            global_count == len(rows)),
        "rank_bridge_selected": False,
        "rank_bridge_rejection_reason": (
            "source-only finite-sample rank conditions were vacuous"),
        "rows": sorted(rows, key=lambda row: row["domain"]),
    }


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = aggregate(args.audit)
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "domain_count": payload["domain_count"],
        "global_theorem_claim_mode": payload[
            "global_theorem_claim_mode"],
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
