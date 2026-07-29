#!/usr/bin/env python3
"""Freeze one hash-addressed, fail-closed paper evidence release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


FINAL_CONTRACT_ID = "or_transfer_frontend_saas_v1"
UNIFORM_VERIFIER_CONTRACT_ID = "uniform_two_policy_external_verifier_v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(condition, message, failures):
    if not condition:
        failures.append(str(message))


def validate_release_inputs(
    method_contract,
    registry,
    audit,
    statistics,
    convergence,
    traffic,
    proposal_coverage,
    proof_receipt,
):
    failures = []
    _require(
        method_contract.get("contract_id") == FINAL_CONTRACT_ID,
        "final method identity is not frozen to the registered contract",
        failures,
    )
    _require(
        method_contract.get("online_backend", {}).get("refit_schedule")
        == "every_iteration",
        "headline backend is not canonical every-iteration SAASBO",
        failures,
    )
    _require(
        audit.get("registry_id") == registry.get("registry_id"),
        "audit and experiment registry identities differ",
        failures,
    )
    _require(
        audit.get("status") == "pass",
        "one or more registered experiment tracks are incomplete or failed",
        failures,
    )
    _require(
        all(row.get("status") == "pass"
            for row in audit.get("track_audits", ())),
        "one or more track audits failed",
        failures,
    )
    _require(
        statistics.get("status") == "complete",
        "paired statistics are incomplete",
        failures,
    )
    _require(
        all(row.get("status") == "pass"
            for row in statistics.get("comparison_audits", ())),
        "one or more preregistered comparisons are incomplete",
        failures,
    )
    final_track = next((
        track for track in registry.get("tracks", ())
        if track.get("track_id")
        == "final_frontend_backend_factorial_d1000_n13"
    ), {})
    _require(
        bool(final_track),
        "experiment registry is missing the headline factorial track",
        failures,
    )
    _require(
        convergence.get("contract_id")
        == "post_run_search_convergence_v1"
        and convergence.get("status") == "complete",
        "final search convergence artifact is incomplete",
        failures,
    )
    _require(
        convergence.get("track_id")
        == "final_frontend_backend_factorial_d1000_n13",
        "convergence artifact does not describe the headline track",
        failures,
    )
    _require(
        set(convergence.get("method_identities", ()))
        == set(final_track.get("expected_method_identities", ())),
        "convergence artifact does not contain every headline method",
        failures,
    )
    final_records = [
        row for row in audit.get("records", ())
        if row.get("track_id")
        == "final_frontend_backend_factorial_d1000_n13"
    ]
    expected_final_results = (
        len(final_track.get("expected_method_identities", ()))
        * len(final_track.get("expected_domains", ()))
        * len(final_track.get("expected_dimensions", (1,)))
        * len(final_track.get("expected_seeds", ()))
    )
    _require(
        len(final_records) == expected_final_results
        and expected_final_results > 0,
        "compact audit does not contain the complete headline factorial",
        failures,
    )
    _require(
        int(convergence.get("result_count", 0)) == len(final_records)
        and int(convergence.get("completed_trace_count", 0))
        == len(final_records),
        "convergence artifact does not contain every headline result",
        failures,
    )
    _require(
        int(convergence.get("trace_row_count", -1))
        == int(convergence.get("expected_trace_row_count", -2)),
        "convergence artifact does not cover every target search call",
        failures,
    )
    _require(
        int(convergence.get(
            "terminal_validation_failure_count", -1)) == 0,
        "convergence reconstruction failed a terminal truth check",
        failures,
    )
    _require(
        convergence.get("target_truth_used_post_run_only") is True
        and convergence.get(
            "target_truth_used_for_search_or_selection") is False
        and convergence.get("verification_samples_included") is False
        and convergence.get("policy_vectors_exported") is False,
        "convergence artifact violates the post-run compact-data contract",
        failures,
    )
    expected_result_receipt = hashlib.sha256(json.dumps(
        sorted(str(row["result_sha256"]) for row in final_records),
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    _require(
        convergence.get("result_receipts_sha256")
        == expected_result_receipt,
        "convergence result receipts differ from the compact audit",
        failures,
    )
    _require(
        proof_receipt.get("status") == "pass",
        "Lean proof receipt did not pass",
        failures,
    )
    _require(
        proof_receipt.get("forbidden_declaration_count") == 0,
        "Lean proof tree contains sorry, admit, or axiom",
        failures,
    )
    _require(
        proof_receipt.get("build", {}).get("executed") is True
        and proof_receipt.get("build", {}).get("returncode") == 0,
        "Lean proof receipt does not contain a successful lake build",
        failures,
    )
    _require(
        traffic.get("status") == "complete",
        "strict no-history traffic audit is incomplete",
        failures,
    )
    _require(
        int(traffic.get("n_seeds", 0)) >= 20,
        "strict no-history traffic audit has fewer than 20 search seeds",
        failures,
    )
    _require(
        int(traffic.get("source_calls_per_run", -1)) == 384
        and int(traffic.get("target_search_calls_per_run", -1)) == 13,
        "traffic source/search budgets differ from the registered contract",
        failures,
    )
    traffic_information = traffic.get("information_contract", {})
    _require(
        traffic_information.get("track")
        == "domain_blind_external_holdout"
        and traffic_information.get(
            "heldout_task_family_identifier_used_by_proposal") is False
        and traffic_information.get(
            "target_labels_used_to_fit_proposal") is False
        and traffic_information.get("target_oracle_used") is False
        and traffic_information.get(
            "excluded_nearest_source_analogue")
        == "QueueResourceControl",
        "traffic experiment is not the registered domain-blind holdout",
        failures,
    )
    _require(
        proposal_coverage.get("contract_id")
        == "source_target_geometric_atlas_coverage_v1",
        "proposal coverage audit uses the wrong theory contract",
        failures,
    )
    _require(
        proposal_coverage.get("status") in {
            "complete",
            "complete_with_conditional_global_bound",
        },
        "proposal coverage audit is incomplete",
        failures,
    )
    proposal_rows = list(proposal_coverage.get("rows", ()))
    _require(
        len(proposal_rows) == 3
        and int(proposal_coverage.get(
            "finite_library_condition_pass_count", 0)) == 3,
        "proposal coverage audit lacks three passing finite-library domains",
        failures,
    )
    _require(
        bool(proposal_rows)
        and all(
            row.get("deterministic_atlas") is True
            and row.get("target_truth_used_post_run_only") is True
            and row.get(
                "target_truth_used_for_proposal_or_selection") is False
            for row in proposal_rows
        ),
        "proposal coverage audit violates the frozen post-run truth contract",
        failures,
    )
    if proposal_coverage.get(
        "unconditional_global_coverage_claim_allowed"
    ) is not True:
        _require(
            proposal_coverage.get("global_theorem_claim_mode")
            == "conditional_theorem_only",
            "uncertified global proposal coverage is not labelled conditional",
            failures,
        )
    traffic_rows = list(traffic.get("rows", ()))
    _require(
        bool(traffic_rows)
        and all(
            row.get("certificate")
            == "one_sided_clopper_pearson_bonferroni"
            and row.get("fixed_shortlist_order") is True
            and row.get("verification_samples_update_optimizer") is False
            and row.get(
                "verification_samples_used_to_reorder_shortlist") is False
            for row in traffic_rows
        ),
        "traffic verifier is not a fixed fresh-seed external certificate",
        failures,
    )
    uniform_rows = [
        row for row in audit.get("records", ())
        if row.get("track_id")
        == "uniform_external_total_cost_d1000_n397"
    ]
    _require(
        len(uniform_rows) == 240,
        "uniform total-cost verifier does not contain 240 results",
        failures,
    )
    _require(
        all(
            row.get("optimization_calls_excluding_verification") == 397
            for row in uniform_rows
        ),
        "uniform total-cost comparison violates the 397-call budget",
        failures,
    )
    _require(
        len({
            row.get("verifier_signature") for row in uniform_rows
        }) == 1,
        "uniform total-cost comparison uses different verifiers",
        failures,
    )
    for record in audit.get("records", ()):
        path = Path(str(record.get("path", "")))
        if not path.is_file():
            failures.append(f"audited result is missing: {path}")
            continue
        if _sha256(path) != record.get("result_sha256"):
            failures.append(f"audited result changed after audit: {path}")
    return failures


def _record_receipts(audit):
    rows = []
    for record in audit.get("records", ()):
        rows.append({
            "track_id": record["track_id"],
            "method_identity": record["method_identity"],
            "domain": record["domain"],
            "target_dimension": record["target_dimension"],
            "seed": record["seed"],
            "source_calls": record["source_calls"],
            "target_search_calls": record["target_search_calls"],
            "target_verification_calls": record[
                "target_verification_calls"],
            "optimization_calls_excluding_verification": record[
                "optimization_calls_excluding_verification"],
            "result_sha256": record["result_sha256"],
        })
    return sorted(rows, key=lambda row: (
        row["track_id"],
        row["method_identity"],
        row["domain"],
        -1 if row["target_dimension"] is None else row["target_dimension"],
        -1 if row["seed"] is None else row["seed"],
    ))


def build_release(
    *,
    method_contract_path,
    registry_path,
    audit_path,
    statistics_path,
    convergence_path,
    traffic_path,
    proposal_coverage_path,
    proof_receipt_path,
    repository_commit,
):
    method_contract = _load(method_contract_path)
    registry = _load(registry_path)
    audit = _load(audit_path)
    statistics = _load(statistics_path)
    convergence = _load(convergence_path)
    traffic = _load(traffic_path)
    proposal_coverage = _load(proposal_coverage_path)
    proof_receipt = _load(proof_receipt_path)
    failures = validate_release_inputs(
        method_contract,
        registry,
        audit,
        statistics,
        convergence,
        traffic,
        proposal_coverage,
        proof_receipt,
    )
    if failures:
        raise ValueError(
            "paper release validation failed:\n- " + "\n- ".join(failures))
    paths = {
        "method_contract": Path(method_contract_path),
        "experiment_registry": Path(registry_path),
        "compact_audit": Path(audit_path),
        "paired_statistics": Path(statistics_path),
        "search_convergence": Path(convergence_path),
        "external_traffic_audit": Path(traffic_path),
        "proposal_coverage_audit": Path(proposal_coverage_path),
        "lean_proof_receipt": Path(proof_receipt_path),
    }
    records = _record_receipts(audit)
    record_digest = hashlib.sha256(
        json.dumps(
            records, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 2,
        "status": "ready_for_manuscript_lock",
        "release_contract_id": "or_submission_evidence_release_v2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repository_commit": str(repository_commit),
        "headline_method_contract_id": FINAL_CONTRACT_ID,
        "headline_method": (
            "source-learned dimension-equivariant structural proposal + "
            "canonical SAASBO backend + independent verifier"
        ),
        "claim_boundaries": method_contract["claim_boundaries"],
        "registry_id": registry["registry_id"],
        "registered_track_count": len(registry["tracks"]),
        "registered_comparison_count": len(
            registry.get("primary_comparisons", ())),
        "audited_result_count": len(records),
        "audited_result_receipt_sha256": record_digest,
        "audited_result_receipts": records,
        "failed_or_timeout_result_count": int(sum(
            row.get("status") != "ok"
            for row in audit.get("records", ()))),
        "false_certificate_count": int(sum(
            bool(row.get("false_certificate"))
            for row in audit.get("records", ()))),
        "external_traffic": {
            "n_seeds": int(traffic["n_seeds"]),
            "certified_seed_count": int(
                traffic["certified_seed_count"]),
            "source_calls_per_run": int(
                traffic["source_calls_per_run"]),
            "target_search_calls_per_run": int(
                traffic["target_search_calls_per_run"]),
            "target_verification_calls_per_run": int(
                traffic["target_verification_calls_per_run"]),
            "information_contract": traffic["information_contract"],
        },
        "proposal_coverage": {
            "contract_id": proposal_coverage["contract_id"],
            "domain_count": int(proposal_coverage["domain_count"]),
            "finite_library_condition_pass_count": int(
                proposal_coverage[
                    "finite_library_condition_pass_count"]),
            "global_lipschitz_certified_count": int(
                proposal_coverage[
                    "global_lipschitz_certified_count"]),
            "global_theorem_claim_mode": proposal_coverage[
                "global_theorem_claim_mode"],
            "unconditional_global_coverage_claim_allowed": bool(
                proposal_coverage[
                    "unconditional_global_coverage_claim_allowed"]),
        },
        "search_convergence": {
            "contract_id": convergence["contract_id"],
            "track_id": convergence["track_id"],
            "result_count": int(convergence["result_count"]),
            "trace_row_count": int(convergence["trace_row_count"]),
            "target_truth_used_post_run_only": True,
            "verification_samples_included": False,
            "result_receipts_sha256": convergence[
                "result_receipts_sha256"],
        },
        "proof": {
            "lean_source_count": int(
                proof_receipt["lean_source_count"]),
            "lean_source_tree_sha256": proof_receipt[
                "lean_source_tree_sha256"],
            "lake_build_returncode": int(
                proof_receipt["build"]["returncode"]),
            "forbidden_declaration_count": int(
                proof_receipt["forbidden_declaration_count"]),
        },
        "artifact_sha256": {
            name: _sha256(path) for name, path in paths.items()
        },
        "statistics_holm_family": statistics["holm_family"],
        "statistics_rows": statistics["rows"],
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
    parser.add_argument("--method-contract", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--convergence", required=True)
    parser.add_argument("--traffic", required=True)
    parser.add_argument("--proposal-coverage", required=True)
    parser.add_argument("--proof-receipt", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=args.repository_root,
        text=True,
    ).strip()
    release = build_release(
        method_contract_path=args.method_contract,
        registry_path=args.registry,
        audit_path=args.audit,
        statistics_path=args.statistics,
        convergence_path=args.convergence,
        traffic_path=args.traffic,
        proposal_coverage_path=args.proposal_coverage,
        proof_receipt_path=args.proof_receipt,
        repository_commit=commit,
    )
    _atomic_json(args.out, release)
    print(json.dumps({
        "status": release["status"],
        "repository_commit": release["repository_commit"],
        "audited_result_count": release["audited_result_count"],
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
