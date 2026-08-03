#!/usr/bin/env python3
"""Fail-closed audit of the frozen evidence without drafting a manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from finalize_paper_submission_release import (
        FINAL_CONTRACT_ID,
        FINAL_HEADLINE_TRACK,
        _convergence_contract_is_valid,
        _dimension_evidence_summary,
        _hvd_gate_summary,
        _hvd_release_role,
    )
except ModuleNotFoundError:
    from .finalize_paper_submission_release import (
        FINAL_CONTRACT_ID,
        FINAL_HEADLINE_TRACK,
        _convergence_contract_is_valid,
        _dimension_evidence_summary,
        _hvd_gate_summary,
        _hvd_release_role,
    )


CONTRACT_ID = "or_submission_readiness_without_manuscript_v1"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _check(condition, message, failures):
    if not condition:
        failures.append(str(message))


def build_readiness(
    *,
    method_contract_path,
    registry_path,
    audit_path,
    statistics_path,
    convergence_path,
    proposal_coverage_path,
    proof_receipt_path,
    hvd_causal_gate_path,
    dimension_evidence_path,
    external_disposition_path,
):
    paths = {
        "method_contract": Path(method_contract_path),
        "experiment_registry": Path(registry_path),
        "compact_result_audit": Path(audit_path),
        "paired_statistics": Path(statistics_path),
        "search_convergence": Path(convergence_path),
        "proposal_coverage": Path(proposal_coverage_path),
        "lean_proof_receipt": Path(proof_receipt_path),
        "hvd_causal_gate": Path(hvd_causal_gate_path),
        "dimension_evidence": Path(dimension_evidence_path),
        "external_disposition": Path(external_disposition_path),
    }
    payloads = {name: _read(path) for name, path in paths.items()}
    method = payloads["method_contract"]
    registry = payloads["experiment_registry"]
    audit = payloads["compact_result_audit"]
    statistics = payloads["paired_statistics"]
    convergence = payloads["search_convergence"]
    coverage = payloads["proposal_coverage"]
    proof = payloads["lean_proof_receipt"]
    hvd = payloads["hvd_causal_gate"]
    dimension = payloads["dimension_evidence"]
    external = payloads["external_disposition"]

    failures = []
    explicit_required_tracks = [
        row for row in audit.get("track_audits", ())
        if row.get("release_required") is True
    ]
    required_tracks = (
        explicit_required_tracks
        if explicit_required_tracks
        else list(audit.get("track_audits", ()))
    )
    explicit_required_comparisons = [
        row for row in statistics.get("comparison_audits", ())
        if row.get("release_required") is True
    ]
    required_comparisons = (
        explicit_required_comparisons
        if explicit_required_comparisons
        else list(statistics.get("comparison_audits", ()))
    )
    _check(
        method.get("contract_id") == FINAL_CONTRACT_ID,
        "final method contract is not frozen",
        failures,
    )
    _check(
        registry.get("registry_id") == audit.get("registry_id")
        == statistics.get("registry_id"),
        "registry, audit, and statistics identities differ",
        failures,
    )
    _check(
        audit.get("status") == "pass"
        and len(required_tracks) == int(audit.get(
            "release_required_track_count", len(required_tracks)))
        and bool(required_tracks)
        and all(row.get("status") == "pass" for row in required_tracks),
        "registered compact result audit is incomplete",
        failures,
    )
    _check(
        statistics.get("status") == "complete"
        and statistics.get("audit_status") == "pass"
        and int(statistics.get("bootstrap_samples", 0)) >= 10000
        and len(required_comparisons) == int(statistics.get(
            "release_required_comparison_count", len(required_comparisons)))
        and bool(required_comparisons)
        and all(row.get("status") == "pass"
                for row in required_comparisons),
        "paired statistics or multiplicity audit is incomplete",
        failures,
    )

    headline_records = [
        row for row in audit.get("records", ())
        if row.get("track_id") == FINAL_HEADLINE_TRACK
    ]
    expected_receipt = hashlib.sha256(json.dumps(
        sorted(str(row["result_sha256"]) for row in headline_records),
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    _check(
        _convergence_contract_is_valid(convergence)
        and convergence.get("status") == "complete"
        and convergence.get("track_id") == FINAL_HEADLINE_TRACK
        and int(convergence.get("result_count", -1))
        == int(convergence.get("completed_trace_count", -2))
        == len(headline_records) == 180
        and int(convergence.get("trace_row_count", -1))
        == int(convergence.get("expected_trace_row_count", -2)) == 2160
        and int(convergence.get(
            "terminal_validation_failure_count", -1)) == 0
        and convergence.get("result_receipts_sha256") == expected_receipt
        and convergence.get("source_audit_sha256") == _sha256(audit_path)
        and convergence.get("target_truth_used_post_run_only") is True
        and convergence.get(
            "target_truth_used_for_search_or_selection") is False
        and convergence.get("verification_samples_included") is False
        and convergence.get("policy_vectors_exported") is False,
        "headline convergence is incomplete or not receipt-bound",
        failures,
    )

    _check(
        proof.get("status") == "pass"
        and int(proof.get("forbidden_declaration_count", -1)) == 0
        and proof.get("build", {}).get("executed") is True
        and int(proof.get("build", {}).get("returncode", -1)) == 0,
        "Lean proof receipt is incomplete",
        failures,
    )
    _check(
        coverage.get("contract_id")
        == "source_target_geometric_atlas_coverage_v1"
        and coverage.get("status")
        == "complete_with_conditional_global_bound"
        and int(coverage.get("domain_count", 0)) == 3
        and int(coverage.get("finite_library_condition_pass_count", 0)) == 3
        and coverage.get("unconditional_global_coverage_claim_allowed")
        is False,
        "source-to-target coverage evidence is incomplete or overclaims",
        failures,
    )

    try:
        hvd_summary = _hvd_gate_summary(hvd)
        hvd_role = _hvd_release_role(hvd)
    except ValueError as error:
        failures.append(f"HVD causal decision is inconsistent: {error}")
        hvd_summary = {}
        hvd_role = "invalid"
    _check(
        int(hvd_summary.get("complete_pair_count", 0)) == 60
        and hvd_summary.get("promote_hvd_as_core") is False
        and hvd_summary.get("retain_optional") is True,
        "HVD has not been validly demoted after the complete causal gate",
        failures,
    )

    dimension_summary = _dimension_evidence_summary(dimension)
    _check(
        dimension_summary.get("complete") is True,
        "stratified dimension/budget evidence is incomplete",
        failures,
    )

    external_failed = bool(
        external.get("status") == "complete"
        and external.get("external_validity_status")
        == "failed_not_promoted"
        and external.get("submission_release_status")
        == "blocked_by_external_validity"
        and external.get(
            "confirmatory_external_traffic_evidence_available") is False
        and external.get("decision_contract", {}).get(
            "paper_release_must_fail_closed") is True
        and external.get("decision_contract", {}).get(
            "posthoc_outcomes_do_not_select_or_modify_the_method") is True
    )
    _check(
        external_failed,
        "external-validity disposition is missing or not fail-closed",
        failures,
    )
    _check(
        method.get("claim_boundaries", {}).get("external_traffic")
        and method.get("supporting_evidence", {}).get(
            "external_validity_contract", {}).get("status")
        == "failed_not_promoted",
        "method contract does not disclose the external-validity failure",
        failures,
    )

    external_blockers = ([{
        "id": "strict_no_history_sumo_external_validity",
        "status": "failed_not_promoted",
        "confirmatory_claim_allowed": False,
        "reason": (
            "The preregistered V2 development gate certified 0/5 seeds; "
            "a diagnostic frozen 111-policy library contained no policy "
            "with empirical feasibility probability at least 0.95."
        ),
    }] if external_failed else [])
    if failures:
        status = "blocked_by_internal_evidence"
    elif external_blockers:
        status = "blocked_by_external_validity"
    else:
        status = "evidence_complete"

    registry_sha256 = _sha256(registry_path)
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": status,
        "ready_for_manuscript_lock": status == "evidence_complete",
        "manuscript_generation_performed": False,
        "non_external_failure_count": len(failures),
        "non_external_failures": failures,
        "external_blockers": external_blockers,
        "registry_status_overlay": {
            "schema_version": 1,
            "contract_id": "immutable_registry_status_overlay_v1",
            "base_registry_id": registry.get("registry_id"),
            "base_registry_sha256": registry_sha256,
            "base_registry_modified": False,
            "status": status,
            "registered_track_count": len(registry.get("tracks", ())),
            "audited_record_count": int(audit.get("record_count", 0)),
        },
        "evidence": {
            "registered_tracks": len(registry.get("tracks", ())),
            "audited_records": int(audit.get("record_count", 0)),
            "paired_comparisons": len(
                statistics.get("comparison_audits", ())),
            "nonrequired_incomplete_tracks": sum(
                row.get("release_required") is not True
                and row.get("status") != "pass"
                for row in audit.get("track_audits", ())),
            "nonrequired_incomplete_comparisons": sum(
                row.get("release_required") is not True
                and row.get("status") != "pass"
                for row in statistics.get("comparison_audits", ())),
            "convergence_results": int(convergence.get("result_count", 0)),
            "convergence_search_calls": int(
                convergence.get("trace_row_count", 0)),
            "dimension_budget": dimension_summary,
            "proposal_coverage_status": coverage.get("status"),
            "hvd": {
                "paper_role": hvd_role,
                "promote_as_core": bool(
                    hvd_summary.get("promote_hvd_as_core")),
                "retain_optional": bool(hvd_summary.get("retain_optional")),
            },
            "lean": {
                "source_count": int(proof.get("lean_source_count", 0)),
                "forbidden_declaration_count": int(
                    proof.get("forbidden_declaration_count", -1)),
                "lake_build_returncode": int(
                    proof.get("build", {}).get("returncode", -1)),
            },
        },
        "artifact_sha256": {
            name: _sha256(path) for name, path in paths.items()
        },
        "information_contract": {
            "runtime_checkpoints_or_model_weights_read": False,
            "policy_vectors_exported": False,
            "target_oracle_used_for_selection": False,
            "posthoc_external_outcomes_used_to_change_method": False,
            "saas_or_gpu_jobs_launched_by_audit": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-contract", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--convergence", required=True)
    parser.add_argument("--proposal-coverage", required=True)
    parser.add_argument("--proof-receipt", required=True)
    parser.add_argument("--hvd-causal-gate", required=True)
    parser.add_argument("--dimension-evidence", required=True)
    parser.add_argument("--external-disposition", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-registry-overlay", required=True)
    args = parser.parse_args()
    readiness = build_readiness(
        method_contract_path=args.method_contract,
        registry_path=args.registry,
        audit_path=args.audit,
        statistics_path=args.statistics,
        convergence_path=args.convergence,
        proposal_coverage_path=args.proposal_coverage,
        proof_receipt_path=args.proof_receipt,
        hvd_causal_gate_path=args.hvd_causal_gate,
        dimension_evidence_path=args.dimension_evidence,
        external_disposition_path=args.external_disposition,
    )
    _atomic_json(args.out, readiness)
    _atomic_json(args.out_registry_overlay, readiness["registry_status_overlay"])
    print(json.dumps({
        "status": readiness["status"],
        "non_external_failure_count": readiness[
            "non_external_failure_count"],
        "external_blocker_count": len(readiness["external_blockers"]),
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
