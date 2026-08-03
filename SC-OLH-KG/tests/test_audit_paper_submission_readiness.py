import hashlib
import json

from performance.audit_paper_submission_readiness import build_readiness


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture(tmp_path):
    result_shas = [f"{index:064x}" for index in range(180)]
    records = [{
        "track_id": "final_frozen_source_frontend_backend_d1000_n13",
        "result_sha256": value,
    } for value in result_shas]
    audit = _write(tmp_path / "audit.json", {
        "status": "pass",
        "registry_id": "paper_submission_experiment_registry_v1",
        "record_count": 180,
        "records": records,
        "track_audits": [{"status": "pass"}],
    })
    audit_sha = hashlib.sha256(audit.read_bytes()).hexdigest()
    receipt = hashlib.sha256(json.dumps(
        sorted(result_shas), separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    paths = {
        "method_contract_path": _write(tmp_path / "method.json", {
            "contract_id": "or_transfer_frontend_saas_v1",
            "claim_boundaries": {"external_traffic": "failed"},
            "supporting_evidence": {
                "external_validity_contract": {
                    "status": "failed_not_promoted",
                },
            },
        }),
        "registry_path": _write(tmp_path / "registry.json", {
            "registry_id": "paper_submission_experiment_registry_v1",
            "tracks": [{"track_id": "track"}],
        }),
        "audit_path": audit,
        "statistics_path": _write(tmp_path / "stats.json", {
            "status": "complete",
            "audit_status": "pass",
            "registry_id": "paper_submission_experiment_registry_v1",
            "bootstrap_samples": 10000,
            "comparison_audits": [{"status": "pass"}],
        }),
        "convergence_path": _write(tmp_path / "convergence.json", {
            "contract_id": "post_run_search_convergence_distributed_v1",
            "source_contract_id": "post_run_search_convergence_v1",
            "status": "complete",
            "track_id": "final_frozen_source_frontend_backend_d1000_n13",
            "result_count": 180,
            "completed_trace_count": 180,
            "trace_row_count": 2160,
            "expected_trace_row_count": 2160,
            "terminal_validation_failure_count": 0,
            "result_receipts_sha256": receipt,
            "source_audit_sha256": audit_sha,
            "target_truth_used_post_run_only": True,
            "target_truth_used_for_search_or_selection": False,
            "verification_samples_included": False,
            "policy_vectors_exported": False,
        }),
        "proposal_coverage_path": _write(tmp_path / "coverage.json", {
            "contract_id": "source_target_geometric_atlas_coverage_v1",
            "status": "complete_with_conditional_global_bound",
            "domain_count": 3,
            "finite_library_condition_pass_count": 3,
            "unconditional_global_coverage_claim_allowed": False,
        }),
        "proof_receipt_path": _write(tmp_path / "proof.json", {
            "status": "pass",
            "lean_source_count": 95,
            "forbidden_declaration_count": 0,
            "build": {"executed": True, "returncode": 0},
        }),
        "hvd_causal_gate_path": _write(tmp_path / "hvd.json", {
            "status": "complete",
            "paired_cells": 60,
            "target_seeds": list(range(20)),
            "paired_contract": {
                "same_frozen_proposal": True,
                "same_source_archive": True,
                "same_independent_terminal_verifier": True,
                "only_changed_object": "aleatoric_variance_head",
                "saas_used": False,
                "gpu_used": False,
            },
            "domain_summaries": {
                domain: {
                    "pooled": {
                        "median_log_variance_rmse": 2.0,
                        "median_variance_shape_correlation": 0.0,
                        "true_feasible": "20/20",
                        "median_feasible_regret": 0.0,
                        "false_certification_count": 0,
                        "mean_verification_calls": 10.0,
                    },
                    "provider_cumulative_factor": {
                        "median_log_variance_rmse": 1.0,
                        "median_variance_shape_correlation": 0.9,
                        "true_feasible": "20/20",
                        "median_feasible_regret": 0.0,
                        "false_certification_count": 0,
                        "mean_verification_calls": 11.0,
                    },
                }
                for domain in ("FactorShock", "Inventory", "Queue")
            },
            "decision": {
                "variance_calibration_and_shape_recovered_in_all_domains": True,
                "verification_cost_noninferior_in_all_domains": False,
                "promote_hvd_as_core_contribution": False,
                "retain_as_optional_risk_diagnostic": True,
            },
        }),
        "dimension_evidence_path": _write(tmp_path / "dimension.json", {
            "contract_id": "stratified_final_dimension_budget_evidence_v1",
            "status": "complete",
            "headline_dimensions": [1000, 10000],
            "headline_seed_counts": {"1000": 20, "10000": 10},
            "release_cell_count": 6,
            "failures": [], "missing_keys": [], "unexpected_keys": [],
            "duplicate_keys": [],
            "all_release_rows_ok": True,
            "all_release_rows_false_certificate_free": True,
            "all_release_rows_frozen": True,
            "cells": {
                f"cell{index}": {
                    "dimension": 1000 if index < 2 else 10000,
                    "all_rows_ok": True,
                    "false_certificate_count": 0,
                    "frozen_execution_provenance_pass": True,
                } for index in range(6)
            },
        }),
        "external_disposition_path": _write(tmp_path / "external.json", {
            "status": "complete",
            "external_validity_status": "failed_not_promoted",
            "submission_release_status": "blocked_by_external_validity",
            "confirmatory_external_traffic_evidence_available": False,
            "decision_contract": {
                "paper_release_must_fail_closed": True,
                "posthoc_outcomes_do_not_select_or_modify_the_method": True,
            },
        }),
    }
    return paths


def test_readiness_closes_internal_evidence_but_fails_external(tmp_path):
    readiness = build_readiness(**_fixture(tmp_path))
    assert readiness["status"] == "blocked_by_external_validity"
    assert readiness["non_external_failure_count"] == 0
    assert len(readiness["external_blockers"]) == 1
    assert readiness["ready_for_manuscript_lock"] is False
    assert readiness["registry_status_overlay"]["base_registry_modified"] is False


def test_readiness_fails_closed_on_receipt_drift(tmp_path):
    paths = _fixture(tmp_path)
    convergence = json.loads(paths["convergence_path"].read_text())
    convergence["result_receipts_sha256"] = "0" * 64
    paths["convergence_path"].write_text(json.dumps(convergence))
    readiness = build_readiness(**paths)
    assert readiness["status"] == "blocked_by_internal_evidence"
    assert readiness["non_external_failure_count"] == 1


def test_confirmed_external_energy_closes_external_blocker(tmp_path):
    paths = _fixture(tmp_path)
    method = json.loads(paths["method_contract_path"].read_text())
    method["claim_boundaries"]["external_energy"] = "confirmed"
    method["supporting_evidence"]["external_energy_validity_contract"] = {
        "status": "passed_confirmatory",
    }
    paths["method_contract_path"].write_text(json.dumps(method))
    energy = {
        "status": "complete_confirmatory_external_energy_pass",
        "external_validity_status": "passed_confirmatory",
        "submission_release_status": "evidence_complete",
        "confirmatory_external_energy_evidence_available": True,
        "confirmatory_result": {
            "status": "pass",
            "frozen_independently_certified": 20,
            "frozen_false_certificates": 0,
            "paired_frozen_wins": 20,
            "method_repair_after_target_opened": False,
        },
        "post_gate_no_regression_result": {
            "status": "pass",
            "domain_count": 3,
            "identical_domain_seed_designs": 60,
            "target_simulator_calls_used": 0,
        },
        "decision_contract": {
            "confirmatory_target_frozen_before_outcomes": True,
            "posthoc_outcomes_do_not_select_or_modify_the_method": True,
        },
    }
    paths["external_disposition_path"].write_text(json.dumps(energy))
    readiness = build_readiness(**paths)
    assert readiness["status"] == "evidence_complete"
    assert readiness["external_blockers"] == []
    assert readiness["ready_for_manuscript_lock"] is True
    assert readiness["evidence"]["external_validity"]["status"] == (
        "passed_confirmatory_energy")
