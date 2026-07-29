import json
from pathlib import Path

import pytest

from performance.finalize_paper_submission_release import build_release


def _write(path, payload):
    path = Path(path)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixtures(tmp_path):
    result = _write(tmp_path / "result.json", {"status": "ok"})
    method = _write(tmp_path / "method.json", {
        "contract_id": "or_transfer_frontend_saas_v1",
        "online_backend": {"refit_schedule": "every_iteration"},
        "claim_boundaries": {"kg": "ablation"},
    })
    registry = _write(tmp_path / "registry.json", {
        "registry_id": "registry",
        "tracks": [{
            "track_id": "final_frontend_backend_factorial_d1000_n13",
            "expected_method_identities": ["headline"],
            "expected_domains": ["Domain"],
            "expected_dimensions": [1000],
            "expected_seeds": [80],
        }],
        "primary_comparisons": [{}],
    })
    import hashlib
    result_hash = hashlib.sha256(result.read_bytes()).hexdigest()
    records = []
    for method_identity in (
        "uniform_verified::canonical_saasbo_every_iteration",
        "uniform_verified::botorch_turbo:canonical_turbo1_ts",
        "uniform_verified::botorch_scbo:canonical_scbo_constrained_ts",
        "uniform_verified::saasbo_periodic_capped",
    ):
        for seed in range(60):
            records.append({
                "track_id": "uniform_external_total_cost_d1000_n397",
                "path": str(result),
                "result_sha256": result_hash,
                "status": "ok",
                "method_identity": method_identity,
                "domain": "Domain",
                "target_dimension": 1000,
                "seed": seed,
                "source_calls": 384 if "canonical_saasbo" in method_identity else 0,
                "target_search_calls": 13 if "canonical_saasbo" in method_identity else 397,
                "target_verification_calls": 256,
                "optimization_calls_excluding_verification": 397,
                "verifier_signature": "uniform",
                "false_certificate": False,
            })
    records.append({
        "track_id": "final_frontend_backend_factorial_d1000_n13",
        "path": str(result),
        "result_sha256": result_hash,
        "status": "ok",
        "method_identity": "headline",
        "domain": "Domain",
        "target_dimension": 1000,
        "seed": 80,
        "source_calls": 384,
        "target_search_calls": 13,
        "target_verification_calls": 256,
        "optimization_calls_excluding_verification": 397,
        "verifier_signature": "uniform",
        "false_certificate": False,
    })
    audit = _write(tmp_path / "audit.json", {
        "registry_id": "registry",
        "status": "pass",
        "track_audits": [{"status": "pass"}],
        "records": records,
    })
    statistics = _write(tmp_path / "statistics.json", {
        "status": "complete",
        "comparison_audits": [{"status": "pass"}],
        "holm_family": "family",
        "rows": [],
    })
    convergence_receipt = hashlib.sha256(json.dumps(
        sorted(
            record["result_sha256"] for record in records
            if record["track_id"]
            == "final_frontend_backend_factorial_d1000_n13"
        ),
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    convergence = _write(tmp_path / "convergence.json", {
        "contract_id": "post_run_search_convergence_v1",
        "status": "complete",
        "track_id": "final_frontend_backend_factorial_d1000_n13",
        "method_identities": ["headline"],
        "result_count": 1,
        "completed_trace_count": 1,
        "trace_row_count": 13,
        "expected_trace_row_count": 13,
        "terminal_validation_failure_count": 0,
        "target_truth_used_post_run_only": True,
        "target_truth_used_for_search_or_selection": False,
        "verification_samples_included": False,
        "policy_vectors_exported": False,
        "result_receipts_sha256": convergence_receipt,
    })
    traffic_rows = [{
        "certificate": "one_sided_clopper_pearson_bonferroni",
        "fixed_shortlist_order": True,
        "verification_samples_update_optimizer": False,
        "verification_samples_used_to_reorder_shortlist": False,
    } for _ in range(20)]
    traffic = _write(tmp_path / "traffic.json", {
        "status": "complete",
        "n_seeds": 20,
        "certified_seed_count": 18,
        "source_calls_per_run": 384,
        "target_search_calls_per_run": 13,
        "target_verification_calls_per_run": 300,
        "policy_vectors_exported": False,
        "information_contract": {
            "track": "descriptor_conditional_external_holdout",
            "source_selection_mode": "descriptor_nearest",
            "source_domains": [
                "QueueResourceControl",
                "InventorySupplyChain",
            ],
            "excluded_nearest_source_analogue": (
                "FactorShockStatePolicyRZDT1"),
            "source_split_heldout": "FactorShockStatePolicyRZDT1",
            "target_domain": "Ingolstadt21Traffic",
            "heldout_task_family_identifier_used_by_proposal": True,
            "target_labels_used_to_fit_proposal": False,
            "target_oracle_used": False,
            "historical_target_anchor_used": False,
        },
        "rows": traffic_rows,
    })
    traffic_negative_control = _write(
        tmp_path / "traffic_negative_control.json",
        {
            "status": "complete",
            "n_seeds": 5,
            "certified_seed_count": 0,
            "source_calls_per_run": 384,
            "target_search_calls_per_run": 13,
            "target_verification_calls_per_run": 300,
            "policy_vectors_exported": False,
            "information_contract": {
                "track": "domain_blind_external_holdout",
                "source_selection_mode": "domain_blind_exclude_nearest",
                "source_domains": [
                    "FactorShockStatePolicyRZDT1",
                    "InventorySupplyChain",
                ],
                "excluded_nearest_source_analogue": (
                    "QueueResourceControl"),
                "source_split_heldout": "QueueResourceControl",
                "target_domain": "Ingolstadt21Traffic",
                "heldout_task_family_identifier_used_by_proposal": False,
                "target_labels_used_to_fit_proposal": False,
                "target_oracle_used": False,
                "historical_target_anchor_used": False,
            },
            "rows": traffic_rows[:5],
        },
    )
    proposal_coverage = _write(tmp_path / "proposal_coverage.json", {
        "status": "complete_with_conditional_global_bound",
        "contract_id": "source_target_geometric_atlas_coverage_v1",
        "domain_count": 3,
        "finite_library_condition_pass_count": 3,
        "global_lipschitz_certified_count": 0,
        "global_theorem_claim_mode": "conditional_theorem_only",
        "unconditional_global_coverage_claim_allowed": False,
        "rows": [{
            "deterministic_atlas": True,
            "target_truth_used_post_run_only": True,
            "target_truth_used_for_proposal_or_selection": False,
        } for _ in range(3)],
    })
    proof = _write(tmp_path / "proof.json", {
        "status": "pass",
        "lean_source_count": 10,
        "lean_source_tree_sha256": "a" * 64,
        "forbidden_declaration_count": 0,
        "build": {"executed": True, "returncode": 0},
    })
    return (
        method,
        registry,
        audit,
        statistics,
        convergence,
        traffic,
        traffic_negative_control,
        proposal_coverage,
        proof,
    )


def test_release_finalizer_is_fail_closed_and_hash_addressed(tmp_path):
    paths = _fixtures(tmp_path)
    release = build_release(
        method_contract_path=paths[0],
        registry_path=paths[1],
        audit_path=paths[2],
        statistics_path=paths[3],
        convergence_path=paths[4],
        traffic_path=paths[5],
        traffic_negative_control_path=paths[6],
        proposal_coverage_path=paths[7],
        proof_receipt_path=paths[8],
        repository_commit="commit",
    )
    assert release["status"] == "ready_for_manuscript_lock"
    assert release["audited_result_count"] == 241
    assert release["failed_or_timeout_result_count"] == 0
    assert len(release["audited_result_receipt_sha256"]) == 64

    traffic = json.loads(paths[5].read_text(encoding="utf-8"))
    traffic["n_seeds"] = 5
    paths[5].write_text(json.dumps(traffic), encoding="utf-8")
    with pytest.raises(ValueError, match="fewer than 20"):
        build_release(
            method_contract_path=paths[0],
            registry_path=paths[1],
            audit_path=paths[2],
            statistics_path=paths[3],
            convergence_path=paths[4],
            traffic_path=paths[5],
            traffic_negative_control_path=paths[6],
            proposal_coverage_path=paths[7],
            proof_receipt_path=paths[8],
            repository_commit="commit",
        )


def test_release_accepts_hash_bound_remote_compact_shards(tmp_path):
    paths = _fixtures(tmp_path)
    audit = json.loads(paths[2].read_text(encoding="utf-8"))
    shard = _write(tmp_path / "remote_shard.json", {
        "schema_version": 1,
        "records": audit["records"],
    })
    import hashlib
    audit["source_mode"] = "remote_compact_record_shards"
    audit["record_shard_receipts"] = [{
        "path": str(shard),
        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
        "record_count": len(audit["records"]),
    }]
    for record in audit["records"]:
        record["path"] = "/remote/results/result.json"
        record["content_verified_at_extraction"] = True
    paths[2].write_text(json.dumps(audit), encoding="utf-8")
    (tmp_path / "result.json").unlink()

    release = build_release(
        method_contract_path=paths[0],
        registry_path=paths[1],
        audit_path=paths[2],
        statistics_path=paths[3],
        convergence_path=paths[4],
        traffic_path=paths[5],
        traffic_negative_control_path=paths[6],
        proposal_coverage_path=paths[7],
        proof_receipt_path=paths[8],
        repository_commit="commit",
    )

    assert release["status"] == "ready_for_manuscript_lock"
