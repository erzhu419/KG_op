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
        "tracks": [{}],
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
        "rows": traffic_rows,
    })
    proof = _write(tmp_path / "proof.json", {
        "status": "pass",
        "lean_source_count": 10,
        "lean_source_tree_sha256": "a" * 64,
        "forbidden_declaration_count": 0,
        "build": {"executed": True, "returncode": 0},
    })
    return method, registry, audit, statistics, traffic, proof


def test_release_finalizer_is_fail_closed_and_hash_addressed(tmp_path):
    paths = _fixtures(tmp_path)
    release = build_release(
        method_contract_path=paths[0],
        registry_path=paths[1],
        audit_path=paths[2],
        statistics_path=paths[3],
        traffic_path=paths[4],
        proof_receipt_path=paths[5],
        repository_commit="commit",
    )
    assert release["status"] == "ready_for_manuscript_lock"
    assert release["audited_result_count"] == 240
    assert release["failed_or_timeout_result_count"] == 0
    assert len(release["audited_result_receipt_sha256"]) == 64

    traffic = json.loads(paths[4].read_text(encoding="utf-8"))
    traffic["n_seeds"] = 5
    paths[4].write_text(json.dumps(traffic), encoding="utf-8")
    with pytest.raises(ValueError, match="fewer than 20"):
        build_release(
            method_contract_path=paths[0],
            registry_path=paths[1],
            audit_path=paths[2],
            statistics_path=paths[3],
            traffic_path=paths[4],
            proof_receipt_path=paths[5],
            repository_commit="commit",
        )
