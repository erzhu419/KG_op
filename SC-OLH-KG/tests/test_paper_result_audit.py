import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.paper_result_audit import (  # noqa: E402
    build_audit,
    extract_result_record,
)


def _write_result(path, *, method, seed, schedule="every_iteration"):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "status": "ok",
        "method": method,
        "heldout": "QueueResourceControl",
        "seed": seed,
        "initial_points_fingerprint": f"initial-{seed}",
        "source_archive_fingerprint": "archive",
        "information_contract": {
            "offline_source_calls": 384,
            "target_search_calls": 13,
            "target_verification_calls": 80,
            "target_total_calls": 93,
        },
        "result": {
            "method": method,
            "algorithm_fidelity": (
                "saas_fully_bayesian_nuts_constrained_qlogei"),
            "saas_nuts_schedule": {
                "hyperposterior_refit_schedule": schedule,
                "posterior_conditions_on_every_observation": True,
            },
            "n_search_simulations": 13,
            "n_verification_simulations": 80,
            "n_target_simulations_total": 93,
            "true_feasible": True,
            "feasible_regret": 0.01,
            "terminal_verification": {
                "certified": True,
                "method": "verifier",
                "protocol": "fixed",
                "familywise_delta": 0.05,
                "candidate_verification_budgets": [80, 128, 128],
                "shortlist_mode": "fixed",
                "posterior_updated_from_verification": False,
                "search_samples_reused": False,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_result_audit_keeps_canonical_and_periodic_saas_separate(tmp_path):
    canonical = tmp_path / "canonical" / "result.json"
    periodic = tmp_path / "periodic" / "result.json"
    _write_result(
        canonical, method="botorch_saasbo", seed=80)
    _write_result(
        periodic, method="botorch_saasbo", seed=80, schedule="doubling")
    canonical_row = extract_result_record(canonical, track_id="canonical")
    periodic_row = extract_result_record(periodic, track_id="periodic")
    assert canonical_row["method_identity"] == (
        "canonical_saasbo_every_iteration")
    assert periodic_row["method_identity"] == "saasbo_periodic_capped"
    assert canonical_row["source_plus_target_total_calls"] == 477


def test_track_audit_requires_paired_information_contracts(tmp_path):
    for seed in (80, 81):
        _write_result(
            tmp_path / "track" / "a" / f"seed{seed}" / "result.json",
            method="botorch_saasbo",
            seed=seed,
        )
    registry = {
        "registry_id": "unit",
        "tracks": [{
            "track_id": "paired",
            "result_root": "track",
            "expected_method_identities": [
                "canonical_saasbo_every_iteration",
            ],
            "expected_domains": ["QueueResourceControl"],
            "expected_seeds": [80, 81],
            "required_source_calls": 384,
            "required_search_calls": 13,
            "paired_equality_fields": [
                "source_archive_fingerprint",
                "initial_design_fingerprint",
                "verifier_signature",
            ],
        }],
    }
    audit = build_audit(registry, root=tmp_path)
    assert audit["status"] == "pass", audit
    assert audit["record_count"] == 2
    assert audit["track_audits"][0]["status"] == "pass"


def test_result_level_source_calls_and_total_optimization_contract(tmp_path):
    path = tmp_path / "uniform" / "result.json"
    _write_result(
        path,
        method="uniform_verified::canonical_saasbo_every_iteration",
        seed=80,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["information_contract"].pop("offline_source_calls")
    payload["result"]["source_calls"] = 384
    payload["result"].pop("algorithm_fidelity")
    payload["result"].pop("saas_nuts_schedule")
    path.write_text(json.dumps(payload), encoding="utf-8")

    registry = {
        "registry_id": "uniform",
        "tracks": [{
            "track_id": "uniform",
            "result_root": "uniform",
            "expected_method_identities": [
                "uniform_verified::canonical_saasbo_every_iteration",
            ],
            "expected_domains": ["QueueResourceControl"],
            "expected_seeds": [80],
            "required_optimization_calls": 397,
        }],
    }
    audit = build_audit(registry, root=tmp_path)
    assert audit["status"] == "pass"
    record = audit["records"][0]
    assert record["source_calls"] == 384
    assert record["optimization_calls_excluding_verification"] == 397
    assert len(record["result_sha256"]) == 64

    registry["tracks"][0]["required_optimization_calls"] = 398
    failed = build_audit(registry, root=tmp_path)
    assert failed["status"] == "incomplete_or_failed"
    assert failed["track_audits"][0]["failures"] == [{
        "kind": "optimization_budget_mismatch",
        "count": 1,
    }]


def test_track_audit_enforces_method_specific_source_budgets(tmp_path):
    universal = tmp_path / "causal" / "universal" / "result.json"
    source = tmp_path / "causal" / "source" / "result.json"
    _write_result(universal, method="universal", seed=80)
    _write_result(source, method="source", seed=80)
    universal_payload = json.loads(universal.read_text(encoding="utf-8"))
    universal_payload["information_contract"]["offline_source_calls"] = 0
    universal_payload["information_contract"]["target_search_calls"] = 10
    universal_payload["result"]["n_search_simulations"] = 10
    universal_payload["result"].pop("algorithm_fidelity")
    universal_payload["result"].pop("saas_nuts_schedule")
    universal.write_text(
        json.dumps(universal_payload), encoding="utf-8")
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    source_payload["information_contract"]["target_search_calls"] = 10
    source_payload["result"]["n_search_simulations"] = 10
    source_payload["result"].pop("algorithm_fidelity")
    source_payload["result"].pop("saas_nuts_schedule")
    source.write_text(json.dumps(source_payload), encoding="utf-8")

    registry = {
        "registry_id": "causal",
        "tracks": [{
            "track_id": "causal",
            "result_root": "causal",
            "expected_method_identities": ["universal", "source"],
            "expected_domains": ["QueueResourceControl"],
            "expected_seeds": [80],
            "required_source_calls_by_method": {
                "universal": 0,
                "source": 384,
            },
            "required_search_calls": 10,
            "paired_equality_fields": ["verifier_signature"],
        }],
    }
    audit = build_audit(registry, root=tmp_path)
    assert audit["status"] == "pass", audit

    registry["tracks"][0]["required_source_calls_by_method"]["universal"] = 1
    failed = build_audit(registry, root=tmp_path)
    assert failed["status"] == "incomplete_or_failed"
    assert any(
        row["kind"] == "method_source_budget_mismatch"
        for row in failed["track_audits"][0]["failures"]
    )
