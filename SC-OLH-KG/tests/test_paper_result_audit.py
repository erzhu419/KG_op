import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.paper_result_audit import (  # noqa: E402
    build_audit,
    build_audit_from_records,
    extract_result_record,
    load_record_shards,
)
from performance.paper_result_record_shard import (  # noqa: E402
    build_record_shard,
)


def _write_result(
    path,
    *,
    method,
    seed,
    schedule="every_iteration",
    frozen_execution=False,
):
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
    if frozen_execution:
        payload["execution_provenance"] = {
            "status": "frozen",
            "repository_commit": "a" * 40,
            "scolhkg_tree": "b" * 40,
            "proof_tree": "c" * 40,
            "scripts_tree": "d" * 40,
            "method_contract_id": "method-v1",
            "theory_contract_id": "theory-v1",
            "snapshot_root": "/immutable/a",
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
    assert canonical_row["problem_contract"]["shared_shock_scale"] is None
    assert len(canonical_row["problem_contract_fingerprint"]) == 64


def test_result_audit_detects_factor_shock_scenario_mismatch(tmp_path):
    canonical = tmp_path / "track" / "canonical" / "result.json"
    scolh = tmp_path / "track" / "scolh" / "result.json"
    _write_result(
        canonical, method="botorch_saasbo", seed=80)
    canonical_payload = json.loads(canonical.read_text(encoding="utf-8"))
    canonical_payload["heldout"] = "FactorShockStatePolicyRZDT1"
    canonical_payload["information_contract"]["target_dimension"] = 1000
    canonical.write_text(json.dumps(canonical_payload), encoding="utf-8")
    _write_result(scolh, method="scolh", seed=80)
    scolh_payload = json.loads(scolh.read_text(encoding="utf-8"))
    scolh_payload["heldout"] = "FactorShockStatePolicyRZDT1"
    scolh_payload["information_contract"]["target_dimension"] = 1000
    scolh_payload["config"] = {
        "d": 1000,
        "target_shared_shock_scale": 0.0,
    }
    scolh_payload["result"].pop("algorithm_fidelity")
    scolh_payload["result"].pop("saas_nuts_schedule")
    scolh.write_text(json.dumps(scolh_payload), encoding="utf-8")
    registry = {
        "registry_id": "scenario",
        "tracks": [{
            "track_id": "scenario",
            "result_root": "track",
            "expected_method_identities": [
                "canonical_saasbo_every_iteration",
                "scolh",
            ],
            "expected_domains": ["FactorShockStatePolicyRZDT1"],
            "expected_seeds": [80],
            "paired_equality_fields": [
                "problem_contract_fingerprint",
            ],
        }],
    }
    audit = build_audit(registry, root=tmp_path)
    assert audit["status"] == "incomplete_or_failed"
    assert {
        row["kind"] for row in audit["track_audits"][0]["failures"]
    } == {"paired_problem_contract_fingerprint_mismatch"}


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


def test_track_audit_requires_frozen_execution_contract(tmp_path):
    result = tmp_path / "frozen" / "result.json"
    _write_result(
        result,
        method="botorch_saasbo",
        seed=80,
        frozen_execution=True,
    )
    registry = {
        "registry_id": "frozen",
        "tracks": [{
            "track_id": "frozen",
            "result_root": "frozen",
            "expected_method_identities": [
                "canonical_saasbo_every_iteration",
            ],
            "expected_domains": ["QueueResourceControl"],
            "expected_seeds": [80],
            "required_execution_provenance_status": "frozen",
            "allowed_execution_commits": ["a" * 40],
            "required_scolhkg_tree": "b" * 40,
            "required_method_contract_id": "method-v1",
            "required_theory_contract_id": "theory-v1",
        }],
    }
    audit = build_audit(registry, root=tmp_path)
    assert audit["status"] == "pass", audit
    row = audit["records"][0]
    assert row["execution_repository_commit"] == "a" * 40
    assert row["execution_scolhkg_tree"] == "b" * 40

    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["execution_provenance"]["repository_commit"] = "e" * 40
    result.write_text(json.dumps(payload), encoding="utf-8")
    failed = build_audit(registry, root=tmp_path)
    assert failed["status"] == "incomplete_or_failed"
    assert any(
        row["kind"] == "execution_commit_mismatch"
        for row in failed["track_audits"][0]["failures"]
    )


def test_track_audit_accepts_backend_specific_execution_contracts(tmp_path):
    first = tmp_path / "contracts" / "first" / "result.json"
    second = tmp_path / "contracts" / "second" / "result.json"
    _write_result(
        first,
        method="botorch_turbo",
        seed=80,
        frozen_execution=True,
    )
    _write_result(
        second,
        method="botorch_scbo",
        seed=80,
        frozen_execution=True,
    )
    second_payload = json.loads(second.read_text(encoding="utf-8"))
    second_payload["execution_provenance"][
        "method_contract_id"
    ] = "method-v2"
    second.write_text(json.dumps(second_payload), encoding="utf-8")
    registry = {
        "registry_id": "method-contracts",
        "tracks": [{
            "track_id": "contracts",
            "result_root": "contracts",
            "expected_method_identities": [
                "botorch_turbo:canonical_turbo1_ts",
                "botorch_scbo:canonical_scbo_constrained_ts",
            ],
            "expected_domains": ["QueueResourceControl"],
            "expected_seeds": [80],
            "required_method_contract_by_method": {
                "botorch_turbo:canonical_turbo1_ts": "method-v1",
                "botorch_scbo:canonical_scbo_constrained_ts": "method-v2",
            },
        }],
    }
    audit = build_audit(registry, root=tmp_path)
    assert audit["status"] == "pass", audit


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


def test_result_sources_select_repaired_methods_without_duplicate_cells(
    tmp_path,
):
    old = tmp_path / "old"
    repaired = tmp_path / "repaired"
    _write_result(
        old / "stable" / "result.json",
        method="botorch_turbo",
        seed=80,
    )
    _write_result(
        old / "broken" / "result.json",
        method="botorch_scbo",
        seed=80,
    )
    broken = json.loads(
        (old / "broken" / "result.json").read_text(encoding="utf-8"))
    broken["status"] = "failed"
    (old / "broken" / "result.json").write_text(
        json.dumps(broken), encoding="utf-8")
    _write_result(
        repaired / "fixed" / "result.json",
        method="botorch_scbo",
        seed=80,
    )
    registry = {
        "registry_id": "repaired",
        "tracks": [{
            "track_id": "track",
            "result_sources": [
                {
                    "result_root": "old",
                    "include_method_identities": [
                        "botorch_turbo:canonical_turbo1_ts"
                    ],
                },
                {
                    "result_root": "repaired",
                    "include_method_identities": [
                        "botorch_scbo:canonical_scbo_constrained_ts"
                    ],
                },
            ],
            "expected_method_identities": [
                "botorch_turbo:canonical_turbo1_ts",
                "botorch_scbo:canonical_scbo_constrained_ts",
            ],
            "expected_domains": ["QueueResourceControl"],
            "expected_seeds": [80],
        }],
    }

    audit = build_audit(registry, root=tmp_path)

    assert audit["status"] == "pass", audit
    assert audit["record_count"] == 2
    assert {row["method_identity"] for row in audit["records"]} == {
        "botorch_turbo:canonical_turbo1_ts",
        "botorch_scbo:canonical_scbo_constrained_ts",
    }


def test_remote_compact_record_shard_is_hash_bound(tmp_path):
    _write_result(
        tmp_path / "track" / "result.json",
        method="botorch_turbo",
        seed=80,
    )
    registry = {
        "registry_id": "remote",
        "tracks": [{
            "track_id": "track",
            "result_root": "track",
            "expected_method_identities": [
                "botorch_turbo:canonical_turbo1_ts"
            ],
            "expected_domains": ["QueueResourceControl"],
            "expected_seeds": [80],
        }],
    }
    shard_payload = build_record_shard(
        registry,
        root=tmp_path,
        origin="node001",
    )
    shard = tmp_path / "record_shard.json"
    shard.write_text(json.dumps(shard_payload), encoding="utf-8")

    records, receipts, sources = load_record_shards(
        [shard],
        registry=registry,
    )
    audit = build_audit_from_records(
        registry,
        records,
        source_mode="remote_compact_record_shards",
        record_shard_receipts=receipts,
        source_receipts=sources,
    )
    assert audit["status"] == "pass"
    assert audit["source_mode"] == "remote_compact_record_shards"
    assert audit["records"][0]["extraction_origin"] == "node001"

    shard_payload["records"][0]["seed"] = 81
    shard.write_text(json.dumps(shard_payload), encoding="utf-8")
    try:
        load_record_shards([shard], registry=registry)
    except ValueError as error:
        assert "payload hash mismatch" in str(error)
    else:
        raise AssertionError("tampered compact record shard was accepted")
