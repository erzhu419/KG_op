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
    assert audit["status"] == "pass"
    assert audit["record_count"] == 2
    assert audit["track_audits"][0]["status"] == "pass"
