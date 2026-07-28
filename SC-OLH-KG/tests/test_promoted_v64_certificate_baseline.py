from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def test_promoted_v64_certificate_record_matches_fresh_gate():
    record = json.loads((
        REPO
        / "SC-OLH-KG/performance/promoted_certificate_baseline.json"
    ).read_text(encoding="utf-8"))
    evidence = json.loads((
        REPO
        / "SC-OLH-KG/performance/baselines/"
        "v64_powered_safe_interior_evidence.json"
    ).read_text(encoding="utf-8"))
    registry = json.loads((
        REPO
        / "SC-OLH-KG/performance/manifests/theory_contract_registry.json"
    ).read_text(encoding="utf-8"))
    assert record["name"] == evidence["name"]
    assert record["role"] == "certified_deployment_baseline"
    assert record["search_performance_baseline"] == (
        "v51_observed_terminal_closure")
    assert record["verification"]["primary_budget"] == 80
    assert record["verification"]["support_budget"] == 96
    assert record["verification"]["familywise_delta"] == 0.05
    assert record["evidence"]["formal_gate_passed"] is True
    assert evidence["verification"]["certified"] == 60
    assert evidence["verification"]["false_certificates"] == 0
    assert evidence["paired_against_v51"]["losses"] == 0
    assert evidence["contracts"]["search_trajectories_identical"] is True
    assert evidence["contracts"]["target_oracle_used"] is False
    assert record["budget_interpretation"][
        "verification_calls_must_be_included_in_total_cost"
    ] is True
    contracts = {
        item["id"]: item for item in registry["contracts"]
    }
    contract = contracts[record["theory_contract_id"]]
    assert contract["implementation_contract_id"] == (
        record["implementation_contract_id"])
    assert contract["status"] == "fresh_gate_passed"
    assert evidence["run_id"] in contract["runs"]
