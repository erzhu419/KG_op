import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.structural_hypothesis_loop import (  # noqa: E402
    ContractValidationError,
    run_structural_hypothesis_loop,
    verify_audit_chain,
    verify_report_integrity,
)


CONTRACT_PATH = (
    ROOT
    / "performance"
    / "manifests"
    / "structural_hypothesis_loop_v1.json"
)
HISTORICAL_CSV = (
    ROOT
    / "results"
    / "completed_non_online_sota_20260716"
    / "structural_backend"
    / "rows.csv"
)
RUNNER = ROOT / "runners" / "run_structural_hypothesis_loop.py"


PROFILES = (
    "none",
    "low_frequency_only",
    "orthogonality_only",
    "sparsity_only",
    "additivity_only",
    "full",
    "leave_out_low_frequency",
    "leave_out_orthogonality",
    "leave_out_sparsity",
    "leave_out_additivity",
)


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _rows(
    *,
    feasible_per_domain=None,
    losses=None,
    regrets=None,
    omit=(),
):
    feasible_per_domain = feasible_per_domain or {name: 8 for name in PROFILES}
    losses = losses or {name: 0 for name in PROFILES}
    regrets = regrets or {name: 1.0 for name in PROFILES}
    omitted = set(omit)
    contract = _contract()
    scope = contract["evidence_scope"]
    rows = []
    for profile in PROFILES:
        if profile in omitted:
            continue
        feasible = feasible_per_domain[profile]
        loss_count = losses[profile]
        row_index = 0
        for domain in scope["domains"]:
            for seed in scope["seeds"]:
                rows.append({
                    "track": scope["track"],
                    "run_id": scope["run_id"],
                    "variant": scope["variant_template"].format(
                        profile=profile
                    ),
                    "method": profile,
                    "structural_prior_profile": profile,
                    "domain": domain,
                    "seed": str(seed),
                    "d": str(scope["d"]),
                    "N": str(scope["N"]),
                    "n0": str(scope["n0"]),
                    "source_calls": str(scope["source_calls"]),
                    "implementation": scope["implementation"],
                    "initial_design": scope["initial_design"],
                    "decision_backend": scope["decision_backend"],
                    "status": "ok",
                    "true_feasible": seed < feasible,
                    "adaptive_loss": row_index < loss_count,
                    "feasible_regret": str(regrets[profile]),
                    **scope["fixed_row_values"],
                })
                row_index += 1
    return rows


def _write_csv(path, rows):
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _report(rows):
    return run_structural_hypothesis_loop(rows, _contract()).to_dict()


def _supported_rows():
    feasible = {name: 9 for name in PROFILES}
    regrets = {name: 1.0 for name in PROFILES}
    feasible["none"] = 8
    feasible["full"] = 10
    regrets["none"] = 2.0
    regrets["full"] = 0.0
    return _rows(feasible_per_domain=feasible, regrets=regrets)


def _assert_valid_audit(report):
    audit = report["audit"]
    assert verify_audit_chain(audit["events"], audit["head"])
    assert verify_report_integrity(report)
    assert len(audit["events"]) == 19
    assert [event["seq"] for event in audit["events"]] == list(range(19))


def test_graph_has_frozen_nine_node_order_and_parentage():
    report = _report(_supported_rows())
    expected = [
        ("COMPOSITE", None, "full", "none"),
        ("NECESSITY", "low_frequency", "full", "leave_out_low_frequency"),
        ("STANDALONE", "low_frequency", "low_frequency_only", "none"),
        ("NECESSITY", "orthogonality", "full", "leave_out_orthogonality"),
        ("STANDALONE", "orthogonality", "orthogonality_only", "none"),
        ("NECESSITY", "sparsity", "full", "leave_out_sparsity"),
        ("STANDALONE", "sparsity", "sparsity_only", "none"),
        ("NECESSITY", "additivity", "full", "leave_out_additivity"),
        ("STANDALONE", "additivity", "additivity_only", "none"),
    ]
    hypotheses = report["hypotheses"]
    assert [
        (
            item["kind"], item["component"], item["challenger_profile"],
            item["reference_profile"],
        )
        for item in hypotheses
    ] == expected
    root_id = hypotheses[0]["hypothesis_id"]
    assert hypotheses[0]["parent_ids"] == []
    assert all(item["parent_ids"] == [root_id] for item in hypotheses[1:])
    assert all(
        item["trigger_disposition"] == "SUPPORTED_SCOPED"
        for item in hypotheses[1:]
    )
    assert len({item["hypothesis_id"] for item in hypotheses}) == 9
    assert [item["hypothesis_id"] for item in report["decisions"]] == [
        item["hypothesis_id"] for item in hypotheses
    ]


def test_root_verdict_drives_child_revision_order_and_trigger():
    supported = _supported_rows()
    needs = _rows(omit=("full",))
    refuted = _rows()
    invalid = _supported_rows()
    invalid[0]["hvd_profile"] = "poisoned"
    cases = (
        (supported, "SUPPORTED_SCOPED", ("NECESSITY", "STANDALONE")),
        (needs, "NEEDS_EVIDENCE", ("STANDALONE", "NECESSITY")),
        (refuted, "REFUTED_SCOPED", ("STANDALONE", "NECESSITY")),
        (invalid, "INVALID_EVIDENCE", ("STANDALONE", "NECESSITY")),
    )
    for rows, root_disposition, pair_order in cases:
        report = _report(rows)
        assert report["decisions"][0]["disposition"] == root_disposition
        children = report["hypotheses"][1:]
        assert all(
            item["trigger_disposition"] == root_disposition
            for item in children
        )
        assert [item["kind"] for item in children] == list(pair_order) * 4


def test_complete_positive_evidence_supports_all_nine_hypotheses():
    report = _report(_supported_rows())
    assert report["status"] == "COMPLETED"
    assert report["stop_reason"] == "FINITE_GRAPH_EXHAUSTED"
    assert report["verdict_counts"] == {
        "SUPPORTED_SCOPED": 9,
        "REFUTED_SCOPED": 0,
        "NEEDS_EVIDENCE": 0,
        "INVALID_EVIDENCE": 0,
    }
    assert all(
        item["disposition"] == "SUPPORTED_SCOPED"
        for item in report["decisions"]
    )
    _assert_valid_audit(report)


def test_unsafe_negative_feasibility_is_refuted_despite_better_regret():
    feasible = {name: 9 for name in PROFILES}
    feasible["none"] = 10
    for name in PROFILES:
        if name.startswith("leave_out_"):
            feasible[name] = 10
    losses = {name: 3 for name in PROFILES}
    losses["none"] = 0
    for name in PROFILES:
        if name.startswith("leave_out_"):
            losses[name] = 0
    regrets = {name: 0.0 for name in PROFILES}
    regrets["none"] = 10.0
    for name in PROFILES:
        if name.startswith("leave_out_"):
            regrets[name] = 10.0

    report = _report(_rows(
        feasible_per_domain=feasible,
        losses=losses,
        regrets=regrets,
    ))
    assert report["verdict_counts"]["REFUTED_SCOPED"] == 9
    assert all(
        item["disposition"] == "REFUTED_SCOPED"
        for item in report["decisions"]
    )
    root = report["decisions"][0]
    assert root["metrics"]["challenger_safety"]["pass"] is False
    assert root["metrics"]["paired_feasibility"]["net"] < 0
    assert root["metrics"]["conditional_regret"]["wins"] > 0
    assert "challenger_safety_gate_failed" in root["reason_codes"]
    assert "paired_feasibility_net_negative" in root["reason_codes"]


def test_missing_full_is_needs_evidence_but_other_nodes_continue():
    feasible = {name: 9 for name in PROFILES}
    feasible["none"] = 8
    report = _report(_rows(feasible_per_domain=feasible, omit=("full",)))
    dispositions = [item["disposition"] for item in report["decisions"]]
    assert dispositions == [
        "NEEDS_EVIDENCE",
        "SUPPORTED_SCOPED", "NEEDS_EVIDENCE",
        "SUPPORTED_SCOPED", "NEEDS_EVIDENCE",
        "SUPPORTED_SCOPED", "NEEDS_EVIDENCE",
        "SUPPORTED_SCOPED", "NEEDS_EVIDENCE",
    ]
    assert report["verdict_counts"]["NEEDS_EVIDENCE"] == 5
    assert report["verdict_counts"]["REFUTED_SCOPED"] == 0
    assert report["status"] == "COMPLETED_WITH_EVIDENCE_GAPS"
    assert report["stop_reason"] == "FINITE_GRAPH_EXHAUSTED"
    # The same absent full cells are requested by five hypotheses but are
    # deduplicated into one executable evidence plan.
    assert len(report["pending_evidence"]) == 30


def test_failed_reference_requests_reexecution_instead_of_manufacturing_wins():
    rows = _supported_rows()
    for row in rows:
        if row["method"] == "none":
            row["status"] = "crashed"
            row["true_feasible"] = False
            row["feasible_regret"] = ""
    report = _report(rows)
    root = report["decisions"][0]
    assert root["disposition"] == "NEEDS_EVIDENCE"
    assert root["reason_codes"] == ["comparison_execution_incomplete"]
    assert root["metrics"]["non_ok_row_count"] == 30
    for hypothesis, decision in zip(report["hypotheses"], report["decisions"]):
        if hypothesis["reference_profile"] == "none":
            assert decision["disposition"] == "NEEDS_EVIDENCE"


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("hvd_profile", "class"),
        ("total_calls", 999),
        ("risk_penalty", 0.0),
        ("utility_weight", 9.0),
        ("source_discrepancy_update", False),
        ("adaptive_replication_voi", True),
        ("posterior_dominance_enabled", True),
        ("recheck_top_k", 4),
    ),
)
def test_non_treatment_configuration_drift_is_invalid(field, replacement):
    rows = _supported_rows()
    rows[0][field] = replacement
    report = _report(rows)
    assert report["verdict_counts"]["INVALID_EVIDENCE"] > 0


def test_negative_regret_is_invalid_evidence():
    rows = _supported_rows()
    rows[0]["feasible_regret"] = "-0.1"
    report = _report(rows)
    assert report["verdict_counts"]["INVALID_EVIDENCE"] > 0


def test_contract_rejects_additional_variant_template_placeholders():
    contract = _contract()
    contract["evidence_scope"]["variant_template"] = (
        "structural_backend/priors/{profile}/{poison}"
    )
    with pytest.raises(ContractValidationError, match="V1 template"):
        run_structural_hypothesis_loop(_supported_rows(), contract)


@pytest.mark.parametrize(
    "corruption",
    ("duplicate", "protocol_drift", "malformed_bool", "nonfinite"),
)
def test_integrity_failures_are_invalid_evidence(corruption):
    rows = _supported_rows()
    if corruption == "duplicate":
        rows.append(dict(rows[0]))
    elif corruption == "protocol_drift":
        rows[0]["decision_backend"] = "not_the_frozen_backend"
    elif corruption == "malformed_bool":
        rows[0]["true_feasible"] = "yes"
    elif corruption == "nonfinite":
        rows[0]["feasible_regret"] = "nan"

    report = _report(rows)
    assert report["status"] == "COMPLETED_WITH_INVALID_EVIDENCE"
    assert report["verdict_counts"]["INVALID_EVIDENCE"] > 0
    invalid = [
        item for item in report["decisions"]
        if item["disposition"] == "INVALID_EVIDENCE"
    ]
    assert invalid
    assert all(item["invalid_issues"] for item in invalid)
    _assert_valid_audit(report)


def test_report_and_audit_chain_are_deterministic_and_tamper_evident():
    first = _report(_supported_rows())
    second = _report(_supported_rows())
    assert first == second
    _assert_valid_audit(first)

    tampered_events = json.loads(json.dumps(first["audit"]["events"]))
    tampered_events[1]["reason_codes"].append("fabricated_reason")
    assert not verify_audit_chain(tampered_events, first["audit"]["head"])
    assert not verify_audit_chain(
        first["audit"]["events"], "sha256:" + "f" * 64
    )
    for mutation in ("metrics", "counts", "nonclaims", "synthesis"):
        tampered = json.loads(json.dumps(first))
        if mutation == "metrics":
            tampered["decisions"][0]["metrics"]["fabricated"] = True
        elif mutation == "counts":
            tampered["verdict_counts"]["SUPPORTED_SCOPED"] = 0
        elif mutation == "nonclaims":
            tampered["nonclaims"] = []
        else:
            tampered["synthesis"]["composite"] = "FABRICATED"
        assert not verify_report_integrity(tampered)


@pytest.mark.skipif(
    not HISTORICAL_CSV.is_file(),
    reason="local ignored historical evidence is not present",
)
def test_historical_csv_refutes_standalones_and_marks_full_comparisons_missing():
    with HISTORICAL_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    report = _report(rows)
    assert len(report["decisions"]) == 9
    assert report["verdict_counts"] == {
        "SUPPORTED_SCOPED": 0,
        "REFUTED_SCOPED": 4,
        "NEEDS_EVIDENCE": 5,
        "INVALID_EVIDENCE": 0,
    }
    assert [item["disposition"] for item in report["decisions"]] == [
        "NEEDS_EVIDENCE",
        "REFUTED_SCOPED", "NEEDS_EVIDENCE",
        "REFUTED_SCOPED", "NEEDS_EVIDENCE",
        "REFUTED_SCOPED", "NEEDS_EVIDENCE",
        "REFUTED_SCOPED", "NEEDS_EVIDENCE",
    ]
    assert report["synthesis"]["ignored_out_of_scope_rows"] == 1080
    _assert_valid_audit(report)


def test_cli_writes_complete_offline_report_to_out(tmp_path):
    evidence = tmp_path / "evidence.csv"
    output = tmp_path / "report.json"
    _write_csv(evidence, _supported_rows())
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--evidence-csv", str(evidence),
            "--contract", str(CONTRACT_PATH),
            "--out", str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert "SUPPORTED_SCOPED=9" in completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    artifact = report["input_artifacts"]["evidence_csv"]
    assert artifact["path"] == str(evidence.resolve())
    assert artifact["bytes"] == evidence.stat().st_size
    assert len(artifact["sha256"]) == 64
    assert report["verdict_counts"]["SUPPORTED_SCOPED"] == 9
    _assert_valid_audit(report)


def test_cli_returns_two_but_still_writes_report_for_invalid_evidence(tmp_path):
    evidence = tmp_path / "invalid-evidence.csv"
    output = tmp_path / "invalid-report.json"
    rows = _supported_rows()
    rows[0]["true_feasible"] = "not-a-boolean"
    _write_csv(evidence, rows)
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--evidence-csv", str(evidence),
            "--contract", str(CONTRACT_PATH),
            "--out", str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["verdict_counts"]["INVALID_EVIDENCE"] > 0
    _assert_valid_audit(report)


def test_cli_does_not_confuse_non_ok_status_text_with_invalid_disposition(tmp_path):
    evidence = tmp_path / "non-ok-evidence.csv"
    output = tmp_path / "non-ok-report.json"
    rows = _supported_rows()
    rows[0]["status"] = "INVALID_EVIDENCE"
    rows[0]["true_feasible"] = False
    rows[0]["feasible_regret"] = ""
    _write_csv(evidence, rows)
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--evidence-csv", str(evidence),
            "--contract", str(CONTRACT_PATH),
            "--out", str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["verdict_counts"]["INVALID_EVIDENCE"] == 0
    assert report["verdict_counts"]["NEEDS_EVIDENCE"] > 0
    _assert_valid_audit(report)


def test_cli_rejects_duplicate_csv_headers(tmp_path):
    evidence = tmp_path / "duplicate-header.csv"
    evidence.write_text("track,track\npriors,priors\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--evidence-csv", str(evidence),
            "--contract", str(CONTRACT_PATH),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "duplicate column names" in completed.stderr


def test_cli_rejects_missing_required_headers(tmp_path):
    evidence = tmp_path / "missing-headers.csv"
    evidence.write_text("track,run_id\npriors,run\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--evidence-csv", str(evidence),
            "--contract", str(CONTRACT_PATH),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "lacks required columns" in completed.stderr


def test_cli_refuses_output_hard_linked_to_evidence(tmp_path):
    evidence = tmp_path / "evidence.csv"
    output = tmp_path / "report.json"
    _write_csv(evidence, _supported_rows())
    before = evidence.read_bytes()
    os.link(evidence, output)
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--evidence-csv", str(evidence),
            "--contract", str(CONTRACT_PATH),
            "--out", str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "hard link" in completed.stderr
    assert evidence.read_bytes() == before


def test_cli_rejects_duplicate_contract_json_keys(tmp_path):
    evidence = tmp_path / "evidence.csv"
    contract = tmp_path / "duplicate-contract.json"
    _write_csv(evidence, _supported_rows())
    contract.write_text(
        '{"schema_version":"first","schema_version":"second"}\n',
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--evidence-csv", str(evidence),
            "--contract", str(contract),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "duplicate key 'schema_version'" in completed.stderr
