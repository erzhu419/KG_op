import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.shadow_theory_transition import (  # noqa: E402
    ShadowTransitionDisposition,
    ShadowTransitionKind,
    ShadowTransitionResult,
    ShadowTransitionValidationError,
    canonical_json_bytes,
    materialize_shadow_theory_transition,
    validate_shadow_transition_contract,
    verify_shadow_theory_transition,
)
from performance.theory_operation_competition import (  # noqa: E402
    SCHEMA_VERSION as COMPETITION_INPUT_SCHEMA,
    run_theory_operation_competition,
)


COMPETITION_CONTRACT = (
    ROOT / "performance/manifests/theory_operation_competition_v1.json"
)
TRANSITION_CONTRACT = (
    ROOT / "performance/manifests/shadow_theory_transition_v1.json"
)
COMPETITION_RUNNER = ROOT / "runners/run_theory_operation_competition.py"
TRANSITION_RUNNER = ROOT / "runners/run_shadow_theory_transition.py"
TRANSITION_CORE = ROOT / "performance/shadow_theory_transition.py"
OLD_BENCHMARK = ROOT / "performance/benchmark_lodo_meta_prior.py"
FIRST_SLICE_FILES = (
    ROOT / "performance/theory_operation_competition.py",
    COMPETITION_CONTRACT,
    COMPETITION_RUNNER,
    ROOT / "tests/test_theory_operation_competition.py",
    ROOT / "docs/theory_operation_competition_v1.md",
)

REPORT_KEYS = {
    "schema_version",
    "contract_id",
    "contract_digest",
    "source_competition",
    "parent_theory_state",
    "parent_theory_state_digest",
    "disposition",
    "operation_kind",
    "transition_kind",
    "selected_candidate_id",
    "child_theory_state",
    "child_theory_state_digest",
    "preservation_certificate",
    "reduction_certificate",
    "evaluator_gate",
    "adoption_status",
    "nonclaims",
    "input_artifacts",
    "audit_events",
    "audit_head",
    "report_digest",
}

CHILD_KEYS = {
    "schema_version",
    "theory_id",
    "task_id",
    "evaluator_epoch",
    "evaluator_status",
    "fixed_anchor",
    "object_space",
    "model_class",
    "probe_ids",
    "violation_functionals",
    "scope_ids",
    "removable_feature_ids",
    "evidence_reuse_policy",
    "operational_probe_status",
    "transition_lineage",
}


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest_value(value):
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context(x, nuisance):
    return {"x": x, "nuisance": nuisance}


def _row(split, index, context, observed):
    return {
        "observation_id": f"{split}-{index:02d}",
        "evaluator_epoch": "evaluator-epoch-1",
        "fixed_anchor": "fixed-anchor-1",
        "scope_id": "registered-scope",
        "context": copy.deepcopy(context),
        "observed_value": observed,
    }


def _case(kind):
    if kind == "robustification":
        predictions = (0.0, 10.0, 2.0, 12.0)
        residuals = (-1.0, -0.3, 0.3, 1.0)
        discovery = tuple(a + b for a, b in zip(predictions, residuals))
        heldout = discovery
    elif kind == "idealization":
        predictions = (-0.1, 0.1, 0.9, 1.1)
        discovery = predictions
        heldout = (0.0, 0.0, 1.0, 1.0)
    else:
        raise AssertionError(kind)
    contexts = [_context(0, 0), _context(0, 1), _context(1, 0), _context(1, 1)]
    evidence = {}
    for split in ("discovery", "validation", "stress"):
        values = discovery if split == "discovery" else heldout
        evidence[split] = [
            _row(split, 2 * index + repeat, context, value)
            for index, (context, value) in enumerate(zip(contexts, values))
            for repeat in range(2)
        ]
    return {
        "schema_version": COMPETITION_INPUT_SCHEMA,
        "case_id": f"transition-{kind}",
        "theory_state": {
            "theory_id": "parent-finite-theory",
            "task_id": "bounded-regression-task",
            "evaluator_epoch": "evaluator-epoch-1",
            "fixed_anchor": "fixed-anchor-1",
            "object_space": {
                "feature_ids": ["x", "nuisance"],
                "contexts": contexts,
            },
            "model_class": {
                "kind": "finite_point_table",
                "predictions": [
                    {"context": copy.deepcopy(context), "value": value}
                    for context, value in zip(contexts, predictions)
                ],
            },
            "probe_ids": ["absolute_error_point_prediction"],
            "violation_functionals": [
                {"functional_id": "absolute_error", "threshold": 0.2}
            ],
            "scope_ids": ["registered-scope"],
            "removable_feature_ids": ["nuisance"],
        },
        "evidence": evidence,
    }


def _competition(case):
    contract = _load(COMPETITION_CONTRACT)
    report = run_theory_operation_competition(case, contract).to_dict()
    return contract, report


def _materialize(case):
    competition_contract, competition_report = _competition(case)
    transition_contract = _load(TRANSITION_CONTRACT)
    report = materialize_shadow_theory_transition(
        case,
        competition_contract,
        competition_report,
        transition_contract,
        expected_competition_contract_digest=_digest_value(competition_contract),
        expected_competition_report_digest=competition_report["report_digest"],
        expected_competition_input_artifacts=None,
        expected_transition_contract_digest=_digest_value(transition_contract),
        input_artifacts=None,
    ).to_dict()
    return competition_contract, competition_report, transition_contract, report


def _write(path, value):
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def test_contract_and_robust_child_exact_reduction():
    contract = _load(TRANSITION_CONTRACT)
    assert validate_shadow_transition_contract(contract) == contract
    _, source, _, report = _materialize(_case("robustification"))

    assert set(report) == REPORT_KEYS
    assert report["disposition"] == "MATERIALIZED_SHADOW_ROBUSTIFICATION"
    assert report["operation_kind"] == "expand"
    assert report["transition_kind"] == "ROBUST_INTERVAL_EXPANSION"
    child = report["child_theory_state"]
    assert set(child) == CHILD_KEYS
    assert child["model_class"] == {
        "kind": "finite_interval_table",
        "center_predictions": report["parent_theory_state"]["model_class"][
            "predictions"
        ],
        "radius_grouping": source["selected_candidate"]["grouping"],
        "radii": source["selected_candidate"]["radii"],
    }
    assert child["evaluator_epoch"] is None
    assert child["fixed_anchor"] == "fixed-anchor-1"
    assert child["evidence_reuse_policy"] == {
        "source_evidence_role": "QUALIFICATION_ONLY",
        "source_evidence_allowed_for_child_scoring": False,
        "old_new_records_pooled": False,
    }
    assert report["preservation_certificate"]["certificate_kind"] == (
        "EXACT_CENTER_CONSERVATIVE_EXTENSION"
    )
    assert report["preservation_certificate"][
        "operational_probe_preservation_certified"
    ] is False
    assert report["reduction_certificate"]["map_kind"] == (
        "COLLAPSE_INTERVAL_AT_RADIUS_MULTIPLIER_ZERO"
    )
    assert report["reduction_certificate"][
        "exact_parent_model_class_recovered"
    ] is True


def test_ideal_child_recomputes_quotient_and_snapshot_recovery():
    _, source, _, report = _materialize(_case("idealization"))

    assert report["disposition"] == "MATERIALIZED_SHADOW_IDEALIZATION"
    assert report["operation_kind"] == "quotient"
    assert report["transition_kind"] == "QUOTIENT_IDEALIZATION"
    child = report["child_theory_state"]
    assert set(report) == REPORT_KEYS
    assert set(child) == CHILD_KEYS
    assert child["object_space"]["feature_ids"] == ["x"]
    assert child["model_class"]["kind"] == "finite_point_table"
    assert child["model_class"]["predictions"] == source["selected_candidate"][
        "quotient_predictions"
    ]
    assert child["removable_feature_ids"] == []
    preservation = report["preservation_certificate"]
    assert preservation["certificate_kind"] == "FINITE_PREDICTION_PROXY_BOUND"
    assert preservation["within_bound"] is True
    assert preservation["operational_probe_preservation_certified"] is False
    reduction = report["reduction_certificate"]
    assert reduction["map_kind"] == (
        "QUOTIENT_PROJECTION_WITH_FROZEN_PARENT_SNAPSHOT"
    )
    assert reduction["exact_parent_recovery_from_snapshot_verified"] is True
    assert reduction["lossy_quotient_requires_parent_snapshot"] is True
    assert reduction["quotient_alone_recovers_parent"] is False
    assert len(reduction["quotient_fiber_map"]) == 4


def test_public_types_and_exact_enum_values():
    assert {item.value for item in ShadowTransitionKind} == {
        "ROBUST_INTERVAL_EXPANSION",
        "QUOTIENT_IDEALIZATION",
    }
    assert {item.value for item in ShadowTransitionDisposition} == {
        "MATERIALIZED_SHADOW_ROBUSTIFICATION",
        "MATERIALIZED_SHADOW_IDEALIZATION",
        "NOT_MATERIALIZED_NEEDS_EVIDENCE",
        "NOT_MATERIALIZED_INCOMPARABLE_EVALUATOR_EPOCH",
    }
    _, _, _, report = _materialize(_case("robustification"))
    result = ShadowTransitionResult(report=report)
    assert result.disposition == report["disposition"]
    assert result.report_digest == report["report_digest"]
    assert result.to_dict() == report
    assert result.to_dict() is not report


def test_unassigned_evaluator_gate_and_child_id_are_deterministic():
    case = _case("idealization")
    _, _, _, first = _materialize(case)
    reordered = copy.deepcopy(case)
    reordered["theory_state"]["object_space"]["contexts"].reverse()
    reordered["theory_state"]["model_class"]["predictions"].reverse()
    for rows in reordered["evidence"].values():
        rows.reverse()
    _, _, _, second = _materialize(reordered)

    child = first["child_theory_state"]
    payload = {key: value for key, value in child.items() if key != "theory_id"}
    assert child["theory_id"] == "shadow-theory:" + hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    assert child["theory_id"] == second["child_theory_state"]["theory_id"]
    assert first["child_theory_state_digest"] == second[
        "child_theory_state_digest"
    ]
    assert first["evaluator_gate"]["adoption_blocked"] is True
    assert first["evaluator_gate"]["selective_erasure_applied"] is False
    assert first["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"


def test_nonselected_sources_produce_no_child_or_certificates():
    bias = _case("robustification")
    lookup = {
        canonical_json_bytes(item["context"]): item["value"]
        for item in bias["theory_state"]["model_class"]["predictions"]
    }
    for rows in bias["evidence"].values():
        for row in rows:
            row["observed_value"] = lookup[canonical_json_bytes(row["context"])] + 1
    _, _, _, needs = _materialize(bias)
    assert needs["disposition"] == "NOT_MATERIALIZED_NEEDS_EVIDENCE"

    cross = _case("robustification")
    cross["evidence"]["stress"][0]["evaluator_epoch"] = "another-epoch"
    _, _, _, incomparable = _materialize(cross)
    assert incomparable["disposition"] == (
        "NOT_MATERIALIZED_INCOMPARABLE_EVALUATOR_EPOCH"
    )
    for report in (needs, incomparable):
        for key in (
            "operation_kind",
            "transition_kind",
            "selected_candidate_id",
            "child_theory_state",
            "child_theory_state_digest",
            "preservation_certificate",
            "reduction_certificate",
            "evaluator_gate",
        ):
            assert report[key] is None


def test_verifier_exact_replay_and_all_independent_digest_gates():
    case = _case("idealization")
    competition_contract, competition_report, transition_contract, report = (
        _materialize(case)
    )
    verified = verify_shadow_theory_transition(
        case,
        competition_contract,
        competition_report,
        transition_contract,
        report,
        expected_competition_contract_digest=_digest_value(competition_contract),
        expected_competition_report_digest=competition_report["report_digest"],
        expected_competition_input_artifacts=None,
        expected_transition_contract_digest=_digest_value(transition_contract),
        expected_transition_report_digest=report["report_digest"],
        expected_transition_input_artifacts=None,
    )
    assert verified["status"] == "VERIFIED_MATERIALIZED_SHADOW_IDEALIZATION"

    for path, forged_value in (
        (("child_theory_state", "evaluator_status"), "forged-status"),
        (("preservation_certificate", "within_bound"), False),
        (("reduction_certificate", "quotient_alone_recovers_parent"), True),
        (("evaluator_gate", "adoption_blocked"), False),
    ):
        tampered = copy.deepcopy(report)
        target = tampered
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = forged_value
        tampered["report_digest"] = _digest_value(
            {key: value for key, value in tampered.items() if key != "report_digest"}
        )
        with pytest.raises(ShadowTransitionValidationError):
            verify_shadow_theory_transition(
                case,
                competition_contract,
                competition_report,
                transition_contract,
                tampered,
                expected_competition_contract_digest=_digest_value(
                    competition_contract
                ),
                expected_competition_report_digest=competition_report[
                    "report_digest"
                ],
                expected_competition_input_artifacts=None,
                expected_transition_contract_digest=_digest_value(
                    transition_contract
                ),
                expected_transition_report_digest=tampered["report_digest"],
                expected_transition_input_artifacts=None,
            )

    for changed_key in (
        "expected_competition_contract_digest",
        "expected_competition_report_digest",
        "expected_transition_contract_digest",
    ):
        kwargs = {
            "expected_competition_contract_digest": _digest_value(
                competition_contract
            ),
            "expected_competition_report_digest": competition_report[
                "report_digest"
            ],
            "expected_competition_input_artifacts": None,
            "expected_transition_contract_digest": _digest_value(
                transition_contract
            ),
            "input_artifacts": None,
        }
        kwargs[changed_key] = "sha256:" + "0" * 64
        with pytest.raises(ShadowTransitionValidationError):
            materialize_shadow_theory_transition(
                case,
                competition_contract,
                competition_report,
                transition_contract,
                **kwargs,
            )


def test_forged_source_self_hash_and_independent_artifact_mismatches_fail_closed():
    case = _case("idealization")
    competition_contract, competition_report = _competition(case)
    transition_contract = _load(TRANSITION_CONTRACT)

    forged = copy.deepcopy(competition_report)
    forged["selected_candidate"]["candidate_id"] = "forged-candidate"
    forged["report_digest"] = _digest_value(
        {key: value for key, value in forged.items() if key != "report_digest"}
    )
    with pytest.raises(ShadowTransitionValidationError):
        materialize_shadow_theory_transition(
            case,
            competition_contract,
            forged,
            transition_contract,
            expected_competition_contract_digest=_digest_value(
                competition_contract
            ),
            expected_competition_report_digest=forged["report_digest"],
            expected_competition_input_artifacts=None,
            expected_transition_contract_digest=_digest_value(
                transition_contract
            ),
            input_artifacts=None,
        )

    with pytest.raises(ShadowTransitionValidationError):
        materialize_shadow_theory_transition(
            case,
            competition_contract,
            competition_report,
            transition_contract,
            expected_competition_contract_digest=_digest_value(
                competition_contract
            ),
            expected_competition_report_digest=competition_report[
                "report_digest"
            ],
            expected_competition_input_artifacts={"forged": True},
            expected_transition_contract_digest=_digest_value(
                transition_contract
            ),
            input_artifacts=None,
        )

    artifacts = {"transition_contract_json": {"sha256": "a" * 64}}
    report = materialize_shadow_theory_transition(
        case,
        competition_contract,
        competition_report,
        transition_contract,
        expected_competition_contract_digest=_digest_value(competition_contract),
        expected_competition_report_digest=competition_report["report_digest"],
        expected_competition_input_artifacts=None,
        expected_transition_contract_digest=_digest_value(transition_contract),
        input_artifacts=artifacts,
    ).to_dict()
    with pytest.raises(ShadowTransitionValidationError):
        verify_shadow_theory_transition(
            case,
            competition_contract,
            competition_report,
            transition_contract,
            report,
            expected_competition_contract_digest=_digest_value(
                competition_contract
            ),
            expected_competition_report_digest=competition_report[
                "report_digest"
            ],
            expected_competition_input_artifacts=None,
            expected_transition_contract_digest=_digest_value(
                transition_contract
            ),
            expected_transition_report_digest=report["report_digest"],
            expected_transition_input_artifacts={"forged": True},
        )


def test_cli_canonical_stdout_and_identical_atomic_out(tmp_path):
    case_path = (tmp_path / "case.json").resolve()
    competition_report_path = (tmp_path / "competition.json").resolve()
    transition_report_path = (tmp_path / "transition.json").resolve()
    _write(case_path, _case("idealization"))
    competition_contract = _load(COMPETITION_CONTRACT)
    source = subprocess.run(
        [
            sys.executable,
            str(COMPETITION_RUNNER),
            "--input",
            str(case_path),
            "--contract",
            str(COMPETITION_CONTRACT.resolve()),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert source.returncode == 0, source.stderr.decode()
    competition_report_path.write_bytes(source.stdout)
    source_report = json.loads(source.stdout)
    transition_contract = _load(TRANSITION_CONTRACT)

    completed = subprocess.run(
        [
            sys.executable,
            str(TRANSITION_RUNNER),
            "--competition-input",
            str(case_path),
            "--competition-contract",
            str(COMPETITION_CONTRACT.resolve()),
            "--competition-report",
            str(competition_report_path),
            "--transition-contract",
            str(TRANSITION_CONTRACT.resolve()),
            "--expected-competition-contract-digest",
            _digest_value(competition_contract),
            "--expected-competition-report-digest",
            source_report["report_digest"],
            "--expected-transition-contract-digest",
            _digest_value(transition_contract),
            "--out",
            str(transition_report_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    parsed = json.loads(completed.stdout)
    assert completed.stdout == canonical_json_bytes(parsed) + b"\n"
    assert transition_report_path.read_bytes() == completed.stdout


def test_cli_rejects_hardlink_alias_and_surface_imports_no_execution(tmp_path):
    case_path = (tmp_path / "case.json").resolve()
    report_path = (tmp_path / "report.json").resolve()
    _write(case_path, _case("idealization"))
    os.link(case_path, report_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(TRANSITION_RUNNER),
            "--competition-input",
            str(case_path),
            "--competition-report",
            str(case_path),
            "--expected-competition-contract-digest",
            "sha256:" + "0" * 64,
            "--expected-competition-report-digest",
            "sha256:" + "0" * 64,
            "--expected-transition-contract-digest",
            "sha256:" + "0" * 64,
            "--out",
            str(report_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert completed.stdout == b""

    for path in (TRANSITION_CORE, TRANSITION_RUNNER):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "performance.benchmark_lodo_meta_prior" not in imports
        assert "run_one" not in calls


@pytest.mark.parametrize(
    "raw",
    (
        b'{"case_id":"a","case_id":"b"}\n',
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
    ),
)
def test_cli_rejects_duplicate_and_nonfinite_json(raw, tmp_path):
    case_path = (tmp_path / "invalid.json").resolve()
    report_path = (tmp_path / "report.json").resolve()
    case_path.write_bytes(raw)
    report_path.write_text("{}\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(TRANSITION_RUNNER),
            "--competition-input",
            str(case_path),
            "--competition-contract",
            str(COMPETITION_CONTRACT.resolve()),
            "--competition-report",
            str(report_path),
            "--transition-contract",
            str(TRANSITION_CONTRACT.resolve()),
            "--expected-competition-contract-digest",
            "sha256:" + "0" * 64,
            "--expected-competition-report-digest",
            "sha256:" + "0" * 64,
            "--expected-transition-contract-digest",
            "sha256:" + "0" * 64,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert completed.stdout == b""


def test_cli_rejects_symlink_input(tmp_path):
    real_case = (tmp_path / "case.json").resolve()
    linked_case = (tmp_path / "case-link.json").resolve()
    report_path = (tmp_path / "report.json").resolve()
    _write(real_case, _case("idealization"))
    linked_case.symlink_to(real_case)
    report_path.write_text("{}\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(TRANSITION_RUNNER),
            "--competition-input",
            str(linked_case),
            "--competition-contract",
            str(COMPETITION_CONTRACT.resolve()),
            "--competition-report",
            str(report_path),
            "--transition-contract",
            str(TRANSITION_CONTRACT.resolve()),
            "--expected-competition-contract-digest",
            "sha256:" + "0" * 64,
            "--expected-competition-report-digest",
            "sha256:" + "0" * 64,
            "--expected-transition-contract-digest",
            "sha256:" + "0" * 64,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert completed.stdout == b""


def test_cli_rejects_symlink_output_and_relative_input(tmp_path):
    case_path = (tmp_path / "case.json").resolve()
    report_path = (tmp_path / "report.json").resolve()
    out_link = (tmp_path / "out-link.json").resolve()
    _write(case_path, _case("idealization"))
    report_path.write_text("{}\n", encoding="utf-8")
    out_link.symlink_to(case_path)

    common = [
        sys.executable,
        str(TRANSITION_RUNNER),
        "--competition-contract",
        str(COMPETITION_CONTRACT.resolve()),
        "--competition-report",
        str(report_path),
        "--transition-contract",
        str(TRANSITION_CONTRACT.resolve()),
        "--expected-competition-contract-digest",
        "sha256:" + "0" * 64,
        "--expected-competition-report-digest",
        "sha256:" + "0" * 64,
        "--expected-transition-contract-digest",
        "sha256:" + "0" * 64,
    ]
    symlink_out = subprocess.run(
        [
            *common,
            "--competition-input",
            str(case_path),
            "--out",
            str(out_link),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert symlink_out.returncode == 2
    assert symlink_out.stdout == b""

    relative_input = subprocess.run(
        [*common, "--competition-input", case_path.name],
        cwd=tmp_path,
        check=False,
        capture_output=True,
    )
    assert relative_input.returncode == 2
    assert relative_input.stdout == b""


def test_first_slice_and_old_benchmark_bytes_are_unchanged():
    before = {path: _digest_file(path) for path in (*FIRST_SLICE_FILES, OLD_BENCHMARK)}
    _materialize(_case("robustification"))
    _materialize(_case("idealization"))
    assert {path: _digest_file(path) for path in before} == before
