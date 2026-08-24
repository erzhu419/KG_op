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

from performance.theory_operation_competition import (  # noqa: E402
    SCHEMA_VERSION,
    CompetitionDisposition,
    ContractValidationError,
    EvidenceValidationError,
    OperationKind,
    canonical_json_bytes,
    run_theory_operation_competition,
    synthesize_theory_operation_candidates,
    validate_contract,
    verify_theory_operation_competition,
)


RUNNER = ROOT / "runners/run_theory_operation_competition.py"
CORE = ROOT / "performance/theory_operation_competition.py"
CONTRACT = (
    ROOT
    / "performance/manifests/theory_operation_competition_v1.json"
)
OR_BASELINE_SOURCE = ROOT / "performance/benchmark_lodo_meta_prior.py"


def _contract():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _context(x, nuisance):
    return {"x": x, "nuisance": nuisance}


def _row(split, index, context, observed_value):
    return {
        "observation_id": f"{split}-{index:02d}",
        "evaluator_epoch": "evaluator-epoch-1",
        "fixed_anchor": "fixed-anchor-1",
        "scope_id": "registered-scope",
        "context": copy.deepcopy(context),
        "observed_value": observed_value,
    }


def _case(kind):
    if kind == "robustification":
        prediction_values = (0.0, 10.0, 2.0, 12.0)
        residuals = (-1.0, -0.3, 0.3, 1.0)
        discovery_values = tuple(
            value + residual
            for value, residual in zip(prediction_values, residuals)
        )
        heldout_values = discovery_values
        case_id = "kat-select-robustification"
    elif kind == "idealization":
        prediction_values = (-0.1, 0.1, 0.9, 1.1)
        discovery_values = prediction_values
        heldout_values = (0.0, 0.0, 1.0, 1.0)
        case_id = "kat-select-idealization"
    else:
        raise AssertionError(f"unknown fixture kind: {kind}")

    contexts = [
        _context(0, 0),
        _context(0, 1),
        _context(1, 0),
        _context(1, 1),
    ]
    predictions = [
        {"context": copy.deepcopy(context), "value": value}
        for context, value in zip(contexts, prediction_values)
    ]

    evidence = {}
    for split in ("discovery", "validation", "stress"):
        observed_values = (
            discovery_values if split == "discovery" else heldout_values
        )
        evidence[split] = [
            _row(split, 2 * index + repeat, context, observed_value)
            for index, (context, observed_value) in enumerate(
                zip(contexts, observed_values)
            )
            for repeat in range(2)
        ]

    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "theory_state": {
            "theory_id": "finite-table-theory-1",
            "task_id": "bounded-regression-task-1",
            "evaluator_epoch": "evaluator-epoch-1",
            "fixed_anchor": "fixed-anchor-1",
            "object_space": {
                "feature_ids": ["x", "nuisance"],
                "contexts": contexts,
            },
            "model_class": {
                "kind": "finite_point_table",
                "predictions": predictions,
            },
            "probe_ids": ["absolute_error_point_prediction"],
            "violation_functionals": [
                {
                    "functional_id": "absolute_error",
                    "threshold": 0.2,
                }
            ],
            "scope_ids": ["registered-scope"],
            "removable_feature_ids": ["nuisance"],
        },
        "evidence": evidence,
    }


def _report(case):
    return run_theory_operation_competition(case, _contract()).to_dict()


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_digest(value):
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _write_json(path, value):
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _run_cli(input_path, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--input",
            str(input_path),
            "--contract",
            str(CONTRACT.resolve()),
            *map(str, extra),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )


def _candidate_commitments(report):
    return report["candidate_commitments"]


def _parent_prediction(case, context):
    for item in case["theory_state"]["model_class"]["predictions"]:
        if item["context"] == context:
            return item["value"]
    raise AssertionError("fixture context has no parent prediction")


def test_frozen_public_api_and_contract_vocabulary():
    contract = _contract()

    assert SCHEMA_VERSION == contract["input_schema_version"]
    assert validate_contract(contract) == contract
    assert [item.value for item in OperationKind] == contract["operation_kinds"]
    assert {item.value for item in CompetitionDisposition} == {
        "SELECT_ROBUSTIFICATION",
        "SELECT_IDEALIZATION",
        "NEEDS_EVIDENCE",
        "INCOMPARABLE_EVALUATOR_EPOCH",
    }

    changed = copy.deepcopy(contract)
    changed["thresholds"]["min_score_margin"] = -1
    with pytest.raises(ContractValidationError):
        validate_contract(changed)


def test_computed_dataset_selects_robustification():
    report = _report(_case("robustification"))

    assert report["disposition"] == "SELECT_ROBUSTIFICATION"
    assert report["selected_candidate"]["family"] == "robust_interval"
    assert report["selected_candidate"]["mechanism"] == "interval"
    assert report["promotion_status"] == "NOT_PROMOTED_SHADOW_ONLY"


def test_computed_dataset_selects_idealization_and_has_nine_field_contract():
    contract = _contract()
    report = _report(_case("idealization"))

    assert report["disposition"] == "SELECT_IDEALIZATION"
    assert report["selected_candidate"]["family"] == "idealization_quotient"
    assert report["selected_candidate"]["mechanism"] == "quotient"
    idealization = report["selected_candidate"]["idealization_contract"]
    assert set(idealization) == set(contract["idealization_contract_fields"])
    assert idealization["deleted_degrees_of_freedom"] == ["nuisance"]
    recovery = idealization["full_model_recovery_method"]
    assert recovery["kind"] == "restore_frozen_parent_point_table"
    assert recovery["lossy_quotient_requires_parent_snapshot"] is True
    assert recovery["parent_theory_state_digest"] == report[
        "theory_state_digest"
    ]
    assert recovery["parent_predictions"] == report["theory_state"][
        "model_class"
    ]["predictions"]
    fiber_map = recovery["quotient_fiber_map"]
    assert len(fiber_map) == len(
        report["theory_state"]["object_space"]["contexts"]
    )
    assert {
        canonical_json_bytes(item["parent_context"]) for item in fiber_map
    } == {
        canonical_json_bytes(context)
        for context in report["theory_state"]["object_space"]["contexts"]
    }


def test_robust_radii_are_exact_discovery_residual_statistics():
    report = _report(_case("idealization"))

    assert report["robustification_candidates"]
    for candidate in report["robustification_candidates"]:
        assert all(
            item["radius"] == pytest.approx(0.0)
            for item in candidate["radii"]
        )


def test_global_bias_is_not_mislabeled_as_robustification():
    case = _case("robustification")
    for rows in case["evidence"].values():
        for row in rows:
            row["observed_value"] = _parent_prediction(
                case, row["context"]
            ) + 1.0

    report = _report(case)

    assert report["disposition"] == "NEEDS_EVIDENCE"
    assert report["selected_candidate"] is None
    assert report["diagnostic_trace"][0]["stage"] == "reestimate"
    assert report["diagnostic_trace"][0]["disposition"] != (
        "EXCLUDED_BY_DISCOVERY"
    )
    assert next(
        item
        for item in report["diagnostic_trace"]
        if item["stage"] == "robustify"
    )["disposition"] == "BLOCKED_BY_EARLIER_DIAGNOSIS"


def test_cross_epoch_is_incomparable_without_pooling_numbers():
    case = _case("robustification")
    case["evidence"]["discovery"][0]["evaluator_epoch"] = (
        "evaluator-epoch-2"
    )

    report = _report(case)

    assert report["disposition"] == "INCOMPARABLE_EVALUATOR_EPOCH"
    assert report["selected_candidate"] is None
    assert report["candidate_commitments"]["robustification_count"] == 0
    assert report["candidate_commitments"]["idealization_count"] == 0
    assert report["robustification_candidates"] == []
    assert report["idealization_candidates"] == []
    assert report["baseline_metrics"] == {
        "validation": None,
        "stress": None,
    }


def test_direct_synthesis_rejects_discovery_from_another_epoch():
    case = _case("robustification")
    case["evidence"]["discovery"][0]["evaluator_epoch"] = (
        "evaluator-epoch-2"
    )

    with pytest.raises(EvidenceValidationError):
        synthesize_theory_operation_candidates(
            case["theory_state"],
            case["evidence"]["discovery"],
            _contract(),
        )


def test_full_object_space_counterexample_blocks_hidden_idealization():
    case = _case("idealization")
    contexts = case["theory_state"]["object_space"]["contexts"]
    prediction_values = (0.0, 10.0, 0.0, 0.0)
    case["theory_state"]["model_class"]["predictions"] = [
        {"context": copy.deepcopy(context), "value": value}
        for context, value in zip(contexts, prediction_values)
    ]
    for row in case["evidence"]["discovery"]:
        row["observed_value"] = _parent_prediction(case, row["context"])
    for split in ("validation", "stress"):
        case["evidence"][split] = [
            _row(split, repeat, contexts[2 + repeat % 2], 0.0)
            for repeat in range(4)
        ]

    report = _report(case)

    assert report["disposition"] == "NEEDS_EVIDENCE"
    assert report["selected_candidate"] is None
    assert max(
        item["idealization_contract"]["approximation_error"]
        for item in report["idealization_candidates"]
    ) == pytest.approx(5.0)


def test_unimplemented_probe_and_uncovered_registered_scope_fail_closed():
    unsupported_probe = _case("idealization")
    unsupported_probe["theory_state"]["probe_ids"].append(
        "unimplemented_causal_effect"
    )
    with pytest.raises(EvidenceValidationError):
        _report(unsupported_probe)

    uncovered_scope = _case("idealization")
    uncovered_scope["theory_state"]["scope_ids"].append("unseen-scope")
    report = _report(uncovered_scope)
    assert report["disposition"] == "NEEDS_EVIDENCE"
    assert report["selected_candidate"] is None
    assert report["evidence_coverage"] == {
        "scope_ids_by_split": {
            split: ["registered-scope"]
            for split in ("discovery", "validation", "stress")
        },
        "missing_scope_ids_by_split": {
            split: ["unseen-scope"]
            for split in ("discovery", "validation", "stress")
        },
        "all_registered_scopes_in_every_split": False,
    }
    assert report["next_probe_spec"]["reason"] == (
        "REGISTERED_SCOPE_COVERAGE_INCOMPLETE"
    )


def test_insufficient_splits_need_evidence():
    case = _case("robustification")
    case["evidence"] = {
        split: rows[:1] for split, rows in case["evidence"].items()
    }

    report = _report(case)

    assert report["disposition"] == "NEEDS_EVIDENCE"
    assert report["selected_candidate"] is None


def test_heldout_rows_do_not_enter_candidate_synthesis():
    original = _case("idealization")
    changed = copy.deepcopy(original)
    for split in ("validation", "stress"):
        for index, row in enumerate(changed["evidence"][split]):
            row["observed_value"] = 1000.0 + index

    original_report = _report(original)
    changed_report = _report(changed)

    assert _candidate_commitments(original_report)
    assert _candidate_commitments(original_report) == _candidate_commitments(
        changed_report
    )


def test_verifier_replays_and_rejects_report_tampering():
    case = _case("idealization")
    contract = _contract()
    artifacts = {
        "case_json": {
            "sha256": "sha256:" + hashlib.sha256(
                canonical_json_bytes(case)
            ).hexdigest()
        }
    }
    report = run_theory_operation_competition(
        case, contract, input_artifacts=artifacts
    ).to_dict()

    verified = verify_theory_operation_competition(
        case,
        contract,
        report,
        expected_contract_digest=_json_digest(contract),
        expected_report_digest=report["report_digest"],
        expected_input_artifacts=artifacts,
    )
    assert isinstance(verified, dict)

    tampered = copy.deepcopy(report)
    tampered["disposition"] = "SELECT_ROBUSTIFICATION"
    with pytest.raises(EvidenceValidationError):
        verify_theory_operation_competition(
            case,
            contract,
            tampered,
            expected_contract_digest=_json_digest(contract),
            expected_report_digest=report["report_digest"],
            expected_input_artifacts=artifacts,
        )

    with pytest.raises(EvidenceValidationError):
        verify_theory_operation_competition(
            case,
            contract,
            report,
            expected_contract_digest=_json_digest(contract),
            expected_report_digest=report["report_digest"],
            expected_input_artifacts={"case_json": {"sha256": "wrong"}},
        )

    with pytest.raises(EvidenceValidationError):
        verify_theory_operation_competition(
            case,
            contract,
            report,
            expected_contract_digest="sha256:" + "0" * 64,
            expected_report_digest=report["report_digest"],
            expected_input_artifacts=artifacts,
        )


def test_forged_artifact_map_is_not_self_authenticating_even_with_new_digest():
    case = _case("idealization")
    contract = _contract()
    forged = _report(case)
    forged["input_artifacts"] = {"forged": {"sha256": "attacker-value"}}
    forged_body = {
        key: value for key, value in forged.items() if key != "report_digest"
    }
    forged["report_digest"] = "sha256:" + hashlib.sha256(
        canonical_json_bytes(forged_body)
    ).hexdigest()

    with pytest.raises(EvidenceValidationError):
        verify_theory_operation_competition(
            case,
            contract,
            forged,
            expected_contract_digest=_json_digest(contract),
            expected_report_digest=forged["report_digest"],
            expected_input_artifacts=None,
        )


def test_semantic_result_is_invariant_to_input_order():
    case = _case("idealization")
    reordered = copy.deepcopy(case)
    reordered["theory_state"]["object_space"]["contexts"].reverse()
    reordered["theory_state"]["model_class"]["predictions"].reverse()
    reordered["theory_state"]["probe_ids"].reverse()
    for rows in reordered["evidence"].values():
        rows.reverse()

    first = _report(case)
    second = _report(reordered)

    assert first["disposition"] == second["disposition"]
    assert first["selected_candidate"] == second["selected_candidate"]
    assert first["candidate_commitments"] == second["candidate_commitments"]
    assert first["diagnostic_trace"] == second["diagnostic_trace"]


def test_duplicate_semantic_ids_and_nonfinite_values_fail_closed():
    duplicate = _case("idealization")
    duplicate["evidence"]["validation"][1]["observation_id"] = (
        duplicate["evidence"]["validation"][0]["observation_id"]
    )
    with pytest.raises(EvidenceValidationError):
        _report(duplicate)

    nonfinite = _case("idealization")
    nonfinite["theory_state"]["model_class"]["predictions"][0][
        "value"
    ] = float("nan")
    with pytest.raises(EvidenceValidationError):
        _report(nonfinite)


def test_language_is_last_and_never_selected_by_v1():
    report = _report(_case("idealization"))

    stages = [entry["stage"] for entry in report["diagnostic_trace"]]
    assert stages == _contract()["diagnostic_order"]
    assert stages[-1] == "language_last"
    assert report["selected_candidate"]["operation_kind"] != "language"


def test_cli_stdout_is_canonical_and_optional_out_is_identical_atomic_copy(
    tmp_path,
):
    input_path = (tmp_path / "case.json").resolve()
    output_path = (tmp_path / "report.json").resolve()
    _write_json(input_path, _case("idealization"))
    before_input = input_path.read_bytes()
    before_contract = CONTRACT.read_bytes()

    completed = _run_cli(input_path, "--out", output_path)

    assert completed.returncode == 0, completed.stderr.decode()
    parsed = json.loads(completed.stdout)
    assert completed.stdout == canonical_json_bytes(parsed) + b"\n"
    assert output_path.read_bytes() == completed.stdout
    assert input_path.read_bytes() == before_input
    assert CONTRACT.read_bytes() == before_contract


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema_version":"a","schema_version":"b"}',
        b'{"x":NaN}',
    ],
)
def test_cli_rejects_duplicate_keys_and_nonfinite_json_without_stdout(
    tmp_path, raw
):
    input_path = (tmp_path / "invalid.json").resolve()
    input_path.write_bytes(raw)

    completed = _run_cli(input_path)

    assert completed.returncode == 2
    assert completed.stdout == b""


def test_cli_refuses_hardlinked_output_and_preserves_input(tmp_path):
    input_path = (tmp_path / "case.json").resolve()
    output_path = (tmp_path / "hardlink.json").resolve()
    _write_json(input_path, _case("idealization"))
    before = input_path.read_bytes()
    os.link(input_path, output_path)

    completed = _run_cli(input_path, "--out", output_path)

    assert completed.returncode == 2
    assert completed.stdout == b""
    assert input_path.read_bytes() == before
    assert output_path.read_bytes() == before


def test_surface_neither_imports_nor_calls_old_execution_and_mutates_nothing():
    before = _digest(OR_BASELINE_SOURCE)

    forbidden_import = "performance.benchmark_lodo_meta_prior"
    for path in (CORE, RUNNER):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        called.update(
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        )
        assert forbidden_import not in imported
        assert "run_one" not in called

    _report(_case("robustification"))

    assert _digest(OR_BASELINE_SOURCE) == before
