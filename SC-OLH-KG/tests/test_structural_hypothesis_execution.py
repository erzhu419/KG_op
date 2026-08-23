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

from performance.structural_hypothesis_execution import (  # noqa: E402
    authorize_plan,
    build_execution_plan,
    execute_authorized_plan,
    reingest_successful_receipts,
    verify_authorization_integrity,
    verify_plan_integrity,
    verify_receipt_integrity,
    verify_reingestion_integrity,
)
from performance.structural_hypothesis_loop import (  # noqa: E402
    canonical_json_bytes,
    run_structural_hypothesis_loop,
    verify_report_integrity,
)


HYPOTHESIS_CONTRACT_PATH = (
    ROOT / "performance/manifests/structural_hypothesis_loop_v1.json"
)
EXECUTOR_CONTRACT_PATH = (
    ROOT / "performance/manifests/structural_hypothesis_executor_v1.json"
)
RUNNER = ROOT / "runners/run_structural_hypothesis_plan.py"
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


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value):
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _hypothesis_contract():
    return _json(HYPOTHESIS_CONTRACT_PATH)


def _executor_contract():
    return _json(EXECUTOR_CONTRACT_PATH)


def _rows(*, omit=("full",), poison=False):
    contract = _hypothesis_contract()
    scope = contract["evidence_scope"]
    rows = []
    for profile in PROFILES:
        if profile in omit:
            continue
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
                    "true_feasible": seed < 9,
                    "adaptive_loss": False,
                    "feasible_regret": "1.0",
                    **scope["fixed_row_values"],
                })
    if poison:
        rows[0]["hvd_profile"] = "unregistered"
    return rows


def _report(*, poison=False):
    result = run_structural_hypothesis_loop(
        _rows(poison=poison), _hypothesis_contract()
    )
    report = result.to_dict()
    assert verify_report_integrity(report)
    return report


def _build(report=None, materializer=None):
    return build_execution_plan(
        report or _report(),
        _hypothesis_contract(),
        _executor_contract(),
        task_materializer=materializer,
    )


def _materialize(cell):
    fixed = _executor_contract()["fixed_task_inputs"]
    return {
        "args": {
            **fixed,
            "d": cell["d"],
            "N": cell["N"],
            "n0": cell["n0"],
            "initial_design": "source_informed",
            # A real materializer must provide the complete raw template.  The
            # nested object makes this fake explicit and non-opaque for KATs.
            "test_only_raw_template": {
                "domain": cell["domain"],
                "source_design": [cell["seed"], cell["d"]],
            },
        },
        "heldout": cell["domain"],
        "line": cell["line"],
        "seed": cell["seed"],
    }


def _normalized_full_row(task):
    scope = _hypothesis_contract()["evidence_scope"]
    profile = "full"
    return {
        "track": scope["track"],
        "run_id": scope["run_id"],
        "variant": scope["variant_template"].format(profile=profile),
        "method": profile,
        "structural_prior_profile": profile,
        "domain": task["heldout"],
        "seed": task["seed"],
        "d": scope["d"],
        "N": scope["N"],
        "n0": scope["n0"],
        "source_calls": scope["source_calls"],
        "implementation": scope["implementation"],
        "initial_design": scope["initial_design"],
        "decision_backend": scope["decision_backend"],
        "status": "ok",
        "true_feasible": True,
        "adaptive_loss": False,
        "feasible_regret": 0.0,
        **scope["fixed_row_values"],
    }


def _raw_run_one_result(task):
    args = task["args"]
    return {
        "line": task["line"],
        "heldout": task["heldout"],
        "seed": task["seed"],
        "n_search_simulations": args["N"],
        "n_simulations": args["N"],
        "n0": args["n0"],
        "source_target_adaptation_contract": {
            "source_simulator_calls": 384,
        },
        "true_feasible": True,
        "feasible_simple_regret": 0.0,
        "adaptive_loss": False,
        "decision_backend": args["decision_backend"],
        "structural_prior_profile": args["structural_prior_profile"],
        "hvd_ablation_profile": args["hvd_ablation_profile"],
        "source_discrepancy_update": args["source_discrepancy_update"],
        "certification_recheck_top_k": args[
            "certification_recheck_top_k"
        ],
        "decision_risk_penalty": args["decision_risk_penalty"],
        "decision_source_utility_weight": args[
            "decision_source_utility_weight"
        ],
        "adaptive_replication_voi_enabled": args[
            "adaptive_replication_voi"
        ],
        "posterior_dominance_enabled": args[
            "posterior_dominance_enabled"
        ],
        "posterior_dominance_switch_count": 0,
    }


def _authorize(
    plan,
    task_ids=None,
    authorization_id="test-local-consent",
    source_report=None,
):
    selected = task_ids or [task["task_id"] for task in plan["tasks"]]
    return authorize_plan(
        plan,
        _hypothesis_contract(),
        source_report or _report(),
        _executor_contract(),
        expected_plan_digest=plan["integrity"]["plan_digest"],
        authorization_id=authorization_id,
        authorized_task_ids=selected,
    )


def test_plan_only_proposal_is_blocked_deterministic_and_native_ordered():
    first = _build()
    second = _build()
    assert first == second
    assert first["status"] == "AWAITING_TASK_TEMPLATE"
    assert verify_plan_integrity(
        first, _hypothesis_contract(), _executor_contract(), _report()
    )
    assert len(first["tasks"]) == 30
    assert all(
        cell["status"] == "BLOCKED_NO_TASK_TEMPLATE"
        for cell in first["tasks"]
    )
    assert [
        (cell["cell"]["domain"], cell["cell"]["seed"])
        for cell in first["tasks"]
    ] == [
        (domain, seed)
        for domain in _hypothesis_contract()["evidence_scope"]["domains"]
        for seed in range(10)
    ]
    assert all(cell["cell"]["profile"] == "full" for cell in first["tasks"])
    assert all(cell["run_one_task"] is None for cell in first["tasks"])
    assert all(cell["task_digest"] is None for cell in first["tasks"])
    assert "no_exact_historical_runtime_reconstruction" in first["nonclaims"]
    assert "no_bundled_real_task_template" in first["nonclaims"]


def test_plan_rejects_tampered_report_and_invalid_evidence():
    tampered = _report()
    tampered["pending_evidence"][0]["seed"] = 99
    assert not verify_report_integrity(tampered)
    with pytest.raises(ValueError):
        _build(tampered)

    invalid = _report(poison=True)
    assert invalid["verdict_counts"]["INVALID_EVIDENCE"] > 0
    with pytest.raises(ValueError):
        _build(invalid)


def test_plan_digest_covers_cells_and_contract():
    plan = _build()
    mutated = copy.deepcopy(plan)
    mutated["tasks"][0]["cell"]["seed"] = 7
    assert not verify_plan_integrity(
        mutated, _hypothesis_contract(), _executor_contract(), _report()
    )

    changed_contract = _executor_contract()
    changed_contract["mechanics"]["network_access"] = True
    assert not verify_plan_integrity(
        plan, _hypothesis_contract(), changed_contract, _report()
    )


def test_plan_verifier_rejects_self_consistent_pending_cell_omission():
    source_report = _report()
    plan = _build(source_report)
    omitted = copy.deepcopy(plan)
    omitted["tasks"].pop()
    omitted["proposal_count"] = len(omitted["tasks"])
    identity = {
        "source_report_binding": omitted["source_report_binding"],
        "executor_contract_digest": omitted["executor_contract_digest"],
        "status": omitted["status"],
        "tasks": omitted["tasks"],
    }
    omitted["plan_id"] = "plan:" + _digest(identity).split(":", 1)[1][:24]
    body = {key: value for key, value in omitted.items() if key != "integrity"}
    omitted["integrity"]["plan_digest"] = _digest(body)
    assert not verify_plan_integrity(
        omitted,
        _hypothesis_contract(),
        _executor_contract(),
        source_report,
    )


def test_plan_verifier_rejects_report_pending_decision_mismatch():
    source_report = _report()
    source_report["pending_evidence"].pop()
    report_body = {
        key: value for key, value in source_report.items() if key != "audit"
    }
    report_body_digest = _digest(report_body)
    source_report["audit"]["report_body_digest"] = report_body_digest
    final_event = source_report["audit"]["events"][-1]
    final_event["report_body_digest"] = report_body_digest
    final_event_body = {
        key: value for key, value in final_event.items() if key != "event_hash"
    }
    final_event["event_hash"] = _digest(final_event_body)
    source_report["audit"]["head"] = final_event["event_hash"]
    assert verify_report_integrity(source_report)

    forged = _build()
    forged["tasks"].pop()
    forged["proposal_count"] = len(forged["tasks"])
    forged["source_report_binding"]["report_body_digest"] = report_body_digest
    forged["source_report_binding"]["audit_head"] = source_report["audit"][
        "head"
    ]
    identity = {
        "source_report_binding": forged["source_report_binding"],
        "executor_contract_digest": forged["executor_contract_digest"],
    }
    for task in forged["tasks"]:
        task["task_id"] = "task:" + _digest(
            {**identity, "cell": task["cell"]}
        ).split(":", 1)[1][:24]
    plan_identity = {
        **identity,
        "status": forged["status"],
        "tasks": forged["tasks"],
    }
    forged["plan_id"] = "plan:" + _digest(plan_identity).split(":", 1)[1][
        :24
    ]
    plan_body = {
        key: value for key, value in forged.items() if key != "integrity"
    }
    forged["integrity"]["plan_digest"] = _digest(plan_body)

    assert not verify_plan_integrity(
        forged,
        _hypothesis_contract(),
        _executor_contract(),
        source_report,
    )


def test_blocked_plan_cannot_be_authorized():
    plan = _build()
    with pytest.raises(ValueError):
        authorize_plan(
            plan,
            _hypothesis_contract(),
            _report(),
            _executor_contract(),
            expected_plan_digest=plan["integrity"]["plan_digest"],
            authorized_task_ids=[],
            authorization_id="local-test-authorization",
        )


def test_materialized_mechanics_tasks_are_explicit_and_tamper_evident():
    first = _build(materializer=_materialize)
    second = _build(materializer=_materialize)
    assert first == second
    assert first["status"] == "READY_FOR_AUTHORIZATION"
    assert verify_plan_integrity(
        first, _hypothesis_contract(), _executor_contract(), _report()
    )
    assert all(
        task["status"] == "READY_FOR_AUTHORIZATION"
        for task in first["tasks"]
    )
    assert all(
        set(task["run_one_task"]) == {"args", "heldout", "line", "seed"}
        for task in first["tasks"]
    )
    assert all(
        isinstance(
            task["run_one_task"]["args"]["test_only_raw_template"], dict
        )
        for task in first["tasks"]
    )

    mutated = copy.deepcopy(first)
    mutated["tasks"][0]["run_one_task"]["args"]["d"] = 99
    assert not verify_plan_integrity(
        mutated, _hypothesis_contract(), _executor_contract(), _report()
    )


def test_authorization_binds_exact_plan_digest_and_explicit_subset():
    plan = _build(materializer=_materialize)
    subset = [plan["tasks"][2]["task_id"], plan["tasks"][0]["task_id"]]
    authorization = _authorize(plan, subset)
    assert verify_authorization_integrity(authorization)
    assert [
        item["task_id"] for item in authorization["authorized_tasks"]
    ] == [plan["tasks"][0]["task_id"], plan["tasks"][2]["task_id"]]

    with pytest.raises(ValueError):
        authorize_plan(
            plan,
            _hypothesis_contract(),
            _report(),
            _executor_contract(),
            expected_plan_digest="sha256:" + "0" * 64,
            authorization_id="wrong-plan",
            authorized_task_ids=subset,
        )
    with pytest.raises(ValueError):
        authorize_plan(
            plan,
            _hypothesis_contract(),
            _report(),
            _executor_contract(),
            expected_plan_digest=plan["integrity"]["plan_digest"],
            authorization_id="duplicate-subset",
            authorized_task_ids=[subset[0], subset[0]],
        )

    tampered = copy.deepcopy(authorization)
    tampered["authorized_tasks"][0]["task_digest"] = "sha256:" + "f" * 64
    assert not verify_authorization_integrity(tampered)


def test_preexecution_binding_failure_invokes_zero_callbacks():
    plan = _build(materializer=_materialize)
    authorization = _authorize(plan, [plan["tasks"][0]["task_id"]])
    calls = []

    def executor(task):
        calls.append(task)
        return _raw_run_one_result(task)

    with pytest.raises(ValueError):
        execute_authorized_plan(
            plan,
            authorization,
            expected_authorization_digest="sha256:" + "0" * 64,
            executor=executor,
            hypothesis_contract=_hypothesis_contract(),
            source_report=_report(),
            executor_contract=_executor_contract(),
        )
    assert calls == []

    tampered_plan = copy.deepcopy(plan)
    tampered_plan["tasks"][0]["task_digest"] = "sha256:" + "1" * 64
    with pytest.raises(ValueError):
        execute_authorized_plan(
            tampered_plan,
            authorization,
            expected_authorization_digest=(
                authorization["integrity"]["authorization_digest"]
            ),
            executor=executor,
            hypothesis_contract=_hypothesis_contract(),
            source_report=_report(),
            executor_contract=_executor_contract(),
        )
    assert calls == []


def test_callback_exception_and_wrong_row_are_evidence_neutral_failures():
    plan = _build(materializer=_materialize)
    task_id = plan["tasks"][0]["task_id"]
    authorization = _authorize(plan, [task_id])
    auth_digest = authorization["integrity"]["authorization_digest"]

    def crash(_task):
        raise RuntimeError("synthetic local failure")

    crashed = execute_authorized_plan(
        plan,
        authorization,
        expected_authorization_digest=auth_digest,
        executor=crash,
        hypothesis_contract=_hypothesis_contract(),
        source_report=_report(),
        executor_contract=_executor_contract(),
    )
    assert crashed["status"] == "COMPLETED_WITH_FAILURES"
    assert crashed["summary"] == {"authorized": 1, "succeeded": 0, "failed": 1}
    assert crashed["results"][0]["status"] == "FAILED"
    assert crashed["results"][0]["evidence_row"] is None
    assert crashed["results"][0]["error"]["type"] == "RuntimeError"
    assert crashed["results"][0]["error"]["code"] == "EXECUTOR_EXCEPTION"
    assert verify_receipt_integrity(crashed)
    assert verify_receipt_integrity(
        crashed,
        plan,
        authorization,
        _report(),
        _hypothesis_contract(),
        _executor_contract(),
    )

    def wrong_seed(task):
        row = _raw_run_one_result(task)
        row["seed"] = 999
        return row

    wrong = execute_authorized_plan(
        plan,
        authorization,
        expected_authorization_digest=auth_digest,
        executor=wrong_seed,
        hypothesis_contract=_hypothesis_contract(),
        source_report=_report(),
        executor_contract=_executor_contract(),
    )
    assert wrong["results"][0]["status"] == "FAILED"
    assert wrong["results"][0]["error"]["code"] == "RESULT_REJECTED"
    assert verify_receipt_integrity(wrong)
    assert verify_receipt_integrity(
        wrong,
        plan,
        authorization,
        _report(),
        _hypothesis_contract(),
        _executor_contract(),
    )

    tampered = copy.deepcopy(crashed)
    tampered["summary"]["failed"] = 0
    assert not verify_receipt_integrity(tampered)


def test_non_ok_native_result_is_failed_before_reingestion():
    plan = _build(materializer=_materialize)
    authorization = _authorize(plan, [plan["tasks"][0]["task_id"]])

    def non_ok(task):
        raw = _raw_run_one_result(task)
        raw["status"] = "runtime_failed"
        return raw

    receipt = execute_authorized_plan(
        plan,
        authorization,
        expected_authorization_digest=(
            authorization["integrity"]["authorization_digest"]
        ),
        executor=non_ok,
        hypothesis_contract=_hypothesis_contract(),
        source_report=_report(),
        executor_contract=_executor_contract(),
    )
    assert receipt["summary"] == {
        "authorized": 1, "succeeded": 0, "failed": 1,
    }
    assert receipt["results"][0]["error"]["code"] == "RESULT_REJECTED"
    with pytest.raises(ValueError, match="at least one successful"):
        reingest_successful_receipts(
            _rows(),
            [receipt],
            _hypothesis_contract(),
            _report(),
            plan=plan,
            authorization=authorization,
            executor_contract=_executor_contract(),
        )


def test_execution_uses_preflight_snapshot_for_every_authorized_task():
    plan = _build(materializer=_materialize)
    authorization = _authorize(
        plan, [plan["tasks"][0]["task_id"], plan["tasks"][1]["task_id"]]
    )
    observed_penalties = []

    def mutate_original_plan(task):
        observed_penalties.append(task["args"]["decision_risk_penalty"])
        if len(observed_penalties) == 1:
            plan["tasks"][1]["run_one_task"]["args"][
                "decision_risk_penalty"
            ] = 999.0
        return _raw_run_one_result(task)

    receipt = execute_authorized_plan(
        plan,
        authorization,
        expected_authorization_digest=(
            authorization["integrity"]["authorization_digest"]
        ),
        executor=mutate_original_plan,
        hypothesis_contract=_hypothesis_contract(),
        source_report=_report(),
        executor_contract=_executor_contract(),
    )
    assert observed_penalties == [5.0, 5.0]
    assert receipt["summary"]["succeeded"] == 2


def test_strong_receipt_verification_rejects_unapproved_task_rewrite():
    plan = _build(materializer=_materialize)
    authorization = _authorize(plan, [plan["tasks"][0]["task_id"]])
    receipt = execute_authorized_plan(
        plan,
        authorization,
        expected_authorization_digest=(
            authorization["integrity"]["authorization_digest"]
        ),
        executor=_raw_run_one_result,
        hypothesis_contract=_hypothesis_contract(),
        source_report=_report(),
        executor_contract=_executor_contract(),
    )
    forged = copy.deepcopy(receipt)
    unauthorized = plan["tasks"][1]
    forged_row = _normalized_full_row(unauthorized["run_one_task"])
    forged["results"][0].update({
        "task_id": unauthorized["task_id"],
        "task_digest": unauthorized["task_digest"],
        "evidence_row": forged_row,
        "evidence_digest": _digest(forged_row),
    })
    forged_body = {key: value for key, value in forged.items() if key != "integrity"}
    forged_body["receipt_id"] = "receipt:" + _digest({
        "plan": forged_body["plan_binding"]["plan_id"],
        "authorization": forged_body["authorization_binding"][
            "authorization_digest"
        ],
        "results": forged_body["results"],
    }).split(":", 1)[1][:24]
    forged = {
        **forged_body,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "receipt_digest": _digest(forged_body),
        },
    }
    # A local digest is not an authority signature, so structural self-checks
    # alone can be recomputed.  Full-chain verification closes the stage link.
    assert verify_receipt_integrity(forged)
    assert not verify_receipt_integrity(
        forged,
        plan,
        authorization,
        _report(),
        _hypothesis_contract(),
        _executor_contract(),
    )
    with pytest.raises(ValueError, match="receipt integrity"):
        reingest_successful_receipts(
            _rows(),
            [forged],
            _hypothesis_contract(),
            _report(),
            plan=plan,
            authorization=authorization,
            executor_contract=_executor_contract(),
        )

    empty_body = {
        key: copy.deepcopy(value)
        for key, value in receipt.items()
        if key != "integrity"
    }
    empty_body["results"] = []
    empty_body["summary"] = {"authorized": 0, "succeeded": 0, "failed": 0}
    empty_body["status"] = "COMPLETED"
    empty_body["receipt_id"] = "receipt:" + _digest({
        "plan": empty_body["plan_binding"]["plan_id"],
        "authorization": empty_body["authorization_binding"][
            "authorization_digest"
        ],
        "results": [],
    }).split(":", 1)[1][:24]
    empty = {
        **empty_body,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "receipt_digest": _digest(empty_body),
        },
    }
    assert not verify_receipt_integrity(empty)


def test_success_receipt_reingests_and_only_then_changes_verdicts():
    base_rows = _rows()
    source_report = _report()
    plan = _build(source_report, materializer=_materialize)
    authorization = _authorize(plan, source_report=source_report)
    calls = []

    def executor(task):
        calls.append((task["heldout"], task["seed"]))
        return _raw_run_one_result(task)

    receipt = execute_authorized_plan(
        plan,
        authorization,
        expected_authorization_digest=(
            authorization["integrity"]["authorization_digest"]
        ),
        executor=executor,
        hypothesis_contract=_hypothesis_contract(),
        source_report=source_report,
        executor_contract=_executor_contract(),
    )
    assert receipt["status"] == "COMPLETED"
    assert receipt["summary"] == {"authorized": 30, "succeeded": 30, "failed": 0}
    assert verify_receipt_integrity(receipt)
    assert verify_receipt_integrity(
        receipt,
        plan,
        authorization,
        source_report,
        _hypothesis_contract(),
        _executor_contract(),
    )
    assert len(calls) == 30
    # Execution did not mutate the already-issued source report.
    assert len(source_report["pending_evidence"]) == 30

    result = reingest_successful_receipts(
        base_rows,
        [receipt],
        _hypothesis_contract(),
        source_report,
        plan=plan,
        authorization=authorization,
        executor_contract=_executor_contract(),
    )
    updated = result["report"]
    assert verify_report_integrity(updated)
    assert updated["pending_evidence"] == []
    assert updated["verdict_counts"]["NEEDS_EVIDENCE"] == 0
    reingestion = result["reingestion_receipt"]
    assert reingestion["accepted_successful_rows"] == 30
    assert reingestion["ignored_failed_attempts"] == 0
    assert verify_reingestion_integrity(
        reingestion,
        source_report=source_report,
        base_rows=base_rows,
        plan=plan,
        authorization=authorization,
        receipts=[receipt],
        output_report=updated,
        hypothesis_contract=_hypothesis_contract(),
        executor_contract=_executor_contract(),
    )

    mutated = copy.deepcopy(receipt)
    mutated["results"][0]["evidence_row"]["seed"] = 999
    with pytest.raises(ValueError):
        reingest_successful_receipts(
            base_rows,
            [mutated],
            _hypothesis_contract(),
            source_report,
            plan=plan,
            authorization=authorization,
            executor_contract=_executor_contract(),
        )

    tampered_reingestion = copy.deepcopy(reingestion)
    tampered_reingestion["accepted_successful_rows"] = 29
    assert not verify_reingestion_integrity(tampered_reingestion)

    omitted_rows = copy.deepcopy(reingestion)
    omitted_rows["combined_evidence_digest"] = source_report[
        "evidence_digest"
    ]
    omitted_rows["output_report_body_digest"] = source_report["audit"][
        "report_body_digest"
    ]
    omitted_body = {
        key: value for key, value in omitted_rows.items()
        if key != "integrity"
    }
    omitted_rows["integrity"]["reingestion_digest"] = _digest(omitted_body)
    assert verify_reingestion_integrity(omitted_rows)
    assert not verify_reingestion_integrity(
        omitted_rows,
        source_report=source_report,
        base_rows=base_rows,
        plan=plan,
        authorization=authorization,
        receipts=[receipt],
        output_report=source_report,
        hypothesis_contract=_hypothesis_contract(),
        executor_contract=_executor_contract(),
    )


def test_partial_success_reingests_then_replans_only_remaining_cells():
    base_rows = _rows()
    source_report = _report()
    plan = _build(source_report, materializer=_materialize)
    selected = [plan["tasks"][0]["task_id"], plan["tasks"][1]["task_id"]]
    authorization = _authorize(plan, selected, source_report=source_report)

    def one_success(task):
        if task["seed"] == 1:
            raise RuntimeError("synthetic second-cell failure")
        return _raw_run_one_result(task)

    receipt = execute_authorized_plan(
        plan,
        authorization,
        expected_authorization_digest=(
            authorization["integrity"]["authorization_digest"]
        ),
        executor=one_success,
        hypothesis_contract=_hypothesis_contract(),
        source_report=source_report,
        executor_contract=_executor_contract(),
    )
    result = reingest_successful_receipts(
        base_rows,
        [receipt],
        _hypothesis_contract(),
        source_report,
        plan=plan,
        authorization=authorization,
        executor_contract=_executor_contract(),
    )
    assert len(result["report"]["pending_evidence"]) == 29
    next_plan = build_execution_plan(
        result["report"],
        _hypothesis_contract(),
        _executor_contract(),
    )
    assert next_plan["proposal_count"] == 29
    assert next_plan["tasks"][0]["cell"]["seed"] == 1
    assert verify_plan_integrity(
        next_plan,
        _hypothesis_contract(),
        _executor_contract(),
        result["report"],
    )


def test_plan_cli_writes_and_verifies_blocked_proposal(tmp_path):
    report_path = tmp_path / "report.json"
    output = tmp_path / "plan.json"
    report_path.write_text(
        json.dumps(_report(), ensure_ascii=False), encoding="utf-8"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "plan",
            "--report", str(report_path),
            "--hypothesis-contract", str(HYPOTHESIS_CONTRACT_PATH),
            "--contract", str(EXECUTOR_CONTRACT_PATH),
            "--out", str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert "status=AWAITING_TASK_TEMPLATE" in completed.stderr
    plan = _json(output)
    assert len(plan["tasks"]) == 30

    verified = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "verify",
            "--plan", str(output),
            "--report", str(report_path),
            "--hypothesis-contract", str(HYPOTHESIS_CONTRACT_PATH),
            "--contract", str(EXECUTOR_CONTRACT_PATH),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr
    assert verified.stdout == ""
    assert "action=verify" in verified.stderr


def test_plan_cli_rejects_duplicate_keys_and_output_hardlink(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"audit": {}, "audit": {}}\n', encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "plan",
            "--report", str(duplicate),
            "--hypothesis-contract", str(HYPOTHESIS_CONTRACT_PATH),
            "--contract", str(EXECUTOR_CONTRACT_PATH),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert rejected.stdout == ""
    assert "duplicate key 'audit'" in rejected.stderr
    assert "Traceback" not in rejected.stderr

    report_path = tmp_path / "report.json"
    output = tmp_path / "alias.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    os.link(report_path, output)
    before = report_path.read_bytes()
    rejected = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "plan",
            "--report", str(report_path),
            "--out", str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "hard link" in rejected.stderr
    assert report_path.read_bytes() == before


def test_plan_cli_rejects_unanchored_hypothesis_contract(tmp_path):
    report_path = tmp_path / "report.json"
    contract_path = tmp_path / "unanchored-hypothesis-contract.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    changed = _hypothesis_contract()
    changed["gate"]["min_overall_feasible"] = 23
    contract_path.write_text(json.dumps(changed), encoding="utf-8")

    rejected = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "plan",
            "--report", str(report_path),
            "--hypothesis-contract", str(contract_path),
            "--contract", str(EXECUTOR_CONTRACT_PATH),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert rejected.stdout == ""
    assert "source binding" in rejected.stderr
    assert "Traceback" not in rejected.stderr


def test_runner_has_no_real_execution_surface():
    source = RUNNER.read_text(encoding="utf-8")
    assert "import benchmark_lodo_meta_prior" not in source
    assert "from performance.benchmark_lodo_meta_prior" not in source
    assert "subprocess" not in source
