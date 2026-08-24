import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST_ROOT))

import runners.run_structural_hypothesis_recursive_campaign as runner  # noqa: E402
import test_structural_hypothesis_recursive_report_advance as advance_test  # noqa: E402
import test_structural_hypothesis_recursive_successor_materializer as successor_test  # noqa: E402
from test_structural_hypothesis_recursive_report_advance import (  # noqa: E402,F401
    fake_completed_successor_case,
)
from test_structural_hypothesis_recursive_successor_materializer import (  # noqa: E402,F401
    fake_recursive_advance_case,
)


RUNNER = ROOT / "runners/run_structural_hypothesis_recursive_campaign.py"
CORE = ROOT / "performance/structural_hypothesis_recursive_campaign.py"
CAMPAIGN_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_recursive_campaign_v1.json"
)
RUNTIME_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_single_task_runtime_v1.json"
)

INSPECTED = (
    "INSPECTED_RECURSIVE_CAMPAIGN_SOURCE_NONTERMINAL_NOT_AUTHORIZED"
)
AUTHORIZED = "RECURSIVE_CAMPAIGN_AUTHORIZED_ONE_CALLBACK_START_LEASED"
EXECUTED = (
    "RECURSIVE_CAMPAIGN_CALLBACK_START_CLAIMED_"
    "RUNTIME_COMPLETED_HARD_STOP"
)
ADVANCED = "ADVANCED_RECURSIVE_CAMPAIGN_ONE_STEP_NONTERMINAL_HARD_STOP"
TERMINAL = "ADVANCED_RECURSIVE_CAMPAIGN_ONE_STEP_TERMINAL_HARD_STOP"


@pytest.fixture(autouse=True)
def _pinned_campaign_environment(monkeypatch):
    for key, value in runner.REQUIRED_EXECUTION_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)


def _digest(label):
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(value) + "\n", encoding="utf-8")
    return path.resolve()


def _tree_observation(root):
    if not root.exists():
        return None
    return {
        str(path.relative_to(root)): (
            "directory" if path.is_dir() else path.read_bytes(),
            path.stat().st_mode,
            path.stat().st_nlink,
        )
        for path in root.rglob("*")
    }


def _recursive_successor_positionals(case):
    return (
        case["publication"],
        advance_test.ADOPTION_CONTRACT,
        case["adoption"],
        advance_test.SUCCESSOR_CONTRACT,
        case["successor"],
        case["evidence"],
        case["source_attempt"],
        advance_test.HYPOTHESIS_CONTRACT,
        advance_test.EXECUTOR_CONTRACT,
        advance_test.RUNTIME_CONTRACT,
        advance_test.PUBLISHER_CONTRACT,
        advance_test.MATERIALIZER_CONTRACT,
        advance_test.BRIDGE_CONTRACT,
        advance_test.BASE_MANIFEST,
        advance_test.ASSET_ROOT,
        case["completed_attempt"],
        advance_test.ADVANCE_CONTRACT,
        case["advance_root"],
        successor_test.RECURSIVE_SUCCESSOR_CONTRACT,
        case["recursive_successor_root"],
        case["future_attempt_root"],
    )


def _recursive_successor_identity_kwargs(case):
    return {
        "adoption_id": case["adoption_id"],
        "source_successor_id": case["successor_id"],
        "advance_id": case["advance_id"],
        "recursive_successor_id": case["recursive_successor_id"],
        "completed_task_id": case["expected"]["task_id"],
    }


@pytest.fixture
def fake_campaign_seed2_case(fake_recursive_advance_case):
    """Materialize the real 1,352-row/28-task seed-2 source, fake-only."""
    from performance import structural_hypothesis_recursive_successor_materializer as recursive_core

    case = fake_recursive_advance_case
    materialized = recursive_core.materialize_recursive_successor(
        *_recursive_successor_positionals(case),
        **_recursive_successor_identity_kwargs(case),
        **case["recursive_expected"],
        confirm_recursive_successor_materialization=True,
    )
    verify_kwargs = {
        **_recursive_successor_identity_kwargs(case),
        **case["recursive_expected"],
        "expected_recursive_successor_digest": materialized[
            "recursive_successor_digest"
        ],
        "expected_next_bundle_digest": materialized["bundle_digest"],
        "expected_next_plan_digest": materialized["plan_digest"],
    }
    descriptor = {
        "schema_version": (
            "sc-olh-kg.structural-hypothesis-recursive-campaign-source/1"
        ),
        "source_kind": "recursive_successor_v1",
        "dependencies": {
            "hypothesis_contract_path": str(advance_test.HYPOTHESIS_CONTRACT),
            "executor_contract_path": str(advance_test.EXECUTOR_CONTRACT),
            "runtime_contract_path": str(advance_test.RUNTIME_CONTRACT),
            "materializer_contract_path": str(
                advance_test.MATERIALIZER_CONTRACT
            ),
            "base_manifest_path": str(advance_test.BASE_MANIFEST),
            "asset_root": str(advance_test.ASSET_ROOT),
        },
        "verify_args": [
            str(path) for path in _recursive_successor_positionals(case)
        ],
        "verify_kwargs": verify_kwargs,
    }
    descriptor_path = _write_json(
        case["state_home"] / "campaign-inputs/seed2-source.json",
        descriptor,
    )
    campaign_id = "fake-recursive-campaign-seed2"
    campaign_root = (
        case["state_home"]
        / "kg-op/structural-hypothesis-recursive-campaign/v1"
        / campaign_id
    )
    return {
        **case,
        "recursive_materialized": materialized,
        "campaign_source": descriptor,
        "campaign_source_path": descriptor_path,
        "campaign_id": campaign_id,
        "campaign_root": campaign_root,
    }


def _expected_nonterminal_advance(case, next_attempt_root=None):
    """Independently replay one successful receipt and next materialization."""
    from performance import structural_hypothesis_execution as execution_core
    from performance import structural_hypothesis_recursive_campaign as campaign_core
    from performance.structural_hypothesis_task_materializer import (
        materialize_task_bundle,
        verify_materialized_task_bundle,
    )

    campaign_root = case["campaign_root"]
    attempt_root = case["future_attempt_root"]
    rows = json.loads(
        (campaign_root / "source/rows.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (campaign_root / "source/report.json").read_text(encoding="utf-8")
    )
    bundle = json.loads(
        (campaign_root / "source/bundle.json").read_text(encoding="utf-8")
    )
    authorization = json.loads(
        (attempt_root / "authorization.json").read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (attempt_root / "receipt.json").read_text(encoding="utf-8")
    )
    hypothesis = json.loads(
        advance_test.HYPOTHESIS_CONTRACT.read_text(encoding="utf-8")
    )
    executor = json.loads(
        advance_test.EXECUTOR_CONTRACT.read_text(encoding="utf-8")
    )
    materializer = json.loads(
        advance_test.MATERIALIZER_CONTRACT.read_text(encoding="utf-8")
    )
    reingested = execution_core.reingest_successful_receipts(
        rows,
        [receipt],
        hypothesis,
        report,
        plan=bundle["plan"],
        authorization=authorization,
        executor_contract=executor,
    )
    output_report = reingested["report"]
    reingestion_receipt = reingested["reingestion_receipt"]
    accepted = [
        copy.deepcopy(result["evidence_row"])
        for result in receipt["results"]
        if result["status"] == "SUCCEEDED"
    ]
    assert len(accepted) == 1
    combined_rows = [*copy.deepcopy(rows), *accepted]
    assert campaign_core._digest(combined_rows) == output_report[
        "evidence_digest"
    ]
    if next_attempt_root is None:
        provisional = materialize_task_bundle(
            output_report,
            hypothesis,
            executor,
            materializer,
            advance_test.BASE_MANIFEST,
            advance_test.ASSET_ROOT,
            attempt_root.parent / "preview-only/checkpoints",
        )
        next_task_id = provisional["plan"]["tasks"][0]["task_id"]
        next_attempt_root = attempt_root.parent / (
            "recursive-" + next_task_id.split(":", 1)[1]
        )
    next_bundle = materialize_task_bundle(
        output_report,
        hypothesis,
        executor,
        materializer,
        advance_test.BASE_MANIFEST,
        advance_test.ASSET_ROOT,
        next_attempt_root / "checkpoints",
    )
    assert verify_materialized_task_bundle(
        next_bundle,
        output_report,
        hypothesis,
        executor,
        materializer,
        advance_test.BASE_MANIFEST,
        advance_test.ASSET_ROOT,
        next_attempt_root / "checkpoints",
    )
    first = output_report["pending_evidence"][0]
    projection = {
        "profile": first["profile"],
        "domain": first["domain"],
        "line": "lodo",
        "seed": first["seed"],
        "d": first["d"],
        "N": first["N"],
        "n0": first["n0"],
    }
    return {
        "expected_output_evidence_digest": output_report["evidence_digest"],
        "expected_output_report_body_digest": output_report["audit"][
            "report_body_digest"
        ],
        "expected_output_audit_head": output_report["audit"]["head"],
        "expected_reingestion_digest": reingestion_receipt["integrity"][
            "reingestion_digest"
        ],
        "expected_next_pending_evidence_digest": campaign_core._digest(
            output_report["pending_evidence"]
        ),
        "expected_next_first_pending_projection_digest": (
            campaign_core._digest(projection)
        ),
        "expected_next_bundle_digest": next_bundle["integrity"][
            "bundle_digest"
        ],
        "expected_next_plan_digest": next_bundle["plan"]["integrity"][
            "plan_digest"
        ],
        "next_attempt_root": next_attempt_root,
        "combined_rows": combined_rows,
        "output_report": output_report,
        "reingestion_receipt": reingestion_receipt,
        "next_bundle": next_bundle,
    }


def _unit_source(tmp_path):
    dependencies = {}
    for name in (
        "hypothesis_contract_path",
        "executor_contract_path",
        "runtime_contract_path",
        "materializer_contract_path",
        "base_manifest_path",
    ):
        dependencies[name] = str(
            _write_json(tmp_path / "dependencies" / f"{name}.json", {})
        )
    asset_root = (tmp_path / "dependencies/assets").resolve()
    asset_root.mkdir(parents=True)
    dependencies["asset_root"] = str(asset_root)
    return {
        "schema_version": (
            "sc-olh-kg.structural-hypothesis-recursive-campaign-source/1"
        ),
        "source_kind": "recursive_successor_v1",
        "dependencies": dependencies,
        "verify_args": [str((tmp_path / f"arg-{index}").resolve()) for index in range(21)],
        "verify_kwargs": {"unit_only": True},
    }


def _unit_case(tmp_path, *, committed=False):
    source = _unit_source(tmp_path)
    source_path = _write_json(tmp_path / "source.json", source)
    contract_path = _write_json(tmp_path / "campaign-contract.json", {})
    runtime_path = _write_json(tmp_path / "runtime-contract.json", {})
    campaign_id = "campaign-unit-0001"
    campaign_root = (tmp_path / campaign_id).resolve()
    if committed:
        campaign_root.mkdir(mode=0o700)
    next_attempt = (tmp_path / "attempts/campaign-unit-0002").resolve()
    return {
        "source": source,
        "source_path": source_path,
        "contract": contract_path,
        "runtime": runtime_path,
        "campaign_id": campaign_id,
        "campaign_root": campaign_root,
        "next_attempt": next_attempt,
        "task_id": "task:" + "a" * 24,
        "authorization_id": "recursive-campaign-v1:" + "b" * 64,
        "digests": {
            name: _digest(name)
            for name in {
                *runner._AUTHORIZE_DIGEST_NAMES,
                *runner._EXECUTE_DIGEST_NAMES,
                *runner._ADVANCE_DIGEST_NAMES,
                *runner._ADVANCE_NEXT_DIGEST_NAMES,
                *runner._VERIFY_OPTIONAL_DIGEST_NAMES,
            }
        },
    }


def _base_cli(case):
    return [
        "--source-descriptor", str(case["source_path"]),
        "--campaign-contract", str(case["contract"]),
        "--campaign-root", str(case["campaign_root"]),
    ]


def _digests_cli(case, names):
    values = []
    for name in names:
        values.extend((
            "--" + name.replace("_", "-"), case["digests"][name]
        ))
    return values


def _inspect_result(case, *, status=INSPECTED):
    return {
        "status": status,
        "source_kind": "recursive_successor_v1",
        "source_state_digest": case["digests"][
            "expected_source_state_digest"
        ],
        "bundle_digest": case["digests"]["expected_bundle_digest"],
        "plan_digest": case["digests"]["expected_plan_digest"],
        "task_count": 28,
        "task_id": case["task_id"],
        "task_digest": case["digests"]["expected_task_digest"],
        "next_attempt_root": str(case["next_attempt"]),
        "checkpoint_root": str(case["next_attempt"] / "checkpoints"),
        "provenance_binding": {"unit": "only"},
        "provenance_binding_digest": case["digests"][
            "expected_provenance_binding_digest"
        ],
        "required_authorization_id": case["authorization_id"],
        "terminal_status": "NONTERMINAL",
    }


def _authorize_result(case):
    return {
        **_inspect_result(case, status=AUTHORIZED),
        "campaign_root": str(case["campaign_root"]),
        "campaign_digest": case["digests"]["expected_campaign_digest"],
        "lease_digest": case["digests"]["expected_lease_digest"],
        "authorization_digest": case["digests"][
            "expected_authorization_digest"
        ],
        "attempt_digest": case["digests"]["expected_attempt_digest"],
        "authorization_status": "AUTHORIZED",
        "execution_status": "NOT_EXECUTED",
    }


def _execute_result(case):
    return {
        "status": EXECUTED,
        "campaign_root": str(case["campaign_root"]),
        "campaign_digest": case["digests"]["expected_campaign_digest"],
        "lease_digest": case["digests"]["expected_lease_digest"],
        "callback_start_claim_digest": case["digests"][
            "expected_callback_start_claim_digest"
        ],
        "provenance_binding_digest": case["digests"][
            "expected_provenance_binding_digest"
        ],
        "authorization_digest": case["digests"][
            "expected_authorization_digest"
        ],
        "attempt_digest": case["digests"]["expected_attempt_digest"],
        "receipt_digest": case["digests"]["expected_receipt_digest"],
        "journal_head_digest": case["digests"][
            "expected_journal_head_digest"
        ],
        "task_id": case["task_id"],
        "execution_status": "COMPLETED_SUCCESS_AWAITING_ADVANCE",
    }


def _advance_result(case, *, status=ADVANCED, verified=False):
    phase = {
        ADVANCED: "ADVANCED_NONTERMINAL",
        TERMINAL: "ADVANCED_TERMINAL",
    }[status]
    value = {
        "status": ("VERIFIED_" + status) if verified else status,
        "phase": phase,
        "campaign_root": str(case["campaign_root"]),
        "campaign_digest": case["digests"]["expected_campaign_digest"],
        "lease_digest": case["digests"]["expected_lease_digest"],
        "callback_start_claim_digest": case["digests"][
            "expected_callback_start_claim_digest"
        ],
        "provenance_binding_digest": case["digests"][
            "expected_provenance_binding_digest"
        ],
        "authorization_digest": case["digests"][
            "expected_authorization_digest"
        ],
        "attempt_digest": case["digests"]["expected_attempt_digest"],
        "receipt_digest": case["digests"]["expected_receipt_digest"],
        "journal_head_digest": case["digests"][
            "expected_journal_head_digest"
        ],
        "advance_digest": case["digests"]["expected_advance_digest"],
        "reingestion_digest": case["digests"][
            "expected_reingestion_digest"
        ],
        "output_evidence_digest": case["digests"][
            "expected_output_evidence_digest"
        ],
        "output_report_body_digest": case["digests"][
            "expected_output_report_body_digest"
        ],
        "output_audit_head": case["digests"][
            "expected_output_audit_head"
        ],
        "next_pending_evidence_digest": case["digests"][
            "expected_next_pending_evidence_digest"
        ],
        "next_first_pending_projection_digest": case["digests"][
            "expected_next_first_pending_projection_digest"
        ],
        "next_bundle_digest": case["digests"][
            "expected_next_bundle_digest"
        ],
        "next_plan_digest": case["digests"]["expected_next_plan_digest"],
        "remaining_task_count": 27,
        "terminal_status": "NONTERMINAL",
        "next_attempt_root": str(case["next_attempt"]),
        "execution_status": "COMPLETED_AND_ADVANCED_HARD_STOP",
    }
    return value


def test_runner_forwards_exact_five_action_surface(tmp_path, monkeypatch, capsys):
    calls = []

    def capture(name, result):
        def called(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        return called

    # Inspect is the only action whose target must remain absent and whose
    # result intentionally has no campaign_root field.
    inspect_case = _unit_case(tmp_path / "inspect")
    inspect_before = _tree_observation(tmp_path / "inspect")
    fake = SimpleNamespace(
        inspect_recursive_campaign=capture(
            "inspect", _inspect_result(inspect_case)
        )
    )
    monkeypatch.setattr(runner, "_load_campaign_core", lambda: fake)
    assert runner.main([
        "inspect",
        *_base_cli(inspect_case),
        "--campaign-id", inspect_case["campaign_id"],
        "--next-attempt-root", str(inspect_case["next_attempt"]),
    ]) == 0
    observed = capsys.readouterr()
    assert observed.out == _canonical(_inspect_result(inspect_case)) + "\n"
    assert "action=inspect" in observed.err
    assert _tree_observation(tmp_path / "inspect") == inspect_before
    assert calls[-1] == (
        "inspect",
        (
            inspect_case["source"],
            inspect_case["contract"],
            inspect_case["campaign_root"],
        ),
        {
            "campaign_id": inspect_case["campaign_id"],
            "next_attempt_root": inspect_case["next_attempt"],
        },
    )

    authorize_case = _unit_case(tmp_path / "authorize")
    fake = SimpleNamespace(
        authorize_recursive_campaign_task=capture(
            "authorize", _authorize_result(authorize_case)
        )
    )
    monkeypatch.setattr(runner, "_load_campaign_core", lambda: fake)
    authorize_cli = [
        "authorize", *_base_cli(authorize_case),
        "--campaign-id", authorize_case["campaign_id"],
        *_digests_cli(authorize_case, runner._AUTHORIZE_DIGEST_NAMES),
        "--task-id", authorize_case["task_id"],
        "--authorization-id", authorize_case["authorization_id"],
        "--confirm-explicit-local-task-authorization",
    ]
    assert runner.main(authorize_cli) == 0
    observed = capsys.readouterr()
    assert observed.out == _canonical(_authorize_result(authorize_case)) + "\n"
    assert calls[-1] == (
        "authorize",
        (
            authorize_case["source"],
            authorize_case["contract"],
            authorize_case["campaign_root"],
        ),
        {
            "campaign_id": authorize_case["campaign_id"],
            **{
                name: authorize_case["digests"][name]
                for name in runner._AUTHORIZE_DIGEST_NAMES
            },
            "task_id": authorize_case["task_id"],
            "authorization_id": authorize_case["authorization_id"],
            "confirm_explicit_local_task_authorization": True,
        },
    )

    committed = _unit_case(tmp_path / "committed", committed=True)
    execute_result = _execute_result(committed)
    fake = SimpleNamespace(
        execute_recursive_campaign_task=capture("execute", execute_result)
    )
    monkeypatch.setattr(runner, "_load_campaign_core", lambda: fake)
    assert runner.main([
        "execute", *_base_cli(committed),
        "--runtime-contract", str(committed["runtime"]),
        *_digests_cli(committed, runner._EXECUTE_DIGEST_NAMES),
        "--confirm-real-local-execution",
    ]) == 0
    observed = capsys.readouterr()
    assert observed.out == _canonical(execute_result) + "\n"
    assert calls[-1][0] == "execute"
    assert calls[-1][1] == (
        committed["source"],
        committed["contract"],
        committed["campaign_root"],
        committed["runtime"],
    )
    assert calls[-1][2] == {
        **{
            name: committed["digests"][name]
            for name in runner._EXECUTE_DIGEST_NAMES
        },
        "confirm_real_local_execution": True,
    }

    advanced_result = _advance_result(committed)
    fake = SimpleNamespace(
        advance_recursive_campaign=capture("advance", advanced_result)
    )
    monkeypatch.setattr(runner, "_load_campaign_core", lambda: fake)
    assert runner.main([
        "advance", *_base_cli(committed),
        "--next-attempt-root", str(committed["next_attempt"]),
        *_digests_cli(committed, runner._ADVANCE_DIGEST_NAMES),
        *_digests_cli(committed, runner._ADVANCE_NEXT_DIGEST_NAMES),
        "--confirm-immutable-one-step-advance",
    ]) == 0
    observed = capsys.readouterr()
    assert observed.out == _canonical(advanced_result) + "\n"
    assert calls[-1][0] == "advance"
    assert calls[-1][1] == (
        committed["source"],
        committed["contract"],
        committed["campaign_root"],
        committed["next_attempt"],
    )
    assert calls[-1][2] == {
        **{
            name: committed["digests"][name]
            for name in (
                *runner._ADVANCE_DIGEST_NAMES,
                *runner._ADVANCE_NEXT_DIGEST_NAMES,
            )
        },
        "confirm_immutable_one_step_advance": True,
    }

    verified_result = _advance_result(committed, verified=True)
    fake = SimpleNamespace(
        verify_recursive_campaign=capture("verify", verified_result)
    )
    monkeypatch.setattr(runner, "_load_campaign_core", lambda: fake)
    verify_names = (
        "expected_campaign_digest",
        "expected_lease_digest",
        *runner._VERIFY_OPTIONAL_DIGEST_NAMES,
    )
    before = _tree_observation(committed["campaign_root"])
    assert runner.main([
        "verify", *_base_cli(committed),
        *_digests_cli(committed, verify_names),
    ]) == 0
    observed = capsys.readouterr()
    assert observed.out == _canonical(verified_result) + "\n"
    assert "action=verify" in observed.err
    assert _tree_observation(committed["campaign_root"]) == before
    assert calls[-1] == (
        "verify",
        (
            committed["source"],
            committed["contract"],
            committed["campaign_root"],
        ),
        {
            name: committed["digests"][name] for name in verify_names
        },
    )


def test_runner_confirmation_duplicate_anchor_path_and_result_gates(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)
    authorize = [
        "authorize", *_base_cli(case),
        "--campaign-id", case["campaign_id"],
        *_digests_cli(case, runner._AUTHORIZE_DIGEST_NAMES),
        "--task-id", case["task_id"],
        "--authorization-id", case["authorization_id"],
    ]
    assert runner.main(authorize) == 2
    parse_rejected = capsys.readouterr()
    assert parse_rejected.out == ""
    assert parse_rejected.err.count("\n") == 1
    assert "invalid command line" in parse_rejected.err

    called = []
    monkeypatch.setattr(
        runner, "_load_campaign_core", lambda: called.append(True)
    )
    bad_anchor = [
        *authorize,
        "--confirm-explicit-local-task-authorization",
    ]
    option = "--expected-source-state-digest"
    bad_anchor[bad_anchor.index(option) + 1] = "sha256:BAD"
    assert runner.main(bad_anchor) == 2
    rejected = capsys.readouterr()
    assert rejected.out == ""
    assert "lowercase sha256 digest" in rejected.err
    assert called == []
    assert not case["campaign_root"].exists()

    duplicate = case["source_path"]
    duplicate.write_text('{"source_kind":"a","source_kind":"b"}\n')
    assert runner.main([
        "inspect", *_base_cli(case),
        "--campaign-id", case["campaign_id"],
    ]) == 2
    rejected = capsys.readouterr()
    assert rejected.out == ""
    assert "duplicate key" in rejected.err
    assert called == []
    duplicate.write_text(_canonical(case["source"]) + "\n", encoding="utf-8")

    # A partially specified nonterminal next boundary is neither terminal nor
    # a complete nonterminal request and must fail before loading the core.
    committed = _unit_case(tmp_path / "partial", committed=True)
    partial = [
        "advance", *_base_cli(committed),
        "--next-attempt-root", str(committed["next_attempt"]),
        *_digests_cli(committed, runner._ADVANCE_DIGEST_NAMES),
        "--confirm-immutable-one-step-advance",
    ]
    assert runner.main(partial) == 2
    rejected = capsys.readouterr()
    assert rejected.out == ""
    assert "all four next-round digests" in rejected.err

    one_completed_anchor = [
        "verify", *_base_cli(committed),
        "--expected-campaign-digest",
        committed["digests"]["expected_campaign_digest"],
        "--expected-lease-digest",
        committed["digests"]["expected_lease_digest"],
        "--expected-receipt-digest",
        committed["digests"]["expected_receipt_digest"],
    ]
    assert runner.main(one_completed_anchor) == 2
    rejected = capsys.readouterr()
    assert rejected.out == ""
    assert "supplied together" in rejected.err

    # Relative input and a result with any extra or missing field fail closed.
    relative = list(authorize)
    relative[relative.index("--source-descriptor") + 1] = "source.json"
    relative.append("--confirm-explicit-local-task-authorization")
    assert runner.main(relative) == 2
    capsys.readouterr()

    malformed = _authorize_result(case)
    malformed["unexpected"] = True
    monkeypatch.setattr(
        runner,
        "_load_campaign_core",
        lambda: SimpleNamespace(
            authorize_recursive_campaign_task=lambda *a, **k: malformed
        ),
    )
    assert runner.main([
        *authorize,
        "--confirm-explicit-local-task-authorization",
    ]) == 2
    rejected = capsys.readouterr()
    assert rejected.out == ""
    assert "result keys" in rejected.err


def test_runner_terminal_advance_omits_all_next_boundary_values(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path, committed=True)
    terminal = _advance_result(case, status=TERMINAL)
    for key in (
        "next_pending_evidence_digest",
        "next_first_pending_projection_digest",
        "next_bundle_digest",
        "next_plan_digest",
        "next_attempt_root",
    ):
        terminal[key] = None
    terminal["remaining_task_count"] = 0
    terminal["terminal_status"] = "TERMINAL"
    terminal["phase"] = "ADVANCED_TERMINAL"
    observed = {}

    def advance(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return terminal

    monkeypatch.setattr(
        runner,
        "_load_campaign_core",
        lambda: SimpleNamespace(advance_recursive_campaign=advance),
    )
    assert runner.main([
        "advance", *_base_cli(case),
        *_digests_cli(case, runner._ADVANCE_DIGEST_NAMES),
        "--confirm-immutable-one-step-advance",
    ]) == 0
    captured = capsys.readouterr()
    assert captured.out == _canonical(terminal) + "\n"
    assert observed["args"][-1] is None
    for name in runner._ADVANCE_NEXT_DIGEST_NAMES:
        assert observed["kwargs"][name] is None


def test_runner_authorize_forwards_committed_root_for_core_recovery(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path, committed=True)
    calls = []
    monkeypatch.setattr(
        runner,
        "_load_campaign_core",
        lambda: SimpleNamespace(
            authorize_recursive_campaign_task=lambda *args, **kwargs: (
                calls.append((args, kwargs)) or _authorize_result(case)
            )
        ),
    )
    assert runner.main([
        "authorize", *_base_cli(case),
        "--campaign-id", case["campaign_id"],
        *_digests_cli(case, runner._AUTHORIZE_DIGEST_NAMES),
        "--task-id", case["task_id"],
        "--authorization-id", case["authorization_id"],
        "--confirm-explicit-local-task-authorization",
    ]) == 0
    captured = capsys.readouterr()
    assert captured.out == _canonical(_authorize_result(case)) + "\n"
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("base_status", "phase"),
    (
        (AUTHORIZED, "AUTHORIZED"),
        (
            "RECURSIVE_CAMPAIGN_CALLBACK_START_CLAIMED_"
            "RUNTIME_INCOMPLETE_HARD_STOP",
            "CALLBACK_INCOMPLETE",
        ),
        (EXECUTED, "CALLBACK_COMPLETED"),
        (ADVANCED, "ADVANCED_NONTERMINAL"),
        (TERMINAL, "ADVANCED_TERMINAL"),
    ),
)
def test_runner_verify_accepts_uniform_phase_aware_surface(
    tmp_path, monkeypatch, capsys, base_status, phase
):
    case = _unit_case(tmp_path, committed=True)
    result = _advance_result(
        case,
        status=TERMINAL if phase == "ADVANCED_TERMINAL" else ADVANCED,
        verified=True,
    )
    result["status"] = "VERIFIED_" + base_status
    result["phase"] = phase
    result["execution_status"] = {
        "AUTHORIZED": "NOT_EXECUTED",
        "CALLBACK_INCOMPLETE": "CALLBACK_CLAIMED_INCOMPLETE_NO_REENTRY",
        "CALLBACK_COMPLETED": (
            "COMPLETED_SUCCESS_PREVIEW_RECOVERED_LOCAL_ANCHORS_"
            "NOT_INDEPENDENT_HARD_STOP"
        ),
        "ADVANCED_NONTERMINAL": "COMPLETED_AND_ADVANCED_HARD_STOP",
        "ADVANCED_TERMINAL": "COMPLETED_AND_ADVANCED_HARD_STOP",
    }[phase]
    if phase == "AUTHORIZED":
        result["callback_start_claim_digest"] = None
    if phase in {"AUTHORIZED", "CALLBACK_INCOMPLETE"}:
        result["receipt_digest"] = None
        result["journal_head_digest"] = None
    if not phase.startswith("ADVANCED_"):
        result["advance_digest"] = None
    if phase in {"AUTHORIZED", "CALLBACK_INCOMPLETE"}:
        for key in (
            "reingestion_digest",
            "output_evidence_digest",
            "output_report_body_digest",
            "output_audit_head",
            "next_pending_evidence_digest",
            "next_first_pending_projection_digest",
            "next_bundle_digest",
            "next_plan_digest",
        ):
            result[key] = None
        result["next_attempt_root"] = None
    if phase == "ADVANCED_TERMINAL":
        for key in (
            "next_pending_evidence_digest",
            "next_first_pending_projection_digest",
            "next_bundle_digest",
            "next_plan_digest",
            "next_attempt_root",
        ):
            result[key] = None
        result["remaining_task_count"] = 0
        result["terminal_status"] = "TERMINAL"

    observed = {}

    def verify(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return result

    monkeypatch.setattr(
        runner,
        "_load_campaign_core",
        lambda: SimpleNamespace(
            verify_recursive_campaign=verify
        ),
    )
    argv = [
        "verify", *_base_cli(case),
        "--expected-campaign-digest",
        case["digests"]["expected_campaign_digest"],
        "--expected-lease-digest",
        case["digests"]["expected_lease_digest"],
    ]
    if phase == "CALLBACK_COMPLETED":
        argv.extend(("--next-attempt-root", str(case["next_attempt"])))
    assert runner.main(argv) == 0
    captured = capsys.readouterr()
    assert captured.out == _canonical(result) + "\n"
    assert "status=VERIFIED_" + base_status in captured.err
    if phase == "CALLBACK_COMPLETED":
        assert observed["kwargs"]["next_attempt_root"] == case[
            "next_attempt"
        ]
    else:
        assert "next_attempt_root" not in observed["kwargs"]
    wrong_execution = copy.deepcopy(result)
    wrong_execution["execution_status"] = "DRIFTED_PHASE_STATUS"
    with pytest.raises(ValueError, match="execution status"):
        runner._validated_result(
            wrong_execution, runner._parser().parse_args(argv)
        )


def test_runner_accepts_evidence_neutral_failed_completion_only_without_preview(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path, committed=True)
    args = runner._parser().parse_args([
        "verify", *_base_cli(case),
        "--expected-campaign-digest",
        case["digests"]["expected_campaign_digest"],
        "--expected-lease-digest",
        case["digests"]["expected_lease_digest"],
    ])
    failed = _advance_result(case, verified=True)
    failed["status"] = "VERIFIED_" + EXECUTED
    failed["phase"] = "CALLBACK_COMPLETED"
    failed["execution_status"] = (
        "COMPLETED_FAILED_EVIDENCE_NEUTRAL_RECOVERED_LOCAL_ANCHORS_"
        "NOT_INDEPENDENT_HARD_STOP"
    )
    failed["advance_digest"] = None
    failed["next_attempt_root"] = None
    for key in (
        "reingestion_digest",
        "output_evidence_digest",
        "output_report_body_digest",
        "output_audit_head",
        "next_pending_evidence_digest",
        "next_first_pending_projection_digest",
        "next_bundle_digest",
        "next_plan_digest",
    ):
        failed[key] = None
    assert runner._validated_result(failed, args) == failed

    calls = []

    def verify_failed(*call_args, **call_kwargs):
        calls.append((call_args, call_kwargs))
        value = copy.deepcopy(failed)
        if call_kwargs.get("expected_receipt_digest") is not None:
            value["execution_status"] = (
                "COMPLETED_FAILED_EVIDENCE_NEUTRAL_"
                "INDEPENDENTLY_ANCHORED_HARD_STOP"
            )
        return value

    monkeypatch.setattr(
        runner,
        "_load_campaign_core",
        lambda: SimpleNamespace(verify_recursive_campaign=verify_failed),
    )
    base_argv = [
        "verify", *_base_cli(case),
        "--expected-campaign-digest",
        case["digests"]["expected_campaign_digest"],
        "--expected-lease-digest",
        case["digests"]["expected_lease_digest"],
    ]
    assert runner.main(base_argv) == 0
    recovered_stdout = json.loads(capsys.readouterr().out)
    assert "RECOVERED_LOCAL_ANCHORS_NOT_INDEPENDENT" in (
        recovered_stdout["execution_status"]
    )

    anchored_argv = [
        *base_argv,
        "--expected-receipt-digest",
        case["digests"]["expected_receipt_digest"],
        "--expected-journal-head-digest",
        case["digests"]["expected_journal_head_digest"],
    ]
    assert runner.main(anchored_argv) == 0
    anchored_stdout = json.loads(capsys.readouterr().out)
    assert "INDEPENDENTLY_ANCHORED" in anchored_stdout["execution_status"]
    assert calls[-1][1]["expected_receipt_digest"] == case["digests"][
        "expected_receipt_digest"
    ]
    assert calls[-1][1]["expected_journal_head_digest"] == case["digests"][
        "expected_journal_head_digest"
    ]

    contradictory = copy.deepcopy(failed)
    contradictory["output_evidence_digest"] = case["digests"][
        "expected_output_evidence_digest"
    ]
    with pytest.raises(ValueError, match="future anchor"):
        runner._validated_result(contradictory, args)


def test_runner_rejects_phase_anchor_contradiction(tmp_path):
    case = _unit_case(tmp_path, committed=True)
    args = runner._parser().parse_args([
        "verify", *_base_cli(case),
        "--expected-campaign-digest",
        case["digests"]["expected_campaign_digest"],
        "--expected-lease-digest",
        case["digests"]["expected_lease_digest"],
    ])
    contradictory = _advance_result(case, verified=True)
    contradictory["status"] = "VERIFIED_" + AUTHORIZED
    contradictory["phase"] = "AUTHORIZED"
    contradictory["execution_status"] = "NOT_EXECUTED"
    contradictory["next_attempt_root"] = None
    # A not-yet-executed authorization cannot expose callback, receipt,
    # preview, or advance anchors.
    with pytest.raises(ValueError, match="future anchor"):
        runner._validated_result(contradictory, args)


def test_runner_startup_is_lazy_and_has_no_native_or_network_primitive():
    source = RUNNER.read_text(encoding="utf-8")
    for forbidden in (
        "benchmark_lodo_meta_prior",
        "run_one(",
        "import subprocess",
        "import requests",
        "import socket",
        "from scheduler",
        "import scheduler",
    ):
        assert forbidden not in source.lower()
    assert set(runner._parser()._subparsers._group_actions[0].choices) == {
        "inspect",
        "authorize",
        "execute",
        "advance",
        "verify",
    }

    script = r'''
import builtins
blocked = {
    "numpy",
    "performance.benchmark_lodo_meta_prior",
    "performance.benchmark_quality",
}
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name in blocked:
        raise AssertionError("campaign CLI startup crossed callback boundary")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
from runners.run_structural_hypothesis_recursive_campaign import _parser
assert set(_parser()._subparsers._group_actions[0].choices) == {
    "inspect", "authorize", "execute", "advance", "verify"
}
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT),
            "SCOLHKG_OFFLINE": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""


def test_runner_environment_gate_precedes_core_import_and_state_write(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)
    callback_starts = []
    core_imports = []
    monkeypatch.delenv("OPENBLAS_NUM_THREADS")
    monkeypatch.setattr(
        runner,
        "_load_campaign_core",
        lambda: core_imports.append(True),
    )
    argv = [
        "authorize", *_base_cli(case),
        "--campaign-id", case["campaign_id"],
        *_digests_cli(case, runner._AUTHORIZE_DIGEST_NAMES),
        "--task-id", case["task_id"],
        "--authorization-id", case["authorization_id"],
        "--confirm-explicit-local-task-authorization",
    ]
    assert runner.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "before Python startup" in captured.err
    assert core_imports == []
    assert callback_starts == []
    assert not case["campaign_root"].exists()


def test_campaign_path_is_exact_xdg_prefix_and_brand_new_inspection_is_zero_write(
    tmp_path, monkeypatch
):
    from performance import structural_hypothesis_recursive_campaign as core

    state_home = (tmp_path / "brand-new-state").resolve()
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    campaign_id = "path-kat-0001"
    planned = (
        state_home
        / "kg-op/structural-hypothesis-recursive-campaign/v1"
        / campaign_id
    )
    assert not state_home.exists()
    assert core._campaign_root(
        planned, campaign_id, fresh=True
    ) == planned
    assert not state_home.exists()

    fake_suffix = (
        tmp_path
        / "x/kg-op/structural-hypothesis-recursive-campaign/v1"
        / campaign_id
    ).resolve()
    with pytest.raises(core.RecursiveCampaignError):
        core._campaign_root(fake_suffix, campaign_id, fresh=True)

    dotted = str(
        state_home
        / "kg-op/structural-hypothesis-recursive-campaign/v1"
        / ".."
        / "v1"
        / campaign_id
    )
    with pytest.raises(core.RecursiveCampaignError):
        core._campaign_root(dotted, campaign_id, fresh=True)

    real_state = tmp_path / "real-state"
    real_state.mkdir(mode=0o700)
    aliased_state = tmp_path / "aliased-state"
    aliased_state.symlink_to(real_state, target_is_directory=True)
    monkeypatch.setenv("XDG_STATE_HOME", str(aliased_state))
    with pytest.raises(core.RecursiveCampaignError):
        core._campaign_root(
            aliased_state
            / "kg-op/structural-hypothesis-recursive-campaign/v1"
            / campaign_id,
            campaign_id,
            fresh=True,
        )


def test_campaign_path_accepts_shared_owned_nonwritable_kg_op_ancestor(
    monkeypatch,
):
    """Keep shared kg-op ancestry intact while owning the module subtree."""
    from performance import structural_hypothesis_recursive_campaign as core

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=".recursive-campaign-shared-", dir=Path.home()
        )
    )
    try:
        temporary_root.chmod(0o700)
        state_home = (temporary_root / "shared-state").resolve()
        state_home.mkdir(mode=0o700)
        shared = state_home / "kg-op"
        shared.mkdir(mode=0o755)
        monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
        campaign_id = "shared-ancestor-kat"
        planned = (
            shared
            / "structural-hypothesis-recursive-campaign/v1"
            / campaign_id
        )
        before = _tree_observation(state_home)

        assert core._campaign_root(
            planned, campaign_id, fresh=True
        ) == planned
        assert _tree_observation(state_home) == before
        assert (shared.stat().st_mode & 0o777) == 0o755

        # These are the exact path-creation helpers used by authorization
        # after all independent source/runtime anchors have passed.
        core._ensure_campaign_parent(planned)
        module = shared / "structural-hypothesis-recursive-campaign"
        version = module / "v1"
        assert (shared.stat().st_mode & 0o777) == 0o755
        assert (module.stat().st_mode & 0o777) == 0o700
        assert (version.stat().st_mode & 0o777) == 0o700
        core._mkdir_new(planned)
        assert (planned.stat().st_mode & 0o777) == 0o700

        unsafe_state = (temporary_root / "unsafe-shared-state").resolve()
        unsafe_state.mkdir(mode=0o700)
        unsafe_shared = unsafe_state / "kg-op"
        unsafe_shared.mkdir(mode=0o775)
        unsafe_shared.chmod(0o775)
        monkeypatch.setenv("XDG_STATE_HOME", str(unsafe_state))
        unsafe_planned = (
            unsafe_shared
            / "structural-hypothesis-recursive-campaign/v1"
            / campaign_id
        )
        with pytest.raises(core.RecursiveCampaignError):
            core._campaign_root(unsafe_planned, campaign_id, fresh=True)

        symlink_state = (temporary_root / "symlink-shared-state").resolve()
        symlink_state.mkdir(mode=0o700)
        symlink_target = temporary_root / "shared-target"
        symlink_target.mkdir(mode=0o755)
        (symlink_state / "kg-op").symlink_to(
            symlink_target, target_is_directory=True
        )
        monkeypatch.setenv("XDG_STATE_HOME", str(symlink_state))
        symlink_planned = (
            symlink_state
            / "kg-op/structural-hypothesis-recursive-campaign/v1"
            / campaign_id
        )
        with pytest.raises(core.RecursiveCampaignError):
            core._campaign_root(symlink_planned, campaign_id, fresh=True)

        wrong_module_state = (
            temporary_root / "wrong-module-state"
        ).resolve()
        wrong_module_state.mkdir(mode=0o700)
        wrong_shared = wrong_module_state / "kg-op"
        wrong_shared.mkdir(mode=0o755)
        wrong_module = (
            wrong_shared / "structural-hypothesis-recursive-campaign"
        )
        wrong_module.mkdir(mode=0o755)
        monkeypatch.setenv("XDG_STATE_HOME", str(wrong_module_state))
        wrong_module_planned = wrong_module / "v1" / campaign_id
        with pytest.raises(core.RecursiveCampaignError):
            core._campaign_root(
                wrong_module_planned, campaign_id, fresh=True
            )
    finally:
        shutil.rmtree(temporary_root)


def test_campaign_path_atomic_writer_survives_process_interruptions(
    monkeypatch
):
    """Exercise both sides of the single rename-no-replace commit point."""
    from performance import structural_hypothesis_recursive_campaign as core

    if not hasattr(os, "fork"):
        pytest.skip("atomic campaign writer KAT requires fork")
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=".recursive-campaign-atomic-", dir=Path.home()
        )
    )
    try:
        temporary_root.chmod(0o700)
        state_home = temporary_root / "state"
        state_home.mkdir(mode=0o700)
        monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
        prefix = state_home
        for part in core._STATE_PREFIX.parts:
            prefix = prefix / part
            prefix.mkdir(mode=0o700)
            prefix.chmod(0o700)
        root = prefix / "atomic-writer-kat"
        root.mkdir(mode=0o700)
        root_identity = core._directory_identity(root)
        value = {
            "schema_version": "atomic-writer-kat/1",
            "status": "EXACT_TEST_VALUE",
            "ordinal": 1,
        }
        raw = core._canonical_file(value)

        wrong_parent = prefix / "wrong-parent"
        wrong_parent.mkdir(mode=0o700)
        wrong_target = wrong_parent / "identity-mismatch.json"
        with pytest.raises(core.RecursiveCampaignError):
            core._write_new(
                wrong_target,
                value,
                expected_parent_identity=root_identity,
            )
        assert not wrong_target.exists()
        assert list(wrong_parent.iterdir()) == []

        def interrupted_writer(target, *, after_publish):
            child = os.fork()
            if child == 0:  # pragma: no cover - assertions run in parent
                real_rename = core._rename_noreplace

                def interrupt_at_commit(*args, **kwargs):
                    if after_publish:
                        real_rename(*args, **kwargs)
                    os.kill(os.getpid(), signal.SIGKILL)
                    os._exit(97)

                core._rename_noreplace = interrupt_at_commit
                try:
                    core._write_new(
                        target,
                        value,
                        expected_parent_identity=root_identity,
                    )
                except BaseException:
                    os._exit(96)
                os._exit(95)
            waited, status = os.waitpid(child, 0)
            assert waited == child
            assert os.WIFSIGNALED(status)
            assert os.WTERMSIG(status) == signal.SIGKILL

        pre_target = root / "pre-publish.json"
        interrupted_writer(pre_target, after_publish=False)
        assert not pre_target.exists()
        assert list(root.iterdir()) == []
        pre_orphans = {
            path
            for path in prefix.iterdir()
            if path.name.startswith(".tmp-")
        }
        assert len(pre_orphans) == 1
        pre_orphan = next(iter(pre_orphans))
        assert pre_orphan.read_bytes() == raw
        assert (pre_orphan.stat().st_mode & 0o777) == 0o600
        assert pre_orphan.stat().st_nlink == 1

        # The sibling staging orphan is evidence of an interrupted process,
        # not a campaign-root leaf, and cannot prevent exact retry.
        core._publish_or_match(
            pre_target,
            value,
            expected_parent_identity=root_identity,
        )
        assert pre_target.read_bytes() == raw
        assert pre_target.stat().st_nlink == 1
        assert {
            path
            for path in prefix.iterdir()
            if path.name.startswith(".tmp-")
        } == pre_orphans

        post_target = root / "post-publish.json"
        interrupted_writer(post_target, after_publish=True)
        assert post_target.read_bytes() == raw
        assert (post_target.stat().st_mode & 0o777) == 0o600
        assert post_target.stat().st_nlink == 1
        assert {
            path
            for path in prefix.iterdir()
            if path.name.startswith(".tmp-")
        } == pre_orphans

        committed = (
            post_target.stat().st_dev,
            post_target.stat().st_ino,
            post_target.read_bytes(),
        )
        core._publish_or_match(
            post_target,
            value,
            expected_parent_identity=root_identity,
        )
        assert (
            post_target.stat().st_dev,
            post_target.stat().st_ino,
            post_target.read_bytes(),
        ) == committed
        with pytest.raises(core.RecursiveCampaignError):
            core._write_new(
                post_target,
                {**value, "ordinal": 2},
                expected_parent_identity=root_identity,
            )
        assert (
            post_target.stat().st_dev,
            post_target.stat().st_ino,
            post_target.read_bytes(),
        ) == committed
        assert post_target.stat().st_nlink == 1
        assert {
            path
            for path in prefix.iterdir()
            if path.name.startswith(".tmp-")
        } == pre_orphans
    finally:
        shutil.rmtree(temporary_root)


def test_recursive_source_semantic_verification_is_linear_and_detects_oldest_drift(
    monkeypatch,
):
    """One public inspect admits each predecessor generation exactly once."""
    from performance import structural_hypothesis_recursive_campaign as core

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=".recursive-campaign-linear-", dir=Path.home()
        )
    )
    try:
        temporary_root.chmod(0o700)
        state_home = temporary_root / "state"
        state_home.mkdir(mode=0o700)
        monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
        shared = state_home / "kg-op"
        shared.mkdir(mode=0o755)
        module = shared / "structural-hypothesis-recursive-campaign"
        module.mkdir(mode=0o700)
        prefix = module / "v1"
        prefix.mkdir(mode=0o700)

        dependencies = {
            "hypothesis_contract_path": str(
                advance_test.HYPOTHESIS_CONTRACT
            ),
            "executor_contract_path": str(
                advance_test.EXECUTOR_CONTRACT
            ),
            "runtime_contract_path": str(advance_test.RUNTIME_CONTRACT),
            "materializer_contract_path": str(
                advance_test.MATERIALIZER_CONTRACT
            ),
            "base_manifest_path": str(advance_test.BASE_MANIFEST),
            "asset_root": str(advance_test.ASSET_ROOT),
        }
        bootstrap_args = [
            str((temporary_root / f"bootstrap-arg-{index}").resolve())
            for index in range(21)
        ]
        for index, key in {
            7: "hypothesis_contract_path",
            8: "executor_contract_path",
            9: "runtime_contract_path",
            11: "materializer_contract_path",
            13: "base_manifest_path",
            14: "asset_root",
        }.items():
            bootstrap_args[index] = dependencies[key]
        bootstrap_kwargs = {
            key: (
                _digest("bootstrap-" + key)
                if key.endswith("_digest") or key.endswith("_head")
                else "synthetic-" + key
            )
            for key in core._SUCCESSOR_VERIFY_KWARGS
        }
        bootstrap_source = {
            "schema_version": core.SOURCE_SCHEMA_VERSION,
            "source_kind": "recursive_successor_v1",
            "dependencies": copy.deepcopy(dependencies),
            "verify_args": bootstrap_args,
            "verify_kwargs": bootstrap_kwargs,
        }

        bundle_digest = _digest("linear-bundle")
        plan_digest = _digest("linear-plan")
        evidence_digest = _digest("linear-evidence")
        task = {
            "task_id": "task:" + "a" * 24,
            "task_digest": _digest("linear-task"),
            "ordinal": 0,
            "cell": {
                "profile": "LinearDepthPolicy",
                "domain": "synthetic",
                "seed": 1,
                "d": 1,
                "N": 1,
                "n0": 1,
            },
        }
        rows = [{"synthetic_row": 1}]
        report = {
            "evidence_digest": evidence_digest,
            "pending_evidence": [copy.deepcopy(task["cell"])],
            "audit": {
                "report_body_digest": _digest("linear-report"),
                "head": _digest("linear-audit"),
            },
        }
        bundle = {
            "status": "MATERIALIZED_NOT_AUTHORIZED",
            "task_count": 1,
            "integrity": {"bundle_digest": bundle_digest},
            "plan": {
                "proposal_count": 1,
                "integrity": {"plan_digest": plan_digest},
                "tasks": [task],
            },
        }
        bootstrap_attempt = (
            state_home
            / "kg-op/structural-hypothesis-execution/v1"
            / "linear-bootstrap"
        )

        def synthetic_successor(_descriptor, *, require_attempt_absent):
            if require_attempt_absent and bootstrap_attempt.exists():
                raise core.RecursiveCampaignError(
                    "synthetic bootstrap attempt unexpectedly exists"
                )
            return {
                "source_kind": "recursive_successor_v1",
                "source_state_digest": bootstrap_kwargs[
                    "expected_recursive_successor_digest"
                ],
                "source_verification": {
                    "status": "VERIFIED_SYNTHETIC_BOOTSTRAP"
                },
                "rows": copy.deepcopy(rows),
                "report": copy.deepcopy(report),
                "bundle": copy.deepcopy(bundle),
                "next_attempt_root": bootstrap_attempt,
                "checkpoint_root": bootstrap_attempt / "checkpoints",
            }

        roots = []
        source_by_root = {}
        expected_by_root = {}
        attempt_by_root = {}
        anchor_names = (
            "expected_campaign_digest",
            "expected_lease_digest",
            "expected_callback_start_claim_digest",
            "expected_advance_digest",
            "expected_output_evidence_digest",
            "expected_output_report_body_digest",
            "expected_output_audit_head",
            "expected_reingestion_digest",
            "expected_next_bundle_digest",
            "expected_next_plan_digest",
        )

        for depth in range(1, 7):
            root = prefix / f"linear-prior-{depth}"
            root.mkdir(mode=0o700)
            source_dir = root / "source"
            source_dir.mkdir(mode=0o700)
            prior_source = (
                bootstrap_source
                if depth == 1
                else source_by_root[str(roots[-1].resolve())]
            )
            source_path = _write_json(
                source_dir / "descriptor.json", prior_source
            )
            source_path.chmod(0o600)
            attempt_by_root[str(root.resolve())] = (
                state_home
                / "kg-op/structural-hypothesis-execution/v1"
                / f"linear-next-{depth}"
            )
            descriptor = {
                "schema_version": core.SOURCE_SCHEMA_VERSION,
                "source_kind": "recursive_campaign_v1",
                "dependencies": copy.deepcopy(dependencies),
                "campaign_contract_path": str(CAMPAIGN_CONTRACT),
                "campaign_root": str(root),
                **{
                    name: (
                        evidence_digest
                        if name == "expected_output_evidence_digest"
                        else bundle_digest
                        if name == "expected_next_bundle_digest"
                        else plan_digest
                        if name == "expected_next_plan_digest"
                        else _digest(f"linear-{depth}-{name}")
                    )
                    for name in anchor_names
                },
            }
            resolved = str(root.resolve())
            roots.append(root)
            source_by_root[resolved] = descriptor
            expected_by_root[resolved] = descriptor

        real_uncached = core._verify_campaign_internal_uncached
        real_check_freshness = core._check_campaign_freshness_record
        real_validate_snapshots = core._validate_report_rows_bundle
        real_record_miss = core._record_semantic_cache_miss
        real_verify_freshness = core._verify_context_freshness

        def synthetic_validate_snapshots(
            *, rows, report, bundle, dependencies, checkpoint_root
        ):
            return (
                copy.deepcopy(rows),
                copy.deepcopy(report),
                copy.deepcopy(bundle),
            )

        def synthetic_uncached(
            source_value,
            campaign_contract_path,
            campaign_root,
            *,
            campaign_id,
            depth=0,
            visited=None,
            verification_context=None,
            **expected,
        ):
            context = core._require_verification_context(
                verification_context
            )
            resolved = str(Path(campaign_root).resolve(strict=True))
            descriptor = expected_by_root[resolved]
            for name in anchor_names:
                assert expected[name] == descriptor[name]

            # Perform the predecessor traversal once, then replay the exact
            # predecessor semantic request.  The replay must be a memo hit;
            # without the per-call cache this recurrence doubles by depth.
            core._derive_source(
                source_value,
                require_attempt_absent=False,
                depth=depth,
                visited=visited,
                verification_context=context,
            )
            if source_value["source_kind"] == "recursive_campaign_v1":
                predecessor = Path(source_value["campaign_root"])
                predecessor_source = core._read_json(
                    predecessor / "source/descriptor.json",
                    "synthetic predecessor descriptor",
                    exact_mode=0o600,
                )
                core._verify_campaign_internal(
                    predecessor_source,
                    source_value["campaign_contract_path"],
                    predecessor,
                    campaign_id=predecessor.name,
                    depth=depth + 1,
                    visited=set() if visited is None else set(visited),
                    verification_context=context,
                    **{
                        name: source_value[name] for name in anchor_names
                    },
                )

            result = {
                "status": "VERIFIED_SYNTHETIC_ADVANCED_NONTERMINAL",
                "phase": "ADVANCED_NONTERMINAL",
                "terminal_status": "NONTERMINAL",
                "remaining_task_count": 1,
                "next_attempt_root": str(attempt_by_root[resolved]),
                "advance_digest": expected["expected_advance_digest"],
                "output_evidence_digest": evidence_digest,
                "output_report_body_digest": expected[
                    "expected_output_report_body_digest"
                ],
                "output_audit_head": expected[
                    "expected_output_audit_head"
                ],
                "reingestion_digest": expected[
                    "expected_reingestion_digest"
                ],
                "next_bundle_digest": bundle_digest,
                "next_plan_digest": plan_digest,
            }
            source_leaf = Path(campaign_root) / "source/descriptor.json"
            record = {
                "resolved_root": resolved,
                "source_leaf": source_leaf,
                "source_raw": source_leaf.read_bytes(),
            }
            return {
                "result": result,
                "capture": {
                    "advance_values": {
                        "combined_rows": copy.deepcopy(rows),
                        "output_report": copy.deepcopy(report),
                        "next_bundle": copy.deepcopy(bundle),
                    }
                },
                "base": {"synthetic_root": resolved},
                "freshness_record": record,
            }

        def synthetic_check_freshness(record):
            if record["source_leaf"].read_bytes() != record["source_raw"]:
                raise core.RecursiveCampaignError(
                    "synthetic oldest predecessor changed"
                )
            return {"synthetic_root": record["resolved_root"]}

        misses = []
        freshness_observations = []

        def observe_miss(context, resolved_root):
            real_record_miss(context, resolved_root)
            misses.append(resolved_root)

        def observe_final_freshness(context):
            freshness_observations.append(
                {
                    "misses": list(context["semantic_cache_miss_order"]),
                    "roots": list(context["campaign_freshness"]),
                }
            )
            return real_verify_freshness(context)

        monkeypatch.setattr(
            core, "_derive_successor_source", synthetic_successor
        )
        monkeypatch.setattr(
            core, "_validate_report_rows_bundle", synthetic_validate_snapshots
        )
        monkeypatch.setattr(
            core, "_verify_campaign_internal_uncached", synthetic_uncached
        )
        monkeypatch.setattr(
            core, "_check_campaign_freshness_record", synthetic_check_freshness
        )
        monkeypatch.setattr(
            core, "_record_semantic_cache_miss", observe_miss
        )
        monkeypatch.setattr(
            core, "_verify_context_freshness", observe_final_freshness
        )

        for depth in range(1, 7):
            misses = []
            freshness_observations = []
            source_value = source_by_root[str(roots[depth - 1].resolve())]
            inspection_root = prefix / f"linear-inspect-{depth}"
            before = _tree_observation(state_home)
            inspected = core.inspect_recursive_campaign(
                source_value,
                CAMPAIGN_CONTRACT,
                inspection_root,
                campaign_id=inspection_root.name,
                next_attempt_root=attempt_by_root[
                    str(roots[depth - 1].resolve())
                ],
            )
            expected_order = [
                str(root.resolve()) for root in reversed(roots[:depth])
            ]
            assert misses == expected_order
            assert len(set(misses)) == depth
            assert freshness_observations == [
                {
                    "misses": expected_order,
                    "roots": [
                        str(root.resolve()) for root in roots[:depth]
                    ],
                }
            ]
            assert inspected["source_kind"] == "recursive_campaign_v1"
            assert inspected["task_count"] == 1
            assert inspected["task_id"] == task["task_id"]
            assert not inspection_root.exists()
            assert _tree_observation(state_home) == before

        # Change the oldest admitted predecessor only after semantic traversal
        # has completed.  The final O(depth) freshness ledger must still catch
        # it before public inspect returns.
        oldest_leaf = roots[0] / "source/descriptor.json"
        oldest_raw = oldest_leaf.read_bytes()
        real_provenance = core._provenance_binding
        drifted = []

        def drift_oldest_before_final_freshness(*args, **kwargs):
            value = real_provenance(*args, **kwargs)
            if not drifted:
                drifted.append(True)
                oldest_leaf.write_bytes(oldest_raw + b"\n")
                oldest_leaf.chmod(0o600)
            return value

        monkeypatch.setattr(
            core, "_provenance_binding", drift_oldest_before_final_freshness
        )
        misses = []
        freshness_observations = []
        drift_root = prefix / "linear-inspect-oldest-drift"
        before_drift = _tree_observation(state_home)
        with pytest.raises(
            core.RecursiveCampaignError,
            match="oldest predecessor changed",
        ):
            core.inspect_recursive_campaign(
                source_by_root[str(roots[-1].resolve())],
                CAMPAIGN_CONTRACT,
                drift_root,
                campaign_id=drift_root.name,
                next_attempt_root=attempt_by_root[
                    str(roots[-1].resolve())
                ],
            )
        assert drifted == [True]
        assert misses == [
            str(root.resolve()) for root in reversed(roots)
        ]
        assert not drift_root.exists()
        oldest_leaf.write_bytes(oldest_raw)
        oldest_leaf.chmod(0o600)
        monkeypatch.setattr(core, "_provenance_binding", real_provenance)
        assert _tree_observation(state_home) == before_drift

        # Keep explicit references so future internal refactors cannot make
        # this synthetic KAT silently shadow unused originals.
        assert callable(real_uncached)
        assert callable(real_check_freshness)
        assert callable(real_validate_snapshots)
    finally:
        shutil.rmtree(temporary_root)


def test_callback_claim_publication_guards_late_campaign_generation_drift(
    monkeypatch, request
):
    """Reject both sides of the final guarded claim-publication window."""
    from performance import structural_hypothesis_recursive_campaign as core

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix="kgop-recursive-claim-guard.", dir=Path.home()
        )
    ).resolve()
    temporary_root.chmod(0o700)
    request.addfinalizer(lambda: shutil.rmtree(temporary_root))
    state_home = temporary_root / "state"
    state_home.mkdir(mode=0o700)
    shared = state_home / "kg-op"
    shared.mkdir(mode=0o755)
    module = shared / "structural-hypothesis-recursive-campaign"
    module.mkdir(mode=0o700)
    prefix = module / "v1"
    prefix.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    root = prefix / "claim-publication-window"
    root.mkdir(mode=0o700)
    source_root = root / "source"
    source_root.mkdir(mode=0o700)
    source_values = {
        "descriptor": {"fixture": "descriptor"},
        "commit": {"fixture": "commit"},
        "rows": [{"fixture": "row"}],
        "report": {"fixture": "report"},
        "bundle": {"fixture": "bundle"},
    }
    for label, relative in core._SOURCE_LAYOUT.items():
        leaf = _write_json(source_root / relative, source_values[label])
        leaf.chmod(0o600)
    root_values = {
        "campaign_contract.json": {"fixture": "contract"},
        "lease.json": {"fixture": "lease"},
        "campaign.json": {"fixture": "campaign"},
    }
    for name, value in root_values.items():
        leaf = _write_json(root / name, value)
        leaf.chmod(0o600)

    attempt_root = state_home / "attempt-never-entered"
    task_id = "task:" + "c" * 24
    authorization_digest = _digest("claim-window-authorization")
    attempt_digest = _digest("claim-window-attempt")
    base_template = {
        "capture": {
            "has_claim": False,
            "has_advance": False,
            "root_identity": core._directory_identity(root),
            "source_identity": core._directory_identity(source_root),
            "raws": {
                "campaign_contract": (root / "campaign_contract.json").read_bytes(),
                "lease": (root / "lease.json").read_bytes(),
                "campaign_commit": (root / "campaign.json").read_bytes(),
                **{
                    "source_" + label: (source_root / relative).read_bytes()
                    for label, relative in core._SOURCE_LAYOUT.items()
                },
            },
        },
        "derived": {
            "descriptor": {"dependencies": {"fixture": "only"}},
            "rows": copy.deepcopy(source_values["rows"]),
            "report": copy.deepcopy(source_values["report"]),
            "bundle": copy.deepcopy(source_values["bundle"]),
            "next_attempt_root": attempt_root,
        },
        "campaign_digest": _digest("claim-window-campaign"),
        "lease_digest": _digest("claim-window-lease"),
        "provenance_binding": {"task": {"task_id": task_id}},
        "provenance_binding_digest": _digest("claim-window-provenance"),
        "authorization_id": "recursive-campaign-v1:" + "d" * 64,
        "authorization_digest": authorization_digest,
        "attempt_digest": attempt_digest,
    }
    claim_path = root / "callback_start_claim.json"
    validation_calls = []

    def synthetic_base(*args, **kwargs):
        validation_calls.append((args, kwargs))
        value = copy.deepcopy(base_template)
        value["capture"]["has_claim"] = claim_path.exists()
        return value

    dependency_objects = {
        "runtime": {"fixture": "runtime"},
        "paths": {
            "runtime_contract_path": RUNTIME_CONTRACT.resolve(),
            "base_manifest_path": advance_test.BASE_MANIFEST.resolve(),
            "asset_root": advance_test.ASSET_ROOT.resolve(),
        },
    }
    runtime_verifications = []

    def authorized_runtime(*args, **kwargs):
        runtime_verifications.append((args, kwargs))
        return {
            "status": "VERIFIED_AUTHORIZED_NOT_EXECUTED",
            "authorization_digest": authorization_digest,
            "attempt_digest": attempt_digest,
        }

    delegate_calls = []

    def forbidden_delegate(*args, **kwargs):
        delegate_calls.append((args, kwargs))
        raise AssertionError("runtime delegate crossed a rejected claim boundary")

    monkeypatch.setattr(core, "_validate_campaign_base", synthetic_base)
    monkeypatch.setattr(
        core, "_load_dependency_objects", lambda _dependencies: dependency_objects
    )
    monkeypatch.setattr(core, "_R_VERIFY", authorized_runtime)
    monkeypatch.setattr(core, "_R_EXECUTE", forbidden_delegate)
    execute_kwargs = {
        "expected_campaign_digest": base_template["campaign_digest"],
        "expected_lease_digest": base_template["lease_digest"],
        "expected_provenance_binding_digest": base_template[
            "provenance_binding_digest"
        ],
        "expected_authorization_digest": authorization_digest,
        "expected_attempt_digest": attempt_digest,
        "confirm_real_local_execution": True,
    }
    root_before = _tree_observation(root)
    attempt_before = _tree_observation(attempt_root)

    # Boundary A: drift immediately after the final public freshness pass is
    # still before publication.  The held source guard must prevent any claim.
    bundle_path = source_root / core._SOURCE_LAYOUT["bundle"]
    bundle_raw = bundle_path.read_bytes()
    real_verify_freshness = core._verify_context_freshness
    freshness_mutations = []

    def mutate_after_final_freshness(context):
        real_verify_freshness(context)
        if not freshness_mutations:
            freshness_mutations.append(True)
            bundle_path.write_bytes(bundle_raw + b"\n")

    monkeypatch.setattr(
        core, "_verify_context_freshness", mutate_after_final_freshness
    )
    with pytest.raises(core.RecursiveCampaignError):
        core.execute_recursive_campaign_task(
            {"fixture": "source"},
            CAMPAIGN_CONTRACT,
            root,
            RUNTIME_CONTRACT,
            **execute_kwargs,
        )
    assert freshness_mutations == [True]
    assert len(validation_calls) == 2
    assert len(runtime_verifications) == 2
    assert delegate_calls == []
    assert not claim_path.exists()
    assert _tree_observation(attempt_root) == attempt_before
    bundle_path.write_bytes(bundle_raw)
    bundle_path.chmod(0o600)
    assert _tree_observation(root) == root_before

    # Boundary B: mutate after the held-directory pre-read but before the
    # atomic rename.  The post-read must reject before the delegate.  Because
    # the no-replace rename already committed, this is deliberately a spent
    # hard stop rather than an absent/retryable claim.
    monkeypatch.setattr(
        core, "_verify_context_freshness", real_verify_freshness
    )
    real_rename = core._rename_noreplace
    lease_path = root / "lease.json"
    lease_raw = lease_path.read_bytes()
    post_guard_mutations = []

    def mutate_after_claim_guard(*args, **kwargs):
        guard_before = kwargs.get("guard_before")
        if args[3] == "callback_start_claim.json":
            assert guard_before is not None
            guard_before()
            post_guard_mutations.append(True)
            lease_path.write_bytes(lease_raw + b"\n")
            forwarded = dict(kwargs)
            forwarded["guard_before"] = None
            return real_rename(*args, **forwarded)
        return real_rename(*args, **kwargs)

    monkeypatch.setattr(core, "_rename_noreplace", mutate_after_claim_guard)
    with pytest.raises(core.RecursiveCampaignError):
        core.execute_recursive_campaign_task(
            {"fixture": "source"},
            CAMPAIGN_CONTRACT,
            root,
            RUNTIME_CONTRACT,
            **execute_kwargs,
        )
    assert post_guard_mutations == [True]
    assert len(validation_calls) == 4
    assert len(runtime_verifications) == 4
    assert delegate_calls == []
    assert claim_path.is_file()
    assert (claim_path.stat().st_mode & 0o777) == 0o600
    assert claim_path.stat().st_nlink == 1
    assert _tree_observation(attempt_root) == attempt_before
    lease_path.write_bytes(lease_raw)
    lease_path.chmod(0o600)
    spent_tree = _tree_observation(root)
    claim_observation = spent_tree.pop("callback_start_claim.json")
    assert spent_tree == root_before
    assert claim_observation[0] != b""

    # The durable claim is the only irreversible state: a later execute is
    # rejected from the captured phase before runtime verification/delegation.
    with pytest.raises(core.RecursiveCampaignError):
        core.execute_recursive_campaign_task(
            {"fixture": "source"},
            CAMPAIGN_CONTRACT,
            root,
            RUNTIME_CONTRACT,
            **execute_kwargs,
        )
    assert len(validation_calls) == 5
    assert len(runtime_verifications) == 4
    assert delegate_calls == []
    assert _tree_observation(attempt_root) == attempt_before


def test_fake_only_recursive_campaign_one_step_kat_and_hard_stops(
    fake_campaign_seed2_case, monkeypatch
):
    """Exercise 1,352/28 -> 1,353/27 without a native callback."""
    from performance import structural_hypothesis_recursive_campaign as core
    from performance import structural_hypothesis_single_task_runtime as runtime_core

    case = fake_campaign_seed2_case
    source = case["campaign_source"]
    root = case["campaign_root"]
    attempt_root = case["future_attempt_root"]
    campaign_prefix = root.parent
    assert not campaign_prefix.exists()
    assert not root.exists()
    assert not attempt_root.exists()

    real_derive_source = core._derive_source
    inspected_source_captures = []

    def capture_inspected_source(source_value, **kwargs):
        value = real_derive_source(source_value, **kwargs)
        if source_value == source:
            inspected_source_captures.append(copy.deepcopy(value))
        return value

    monkeypatch.setattr(core, "_derive_source", capture_inspected_source)
    before_inspect = _tree_observation(case["state_home"])
    inspected = core.inspect_recursive_campaign(
        source,
        CAMPAIGN_CONTRACT,
        root,
        campaign_id=case["campaign_id"],
        next_attempt_root=attempt_root,
    )
    assert inspected["status"] == INSPECTED
    assert inspected["source_kind"] == "recursive_successor_v1"
    assert inspected["task_count"] == 28
    assert inspected["terminal_status"] == "NONTERMINAL"
    assert inspected["next_attempt_root"] == str(attempt_root)
    assert inspected["checkpoint_root"] == str(attempt_root / "checkpoints")
    assert inspected["task_id"] == case["recursive_materialized"][
        "first_task_id"
    ]
    assert _tree_observation(case["state_home"]) == before_inspect
    assert not campaign_prefix.exists()
    assert not attempt_root.exists()
    assert inspected_source_captures
    inspected_source = inspected_source_captures[-1]
    monkeypatch.setattr(core, "_derive_source", real_derive_source)

    authorize_kwargs = {
        "campaign_id": case["campaign_id"],
        "expected_source_state_digest": inspected["source_state_digest"],
        "expected_bundle_digest": inspected["bundle_digest"],
        "expected_plan_digest": inspected["plan_digest"],
        "task_id": inspected["task_id"],
        "expected_task_digest": inspected["task_digest"],
        "expected_provenance_binding_digest": inspected[
            "provenance_binding_digest"
        ],
        "authorization_id": inspected["required_authorization_id"],
    }
    with pytest.raises(core.RecursiveCampaignError):
        core.authorize_recursive_campaign_task(
            source,
            CAMPAIGN_CONTRACT,
            root,
            **authorize_kwargs,
            confirm_explicit_local_task_authorization=False,
        )
    assert not root.exists()
    assert not attempt_root.exists()

    wrong = dict(authorize_kwargs)
    wrong["expected_source_state_digest"] = _digest("wrong-source-state")
    with pytest.raises(core.RecursiveCampaignError):
        core.authorize_recursive_campaign_task(
            source,
            CAMPAIGN_CONTRACT,
            root,
            **wrong,
            confirm_explicit_local_task_authorization=True,
        )
    assert not root.exists()
    assert not attempt_root.exists()

    # Crash after deterministic runtime authorization but before campaign
    # publication.  Retry verifies and reuses that AUTHORIZED attempt; prepare
    # is never called twice.
    prepare_calls = 0
    real_prepare = core._R_PREPARE

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return real_prepare(*args, **kwargs)

    real_ensure_parent = core._ensure_campaign_parent

    def crash_before_campaign(_root):
        raise core.RecursiveCampaignError("injected cross-capsule crash")

    def replay_inspected_source(source_value, **kwargs):
        if source_value == source:
            return copy.deepcopy(inspected_source)
        return real_derive_source(source_value, **kwargs)

    monkeypatch.setattr(core, "_R_PREPARE", counted_prepare)
    monkeypatch.setattr(core, "_derive_source", replay_inspected_source)
    monkeypatch.setattr(core, "_ensure_campaign_parent", crash_before_campaign)
    with pytest.raises(core.RecursiveCampaignError):
        core.authorize_recursive_campaign_task(
            source,
            CAMPAIGN_CONTRACT,
            root,
            **authorize_kwargs,
            confirm_explicit_local_task_authorization=True,
        )
    assert prepare_calls == 1
    assert attempt_root.is_dir()
    assert not root.exists()

    # The successful authorization below is the lifecycle's real full source
    # admission.  Capture its already-verified immutable source for generic
    # publication recovery calls that follow.
    authorization_source_captures = []

    def capture_authorization_source(source_value, **kwargs):
        value = real_derive_source(source_value, **kwargs)
        if source_value == source:
            authorization_source_captures.append(copy.deepcopy(value))
        return value

    monkeypatch.setattr(core, "_derive_source", capture_authorization_source)
    monkeypatch.setattr(core, "_ensure_campaign_parent", real_ensure_parent)
    authorized = core.authorize_recursive_campaign_task(
        source,
        CAMPAIGN_CONTRACT,
        root,
        **authorize_kwargs,
        confirm_explicit_local_task_authorization=True,
    )
    assert prepare_calls == 1
    assert authorized["status"] == AUTHORIZED
    assert authorized["authorization_status"] == "AUTHORIZED"
    assert authorized["execution_status"] == "NOT_EXECUTED"
    assert root.is_dir() and attempt_root.is_dir()
    assert authorization_source_captures
    authorization_recovery_source = authorization_source_captures[-1]

    def replay_authorization_source(source_value, **kwargs):
        if source_value == source:
            return copy.deepcopy(authorization_recovery_source)
        return real_derive_source(source_value, **kwargs)

    monkeypatch.setattr(core, "_derive_source", replay_authorization_source)
    assert {path.name for path in root.iterdir()} == {
        "campaign_contract.json",
        "source",
        "lease.json",
        "campaign.json",
    }
    assert {path.name for path in (root / "source").iterdir()} == {
        "descriptor.json",
        "commit.json",
        "rows.json",
        "report.json",
        "bundle.json",
    }
    assert (root.stat().st_mode & 0o777) == 0o700
    assert all(
        (path.stat().st_mode & 0o777) == (0o700 if path.is_dir() else 0o600)
        for path in root.rglob("*")
    )

    committed_before = _tree_observation(root)
    attempt_before = _tree_observation(attempt_root)
    repeated = core.authorize_recursive_campaign_task(
        source,
        CAMPAIGN_CONTRACT,
        root,
        **authorize_kwargs,
        confirm_explicit_local_task_authorization=True,
    )
    assert repeated == authorized
    assert prepare_calls == 1
    assert _tree_observation(root) == committed_before
    assert _tree_observation(attempt_root) == attempt_before

    # Every publication boundary before campaign.json is recoverable from the
    # already-authorized runtime attempt.  Existing leaves must match exactly,
    # no second prepare is permitted, and the campaign marker is always last.
    authorized_backup = case["state_home"] / "campaign-complete-backup"
    shutil.copytree(root, authorized_backup)
    real_authorization_write_new = core._write_new
    authorization_marker_links = []

    def observe_authorization_marker_last(path, value, *args, **kwargs):
        path = Path(path)
        is_marker = path == root / "campaign.json"
        if is_marker:
            assert set(os.listdir(root)) == {
                "campaign_contract.json",
                "source",
                "lease.json",
            }
            assert set(os.listdir(root / "source")) == {
                "descriptor.json",
                "commit.json",
                "rows.json",
                "report.json",
                "bundle.json",
            }
        result = real_authorization_write_new(
            path, value, *args, **kwargs
        )
        if is_marker:
            authorization_marker_links.append(True)
        return result

    monkeypatch.setattr(core, "_derive_source", replay_authorization_source)
    monkeypatch.setattr(
        core, "_write_new", observe_authorization_marker_last
    )
    publication_order = (
        "campaign_contract.json",
        "source/descriptor.json",
        "source/commit.json",
        "source/rows.json",
        "source/report.json",
        "source/bundle.json",
        "lease.json",
    )

    def install_partial_authorization(leaf_count):
        shutil.rmtree(root)
        root.mkdir(mode=0o700)
        (root / "source").mkdir(mode=0o700)
        for relative in publication_order[:leaf_count]:
            destination = root / relative
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copy2(authorized_backup / relative, destination)

    authorization_prefix_samples = (
        0,
        len(publication_order) // 2,
        len(publication_order),
    )
    for leaf_count in authorization_prefix_samples:
        install_partial_authorization(leaf_count)
        recovered = core.authorize_recursive_campaign_task(
            source,
            CAMPAIGN_CONTRACT,
            root,
            **authorize_kwargs,
            confirm_explicit_local_task_authorization=True,
        )
        assert recovered == authorized
        assert prepare_calls == 1
        assert _tree_observation(root) == _tree_observation(authorized_backup)

    # At the marker boundary, an exact-byte replacement source directory with
    # a different identity must be rejected before campaign.json is visible.
    install_partial_authorization(len(publication_order))
    authorization_partial_before = _tree_observation(root)
    displaced_authorization_source = (
        case["state_home"] / "authorization-source-before-marker"
    )
    real_authorization_rename = core._rename_noreplace
    authorization_swaps = []

    def replace_authorization_source_before_marker(*args, **kwargs):
        if args[3] == "campaign.json" and not authorization_swaps:
            authorization_swaps.append(True)
            (root / "source").rename(displaced_authorization_source)
            shutil.copytree(authorized_backup / "source", root / "source")
        return real_authorization_rename(*args, **kwargs)

    monkeypatch.setattr(
        core,
        "_rename_noreplace",
        replace_authorization_source_before_marker,
    )
    with pytest.raises(core.RecursiveCampaignError):
        core.authorize_recursive_campaign_task(
            source,
            CAMPAIGN_CONTRACT,
            root,
            **authorize_kwargs,
            confirm_explicit_local_task_authorization=True,
        )
    assert authorization_swaps == [True]
    assert not (root / "campaign.json").exists()
    assert _tree_observation(root) == authorization_partial_before
    shutil.rmtree(root / "source")
    displaced_authorization_source.rename(root / "source")
    monkeypatch.setattr(
        core, "_rename_noreplace", real_authorization_rename
    )
    assert _tree_observation(root) == authorization_partial_before

    # A marker-first crash image is not a recoverable prefix: accepting it
    # would allow a committed authorization to hide missing immutable leaves.
    shutil.rmtree(root)
    root.mkdir(mode=0o700)
    shutil.copy2(
        authorized_backup / "campaign.json", root / "campaign.json"
    )
    with pytest.raises(core.RecursiveCampaignError):
        core.authorize_recursive_campaign_task(
            source,
            CAMPAIGN_CONTRACT,
            root,
            **authorize_kwargs,
            confirm_explicit_local_task_authorization=True,
        )
    assert prepare_calls == 1
    shutil.rmtree(root)
    shutil.copytree(authorized_backup, root)
    monkeypatch.setattr(core, "_write_new", real_authorization_write_new)
    assert len(authorization_marker_links) == len(
        authorization_prefix_samples
    )

    verified_authorized = core.verify_recursive_campaign(
        source,
        CAMPAIGN_CONTRACT,
        root,
        expected_campaign_digest=authorized["campaign_digest"],
        expected_lease_digest=authorized["lease_digest"],
    )
    assert verified_authorized["status"] == "VERIFIED_" + AUTHORIZED
    assert verified_authorized["phase"] == "AUTHORIZED"
    assert verified_authorized["callback_start_claim_digest"] is None
    assert verified_authorized["advance_digest"] is None
    monkeypatch.setattr(core, "_derive_source", real_derive_source)

    execute_kwargs = {
        "expected_campaign_digest": authorized["campaign_digest"],
        "expected_lease_digest": authorized["lease_digest"],
        "expected_provenance_binding_digest": authorized[
            "provenance_binding_digest"
        ],
        "expected_authorization_digest": authorized[
            "authorization_digest"
        ],
        "expected_attempt_digest": authorized["attempt_digest"],
    }

    # A generation swap after the first runtime verification but before the
    # durable callback claim must consume neither the campaign lease nor the
    # callback.  The second full base capture is the admission boundary.
    real_campaign_runtime_verify = core._R_VERIFY
    real_campaign_runtime_execute = core._R_EXECUTE
    swapped_once = []
    forbidden_preclaim_callbacks = []

    def verify_then_swap(*args, **kwargs):
        verified = real_campaign_runtime_verify(*args, **kwargs)
        if not swapped_once and verified.get("status") == (
            "VERIFIED_AUTHORIZED_NOT_EXECUTED"
        ):
            swapped_once.append(True)
            bundle_path = root / "source/bundle.json"
            bundle_path.write_bytes(bundle_path.read_bytes() + b"\n")
        return verified

    def forbidden_preclaim_execute(*args, **kwargs):
        forbidden_preclaim_callbacks.append(True)
        raise AssertionError("delegate crossed after generation swap")

    monkeypatch.setattr(core, "_R_VERIFY", verify_then_swap)
    monkeypatch.setattr(core, "_R_EXECUTE", forbidden_preclaim_execute)
    with pytest.raises(core.RecursiveCampaignError):
        core.execute_recursive_campaign_task(
            source,
            CAMPAIGN_CONTRACT,
            root,
            RUNTIME_CONTRACT,
            **execute_kwargs,
            confirm_real_local_execution=True,
        )
    assert swapped_once == [True]
    assert forbidden_preclaim_callbacks == []
    assert not (root / "callback_start_claim.json").exists()
    shutil.rmtree(root)
    shutil.copytree(authorized_backup, root)
    monkeypatch.setattr(core, "_R_VERIFY", real_campaign_runtime_verify)
    monkeypatch.setattr(core, "_R_EXECUTE", real_campaign_runtime_execute)

    with pytest.raises(core.RecursiveCampaignError):
        core.execute_recursive_campaign_task(
            source,
            CAMPAIGN_CONTRACT,
            root,
            RUNTIME_CONTRACT,
            **execute_kwargs,
            confirm_real_local_execution=False,
        )
    assert not (root / "callback_start_claim.json").exists()

    wrong_execute = dict(execute_kwargs)
    wrong_execute["expected_lease_digest"] = _digest("wrong-lease")
    monkeypatch.setattr(core, "_derive_source", replay_authorization_source)
    with pytest.raises(core.RecursiveCampaignError):
        core.execute_recursive_campaign_task(
            source,
            CAMPAIGN_CONTRACT,
            root,
            RUNTIME_CONTRACT,
            **wrong_execute,
            confirm_real_local_execution=True,
        )
    assert not (root / "callback_start_claim.json").exists()
    monkeypatch.setattr(core, "_derive_source", real_derive_source)

    runtime_contract = json.loads(
        RUNTIME_CONTRACT.read_text(encoding="utf-8")
    )
    callback_starts = []

    def fake_executor(task):
        claim_path = root / "callback_start_claim.json"
        assert claim_path.is_file()
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        assert claim["status"] == (
            "CALLBACK_START_CLAIMED_HARD_STOP_NO_REENTRY"
        )
        callback_starts.append(copy.deepcopy(task))
        return advance_test._fake_native_result(task)

    def fake_preflight(*args, **kwargs):
        task = kwargs["task"]
        prepared = {
            "attempt_digest": kwargs["attempt_digest"],
            "authorization_digest": kwargs["authorization_digest"],
        }
        return advance_test._valid_fake_preflight(
            runtime_core, runtime_contract, task, prepared
        )

    monkeypatch.setattr(runtime_core, "_load_real_executor", lambda _c: fake_executor)
    monkeypatch.setattr(runtime_core, "_run_preflight", fake_preflight)
    execution_source_captures = []

    def capture_execution_source(source_value, **kwargs):
        value = real_derive_source(source_value, **kwargs)
        if source_value == source:
            execution_source_captures.append(copy.deepcopy(value))
        return value

    monkeypatch.setattr(core, "_derive_source", capture_execution_source)
    executed = core.execute_recursive_campaign_task(
        source,
        CAMPAIGN_CONTRACT,
        root,
        RUNTIME_CONTRACT,
        **execute_kwargs,
        confirm_real_local_execution=True,
    )
    assert executed["status"] == EXECUTED
    assert executed["execution_status"] == (
        "COMPLETED_SUCCESS_AWAITING_ADVANCE"
    )
    assert len(callback_starts) == 1
    assert (root / "callback_start_claim.json").is_file()
    assert execution_source_captures
    completed_source = execution_source_captures[-1]

    def replay_completed_source(source_value, **kwargs):
        if source_value == source:
            return copy.deepcopy(completed_source)
        return real_derive_source(source_value, **kwargs)

    monkeypatch.setattr(core, "_derive_source", replay_completed_source)

    with pytest.raises(core.RecursiveCampaignError):
        core.execute_recursive_campaign_task(
            source,
            CAMPAIGN_CONTRACT,
            root,
            RUNTIME_CONTRACT,
            **execute_kwargs,
            confirm_real_local_execution=True,
        )
    assert len(callback_starts) == 1

    expected_advance = _expected_nonterminal_advance(case)
    next_attempt_root = expected_advance["next_attempt_root"]
    preview_before = _tree_observation(root)
    real_attempt_snapshots = core._P_ATTEMPT_SNAPSHOTS
    generation_captures = []

    def mismatched_receipt_generation(attempt):
        objects, raws = real_attempt_snapshots(attempt)
        generation_captures.append(True)
        if len(generation_captures) == 2:
            raws = dict(raws)
            raws["execution_receipt"] = (
                raws["execution_receipt"] + b"\n"
            )
        return objects, raws

    monkeypatch.setattr(
        core, "_P_ATTEMPT_SNAPSHOTS", mismatched_receipt_generation
    )
    with pytest.raises(core.RecursiveCampaignError):
        core.verify_recursive_campaign(
            source,
            CAMPAIGN_CONTRACT,
            root,
            next_attempt_root=next_attempt_root,
            expected_campaign_digest=authorized["campaign_digest"],
            expected_lease_digest=authorized["lease_digest"],
            expected_callback_start_claim_digest=executed[
                "callback_start_claim_digest"
            ],
            expected_receipt_digest=executed["receipt_digest"],
            expected_journal_head_digest=executed["journal_head_digest"],
        )
    assert generation_captures == [True, True]
    assert _tree_observation(root) == preview_before
    monkeypatch.setattr(
        core, "_P_ATTEMPT_SNAPSHOTS", real_attempt_snapshots
    )

    with pytest.raises(core.RecursiveCampaignError):
        core.verify_recursive_campaign(
            source,
            CAMPAIGN_CONTRACT,
            root,
            next_attempt_root=next_attempt_root,
            expected_campaign_digest=authorized["campaign_digest"],
            expected_lease_digest=authorized["lease_digest"],
            expected_callback_start_claim_digest=executed[
                "callback_start_claim_digest"
            ],
            expected_receipt_digest=_digest("wrong-completed-receipt"),
            expected_journal_head_digest=executed["journal_head_digest"],
        )
    assert _tree_observation(root) == preview_before

    preview_source_captures = []

    def capture_preview_source(source_value, **kwargs):
        value = real_derive_source(source_value, **kwargs)
        if source_value == source:
            preview_source_captures.append(copy.deepcopy(value))
        return value

    monkeypatch.setattr(core, "_derive_source", capture_preview_source)
    verified_completed = core.verify_recursive_campaign(
        source,
        CAMPAIGN_CONTRACT,
        root,
        next_attempt_root=next_attempt_root,
        expected_campaign_digest=authorized["campaign_digest"],
        expected_lease_digest=authorized["lease_digest"],
        expected_callback_start_claim_digest=executed[
            "callback_start_claim_digest"
        ],
        expected_receipt_digest=executed["receipt_digest"],
        expected_journal_head_digest=executed["journal_head_digest"],
    )
    assert verified_completed["status"] == "VERIFIED_" + EXECUTED
    assert verified_completed["phase"] == "CALLBACK_COMPLETED"
    assert verified_completed["execution_status"] == (
        "COMPLETED_SUCCESS_PREVIEW_INDEPENDENTLY_ANCHORED_HARD_STOP"
    )
    assert verified_completed["next_attempt_root"] == str(next_attempt_root)
    assert verified_completed["remaining_task_count"] == 27
    assert verified_completed["advance_digest"] is None
    for expected_name in (
        *runner._ADVANCE_DIGEST_NAMES[7:],
        *runner._ADVANCE_NEXT_DIGEST_NAMES,
    ):
        assert verified_completed[
            expected_name.removeprefix("expected_")
        ] == expected_advance[expected_name]
    assert _tree_observation(root) == preview_before
    assert not next_attempt_root.exists()
    assert preview_source_captures
    preview_source = preview_source_captures[-1]

    def replay_preview_source(source_value, **kwargs):
        if source_value == source:
            return copy.deepcopy(preview_source)
        return real_derive_source(source_value, **kwargs)

    monkeypatch.setattr(core, "_derive_source", replay_preview_source)

    recovered_preview = core.verify_recursive_campaign(
        source,
        CAMPAIGN_CONTRACT,
        root,
        next_attempt_root=next_attempt_root,
        expected_campaign_digest=authorized["campaign_digest"],
        expected_lease_digest=authorized["lease_digest"],
        expected_callback_start_claim_digest=executed[
            "callback_start_claim_digest"
        ],
    )
    assert recovered_preview["execution_status"] == (
        "COMPLETED_SUCCESS_PREVIEW_RECOVERED_LOCAL_ANCHORS_"
        "NOT_INDEPENDENT_HARD_STOP"
    )
    assert {
        key: recovered_preview[key]
        for key in runner._ADVANCE_RESULT_KEYS
        if key not in {"execution_status"}
    } == {
        key: verified_completed[key]
        for key in runner._ADVANCE_RESULT_KEYS
        if key not in {"execution_status"}
    }
    assert _tree_observation(root) == preview_before

    advance_kwargs = {
        **execute_kwargs,
        "expected_receipt_digest": executed["receipt_digest"],
        "expected_journal_head_digest": executed["journal_head_digest"],
        **{
            name: verified_completed[name.removeprefix("expected_")]
            for name in (
                *runner._ADVANCE_DIGEST_NAMES[7:],
                *runner._ADVANCE_NEXT_DIGEST_NAMES,
            )
        },
    }
    # The slice above starts at output evidence; campaign/lease/provenance,
    # authorization/attempt and receipt/journal are already explicit.
    assert set(advance_kwargs) == {
        *runner._ADVANCE_DIGEST_NAMES,
        *runner._ADVANCE_NEXT_DIGEST_NAMES,
    }
    with pytest.raises(core.RecursiveCampaignError):
        core.advance_recursive_campaign(
            source,
            CAMPAIGN_CONTRACT,
            root,
            next_attempt_root,
            **advance_kwargs,
            confirm_immutable_one_step_advance=False,
        )
    assert not (root / "advance").exists()

    wrong_advance = dict(advance_kwargs)
    wrong_advance["expected_next_plan_digest"] = _digest("wrong-next-plan")
    with pytest.raises(core.RecursiveCampaignError):
        core.advance_recursive_campaign(
            source,
            CAMPAIGN_CONTRACT,
            root,
            next_attempt_root,
            **wrong_advance,
            confirm_immutable_one_step_advance=True,
        )
    assert not (root / "advance").exists()

    real_completed_preview = core._preview_completed_advance
    captured_nonterminal_preview = []
    advance_source_captures = []

    def capture_nonterminal_preview(*args, **kwargs):
        value = real_completed_preview(*args, **kwargs)
        captured_nonterminal_preview.append(copy.deepcopy(value))
        return value

    def capture_advance_source(source_value, **kwargs):
        value = real_derive_source(source_value, **kwargs)
        if source_value == source:
            advance_source_captures.append(copy.deepcopy(value))
        return value

    monkeypatch.setattr(
        core, "_preview_completed_advance", capture_nonterminal_preview
    )
    monkeypatch.setattr(core, "_derive_source", capture_advance_source)
    advanced = core.advance_recursive_campaign(
        source,
        CAMPAIGN_CONTRACT,
        root,
        next_attempt_root,
        **advance_kwargs,
        confirm_immutable_one_step_advance=True,
    )
    monkeypatch.setattr(
        core, "_preview_completed_advance", real_completed_preview
    )
    assert len(captured_nonterminal_preview) == 1
    assert advance_source_captures
    nonterminal_preview = captured_nonterminal_preview[0]
    advance_recovery_source = advance_source_captures[-1]
    assert advanced["status"] == ADVANCED
    assert advanced["phase"] == "ADVANCED_NONTERMINAL"
    assert advanced["remaining_task_count"] == 27
    assert advanced["terminal_status"] == "NONTERMINAL"
    assert advanced["next_attempt_root"] == str(next_attempt_root)
    assert advanced["advance_digest"].startswith("sha256:")
    assert not next_attempt_root.exists()
    combined = json.loads(
        (root / "advance/combined_rows.json").read_text(encoding="utf-8")
    )
    output_report = json.loads(
        (root / "advance/output_report.json").read_text(encoding="utf-8")
    )
    next_bundle = json.loads(
        (root / "advance/next_bundle.json").read_text(encoding="utf-8")
    )
    assert len(combined) == 1353
    assert len(output_report["pending_evidence"]) == 27
    assert len(next_bundle["plan"]["tasks"]) == 27
    assert next_bundle["plan"]["tasks"][0]["cell"]["seed"] == 3
    assert {path.name for path in (root / "advance").iterdir()} == {
        "execution_receipt.json",
        "combined_rows.json",
        "output_report.json",
        "reingestion_receipt.json",
        "next_bundle.json",
        "advance.json",
    }

    # Recover representative empty/middle/full noncommit prefixes without
    # rerunning scientific reingestion/materialization.  The direct atomic
    # writer KAT covers each individual no-replace publication boundary.
    nonterminal_advance_backup = (
        case["state_home"] / "nonterminal-advance-complete-backup"
    )
    shutil.copytree(root / "advance", nonterminal_advance_backup)
    nonterminal_leaf_order = (
        "execution_receipt.json",
        "combined_rows.json",
        "output_report.json",
        "reingestion_receipt.json",
        "next_bundle.json",
    )

    def install_nonterminal_advance_prefix(leaf_count):
        shutil.rmtree(root / "advance")
        (root / "advance").mkdir(mode=0o700)
        for name in nonterminal_leaf_order[:leaf_count]:
            shutil.copy2(
                nonterminal_advance_backup / name, root / "advance" / name
            )

    preview_replays = []

    def replay_nonterminal_preview(*args, **kwargs):
        preview_replays.append(True)
        return copy.deepcopy(nonterminal_preview)

    prepare_before_advance_recovery = prepare_calls
    callback_before_advance_recovery = len(callback_starts)
    real_advance_recovery_derive = real_derive_source

    def replay_advance_recovery_source(source_value, **kwargs):
        if source_value == source:
            return copy.deepcopy(advance_recovery_source)
        return real_advance_recovery_derive(source_value, **kwargs)

    real_campaign_write_new = core._write_new
    nonterminal_marker_links = []

    def observe_nonterminal_marker_last(path, value, *args, **kwargs):
        path = Path(path)
        is_marker = path == root / "advance/advance.json"
        if is_marker:
            assert set(os.listdir(path.parent)) == set(
                nonterminal_leaf_order
            )
        result = real_campaign_write_new(path, value, *args, **kwargs)
        if is_marker:
            nonterminal_marker_links.append(True)
        return result

    monkeypatch.setattr(
        core, "_preview_completed_advance", replay_nonterminal_preview
    )
    monkeypatch.setattr(
        core, "_derive_source", replay_advance_recovery_source
    )
    monkeypatch.setattr(core, "_write_new", observe_nonterminal_marker_last)
    nonterminal_prefix_samples = (
        0,
        len(nonterminal_leaf_order) // 2,
        len(nonterminal_leaf_order),
    )
    for leaf_count in nonterminal_prefix_samples:
        install_nonterminal_advance_prefix(leaf_count)
        recovered_advance = core.advance_recursive_campaign(
            source,
            CAMPAIGN_CONTRACT,
            root,
            next_attempt_root,
            **advance_kwargs,
            confirm_immutable_one_step_advance=True,
        )
        assert recovered_advance == advanced
        assert _tree_observation(root / "advance") == _tree_observation(
            nonterminal_advance_backup
        )
        assert not next_attempt_root.exists()

    # Replacing the immutable source directory with exact bytes but a new
    # identity at the advance marker boundary cannot expose advance.json.
    install_nonterminal_advance_prefix(len(nonterminal_leaf_order))
    advance_partial_before = _tree_observation(root)
    displaced_advance_source = (
        case["state_home"] / "advance-source-before-marker"
    )
    real_advance_rename = core._rename_noreplace
    advance_source_swaps = []

    def replace_advance_source_before_marker(*args, **kwargs):
        if args[3] == "advance.json" and not advance_source_swaps:
            advance_source_swaps.append(True)
            (root / "source").rename(displaced_advance_source)
            shutil.copytree(authorized_backup / "source", root / "source")
        return real_advance_rename(*args, **kwargs)

    monkeypatch.setattr(
        core, "_rename_noreplace", replace_advance_source_before_marker
    )
    with pytest.raises(core.RecursiveCampaignError):
        core.advance_recursive_campaign(
            source,
            CAMPAIGN_CONTRACT,
            root,
            next_attempt_root,
            **advance_kwargs,
            confirm_immutable_one_step_advance=True,
        )
    assert advance_source_swaps == [True]
    assert not (root / "advance/advance.json").exists()
    assert _tree_observation(root) == advance_partial_before
    shutil.rmtree(root / "source")
    displaced_advance_source.rename(root / "source")
    monkeypatch.setattr(core, "_rename_noreplace", real_advance_rename)
    assert _tree_observation(root) == advance_partial_before

    # Marker-first and a corrupt later leaf fail before filling any missing
    # file.  An occupied future attempt may leave exact staged leaves, but can
    # never acquire the advance commit marker.
    shutil.rmtree(root / "advance")
    (root / "advance").mkdir(mode=0o700)
    shutil.copy2(
        nonterminal_advance_backup / "advance.json",
        root / "advance/advance.json",
    )
    marker_first_before = _tree_observation(root / "advance")
    with pytest.raises(core.RecursiveCampaignError):
        core.advance_recursive_campaign(
            source,
            CAMPAIGN_CONTRACT,
            root,
            next_attempt_root,
            **advance_kwargs,
            confirm_immutable_one_step_advance=True,
        )
    assert _tree_observation(root / "advance") == marker_first_before

    shutil.rmtree(root / "advance")
    (root / "advance").mkdir(mode=0o700)
    corrupt_late = root / "advance/output_report.json"
    corrupt_late.write_bytes(
        (nonterminal_advance_backup / "output_report.json").read_bytes()
        + b"\n"
    )
    corrupt_late.chmod(0o600)
    corrupt_subset_before = _tree_observation(root / "advance")
    with pytest.raises(core.RecursiveCampaignError):
        core.advance_recursive_campaign(
            source,
            CAMPAIGN_CONTRACT,
            root,
            next_attempt_root,
            **advance_kwargs,
            confirm_immutable_one_step_advance=True,
        )
    assert _tree_observation(root / "advance") == corrupt_subset_before

    install_nonterminal_advance_prefix(2)
    next_attempt_root.mkdir(mode=0o700)
    with pytest.raises(core.RecursiveCampaignError):
        core.advance_recursive_campaign(
            source,
            CAMPAIGN_CONTRACT,
            root,
            next_attempt_root,
            **advance_kwargs,
            confirm_immutable_one_step_advance=True,
        )
    assert not (root / "advance/advance.json").exists()
    shutil.rmtree(next_attempt_root)
    shutil.rmtree(root / "advance")
    shutil.copytree(nonterminal_advance_backup, root / "advance")
    monkeypatch.setattr(
        core, "_preview_completed_advance", real_completed_preview
    )
    monkeypatch.setattr(core, "_derive_source", real_advance_recovery_derive)
    monkeypatch.setattr(core, "_write_new", real_campaign_write_new)
    assert len(preview_replays) >= len(nonterminal_prefix_samples)
    assert len(nonterminal_marker_links) == len(
        nonterminal_prefix_samples
    )
    assert prepare_calls == prepare_before_advance_recovery
    assert len(callback_starts) == callback_before_advance_recovery

    verify_advanced_kwargs = {
        "expected_campaign_digest": advanced["campaign_digest"],
        "expected_lease_digest": advanced["lease_digest"],
        "expected_callback_start_claim_digest": advanced[
            "callback_start_claim_digest"
        ],
        "expected_advance_digest": advanced["advance_digest"],
        "expected_output_evidence_digest": advanced[
            "output_evidence_digest"
        ],
        "expected_output_report_body_digest": advanced[
            "output_report_body_digest"
        ],
        "expected_output_audit_head": advanced["output_audit_head"],
        "expected_reingestion_digest": advanced["reingestion_digest"],
        "expected_next_bundle_digest": advanced["next_bundle_digest"],
        "expected_next_plan_digest": advanced["next_plan_digest"],
    }
    advanced_before = _tree_observation(root)
    verified_advanced = core.verify_recursive_campaign(
        source,
        CAMPAIGN_CONTRACT,
        root,
        **verify_advanced_kwargs,
    )
    assert verified_advanced["status"] == "VERIFIED_" + ADVANCED
    assert verified_advanced["phase"] == "ADVANCED_NONTERMINAL"
    assert _tree_observation(root) == advanced_before

    previous_source = {
        "schema_version": core.SOURCE_SCHEMA_VERSION,
        "source_kind": "recursive_campaign_v1",
        "dependencies": copy.deepcopy(source["dependencies"]),
        "campaign_contract_path": str(CAMPAIGN_CONTRACT),
        "campaign_root": str(root),
        **verify_advanced_kwargs,
    }
    next_campaign_id = "fake-recursive-campaign-seed3"
    next_campaign_root = campaign_prefix / next_campaign_id
    before_recursive_inspect = _tree_observation(case["state_home"])
    previous_source_captures = []

    def capture_previous_campaign_source(source_value, **kwargs):
        value = real_derive_source(source_value, **kwargs)
        if source_value == previous_source:
            previous_source_captures.append(copy.deepcopy(value))
        return value

    monkeypatch.setattr(
        core, "_derive_source", capture_previous_campaign_source
    )
    next_inspected = core.inspect_recursive_campaign(
        previous_source,
        CAMPAIGN_CONTRACT,
        next_campaign_root,
        campaign_id=next_campaign_id,
        next_attempt_root=next_attempt_root,
    )
    assert next_inspected["status"] == INSPECTED
    assert next_inspected["source_kind"] == "recursive_campaign_v1"
    assert next_inspected["source_state_digest"] == advanced[
        "advance_digest"
    ]
    assert next_inspected["task_count"] == 27
    assert next_inspected["task_id"] == next_bundle["plan"]["tasks"][0][
        "task_id"
    ]
    assert _tree_observation(case["state_home"]) == before_recursive_inspect
    assert not next_campaign_root.exists()
    assert not next_attempt_root.exists()
    assert previous_source_captures
    previous_campaign_derived = previous_source_captures[-1]
    monkeypatch.setattr(core, "_derive_source", real_derive_source)

    # Byte tamper, hardlink alias and full-root reuse are all fail-closed and
    # never overwrite a committed capsule.
    next_bundle_path = root / "advance/next_bundle.json"
    next_bundle_raw = next_bundle_path.read_bytes()
    next_bundle_path.write_bytes(next_bundle_raw + b"\n")
    next_bundle_path.chmod(0o600)
    with pytest.raises(core.RecursiveCampaignError):
        core.verify_recursive_campaign(
            source,
            CAMPAIGN_CONTRACT,
            root,
            **verify_advanced_kwargs,
        )
    next_bundle_path.write_bytes(next_bundle_raw)
    next_bundle_path.chmod(0o600)

    alias = case["state_home"] / "next-bundle-hardlink"
    os.link(next_bundle_path, alias)
    try:
        with pytest.raises(core.RecursiveCampaignError):
            core.verify_recursive_campaign(
                source,
                CAMPAIGN_CONTRACT,
                root,
                **verify_advanced_kwargs,
            )
    finally:
        alias.unlink()
    assert _tree_observation(root) == advanced_before

    # Authorize the next source once, then fork its temporary state to test
    # both an uncertain delegate crash and a completed failed receipt.  Both
    # paths consume the one lease and forbid retry/advance.
    next_authorize_kwargs = {
        "campaign_id": next_campaign_id,
        "expected_source_state_digest": next_inspected[
            "source_state_digest"
        ],
        "expected_bundle_digest": next_inspected["bundle_digest"],
        "expected_plan_digest": next_inspected["plan_digest"],
        "task_id": next_inspected["task_id"],
        "expected_task_digest": next_inspected["task_digest"],
        "expected_provenance_binding_digest": next_inspected[
            "provenance_binding_digest"
        ],
        "authorization_id": next_inspected["required_authorization_id"],
    }

    def replay_previous_campaign_source(source_value, **kwargs):
        if source_value == previous_source:
            return copy.deepcopy(previous_campaign_derived)
        return real_derive_source(source_value, **kwargs)

    monkeypatch.setattr(
        core, "_derive_source", replay_previous_campaign_source
    )
    next_authorized = core.authorize_recursive_campaign_task(
        previous_source,
        CAMPAIGN_CONTRACT,
        next_campaign_root,
        **next_authorize_kwargs,
        confirm_explicit_local_task_authorization=True,
    )
    backup_root = case["state_home"] / "campaign-authorized-backup"
    backup_attempt = case["state_home"] / "attempt-authorized-backup"
    shutil.copytree(next_campaign_root, backup_root)
    shutil.copytree(next_attempt_root, backup_attempt)

    real_execute_delegate = core._R_EXECUTE
    next_execute_kwargs = {
        "expected_campaign_digest": next_authorized["campaign_digest"],
        "expected_lease_digest": next_authorized["lease_digest"],
        "expected_provenance_binding_digest": next_authorized[
            "provenance_binding_digest"
        ],
        "expected_authorization_digest": next_authorized[
            "authorization_digest"
        ],
        "expected_attempt_digest": next_authorized["attempt_digest"],
    }

    def restore_next_authorized():
        if next_campaign_root.exists():
            shutil.rmtree(next_campaign_root)
        if next_attempt_root.exists():
            shutil.rmtree(next_attempt_root)
        shutil.copytree(backup_root, next_campaign_root)
        shutil.copytree(backup_attempt, next_attempt_root)

    # A delegate cannot earn a completed campaign return by merely claiming
    # receipt digests.  The core must publicly verify the actual runtime tree
    # after every delegate return; this lying producer writes nothing.
    lying_delegate_calls = []

    def lying_delegate(*args, **kwargs):
        assert (next_campaign_root / "callback_start_claim.json").is_file()
        lying_delegate_calls.append(True)
        return {
            "status": "EXECUTED_RECEIPT_WRITTEN",
            "authorization_digest": next_authorized[
                "authorization_digest"
            ],
            "attempt_digest": next_authorized["attempt_digest"],
            "receipt_digest": _digest("lying-producer-receipt"),
            "journal_head_digest": _digest("lying-producer-journal"),
        }

    monkeypatch.setattr(core, "_R_EXECUTE", lying_delegate)
    with pytest.raises(core.RecursiveCampaignError):
        core.execute_recursive_campaign_task(
            previous_source,
            CAMPAIGN_CONTRACT,
            next_campaign_root,
            RUNTIME_CONTRACT,
            **next_execute_kwargs,
            confirm_real_local_execution=True,
        )
    assert lying_delegate_calls == [True]
    lying_claim = json.loads(
        (next_campaign_root / "callback_start_claim.json").read_text(
            encoding="utf-8"
        )
    )
    lying_incomplete = core.verify_recursive_campaign(
        previous_source,
        CAMPAIGN_CONTRACT,
        next_campaign_root,
        expected_campaign_digest=next_authorized["campaign_digest"],
        expected_lease_digest=next_authorized["lease_digest"],
        expected_callback_start_claim_digest=lying_claim["integrity"][
            "callback_start_claim_digest"
        ],
    )
    assert lying_incomplete["phase"] == "CALLBACK_INCOMPLETE"
    with pytest.raises(core.RecursiveCampaignError):
        core.execute_recursive_campaign_task(
            previous_source,
            CAMPAIGN_CONTRACT,
            next_campaign_root,
            RUNTIME_CONTRACT,
            **next_execute_kwargs,
            confirm_real_local_execution=True,
        )
    assert lying_delegate_calls == [True]
    restore_next_authorized()

    delegate_calls = []

    def uncertain_delegate(*args, **kwargs):
        assert (next_campaign_root / "callback_start_claim.json").is_file()
        delegate_calls.append(True)
        raise RuntimeError("fake uncertain delegate crash")

    monkeypatch.setattr(core, "_R_EXECUTE", uncertain_delegate)
    with pytest.raises(core.RecursiveCampaignError):
        core.execute_recursive_campaign_task(
            previous_source,
            CAMPAIGN_CONTRACT,
            next_campaign_root,
            RUNTIME_CONTRACT,
            **next_execute_kwargs,
            confirm_real_local_execution=True,
        )
    assert delegate_calls == [True]
    claim = json.loads(
        (next_campaign_root / "callback_start_claim.json").read_text(
            encoding="utf-8"
        )
    )
    claim_digest = claim["integrity"]["callback_start_claim_digest"]
    uncertain = core.verify_recursive_campaign(
        previous_source,
        CAMPAIGN_CONTRACT,
        next_campaign_root,
        expected_campaign_digest=next_authorized["campaign_digest"],
        expected_lease_digest=next_authorized["lease_digest"],
        expected_callback_start_claim_digest=claim_digest,
    )
    assert uncertain["phase"] == "CALLBACK_INCOMPLETE"
    assert uncertain["status"].endswith(
        "RUNTIME_INCOMPLETE_HARD_STOP"
    )
    assert uncertain["execution_status"] == (
        "CALLBACK_CLAIMED_INCOMPLETE_NO_REENTRY"
    )
    assert uncertain["remaining_task_count"] == 27
    assert uncertain["terminal_status"] == "NONTERMINAL"
    assert uncertain["next_attempt_root"] is None
    authorized_after_claim = runtime_core.verify_single_task_attempt(
        next_attempt_root,
        runtime_contract,
        advance_test.BASE_MANIFEST,
        advance_test.ASSET_ROOT,
        expected_authorization_digest=next_authorized[
            "authorization_digest"
        ],
    )
    assert authorized_after_claim["status"] == (
        "VERIFIED_AUTHORIZED_NOT_EXECUTED"
    )
    with pytest.raises(core.RecursiveCampaignError):
        core.execute_recursive_campaign_task(
            previous_source,
            CAMPAIGN_CONTRACT,
            next_campaign_root,
            RUNTIME_CONTRACT,
            **next_execute_kwargs,
            confirm_real_local_execution=True,
        )
    assert delegate_calls == [True]

    # Restore the same authorization and crash at the next two durable runtime
    # boundaries.  PREFLIGHT and RUNNING are distinct runtime generations but
    # both are one-lease, no-reentry campaign states.
    restore_next_authorized()
    monkeypatch.setattr(core, "_R_EXECUTE", real_execute_delegate)
    monkeypatch.setattr(
        runtime_core,
        "_load_real_executor",
        lambda _contract: (lambda _task: None),
    )
    real_runtime_write = runtime_core._write_new_json

    def crash_before_running(path, payload):
        if Path(path) == next_attempt_root / "journal/0001_RUNNING.json":
            raise RuntimeError("injected crash after PREFLIGHT publication")
        return real_runtime_write(path, payload)

    monkeypatch.setattr(runtime_core, "_write_new_json", crash_before_running)
    with pytest.raises(core.RecursiveCampaignError):
        core.execute_recursive_campaign_task(
            previous_source,
            CAMPAIGN_CONTRACT,
            next_campaign_root,
            RUNTIME_CONTRACT,
            **next_execute_kwargs,
            confirm_real_local_execution=True,
        )
    assert (next_attempt_root / "preflight.json").is_file()
    assert not (next_attempt_root / "journal/0001_RUNNING.json").exists()
    preflight_runtime = runtime_core.verify_single_task_attempt(
        next_attempt_root,
        runtime_contract,
        advance_test.BASE_MANIFEST,
        advance_test.ASSET_ROOT,
        expected_authorization_digest=next_authorized[
            "authorization_digest"
        ],
    )
    assert preflight_runtime["status"] == (
        "VERIFIED_PREFLIGHT_PASSED_NO_CALLBACK"
    )
    preflight_claim = json.loads(
        (next_campaign_root / "callback_start_claim.json").read_text(
            encoding="utf-8"
        )
    )
    preflight_before_verify = (
        _tree_observation(next_campaign_root),
        _tree_observation(next_attempt_root),
    )
    preflight_campaign = core.verify_recursive_campaign(
        previous_source,
        CAMPAIGN_CONTRACT,
        next_campaign_root,
        expected_campaign_digest=next_authorized["campaign_digest"],
        expected_lease_digest=next_authorized["lease_digest"],
        expected_callback_start_claim_digest=preflight_claim["integrity"][
            "callback_start_claim_digest"
        ],
    )
    assert preflight_campaign["phase"] == "CALLBACK_INCOMPLETE"
    assert preflight_campaign["receipt_digest"] is None
    assert (
        _tree_observation(next_campaign_root),
        _tree_observation(next_attempt_root),
    ) == preflight_before_verify
    with pytest.raises(core.RecursiveCampaignError):
        core.execute_recursive_campaign_task(
            previous_source,
            CAMPAIGN_CONTRACT,
            next_campaign_root,
            RUNTIME_CONTRACT,
            **next_execute_kwargs,
            confirm_real_local_execution=True,
        )

    restore_next_authorized()
    monkeypatch.setattr(runtime_core, "_write_new_json", real_runtime_write)
    real_captured_execute = runtime_core._CAPTURED_EXECUTE_AUTHORIZED_PLAN
    running_boundary_calls = []

    def crash_after_running(*args, **kwargs):
        running_boundary_calls.append(True)
        raise RuntimeError("injected crash after RUNNING publication")

    monkeypatch.setattr(
        runtime_core,
        "_CAPTURED_EXECUTE_AUTHORIZED_PLAN",
        crash_after_running,
    )
    with pytest.raises(core.RecursiveCampaignError):
        core.execute_recursive_campaign_task(
            previous_source,
            CAMPAIGN_CONTRACT,
            next_campaign_root,
            RUNTIME_CONTRACT,
            **next_execute_kwargs,
            confirm_real_local_execution=True,
        )
    assert running_boundary_calls == [True]
    assert (next_attempt_root / "preflight.json").is_file()
    assert (next_attempt_root / "journal/0001_RUNNING.json").is_file()
    running_runtime = runtime_core.verify_single_task_attempt(
        next_attempt_root,
        runtime_contract,
        advance_test.BASE_MANIFEST,
        advance_test.ASSET_ROOT,
        expected_authorization_digest=next_authorized[
            "authorization_digest"
        ],
    )
    assert running_runtime["status"] == (
        "VERIFIED_RUNNING_INCOMPLETE_NO_REENTRY"
    )
    running_claim = json.loads(
        (next_campaign_root / "callback_start_claim.json").read_text(
            encoding="utf-8"
        )
    )
    running_before_verify = (
        _tree_observation(next_campaign_root),
        _tree_observation(next_attempt_root),
    )
    running_campaign = core.verify_recursive_campaign(
        previous_source,
        CAMPAIGN_CONTRACT,
        next_campaign_root,
        expected_campaign_digest=next_authorized["campaign_digest"],
        expected_lease_digest=next_authorized["lease_digest"],
        expected_callback_start_claim_digest=running_claim["integrity"][
            "callback_start_claim_digest"
        ],
    )
    assert running_campaign["phase"] == "CALLBACK_INCOMPLETE"
    assert running_campaign["receipt_digest"] is None
    assert (
        _tree_observation(next_campaign_root),
        _tree_observation(next_attempt_root),
    ) == running_before_verify
    with pytest.raises(core.RecursiveCampaignError):
        core.execute_recursive_campaign_task(
            previous_source,
            CAMPAIGN_CONTRACT,
            next_campaign_root,
            RUNTIME_CONTRACT,
            **next_execute_kwargs,
            confirm_real_local_execution=True,
        )

    # Restore the exact AUTHORIZED snapshot at the same absolute paths to run
    # an independent failed-receipt branch in this temporary KAT only.
    restore_next_authorized()
    monkeypatch.setattr(
        runtime_core,
        "_CAPTURED_EXECUTE_AUTHORIZED_PLAN",
        real_captured_execute,
    )
    failed_starts = []

    def failing_executor(_task):
        assert (next_campaign_root / "callback_start_claim.json").is_file()
        failed_starts.append(True)
        raise RuntimeError("fake callback failure")

    monkeypatch.setattr(
        runtime_core, "_load_real_executor", lambda _c: failing_executor
    )
    failed_execution = core.execute_recursive_campaign_task(
        previous_source,
        CAMPAIGN_CONTRACT,
        next_campaign_root,
        RUNTIME_CONTRACT,
        **next_execute_kwargs,
        confirm_real_local_execution=True,
    )
    assert failed_starts == [True]
    assert failed_execution["status"] == EXECUTED
    assert failed_execution["execution_status"] == (
        "COMPLETED_FAILED_EVIDENCE_NEUTRAL_HARD_STOP"
    )
    failed_receipt = json.loads(
        (next_attempt_root / "receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        result["status"] == "FAILED"
        for result in failed_receipt["results"]
    )
    failed_before = _tree_observation(next_campaign_root)
    failed_verified = core.verify_recursive_campaign(
        previous_source,
        CAMPAIGN_CONTRACT,
        next_campaign_root,
        expected_campaign_digest=next_authorized["campaign_digest"],
        expected_lease_digest=next_authorized["lease_digest"],
        expected_callback_start_claim_digest=failed_execution[
            "callback_start_claim_digest"
        ],
        expected_receipt_digest=failed_execution["receipt_digest"],
        expected_journal_head_digest=failed_execution[
            "journal_head_digest"
        ],
    )
    assert failed_verified["status"] == "VERIFIED_" + EXECUTED
    assert failed_verified["phase"] == "CALLBACK_COMPLETED"
    assert failed_verified["execution_status"] == (
        "COMPLETED_FAILED_EVIDENCE_NEUTRAL_"
        "INDEPENDENTLY_ANCHORED_HARD_STOP"
    )
    assert failed_verified["receipt_digest"] == failed_execution[
        "receipt_digest"
    ]
    assert failed_verified["journal_head_digest"] == failed_execution[
        "journal_head_digest"
    ]
    assert failed_verified["remaining_task_count"] == 27
    assert failed_verified["terminal_status"] == "NONTERMINAL"
    for key in (
        "advance_digest",
        "reingestion_digest",
        "output_evidence_digest",
        "output_report_body_digest",
        "output_audit_head",
        "next_pending_evidence_digest",
        "next_first_pending_projection_digest",
        "next_bundle_digest",
        "next_plan_digest",
        "next_attempt_root",
    ):
        assert failed_verified[key] is None
    assert _tree_observation(next_campaign_root) == failed_before

    failed_recovered = core.verify_recursive_campaign(
        previous_source,
        CAMPAIGN_CONTRACT,
        next_campaign_root,
        expected_campaign_digest=next_authorized["campaign_digest"],
        expected_lease_digest=next_authorized["lease_digest"],
        expected_callback_start_claim_digest=failed_execution[
            "callback_start_claim_digest"
        ],
    )
    assert failed_recovered["execution_status"] == (
        "COMPLETED_FAILED_EVIDENCE_NEUTRAL_RECOVERED_LOCAL_ANCHORS_"
        "NOT_INDEPENDENT_HARD_STOP"
    )
    assert _tree_observation(next_campaign_root) == failed_before
    with pytest.raises(core.RecursiveCampaignError):
        core.advance_recursive_campaign(
            previous_source,
            CAMPAIGN_CONTRACT,
            next_campaign_root,
            None,
            expected_campaign_digest=next_authorized["campaign_digest"],
            expected_lease_digest=next_authorized["lease_digest"],
            expected_provenance_binding_digest=next_authorized[
                "provenance_binding_digest"
            ],
            expected_authorization_digest=next_authorized[
                "authorization_digest"
            ],
            expected_attempt_digest=next_authorized["attempt_digest"],
            expected_receipt_digest=failed_execution["receipt_digest"],
            expected_journal_head_digest=failed_execution[
                "journal_head_digest"
            ],
            expected_output_evidence_digest=_digest("not-evidence"),
            expected_output_report_body_digest=_digest("not-report"),
            expected_output_audit_head=_digest("not-audit"),
            expected_reingestion_digest=_digest("not-reingestion"),
            expected_next_pending_evidence_digest=None,
            expected_next_first_pending_projection_digest=None,
            expected_next_bundle_digest=None,
            expected_next_plan_digest=None,
            confirm_immutable_one_step_advance=True,
        )
    assert not (next_campaign_root / "advance").exists()
    with pytest.raises(core.RecursiveCampaignError):
        core.execute_recursive_campaign_task(
            previous_source,
            CAMPAIGN_CONTRACT,
            next_campaign_root,
            RUNTIME_CONTRACT,
            **next_execute_kwargs,
            confirm_real_local_execution=True,
        )
    assert failed_starts == [True]
    monkeypatch.setattr(core, "_derive_source", real_derive_source)

    # Build a scientifically valid one-gap report/bundle from the same frozen
    # evidence scope.  Admission is injected at the internal source boundary
    # because recursive-successor V1 is intentionally frozen to 28 tasks; the
    # authorization, runtime, preview, terminal publication and verification
    # below are the real public campaign/runtime paths.
    from performance.structural_hypothesis_loop import (
        run_structural_hypothesis_loop,
        verify_report_integrity,
    )
    from performance.structural_hypothesis_task_materializer import (
        materialize_task_bundle,
        verify_materialized_task_bundle,
    )

    initial_rows = json.loads(
        (root / "source/rows.json").read_text(encoding="utf-8")
    )
    initial_report = json.loads(
        (root / "source/report.json").read_text(encoding="utf-8")
    )
    hypothesis = json.loads(
        advance_test.HYPOTHESIS_CONTRACT.read_text(encoding="utf-8")
    )
    executor = json.loads(
        advance_test.EXECUTOR_CONTRACT.read_text(encoding="utf-8")
    )
    materializer = json.loads(
        advance_test.MATERIALIZER_CONTRACT.read_text(encoding="utf-8")
    )
    filled = [
        advance_test._synthetic_row(
            hypothesis["evidence_scope"],
            cell["profile"],
            cell["domain"],
            cell["seed"],
        )
        for cell in initial_report["pending_evidence"][:27]
    ]
    one_gap_rows = [*copy.deepcopy(initial_rows), *filled]
    one_gap_report = run_structural_hypothesis_loop(
        one_gap_rows, hypothesis
    ).to_dict()
    assert verify_report_integrity(one_gap_report)
    assert len(one_gap_rows) == 1379
    assert len(one_gap_report["pending_evidence"]) == 1

    terminal_attempt_root = (
        case["state_home"]
        / "kg-op/structural-hypothesis-execution/v1"
        / "terminal-one-gap-source"
    )
    one_gap_bundle = materialize_task_bundle(
        one_gap_report,
        hypothesis,
        executor,
        materializer,
        advance_test.BASE_MANIFEST,
        advance_test.ASSET_ROOT,
        terminal_attempt_root / "checkpoints",
    )
    assert verify_materialized_task_bundle(
        one_gap_bundle,
        one_gap_report,
        hypothesis,
        executor,
        materializer,
        advance_test.BASE_MANIFEST,
        advance_test.ASSET_ROOT,
        terminal_attempt_root / "checkpoints",
    )
    one_gap_derived = {
        "source_kind": "recursive_successor_v1",
        "source_state_digest": _digest("valid-one-gap-source"),
        "source_verification": {"status": "TEST_ONLY_VALID_SOURCE_INJECTION"},
        "rows": one_gap_rows,
        "report": one_gap_report,
        "bundle": one_gap_bundle,
        "next_attempt_root": terminal_attempt_root,
        "checkpoint_root": terminal_attempt_root / "checkpoints",
        "descriptor": copy.deepcopy(source),
        "descriptor_digest": core._digest(source),
    }
    real_derive_source = core._derive_source

    def one_gap_source(source_value, *, require_attempt_absent, **kwargs):
        if source_value == source:
            if require_attempt_absent and terminal_attempt_root.exists():
                raise core.RecursiveCampaignError(
                    "one-gap source-bound attempt already exists"
                )
            return copy.deepcopy(one_gap_derived)
        return real_derive_source(
            source_value,
            require_attempt_absent=require_attempt_absent,
            **kwargs,
        )

    monkeypatch.setattr(core, "_derive_source", one_gap_source)
    terminal_campaign_id = "terminal-one-gap-campaign"
    terminal_campaign_root = campaign_prefix / terminal_campaign_id
    terminal_inspected = core.inspect_recursive_campaign(
        source,
        CAMPAIGN_CONTRACT,
        terminal_campaign_root,
        campaign_id=terminal_campaign_id,
        next_attempt_root=terminal_attempt_root,
    )
    assert terminal_inspected["task_count"] == 1
    terminal_authorized = core.authorize_recursive_campaign_task(
        source,
        CAMPAIGN_CONTRACT,
        terminal_campaign_root,
        campaign_id=terminal_campaign_id,
        expected_source_state_digest=terminal_inspected[
            "source_state_digest"
        ],
        expected_bundle_digest=terminal_inspected["bundle_digest"],
        expected_plan_digest=terminal_inspected["plan_digest"],
        task_id=terminal_inspected["task_id"],
        expected_task_digest=terminal_inspected["task_digest"],
        expected_provenance_binding_digest=terminal_inspected[
            "provenance_binding_digest"
        ],
        authorization_id=terminal_inspected["required_authorization_id"],
        confirm_explicit_local_task_authorization=True,
    )

    terminal_starts = []

    def terminal_executor(task):
        assert (terminal_campaign_root / "callback_start_claim.json").is_file()
        terminal_starts.append(True)
        return advance_test._fake_native_result(task)

    monkeypatch.setattr(
        runtime_core, "_load_real_executor", lambda _c: terminal_executor
    )
    terminal_execute_kwargs = {
        "expected_campaign_digest": terminal_authorized["campaign_digest"],
        "expected_lease_digest": terminal_authorized["lease_digest"],
        "expected_provenance_binding_digest": terminal_authorized[
            "provenance_binding_digest"
        ],
        "expected_authorization_digest": terminal_authorized[
            "authorization_digest"
        ],
        "expected_attempt_digest": terminal_authorized["attempt_digest"],
    }
    terminal_executed = core.execute_recursive_campaign_task(
        source,
        CAMPAIGN_CONTRACT,
        terminal_campaign_root,
        RUNTIME_CONTRACT,
        **terminal_execute_kwargs,
        confirm_real_local_execution=True,
    )
    assert terminal_starts == [True]
    terminal_preview_before = _tree_observation(terminal_campaign_root)
    terminal_preview = core.verify_recursive_campaign(
        source,
        CAMPAIGN_CONTRACT,
        terminal_campaign_root,
        next_attempt_root=None,
        expected_campaign_digest=terminal_authorized["campaign_digest"],
        expected_lease_digest=terminal_authorized["lease_digest"],
        expected_callback_start_claim_digest=terminal_executed[
            "callback_start_claim_digest"
        ],
        expected_receipt_digest=terminal_executed["receipt_digest"],
        expected_journal_head_digest=terminal_executed[
            "journal_head_digest"
        ],
    )
    assert terminal_preview["phase"] == "CALLBACK_COMPLETED"
    assert terminal_preview["terminal_status"] == "TERMINAL"
    assert terminal_preview["remaining_task_count"] == 0
    assert terminal_preview["execution_status"] == (
        "COMPLETED_SUCCESS_PREVIEW_INDEPENDENTLY_ANCHORED_HARD_STOP"
    )
    assert terminal_preview["next_attempt_root"] is None
    assert terminal_preview["next_pending_evidence_digest"] is None
    assert terminal_preview["next_first_pending_projection_digest"] is None
    assert terminal_preview["next_bundle_digest"] is None
    assert terminal_preview["next_plan_digest"] is None
    assert _tree_observation(terminal_campaign_root) == terminal_preview_before

    terminal_advance_kwargs = {
        **terminal_execute_kwargs,
        "expected_receipt_digest": terminal_executed["receipt_digest"],
        "expected_journal_head_digest": terminal_executed[
            "journal_head_digest"
        ],
        "expected_output_evidence_digest": terminal_preview[
            "output_evidence_digest"
        ],
        "expected_output_report_body_digest": terminal_preview[
            "output_report_body_digest"
        ],
        "expected_output_audit_head": terminal_preview[
            "output_audit_head"
        ],
        "expected_reingestion_digest": terminal_preview[
            "reingestion_digest"
        ],
        "expected_next_pending_evidence_digest": None,
        "expected_next_first_pending_projection_digest": None,
        "expected_next_bundle_digest": None,
        "expected_next_plan_digest": None,
    }
    real_terminal_completed_preview = core._preview_completed_advance
    captured_terminal_preview = []

    def capture_terminal_completed_preview(*args, **kwargs):
        value = real_terminal_completed_preview(*args, **kwargs)
        captured_terminal_preview.append(copy.deepcopy(value))
        return value

    monkeypatch.setattr(
        core,
        "_preview_completed_advance",
        capture_terminal_completed_preview,
    )
    terminal_advanced = core.advance_recursive_campaign(
        source,
        CAMPAIGN_CONTRACT,
        terminal_campaign_root,
        None,
        **terminal_advance_kwargs,
        confirm_immutable_one_step_advance=True,
    )
    monkeypatch.setattr(
        core, "_preview_completed_advance", real_terminal_completed_preview
    )
    assert len(captured_terminal_preview) == 1
    terminal_internal_preview = captured_terminal_preview[0]
    assert terminal_advanced["status"] == TERMINAL
    assert terminal_advanced["phase"] == "ADVANCED_TERMINAL"
    assert terminal_advanced["remaining_task_count"] == 0
    assert terminal_advanced["next_bundle_digest"] is None
    assert terminal_advanced["next_plan_digest"] is None
    assert terminal_advanced["next_attempt_root"] is None
    assert {path.name for path in (terminal_campaign_root / "advance").iterdir()} == {
        "execution_receipt.json",
        "combined_rows.json",
        "output_report.json",
        "reingestion_receipt.json",
        "advance.json",
    }
    assert not (terminal_campaign_root / "advance/next_bundle.json").exists()

    terminal_advance_backup = (
        case["state_home"] / "terminal-advance-complete-backup"
    )
    shutil.copytree(
        terminal_campaign_root / "advance", terminal_advance_backup
    )
    terminal_leaf_order = (
        "execution_receipt.json",
        "combined_rows.json",
        "output_report.json",
        "reingestion_receipt.json",
    )

    def install_terminal_advance_prefix(leaf_count):
        shutil.rmtree(terminal_campaign_root / "advance")
        (terminal_campaign_root / "advance").mkdir(mode=0o700)
        for name in terminal_leaf_order[:leaf_count]:
            shutil.copy2(
                terminal_advance_backup / name,
                terminal_campaign_root / "advance" / name,
            )

    def replay_terminal_preview(*args, **kwargs):
        return copy.deepcopy(terminal_internal_preview)

    prepare_before_terminal_recovery = prepare_calls
    callback_before_terminal_recovery = len(terminal_starts)
    real_terminal_write_new = core._write_new
    terminal_marker_links = []

    def observe_terminal_marker_last(path, value, *args, **kwargs):
        path = Path(path)
        is_marker = path == terminal_campaign_root / "advance/advance.json"
        if is_marker:
            assert set(os.listdir(path.parent)) == set(terminal_leaf_order)
            assert "next_bundle.json" not in os.listdir(path.parent)
        result = real_terminal_write_new(path, value, *args, **kwargs)
        if is_marker:
            terminal_marker_links.append(True)
        return result

    monkeypatch.setattr(
        core, "_preview_completed_advance", replay_terminal_preview
    )
    monkeypatch.setattr(core, "_write_new", observe_terminal_marker_last)
    terminal_prefix_samples = (
        0,
        len(terminal_leaf_order) // 2,
        len(terminal_leaf_order),
    )
    for leaf_count in terminal_prefix_samples:
        install_terminal_advance_prefix(leaf_count)
        recovered_terminal = core.advance_recursive_campaign(
            source,
            CAMPAIGN_CONTRACT,
            terminal_campaign_root,
            None,
            **terminal_advance_kwargs,
            confirm_immutable_one_step_advance=True,
        )
        assert recovered_terminal == terminal_advanced
        assert not (
            terminal_campaign_root / "advance/next_bundle.json"
        ).exists()
        assert _tree_observation(
            terminal_campaign_root / "advance"
        ) == _tree_observation(terminal_advance_backup)

    shutil.rmtree(terminal_campaign_root / "advance")
    (terminal_campaign_root / "advance").mkdir(mode=0o700)
    shutil.copy2(
        terminal_advance_backup / "advance.json",
        terminal_campaign_root / "advance/advance.json",
    )
    terminal_marker_first = _tree_observation(
        terminal_campaign_root / "advance"
    )
    with pytest.raises(core.RecursiveCampaignError):
        core.advance_recursive_campaign(
            source,
            CAMPAIGN_CONTRACT,
            terminal_campaign_root,
            None,
            **terminal_advance_kwargs,
            confirm_immutable_one_step_advance=True,
        )
    assert _tree_observation(
        terminal_campaign_root / "advance"
    ) == terminal_marker_first

    shutil.rmtree(terminal_campaign_root / "advance")
    (terminal_campaign_root / "advance").mkdir(mode=0o700)
    terminal_corrupt = terminal_campaign_root / "advance/output_report.json"
    terminal_corrupt.write_bytes(
        (terminal_advance_backup / "output_report.json").read_bytes()
        + b"\n"
    )
    terminal_corrupt.chmod(0o600)
    terminal_corrupt_before = _tree_observation(
        terminal_campaign_root / "advance"
    )
    with pytest.raises(core.RecursiveCampaignError):
        core.advance_recursive_campaign(
            source,
            CAMPAIGN_CONTRACT,
            terminal_campaign_root,
            None,
            **terminal_advance_kwargs,
            confirm_immutable_one_step_advance=True,
        )
    assert _tree_observation(
        terminal_campaign_root / "advance"
    ) == terminal_corrupt_before
    shutil.rmtree(terminal_campaign_root / "advance")
    shutil.copytree(
        terminal_advance_backup, terminal_campaign_root / "advance"
    )
    monkeypatch.setattr(
        core, "_preview_completed_advance", real_terminal_completed_preview
    )
    monkeypatch.setattr(core, "_write_new", real_terminal_write_new)
    assert len(terminal_marker_links) == len(terminal_prefix_samples)
    assert prepare_calls == prepare_before_terminal_recovery
    assert len(terminal_starts) == callback_before_terminal_recovery

    terminal_verified = core.verify_recursive_campaign(
        source,
        CAMPAIGN_CONTRACT,
        terminal_campaign_root,
        expected_campaign_digest=terminal_advanced["campaign_digest"],
        expected_lease_digest=terminal_advanced["lease_digest"],
        expected_callback_start_claim_digest=terminal_advanced[
            "callback_start_claim_digest"
        ],
        expected_advance_digest=terminal_advanced["advance_digest"],
        expected_output_evidence_digest=terminal_advanced[
            "output_evidence_digest"
        ],
        expected_output_report_body_digest=terminal_advanced[
            "output_report_body_digest"
        ],
        expected_output_audit_head=terminal_advanced["output_audit_head"],
        expected_reingestion_digest=terminal_advanced[
            "reingestion_digest"
        ],
        expected_next_bundle_digest=None,
        expected_next_plan_digest=None,
    )
    assert terminal_verified["status"] == "VERIFIED_" + TERMINAL
    assert terminal_verified["phase"] == "ADVANCED_TERMINAL"

    terminal_source = {
        "schema_version": core.SOURCE_SCHEMA_VERSION,
        "source_kind": "recursive_campaign_v1",
        "dependencies": copy.deepcopy(source["dependencies"]),
        "campaign_contract_path": str(CAMPAIGN_CONTRACT),
        "campaign_root": str(terminal_campaign_root),
        "expected_campaign_digest": terminal_advanced["campaign_digest"],
        "expected_lease_digest": terminal_advanced["lease_digest"],
        "expected_callback_start_claim_digest": terminal_advanced[
            "callback_start_claim_digest"
        ],
        "expected_advance_digest": terminal_advanced["advance_digest"],
        "expected_output_evidence_digest": terminal_advanced[
            "output_evidence_digest"
        ],
        "expected_output_report_body_digest": terminal_advanced[
            "output_report_body_digest"
        ],
        "expected_output_audit_head": terminal_advanced[
            "output_audit_head"
        ],
        "expected_reingestion_digest": terminal_advanced[
            "reingestion_digest"
        ],
        # The strict source schema has digest fields, but a terminal capsule
        # has no such artifacts.  Any proposed values must fail closed.
        "expected_next_bundle_digest": _digest("terminal-has-no-bundle"),
        "expected_next_plan_digest": _digest("terminal-has-no-plan"),
    }
    with pytest.raises(core.RecursiveCampaignError):
        core.inspect_recursive_campaign(
            terminal_source,
            CAMPAIGN_CONTRACT,
            campaign_prefix / "forbidden-after-terminal",
            campaign_id="forbidden-after-terminal",
        )
