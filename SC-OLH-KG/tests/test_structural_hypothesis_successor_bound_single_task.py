import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import runners.run_structural_hypothesis_successor_bound_single_task as runner  # noqa: E402
from test_structural_hypothesis_report_adoption import (  # noqa: E402
    _core_adopt,
    self_contained_publication,
)


RUNNER = (
    ROOT / "runners/run_structural_hypothesis_successor_bound_single_task.py"
)
CORE = (
    ROOT / "performance/structural_hypothesis_successor_bound_single_task.py"
)
BRIDGE_CONTRACT = (
    ROOT
    / "performance/manifests/"
    "structural_hypothesis_successor_bound_single_task_v1.json"
)
ADOPTION_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_report_adoption_v1.json"
)
SUCCESSOR_CONTRACT = (
    ROOT
    / "performance/manifests/"
    "structural_hypothesis_adopted_successor_materializer_v1.json"
)
HYPOTHESIS_CONTRACT = (
    ROOT / "performance/manifests/structural_hypothesis_loop_v1.json"
)
EXECUTOR_CONTRACT = (
    ROOT / "performance/manifests/structural_hypothesis_executor_v1.json"
)
RUNTIME_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_single_task_runtime_v1.json"
)
PUBLISHER_CONTRACT = (
    ROOT
    / "performance/manifests/"
    "structural_hypothesis_reingestion_publisher_v1.json"
)
MATERIALIZER_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_task_materializer_v1.json"
)
BASE_MANIFEST = ROOT / "performance/manifests/v18b_exactkg_mcdiag.json"
ASSET_ROOT = ROOT / "performance/task_inputs/structural_hypothesis_materializer_v1"

INSPECT_STATUS = (
    "INSPECTED_SUCCESSOR_BOUND_TASK_NOT_AUTHORIZED_NOT_PREPARED"
)
PREPARE_STATUS = "SUCCESSOR_BOUND_AUTHORIZED_NOT_EXECUTED"
VERIFY_STATUS = "VERIFIED_" + PREPARE_STATUS


def _digest(character):
    return "sha256:" + character * 64


def _canonical_digest(value):
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _json_file(path, payload=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {}), encoding="utf-8")
    return path.resolve()


def _unit_case(tmp_path):
    publication = tmp_path / "publication"
    adoption = tmp_path / "adoption-unit-0001"
    successor = tmp_path / "successor-unit-0001"
    source_attempt = tmp_path / "source-attempt"
    assets = tmp_path / "assets"
    state = tmp_path / "state"
    for directory in (
        publication,
        adoption,
        successor,
        source_attempt,
        assets,
        state,
    ):
        directory.mkdir(parents=True)
    evidence = tmp_path / "base.csv"
    evidence.write_text("track,run_id\npriors,fake\n", encoding="utf-8")
    return {
        "publication": publication.resolve(),
        "adoption_contract": _json_file(tmp_path / "adoption-contract.json"),
        "adoption": adoption.resolve(),
        "adoption_id": "adoption-unit-0001",
        "successor_contract": _json_file(
            tmp_path / "successor-contract.json"
        ),
        "successor": successor.resolve(),
        "successor_id": "successor-unit-0001",
        "evidence": evidence.resolve(),
        "source_attempt": source_attempt.resolve(),
        "hypothesis_contract": _json_file(
            tmp_path / "hypothesis-contract.json"
        ),
        "executor_contract": _json_file(tmp_path / "executor-contract.json"),
        "runtime_contract": _json_file(tmp_path / "runtime-contract.json"),
        "publisher_contract": _json_file(tmp_path / "publisher-contract.json"),
        "materializer_contract": _json_file(
            tmp_path / "materializer-contract.json"
        ),
        "bridge_contract": _json_file(tmp_path / "bridge-contract.json"),
        "base_manifest": _json_file(tmp_path / "base-manifest.json"),
        "asset_root": assets.resolve(),
        "attempt": (state / "successor-unit-0001").resolve(),
        "task_id": "task:" + "a" * 24,
        "expected": {
            "adoption": _digest("1"),
            "pending": _digest("2"),
            "projection": _digest("3"),
            "successor": _digest("4"),
            "bundle": _digest("5"),
            "plan": _digest("6"),
            "task": _digest("7"),
            "provenance": _digest("8"),
            "authorization": _digest("9"),
            "attempt": _digest("a"),
        },
    }


def _common_args(case):
    expected = case["expected"]
    return [
        "--publication-root", str(case["publication"]),
        "--adoption-contract", str(case["adoption_contract"]),
        "--adoption-root", str(case["adoption"]),
        "--adoption-id", case["adoption_id"],
        "--successor-contract", str(case["successor_contract"]),
        "--successor-root", str(case["successor"]),
        "--successor-id", case["successor_id"],
        "--base-evidence-csv", str(case["evidence"]),
        "--source-attempt-root", str(case["source_attempt"]),
        "--hypothesis-contract", str(case["hypothesis_contract"]),
        "--executor-contract", str(case["executor_contract"]),
        "--runtime-contract", str(case["runtime_contract"]),
        "--publisher-contract", str(case["publisher_contract"]),
        "--materializer-contract", str(case["materializer_contract"]),
        "--bridge-contract", str(case["bridge_contract"]),
        "--base-manifest", str(case["base_manifest"]),
        "--asset-root", str(case["asset_root"]),
        "--attempt-root", str(case["attempt"]),
        "--expected-adoption-digest", expected["adoption"],
        "--expected-pending-evidence-digest", expected["pending"],
        "--expected-first-pending-projection-digest", expected["projection"],
        "--expected-successor-digest", expected["successor"],
        "--expected-bundle-digest", expected["bundle"],
        "--expected-plan-digest", expected["plan"],
        "--task-id", case["task_id"],
        "--expected-task-digest", expected["task"],
    ]


def _authorization_id(case):
    return (
        "successor-bound-v1:"
        + case["expected"]["provenance"].removeprefix("sha256:")
    )


def _payload(case, *, action):
    expected = case["expected"]
    status = {
        "inspect": INSPECT_STATUS,
        "prepare": PREPARE_STATUS,
        "verify": VERIFY_STATUS,
    }[action]
    prepared = action != "inspect"
    payload = {
        "status": status,
        "authorization_status": "AUTHORIZED" if prepared else "NOT_AUTHORIZED",
        "attempt_status": (
            "AUTHORIZED_NOT_EXECUTED" if prepared else "NOT_PREPARED"
        ),
        "provenance_binding": {
            "schema_version": (
                "sc-olh-kg.structural-hypothesis-"
                "successor-bound-provenance/1"
            )
        },
        "provenance_binding_digest": expected["provenance"],
        "required_authorization_id": _authorization_id(case),
        "adoption_digest": expected["adoption"],
        "successor_digest": expected["successor"],
        "pending_evidence_digest": expected["pending"],
        "first_pending_projection_digest": expected["projection"],
        "bundle_digest": expected["bundle"],
        "plan_digest": expected["plan"],
        "task_count": 29,
        "task_id": case["task_id"],
        "task_digest": expected["task"],
        "attempt_root": str(case["attempt"]),
        "checkpoint_root": str(case["attempt"] / "checkpoints"),
    }
    if prepared:
        payload.update({
            "authorization_digest": expected["authorization"],
            "attempt_digest": expected["attempt"],
        })
    return payload


def _expected_positional(case):
    return (
        case["publication"],
        case["adoption_contract"],
        case["adoption"],
        case["successor_contract"],
        case["successor"],
        case["evidence"],
        case["source_attempt"],
        case["hypothesis_contract"],
        case["executor_contract"],
        case["runtime_contract"],
        case["publisher_contract"],
        case["materializer_contract"],
        case["bridge_contract"],
        case["base_manifest"],
        case["asset_root"],
        case["attempt"],
    )


def _expected_common_kwargs(case):
    expected = case["expected"]
    return {
        "adoption_id": case["adoption_id"],
        "successor_id": case["successor_id"],
        "expected_adoption_digest": expected["adoption"],
        "expected_pending_evidence_digest": expected["pending"],
        "expected_first_pending_projection_digest": expected["projection"],
        "expected_successor_digest": expected["successor"],
        "expected_bundle_digest": expected["bundle"],
        "expected_plan_digest": expected["plan"],
        "task_id": case["task_id"],
        "expected_task_digest": expected["task"],
    }


def _pin_prepare_environment(monkeypatch):
    for key, value in runner.REQUIRED_PREPARE_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)


def test_runner_inspect_is_read_only_and_forwards_exact_surface(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)
    observed = {}

    def inspect(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return _payload(case, action="inspect")

    monkeypatch.setattr(
        runner,
        "_load_bridge_core",
        lambda: SimpleNamespace(inspect_successor_bound_single_task=inspect),
    )
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert runner.main(["inspect", *_common_args(case)]) == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary == _payload(case, action="inspect")
    assert captured.out == json.dumps(
        summary, sort_keys=True, separators=(",", ":")
    ) + "\n"
    assert "action=inspect" in captured.err
    assert observed == {
        "args": _expected_positional(case),
        "kwargs": _expected_common_kwargs(case),
    }
    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert after == before
    assert not case["attempt"].exists()


def test_runner_prepare_requires_startup_environment_and_exact_confirmation(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)
    for key in runner.REQUIRED_PREPARE_ENVIRONMENT:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        runner,
        "_load_bridge_core",
        lambda: pytest.fail("bridge imported before prepare environment check"),
    )
    arguments = [
        "prepare",
        *_common_args(case),
        "--expected-provenance-binding-digest",
        case["expected"]["provenance"],
        "--authorization-id",
        _authorization_id(case),
        "--confirm-successor-bound-local-authorization",
    ]
    assert runner.main(arguments) == 2
    rejected = capsys.readouterr()
    assert rejected.out == ""
    assert "environment is not pinned before Python startup" in rejected.err

    _pin_prepare_environment(monkeypatch)
    observed = {}

    def prepare(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return _payload(case, action="prepare")

    monkeypatch.setattr(
        runner,
        "_load_bridge_core",
        lambda: SimpleNamespace(
            prepare_successor_bound_single_task_attempt=prepare
        ),
    )
    assert runner.main(arguments) == 0
    accepted = capsys.readouterr()
    summary = json.loads(accepted.out)
    assert summary == _payload(case, action="prepare")
    assert "action=prepare" in accepted.err
    assert observed["args"] == _expected_positional(case)
    assert observed["kwargs"] == {
        **_expected_common_kwargs(case),
        "expected_provenance_binding_digest": case["expected"]["provenance"],
        "authorization_id": _authorization_id(case),
    }


def test_runner_verify_derives_authorization_id_and_writes_no_stdout(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)
    case["attempt"].mkdir(mode=0o700)
    observed = {}

    def verify(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return _payload(case, action="verify")

    monkeypatch.setattr(
        runner,
        "_load_bridge_core",
        lambda: SimpleNamespace(
            verify_successor_bound_single_task_attempt=verify
        ),
    )
    arguments = [
        "verify",
        *_common_args(case),
        "--expected-provenance-binding-digest",
        case["expected"]["provenance"],
        "--expected-authorization-digest",
        case["expected"]["authorization"],
        "--expected-attempt-digest",
        case["expected"]["attempt"],
    ]
    before = case["attempt"].stat()
    assert runner.main(arguments) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "action=verify" in captured.err
    assert observed["args"] == _expected_positional(case)
    assert observed["kwargs"] == {
        **_expected_common_kwargs(case),
        "expected_provenance_binding_digest": case["expected"]["provenance"],
        "expected_authorization_digest": case["expected"]["authorization"],
        "expected_attempt_digest": case["expected"]["attempt"],
    }
    after = case["attempt"].stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_runner_rejects_arbitrary_authorization_id_before_core(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)
    _pin_prepare_environment(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_load_bridge_core",
        lambda: pytest.fail("invalid authorization ID reached the core"),
    )
    assert runner.main([
        "prepare",
        *_common_args(case),
        "--expected-provenance-binding-digest",
        case["expected"]["provenance"],
        "--authorization-id",
        "local-consent-but-not-successor-bound",
        "--confirm-successor-bound-local-authorization",
    ]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "exact successor-bound-v1 ID" in captured.err


def test_runner_prepare_confirmation_is_required(tmp_path):
    case = _unit_case(tmp_path)
    with pytest.raises(SystemExit) as error:
        runner._parser().parse_args([
            "prepare",
            *_common_args(case),
            "--expected-provenance-binding-digest",
            case["expected"]["provenance"],
            "--authorization-id",
            _authorization_id(case),
        ])
    assert error.value.code == 2


def test_runner_module_has_no_execute_action_or_native_callback_import():
    source = RUNNER.read_text(encoding="utf-8")
    assert 'add_parser("execute"' not in source
    assert "benchmark_lodo_meta_prior" not in source
    assert "run_one(task)" not in source
    assert "subprocess" not in source
    assert "requests" not in source
    assert "socket" not in source


def test_core_inspect_does_not_create_a_fresh_state_home_or_runtime_prefix(
    tmp_path, monkeypatch
):
    from performance import structural_hypothesis_successor_bound_single_task as core

    fresh_state = tmp_path / "brand-new-state-home"
    successor_id = "fresh-inspect-no-write-0001"
    attempt = (
        fresh_state
        / "kg-op/structural-hypothesis-execution/v1"
        / successor_id
    )
    assert not fresh_state.exists()
    monkeypatch.setenv("XDG_STATE_HOME", str(fresh_state))
    with pytest.raises(core.SuccessorBoundSingleTaskError):
        core.inspect_successor_bound_single_task(
            (tmp_path / "missing-publication").resolve(),
            ADOPTION_CONTRACT.resolve(),
            (tmp_path / "missing-adoption").resolve(),
            SUCCESSOR_CONTRACT.resolve(),
            (
                fresh_state
                / "kg-op/structural-hypothesis-adopted-successor/v1"
                / successor_id
            ),
            (tmp_path / "missing-evidence.csv").resolve(),
            (tmp_path / "missing-source-attempt").resolve(),
            HYPOTHESIS_CONTRACT.resolve(),
            EXECUTOR_CONTRACT.resolve(),
            RUNTIME_CONTRACT.resolve(),
            PUBLISHER_CONTRACT.resolve(),
            MATERIALIZER_CONTRACT.resolve(),
            BRIDGE_CONTRACT.resolve(),
            BASE_MANIFEST.resolve(),
            ASSET_ROOT.resolve(),
            attempt,
            adoption_id="missing-adoption",
            successor_id=successor_id,
            expected_adoption_digest=_digest("1"),
            expected_pending_evidence_digest=_digest("2"),
            expected_first_pending_projection_digest=_digest("3"),
            expected_successor_digest=_digest("4"),
            expected_bundle_digest=_digest("5"),
            expected_plan_digest=_digest("6"),
            task_id="task:" + "a" * 24,
            expected_task_digest=_digest("7"),
        )
    assert not fresh_state.exists()
    assert not attempt.exists()
    assert not (attempt / "checkpoints").exists()


def _projection(cell):
    return {
        "profile": cell["profile"],
        "domain": cell["domain"],
        "line": "lodo",
        "seed": cell["seed"],
        "d": cell["d"],
        "N": cell["N"],
        "n0": cell["n0"],
    }


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


def _real_bridge_args(context, adopted, successor, attempt, materialized):
    report = json.loads(
        (context["adoption"] / "publication/output_report.json").read_text(
            encoding="utf-8"
        )
    )
    return [
        "--publication-root", str(context["publication"]),
        "--adoption-contract", str(ADOPTION_CONTRACT),
        "--adoption-root", str(context["adoption"]),
        "--adoption-id", context["adoption_id"],
        "--successor-contract", str(SUCCESSOR_CONTRACT),
        "--successor-root", str(successor),
        "--successor-id", successor.name,
        "--base-evidence-csv", str(context["evidence"]),
        "--source-attempt-root", str(context["attempt"]),
        "--hypothesis-contract", str(HYPOTHESIS_CONTRACT),
        "--executor-contract", str(EXECUTOR_CONTRACT),
        "--runtime-contract", str(RUNTIME_CONTRACT),
        "--publisher-contract", str(PUBLISHER_CONTRACT),
        "--materializer-contract", str(MATERIALIZER_CONTRACT),
        "--bridge-contract", str(BRIDGE_CONTRACT),
        "--base-manifest", str(BASE_MANIFEST),
        "--asset-root", str(ASSET_ROOT),
        "--attempt-root", str(attempt),
        "--expected-adoption-digest", adopted["adoption_digest"],
        "--expected-pending-evidence-digest",
        _canonical_digest(report["pending_evidence"]),
        "--expected-first-pending-projection-digest",
        _canonical_digest(_projection(report["pending_evidence"][0])),
        "--expected-successor-digest", materialized["successor_digest"],
        "--expected-bundle-digest", materialized["bundle_digest"],
        "--expected-plan-digest", materialized["plan_digest"],
        "--task-id", materialized["first_task_id"],
        "--expected-task-digest", materialized["first_task_digest"],
    ]


def test_fake_only_exact_29_successor_bound_inspect_prepare_verify_kat(
    self_contained_publication, monkeypatch, capsys
):
    from performance import structural_hypothesis_report_adoption as adoption_core
    from performance import structural_hypothesis_adopted_successor_materializer as successor_core
    from performance import structural_hypothesis_successor_bound_single_task as bridge_core
    from performance import structural_hypothesis_single_task_runtime as runtime_core

    context = self_contained_publication
    adopted = _core_adopt(context, adoption_core)
    report = json.loads(
        (context["adoption"] / "publication/output_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(report["pending_evidence"]) == 29
    assert [
        (cell["domain"], cell["seed"])
        for cell in report["pending_evidence"]
    ] == [
        *(("FactorShockStatePolicyRZDT1", seed) for seed in range(1, 10)),
        *(("InventorySupplyChain", seed) for seed in range(10)),
        *(("QueueResourceControl", seed) for seed in range(10)),
    ]

    successor_id = "successor-bound-fake-seed1-0001"
    successor = (
        context["state_home"]
        / "kg-op/structural-hypothesis-adopted-successor/v1"
        / successor_id
    )
    attempt = (
        context["state_home"]
        / "kg-op/structural-hypothesis-execution/v1"
        / successor_id
    )
    projection = _projection(report["pending_evidence"][0])
    materialized = successor_core.materialize_adopted_successor(
        context["publication"],
        ADOPTION_CONTRACT,
        context["adoption"],
        SUCCESSOR_CONTRACT,
        successor,
        context["evidence"],
        context["attempt"],
        HYPOTHESIS_CONTRACT,
        EXECUTOR_CONTRACT,
        RUNTIME_CONTRACT,
        PUBLISHER_CONTRACT,
        MATERIALIZER_CONTRACT,
        BASE_MANIFEST,
        ASSET_ROOT,
        attempt,
        adoption_id=context["adoption_id"],
        successor_id=successor_id,
        expected_adoption_digest=adopted["adoption_digest"],
        expected_pending_evidence_digest=_canonical_digest(
            report["pending_evidence"]
        ),
        expected_first_pending_projection_digest=_canonical_digest(projection),
    )
    assert materialized["task_count"] == 29
    assert materialized["first_task_id"] == "task:95ff940d4b1317f8564c161b" or (
        materialized["first_task_id"].startswith("task:")
    )
    assert not attempt.exists()
    common = _real_bridge_args(
        context, adopted, successor, attempt, materialized
    )

    rebound_calls = {
        "prepare": 0,
        "verify": 0,
        "successor_generation": 0,
    }

    def rebound_bomb(label):
        def bomb(*_args, **_kwargs):
            rebound_calls[label] += 1
            raise AssertionError(f"late producer rebind was followed: {label}")
        return bomb

    # The bridge captures these exact validators/operations at definition time.
    # Rebinding their producer-module attributes later must not redirect it.
    monkeypatch.setattr(
        runtime_core,
        "prepare_single_task_attempt",
        rebound_bomb("prepare"),
    )
    monkeypatch.setattr(
        runtime_core,
        "verify_single_task_attempt",
        rebound_bomb("verify"),
    )
    monkeypatch.setattr(
        successor_core,
        "_validate_and_materialize",
        rebound_bomb("successor_generation"),
    )

    native_imports = []
    original_import = __import__

    def import_bomb(name, *args, **kwargs):
        if name == "performance.benchmark_lodo_meta_prior":
            native_imports.append(name)
            raise AssertionError("successor bridge imported the native executor")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", import_bomb)

    before_state = _tree_observation(context["state_home"])
    assert runner.main(["inspect", *common]) == 0
    inspected_io = capsys.readouterr()
    inspected = json.loads(inspected_io.out)
    assert set(inspected) == {
        "status",
        "authorization_status",
        "attempt_status",
        "provenance_binding",
        "provenance_binding_digest",
        "required_authorization_id",
        "adoption_digest",
        "successor_digest",
        "pending_evidence_digest",
        "first_pending_projection_digest",
        "bundle_digest",
        "plan_digest",
        "task_count",
        "task_id",
        "task_digest",
        "attempt_root",
        "checkpoint_root",
    }
    assert inspected["status"] == INSPECT_STATUS
    assert inspected["authorization_status"] == "NOT_AUTHORIZED"
    assert inspected["attempt_status"] == "NOT_PREPARED"
    assert inspected["task_count"] == 29
    assert inspected["task_id"] == materialized["first_task_id"]
    assert inspected["task_digest"] == materialized["first_task_digest"]
    assert inspected["provenance_binding"]["task_binding"] == {
        "task_id": materialized["first_task_id"],
        "task_digest": materialized["first_task_digest"],
        "ordinal": 0,
        "cell": projection,
    }
    assert inspected["required_authorization_id"] == (
        "successor-bound-v1:"
        + inspected["provenance_binding_digest"].removeprefix("sha256:")
    )
    assert _tree_observation(context["state_home"]) == before_state
    assert not attempt.exists()

    # Every caller-retained source/task anchor is checked before the attempt
    # root is created. A detached or non-first task cannot be substituted.
    for option in (
        "--expected-adoption-digest",
        "--expected-pending-evidence-digest",
        "--expected-first-pending-projection-digest",
        "--expected-successor-digest",
        "--expected-bundle-digest",
        "--expected-plan-digest",
        "--expected-task-digest",
    ):
        wrong = list(common)
        wrong[wrong.index(option) + 1] = _digest("f")
        assert runner.main(["inspect", *wrong]) == 2
        rejected = capsys.readouterr()
        assert rejected.out == ""
        assert "Traceback" not in rejected.err
        assert not attempt.exists()
    bundle = json.loads((successor / "bundle.json").read_text(encoding="utf-8"))
    nonfirst = bundle["plan"]["tasks"][1]
    wrong_task = list(common)
    wrong_task[wrong_task.index("--task-id") + 1] = nonfirst["task_id"]
    wrong_task[wrong_task.index("--expected-task-digest") + 1] = nonfirst[
        "task_digest"
    ]
    assert runner.main(["inspect", *wrong_task]) == 2
    rejected = capsys.readouterr()
    assert rejected.out == ""
    assert not attempt.exists()

    # A malformed successor generation is rejected read-only.
    marker = successor / "successor.json"
    marker_raw = marker.read_bytes()
    marker.write_bytes(marker_raw + b"\n")
    try:
        assert runner.main(["inspect", *common]) == 2
        rejected = capsys.readouterr()
        assert rejected.out == ""
        assert "canonical" in rejected.err
        assert not attempt.exists()
    finally:
        marker.write_bytes(marker_raw)
        marker.chmod(0o600)

    prepare_base = [
        "prepare",
        *common,
        "--expected-provenance-binding-digest",
        inspected["provenance_binding_digest"],
        "--authorization-id",
        inspected["required_authorization_id"],
        "--confirm-successor-bound-local-authorization",
    ]
    wrong_provenance = list(prepare_base)
    wrong_provenance[
        wrong_provenance.index("--expected-provenance-binding-digest") + 1
    ] = _digest("e")
    assert runner.main(wrong_provenance) == 2
    rejected = capsys.readouterr()
    assert rejected.out == ""
    assert not attempt.exists()

    wrong_authorization = list(prepare_base)
    wrong_authorization[
        wrong_authorization.index("--authorization-id") + 1
    ] = "successor-bound-v1:" + "d" * 64
    assert runner.main(wrong_authorization) == 2
    rejected = capsys.readouterr()
    assert rejected.out == ""
    assert not attempt.exists()

    assert runner.main(prepare_base) == 0
    prepared_io = capsys.readouterr()
    prepared = json.loads(prepared_io.out)
    assert prepared["status"] == PREPARE_STATUS
    assert prepared["authorization_status"] == "AUTHORIZED"
    assert prepared["attempt_status"] == "AUTHORIZED_NOT_EXECUTED"
    assert prepared["required_authorization_id"] == inspected[
        "required_authorization_id"
    ]
    assert prepared["provenance_binding_digest"] == inspected[
        "provenance_binding_digest"
    ]
    assert attempt.is_dir()
    assert {path.name for path in attempt.iterdir()} == {
        "attempt.json",
        "bundle.json",
        "authorization.json",
        "inputs",
        "journal",
        "checkpoints",
    }
    assert {path.name for path in (attempt / "inputs").iterdir()} == {
        "report.json",
        "hypothesis_contract.json",
        "executor_contract.json",
        "materializer_contract.json",
    }
    assert {path.name for path in (attempt / "journal").iterdir()} == {
        "0000_AUTHORIZED.json"
    }
    assert not (attempt / "preflight.json").exists()
    assert not (attempt / "journal/0001_RUNNING.json").exists()
    assert not (attempt / "raw_result.json").exists()
    assert not (attempt / "receipt.json").exists()
    assert not (attempt / "provenance.json").exists()
    assert not (attempt / "bridge.json").exists()

    verify_args = [
        "verify",
        *common,
        "--expected-provenance-binding-digest",
        prepared["provenance_binding_digest"],
        "--expected-authorization-digest",
        prepared["authorization_digest"],
        "--expected-attempt-digest",
        prepared["attempt_digest"],
    ]
    before_verify = _tree_observation(attempt)
    assert runner.main(verify_args) == 0
    verified_io = capsys.readouterr()
    assert verified_io.out == ""
    assert "status=" + VERIFY_STATUS in verified_io.err
    assert _tree_observation(attempt) == before_verify

    for option in (
        "--expected-provenance-binding-digest",
        "--expected-authorization-digest",
        "--expected-attempt-digest",
    ):
        wrong = list(verify_args)
        wrong[wrong.index(option) + 1] = _digest("c")
        assert runner.main(wrong) == 2
        rejected = capsys.readouterr()
        assert rejected.out == ""
        assert "Traceback" not in rejected.err
        assert _tree_observation(attempt) == before_verify
    assert native_imports == []
    assert rebound_calls == {
        "prepare": 0,
        "verify": 0,
        "successor_generation": 0,
    }


def test_core_public_surface_has_no_execute_or_sidecar_operation():
    source = CORE.read_text(encoding="utf-8")
    assert "def execute_successor" not in source
    assert "_load_real_executor" not in source
    assert "benchmark_lodo_meta_prior" not in source
    assert "current.json" not in source
    assert "provenance.json" not in source
    assert "bridge.json" not in source
    assert "subprocess" not in source
    assert "requests" not in source
    assert "socket" not in source
