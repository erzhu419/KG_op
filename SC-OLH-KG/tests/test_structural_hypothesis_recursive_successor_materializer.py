import copy
import hashlib
import json
import os
from pathlib import Path
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

import runners.run_structural_hypothesis_recursive_successor_materializer as runner  # noqa: E402
import test_structural_hypothesis_recursive_report_advance as advance_test  # noqa: E402
from test_structural_hypothesis_recursive_report_advance import (  # noqa: E402,F401
    fake_completed_successor_case,
)


RUNNER = (
    ROOT
    / "runners/"
    "run_structural_hypothesis_recursive_successor_materializer.py"
)
CORE = (
    ROOT
    / "performance/"
    "structural_hypothesis_recursive_successor_materializer.py"
)
RECURSIVE_SUCCESSOR_CONTRACT = (
    ROOT
    / "performance/manifests/"
    "structural_hypothesis_recursive_successor_materializer_v1.json"
)

STATUS = (
    "RECURSIVE_SUCCESSOR_MATERIALIZED_FROM_VERIFIED_ADVANCE_"
    "NOT_AUTHORIZED"
)
VERIFY_STATUS = "VERIFIED_" + STATUS
RESULT_KEYS = (
    "status",
    "recursive_successor_root",
    "recursive_successor_digest",
    "advance_digest",
    "advance_output_evidence_digest",
    "pending_evidence_digest",
    "first_pending_projection_digest",
    "bundle_digest",
    "plan_digest",
    "first_task_id",
    "first_task_digest",
    "task_count",
    "future_attempt_root",
    "checkpoint_root",
    "current_status",
    "authorization_status",
    "attempt_status",
    "execution_status",
)
COMMON_DIGEST_NAMES = runner._COMMON_DIGEST_NAMES
VERIFY_DIGEST_NAMES = runner._VERIFY_DIGEST_NAMES


def _digest(label):
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical_digest(value):
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


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


def _json_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    return path.resolve()


def _unit_case(tmp_path, *, committed=False):
    adoption_id = "adoption-unit-0001"
    source_successor_id = "source-successor-unit-0001"
    advance_id = "advance-unit-0001"
    recursive_successor_id = "recursive-successor-unit-0001"
    directories = {
        "publication": (tmp_path / "publication").resolve(),
        "adoption": (tmp_path / adoption_id).resolve(),
        "source_successor": (tmp_path / source_successor_id).resolve(),
        "source_attempt": (tmp_path / "source-attempt").resolve(),
        "asset_root": (tmp_path / "assets").resolve(),
        "completed_attempt": (tmp_path / "completed-attempt").resolve(),
        "advance": (tmp_path / advance_id).resolve(),
    }
    for path in directories.values():
        path.mkdir(parents=True)
    recursive_successor = (tmp_path / recursive_successor_id).resolve()
    if committed:
        recursive_successor.mkdir()
    future_attempt = (
        tmp_path / "future-attempts" / recursive_successor_id
    ).resolve()
    file_names = (
        "adoption_contract",
        "source_successor_contract",
        "hypothesis_contract",
        "executor_contract",
        "runtime_contract",
        "publisher_contract",
        "materializer_contract",
        "bridge_contract",
        "base_manifest",
        "advance_contract",
        "recursive_successor_contract",
    )
    files = {
        name: _json_file(tmp_path / "contracts" / f"{name}.json")
        for name in file_names
    }
    evidence = tmp_path / "base.csv"
    evidence.write_text("track,run_id\npriors,fake\n", encoding="utf-8")
    digests = {
        name: _digest(name)
        for name in (*COMMON_DIGEST_NAMES, *VERIFY_DIGEST_NAMES)
    }
    return {
        **directories,
        **files,
        "evidence": evidence.resolve(),
        "recursive_successor": recursive_successor,
        "future_attempt": future_attempt,
        "adoption_id": adoption_id,
        "source_successor_id": source_successor_id,
        "advance_id": advance_id,
        "recursive_successor_id": recursive_successor_id,
        "completed_task_id": "task:" + "a" * 24,
        "digests": digests,
    }


def _unit_common_cli(case):
    values = [
        "--publication-root", str(case["publication"]),
        "--adoption-contract", str(case["adoption_contract"]),
        "--adoption-root", str(case["adoption"]),
        "--adoption-id", case["adoption_id"],
        "--source-successor-contract",
        str(case["source_successor_contract"]),
        "--source-successor-root", str(case["source_successor"]),
        "--source-successor-id", case["source_successor_id"],
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
        "--completed-attempt-root", str(case["completed_attempt"]),
        "--advance-contract", str(case["advance_contract"]),
        "--advance-root", str(case["advance"]),
        "--advance-id", case["advance_id"],
        "--recursive-successor-contract",
        str(case["recursive_successor_contract"]),
        "--recursive-successor-root", str(case["recursive_successor"]),
        "--recursive-successor-id", case["recursive_successor_id"],
        "--future-attempt-root", str(case["future_attempt"]),
        "--completed-task-id", case["completed_task_id"],
    ]
    for name in COMMON_DIGEST_NAMES:
        values.extend((
            "--" + name.replace("_", "-"), case["digests"][name]
        ))
    return values


def _unit_verify_cli(case):
    values = [*_unit_common_cli(case)]
    for name in VERIFY_DIGEST_NAMES:
        values.extend((
            "--" + name.replace("_", "-"), case["digests"][name]
        ))
    return values


def _unit_positional(case):
    return (
        case["publication"],
        case["adoption_contract"],
        case["adoption"],
        case["source_successor_contract"],
        case["source_successor"],
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
        case["completed_attempt"],
        case["advance_contract"],
        case["advance"],
        case["recursive_successor_contract"],
        case["recursive_successor"],
        case["future_attempt"],
    )


def _unit_kwargs(case):
    digests = case["digests"]
    return {
        "adoption_id": case["adoption_id"],
        "source_successor_id": case["source_successor_id"],
        "advance_id": case["advance_id"],
        "recursive_successor_id": case["recursive_successor_id"],
        **{name: digests[name] for name in COMMON_DIGEST_NAMES[:6]},
        "completed_task_id": case["completed_task_id"],
        **{name: digests[name] for name in COMMON_DIGEST_NAMES[6:]},
    }


def _unit_payload(case, *, verified=False):
    digests = case["digests"]
    return {
        "status": VERIFY_STATUS if verified else STATUS,
        "recursive_successor_root": str(case["recursive_successor"]),
        "recursive_successor_digest": digests[
            "expected_recursive_successor_digest"
        ],
        "advance_digest": digests["expected_advance_digest"],
        "advance_output_evidence_digest": digests[
            "expected_advance_output_evidence_digest"
        ],
        "pending_evidence_digest": digests[
            "expected_next_pending_evidence_digest"
        ],
        "first_pending_projection_digest": digests[
            "expected_next_first_pending_projection_digest"
        ],
        "bundle_digest": digests["expected_next_bundle_digest"],
        "plan_digest": digests["expected_next_plan_digest"],
        "first_task_id": "task:" + "b" * 24,
        "first_task_digest": _digest("first-task"),
        "task_count": 28,
        "future_attempt_root": str(case["future_attempt"]),
        "checkpoint_root": str(case["future_attempt"] / "checkpoints"),
        "current_status": "NOT_CURRENT",
        "authorization_status": "NOT_AUTHORIZED",
        "attempt_status": "NOT_PREPARED",
        "execution_status": "NOT_EXECUTED",
    }


def test_runner_forwards_exact_frozen_api_and_verify_is_read_only(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)
    calls = []

    def materialize(*args, **kwargs):
        calls.append(("materialize", args, kwargs))
        return {**_unit_payload(case), "ignored": "not-on-surface"}

    def verify(*args, **kwargs):
        calls.append(("verify", args, kwargs))
        return _unit_payload(case, verified=True)

    monkeypatch.setattr(
        runner,
        "_load_recursive_successor_core",
        lambda: SimpleNamespace(
            materialize_recursive_successor=materialize,
            verify_recursive_successor=verify,
        ),
    )
    assert runner.main([
        "materialize",
        *_unit_common_cli(case),
        "--confirm-recursive-successor-materialization",
    ]) == 0
    made = capsys.readouterr()
    expected = _unit_payload(case)
    assert made.out == json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ) + "\n"
    assert "action=materialize" in made.err
    assert calls == [(
        "materialize",
        _unit_positional(case),
        {
            **_unit_kwargs(case),
            "confirm_recursive_successor_materialization": True,
        },
    )]

    case["recursive_successor"].mkdir(mode=0o700)
    before = _tree_observation(case["recursive_successor"])
    assert runner.main(["verify", *_unit_verify_cli(case)]) == 0
    checked = capsys.readouterr()
    assert checked.out == ""
    assert "action=verify" in checked.err
    assert "status=" + VERIFY_STATUS in checked.err
    assert _tree_observation(case["recursive_successor"]) == before
    assert calls[-1] == (
        "verify",
        _unit_positional(case),
        {
            **_unit_kwargs(case),
            **{
                name: case["digests"][name]
                for name in VERIFY_DIGEST_NAMES
            },
        },
    )


def test_runner_requires_confirmation_and_fails_closed_before_core(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)
    with pytest.raises(SystemExit) as error:
        runner._parser().parse_args([
            "materialize", *_unit_common_cli(case)
        ])
    assert error.value.code == 2

    called = []
    monkeypatch.setattr(
        runner,
        "_load_recursive_successor_core",
        lambda: called.append(True),
    )
    argv = [
        "materialize",
        *_unit_common_cli(case),
        "--confirm-recursive-successor-materialization",
    ]
    option = "--expected-advance-digest"
    argv[argv.index(option) + 1] = "sha256:BAD"
    assert runner.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "lowercase sha256 digest" in captured.err
    assert called == []
    assert not case["recursive_successor"].exists()


def test_runner_startup_has_no_execution_or_network_imports():
    source = RUNNER.read_text(encoding="utf-8")
    for forbidden in (
        "benchmark_lodo_meta_prior",
        "run_one",
        "subprocess",
        "requests",
        "socket",
        "import scheduler",
        "from scheduler",
    ):
        assert forbidden not in source.lower()
    assert set(runner._parser()._subparsers._group_actions[0].choices) == {
        "materialize",
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
        raise AssertionError("recursive successor startup crossed boundary")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
from runners.run_structural_hypothesis_recursive_successor_materializer import _parser
assert set(_parser()._subparsers._group_actions[0].choices) == {"materialize", "verify"}
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


@pytest.fixture
def fake_recursive_advance_case(fake_completed_successor_case):
    """Reuse the fake-only seed-0/seed-1 chain, then commit its advance."""
    from performance import structural_hypothesis_recursive_report_advance as advance_core

    source = fake_completed_successor_case
    advanced = advance_core.advance_recursive_report_version(
        *advance_test._real_source_positional(source),
        advance_test.ADVANCE_CONTRACT,
        source["advance_root"],
        advance_id=source["advance_id"],
        adoption_id=source["adoption_id"],
        successor_id=source["successor_id"],
        task_id=source["expected"]["task_id"],
        **advance_test._real_core_expected(source),
        confirm_immutable_local_report_advance=True,
    )
    output_report = json.loads(
        (source["advance_root"] / "output_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(output_report["pending_evidence"]) == 28
    projection = advance_test._projection(
        output_report["pending_evidence"][0]
    )
    assert (projection["domain"], projection["seed"]) == (
        "FactorShockStatePolicyRZDT1",
        2,
    )
    recursive_successor_id = "fake-recursive-successor-seed2"
    recursive_successor_root = (
        source["state_home"]
        / "kg-op/structural-hypothesis-recursive-successor/v1"
        / recursive_successor_id
    )
    future_attempt_root = (
        source["state_home"]
        / "kg-op/structural-hypothesis-execution/v1"
        / recursive_successor_id
    )
    expected = {
        "expected_adoption_digest": source["expected"]["adoption_digest"],
        "expected_source_pending_evidence_digest": source["expected"][
            "pending_evidence_digest"
        ],
        "expected_source_first_pending_projection_digest": source[
            "expected"
        ]["first_pending_projection_digest"],
        "expected_source_successor_digest": source["expected"][
            "successor_digest"
        ],
        "expected_source_bundle_digest": source["expected"][
            "bundle_digest"
        ],
        "expected_source_plan_digest": source["expected"]["plan_digest"],
        "expected_completed_task_digest": source["expected"]["task_digest"],
        "expected_source_provenance_binding_digest": source["expected"][
            "provenance_binding_digest"
        ],
        "expected_source_authorization_digest": source["expected"][
            "authorization_digest"
        ],
        "expected_source_attempt_digest": source["expected"][
            "attempt_digest"
        ],
        "expected_source_execution_receipt_digest": source["expected"][
            "execution_receipt_digest"
        ],
        "expected_source_execution_journal_head_digest": source[
            "expected"
        ]["execution_journal_head_digest"],
        "expected_advance_digest": advanced["advance_digest"],
        "expected_advance_reingestion_digest": advanced[
            "reingestion_digest"
        ],
        "expected_advance_output_report_body_digest": advanced[
            "output_report_body_digest"
        ],
        "expected_advance_output_audit_head": advanced[
            "output_audit_head"
        ],
        "expected_advance_output_evidence_digest": advanced[
            "output_evidence_digest"
        ],
        "expected_next_pending_evidence_digest": _canonical_digest(
            output_report["pending_evidence"]
        ),
        "expected_next_first_pending_projection_digest": (
            _canonical_digest(projection)
        ),
    }
    return {
        **source,
        "advanced": advanced,
        "advance_output_report": output_report,
        "recursive_successor_id": recursive_successor_id,
        "recursive_successor_root": recursive_successor_root,
        "future_attempt_root": future_attempt_root,
        "recursive_expected": expected,
    }


def _real_common_cli(
    case,
    *,
    recursive_successor_root=None,
    recursive_successor_id=None,
    future_attempt_root=None,
):
    root = recursive_successor_root or case["recursive_successor_root"]
    successor_id = recursive_successor_id or case["recursive_successor_id"]
    future = future_attempt_root or case["future_attempt_root"]
    values = [
        "--publication-root", str(case["publication"]),
        "--adoption-contract", str(advance_test.ADOPTION_CONTRACT),
        "--adoption-root", str(case["adoption"]),
        "--adoption-id", case["adoption_id"],
        "--source-successor-contract", str(advance_test.SUCCESSOR_CONTRACT),
        "--source-successor-root", str(case["successor"]),
        "--source-successor-id", case["successor_id"],
        "--base-evidence-csv", str(case["evidence"]),
        "--source-attempt-root", str(case["source_attempt"]),
        "--hypothesis-contract", str(advance_test.HYPOTHESIS_CONTRACT),
        "--executor-contract", str(advance_test.EXECUTOR_CONTRACT),
        "--runtime-contract", str(advance_test.RUNTIME_CONTRACT),
        "--publisher-contract", str(advance_test.PUBLISHER_CONTRACT),
        "--materializer-contract", str(advance_test.MATERIALIZER_CONTRACT),
        "--bridge-contract", str(advance_test.BRIDGE_CONTRACT),
        "--base-manifest", str(advance_test.BASE_MANIFEST),
        "--asset-root", str(advance_test.ASSET_ROOT),
        "--completed-attempt-root", str(case["completed_attempt"]),
        "--advance-contract", str(advance_test.ADVANCE_CONTRACT),
        "--advance-root", str(case["advance_root"]),
        "--advance-id", case["advance_id"],
        "--recursive-successor-contract", str(RECURSIVE_SUCCESSOR_CONTRACT),
        "--recursive-successor-root", str(root),
        "--recursive-successor-id", successor_id,
        "--future-attempt-root", str(future),
        "--completed-task-id", case["expected"]["task_id"],
    ]
    for name in COMMON_DIGEST_NAMES:
        values.extend((
            "--" + name.replace("_", "-"),
            case["recursive_expected"][name],
        ))
    return values


def _real_verify_cli(case, materialized):
    return [
        *_real_common_cli(case),
        "--expected-recursive-successor-digest",
        materialized["recursive_successor_digest"],
        "--expected-next-bundle-digest", materialized["bundle_digest"],
        "--expected-next-plan-digest", materialized["plan_digest"],
    ]


def _materialize(case, capsys, **overrides):
    assert runner.main([
        "materialize",
        *_real_common_cli(case, **overrides),
        "--confirm-recursive-successor-materialization",
    ]) == 0
    captured = capsys.readouterr()
    assert "action=materialize" in captured.err
    return json.loads(captured.out)


def _verify_rejects(case, materialized, capsys, *, argv=None):
    assert runner.main(
        argv or ["verify", *_real_verify_cli(case, materialized)]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err


def test_fake_only_recursive_successor_kat_all_anchors_and_hardening(
    fake_recursive_advance_case, monkeypatch, capsys
):
    from performance import structural_hypothesis_recursive_successor_materializer as core

    case = fake_recursive_advance_case
    prefix = case["recursive_successor_root"].parent
    future_prefix = case["future_attempt_root"].parent

    # Every independently retained source/advance/next-pending anchor is
    # mandatory before publication of any recursive successor root.
    for ordinal, name in enumerate(COMMON_DIGEST_NAMES):
        successor_id = f"wrong-recursive-anchor-{ordinal:02d}"
        target = prefix / successor_id
        future = future_prefix / successor_id
        argv = [
            "materialize",
            *_real_common_cli(
                case,
                recursive_successor_root=target,
                recursive_successor_id=successor_id,
                future_attempt_root=future,
            ),
            "--confirm-recursive-successor-materialization",
        ]
        option = "--" + name.replace("_", "-")
        argv[argv.index(option) + 1] = _digest("wrong-" + name)
        assert runner.main(argv) == 2, name
        rejected = capsys.readouterr()
        assert rejected.out == "", name
        assert "Traceback" not in rejected.err, name
        assert not target.exists(), name
        assert not future.exists(), name

    # The two source-generation captures must agree before root creation.
    real_validate_generation = core._validate_generation
    generation_calls = 0

    def changed_second_generation(*args, **kwargs):
        nonlocal generation_calls
        generation_calls += 1
        generation = real_validate_generation(*args, **kwargs)
        if generation_calls == 2:
            generation = copy.deepcopy(generation)
            generation["bundle"]["plan"]["tasks"][0]["ordinal"] = 999
        return generation

    monkeypatch.setattr(core, "_validate_generation", changed_second_generation)
    changed_id = "recursive-generation-changed"
    changed_root = prefix / changed_id
    changed_future = future_prefix / changed_id
    assert runner.main([
        "materialize",
        *_real_common_cli(
            case,
            recursive_successor_root=changed_root,
            recursive_successor_id=changed_id,
            future_attempt_root=changed_future,
        ),
        "--confirm-recursive-successor-materialization",
    ]) == 2
    changed = capsys.readouterr()
    assert changed.out == ""
    assert "generation" in changed.err.lower()
    assert not changed_root.exists()
    monkeypatch.setattr(core, "_validate_generation", real_validate_generation)

    source_before = {
        name: _tree_observation(case[name])
        for name in (
            "publication",
            "adoption",
            "successor",
            "source_attempt",
            "completed_attempt",
            "advance_root",
        )
    }
    materialized = _materialize(case, capsys)
    assert set(materialized) == set(RESULT_KEYS)
    assert materialized["status"] == STATUS
    assert materialized["task_count"] == 28
    assert materialized["current_status"] == "NOT_CURRENT"
    assert materialized["authorization_status"] == "NOT_AUTHORIZED"
    assert materialized["attempt_status"] == "NOT_PREPARED"
    assert materialized["execution_status"] == "NOT_EXECUTED"
    assert materialized["advance_digest"] == case["advanced"][
        "advance_digest"
    ]
    assert materialized["advance_output_evidence_digest"] == case[
        "advanced"
    ]["output_evidence_digest"]
    assert materialized["pending_evidence_digest"] == case[
        "recursive_expected"
    ]["expected_next_pending_evidence_digest"]
    assert materialized["first_pending_projection_digest"] == case[
        "recursive_expected"
    ]["expected_next_first_pending_projection_digest"]
    assert not case["future_attempt_root"].exists()
    assert not (case["future_attempt_root"] / "checkpoints").exists()

    root = case["recursive_successor_root"]
    assert {path.name for path in root.iterdir()} == {
        "recursive_successor_contract.json",
        "bundle.json",
        "successor.json",
    }
    assert (root.stat().st_mode & 0o777) == 0o700
    assert all(
        (path.stat().st_mode & 0o777) == 0o600
        for path in root.iterdir()
    )
    bundle = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    tasks = bundle["plan"]["tasks"]
    assert len(tasks) == 28
    assert tasks[0]["cell"] == {
        "profile": "full",
        "domain": "FactorShockStatePolicyRZDT1",
        "line": "lodo",
        "seed": 2,
        "d": 50,
        "N": 20,
        "n0": 10,
    }
    assert tasks[0]["task_id"] == materialized["first_task_id"]
    assert tasks[0]["task_digest"] == materialized["first_task_digest"]
    assert len(tasks[0]["run_one_task"]["args"]) == 438
    assert all(len(task["run_one_task"]["args"]) == 438 for task in tasks)

    marker = json.loads(
        (root / "successor.json").read_text(encoding="utf-8")
    )
    assert marker["status"] == STATUS
    assert marker["current_status"] == "NOT_CURRENT"
    assert marker["authorization_status"] == "NOT_AUTHORIZED"
    assert marker["attempt_status"] == "NOT_PREPARED"
    assert marker["execution_status"] == "NOT_EXECUTED"
    assert marker["future_attempt_binding"][
        "future_attempt_absent_at_final_precommit_observation"
    ] is True
    assert marker["integrity"]["recursive_successor_digest"] == (
        materialized["recursive_successor_digest"]
    )

    committed_before = _tree_observation(root)
    assert runner.main(["verify", *_real_verify_cli(case, materialized)]) == 0
    verified = capsys.readouterr()
    assert verified.out == ""
    assert "status=" + VERIFY_STATUS in verified.err
    assert _tree_observation(root) == committed_before
    assert {
        name: _tree_observation(case[name]) for name in source_before
    } == source_before

    # All three independently retained output anchors are required by verify.
    for name in VERIFY_DIGEST_NAMES:
        argv = ["verify", *_real_verify_cli(case, materialized)]
        option = "--" + name.replace("_", "-")
        argv[argv.index(option) + 1] = _digest("wrong-" + name)
        _verify_rejects(case, materialized, capsys, argv=argv)
        assert _tree_observation(root) == committed_before

    bundle_path = root / "bundle.json"
    bundle_raw = bundle_path.read_bytes()
    bundle_path.write_bytes(bundle_raw + b"\n")
    bundle_path.chmod(0o600)
    _verify_rejects(case, materialized, capsys)
    bundle_path.write_bytes(bundle_raw)
    bundle_path.chmod(0o600)

    external_link = case["state_home"] / "recursive-bundle-hardlink"
    os.link(bundle_path, external_link)
    try:
        _verify_rejects(case, materialized, capsys)
    finally:
        external_link.unlink()

    bundle_path.unlink()
    os.mkfifo(bundle_path, mode=0o600)
    try:
        _verify_rejects(case, materialized, capsys)
    finally:
        bundle_path.unlink()
        bundle_path.write_bytes(bundle_raw)
        bundle_path.chmod(0o600)

    assert _tree_observation(root) == committed_before
    assert runner.main([
        "materialize",
        *_real_common_cli(case),
        "--confirm-recursive-successor-materialization",
    ]) == 2
    no_clobber = capsys.readouterr()
    assert no_clobber.out == ""
    assert "refusing to overwrite" in no_clobber.err
    assert _tree_observation(root) == committed_before

    # The commit marker is the last write.  A future attempt created after
    # staging but immediately before the marker link is detected by the final
    # point-in-time observation.  The marker remains absent and the incomplete
    # recursive-successor root cannot be reused.
    real_write = core._write_new_bytes
    failed_id = "recursive-commit-last-failure"
    failed_root = prefix / failed_id
    failed_future = future_prefix / failed_id

    def fail_final_marker(path, raw, *, prepublish=None):
        if Path(path).name == "successor.json":
            assert prepublish is not None

            def create_future_then_observe():
                failed_future.mkdir(mode=0o700)
                prepublish()

            return real_write(
                path, raw, prepublish=create_future_then_observe
            )
        return real_write(path, raw, prepublish=prepublish)

    monkeypatch.setattr(core, "_write_new_bytes", fail_final_marker)
    assert runner.main([
        "materialize",
        *_real_common_cli(
            case,
            recursive_successor_root=failed_root,
            recursive_successor_id=failed_id,
            future_attempt_root=failed_future,
        ),
        "--confirm-recursive-successor-materialization",
    ]) == 2
    failed = capsys.readouterr()
    assert failed.out == ""
    assert "must be absent" in failed.err
    assert failed_future.is_dir()
    failed_future.rmdir()
    assert failed_root.is_dir()
    assert {path.name for path in failed_root.iterdir()} == {
        "recursive_successor_contract.json",
        "bundle.json",
    }
    assert not (failed_root / "successor.json").exists()
    assert not failed_future.exists()
    monkeypatch.setattr(core, "_write_new_bytes", real_write)
    assert runner.main([
        "materialize",
        *_real_common_cli(
            case,
            recursive_successor_root=failed_root,
            recursive_successor_id=failed_id,
            future_attempt_root=failed_future,
        ),
        "--confirm-recursive-successor-materialization",
    ]) == 2
    reused = capsys.readouterr()
    assert reused.out == ""
    assert "refusing to overwrite" in reused.err
    assert not (failed_root / "successor.json").exists()


def test_reader_rejects_fifo_hardlink_and_lstat_open_generation_swap(
    monkeypatch,
):
    from performance import structural_hypothesis_recursive_successor_materializer as core

    root = Path(tempfile.mkdtemp(
        prefix="kgop-recursive-successor-reader.", dir="/tmp"
    ))
    root.chmod(0o700)
    try:
        fifo = root / "fifo.json"
        os.mkfifo(fifo, mode=0o600)
        with pytest.raises(core.RecursiveSuccessorMaterializationError):
            core._read_regular(fifo, "fifo", exact_mode=0o600)

        hardlinked = root / "hardlinked.json"
        hardlinked.write_bytes(b"{}\n")
        hardlinked.chmod(0o600)
        alias = root / "hardlinked.alias"
        os.link(hardlinked, alias)
        with pytest.raises(core.RecursiveSuccessorMaterializationError):
            core._read_regular(hardlinked, "hardlink", exact_mode=0o600)

        observed = root / "observed.json"
        replacement = root / "replacement.json"
        observed.write_bytes(b'{"same":true}\n')
        replacement.write_bytes(b'{"same":true}\n')
        observed.chmod(0o600)
        replacement.chmod(0o600)
        real_open = os.open
        swapped = False

        def swap_before_open(path, flags, *args, **kwargs):
            nonlocal swapped
            if Path(path) == observed and not swapped:
                swapped = True
                observed.unlink()
                replacement.replace(observed)
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(core.os, "open", swap_before_open)
        with pytest.raises(core.RecursiveSuccessorMaterializationError):
            core._read_regular(
                observed, "generation swap", exact_mode=0o600
            )
        assert swapped is True
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_future_attempt_policy_rejects_existing_nested_and_symlink_paths(
    tmp_path, monkeypatch
):
    from performance import (
        structural_hypothesis_recursive_successor_materializer as core,
    )

    state_home = tmp_path / "state"
    runtime_prefix = (
        state_home / "kg-op" / "structural-hypothesis-execution" / "v1"
    )
    runtime_prefix.mkdir(parents=True, mode=0o700)
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    candidate = runtime_prefix / "recursive-seed2"
    observed, checkpoint = core._validate_future_attempt(
        candidate, "recursive-seed2", require_absent=True
    )
    assert observed == candidate
    assert checkpoint == candidate / "checkpoints"

    candidate.mkdir(mode=0o700)
    with pytest.raises(core.RecursiveSuccessorMaterializationError):
        core._validate_future_attempt(
            candidate, "recursive-seed2", require_absent=True
        )
    candidate.rmdir()

    nested = runtime_prefix / "nested" / "recursive-seed2"
    with pytest.raises(core.RecursiveSuccessorMaterializationError):
        core._validate_future_attempt(
            nested, "recursive-seed2", require_absent=True
        )

    target = tmp_path / "outside"
    target.mkdir(mode=0o700)
    candidate.symlink_to(target, target_is_directory=True)
    with pytest.raises(core.RecursiveSuccessorMaterializationError):
        core._validate_future_attempt(
            candidate, "recursive-seed2", require_absent=True
        )


def test_core_surface_contains_no_benchmark_execution_primitive():
    source = CORE.read_text(encoding="utf-8")
    for forbidden in (
        "benchmark_lodo_meta_prior",
        "run_one(",
        "import subprocess",
        "import requests",
        "import socket",
        "_load_real_executor",
        "execute_single_task_attempt(",
        "prepare_single_task_attempt(",
    ):
        assert forbidden not in source
