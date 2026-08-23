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

import runners.run_structural_hypothesis_adopted_successor_materializer as runner  # noqa: E402
from test_structural_hypothesis_report_adoption import (  # noqa: E402
    _core_adopt,
    self_contained_publication,
)


RUNNER = (
    ROOT
    / "runners/run_structural_hypothesis_adopted_successor_materializer.py"
)
CORE = (
    ROOT
    / "performance/structural_hypothesis_adopted_successor_materializer.py"
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
    / "performance/manifests/structural_hypothesis_reingestion_publisher_v1.json"
)
MATERIALIZER_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_task_materializer_v1.json"
)
BASE_MANIFEST = ROOT / "performance/manifests/v18b_exactkg_mcdiag.json"
ASSET_ROOT = ROOT / "performance/task_inputs/structural_hypothesis_materializer_v1"

STATUS = "SUCCESSOR_MATERIALIZED_FROM_VERIFIED_ADOPTION_NOT_AUTHORIZED"
VERIFIED_STATUS = "VERIFIED_" + STATUS
FIRST_PROJECTION_KEYS = ("profile", "domain", "line", "seed", "d", "N", "n0")


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
    source_attempt = tmp_path / "source-attempt"
    assets = tmp_path / "assets"
    runtime = tmp_path / "runtime"
    for directory in (publication, adoption, source_attempt, assets, runtime):
        directory.mkdir(parents=True)
    evidence = tmp_path / "base.csv"
    evidence.write_text("track,run_id\npriors,fake\n", encoding="utf-8")
    return {
        "publication": publication.resolve(),
        "adoption_contract": _json_file(tmp_path / "adoption-contract.json"),
        "adoption": adoption.resolve(),
        "adoption_id": "adoption-unit-0001",
        "successor_contract": _json_file(tmp_path / "successor-contract.json"),
        "successor": (tmp_path / "successor-unit-0001").resolve(),
        "successor_id": "successor-unit-0001",
        "evidence": evidence.resolve(),
        "source_attempt": source_attempt.resolve(),
        "hypothesis_contract": _json_file(tmp_path / "hypothesis-contract.json"),
        "executor_contract": _json_file(tmp_path / "executor-contract.json"),
        "runtime_contract": _json_file(tmp_path / "runtime-contract.json"),
        "publisher_contract": _json_file(tmp_path / "publisher-contract.json"),
        "materializer_contract": _json_file(
            tmp_path / "materializer-contract.json"
        ),
        "base_manifest": _json_file(tmp_path / "base-manifest.json"),
        "asset_root": assets.resolve(),
        "future_attempt": (
            runtime / "successor-unit-0001"
        ).resolve(),
    }


def _common_args(case):
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
        "--base-manifest", str(case["base_manifest"]),
        "--asset-root", str(case["asset_root"]),
        "--future-attempt-root", str(case["future_attempt"]),
        "--expected-adoption-digest", _digest("1"),
        "--expected-pending-evidence-digest", _digest("2"),
        "--expected-first-pending-projection-digest", _digest("3"),
    ]


def _materialize_args(case):
    return [
        "materialize",
        *_common_args(case),
        "--confirm-successor-materialization",
    ]


def _verify_args(case):
    return [
        "verify",
        *_common_args(case),
        "--expected-successor-digest", _digest("4"),
        "--expected-bundle-digest", _digest("5"),
        "--expected-plan-digest", _digest("6"),
    ]


def _payload(case, *, verified=False):
    return {
        "status": VERIFIED_STATUS if verified else STATUS,
        "successor_root": str(case["successor"]),
        "successor_digest": _digest("4"),
        "adoption_digest": _digest("1"),
        "pending_evidence_digest": _digest("2"),
        "first_pending_projection_digest": _digest("3"),
        "bundle_digest": _digest("5"),
        "plan_digest": _digest("6"),
        "first_task_id": "task:unit-first-task",
        "first_task_digest": _digest("7"),
        "task_count": 29,
        "future_attempt_root": str(case["future_attempt"]),
        "checkpoint_root": str(case["future_attempt"] / "checkpoints"),
        "authorization_status": "NOT_AUTHORIZED",
    }


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
        case["base_manifest"],
        case["asset_root"],
        case["future_attempt"],
    )


def _expected_common_kwargs(case):
    return {
        "adoption_id": case["adoption_id"],
        "successor_id": case["successor_id"],
        "expected_adoption_digest": _digest("1"),
        "expected_pending_evidence_digest": _digest("2"),
        "expected_first_pending_projection_digest": _digest("3"),
    }


def test_runner_exact_calls_canonical_stdout_and_read_only_verify(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)
    calls = []

    def materialize(*args, **kwargs):
        calls.append(("materialize", args, kwargs))
        return {**_payload(case), "ignored_extra": "not-an-output-field"}

    def verify(*args, **kwargs):
        calls.append(("verify", args, kwargs))
        return _payload(case, verified=True)

    monkeypatch.setattr(
        runner,
        "_load_successor_core",
        lambda: SimpleNamespace(
            materialize_adopted_successor=materialize,
            verify_adopted_successor=verify,
        ),
    )
    assert runner.main(_materialize_args(case)) == 0
    made = capsys.readouterr()
    expected = _payload(case)
    assert made.out == json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ) + "\n"
    assert "action=materialize" in made.err
    assert "first_task_id=task:unit-first-task" in made.err
    assert calls == [
        (
            "materialize",
            _expected_positional(case),
            _expected_common_kwargs(case),
        )
    ]

    case["successor"].mkdir(mode=0o700)
    assert runner.main(_verify_args(case)) == 0
    checked = capsys.readouterr()
    assert checked.out == ""
    assert "action=verify" in checked.err
    assert "status=" + VERIFIED_STATUS in checked.err
    assert calls[-1] == (
        "verify",
        _expected_positional(case),
        {
            "expected_successor_digest": _digest("4"),
            "expected_bundle_digest": _digest("5"),
            "expected_plan_digest": _digest("6"),
            **_expected_common_kwargs(case),
        },
    )


def test_runner_surface_starts_without_execution_imports():
    source = RUNNER.read_text(encoding="utf-8")
    assert "import numpy" not in source
    assert "import subprocess" not in source
    assert "import socket" not in source
    assert "import requests" not in source
    choices = set(runner._parser()._subparsers._group_actions[0].choices)
    assert choices == {"materialize", "verify"}

    script = r'''
import builtins

blocked = {
    "numpy",
    "performance.benchmark_lodo_meta_prior",
    "performance.benchmark_quality",
}
original_import = builtins.__import__

def bomb_import(name, *args, **kwargs):
    if name in blocked:
        raise AssertionError("successor startup crossed execution boundary")
    return original_import(name, *args, **kwargs)

builtins.__import__ = bomb_import
from runners.run_structural_hypothesis_adopted_successor_materializer import _parser
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


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda case, args: case["successor"].mkdir(), "refusing to overwrite"),
        (
            lambda case, args: case["future_attempt"].mkdir(),
            "future attempt root must be absent",
        ),
        (
            lambda case, args: args.__setitem__(
                args.index("--adoption-id") + 1, "../bad"
            ),
            "non-path local mechanics label",
        ),
        (
            lambda case, args: args.__setitem__(
                args.index("--expected-adoption-digest") + 1, "sha256:BAD"
            ),
            "lowercase sha256 digest",
        ),
        (
            lambda case, args: args.__setitem__(
                args.index("--base-evidence-csv") + 1, "relative.csv"
            ),
            "absolute path",
        ),
    ],
)
def test_runner_fails_closed_before_core(tmp_path, monkeypatch, capsys, mutate, needle):
    case = _unit_case(tmp_path)
    args = _materialize_args(case)
    mutate(case, args)
    called = []
    monkeypatch.setattr(
        runner,
        "_load_successor_core",
        lambda: called.append(True),
    )
    assert runner.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert needle in captured.err
    assert called == []


def test_runner_core_rejection_and_bad_result_never_publish_stdout(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)

    def reject(*_args, **_kwargs):
        raise ValueError("fake exact-chain rejection")

    monkeypatch.setattr(
        runner,
        "_load_successor_core",
        lambda: SimpleNamespace(materialize_adopted_successor=reject),
    )
    assert runner.main(_materialize_args(case)) == 2
    rejected = capsys.readouterr()
    assert rejected.out == ""
    assert "fake exact-chain rejection" in rejected.err
    assert "Traceback" not in rejected.err

    monkeypatch.setattr(
        runner,
        "_load_successor_core",
        lambda: SimpleNamespace(
            materialize_adopted_successor=lambda *_args, **_kwargs: {
                **_payload(case),
                "first_task_digest": "invalid",
            }
        ),
    )
    assert runner.main(_materialize_args(case)) == 2
    malformed = capsys.readouterr()
    assert malformed.out == ""
    assert "result first_task_digest" in malformed.err


def _projection_cell(cell):
    return {
        "profile": cell["profile"],
        "domain": cell["domain"],
        "line": "lodo",
        "seed": cell["seed"],
        "d": cell["d"],
        "N": cell["N"],
        "n0": cell["n0"],
    }


def _projection(report):
    return _projection_cell(report["pending_evidence"][0])


def _real_common_args(context, adopted, successor, successor_id, future_attempt):
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
        "--successor-id", successor_id,
        "--base-evidence-csv", str(context["evidence"]),
        "--source-attempt-root", str(context["attempt"]),
        "--hypothesis-contract", str(HYPOTHESIS_CONTRACT),
        "--executor-contract", str(EXECUTOR_CONTRACT),
        "--runtime-contract", str(RUNTIME_CONTRACT),
        "--publisher-contract", str(PUBLISHER_CONTRACT),
        "--materializer-contract", str(MATERIALIZER_CONTRACT),
        "--base-manifest", str(BASE_MANIFEST),
        "--asset-root", str(ASSET_ROOT),
        "--future-attempt-root", str(future_attempt),
        "--expected-adoption-digest", adopted["adoption_digest"],
        "--expected-pending-evidence-digest",
        _canonical_digest(report["pending_evidence"]),
        "--expected-first-pending-projection-digest",
        _canonical_digest(_projection(report)),
    ]


def _run(args, context):
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT),
            "XDG_STATE_HOME": str(context["state_home"]),
            "SCOLHKG_OFFLINE": "1",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _tree_observation(root):
    return {
        str(path.relative_to(root)): (
            "directory" if path.is_dir() else path.read_bytes(),
            path.stat().st_mode,
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
    }


def test_fake_only_full_chain_exact_29_successor_and_negative_anchors(
    self_contained_publication, monkeypatch
):
    from performance import structural_hypothesis_report_adoption as adoption_core
    from performance import structural_hypothesis_adopted_successor_materializer as successor_core

    context = self_contained_publication
    adopted = _core_adopt(context, adoption_core)
    report = json.loads(
        (context["adoption"] / "publication/output_report.json").read_text(
            encoding="utf-8"
        )
    )
    projection = _projection(report)
    assert tuple(projection) == FIRST_PROJECTION_KEYS
    assert len(report["pending_evidence"]) == 29
    assert projection["seed"] == 1
    assert [
        (cell["domain"], cell["seed"])
        for cell in report["pending_evidence"]
    ] == [
        *(('FactorShockStatePolicyRZDT1', seed) for seed in range(1, 10)),
        *(('InventorySupplyChain', seed) for seed in range(10)),
        *(('QueueResourceControl', seed) for seed in range(10)),
    ]

    prefix = (
        context["state_home"]
        / "kg-op/structural-hypothesis-adopted-successor/v1"
    )
    for index, option in enumerate(
        (
            "--expected-adoption-digest",
            "--expected-pending-evidence-digest",
            "--expected-first-pending-projection-digest",
        ),
        start=1,
    ):
        bad_id = f"successor-bad-anchor-{index:04d}"
        bad_root = prefix / bad_id
        bad_future = (
            context["state_home"]
            / "kg-op/structural-hypothesis-execution/v1"
            / bad_id
        )
        bad_args = [
            "materialize",
            *_real_common_args(
                context, adopted, bad_root, bad_id, bad_future
            ),
            "--confirm-successor-materialization",
        ]
        bad_args[bad_args.index(option) + 1] = _digest("f")
        rejected = _run(bad_args, context)
        assert rejected.returncode == 2
        assert rejected.stdout == ""
        assert "Traceback" not in rejected.stderr
        assert not bad_root.exists()
        assert not bad_future.exists()

    nested_id = "successor-nested-future-0001"
    nested_root = prefix / nested_id
    nested_future = (
        context["state_home"]
        / "kg-op/structural-hypothesis-execution/v1/nested"
        / nested_id
    )
    nested = _run(
        [
            "materialize",
            *_real_common_args(
                context, adopted, nested_root, nested_id, nested_future
            ),
            "--confirm-successor-materialization",
        ],
        context,
    )
    assert nested.returncode == 2
    assert nested.stdout == ""
    assert "direct runtime-prefix child" in nested.stderr
    assert not nested_root.exists()
    assert not nested_future.exists()

    wrong_name_id = "successor-wrong-future-name-0001"
    wrong_name_root = prefix / wrong_name_id
    wrong_name_future = (
        context["state_home"]
        / "kg-op/structural-hypothesis-execution/v1/not-the-successor-id"
    )
    wrong_name = _run(
        [
            "materialize",
            *_real_common_args(
                context,
                adopted,
                wrong_name_root,
                wrong_name_id,
                wrong_name_future,
            ),
            "--confirm-successor-materialization",
        ],
        context,
    )
    assert wrong_name.returncode == 2
    assert wrong_name.stdout == ""
    assert "basename must equal" in wrong_name.stderr
    assert not wrong_name_root.exists()
    assert not wrong_name_future.exists()

    changing_id = "successor-changing-generation-0001"
    changing_root = prefix / changing_id
    changing_future = (
        context["state_home"]
        / "kg-op/structural-hypothesis-execution/v1"
        / changing_id
    )
    original_materialize = successor_core._materialize_bundle
    generations = []

    def change_second_generation(*args, **kwargs):
        bundle, pending = original_materialize(*args, **kwargs)
        generations.append(bundle["integrity"]["bundle_digest"])
        if len(generations) == 2:
            bundle = json.loads(json.dumps(bundle))
            bundle["bundle_id"] += "-changed-generation"
        return bundle, pending

    with monkeypatch.context() as local_patch:
        local_patch.setattr(
            successor_core, "_materialize_bundle", change_second_generation
        )
        with pytest.raises(
            successor_core.AdoptedSuccessorMaterializationError,
            match="rebuilt bundle changed",
        ):
            successor_core.materialize_adopted_successor(
                context["publication"],
                ADOPTION_CONTRACT,
                context["adoption"],
                SUCCESSOR_CONTRACT,
                changing_root,
                context["evidence"],
                context["attempt"],
                HYPOTHESIS_CONTRACT,
                EXECUTOR_CONTRACT,
                RUNTIME_CONTRACT,
                PUBLISHER_CONTRACT,
                MATERIALIZER_CONTRACT,
                BASE_MANIFEST,
                ASSET_ROOT,
                changing_future,
                adoption_id=context["adoption_id"],
                successor_id=changing_id,
                expected_adoption_digest=adopted["adoption_digest"],
                expected_pending_evidence_digest=_canonical_digest(
                    report["pending_evidence"]
                ),
                expected_first_pending_projection_digest=(
                    _canonical_digest(projection)
                ),
            )
    assert generations[0] == generations[1]
    assert changing_root.is_dir()
    assert not (changing_root / "successor.json").exists()
    assert not changing_future.exists()

    successor_id = "successor-fake-seed1-0001"
    successor = prefix / successor_id
    future = (
        context["state_home"]
        / "kg-op/structural-hypothesis-execution/v1"
        / successor_id
    )
    common = _real_common_args(
        context, adopted, successor, successor_id, future
    )
    made = _run(
        ["materialize", *common, "--confirm-successor-materialization"],
        context,
    )
    assert made.returncode == 0, made.stderr
    summary = json.loads(made.stdout)
    assert set(summary) == {
        "status",
        "successor_root",
        "successor_digest",
        "adoption_digest",
        "pending_evidence_digest",
        "first_pending_projection_digest",
        "bundle_digest",
        "plan_digest",
        "first_task_id",
        "first_task_digest",
        "task_count",
        "future_attempt_root",
        "checkpoint_root",
        "authorization_status",
    }
    assert summary["status"] == STATUS
    assert summary["authorization_status"] == "NOT_AUTHORIZED"
    assert summary["task_count"] == 29
    assert summary["adoption_digest"] == adopted["adoption_digest"]
    assert summary["pending_evidence_digest"] == _canonical_digest(
        report["pending_evidence"]
    )
    assert summary["first_pending_projection_digest"] == _canonical_digest(
        projection
    )
    assert made.stdout == json.dumps(
        summary, sort_keys=True, separators=(",", ":")
    ) + "\n"
    assert "action=materialize" in made.stderr

    assert successor.is_dir()
    assert oct(successor.stat().st_mode & 0o777) == "0o700"
    assert {path.name for path in successor.iterdir()} == {
        "successor_contract.json",
        "bundle.json",
        "successor.json",
    }
    assert all(
        oct(path.stat().st_mode & 0o777) == "0o600"
        for path in successor.iterdir()
    )
    bundle = json.loads((successor / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["status"] == "MATERIALIZED_NOT_AUTHORIZED"
    assert bundle["task_count"] == 29
    assert bundle["plan"]["status"] == "READY_FOR_AUTHORIZATION"
    assert len(bundle["plan"]["tasks"]) == 29
    assert [
        task["cell"] for task in bundle["plan"]["tasks"]
    ] == [_projection_cell(cell) for cell in report["pending_evidence"]]
    first_task = bundle["plan"]["tasks"][0]
    assert first_task["cell"] == projection
    assert first_task["task_id"] == summary["first_task_id"]
    assert first_task["task_digest"] == summary["first_task_digest"]
    assert len(first_task["run_one_task"]["args"]) == 438
    assert first_task["run_one_task"]["seed"] == 1
    assert not (successor / "current.json").exists()
    assert not future.exists()
    assert not (future / "checkpoints").exists()

    before_repeat = _tree_observation(successor)
    repeated = _run(
        ["materialize", *common, "--confirm-successor-materialization"],
        context,
    )
    assert repeated.returncode == 2
    assert repeated.stdout == ""
    assert "overwrite" in repeated.stderr or "already exists" in repeated.stderr
    assert before_repeat == _tree_observation(successor)

    verify_args = [
        "verify",
        *common,
        "--expected-successor-digest", summary["successor_digest"],
        "--expected-bundle-digest", summary["bundle_digest"],
        "--expected-plan-digest", summary["plan_digest"],
    ]
    before_verify = _tree_observation(successor)
    verified = _run(verify_args, context)
    assert verified.returncode == 0, verified.stderr
    assert verified.stdout == ""
    assert "status=" + VERIFIED_STATUS in verified.stderr
    assert before_verify == _tree_observation(successor)

    captured_adoption = successor_core._capture_adoption(context["adoption"])
    captured_adoption["publication_values"]["output_report"] = json.loads(
        json.dumps(captured_adoption["publication_values"]["output_report"])
    )
    captured_adoption["publication_values"]["output_report"]["stop_reason"] = (
        "TAMPERED"
    )
    with pytest.raises(
        successor_core.AdoptedSuccessorMaterializationError,
        match="do not exactly replay",
    ):
        successor_core._replay_report(captured_adoption)

    adoption_marker = context["adoption"] / "adoption.json"
    adoption_marker_raw = adoption_marker.read_bytes()
    adoption_marker.write_bytes(adoption_marker_raw + b"\n")
    noncanonical_adoption = _run(verify_args, context)
    assert noncanonical_adoption.returncode == 2
    assert noncanonical_adoption.stdout == ""
    assert "canonical" in noncanonical_adoption.stderr
    adoption_marker.write_bytes(adoption_marker_raw)

    unexpected = successor / "current.json"
    unexpected.write_text("{}\n", encoding="utf-8")
    unexpected.chmod(0o600)
    extra = _run(verify_args, context)
    assert extra.returncode == 2
    assert extra.stdout == ""
    assert "unexpected artifacts" in extra.stderr
    unexpected.unlink()

    bundle_path = successor / "bundle.json"
    bundle_raw = bundle_path.read_bytes()
    hardlink_source = context["state_home"] / "successor-bundle-hardlink"
    hardlink_source.write_bytes(bundle_raw)
    hardlink_source.chmod(0o600)
    bundle_path.unlink()
    os.link(hardlink_source, bundle_path)
    hardlinked = _run(verify_args, context)
    assert hardlinked.returncode == 2
    assert hardlinked.stdout == ""
    assert (
        "link count" in hardlinked.stderr
        or "regular file" in hardlinked.stderr
    )
    bundle_path.unlink()
    bundle_path.write_bytes(bundle_raw)
    bundle_path.chmod(0o600)
    hardlink_source.unlink()

    marker_path = successor / "successor.json"
    marker_raw = marker_path.read_bytes()
    marker_path.write_bytes(marker_raw + b"\n")
    noncanonical_marker = _run(verify_args, context)
    assert noncanonical_marker.returncode == 2
    assert noncanonical_marker.stdout == ""
    assert "canonical" in noncanonical_marker.stderr
    marker_path.write_bytes(marker_raw)
    marker_path.chmod(0o600)
    before_verify = _tree_observation(successor)

    for option in (
        "--expected-successor-digest",
        "--expected-bundle-digest",
        "--expected-plan-digest",
    ):
        wrong = list(verify_args)
        wrong[wrong.index(option) + 1] = _digest("e")
        rejected = _run(wrong, context)
        assert rejected.returncode == 2
        assert rejected.stdout == ""
        assert "Traceback" not in rejected.stderr
        assert before_verify == _tree_observation(successor)

    bundle_path.write_bytes(bundle_path.read_bytes() + b"\n")
    tampered_before = _tree_observation(successor)
    tampered = _run(verify_args, context)
    assert tampered.returncode == 2
    assert tampered.stdout == ""
    assert "Traceback" not in tampered.stderr
    assert tampered_before == _tree_observation(successor)


def test_core_surface_has_no_executor_operation_or_ambient_current():
    source = CORE.read_text(encoding="utf-8")
    assert "def run_one" not in source
    assert "def authorize" not in source
    assert "def prepare" not in source
    assert "current.json" not in source
    assert "subprocess" not in source
    assert "requests" not in source
    assert "socket" not in source
