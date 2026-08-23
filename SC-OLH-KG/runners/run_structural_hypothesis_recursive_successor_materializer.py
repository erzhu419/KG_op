"""Materialize or verify a successor from one recursive report advance.

This offline CLI only publishes a fresh, not-authorized successor capsule.
It never imports the benchmark, invokes a native execution callable, prepares
an attempt, executes a task, or changes a current pointer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

DEFAULT_ADOPTION_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_report_adoption_v1.json"
)
DEFAULT_SOURCE_SUCCESSOR_CONTRACT = (
    ROOT
    / "performance/manifests/"
    "structural_hypothesis_adopted_successor_materializer_v1.json"
)
DEFAULT_HYPOTHESIS_CONTRACT = (
    ROOT / "performance/manifests/structural_hypothesis_loop_v1.json"
)
DEFAULT_EXECUTOR_CONTRACT = (
    ROOT / "performance/manifests/structural_hypothesis_executor_v1.json"
)
DEFAULT_RUNTIME_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_single_task_runtime_v1.json"
)
DEFAULT_PUBLISHER_CONTRACT = (
    ROOT
    / "performance/manifests/"
    "structural_hypothesis_reingestion_publisher_v1.json"
)
DEFAULT_MATERIALIZER_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_task_materializer_v1.json"
)
DEFAULT_BRIDGE_CONTRACT = (
    ROOT
    / "performance/manifests/"
    "structural_hypothesis_successor_bound_single_task_v1.json"
)
DEFAULT_BASE_MANIFEST = (
    ROOT / "performance/manifests/v18b_exactkg_mcdiag.json"
)
DEFAULT_ASSET_ROOT = (
    ROOT / "performance/task_inputs/structural_hypothesis_materializer_v1"
)
DEFAULT_ADVANCE_CONTRACT = (
    ROOT
    / "performance/manifests/"
    "structural_hypothesis_recursive_report_advance_v1.json"
)
DEFAULT_RECURSIVE_SUCCESSOR_CONTRACT = (
    ROOT
    / "performance/manifests/"
    "structural_hypothesis_recursive_successor_materializer_v1.json"
)

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_TASK_ID_PATTERN = re.compile(r"task:[0-9a-f]{24}\Z")
_STATUS = (
    "RECURSIVE_SUCCESSOR_MATERIALIZED_FROM_VERIFIED_ADVANCE_"
    "NOT_AUTHORIZED"
)
_VERIFIED_STATUS = "VERIFIED_" + _STATUS

_COMMON_DIGEST_NAMES = (
    "expected_adoption_digest",
    "expected_source_pending_evidence_digest",
    "expected_source_first_pending_projection_digest",
    "expected_source_successor_digest",
    "expected_source_bundle_digest",
    "expected_source_plan_digest",
    "expected_completed_task_digest",
    "expected_source_provenance_binding_digest",
    "expected_source_authorization_digest",
    "expected_source_attempt_digest",
    "expected_source_execution_receipt_digest",
    "expected_source_execution_journal_head_digest",
    "expected_advance_digest",
    "expected_advance_reingestion_digest",
    "expected_advance_output_report_body_digest",
    "expected_advance_output_audit_head",
    "expected_advance_output_evidence_digest",
    "expected_next_pending_evidence_digest",
    "expected_next_first_pending_projection_digest",
)
_VERIFY_DIGEST_NAMES = (
    "expected_recursive_successor_digest",
    "expected_next_bundle_digest",
    "expected_next_plan_digest",
)
_RESULT_KEYS = (
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


def _load_recursive_successor_core():
    from performance import (
        structural_hypothesis_recursive_successor_materializer,
    )

    return structural_hypothesis_recursive_successor_materializer


def _require_absolute(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")


def _require_source_directory(path: Path, label: str) -> None:
    _require_absolute(path, label)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be an existing non-symlink directory")


def _require_source_file(path: Path, label: str) -> None:
    _require_absolute(path, label)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an existing non-symlink file")


def _require_digest(value: Any, label: str) -> str:
    if type(value) is not str or not _DIGEST_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase sha256 digest")
    return value


def _validate_paths(args: argparse.Namespace) -> None:
    for label, path in (
        ("--publication-root", args.publication_root),
        ("--adoption-root", args.adoption_root),
        ("--source-successor-root", args.source_successor_root),
        ("--source-attempt-root", args.source_attempt_root),
        ("--asset-root", args.asset_root),
        ("--completed-attempt-root", args.completed_attempt_root),
        ("--advance-root", args.advance_root),
    ):
        _require_source_directory(path, label)
    for label, path in (
        ("--adoption-contract", args.adoption_contract),
        ("--source-successor-contract", args.source_successor_contract),
        ("--base-evidence-csv", args.base_evidence_csv),
        ("--hypothesis-contract", args.hypothesis_contract),
        ("--executor-contract", args.executor_contract),
        ("--runtime-contract", args.runtime_contract),
        ("--publisher-contract", args.publisher_contract),
        ("--materializer-contract", args.materializer_contract),
        ("--bridge-contract", args.bridge_contract),
        ("--base-manifest", args.base_manifest),
        ("--advance-contract", args.advance_contract),
        (
            "--recursive-successor-contract",
            args.recursive_successor_contract,
        ),
    ):
        _require_source_file(path, label)

    _require_absolute(
        args.recursive_successor_root, "--recursive-successor-root"
    )
    _require_absolute(args.future_attempt_root, "--future-attempt-root")
    if args.action == "materialize":
        if (
            args.recursive_successor_root.exists()
            or args.recursive_successor_root.is_symlink()
        ):
            raise ValueError(
                "fresh recursive successor root is required; refusing to "
                "overwrite"
            )
        if (
            args.future_attempt_root.exists()
            or args.future_attempt_root.is_symlink()
        ):
            raise ValueError(
                "future attempt root must be absent during materialization"
            )
    elif (
        args.recursive_successor_root.is_symlink()
        or not args.recursive_successor_root.is_dir()
    ):
        raise ValueError(
            "recursive successor root must be an existing non-symlink "
            "directory"
        )


def _validate_explicit_bindings(args: argparse.Namespace) -> None:
    bindings = (
        ("--adoption-id", args.adoption_id, args.adoption_root),
        (
            "--source-successor-id",
            args.source_successor_id,
            args.source_successor_root,
        ),
        ("--advance-id", args.advance_id, args.advance_root),
        (
            "--recursive-successor-id",
            args.recursive_successor_id,
            args.recursive_successor_root,
        ),
    )
    for option, value, root in bindings:
        if type(value) is not str or not _ID_PATTERN.fullmatch(value):
            raise ValueError(f"{option} must be a non-path local label")
        if root.name != value:
            raise ValueError(f"{option} must equal its root basename")
    if args.future_attempt_root.name != args.recursive_successor_id:
        raise ValueError(
            "--future-attempt-root basename must equal "
            "--recursive-successor-id"
        )
    if (
        type(args.completed_task_id) is not str
        or not _TASK_ID_PATTERN.fullmatch(args.completed_task_id)
    ):
        raise ValueError(
            "--completed-task-id must be an exact task:<24-hex> identifier"
        )
    names = list(_COMMON_DIGEST_NAMES)
    if args.action == "verify":
        names.extend(_VERIFY_DIGEST_NAMES)
    for name in names:
        _require_digest(
            getattr(args, name), "--" + name.replace("_", "-")
        )


def _add_common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--publication-root", type=Path, required=True)
    parser.add_argument(
        "--adoption-contract", type=Path, default=DEFAULT_ADOPTION_CONTRACT
    )
    parser.add_argument("--adoption-root", type=Path, required=True)
    parser.add_argument("--adoption-id", required=True)
    parser.add_argument(
        "--source-successor-contract",
        type=Path,
        default=DEFAULT_SOURCE_SUCCESSOR_CONTRACT,
    )
    parser.add_argument("--source-successor-root", type=Path, required=True)
    parser.add_argument("--source-successor-id", required=True)
    parser.add_argument("--base-evidence-csv", type=Path, required=True)
    parser.add_argument("--source-attempt-root", type=Path, required=True)
    parser.add_argument(
        "--hypothesis-contract",
        type=Path,
        default=DEFAULT_HYPOTHESIS_CONTRACT,
    )
    parser.add_argument(
        "--executor-contract", type=Path, default=DEFAULT_EXECUTOR_CONTRACT
    )
    parser.add_argument(
        "--runtime-contract", type=Path, default=DEFAULT_RUNTIME_CONTRACT
    )
    parser.add_argument(
        "--publisher-contract", type=Path, default=DEFAULT_PUBLISHER_CONTRACT
    )
    parser.add_argument(
        "--materializer-contract",
        type=Path,
        default=DEFAULT_MATERIALIZER_CONTRACT,
    )
    parser.add_argument(
        "--bridge-contract", type=Path, default=DEFAULT_BRIDGE_CONTRACT
    )
    parser.add_argument(
        "--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST
    )
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--completed-attempt-root", type=Path, required=True)
    parser.add_argument(
        "--advance-contract", type=Path, default=DEFAULT_ADVANCE_CONTRACT
    )
    parser.add_argument("--advance-root", type=Path, required=True)
    parser.add_argument("--advance-id", required=True)
    parser.add_argument(
        "--recursive-successor-contract",
        type=Path,
        default=DEFAULT_RECURSIVE_SUCCESSOR_CONTRACT,
    )
    parser.add_argument(
        "--recursive-successor-root", type=Path, required=True
    )
    parser.add_argument("--recursive-successor-id", required=True)
    parser.add_argument("--future-attempt-root", type=Path, required=True)
    parser.add_argument("--completed-task-id", required=True)
    for name in _COMMON_DIGEST_NAMES:
        parser.add_argument("--" + name.replace("_", "-"), required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize or verify one exact not-authorized successor from "
            "a fully verified recursive report advance."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    materialize = subparsers.add_parser(
        "materialize", help="publish one fresh recursive successor capsule"
    )
    _add_common_inputs(materialize)
    materialize.add_argument(
        "--confirm-recursive-successor-materialization",
        action="store_true",
        required=True,
        help=(
            "acknowledge local successor materialization; this grants no "
            "authorization and permits no task execution"
        ),
    )

    verify = subparsers.add_parser(
        "verify", help="verify one recursive successor without writes"
    )
    _add_common_inputs(verify)
    for name in _VERIFY_DIGEST_NAMES:
        verify.add_argument("--" + name.replace("_", "-"), required=True)
    return parser


def _positional(args: argparse.Namespace) -> tuple[Path, ...]:
    return (
        args.publication_root,
        args.adoption_contract,
        args.adoption_root,
        args.source_successor_contract,
        args.source_successor_root,
        args.base_evidence_csv,
        args.source_attempt_root,
        args.hypothesis_contract,
        args.executor_contract,
        args.runtime_contract,
        args.publisher_contract,
        args.materializer_contract,
        args.bridge_contract,
        args.base_manifest,
        args.asset_root,
        args.completed_attempt_root,
        args.advance_contract,
        args.advance_root,
        args.recursive_successor_contract,
        args.recursive_successor_root,
        args.future_attempt_root,
    )


def _common_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "adoption_id": args.adoption_id,
        "source_successor_id": args.source_successor_id,
        "advance_id": args.advance_id,
        "recursive_successor_id": args.recursive_successor_id,
        **{name: getattr(args, name) for name in _COMMON_DIGEST_NAMES[:6]},
        "completed_task_id": args.completed_task_id,
        **{name: getattr(args, name) for name in _COMMON_DIGEST_NAMES[6:]},
    }


def _validated_result(
    payload: Any, args: argparse.Namespace
) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("recursive successor core returned a non-object")
    expected_status = _STATUS if args.action == "materialize" else _VERIFIED_STATUS
    if payload.get("status") != expected_status:
        raise ValueError("recursive successor status differs from frozen V1")
    expected_states = {
        "current_status": "NOT_CURRENT",
        "authorization_status": "NOT_AUTHORIZED",
        "attempt_status": "NOT_PREPARED",
        "execution_status": "NOT_EXECUTED",
    }
    for key, expected in expected_states.items():
        if payload.get(key) != expected:
            raise ValueError(f"recursive successor has invalid {key}")

    string_keys = tuple(key for key in _RESULT_KEYS if key != "task_count")
    for key in string_keys:
        value = payload.get(key)
        if type(value) is not str or not value:
            raise ValueError(f"recursive successor has invalid {key}")
    task_count = payload.get("task_count")
    if type(task_count) is not int or not 1 <= task_count <= 30:
        raise ValueError("recursive successor has invalid task_count")
    for key in (
        "recursive_successor_digest",
        "advance_digest",
        "advance_output_evidence_digest",
        "pending_evidence_digest",
        "first_pending_projection_digest",
        "bundle_digest",
        "plan_digest",
        "first_task_digest",
    ):
        _require_digest(payload[key], f"result {key}")
    if not _TASK_ID_PATTERN.fullmatch(payload["first_task_id"]):
        raise ValueError("recursive successor has invalid first_task_id")
    if Path(payload["recursive_successor_root"]) != (
        args.recursive_successor_root.resolve()
    ):
        raise ValueError("recursive successor result root differs")
    future = args.future_attempt_root.resolve()
    if Path(payload["future_attempt_root"]) != future:
        raise ValueError("recursive successor future attempt root differs")
    if Path(payload["checkpoint_root"]) != future / "checkpoints":
        raise ValueError("recursive successor checkpoint root differs")

    expected_bindings = {
        "advance_digest": args.expected_advance_digest,
        "advance_output_evidence_digest": (
            args.expected_advance_output_evidence_digest
        ),
        "pending_evidence_digest": args.expected_next_pending_evidence_digest,
        "first_pending_projection_digest": (
            args.expected_next_first_pending_projection_digest
        ),
    }
    if args.action == "verify":
        expected_bindings.update({
            "recursive_successor_digest": (
                args.expected_recursive_successor_digest
            ),
            "bundle_digest": args.expected_next_bundle_digest,
            "plan_digest": args.expected_next_plan_digest,
        })
    for key, expected in expected_bindings.items():
        if payload[key] != expected:
            raise ValueError(f"recursive successor result {key} differs")
    return {key: payload[key] for key in _RESULT_KEYS}


def _write_summary(payload: dict[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def _human_summary(payload: dict[str, Any], action: str) -> str:
    return " ".join((
        "structural_hypothesis_recursive_successor_materializer",
        f"action={action}",
        f"status={payload['status']}",
        f"recursive_successor_digest={payload['recursive_successor_digest']}",
        f"first_task_id={payload['first_task_id']}",
        f"task_count={payload['task_count']}",
        f"authorization_status={payload['authorization_status']}",
        f"execution_status={payload['execution_status']}",
    ))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_paths(args)
        _validate_explicit_bindings(args)
        core = _load_recursive_successor_core()
        kwargs = _common_kwargs(args)
        if args.action == "materialize":
            payload = core.materialize_recursive_successor(
                *_positional(args),
                **kwargs,
                confirm_recursive_successor_materialization=True,
            )
        else:
            payload = core.verify_recursive_successor(
                *_positional(args),
                **kwargs,
                **{
                    name: getattr(args, name)
                    for name in _VERIFY_DIGEST_NAMES
                },
            )
        summary = _validated_result(payload, args)
        if args.action == "materialize":
            _write_summary(summary)
        print(_human_summary(summary, args.action), file=sys.stderr)
        return 0
    except (
        ImportError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        print(
            "invalid structural-hypothesis recursive successor: " + str(exc),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
