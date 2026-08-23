"""Advance or verify one immutable local recursive report version.

The CLI accepts only a completed, successor-bound single-task attempt.  It
does not execute a task, choose another task, update a current pointer, or
access a network, shell, scheduler, or credential source.
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

DEFAULT_ADVANCE_CONTRACT = (
    ROOT
    / "performance/manifests/"
    "structural_hypothesis_recursive_report_advance_v1.json"
)
DEFAULT_ADOPTION_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_report_adoption_v1.json"
)
DEFAULT_SUCCESSOR_CONTRACT = (
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

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_TASK_ID_PATTERN = re.compile(r"task:[0-9a-f]{24}\Z")
_STATUS = (
    "ADVANCED_AS_IMMUTABLE_LOCAL_REPORT_VERSION_"
    "NOT_CURRENT_NOT_PLANNED"
)
_VERIFIED_STATUS = "VERIFIED_" + _STATUS


def _load_advance_core():
    from performance import structural_hypothesis_recursive_report_advance

    return structural_hypothesis_recursive_report_advance


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


def _validate_paths(args) -> None:
    for label, path in (
        ("--publication-root", args.publication_root),
        ("--adoption-root", args.adoption_root),
        ("--successor-root", args.successor_root),
        ("--source-attempt-root", args.source_attempt_root),
        ("--asset-root", args.asset_root),
        ("--completed-attempt-root", args.completed_attempt_root),
    ):
        _require_source_directory(path, label)
    for label, path in (
        ("--adoption-contract", args.adoption_contract),
        ("--successor-contract", args.successor_contract),
        ("--base-evidence-csv", args.base_evidence_csv),
        ("--hypothesis-contract", args.hypothesis_contract),
        ("--executor-contract", args.executor_contract),
        ("--runtime-contract", args.runtime_contract),
        ("--publisher-contract", args.publisher_contract),
        ("--materializer-contract", args.materializer_contract),
        ("--bridge-contract", args.bridge_contract),
        ("--base-manifest", args.base_manifest),
        ("--advance-contract", args.advance_contract),
    ):
        _require_source_file(path, label)
    _require_absolute(args.advance_root, "--advance-root")
    if args.action == "advance":
        if args.advance_root.exists() or args.advance_root.is_symlink():
            raise ValueError(
                "fresh advance root is required; refusing to overwrite"
            )
    elif args.advance_root.is_symlink() or not args.advance_root.is_dir():
        raise ValueError(
            "advance root must be an existing non-symlink directory"
        )


def _require_digest(value: str, label: str) -> str:
    if not _DIGEST_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase sha256 digest")
    return value


_COMMON_DIGEST_NAMES = (
    "expected_adoption_digest",
    "expected_pending_evidence_digest",
    "expected_first_pending_projection_digest",
    "expected_successor_digest",
    "expected_bundle_digest",
    "expected_plan_digest",
    "expected_task_digest",
    "expected_provenance_binding_digest",
    "expected_authorization_digest",
    "expected_attempt_digest",
    "expected_execution_receipt_digest",
    "expected_execution_journal_head_digest",
)

_VERIFY_DIGEST_NAMES = (
    "expected_advance_digest",
    "expected_reingestion_digest",
    "expected_output_report_body_digest",
    "expected_output_audit_head",
    "expected_output_evidence_digest",
)


def _validate_explicit_bindings(args) -> None:
    for name in ("advance_id", "adoption_id", "successor_id"):
        if not _ID_PATTERN.fullmatch(getattr(args, name)):
            raise ValueError(
                f"--{name.replace('_', '-')} must be a non-path local label"
            )
    if not _TASK_ID_PATTERN.fullmatch(args.task_id):
        raise ValueError("--task-id must be an exact task:<24-hex> identifier")
    digest_names = list(_COMMON_DIGEST_NAMES)
    if args.action == "verify":
        digest_names.extend(_VERIFY_DIGEST_NAMES)
    for name in digest_names:
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
        "--successor-contract", type=Path, default=DEFAULT_SUCCESSOR_CONTRACT
    )
    parser.add_argument("--successor-root", type=Path, required=True)
    parser.add_argument("--successor-id", required=True)
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
    parser.add_argument("--expected-adoption-digest", required=True)
    parser.add_argument("--expected-pending-evidence-digest", required=True)
    parser.add_argument(
        "--expected-first-pending-projection-digest", required=True
    )
    parser.add_argument("--expected-successor-digest", required=True)
    parser.add_argument("--expected-bundle-digest", required=True)
    parser.add_argument("--expected-plan-digest", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--expected-task-digest", required=True)
    parser.add_argument(
        "--expected-provenance-binding-digest", required=True
    )
    parser.add_argument("--expected-authorization-digest", required=True)
    parser.add_argument("--expected-attempt-digest", required=True)
    parser.add_argument(
        "--expected-execution-receipt-digest", required=True
    )
    parser.add_argument(
        "--expected-execution-journal-head-digest", required=True
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Advance or verify one immutable local report version from an "
            "exact completed successor-bound task; never make it current or "
            "plan another task."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    advance = subparsers.add_parser(
        "advance", help="commit one fresh immutable recursive report version"
    )
    _add_common_inputs(advance)
    advance.add_argument(
        "--confirm-immutable-local-report-advance",
        action="store_true",
        required=True,
        help=(
            "acknowledge a local non-current, non-planned report advance"
        ),
    )

    verify = subparsers.add_parser(
        "verify", help="verify one committed report advance without writes"
    )
    _add_common_inputs(verify)
    for name in _VERIFY_DIGEST_NAMES:
        verify.add_argument("--" + name.replace("_", "-"), required=True)
    return parser


def _positional(args) -> tuple[Path, ...]:
    return (
        args.publication_root,
        args.adoption_contract,
        args.adoption_root,
        args.successor_contract,
        args.successor_root,
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
    )


def _common_kwargs(args) -> dict[str, str]:
    return {
        "advance_id": args.advance_id,
        "adoption_id": args.adoption_id,
        "successor_id": args.successor_id,
        **{name: getattr(args, name) for name in _COMMON_DIGEST_NAMES},
        "task_id": args.task_id,
    }


def _validated_result(payload: Any, args) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("recursive report advance core returned a non-object")
    expected_status = _STATUS if args.action == "advance" else _VERIFIED_STATUS
    if payload.get("status") != expected_status:
        raise ValueError("recursive report advance status differs from V1")
    if payload.get("current_status") != "NOT_CURRENT":
        raise ValueError("recursive report advance unexpectedly reports current")
    if payload.get("planning_status") != "NOT_PLANNED":
        raise ValueError("recursive report advance unexpectedly reports planning")
    root = payload.get("advance_root")
    if type(root) is not str or Path(root) != args.advance_root.resolve():
        raise ValueError("recursive report advance root differs")
    for key in (
        "advance_digest",
        "reingestion_digest",
        "output_report_body_digest",
        "output_audit_head",
        "output_evidence_digest",
    ):
        _require_digest(payload.get(key), f"result {key}")
    for key in ("typed_row_count", "pending_evidence_count"):
        value = payload.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"recursive report advance has invalid {key}")
    keys = (
        "status",
        "advance_root",
        "advance_digest",
        "reingestion_digest",
        "output_report_body_digest",
        "output_audit_head",
        "output_evidence_digest",
        "typed_row_count",
        "pending_evidence_count",
        "current_status",
        "planning_status",
    )
    return {key: payload[key] for key in keys}


def _write_summary(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_paths(args)
        _validate_explicit_bindings(args)
        core = _load_advance_core()
        kwargs = _common_kwargs(args)
        if args.action == "advance":
            payload = core.advance_recursive_report_version(
                *_positional(args),
                **kwargs,
                confirm_immutable_local_report_advance=True,
            )
        else:
            payload = core.verify_recursive_report_advance(
                *_positional(args),
                **kwargs,
                **{
                    name: getattr(args, name)
                    for name in _VERIFY_DIGEST_NAMES
                },
            )
        summary = _validated_result(payload, args)
        _write_summary(summary)
        print(
            "structural_hypothesis_recursive_report_advance "
            f"action={args.action} status={summary['status']} "
            f"advance_digest={summary['advance_digest']} "
            f"planning_status={summary['planning_status']} "
            f"current_status={summary['current_status']}",
            file=sys.stderr,
        )
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
            f"invalid structural-hypothesis recursive report advance: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
