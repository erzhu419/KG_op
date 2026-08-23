"""Materialize or verify one successor bundle from a named report adoption.

This local, offline surface never imports the benchmark executor, invokes
``run_one``, authorizes a task, creates an attempt, or writes a current pointer.
Materialization publishes only a fresh successor capsule; verification is
read-only and requires independently retained output digests.
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
    / "performance/manifests/structural_hypothesis_reingestion_publisher_v1.json"
)
DEFAULT_MATERIALIZER_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_task_materializer_v1.json"
)

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SUCCESS_STATUS = (
    "SUCCESSOR_MATERIALIZED_FROM_VERIFIED_ADOPTION_NOT_AUTHORIZED"
)
_VERIFIED_STATUS = "VERIFIED_" + _SUCCESS_STATUS


def _load_successor_core():
    from performance import structural_hypothesis_adopted_successor_materializer

    return structural_hypothesis_adopted_successor_materializer


def _require_digest(value: str, label: str) -> str:
    if not _DIGEST_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase sha256 digest")
    return value


def _require_absolute(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")


def _validate_paths(args: argparse.Namespace) -> None:
    paths = {
        "--publication-root": args.publication_root,
        "--adoption-contract": args.adoption_contract,
        "--adoption-root": args.adoption_root,
        "--successor-contract": args.successor_contract,
        "--successor-root": args.successor_root,
        "--base-evidence-csv": args.base_evidence_csv,
        "--source-attempt-root": args.source_attempt_root,
        "--hypothesis-contract": args.hypothesis_contract,
        "--executor-contract": args.executor_contract,
        "--runtime-contract": args.runtime_contract,
        "--publisher-contract": args.publisher_contract,
        "--materializer-contract": args.materializer_contract,
        "--base-manifest": args.base_manifest,
        "--asset-root": args.asset_root,
        "--future-attempt-root": args.future_attempt_root,
    }
    for label, path in paths.items():
        _require_absolute(path, label)

    for label, root in (
        ("publication root", args.publication_root),
        ("adoption root", args.adoption_root),
        ("source attempt root", args.source_attempt_root),
    ):
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"{label} must be an existing non-symlink directory")

    if args.action == "materialize":
        if args.successor_root.exists() or args.successor_root.is_symlink():
            raise ValueError(
                "fresh successor root is required; refusing to overwrite"
            )
        if (
            args.future_attempt_root.exists()
            or args.future_attempt_root.is_symlink()
        ):
            raise ValueError(
                "future attempt root must be absent during materialization"
            )
    elif args.successor_root.is_symlink() or not args.successor_root.is_dir():
        raise ValueError(
            "successor root must be an existing non-symlink directory"
        )


def _validate_explicit_bindings(args: argparse.Namespace) -> None:
    for option, value in (
        ("--adoption-id", args.adoption_id),
        ("--successor-id", args.successor_id),
    ):
        if not _ID_PATTERN.fullmatch(value):
            raise ValueError(f"{option} must be a non-path local mechanics label")
    if args.adoption_root.name != args.adoption_id:
        raise ValueError("--adoption-id must equal --adoption-root basename")
    if args.successor_root.name != args.successor_id:
        raise ValueError("--successor-id must equal --successor-root basename")
    if args.future_attempt_root.name != args.successor_id:
        raise ValueError(
            "--future-attempt-root basename must equal --successor-id"
        )

    for name in (
        "expected_adoption_digest",
        "expected_pending_evidence_digest",
        "expected_first_pending_projection_digest",
    ):
        _require_digest(getattr(args, name), "--" + name.replace("_", "-"))
    if args.action == "verify":
        for name in (
            "expected_successor_digest",
            "expected_bundle_digest",
            "expected_plan_digest",
        ):
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
        "--hypothesis-contract", type=Path, default=DEFAULT_HYPOTHESIS_CONTRACT
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
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--future-attempt-root", type=Path, required=True)
    parser.add_argument("--expected-adoption-digest", required=True)
    parser.add_argument("--expected-pending-evidence-digest", required=True)
    parser.add_argument(
        "--expected-first-pending-projection-digest", required=True
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize or verify an exact local successor task bundle from "
            "one fully verified named report adoption, without authorization "
            "or execution."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    materialize = subparsers.add_parser(
        "materialize", help="publish one fresh not-authorized successor capsule"
    )
    _add_common_inputs(materialize)
    materialize.add_argument(
        "--confirm-successor-materialization",
        action="store_true",
        required=True,
        help=(
            "acknowledge local successor planning/materialization; this does "
            "not grant authorization or permit execution"
        ),
    )

    verify = subparsers.add_parser(
        "verify", help="verify one successor capsule without changing it"
    )
    _add_common_inputs(verify)
    verify.add_argument("--expected-successor-digest", required=True)
    verify.add_argument("--expected-bundle-digest", required=True)
    verify.add_argument("--expected-plan-digest", required=True)
    return parser


def _common_kwargs(args: argparse.Namespace) -> dict[str, str]:
    return {
        "adoption_id": args.adoption_id,
        "successor_id": args.successor_id,
        "expected_adoption_digest": args.expected_adoption_digest,
        "expected_pending_evidence_digest": (
            args.expected_pending_evidence_digest
        ),
        "expected_first_pending_projection_digest": (
            args.expected_first_pending_projection_digest
        ),
    }


def _result_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if type(value) is not str or not value:
        raise ValueError(f"successor materialization result has an invalid {key}")
    return value


def _validated_result(
    payload: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    expected_status = (
        _SUCCESS_STATUS if args.action == "materialize" else _VERIFIED_STATUS
    )
    if payload.get("status") != expected_status:
        raise ValueError("successor materialization result status differs from V1")
    if payload.get("authorization_status") != "NOT_AUTHORIZED":
        raise ValueError("successor materialization unexpectedly reports authorization")
    task_count = payload.get("task_count")
    if type(task_count) is not int or not 1 <= task_count <= 30:
        raise ValueError("successor materialization result has an invalid task_count")

    strings = {
        key: _result_string(payload, key)
        for key in (
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
            "future_attempt_root",
            "checkpoint_root",
            "authorization_status",
        )
    }
    if Path(strings["successor_root"]) != args.successor_root.resolve():
        raise ValueError("successor materialization result root differs")
    if Path(strings["future_attempt_root"]) != args.future_attempt_root.resolve():
        raise ValueError("successor materialization future attempt root differs")
    if Path(strings["checkpoint_root"]) != args.future_attempt_root.resolve() / "checkpoints":
        raise ValueError("successor materialization checkpoint root differs")
    if strings["adoption_digest"] != args.expected_adoption_digest:
        raise ValueError("successor materialization adoption digest differs")
    if (
        strings["pending_evidence_digest"]
        != args.expected_pending_evidence_digest
        or strings["first_pending_projection_digest"]
        != args.expected_first_pending_projection_digest
    ):
        raise ValueError("successor materialization pending-cell anchors differ")
    for key in (
        "successor_digest",
        "adoption_digest",
        "pending_evidence_digest",
        "first_pending_projection_digest",
        "bundle_digest",
        "plan_digest",
        "first_task_digest",
    ):
        _require_digest(strings[key], f"result {key}")
    if args.action == "verify" and (
        strings["successor_digest"] != args.expected_successor_digest
        or strings["bundle_digest"] != args.expected_bundle_digest
        or strings["plan_digest"] != args.expected_plan_digest
    ):
        raise ValueError("successor materialization output anchors differ")
    return {**strings, "task_count": task_count}


def _write_canonical_summary(summary: dict[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def _summary(payload: dict[str, Any], action: str) -> str:
    fields = [
        "structural_hypothesis_adopted_successor_materializer",
        f"action={action}",
        f"status={payload.get('status', 'UNKNOWN')}",
    ]
    for key in (
        "successor_digest",
        "bundle_digest",
        "plan_digest",
        "first_task_id",
        "first_task_digest",
        "task_count",
    ):
        value = payload.get(key)
        if type(value) in (str, int):
            fields.append(f"{key}={value}")
    return " ".join(fields)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_paths(args)
        _validate_explicit_bindings(args)
        core = _load_successor_core()
        positional = (
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
            args.base_manifest,
            args.asset_root,
            args.future_attempt_root,
        )
        kwargs = _common_kwargs(args)
        if args.action == "materialize":
            payload = core.materialize_adopted_successor(*positional, **kwargs)
        else:
            payload = core.verify_adopted_successor(
                *positional,
                expected_successor_digest=args.expected_successor_digest,
                expected_bundle_digest=args.expected_bundle_digest,
                expected_plan_digest=args.expected_plan_digest,
                **kwargs,
            )
        if type(payload) is not dict:
            raise TypeError("successor materialization core returned a non-object")
        validated = _validated_result(payload, args)
        if args.action == "materialize":
            _write_canonical_summary(validated)
        print(_summary(validated, args.action), file=sys.stderr)
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
            f"invalid structural-hypothesis adopted successor: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
