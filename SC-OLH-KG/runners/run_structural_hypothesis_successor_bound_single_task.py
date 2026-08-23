"""Inspect, prepare, or verify one successor-bound local task attempt.

``inspect`` is read-only. ``prepare`` grants one explicit local authorization
and delegates attempt creation to the frozen single-task runtime V1, but never
executes the task. ``verify`` is read-only. No action in this runner imports or
invokes the native ``run_one`` callback.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
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

REQUIRED_PREPARE_ENVIRONMENT = {
    "SCOLHKG_OFFLINE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_TASK_ID_PATTERN = re.compile(r"task:[0-9a-f]{24}\Z")
_AUTHORIZATION_ID_PATTERN = re.compile(r"successor-bound-v1:[0-9a-f]{64}\Z")

_INSPECT_STATUS = (
    "INSPECTED_SUCCESSOR_BOUND_TASK_NOT_AUTHORIZED_NOT_PREPARED"
)
_PREPARE_STATUS = "SUCCESSOR_BOUND_AUTHORIZED_NOT_EXECUTED"
_VERIFY_STATUS = "VERIFIED_" + _PREPARE_STATUS


def _load_bridge_core():
    return importlib.import_module(
        "performance.structural_hypothesis_successor_bound_single_task"
    )


def _require_prepare_environment() -> None:
    mismatches = {
        key: os.environ.get(key)
        for key, required in REQUIRED_PREPARE_ENVIRONMENT.items()
        if os.environ.get(key) != required
    }
    if mismatches:
        rendered = ", ".join(
            f"{key}={value!r} (required '1')"
            for key, value in sorted(mismatches.items())
        )
        raise ValueError(
            "offline/single-thread runtime environment is not pinned before "
            f"Python startup: {rendered}"
        )


def _require_digest(value: str, label: str) -> str:
    if type(value) is not str or not _DIGEST_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase sha256 digest")
    return value


def _require_absolute(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")


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
    parser.add_argument(
        "--bridge-contract", type=Path, default=DEFAULT_BRIDGE_CONTRACT
    )
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
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


def _add_expected_provenance(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-provenance-binding-digest", required=True)


def _add_authorization_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--authorization-id",
        required=True,
        help=(
            "exact deterministic successor-bound authorization ID printed by "
            "inspect; it is a local consent binding, not identity or authority"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect, locally authorize/prepare, or verify exactly one "
            "successor-bound first-pending task without executing it."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    inspect = subparsers.add_parser(
        "inspect",
        help=(
            "read-only full-chain inspection and deterministic authorization "
            "binding derivation"
        ),
    )
    _add_common_inputs(inspect)

    prepare = subparsers.add_parser(
        "prepare",
        help="explicitly authorize one exact task and create an AUTHORIZED attempt",
    )
    _add_common_inputs(prepare)
    _add_expected_provenance(prepare)
    _add_authorization_input(prepare)
    prepare.add_argument(
        "--confirm-successor-bound-local-authorization",
        action="store_true",
        required=True,
        help=(
            "grant local authorization for this exact successor-bound task; "
            "this does not permit or invoke execution"
        ),
    )

    verify = subparsers.add_parser(
        "verify",
        help="read-only full-chain verification of an AUTHORIZED attempt",
    )
    _add_common_inputs(verify)
    _add_expected_provenance(verify)
    verify.add_argument("--expected-authorization-digest", required=True)
    verify.add_argument("--expected-attempt-digest", required=True)
    return parser


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
        "--bridge-contract": args.bridge_contract,
        "--base-manifest": args.base_manifest,
        "--asset-root": args.asset_root,
        "--attempt-root": args.attempt_root,
    }
    for label, path in paths.items():
        _require_absolute(path, label)

    for label, root in (
        ("publication root", args.publication_root),
        ("adoption root", args.adoption_root),
        ("successor root", args.successor_root),
        ("source attempt root", args.source_attempt_root),
        ("asset root", args.asset_root),
    ):
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"{label} must be an existing non-symlink directory")

    if args.action in {"inspect", "prepare"}:
        if args.attempt_root.exists() or args.attempt_root.is_symlink():
            raise ValueError(
                "attempt root must be absent before inspection/preparation"
            )
    elif args.attempt_root.is_symlink() or not args.attempt_root.is_dir():
        raise ValueError(
            "attempt root must be an existing non-symlink directory"
        )


def _validate_explicit_bindings(args: argparse.Namespace) -> None:
    for option, value in (
        ("--adoption-id", args.adoption_id),
        ("--successor-id", args.successor_id),
    ):
        if type(value) is not str or not _ID_PATTERN.fullmatch(value):
            raise ValueError(f"{option} must be a non-path local mechanics label")
    if args.adoption_root.name != args.adoption_id:
        raise ValueError("--adoption-id must equal --adoption-root basename")
    if args.successor_root.name != args.successor_id:
        raise ValueError("--successor-id must equal --successor-root basename")
    if args.attempt_root.name != args.successor_id:
        raise ValueError("--attempt-root basename must equal --successor-id")
    if type(args.task_id) is not str or not _TASK_ID_PATTERN.fullmatch(
        args.task_id
    ):
        raise ValueError("--task-id must be an exact task:<24 lowercase hex> ID")

    for name in (
        "expected_adoption_digest",
        "expected_pending_evidence_digest",
        "expected_first_pending_projection_digest",
        "expected_successor_digest",
        "expected_bundle_digest",
        "expected_plan_digest",
        "expected_task_digest",
    ):
        _require_digest(getattr(args, name), "--" + name.replace("_", "-"))
    if args.action in {"prepare", "verify"}:
        _require_digest(
            args.expected_provenance_binding_digest,
            "--expected-provenance-binding-digest",
        )
    if args.action == "prepare":
        if not _AUTHORIZATION_ID_PATTERN.fullmatch(args.authorization_id):
            raise ValueError(
                "--authorization-id must be the exact successor-bound-v1 ID "
                "printed by inspect"
            )
    if args.action == "verify":
        _require_digest(
            args.expected_authorization_digest,
            "--expected-authorization-digest",
        )
        _require_digest(
            args.expected_attempt_digest,
            "--expected-attempt-digest",
        )


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
        "expected_successor_digest": args.expected_successor_digest,
        "expected_bundle_digest": args.expected_bundle_digest,
        "expected_plan_digest": args.expected_plan_digest,
        "task_id": args.task_id,
        "expected_task_digest": args.expected_task_digest,
    }


def _result_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if type(value) is not str or not value:
        raise ValueError(f"successor-bound result has an invalid {key}")
    return value


def _validated_result(
    payload: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    expected_status = {
        "inspect": _INSPECT_STATUS,
        "prepare": _PREPARE_STATUS,
        "verify": _VERIFY_STATUS,
    }[args.action]
    if payload.get("status") != expected_status:
        raise ValueError("successor-bound result status differs from frozen V1")

    expected_authorization_status = (
        "NOT_AUTHORIZED" if args.action == "inspect" else "AUTHORIZED"
    )
    expected_attempt_status = (
        "NOT_PREPARED"
        if args.action == "inspect"
        else "AUTHORIZED_NOT_EXECUTED"
    )
    if payload.get("authorization_status") != expected_authorization_status:
        raise ValueError("successor-bound authorization status differs")
    if payload.get("attempt_status") != expected_attempt_status:
        raise ValueError("successor-bound attempt status differs")

    strings = {
        key: _result_string(payload, key)
        for key in (
            "status",
            "authorization_status",
            "attempt_status",
            "provenance_binding_digest",
            "required_authorization_id",
            "adoption_digest",
            "successor_digest",
            "pending_evidence_digest",
            "first_pending_projection_digest",
            "bundle_digest",
            "plan_digest",
            "task_id",
            "task_digest",
            "attempt_root",
            "checkpoint_root",
        )
    }
    for key in (
        "provenance_binding_digest",
        "adoption_digest",
        "successor_digest",
        "pending_evidence_digest",
        "first_pending_projection_digest",
        "bundle_digest",
        "plan_digest",
        "task_digest",
    ):
        _require_digest(strings[key], f"result {key}")
    if not _AUTHORIZATION_ID_PATTERN.fullmatch(
        strings["required_authorization_id"]
    ):
        raise ValueError("successor-bound result authorization ID is malformed")

    task_count = payload.get("task_count")
    provenance_binding = payload.get("provenance_binding")
    if type(task_count) is not int or not 1 <= task_count <= 30:
        raise ValueError("successor-bound result task_count differs")
    if type(provenance_binding) is not dict:
        raise ValueError("successor-bound provenance binding is not an object")
    if Path(strings["attempt_root"]) != args.attempt_root.resolve():
        raise ValueError("successor-bound result attempt root differs")
    if Path(strings["checkpoint_root"]) != args.attempt_root.resolve() / "checkpoints":
        raise ValueError("successor-bound result checkpoint root differs")
    expected_values = {
        "adoption_digest": args.expected_adoption_digest,
        "successor_digest": args.expected_successor_digest,
        "pending_evidence_digest": args.expected_pending_evidence_digest,
        "first_pending_projection_digest": (
            args.expected_first_pending_projection_digest
        ),
        "bundle_digest": args.expected_bundle_digest,
        "plan_digest": args.expected_plan_digest,
        "task_id": args.task_id,
        "task_digest": args.expected_task_digest,
    }
    if any(strings[key] != value for key, value in expected_values.items()):
        raise ValueError("successor-bound result anchors differ")

    result: dict[str, Any] = {
        **strings,
        "provenance_binding": provenance_binding,
        "task_count": task_count,
    }
    if args.action in {"prepare", "verify"}:
        if strings["provenance_binding_digest"] != (
            args.expected_provenance_binding_digest
        ):
            raise ValueError("successor-bound authorization binding differs")
        expected_authorization_id = (
            "successor-bound-v1:"
            + args.expected_provenance_binding_digest.removeprefix("sha256:")
        )
        if strings["required_authorization_id"] != expected_authorization_id:
            raise ValueError("successor-bound derived authorization ID differs")
        if args.action == "prepare" and (
            strings["required_authorization_id"] != args.authorization_id
        ):
            raise ValueError("successor-bound authorization ID differs")
        for key in ("authorization_digest", "attempt_digest"):
            result[key] = _require_digest(
                _result_string(payload, key), f"result {key}"
            )
    if args.action == "verify" and (
        result["authorization_digest"] != args.expected_authorization_digest
        or result["attempt_digest"] != args.expected_attempt_digest
    ):
        raise ValueError("successor-bound attempt output anchors differ")
    return result


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
        "structural_hypothesis_successor_bound_single_task",
        f"action={action}",
        f"status={payload.get('status', 'UNKNOWN')}",
    ]
    for key in (
        "provenance_binding_digest",
        "required_authorization_id",
        "task_id",
        "task_digest",
        "authorization_digest",
        "attempt_digest",
    ):
        value = payload.get(key)
        if type(value) is str:
            fields.append(f"{key}={value}")
    return " ".join(fields)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_paths(args)
        _validate_explicit_bindings(args)
        if args.action == "prepare":
            # Must happen before importing the bridge and frozen runtime core;
            # importing NumPy before BLAS limits are set is too late.
            _require_prepare_environment()
        core = _load_bridge_core()
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
            args.bridge_contract,
            args.base_manifest,
            args.asset_root,
            args.attempt_root,
        )
        kwargs = _common_kwargs(args)
        if args.action == "inspect":
            payload = core.inspect_successor_bound_single_task(
                *positional, **kwargs
            )
        elif args.action == "prepare":
            payload = core.prepare_successor_bound_single_task_attempt(
                *positional,
                expected_provenance_binding_digest=(
                    args.expected_provenance_binding_digest
                ),
                authorization_id=args.authorization_id,
                **kwargs,
            )
        else:
            payload = core.verify_successor_bound_single_task_attempt(
                *positional,
                expected_provenance_binding_digest=(
                    args.expected_provenance_binding_digest
                ),
                expected_authorization_digest=(
                    args.expected_authorization_digest
                ),
                expected_attempt_digest=args.expected_attempt_digest,
                **kwargs,
            )
        if type(payload) is not dict:
            raise TypeError("successor-bound bridge core returned a non-object")
        validated = _validated_result(payload, args)
        if args.action in {"inspect", "prepare"}:
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
            f"invalid structural-hypothesis successor-bound task: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
