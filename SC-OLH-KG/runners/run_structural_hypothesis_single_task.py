"""Prepare, execute, or verify one real local structural-hypothesis task.

The module intentionally imports only the standard library at startup.  The
runtime core (and therefore NumPy) is imported only after the required offline
and single-thread BLAS environment has been checked.  ``execute`` is the sole
operation that can cross the real ``run_one(task)`` boundary.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))


DEFAULT_HYPOTHESIS_CONTRACT = (
    ROOT / "performance/manifests/structural_hypothesis_loop_v1.json"
)
DEFAULT_EXECUTOR_CONTRACT = (
    ROOT / "performance/manifests/structural_hypothesis_executor_v1.json"
)
DEFAULT_MATERIALIZER_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_task_materializer_v1.json"
)
DEFAULT_RUNTIME_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_single_task_runtime_v1.json"
)

REQUIRED_EXECUTION_ENVIRONMENT = {
    "SCOLHKG_OFFLINE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON has duplicate key {key!r}")
        result[key] = value
    return result


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(
        path.read_bytes().decode("utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object")
    return payload


def _require_execution_environment() -> None:
    mismatches = {
        key: os.environ.get(key)
        for key, required in REQUIRED_EXECUTION_ENVIRONMENT.items()
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


def _validate_attempt_root(path: Path, *, must_be_fresh: bool) -> None:
    if not path.is_absolute():
        raise ValueError("--attempt-root must be an absolute path")
    if must_be_fresh:
        if path.exists() or path.is_symlink():
            raise ValueError(
                "--attempt-root already exists; a fresh attempt is required"
            )
    elif path.is_symlink() or not path.is_dir():
        raise ValueError("--attempt-root must be an existing non-symlink directory")


def _load_runtime_core():
    return importlib.import_module(
        "performance.structural_hypothesis_single_task_runtime"
    )


def _add_prepare_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--hypothesis-contract",
        type=Path,
        default=DEFAULT_HYPOTHESIS_CONTRACT,
    )
    parser.add_argument(
        "--executor-contract", type=Path, default=DEFAULT_EXECUTOR_CONTRACT
    )
    parser.add_argument(
        "--materializer-contract",
        type=Path,
        default=DEFAULT_MATERIALIZER_CONTRACT,
    )
    parser.add_argument(
        "--runtime-contract", type=Path, default=DEFAULT_RUNTIME_CONTRACT
    )
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument(
        "--task-id",
        required=True,
        help="exact first-pending task ID; V1 never authorizes a task set",
    )
    parser.add_argument("--expected-bundle-digest", required=True)
    parser.add_argument("--expected-plan-digest", required=True)
    parser.add_argument(
        "--authorization-id",
        required=True,
        help="local consent label; it is not an identity or signature",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, explicitly execute, or verify exactly one fresh local "
            "structural-hypothesis attempt."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="create one fresh, digest-bound authorization attempt"
    )
    _add_prepare_inputs(prepare)

    execute = subparsers.add_parser(
        "execute", help="run the one authorized real local task"
    )
    execute.add_argument("--attempt-root", type=Path, required=True)
    execute.add_argument(
        "--runtime-contract", type=Path, default=DEFAULT_RUNTIME_CONTRACT
    )
    execute.add_argument("--expected-authorization-digest", required=True)
    execute.add_argument(
        "--confirm-real-local-execution",
        action="store_true",
        required=True,
        help="required acknowledgement that this invokes real run_one(task)",
    )

    verify = subparsers.add_parser(
        "verify", help="verify the saved attempt without importing the benchmark"
    )
    verify.add_argument("--attempt-root", type=Path, required=True)
    verify.add_argument(
        "--runtime-contract", type=Path, default=DEFAULT_RUNTIME_CONTRACT
    )
    verify.add_argument("--base-manifest", type=Path, required=True)
    verify.add_argument("--asset-root", type=Path, required=True)
    verify.add_argument("--expected-authorization-digest", required=True)
    verify.add_argument("--expected-receipt-digest")
    verify.add_argument("--expected-journal-head-digest")
    verify.add_argument("--expected-attempt-digest")
    return parser


def _nested_string(payload: dict[str, Any], *paths: tuple[str, ...]) -> str:
    for path in paths:
        value: Any = payload
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            if isinstance(value, str) and value:
                return value
    raise ValueError(
        "single-task runtime result is missing "
        + "/".join(paths[0])
    )


def _prepare_output(payload: dict[str, Any], args) -> dict[str, str]:
    status = payload.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("single-task runtime result is missing status")
    authorization_digest = _nested_string(
        payload,
        ("authorization_digest",),
        ("integrity", "authorization_digest"),
        ("authorization", "authorization_digest"),
        ("authorization", "integrity", "authorization_digest"),
    )
    return {
        "attempt_root": str(args.attempt_root.resolve()),
        "attempt_digest": _nested_string(
            payload,
            ("attempt_digest",),
            ("integrity", "attempt_digest"),
        ),
        "authorization_digest": authorization_digest,
        "bundle_digest": args.expected_bundle_digest,
        "plan_digest": args.expected_plan_digest,
        "status": status,
        "task_id": args.task_id,
    }


def _execute_output(payload: dict[str, Any], args) -> dict[str, str]:
    return {
        "attempt_digest": _nested_string(
            payload,
            ("attempt_digest",),
            ("integrity", "attempt_digest"),
        ),
        "authorization_digest": _nested_string(
            payload,
            ("authorization_digest",),
            ("integrity", "authorization_digest"),
            ("authorization_binding", "authorization_digest"),
        ),
        "journal_head_digest": _nested_string(
            payload,
            ("journal_head_digest",),
            ("integrity", "journal_head_digest"),
            ("journal", "head_digest"),
        ),
        "receipt_digest": _nested_string(
            payload,
            ("receipt_digest",),
            ("integrity", "receipt_digest"),
            ("receipt", "integrity", "receipt_digest"),
        ),
    }


def _write_canonical_summary(summary: dict[str, str]) -> None:
    sys.stdout.write(json.dumps(
        summary,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n")


def _summary(payload: dict[str, Any], action: str) -> str:
    status = payload.get("status", "UNKNOWN")
    observed = []
    for key in (
        "authorization_digest",
        "receipt_digest",
        "journal_head_digest",
        "attempt_digest",
    ):
        try:
            value = _nested_string(payload, (key,), ("integrity", key))
        except ValueError:
            continue
        observed.append(f"{key}={value}")
    suffix = " " + " ".join(observed) if observed else ""
    return (
        "structural_hypothesis_single_task "
        f"action={action} status={status}{suffix}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action in {"prepare", "execute"}:
            # This check must precede importing the core: its validation chain
            # imports NumPy, after which changing OpenBLAS variables is too late.
            _require_execution_environment()
        _validate_attempt_root(
            args.attempt_root, must_be_fresh=args.action == "prepare"
        )
        core = _load_runtime_core()

        if args.action == "prepare":
            report = _load_object(args.report, "hypothesis report")
            bundle = _load_object(args.bundle, "materialized task bundle")
            hypothesis_contract = _load_object(
                args.hypothesis_contract, "hypothesis contract"
            )
            executor_contract = _load_object(
                args.executor_contract, "executor contract"
            )
            materializer_contract = _load_object(
                args.materializer_contract, "materializer contract"
            )
            runtime_contract = _load_object(
                args.runtime_contract, "single-task runtime contract"
            )
            payload = core.prepare_single_task_attempt(
                report,
                bundle,
                hypothesis_contract,
                executor_contract,
                materializer_contract,
                runtime_contract,
                args.base_manifest,
                args.asset_root,
                args.attempt_root,
                task_id=args.task_id,
                expected_bundle_digest=args.expected_bundle_digest,
                expected_plan_digest=args.expected_plan_digest,
                authorization_id=args.authorization_id,
            )
        elif args.action == "execute":
            runtime_contract = _load_object(
                args.runtime_contract, "single-task runtime contract"
            )
            payload = core.execute_single_task_attempt(
                args.attempt_root,
                runtime_contract,
                expected_authorization_digest=(
                    args.expected_authorization_digest
                ),
            )
        else:
            runtime_contract = _load_object(
                args.runtime_contract, "single-task runtime contract"
            )
            verified = core.verify_single_task_attempt(
                args.attempt_root,
                runtime_contract,
                args.base_manifest,
                args.asset_root,
                expected_authorization_digest=(
                    args.expected_authorization_digest
                ),
                expected_receipt_digest=args.expected_receipt_digest,
                expected_journal_head_digest=(
                    args.expected_journal_head_digest
                ),
                expected_attempt_digest=args.expected_attempt_digest,
            )
            if isinstance(verified, dict):
                payload = verified
            elif verified is True:
                payload = {"status": "VERIFIED", "integrity": {}}
            else:
                raise ValueError("single-task attempt failed verification")

        if not isinstance(payload, dict):
            raise TypeError("single-task runtime core returned a non-object")
        if args.action == "prepare":
            _write_canonical_summary(_prepare_output(payload, args))
        elif args.action == "execute":
            _write_canonical_summary(_execute_output(payload, args))
        print(_summary(payload, args.action), file=sys.stderr)
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
        print(f"invalid structural-hypothesis single-task runtime: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
