"""Publish or verify one local structural-hypothesis reingestion.

This CLI has no execution, adoption, replan, network, scheduler, shell, or
credential surface.  ``publish`` consumes exactly one completed single-task
attempt and creates a fresh committed publication.  ``verify`` is read-only.
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

DEFAULT_PUBLISHER_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_reingestion_publisher_v1.json"
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

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PUBLICATION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


def _load_publisher_core():
    from performance import structural_hypothesis_reingestion_publisher

    return structural_hypothesis_reingestion_publisher


def _require_digest(value: str, label: str) -> str:
    if not _DIGEST_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase sha256 digest")
    return value


def _require_absolute(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")


def _validate_paths(args) -> None:
    paths = {
        "--base-evidence-csv": args.base_evidence_csv,
        "--attempt-root": args.attempt_root,
        "--hypothesis-contract": args.hypothesis_contract,
        "--executor-contract": args.executor_contract,
        "--publisher-contract": args.publisher_contract,
        "--runtime-contract": args.runtime_contract,
        "--base-manifest": args.base_manifest,
        "--asset-root": args.asset_root,
        "--publication-root": args.publication_root,
    }
    for label, path in paths.items():
        _require_absolute(path, label)

    root = args.publication_root
    if args.action == "publish":
        if root.exists() or root.is_symlink():
            raise ValueError(
                "fresh publication root is required; refusing to overwrite"
            )
    elif root.is_symlink() or not root.is_dir():
        raise ValueError(
            "publication root must be an existing non-symlink directory"
        )


def _validate_explicit_bindings(args) -> None:
    for name in (
        "expected_source_evidence_digest",
        "expected_plan_digest",
        "expected_authorization_digest",
        "expected_execution_receipt_digest",
        "expected_execution_journal_head_digest",
        "expected_execution_attempt_digest",
    ):
        _require_digest(getattr(args, name), "--" + name.replace("_", "-"))
    if args.action == "publish":
        if not _PUBLICATION_ID_PATTERN.fullmatch(args.publication_id):
            raise ValueError(
                "--publication-id must be a non-path local mechanics label"
            )
    else:
        for name in (
            "expected_publication_digest",
            "expected_reingestion_digest",
            "expected_output_report_body_digest",
            "expected_output_audit_head",
            "expected_output_evidence_digest",
        ):
            _require_digest(
                getattr(args, name), "--" + name.replace("_", "-")
            )


def _add_contract_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-evidence-csv", type=Path, required=True)
    parser.add_argument(
        "--hypothesis-contract",
        type=Path,
        default=DEFAULT_HYPOTHESIS_CONTRACT,
    )
    parser.add_argument(
        "--executor-contract",
        type=Path,
        default=DEFAULT_EXECUTOR_CONTRACT,
    )
    parser.add_argument(
        "--publisher-contract",
        type=Path,
        default=DEFAULT_PUBLISHER_CONTRACT,
    )
    parser.add_argument(
        "--runtime-contract", type=Path, default=DEFAULT_RUNTIME_CONTRACT
    )
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--publication-root", type=Path, required=True)
    parser.add_argument("--expected-source-evidence-digest", required=True)
    parser.add_argument("--expected-plan-digest", required=True)
    parser.add_argument("--expected-authorization-digest", required=True)
    parser.add_argument("--expected-execution-receipt-digest", required=True)
    parser.add_argument(
        "--expected-execution-journal-head-digest", required=True
    )
    parser.add_argument("--expected-execution-attempt-digest", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish or verify exactly one successful local reingestion "
            "without adopting it as current or replanning."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    publish = subparsers.add_parser(
        "publish", help="commit one fresh local reingestion publication"
    )
    _add_contract_inputs(publish)
    publish.add_argument("--publication-id", required=True)
    publish.add_argument(
        "--confirm-local-reingestion",
        action="store_true",
        required=True,
        help=(
            "acknowledge the local evidence/report transition; this does not "
            "adopt a current report"
        ),
    )

    verify = subparsers.add_parser(
        "verify", help="verify a committed publication without execution"
    )
    _add_contract_inputs(verify)
    verify.add_argument("--expected-publication-digest", required=True)
    verify.add_argument("--expected-reingestion-digest", required=True)
    verify.add_argument(
        "--expected-output-report-body-digest", required=True
    )
    verify.add_argument("--expected-output-audit-head", required=True)
    verify.add_argument("--expected-output-evidence-digest", required=True)
    return parser


def _nested_value(payload: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = payload
        for key in path:
            if type(value) is not dict or key not in value:
                break
            value = value[key]
        else:
            return value
    raise ValueError(
        "reingestion publisher result is missing " + "/".join(paths[0])
    )


def _nested_string(payload: dict[str, Any], *paths: tuple[str, ...]) -> str:
    value = _nested_value(payload, *paths)
    if type(value) is not str or not value:
        raise ValueError(
            "reingestion publisher result has an invalid "
            + "/".join(paths[0])
        )
    return value


def _nested_nonnegative_int(
    payload: dict[str, Any], *paths: tuple[str, ...]
) -> int:
    value = _nested_value(payload, *paths)
    if type(value) is not int or value < 0:
        raise ValueError(
            "reingestion publisher result has an invalid "
            + "/".join(paths[0])
        )
    return value


def _publish_output(payload: dict[str, Any], args) -> dict[str, Any]:
    return {
        "accepted_successful_rows": _nested_nonnegative_int(
            payload,
            ("accepted_successful_rows",),
            ("reingestion_receipt", "accepted_successful_rows"),
        ),
        "authorization_digest": args.expected_authorization_digest,
        "execution_receipt_digest": args.expected_execution_receipt_digest,
        "ignored_failed_attempts": _nested_nonnegative_int(
            payload,
            ("ignored_failed_attempts",),
            ("reingestion_receipt", "ignored_failed_attempts"),
        ),
        "output_audit_head": _nested_string(
            payload,
            ("output_audit_head",),
            ("report", "audit", "head"),
            ("output_report", "audit", "head"),
        ),
        "output_evidence_digest": _nested_string(
            payload,
            ("output_evidence_digest",),
            ("report", "evidence_digest"),
            ("output_report", "evidence_digest"),
        ),
        "output_report_body_digest": _nested_string(
            payload,
            ("output_report_body_digest",),
            ("report", "audit", "report_body_digest"),
            ("output_report", "audit", "report_body_digest"),
        ),
        "plan_digest": args.expected_plan_digest,
        "publication_digest": _nested_string(
            payload,
            ("publication_digest",),
            ("integrity", "publication_digest"),
            ("publication", "integrity", "publication_digest"),
        ),
        "publication_root": str(args.publication_root.resolve()),
        "reingestion_digest": _nested_string(
            payload,
            ("reingestion_digest",),
            ("reingestion_receipt", "integrity", "reingestion_digest"),
        ),
        "status": _nested_string(payload, ("status",)),
    }


def _write_canonical_summary(summary: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(
        summary,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n")


def _summary(payload: dict[str, Any], action: str) -> str:
    fields = [
        "structural_hypothesis_reingestion",
        f"action={action}",
        f"status={payload.get('status', 'UNKNOWN')}",
    ]
    for key in (
        "publication_digest",
        "reingestion_digest",
        "output_report_body_digest",
    ):
        try:
            value = _nested_string(
                payload,
                (key,),
                ("integrity", key),
                ("reingestion_receipt", "integrity", key),
                ("report", "audit", key),
                ("output_report", "audit", key),
            )
        except ValueError:
            continue
        fields.append(f"{key}={value}")
    return " ".join(fields)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_paths(args)
        _validate_explicit_bindings(args)
        core = _load_publisher_core()
        if args.action == "publish":
            payload = core.publish_single_task_reingestion(
                args.base_evidence_csv,
                args.attempt_root,
                args.hypothesis_contract,
                args.executor_contract,
                args.runtime_contract,
                args.publisher_contract,
                args.base_manifest,
                args.asset_root,
                args.publication_root,
                expected_source_evidence_digest=(
                    args.expected_source_evidence_digest
                ),
                expected_plan_digest=args.expected_plan_digest,
                expected_authorization_digest=(
                    args.expected_authorization_digest
                ),
                expected_execution_receipt_digest=(
                    args.expected_execution_receipt_digest
                ),
                expected_execution_journal_head_digest=(
                    args.expected_execution_journal_head_digest
                ),
                expected_execution_attempt_digest=(
                    args.expected_execution_attempt_digest
                ),
                publication_id=args.publication_id,
            )
        else:
            payload = core.verify_single_task_reingestion_publication(
                args.base_evidence_csv,
                args.attempt_root,
                args.hypothesis_contract,
                args.executor_contract,
                args.runtime_contract,
                args.publisher_contract,
                args.base_manifest,
                args.asset_root,
                args.publication_root,
                expected_source_evidence_digest=(
                    args.expected_source_evidence_digest
                ),
                expected_plan_digest=args.expected_plan_digest,
                expected_authorization_digest=(
                    args.expected_authorization_digest
                ),
                expected_execution_receipt_digest=(
                    args.expected_execution_receipt_digest
                ),
                expected_execution_journal_head_digest=(
                    args.expected_execution_journal_head_digest
                ),
                expected_execution_attempt_digest=(
                    args.expected_execution_attempt_digest
                ),
                expected_publication_digest=args.expected_publication_digest,
                expected_reingestion_digest=args.expected_reingestion_digest,
                expected_output_report_body_digest=(
                    args.expected_output_report_body_digest
                ),
                expected_output_audit_head=args.expected_output_audit_head,
                expected_output_evidence_digest=(
                    args.expected_output_evidence_digest
                ),
            )
        if payload is True:
            payload = {"status": "VERIFIED_PUBLISHED_NOT_ADOPTED"}
        if type(payload) is not dict:
            raise TypeError("reingestion publisher core returned a non-object")
        if args.action == "publish":
            _write_canonical_summary(_publish_output(payload, args))
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
        print(
            f"invalid structural-hypothesis reingestion publisher: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
