"""Adopt or verify one local structural-hypothesis report version.

This CLI copies one fully verified reingestion publication into a fresh,
versioned local adoption capsule.  It never writes an ambient current pointer,
plans a task, executes a benchmark, or accesses external systems.
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
_ADOPTION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SUCCESS_STATUS = "ADOPTED_AS_LOCAL_REPORT_VERSION_NOT_PLANNED"
_VERIFIED_STATUS = "VERIFIED_" + _SUCCESS_STATUS


def _load_adoption_core():
    from performance import structural_hypothesis_report_adoption

    return structural_hypothesis_report_adoption


def _require_digest(value: str, label: str) -> str:
    if not _DIGEST_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase sha256 digest")
    return value


def _require_absolute(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")


def _validate_paths(args) -> None:
    paths = {
        "--publication-root": args.publication_root,
        "--adoption-contract": args.adoption_contract,
        "--adoption-root": args.adoption_root,
        "--base-evidence-csv": args.base_evidence_csv,
        "--attempt-root": args.attempt_root,
        "--hypothesis-contract": args.hypothesis_contract,
        "--executor-contract": args.executor_contract,
        "--runtime-contract": args.runtime_contract,
        "--publisher-contract": args.publisher_contract,
        "--base-manifest": args.base_manifest,
        "--asset-root": args.asset_root,
    }
    for label, path in paths.items():
        _require_absolute(path, label)

    if args.publication_root.is_symlink() or not args.publication_root.is_dir():
        raise ValueError(
            "publication root must be an existing non-symlink directory"
        )
    if args.action == "adopt":
        if args.adoption_root.exists() or args.adoption_root.is_symlink():
            raise ValueError(
                "fresh adoption root is required; refusing to overwrite"
            )
    elif args.adoption_root.is_symlink() or not args.adoption_root.is_dir():
        raise ValueError(
            "adoption root must be an existing non-symlink directory"
        )


def _validate_explicit_bindings(args) -> None:
    if not _ADOPTION_ID_PATTERN.fullmatch(args.adoption_id):
        raise ValueError(
            "--adoption-id must be a non-path local mechanics label"
        )
    for name in (
        "expected_source_evidence_digest",
        "expected_plan_digest",
        "expected_authorization_digest",
        "expected_execution_receipt_digest",
        "expected_execution_journal_head_digest",
        "expected_execution_attempt_digest",
        "expected_publication_digest",
        "expected_reingestion_digest",
        "expected_output_report_body_digest",
        "expected_output_audit_head",
        "expected_output_evidence_digest",
        "expected_publication_marker_raw_sha256",
        "expected_combined_rows_raw_sha256",
        "expected_output_report_raw_sha256",
        "expected_reingestion_receipt_raw_sha256",
    ):
        _require_digest(getattr(args, name), "--" + name.replace("_", "-"))
    if args.action == "verify":
        _require_digest(
            args.expected_adoption_digest,
            "--expected-adoption-digest",
        )


def _add_full_chain_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--publication-root", type=Path, required=True)
    parser.add_argument(
        "--adoption-contract",
        type=Path,
        default=DEFAULT_ADOPTION_CONTRACT,
    )
    parser.add_argument("--adoption-root", type=Path, required=True)
    parser.add_argument("--adoption-id", required=True)
    parser.add_argument("--base-evidence-csv", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
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
        "--runtime-contract",
        type=Path,
        default=DEFAULT_RUNTIME_CONTRACT,
    )
    parser.add_argument(
        "--publisher-contract",
        type=Path,
        default=DEFAULT_PUBLISHER_CONTRACT,
    )
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--expected-source-evidence-digest", required=True)
    parser.add_argument("--expected-plan-digest", required=True)
    parser.add_argument("--expected-authorization-digest", required=True)
    parser.add_argument("--expected-execution-receipt-digest", required=True)
    parser.add_argument(
        "--expected-execution-journal-head-digest", required=True
    )
    parser.add_argument("--expected-execution-attempt-digest", required=True)
    parser.add_argument("--expected-publication-digest", required=True)
    parser.add_argument("--expected-reingestion-digest", required=True)
    parser.add_argument(
        "--expected-output-report-body-digest", required=True
    )
    parser.add_argument("--expected-output-audit-head", required=True)
    parser.add_argument("--expected-output-evidence-digest", required=True)
    parser.add_argument(
        "--expected-publication-marker-raw-sha256", required=True
    )
    parser.add_argument("--expected-combined-rows-raw-sha256", required=True)
    parser.add_argument("--expected-output-report-raw-sha256", required=True)
    parser.add_argument(
        "--expected-reingestion-receipt-raw-sha256", required=True
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Adopt or verify one fully anchored publication as a versioned "
            "local report without planning or changing a current pointer."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    adopt = subparsers.add_parser(
        "adopt", help="commit one fresh local report-version capsule"
    )
    _add_full_chain_inputs(adopt)
    adopt.add_argument(
        "--confirm-local-report-adoption",
        action="store_true",
        required=True,
        help=(
            "acknowledge a local version adoption; this does not establish "
            "a global current report or admit a next task"
        ),
    )

    verify = subparsers.add_parser(
        "verify", help="verify one adoption capsule without changing it"
    )
    _add_full_chain_inputs(verify)
    verify.add_argument("--expected-adoption-digest", required=True)
    return parser


def _result_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if type(value) is not str or not value:
        raise ValueError(f"report adoption result has an invalid {key}")
    return value


def _adopt_output(payload: dict[str, Any], args) -> dict[str, str]:
    output = {
        key: _result_string(payload, key)
        for key in (
            "status",
            "adoption_root",
            "adoption_digest",
            "publication_digest",
            "reingestion_digest",
            "output_report_body_digest",
            "output_audit_head",
            "output_evidence_digest",
            "planning_status",
        )
    }
    if output["status"] != _SUCCESS_STATUS:
        raise ValueError("report adoption result status differs from V1")
    if output["planning_status"] != "NOT_PLANNED":
        raise ValueError("report adoption unexpectedly reports planning")
    if Path(output["adoption_root"]) != args.adoption_root.resolve():
        raise ValueError("report adoption result root differs")
    return output


def _write_canonical_summary(summary: dict[str, str]) -> None:
    sys.stdout.write(json.dumps(
        summary,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n")


def _summary(payload: dict[str, Any], action: str) -> str:
    fields = [
        "structural_hypothesis_report_adoption",
        f"action={action}",
        f"status={payload.get('status', 'UNKNOWN')}",
    ]
    for key in ("adoption_digest", "publication_digest", "planning_status"):
        value = payload.get(key)
        if type(value) is str and value:
            fields.append(f"{key}={value}")
    return " ".join(fields)


def _common_kwargs(args) -> dict[str, Any]:
    return {
        "adoption_id": args.adoption_id,
        "expected_source_evidence_digest": (
            args.expected_source_evidence_digest
        ),
        "expected_plan_digest": args.expected_plan_digest,
        "expected_authorization_digest": args.expected_authorization_digest,
        "expected_execution_receipt_digest": (
            args.expected_execution_receipt_digest
        ),
        "expected_execution_journal_head_digest": (
            args.expected_execution_journal_head_digest
        ),
        "expected_execution_attempt_digest": (
            args.expected_execution_attempt_digest
        ),
        "expected_publication_digest": args.expected_publication_digest,
        "expected_reingestion_digest": args.expected_reingestion_digest,
        "expected_output_report_body_digest": (
            args.expected_output_report_body_digest
        ),
        "expected_output_audit_head": args.expected_output_audit_head,
        "expected_output_evidence_digest": (
            args.expected_output_evidence_digest
        ),
        "expected_publication_marker_raw_sha256": (
            args.expected_publication_marker_raw_sha256
        ),
        "expected_combined_rows_raw_sha256": (
            args.expected_combined_rows_raw_sha256
        ),
        "expected_output_report_raw_sha256": (
            args.expected_output_report_raw_sha256
        ),
        "expected_reingestion_receipt_raw_sha256": (
            args.expected_reingestion_receipt_raw_sha256
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_paths(args)
        _validate_explicit_bindings(args)
        core = _load_adoption_core()
        kwargs = _common_kwargs(args)
        if args.action == "adopt":
            payload = core.adopt_structural_hypothesis_report(
                args.publication_root,
                args.adoption_contract,
                args.adoption_root,
                args.base_evidence_csv,
                args.attempt_root,
                args.hypothesis_contract,
                args.executor_contract,
                args.runtime_contract,
                args.publisher_contract,
                args.base_manifest,
                args.asset_root,
                **kwargs,
            )
        else:
            payload = core.verify_structural_hypothesis_report_adoption(
                args.publication_root,
                args.adoption_contract,
                args.adoption_root,
                args.base_evidence_csv,
                args.attempt_root,
                args.hypothesis_contract,
                args.executor_contract,
                args.runtime_contract,
                args.publisher_contract,
                args.base_manifest,
                args.asset_root,
                expected_adoption_digest=args.expected_adoption_digest,
                **kwargs,
            )
        if payload is True:
            payload = {
                "status": _VERIFIED_STATUS,
                "planning_status": "NOT_PLANNED",
            }
        if type(payload) is not dict:
            raise TypeError("report adoption core returned a non-object")
        expected_status = (
            _SUCCESS_STATUS if args.action == "adopt" else _VERIFIED_STATUS
        )
        if payload.get("status") != expected_status:
            raise ValueError("report adoption result status differs from V1")
        if payload.get("planning_status") != "NOT_PLANNED":
            raise ValueError("report adoption unexpectedly reports planning")
        if args.action == "adopt":
            _write_canonical_summary(_adopt_output(payload, args))
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
            f"invalid structural-hypothesis report adoption: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
