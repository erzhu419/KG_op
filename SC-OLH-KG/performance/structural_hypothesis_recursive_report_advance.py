"""Advance one verified successor receipt into an immutable local report.

This module is deliberately a terminal, local-only bridge.  It verifies the
already-adopted report, its materialized successor, the deterministic bridge
authorization, and the completed runtime attempt before reusing the frozen
``reingest_successful_receipts`` API.  It never executes, schedules, plans, or
selects a current report.  ``advance.json`` is the last and only commit marker.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping, Sequence

from . import structural_hypothesis_adopted_successor_materializer as _successor
from . import structural_hypothesis_execution as _execution
from . import structural_hypothesis_report_adoption as _adoption
from . import structural_hypothesis_reingestion_publisher as _publisher
from . import structural_hypothesis_single_task_runtime as _runtime
from . import structural_hypothesis_successor_bound_single_task as _bridge
from .structural_hypothesis_loop import (
    canonical_json_bytes,
    verify_report_integrity,
)


ADVANCE_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-recursive-report-advance/1"
)
ADVANCE_CAPSULE_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-recursive-report-advance-capsule/1"
)
ADVANCE_CONTRACT_ID = "structural_hypothesis_recursive_report_advance_v1"

_ADVANCE_CONTRACT_DIGEST = (
    "sha256:54bfe070aeb4cca2c86ef50110000af3be23fe9026fecd56d9829d8eb480b519"
)
_ADOPTION_CONTRACT_DIGEST = (
    "sha256:66ed58df701c5e39924aa7b61574eb63126770290b1d9d1fabb60a9ebf8e13a1"
)
_SUCCESSOR_CONTRACT_DIGEST = (
    "sha256:6a8380ee1d42b188f7bbd4818adc93949ed6f31c5981ddf6a17691d147043411"
)
_BRIDGE_CONTRACT_DIGEST = (
    "sha256:3969b99401f92ce7d193f8e6fdd52d3fe4f63b048da539fce2a17a2c9718baf1"
)
_RUNTIME_CONTRACT_DIGEST = (
    "sha256:d03529c64e6ea63b9997ded35fb6c0b44c6e17fb828f9e9db8960adb764a8c6b"
)
_HYPOTHESIS_CONTRACT_DIGEST = (
    "sha256:4242f6af8424acca5c93136f0d4eb354f8c2203431f1c5145290c4a3f248cf26"
)
_EXECUTOR_CONTRACT_DIGEST = (
    "sha256:ede48b8b1fb0bb788f91a3834d5a41f336e55b331183922237176aec12624030"
)

_SOURCE_FILES = {
    "performance/structural_hypothesis_loop.py": (
        "f23a29c2f85b9bdce96398c6b901190fc34c2e2f97880dce5dbf2594b76c7635"
    ),
    "performance/structural_hypothesis_execution.py": (
        "7c5cc27f8e97da9b51e57975f63e860a23463d5e7728d1089f32978146f27c9b"
    ),
    "performance/structural_hypothesis_report_adoption.py": (
        "78acadfcd6b094cd6521ef29b49f164b58b0e6816cc257fe7dc7709ea37aa725"
    ),
    "performance/structural_hypothesis_adopted_successor_materializer.py": (
        "0d3fd169e31937f597a2a1b71d1106013125977f9cb197a8218e4357056fe55f"
    ),
    "performance/structural_hypothesis_reingestion_publisher.py": (
        "05d2ca6668414167cbe56a042f9564ff56728faf73ff5dd6fb167e9a16c3dd4d"
    ),
    "performance/structural_hypothesis_single_task_runtime.py": (
        "618c336aee3558efb05a5415201cdf6b5cb1a7e028b90d94f2ac204b072e7fe4"
    ),
    "performance/structural_hypothesis_successor_bound_single_task.py": (
        "472cbf62ef86ecdc3527689a49858cf935b78e1936f57346571d875acadd6763"
    ),
}

_STATE_PREFIX = Path("kg-op/structural-hypothesis-recursive-report-advance/v1")
_ADVANCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TASK_ID = re.compile(r"task:[0-9a-f]{24}\Z")
_COMMIT_STATUS = (
    "ADVANCED_AS_IMMUTABLE_LOCAL_REPORT_VERSION_NOT_CURRENT_NOT_PLANNED"
)
_VERIFIED_STATUS = "VERIFIED_" + _COMMIT_STATUS
_SOURCE_ROWS = 1351
_OUTPUT_ROWS = 1352
_SOURCE_PENDING = 29
_OUTPUT_PENDING = 28

_ARTIFACT_LAYOUT = {
    "advance_contract": "advance_contract.json",
    "source_directory": "source",
    "source_adoption": "source/adoption.json",
    "source_successor": "source/successor.json",
    "source_execution_directory": "source/execution",
    "source_execution_attempt": "source/execution/attempt.json",
    "source_execution_authorization": "source/execution/authorization.json",
    "source_execution_receipt": "source/execution/receipt.json",
    "source_execution_journal_directory": "source/execution/journal",
    "source_execution_completed_event": (
        "source/execution/journal/0002_COMPLETED.json"
    ),
    "combined_rows": "combined_rows.json",
    "output_report": "output_report.json",
    "reingestion_receipt": "reingestion_receipt.json",
    "advance_commit": "advance.json",
}
_DIRECTORIES = {
    _ARTIFACT_LAYOUT["source_directory"],
    _ARTIFACT_LAYOUT["source_execution_directory"],
    _ARTIFACT_LAYOUT["source_execution_journal_directory"],
}
_NONCLAIMS = [
    "immutable_local_report_version_is_not_global_current",
    "immutable_local_report_version_is_not_planned",
    "local_confirmation_is_not_external_authority",
    "local_confirmation_is_not_scientific_confirmation",
    "local_confirmation_is_not_subsequent_execution_authorization",
    "single_success_is_not_scientific_confirmation",
    "single_success_is_not_scientific_refutation",
    "output_report_may_retain_evidence_gaps",
    "reingestion_is_not_external_verification",
    "local_digest_is_not_signature",
    "no_external_authority",
    "no_currentness_claim",
    "no_network_access",
    "no_scheduler_access",
    "no_credential_access",
    "no_shell_execution",
    "no_task_materialization",
    "no_task_authorization",
    "no_task_execution",
    "no_benchmark_execution",
    "no_run_one_invocation",
    "no_automatic_successor_creation",
    "no_paper_promotion",
]

# Capture the audited implementation surfaces at import time.  In particular,
# no executor callable is imported by this module.
_DERIVE_SUCCESSOR_BOUND = _bridge._derive
_ASSERT_RUNTIME_CHAIN = _bridge._assert_runtime_chain
_VERIFY_COMPLETED_ATTEMPT = _runtime.verify_single_task_attempt
_REINGEST_SUCCESSFUL_RECEIPTS = _execution.reingest_successful_receipts
_VERIFY_REINGESTION_INTEGRITY = _execution.verify_reingestion_integrity
_CAPTURE_ATTEMPT = _publisher._attempt_snapshots
_VERIFY_CAPTURED_ATTEMPT = _publisher._validate_captured_runtime_capsule


class RecursiveReportAdvanceError(ValueError):
    """Raised when the recursive local report advance fails closed."""


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _raw_digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _canonical_file(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _json_clone(value: Any, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RecursiveReportAdvanceError(
            f"{label} is not strict JSON"
        ) from error


def _require_digest(value: Any, label: str) -> str:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise RecursiveReportAdvanceError(
            f"{label} must be a lowercase sha256 digest"
        )
    return value


def _absolute(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise RecursiveReportAdvanceError(f"{label} must be an absolute path")
    normalized = Path(os.path.normpath(os.fspath(path)))
    if path != normalized:
        raise RecursiveReportAdvanceError(
            f"{label} must be a canonical absolute path"
        )
    return path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_regular(
    path: Path, label: str, *, exact_mode: int | None = None
) -> bytes:
    candidate = _absolute(path, label)
    _reject_symlink_components(candidate)
    try:
        observed = candidate.lstat()
    except OSError as error:
        raise RecursiveReportAdvanceError(f"cannot inspect {label}") from error
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise RecursiveReportAdvanceError(f"{label} is not a regular file")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise RecursiveReportAdvanceError(f"cannot open {label}") from error
    try:
        before = os.fstat(descriptor)
        mode = stat.S_IMODE(before.st_mode)
        observed_identity = (
            observed.st_dev,
            observed.st_ino,
            observed.st_mode,
            observed.st_uid,
            observed.st_nlink,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
        )
        opened_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or opened_identity != observed_identity
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or mode & 0o022
            or exact_mode is not None
            and mode != exact_mode
        ):
            raise RecursiveReportAdvanceError(
                f"{label} ownership, type, mode, or link count differs"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        stable_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        stable_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        raw = b"".join(chunks)
        if stable_before != stable_after:
            raise RecursiveReportAdvanceError(f"{label} changed while read")
        if len(raw) != before.st_size:
            raise RecursiveReportAdvanceError(f"{label} byte count changed")
        return raw
    finally:
        os.close(descriptor)


def _parse_json(raw: bytes, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise RecursiveReportAdvanceError(f"{label} is not strict JSON") from error


def _read_json(
    path: Path, label: str, *, exact_mode: int | None = None
) -> tuple[Any, bytes]:
    raw = _read_regular(path, label, exact_mode=exact_mode)
    return _parse_json(raw, label), raw


def _state_base() -> Path:
    configured = os.environ.get("XDG_STATE_HOME")
    if configured:
        base = Path(configured)
        if not base.is_absolute():
            raise RecursiveReportAdvanceError(
                "XDG_STATE_HOME must be absolute"
            )
        return base
    return Path.home() / ".local/state"


def _secure_directory(path: Path, *, exact_mode: int | None = None) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise RecursiveReportAdvanceError(
            f"missing advance directory: {path}"
        ) from error
    mode = stat.S_IMODE(info.st_mode)
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or mode & 0o022
        or exact_mode is not None
        and mode != exact_mode
    ):
        raise RecursiveReportAdvanceError(
            f"advance directory ownership or mode differs: {path}"
        )


def _reject_symlink_components(path: Path) -> None:
    cursor = path
    while True:
        if cursor.is_symlink():
            raise RecursiveReportAdvanceError(
                f"advance path alias is forbidden: {cursor}"
            )
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent


def _ensure_secure_parent(target: Path) -> None:
    base = _state_base()
    prefix = base / _STATE_PREFIX
    if target.parent != prefix:
        raise RecursiveReportAdvanceError(
            "advance_root is outside the frozen local state prefix"
        )
    _reject_symlink_components(target)
    missing: list[Path] = []
    cursor = prefix
    while not cursor.exists():
        if cursor.is_symlink():
            raise RecursiveReportAdvanceError(
                f"advance path alias is forbidden: {cursor}"
            )
        missing.append(cursor)
        if cursor == base:
            break
        cursor = cursor.parent
    if not cursor.exists():
        raise RecursiveReportAdvanceError("state-home ancestor is missing")
    _secure_directory(cursor)
    for path in reversed(missing):
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
        os.chmod(path, 0o700, follow_symlinks=False)
    chain = [prefix]
    cursor = prefix.parent
    while cursor != base and base in cursor.parents:
        chain.append(cursor)
        cursor = cursor.parent
    chain.append(base)
    for path in chain:
        _secure_directory(path)


def _validate_advance_location(
    advance_root: str | Path, advance_id: str, *, fresh: bool
) -> Path:
    root = _absolute(advance_root, "advance_root")
    if type(advance_id) is not str or not _ADVANCE_ID.fullmatch(advance_id):
        raise RecursiveReportAdvanceError("advance_id is invalid")
    if root.name != advance_id:
        raise RecursiveReportAdvanceError(
            "advance_id must equal advance_root basename"
        )
    if root.parent != _state_base() / _STATE_PREFIX:
        raise RecursiveReportAdvanceError(
            "advance_root is outside the frozen local state prefix"
        )
    if fresh:
        if root.exists() or root.is_symlink():
            raise RecursiveReportAdvanceError("advance_root already exists")
        _ensure_secure_parent(root)
    else:
        _reject_symlink_components(root)
        _secure_directory(root, exact_mode=0o700)
    return root


def _mkdir_new(path: Path) -> None:
    try:
        os.mkdir(path, 0o700)
        os.chmod(path, 0o700, follow_symlinks=False)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise RecursiveReportAdvanceError(
            f"cannot create fresh advance directory: {path}"
        ) from error
    _secure_directory(path, exact_mode=0o700)


def _write_new_bytes(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RecursiveReportAdvanceError(
            f"advance artifact already exists: {path}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _artifact(path: str, raw: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": _raw_digest(raw), "bytes": len(raw)}


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        not isinstance(contract, Mapping)
        or contract.get("schema_version") != ADVANCE_SCHEMA_VERSION
        or contract.get("contract_id") != ADVANCE_CONTRACT_ID
        or contract.get("source_files") != _SOURCE_FILES
        or contract.get("artifact_layout") != _ARTIFACT_LAYOUT
        or contract.get("nonclaims") != _NONCLAIMS
        or _digest(contract) != _ADVANCE_CONTRACT_DIGEST
    ):
        raise RecursiveReportAdvanceError(
            "recursive report advance contract differs from frozen V1"
        )
    if contract.get("source_contracts") != {
        "adoption_contract_id": _adoption.ADOPTION_CONTRACT_ID,
        "adoption_contract_digest": _ADOPTION_CONTRACT_DIGEST,
        "successor_contract_id": _successor.SUCCESSOR_CONTRACT_ID,
        "successor_contract_digest": _SUCCESSOR_CONTRACT_DIGEST,
        "bridge_contract_id": _bridge.BRIDGE_CONTRACT_ID,
        "bridge_contract_digest": _BRIDGE_CONTRACT_DIGEST,
        "runtime_contract_id": "structural_hypothesis_single_task_runtime_v1",
        "runtime_contract_digest": _RUNTIME_CONTRACT_DIGEST,
        "hypothesis_contract_id": "structural_hypothesis_loop_v1",
        "hypothesis_contract_digest": _HYPOTHESIS_CONTRACT_DIGEST,
        "executor_contract_id": "structural_hypothesis_executor_v1",
        "executor_contract_digest": _EXECUTOR_CONTRACT_DIGEST,
        "reingestion_schema_version": _execution.REINGESTION_SCHEMA_VERSION,
    }:
        raise RecursiveReportAdvanceError(
            "recursive report advance source contracts differ"
        )
    admission = contract.get("admission")
    required_admission = {
        "source_adoption_status": (
            "ADOPTED_AS_LOCAL_REPORT_VERSION_NOT_PLANNED"
        ),
        "source_successor_status": (
            "SUCCESSOR_MATERIALIZED_FROM_VERIFIED_ADOPTION_NOT_AUTHORIZED"
        ),
        "completed_runtime_status": "VERIFIED_COMPLETED",
        "full_adoption_and_successor_verification_required": True,
        "deterministic_successor_bridge_rederivation_required": True,
        "independent_adoption_digest_required": True,
        "independent_pending_evidence_digest_required": True,
        "independent_first_pending_projection_digest_required": True,
        "independent_successor_digest_required": True,
        "independent_bundle_digest_required": True,
        "independent_plan_digest_required": True,
        "independent_task_digest_required": True,
        "independent_provenance_binding_digest_required": True,
        "independent_authorization_digest_required": True,
        "independent_attempt_digest_required": True,
        "independent_execution_receipt_digest_required": True,
        "independent_execution_journal_head_digest_required": True,
        "required_authorization_id": (
            "successor-bound-v1:<provenance-binding-hex>"
        ),
        "execution_receipt_count": 1,
        "authorized_task_count": 1,
        "successful_result_count": 1,
        "failed_result_count": 0,
        "accepted_successful_rows": 1,
        "ignored_failed_attempts": 0,
        "source_typed_row_count": _SOURCE_ROWS,
        "output_typed_row_count": _OUTPUT_ROWS,
        "source_pending_count": _SOURCE_PENDING,
        "output_pending_count": _OUTPUT_PENDING,
        "explicit_local_confirmation_required": True,
        "committed_status": _COMMIT_STATUS,
        "current_status": "NOT_CURRENT",
        "planning_status": "NOT_PLANNED",
    }
    if admission != required_admission:
        raise RecursiveReportAdvanceError(
            "recursive report advance admission differs"
        )
    mechanics = contract.get("mechanics")
    if not isinstance(mechanics, Mapping) or mechanics != {
        "local_files_only": True,
        "immutable_named_report_version_only": True,
        "global_current_pointer_written": False,
        "planning_performed": False,
        "network_access": False,
        "scheduler_access": False,
        "credential_access": False,
        "shell_execution": False,
        "task_materialization": False,
        "task_authorization": False,
        "task_execution": False,
        "benchmark_execution": False,
        "run_one_invocation": False,
        "automatic_successor_creation": False,
    }:
        raise RecursiveReportAdvanceError(
            "recursive report advance mechanics differ"
        )
    for relative, expected in _SOURCE_FILES.items():
        raw = _read_regular(_repo_root() / relative, f"source file {relative}")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RecursiveReportAdvanceError(
                f"recursive report advance source file differs: {relative}"
            )


def _expected_digests(**values: str) -> dict[str, str]:
    return {
        label: _require_digest(value, label)
        for label, value in values.items()
    }


def _source_raws(
    derived: Mapping[str, Any], attempt_raws: Mapping[str, bytes]
) -> dict[str, bytes]:
    captured_adoption = derived["captured_adoption"]
    captured_successor = derived["captured_successor"]
    sources = {
        "source_adoption": captured_adoption["raws"]["adoption_commit"],
        "source_successor": captured_successor["raws"]["successor_commit"],
    }
    for label, source_label in {
        "source_execution_attempt": "execution_attempt",
        "source_execution_authorization": "execution_authorization",
        "source_execution_receipt": "execution_receipt",
        "source_execution_completed_event": "execution_completed_event",
    }.items():
        raw = attempt_raws.get(source_label)
        if type(raw) is not bytes:
            raise RecursiveReportAdvanceError(
                f"captured attempt lacks {source_label} bytes"
            )
        sources[label] = raw
    return sources


def _validate_generation(
    publication_root,
    adoption_contract_path,
    adoption_root,
    successor_contract_path,
    successor_root,
    base_evidence_csv,
    source_attempt_root,
    hypothesis_contract_path,
    executor_contract_path,
    runtime_contract_path,
    publisher_contract_path,
    materializer_contract_path,
    bridge_contract_path,
    base_manifest_path,
    asset_root,
    completed_attempt_root,
    *,
    adoption_id: str,
    successor_id: str,
    expected: Mapping[str, str],
    task_id: str,
) -> dict[str, Any]:
    if type(task_id) is not str or not _TASK_ID.fullmatch(task_id):
        raise RecursiveReportAdvanceError("task_id is invalid")
    completed_root = _absolute(
        completed_attempt_root, "completed_attempt_root"
    )
    try:
        attempt_objects, attempt_raws = _CAPTURE_ATTEMPT(completed_root)
        derived = _DERIVE_SUCCESSOR_BOUND(
            publication_root,
            adoption_contract_path,
            adoption_root,
            successor_contract_path,
            successor_root,
            base_evidence_csv,
            source_attempt_root,
            hypothesis_contract_path,
            executor_contract_path,
            runtime_contract_path,
            publisher_contract_path,
            materializer_contract_path,
            bridge_contract_path,
            base_manifest_path,
            asset_root,
            completed_attempt_root,
            adoption_id=adoption_id,
            successor_id=successor_id,
            expected_adoption_digest=expected["expected_adoption_digest"],
            expected_pending_evidence_digest=expected[
                "expected_pending_evidence_digest"
            ],
            expected_first_pending_projection_digest=expected[
                "expected_first_pending_projection_digest"
            ],
            expected_successor_digest=expected[
                "expected_successor_digest"
            ],
            expected_bundle_digest=expected["expected_bundle_digest"],
            expected_plan_digest=expected["expected_plan_digest"],
            task_id=task_id,
            expected_task_digest=expected["expected_task_digest"],
            require_attempt_absent=False,
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise RecursiveReportAdvanceError(
            "source adoption/successor bridge failed full verification"
        ) from error
    if (
        derived.get("provenance_binding_digest")
        != expected["expected_provenance_binding_digest"]
        or derived.get("required_authorization_id")
        != "successor-bound-v1:"
        + expected["expected_provenance_binding_digest"].split(":", 1)[1]
    ):
        raise RecursiveReportAdvanceError(
            "deterministic bridge provenance or authorization ID differs"
        )
    try:
        chain = _ASSERT_RUNTIME_CHAIN(
            derived,
            expected_authorization_digest=expected[
                "expected_authorization_digest"
            ],
        )
        _VERIFY_CAPTURED_ATTEMPT(
            attempt_objects,
            derived["runtime_contract"],
            _absolute(base_manifest_path, "base_manifest_path"),
            _absolute(asset_root, "asset_root"),
            completed_root,
            expected_plan_digest=expected["expected_plan_digest"],
            expected_authorization_digest=expected[
                "expected_authorization_digest"
            ],
            expected_execution_receipt_digest=expected[
                "expected_execution_receipt_digest"
            ],
            expected_execution_journal_head_digest=expected[
                "expected_execution_journal_head_digest"
            ],
            expected_execution_attempt_digest=expected[
                "expected_attempt_digest"
            ],
        )
        completed = _VERIFY_COMPLETED_ATTEMPT(
            completed_attempt_root,
            derived["runtime_contract"],
            base_manifest_path,
            asset_root,
            expected_authorization_digest=expected[
                "expected_authorization_digest"
            ],
            expected_receipt_digest=expected[
                "expected_execution_receipt_digest"
            ],
            expected_journal_head_digest=expected[
                "expected_execution_journal_head_digest"
            ],
            expected_attempt_digest=expected["expected_attempt_digest"],
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise RecursiveReportAdvanceError(
            "completed successor-bound runtime chain failed verification"
        ) from error
    if completed.get("status") != "VERIFIED_COMPLETED":
        raise RecursiveReportAdvanceError(
            "source attempt is not independently verified COMPLETED"
        )
    authorization = chain.get("authorization")
    receipt = attempt_objects.get("execution_receipt")
    attempt = chain.get("attempt")
    captured_chain = {
        "attempt": "execution_attempt",
        "bundle": "execution_bundle",
        "authorization": "execution_authorization",
        "report": "source_report",
        "hypothesis_contract": "execution_hypothesis_contract",
        "executor_contract": "execution_executor_contract",
        "materializer_contract": "execution_materializer_contract",
    }
    if any(
        chain.get(chain_label) != attempt_objects.get(snapshot_label)
        for chain_label, snapshot_label in captured_chain.items()
    ):
        raise RecursiveReportAdvanceError(
            "captured runtime bytes differ from successor-bound runtime chain"
        )
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("authorization_id")
        != derived["required_authorization_id"]
        or authorization.get("integrity", {}).get("authorization_digest")
        != expected["expected_authorization_digest"]
        or not isinstance(attempt, Mapping)
        or attempt.get("integrity", {}).get("attempt_digest")
        != expected["expected_attempt_digest"]
        or not isinstance(receipt, Mapping)
        or receipt.get("integrity", {}).get("receipt_digest")
        != expected["expected_execution_receipt_digest"]
    ):
        raise RecursiveReportAdvanceError(
            "completed attempt differs from deterministic bridge authorization"
        )
    captured = derived["captured_adoption"]["publication_values"]
    base_rows = captured.get("combined_rows")
    source_report = captured.get("output_report")
    hypothesis_contract = captured.get("source_hypothesis_contract")
    executor_contract = captured.get("source_executor_contract")
    plan = derived["bundle"].get("plan")
    if (
        type(base_rows) is not list
        or len(base_rows) != _SOURCE_ROWS
        or not isinstance(source_report, Mapping)
        or source_report.get("status") != "COMPLETED_WITH_EVIDENCE_GAPS"
        or not verify_report_integrity(source_report)
        or _digest(base_rows) != source_report.get("evidence_digest")
        or type(source_report.get("pending_evidence")) is not list
        or len(source_report["pending_evidence"]) != _SOURCE_PENDING
        or not isinstance(plan, Mapping)
    ):
        raise RecursiveReportAdvanceError(
            "adopted typed evidence or source report differs from V1 admission"
        )
    task = derived.get("task")
    first_pending = source_report["pending_evidence"][0]
    if (
        not isinstance(task, Mapping)
        or task.get("task_id") != task_id
        or task.get("task_digest") != expected["expected_task_digest"]
        or any(
            first_pending.get(key) != value
            for key, value in task.get("cell", {}).items()
            if key != "line"
        )
    ):
        raise RecursiveReportAdvanceError(
            "completed task is not the exact first pending adopted cell"
        )
    results = receipt.get("results")
    summary = receipt.get("summary")
    if (
        receipt.get("status") != "COMPLETED"
        or summary != {"authorized": 1, "failed": 0, "succeeded": 1}
        or type(results) is not list
        or len(results) != 1
        or results[0].get("status") != "SUCCEEDED"
        or results[0].get("task_id") != task_id
        or results[0].get("task_digest")
        != expected["expected_task_digest"]
        or type(results[0].get("evidence_row")) is not dict
    ):
        raise RecursiveReportAdvanceError(
            "completed receipt is not exactly one successful authorized result"
        )
    try:
        advanced = _REINGEST_SUCCESSFUL_RECEIPTS(
            base_rows,
            [receipt],
            hypothesis_contract,
            source_report,
            plan=plan,
            authorization=authorization,
            executor_contract=executor_contract,
        )
    except (ValueError, KeyError, TypeError) as error:
        raise RecursiveReportAdvanceError(
            "typed successor receipt reingestion failed"
        ) from error
    output_report = advanced.get("report")
    reingestion_receipt = advanced.get("reingestion_receipt")
    appended_row = _json_clone(results[0]["evidence_row"], "evidence row")
    combined_rows = _json_clone(
        [*base_rows, appended_row], "combined typed evidence"
    )
    if (
        type(combined_rows) is not list
        or len(combined_rows) != _OUTPUT_ROWS
        or combined_rows[:-1] != base_rows
        or combined_rows[-1] != results[0]["evidence_row"]
        or not isinstance(output_report, Mapping)
        or not verify_report_integrity(output_report)
        or output_report.get("evidence_digest") != _digest(combined_rows)
        or type(output_report.get("pending_evidence")) is not list
        or len(output_report["pending_evidence"]) != _OUTPUT_PENDING
        or output_report["pending_evidence"]
        != source_report["pending_evidence"][1:]
        or not isinstance(reingestion_receipt, Mapping)
        or reingestion_receipt.get("accepted_successful_rows") != 1
        or reingestion_receipt.get("ignored_failed_attempts") != 0
        or reingestion_receipt.get("combined_evidence_digest")
        != output_report.get("evidence_digest")
        or not _VERIFY_REINGESTION_INTEGRITY(
            reingestion_receipt,
            source_report=source_report,
            base_rows=base_rows,
            plan=plan,
            authorization=authorization,
            receipts=[receipt],
            output_report=output_report,
            hypothesis_contract=hypothesis_contract,
            executor_contract=executor_contract,
        )
    ):
        raise RecursiveReportAdvanceError(
            "advanced report does not satisfy the exact 1352/28 transition"
        )
    return {
        "derived": derived,
        "combined_rows": combined_rows,
        "output_report": _json_clone(output_report, "output report"),
        "reingestion_receipt": _json_clone(
            reingestion_receipt, "reingestion receipt"
        ),
        "source_raws": _source_raws(derived, attempt_raws),
        "appended_evidence_row_digest": _digest(appended_row),
    }


def _same_generation(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    for key in (
        "combined_rows",
        "output_report",
        "reingestion_receipt",
        "source_raws",
        "appended_evidence_row_digest",
    ):
        if first[key] != second[key]:
            raise RecursiveReportAdvanceError(
                "source generation changed before local report commit"
            )
    for key in (
        "provenance_binding",
        "provenance_binding_digest",
        "required_authorization_id",
        "bundle",
        "task",
    ):
        if first["derived"][key] != second["derived"][key]:
            raise RecursiveReportAdvanceError(
                "successor provenance changed before local report commit"
            )


def _artifact_raws(
    contract_raw: bytes, generation: Mapping[str, Any]
) -> dict[str, bytes]:
    return {
        "advance_contract": contract_raw,
        **generation["source_raws"],
        "combined_rows": _canonical_file(generation["combined_rows"]),
        "output_report": _canonical_file(generation["output_report"]),
        "reingestion_receipt": _canonical_file(
            generation["reingestion_receipt"]
        ),
    }


def _expected_names(relative: str, *, committed: bool) -> set[str]:
    if relative == ".":
        names = {
            "advance_contract.json",
            "source",
            "combined_rows.json",
            "output_report.json",
            "reingestion_receipt.json",
        }
        if committed:
            names.add("advance.json")
        return names
    if relative == "source":
        return {"adoption.json", "successor.json", "execution"}
    if relative == "source/execution":
        return {"attempt.json", "authorization.json", "receipt.json", "journal"}
    if relative == "source/execution/journal":
        return {"0002_COMPLETED.json"}
    raise AssertionError(relative)


def _layout(root: Path, *, committed: bool) -> None:
    for relative in (".", "source", "source/execution", "source/execution/journal"):
        path = root if relative == "." else root / relative
        _secure_directory(path, exact_mode=0o700)
        try:
            names = {entry.name for entry in path.iterdir()}
        except OSError as error:
            raise RecursiveReportAdvanceError(
                f"cannot inspect advance directory: {path}"
            ) from error
        if names != _expected_names(relative, committed=committed):
            raise RecursiveReportAdvanceError(
                f"advance artifact layout differs: {relative}"
            )
    for label, relative in _ARTIFACT_LAYOUT.items():
        if label.endswith("directory"):
            continue
        if label == "advance_commit" and not committed:
            continue
        _read_regular(root / relative, label.replace("_", " "), exact_mode=0o600)


def _source_binding(
    generation: Mapping[str, Any], expected: Mapping[str, str], task_id: str
) -> dict[str, Any]:
    derived = generation["derived"]
    return {
        "adoption_digest": expected["expected_adoption_digest"],
        "successor_digest": expected["expected_successor_digest"],
        "pending_evidence_digest": expected[
            "expected_pending_evidence_digest"
        ],
        "first_pending_projection_digest": expected[
            "expected_first_pending_projection_digest"
        ],
        "bundle_digest": expected["expected_bundle_digest"],
        "plan_digest": expected["expected_plan_digest"],
        "task_id": task_id,
        "task_digest": expected["expected_task_digest"],
        "provenance_binding": derived["provenance_binding"],
        "provenance_binding_digest": expected[
            "expected_provenance_binding_digest"
        ],
        "required_authorization_id": derived["required_authorization_id"],
        "authorization_digest": expected["expected_authorization_digest"],
        "attempt_digest": expected["expected_attempt_digest"],
        "execution_receipt_digest": expected[
            "expected_execution_receipt_digest"
        ],
        "execution_journal_head_digest": expected[
            "expected_execution_journal_head_digest"
        ],
    }


def _marker_body(
    *,
    advance_id: str,
    contract: Mapping[str, Any],
    generation: Mapping[str, Any],
    expected: Mapping[str, str],
    task_id: str,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    report = generation["output_report"]
    reingestion = generation["reingestion_receipt"]
    return {
        "schema_version": ADVANCE_CAPSULE_SCHEMA_VERSION,
        "status": _COMMIT_STATUS,
        "advance_id": advance_id,
        "current_status": "NOT_CURRENT",
        "planning_status": "NOT_PLANNED",
        "confirmation": {
            "kind": "LOCAL_EXPLICIT_CONFIRMATION_NOT_AUTHORITY",
            "immutable_local_report_advance": True,
            "current_status": "NOT_CURRENT",
            "planning_status": "NOT_PLANNED",
        },
        "advance_contract_binding": {
            "contract_id": contract["contract_id"],
            "contract_digest": _digest(contract),
        },
        "source_binding": _source_binding(generation, expected, task_id),
        "transition": {
            "source_typed_row_count": _SOURCE_ROWS,
            "output_typed_row_count": _OUTPUT_ROWS,
            "source_pending_count": _SOURCE_PENDING,
            "output_pending_count": _OUTPUT_PENDING,
            "accepted_successful_rows": 1,
            "ignored_failed_attempts": 0,
            "task_id": task_id,
            "task_digest": expected["expected_task_digest"],
            "appended_evidence_row_digest": generation[
                "appended_evidence_row_digest"
            ],
        },
        "adopted_report": {
            "schema_version": report["schema_version"],
            "status": report["status"],
            "evidence_digest": report["evidence_digest"],
            "report_body_digest": report["audit"]["report_body_digest"],
            "audit_head": report["audit"]["head"],
            "reingestion_digest": reingestion["integrity"][
                "reingestion_digest"
            ],
            "typed_row_count": _OUTPUT_ROWS,
            "pending_evidence_count": _OUTPUT_PENDING,
            "combined_rows_artifact": "combined_rows.json",
            "report_artifact": "output_report.json",
        },
        "artifacts": dict(artifacts),
        "nonclaims": list(_NONCLAIMS),
    }


def _result(marker: Mapping[str, Any], root: Path, *, verified: bool) -> dict[str, Any]:
    adopted = marker["adopted_report"]
    return {
        "status": _VERIFIED_STATUS if verified else _COMMIT_STATUS,
        "advance_root": str(root),
        "advance_digest": marker["integrity"]["advance_digest"],
        "reingestion_digest": adopted["reingestion_digest"],
        "output_report_body_digest": adopted["report_body_digest"],
        "output_audit_head": adopted["audit_head"],
        "output_evidence_digest": adopted["evidence_digest"],
        "typed_row_count": adopted["typed_row_count"],
        "pending_evidence_count": adopted["pending_evidence_count"],
        "current_status": marker["current_status"],
        "planning_status": marker["planning_status"],
    }


def _common_expected(
    *,
    expected_adoption_digest,
    expected_pending_evidence_digest,
    expected_first_pending_projection_digest,
    expected_successor_digest,
    expected_bundle_digest,
    expected_plan_digest,
    expected_task_digest,
    expected_provenance_binding_digest,
    expected_authorization_digest,
    expected_attempt_digest,
    expected_execution_receipt_digest,
    expected_execution_journal_head_digest,
) -> dict[str, str]:
    return _expected_digests(
        expected_adoption_digest=expected_adoption_digest,
        expected_pending_evidence_digest=expected_pending_evidence_digest,
        expected_first_pending_projection_digest=(
            expected_first_pending_projection_digest
        ),
        expected_successor_digest=expected_successor_digest,
        expected_bundle_digest=expected_bundle_digest,
        expected_plan_digest=expected_plan_digest,
        expected_task_digest=expected_task_digest,
        expected_provenance_binding_digest=(
            expected_provenance_binding_digest
        ),
        expected_authorization_digest=expected_authorization_digest,
        expected_attempt_digest=expected_attempt_digest,
        expected_execution_receipt_digest=(
            expected_execution_receipt_digest
        ),
        expected_execution_journal_head_digest=(
            expected_execution_journal_head_digest
        ),
    )


def advance_recursive_report_version(
    publication_root,
    adoption_contract_path,
    adoption_root,
    successor_contract_path,
    successor_root,
    base_evidence_csv,
    source_attempt_root,
    hypothesis_contract_path,
    executor_contract_path,
    runtime_contract_path,
    publisher_contract_path,
    materializer_contract_path,
    bridge_contract_path,
    base_manifest_path,
    asset_root,
    completed_attempt_root,
    advance_contract_path,
    advance_root,
    *,
    advance_id: str,
    adoption_id: str,
    successor_id: str,
    expected_adoption_digest: str,
    expected_pending_evidence_digest: str,
    expected_first_pending_projection_digest: str,
    expected_successor_digest: str,
    expected_bundle_digest: str,
    expected_plan_digest: str,
    task_id: str,
    expected_task_digest: str,
    expected_provenance_binding_digest: str,
    expected_authorization_digest: str,
    expected_attempt_digest: str,
    expected_execution_receipt_digest: str,
    expected_execution_journal_head_digest: str,
    confirm_immutable_local_report_advance: bool,
) -> dict[str, Any]:
    """Commit the exact next typed report without selecting it as current."""
    if confirm_immutable_local_report_advance is not True:
        raise RecursiveReportAdvanceError(
            "explicit immutable local report advance confirmation is required"
        )
    contract_path = _absolute(advance_contract_path, "advance_contract_path")
    contract, contract_raw = _read_json(
        contract_path, "recursive report advance contract"
    )
    _validate_contract(contract)
    expected = _common_expected(
        expected_adoption_digest=expected_adoption_digest,
        expected_pending_evidence_digest=expected_pending_evidence_digest,
        expected_first_pending_projection_digest=(
            expected_first_pending_projection_digest
        ),
        expected_successor_digest=expected_successor_digest,
        expected_bundle_digest=expected_bundle_digest,
        expected_plan_digest=expected_plan_digest,
        expected_task_digest=expected_task_digest,
        expected_provenance_binding_digest=(
            expected_provenance_binding_digest
        ),
        expected_authorization_digest=expected_authorization_digest,
        expected_attempt_digest=expected_attempt_digest,
        expected_execution_receipt_digest=(
            expected_execution_receipt_digest
        ),
        expected_execution_journal_head_digest=(
            expected_execution_journal_head_digest
        ),
    )
    common = (
        publication_root,
        adoption_contract_path,
        adoption_root,
        successor_contract_path,
        successor_root,
        base_evidence_csv,
        source_attempt_root,
        hypothesis_contract_path,
        executor_contract_path,
        runtime_contract_path,
        publisher_contract_path,
        materializer_contract_path,
        bridge_contract_path,
        base_manifest_path,
        asset_root,
        completed_attempt_root,
    )
    first = _validate_generation(
        *common,
        adoption_id=adoption_id,
        successor_id=successor_id,
        expected=expected,
        task_id=task_id,
    )
    second = _validate_generation(
        *common,
        adoption_id=adoption_id,
        successor_id=successor_id,
        expected=expected,
        task_id=task_id,
    )
    _same_generation(first, second)
    root = _validate_advance_location(advance_root, advance_id, fresh=True)
    raws = _artifact_raws(contract_raw, second)
    artifacts = {
        label: _artifact(_ARTIFACT_LAYOUT[label], raw)
        for label, raw in raws.items()
    }
    _mkdir_new(root)
    _mkdir_new(root / "source")
    _mkdir_new(root / "source/execution")
    _mkdir_new(root / "source/execution/journal")
    for label, raw in raws.items():
        _write_new_bytes(root / _ARTIFACT_LAYOUT[label], raw)
    _layout(root, committed=False)
    for label, raw in raws.items():
        if _read_regular(
            root / _ARTIFACT_LAYOUT[label],
            f"staged {label}",
            exact_mode=0o600,
        ) != raw:
            raise RecursiveReportAdvanceError(
                f"staged advance artifact differs: {label}"
            )
    marker_body = _marker_body(
        advance_id=advance_id,
        contract=contract,
        generation=second,
        expected=expected,
        task_id=task_id,
        artifacts=artifacts,
    )
    marker = {
        **marker_body,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "advance_digest": _digest(marker_body),
        },
    }
    marker_raw = _canonical_file(marker)
    _write_new_bytes(root / "advance.json", marker_raw)
    _layout(root, committed=True)
    for label, raw in raws.items():
        if _read_regular(
            root / _ARTIFACT_LAYOUT[label],
            f"committed {label}",
            exact_mode=0o600,
        ) != raw:
            raise RecursiveReportAdvanceError(
                f"committed advance artifact differs: {label}"
            )
    if _read_regular(
        root / "advance.json", "committed advance marker", exact_mode=0o600
    ) != marker_raw:
        raise RecursiveReportAdvanceError("committed advance marker differs")
    return _result(marker, root, verified=False)


def verify_recursive_report_advance(
    publication_root,
    adoption_contract_path,
    adoption_root,
    successor_contract_path,
    successor_root,
    base_evidence_csv,
    source_attempt_root,
    hypothesis_contract_path,
    executor_contract_path,
    runtime_contract_path,
    publisher_contract_path,
    materializer_contract_path,
    bridge_contract_path,
    base_manifest_path,
    asset_root,
    completed_attempt_root,
    advance_contract_path,
    advance_root,
    *,
    advance_id: str,
    adoption_id: str,
    successor_id: str,
    expected_adoption_digest: str,
    expected_pending_evidence_digest: str,
    expected_first_pending_projection_digest: str,
    expected_successor_digest: str,
    expected_bundle_digest: str,
    expected_plan_digest: str,
    task_id: str,
    expected_task_digest: str,
    expected_provenance_binding_digest: str,
    expected_authorization_digest: str,
    expected_attempt_digest: str,
    expected_execution_receipt_digest: str,
    expected_execution_journal_head_digest: str,
    expected_advance_digest: str,
    expected_reingestion_digest: str,
    expected_output_report_body_digest: str,
    expected_output_audit_head: str,
    expected_output_evidence_digest: str,
) -> dict[str, Any]:
    """Read-only full-chain verification of one committed advance capsule."""
    contract_path = _absolute(advance_contract_path, "advance_contract_path")
    contract, contract_raw = _read_json(
        contract_path, "recursive report advance contract"
    )
    _validate_contract(contract)
    expected = _common_expected(
        expected_adoption_digest=expected_adoption_digest,
        expected_pending_evidence_digest=expected_pending_evidence_digest,
        expected_first_pending_projection_digest=(
            expected_first_pending_projection_digest
        ),
        expected_successor_digest=expected_successor_digest,
        expected_bundle_digest=expected_bundle_digest,
        expected_plan_digest=expected_plan_digest,
        expected_task_digest=expected_task_digest,
        expected_provenance_binding_digest=(
            expected_provenance_binding_digest
        ),
        expected_authorization_digest=expected_authorization_digest,
        expected_attempt_digest=expected_attempt_digest,
        expected_execution_receipt_digest=(
            expected_execution_receipt_digest
        ),
        expected_execution_journal_head_digest=(
            expected_execution_journal_head_digest
        ),
    )
    final_expected = _expected_digests(
        expected_advance_digest=expected_advance_digest,
        expected_reingestion_digest=expected_reingestion_digest,
        expected_output_report_body_digest=expected_output_report_body_digest,
        expected_output_audit_head=expected_output_audit_head,
        expected_output_evidence_digest=expected_output_evidence_digest,
    )
    common = (
        publication_root,
        adoption_contract_path,
        adoption_root,
        successor_contract_path,
        successor_root,
        base_evidence_csv,
        source_attempt_root,
        hypothesis_contract_path,
        executor_contract_path,
        runtime_contract_path,
        publisher_contract_path,
        materializer_contract_path,
        bridge_contract_path,
        base_manifest_path,
        asset_root,
        completed_attempt_root,
    )
    first = _validate_generation(
        *common,
        adoption_id=adoption_id,
        successor_id=successor_id,
        expected=expected,
        task_id=task_id,
    )
    second = _validate_generation(
        *common,
        adoption_id=adoption_id,
        successor_id=successor_id,
        expected=expected,
        task_id=task_id,
    )
    _same_generation(first, second)
    root = _validate_advance_location(advance_root, advance_id, fresh=False)
    _layout(root, committed=True)
    raws = _artifact_raws(contract_raw, second)
    artifacts = {
        label: _artifact(_ARTIFACT_LAYOUT[label], raw)
        for label, raw in raws.items()
    }
    for label, raw in raws.items():
        if _read_regular(
            root / _ARTIFACT_LAYOUT[label],
            f"capsule {label}",
            exact_mode=0o600,
        ) != raw:
            raise RecursiveReportAdvanceError(
                f"capsule advance artifact differs: {label}"
            )
    marker, marker_raw = _read_json(
        root / "advance.json", "advance commit", exact_mode=0o600
    )
    marker_body = _marker_body(
        advance_id=advance_id,
        contract=contract,
        generation=second,
        expected=expected,
        task_id=task_id,
        artifacts=artifacts,
    )
    expected_marker = {
        **marker_body,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "advance_digest": _digest(marker_body),
        },
    }
    if (
        marker != expected_marker
        or marker_raw != _canonical_file(expected_marker)
        or marker["integrity"]["advance_digest"]
        != final_expected["expected_advance_digest"]
        or marker["adopted_report"]["reingestion_digest"]
        != final_expected["expected_reingestion_digest"]
        or marker["adopted_report"]["report_body_digest"]
        != final_expected["expected_output_report_body_digest"]
        or marker["adopted_report"]["audit_head"]
        != final_expected["expected_output_audit_head"]
        or marker["adopted_report"]["evidence_digest"]
        != final_expected["expected_output_evidence_digest"]
    ):
        raise RecursiveReportAdvanceError(
            "advance marker or independent output anchors differ"
        )
    return _result(marker, root, verified=True)


__all__ = [
    "ADVANCE_CAPSULE_SCHEMA_VERSION",
    "ADVANCE_CONTRACT_ID",
    "ADVANCE_SCHEMA_VERSION",
    "RecursiveReportAdvanceError",
    "advance_recursive_report_version",
    "verify_recursive_report_advance",
]
