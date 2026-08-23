"""Materialize the next non-authorized successor from a verified advance.

The module is deliberately local and terminal.  It captures an immutable
recursive-report advance through held directory descriptors, runs the public
full-chain advance verifier, replays the captured report, and reuses the
frozen public task materializer.  It does not authorize, prepare, execute,
schedule, select a current report, or invoke ``run_one``.  ``successor.json``
is the last and only commit marker in a three-leaf capsule.
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

from . import structural_hypothesis_recursive_report_advance as _advance
from . import structural_hypothesis_task_materializer as _materializer
from .structural_hypothesis_loop import (
    canonical_json_bytes,
    run_structural_hypothesis_loop,
    verify_report_integrity,
)


RECURSIVE_SUCCESSOR_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-recursive-successor-materializer/1"
)
RECURSIVE_SUCCESSOR_CAPSULE_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-recursive-successor-materialization/1"
)
RECURSIVE_SUCCESSOR_CONTRACT_ID = (
    "structural_hypothesis_recursive_successor_materializer_v1"
)

_RECURSIVE_SUCCESSOR_CONTRACT_DIGEST = (
    "sha256:dfc58175e637df625758bcd77e41cacd11e1ad9d71a1b8224d66c51eec0cfd1c"
)
_ADVANCE_CONTRACT_DIGEST = (
    "sha256:54bfe070aeb4cca2c86ef50110000af3be23fe9026fecd56d9829d8eb480b519"
)
_MATERIALIZER_CONTRACT_DIGEST = (
    "sha256:30c65d77e6cbdbc13b95e9083604f6f99835b0982d52319b99d0040491c1d013"
)
_HYPOTHESIS_CONTRACT_DIGEST = (
    "sha256:4242f6af8424acca5c93136f0d4eb354f8c2203431f1c5145290c4a3f248cf26"
)
_EXECUTOR_CONTRACT_DIGEST = (
    "sha256:ede48b8b1fb0bb788f91a3834d5a41f336e55b331183922237176aec12624030"
)

_SOURCE_CONTRACTS = {
    "advance_contract_id": "structural_hypothesis_recursive_report_advance_v1",
    "advance_contract_digest": _ADVANCE_CONTRACT_DIGEST,
    "materializer_contract_id": "structural_hypothesis_task_materializer_v1",
    "materializer_contract_digest": _MATERIALIZER_CONTRACT_DIGEST,
    "hypothesis_contract_id": "structural_hypothesis_loop_v1",
    "hypothesis_contract_digest": _HYPOTHESIS_CONTRACT_DIGEST,
    "executor_contract_id": "structural_hypothesis_executor_v1",
    "executor_contract_digest": _EXECUTOR_CONTRACT_DIGEST,
}
_SOURCE_FILES = {
    "performance/structural_hypothesis_loop.py": (
        "f23a29c2f85b9bdce96398c6b901190fc34c2e2f97880dce5dbf2594b76c7635"
    ),
    "performance/structural_hypothesis_task_materializer.py": (
        "93345ef22df0cc9c5665be35722ad4646fb233d7a8bbd1294f0ca051cf64f20b"
    ),
    "performance/structural_hypothesis_recursive_report_advance.py": (
        "7a101890c8220b18c04febebe82e36e95fe199e9f1165658d2fd72643888d225"
    ),
}

_STATE_PREFIX = Path("kg-op/structural-hypothesis-recursive-successor/v1")
_ADVANCE_STATE_PREFIX = Path(
    "kg-op/structural-hypothesis-recursive-report-advance/v1"
)
_RUNTIME_STATE_PREFIX = Path("kg-op/structural-hypothesis-execution/v1")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TASK_ID = re.compile(r"task:[0-9a-f]{24}\Z")
_STATUS = (
    "RECURSIVE_SUCCESSOR_MATERIALIZED_FROM_VERIFIED_ADVANCE_NOT_AUTHORIZED"
)
_VERIFIED_STATUS = "VERIFIED_" + _STATUS
_ADVANCE_STATUS = (
    "ADVANCED_AS_IMMUTABLE_LOCAL_REPORT_VERSION_NOT_CURRENT_NOT_PLANNED"
)
_VERIFIED_ADVANCE_STATUS = "VERIFIED_" + _ADVANCE_STATUS
_SOURCE_TYPED_ROWS = 1352
_PENDING_TASKS = 28

_ARTIFACT_LAYOUT = {
    "recursive_successor_contract": "recursive_successor_contract.json",
    "task_bundle": "bundle.json",
    "recursive_successor_commit": "successor.json",
}
_ADVANCE_LAYOUT = {
    "advance_contract": "advance_contract.json",
    "source_adoption": "source/adoption.json",
    "source_successor": "source/successor.json",
    "source_execution_attempt": "source/execution/attempt.json",
    "source_execution_authorization": "source/execution/authorization.json",
    "source_execution_receipt": "source/execution/receipt.json",
    "source_execution_completed_event": (
        "source/execution/journal/0002_COMPLETED.json"
    ),
    "combined_rows": "combined_rows.json",
    "output_report": "output_report.json",
    "reingestion_receipt": "reingestion_receipt.json",
    "advance_commit": "advance.json",
}
_NONCLAIMS = [
    "recursive_successor_materialization_is_not_authorization",
    "recursive_successor_materialization_is_not_attempt_preparation",
    "recursive_successor_materialization_is_not_execution",
    "recursive_successor_materialization_is_not_global_current",
    "advance_is_not_global_current",
    "detached_bundle_alone_has_no_advance_provenance",
    "transitively_derived_leaf_anchors_are_not_independent_observations",
    "original_advance_and_full_source_chain_are_required_for_full_verification",
    "ready_for_authorization_mechanics_is_not_runtime_readiness",
    "future_attempt_absence_is_not_a_reservation",
    "local_digest_is_not_signature",
    "no_external_authority",
    "no_currentness_claim",
    "no_runtime_readiness_claim",
    "no_scientific_claim",
    "future_attempt_absence_is_a_point_in_time_final_precommit_observation",
    "future_attempt_and_checkpoint_paths_are_not_created",
    "no_network_access",
    "no_scheduler_access",
    "no_credential_access",
    "no_shell_execution",
    "no_task_authorization",
    "no_attempt_preparation",
    "no_task_execution",
    "no_benchmark_execution",
    "no_run_one_invocation",
    "no_paper_promotion",
]

# Capture only audited, non-executing public surfaces at definition time.
_VERIFY_RECURSIVE_REPORT_ADVANCE = _advance.verify_recursive_report_advance
_MATERIALIZE_TASK_BUNDLE = _materializer.materialize_task_bundle
_VERIFY_MATERIALIZED_TASK_BUNDLE = _materializer.verify_materialized_task_bundle


class RecursiveSuccessorMaterializationError(ValueError):
    """Raised when recursive successor materialization fails closed."""


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
        raise RecursiveSuccessorMaterializationError(
            f"{label} is not strict JSON"
        ) from error


def _require_digest(value: Any, label: str) -> str:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise RecursiveSuccessorMaterializationError(
            f"{label} must be a lowercase sha256 digest"
        )
    return value


def _require_identifier(value: Any, label: str) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise RecursiveSuccessorMaterializationError(f"{label} is invalid")
    return value


def _absolute(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise RecursiveSuccessorMaterializationError(
            f"{label} must be an absolute path"
        )
    normalized = Path(os.path.normpath(os.fspath(path)))
    if path != normalized or ".." in path.parts:
        raise RecursiveSuccessorMaterializationError(
            f"{label} must be a canonical absolute path"
        )
    return path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _state_base() -> Path:
    configured = os.environ.get("XDG_STATE_HOME")
    base = Path(configured) if configured else Path.home() / ".local/state"
    base = _absolute(base, "state home")
    if base != base.resolve(strict=False):
        raise RecursiveSuccessorMaterializationError(
            "state home contains an alias or symlink component"
        )
    return base


def _reject_symlink_components(path: Path) -> None:
    cursor = path
    while True:
        try:
            if cursor.is_symlink():
                raise RecursiveSuccessorMaterializationError(
                    f"path alias is forbidden: {cursor}"
                )
        except OSError as error:
            raise RecursiveSuccessorMaterializationError(
                f"cannot inspect path component: {cursor}"
            ) from error
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent


def _secure_directory(path: Path, label: str, *, exact_mode=None) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise RecursiveSuccessorMaterializationError(
            f"cannot inspect {label}"
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
        raise RecursiveSuccessorMaterializationError(
            f"{label} ownership, type, or mode differs"
        )


def _read_regular(
    path: Path, label: str, *, exact_mode: int | None = None
) -> bytes:
    candidate = _absolute(path, label)
    _reject_symlink_components(candidate)
    try:
        observed = candidate.lstat()
    except OSError as error:
        raise RecursiveSuccessorMaterializationError(
            f"cannot inspect {label}"
        ) from error
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise RecursiveSuccessorMaterializationError(
            f"cannot open {label}"
        ) from error
    try:
        before = os.fstat(descriptor)
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
        mode = stat.S_IMODE(before.st_mode)
        if (
            not stat.S_ISREG(before.st_mode)
            or opened_identity != observed_identity
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or mode & 0o022
            or exact_mode is not None
            and mode != exact_mode
        ):
            raise RecursiveSuccessorMaterializationError(
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
        if stable_before != stable_after or len(raw) != before.st_size:
            raise RecursiveSuccessorMaterializationError(
                f"{label} changed while captured"
            )
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
        raise RecursiveSuccessorMaterializationError(
            f"{label} is not strict JSON"
        ) from error


def _read_json(
    path: Path, label: str, *, exact_mode: int | None = None
) -> tuple[Any, bytes]:
    raw = _read_regular(path, label, exact_mode=exact_mode)
    return _parse_json(raw, label), raw


def _check_open_directory(descriptor: int, label: str) -> None:
    info = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise RecursiveSuccessorMaterializationError(
            f"{label} ownership, type, or mode differs"
        )


def _open_directory_at(parent: int, name: str, label: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent)
    except OSError as error:
        raise RecursiveSuccessorMaterializationError(
            f"cannot open {label}"
        ) from error
    try:
        _check_open_directory(descriptor, label)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_regular_at(parent: int, name: str, label: str) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent)
    except OSError as error:
        raise RecursiveSuccessorMaterializationError(
            f"cannot open {label}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise RecursiveSuccessorMaterializationError(
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
        if stable_before != stable_after or len(raw) != before.st_size:
            raise RecursiveSuccessorMaterializationError(
                f"{label} changed while captured"
            )
        return raw
    finally:
        os.close(descriptor)


def _require_names(descriptor: int, expected: set[str], label: str) -> None:
    try:
        observed = set(os.listdir(descriptor))
    except OSError as error:
        raise RecursiveSuccessorMaterializationError(
            f"cannot enumerate {label}"
        ) from error
    if observed != expected:
        raise RecursiveSuccessorMaterializationError(
            f"{label} has missing or unexpected artifacts"
        )


def _artifact(path: str, raw: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": _raw_digest(raw), "bytes": len(raw)}


def _validate_recursive_successor_contract(contract: Mapping[str, Any]) -> None:
    if (
        type(contract) is not dict
        or set(contract) != {
            "schema_version",
            "contract_id",
            "source_contracts",
            "source_files",
            "admission",
            "future_attempt_policy",
            "recursive_successor_root_policy",
            "artifact_layout",
            "commit_protocol",
            "mechanics",
            "nonclaims",
        }
        or contract.get("schema_version") != RECURSIVE_SUCCESSOR_SCHEMA_VERSION
        or contract.get("contract_id") != RECURSIVE_SUCCESSOR_CONTRACT_ID
        or contract.get("source_contracts") != _SOURCE_CONTRACTS
        or contract.get("source_files") != _SOURCE_FILES
        or contract.get("artifact_layout") != _ARTIFACT_LAYOUT
        or contract.get("nonclaims") != _NONCLAIMS
        or _digest(contract) != _RECURSIVE_SUCCESSOR_CONTRACT_DIGEST
    ):
        raise RecursiveSuccessorMaterializationError(
            "recursive successor contract differs from frozen V1"
        )
    root = _repo_root()
    for relative, expected in _SOURCE_FILES.items():
        raw = _read_regular(root / relative, f"source file {relative}")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RecursiveSuccessorMaterializationError(
                f"recursive successor source file differs: {relative}"
            )


def _validate_pinned_contract(
    value: Any, *, contract_id: str, digest: str, label: str
) -> dict[str, Any]:
    if (
        type(value) is not dict
        or value.get("contract_id") != contract_id
        or _digest(value) != digest
    ):
        raise RecursiveSuccessorMaterializationError(
            f"{label} differs from frozen V1"
        )
    return value


def _load_contracts(paths: Mapping[str, Path]) -> dict[str, Any]:
    specifications = {
        "advance": (
            "advance_contract_path",
            "structural_hypothesis_recursive_report_advance_v1",
            _ADVANCE_CONTRACT_DIGEST,
        ),
        "hypothesis": (
            "hypothesis_contract_path",
            "structural_hypothesis_loop_v1",
            _HYPOTHESIS_CONTRACT_DIGEST,
        ),
        "executor": (
            "executor_contract_path",
            "structural_hypothesis_executor_v1",
            _EXECUTOR_CONTRACT_DIGEST,
        ),
        "materializer": (
            "materializer_contract_path",
            "structural_hypothesis_task_materializer_v1",
            _MATERIALIZER_CONTRACT_DIGEST,
        ),
        "recursive_successor": (
            "recursive_successor_contract_path",
            RECURSIVE_SUCCESSOR_CONTRACT_ID,
            _RECURSIVE_SUCCESSOR_CONTRACT_DIGEST,
        ),
    }
    values: dict[str, Any] = {}
    raws: dict[str, bytes] = {}
    for label, (path_key, contract_id, digest) in specifications.items():
        value, raw = _read_json(paths[path_key], f"{label} contract")
        values[label] = _validate_pinned_contract(
            value, contract_id=contract_id, digest=digest, label=label
        )
        raws[label] = raw
    _validate_recursive_successor_contract(values["recursive_successor"])
    try:
        _materializer.validate_materializer_contract(values["materializer"])
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise RecursiveSuccessorMaterializationError(
            "task materializer contract failed public validation"
        ) from error
    return {"values": values, "raws": raws}


def _same_contracts(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    if first["raws"] != second["raws"] or first["values"] != second["values"]:
        raise RecursiveSuccessorMaterializationError(
            "source or recursive successor contract generation changed"
        )


def _validate_advance_location(
    advance_root: str | Path, advance_id: str
) -> Path:
    root = _absolute(advance_root, "advance_root")
    _require_identifier(advance_id, "advance_id")
    if root.name != advance_id:
        raise RecursiveSuccessorMaterializationError(
            "advance_id must equal advance_root basename"
        )
    if root.parent != _state_base() / _ADVANCE_STATE_PREFIX:
        raise RecursiveSuccessorMaterializationError(
            "advance_root is outside the frozen local state prefix"
        )
    _reject_symlink_components(root)
    _secure_directory(root, "advance root", exact_mode=0o700)
    return root


def _ensure_secure_parent(target: Path) -> None:
    base = _state_base()
    prefix = base / _STATE_PREFIX
    if target.parent != prefix:
        raise RecursiveSuccessorMaterializationError(
            "recursive_successor_root is outside the frozen local state prefix"
        )
    _reject_symlink_components(target)
    missing: list[Path] = []
    cursor = prefix
    while not cursor.exists():
        if cursor.is_symlink():
            raise RecursiveSuccessorMaterializationError(
                f"recursive successor path alias is forbidden: {cursor}"
            )
        missing.append(cursor)
        if cursor == base:
            break
        cursor = cursor.parent
    if not cursor.exists():
        raise RecursiveSuccessorMaterializationError(
            "state-home ancestor is missing"
        )
    _secure_directory(cursor, "state-home ancestor")
    for path in reversed(missing):
        try:
            os.mkdir(path, 0o700)
            os.chmod(path, 0o700, follow_symlinks=False)
            _fsync_directory(path.parent)
        except FileExistsError:
            pass
        except OSError as error:
            raise RecursiveSuccessorMaterializationError(
                f"cannot create recursive successor prefix: {path}"
            ) from error
    chain = [prefix]
    cursor = prefix.parent
    while cursor != base and base in cursor.parents:
        chain.append(cursor)
        cursor = cursor.parent
    chain.append(base)
    for path in chain:
        _secure_directory(path, f"state directory {path}")


def _validate_recursive_successor_location(
    recursive_successor_root: str | Path,
    recursive_successor_id: str,
    *,
    fresh: bool,
) -> Path:
    root = _absolute(recursive_successor_root, "recursive_successor_root")
    _require_identifier(recursive_successor_id, "recursive_successor_id")
    if root.name != recursive_successor_id:
        raise RecursiveSuccessorMaterializationError(
            "recursive_successor_id must equal recursive_successor_root basename"
        )
    if root.parent != _state_base() / _STATE_PREFIX:
        raise RecursiveSuccessorMaterializationError(
            "recursive_successor_root is outside the frozen local state prefix"
        )
    if fresh:
        if root.exists() or root.is_symlink():
            raise RecursiveSuccessorMaterializationError(
                "recursive_successor_root already exists"
            )
        _ensure_secure_parent(root)
    else:
        _reject_symlink_components(root)
        _secure_directory(root, "recursive successor root", exact_mode=0o700)
    return root


def _validate_future_attempt(
    future_attempt_root: str | Path,
    recursive_successor_id: str,
    *,
    require_absent: bool,
) -> tuple[Path, Path]:
    attempt = _absolute(future_attempt_root, "future_attempt_root")
    if attempt.parent != _state_base() / _RUNTIME_STATE_PREFIX:
        raise RecursiveSuccessorMaterializationError(
            "future_attempt_root must be a direct runtime-prefix child"
        )
    if attempt.name != recursive_successor_id:
        raise RecursiveSuccessorMaterializationError(
            "future_attempt_root basename must equal recursive_successor_id"
        )
    _reject_symlink_components(attempt)
    if require_absent and (attempt.exists() or attempt.is_symlink()):
        raise RecursiveSuccessorMaterializationError(
            "future_attempt_root must be absent at materialization"
        )
    return attempt, attempt / "checkpoints"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_new(path: Path) -> None:
    try:
        os.mkdir(path, 0o700)
        os.chmod(path, 0o700, follow_symlinks=False)
        _fsync_directory(path.parent)
        _fsync_directory(path)
    except OSError as error:
        raise RecursiveSuccessorMaterializationError(
            f"cannot create fresh recursive successor directory: {path}"
        ) from error
    _secure_directory(path, "recursive successor root", exact_mode=0o700)


def _write_new_bytes(path: Path, raw: bytes, *, prepublish=None) -> None:
    if type(raw) is not bytes:
        raise RecursiveSuccessorMaterializationError(
            "recursive successor artifact payload must be bytes"
        )
    if path.exists() or path.is_symlink():
        raise RecursiveSuccessorMaterializationError(
            f"recursive successor artifact already exists: {path}"
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
        if prepublish is not None:
            prepublish()
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _capture_advance(
    advance_root: str | Path, advance_id: str
) -> dict[str, Any]:
    """Capture one exact advance generation through held directory fds."""
    root = _validate_advance_location(advance_root, advance_id)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(root, flags)
    except OSError as error:
        raise RecursiveSuccessorMaterializationError(
            "cannot open source advance root"
        ) from error
    descriptors = [root_fd]
    try:
        _check_open_directory(root_fd, "source advance root")
        source_fd = _open_directory_at(root_fd, "source", "advance source")
        descriptors.append(source_fd)
        execution_fd = _open_directory_at(
            source_fd, "execution", "advance source execution"
        )
        descriptors.append(execution_fd)
        journal_fd = _open_directory_at(
            execution_fd, "journal", "advance source execution journal"
        )
        descriptors.append(journal_fd)
        directory_fds = {
            "": root_fd,
            "source": source_fd,
            "source/execution": execution_fd,
            "source/execution/journal": journal_fd,
        }
        expected_names = {key: set() for key in directory_fds}
        expected_names[""].add("source")
        expected_names["source"].add("execution")
        expected_names["source/execution"].add("journal")
        for relative in _ADVANCE_LAYOUT.values():
            path = Path(relative)
            parent = os.fspath(path.parent)
            if parent == ".":
                parent = ""
            expected_names[parent].add(path.name)
        raws: dict[str, bytes] = {}
        for label, relative in _ADVANCE_LAYOUT.items():
            path = Path(relative)
            parent = os.fspath(path.parent)
            if parent == ".":
                parent = ""
            raws[label] = _read_regular_at(
                directory_fds[parent], path.name, f"advance {label}"
            )
        for relative, descriptor in directory_fds.items():
            _require_names(
                descriptor,
                expected_names[relative],
                f"advance directory {relative or '.'}",
            )
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    values = {
        label: _parse_json(raw, f"captured advance {label}")
        for label, raw in raws.items()
    }
    return {"root": root, "raws": raws, "values": values}


def _same_advance(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    if first["root"] != second["root"] or first["raws"] != second["raws"]:
        raise RecursiveSuccessorMaterializationError(
            "source advance generation changed during materialization"
        )


def _expected_source_binding(
    expected: Mapping[str, str], identities: Mapping[str, str]
) -> dict[str, str]:
    return {
        "adoption_digest": expected["expected_adoption_digest"],
        "successor_digest": expected["expected_source_successor_digest"],
        "pending_evidence_digest": expected[
            "expected_source_pending_evidence_digest"
        ],
        "first_pending_projection_digest": expected[
            "expected_source_first_pending_projection_digest"
        ],
        "bundle_digest": expected["expected_source_bundle_digest"],
        "plan_digest": expected["expected_source_plan_digest"],
        "task_id": identities["completed_task_id"],
        "task_digest": expected["expected_completed_task_digest"],
        "provenance_binding_digest": expected[
            "expected_source_provenance_binding_digest"
        ],
        "authorization_digest": expected[
            "expected_source_authorization_digest"
        ],
        "attempt_digest": expected["expected_source_attempt_digest"],
        "execution_receipt_digest": expected[
            "expected_source_execution_receipt_digest"
        ],
        "execution_journal_head_digest": expected[
            "expected_source_execution_journal_head_digest"
        ],
    }


def _validate_captured_advance(
    captured: Mapping[str, Any],
    *,
    external_advance_contract_raw: bytes,
    identities: Mapping[str, str],
    expected: Mapping[str, str],
) -> None:
    raws = captured["raws"]
    values = captured["values"]
    if raws["advance_contract"] != external_advance_contract_raw:
        raise RecursiveSuccessorMaterializationError(
            "captured and external advance contracts differ"
        )
    for label, value in values.items():
        if label == "combined_rows":
            if type(value) is not list:
                raise RecursiveSuccessorMaterializationError(
                    "captured advance combined_rows must be an array"
                )
        elif type(value) is not dict:
            raise RecursiveSuccessorMaterializationError(
                f"captured advance {label} must be an object"
            )
        if label != "advance_contract" and raws[label] != _canonical_file(value):
            raise RecursiveSuccessorMaterializationError(
                f"captured advance {label} is not canonical JSON"
            )
    contract = values["advance_contract"]
    marker = values["advance_commit"]
    body = {key: marker[key] for key in marker if key != "integrity"}
    integrity = marker.get("integrity")
    if (
        _digest(contract) != _ADVANCE_CONTRACT_DIGEST
        or contract.get("contract_id")
        != "structural_hypothesis_recursive_report_advance_v1"
        or type(integrity) is not dict
        or set(integrity) != {"algorithm", "advance_digest"}
        or integrity.get("algorithm") != "sha256-canonical-json-v1"
        or _digest(body) != integrity.get("advance_digest")
        or integrity.get("advance_digest")
        != expected["expected_advance_digest"]
        or marker.get("schema_version")
        != _advance.ADVANCE_CAPSULE_SCHEMA_VERSION
        or marker.get("status") != _ADVANCE_STATUS
        or marker.get("advance_id") != identities["advance_id"]
        or captured["root"].name != identities["advance_id"]
        or marker.get("current_status") != "NOT_CURRENT"
        or marker.get("planning_status") != "NOT_PLANNED"
    ):
        raise RecursiveSuccessorMaterializationError(
            "captured advance marker or independent digest differs"
        )
    expected_artifacts = {
        label: _artifact(_ADVANCE_LAYOUT[label], raws[label])
        for label in _ADVANCE_LAYOUT
        if label != "advance_commit"
    }
    if marker.get("artifacts") != expected_artifacts:
        raise RecursiveSuccessorMaterializationError(
            "captured advance artifact map differs"
        )
    if marker.get("advance_contract_binding") != {
        "contract_id": "structural_hypothesis_recursive_report_advance_v1",
        "contract_digest": _ADVANCE_CONTRACT_DIGEST,
    }:
        raise RecursiveSuccessorMaterializationError(
            "captured advance contract binding differs"
        )
    source_binding = marker.get("source_binding")
    required_source = _expected_source_binding(expected, identities)
    if not isinstance(source_binding, Mapping) or any(
        source_binding.get(key) != value
        for key, value in required_source.items()
    ):
        raise RecursiveSuccessorMaterializationError(
            "captured advance source anchors differ"
        )
    required_authorization_id = (
        "successor-bound-v1:"
        + expected["expected_source_provenance_binding_digest"].split(":", 1)[1]
    )
    if source_binding.get("required_authorization_id") != required_authorization_id:
        raise RecursiveSuccessorMaterializationError(
            "captured advance authorization identity differs"
        )
    transition = marker.get("transition")
    if (
        not isinstance(transition, Mapping)
        or transition.get("source_typed_row_count") != 1351
        or transition.get("output_typed_row_count") != _SOURCE_TYPED_ROWS
        or transition.get("source_pending_count") != 29
        or transition.get("output_pending_count") != _PENDING_TASKS
        or transition.get("accepted_successful_rows") != 1
        or transition.get("ignored_failed_attempts") != 0
        or transition.get("task_id") != identities["completed_task_id"]
        or transition.get("task_digest")
        != expected["expected_completed_task_digest"]
    ):
        raise RecursiveSuccessorMaterializationError(
            "captured advance transition differs from frozen V1"
        )
    rows = values["combined_rows"]
    report = values["output_report"]
    receipt = values["reingestion_receipt"]
    adopted = marker.get("adopted_report")
    if (
        len(rows) != _SOURCE_TYPED_ROWS
        or not verify_report_integrity(report)
        or report.get("evidence_digest") != _digest(rows)
        or report.get("status") != "COMPLETED_WITH_EVIDENCE_GAPS"
        or type(report.get("pending_evidence")) is not list
        or len(report["pending_evidence"]) != _PENDING_TASKS
        or not isinstance(receipt, Mapping)
        or receipt.get("integrity", {}).get("reingestion_digest")
        != expected["expected_advance_reingestion_digest"]
        or not isinstance(adopted, Mapping)
        or adopted.get("evidence_digest")
        != expected["expected_advance_output_evidence_digest"]
        or adopted.get("report_body_digest")
        != expected["expected_advance_output_report_body_digest"]
        or adopted.get("audit_head")
        != expected["expected_advance_output_audit_head"]
        or adopted.get("reingestion_digest")
        != expected["expected_advance_reingestion_digest"]
        or adopted.get("typed_row_count") != _SOURCE_TYPED_ROWS
        or adopted.get("pending_evidence_count") != _PENDING_TASKS
        or report.get("evidence_digest")
        != expected["expected_advance_output_evidence_digest"]
        or report.get("audit", {}).get("report_body_digest")
        != expected["expected_advance_output_report_body_digest"]
        or report.get("audit", {}).get("head")
        != expected["expected_advance_output_audit_head"]
    ):
        raise RecursiveSuccessorMaterializationError(
            "captured advance output report or anchors differ"
        )


def _paths(
    publication_root,
    adoption_contract_path,
    adoption_root,
    source_successor_contract_path,
    source_successor_root,
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
    recursive_successor_contract_path,
    recursive_successor_root,
    future_attempt_root,
) -> dict[str, Path]:
    return {
        name: _absolute(value, name)
        for name, value in locals().items()
    }


def _identities(
    *,
    adoption_id,
    source_successor_id,
    advance_id,
    recursive_successor_id,
    completed_task_id,
) -> dict[str, str]:
    values = {
        "adoption_id": adoption_id,
        "source_successor_id": source_successor_id,
        "advance_id": advance_id,
        "recursive_successor_id": recursive_successor_id,
    }
    for label, value in values.items():
        values[label] = _require_identifier(value, label)
    if type(completed_task_id) is not str or not _TASK_ID.fullmatch(
        completed_task_id
    ):
        raise RecursiveSuccessorMaterializationError(
            "completed_task_id is invalid"
        )
    values["completed_task_id"] = completed_task_id
    return values


def _expectations(**values: str) -> dict[str, str]:
    return {
        label: _require_digest(value, label)
        for label, value in values.items()
    }


def _verify_public_advance(
    paths: Mapping[str, Path],
    identities: Mapping[str, str],
    expected: Mapping[str, str],
) -> dict[str, Any]:
    try:
        result = _VERIFY_RECURSIVE_REPORT_ADVANCE(
            paths["publication_root"],
            paths["adoption_contract_path"],
            paths["adoption_root"],
            paths["source_successor_contract_path"],
            paths["source_successor_root"],
            paths["base_evidence_csv"],
            paths["source_attempt_root"],
            paths["hypothesis_contract_path"],
            paths["executor_contract_path"],
            paths["runtime_contract_path"],
            paths["publisher_contract_path"],
            paths["materializer_contract_path"],
            paths["bridge_contract_path"],
            paths["base_manifest_path"],
            paths["asset_root"],
            paths["completed_attempt_root"],
            paths["advance_contract_path"],
            paths["advance_root"],
            advance_id=identities["advance_id"],
            adoption_id=identities["adoption_id"],
            successor_id=identities["source_successor_id"],
            expected_adoption_digest=expected["expected_adoption_digest"],
            expected_pending_evidence_digest=expected[
                "expected_source_pending_evidence_digest"
            ],
            expected_first_pending_projection_digest=expected[
                "expected_source_first_pending_projection_digest"
            ],
            expected_successor_digest=expected[
                "expected_source_successor_digest"
            ],
            expected_bundle_digest=expected["expected_source_bundle_digest"],
            expected_plan_digest=expected["expected_source_plan_digest"],
            task_id=identities["completed_task_id"],
            expected_task_digest=expected["expected_completed_task_digest"],
            expected_provenance_binding_digest=expected[
                "expected_source_provenance_binding_digest"
            ],
            expected_authorization_digest=expected[
                "expected_source_authorization_digest"
            ],
            expected_attempt_digest=expected[
                "expected_source_attempt_digest"
            ],
            expected_execution_receipt_digest=expected[
                "expected_source_execution_receipt_digest"
            ],
            expected_execution_journal_head_digest=expected[
                "expected_source_execution_journal_head_digest"
            ],
            expected_advance_digest=expected["expected_advance_digest"],
            expected_reingestion_digest=expected[
                "expected_advance_reingestion_digest"
            ],
            expected_output_report_body_digest=expected[
                "expected_advance_output_report_body_digest"
            ],
            expected_output_audit_head=expected[
                "expected_advance_output_audit_head"
            ],
            expected_output_evidence_digest=expected[
                "expected_advance_output_evidence_digest"
            ],
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise RecursiveSuccessorMaterializationError(
            "source advance failed public full-chain verification"
        ) from error
    required = {
        "status": _VERIFIED_ADVANCE_STATUS,
        "advance_root": str(paths["advance_root"]),
        "advance_digest": expected["expected_advance_digest"],
        "reingestion_digest": expected["expected_advance_reingestion_digest"],
        "output_report_body_digest": expected[
            "expected_advance_output_report_body_digest"
        ],
        "output_audit_head": expected["expected_advance_output_audit_head"],
        "output_evidence_digest": expected[
            "expected_advance_output_evidence_digest"
        ],
        "typed_row_count": _SOURCE_TYPED_ROWS,
        "pending_evidence_count": _PENDING_TASKS,
        "current_status": "NOT_CURRENT",
        "planning_status": "NOT_PLANNED",
    }
    if type(result) is not dict or result != required:
        raise RecursiveSuccessorMaterializationError(
            "public advance verifier returned an unexpected surface"
        )
    return result


def _materialize_next_bundle(
    captured: Mapping[str, Any],
    contracts: Mapping[str, Any],
    paths: Mapping[str, Path],
    expected: Mapping[str, str],
    checkpoint_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = captured["values"]["combined_rows"]
    report = captured["values"]["output_report"]
    hypothesis = contracts["values"]["hypothesis"]
    executor = contracts["values"]["executor"]
    materializer = contracts["values"]["materializer"]
    try:
        replayed = run_structural_hypothesis_loop(
            rows, hypothesis, input_artifacts=None
        ).to_dict()
    except (ValueError, KeyError, TypeError) as error:
        raise RecursiveSuccessorMaterializationError(
            "captured advance rows failed exact report replay"
        ) from error
    pending = replayed.get("pending_evidence")
    if (
        replayed != report
        or not verify_report_integrity(replayed)
        or replayed.get("evidence_digest") != _digest(rows)
        or type(rows) is not list
        or len(rows) != _SOURCE_TYPED_ROWS
        or type(pending) is not list
        or len(pending) != _PENDING_TASKS
    ):
        raise RecursiveSuccessorMaterializationError(
            "captured advance does not replay the exact 1352/28 report"
        )
    try:
        first_projection = {
            "profile": pending[0]["profile"],
            "domain": pending[0]["domain"],
            "line": executor["execution_scope"]["line"],
            "seed": pending[0]["seed"],
            "d": pending[0]["d"],
            "N": pending[0]["N"],
            "n0": pending[0]["n0"],
        }
    except (KeyError, TypeError, IndexError) as error:
        raise RecursiveSuccessorMaterializationError(
            "first pending projection is malformed"
        ) from error
    pending_digest = _digest(pending)
    projection_digest = _digest(first_projection)
    if (
        pending_digest != expected["expected_next_pending_evidence_digest"]
        or projection_digest
        != expected["expected_next_first_pending_projection_digest"]
    ):
        raise RecursiveSuccessorMaterializationError(
            "independent next-pending anchors differ"
        )
    try:
        bundle = _MATERIALIZE_TASK_BUNDLE(
            replayed,
            hypothesis,
            executor,
            materializer,
            paths["base_manifest_path"],
            paths["asset_root"],
            checkpoint_root,
        )
        strongly_verified = _VERIFY_MATERIALIZED_TASK_BUNDLE(
            bundle,
            replayed,
            hypothesis,
            executor,
            materializer,
            paths["base_manifest_path"],
            paths["asset_root"],
            checkpoint_root,
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise RecursiveSuccessorMaterializationError(
            "public task materializer rejected the verified advance report"
        ) from error
    plan = bundle.get("plan") if isinstance(bundle, Mapping) else None
    tasks = plan.get("tasks") if isinstance(plan, Mapping) else None
    if (
        bundle.get("status") != "MATERIALIZED_NOT_AUTHORIZED"
        or bundle.get("task_count") != _PENDING_TASKS
        or not isinstance(plan, Mapping)
        or plan.get("proposal_count") != _PENDING_TASKS
        or type(tasks) is not list
        or len(tasks) != _PENDING_TASKS
        or strongly_verified is not True
        or tasks[0].get("cell") != first_projection
    ):
        raise RecursiveSuccessorMaterializationError(
            "next bundle failed strong frozen-materializer verification"
        )
    first_task_id = tasks[0].get("task_id")
    first_task_digest = tasks[0].get("task_digest")
    if (
        type(first_task_id) is not str
        or not _TASK_ID.fullmatch(first_task_id)
        or type(first_task_digest) is not str
        or not _DIGEST.fullmatch(first_task_digest)
    ):
        raise RecursiveSuccessorMaterializationError(
            "next bundle first-task identity is malformed"
        )
    return _json_clone(bundle, "next task bundle"), {
        "pending_evidence_digest": pending_digest,
        "first_pending_projection_digest": projection_digest,
        "first_pending_projection": first_projection,
        "first_task_id": first_task_id,
        "first_task_digest": first_task_digest,
    }


def _validate_generation(
    paths: Mapping[str, Path],
    *,
    identities: Mapping[str, str],
    expected: Mapping[str, str],
    require_future_absent: bool,
) -> dict[str, Any]:
    """Validate, verify, replay, and materialize one exact source generation."""
    contracts = _load_contracts(paths)
    captured = _capture_advance(
        paths["advance_root"], identities["advance_id"]
    )
    _validate_captured_advance(
        captured,
        external_advance_contract_raw=contracts["raws"]["advance"],
        identities=identities,
        expected=expected,
    )
    verified_advance = _verify_public_advance(paths, identities, expected)
    future_attempt, checkpoint_root = _validate_future_attempt(
        paths["future_attempt_root"],
        identities["recursive_successor_id"],
        require_absent=require_future_absent,
    )
    bundle, pending = _materialize_next_bundle(
        captured, contracts, paths, expected, checkpoint_root
    )
    return {
        "advance_capture": captured,
        "contracts": contracts,
        "verified_advance": verified_advance,
        "bundle": bundle,
        "pending": pending,
        "future_attempt": future_attempt,
        "checkpoint_root": checkpoint_root,
    }


def _same_generation(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    _same_advance(first["advance_capture"], second["advance_capture"])
    _same_contracts(first["contracts"], second["contracts"])
    for label in (
        "verified_advance",
        "bundle",
        "pending",
        "future_attempt",
        "checkpoint_root",
    ):
        if first[label] != second[label]:
            raise RecursiveSuccessorMaterializationError(
                f"validated source generation changed: {label}"
            )


def _revalidate_staged_generation(
    generation: Mapping[str, Any],
    paths: Mapping[str, Path],
    identities: Mapping[str, str],
    expected: Mapping[str, str],
    *,
    require_future_absent: bool,
) -> None:
    captured_before = _capture_advance(
        paths["advance_root"], identities["advance_id"]
    )
    _same_advance(generation["advance_capture"], captured_before)
    contracts_before = _load_contracts(paths)
    _same_contracts(generation["contracts"], contracts_before)
    _validate_captured_advance(
        captured_before,
        external_advance_contract_raw=contracts_before["raws"]["advance"],
        identities=identities,
        expected=expected,
    )
    verified_advance = _verify_public_advance(paths, identities, expected)
    if verified_advance != generation["verified_advance"]:
        raise RecursiveSuccessorMaterializationError(
            "public advance verification changed after staging"
        )
    captured_after = _capture_advance(
        paths["advance_root"], identities["advance_id"]
    )
    _same_advance(captured_before, captured_after)
    _same_advance(generation["advance_capture"], captured_after)
    contracts_after = _load_contracts(paths)
    _same_contracts(contracts_before, contracts_after)
    _same_contracts(generation["contracts"], contracts_after)
    _validate_captured_advance(
        captured_after,
        external_advance_contract_raw=contracts_after["raws"]["advance"],
        identities=identities,
        expected=expected,
    )
    future_attempt, checkpoint_root = _validate_future_attempt(
        paths["future_attempt_root"],
        identities["recursive_successor_id"],
        require_absent=require_future_absent,
    )
    rebuilt_bundle, rebuilt_pending = _materialize_next_bundle(
        captured_after, contracts_after, paths, expected, checkpoint_root
    )
    if (
        rebuilt_bundle != generation["bundle"]
        or rebuilt_pending != generation["pending"]
        or future_attempt != generation["future_attempt"]
        or checkpoint_root != generation["checkpoint_root"]
    ):
        raise RecursiveSuccessorMaterializationError(
            "source generation changed after staging"
        )


def _layout(root: Path, *, committed: bool) -> None:
    _secure_directory(root, "recursive successor root", exact_mode=0o700)
    expected = {
        _ARTIFACT_LAYOUT["recursive_successor_contract"],
        _ARTIFACT_LAYOUT["task_bundle"],
    }
    if committed:
        expected.add(_ARTIFACT_LAYOUT["recursive_successor_commit"])
    try:
        observed = set(os.listdir(root))
    except OSError as error:
        raise RecursiveSuccessorMaterializationError(
            "cannot enumerate recursive successor root"
        ) from error
    if observed != expected:
        raise RecursiveSuccessorMaterializationError(
            "recursive successor root has missing or unexpected artifacts"
        )


def _marker_body(
    *,
    recursive_successor_id: str,
    contract: Mapping[str, Any],
    contract_raw: bytes,
    generation: Mapping[str, Any],
    expected: Mapping[str, str],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    pending = generation["pending"]
    bundle = generation["bundle"]
    plan = bundle["plan"]
    future_attempt = generation["future_attempt"]
    checkpoint_root = generation["checkpoint_root"]
    return {
        "schema_version": RECURSIVE_SUCCESSOR_CAPSULE_SCHEMA_VERSION,
        "status": _STATUS,
        "recursive_successor_id": recursive_successor_id,
        "current_status": "NOT_CURRENT",
        "authorization_status": "NOT_AUTHORIZED",
        "attempt_status": "NOT_PREPARED",
        "execution_status": "NOT_EXECUTED",
        "recursive_successor_contract_binding": {
            "contract_id": contract["contract_id"],
            "contract_digest": _digest(contract),
            "raw_sha256": _raw_digest(contract_raw),
            "bytes": len(contract_raw),
        },
        "source_advance": {
            "advance_id": generation["advance_capture"]["values"][
                "advance_commit"
            ]["advance_id"],
            "advance_digest": expected["expected_advance_digest"],
            "advance_status": _ADVANCE_STATUS,
            "advance_output_evidence_digest": expected[
                "expected_advance_output_evidence_digest"
            ],
            "advance_output_report_body_digest": expected[
                "expected_advance_output_report_body_digest"
            ],
            "advance_output_audit_head": expected[
                "expected_advance_output_audit_head"
            ],
            "advance_reingestion_digest": expected[
                "expected_advance_reingestion_digest"
            ],
            "typed_row_count": _SOURCE_TYPED_ROWS,
            "pending_evidence_count": _PENDING_TASKS,
            "current_status": "NOT_CURRENT",
            "planning_status": "NOT_PLANNED",
        },
        "pending_binding": dict(pending),
        "bundle_binding": {
            "bundle_id": bundle["bundle_id"],
            "bundle_digest": bundle["integrity"]["bundle_digest"],
            "plan_digest": plan["integrity"]["plan_digest"],
            "task_count": bundle["task_count"],
        },
        "future_attempt_binding": {
            "future_attempt_root": str(future_attempt),
            "checkpoint_root": str(checkpoint_root),
            "future_attempt_absent_at_final_precommit_observation": True,
            "future_attempt_created": False,
            "checkpoint_root_created": False,
        },
        "artifacts": dict(artifacts),
        "nonclaims": list(_NONCLAIMS),
    }


def _result(
    marker: Mapping[str, Any], root: Path, *, verified: bool
) -> dict[str, Any]:
    source = marker["source_advance"]
    pending = marker["pending_binding"]
    bundle = marker["bundle_binding"]
    future = marker["future_attempt_binding"]
    return {
        "status": _VERIFIED_STATUS if verified else _STATUS,
        "recursive_successor_root": str(root),
        "recursive_successor_digest": marker["integrity"][
            "recursive_successor_digest"
        ],
        "advance_digest": source["advance_digest"],
        "advance_output_evidence_digest": source[
            "advance_output_evidence_digest"
        ],
        "pending_evidence_digest": pending["pending_evidence_digest"],
        "first_pending_projection_digest": pending[
            "first_pending_projection_digest"
        ],
        "bundle_digest": bundle["bundle_digest"],
        "plan_digest": bundle["plan_digest"],
        "first_task_id": pending["first_task_id"],
        "first_task_digest": pending["first_task_digest"],
        "task_count": bundle["task_count"],
        "future_attempt_root": future["future_attempt_root"],
        "checkpoint_root": future["checkpoint_root"],
        "current_status": marker["current_status"],
        "authorization_status": marker["authorization_status"],
        "attempt_status": marker["attempt_status"],
        "execution_status": marker["execution_status"],
    }


def _capture_recursive_successor(
    root: Path, recursive_successor_id: str
) -> dict[str, Any]:
    capsule_root = _validate_recursive_successor_location(
        root, recursive_successor_id, fresh=False
    )
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(capsule_root, flags)
    except OSError as error:
        raise RecursiveSuccessorMaterializationError(
            "cannot open recursive successor capsule"
        ) from error
    try:
        _check_open_directory(root_fd, "recursive successor capsule")
        raws = {
            label: _read_regular_at(
                root_fd, relative, f"recursive successor {label}"
            )
            for label, relative in _ARTIFACT_LAYOUT.items()
        }
        _require_names(
            root_fd,
            set(_ARTIFACT_LAYOUT.values()),
            "recursive successor capsule",
        )
    finally:
        os.close(root_fd)
    values = {
        label: _parse_json(raw, f"recursive successor {label}")
        for label, raw in raws.items()
    }
    return {"root": capsule_root, "raws": raws, "values": values}


def materialize_recursive_successor(
    publication_root,
    adoption_contract_path,
    adoption_root,
    source_successor_contract_path,
    source_successor_root,
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
    recursive_successor_contract_path,
    recursive_successor_root,
    future_attempt_root,
    *,
    adoption_id: str,
    source_successor_id: str,
    advance_id: str,
    recursive_successor_id: str,
    expected_adoption_digest: str,
    expected_source_pending_evidence_digest: str,
    expected_source_first_pending_projection_digest: str,
    expected_source_successor_digest: str,
    expected_source_bundle_digest: str,
    expected_source_plan_digest: str,
    completed_task_id: str,
    expected_completed_task_digest: str,
    expected_source_provenance_binding_digest: str,
    expected_source_authorization_digest: str,
    expected_source_attempt_digest: str,
    expected_source_execution_receipt_digest: str,
    expected_source_execution_journal_head_digest: str,
    expected_advance_digest: str,
    expected_advance_reingestion_digest: str,
    expected_advance_output_report_body_digest: str,
    expected_advance_output_audit_head: str,
    expected_advance_output_evidence_digest: str,
    expected_next_pending_evidence_digest: str,
    expected_next_first_pending_projection_digest: str,
    confirm_recursive_successor_materialization: bool,
) -> dict[str, Any]:
    """Commit a next-generation task bundle without authorizing it."""
    if confirm_recursive_successor_materialization is not True:
        raise RecursiveSuccessorMaterializationError(
            "explicit recursive successor materialization confirmation is required"
        )
    paths = _paths(
        publication_root,
        adoption_contract_path,
        adoption_root,
        source_successor_contract_path,
        source_successor_root,
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
        recursive_successor_contract_path,
        recursive_successor_root,
        future_attempt_root,
    )
    identities = _identities(
        adoption_id=adoption_id,
        source_successor_id=source_successor_id,
        advance_id=advance_id,
        recursive_successor_id=recursive_successor_id,
        completed_task_id=completed_task_id,
    )
    expected = _expectations(
        expected_adoption_digest=expected_adoption_digest,
        expected_source_pending_evidence_digest=(
            expected_source_pending_evidence_digest
        ),
        expected_source_first_pending_projection_digest=(
            expected_source_first_pending_projection_digest
        ),
        expected_source_successor_digest=expected_source_successor_digest,
        expected_source_bundle_digest=expected_source_bundle_digest,
        expected_source_plan_digest=expected_source_plan_digest,
        expected_completed_task_digest=expected_completed_task_digest,
        expected_source_provenance_binding_digest=(
            expected_source_provenance_binding_digest
        ),
        expected_source_authorization_digest=(
            expected_source_authorization_digest
        ),
        expected_source_attempt_digest=expected_source_attempt_digest,
        expected_source_execution_receipt_digest=(
            expected_source_execution_receipt_digest
        ),
        expected_source_execution_journal_head_digest=(
            expected_source_execution_journal_head_digest
        ),
        expected_advance_digest=expected_advance_digest,
        expected_advance_reingestion_digest=(
            expected_advance_reingestion_digest
        ),
        expected_advance_output_report_body_digest=(
            expected_advance_output_report_body_digest
        ),
        expected_advance_output_audit_head=(
            expected_advance_output_audit_head
        ),
        expected_advance_output_evidence_digest=(
            expected_advance_output_evidence_digest
        ),
        expected_next_pending_evidence_digest=(
            expected_next_pending_evidence_digest
        ),
        expected_next_first_pending_projection_digest=(
            expected_next_first_pending_projection_digest
        ),
    )
    first = _validate_generation(
        paths,
        identities=identities,
        expected=expected,
        require_future_absent=True,
    )
    second = _validate_generation(
        paths,
        identities=identities,
        expected=expected,
        require_future_absent=True,
    )
    _same_generation(first, second)
    root = _validate_recursive_successor_location(
        paths["recursive_successor_root"],
        recursive_successor_id,
        fresh=True,
    )
    contract = second["contracts"]["values"]["recursive_successor"]
    contract_raw = second["contracts"]["raws"]["recursive_successor"]
    bundle_raw = _canonical_file(second["bundle"])
    artifacts = {
        "recursive_successor_contract": _artifact(
            _ARTIFACT_LAYOUT["recursive_successor_contract"], contract_raw
        ),
        "task_bundle": _artifact(
            _ARTIFACT_LAYOUT["task_bundle"], bundle_raw
        ),
    }
    _mkdir_new(root)
    _write_new_bytes(
        root / _ARTIFACT_LAYOUT["recursive_successor_contract"], contract_raw
    )
    _write_new_bytes(root / _ARTIFACT_LAYOUT["task_bundle"], bundle_raw)
    _layout(root, committed=False)
    for label, raw in {
        "recursive_successor_contract": contract_raw,
        "task_bundle": bundle_raw,
    }.items():
        if _read_regular(
            root / _ARTIFACT_LAYOUT[label],
            f"staged recursive successor {label}",
            exact_mode=0o600,
        ) != raw:
            raise RecursiveSuccessorMaterializationError(
                f"staged recursive successor artifact differs: {label}"
            )
    _revalidate_staged_generation(
        second,
        paths,
        identities,
        expected,
        require_future_absent=True,
    )
    _layout(root, committed=False)
    for label, raw in {
        "recursive_successor_contract": contract_raw,
        "task_bundle": bundle_raw,
    }.items():
        if _read_regular(
            root / _ARTIFACT_LAYOUT[label],
            f"final staged recursive successor {label}",
            exact_mode=0o600,
        ) != raw:
            raise RecursiveSuccessorMaterializationError(
                f"final staged recursive successor artifact differs: {label}"
            )
    marker_body = _marker_body(
        recursive_successor_id=recursive_successor_id,
        contract=contract,
        contract_raw=contract_raw,
        generation=second,
        expected=expected,
        artifacts=artifacts,
    )
    marker = {
        **marker_body,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "recursive_successor_digest": _digest(marker_body),
        },
    }
    marker_raw = _canonical_file(marker)
    def observe_future_absence_before_commit() -> None:
        final_future_attempt, final_checkpoint_root = _validate_future_attempt(
            paths["future_attempt_root"],
            recursive_successor_id,
            require_absent=True,
        )
        if (
            final_future_attempt != second["future_attempt"]
            or final_checkpoint_root != second["checkpoint_root"]
        ):
            raise RecursiveSuccessorMaterializationError(
                "future attempt binding changed before commit"
            )

    _write_new_bytes(
        root / _ARTIFACT_LAYOUT["recursive_successor_commit"],
        marker_raw,
        prepublish=observe_future_absence_before_commit,
    )
    _layout(root, committed=True)
    committed = _capture_recursive_successor(root, recursive_successor_id)
    expected_raws = {
        "recursive_successor_contract": contract_raw,
        "task_bundle": bundle_raw,
        "recursive_successor_commit": marker_raw,
    }
    if committed["raws"] != expected_raws:
        raise RecursiveSuccessorMaterializationError(
            "committed recursive successor raw artifacts differ"
        )
    return _result(marker, root, verified=False)


def verify_recursive_successor(
    publication_root,
    adoption_contract_path,
    adoption_root,
    source_successor_contract_path,
    source_successor_root,
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
    recursive_successor_contract_path,
    recursive_successor_root,
    future_attempt_root,
    *,
    adoption_id: str,
    source_successor_id: str,
    advance_id: str,
    recursive_successor_id: str,
    expected_adoption_digest: str,
    expected_source_pending_evidence_digest: str,
    expected_source_first_pending_projection_digest: str,
    expected_source_successor_digest: str,
    expected_source_bundle_digest: str,
    expected_source_plan_digest: str,
    completed_task_id: str,
    expected_completed_task_digest: str,
    expected_source_provenance_binding_digest: str,
    expected_source_authorization_digest: str,
    expected_source_attempt_digest: str,
    expected_source_execution_receipt_digest: str,
    expected_source_execution_journal_head_digest: str,
    expected_advance_digest: str,
    expected_advance_reingestion_digest: str,
    expected_advance_output_report_body_digest: str,
    expected_advance_output_audit_head: str,
    expected_advance_output_evidence_digest: str,
    expected_next_pending_evidence_digest: str,
    expected_next_first_pending_projection_digest: str,
    expected_recursive_successor_digest: str,
    expected_next_bundle_digest: str,
    expected_next_plan_digest: str,
) -> dict[str, Any]:
    """Read-only full verification of a recursive successor capsule."""
    paths = _paths(
        publication_root,
        adoption_contract_path,
        adoption_root,
        source_successor_contract_path,
        source_successor_root,
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
        recursive_successor_contract_path,
        recursive_successor_root,
        future_attempt_root,
    )
    identities = _identities(
        adoption_id=adoption_id,
        source_successor_id=source_successor_id,
        advance_id=advance_id,
        recursive_successor_id=recursive_successor_id,
        completed_task_id=completed_task_id,
    )
    expected = _expectations(
        expected_adoption_digest=expected_adoption_digest,
        expected_source_pending_evidence_digest=(
            expected_source_pending_evidence_digest
        ),
        expected_source_first_pending_projection_digest=(
            expected_source_first_pending_projection_digest
        ),
        expected_source_successor_digest=expected_source_successor_digest,
        expected_source_bundle_digest=expected_source_bundle_digest,
        expected_source_plan_digest=expected_source_plan_digest,
        expected_completed_task_digest=expected_completed_task_digest,
        expected_source_provenance_binding_digest=(
            expected_source_provenance_binding_digest
        ),
        expected_source_authorization_digest=(
            expected_source_authorization_digest
        ),
        expected_source_attempt_digest=expected_source_attempt_digest,
        expected_source_execution_receipt_digest=(
            expected_source_execution_receipt_digest
        ),
        expected_source_execution_journal_head_digest=(
            expected_source_execution_journal_head_digest
        ),
        expected_advance_digest=expected_advance_digest,
        expected_advance_reingestion_digest=(
            expected_advance_reingestion_digest
        ),
        expected_advance_output_report_body_digest=(
            expected_advance_output_report_body_digest
        ),
        expected_advance_output_audit_head=(
            expected_advance_output_audit_head
        ),
        expected_advance_output_evidence_digest=(
            expected_advance_output_evidence_digest
        ),
        expected_next_pending_evidence_digest=(
            expected_next_pending_evidence_digest
        ),
        expected_next_first_pending_projection_digest=(
            expected_next_first_pending_projection_digest
        ),
        expected_recursive_successor_digest=(
            expected_recursive_successor_digest
        ),
        expected_next_bundle_digest=expected_next_bundle_digest,
        expected_next_plan_digest=expected_next_plan_digest,
    )
    first = _validate_generation(
        paths,
        identities=identities,
        expected=expected,
        require_future_absent=False,
    )
    second = _validate_generation(
        paths,
        identities=identities,
        expected=expected,
        require_future_absent=False,
    )
    _same_generation(first, second)
    capsule = _capture_recursive_successor(
        paths["recursive_successor_root"], recursive_successor_id
    )
    contract_raw = second["contracts"]["raws"]["recursive_successor"]
    bundle_raw = _canonical_file(second["bundle"])
    marker_raw = capsule["raws"]["recursive_successor_commit"]
    marker = capsule["values"]["recursive_successor_commit"]
    if (
        capsule["raws"]["recursive_successor_contract"] != contract_raw
        or capsule["raws"]["task_bundle"] != bundle_raw
        or marker_raw != _canonical_file(marker)
    ):
        raise RecursiveSuccessorMaterializationError(
            "recursive successor capsule differs from verified generation"
        )
    artifacts = {
        "recursive_successor_contract": _artifact(
            _ARTIFACT_LAYOUT["recursive_successor_contract"], contract_raw
        ),
        "task_bundle": _artifact(
            _ARTIFACT_LAYOUT["task_bundle"], bundle_raw
        ),
    }
    marker_body = _marker_body(
        recursive_successor_id=recursive_successor_id,
        contract=second["contracts"]["values"]["recursive_successor"],
        contract_raw=contract_raw,
        generation=second,
        expected=expected,
        artifacts=artifacts,
    )
    expected_marker = {
        **marker_body,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "recursive_successor_digest": _digest(marker_body),
        },
    }
    if (
        marker != expected_marker
        or marker_raw != _canonical_file(expected_marker)
        or marker["integrity"]["recursive_successor_digest"]
        != expected["expected_recursive_successor_digest"]
        or marker["bundle_binding"]["bundle_digest"]
        != expected["expected_next_bundle_digest"]
        or marker["bundle_binding"]["plan_digest"]
        != expected["expected_next_plan_digest"]
    ):
        raise RecursiveSuccessorMaterializationError(
            "recursive successor marker or independent output anchors differ"
        )
    _revalidate_staged_generation(
        second,
        paths,
        identities,
        expected,
        require_future_absent=False,
    )
    again = _capture_recursive_successor(
        paths["recursive_successor_root"], recursive_successor_id
    )
    if again["raws"] != capsule["raws"]:
        raise RecursiveSuccessorMaterializationError(
            "recursive successor capsule changed during verification"
        )
    return _result(marker, capsule["root"], verified=True)


__all__ = [
    "RECURSIVE_SUCCESSOR_CAPSULE_SCHEMA_VERSION",
    "RECURSIVE_SUCCESSOR_CONTRACT_ID",
    "RECURSIVE_SUCCESSOR_SCHEMA_VERSION",
    "RecursiveSuccessorMaterializationError",
    "materialize_recursive_successor",
    "verify_recursive_successor",
]
