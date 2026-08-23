"""Immutable local adoption of one verified hypothesis report publication.

Adoption selects the report in one exact reingestion publication as a named
local report version.  It snapshots every raw publication leaf and commits
``adoption.json`` last.  It does not set a mutable current pointer or plan,
materialize, authorize, or execute any subsequent task.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from . import structural_hypothesis_reingestion_publisher as _publisher
from .structural_hypothesis_loop import canonical_json_bytes
from .structural_hypothesis_reingestion_publisher import (
    ReingestionPublicationError,
    verify_single_task_reingestion_publication,
)


ADOPTION_SCHEMA_VERSION = "sc-olh-kg.structural-hypothesis-report-adoption/1"
ADOPTION_CAPSULE_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-report-adoption-capsule/1"
)
ADOPTION_CONTRACT_ID = "structural_hypothesis_report_adoption_v1"

_ADOPTION_CONTRACT_DIGEST = (
    "sha256:66ed58df701c5e39924aa7b61574eb63126770290b1d9d1fabb60a9ebf8e13a1"
)
_PUBLISHER_CONTRACT_DIGEST = (
    "sha256:b94f745b0d07a1c21ecdcc3eb6ff20658d61fca490364ad20b1f86958489ff54"
)
_PUBLISHER_SOURCE_SHA256 = (
    "05d2ca6668414167cbe56a042f9564ff56728faf73ff5dd6fb167e9a16c3dd4d"
)

_STATE_PREFIX = Path("kg-op/structural-hypothesis-report-adoption/v1")
_ADOPTION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT_STATUS = "ADOPTED_AS_LOCAL_REPORT_VERSION_NOT_PLANNED"
_VERIFIED_STATUS = "VERIFIED_ADOPTED_AS_LOCAL_REPORT_VERSION_NOT_PLANNED"

_ARTIFACT_LAYOUT = {
    "adoption_contract": "adoption_contract.json",
    "publication_directory": "publication",
    "publication_execution_directory": "publication/execution",
    "publication_execution_input_directory": "publication/execution/inputs",
    "publication_execution_journal_directory": (
        "publication/execution/journal"
    ),
    "adoption_commit": "adoption.json",
}

_NONCLAIMS = [
    "local_report_version_is_not_global_current",
    "adoption_is_not_external_verification",
    "local_digest_is_not_signature",
    "no_external_authority",
    "no_currentness_claim",
    "single_success_is_not_scientific_confirmation",
    "single_success_is_not_scientific_refutation",
    "adopted_report_may_retain_evidence_gaps",
    "original_publication_and_full_source_chain_are_required_for_full_verification",
    "no_same_user_rewrite_defense_without_independent_expected_digests",
    "no_network_access",
    "no_scheduler_access",
    "no_credential_access",
    "no_shell_execution",
    "no_task_planning",
    "no_task_materialization",
    "no_task_authorization",
    "no_task_execution",
    "no_benchmark_execution",
    "no_next_task_admission",
    "no_paper_promotion",
]


class StructuralHypothesisReportAdoptionError(ValueError):
    """Raised when V1 local report adoption fails closed."""


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _raw_digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _canonical_file(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _require_digest(value: Any, label: str) -> str:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise StructuralHypothesisReportAdoptionError(
            f"{label} must be a lowercase sha256 digest"
        )
    return value


def _translate(action):
    try:
        return action()
    except ReingestionPublicationError as error:
        raise StructuralHypothesisReportAdoptionError(str(error)) from error


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _validate_adoption_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema_version") != ADOPTION_SCHEMA_VERSION
        or contract.get("contract_id") != ADOPTION_CONTRACT_ID
        or contract.get("artifact_layout") != _ARTIFACT_LAYOUT
        or contract.get("nonclaims") != _NONCLAIMS
        or _digest(contract) != _ADOPTION_CONTRACT_DIGEST
    ):
        raise StructuralHypothesisReportAdoptionError(
            "adoption contract differs from frozen V1"
        )
    source = contract.get("source_contract")
    if type(source) is not dict or source != {
        "publisher_contract_id": _publisher.PUBLISHER_CONTRACT_ID,
        "publisher_contract_digest": _PUBLISHER_CONTRACT_DIGEST,
        "publication_schema_version": _publisher.PUBLICATION_SCHEMA_VERSION,
        "required_publication_status": "PUBLISHED_NOT_ADOPTED",
    }:
        raise StructuralHypothesisReportAdoptionError(
            "adoption source contract differs"
        )
    admission = contract.get("admission")
    if type(admission) is not dict or admission != {
        "full_publication_verification_required": True,
        "independent_publication_anchors_required": True,
        "independent_publication_raw_anchors_required": True,
        "exact_publication_raw_leaf_snapshot_required": True,
        "adopted_report_artifact": "publication/output_report.json",
        "adopted_evidence_artifact": "publication/combined_rows.json",
        "adopted_status": _COMMIT_STATUS,
        "planning_status": "NOT_PLANNED",
    }:
        raise StructuralHypothesisReportAdoptionError(
            "adoption admission differs"
        )
    if contract.get("publication_snapshot_layout") != (
        _publisher._ARTIFACT_LAYOUT
    ):
        raise StructuralHypothesisReportAdoptionError(
            "adoption publication snapshot layout differs"
        )
    source_files = contract.get("source_files")
    expected_source_files = {
        "performance/structural_hypothesis_reingestion_publisher.py": (
            _PUBLISHER_SOURCE_SHA256
        )
    }
    if source_files != expected_source_files:
        raise StructuralHypothesisReportAdoptionError(
            "adoption source-file commitments differ"
        )
    publisher_source = _repo_root() / next(iter(expected_source_files))
    raw = _translate(
        lambda: _publisher._read_regular(
            publisher_source, "adoption publisher source"
        )
    )
    if hashlib.sha256(raw).hexdigest() != _PUBLISHER_SOURCE_SHA256:
        raise StructuralHypothesisReportAdoptionError(
            "adoption publisher source differs"
        )


def _absolute_path(value: str | Path, label: str) -> Path:
    return _translate(lambda: _publisher._absolute_path(value, label))


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    return _translate(lambda: _publisher._read_json(path, label))


def _state_base() -> Path:
    configured = os.environ.get("XDG_STATE_HOME")
    if configured:
        base = Path(configured)
        if not base.is_absolute():
            raise StructuralHypothesisReportAdoptionError(
                "XDG_STATE_HOME must be absolute"
            )
        return base
    return Path.home() / ".local/state"


def _secure_directory(path: Path, *, exact_mode: int | None = None) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise StructuralHypothesisReportAdoptionError(
            f"missing adoption directory: {path}"
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
        raise StructuralHypothesisReportAdoptionError(
            f"adoption directory ownership or mode differs: {path}"
        )


def _ensure_secure_parent(target: Path) -> None:
    base = _state_base()
    prefix = base / _STATE_PREFIX
    if target.parent != prefix:
        raise StructuralHypothesisReportAdoptionError(
            "adoption_root is outside the frozen local state prefix"
        )
    _translate(lambda: _publisher._reject_symlink_components(target))
    missing = []
    cursor = prefix
    while not cursor.exists():
        if cursor.is_symlink():
            raise StructuralHypothesisReportAdoptionError(
                f"adoption directory alias is forbidden: {cursor}"
            )
        missing.append(cursor)
        if cursor == base:
            break
        cursor = cursor.parent
    if not cursor.exists():
        raise StructuralHypothesisReportAdoptionError(
            "state-home ancestor is missing"
        )
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


def _validate_adoption_location(
    adoption_root: str | Path,
    adoption_id: str | None,
    *,
    fresh: bool,
) -> Path:
    root = _absolute_path(adoption_root, "adoption_root")
    if adoption_id is not None:
        if type(adoption_id) is not str or not _ADOPTION_ID.fullmatch(
            adoption_id
        ):
            raise StructuralHypothesisReportAdoptionError(
                "adoption_id is invalid"
            )
        if root.name != adoption_id:
            raise StructuralHypothesisReportAdoptionError(
                "adoption_id must equal adoption_root basename"
            )
    expected_parent = _state_base() / _STATE_PREFIX
    if root.parent != expected_parent:
        raise StructuralHypothesisReportAdoptionError(
            "adoption_root is outside the frozen local state prefix"
        )
    if fresh:
        if root.exists() or root.is_symlink():
            raise StructuralHypothesisReportAdoptionError(
                "adoption_root already exists"
            )
        _ensure_secure_parent(root)
    else:
        _translate(lambda: _publisher._reject_symlink_components(root))
        _secure_directory(root, exact_mode=0o700)
    return root


def _mkdir_new(path: Path) -> None:
    _translate(lambda: _publisher._mkdir_new(path))


def _write_new_bytes(path: Path, raw: bytes) -> None:
    _translate(lambda: _publisher._write_new_bytes(path, raw))


def _read_regular(
    path: Path, label: str, *, exact_mode: int | None = None
) -> bytes:
    return _translate(
        lambda: _publisher._read_regular(
            path, label, exact_mode=exact_mode
        )
    )


def _publication_relative_files() -> dict[str, str]:
    return {
        label: relative
        for label, relative in _publisher._ARTIFACT_LAYOUT.items()
        if label
        not in {
            "execution_directory",
            "execution_input_directory",
            "execution_journal_directory",
        }
    }


def _check_open_directory(descriptor: int, label: str) -> None:
    info = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise StructuralHypothesisReportAdoptionError(
            f"{label} ownership, type, or mode differs"
        )


def _open_directory_at(parent: int, name: str, label: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=parent)
    except OSError as error:
        raise StructuralHypothesisReportAdoptionError(
            f"cannot open {label}"
        ) from error
    try:
        _check_open_directory(descriptor, label)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_regular_at(parent: int, name: str, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=parent)
    except OSError as error:
        raise StructuralHypothesisReportAdoptionError(
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
            raise StructuralHypothesisReportAdoptionError(
                f"{label} ownership, type, mode, or link count differs"
            )
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise StructuralHypothesisReportAdoptionError(
                f"{label} changed while captured"
            )
        raw = b"".join(chunks)
        if len(raw) != before.st_size:
            raise StructuralHypothesisReportAdoptionError(
                f"{label} byte count changed"
            )
        return raw
    finally:
        os.close(descriptor)


def _require_names(
    descriptor: int, expected: set[str], label: str
) -> None:
    try:
        observed = set(os.listdir(descriptor))
    except OSError as error:
        raise StructuralHypothesisReportAdoptionError(
            f"cannot enumerate {label}"
        ) from error
    if observed != expected:
        raise StructuralHypothesisReportAdoptionError(
            f"{label} has missing or unexpected artifacts"
        )


def _capture_publication(publication_root: Path) -> dict[str, Any]:
    """Capture one generation through held O_NOFOLLOW directory handles."""
    root = _translate(
        lambda: _publisher._validate_publication_location(
            publication_root, None, fresh=False
        )
    )
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        root_fd = os.open(root, flags)
    except OSError as error:
        raise StructuralHypothesisReportAdoptionError(
            "cannot open source publication root"
        ) from error
    descriptors = [root_fd]
    try:
        _check_open_directory(root_fd, "source publication root")
        execution_fd = _open_directory_at(
            root_fd, "execution", "source publication execution"
        )
        descriptors.append(execution_fd)
        inputs_fd = _open_directory_at(
            execution_fd, "inputs", "source publication inputs"
        )
        descriptors.append(inputs_fd)
        journal_fd = _open_directory_at(
            execution_fd, "journal", "source publication journal"
        )
        descriptors.append(journal_fd)
        directory_fds = {
            "": root_fd,
            "execution": execution_fd,
            "execution/inputs": inputs_fd,
            "execution/journal": journal_fd,
        }
        expected_names = {key: set() for key in directory_fds}
        for relative in _publisher._ARTIFACT_LAYOUT.values():
            path = Path(relative)
            parent = os.fspath(path.parent)
            if parent == ".":
                parent = ""
            expected_names[parent].add(path.name)
        raws = {}
        for label, relative in _publication_relative_files().items():
            path = Path(relative)
            parent = os.fspath(path.parent)
            if parent == ".":
                parent = ""
            raws[label] = _read_regular_at(
                directory_fds[parent], path.name, f"publication {label}"
            )
        # Re-enumerate after all leaf reads while every directory generation
        # remains held.  The marker below binds the resulting exact byte map.
        for relative, descriptor in directory_fds.items():
            _require_names(
                descriptor,
                expected_names[relative],
                f"publication directory {relative or '.'}",
            )
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    values = {}
    for label, raw in raws.items():
        if label == "base_evidence_csv":
            continue
        values[label] = _translate(
            lambda raw=raw, label=label: (
                _publisher._parse_json_value(raw, label)
                if label == "combined_rows"
                else _publisher._parse_json(raw, label)
            )
        )
    if type(values["combined_rows"]) is not list:
        raise StructuralHypothesisReportAdoptionError(
            "captured combined_rows JSON must be an array"
        )
    return {"root": root, "values": values, "raws": raws}


def _same_publication_snapshot(captured: Mapping[str, Any]) -> None:
    again = _capture_publication(captured["root"])
    if again["raws"] != captured["raws"]:
        raise StructuralHypothesisReportAdoptionError(
            "source publication changed before adoption commit"
        )


def _validate_captured_publication(
    captured: Mapping[str, Any],
    *,
    base_evidence_csv: Path,
    attempt_root: Path,
    hypothesis_contract_path: Path,
    executor_contract_path: Path,
    runtime_contract_path: Path,
    publisher_contract_path: Path,
    base_manifest_path: Path,
    asset_root: Path,
    expectations: Mapping[str, str],
) -> None:
    """Replay publication semantics against the exact captured raw map."""
    source = _translate(
        lambda: _publisher._snapshot_inputs(
            base_evidence_csv,
            attempt_root,
            hypothesis_contract_path,
            executor_contract_path,
            publisher_contract_path,
            runtime_contract_path,
        )
    )
    _translate(
        lambda: _publisher._validate_captured_runtime_capsule(
            source["objects"],
            source["runtime_contract"],
            base_manifest_path,
            asset_root,
            attempt_root,
            expected_plan_digest=expectations["expected_plan_digest"],
            expected_authorization_digest=expectations[
                "expected_authorization_digest"
            ],
            expected_execution_receipt_digest=expectations[
                "expected_execution_receipt_digest"
            ],
            expected_execution_journal_head_digest=expectations[
                "expected_execution_journal_head_digest"
            ],
            expected_execution_attempt_digest=expectations[
                "expected_execution_attempt_digest"
            ],
        )
    )
    _translate(
        lambda: _publisher._verify_runtime_attempt(
            attempt_root,
            source["runtime_contract"],
            base_manifest_path,
            asset_root,
            expected_authorization_digest=expectations[
                "expected_authorization_digest"
            ],
            expected_execution_receipt_digest=expectations[
                "expected_execution_receipt_digest"
            ],
            expected_execution_journal_head_digest=expectations[
                "expected_execution_journal_head_digest"
            ],
            expected_execution_attempt_digest=expectations[
                "expected_execution_attempt_digest"
            ],
        )
    )
    output, reingestion, combined = _translate(
        lambda: _publisher._validate_chain_objects(
            source["objects"],
            source["publisher_contract"],
            source["runtime_contract"],
            source["base_rows"],
            source["base_raw"],
            base_evidence_csv,
            source["hypothesis_raw"],
            hypothesis_contract_path,
            expected_source_evidence_digest=expectations[
                "expected_source_evidence_digest"
            ],
            expected_plan_digest=expectations["expected_plan_digest"],
            expected_authorization_digest=expectations[
                "expected_authorization_digest"
            ],
            expected_execution_receipt_digest=expectations[
                "expected_execution_receipt_digest"
            ],
        )
    )
    expected_raws = {
        "publisher_contract": source["publisher_raw"],
        "runtime_contract": source["runtime_raw"],
        "materializer_contract": source["attempt_raws"][
            "execution_materializer_contract"
        ],
        "source_hypothesis_contract": source["hypothesis_raw"],
        "source_executor_contract": source["executor_raw"],
        "base_evidence_csv": source["base_raw"],
        **source["attempt_raws"],
        "combined_rows": _canonical_file(combined),
        "output_report": _canonical_file(output),
        "reingestion_receipt": _canonical_file(reingestion),
    }
    for label, expected_raw in expected_raws.items():
        if captured["raws"].get(label) != expected_raw:
            raise StructuralHypothesisReportAdoptionError(
                f"captured publication semantic artifact differs: {label}"
            )
    marker = captured["values"]["publication_commit"]
    marker_raw = captured["raws"]["publication_commit"]
    integrity = marker.get("integrity")
    if (
        type(integrity) is not dict
        or set(integrity) != {"algorithm", "publication_digest"}
        or integrity.get("algorithm") != "sha256-canonical-json-v1"
    ):
        raise StructuralHypothesisReportAdoptionError(
            "captured publication integrity is malformed"
        )
    marker_body = {key: marker[key] for key in marker if key != "integrity"}
    if (
        _digest(marker_body) != integrity.get("publication_digest")
        or integrity.get("publication_digest")
        != expectations["expected_publication_digest"]
        or marker.get("schema_version")
        != _publisher.PUBLICATION_SCHEMA_VERSION
        or marker.get("status") != "PUBLISHED_NOT_ADOPTED"
        or marker.get("publication_id") != captured["root"].name
    ):
        raise StructuralHypothesisReportAdoptionError(
            "captured publication marker or independent digest differs"
        )
    expected_artifacts = {
        label: {
            "path": _publisher._ARTIFACT_LAYOUT[label],
            "sha256": _raw_digest(raw),
            "bytes": len(raw),
        }
        for label, raw in expected_raws.items()
    }
    if marker.get("artifacts") != expected_artifacts:
        raise StructuralHypothesisReportAdoptionError(
            "captured publication raw leaves differ from marker"
        )
    rebuilt_body = _publisher._marker_body(
        publication_id=marker["publication_id"],
        artifacts=expected_artifacts,
        publisher_contract=source["publisher_contract"],
        publisher_raw=source["publisher_raw"],
        hypothesis_raw=source["hypothesis_raw"],
        executor_raw=source["executor_raw"],
        base_raw=source["base_raw"],
        base_rows=source["base_rows"],
        objects=source["objects"],
        output=output,
        reingestion=reingestion,
        combined=combined,
        expected_plan_digest=expectations["expected_plan_digest"],
        expected_authorization_digest=expectations[
            "expected_authorization_digest"
        ],
        expected_execution_receipt_digest=expectations[
            "expected_execution_receipt_digest"
        ],
        expected_execution_journal_head_digest=expectations[
            "expected_execution_journal_head_digest"
        ],
        expected_execution_attempt_digest=expectations[
            "expected_execution_attempt_digest"
        ],
    )
    if marker_body != rebuilt_body:
        raise StructuralHypothesisReportAdoptionError(
            "captured publication marker does not reproduce its full chain"
        )
    raw_anchors = {
        "expected_publication_marker_raw_sha256": _raw_digest(marker_raw),
        "expected_combined_rows_raw_sha256": _raw_digest(
            captured["raws"]["combined_rows"]
        ),
        "expected_output_report_raw_sha256": _raw_digest(
            captured["raws"]["output_report"]
        ),
        "expected_reingestion_receipt_raw_sha256": _raw_digest(
            captured["raws"]["reingestion_receipt"]
        ),
    }
    if any(
        expectations[label] != observed
        for label, observed in raw_anchors.items()
    ):
        raise StructuralHypothesisReportAdoptionError(
            "captured publication independent raw anchor differs"
        )
    if (
        reingestion["integrity"]["reingestion_digest"]
        != expectations["expected_reingestion_digest"]
        or output["audit"]["report_body_digest"]
        != expectations["expected_output_report_body_digest"]
        or output["audit"]["head"]
        != expectations["expected_output_audit_head"]
        or output["evidence_digest"]
        != expectations["expected_output_evidence_digest"]
    ):
        raise StructuralHypothesisReportAdoptionError(
            "captured publication expected output digest differs"
        )


def _verify_publication(
    *,
    publication_root: Path,
    base_evidence_csv: Path,
    attempt_root: Path,
    hypothesis_contract_path: Path,
    executor_contract_path: Path,
    runtime_contract_path: Path,
    publisher_contract_path: Path,
    base_manifest_path: Path,
    asset_root: Path,
    expectations: Mapping[str, str],
) -> dict[str, Any]:
    try:
        publication_expectations = {
            key: value
            for key, value in expectations.items()
            if key
            not in {
                "expected_publication_marker_raw_sha256",
                "expected_combined_rows_raw_sha256",
                "expected_output_report_raw_sha256",
                "expected_reingestion_receipt_raw_sha256",
            }
        }
        return verify_single_task_reingestion_publication(
            base_evidence_csv,
            attempt_root,
            hypothesis_contract_path,
            executor_contract_path,
            runtime_contract_path,
            publisher_contract_path,
            base_manifest_path,
            asset_root,
            publication_root,
            **publication_expectations,
        )
    except (ReingestionPublicationError, ValueError) as error:
        raise StructuralHypothesisReportAdoptionError(
            "source publication failed full verification"
        ) from error


def _artifact(path: str, raw: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": _raw_digest(raw), "bytes": len(raw)}


def _root_names(root: Path) -> set[str]:
    _secure_directory(root, exact_mode=0o700)
    try:
        return {child.name for child in root.iterdir()}
    except OSError as error:
        raise StructuralHypothesisReportAdoptionError(
            f"cannot enumerate adoption directory: {root}"
        ) from error


def _validate_capsule_layout(root: Path, *, committed: bool = True) -> None:
    expected = {"adoption_contract.json", "publication"}
    if committed:
        expected.add("adoption.json")
    if _root_names(root) != expected:
        raise StructuralHypothesisReportAdoptionError(
            "adoption root has missing or unexpected artifacts"
        )
    publication = root / "publication"
    expected_publication = {
        Path(relative).parts[0]
        for relative in _publication_relative_files().values()
    }
    if _root_names(publication) != expected_publication:
        raise StructuralHypothesisReportAdoptionError(
            "adoption publication snapshot layout differs"
        )
    if _root_names(publication / "execution") != {
        "attempt.json",
        "bundle.json",
        "authorization.json",
        "preflight.json",
        "raw_result.json",
        "receipt.json",
        "inputs",
        "journal",
    }:
        raise StructuralHypothesisReportAdoptionError(
            "adoption execution snapshot layout differs"
        )
    if _root_names(publication / "execution/inputs") != {
        "report.json",
        "hypothesis_contract.json",
        "executor_contract.json",
        "materializer_contract.json",
    }:
        raise StructuralHypothesisReportAdoptionError(
            "adoption input snapshot layout differs"
        )
    if _root_names(publication / "execution/journal") != {
        "0000_AUTHORIZED.json",
        "0001_RUNNING.json",
        "0002_COMPLETED.json",
    }:
        raise StructuralHypothesisReportAdoptionError(
            "adoption journal snapshot layout differs"
        )


def _marker_body(
    *,
    adoption_id: str,
    adoption_contract: Mapping[str, Any],
    adoption_contract_raw: bytes,
    publication_marker: Mapping[str, Any],
    publication_raws: Mapping[str, bytes],
    output_report: Mapping[str, Any],
    combined_rows: list[Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    outputs = publication_marker["outputs"]
    return {
        "schema_version": ADOPTION_CAPSULE_SCHEMA_VERSION,
        "status": _COMMIT_STATUS,
        "planning_status": "NOT_PLANNED",
        "adoption_id": adoption_id,
        "adoption_contract_binding": {
            "contract_id": adoption_contract["contract_id"],
            "contract_digest": _digest(adoption_contract),
            "raw_sha256": _raw_digest(adoption_contract_raw),
            "bytes": len(adoption_contract_raw),
        },
        "source_publication": {
            "publication_id": publication_marker["publication_id"],
            "publication_status": publication_marker["status"],
            "publication_digest": publication_marker["integrity"][
                "publication_digest"
            ],
            "publication_marker_raw_sha256": _raw_digest(
                publication_raws["publication_commit"]
            ),
            "publication_marker_bytes": len(
                publication_raws["publication_commit"]
            ),
            "reingestion_digest": outputs["reingestion_digest"],
        },
        "adopted_report": {
            "report_artifact": "publication/output_report.json",
            "combined_evidence_artifact": "publication/combined_rows.json",
            "status": output_report["status"],
            "report_body_digest": output_report["audit"][
                "report_body_digest"
            ],
            "audit_head": output_report["audit"]["head"],
            "evidence_digest": output_report["evidence_digest"],
            "combined_rows_count": len(combined_rows),
            "pending_evidence_count": len(output_report["pending_evidence"]),
        },
        "artifacts": dict(artifacts),
        "nonclaims": list(_NONCLAIMS),
    }


def _result(
    marker: Mapping[str, Any], root: Path, *, verified: bool = False
) -> dict[str, Any]:
    report = marker["adopted_report"]
    source = marker["source_publication"]
    return {
        "status": _VERIFIED_STATUS if verified else _COMMIT_STATUS,
        "adoption_root": str(root),
        "adoption_digest": marker["integrity"]["adoption_digest"],
        "publication_digest": source["publication_digest"],
        "reingestion_digest": source["reingestion_digest"],
        "output_report_body_digest": report["report_body_digest"],
        "output_audit_head": report["audit_head"],
        "output_evidence_digest": report["evidence_digest"],
        "planning_status": marker["planning_status"],
    }


def _normalize_inputs(
    *,
    publication_root: str | Path,
    adoption_contract_path: str | Path,
    adoption_root: str | Path,
    base_evidence_csv: str | Path,
    attempt_root: str | Path,
    hypothesis_contract_path: str | Path,
    executor_contract_path: str | Path,
    runtime_contract_path: str | Path,
    publisher_contract_path: str | Path,
    base_manifest_path: str | Path,
    asset_root: str | Path,
) -> dict[str, Path]:
    return {
        "publication_root": _absolute_path(
            publication_root, "publication_root"
        ),
        "adoption_contract_path": _absolute_path(
            adoption_contract_path, "adoption_contract_path"
        ),
        "adoption_root": _absolute_path(adoption_root, "adoption_root"),
        "base_evidence_csv": _absolute_path(
            base_evidence_csv, "base_evidence_csv"
        ),
        "attempt_root": _absolute_path(attempt_root, "attempt_root"),
        "hypothesis_contract_path": _absolute_path(
            hypothesis_contract_path, "hypothesis_contract_path"
        ),
        "executor_contract_path": _absolute_path(
            executor_contract_path, "executor_contract_path"
        ),
        "runtime_contract_path": _absolute_path(
            runtime_contract_path, "runtime_contract_path"
        ),
        "publisher_contract_path": _absolute_path(
            publisher_contract_path, "publisher_contract_path"
        ),
        "base_manifest_path": _absolute_path(
            base_manifest_path, "base_manifest_path"
        ),
        "asset_root": _absolute_path(asset_root, "asset_root"),
    }


def _expectations(**values: str) -> dict[str, str]:
    return {
        label: _require_digest(value, label)
        for label, value in values.items()
    }


def adopt_structural_hypothesis_report(
    publication_root: str | Path,
    adoption_contract_path: str | Path,
    adoption_root: str | Path,
    base_evidence_csv: str | Path,
    attempt_root: str | Path,
    hypothesis_contract_path: str | Path,
    executor_contract_path: str | Path,
    runtime_contract_path: str | Path,
    publisher_contract_path: str | Path,
    base_manifest_path: str | Path,
    asset_root: str | Path,
    *,
    adoption_id: str,
    expected_source_evidence_digest: str,
    expected_plan_digest: str,
    expected_authorization_digest: str,
    expected_execution_receipt_digest: str,
    expected_execution_journal_head_digest: str,
    expected_execution_attempt_digest: str,
    expected_publication_digest: str,
    expected_reingestion_digest: str,
    expected_output_report_body_digest: str,
    expected_output_audit_head: str,
    expected_output_evidence_digest: str,
    expected_publication_marker_raw_sha256: str,
    expected_combined_rows_raw_sha256: str,
    expected_output_report_raw_sha256: str,
    expected_reingestion_receipt_raw_sha256: str,
) -> dict[str, Any]:
    """Adopt one fully verified publication as a named local report version."""
    expected = _expectations(
        expected_source_evidence_digest=expected_source_evidence_digest,
        expected_plan_digest=expected_plan_digest,
        expected_authorization_digest=expected_authorization_digest,
        expected_execution_receipt_digest=expected_execution_receipt_digest,
        expected_execution_journal_head_digest=(
            expected_execution_journal_head_digest
        ),
        expected_execution_attempt_digest=expected_execution_attempt_digest,
        expected_publication_digest=expected_publication_digest,
        expected_reingestion_digest=expected_reingestion_digest,
        expected_output_report_body_digest=expected_output_report_body_digest,
        expected_output_audit_head=expected_output_audit_head,
        expected_output_evidence_digest=expected_output_evidence_digest,
        expected_publication_marker_raw_sha256=(
            expected_publication_marker_raw_sha256
        ),
        expected_combined_rows_raw_sha256=(
            expected_combined_rows_raw_sha256
        ),
        expected_output_report_raw_sha256=expected_output_report_raw_sha256,
        expected_reingestion_receipt_raw_sha256=(
            expected_reingestion_receipt_raw_sha256
        ),
    )
    paths = _normalize_inputs(
        publication_root=publication_root,
        adoption_contract_path=adoption_contract_path,
        adoption_root=adoption_root,
        base_evidence_csv=base_evidence_csv,
        attempt_root=attempt_root,
        hypothesis_contract_path=hypothesis_contract_path,
        executor_contract_path=executor_contract_path,
        runtime_contract_path=runtime_contract_path,
        publisher_contract_path=publisher_contract_path,
        base_manifest_path=base_manifest_path,
        asset_root=asset_root,
    )
    root = _validate_adoption_location(
        paths["adoption_root"], adoption_id, fresh=True
    )
    contract, contract_raw = _read_json(
        paths["adoption_contract_path"], "adoption contract"
    )
    _validate_adoption_contract(contract)
    captured = _capture_publication(paths["publication_root"])
    _validate_captured_publication(
        captured,
        base_evidence_csv=paths["base_evidence_csv"],
        attempt_root=paths["attempt_root"],
        hypothesis_contract_path=paths["hypothesis_contract_path"],
        executor_contract_path=paths["executor_contract_path"],
        runtime_contract_path=paths["runtime_contract_path"],
        publisher_contract_path=paths["publisher_contract_path"],
        base_manifest_path=paths["base_manifest_path"],
        asset_root=paths["asset_root"],
        expectations=expected,
    )
    _verify_publication(
        publication_root=paths["publication_root"],
        base_evidence_csv=paths["base_evidence_csv"],
        attempt_root=paths["attempt_root"],
        hypothesis_contract_path=paths["hypothesis_contract_path"],
        executor_contract_path=paths["executor_contract_path"],
        runtime_contract_path=paths["runtime_contract_path"],
        publisher_contract_path=paths["publisher_contract_path"],
        base_manifest_path=paths["base_manifest_path"],
        asset_root=paths["asset_root"],
        expectations=expected,
    )
    _same_publication_snapshot(captured)
    contract_again, contract_raw_again = _read_json(
        paths["adoption_contract_path"], "adoption contract"
    )
    if contract_again != contract or contract_raw_again != contract_raw:
        raise StructuralHypothesisReportAdoptionError(
            "adoption contract changed before root creation"
        )

    _mkdir_new(root)
    _mkdir_new(root / "publication")
    _mkdir_new(root / "publication/execution")
    _mkdir_new(root / "publication/execution/inputs")
    _mkdir_new(root / "publication/execution/journal")

    raw_files = {"adoption_contract": contract_raw}
    artifacts = {
        "adoption_contract": _artifact("adoption_contract.json", contract_raw)
    }
    _write_new_bytes(root / "adoption_contract.json", contract_raw)
    for label, relative in _publication_relative_files().items():
        raw = captured["raws"][label]
        capsule_relative = f"publication/{relative}"
        raw_files[f"publication:{label}"] = raw
        artifacts[f"publication:{label}"] = _artifact(
            capsule_relative, raw
        )
        _write_new_bytes(root / capsule_relative, raw)

    _validate_capsule_layout(root, committed=False)
    for label, raw in raw_files.items():
        relative = artifacts[label]["path"]
        if _read_regular(
            root / relative,
            f"pre-commit adoption artifact {label}",
            exact_mode=0o600,
        ) != raw:
            raise StructuralHypothesisReportAdoptionError(
                f"pre-commit adoption artifact differs: {label}"
            )
    _verify_publication(
        publication_root=paths["publication_root"],
        base_evidence_csv=paths["base_evidence_csv"],
        attempt_root=paths["attempt_root"],
        hypothesis_contract_path=paths["hypothesis_contract_path"],
        executor_contract_path=paths["executor_contract_path"],
        runtime_contract_path=paths["runtime_contract_path"],
        publisher_contract_path=paths["publisher_contract_path"],
        base_manifest_path=paths["base_manifest_path"],
        asset_root=paths["asset_root"],
        expectations=expected,
    )
    _same_publication_snapshot(captured)
    # The final source verification can be comparatively slow.  Re-check the
    # exact staged layout and every staged byte immediately before allowing
    # the commit marker to appear.
    _validate_capsule_layout(root, committed=False)
    for label, raw in raw_files.items():
        relative = artifacts[label]["path"]
        if _read_regular(
            root / relative,
            f"final pre-commit adoption artifact {label}",
            exact_mode=0o600,
        ) != raw:
            raise StructuralHypothesisReportAdoptionError(
                f"final pre-commit adoption artifact differs: {label}"
            )

    marker_body = _marker_body(
        adoption_id=adoption_id,
        adoption_contract=contract,
        adoption_contract_raw=contract_raw,
        publication_marker=captured["values"]["publication_commit"],
        publication_raws=captured["raws"],
        output_report=captured["values"]["output_report"],
        combined_rows=captured["values"]["combined_rows"],
        artifacts=artifacts,
    )
    marker = {
        **marker_body,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "adoption_digest": _digest(marker_body),
        },
    }
    marker_raw = _canonical_file(marker)
    _write_new_bytes(root / "adoption.json", marker_raw)
    _validate_capsule_layout(root)
    for label, raw in {
        **raw_files,
        "adoption_commit": marker_raw,
    }.items():
        relative = (
            "adoption.json"
            if label == "adoption_commit"
            else artifacts[label]["path"]
        )
        if _read_regular(
            root / relative,
            f"committed adoption artifact {label}",
            exact_mode=0o600,
        ) != raw:
            raise StructuralHypothesisReportAdoptionError(
                f"committed adoption artifact differs: {label}"
            )
    return _result(marker, root)


def verify_structural_hypothesis_report_adoption(
    publication_root: str | Path,
    adoption_contract_path: str | Path,
    adoption_root: str | Path,
    base_evidence_csv: str | Path,
    attempt_root: str | Path,
    hypothesis_contract_path: str | Path,
    executor_contract_path: str | Path,
    runtime_contract_path: str | Path,
    publisher_contract_path: str | Path,
    base_manifest_path: str | Path,
    asset_root: str | Path,
    *,
    adoption_id: str,
    expected_source_evidence_digest: str,
    expected_plan_digest: str,
    expected_authorization_digest: str,
    expected_execution_receipt_digest: str,
    expected_execution_journal_head_digest: str,
    expected_execution_attempt_digest: str,
    expected_publication_digest: str,
    expected_reingestion_digest: str,
    expected_output_report_body_digest: str,
    expected_output_audit_head: str,
    expected_output_evidence_digest: str,
    expected_publication_marker_raw_sha256: str,
    expected_combined_rows_raw_sha256: str,
    expected_output_report_raw_sha256: str,
    expected_reingestion_receipt_raw_sha256: str,
    expected_adoption_digest: str,
) -> dict[str, Any]:
    """Read-only verification of one committed local report adoption."""
    expected = _expectations(
        expected_source_evidence_digest=expected_source_evidence_digest,
        expected_plan_digest=expected_plan_digest,
        expected_authorization_digest=expected_authorization_digest,
        expected_execution_receipt_digest=expected_execution_receipt_digest,
        expected_execution_journal_head_digest=(
            expected_execution_journal_head_digest
        ),
        expected_execution_attempt_digest=expected_execution_attempt_digest,
        expected_publication_digest=expected_publication_digest,
        expected_reingestion_digest=expected_reingestion_digest,
        expected_output_report_body_digest=expected_output_report_body_digest,
        expected_output_audit_head=expected_output_audit_head,
        expected_output_evidence_digest=expected_output_evidence_digest,
        expected_publication_marker_raw_sha256=(
            expected_publication_marker_raw_sha256
        ),
        expected_combined_rows_raw_sha256=(
            expected_combined_rows_raw_sha256
        ),
        expected_output_report_raw_sha256=expected_output_report_raw_sha256,
        expected_reingestion_receipt_raw_sha256=(
            expected_reingestion_receipt_raw_sha256
        ),
    )
    expected_adoption_digest = _require_digest(
        expected_adoption_digest, "expected_adoption_digest"
    )
    paths = _normalize_inputs(
        publication_root=publication_root,
        adoption_contract_path=adoption_contract_path,
        adoption_root=adoption_root,
        base_evidence_csv=base_evidence_csv,
        attempt_root=attempt_root,
        hypothesis_contract_path=hypothesis_contract_path,
        executor_contract_path=executor_contract_path,
        runtime_contract_path=runtime_contract_path,
        publisher_contract_path=publisher_contract_path,
        base_manifest_path=base_manifest_path,
        asset_root=asset_root,
    )
    root = _validate_adoption_location(
        paths["adoption_root"], adoption_id, fresh=False
    )
    _validate_capsule_layout(root)
    contract, contract_raw = _read_json(
        paths["adoption_contract_path"], "adoption contract"
    )
    _validate_adoption_contract(contract)
    captured = _capture_publication(paths["publication_root"])
    _validate_captured_publication(
        captured,
        base_evidence_csv=paths["base_evidence_csv"],
        attempt_root=paths["attempt_root"],
        hypothesis_contract_path=paths["hypothesis_contract_path"],
        executor_contract_path=paths["executor_contract_path"],
        runtime_contract_path=paths["runtime_contract_path"],
        publisher_contract_path=paths["publisher_contract_path"],
        base_manifest_path=paths["base_manifest_path"],
        asset_root=paths["asset_root"],
        expectations=expected,
    )
    _verify_publication(
        publication_root=paths["publication_root"],
        base_evidence_csv=paths["base_evidence_csv"],
        attempt_root=paths["attempt_root"],
        hypothesis_contract_path=paths["hypothesis_contract_path"],
        executor_contract_path=paths["executor_contract_path"],
        runtime_contract_path=paths["runtime_contract_path"],
        publisher_contract_path=paths["publisher_contract_path"],
        base_manifest_path=paths["base_manifest_path"],
        asset_root=paths["asset_root"],
        expectations=expected,
    )
    _same_publication_snapshot(captured)

    capsule_contract_raw = _read_regular(
        root / "adoption_contract.json",
        "capsule adoption contract",
        exact_mode=0o600,
    )
    if capsule_contract_raw != contract_raw:
        raise StructuralHypothesisReportAdoptionError(
            "capsule adoption contract differs"
        )
    artifacts = {
        "adoption_contract": _artifact(
            "adoption_contract.json", capsule_contract_raw
        )
    }
    for label, relative in _publication_relative_files().items():
        capsule_relative = f"publication/{relative}"
        raw = _read_regular(
            root / capsule_relative,
            f"capsule publication artifact {label}",
            exact_mode=0o600,
        )
        if raw != captured["raws"][label]:
            raise StructuralHypothesisReportAdoptionError(
                f"capsule publication artifact differs: {label}"
            )
        artifacts[f"publication:{label}"] = _artifact(
            capsule_relative, raw
        )
    marker_raw = _read_regular(
        root / "adoption.json", "adoption commit", exact_mode=0o600
    )
    marker = _translate(
        lambda: _publisher._parse_json(marker_raw, "adoption commit")
    )
    if marker_raw != _canonical_file(marker):
        raise StructuralHypothesisReportAdoptionError(
            "adoption commit is not canonical JSON"
        )
    integrity = marker.get("integrity")
    if (
        type(integrity) is not dict
        or set(integrity) != {"algorithm", "adoption_digest"}
        or integrity.get("algorithm") != "sha256-canonical-json-v1"
    ):
        raise StructuralHypothesisReportAdoptionError(
            "adoption integrity is malformed"
        )
    marker_body = {key: marker[key] for key in marker if key != "integrity"}
    if (
        _digest(marker_body) != integrity.get("adoption_digest")
        or integrity.get("adoption_digest") != expected_adoption_digest
    ):
        raise StructuralHypothesisReportAdoptionError(
            "adoption digest or independent anchor differs"
        )
    rebuilt = _marker_body(
        adoption_id=marker.get("adoption_id"),
        adoption_contract=contract,
        adoption_contract_raw=contract_raw,
        publication_marker=captured["values"]["publication_commit"],
        publication_raws=captured["raws"],
        output_report=captured["values"]["output_report"],
        combined_rows=captured["values"]["combined_rows"],
        artifacts=artifacts,
    )
    if (
        marker_body != rebuilt
        or marker.get("schema_version") != ADOPTION_CAPSULE_SCHEMA_VERSION
        or marker.get("status") != _COMMIT_STATUS
        or marker.get("planning_status") != "NOT_PLANNED"
        or marker.get("adoption_id") != root.name
        or marker.get("nonclaims") != _NONCLAIMS
    ):
        raise StructuralHypothesisReportAdoptionError(
            "adoption marker does not reproduce its full chain"
        )
    return _result(marker, root, verified=True)


__all__ = [
    "ADOPTION_CAPSULE_SCHEMA_VERSION",
    "ADOPTION_CONTRACT_ID",
    "ADOPTION_SCHEMA_VERSION",
    "StructuralHypothesisReportAdoptionError",
    "adopt_structural_hypothesis_report",
    "verify_structural_hypothesis_report_adoption",
]
