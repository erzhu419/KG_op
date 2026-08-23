"""Atomic local publication for one verified hypothesis reingestion.

The publisher consumes one completed single-task attempt and the exact base
CSV used by its source report.  It does not execute a task, adopt a report, or
materialize a subsequent plan.  ``publication.json`` is written last and is
the only commit marker.  All digests are local integrity commitments, not
signatures or external authority.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping, Sequence

from . import structural_hypothesis_single_task_runtime as _runtime_core
from .structural_hypothesis_execution import (
    ExecutionValidationError,
    reingest_successful_receipts,
    verify_reingestion_integrity,
)
from .structural_hypothesis_loop import (
    REQUIRED_EVIDENCE_FIELDS,
    canonical_json_bytes,
)
from .structural_hypothesis_single_task_runtime import (
    SingleTaskRuntimeValidationError,
    verify_single_task_attempt,
)


PUBLISHER_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-reingestion-publisher/1"
)
PUBLICATION_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-reingestion-publication/1"
)
PUBLISHER_CONTRACT_ID = "structural_hypothesis_reingestion_publisher_v1"

# Filled after the checked-in contract is frozen.  Keeping this commitment in
# code prevents a self-consistently rewritten local contract from silently
# changing V1 mechanics.
_PUBLISHER_CONTRACT_DIGEST = (
    "sha256:b94f745b0d07a1c21ecdcc3eb6ff20658d61fca490364ad20b1f86958489ff54"
)
_HYPOTHESIS_CONTRACT_DIGEST = (
    "sha256:4242f6af8424acca5c93136f0d4eb354f8c2203431f1c5145290c4a3f248cf26"
)
_EXECUTOR_CONTRACT_DIGEST = (
    "sha256:ede48b8b1fb0bb788f91a3834d5a41f336e55b331183922237176aec12624030"
)
_MATERIALIZER_CONTRACT_DIGEST = (
    "sha256:30c65d77e6cbdbc13b95e9083604f6f99835b0982d52319b99d0040491c1d013"
)
_RUNTIME_CONTRACT_DIGEST = (
    "sha256:d03529c64e6ea63b9997ded35fb6c0b44c6e17fb828f9e9db8960adb764a8c6b"
)

_STATE_PREFIX = Path("kg-op/structural-hypothesis-reingestion/v1")
_PUBLICATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")

_SOURCE_FILES = {
    "performance/structural_hypothesis_loop.py": (
        "f23a29c2f85b9bdce96398c6b901190fc34c2e2f97880dce5dbf2594b76c7635"
    ),
    "performance/structural_hypothesis_execution.py": (
        "7c5cc27f8e97da9b51e57975f63e860a23463d5e7728d1089f32978146f27c9b"
    ),
    "performance/structural_hypothesis_task_materializer.py": (
        "93345ef22df0cc9c5665be35722ad4646fb233d7a8bbd1294f0ca051cf64f20b"
    ),
    "performance/structural_hypothesis_single_task_runtime.py": (
        "618c336aee3558efb05a5415201cdf6b5cb1a7e028b90d94f2ac204b072e7fe4"
    ),
}

_ARTIFACT_LAYOUT = {
    "publisher_contract": "publisher_contract.json",
    "runtime_contract": "runtime_contract.json",
    "materializer_contract": "materializer_contract.json",
    "source_hypothesis_contract": "hypothesis_contract.json",
    "source_executor_contract": "executor_contract.json",
    "base_evidence_csv": "base_evidence.csv",
    "execution_directory": "execution",
    "execution_attempt": "execution/attempt.json",
    "execution_bundle": "execution/bundle.json",
    "execution_authorization": "execution/authorization.json",
    "execution_preflight": "execution/preflight.json",
    "execution_raw_result": "execution/raw_result.json",
    "execution_receipt": "execution/receipt.json",
    "execution_input_directory": "execution/inputs",
    "source_report": "execution/inputs/report.json",
    "execution_hypothesis_contract": (
        "execution/inputs/hypothesis_contract.json"
    ),
    "execution_executor_contract": "execution/inputs/executor_contract.json",
    "execution_materializer_contract": (
        "execution/inputs/materializer_contract.json"
    ),
    "execution_journal_directory": "execution/journal",
    "execution_authorized_event": "execution/journal/0000_AUTHORIZED.json",
    "execution_running_event": "execution/journal/0001_RUNNING.json",
    "execution_completed_event": "execution/journal/0002_COMPLETED.json",
    "combined_rows": "combined_rows.json",
    "output_report": "output_report.json",
    "reingestion_receipt": "reingestion_receipt.json",
    "publication_commit": "publication.json",
}

_ATTEMPT_FILES = {
    "execution_attempt": "attempt.json",
    "execution_bundle": "bundle.json",
    "execution_authorization": "authorization.json",
    "execution_preflight": "preflight.json",
    "execution_raw_result": "raw_result.json",
    "execution_receipt": "receipt.json",
    "source_report": "inputs/report.json",
    "execution_hypothesis_contract": "inputs/hypothesis_contract.json",
    "execution_executor_contract": "inputs/executor_contract.json",
    "execution_materializer_contract": "inputs/materializer_contract.json",
    "execution_authorized_event": "journal/0000_AUTHORIZED.json",
    "execution_running_event": "journal/0001_RUNNING.json",
    "execution_completed_event": "journal/0002_COMPLETED.json",
}

_NONCLAIMS = [
    "publication_is_not_adoption",
    "reingestion_is_not_external_verification",
    "local_digest_is_not_signature",
    "no_external_authority",
    "no_currentness_claim",
    "single_success_is_not_scientific_confirmation",
    "single_success_is_not_scientific_refutation",
    "output_report_may_retain_evidence_gaps",
    "source_csv_path_is_local_provenance_not_identity",
    "original_attempt_and_csv_are_required_for_full_verification",
    "no_same_user_rewrite_defense_without_independent_expected_digests",
    "no_network_access",
    "no_scheduler_access",
    "no_credential_access",
    "no_shell_execution",
    "no_task_execution",
    "no_automatic_replanning",
    "no_automatic_materialization",
    "no_automatic_adoption",
    "checkpoint_subtree_is_not_publication_evidence",
    "no_paper_promotion",
]


class ReingestionPublicationError(ValueError):
    """Raised when V1 publication or verification fails closed."""


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _raw_digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _require_digest(value: Any, label: str) -> str:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ReingestionPublicationError(
            f"{label} must be a lowercase sha256 digest"
        )
    return value


def _json_clone(value: Any, label: str = "value") -> Any:
    def check(item: Any, path: str) -> None:
        if item is None or type(item) in (str, int, bool):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ReingestionPublicationError(f"{path} must be finite")
            return
        if type(item) is list:
            for index, child in enumerate(item):
                check(child, f"{path}[{index}]")
            return
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ReingestionPublicationError(
                    f"{path} keys must be strings"
                )
            for key, child in item.items():
                check(child, f"{path}.{key}")
            return
        raise ReingestionPublicationError(f"{path} is not native JSON")

    check(value, label)
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReingestionPublicationError(
                f"JSON has duplicate key {key!r}"
            )
        result[key] = value
    return result


def _parse_json_value(raw: bytes, label: str) -> Any:
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReingestionPublicationError(f"{label} JSON is invalid") from error
    return _json_clone(value, label)


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    value = _parse_json_value(raw, label)
    if type(value) is not dict:
        raise ReingestionPublicationError(f"{label} JSON must be an object")
    return value


def _canonical_file(value: Any) -> bytes:
    return canonical_json_bytes(_json_clone(value)) + b"\n"


def _absolute_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ReingestionPublicationError(f"{label} must be an absolute path")
    normalized = Path(os.path.normpath(os.fspath(path)))
    if path != normalized:
        raise ReingestionPublicationError(
            f"{label} must be a canonical absolute path"
        )
    return path


def _reject_symlink_components(path: Path) -> None:
    cursor = path
    while True:
        if cursor.is_symlink():
            raise ReingestionPublicationError(
                f"path alias is forbidden: {cursor}"
            )
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent


def _read_regular(
    path: str | Path, label: str, *, exact_mode: int | None = None
) -> bytes:
    candidate = _absolute_path(path, label)
    _reject_symlink_components(candidate)
    try:
        observed_path = candidate.lstat()
    except OSError as error:
        raise ReingestionPublicationError(f"cannot inspect {label}") from error
    if stat.S_ISLNK(observed_path.st_mode) or not stat.S_ISREG(
        observed_path.st_mode
    ):
        raise ReingestionPublicationError(f"{label} is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise ReingestionPublicationError(f"cannot open {label}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino)
            != (observed_path.st_dev, observed_path.st_ino)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o022
            or exact_mode is not None
            and stat.S_IMODE(before.st_mode) != exact_mode
        ):
            raise ReingestionPublicationError(
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
            raise ReingestionPublicationError(f"{label} changed while read")
        raw = b"".join(chunks)
        if len(raw) != before.st_size:
            raise ReingestionPublicationError(f"{label} byte count changed")
        return raw
    finally:
        os.close(descriptor)


def _read_json(path: str | Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, label)
    return _parse_json(raw, label), raw


def _load_csv(raw: bytes) -> list[dict[str, str]]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ReingestionPublicationError(
            "base evidence CSV is not UTF-8"
        ) from error
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    names = reader.fieldnames
    if not names or any(type(name) is not str or not name for name in names):
        raise ReingestionPublicationError(
            "base evidence CSV has invalid column names"
        )
    if len(names) != len(set(names)):
        raise ReingestionPublicationError(
            "base evidence CSV has duplicate column names"
        )
    missing = sorted(REQUIRED_EVIDENCE_FIELDS.difference(names))
    if missing:
        raise ReingestionPublicationError(
            f"base evidence CSV lacks required columns: {missing}"
        )
    rows = []
    try:
        for row in reader:
            if None in row:
                raise ReingestionPublicationError(
                    "base evidence CSV row exceeds its header"
                )
            rows.append(dict(row))
    except csv.Error as error:
        raise ReingestionPublicationError(
            "base evidence CSV is malformed"
        ) from error
    return rows


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _validate_source_files(contract: Mapping[str, Any]) -> None:
    if contract.get("source_files") != _SOURCE_FILES:
        raise ReingestionPublicationError(
            "publisher source-file commitments differ"
        )
    root = _repo_root()
    for relative, expected in _SOURCE_FILES.items():
        path = root / relative
        if hashlib.sha256(
            _read_regular(path, f"publisher source file {relative}")
        ).hexdigest() != expected:
            raise ReingestionPublicationError(
                f"publisher source file differs: {relative}"
            )


def _validate_publisher_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema_version") != PUBLISHER_SCHEMA_VERSION
        or contract.get("contract_id") != PUBLISHER_CONTRACT_ID
        or contract.get("artifact_layout") != _ARTIFACT_LAYOUT
        or contract.get("nonclaims") != _NONCLAIMS
    ):
        raise ReingestionPublicationError(
            "publisher contract differs from frozen V1"
        )
    if _digest(contract) != _PUBLISHER_CONTRACT_DIGEST:
        raise ReingestionPublicationError(
            "publisher contract digest differs from frozen V1"
        )
    source = contract.get("source_contracts")
    if type(source) is not dict or (
        source.get("hypothesis_contract_digest")
        != _HYPOTHESIS_CONTRACT_DIGEST
        or source.get("executor_contract_digest")
        != _EXECUTOR_CONTRACT_DIGEST
        or source.get("materializer_contract_digest")
        != _MATERIALIZER_CONTRACT_DIGEST
        or source.get("runtime_contract_digest")
        != _RUNTIME_CONTRACT_DIGEST
    ):
        raise ReingestionPublicationError(
            "publisher source-contract commitments differ"
        )
    admission = contract.get("admission")
    if type(admission) is not dict or admission != {
        "execution_receipt_count": 1,
        "authorized_task_count": 1,
        "successful_result_count": 1,
        "failed_result_count": 0,
        "accepted_successful_rows": 1,
        "ignored_failed_attempts": 0,
        "pending_evidence_delta": 1,
        "source_report_status": "COMPLETED_WITH_EVIDENCE_GAPS",
        "output_statuses": ["COMPLETED_WITH_EVIDENCE_GAPS", "COMPLETED"],
    }:
        raise ReingestionPublicationError("publisher admission differs")
    _validate_source_files(contract)


def _state_base() -> Path:
    configured = os.environ.get("XDG_STATE_HOME")
    if configured:
        base = Path(configured)
        if not base.is_absolute():
            raise ReingestionPublicationError(
                "XDG_STATE_HOME must be absolute"
            )
        return base
    return Path.home() / ".local/state"


def _secure_directory(path: Path, *, exact_mode: int | None = None) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise ReingestionPublicationError(
            f"missing publication directory: {path}"
        ) from error
    mode = stat.S_IMODE(info.st_mode)
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or mode & 0o022
        or exact_mode is not None and mode != exact_mode
    ):
        raise ReingestionPublicationError(
            f"publication directory ownership or mode differs: {path}"
        )


def _ensure_secure_parent(target: Path) -> None:
    base = _state_base()
    prefix = base / _STATE_PREFIX
    if target.parent != prefix:
        raise ReingestionPublicationError(
            "publication_root is outside the frozen local state prefix"
        )
    _reject_symlink_components(target)
    missing = []
    cursor = prefix
    while not cursor.exists():
        if cursor.is_symlink():
            raise ReingestionPublicationError(
                f"publication directory alias is forbidden: {cursor}"
            )
        missing.append(cursor)
        if cursor == base:
            break
        cursor = cursor.parent
    if not cursor.exists():
        raise ReingestionPublicationError("state-home ancestor is missing")
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


def _validate_publication_location(
    publication_root: str | Path,
    publication_id: str | None,
    *,
    fresh: bool,
) -> Path:
    root = _absolute_path(publication_root, "publication_root")
    if publication_id is not None:
        if type(publication_id) is not str or not _PUBLICATION_ID.fullmatch(
            publication_id
        ):
            raise ReingestionPublicationError("publication_id is invalid")
        if root.name != publication_id:
            raise ReingestionPublicationError(
                "publication_id must equal publication_root basename"
            )
    expected_parent = _state_base() / _STATE_PREFIX
    if root.parent != expected_parent:
        raise ReingestionPublicationError(
            "publication_root is outside the frozen local state prefix"
        )
    if fresh:
        if root.exists() or root.is_symlink():
            raise ReingestionPublicationError("publication_root already exists")
        _ensure_secure_parent(root)
    else:
        _reject_symlink_components(root)
        _secure_directory(root, exact_mode=0o700)
    return root


def _mkdir_new(path: Path) -> None:
    try:
        os.mkdir(path, 0o700)
        os.chmod(path, 0o700, follow_symlinks=False)
    except OSError as error:
        raise ReingestionPublicationError(
            f"cannot create fresh publication directory: {path}"
        ) from error
    _secure_directory(path, exact_mode=0o700)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_new_bytes(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ReingestionPublicationError(
            f"publication artifact already exists: {path.name}"
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


def _publication_artifact(path: str, raw: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": _raw_digest(raw), "bytes": len(raw)}


def _attempt_snapshots(attempt_root: Path) -> tuple[
    dict[str, dict[str, Any]], dict[str, bytes]
]:
    objects = {}
    raws = {}
    for label, relative in _ATTEMPT_FILES.items():
        value, raw = _read_json(attempt_root / relative, label)
        objects[label] = value
        raws[label] = raw
    return objects, raws


def _validate_chain_objects(
    objects: Mapping[str, Mapping[str, Any]],
    publisher_contract: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    base_rows: Sequence[Mapping[str, Any]],
    base_raw: bytes,
    base_path: Path,
    hypothesis_raw: bytes,
    hypothesis_path: Path,
    *,
    expected_source_evidence_digest: str,
    expected_plan_digest: str,
    expected_authorization_digest: str,
    expected_execution_receipt_digest: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    report = objects["source_report"]
    bundle = objects["execution_bundle"]
    authorization = objects["execution_authorization"]
    receipt = objects["execution_receipt"]
    hypothesis_contract = objects["execution_hypothesis_contract"]
    executor_contract = objects["execution_executor_contract"]
    materializer_contract = objects["execution_materializer_contract"]
    attempt = objects["execution_attempt"]

    if _digest(hypothesis_contract) != _HYPOTHESIS_CONTRACT_DIGEST:
        raise ReingestionPublicationError("hypothesis contract differs")
    if _digest(executor_contract) != _EXECUTOR_CONTRACT_DIGEST:
        raise ReingestionPublicationError("executor contract differs")
    if _digest(materializer_contract) != _MATERIALIZER_CONTRACT_DIGEST:
        raise ReingestionPublicationError("materializer contract differs")
    if _digest(runtime_contract) != _RUNTIME_CONTRACT_DIGEST:
        raise ReingestionPublicationError("runtime contract differs")
    if report.get("status") != publisher_contract["admission"][
        "source_report_status"
    ]:
        raise ReingestionPublicationError("source report status differs")
    if report.get("evidence_digest") != expected_source_evidence_digest:
        raise ReingestionPublicationError("source evidence digest differs")
    input_artifacts = report.get("input_artifacts")
    if type(input_artifacts) is not dict:
        raise ReingestionPublicationError(
            "source report raw input artifacts differ"
        )
    expected_evidence_artifact = {
        "path": str(base_path.resolve(strict=True)),
        "sha256": hashlib.sha256(base_raw).hexdigest(),
        "bytes": len(base_raw),
    }
    if input_artifacts.get("evidence_csv") != expected_evidence_artifact:
        raise ReingestionPublicationError(
            "raw evidence CSV digest differs"
        )
    expected_contract_artifact = {
        "path": str(hypothesis_path.resolve(strict=True)),
        "sha256": hashlib.sha256(hypothesis_raw).hexdigest(),
        "bytes": len(hypothesis_raw),
    }
    if (
        input_artifacts.get("contract_json") != expected_contract_artifact
        or set(input_artifacts) != {"evidence_csv", "contract_json"}
    ):
        raise ReingestionPublicationError(
            "raw hypothesis contract digest differs"
        )
    if _digest(base_rows) != expected_source_evidence_digest:
        raise ReingestionPublicationError(
            "base evidence rows do not reproduce source evidence digest"
        )
    plan = bundle.get("plan")
    if type(plan) is not dict or plan.get("integrity", {}).get(
        "plan_digest"
    ) != expected_plan_digest:
        raise ReingestionPublicationError("execution plan digest differs")
    if authorization.get("integrity", {}).get(
        "authorization_digest"
    ) != expected_authorization_digest:
        raise ReingestionPublicationError("authorization digest differs")
    if receipt.get("integrity", {}).get(
        "receipt_digest"
    ) != expected_execution_receipt_digest:
        raise ReingestionPublicationError("execution receipt digest differs")
    if attempt.get("plan_binding", {}).get(
        "plan_digest"
    ) != expected_plan_digest:
        raise ReingestionPublicationError("attempt plan binding differs")
    if attempt.get("authorization_binding", {}).get(
        "authorization_digest"
    ) != expected_authorization_digest:
        raise ReingestionPublicationError(
            "attempt authorization binding differs"
        )
    if receipt.get("status") != "COMPLETED" or receipt.get("summary") != {
        "authorized": 1,
        "succeeded": 1,
        "failed": 0,
    }:
        raise ReingestionPublicationError(
            "V1 requires one completed successful receipt"
        )
    results = receipt.get("results")
    if (
        type(results) is not list
        or len(results) != 1
        or type(results[0]) is not dict
        or results[0].get("status") != "SUCCEEDED"
        or type(results[0].get("evidence_row")) is not dict
    ):
        raise ReingestionPublicationError(
            "V1 requires one successful evidence row"
        )
    try:
        generated = reingest_successful_receipts(
            base_rows,
            [receipt],
            hypothesis_contract,
            report,
            plan=plan,
            authorization=authorization,
            executor_contract=executor_contract,
        )
    except ExecutionValidationError as error:
        raise ReingestionPublicationError(
            "reingestion full-chain validation failed"
        ) from error
    output = generated["report"]
    reingestion = generated["reingestion_receipt"]
    if (
        reingestion.get("accepted_successful_rows") != 1
        or reingestion.get("ignored_failed_attempts") != 0
        or output.get("status")
        not in publisher_contract["admission"]["output_statuses"]
        or len(report.get("pending_evidence", []))
        - len(output.get("pending_evidence", []))
        != 1
    ):
        raise ReingestionPublicationError(
            "reingestion transition differs from V1 admission"
        )
    combined = [_json_clone(dict(row), "base row") for row in base_rows]
    combined.append(
        _json_clone(results[0]["evidence_row"], "successful evidence row")
    )
    if _digest(combined) != output.get("evidence_digest"):
        raise ReingestionPublicationError(
            "combined rows do not reproduce output evidence digest"
        )
    if not verify_reingestion_integrity(
        reingestion,
        source_report=report,
        base_rows=base_rows,
        plan=plan,
        authorization=authorization,
        receipts=[receipt],
        output_report=output,
        hypothesis_contract=hypothesis_contract,
        executor_contract=executor_contract,
    ):
        raise ReingestionPublicationError(
            "generated reingestion receipt failed full-chain verification"
        )
    return output, reingestion, combined


def _verify_runtime_attempt(
    attempt_root: Path,
    runtime_contract: Mapping[str, Any],
    base_manifest: Path,
    asset_root: Path,
    *,
    expected_authorization_digest: str,
    expected_execution_receipt_digest: str,
    expected_execution_journal_head_digest: str,
    expected_execution_attempt_digest: str,
) -> dict[str, Any]:
    try:
        verified = verify_single_task_attempt(
            attempt_root,
            runtime_contract,
            base_manifest,
            asset_root,
            expected_authorization_digest=expected_authorization_digest,
            expected_receipt_digest=expected_execution_receipt_digest,
            expected_journal_head_digest=(
                expected_execution_journal_head_digest
            ),
            expected_attempt_digest=expected_execution_attempt_digest,
        )
    except (SingleTaskRuntimeValidationError, ValueError) as error:
        raise ReingestionPublicationError(
            "completed execution attempt failed strong verification"
        ) from error
    if verified.get("status") != "VERIFIED_COMPLETED":
        raise ReingestionPublicationError(
            "execution attempt is not verified completed"
        )
    return verified


def _validate_captured_runtime_capsule(
    objects: Mapping[str, Mapping[str, Any]],
    runtime_contract: Mapping[str, Any],
    base_manifest: Path,
    asset_root: Path,
    attempt_root: Path,
    *,
    expected_plan_digest: str,
    expected_authorization_digest: str,
    expected_execution_receipt_digest: str,
    expected_execution_journal_head_digest: str,
    expected_execution_attempt_digest: str,
) -> None:
    """Verify the captured bytes, not a later generation at their paths."""
    attempt = objects["execution_attempt"]
    bundle = objects["execution_bundle"]
    authorization = objects["execution_authorization"]
    report = objects["source_report"]
    hypothesis = objects["execution_hypothesis_contract"]
    executor_contract = objects["execution_executor_contract"]
    materializer = objects["execution_materializer_contract"]
    receipt = objects["execution_receipt"]
    try:
        task_id = attempt.get("task_binding", {}).get("task_id")
        task = _runtime_core._selected_task(bundle, task_id)
        _runtime_core._validate_bound_contracts(
            hypothesis,
            executor_contract,
            materializer,
            runtime_contract,
        )
        bound_base = _runtime_core._path(
            base_manifest, "captured base manifest"
        )
        bound_assets = _runtime_core._path(
            asset_root, "captured asset root"
        )
        _state_base, state_root = _runtime_core._state_base_and_prefix(
            runtime_contract
        )
        checkpoint_root = attempt_root / _runtime_core._LAYOUT[
            "checkpoint_root"
        ]
        _runtime_core._validate_bundle(
            report,
            bundle,
            hypothesis,
            executor_contract,
            materializer,
            bound_base,
            bound_assets,
            checkpoint_root,
        )
        _runtime_core._validate_task_runtime(task, checkpoint_root)
        expected_authorization = _runtime_core._CAPTURED_AUTHORIZE_PLAN(
            bundle["plan"],
            hypothesis,
            report,
            executor_contract,
            expected_plan_digest=expected_plan_digest,
            authorization_id=authorization.get("authorization_id"),
            authorized_task_ids=[task_id],
        )
        if authorization != expected_authorization:
            raise ReingestionPublicationError(
                "captured authorization differs"
            )
        expected_attempt = _runtime_core._attempt_binding(
            bundle=bundle,
            task=task,
            authorization=authorization,
            base_manifest_path=bound_base,
            asset_root=bound_assets,
            checkpoint_root=checkpoint_root,
            state_root=state_root,
        )
        if attempt != expected_attempt:
            raise ReingestionPublicationError(
                "captured attempt binding differs"
            )
        attempt_digest = attempt["integrity"]["attempt_digest"]
        authorization_digest = authorization["integrity"][
            "authorization_digest"
        ]
        preflight_digest = _runtime_core._validate_preflight_artifact(
            objects["execution_preflight"],
            task=task,
            runtime_contract=runtime_contract,
            attempt_digest=attempt_digest,
            authorization_digest=authorization_digest,
        )
        native = _runtime_core._validate_raw_envelope(
            objects["execution_raw_result"],
            task,
            authorization_digest,
        )
        replayed = _runtime_core._CAPTURED_EXECUTE_AUTHORIZED_PLAN(
            bundle["plan"],
            authorization,
            expected_authorization_digest=authorization_digest,
            executor=lambda _task: _json_clone(native),
            hypothesis_contract=hypothesis,
            source_report=report,
            executor_contract=executor_contract,
        )
        if replayed != receipt:
            raise ReingestionPublicationError(
                "captured raw result does not reproduce receipt"
            )
        receipt_digest = receipt["integrity"]["receipt_digest"]
        raw_digest = objects["execution_raw_result"]["integrity"][
            "raw_result_digest"
        ]
        authorized = _runtime_core._event(
            sequence=0,
            state="AUTHORIZED",
            previous_event_digest=None,
            attempt_digest=attempt_digest,
            authorization_digest=authorization_digest,
            task_id=task["task_id"],
            task_digest=task["task_digest"],
        )
        running = _runtime_core._event(
            sequence=1,
            state="RUNNING",
            previous_event_digest=authorized["integrity"]["event_digest"],
            attempt_digest=attempt_digest,
            authorization_digest=authorization_digest,
            task_id=task["task_id"],
            task_digest=task["task_digest"],
            preflight_digest=preflight_digest,
        )
        completed = _runtime_core._event(
            sequence=2,
            state="COMPLETED",
            previous_event_digest=running["integrity"]["event_digest"],
            attempt_digest=attempt_digest,
            authorization_digest=authorization_digest,
            task_id=task["task_id"],
            task_digest=task["task_digest"],
            preflight_digest=preflight_digest,
            raw_result_digest=raw_digest,
            receipt_digest=receipt_digest,
            runtime_error_code=None,
        )
        if (
            objects["execution_authorized_event"] != authorized
            or objects["execution_running_event"] != running
            or objects["execution_completed_event"] != completed
            or bundle["plan"]["integrity"]["plan_digest"]
            != expected_plan_digest
            or authorization_digest != expected_authorization_digest
            or receipt_digest != expected_execution_receipt_digest
            or completed["integrity"]["event_digest"]
            != expected_execution_journal_head_digest
            or attempt_digest != expected_execution_attempt_digest
        ):
            raise ReingestionPublicationError(
                "captured execution capsule or independent anchor differs"
            )
    except (
        ExecutionValidationError,
        SingleTaskRuntimeValidationError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        if isinstance(error, ReingestionPublicationError):
            raise
        raise ReingestionPublicationError(
            "captured execution capsule failed strong verification"
        ) from error


def _snapshot_inputs(
    base_evidence_csv: Path,
    attempt_root: Path,
    hypothesis_contract_path: Path,
    executor_contract_path: Path,
    publisher_contract_path: Path,
    runtime_contract_path: Path,
) -> dict[str, Any]:
    hypothesis_contract, hypothesis_raw = _read_json(
        hypothesis_contract_path, "hypothesis contract"
    )
    executor_contract, executor_raw = _read_json(
        executor_contract_path, "executor contract"
    )
    publisher_contract, publisher_raw = _read_json(
        publisher_contract_path, "publisher contract"
    )
    runtime_contract, runtime_raw = _read_json(
        runtime_contract_path, "runtime contract"
    )
    _validate_publisher_contract(publisher_contract)
    if _digest(runtime_contract) != _RUNTIME_CONTRACT_DIGEST:
        raise ReingestionPublicationError("runtime contract differs")
    base_raw = _read_regular(base_evidence_csv, "base evidence CSV")
    base_rows = _load_csv(base_raw)
    objects, attempt_raws = _attempt_snapshots(attempt_root)
    if (
        _digest(hypothesis_contract) != _HYPOTHESIS_CONTRACT_DIGEST
        or hypothesis_contract != objects["execution_hypothesis_contract"]
    ):
        raise ReingestionPublicationError(
            "external hypothesis contract differs from attempt snapshot"
        )
    if (
        _digest(executor_contract) != _EXECUTOR_CONTRACT_DIGEST
        or executor_contract != objects["execution_executor_contract"]
    ):
        raise ReingestionPublicationError(
            "external executor contract differs from attempt snapshot"
        )
    return {
        "hypothesis_contract": hypothesis_contract,
        "hypothesis_raw": hypothesis_raw,
        "executor_contract": executor_contract,
        "executor_raw": executor_raw,
        "publisher_contract": publisher_contract,
        "publisher_raw": publisher_raw,
        "runtime_contract": runtime_contract,
        "runtime_raw": runtime_raw,
        "base_raw": base_raw,
        "base_rows": base_rows,
        "objects": objects,
        "attempt_raws": attempt_raws,
    }


def _same_input_snapshot(
    captured: Mapping[str, Any],
    base_evidence_csv: Path,
    attempt_root: Path,
    hypothesis_contract_path: Path,
    executor_contract_path: Path,
    publisher_contract_path: Path,
    runtime_contract_path: Path,
) -> None:
    again = _snapshot_inputs(
        base_evidence_csv,
        attempt_root,
        hypothesis_contract_path,
        executor_contract_path,
        publisher_contract_path,
        runtime_contract_path,
    )
    for key in (
        "hypothesis_raw",
        "executor_raw",
        "publisher_raw",
        "runtime_raw",
        "base_raw",
    ):
        if again[key] != captured[key]:
            raise ReingestionPublicationError(
                "publisher input changed before commit"
            )
    if again["attempt_raws"] != captured["attempt_raws"]:
        raise ReingestionPublicationError(
            "completed attempt changed before commit"
        )


def _root_expected_names(*, committed: bool) -> set[str]:
    names = {
        "publisher_contract.json",
        "runtime_contract.json",
        "materializer_contract.json",
        "hypothesis_contract.json",
        "executor_contract.json",
        "base_evidence.csv",
        "execution",
        "combined_rows.json",
        "output_report.json",
        "reingestion_receipt.json",
    }
    if committed:
        names.add("publication.json")
    return names


def _directory_names(path: Path) -> set[str]:
    _secure_directory(path, exact_mode=0o700)
    try:
        return {child.name for child in path.iterdir()}
    except OSError as error:
        raise ReingestionPublicationError(
            f"cannot enumerate publication directory: {path}"
        ) from error


def _validate_directory_layout(root: Path, *, committed: bool = True) -> None:
    if _directory_names(root) != _root_expected_names(committed=committed):
        raise ReingestionPublicationError(
            "publication root has missing or unexpected artifacts"
        )
    execution = root / "execution"
    if _directory_names(execution) != {
        "attempt.json",
        "bundle.json",
        "authorization.json",
        "preflight.json",
        "raw_result.json",
        "receipt.json",
        "inputs",
        "journal",
    }:
        raise ReingestionPublicationError(
            "publication execution snapshot layout differs"
        )
    if _directory_names(execution / "inputs") != {
        "report.json",
        "hypothesis_contract.json",
        "executor_contract.json",
        "materializer_contract.json",
    }:
        raise ReingestionPublicationError(
            "publication input snapshot layout differs"
        )
    if _directory_names(execution / "journal") != {
        "0000_AUTHORIZED.json",
        "0001_RUNNING.json",
        "0002_COMPLETED.json",
    }:
        raise ReingestionPublicationError(
            "publication journal snapshot layout differs"
        )


def _load_publication_artifacts(root: Path) -> tuple[
    dict[str, dict[str, Any]], dict[str, bytes]
]:
    values = {}
    raws = {}
    for label, relative in _ARTIFACT_LAYOUT.items():
        if label in {
            "execution_directory",
            "execution_input_directory",
            "execution_journal_directory",
        }:
            continue
        path = root / relative
        if label == "base_evidence_csv":
            raw = _read_regular(path, label, exact_mode=0o600)
            raws[label] = raw
            continue
        raw = _read_regular(path, label, exact_mode=0o600)
        value = (
            _parse_json_value(raw, label)
            if label == "combined_rows"
            else _parse_json(raw, label)
        )
        if label == "combined_rows" and type(value) is not list:
            raise ReingestionPublicationError(
                "combined_rows JSON must be an array"
            )
        values[label] = value
        raws[label] = raw
    return values, raws


def _marker_body(
    *,
    publication_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    publisher_contract: Mapping[str, Any],
    publisher_raw: bytes,
    hypothesis_raw: bytes,
    executor_raw: bytes,
    base_raw: bytes,
    base_rows: Sequence[Mapping[str, Any]],
    objects: Mapping[str, Mapping[str, Any]],
    output: Mapping[str, Any],
    reingestion: Mapping[str, Any],
    combined: Sequence[Mapping[str, Any]],
    expected_plan_digest: str,
    expected_authorization_digest: str,
    expected_execution_receipt_digest: str,
    expected_execution_journal_head_digest: str,
    expected_execution_attempt_digest: str,
) -> dict[str, Any]:
    report = objects["source_report"]
    successful = objects["execution_receipt"]["results"][0][
        "evidence_row"
    ]
    return {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "status": "PUBLISHED_NOT_ADOPTED",
        "publication_id": publication_id,
        "publisher_contract_binding": {
            "contract_id": publisher_contract["contract_id"],
            "contract_digest": _digest(publisher_contract),
            "raw_sha256": _raw_digest(publisher_raw),
            "bytes": len(publisher_raw),
        },
        "source_binding": {
            "external_contracts": {
                "hypothesis_contract": {
                    "canonical_digest": _HYPOTHESIS_CONTRACT_DIGEST,
                    "raw_sha256": _raw_digest(hypothesis_raw),
                    "bytes": len(hypothesis_raw),
                },
                "executor_contract": {
                    "canonical_digest": _EXECUTOR_CONTRACT_DIGEST,
                    "raw_sha256": _raw_digest(executor_raw),
                    "bytes": len(executor_raw),
                },
            },
            "base_evidence": {
                "raw_sha256": _raw_digest(base_raw),
                "bytes": len(base_raw),
                "row_count": len(base_rows),
                "evidence_digest": report["evidence_digest"],
            },
            "source_report": {
                "status": report["status"],
                "evidence_digest": report["evidence_digest"],
                "report_body_digest": report["audit"]["report_body_digest"],
                "audit_head": report["audit"]["head"],
                "pending_evidence_count": len(report["pending_evidence"]),
            },
            "plan_digest": expected_plan_digest,
            "authorization_digest": expected_authorization_digest,
            "execution_receipt_digest": expected_execution_receipt_digest,
            "execution_journal_head_digest": (
                expected_execution_journal_head_digest
            ),
            "execution_attempt_digest": expected_execution_attempt_digest,
        },
        "transition": {
            "accepted_successful_rows": 1,
            "ignored_failed_attempts": 0,
            "successful_cell": {
                "profile": successful["method"],
                "domain": successful["domain"],
                "seed": successful["seed"],
            },
            "source_pending_evidence_count": len(report["pending_evidence"]),
            "output_pending_evidence_count": len(output["pending_evidence"]),
        },
        "outputs": {
            "combined_rows_count": len(combined),
            "combined_evidence_digest": output["evidence_digest"],
            "output_report_status": output["status"],
            "output_report_body_digest": output["audit"][
                "report_body_digest"
            ],
            "output_audit_head": output["audit"]["head"],
            "output_evidence_digest": output["evidence_digest"],
            "reingestion_digest": reingestion["integrity"][
                "reingestion_digest"
            ],
        },
        "artifacts": dict(artifacts),
        "nonclaims": list(_NONCLAIMS),
    }


def _publication_result(
    marker: Mapping[str, Any], root: Path, *, verified: bool = False
) -> dict[str, Any]:
    outputs = marker["outputs"]
    transition = marker["transition"]
    return {
        "status": (
            "VERIFIED_PUBLISHED_NOT_ADOPTED"
            if verified
            else "PUBLISHED_NOT_ADOPTED"
        ),
        "publication_root": str(root),
        "publication_digest": marker["integrity"]["publication_digest"],
        "reingestion_digest": outputs["reingestion_digest"],
        "output_report_body_digest": outputs["output_report_body_digest"],
        "output_audit_head": outputs["output_audit_head"],
        "output_evidence_digest": outputs["output_evidence_digest"],
        "accepted_successful_rows": transition["accepted_successful_rows"],
        "ignored_failed_attempts": transition["ignored_failed_attempts"],
    }


def publish_single_task_reingestion(
    base_evidence_csv: str | Path,
    attempt_root: str | Path,
    hypothesis_contract_path: str | Path,
    executor_contract_path: str | Path,
    runtime_contract_path: str | Path,
    publisher_contract_path: str | Path,
    base_manifest_path: str | Path,
    asset_root: str | Path,
    publication_root: str | Path,
    *,
    publication_id: str,
    expected_source_evidence_digest: str,
    expected_plan_digest: str,
    expected_authorization_digest: str,
    expected_execution_receipt_digest: str,
    expected_execution_journal_head_digest: str,
    expected_execution_attempt_digest: str,
) -> dict[str, Any]:
    """Verify, reingest, and commit exactly one successful local receipt."""
    expected_source_evidence_digest = _require_digest(
        expected_source_evidence_digest, "expected_source_evidence_digest"
    )
    expected_plan_digest = _require_digest(
        expected_plan_digest, "expected_plan_digest"
    )
    expected_authorization_digest = _require_digest(
        expected_authorization_digest, "expected_authorization_digest"
    )
    expected_execution_receipt_digest = _require_digest(
        expected_execution_receipt_digest,
        "expected_execution_receipt_digest",
    )
    expected_execution_journal_head_digest = _require_digest(
        expected_execution_journal_head_digest,
        "expected_execution_journal_head_digest",
    )
    expected_execution_attempt_digest = _require_digest(
        expected_execution_attempt_digest, "expected_execution_attempt_digest"
    )
    base_path = _absolute_path(base_evidence_csv, "base_evidence_csv")
    attempt_path = _absolute_path(attempt_root, "attempt_root")
    publisher_path = _absolute_path(
        publisher_contract_path, "publisher_contract_path"
    )
    hypothesis_path = _absolute_path(
        hypothesis_contract_path, "hypothesis_contract_path"
    )
    executor_path = _absolute_path(
        executor_contract_path, "executor_contract_path"
    )
    runtime_path = _absolute_path(runtime_contract_path, "runtime_contract_path")
    manifest_path = _absolute_path(base_manifest_path, "base_manifest_path")
    assets_path = _absolute_path(asset_root, "asset_root")
    root = _validate_publication_location(
        publication_root, publication_id, fresh=True
    )

    captured = _snapshot_inputs(
        base_path,
        attempt_path,
        hypothesis_path,
        executor_path,
        publisher_path,
        runtime_path,
    )
    _validate_captured_runtime_capsule(
        captured["objects"],
        captured["runtime_contract"],
        manifest_path,
        assets_path,
        attempt_path,
        expected_plan_digest=expected_plan_digest,
        expected_authorization_digest=expected_authorization_digest,
        expected_execution_receipt_digest=expected_execution_receipt_digest,
        expected_execution_journal_head_digest=(
            expected_execution_journal_head_digest
        ),
        expected_execution_attempt_digest=expected_execution_attempt_digest,
    )
    _verify_runtime_attempt(
        attempt_path,
        captured["runtime_contract"],
        manifest_path,
        assets_path,
        expected_authorization_digest=expected_authorization_digest,
        expected_execution_receipt_digest=expected_execution_receipt_digest,
        expected_execution_journal_head_digest=(
            expected_execution_journal_head_digest
        ),
        expected_execution_attempt_digest=expected_execution_attempt_digest,
    )
    output, reingestion, combined = _validate_chain_objects(
        captured["objects"],
        captured["publisher_contract"],
        captured["runtime_contract"],
        captured["base_rows"],
        captured["base_raw"],
        base_path,
        captured["hypothesis_raw"],
        hypothesis_path,
        expected_source_evidence_digest=expected_source_evidence_digest,
        expected_plan_digest=expected_plan_digest,
        expected_authorization_digest=expected_authorization_digest,
        expected_execution_receipt_digest=expected_execution_receipt_digest,
    )

    # Re-read every mutable source and repeat the runtime gate before creating
    # the publication directory.  The bytes captured above are the bytes that
    # are published; a later same-user rewrite is outside local-digest trust.
    _same_input_snapshot(
        captured,
        base_path,
        attempt_path,
        hypothesis_path,
        executor_path,
        publisher_path,
        runtime_path,
    )
    _verify_runtime_attempt(
        attempt_path,
        captured["runtime_contract"],
        manifest_path,
        assets_path,
        expected_authorization_digest=expected_authorization_digest,
        expected_execution_receipt_digest=expected_execution_receipt_digest,
        expected_execution_journal_head_digest=(
            expected_execution_journal_head_digest
        ),
        expected_execution_attempt_digest=expected_execution_attempt_digest,
    )
    _same_input_snapshot(
        captured,
        base_path,
        attempt_path,
        hypothesis_path,
        executor_path,
        publisher_path,
        runtime_path,
    )

    _mkdir_new(root)
    _mkdir_new(root / "execution")
    _mkdir_new(root / "execution/inputs")
    _mkdir_new(root / "execution/journal")

    derived = {
        "combined_rows": _canonical_file(combined),
        "output_report": _canonical_file(output),
        "reingestion_receipt": _canonical_file(reingestion),
    }
    raw_files = {
        "publisher_contract": captured["publisher_raw"],
        "runtime_contract": captured["runtime_raw"],
        "materializer_contract": captured["attempt_raws"][
            "execution_materializer_contract"
        ],
        "source_hypothesis_contract": captured["hypothesis_raw"],
        "source_executor_contract": captured["executor_raw"],
        "base_evidence_csv": captured["base_raw"],
        **captured["attempt_raws"],
        **derived,
    }
    artifacts = {}
    for label, raw in raw_files.items():
        relative = _ARTIFACT_LAYOUT[label]
        _write_new_bytes(root / relative, raw)
        artifacts[label] = _publication_artifact(relative, raw)

    # A publication is not committed yet.  Re-open every leaf through the
    # secure reader and require the exact pre-commit layout before allowing
    # the final marker to appear.
    _validate_directory_layout(root, committed=False)
    for label, expected_raw in raw_files.items():
        observed_raw = _read_regular(
            root / _ARTIFACT_LAYOUT[label],
            f"pre-commit publication artifact {label}",
            exact_mode=0o600,
        )
        if observed_raw != expected_raw:
            raise ReingestionPublicationError(
                f"pre-commit publication artifact differs: {label}"
            )
    _verify_runtime_attempt(
        attempt_path,
        captured["runtime_contract"],
        manifest_path,
        assets_path,
        expected_authorization_digest=expected_authorization_digest,
        expected_execution_receipt_digest=expected_execution_receipt_digest,
        expected_execution_journal_head_digest=(
            expected_execution_journal_head_digest
        ),
        expected_execution_attempt_digest=expected_execution_attempt_digest,
    )
    _same_input_snapshot(
        captured,
        base_path,
        attempt_path,
        hypothesis_path,
        executor_path,
        publisher_path,
        runtime_path,
    )

    marker_body = _marker_body(
        publication_id=publication_id,
        artifacts=artifacts,
        publisher_contract=captured["publisher_contract"],
        publisher_raw=captured["publisher_raw"],
        hypothesis_raw=captured["hypothesis_raw"],
        executor_raw=captured["executor_raw"],
        base_raw=captured["base_raw"],
        base_rows=captured["base_rows"],
        objects=captured["objects"],
        output=output,
        reingestion=reingestion,
        combined=combined,
        expected_plan_digest=expected_plan_digest,
        expected_authorization_digest=expected_authorization_digest,
        expected_execution_receipt_digest=expected_execution_receipt_digest,
        expected_execution_journal_head_digest=(
            expected_execution_journal_head_digest
        ),
        expected_execution_attempt_digest=expected_execution_attempt_digest,
    )
    marker = {
        **marker_body,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "publication_digest": _digest(marker_body),
        },
    }
    marker_raw = _canonical_file(marker)
    _write_new_bytes(
        root / _ARTIFACT_LAYOUT["publication_commit"],
        marker_raw,
    )
    _validate_directory_layout(root, committed=True)
    for label, expected_raw in {
        **raw_files,
        "publication_commit": marker_raw,
    }.items():
        observed_raw = _read_regular(
            root / _ARTIFACT_LAYOUT[label],
            f"committed publication artifact {label}",
            exact_mode=0o600,
        )
        if observed_raw != expected_raw:
            raise ReingestionPublicationError(
                f"committed publication artifact differs: {label}"
            )
    return _publication_result(marker, root)


def verify_single_task_reingestion_publication(
    base_evidence_csv: str | Path,
    attempt_root: str | Path,
    hypothesis_contract_path: str | Path,
    executor_contract_path: str | Path,
    runtime_contract_path: str | Path,
    publisher_contract_path: str | Path,
    base_manifest_path: str | Path,
    asset_root: str | Path,
    publication_root: str | Path,
    *,
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
) -> dict[str, Any]:
    """Read-only full-chain verification of a committed publication."""
    expectations = {
        "expected_source_evidence_digest": expected_source_evidence_digest,
        "expected_plan_digest": expected_plan_digest,
        "expected_authorization_digest": expected_authorization_digest,
        "expected_execution_receipt_digest": (
            expected_execution_receipt_digest
        ),
        "expected_execution_journal_head_digest": (
            expected_execution_journal_head_digest
        ),
        "expected_execution_attempt_digest": expected_execution_attempt_digest,
        "expected_publication_digest": expected_publication_digest,
        "expected_reingestion_digest": expected_reingestion_digest,
        "expected_output_report_body_digest": (
            expected_output_report_body_digest
        ),
        "expected_output_audit_head": expected_output_audit_head,
        "expected_output_evidence_digest": expected_output_evidence_digest,
    }
    for label, value in expectations.items():
        expectations[label] = _require_digest(value, label)

    base_path = _absolute_path(base_evidence_csv, "base_evidence_csv")
    attempt_path = _absolute_path(attempt_root, "attempt_root")
    publisher_path = _absolute_path(
        publisher_contract_path, "publisher_contract_path"
    )
    hypothesis_path = _absolute_path(
        hypothesis_contract_path, "hypothesis_contract_path"
    )
    executor_path = _absolute_path(
        executor_contract_path, "executor_contract_path"
    )
    runtime_path = _absolute_path(runtime_contract_path, "runtime_contract_path")
    manifest_path = _absolute_path(base_manifest_path, "base_manifest_path")
    assets_path = _absolute_path(asset_root, "asset_root")
    root = _validate_publication_location(
        publication_root, None, fresh=False
    )
    _validate_directory_layout(root)

    captured = _snapshot_inputs(
        base_path,
        attempt_path,
        hypothesis_path,
        executor_path,
        publisher_path,
        runtime_path,
    )
    _validate_captured_runtime_capsule(
        captured["objects"],
        captured["runtime_contract"],
        manifest_path,
        assets_path,
        attempt_path,
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
    _verify_runtime_attempt(
        attempt_path,
        captured["runtime_contract"],
        manifest_path,
        assets_path,
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
    output, reingestion, combined = _validate_chain_objects(
        captured["objects"],
        captured["publisher_contract"],
        captured["runtime_contract"],
        captured["base_rows"],
        captured["base_raw"],
        base_path,
        captured["hypothesis_raw"],
        hypothesis_path,
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
    values, raws = _load_publication_artifacts(root)
    marker = values["publication_commit"]
    integrity = marker.get("integrity")
    if type(integrity) is not dict or set(integrity) != {
        "algorithm",
        "publication_digest",
    } or integrity.get("algorithm") != "sha256-canonical-json-v1":
        raise ReingestionPublicationError("publication integrity is malformed")
    marker_body = {key: marker[key] for key in marker if key != "integrity"}
    if _digest(marker_body) != integrity["publication_digest"]:
        raise ReingestionPublicationError("publication digest differs")
    if integrity["publication_digest"] != expectations[
        "expected_publication_digest"
    ]:
        raise ReingestionPublicationError(
            "expected publication digest differs"
        )
    if (
        marker.get("schema_version") != PUBLICATION_SCHEMA_VERSION
        or marker.get("status") != "PUBLISHED_NOT_ADOPTED"
        or marker.get("nonclaims") != _NONCLAIMS
        or root.name != marker.get("publication_id")
    ):
        raise ReingestionPublicationError("publication marker differs from V1")

    expected_raws = {
        "publisher_contract": captured["publisher_raw"],
        "runtime_contract": captured["runtime_raw"],
        "materializer_contract": captured["attempt_raws"][
            "execution_materializer_contract"
        ],
        "source_hypothesis_contract": captured["hypothesis_raw"],
        "source_executor_contract": captured["executor_raw"],
        "base_evidence_csv": captured["base_raw"],
        **captured["attempt_raws"],
        "combined_rows": _canonical_file(combined),
        "output_report": _canonical_file(output),
        "reingestion_receipt": _canonical_file(reingestion),
    }
    for label, expected_raw in expected_raws.items():
        if raws.get(label) != expected_raw:
            raise ReingestionPublicationError(
                f"publication artifact differs: {label}"
            )
    expected_artifacts = {
        label: _publication_artifact(_ARTIFACT_LAYOUT[label], raw)
        for label, raw in expected_raws.items()
    }
    if marker.get("artifacts") != expected_artifacts:
        raise ReingestionPublicationError("publication artifact manifest differs")

    rebuilt_body = _marker_body(
        publication_id=marker["publication_id"],
        artifacts=expected_artifacts,
        publisher_contract=captured["publisher_contract"],
        publisher_raw=captured["publisher_raw"],
        hypothesis_raw=captured["hypothesis_raw"],
        executor_raw=captured["executor_raw"],
        base_raw=captured["base_raw"],
        base_rows=captured["base_rows"],
        objects=captured["objects"],
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
        raise ReingestionPublicationError(
            "publication marker does not reproduce its full chain"
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
        raise ReingestionPublicationError(
            "expected output digest differs"
        )
    return _publication_result(marker, root, verified=True)


__all__ = [
    "PUBLICATION_SCHEMA_VERSION",
    "PUBLISHER_CONTRACT_ID",
    "PUBLISHER_SCHEMA_VERSION",
    "ReingestionPublicationError",
    "publish_single_task_reingestion",
    "verify_single_task_reingestion_publication",
]
