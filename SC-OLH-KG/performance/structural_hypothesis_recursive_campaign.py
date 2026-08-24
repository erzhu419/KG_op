"""Run exactly one provenance-bound recursive structural-hypothesis step.

The campaign layer is deliberately a one-step state machine.  It accepts a
fully verified recursive-successor generation, or the non-terminal output of a
previous campaign step, authorizes only the first pending task, durably spends
one callback-start claim, and then stops after publishing one immutable report
advance.  It never loops and never retries a claimed callback.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, Mapping

from . import structural_hypothesis_execution as _execution
from . import structural_hypothesis_recursive_successor_materializer as _recursive_successor
from . import structural_hypothesis_reingestion_publisher as _publisher
from . import structural_hypothesis_single_task_runtime as _runtime
from . import structural_hypothesis_task_materializer as _materializer
from .structural_hypothesis_loop import canonical_json_bytes, verify_report_integrity


# Capture public delegates at definition time.  Tests may replace these local
# bindings, but post-import rebinding of producer modules cannot redirect a
# campaign operation.
_RS_VERIFY = _recursive_successor.verify_recursive_successor
_R_PREPARE = _runtime.prepare_single_task_attempt
_R_EXECUTE = _runtime.execute_single_task_attempt
_R_VERIFY = _runtime.verify_single_task_attempt
_E_BUILD_PLAN = _execution.build_execution_plan
_E_REINGEST = _execution.reingest_successful_receipts
_E_VERIFY_REINGESTION = _execution.verify_reingestion_integrity
_M_MATERIALIZE = _materializer.materialize_task_bundle
_M_VERIFY = _materializer.verify_materialized_task_bundle
_P_ATTEMPT_SNAPSHOTS = _publisher._attempt_snapshots
_P_VALIDATE_CAPTURED_RUNTIME = _publisher._validate_captured_runtime_capsule


_RENAME_NOREPLACE = 1
try:
    _LIBC = ctypes.CDLL(None, use_errno=True)
    _LIBC_RENAMEAT2 = _LIBC.renameat2
    _LIBC_RENAMEAT2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    _LIBC_RENAMEAT2.restype = ctypes.c_int
except (AttributeError, OSError):
    _LIBC_RENAMEAT2 = None


CAMPAIGN_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-recursive-campaign/1"
)
SOURCE_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-recursive-campaign-source/1"
)
SOURCE_COMMIT_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-recursive-campaign-source-commit/1"
)
PROVENANCE_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-recursive-campaign-provenance/1"
)
LEASE_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-recursive-campaign-lease/1"
)
CAPSULE_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-recursive-campaign-capsule/1"
)
CALLBACK_START_CLAIM_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-recursive-campaign-callback-start-claim/1"
)
ADVANCE_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-recursive-campaign-advance/1"
)
CAMPAIGN_CONTRACT_ID = "structural_hypothesis_recursive_campaign_v1"

# Patched to the canonical manifest digest after the two-file implementation is
# frozen.  Validation always requires exact equality with the external file.
_CAMPAIGN_CONTRACT_DIGEST = (
    "sha256:3950aeab12a9dda13c41a1dd03afce27c89f3d8d1364a1d51c8c8f8d9b9dfbc5"
)
_RECURSIVE_SUCCESSOR_CONTRACT_DIGEST = (
    "sha256:dfc58175e637df625758bcd77e41cacd11e1ad9d71a1b8224d66c51eec0cfd1c"
)
_RUNTIME_CONTRACT_DIGEST = (
    "sha256:d03529c64e6ea63b9997ded35fb6c0b44c6e17fb828f9e9db8960adb764a8c6b"
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
_SOURCE_FILES = {
    "performance/structural_hypothesis_recursive_successor_materializer.py": (
        "de6f16130a49da2e3bd7fe9e5506e573a6db6dd27f9cd701c307976e39bc14c2"
    ),
    "performance/structural_hypothesis_single_task_runtime.py": (
        "618c336aee3558efb05a5415201cdf6b5cb1a7e028b90d94f2ac204b072e7fe4"
    ),
    "performance/structural_hypothesis_execution.py": (
        "7c5cc27f8e97da9b51e57975f63e860a23463d5e7728d1089f32978146f27c9b"
    ),
    "performance/structural_hypothesis_task_materializer.py": (
        "93345ef22df0cc9c5665be35722ad4646fb233d7a8bbd1294f0ca051cf64f20b"
    ),
    "performance/structural_hypothesis_reingestion_publisher.py": (
        "05d2ca6668414167cbe56a042f9564ff56728faf73ff5dd6fb167e9a16c3dd4d"
    ),
}

_STATE_PREFIX = Path("kg-op/structural-hypothesis-recursive-campaign/v1")
_MAX_RECURSIVE_DEPTH = 30
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_TASK_ID = re.compile(r"task:[0-9a-f]{24}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_AUTHORIZATION_ID = re.compile(r"recursive-campaign-v1:[0-9a-f]{64}\Z")

_INSPECTED = "INSPECTED_RECURSIVE_CAMPAIGN_SOURCE_NONTERMINAL_NOT_AUTHORIZED"
_AUTHORIZED = "RECURSIVE_CAMPAIGN_AUTHORIZED_ONE_CALLBACK_START_LEASED"
_CALLBACK_COMPLETED = (
    "RECURSIVE_CAMPAIGN_CALLBACK_START_CLAIMED_RUNTIME_COMPLETED_HARD_STOP"
)
_CALLBACK_INCOMPLETE = (
    "RECURSIVE_CAMPAIGN_CALLBACK_START_CLAIMED_RUNTIME_INCOMPLETE_HARD_STOP"
)
_ADVANCED_NONTERMINAL = (
    "ADVANCED_RECURSIVE_CAMPAIGN_ONE_STEP_NONTERMINAL_HARD_STOP"
)
_ADVANCED_TERMINAL = "ADVANCED_RECURSIVE_CAMPAIGN_ONE_STEP_TERMINAL_HARD_STOP"

_DEPENDENCY_KEYS = {
    "hypothesis_contract_path",
    "executor_contract_path",
    "runtime_contract_path",
    "materializer_contract_path",
    "base_manifest_path",
    "asset_root",
}
_SUCCESSOR_SOURCE_KEYS = {
    "schema_version", "source_kind", "dependencies", "verify_args",
    "verify_kwargs",
}
_CAMPAIGN_SOURCE_KEYS = {
    "schema_version", "source_kind", "dependencies",
    "campaign_contract_path", "campaign_root", "expected_campaign_digest",
    "expected_lease_digest", "expected_callback_start_claim_digest",
    "expected_advance_digest", "expected_output_evidence_digest",
    "expected_output_report_body_digest", "expected_output_audit_head",
    "expected_reingestion_digest", "expected_next_bundle_digest",
    "expected_next_plan_digest",
}
_SUCCESSOR_VERIFY_KWARGS = {
    "adoption_id", "source_successor_id", "advance_id",
    "recursive_successor_id", "expected_adoption_digest",
    "expected_source_pending_evidence_digest",
    "expected_source_first_pending_projection_digest",
    "expected_source_successor_digest", "expected_source_bundle_digest",
    "expected_source_plan_digest", "completed_task_id",
    "expected_completed_task_digest",
    "expected_source_provenance_binding_digest",
    "expected_source_authorization_digest", "expected_source_attempt_digest",
    "expected_source_execution_receipt_digest",
    "expected_source_execution_journal_head_digest", "expected_advance_digest",
    "expected_advance_reingestion_digest",
    "expected_advance_output_report_body_digest",
    "expected_advance_output_audit_head",
    "expected_advance_output_evidence_digest",
    "expected_next_pending_evidence_digest",
    "expected_next_first_pending_projection_digest",
    "expected_recursive_successor_digest", "expected_next_bundle_digest",
    "expected_next_plan_digest",
}

_ROOT_LAYOUT = {
    "campaign_contract": "campaign_contract.json",
    "source_directory": "source",
    "lease": "lease.json",
    "campaign_commit": "campaign.json",
    "callback_start_claim": "callback_start_claim.json",
    "advance_directory": "advance",
}
_SOURCE_LAYOUT = {
    "descriptor": "descriptor.json",
    "commit": "commit.json",
    "rows": "rows.json",
    "report": "report.json",
    "bundle": "bundle.json",
}
_ADVANCE_LAYOUT = {
    "execution_receipt": "execution_receipt.json",
    "combined_rows": "combined_rows.json",
    "output_report": "output_report.json",
    "reingestion_receipt": "reingestion_receipt.json",
    "next_bundle": "next_bundle.json",
    "advance_commit": "advance.json",
}

_NONCLAIMS = [
    "inspection_is_not_authorization",
    "local_authorization_is_not_external_authority",
    "local_digest_is_not_signature",
    "lease_is_not_exactly_once_execution",
    "callback_start_claim_is_not_callback_success",
    "crash_after_claim_is_not_retry_authority",
    "one_step_advance_is_not_run_all",
    "failed_execution_is_not_scientific_refutation",
    "reingestion_is_not_external_verification",
    "recursive_source_is_not_global_currentness",
    "terminal_is_only_frozen_pending_evidence_exhaustion",
    "no_network_access_by_campaign_layer",
    "no_scheduler_access_by_campaign_layer",
    "no_mutation_of_operations_research_artifacts",
]
_REQUIRED_RUNTIME_ENVIRONMENT = {
    "SCOLHKG_OFFLINE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


class RecursiveCampaignError(ValueError):
    """Raised when any recursive campaign gate fails closed."""


def _new_verification_context() -> dict[str, Any]:
    """Create one public-call-scoped semantic memo and freshness ledger.

    Recursive campaign sources form a single predecessor chain.  Re-running a
    predecessor's full semantic verifier from every final recapture would turn
    that chain into an exponential verification tree.  The memo therefore
    admits each exact predecessor generation once, while the freshness ledger
    retains enough local evidence to recapture every admitted generation in a
    linear final pass.
    """
    return {
        "semantic_cache": {},
        "semantic_active": set(),
        "semantic_cache_miss_order": [],
        "source_cache": {},
        "source_attempt_absence": {},
        "successor_freshness": {},
        "campaign_freshness": {},
    }


def _require_verification_context(
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    if context is None:
        return _new_verification_context()
    required = {
        "semantic_cache", "semantic_active", "semantic_cache_miss_order",
        "source_cache", "source_attempt_absence", "successor_freshness",
        "campaign_freshness",
    }
    if type(context) is not dict or set(context) != required:
        raise RecursiveCampaignError("verification context shape differs")
    if (
        type(context["semantic_cache"]) is not dict
        or type(context["semantic_active"]) is not set
        or type(context["semantic_cache_miss_order"]) is not list
        or type(context["source_cache"]) is not dict
        or type(context["source_attempt_absence"]) is not dict
        or type(context["successor_freshness"]) is not dict
        or type(context["campaign_freshness"]) is not dict
    ):
        raise RecursiveCampaignError("verification context members differ")
    return context


def _record_semantic_cache_miss(
    context: dict[str, Any], resolved_root: str
) -> None:
    """Internal deterministic hook used by linear-complexity KATs."""
    context["semantic_cache_miss_order"].append(resolved_root)


def _require_runtime_environment() -> None:
    for key, expected in _REQUIRED_RUNTIME_ENVIRONMENT.items():
        if os.environ.get(key) != expected:
            raise RecursiveCampaignError(
                f"required runtime environment differs: {key}"
            )


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _canonical_file(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _clone(value: Any, label: str = "value") -> Any:
    def check(item: Any, path: str) -> None:
        if item is None or type(item) in (str, int, bool):
            return
        if type(item) is float:
            if item != item or item in (float("inf"), float("-inf")):
                raise RecursiveCampaignError(f"{label} contains non-finite float at {path}")
            return
        if type(item) is list:
            for index, child in enumerate(item):
                check(child, f"{path}[{index}]")
            return
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise RecursiveCampaignError(f"{label} contains non-string key")
                check(child, f"{path}.{key}")
            return
        raise RecursiveCampaignError(f"{label} contains unsupported type at {path}")

    check(value, "$")
    try:
        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (UnicodeError, ValueError, TypeError) as error:
        raise RecursiveCampaignError(f"{label} is not strict JSON") from error


def _decode_strict_json(raw: bytes, label: str) -> Any:
    """Decode UTF-8 JSON while rejecting duplicate keys and non-JSON numbers."""

    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise RecursiveCampaignError(
                    f"{label} contains duplicate object key: {key}"
                )
            value[key] = child
        return value

    def reject_constant(value: str) -> None:
        raise RecursiveCampaignError(
            f"{label} contains non-finite JSON number: {value}"
        )

    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=object_from_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise RecursiveCampaignError(f"{label} is not strict JSON") from error
    return _clone(value, label)


def _exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise RecursiveCampaignError(f"{label} keys differ")
    return value


def _require_digest(value: Any, label: str) -> str:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise RecursiveCampaignError(f"{label} must be a lowercase sha256 digest")
    return value


def _require_identifier(value: Any, label: str) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise RecursiveCampaignError(f"{label} is malformed")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if isinstance(value, Path):
        value = str(value)
    if type(value) is not str or not value or "\x00" in value:
        raise RecursiveCampaignError(f"{label} must be an absolute path string")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise RecursiveCampaignError(f"{label} must be normalized and absolute")
    return path


def _lexists(path: Path) -> bool:
    try:
        os.lstat(path)
        return True
    except FileNotFoundError:
        return False


def _secure_directory(path: Path, *, exact_mode: int | None = None) -> None:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise RecursiveCampaignError(f"cannot inspect directory: {path}") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RecursiveCampaignError(f"directory is aliased or not a directory: {path}")
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o022:
        raise RecursiveCampaignError(f"directory ownership or writable mode differs: {path}")
    if exact_mode is not None and stat.S_IMODE(info.st_mode) != exact_mode:
        raise RecursiveCampaignError(f"directory mode differs: {path}")


def _read_bytes(path: Path, label: str, *, exact_mode: int | None = None) -> bytes:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise RecursiveCampaignError(f"cannot inspect {label}") from error
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise RecursiveCampaignError(f"cannot open {label}") from error
    try:
        info = os.fstat(fd)
        signature = lambda item: (
            item.st_dev, item.st_ino, item.st_mode, item.st_nlink,
            item.st_uid, item.st_gid, item.st_size, item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o022
            or signature(before) != signature(info)
        ):
            raise RecursiveCampaignError(f"{label} must be a single-link regular file")
        if exact_mode is not None and stat.S_IMODE(info.st_mode) != exact_mode:
            raise RecursiveCampaignError(f"{label} mode differs")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        try:
            final_path = os.lstat(path)
        except OSError as error:
            raise RecursiveCampaignError(f"{label} changed during read") from error
        if (
            signature(after) != signature(info)
            or signature(final_path) != signature(info)
            or len(raw) != info.st_size
        ):
            raise RecursiveCampaignError(f"{label} changed during read")
        return raw
    finally:
        os.close(fd)


def _stat_signature(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_gid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_bytes_at(
    directory_fd: int,
    name: str,
    label: str,
    *,
    exact_mode: int = 0o600,
) -> bytes:
    """Read one stable owned leaf relative to an already-held directory."""
    if type(name) is not str or Path(name).name != name or "\x00" in name:
        raise RecursiveCampaignError(f"{label} leaf name is unsafe")
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise RecursiveCampaignError(f"cannot inspect held {label}") from error
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise RecursiveCampaignError(f"cannot open held {label}") from error
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != exact_mode
            or _stat_signature(before) != _stat_signature(info)
        ):
            raise RecursiveCampaignError(
                f"held {label} must be an exact single-link regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        try:
            final = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise RecursiveCampaignError(f"held {label} changed") from error
        if (
            _stat_signature(after) != _stat_signature(info)
            or _stat_signature(final) != _stat_signature(info)
            or len(raw) != info.st_size
        ):
            raise RecursiveCampaignError(f"held {label} changed")
        return raw
    finally:
        os.close(fd)


def _validate_held_publication_directory(
    directory_fd: int,
    guard: Mapping[str, Any],
    *,
    phase: str,
) -> None:
    """Validate one marker generation entirely through a held dirfd."""
    identity = guard["identity"]
    expected_names = guard[f"{phase}_names"]
    expected_raws = guard[f"{phase}_raws"]
    child_identities = guard.get("child_identities", {})
    info = os.fstat(directory_fd)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
        or (info.st_dev, info.st_ino) != identity
    ):
        raise RecursiveCampaignError(
            f"guarded {guard['label']} directory identity differs"
        )
    try:
        names = set(os.listdir(directory_fd))
    except OSError as error:
        raise RecursiveCampaignError(
            f"cannot enumerate guarded {guard['label']}"
        ) from error
    if names != expected_names or set(expected_raws) - names:
        raise RecursiveCampaignError(
            f"guarded {guard['label']} {phase} names differ"
        )
    for name, expected_child_identity in child_identities.items():
        try:
            child = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise RecursiveCampaignError(
                f"cannot inspect guarded {guard['label']} child {name}"
            ) from error
        if (
            not stat.S_ISDIR(child.st_mode)
            or stat.S_ISLNK(child.st_mode)
            or child.st_uid != os.geteuid()
            or stat.S_IMODE(child.st_mode) != 0o700
            or (child.st_dev, child.st_ino) != expected_child_identity
        ):
            raise RecursiveCampaignError(
                f"guarded {guard['label']} child identity differs: {name}"
            )
    for name, expected_raw in expected_raws.items():
        if _read_bytes_at(
            directory_fd,
            name,
            f"{guard['label']} {name}",
        ) != expected_raw:
            raise RecursiveCampaignError(
                f"guarded {guard['label']} leaf differs: {name}"
            )
    # Re-enumerate and rebind child entries after all leaf reads so the guard
    # never accepts a directory assembled from different generations.
    if set(os.listdir(directory_fd)) != expected_names:
        raise RecursiveCampaignError(
            f"guarded {guard['label']} changed during {phase} validation"
        )
    for name, expected_child_identity in child_identities.items():
        child = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (child.st_dev, child.st_ino) != expected_child_identity:
            raise RecursiveCampaignError(
                f"guarded {guard['label']} child changed: {name}"
            )


def _parse_json(raw: bytes, label: str) -> Any:
    try:
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            raise ValueError("noncanonical line ending")
        value = _decode_strict_json(raw, label)
    except (UnicodeError, ValueError, TypeError) as error:
        raise RecursiveCampaignError(f"{label} is not strict JSON") from error
    if raw != _canonical_file(value):
        raise RecursiveCampaignError(f"{label} is not canonical JSON")
    return value


def _read_json(path: Path, label: str, *, exact_mode: int | None = None) -> Any:
    return _parse_json(_read_bytes(path, label, exact_mode=exact_mode), label)


def _read_repo_json_object(path: Path, label: str) -> dict[str, Any]:
    """Read an immutable repository JSON object independent of formatting."""
    value = _decode_strict_json(_read_bytes(path, label), label)
    if type(value) is not dict:
        raise RecursiveCampaignError(f"{label} must be a JSON object")
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir_new(path: Path) -> None:
    try:
        os.mkdir(path, 0o700)
    except OSError as error:
        raise RecursiveCampaignError(f"cannot create fresh directory: {path}") from error
    os.chmod(path, 0o700, follow_symlinks=False)
    _fsync_directory(path.parent)


def _directory_identity(path: Path) -> tuple[int, int]:
    info = os.lstat(path)
    _secure_directory(path, exact_mode=0o700)
    return info.st_dev, info.st_ino


def _rename_noreplace(
    source_directory_fd: int,
    source_name: str,
    target_directory_fd: int,
    target_name: str,
    *,
    guard_before=None,
) -> None:
    """Atomically move one leaf without ever replacing an existing target."""
    if _LIBC_RENAMEAT2 is None:
        raise RecursiveCampaignError(
            "Linux renameat2(RENAME_NOREPLACE) is unavailable"
        )
    if guard_before is not None:
        guard_before()
    ctypes.set_errno(0)
    result = _LIBC_RENAMEAT2(
        source_directory_fd,
        os.fsencode(source_name),
        target_directory_fd,
        os.fsencode(target_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {
        errno.ENOSYS,
        errno.EINVAL,
        errno.EOPNOTSUPP,
        getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
    }:
        raise RecursiveCampaignError(
            "Linux renameat2(RENAME_NOREPLACE) is unsupported"
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        f"{source_name} -> {target_name}",
    )


def _hold_publication_guards(
    guards: tuple[Mapping[str, Any], ...],
    *,
    target_parent: Path,
    target_directory_fd: int,
    directory_flags: int,
) -> list[tuple[int, Mapping[str, Any]]]:
    held: list[tuple[int, Mapping[str, Any]]] = []
    try:
        for guard in guards:
            guard_path = guard.get("path")
            if not isinstance(guard_path, Path) or not guard_path.is_absolute():
                raise RecursiveCampaignError("publication guard path is malformed")
            if guard_path == target_parent:
                directory_fd = os.dup(target_directory_fd)
            else:
                try:
                    directory_fd = os.open(guard_path, directory_flags)
                except OSError as error:
                    raise RecursiveCampaignError(
                        f"cannot hold guarded directory: {guard_path}"
                    ) from error
            held.append((directory_fd, guard))
            try:
                path_info = os.lstat(guard_path)
                fd_info = os.fstat(directory_fd)
            except OSError as error:
                raise RecursiveCampaignError(
                    f"cannot bind guarded directory: {guard_path}"
                ) from error
            if (
                not stat.S_ISDIR(path_info.st_mode)
                or (path_info.st_dev, path_info.st_ino)
                != (fd_info.st_dev, fd_info.st_ino)
                or (fd_info.st_dev, fd_info.st_ino) != guard.get("identity")
            ):
                raise RecursiveCampaignError(
                    f"guarded directory path identity differs: {guard_path}"
                )
        return held
    except BaseException:
        for directory_fd, _guard in held:
            os.close(directory_fd)
        raise


def _write_new(
    path: Path,
    value: Any,
    *,
    expected_parent_identity: tuple[int, int] | None = None,
    publication_guards: tuple[Mapping[str, Any], ...] = (),
) -> bytes:
    raw = _canonical_file(value)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        target_fd = os.open(path.parent, directory_flags)
    except OSError as error:
        raise RecursiveCampaignError(f"cannot hold target directory: {path.parent}") from error
    staging = _state_home() / _STATE_PREFIX
    try:
        staging_fd = os.open(staging, directory_flags)
    except OSError as error:
        os.close(target_fd)
        raise RecursiveCampaignError("cannot hold campaign staging prefix") from error
    temporary_name = f".tmp-{os.getpid()}-{secrets.token_hex(12)}"
    temp_fd = None
    held_guards: list[tuple[int, Mapping[str, Any]]] = []
    try:
        target_info = os.fstat(target_fd)
        path_info = os.lstat(path.parent)
        staging_info = os.fstat(staging_fd)
        staging_path_info = os.lstat(staging)
        identity = (target_info.st_dev, target_info.st_ino)
        if (
            not stat.S_ISDIR(target_info.st_mode)
            or target_info.st_uid != os.geteuid()
            or stat.S_IMODE(target_info.st_mode) != 0o700
            or (path_info.st_dev, path_info.st_ino) != identity
            or (
                expected_parent_identity is not None
                and identity != expected_parent_identity
            )
        ):
            raise RecursiveCampaignError("target directory identity changed")
        if (
            not stat.S_ISDIR(staging_info.st_mode)
            or staging_info.st_uid != os.geteuid()
            or stat.S_IMODE(staging_info.st_mode) != 0o700
            or (staging_path_info.st_dev, staging_path_info.st_ino)
            != (staging_info.st_dev, staging_info.st_ino)
            or staging_info.st_dev != target_info.st_dev
        ):
            raise RecursiveCampaignError(
                "campaign staging directory identity or filesystem differs"
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        temp_fd = os.open(temporary_name, flags, 0o600, dir_fd=staging_fd)
        os.fchmod(temp_fd, 0o600)
        offset = 0
        while offset < len(raw):
            written = os.write(temp_fd, raw[offset:])
            if written <= 0:
                raise OSError("short write")
            offset += written
        os.fsync(temp_fd)
        os.close(temp_fd)
        temp_fd = None
        if publication_guards:
            held_guards = _hold_publication_guards(
                publication_guards,
                target_parent=path.parent,
                target_directory_fd=target_fd,
                directory_flags=directory_flags,
            )
            target_guards = [
                guard
                for _directory_fd, guard in held_guards
                if guard["path"] == path.parent
            ]
            if (
                len(target_guards) != 1
                or target_guards[0]["after_raws"].get(path.name) != raw
            ):
                raise RecursiveCampaignError(
                    "marker publication guard does not bind the target leaf"
                )

        def validate_guards_before() -> None:
            for directory_fd, guard in held_guards:
                _validate_held_publication_directory(
                    directory_fd, guard, phase="before"
                )

        _rename_noreplace(
            staging_fd,
            temporary_name,
            target_fd,
            path.name,
            guard_before=(validate_guards_before if held_guards else None),
        )
        os.fsync(target_fd)
        os.fsync(staging_fd)
        for directory_fd, guard in held_guards:
            _validate_held_publication_directory(
                directory_fd, guard, phase="after"
            )
        final_parent = os.lstat(path.parent)
        if (final_parent.st_dev, final_parent.st_ino) != identity:
            raise RecursiveCampaignError("target directory path changed after publish")
        final_staging = os.lstat(staging)
        if (
            final_staging.st_dev,
            final_staging.st_ino,
        ) != (
            staging_info.st_dev,
            staging_info.st_ino,
        ):
            raise RecursiveCampaignError(
                "campaign staging directory path changed after publish"
            )
    except RecursiveCampaignError:
        raise
    except OSError as error:
        raise RecursiveCampaignError(f"atomic no-clobber publish failed: {path}") from error
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        for directory_fd, _guard in held_guards:
            os.close(directory_fd)
        try:
            os.unlink(temporary_name, dir_fd=staging_fd)
            os.fsync(staging_fd)
        except FileNotFoundError:
            pass
        os.close(staging_fd)
        os.close(target_fd)
    if _read_bytes(path, str(path), exact_mode=0o600) != raw:
        raise RecursiveCampaignError(f"published artifact differs: {path}")
    return raw


def _publish_or_match(
    path: Path,
    value: Any,
    *,
    expected_parent_identity: tuple[int, int] | None = None,
) -> None:
    expected = _canonical_file(value)
    if _lexists(path):
        if _read_bytes(path, str(path), exact_mode=0o600) != expected:
            raise RecursiveCampaignError(f"partial campaign artifact differs: {path}")
        return
    _write_new(
        path, value, expected_parent_identity=expected_parent_identity
    )


def _require_names(path: Path, expected: set[str], label: str) -> None:
    _secure_directory(path, exact_mode=0o700)
    try:
        names = set(os.listdir(path))
    except OSError as error:
        raise RecursiveCampaignError(f"cannot enumerate {label}") from error
    if names != expected:
        raise RecursiveCampaignError(f"{label} exact names differ")


def _state_home() -> Path:
    configured = os.environ.get("XDG_STATE_HOME", "")
    base = Path(configured) if configured else Path.home() / ".local" / "state"
    if not base.is_absolute() or ".." in base.parts or base != base.resolve(strict=False):
        raise RecursiveCampaignError("XDG state home is not canonical and absolute")
    return base


def _validate_campaign_prefix_read_only() -> None:
    """Validate every existing fixed-prefix component without creating it."""
    base = _state_home()
    if not _lexists(base):
        return
    _secure_directory(base)
    cursor = base
    for index, component in enumerate(_STATE_PREFIX.parts):
        cursor = cursor / component
        if not _lexists(cursor):
            return
        # ``kg-op`` is a shared, pre-existing state ancestor and may be 0755;
        # it still has to be owned by this euid and not group/world writable.
        # Campaign-owned descendants are private and remain exactly 0700.
        _secure_directory(cursor, exact_mode=None if index == 0 else 0o700)


def _campaign_root(
    path_value: Any,
    campaign_id: str,
    *,
    fresh: bool,
    allow_existing: bool = False,
) -> Path:
    campaign_id = _require_identifier(campaign_id, "campaign_id")
    root = _absolute_path(path_value, "campaign_root")
    expected = _state_home() / _STATE_PREFIX / campaign_id
    if root != expected:
        raise RecursiveCampaignError("campaign root does not match frozen prefix and ID")
    _validate_campaign_prefix_read_only()
    if fresh:
        if _lexists(root) and not allow_existing:
            raise RecursiveCampaignError("campaign root already exists")
        if _lexists(root):
            _secure_directory(root, exact_mode=0o700)
    else:
        _secure_directory(root, exact_mode=0o700)
    return root


def _ensure_campaign_parent(root: Path) -> None:
    base = _state_home()
    target = root.parent
    if target != base and base not in target.parents:
        raise RecursiveCampaignError("campaign root escapes XDG state home")
    if not _lexists(base):
        try:
            base.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(base, 0o700, follow_symlinks=False)
        except OSError as error:
            raise RecursiveCampaignError("cannot create XDG state home") from error
    _secure_directory(base)
    cursor = target
    missing: list[Path] = []
    while not _lexists(cursor):
        missing.append(cursor)
        if cursor == base:
            break
        cursor = cursor.parent
    _secure_directory(cursor)
    for path in reversed(missing):
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
        os.chmod(path, 0o700, follow_symlinks=False)
        _secure_directory(path, exact_mode=0o700)
        _fsync_directory(path.parent)


def _load_contract(path_value: Any) -> tuple[dict[str, Any], bytes]:
    path = _absolute_path(path_value, "campaign_contract_path")
    contract = _read_repo_json_object(path, "campaign contract")
    if (
        type(contract) is not dict
        or contract.get("schema_version") != CAMPAIGN_SCHEMA_VERSION
        or contract.get("contract_id") != CAMPAIGN_CONTRACT_ID
        or contract.get("state_prefix") != str(_STATE_PREFIX)
        or contract.get("source_files") != _SOURCE_FILES
        or _digest(contract) != _CAMPAIGN_CONTRACT_DIGEST
    ):
        raise RecursiveCampaignError("campaign contract identity differs")
    repo_root = Path(__file__).resolve().parents[1]
    for relative, expected_sha in _SOURCE_FILES.items():
        source_path = repo_root / relative
        source_raw = _read_bytes(source_path, f"pinned source file {relative}")
        if hashlib.sha256(source_raw).hexdigest() != expected_sha:
            raise RecursiveCampaignError(f"pinned source file differs: {relative}")
    return contract, _canonical_file(contract)


def _load_dependency_objects(dependencies: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(dependencies, _DEPENDENCY_KEYS, "source dependencies")
    paths = {key: _absolute_path(value, key) for key, value in dependencies.items()}
    values = {
        "hypothesis": _read_repo_json_object(
            paths["hypothesis_contract_path"], "hypothesis contract"
        ),
        "executor": _read_repo_json_object(
            paths["executor_contract_path"], "executor contract"
        ),
        "runtime": _read_repo_json_object(
            paths["runtime_contract_path"], "runtime contract"
        ),
        "materializer": _read_repo_json_object(
            paths["materializer_contract_path"], "materializer contract"
        ),
    }
    expected = {
        "hypothesis": _HYPOTHESIS_CONTRACT_DIGEST,
        "executor": _EXECUTOR_CONTRACT_DIGEST,
        "runtime": _RUNTIME_CONTRACT_DIGEST,
        "materializer": _MATERIALIZER_CONTRACT_DIGEST,
    }
    for label, digest in expected.items():
        if _digest(values[label]) != digest:
            raise RecursiveCampaignError(f"{label} contract digest differs")
    base = paths["base_manifest_path"]
    asset = paths["asset_root"]
    if base.is_symlink() or not base.is_file() or asset.is_symlink() or not asset.is_dir():
        raise RecursiveCampaignError("base manifest or asset root is unavailable/aliased")
    return {"paths": paths, **values}


def _normalize_source(source: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise RecursiveCampaignError("source must be a JSON mapping")
    cloned = _clone(dict(source), "source descriptor")
    kind = cloned.get("source_kind")
    keys = _SUCCESSOR_SOURCE_KEYS if kind == "recursive_successor_v1" else _CAMPAIGN_SOURCE_KEYS
    _exact_keys(cloned, keys, "source descriptor")
    if cloned.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise RecursiveCampaignError("source descriptor schema differs")
    _load_dependency_objects(cloned["dependencies"])
    if kind == "recursive_successor_v1":
        args = cloned["verify_args"]
        kwargs = cloned["verify_kwargs"]
        if type(args) is not list or len(args) != 21:
            raise RecursiveCampaignError("recursive successor verify_args must contain 21 paths")
        for index, value in enumerate(args):
            _absolute_path(value, f"verify_args[{index}]")
        _exact_keys(kwargs, _SUCCESSOR_VERIFY_KWARGS, "recursive successor verify_kwargs")
        for key, value in kwargs.items():
            if key.endswith("_digest") or key.endswith("_head"):
                _require_digest(value, key)
            elif type(value) is not str or not value:
                raise RecursiveCampaignError(f"{key} must be a nonempty string")
        cross = {
            7: "hypothesis_contract_path", 8: "executor_contract_path",
            9: "runtime_contract_path", 11: "materializer_contract_path",
            13: "base_manifest_path", 14: "asset_root",
        }
        for index, key in cross.items():
            if args[index] != cloned["dependencies"][key]:
                raise RecursiveCampaignError(f"source dependency cross-binding differs: {key}")
    elif kind == "recursive_campaign_v1":
        for key in ("campaign_contract_path", "campaign_root"):
            _absolute_path(cloned[key], key)
        for key in _CAMPAIGN_SOURCE_KEYS - {
            "schema_version", "source_kind", "dependencies",
            "campaign_contract_path", "campaign_root",
        }:
            _require_digest(cloned[key], key)
    else:
        raise RecursiveCampaignError("source_kind is not accepted")
    return cloned


def _task_from_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    try:
        tasks = bundle["plan"]["tasks"]
        task = tasks[0]
    except (KeyError, IndexError, TypeError) as error:
        raise RecursiveCampaignError("source bundle lacks its first task") from error
    if (
        type(tasks) is not list
        or not tasks
        or type(task) is not dict
        or type(task.get("task_id")) is not str
        or not _TASK_ID.fullmatch(task["task_id"])
        or type(task.get("task_digest")) is not str
        or not _DIGEST.fullmatch(task["task_digest"])
        or task.get("ordinal") != 0
    ):
        raise RecursiveCampaignError("source bundle first task is malformed")
    return _clone(task, "first task")


def _validate_report_rows_bundle(
    *,
    rows: Any,
    report: Any,
    bundle: Any,
    dependencies: Mapping[str, Any],
    checkpoint_root: Path,
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    if type(rows) is not list or type(report) is not dict or type(bundle) is not dict:
        raise RecursiveCampaignError("source rows/report/bundle types differ")
    rows = _clone(rows, "source rows")
    report = _clone(report, "source report")
    bundle = _clone(bundle, "source bundle")
    if (
        not verify_report_integrity(report)
        or report.get("evidence_digest") != _digest(rows)
        or type(report.get("pending_evidence")) is not list
        or not report["pending_evidence"]
    ):
        raise RecursiveCampaignError("source rows/report integrity or nonterminal state differs")
    contracts = _load_dependency_objects(dependencies)
    try:
        verified = _M_VERIFY(
            bundle,
            report,
            contracts["hypothesis"],
            contracts["executor"],
            contracts["materializer"],
            contracts["paths"]["base_manifest_path"],
            contracts["paths"]["asset_root"],
            checkpoint_root,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError("public task bundle verification failed") from error
    tasks = bundle.get("plan", {}).get("tasks")
    if (
        verified is not True
        or bundle.get("status") != "MATERIALIZED_NOT_AUTHORIZED"
        or type(tasks) is not list
        or not tasks
        or bundle.get("task_count") != len(tasks)
        or bundle["plan"].get("proposal_count") != len(tasks)
        or len(report["pending_evidence"]) != len(tasks)
    ):
        raise RecursiveCampaignError("source bundle does not strongly match the report")
    _task_from_bundle(bundle)
    return rows, report, bundle


def _derive_successor_source(
    descriptor: Mapping[str, Any], *, require_attempt_absent: bool
) -> dict[str, Any]:
    args = descriptor["verify_args"]
    kwargs = descriptor["verify_kwargs"]
    try:
        first = _RS_VERIFY(*args, **kwargs)
        second = _RS_VERIFY(*args, **kwargs)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError("public recursive successor verification failed") from error
    if first != second or not str(first.get("status", "")).startswith("VERIFIED_"):
        raise RecursiveCampaignError("recursive successor generation changed or is unverified")
    required = {
        "recursive_successor_digest": kwargs["expected_recursive_successor_digest"],
        "pending_evidence_digest": kwargs["expected_next_pending_evidence_digest"],
        "first_pending_projection_digest": kwargs[
            "expected_next_first_pending_projection_digest"
        ],
        "bundle_digest": kwargs["expected_next_bundle_digest"],
        "plan_digest": kwargs["expected_next_plan_digest"],
    }
    for key, expected in required.items():
        if first.get(key) != expected:
            raise RecursiveCampaignError(f"recursive successor result anchor differs: {key}")
    attempt_root = _absolute_path(first.get("future_attempt_root"), "future_attempt_root")
    checkpoint_root = _absolute_path(first.get("checkpoint_root"), "checkpoint_root")
    if checkpoint_root != attempt_root / "checkpoints":
        raise RecursiveCampaignError("recursive successor checkpoint binding differs")
    if require_attempt_absent and _lexists(attempt_root):
        raise RecursiveCampaignError("source-bound next attempt root already exists")
    advance_root = _absolute_path(args[17], "advance_root")
    successor_root = _absolute_path(args[19], "recursive_successor_root")
    rows = _read_json(advance_root / "combined_rows.json", "source combined rows")
    report = _read_json(advance_root / "output_report.json", "source output report")
    bundle = _read_json(successor_root / "bundle.json", "source task bundle")
    rows, report, bundle = _validate_report_rows_bundle(
        rows=rows,
        report=report,
        bundle=bundle,
        dependencies=descriptor["dependencies"],
        checkpoint_root=checkpoint_root,
    )
    if (
        bundle["integrity"]["bundle_digest"] != first["bundle_digest"]
        or bundle["plan"]["integrity"]["plan_digest"] != first["plan_digest"]
        or report["evidence_digest"] != first["advance_output_evidence_digest"]
        or len(bundle["plan"]["tasks"]) != first["task_count"]
    ):
        raise RecursiveCampaignError("recursive successor snapshots differ from public result")
    try:
        final = _RS_VERIFY(*args, **kwargs)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError("recursive successor post-capture verification failed") from error
    if final != first:
        raise RecursiveCampaignError("recursive successor changed during capture")
    return {
        "source_kind": "recursive_successor_v1",
        "source_state_digest": first["recursive_successor_digest"],
        "source_verification": first,
        "rows": rows,
        "report": report,
        "bundle": bundle,
        "next_attempt_root": attempt_root,
        "checkpoint_root": checkpoint_root,
    }


def _campaign_source_traversal_guard(
    descriptor: Mapping[str, Any],
    *,
    depth: int,
    visited: set[str],
) -> tuple[Path, str]:
    """Apply cycle/depth gates even when a derived-source memo is reused."""
    root = _absolute_path(descriptor["campaign_root"], "prior campaign_root")
    root = _campaign_root(root, root.name, fresh=False)
    try:
        resolved = str(root.resolve(strict=True))
    except OSError as error:
        raise RecursiveCampaignError("prior campaign root is unavailable") from error
    if resolved in visited:
        raise RecursiveCampaignError("recursive campaign source cycle detected")
    if depth >= _MAX_RECURSIVE_DEPTH:
        raise RecursiveCampaignError("recursive campaign source depth exceeds 30")
    return root, resolved


def _derive_campaign_source(
    descriptor: Mapping[str, Any],
    *,
    require_attempt_absent: bool,
    depth: int,
    visited: set[str],
    verification_context: dict[str, Any],
) -> dict[str, Any]:
    # Reject an out-of-policy source root before resolving or reading any of
    # its leaves.  Recursive input cannot widen the frozen XDG state boundary.
    root, resolved = _campaign_source_traversal_guard(
        descriptor, depth=depth, visited=visited
    )
    prior_descriptor = _read_json(
        root / _ROOT_LAYOUT["source_directory"] / _SOURCE_LAYOUT["descriptor"],
        "prior source descriptor",
        exact_mode=0o600,
    )
    if prior_descriptor.get("dependencies") != descriptor["dependencies"]:
        raise RecursiveCampaignError("recursive campaign dependencies drifted")
    expected = {
        key: descriptor[key]
        for key in (
            "expected_campaign_digest", "expected_lease_digest",
            "expected_callback_start_claim_digest", "expected_advance_digest",
            "expected_output_evidence_digest", "expected_output_report_body_digest",
            "expected_output_audit_head", "expected_reingestion_digest",
            "expected_next_bundle_digest", "expected_next_plan_digest",
        )
    }
    prior = _verify_campaign_internal(
        prior_descriptor,
        descriptor["campaign_contract_path"],
        root,
        campaign_id=root.name,
        depth=depth + 1,
        visited=visited | {resolved},
        verification_context=verification_context,
        **expected,
    )
    result = prior["result"]
    if result.get("terminal_status") != "NONTERMINAL":
        raise RecursiveCampaignError("terminal campaign output cannot be a source")
    attempt_root = _absolute_path(result.get("next_attempt_root"), "next_attempt_root")
    if require_attempt_absent and _lexists(attempt_root):
        raise RecursiveCampaignError("source-bound next attempt root already exists")
    advance_values = prior["capture"]["advance_values"]
    rows = advance_values["combined_rows"]
    report = advance_values["output_report"]
    bundle = advance_values["next_bundle"]
    rows, report, bundle = _validate_report_rows_bundle(
        rows=rows,
        report=report,
        bundle=bundle,
        dependencies=descriptor["dependencies"],
        checkpoint_root=attempt_root / "checkpoints",
    )
    if (
        bundle["integrity"]["bundle_digest"] != result["next_bundle_digest"]
        or bundle["plan"]["integrity"]["plan_digest"] != result["next_plan_digest"]
        or report["evidence_digest"] != result["output_evidence_digest"]
        or len(bundle["plan"]["tasks"]) != result["remaining_task_count"]
    ):
        raise RecursiveCampaignError("prior campaign output snapshots differ")
    return {
        "source_kind": "recursive_campaign_v1",
        "source_state_digest": result["advance_digest"],
        "source_verification": result,
        "rows": rows,
        "report": report,
        "bundle": bundle,
        "next_attempt_root": attempt_root,
        "checkpoint_root": attempt_root / "checkpoints",
    }


def _derive_source(
    source: Mapping[str, Any],
    *,
    require_attempt_absent: bool,
    depth: int = 0,
    visited: set[str] | None = None,
    verification_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verification_context = _require_verification_context(verification_context)
    descriptor = _normalize_source(source)
    visited = set() if visited is None else set(visited)
    source_key = _digest(descriptor)
    if descriptor["source_kind"] == "recursive_campaign_v1":
        # A cache hit must never bypass traversal policy.  The descriptor's
        # predecessor location is rechecked against this call's live ancestry
        # before any memoized semantic result is admitted.
        _campaign_source_traversal_guard(
            descriptor, depth=depth, visited=visited
        )
    cached = verification_context["source_cache"].get(source_key)
    if cached is not None:
        if require_attempt_absent and _lexists(cached["next_attempt_root"]):
            raise RecursiveCampaignError(
                "source-bound next attempt root already exists"
            )
        if require_attempt_absent:
            verification_context["source_attempt_absence"][source_key] = (
                cached["next_attempt_root"]
            )
        return cached
    if descriptor["source_kind"] == "recursive_successor_v1":
        derived = _derive_successor_source(
            descriptor, require_attempt_absent=require_attempt_absent
        )
    else:
        derived = _derive_campaign_source(
            descriptor,
            require_attempt_absent=require_attempt_absent,
            depth=depth,
            visited=visited,
            verification_context=verification_context,
        )
    derived["descriptor"] = descriptor
    derived["descriptor_digest"] = _digest(descriptor)
    verification_context["source_cache"][source_key] = derived
    if require_attempt_absent:
        verification_context["source_attempt_absence"][source_key] = (
            derived["next_attempt_root"]
        )
    if descriptor["source_kind"] == "recursive_successor_v1":
        verification_context["successor_freshness"].setdefault(
            source_key,
            {
                "descriptor": descriptor,
                "derived": derived,
            },
        )
    return derived


def _provenance_binding(
    derived: Mapping[str, Any],
    *,
    campaign_id: str,
    contract_digest: str,
) -> dict[str, Any]:
    bundle = derived["bundle"]
    task = _task_from_bundle(bundle)
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "campaign_contract": {
            "id": CAMPAIGN_CONTRACT_ID,
            "digest": contract_digest,
        },
        "campaign_id": campaign_id,
        "source": {
            "kind": derived["source_kind"],
            "state_digest": derived["source_state_digest"],
            "descriptor_digest": derived["descriptor_digest"],
        },
        "bundle": {
            "bundle_digest": bundle["integrity"]["bundle_digest"],
            "plan_digest": bundle["plan"]["integrity"]["plan_digest"],
            "task_count": len(bundle["plan"]["tasks"]),
        },
        "task": {
            "task_id": task["task_id"],
            "task_digest": task["task_digest"],
            "ordinal": 0,
            "cell": task["cell"],
        },
        "attempt": {
            "root": str(derived["next_attempt_root"]),
            "checkpoint_root": str(derived["checkpoint_root"]),
            "runtime_contract_id": "structural_hypothesis_single_task_runtime_v1",
            "runtime_contract_digest": _RUNTIME_CONTRACT_DIGEST,
        },
    }


def _inspect(
    source: Mapping[str, Any],
    campaign_contract_path,
    campaign_root,
    *,
    campaign_id: str,
    next_attempt_root=None,
    require_attempt_absent: bool,
    campaign_root_fresh: bool | None = None,
    allow_existing_campaign_root: bool = False,
    depth: int = 0,
    visited: set[str] | None = None,
    verification_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], bytes, Path]:
    campaign_id = _require_identifier(campaign_id, "campaign_id")
    _contract, contract_raw = _load_contract(campaign_contract_path)
    if campaign_root_fresh is None:
        campaign_root_fresh = require_attempt_absent
    root = _campaign_root(
        campaign_root,
        campaign_id,
        fresh=campaign_root_fresh,
        allow_existing=allow_existing_campaign_root,
    )
    derived = _derive_source(
        source,
        require_attempt_absent=require_attempt_absent,
        depth=depth,
        visited=visited,
        verification_context=verification_context,
    )
    if next_attempt_root is not None:
        supplied = _absolute_path(next_attempt_root, "next_attempt_root")
        if supplied != derived["next_attempt_root"]:
            raise RecursiveCampaignError("next attempt root differs from source binding")
    provenance = _provenance_binding(
        derived,
        campaign_id=campaign_id,
        contract_digest=_CAMPAIGN_CONTRACT_DIGEST,
    )
    provenance_digest = _digest(provenance)
    required_authorization_id = (
        "recursive-campaign-v1:" + provenance_digest.split(":", 1)[1]
    )
    task = _task_from_bundle(derived["bundle"])
    result = {
        "status": _INSPECTED,
        "source_kind": derived["source_kind"],
        "source_state_digest": derived["source_state_digest"],
        "bundle_digest": derived["bundle"]["integrity"]["bundle_digest"],
        "plan_digest": derived["bundle"]["plan"]["integrity"]["plan_digest"],
        "task_count": len(derived["bundle"]["plan"]["tasks"]),
        "task_id": task["task_id"],
        "task_digest": task["task_digest"],
        "next_attempt_root": str(derived["next_attempt_root"]),
        "checkpoint_root": str(derived["checkpoint_root"]),
        "provenance_binding": provenance,
        "provenance_binding_digest": provenance_digest,
        "required_authorization_id": required_authorization_id,
        "terminal_status": "NONTERMINAL",
    }
    return result, derived, contract_raw, root


def inspect_recursive_campaign(
    source,
    campaign_contract_path,
    campaign_root,
    *,
    campaign_id,
    next_attempt_root=None,
) -> dict[str, Any]:
    """Fully inspect one non-terminal source without creating state."""
    verification_context = _new_verification_context()
    result, _derived, _contract_raw, _root = _inspect(
        source,
        campaign_contract_path,
        campaign_root,
        campaign_id=campaign_id,
        next_attempt_root=next_attempt_root,
        require_attempt_absent=True,
        verification_context=verification_context,
    )
    _verify_context_freshness(verification_context)
    return result


def _source_commit(derived: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": SOURCE_COMMIT_SCHEMA_VERSION,
        "status": "VERIFIED_NONTERMINAL_RECURSIVE_CAMPAIGN_SOURCE",
        "source_kind": derived["source_kind"],
        "source_state_digest": derived["source_state_digest"],
        "descriptor_digest": derived["descriptor_digest"],
        "rows_digest": _digest(derived["rows"]),
        "report_body_digest": derived["report"]["audit"]["report_body_digest"],
        "report_audit_head": derived["report"]["audit"]["head"],
        "evidence_digest": derived["report"]["evidence_digest"],
        "bundle_digest": derived["bundle"]["integrity"]["bundle_digest"],
        "plan_digest": derived["bundle"]["plan"]["integrity"]["plan_digest"],
        "task_count": len(derived["bundle"]["plan"]["tasks"]),
        "next_attempt_root": str(derived["next_attempt_root"]),
        "checkpoint_root": str(derived["checkpoint_root"]),
    }
    return {**body, "integrity": {"algorithm": "sha256-canonical-json-v1", "source_commit_digest": _digest(body)}}


def authorize_recursive_campaign_task(
    source,
    campaign_contract_path,
    campaign_root,
    *,
    campaign_id,
    expected_source_state_digest,
    expected_bundle_digest,
    expected_plan_digest,
    task_id,
    expected_task_digest,
    expected_provenance_binding_digest,
    authorization_id,
    confirm_explicit_local_task_authorization,
) -> dict[str, Any]:
    """Prepare exactly one runtime-V1 attempt and publish its immutable lease."""
    if confirm_explicit_local_task_authorization is not True:
        raise RecursiveCampaignError("explicit local task authorization is required")
    _require_runtime_environment()
    verification_context = _new_verification_context()
    expected = {
        "source_state_digest": _require_digest(expected_source_state_digest, "expected_source_state_digest"),
        "bundle_digest": _require_digest(expected_bundle_digest, "expected_bundle_digest"),
        "plan_digest": _require_digest(expected_plan_digest, "expected_plan_digest"),
        "task_digest": _require_digest(expected_task_digest, "expected_task_digest"),
        "provenance_binding_digest": _require_digest(expected_provenance_binding_digest, "expected_provenance_binding_digest"),
    }
    if type(task_id) is not str or not _TASK_ID.fullmatch(task_id):
        raise RecursiveCampaignError("task_id is malformed")
    if type(authorization_id) is not str or not _AUTHORIZATION_ID.fullmatch(authorization_id):
        raise RecursiveCampaignError("authorization_id is malformed")
    inspected, derived, contract_raw, root = _inspect(
        source,
        campaign_contract_path,
        campaign_root,
        campaign_id=campaign_id,
        require_attempt_absent=False,
        campaign_root_fresh=True,
        allow_existing_campaign_root=True,
        verification_context=verification_context,
    )
    observed = {
        "source_state_digest": inspected["source_state_digest"],
        "bundle_digest": inspected["bundle_digest"],
        "plan_digest": inspected["plan_digest"],
        "task_digest": inspected["task_digest"],
        "provenance_binding_digest": inspected["provenance_binding_digest"],
    }
    if (
        observed != expected
        or task_id != inspected["task_id"]
        or authorization_id != inspected["required_authorization_id"]
    ):
        raise RecursiveCampaignError("independent authorization anchors differ")
    dependencies = _load_dependency_objects(derived["descriptor"]["dependencies"])
    if not _lexists(derived["next_attempt_root"]):
        if _lexists(root):
            raise RecursiveCampaignError(
                "campaign root exists without its source-bound runtime attempt"
            )
        try:
            prepared = _R_PREPARE(
                derived["report"],
                derived["bundle"],
                dependencies["hypothesis"],
                dependencies["executor"],
                dependencies["materializer"],
                dependencies["runtime"],
                dependencies["paths"]["base_manifest_path"],
                dependencies["paths"]["asset_root"],
                derived["next_attempt_root"],
                task_id=task_id,
                expected_bundle_digest=expected_bundle_digest,
                expected_plan_digest=expected_plan_digest,
                authorization_id=authorization_id,
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise RecursiveCampaignError("runtime V1 authorization failed") from error
        if prepared.get("status") != "AUTHORIZED":
            raise RecursiveCampaignError("runtime V1 did not return AUTHORIZED")
        authorization_digest = _require_digest(prepared.get("authorization_digest"), "authorization_digest")
        attempt_digest = _require_digest(prepared.get("attempt_digest"), "attempt_digest")
    else:
        # Recover only the exact deterministic AUTHORIZED attempt.  No prepare
        # delegate is called on this path, so a cross-tree crash cannot spend a
        # second authorization or mutate an existing runtime capsule.
        authorization = _read_json(
            derived["next_attempt_root"] / "authorization.json",
            "recoverable runtime authorization",
            exact_mode=0o600,
        )
        if authorization.get("authorization_id") != authorization_id:
            raise RecursiveCampaignError("recoverable authorization ID differs")
        authorization_digest = _require_digest(
            authorization.get("integrity", {}).get("authorization_digest"),
            "authorization_digest",
        )
        attempt_binding = _read_json(
            derived["next_attempt_root"] / "attempt.json",
            "recoverable runtime attempt",
            exact_mode=0o600,
        )
        attempt_digest = _require_digest(
            attempt_binding.get("integrity", {}).get("attempt_digest"),
            "attempt_digest",
        )
    try:
        runtime_verified = _R_VERIFY(
            derived["next_attempt_root"],
            dependencies["runtime"],
            dependencies["paths"]["base_manifest_path"],
            dependencies["paths"]["asset_root"],
            expected_authorization_digest=authorization_digest,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError("prepared runtime V1 state failed verification") from error
    if (
        runtime_verified.get("status") != "VERIFIED_AUTHORIZED_NOT_EXECUTED"
        or runtime_verified.get("attempt_digest") != attempt_digest
    ):
        raise RecursiveCampaignError("prepared runtime state is not AUTHORIZED-only")
    # Revalidate the source after runtime state creation.  The source verifier
    # permits the now-existing bound attempt, but all source snapshots must be
    # byte-for-byte equal to those used for authorization.
    after = _derive_source(
        source,
        require_attempt_absent=False,
        verification_context=verification_context,
    )
    for key in ("descriptor", "source_state_digest", "rows", "report", "bundle", "next_attempt_root", "checkpoint_root"):
        if after[key] != derived[key]:
            raise RecursiveCampaignError(f"source changed during authorization: {key}")
    source_commit = _source_commit(derived)
    lease_body = {
        "schema_version": LEASE_SCHEMA_VERSION,
        "status": "ONE_CALLBACK_START_LEASED_NOT_STARTED",
        "campaign_id": campaign_id,
        "provenance_binding_digest": inspected["provenance_binding_digest"],
        "authorization_id": authorization_id,
        "authorization_digest": authorization_digest,
        "attempt_digest": attempt_digest,
        "attempt_root": inspected["next_attempt_root"],
        "task_id": task_id,
        "task_digest": expected_task_digest,
        "max_callback_starts": 1,
        "auto_retry": False,
        "run_all": False,
        "nonclaims": list(_NONCLAIMS),
    }
    lease = {**lease_body, "integrity": {"algorithm": "sha256-canonical-json-v1", "lease_digest": _digest(lease_body)}}
    artifacts = {
        "campaign_contract_digest": "sha256:" + hashlib.sha256(contract_raw).hexdigest(),
        "source_descriptor_digest": derived["descriptor_digest"],
        "source_commit_digest": source_commit["integrity"]["source_commit_digest"],
        "source_rows_digest": _digest(derived["rows"]),
        "source_report_digest": _digest(derived["report"]),
        "source_bundle_digest": _digest(derived["bundle"]),
        "lease_digest": lease["integrity"]["lease_digest"],
    }
    campaign_body = {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "status": _AUTHORIZED,
        "campaign_id": campaign_id,
        "provenance_binding": inspected["provenance_binding"],
        "provenance_binding_digest": inspected["provenance_binding_digest"],
        "authorization_id": authorization_id,
        "authorization_digest": authorization_digest,
        "attempt_digest": attempt_digest,
        "artifacts": artifacts,
        "authorization_status": "AUTHORIZED",
        "execution_status": "NOT_EXECUTED",
        "terminal_status": "NONTERMINAL",
        "nonclaims": list(_NONCLAIMS),
    }
    campaign = {**campaign_body, "integrity": {"algorithm": "sha256-canonical-json-v1", "campaign_digest": _digest(campaign_body)}}
    # A recoverable partial root is possible only after the exact runtime
    # attempt exists.  Validate every present leaf before writing any missing
    # leaf so a late bad artifact cannot cause partial mutation on retry.
    if _lexists(root):
        _secure_directory(root, exact_mode=0o700)
        allowed_root = {
            _ROOT_LAYOUT["campaign_contract"],
            _ROOT_LAYOUT["source_directory"],
            _ROOT_LAYOUT["lease"],
            _ROOT_LAYOUT["campaign_commit"],
        }
        observed_root = set(os.listdir(root))
        if not observed_root.issubset(allowed_root):
            raise RecursiveCampaignError("partial authorized campaign has extra names")
        expected_root_values = {
            _ROOT_LAYOUT["campaign_contract"]: _parse_json(
                contract_raw, "campaign contract"
            ),
            _ROOT_LAYOUT["lease"]: lease,
            _ROOT_LAYOUT["campaign_commit"]: campaign,
        }
        for name in observed_root - {_ROOT_LAYOUT["source_directory"]}:
            if _read_bytes(
                root / name, f"partial campaign {name}", exact_mode=0o600
            ) != _canonical_file(expected_root_values[name]):
                raise RecursiveCampaignError(f"partial campaign leaf differs: {name}")
        if _ROOT_LAYOUT["source_directory"] in observed_root:
            staged_source = root / _ROOT_LAYOUT["source_directory"]
            _secure_directory(staged_source, exact_mode=0o700)
            observed_source = set(os.listdir(staged_source))
            if not observed_source.issubset(set(_SOURCE_LAYOUT.values())):
                raise RecursiveCampaignError("partial source snapshot has extra names")
            expected_source_values = {
                _SOURCE_LAYOUT["descriptor"]: derived["descriptor"],
                _SOURCE_LAYOUT["commit"]: source_commit,
                _SOURCE_LAYOUT["rows"]: derived["rows"],
                _SOURCE_LAYOUT["report"]: derived["report"],
                _SOURCE_LAYOUT["bundle"]: derived["bundle"],
            }
            for name in observed_source:
                if _read_bytes(
                    staged_source / name,
                    f"partial source {name}",
                    exact_mode=0o600,
                ) != _canonical_file(expected_source_values[name]):
                    raise RecursiveCampaignError(
                        f"partial source leaf differs: {name}"
                    )
        if _ROOT_LAYOUT["campaign_commit"] in observed_root:
            if observed_root != allowed_root:
                raise RecursiveCampaignError(
                    "campaign commit exists before complete authorized layout"
                )
            _require_names(
                root / _ROOT_LAYOUT["source_directory"],
                set(_SOURCE_LAYOUT.values()),
                "committed source snapshot",
            )
    _ensure_campaign_parent(root)
    if not _lexists(root):
        _mkdir_new(root)
    else:
        _secure_directory(root, exact_mode=0o700)
        allowed_partial = {
            _ROOT_LAYOUT["campaign_contract"],
            _ROOT_LAYOUT["source_directory"],
            _ROOT_LAYOUT["lease"],
            _ROOT_LAYOUT["campaign_commit"],
        }
        if not set(os.listdir(root)).issubset(allowed_partial):
            raise RecursiveCampaignError("partial authorized campaign has extra names")
        if _lexists(root / _ROOT_LAYOUT["campaign_commit"]):
            required_complete = allowed_partial
            if set(os.listdir(root)) != required_complete:
                raise RecursiveCampaignError(
                    "campaign commit exists before complete authorized layout"
                )
            committed_source = root / _ROOT_LAYOUT["source_directory"]
            _require_names(
                committed_source,
                set(_SOURCE_LAYOUT.values()),
                "committed source snapshot",
            )
    source_dir = root / _ROOT_LAYOUT["source_directory"]
    if not _lexists(source_dir):
        _mkdir_new(source_dir)
    else:
        _secure_directory(source_dir, exact_mode=0o700)
        if not set(os.listdir(source_dir)).issubset(set(_SOURCE_LAYOUT.values())):
            raise RecursiveCampaignError("partial source snapshot has extra names")
    root_identity = _directory_identity(root)
    source_identity = _directory_identity(source_dir)
    _publish_or_match(
        root / _ROOT_LAYOUT["campaign_contract"],
        _parse_json(contract_raw, "campaign contract"),
        expected_parent_identity=root_identity,
    )
    _publish_or_match(
        source_dir / _SOURCE_LAYOUT["descriptor"],
        derived["descriptor"],
        expected_parent_identity=source_identity,
    )
    _publish_or_match(
        source_dir / _SOURCE_LAYOUT["commit"],
        source_commit,
        expected_parent_identity=source_identity,
    )
    _publish_or_match(
        source_dir / _SOURCE_LAYOUT["rows"],
        derived["rows"],
        expected_parent_identity=source_identity,
    )
    _publish_or_match(
        source_dir / _SOURCE_LAYOUT["report"],
        derived["report"],
        expected_parent_identity=source_identity,
    )
    _publish_or_match(
        source_dir / _SOURCE_LAYOUT["bundle"],
        derived["bundle"],
        expected_parent_identity=source_identity,
    )
    _publish_or_match(
        root / _ROOT_LAYOUT["lease"],
        lease,
        expected_parent_identity=root_identity,
    )
    _require_names(source_dir, set(_SOURCE_LAYOUT.values()), "source snapshot")
    marker_path = root / _ROOT_LAYOUT["campaign_commit"]
    marker_preexisting = _lexists(marker_path)
    expected_noncommit_names = {
        _ROOT_LAYOUT["campaign_contract"],
        _ROOT_LAYOUT["source_directory"],
        _ROOT_LAYOUT["lease"],
    }
    expected_complete_names = expected_noncommit_names | {
        _ROOT_LAYOUT["campaign_commit"]
    }
    _require_names(
        root,
        expected_complete_names if marker_preexisting else expected_noncommit_names,
        "staged authorized campaign",
    )
    if _directory_identity(root) != root_identity or _directory_identity(source_dir) != source_identity:
        raise RecursiveCampaignError("authorized staging directory identity changed")
    final_source_a = _derive_source(
        source,
        require_attempt_absent=False,
        verification_context=verification_context,
    )
    for key in (
        "descriptor", "source_state_digest", "rows", "report", "bundle",
        "next_attempt_root", "checkpoint_root",
    ):
        if final_source_a[key] != derived[key]:
            raise RecursiveCampaignError(f"source changed before campaign commit: {key}")
    try:
        final_runtime = _R_VERIFY(
            derived["next_attempt_root"],
            dependencies["runtime"],
            dependencies["paths"]["base_manifest_path"],
            dependencies["paths"]["asset_root"],
            expected_authorization_digest=authorization_digest,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError("runtime changed before campaign commit") from error
    if (
        final_runtime.get("status") != "VERIFIED_AUTHORIZED_NOT_EXECUTED"
        or final_runtime.get("attempt_digest") != attempt_digest
    ):
        raise RecursiveCampaignError("runtime is not AUTHORIZED at campaign commit")
    final_source_b = _derive_source(
        source,
        require_attempt_absent=False,
        verification_context=verification_context,
    )
    for key in (
        "descriptor", "source_state_digest", "rows", "report", "bundle",
        "next_attempt_root", "checkpoint_root",
    ):
        if final_source_b[key] != derived[key]:
            raise RecursiveCampaignError(f"source changed at campaign commit: {key}")
    _verify_context_freshness(verification_context)
    if not marker_preexisting:
        root_before_raws = {
            _ROOT_LAYOUT["campaign_contract"]: contract_raw,
            _ROOT_LAYOUT["lease"]: _canonical_file(lease),
        }
        root_after_raws = {
            **root_before_raws,
            _ROOT_LAYOUT["campaign_commit"]: _canonical_file(campaign),
        }
        source_guard_raws = {
            _SOURCE_LAYOUT["descriptor"]: _canonical_file(derived["descriptor"]),
            _SOURCE_LAYOUT["commit"]: _canonical_file(source_commit),
            _SOURCE_LAYOUT["rows"]: _canonical_file(derived["rows"]),
            _SOURCE_LAYOUT["report"]: _canonical_file(derived["report"]),
            _SOURCE_LAYOUT["bundle"]: _canonical_file(derived["bundle"]),
        }
        marker_guards = (
            {
                "label": "authorized source snapshot",
                "path": source_dir,
                "identity": source_identity,
                "before_names": set(_SOURCE_LAYOUT.values()),
                "after_names": set(_SOURCE_LAYOUT.values()),
                "before_raws": source_guard_raws,
                "after_raws": source_guard_raws,
                "child_identities": {},
            },
            {
                "label": "authorized campaign root",
                "path": root,
                "identity": root_identity,
                "before_names": expected_noncommit_names,
                "after_names": expected_complete_names,
                "before_raws": root_before_raws,
                "after_raws": root_after_raws,
                "child_identities": {
                    _ROOT_LAYOUT["source_directory"]: source_identity,
                },
            },
        )
        _write_new(
            marker_path,
            campaign,
            expected_parent_identity=root_identity,
            publication_guards=marker_guards,
        )
    _require_names(root, expected_complete_names, "authorized campaign")
    committed_base = _validate_campaign_base(
        source,
        campaign_contract_path,
        root,
        campaign_id=campaign_id,
        expected_campaign_digest=campaign["integrity"]["campaign_digest"],
        expected_lease_digest=lease["integrity"]["lease_digest"],
        verification_context=verification_context,
    )
    if committed_base["capture"]["has_claim"] or committed_base["capture"]["has_advance"]:
        raise RecursiveCampaignError("authorized campaign unexpectedly crossed execution")
    try:
        committed_runtime = _R_VERIFY(
            derived["next_attempt_root"],
            dependencies["runtime"],
            dependencies["paths"]["base_manifest_path"],
            dependencies["paths"]["asset_root"],
            expected_authorization_digest=authorization_digest,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError(
            "runtime changed after authorized campaign commit"
        ) from error
    if (
        committed_runtime.get("status") != "VERIFIED_AUTHORIZED_NOT_EXECUTED"
        or committed_runtime.get("authorization_digest") != authorization_digest
        or committed_runtime.get("attempt_digest") != attempt_digest
    ):
        raise RecursiveCampaignError(
            "runtime is no longer AUTHORIZED after campaign commit"
        )
    _verify_context_freshness(verification_context)
    result = dict(inspected)
    result.update({
        "status": _AUTHORIZED,
        "campaign_root": str(root),
        "campaign_digest": campaign["integrity"]["campaign_digest"],
        "lease_digest": lease["integrity"]["lease_digest"],
        "authorization_digest": authorization_digest,
        "attempt_digest": attempt_digest,
        "authorization_status": "AUTHORIZED",
        "execution_status": "NOT_EXECUTED",
    })
    return result


def _integrity_digest(value: Mapping[str, Any], field: str, label: str) -> str:
    if type(value) is not dict:
        raise RecursiveCampaignError(f"{label} is not an object")
    integrity = value.get("integrity")
    if (
        type(integrity) is not dict
        or set(integrity) != {"algorithm", field}
        or integrity.get("algorithm") != "sha256-canonical-json-v1"
    ):
        raise RecursiveCampaignError(f"{label} integrity shape differs")
    body = {key: value[key] for key in value if key != "integrity"}
    digest = _require_digest(integrity.get(field), f"{label} {field}")
    if digest != _digest(body):
        raise RecursiveCampaignError(f"{label} integrity digest differs")
    return digest


def _capture_campaign(
    root: Path, campaign_id: str, *, allow_partial_advance: bool = False
) -> dict[str, Any]:
    _campaign_root(root, campaign_id, fresh=False)
    root_identity = _directory_identity(root)
    names = set(os.listdir(root))
    authorized_names = {
        _ROOT_LAYOUT["campaign_contract"],
        _ROOT_LAYOUT["source_directory"],
        _ROOT_LAYOUT["lease"],
        _ROOT_LAYOUT["campaign_commit"],
    }
    has_claim = _ROOT_LAYOUT["callback_start_claim"] in names
    has_advance = _ROOT_LAYOUT["advance_directory"] in names
    expected_names = set(authorized_names)
    if has_claim:
        expected_names.add(_ROOT_LAYOUT["callback_start_claim"])
    if has_advance:
        if not has_claim:
            raise RecursiveCampaignError("advanced campaign lacks callback claim")
        expected_names.add(_ROOT_LAYOUT["advance_directory"])
    if names != expected_names:
        raise RecursiveCampaignError("campaign root exact names differ")
    source_dir = root / _ROOT_LAYOUT["source_directory"]
    source_identity = _directory_identity(source_dir)
    _require_names(source_dir, set(_SOURCE_LAYOUT.values()), "campaign source")
    raws: dict[str, bytes] = {}
    values: dict[str, Any] = {}
    top_files = {
        "campaign_contract": root / _ROOT_LAYOUT["campaign_contract"],
        "lease": root / _ROOT_LAYOUT["lease"],
        "campaign_commit": root / _ROOT_LAYOUT["campaign_commit"],
    }
    if has_claim:
        top_files["callback_start_claim"] = root / _ROOT_LAYOUT["callback_start_claim"]
    for label, path in top_files.items():
        raws[label] = _read_bytes(path, label, exact_mode=0o600)
        values[label] = _parse_json(raws[label], label)
    for label, relative in _SOURCE_LAYOUT.items():
        key = "source_" + label
        path = source_dir / relative
        raws[key] = _read_bytes(path, key, exact_mode=0o600)
        values[key] = _parse_json(raws[key], key)
    terminal = None
    advance_partial = False
    advance_identity = None
    advance_values: dict[str, Any] = {}
    advance_raws: dict[str, bytes] = {}
    if has_advance:
        advance_dir = root / _ROOT_LAYOUT["advance_directory"]
        _secure_directory(advance_dir, exact_mode=0o700)
        advance_identity = _directory_identity(advance_dir)
        advance_names = set(os.listdir(advance_dir))
        nonterminal_names = set(_ADVANCE_LAYOUT.values())
        terminal_names = nonterminal_names - {_ADVANCE_LAYOUT["next_bundle"]}
        if advance_names == nonterminal_names:
            terminal = False
        elif advance_names == terminal_names:
            terminal = True
        elif (
            allow_partial_advance
            and _ADVANCE_LAYOUT["advance_commit"] not in advance_names
            and advance_names.issubset(nonterminal_names)
        ):
            advance_partial = True
        else:
            raise RecursiveCampaignError("advance exact names differ")
        for label, relative in _ADVANCE_LAYOUT.items():
            if (terminal and label == "next_bundle") or relative not in advance_names:
                continue
            raw = _read_bytes(advance_dir / relative, f"advance {label}", exact_mode=0o600)
            advance_raws[label] = raw
            advance_values[label] = _parse_json(raw, f"advance {label}")
    if _directory_identity(root) != root_identity:
        raise RecursiveCampaignError("campaign root identity changed during capture")
    if _directory_identity(source_dir) != source_identity:
        raise RecursiveCampaignError("campaign source identity changed during capture")
    _require_names(root, expected_names, "campaign root final capture")
    _require_names(source_dir, set(_SOURCE_LAYOUT.values()), "campaign source final capture")
    if has_advance:
        advance_dir = root / _ROOT_LAYOUT["advance_directory"]
        if _directory_identity(advance_dir) != advance_identity:
            raise RecursiveCampaignError("advance directory identity changed during capture")
        expected_advance_names = (
            set(_ADVANCE_LAYOUT.values())
            if terminal is False
            else set(_ADVANCE_LAYOUT.values()) - {_ADVANCE_LAYOUT["next_bundle"]}
            if terminal is True
            else set(os.listdir(advance_dir))
        )
        _require_names(advance_dir, expected_advance_names, "advance final capture")
    return {
        "root": root,
        "raws": raws,
        "values": values,
        "has_claim": has_claim,
        "has_advance": has_advance,
        "terminal": terminal,
        "advance_partial": advance_partial,
        "advance_raws": advance_raws,
        "advance_values": advance_values,
        "root_identity": root_identity,
        "source_identity": source_identity,
        "advance_identity": advance_identity,
    }


def _validate_campaign_base(
    source: Mapping[str, Any],
    campaign_contract_path,
    campaign_root,
    *,
    campaign_id: str,
    expected_campaign_digest: str,
    expected_lease_digest: str,
    depth: int = 0,
    visited: set[str] | None = None,
    allow_partial_advance: bool = False,
    verification_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_campaign_digest = _require_digest(expected_campaign_digest, "expected_campaign_digest")
    expected_lease_digest = _require_digest(expected_lease_digest, "expected_lease_digest")
    _contract, contract_raw = _load_contract(campaign_contract_path)
    root = _campaign_root(campaign_root, campaign_id, fresh=False)
    capture = _capture_campaign(
        root, campaign_id, allow_partial_advance=allow_partial_advance
    )
    values = capture["values"]
    if capture["raws"]["campaign_contract"] != contract_raw:
        raise RecursiveCampaignError("campaign contract snapshot differs")
    descriptor = _normalize_source(source)
    if values["source_descriptor"] != descriptor:
        raise RecursiveCampaignError("supplied source differs from immutable snapshot")
    campaign = values["campaign_commit"]
    lease = values["lease"]
    _exact_keys(
        campaign,
        {
            "schema_version", "status", "campaign_id",
            "provenance_binding", "provenance_binding_digest",
            "authorization_id", "authorization_digest", "attempt_digest",
            "artifacts", "authorization_status", "execution_status",
            "terminal_status", "nonclaims", "integrity",
        },
        "campaign marker",
    )
    _exact_keys(
        lease,
        {
            "schema_version", "status", "campaign_id",
            "provenance_binding_digest", "authorization_id",
            "authorization_digest", "attempt_digest", "attempt_root",
            "task_id", "task_digest", "max_callback_starts",
            "auto_retry", "run_all", "nonclaims", "integrity",
        },
        "campaign lease",
    )
    campaign_digest = _integrity_digest(campaign, "campaign_digest", "campaign marker")
    lease_digest = _integrity_digest(lease, "lease_digest", "campaign lease")
    if campaign_digest != expected_campaign_digest or lease_digest != expected_lease_digest:
        raise RecursiveCampaignError("independent campaign or lease digest differs")
    if (
        campaign.get("schema_version") != CAPSULE_SCHEMA_VERSION
        or campaign.get("status") != _AUTHORIZED
        or campaign.get("campaign_id") != campaign_id
        or campaign.get("authorization_status") != "AUTHORIZED"
        or campaign.get("execution_status") != "NOT_EXECUTED"
        or campaign.get("terminal_status") != "NONTERMINAL"
        or campaign.get("nonclaims") != _NONCLAIMS
        or lease.get("schema_version") != LEASE_SCHEMA_VERSION
        or lease.get("status") != "ONE_CALLBACK_START_LEASED_NOT_STARTED"
        or lease.get("campaign_id") != campaign_id
        or lease.get("max_callback_starts") != 1
        or lease.get("auto_retry") is not False
        or lease.get("run_all") is not False
        or lease.get("nonclaims") != _NONCLAIMS
    ):
        raise RecursiveCampaignError("campaign marker or lease policy differs")
    derived = _derive_source(
        descriptor,
        require_attempt_absent=False,
        depth=depth,
        visited=visited,
        verification_context=verification_context,
    )
    expected_source_commit = _source_commit(derived)
    if (
        values["source_commit"] != expected_source_commit
        or values["source_rows"] != derived["rows"]
        or values["source_report"] != derived["report"]
        or values["source_bundle"] != derived["bundle"]
    ):
        raise RecursiveCampaignError("campaign source snapshots differ")
    provenance = _provenance_binding(
        derived,
        campaign_id=campaign_id,
        contract_digest=_CAMPAIGN_CONTRACT_DIGEST,
    )
    provenance_digest = _digest(provenance)
    authorization_id = "recursive-campaign-v1:" + provenance_digest.split(":", 1)[1]
    if (
        campaign.get("provenance_binding") != provenance
        or campaign.get("provenance_binding_digest") != provenance_digest
        or campaign.get("authorization_id") != authorization_id
        or lease.get("provenance_binding_digest") != provenance_digest
        or lease.get("authorization_id") != authorization_id
        or lease.get("authorization_digest") != campaign.get("authorization_digest")
        or lease.get("attempt_digest") != campaign.get("attempt_digest")
        or lease.get("attempt_root") != str(derived["next_attempt_root"])
        or lease.get("task_id") != provenance["task"]["task_id"]
        or lease.get("task_digest") != provenance["task"]["task_digest"]
    ):
        raise RecursiveCampaignError("campaign provenance/lease binding differs")
    artifacts = campaign.get("artifacts")
    expected_artifacts = {
        "campaign_contract_digest": "sha256:" + hashlib.sha256(contract_raw).hexdigest(),
        "source_descriptor_digest": derived["descriptor_digest"],
        "source_commit_digest": expected_source_commit["integrity"]["source_commit_digest"],
        "source_rows_digest": _digest(derived["rows"]),
        "source_report_digest": _digest(derived["report"]),
        "source_bundle_digest": _digest(derived["bundle"]),
        "lease_digest": lease_digest,
    }
    if artifacts != expected_artifacts:
        raise RecursiveCampaignError("campaign artifact map differs")
    return {
        "capture": capture,
        "derived": derived,
        "campaign": campaign,
        "campaign_digest": campaign_digest,
        "lease": lease,
        "lease_digest": lease_digest,
        "provenance_binding": provenance,
        "provenance_binding_digest": provenance_digest,
        "authorization_id": authorization_id,
        "authorization_digest": _require_digest(campaign.get("authorization_digest"), "campaign authorization_digest"),
        "attempt_digest": _require_digest(campaign.get("attempt_digest"), "campaign attempt_digest"),
    }


def _validate_claim(base: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    capture = base["capture"]
    if not capture["has_claim"]:
        raise RecursiveCampaignError("callback start has not been claimed")
    claim = capture["values"]["callback_start_claim"]
    claim_digest = _integrity_digest(
        claim, "callback_start_claim_digest", "callback start claim"
    )
    expected_body = {
        "schema_version": CALLBACK_START_CLAIM_SCHEMA_VERSION,
        "status": "CALLBACK_START_CLAIMED_HARD_STOP_NO_REENTRY",
        "campaign_digest": base["campaign_digest"],
        "lease_digest": base["lease_digest"],
        "provenance_binding_digest": base["provenance_binding_digest"],
        "authorization_digest": base["authorization_digest"],
        "attempt_digest": base["attempt_digest"],
        "attempt_root": str(base["derived"]["next_attempt_root"]),
        "task_id": base["provenance_binding"]["task"]["task_id"],
        "callback_start_ordinal": 1,
        "max_callback_starts": 1,
        "auto_retry": False,
    }
    if {key: claim[key] for key in claim if key != "integrity"} != expected_body:
        raise RecursiveCampaignError("callback start claim binding differs")
    return claim, claim_digest


def execute_recursive_campaign_task(
    source,
    campaign_contract_path,
    campaign_root,
    runtime_contract_path,
    *,
    expected_campaign_digest,
    expected_lease_digest,
    expected_provenance_binding_digest,
    expected_authorization_digest,
    expected_attempt_digest,
    confirm_real_local_execution,
) -> dict[str, Any]:
    """Spend one callback-start claim, delegate once to runtime V1, and stop."""
    if confirm_real_local_execution is not True:
        raise RecursiveCampaignError("explicit real local execution confirmation is required")
    _require_runtime_environment()
    verification_context = _new_verification_context()
    root = _absolute_path(campaign_root, "campaign_root")
    campaign_id = root.name
    base = _validate_campaign_base(
        source,
        campaign_contract_path,
        root,
        campaign_id=campaign_id,
        expected_campaign_digest=expected_campaign_digest,
        expected_lease_digest=expected_lease_digest,
        verification_context=verification_context,
    )
    expected_provenance_binding_digest = _require_digest(
        expected_provenance_binding_digest, "expected_provenance_binding_digest"
    )
    expected_authorization_digest = _require_digest(
        expected_authorization_digest, "expected_authorization_digest"
    )
    expected_attempt_digest = _require_digest(expected_attempt_digest, "expected_attempt_digest")
    if (
        base["provenance_binding_digest"] != expected_provenance_binding_digest
        or base["authorization_digest"] != expected_authorization_digest
        or base["attempt_digest"] != expected_attempt_digest
    ):
        raise RecursiveCampaignError("independent execution anchors differ")
    if base["capture"]["has_claim"] or base["capture"]["has_advance"]:
        raise RecursiveCampaignError("callback lease was already claimed")
    dependencies = _load_dependency_objects(base["derived"]["descriptor"]["dependencies"])
    supplied_runtime_path = _absolute_path(runtime_contract_path, "runtime_contract_path")
    if supplied_runtime_path != dependencies["paths"]["runtime_contract_path"]:
        raise RecursiveCampaignError("runtime contract path differs from source dependency")
    # Ensure the runtime capsule is still exactly AUTHORIZED before spending the
    # campaign lease.  Environment checking above deliberately precedes claim.
    try:
        verified = _R_VERIFY(
            base["derived"]["next_attempt_root"],
            dependencies["runtime"],
            dependencies["paths"]["base_manifest_path"],
            dependencies["paths"]["asset_root"],
            expected_authorization_digest=base["authorization_digest"],
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError("authorized runtime verification failed") from error
    if (
        verified.get("status") != "VERIFIED_AUTHORIZED_NOT_EXECUTED"
        or verified.get("attempt_digest") != base["attempt_digest"]
    ):
        raise RecursiveCampaignError("runtime capsule is not fresh AUTHORIZED state")
    final_base = _validate_campaign_base(
        source,
        campaign_contract_path,
        root,
        campaign_id=campaign_id,
        expected_campaign_digest=expected_campaign_digest,
        expected_lease_digest=expected_lease_digest,
        verification_context=verification_context,
    )
    for key in (
        "campaign_digest", "lease_digest", "provenance_binding",
        "provenance_binding_digest", "authorization_id",
        "authorization_digest", "attempt_digest",
    ):
        if final_base[key] != base[key]:
            raise RecursiveCampaignError(f"campaign generation changed before claim: {key}")
    if (
        final_base["capture"]["raws"] != base["capture"]["raws"]
        or final_base["derived"]["descriptor"] != base["derived"]["descriptor"]
        or final_base["derived"]["rows"] != base["derived"]["rows"]
        or final_base["derived"]["report"] != base["derived"]["report"]
        or final_base["derived"]["bundle"] != base["derived"]["bundle"]
    ):
        raise RecursiveCampaignError("campaign source changed before claim")
    base = final_base
    claim_body = {
        "schema_version": CALLBACK_START_CLAIM_SCHEMA_VERSION,
        "status": "CALLBACK_START_CLAIMED_HARD_STOP_NO_REENTRY",
        "campaign_digest": base["campaign_digest"],
        "lease_digest": base["lease_digest"],
        "provenance_binding_digest": base["provenance_binding_digest"],
        "authorization_digest": base["authorization_digest"],
        "attempt_digest": base["attempt_digest"],
        "attempt_root": str(base["derived"]["next_attempt_root"]),
        "task_id": base["provenance_binding"]["task"]["task_id"],
        "callback_start_ordinal": 1,
        "max_callback_starts": 1,
        "auto_retry": False,
    }
    claim = {
        **claim_body,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "callback_start_claim_digest": _digest(claim_body),
        },
    }
    # Recheck the runtime after the final campaign/source capture.  This is the
    # last external read before the durable callback-start claim, so a
    # PREFLIGHT/RUNNING/COMPLETED transition cannot consume the campaign lease
    # on the strength of the earlier verification.
    try:
        claim_ready_runtime = _R_VERIFY(
            base["derived"]["next_attempt_root"],
            dependencies["runtime"],
            dependencies["paths"]["base_manifest_path"],
            dependencies["paths"]["asset_root"],
            expected_authorization_digest=base["authorization_digest"],
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError(
            "runtime changed immediately before callback claim"
        ) from error
    if (
        claim_ready_runtime.get("status") != "VERIFIED_AUTHORIZED_NOT_EXECUTED"
        or claim_ready_runtime.get("authorization_digest")
        != base["authorization_digest"]
        or claim_ready_runtime.get("attempt_digest") != base["attempt_digest"]
    ):
        raise RecursiveCampaignError(
            "runtime is not exact AUTHORIZED state at callback claim"
        )
    _verify_context_freshness(verification_context)
    claim_root_before_names = {
        _ROOT_LAYOUT["campaign_contract"],
        _ROOT_LAYOUT["source_directory"],
        _ROOT_LAYOUT["lease"],
        _ROOT_LAYOUT["campaign_commit"],
    }
    claim_root_after_names = claim_root_before_names | {
        _ROOT_LAYOUT["callback_start_claim"]
    }
    claim_root_before_raws = {
        _ROOT_LAYOUT["campaign_contract"]: base["capture"]["raws"][
            "campaign_contract"
        ],
        _ROOT_LAYOUT["lease"]: base["capture"]["raws"]["lease"],
        _ROOT_LAYOUT["campaign_commit"]: base["capture"]["raws"][
            "campaign_commit"
        ],
    }
    claim_root_after_raws = {
        **claim_root_before_raws,
        _ROOT_LAYOUT["callback_start_claim"]: _canonical_file(claim),
    }
    claim_source_raws = {
        relative: base["capture"]["raws"]["source_" + label]
        for label, relative in _SOURCE_LAYOUT.items()
    }
    claim_publication_guards = (
        {
            "label": "callback-claim source snapshot",
            "path": root / _ROOT_LAYOUT["source_directory"],
            "identity": base["capture"]["source_identity"],
            "before_names": set(_SOURCE_LAYOUT.values()),
            "after_names": set(_SOURCE_LAYOUT.values()),
            "before_raws": claim_source_raws,
            "after_raws": claim_source_raws,
            "child_identities": {},
        },
        {
            "label": "callback-claim campaign root",
            "path": root,
            "identity": base["capture"]["root_identity"],
            "before_names": claim_root_before_names,
            "after_names": claim_root_after_names,
            "before_raws": claim_root_before_raws,
            "after_raws": claim_root_after_raws,
            "child_identities": {
                _ROOT_LAYOUT["source_directory"]: base["capture"][
                    "source_identity"
                ],
            },
        },
    )
    _write_new(
        root / _ROOT_LAYOUT["callback_start_claim"],
        claim,
        expected_parent_identity=base["capture"]["root_identity"],
        publication_guards=claim_publication_guards,
    )
    # The publication primitive has post-read the claim and the exact held
    # campaign/source generation.  There is intentionally no further operation
    # between that durable guarded publication and the captured runtime delegate.
    try:
        executed = _R_EXECUTE(
            base["derived"]["next_attempt_root"],
            dependencies["runtime"],
            expected_authorization_digest=base["authorization_digest"],
        )
    except Exception as error:
        raise RecursiveCampaignError(
            "runtime execution did not return; callback claim remains spent"
        ) from error
    if executed.get("status") != "EXECUTED_RECEIPT_WRITTEN":
        raise RecursiveCampaignError("runtime execution did not durably complete")
    receipt_digest = _require_digest(executed.get("receipt_digest"), "receipt_digest")
    journal_head = _require_digest(executed.get("journal_head_digest"), "journal_head_digest")
    if (
        executed.get("authorization_digest") != base["authorization_digest"]
        or executed.get("attempt_digest") != base["attempt_digest"]
    ):
        raise RecursiveCampaignError("runtime execution result binding differs")
    try:
        completed = _R_VERIFY(
            base["derived"]["next_attempt_root"],
            dependencies["runtime"],
            dependencies["paths"]["base_manifest_path"],
            dependencies["paths"]["asset_root"],
            expected_authorization_digest=base["authorization_digest"],
            expected_receipt_digest=receipt_digest,
            expected_journal_head_digest=journal_head,
            expected_attempt_digest=base["attempt_digest"],
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError(
            "runtime execution return failed full completed verification"
        ) from error
    if (
        completed.get("status") != "VERIFIED_COMPLETED"
        or completed.get("authorization_digest") != base["authorization_digest"]
        or completed.get("receipt_digest") != receipt_digest
        or completed.get("journal_head_digest") != journal_head
        or completed.get("attempt_digest") != base["attempt_digest"]
    ):
        raise RecursiveCampaignError("completed runtime verification anchors differ")
    receipt_value = _read_json(
        base["derived"]["next_attempt_root"] / "receipt.json",
        "post-execution receipt",
        exact_mode=0o600,
    )
    if (
        _integrity_digest(
            receipt_value, "receipt_digest", "post-execution receipt"
        )
        != receipt_digest
    ):
        raise RecursiveCampaignError("post-execution receipt digest differs")
    successful = (
        receipt_value.get("status") == "COMPLETED"
        and receipt_value.get("summary")
        == {"authorized": 1, "succeeded": 1, "failed": 0}
        and type(receipt_value.get("results")) is list
        and len(receipt_value["results"]) == 1
        and receipt_value["results"][0].get("status") == "SUCCEEDED"
    )
    post_base = _validate_campaign_base(
        source,
        campaign_contract_path,
        root,
        campaign_id=campaign_id,
        expected_campaign_digest=expected_campaign_digest,
        expected_lease_digest=expected_lease_digest,
        verification_context=verification_context,
    )
    for key in (
        "campaign_digest", "lease_digest", "provenance_binding_digest",
        "authorization_digest", "attempt_digest",
    ):
        if post_base[key] != base[key]:
            raise RecursiveCampaignError(f"campaign generation changed after callback: {key}")
    for key, raw in base["capture"]["raws"].items():
        if post_base["capture"]["raws"].get(key) != raw:
            raise RecursiveCampaignError(f"campaign artifact changed after callback: {key}")
    if (
        post_base["capture"]["root_identity"]
        != base["capture"]["root_identity"]
        or post_base["capture"]["has_advance"]
    ):
        raise RecursiveCampaignError("campaign root identity/phase changed after callback")
    post_claim, post_claim_digest = _validate_claim(post_base)
    if (
        post_claim != claim
        or post_claim_digest
        != claim["integrity"]["callback_start_claim_digest"]
    ):
        raise RecursiveCampaignError("callback claim changed during execution")
    try:
        return_ready_runtime = _R_VERIFY(
            base["derived"]["next_attempt_root"],
            dependencies["runtime"],
            dependencies["paths"]["base_manifest_path"],
            dependencies["paths"]["asset_root"],
            expected_authorization_digest=base["authorization_digest"],
            expected_receipt_digest=receipt_digest,
            expected_journal_head_digest=journal_head,
            expected_attempt_digest=base["attempt_digest"],
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError(
            "runtime changed at execution return boundary"
        ) from error
    if (
        return_ready_runtime.get("status") != "VERIFIED_COMPLETED"
        or return_ready_runtime.get("authorization_digest")
        != base["authorization_digest"]
        or return_ready_runtime.get("receipt_digest") != receipt_digest
        or return_ready_runtime.get("journal_head_digest") != journal_head
        or return_ready_runtime.get("attempt_digest") != base["attempt_digest"]
    ):
        raise RecursiveCampaignError(
            "completed runtime changed at execution return boundary"
        )
    _verify_context_freshness(verification_context)
    return {
        "status": _CALLBACK_COMPLETED,
        "campaign_root": str(root),
        "campaign_digest": base["campaign_digest"],
        "lease_digest": base["lease_digest"],
        "callback_start_claim_digest": claim["integrity"]["callback_start_claim_digest"],
        "provenance_binding_digest": base["provenance_binding_digest"],
        "authorization_digest": base["authorization_digest"],
        "attempt_digest": base["attempt_digest"],
        "receipt_digest": receipt_digest,
        "journal_head_digest": journal_head,
        "task_id": base["provenance_binding"]["task"]["task_id"],
        "execution_status": (
            "COMPLETED_SUCCESS_AWAITING_ADVANCE"
            if successful
            else "COMPLETED_FAILED_EVIDENCE_NEUTRAL_HARD_STOP"
        ),
    }


def _future_attempt_candidate(
    value: Any,
    runtime_contract: Mapping[str, Any],
    *,
    require_absent: bool,
) -> Path:
    path = _absolute_path(value, "next_attempt_root")
    policy = runtime_contract.get("attempt_root_policy")
    if type(policy) is not dict or policy.get("kind") != "xdg-state-home-relative-v1":
        raise RecursiveCampaignError("runtime attempt-root policy differs")
    relative = Path(policy.get("relative_prefix", ""))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RecursiveCampaignError("runtime attempt prefix is unsafe")
    prefix = _state_home() / relative
    if path.parent != prefix:
        raise RecursiveCampaignError("next attempt must be a direct frozen-prefix child")
    if require_absent and _lexists(path):
        raise RecursiveCampaignError("next attempt candidate already exists")
    return path


def _completed_runtime_anchors(base: Mapping[str, Any]) -> dict[str, str]:
    attempt_root = base["derived"]["next_attempt_root"]
    receipt = _read_json(
        attempt_root / "receipt.json", "completed runtime receipt", exact_mode=0o600
    )
    completed = _read_json(
        attempt_root / "journal" / "0002_COMPLETED.json",
        "completed runtime journal event",
        exact_mode=0o600,
    )
    return {
        "receipt_digest": _require_digest(
            receipt.get("integrity", {}).get("receipt_digest"), "receipt_digest"
        ),
        "journal_head_digest": _require_digest(
            completed.get("integrity", {}).get("event_digest"),
            "journal_head_digest",
        ),
    }


def _preview_completed_advance(
    base: Mapping[str, Any],
    *,
    next_attempt_root=None,
    require_next_absent: bool,
) -> dict[str, Any]:
    dependencies = _load_dependency_objects(base["derived"]["descriptor"]["dependencies"])
    attempt_root = base["derived"]["next_attempt_root"]
    try:
        captured_objects, captured_raws = _P_ATTEMPT_SNAPSHOTS(attempt_root)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError("cannot capture completed runtime generation") from error
    anchors = {
        "receipt_digest": _require_digest(
            captured_objects.get("execution_receipt", {})
            .get("integrity", {})
            .get("receipt_digest"),
            "receipt_digest",
        ),
        "journal_head_digest": _require_digest(
            captured_objects.get("execution_completed_event", {})
            .get("integrity", {})
            .get("event_digest"),
            "journal_head_digest",
        ),
    }
    try:
        _P_VALIDATE_CAPTURED_RUNTIME(
            captured_objects,
            dependencies["runtime"],
            dependencies["paths"]["base_manifest_path"],
            dependencies["paths"]["asset_root"],
            attempt_root,
            expected_plan_digest=base["derived"]["bundle"]["plan"]["integrity"]["plan_digest"],
            expected_authorization_digest=base["authorization_digest"],
            expected_execution_receipt_digest=anchors["receipt_digest"],
            expected_execution_journal_head_digest=anchors["journal_head_digest"],
            expected_execution_attempt_digest=base["attempt_digest"],
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError("captured runtime generation failed verification") from error
    if (
        captured_objects.get("source_report") != base["derived"]["report"]
        or captured_objects.get("execution_bundle") != base["derived"]["bundle"]
        or captured_objects.get("execution_hypothesis_contract") != dependencies["hypothesis"]
        or captured_objects.get("execution_executor_contract") != dependencies["executor"]
        or captured_objects.get("execution_materializer_contract") != dependencies["materializer"]
    ):
        raise RecursiveCampaignError("captured runtime inputs differ from campaign source")
    try:
        runtime_verified = _R_VERIFY(
            attempt_root,
            dependencies["runtime"],
            dependencies["paths"]["base_manifest_path"],
            dependencies["paths"]["asset_root"],
            expected_authorization_digest=base["authorization_digest"],
            expected_receipt_digest=anchors["receipt_digest"],
            expected_journal_head_digest=anchors["journal_head_digest"],
            expected_attempt_digest=base["attempt_digest"],
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError("completed runtime capsule failed verification") from error
    if (
        runtime_verified.get("status") != "VERIFIED_COMPLETED"
        or runtime_verified.get("receipt_digest") != anchors["receipt_digest"]
        or runtime_verified.get("journal_head_digest") != anchors["journal_head_digest"]
        or runtime_verified.get("attempt_digest") != base["attempt_digest"]
        or runtime_verified.get("authorization_digest") != base["authorization_digest"]
    ):
        raise RecursiveCampaignError("completed runtime anchors differ")
    try:
        second_objects, second_raws = _P_ATTEMPT_SNAPSHOTS(attempt_root)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError("cannot recapture completed runtime generation") from error
    if second_raws != captured_raws or second_objects != captured_objects:
        raise RecursiveCampaignError("completed runtime generation changed")
    receipt = captured_objects["execution_receipt"]
    authorization = captured_objects["execution_authorization"]
    results = receipt.get("results")
    if (
        type(results) is not list
        or len(results) != 1
        or type(results[0]) is not dict
        or results[0].get("status") != "SUCCEEDED"
        or type(results[0].get("evidence_row")) is not dict
        or results[0].get("task_id")
        != base["provenance_binding"]["task"]["task_id"]
    ):
        raise RecursiveCampaignError(
            "only one successful completed runtime receipt may be advanced"
        )
    try:
        reingested = _E_REINGEST(
            base["derived"]["rows"],
            [receipt],
            dependencies["hypothesis"],
            base["derived"]["report"],
            plan=base["derived"]["bundle"]["plan"],
            authorization=authorization,
            executor_contract=dependencies["executor"],
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError("public execution reingestion failed") from error
    output_report = _clone(reingested.get("report"), "output report")
    reingestion = _clone(
        reingested.get("reingestion_receipt"), "reingestion receipt"
    )
    combined_rows = _clone(
        list(base["derived"]["rows"]) + [results[0]["evidence_row"]],
        "combined rows",
    )
    try:
        reingestion_verified = _E_VERIFY_REINGESTION(
            reingestion,
            source_report=base["derived"]["report"],
            base_rows=base["derived"]["rows"],
            plan=base["derived"]["bundle"]["plan"],
            authorization=authorization,
            receipts=[receipt],
            output_report=output_report,
            hypothesis_contract=dependencies["hypothesis"],
            executor_contract=dependencies["executor"],
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError("public reingestion verification failed") from error
    source_pending = base["derived"]["report"].get("pending_evidence")
    pending = output_report.get("pending_evidence")
    if (
        reingestion_verified is not True
        or not verify_report_integrity(output_report)
        or output_report.get("evidence_digest") != _digest(combined_rows)
        or type(source_pending) is not list
        or type(pending) is not list
        or len(pending) != len(source_pending) - 1
        or pending != source_pending[1:]
        or reingestion.get("accepted_successful_rows") != 1
        or reingestion.get("ignored_failed_attempts") != 0
    ):
        raise RecursiveCampaignError("one-row report advance invariant differs")
    output_evidence_digest = _require_digest(
        output_report.get("evidence_digest"), "output_evidence_digest"
    )
    output_report_body_digest = _require_digest(
        output_report.get("audit", {}).get("report_body_digest"),
        "output_report_body_digest",
    )
    output_audit_head = _require_digest(
        output_report.get("audit", {}).get("head"), "output_audit_head"
    )
    reingestion_digest = _integrity_digest(
        reingestion, "reingestion_digest", "reingestion receipt"
    )
    terminal = len(pending) == 0
    pending_digest = None if terminal else _digest(pending)
    projection_digest = None
    next_bundle = None
    next_bundle_digest = None
    next_plan_digest = None
    next_path = None
    if terminal:
        if next_attempt_root is not None:
            raise RecursiveCampaignError("terminal preview forbids next_attempt_root")
    else:
        if next_attempt_root is None:
            raise RecursiveCampaignError(
                "nonterminal completed preview requires next_attempt_root"
            )
        try:
            first_projection = {
                "profile": pending[0]["profile"],
                "domain": pending[0]["domain"],
                "line": dependencies["executor"]["execution_scope"]["line"],
                "seed": pending[0]["seed"],
                "d": pending[0]["d"],
                "N": pending[0]["N"],
                "n0": pending[0]["n0"],
            }
        except (KeyError, IndexError, TypeError) as error:
            raise RecursiveCampaignError("next pending projection is malformed") from error
        projection_digest = _digest(first_projection)
        try:
            identity_plan = _E_BUILD_PLAN(
                output_report,
                dependencies["hypothesis"],
                dependencies["executor"],
                None,
            )
            identity_tasks = identity_plan["tasks"]
            next_task_id = identity_tasks[0]["task_id"]
        except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
            raise RecursiveCampaignError(
                "public next-task identity derivation failed"
            ) from error
        if (
            identity_plan.get("status") != "AWAITING_TASK_TEMPLATE"
            or identity_plan.get("proposal_count") != len(pending)
            or type(identity_tasks) is not list
            or len(identity_tasks) != len(pending)
            or type(next_task_id) is not str
            or not _TASK_ID.fullmatch(next_task_id)
            or identity_tasks[0].get("cell") != first_projection
        ):
            raise RecursiveCampaignError(
                "public next-task identity plan differs from pending report"
            )
        expected_basename = "recursive-" + next_task_id.split(":", 1)[1]
        next_path = _future_attempt_candidate(
            next_attempt_root,
            dependencies["runtime"],
            require_absent=require_next_absent,
        )
        if next_path.name != expected_basename:
            raise RecursiveCampaignError(
                "next attempt basename differs from next task ID"
            )
        try:
            next_bundle = _M_MATERIALIZE(
                output_report,
                dependencies["hypothesis"],
                dependencies["executor"],
                dependencies["materializer"],
                dependencies["paths"]["base_manifest_path"],
                dependencies["paths"]["asset_root"],
                next_path / "checkpoints",
            )
            strong = _M_VERIFY(
                next_bundle,
                output_report,
                dependencies["hypothesis"],
                dependencies["executor"],
                dependencies["materializer"],
                dependencies["paths"]["base_manifest_path"],
                dependencies["paths"]["asset_root"],
                next_path / "checkpoints",
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise RecursiveCampaignError("public next-task materialization failed") from error
        if (
            strong is not True
            or next_bundle.get("task_count") != len(pending)
            or next_bundle.get("plan", {}).get("proposal_count") != len(pending)
            or len(next_bundle.get("plan", {}).get("tasks", [])) != len(pending)
            or next_bundle["plan"]["tasks"][0].get("cell") != first_projection
            or next_bundle["plan"]["tasks"][0].get("task_id") != next_task_id
        ):
            raise RecursiveCampaignError("next task bundle does not match pending report")
        next_bundle_digest = _require_digest(
            next_bundle.get("integrity", {}).get("bundle_digest"),
            "next_bundle_digest",
        )
        next_plan_digest = _require_digest(
            next_bundle.get("plan", {}).get("integrity", {}).get("plan_digest"),
            "next_plan_digest",
        )
    try:
        final_objects, final_raws = _P_ATTEMPT_SNAPSHOTS(attempt_root)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError("cannot final-recapture runtime generation") from error
    if final_raws != captured_raws or final_objects != captured_objects:
        raise RecursiveCampaignError("runtime generation changed during preview")
    try:
        _P_VALIDATE_CAPTURED_RUNTIME(
            final_objects,
            dependencies["runtime"],
            dependencies["paths"]["base_manifest_path"],
            dependencies["paths"]["asset_root"],
            attempt_root,
            expected_plan_digest=base["derived"]["bundle"]["plan"]["integrity"]["plan_digest"],
            expected_authorization_digest=base["authorization_digest"],
            expected_execution_receipt_digest=anchors["receipt_digest"],
            expected_execution_journal_head_digest=anchors["journal_head_digest"],
            expected_execution_attempt_digest=base["attempt_digest"],
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError("final captured runtime generation failed verification") from error
    return {
        **anchors,
        "receipt": receipt,
        "authorization": authorization,
        "combined_rows": combined_rows,
        "output_report": output_report,
        "reingestion_receipt": reingestion,
        "reingestion_digest": reingestion_digest,
        "output_evidence_digest": output_evidence_digest,
        "output_report_body_digest": output_report_body_digest,
        "output_audit_head": output_audit_head,
        "next_pending_evidence_digest": pending_digest,
        "next_first_pending_projection_digest": projection_digest,
        "next_bundle": next_bundle,
        "next_bundle_digest": next_bundle_digest,
        "next_plan_digest": next_plan_digest,
        "remaining_task_count": len(pending),
        "terminal_status": "TERMINAL" if terminal else "NONTERMINAL",
        "next_attempt_root": None if terminal else str(next_path),
        "runtime_raws": captured_raws,
    }


def _advanced_result(
    base: Mapping[str, Any],
    claim_digest: str,
    preview: Mapping[str, Any],
    advance_digest: str,
    *,
    verified: bool,
) -> dict[str, Any]:
    status = (
        _ADVANCED_TERMINAL
        if preview["terminal_status"] == "TERMINAL"
        else _ADVANCED_NONTERMINAL
    )
    if verified:
        status = "VERIFIED_" + status
    return {
        "status": status,
        "phase": (
            "ADVANCED_TERMINAL"
            if preview["terminal_status"] == "TERMINAL"
            else "ADVANCED_NONTERMINAL"
        ),
        "campaign_root": str(base["capture"]["root"]),
        "campaign_digest": base["campaign_digest"],
        "lease_digest": base["lease_digest"],
        "callback_start_claim_digest": claim_digest,
        "provenance_binding_digest": base["provenance_binding_digest"],
        "authorization_digest": base["authorization_digest"],
        "attempt_digest": base["attempt_digest"],
        "receipt_digest": preview["receipt_digest"],
        "journal_head_digest": preview["journal_head_digest"],
        "advance_digest": advance_digest,
        "reingestion_digest": preview["reingestion_digest"],
        "output_evidence_digest": preview["output_evidence_digest"],
        "output_report_body_digest": preview["output_report_body_digest"],
        "output_audit_head": preview["output_audit_head"],
        "next_pending_evidence_digest": preview["next_pending_evidence_digest"],
        "next_first_pending_projection_digest": preview[
            "next_first_pending_projection_digest"
        ],
        "next_bundle_digest": preview["next_bundle_digest"],
        "next_plan_digest": preview["next_plan_digest"],
        "remaining_task_count": preview["remaining_task_count"],
        "terminal_status": preview["terminal_status"],
        "next_attempt_root": preview["next_attempt_root"],
        "execution_status": "COMPLETED_AND_ADVANCED_HARD_STOP",
    }


def advance_recursive_campaign(
    source,
    campaign_contract_path,
    campaign_root,
    next_attempt_root,
    *,
    expected_campaign_digest,
    expected_lease_digest,
    expected_provenance_binding_digest,
    expected_authorization_digest,
    expected_attempt_digest,
    expected_receipt_digest,
    expected_journal_head_digest,
    expected_output_evidence_digest,
    expected_output_report_body_digest,
    expected_output_audit_head,
    expected_reingestion_digest,
    expected_next_pending_evidence_digest,
    expected_next_first_pending_projection_digest,
    expected_next_bundle_digest,
    expected_next_plan_digest,
    confirm_immutable_one_step_advance,
) -> dict[str, Any]:
    """Publish one immutable report advance and then hard-stop."""
    if confirm_immutable_one_step_advance is not True:
        raise RecursiveCampaignError("explicit immutable one-step advance confirmation is required")
    verification_context = _new_verification_context()
    root = _absolute_path(campaign_root, "campaign_root")
    campaign_id = root.name
    base = _validate_campaign_base(
        source,
        campaign_contract_path,
        root,
        campaign_id=campaign_id,
        expected_campaign_digest=expected_campaign_digest,
        expected_lease_digest=expected_lease_digest,
        allow_partial_advance=True,
        verification_context=verification_context,
    )
    dependencies = _load_dependency_objects(
        base["derived"]["descriptor"]["dependencies"]
    )
    _claim, claim_digest = _validate_claim(base)
    anchors = {
        "provenance_binding_digest": _require_digest(
            expected_provenance_binding_digest,
            "expected_provenance_binding_digest",
        ),
        "authorization_digest": _require_digest(
            expected_authorization_digest, "expected_authorization_digest"
        ),
        "attempt_digest": _require_digest(expected_attempt_digest, "expected_attempt_digest"),
        "receipt_digest": _require_digest(expected_receipt_digest, "expected_receipt_digest"),
        "journal_head_digest": _require_digest(
            expected_journal_head_digest, "expected_journal_head_digest"
        ),
        "output_evidence_digest": _require_digest(
            expected_output_evidence_digest, "expected_output_evidence_digest"
        ),
        "output_report_body_digest": _require_digest(
            expected_output_report_body_digest,
            "expected_output_report_body_digest",
        ),
        "output_audit_head": _require_digest(
            expected_output_audit_head, "expected_output_audit_head"
        ),
        "reingestion_digest": _require_digest(
            expected_reingestion_digest, "expected_reingestion_digest"
        ),
    }
    if (
        anchors["provenance_binding_digest"] != base["provenance_binding_digest"]
        or anchors["authorization_digest"] != base["authorization_digest"]
        or anchors["attempt_digest"] != base["attempt_digest"]
    ):
        raise RecursiveCampaignError("independent campaign advance anchors differ")
    preview = _preview_completed_advance(
        base,
        next_attempt_root=next_attempt_root,
        require_next_absent=(
            not base["capture"]["has_advance"]
            or base["capture"]["advance_partial"]
        ),
    )
    for key in (
        "receipt_digest", "journal_head_digest", "output_evidence_digest",
        "output_report_body_digest", "output_audit_head", "reingestion_digest",
    ):
        if preview[key] != anchors[key]:
            raise RecursiveCampaignError(f"independent preview anchor differs: {key}")
    terminal = preview["terminal_status"] == "TERMINAL"
    supplied_next = (
        expected_next_pending_evidence_digest,
        expected_next_first_pending_projection_digest,
        expected_next_bundle_digest,
        expected_next_plan_digest,
    )
    if terminal:
        if any(value is not None for value in supplied_next) or next_attempt_root is not None:
            raise RecursiveCampaignError("terminal advance forbids all next-generation anchors")
    else:
        expected_next = {
            "next_pending_evidence_digest": _require_digest(
                expected_next_pending_evidence_digest,
                "expected_next_pending_evidence_digest",
            ),
            "next_first_pending_projection_digest": _require_digest(
                expected_next_first_pending_projection_digest,
                "expected_next_first_pending_projection_digest",
            ),
            "next_bundle_digest": _require_digest(
                expected_next_bundle_digest, "expected_next_bundle_digest"
            ),
            "next_plan_digest": _require_digest(
                expected_next_plan_digest, "expected_next_plan_digest"
            ),
        }
        if any(preview[key] != value for key, value in expected_next.items()):
            raise RecursiveCampaignError("independent next-generation anchors differ")
    artifact_values = {
        "execution_receipt": preview["receipt"],
        "combined_rows": preview["combined_rows"],
        "output_report": preview["output_report"],
        "reingestion_receipt": preview["reingestion_receipt"],
    }
    if not terminal:
        artifact_values["next_bundle"] = preview["next_bundle"]
    artifact_map = {
        label: {
            "path": _ADVANCE_LAYOUT[label],
            "sha256": "sha256:" + hashlib.sha256(_canonical_file(value)).hexdigest(),
            "bytes": len(_canonical_file(value)),
        }
        for label, value in artifact_values.items()
    }
    advance_body = {
        "schema_version": ADVANCE_SCHEMA_VERSION,
        "status": _ADVANCED_TERMINAL if terminal else _ADVANCED_NONTERMINAL,
        "campaign_id": campaign_id,
        "campaign_digest": base["campaign_digest"],
        "lease_digest": base["lease_digest"],
        "callback_start_claim_digest": claim_digest,
        "provenance_binding_digest": base["provenance_binding_digest"],
        "authorization_digest": base["authorization_digest"],
        "attempt_digest": base["attempt_digest"],
        "receipt_digest": preview["receipt_digest"],
        "journal_head_digest": preview["journal_head_digest"],
        "reingestion_digest": preview["reingestion_digest"],
        "output_evidence_digest": preview["output_evidence_digest"],
        "output_report_body_digest": preview["output_report_body_digest"],
        "output_audit_head": preview["output_audit_head"],
        "next_pending_evidence_digest": preview["next_pending_evidence_digest"],
        "next_first_pending_projection_digest": preview[
            "next_first_pending_projection_digest"
        ],
        "next_bundle_digest": preview["next_bundle_digest"],
        "next_plan_digest": preview["next_plan_digest"],
        "remaining_task_count": preview["remaining_task_count"],
        "terminal_status": preview["terminal_status"],
        "next_attempt_root": preview["next_attempt_root"],
        "artifacts": artifact_map,
        "execution_status": "COMPLETED_AND_ADVANCED_HARD_STOP",
        "nonclaims": list(_NONCLAIMS),
    }
    advance = {
        **advance_body,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "advance_digest": _digest(advance_body),
        },
    }
    advance_dir = root / _ROOT_LAYOUT["advance_directory"]
    expected_names = {
        _ADVANCE_LAYOUT[label] for label in artifact_values
    } | {_ADVANCE_LAYOUT["advance_commit"]}
    if not _lexists(advance_dir):
        _mkdir_new(advance_dir)
    else:
        _secure_directory(advance_dir, exact_mode=0o700)
        observed_names = set(os.listdir(advance_dir))
        if not observed_names.issubset(expected_names):
            raise RecursiveCampaignError("partial advance contains extra names")
        if _ADVANCE_LAYOUT["advance_commit"] in observed_names:
            if observed_names != expected_names:
                raise RecursiveCampaignError(
                    "advance commit exists before complete advance layout"
                )
    advance_identity = _directory_identity(advance_dir)
    expected_values_by_name = {
        _ADVANCE_LAYOUT[label]: value for label, value in artifact_values.items()
    }
    expected_values_by_name[_ADVANCE_LAYOUT["advance_commit"]] = advance
    # Validate every pre-existing partial leaf before filling any missing leaf.
    for name in set(os.listdir(advance_dir)):
        expected_value = expected_values_by_name[name]
        observed_raw = _read_bytes(
            advance_dir / name, f"pre-existing advance {name}", exact_mode=0o600
        )
        if observed_raw != _canonical_file(expected_value):
            raise RecursiveCampaignError(f"pre-existing advance leaf differs: {name}")
    marker_preexisting = _lexists(
        advance_dir / _ADVANCE_LAYOUT["advance_commit"]
    )
    for label, value in artifact_values.items():
        _publish_or_match(
            advance_dir / _ADVANCE_LAYOUT[label],
            value,
            expected_parent_identity=advance_identity,
        )
    noncommit_names = expected_names - {_ADVANCE_LAYOUT["advance_commit"]}
    if set(os.listdir(advance_dir)) not in (noncommit_names, expected_names):
        raise RecursiveCampaignError("advance noncommit layout differs before marker")
    # Immediately before the commit marker, revalidate campaign/source and the
    # exact captured runtime generation, and recheck future-attempt absence.
    final_base = _validate_campaign_base(
        source,
        campaign_contract_path,
        root,
        campaign_id=campaign_id,
        expected_campaign_digest=expected_campaign_digest,
        expected_lease_digest=expected_lease_digest,
        allow_partial_advance=True,
        verification_context=verification_context,
    )
    for key in (
        "campaign_digest", "lease_digest", "provenance_binding_digest",
        "authorization_digest", "attempt_digest",
    ):
        if final_base[key] != base[key]:
            raise RecursiveCampaignError(f"campaign generation changed before advance: {key}")
    for key, raw in base["capture"]["raws"].items():
        if final_base["capture"]["raws"].get(key) != raw:
            raise RecursiveCampaignError(f"campaign artifact changed before advance: {key}")
    try:
        final_objects, final_runtime_raws = _P_ATTEMPT_SNAPSHOTS(
            base["derived"]["next_attempt_root"]
        )
        _P_VALIDATE_CAPTURED_RUNTIME(
            final_objects,
            dependencies["runtime"],
            dependencies["paths"]["base_manifest_path"],
            dependencies["paths"]["asset_root"],
            base["derived"]["next_attempt_root"],
            expected_plan_digest=base["derived"]["bundle"]["plan"]["integrity"]["plan_digest"],
            expected_authorization_digest=base["authorization_digest"],
            expected_execution_receipt_digest=preview["receipt_digest"],
            expected_execution_journal_head_digest=preview["journal_head_digest"],
            expected_execution_attempt_digest=base["attempt_digest"],
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError("runtime recapture failed before advance commit") from error
    if final_runtime_raws != preview["runtime_raws"]:
        raise RecursiveCampaignError("runtime generation changed before advance commit")
    _verify_context_freshness(verification_context)
    if not terminal:
        _future_attempt_candidate(
            preview["next_attempt_root"],
            dependencies["runtime"],
            require_absent=True,
        )
    for name in noncommit_names:
        if _read_bytes(
            advance_dir / name, f"final staged advance {name}", exact_mode=0o600
        ) != _canonical_file(expected_values_by_name[name]):
            raise RecursiveCampaignError(f"staged advance leaf differs: {name}")
    if not marker_preexisting:
        root_guard_names = {
            _ROOT_LAYOUT["campaign_contract"],
            _ROOT_LAYOUT["source_directory"],
            _ROOT_LAYOUT["lease"],
            _ROOT_LAYOUT["campaign_commit"],
            _ROOT_LAYOUT["callback_start_claim"],
            _ROOT_LAYOUT["advance_directory"],
        }
        root_guard_raws = {
            _ROOT_LAYOUT["campaign_contract"]: base["capture"]["raws"][
                "campaign_contract"
            ],
            _ROOT_LAYOUT["lease"]: base["capture"]["raws"]["lease"],
            _ROOT_LAYOUT["campaign_commit"]: base["capture"]["raws"][
                "campaign_commit"
            ],
            _ROOT_LAYOUT["callback_start_claim"]: base["capture"]["raws"][
                "callback_start_claim"
            ],
        }
        source_guard_raws = {
            relative: base["capture"]["raws"]["source_" + label]
            for label, relative in _SOURCE_LAYOUT.items()
        }
        advance_before_raws = {
            name: _canonical_file(expected_values_by_name[name])
            for name in noncommit_names
        }
        advance_after_raws = {
            **advance_before_raws,
            _ADVANCE_LAYOUT["advance_commit"]: _canonical_file(advance),
        }
        marker_guards = (
            {
                "label": "staged campaign advance",
                "path": advance_dir,
                "identity": advance_identity,
                "before_names": noncommit_names,
                "after_names": expected_names,
                "before_raws": advance_before_raws,
                "after_raws": advance_after_raws,
                "child_identities": {},
            },
            {
                "label": "advancing source snapshot",
                "path": root / _ROOT_LAYOUT["source_directory"],
                "identity": base["capture"]["source_identity"],
                "before_names": set(_SOURCE_LAYOUT.values()),
                "after_names": set(_SOURCE_LAYOUT.values()),
                "before_raws": source_guard_raws,
                "after_raws": source_guard_raws,
                "child_identities": {},
            },
            {
                "label": "advancing campaign root",
                "path": root,
                "identity": base["capture"]["root_identity"],
                "before_names": root_guard_names,
                "after_names": root_guard_names,
                "before_raws": root_guard_raws,
                "after_raws": root_guard_raws,
                "child_identities": {
                    _ROOT_LAYOUT["source_directory"]: base["capture"][
                        "source_identity"
                    ],
                    _ROOT_LAYOUT["advance_directory"]: advance_identity,
                },
            },
        )
        _write_new(
            advance_dir / _ADVANCE_LAYOUT["advance_commit"],
            advance,
            expected_parent_identity=advance_identity,
            publication_guards=marker_guards,
        )
    _require_names(advance_dir, expected_names, "published campaign advance")
    # The commit marker is not trusted merely because this process published
    # it.  Re-run the complete source/campaign validation and the captured
    # runtime validator after publication, then match the committed advance
    # against the same preview generation.
    committed_base = _validate_campaign_base(
        source,
        campaign_contract_path,
        root,
        campaign_id=campaign_id,
        expected_campaign_digest=expected_campaign_digest,
        expected_lease_digest=expected_lease_digest,
        verification_context=verification_context,
    )
    for key in (
        "campaign_digest", "lease_digest", "provenance_binding_digest",
        "authorization_digest", "attempt_digest",
    ):
        if committed_base[key] != base[key]:
            raise RecursiveCampaignError(
                f"campaign generation changed after advance commit: {key}"
            )
    if (
        committed_base["capture"]["root_identity"]
        != base["capture"]["root_identity"]
        or committed_base["capture"]["source_identity"]
        != base["capture"]["source_identity"]
        or committed_base["capture"]["advance_identity"] != advance_identity
    ):
        raise RecursiveCampaignError(
            "campaign directory identity changed after advance commit"
        )
    for key, raw in base["capture"]["raws"].items():
        if committed_base["capture"]["raws"].get(key) != raw:
            raise RecursiveCampaignError(
                f"campaign artifact changed after advance commit: {key}"
            )
    for key in (
        "descriptor", "source_state_digest", "rows", "report", "bundle",
        "next_attempt_root", "checkpoint_root",
    ):
        if committed_base["derived"][key] != base["derived"][key]:
            raise RecursiveCampaignError(
                f"campaign source changed after advance commit: {key}"
            )
    try:
        committed_objects, committed_runtime_raws = _P_ATTEMPT_SNAPSHOTS(
            base["derived"]["next_attempt_root"]
        )
        _P_VALIDATE_CAPTURED_RUNTIME(
            committed_objects,
            dependencies["runtime"],
            dependencies["paths"]["base_manifest_path"],
            dependencies["paths"]["asset_root"],
            base["derived"]["next_attempt_root"],
            expected_plan_digest=base["derived"]["bundle"]["plan"]["integrity"]["plan_digest"],
            expected_authorization_digest=base["authorization_digest"],
            expected_execution_receipt_digest=preview["receipt_digest"],
            expected_execution_journal_head_digest=preview["journal_head_digest"],
            expected_execution_attempt_digest=base["attempt_digest"],
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError(
            "runtime changed after advance commit"
        ) from error
    if committed_runtime_raws != preview["runtime_raws"]:
        raise RecursiveCampaignError(
            "runtime generation changed after advance commit"
        )
    _verify_context_freshness(verification_context)
    _committed_marker, committed_advance_digest = _validate_advanced_capture(
        committed_base, claim_digest, preview
    )
    if committed_advance_digest != advance["integrity"]["advance_digest"]:
        raise RecursiveCampaignError("published advance marker digest differs")
    return _advanced_result(
        committed_base,
        claim_digest,
        preview,
        advance["integrity"]["advance_digest"],
        verified=False,
    )


def _phase_result(
    base: Mapping[str, Any],
    *,
    status: str,
    phase: str,
    claim_digest: str | None,
    execution_status: str,
    receipt_digest: str | None = None,
    journal_head_digest: str | None = None,
    preview: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    has_preview = preview is not None
    preview = {} if preview is None else preview
    source_remaining = len(base["derived"]["report"]["pending_evidence"])
    return {
        "status": status,
        "phase": phase,
        "campaign_root": str(base["capture"]["root"]),
        "campaign_digest": base["campaign_digest"],
        "lease_digest": base["lease_digest"],
        "callback_start_claim_digest": claim_digest,
        "provenance_binding_digest": base["provenance_binding_digest"],
        "authorization_digest": base["authorization_digest"],
        "attempt_digest": base["attempt_digest"],
        "receipt_digest": receipt_digest,
        "journal_head_digest": journal_head_digest,
        "advance_digest": None,
        "reingestion_digest": preview.get("reingestion_digest"),
        "output_evidence_digest": preview.get("output_evidence_digest"),
        "output_report_body_digest": preview.get("output_report_body_digest"),
        "output_audit_head": preview.get("output_audit_head"),
        "next_pending_evidence_digest": preview.get(
            "next_pending_evidence_digest"
        ),
        "next_first_pending_projection_digest": preview.get(
            "next_first_pending_projection_digest"
        ),
        "next_bundle_digest": preview.get("next_bundle_digest"),
        "next_plan_digest": preview.get("next_plan_digest"),
        "remaining_task_count": (
            preview.get("remaining_task_count")
            if has_preview
            else source_remaining
        ),
        "terminal_status": (
            preview.get("terminal_status") if has_preview else "NONTERMINAL"
        ),
        "next_attempt_root": preview.get("next_attempt_root"),
        "execution_status": execution_status,
    }


_LOCAL_CAPTURE_COMPARE_KEYS = (
    "raws", "has_claim", "has_advance", "terminal", "advance_partial",
    "advance_raws", "root_identity", "source_identity", "advance_identity",
)


def _campaign_freshness_record(
    base: Mapping[str, Any],
    campaign_contract_path,
    *,
    preview: Mapping[str, Any] | None,
    expected_runtime_statuses: set[str] | None,
    expected_runtime_receipt_digest: str | None,
    expected_runtime_journal_head_digest: str | None,
) -> dict[str, Any]:
    return {
        "resolved_root": str(base["capture"]["root"].resolve(strict=True)),
        "base": base,
        "campaign_contract_path": _absolute_path(
            campaign_contract_path, "campaign_contract_path"
        ),
        "preview": preview,
        "expected_runtime_statuses": (
            None
            if expected_runtime_statuses is None
            else frozenset(expected_runtime_statuses)
        ),
        "expected_runtime_receipt_digest": expected_runtime_receipt_digest,
        "expected_runtime_journal_head_digest": (
            expected_runtime_journal_head_digest
        ),
    }


def _check_campaign_freshness_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Recapture one admitted campaign without re-verifying its ancestors."""
    base = record["base"]
    first_capture = base["capture"]
    root = first_capture["root"]
    final_capture = _capture_campaign(
        root, base["campaign"]["campaign_id"]
    )
    for key in _LOCAL_CAPTURE_COMPARE_KEYS:
        if final_capture[key] != first_capture[key]:
            raise RecursiveCampaignError(
                f"campaign generation changed during verification: {key}"
            )
    _contract, contract_raw = _load_contract(
        record["campaign_contract_path"]
    )
    if final_capture["raws"]["campaign_contract"] != contract_raw:
        raise RecursiveCampaignError(
            "campaign contract changed during verification"
        )
    preview = record["preview"]
    if preview is not None:
        try:
            _objects, runtime_raws = _P_ATTEMPT_SNAPSHOTS(
                base["derived"]["next_attempt_root"]
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise RecursiveCampaignError(
                "runtime generation unavailable at verification return"
            ) from error
        if runtime_raws != preview["runtime_raws"]:
            raise RecursiveCampaignError(
                "runtime generation changed during campaign verification"
            )
        if (
            not first_capture["has_advance"]
            and preview.get("terminal_status") == "NONTERMINAL"
        ):
            dependencies = _load_dependency_objects(
                base["derived"]["descriptor"]["dependencies"]
            )
            _future_attempt_candidate(
                preview.get("next_attempt_root"),
                dependencies["runtime"],
                require_absent=True,
            )
    elif record["expected_runtime_statuses"] is not None:
        dependencies = _load_dependency_objects(
            base["derived"]["descriptor"]["dependencies"]
        )
        verify_kwargs = {
            "expected_authorization_digest": base["authorization_digest"]
        }
        receipt_digest = record["expected_runtime_receipt_digest"]
        journal_digest = record["expected_runtime_journal_head_digest"]
        if receipt_digest is not None:
            if journal_digest is None:
                raise RecursiveCampaignError(
                    "final runtime receipt/journal anchors are not paired"
                )
            verify_kwargs.update(
                expected_receipt_digest=receipt_digest,
                expected_journal_head_digest=journal_digest,
                expected_attempt_digest=base["attempt_digest"],
            )
        try:
            final_runtime = _R_VERIFY(
                base["derived"]["next_attempt_root"],
                dependencies["runtime"],
                dependencies["paths"]["base_manifest_path"],
                dependencies["paths"]["asset_root"],
                **verify_kwargs,
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise RecursiveCampaignError(
                "runtime generation changed at campaign verification return"
            ) from error
        if (
            final_runtime.get("status")
            not in record["expected_runtime_statuses"]
            or final_runtime.get("authorization_digest")
            != base["authorization_digest"]
            or final_runtime.get("attempt_digest") != base["attempt_digest"]
            or (
                receipt_digest is not None
                and (
                    final_runtime.get("receipt_digest") != receipt_digest
                    or final_runtime.get("journal_head_digest")
                    != journal_digest
                )
            )
        ):
            raise RecursiveCampaignError(
                "runtime phase changed at campaign verification return"
            )
    return final_capture


def _register_campaign_freshness(
    context: dict[str, Any], record: Mapping[str, Any]
) -> None:
    resolved = record["resolved_root"]
    existing = context["campaign_freshness"].get(resolved)
    if existing is not None:
        if (
            existing["base"]["capture"]["raws"]
            != record["base"]["capture"]["raws"]
            or existing["base"]["capture"]["advance_raws"]
            != record["base"]["capture"]["advance_raws"]
            or existing["campaign_contract_path"]
            != record["campaign_contract_path"]
            or existing["expected_runtime_statuses"]
            != record["expected_runtime_statuses"]
            or existing["expected_runtime_receipt_digest"]
            != record["expected_runtime_receipt_digest"]
            or existing["expected_runtime_journal_head_digest"]
            != record["expected_runtime_journal_head_digest"]
            or (
                (existing["preview"] is None) != (record["preview"] is None)
            )
            or (
                existing["preview"] is not None
                and existing["preview"]["runtime_raws"]
                != record["preview"]["runtime_raws"]
            )
        ):
            raise RecursiveCampaignError(
                "one campaign root admitted as two generations"
            )
        return
    context["campaign_freshness"][resolved] = record


def _verify_context_freshness(context: dict[str, Any]) -> None:
    """Linearly recapture the bootstrap and every admitted campaign root."""
    context = _require_verification_context(context)
    for record in context["successor_freshness"].values():
        observed = _derive_successor_source(
            record["descriptor"], require_attempt_absent=False
        )
        expected = record["derived"]
        for key in (
            "source_kind", "source_state_digest", "source_verification",
            "rows", "report", "bundle", "next_attempt_root",
            "checkpoint_root",
        ):
            if observed[key] != expected[key]:
                raise RecursiveCampaignError(
                    f"recursive successor source changed during operation: {key}"
                )
    for record in context["campaign_freshness"].values():
        _check_campaign_freshness_record(record)
    for attempt_root in context["source_attempt_absence"].values():
        if _lexists(attempt_root):
            raise RecursiveCampaignError(
                "source-bound next attempt root changed during operation"
            )


def _finish_phase_verification(
    base: Mapping[str, Any],
    source: Mapping[str, Any],
    campaign_contract_path,
    *,
    result: Mapping[str, Any],
    preview: Mapping[str, Any] | None = None,
    expected_runtime_statuses: set[str] | None = None,
    expected_runtime_receipt_digest: str | None = None,
    expected_runtime_journal_head_digest: str | None = None,
) -> dict[str, Any]:
    record = _campaign_freshness_record(
        base,
        campaign_contract_path,
        preview=preview,
        expected_runtime_statuses=expected_runtime_statuses,
        expected_runtime_receipt_digest=expected_runtime_receipt_digest,
        expected_runtime_journal_head_digest=(
            expected_runtime_journal_head_digest
        ),
    )
    final_capture = _check_campaign_freshness_record(record)
    final = dict(base)
    final["capture"] = final_capture
    return {
        "result": dict(result),
        "capture": final_capture,
        "base": final,
        "freshness_record": record,
        **({"preview": preview} if preview is not None else {}),
    }


def _validate_advanced_capture(
    base: Mapping[str, Any],
    claim_digest: str,
    preview: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    capture = base["capture"]
    if not capture["has_advance"] or capture["advance_partial"]:
        raise RecursiveCampaignError("campaign advance is not fully committed")
    marker = capture["advance_values"].get("advance_commit")
    advance_digest = _integrity_digest(marker, "advance_digest", "advance marker")
    terminal = preview["terminal_status"] == "TERMINAL"
    artifact_values = {
        "execution_receipt": preview["receipt"],
        "combined_rows": preview["combined_rows"],
        "output_report": preview["output_report"],
        "reingestion_receipt": preview["reingestion_receipt"],
    }
    if not terminal:
        artifact_values["next_bundle"] = preview["next_bundle"]
    if set(capture["advance_values"]) != set(artifact_values) | {"advance_commit"}:
        raise RecursiveCampaignError("captured advance artifact set differs")
    for label, expected in artifact_values.items():
        if capture["advance_values"].get(label) != expected:
            raise RecursiveCampaignError(f"captured advance differs: {label}")
    artifact_map = {
        label: {
            "path": _ADVANCE_LAYOUT[label],
            "sha256": "sha256:" + hashlib.sha256(_canonical_file(value)).hexdigest(),
            "bytes": len(_canonical_file(value)),
        }
        for label, value in artifact_values.items()
    }
    expected_body = {
        "schema_version": ADVANCE_SCHEMA_VERSION,
        "status": _ADVANCED_TERMINAL if terminal else _ADVANCED_NONTERMINAL,
        "campaign_id": base["campaign"]["campaign_id"],
        "campaign_digest": base["campaign_digest"],
        "lease_digest": base["lease_digest"],
        "callback_start_claim_digest": claim_digest,
        "provenance_binding_digest": base["provenance_binding_digest"],
        "authorization_digest": base["authorization_digest"],
        "attempt_digest": base["attempt_digest"],
        "receipt_digest": preview["receipt_digest"],
        "journal_head_digest": preview["journal_head_digest"],
        "reingestion_digest": preview["reingestion_digest"],
        "output_evidence_digest": preview["output_evidence_digest"],
        "output_report_body_digest": preview["output_report_body_digest"],
        "output_audit_head": preview["output_audit_head"],
        "next_pending_evidence_digest": preview["next_pending_evidence_digest"],
        "next_first_pending_projection_digest": preview[
            "next_first_pending_projection_digest"
        ],
        "next_bundle_digest": preview["next_bundle_digest"],
        "next_plan_digest": preview["next_plan_digest"],
        "remaining_task_count": preview["remaining_task_count"],
        "terminal_status": preview["terminal_status"],
        "next_attempt_root": preview["next_attempt_root"],
        "artifacts": artifact_map,
        "execution_status": "COMPLETED_AND_ADVANCED_HARD_STOP",
        "nonclaims": list(_NONCLAIMS),
    }
    if {key: marker[key] for key in marker if key != "integrity"} != expected_body:
        raise RecursiveCampaignError("advance marker body differs")
    return marker, advance_digest


def _verification_cache_key(
    source: Mapping[str, Any],
    campaign_contract_path,
    campaign_root,
    *,
    campaign_id: str,
    next_attempt_root,
    expected_campaign_digest,
    expected_lease_digest,
    expected_callback_start_claim_digest,
    expected_receipt_digest,
    expected_journal_head_digest,
    expected_advance_digest,
    expected_output_evidence_digest,
    expected_output_report_body_digest,
    expected_output_audit_head,
    expected_reingestion_digest,
    expected_next_bundle_digest,
    expected_next_plan_digest,
) -> tuple[str, str]:
    if not isinstance(source, Mapping):
        raise RecursiveCampaignError("source must be a JSON mapping")
    root = _absolute_path(campaign_root, "campaign_root")
    try:
        resolved_root = str(root.resolve(strict=True))
    except OSError as error:
        raise RecursiveCampaignError("campaign root is unavailable") from error
    contract_path = _absolute_path(
        campaign_contract_path, "campaign_contract_path"
    )
    next_path = (
        None
        if next_attempt_root is None
        else str(_absolute_path(next_attempt_root, "next_attempt_root"))
    )
    payload = {
        "resolved_root": resolved_root,
        "campaign_contract_path": str(contract_path),
        "campaign_id": campaign_id,
        "source_descriptor": _clone(dict(source), "source descriptor"),
        "next_attempt_root": next_path,
        "expected": {
            "campaign_digest": expected_campaign_digest,
            "lease_digest": expected_lease_digest,
            "callback_start_claim_digest": (
                expected_callback_start_claim_digest
            ),
            "receipt_digest": expected_receipt_digest,
            "journal_head_digest": expected_journal_head_digest,
            "advance_digest": expected_advance_digest,
            "output_evidence_digest": expected_output_evidence_digest,
            "output_report_body_digest": expected_output_report_body_digest,
            "output_audit_head": expected_output_audit_head,
            "reingestion_digest": expected_reingestion_digest,
            "next_bundle_digest": expected_next_bundle_digest,
            "next_plan_digest": expected_next_plan_digest,
        },
    }
    return resolved_root, _digest(payload)


def _verify_campaign_internal(
    source,
    campaign_contract_path,
    campaign_root,
    *,
    campaign_id: str,
    next_attempt_root=None,
    expected_campaign_digest,
    expected_lease_digest,
    expected_callback_start_claim_digest=None,
    expected_receipt_digest=None,
    expected_journal_head_digest=None,
    expected_advance_digest=None,
    expected_output_evidence_digest=None,
    expected_output_report_body_digest=None,
    expected_output_audit_head=None,
    expected_reingestion_digest=None,
    expected_next_bundle_digest=None,
    expected_next_plan_digest=None,
    depth: int = 0,
    visited: set[str] | None = None,
    verification_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Memoize one exact campaign semantic verifier per public operation."""
    context = _require_verification_context(verification_context)
    resolved_root, cache_digest = _verification_cache_key(
        source,
        campaign_contract_path,
        campaign_root,
        campaign_id=campaign_id,
        next_attempt_root=next_attempt_root,
        expected_campaign_digest=expected_campaign_digest,
        expected_lease_digest=expected_lease_digest,
        expected_callback_start_claim_digest=(
            expected_callback_start_claim_digest
        ),
        expected_receipt_digest=expected_receipt_digest,
        expected_journal_head_digest=expected_journal_head_digest,
        expected_advance_digest=expected_advance_digest,
        expected_output_evidence_digest=expected_output_evidence_digest,
        expected_output_report_body_digest=expected_output_report_body_digest,
        expected_output_audit_head=expected_output_audit_head,
        expected_reingestion_digest=expected_reingestion_digest,
        expected_next_bundle_digest=expected_next_bundle_digest,
        expected_next_plan_digest=expected_next_plan_digest,
    )
    cache_key = (resolved_root, cache_digest)
    cached = context["semantic_cache"].get(cache_key)
    if cached is not None:
        _check_campaign_freshness_record(cached["freshness_record"])
        return cached
    if cache_key in context["semantic_active"]:
        raise RecursiveCampaignError("recursive campaign semantic cycle detected")
    _record_semantic_cache_miss(context, resolved_root)
    context["semantic_active"].add(cache_key)
    try:
        verified = _verify_campaign_internal_uncached(
            source,
            campaign_contract_path,
            campaign_root,
            campaign_id=campaign_id,
            next_attempt_root=next_attempt_root,
            expected_campaign_digest=expected_campaign_digest,
            expected_lease_digest=expected_lease_digest,
            expected_callback_start_claim_digest=(
                expected_callback_start_claim_digest
            ),
            expected_receipt_digest=expected_receipt_digest,
            expected_journal_head_digest=expected_journal_head_digest,
            expected_advance_digest=expected_advance_digest,
            expected_output_evidence_digest=expected_output_evidence_digest,
            expected_output_report_body_digest=(
                expected_output_report_body_digest
            ),
            expected_output_audit_head=expected_output_audit_head,
            expected_reingestion_digest=expected_reingestion_digest,
            expected_next_bundle_digest=expected_next_bundle_digest,
            expected_next_plan_digest=expected_next_plan_digest,
            depth=depth,
            visited=visited,
            verification_context=context,
        )
        _register_campaign_freshness(
            context, verified["freshness_record"]
        )
        context["semantic_cache"][cache_key] = verified
        return verified
    finally:
        context["semantic_active"].discard(cache_key)


def _verify_campaign_internal_uncached(
    source,
    campaign_contract_path,
    campaign_root,
    *,
    campaign_id: str,
    next_attempt_root=None,
    expected_campaign_digest,
    expected_lease_digest,
    expected_callback_start_claim_digest=None,
    expected_receipt_digest=None,
    expected_journal_head_digest=None,
    expected_advance_digest=None,
    expected_output_evidence_digest=None,
    expected_output_report_body_digest=None,
    expected_output_audit_head=None,
    expected_reingestion_digest=None,
    expected_next_bundle_digest=None,
    expected_next_plan_digest=None,
    depth: int = 0,
    visited: set[str] | None = None,
    verification_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    receipt_pair = (expected_receipt_digest, expected_journal_head_digest)
    if (receipt_pair[0] is None) != (receipt_pair[1] is None):
        raise RecursiveCampaignError(
            "expected receipt and journal anchors must be supplied together"
        )
    if expected_receipt_digest is not None:
        expected_receipt_digest = _require_digest(
            expected_receipt_digest, "expected_receipt_digest"
        )
        expected_journal_head_digest = _require_digest(
            expected_journal_head_digest, "expected_journal_head_digest"
        )
    base = _validate_campaign_base(
        source,
        campaign_contract_path,
        campaign_root,
        campaign_id=campaign_id,
        expected_campaign_digest=expected_campaign_digest,
        expected_lease_digest=expected_lease_digest,
        depth=depth,
        visited=visited,
        verification_context=verification_context,
    )
    capture = base["capture"]
    dependencies = _load_dependency_objects(base["derived"]["descriptor"]["dependencies"])
    if not capture["has_claim"]:
        optional = (
            expected_callback_start_claim_digest, expected_advance_digest,
            expected_output_evidence_digest, expected_output_report_body_digest,
            expected_output_audit_head, expected_reingestion_digest,
            expected_next_bundle_digest, expected_next_plan_digest,
            expected_receipt_digest, expected_journal_head_digest,
        )
        if any(value is not None for value in optional) or next_attempt_root is not None:
            raise RecursiveCampaignError("authorized verification received future anchors")
        try:
            runtime_verified = _R_VERIFY(
                base["derived"]["next_attempt_root"],
                dependencies["runtime"],
                dependencies["paths"]["base_manifest_path"],
                dependencies["paths"]["asset_root"],
                expected_authorization_digest=base["authorization_digest"],
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise RecursiveCampaignError("authorized campaign runtime verification failed") from error
        if (
            runtime_verified.get("status") != "VERIFIED_AUTHORIZED_NOT_EXECUTED"
            or runtime_verified.get("attempt_digest") != base["attempt_digest"]
        ):
            raise RecursiveCampaignError("unclaimed campaign runtime is not AUTHORIZED-only")
        result = _phase_result(
            base,
            status="VERIFIED_" + _AUTHORIZED,
            phase="AUTHORIZED",
            claim_digest=None,
            execution_status="NOT_EXECUTED",
        )
        return _finish_phase_verification(
            base,
            source,
            campaign_contract_path,
            result=result,
            expected_runtime_statuses={"VERIFIED_AUTHORIZED_NOT_EXECUTED"},
        )
    _claim, claim_digest = _validate_claim(base)
    if expected_callback_start_claim_digest is not None:
        if claim_digest != _require_digest(
            expected_callback_start_claim_digest,
            "expected_callback_start_claim_digest",
        ):
            raise RecursiveCampaignError("independent callback claim digest differs")
    completed_path = (
        base["derived"]["next_attempt_root"] / "journal" / "0002_COMPLETED.json"
    )
    if not _lexists(completed_path):
        if capture["has_advance"] or any(
            value is not None
            for value in (
                expected_advance_digest, expected_output_evidence_digest,
                expected_output_report_body_digest, expected_output_audit_head,
                expected_reingestion_digest, expected_next_bundle_digest,
                expected_next_plan_digest, expected_receipt_digest,
                expected_journal_head_digest, next_attempt_root,
            )
        ):
            raise RecursiveCampaignError("incomplete callback has future anchors")
        try:
            runtime_verified = _R_VERIFY(
                base["derived"]["next_attempt_root"],
                dependencies["runtime"],
                dependencies["paths"]["base_manifest_path"],
                dependencies["paths"]["asset_root"],
                expected_authorization_digest=base["authorization_digest"],
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise RecursiveCampaignError("claimed incomplete runtime verification failed") from error
        if runtime_verified.get("status") not in {
            "VERIFIED_AUTHORIZED_NOT_EXECUTED",
            "VERIFIED_PREFLIGHT_PASSED_NO_CALLBACK",
            "VERIFIED_RUNNING_INCOMPLETE_NO_REENTRY",
        }:
            raise RecursiveCampaignError("claimed runtime phase is not accepted incomplete state")
        result = _phase_result(
            base,
            status="VERIFIED_" + _CALLBACK_INCOMPLETE,
            phase="CALLBACK_INCOMPLETE",
            claim_digest=claim_digest,
            execution_status="CALLBACK_CLAIMED_INCOMPLETE_NO_REENTRY",
        )
        return _finish_phase_verification(
            base,
            source,
            campaign_contract_path,
            result=result,
            expected_runtime_statuses={runtime_verified["status"]},
        )
    local_anchors = _completed_runtime_anchors(base)
    if expected_receipt_digest is not None and (
        local_anchors["receipt_digest"] != expected_receipt_digest
        or local_anchors["journal_head_digest"] != expected_journal_head_digest
    ):
        raise RecursiveCampaignError("independent completed runtime anchors differ")
    try:
        runtime_verified = _R_VERIFY(
            base["derived"]["next_attempt_root"],
            dependencies["runtime"],
            dependencies["paths"]["base_manifest_path"],
            dependencies["paths"]["asset_root"],
            expected_authorization_digest=base["authorization_digest"],
            expected_receipt_digest=local_anchors["receipt_digest"],
            expected_journal_head_digest=local_anchors["journal_head_digest"],
            expected_attempt_digest=base["attempt_digest"],
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RecursiveCampaignError("completed runtime verification failed") from error
    if runtime_verified.get("status") != "VERIFIED_COMPLETED":
        raise RecursiveCampaignError("claimed runtime is not verified COMPLETED")
    receipt = _read_json(
        base["derived"]["next_attempt_root"] / "receipt.json",
        "completed receipt classification",
        exact_mode=0o600,
    )
    if (
        _integrity_digest(
            receipt, "receipt_digest", "completed receipt classification"
        )
        != local_anchors["receipt_digest"]
    ):
        raise RecursiveCampaignError(
            "completed receipt classification generation differs"
        )
    successful = (
        receipt.get("summary") == {"authorized": 1, "succeeded": 1, "failed": 0}
        and type(receipt.get("results")) is list
        and len(receipt["results"]) == 1
        and receipt["results"][0].get("status") == "SUCCEEDED"
    )
    anchored = expected_receipt_digest is not None
    if not successful:
        if capture["has_advance"] or any(
            value is not None
            for value in (
                expected_advance_digest, expected_output_evidence_digest,
                expected_output_report_body_digest, expected_output_audit_head,
                expected_reingestion_digest, expected_next_bundle_digest,
                expected_next_plan_digest, next_attempt_root,
            )
        ):
            raise RecursiveCampaignError("failed completed callback cannot be advanced")
        suffix = (
            "INDEPENDENTLY_ANCHORED_HARD_STOP"
            if anchored
            else "RECOVERED_LOCAL_ANCHORS_NOT_INDEPENDENT_HARD_STOP"
        )
        result = _phase_result(
            base,
            status="VERIFIED_" + _CALLBACK_COMPLETED,
            phase="CALLBACK_COMPLETED",
            claim_digest=claim_digest,
            receipt_digest=local_anchors["receipt_digest"],
            journal_head_digest=local_anchors["journal_head_digest"],
            execution_status=(
                "COMPLETED_FAILED_EVIDENCE_NEUTRAL_" + suffix
            ),
        )
        return _finish_phase_verification(
            base,
            source,
            campaign_contract_path,
            result=result,
            expected_runtime_statuses={"VERIFIED_COMPLETED"},
            expected_runtime_receipt_digest=local_anchors["receipt_digest"],
            expected_runtime_journal_head_digest=local_anchors[
                "journal_head_digest"
            ],
        )
    if capture["has_advance"]:
        marker = capture["advance_values"].get("advance_commit", {})
        bound_next = marker.get("next_attempt_root")
        if next_attempt_root is not None and str(next_attempt_root) != bound_next:
            raise RecursiveCampaignError("supplied next attempt differs from advance marker")
        preview_next = bound_next
    else:
        preview_next = next_attempt_root
    preview = _preview_completed_advance(
        base,
        next_attempt_root=preview_next,
        require_next_absent=not capture["has_advance"],
    )
    if (
        preview["receipt_digest"] != local_anchors["receipt_digest"]
        or preview["journal_head_digest"]
        != local_anchors["journal_head_digest"]
    ):
        raise RecursiveCampaignError(
            "completed preview differs from the anchored runtime generation"
        )
    if not capture["has_advance"]:
        if expected_advance_digest is not None:
            raise RecursiveCampaignError("completed preview cannot have advance digest")
        optional_preview = {
            "output_evidence_digest": expected_output_evidence_digest,
            "output_report_body_digest": expected_output_report_body_digest,
            "output_audit_head": expected_output_audit_head,
            "reingestion_digest": expected_reingestion_digest,
            "next_bundle_digest": expected_next_bundle_digest,
            "next_plan_digest": expected_next_plan_digest,
        }
        for key, expected in optional_preview.items():
            if expected is not None and preview[key] != _require_digest(expected, key):
                raise RecursiveCampaignError(f"completed preview anchor differs: {key}")
        suffix = (
            "INDEPENDENTLY_ANCHORED_HARD_STOP"
            if anchored
            else "RECOVERED_LOCAL_ANCHORS_NOT_INDEPENDENT_HARD_STOP"
        )
        result = _phase_result(
            base,
            status="VERIFIED_" + _CALLBACK_COMPLETED,
            phase="CALLBACK_COMPLETED",
            claim_digest=claim_digest,
            receipt_digest=preview["receipt_digest"],
            journal_head_digest=preview["journal_head_digest"],
            preview=preview,
            execution_status="COMPLETED_SUCCESS_PREVIEW_" + suffix,
        )
        return _finish_phase_verification(
            base,
            source,
            campaign_contract_path,
            result=result,
            preview=preview,
        )
    _marker, advance_digest = _validate_advanced_capture(base, claim_digest, preview)
    expected_advanced = {
        "advance_digest": expected_advance_digest,
        "output_evidence_digest": expected_output_evidence_digest,
        "output_report_body_digest": expected_output_report_body_digest,
        "output_audit_head": expected_output_audit_head,
        "reingestion_digest": expected_reingestion_digest,
        "next_bundle_digest": expected_next_bundle_digest,
        "next_plan_digest": expected_next_plan_digest,
    }
    actual_advanced = {**preview, "advance_digest": advance_digest}
    always_required = {
        "advance_digest", "output_evidence_digest",
        "output_report_body_digest", "output_audit_head",
        "reingestion_digest",
    }
    for key in always_required:
        expected = expected_advanced[key]
        if expected is None or actual_advanced[key] != _require_digest(expected, key):
            raise RecursiveCampaignError(f"advanced verification anchor differs: {key}")
    if preview["terminal_status"] == "TERMINAL":
        if (
            expected_next_bundle_digest is not None
            or expected_next_plan_digest is not None
            or actual_advanced["next_bundle_digest"] is not None
            or actual_advanced["next_plan_digest"] is not None
        ):
            raise RecursiveCampaignError("terminal advanced next anchors must be None")
    else:
        for key in ("next_bundle_digest", "next_plan_digest"):
            expected = expected_advanced[key]
            if expected is None or actual_advanced[key] != _require_digest(expected, key):
                raise RecursiveCampaignError(f"advanced verification anchor differs: {key}")
    result = _advanced_result(
        base, claim_digest, preview, advance_digest, verified=True
    )
    return _finish_phase_verification(
        base,
        source,
        campaign_contract_path,
        result=result,
        preview=preview,
    )


def verify_recursive_campaign(
    source,
    campaign_contract_path,
    campaign_root,
    *,
    next_attempt_root=None,
    expected_campaign_digest,
    expected_lease_digest,
    expected_callback_start_claim_digest=None,
    expected_receipt_digest=None,
    expected_journal_head_digest=None,
    expected_advance_digest=None,
    expected_output_evidence_digest=None,
    expected_output_report_body_digest=None,
    expected_output_audit_head=None,
    expected_reingestion_digest=None,
    expected_next_bundle_digest=None,
    expected_next_plan_digest=None,
) -> dict[str, Any]:
    """Read-only phase-aware verification of one recursive campaign capsule."""
    verification_context = _new_verification_context()
    root = _absolute_path(campaign_root, "campaign_root")
    root = _campaign_root(root, root.name, fresh=False)
    try:
        resolved = str(root.resolve(strict=True))
    except OSError as error:
        raise RecursiveCampaignError("campaign root is unavailable") from error
    verified = _verify_campaign_internal(
        source,
        campaign_contract_path,
        root,
        campaign_id=root.name,
        next_attempt_root=next_attempt_root,
        expected_campaign_digest=expected_campaign_digest,
        expected_lease_digest=expected_lease_digest,
        expected_callback_start_claim_digest=expected_callback_start_claim_digest,
        expected_receipt_digest=expected_receipt_digest,
        expected_journal_head_digest=expected_journal_head_digest,
        expected_advance_digest=expected_advance_digest,
        expected_output_evidence_digest=expected_output_evidence_digest,
        expected_output_report_body_digest=expected_output_report_body_digest,
        expected_output_audit_head=expected_output_audit_head,
        expected_reingestion_digest=expected_reingestion_digest,
        expected_next_bundle_digest=expected_next_bundle_digest,
        expected_next_plan_digest=expected_next_plan_digest,
        depth=0,
        visited={resolved},
        verification_context=verification_context,
    )
    _verify_context_freshness(verification_context)
    return verified["result"]


__all__ = [
    "ADVANCE_SCHEMA_VERSION",
    "CALLBACK_START_CLAIM_SCHEMA_VERSION",
    "CAMPAIGN_CONTRACT_ID",
    "CAMPAIGN_SCHEMA_VERSION",
    "CAPSULE_SCHEMA_VERSION",
    "LEASE_SCHEMA_VERSION",
    "PROVENANCE_SCHEMA_VERSION",
    "RecursiveCampaignError",
    "SOURCE_COMMIT_SCHEMA_VERSION",
    "SOURCE_SCHEMA_VERSION",
    "advance_recursive_campaign",
    "authorize_recursive_campaign_task",
    "execute_recursive_campaign_task",
    "inspect_recursive_campaign",
    "verify_recursive_campaign",
]
