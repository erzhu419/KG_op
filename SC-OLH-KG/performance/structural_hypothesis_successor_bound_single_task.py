"""Bind one verified successor generation to a runtime-V1 authorized attempt.

Inspection is read-only.  Preparation delegates only to runtime V1's prepare
operation and therefore creates an AUTHORIZED attempt but never preflights,
imports the benchmark, executes ``run_one``, or records a result.  The exact
successor/adoption provenance digest is carried by the otherwise unchanged
runtime V1 authorization ID.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from . import structural_hypothesis_adopted_successor_materializer as _successor
from . import structural_hypothesis_report_adoption as _adoption
from . import structural_hypothesis_single_task_runtime as _runtime
from .structural_hypothesis_loop import canonical_json_bytes


# Definition-time captures prevent post-import module-attribute rebinding from
# changing which producer-independent validators or runtime operation we call.
_S_ABSOLUTE_PATH = _successor._absolute_path
_S_ARTIFACT = _successor._artifact
_S_MARKER_BODY = _successor._marker_body
_S_PARSE_OBJECT = _successor._parse_object
_S_PATHS = _successor._paths
_S_READ_JSON = _successor._read_json
_S_SAME_ADOPTION = _successor._same_adoption
_S_VALIDATE_AND_MATERIALIZE = _successor._validate_and_materialize
_S_VALIDATE_CONTRACT = _successor._validate_successor_contract
_S_VALIDATE_FUTURE_ATTEMPT = _successor._validate_future_attempt
_S_VALIDATE_LOCATION = _successor._validate_successor_location
_A_CHECK_OPEN_DIRECTORY = _adoption._check_open_directory
_A_READ_REGULAR_AT = _adoption._read_regular_at
_A_REQUIRE_NAMES = _adoption._require_names
_R_EXPECTED_ATTEMPT = _runtime._expected_attempt
_R_PREPARE_ATTEMPT = _runtime.prepare_single_task_attempt
_R_VALIDATE_CONTRACT = _runtime.validate_runtime_contract
_R_VERIFY_ATTEMPT = _runtime.verify_single_task_attempt
_SUCCESSOR_ERROR = _successor.AdoptedSuccessorMaterializationError
_ADOPTION_ERROR = _adoption.StructuralHypothesisReportAdoptionError
_RUNTIME_ERROR = _runtime.SingleTaskRuntimeValidationError
_SUCCESSOR_CAPSULE_SCHEMA_VERSION = _successor.SUCCESSOR_CAPSULE_SCHEMA_VERSION
_SUCCESSOR_CONTRACT_ID = _successor.SUCCESSOR_CONTRACT_ID
_SUCCESSOR_NONCLAIMS = tuple(_successor._NONCLAIMS)


BRIDGE_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-successor-bound-single-task/1"
)
PROVENANCE_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-successor-bound-provenance/1"
)
BRIDGE_CONTRACT_ID = "structural_hypothesis_successor_bound_single_task_v1"

_BRIDGE_CONTRACT_DIGEST = (
    "sha256:3969b99401f92ce7d193f8e6fdd52d3fe4f63b048da539fce2a17a2c9718baf1"
)
_SUCCESSOR_CONTRACT_DIGEST = (
    "sha256:6a8380ee1d42b188f7bbd4818adc93949ed6f31c5981ddf6a17691d147043411"
)
_RUNTIME_CONTRACT_DIGEST = (
    "sha256:d03529c64e6ea63b9997ded35fb6c0b44c6e17fb828f9e9db8960adb764a8c6b"
)
_SOURCE_FILES = {
    "performance/structural_hypothesis_adopted_successor_materializer.py": (
        "0d3fd169e31937f597a2a1b71d1106013125977f9cb197a8218e4357056fe55f"
    ),
    "performance/structural_hypothesis_single_task_runtime.py": (
        "618c336aee3558efb05a5415201cdf6b5cb1a7e028b90d94f2ac204b072e7fe4"
    ),
}
_SOURCE_CONTRACTS = {
    "successor_contract_id": _SUCCESSOR_CONTRACT_ID,
    "successor_contract_digest": _SUCCESSOR_CONTRACT_DIGEST,
    "runtime_contract_id": "structural_hypothesis_single_task_runtime_v1",
    "runtime_contract_digest": _RUNTIME_CONTRACT_DIGEST,
}
_OPERATIONS = {
    "inspect": "read_only_exact_successor_and_adoption_full_chain",
    "prepare": "runtime_v1_authorized_attempt_without_execution",
    "verify": "runtime_v1_verified_authorized_not_executed_only",
}
_PROVENANCE_POLICY = {
    "schema_version": PROVENANCE_SCHEMA_VERSION,
    "fields": {
        "bridge_contract": ["id", "digest"],
        "source_adoption": ["id", "digest"],
        "source_successor": [
            "id",
            "digest",
            "pending_evidence_digest",
            "first_pending_projection_digest",
        ],
        "bundle_binding": [
            "bundle_id",
            "bundle_digest",
            "plan_id",
            "plan_digest",
            "task_count",
        ],
        "task_binding": ["task_id", "task_digest", "ordinal", "cell"],
        "attempt_binding": [
            "attempt_root",
            "checkpoint_root",
            "runtime_contract_id",
            "runtime_contract_digest",
        ],
    },
    "digest": "sha256-canonical-json-v1",
    "authorization_id": "successor-bound-v1:<provenance-binding-hex>",
}
_ADMISSION = {
    "independent_adoption_digest_required": True,
    "independent_successor_digest_required": True,
    "independent_pending_evidence_digest_required": True,
    "independent_first_pending_projection_digest_required": True,
    "independent_bundle_digest_required": True,
    "independent_plan_digest_required": True,
    "independent_task_digest_required": True,
    "held_dirfd_exact_successor_capture_required": True,
    "full_successor_and_adoption_verification_required": True,
    "same_generation_recheck_required": True,
    "exact_first_task_only": True,
    "exact_successor_future_attempt_root_required": True,
    "exact_authorization_id_required": True,
    "exact_provenance_binding_digest_required_before_prepare": True,
}
_STATE_BOUNDARY = {
    "inspected_status": (
        "INSPECTED_SUCCESSOR_BOUND_TASK_NOT_AUTHORIZED_NOT_PREPARED"
    ),
    "inspect_creates_state": False,
    "inspect_artifact_layout": "NONE",
    "prepare_creates_runtime_v1_authorized_attempt": True,
    "prepare_artifact_layout": (
        "exact_structural_hypothesis_single_task_runtime_v1"
    ),
    "bridge_sidecar_created": False,
    "prepare_creates_empty_checkpoint_directories": True,
    "prepare_performs_resource_preflight": False,
    "post_write_verification_failure_leaves_nonreusable_attempt_root": True,
    "atomic_rollback_claim": False,
    "prepare_executes_task": False,
    "verify_creates_state": False,
    "accepted_runtime_verify_status": "VERIFIED_AUTHORIZED_NOT_EXECUTED",
    "prepared_status": "SUCCESSOR_BOUND_AUTHORIZED_NOT_EXECUTED",
    "verified_status": "VERIFIED_SUCCESSOR_BOUND_AUTHORIZED_NOT_EXECUTED",
}
_MECHANICS = {
    "local_files_only": True,
    "network_access": False,
    "scheduler_access": False,
    "credential_access": False,
    "shell_execution": False,
    "benchmark_import": False,
    "run_one_invocation": False,
    "runtime_v1_unchanged": True,
    "captured_report_and_bundle_passed_to_runtime_prepare": True,
    "bridge_provenance_and_authorization_checks_before_attempt_creation": True,
    "mutable_current_pointer_written": False,
}
_NONCLAIMS = [
    "inspection_is_not_authorization",
    "inspection_is_not_attempt_preparation",
    "local_authorization_is_not_external_authority",
    "local_digest_is_not_signature",
    "authorized_is_not_executed",
    "authorized_is_not_preflight_passed",
    "authorized_is_not_runtime_ready",
    "successor_binding_is_not_currentness",
    "successor_binding_is_not_scientific_validation",
    "checkpoint_directory_creation_is_not_execution",
    "no_external_authority",
    "no_network_access",
    "no_scheduler_access",
    "no_credential_access",
    "no_shell_execution",
    "no_benchmark_execution",
    "no_run_one_invocation",
    "no_reingestion",
    "no_global_current_selection",
    "no_exactly_once_execution_claim",
    "attempt_alone_is_not_successor_provenance",
    "external_bridge_contract_and_full_original_successor_adoption_chain_plus_independent_anchors_required",
    "raw_runtime_execute_is_not_a_successor_provenance_gate_without_immediately_preceding_bridge_verify",
    "bridge_verify_is_a_temporal_observation_not_execution_reservation_or_atomic_handoff",
]

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_AUTHORIZATION_ID = re.compile(r"successor-bound-v1:[0-9a-f]{64}\Z")

_INSPECTED = "INSPECTED_SUCCESSOR_BOUND_TASK_NOT_AUTHORIZED_NOT_PREPARED"
_PREPARED = "SUCCESSOR_BOUND_AUTHORIZED_NOT_EXECUTED"
_VERIFIED = "VERIFIED_SUCCESSOR_BOUND_AUTHORIZED_NOT_EXECUTED"


class SuccessorBoundSingleTaskError(ValueError):
    """Raised when the successor-bound bridge fails closed."""


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _raw_digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _canonical_file(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _require_digest(value: Any, label: str) -> str:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise SuccessorBoundSingleTaskError(
            f"{label} must be a lowercase sha256 digest"
        )
    return value


def _translate(action):
    try:
        return action()
    except (
        _SUCCESSOR_ERROR,
        _RUNTIME_ERROR,
        _ADOPTION_ERROR,
    ) as error:
        raise SuccessorBoundSingleTaskError(str(error)) from error


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _validate_source_files() -> None:
    root = _repo_root()
    for relative, expected in _SOURCE_FILES.items():
        path = root / relative
        if path != path.resolve(strict=False):
            raise SuccessorBoundSingleTaskError(
                f"bridge source path is aliased: {relative}"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise SuccessorBoundSingleTaskError(
                f"cannot read bridge source file: {relative}"
            ) from error
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) & 0o022
            ):
                raise SuccessorBoundSingleTaskError(
                    f"bridge source file type, owner, mode, or links differ: {relative}"
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
                raise SuccessorBoundSingleTaskError(
                    f"bridge source changed while read: {relative}"
                )
            raw = b"".join(chunks)
            if len(raw) != before.st_size:
                raise SuccessorBoundSingleTaskError(
                    f"bridge source byte count changed: {relative}"
                )
        finally:
            os.close(descriptor)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise SuccessorBoundSingleTaskError(
                f"bridge source SHA-256 differs: {relative}"
            )


def validate_bridge_contract(contract: Mapping[str, Any]) -> None:
    if type(contract) is not dict:
        raise SuccessorBoundSingleTaskError(
            "bridge contract must be an exact object"
        )
    if (
        contract.get("schema_version") != BRIDGE_SCHEMA_VERSION
        or contract.get("contract_id") != BRIDGE_CONTRACT_ID
        or contract.get("source_contracts") != _SOURCE_CONTRACTS
        or contract.get("source_files") != _SOURCE_FILES
        or contract.get("operations") != _OPERATIONS
        or contract.get("provenance_binding") != _PROVENANCE_POLICY
        or contract.get("admission") != _ADMISSION
        or contract.get("state_boundary") != _STATE_BOUNDARY
        or contract.get("mechanics") != _MECHANICS
        or contract.get("nonclaims") != _NONCLAIMS
        or set(contract) != {
            "schema_version",
            "contract_id",
            "source_contracts",
            "source_files",
            "operations",
            "provenance_binding",
            "admission",
            "state_boundary",
            "mechanics",
            "nonclaims",
        }
        or _digest(contract) != _BRIDGE_CONTRACT_DIGEST
    ):
        raise SuccessorBoundSingleTaskError(
            "bridge contract differs from frozen V1"
        )
    _validate_source_files()


def _absolute(value: str | Path, label: str) -> Path:
    return _translate(lambda: _S_ABSOLUTE_PATH(value, label))


def _capture_successor(
    successor_root: Path, successor_id: str
) -> dict[str, Any]:
    root = _translate(
        lambda: _S_VALIDATE_LOCATION(
            successor_root, successor_id, fresh=False
        )
    )
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        root_fd = os.open(root, flags)
    except OSError as error:
        raise SuccessorBoundSingleTaskError(
            "cannot open source successor root"
        ) from error
    try:
        _translate(
            lambda: _A_CHECK_OPEN_DIRECTORY(
                root_fd, "source successor root"
            )
        )
        raws = {
            label: _translate(
                lambda name=name, label=label: _A_READ_REGULAR_AT(
                    root_fd, name, f"source successor {label}"
                )
            )
            for label, name in {
                "successor_contract": "successor_contract.json",
                "task_bundle": "bundle.json",
                "successor_commit": "successor.json",
            }.items()
        }
        _translate(
            lambda: _A_REQUIRE_NAMES(
                root_fd,
                {"successor_contract.json", "bundle.json", "successor.json"},
                "source successor root",
            )
        )
    finally:
        os.close(root_fd)
    values = {
        label: _translate(
            lambda raw=raw, label=label: _S_PARSE_OBJECT(raw, label)
        )
        for label, raw in raws.items()
    }
    return {"root": root, "raws": raws, "values": values}


def _same_successor(captured: Mapping[str, Any]) -> None:
    again = _capture_successor(
        captured["root"], captured["values"]["successor_commit"]["successor_id"]
    )
    if again["raws"] != captured["raws"]:
        raise SuccessorBoundSingleTaskError(
            "source successor changed during bridge operation"
        )


def _validate_captured_header(
    captured: Mapping[str, Any],
    *,
    external_contract_raw: bytes,
    adoption_id: str,
    successor_id: str,
    attempt_root: Path,
    expected_adoption_digest: str,
    expected_pending_evidence_digest: str,
    expected_first_pending_projection_digest: str,
    expected_successor_digest: str,
    expected_bundle_digest: str,
    expected_plan_digest: str,
    task_id: str,
    expected_task_digest: str,
) -> dict[str, Any]:
    raws = captured["raws"]
    values = captured["values"]
    contract = values["successor_contract"]
    bundle = values["task_bundle"]
    marker = values["successor_commit"]
    for label in ("task_bundle", "successor_commit"):
        value = values[label]
        if raws[label] != _canonical_file(value):
            raise SuccessorBoundSingleTaskError(
                f"captured {label} is not canonical JSON"
            )
    if raws["successor_contract"] != external_contract_raw:
        raise SuccessorBoundSingleTaskError(
            "captured and external successor contracts differ"
        )
    _translate(lambda: _S_VALIDATE_CONTRACT(contract))
    integrity = marker.get("integrity")
    body = {key: marker[key] for key in marker if key != "integrity"}
    if (
        type(integrity) is not dict
        or set(integrity) != {"algorithm", "successor_digest"}
        or integrity.get("algorithm") != "sha256-canonical-json-v1"
        or _digest(body) != integrity.get("successor_digest")
        or integrity.get("successor_digest") != expected_successor_digest
        or marker.get("schema_version")
        != _SUCCESSOR_CAPSULE_SCHEMA_VERSION
        or marker.get("status")
        != "SUCCESSOR_MATERIALIZED_FROM_VERIFIED_ADOPTION_NOT_AUTHORIZED"
        or marker.get("authorization_status") != "NOT_AUTHORIZED"
        or marker.get("successor_id") != successor_id
        or captured["root"].name != successor_id
        or marker.get("nonclaims") != list(_SUCCESSOR_NONCLAIMS)
    ):
        raise SuccessorBoundSingleTaskError(
            "captured successor marker or independent digest differs"
        )
    artifacts = {
        "successor_contract": _S_ARTIFACT(
            "successor_contract.json", raws["successor_contract"]
        ),
        "task_bundle": _S_ARTIFACT(
            "bundle.json", raws["task_bundle"]
        ),
    }
    contract_binding = marker.get("successor_contract_binding")
    if (
        marker.get("artifacts") != artifacts
        or contract_binding
        != {
            "contract_id": contract["contract_id"],
            "contract_digest": _digest(contract),
            "raw_sha256": _raw_digest(raws["successor_contract"]),
            "bytes": len(raws["successor_contract"]),
        }
    ):
        raise SuccessorBoundSingleTaskError(
            "captured successor raw artifact map differs"
        )
    source = marker.get("source_adoption")
    pending = marker.get("pending_binding")
    binding = marker.get("bundle_binding")
    future = marker.get("future_attempt_binding")
    plan = bundle.get("plan")
    tasks = plan.get("tasks") if isinstance(plan, Mapping) else None
    if type(tasks) is not list or not tasks or type(tasks[0]) is not dict:
        raise SuccessorBoundSingleTaskError("successor has no exact first task")
    first = tasks[0]
    bundle_body = {key: bundle[key] for key in bundle if key != "integrity"}
    plan_body = {key: plan[key] for key in plan if key != "integrity"}
    if (
        source
        != {
            "adoption_id": adoption_id,
            "adoption_digest": expected_adoption_digest,
            "adoption_status": "ADOPTED_AS_LOCAL_REPORT_VERSION_NOT_PLANNED",
        }
        or not isinstance(pending, Mapping)
        or pending.get("pending_evidence_digest")
        != expected_pending_evidence_digest
        or pending.get("first_pending_projection_digest")
        != expected_first_pending_projection_digest
        or pending.get("first_task_id") != task_id
        or pending.get("first_task_digest") != expected_task_digest
        or not isinstance(binding, Mapping)
        or binding.get("bundle_id") != bundle.get("bundle_id")
        or binding.get("bundle_digest") != expected_bundle_digest
        or binding.get("plan_digest") != expected_plan_digest
        or binding.get("task_count") != bundle.get("task_count")
        or bundle.get("integrity", {}).get("bundle_digest")
        != expected_bundle_digest
        or _digest(bundle_body) != expected_bundle_digest
        or plan.get("integrity", {}).get("plan_digest") != expected_plan_digest
        or _digest(plan_body) != expected_plan_digest
        or bundle.get("task_count") != len(tasks)
        or first.get("task_id") != task_id
        or first.get("task_digest") != expected_task_digest
        or first.get("ordinal") != 0
        or first.get("status") != "READY_FOR_AUTHORIZATION"
        or future
        != {
            "future_attempt_root": str(attempt_root),
            "checkpoint_root": str(attempt_root / "checkpoints"),
            "future_attempt_absent_at_materialization": True,
            "future_attempt_created": False,
            "checkpoint_root_created": False,
        }
    ):
        raise SuccessorBoundSingleTaskError(
            "captured successor task, bundle, or future-attempt binding differs"
        )
    return first


def _validate_full_generation(
    captured_successor: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    adoption_id: str,
    successor_id: str,
    expected_adoption_digest: str,
    expected_pending_evidence_digest: str,
    expected_first_pending_projection_digest: str,
    require_attempt_absent: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    successor_paths = _S_PATHS(
        paths["publication_root"],
        paths["adoption_contract_path"],
        paths["adoption_root"],
        paths["successor_contract_path"],
        paths["successor_root"],
        paths["base_evidence_csv"],
        paths["source_attempt_root"],
        paths["hypothesis_contract_path"],
        paths["executor_contract_path"],
        paths["runtime_contract_path"],
        paths["publisher_contract_path"],
        paths["materializer_contract_path"],
        paths["base_manifest_path"],
        paths["asset_root"],
        paths["attempt_root"],
    )
    try:
        adoption, bundle, pending, _raw, future, checkpoint = (
            _S_VALIDATE_AND_MATERIALIZE(
                successor_paths,
                adoption_id=adoption_id,
                successor_id=successor_id,
                expected_adoption_digest=expected_adoption_digest,
                expected_pending_evidence_digest=(
                    expected_pending_evidence_digest
                ),
                expected_first_pending_projection_digest=(
                    expected_first_pending_projection_digest
                ),
                require_future_absent=require_attempt_absent,
            )
        )
    except (ValueError, KeyError, TypeError) as error:
        raise SuccessorBoundSingleTaskError(
            "source successor/adoption failed full verification"
        ) from error
    captured_bundle = captured_successor["values"]["task_bundle"]
    marker = captured_successor["values"]["successor_commit"]
    marker_body = {key: marker[key] for key in marker if key != "integrity"}
    rebuilt = _S_MARKER_BODY(
        successor_id=successor_id,
        successor_contract=captured_successor["values"]["successor_contract"],
        successor_contract_raw=captured_successor["raws"]["successor_contract"],
        adoption_marker=adoption["adoption_marker"],
        bundle=bundle,
        pending_binding=pending,
        future_attempt_root=future,
        checkpoint_root=checkpoint,
        artifacts=marker["artifacts"],
    )
    if (
        captured_bundle != bundle
        or captured_successor["raws"]["task_bundle"] != _canonical_file(bundle)
        or marker_body != rebuilt
    ):
        raise SuccessorBoundSingleTaskError(
            "captured successor does not reproduce the full adoption generation"
        )
    _same_successor(captured_successor)
    _translate(lambda: _S_SAME_ADOPTION(adoption))
    return adoption, bundle, pending


def _provenance_binding(
    *,
    bridge_contract: Mapping[str, Any],
    adoption_id: str,
    successor_id: str,
    successor_digest: str,
    expected_adoption_digest: str,
    pending_evidence_digest: str,
    first_pending_projection_digest: str,
    bundle: Mapping[str, Any],
    task: Mapping[str, Any],
    attempt_root: Path,
    runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "bridge_contract": {
            "id": bridge_contract["contract_id"],
            "digest": _digest(bridge_contract),
        },
        "source_adoption": {
            "id": adoption_id,
            "digest": expected_adoption_digest,
        },
        "source_successor": {
            "id": successor_id,
            "digest": successor_digest,
            "pending_evidence_digest": pending_evidence_digest,
            "first_pending_projection_digest": (
                first_pending_projection_digest
            ),
        },
        "bundle_binding": {
            "bundle_id": bundle["bundle_id"],
            "bundle_digest": bundle["integrity"]["bundle_digest"],
            "plan_id": bundle["plan"]["plan_id"],
            "plan_digest": bundle["plan"]["integrity"]["plan_digest"],
            "task_count": bundle["task_count"],
        },
        "task_binding": {
            "task_id": task["task_id"],
            "task_digest": task["task_digest"],
            "ordinal": 0,
            "cell": task["cell"],
        },
        "attempt_binding": {
            "attempt_root": str(attempt_root),
            "checkpoint_root": str(attempt_root / "checkpoints"),
            "runtime_contract_id": runtime_contract["contract_id"],
            "runtime_contract_digest": _digest(runtime_contract),
        },
    }


def _derive(
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
    attempt_root,
    *,
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
    require_attempt_absent: bool,
) -> dict[str, Any]:
    expected = {
        "expected_adoption_digest": expected_adoption_digest,
        "expected_pending_evidence_digest": expected_pending_evidence_digest,
        "expected_first_pending_projection_digest": (
            expected_first_pending_projection_digest
        ),
        "expected_successor_digest": expected_successor_digest,
        "expected_bundle_digest": expected_bundle_digest,
        "expected_plan_digest": expected_plan_digest,
        "expected_task_digest": expected_task_digest,
    }
    for label, value in expected.items():
        expected[label] = _require_digest(value, label)
    paths = {
        label: _absolute(value, label)
        for label, value in {
            "publication_root": publication_root,
            "adoption_contract_path": adoption_contract_path,
            "adoption_root": adoption_root,
            "successor_contract_path": successor_contract_path,
            "successor_root": successor_root,
            "base_evidence_csv": base_evidence_csv,
            "source_attempt_root": source_attempt_root,
            "hypothesis_contract_path": hypothesis_contract_path,
            "executor_contract_path": executor_contract_path,
            "runtime_contract_path": runtime_contract_path,
            "publisher_contract_path": publisher_contract_path,
            "materializer_contract_path": materializer_contract_path,
            "bridge_contract_path": bridge_contract_path,
            "base_manifest_path": base_manifest_path,
            "asset_root": asset_root,
            "attempt_root": attempt_root,
        }.items()
    }
    bridge_contract, _bridge_raw = _translate(
        lambda: _S_READ_JSON(
            paths["bridge_contract_path"], "bridge contract"
        )
    )
    validate_bridge_contract(bridge_contract)
    external_successor_contract, external_successor_raw = _translate(
        lambda: _S_READ_JSON(
            paths["successor_contract_path"], "successor contract"
        )
    )
    _translate(
        lambda: _S_VALIDATE_CONTRACT(
            external_successor_contract
        )
    )
    runtime_contract, _runtime_raw = _translate(
        lambda: _S_READ_JSON(
            paths["runtime_contract_path"], "runtime contract"
        )
    )
    _translate(lambda: _R_VALIDATE_CONTRACT(runtime_contract))
    if _digest(runtime_contract) != _RUNTIME_CONTRACT_DIGEST:
        raise SuccessorBoundSingleTaskError("runtime contract digest differs")
    attempt, _checkpoint = _translate(
        lambda: _S_VALIDATE_FUTURE_ATTEMPT(
            paths["attempt_root"],
            runtime_contract,
            successor_id=successor_id,
            require_absent=require_attempt_absent,
        )
    )
    captured_successor = _capture_successor(
        paths["successor_root"], successor_id
    )
    first = _validate_captured_header(
        captured_successor,
        external_contract_raw=external_successor_raw,
        adoption_id=adoption_id,
        successor_id=successor_id,
        attempt_root=attempt,
        expected_adoption_digest=expected["expected_adoption_digest"],
        expected_pending_evidence_digest=expected[
            "expected_pending_evidence_digest"
        ],
        expected_first_pending_projection_digest=expected[
            "expected_first_pending_projection_digest"
        ],
        expected_successor_digest=expected["expected_successor_digest"],
        expected_bundle_digest=expected["expected_bundle_digest"],
        expected_plan_digest=expected["expected_plan_digest"],
        task_id=task_id,
        expected_task_digest=expected["expected_task_digest"],
    )
    adoption, bundle, pending = _validate_full_generation(
        captured_successor,
        paths,
        adoption_id=adoption_id,
        successor_id=successor_id,
        expected_adoption_digest=expected["expected_adoption_digest"],
        expected_pending_evidence_digest=expected[
            "expected_pending_evidence_digest"
        ],
        expected_first_pending_projection_digest=expected[
            "expected_first_pending_projection_digest"
        ],
        require_attempt_absent=require_attempt_absent,
    )
    first = bundle["plan"]["tasks"][0]
    provenance = _provenance_binding(
        bridge_contract=bridge_contract,
        adoption_id=adoption_id,
        successor_id=successor_id,
        successor_digest=expected["expected_successor_digest"],
        expected_adoption_digest=expected["expected_adoption_digest"],
        pending_evidence_digest=pending["pending_evidence_digest"],
        first_pending_projection_digest=pending[
            "first_pending_projection_digest"
        ],
        bundle=bundle,
        task=first,
        attempt_root=attempt,
        runtime_contract=runtime_contract,
    )
    provenance_digest = _digest(provenance)
    required_authorization_id = (
        "successor-bound-v1:" + provenance_digest.split(":", 1)[1]
    )
    return {
        "paths": paths,
        "bridge_contract": bridge_contract,
        "captured_successor": captured_successor,
        "captured_adoption": adoption,
        "bundle": bundle,
        "task": first,
        "runtime_contract": runtime_contract,
        "provenance_binding": provenance,
        "provenance_binding_digest": provenance_digest,
        "required_authorization_id": required_authorization_id,
    }


def _inspect_result(derived: Mapping[str, Any]) -> dict[str, Any]:
    provenance = derived["provenance_binding"]
    adoption = provenance["source_adoption"]
    successor = provenance["source_successor"]
    bundle = provenance["bundle_binding"]
    task = provenance["task_binding"]
    attempt = provenance["attempt_binding"]
    return {
        "status": _INSPECTED,
        "authorization_status": "NOT_AUTHORIZED",
        "attempt_status": "NOT_PREPARED",
        "provenance_binding": provenance,
        "provenance_binding_digest": derived["provenance_binding_digest"],
        "required_authorization_id": derived["required_authorization_id"],
        "adoption_digest": adoption["digest"],
        "successor_digest": successor["digest"],
        "pending_evidence_digest": successor["pending_evidence_digest"],
        "first_pending_projection_digest": successor[
            "first_pending_projection_digest"
        ],
        "bundle_digest": bundle["bundle_digest"],
        "plan_digest": bundle["plan_digest"],
        "task_count": bundle["task_count"],
        "task_id": task["task_id"],
        "task_digest": task["task_digest"],
        "attempt_root": attempt["attempt_root"],
        "checkpoint_root": attempt["checkpoint_root"],
    }


def inspect_successor_bound_single_task(
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
    attempt_root,
    *,
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
) -> dict[str, Any]:
    """Inspect the exact first successor task without creating state."""
    derived = _derive(
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
        attempt_root,
        adoption_id=adoption_id,
        successor_id=successor_id,
        expected_adoption_digest=expected_adoption_digest,
        expected_pending_evidence_digest=expected_pending_evidence_digest,
        expected_first_pending_projection_digest=(
            expected_first_pending_projection_digest
        ),
        expected_successor_digest=expected_successor_digest,
        expected_bundle_digest=expected_bundle_digest,
        expected_plan_digest=expected_plan_digest,
        task_id=task_id,
        expected_task_digest=expected_task_digest,
        require_attempt_absent=True,
    )
    return _inspect_result(derived)


def _assert_runtime_chain(
    derived: Mapping[str, Any], *, expected_authorization_digest: str
) -> dict[str, Any]:
    paths = derived["paths"]
    chain = _translate(
        lambda: _R_EXPECTED_ATTEMPT(
            paths["attempt_root"],
            derived["runtime_contract"],
            base_manifest_path=paths["base_manifest_path"],
            asset_root=paths["asset_root"],
        )
    )
    captured = derived["captured_adoption"]["publication_values"]
    if (
        chain["authorization"].get("authorization_id")
        != derived["required_authorization_id"]
        or chain["authorization"].get("integrity", {}).get(
            "authorization_digest"
        )
        != expected_authorization_digest
        or chain["bundle"] != derived["bundle"]
        or chain["report"] != captured["output_report"]
        or chain["hypothesis_contract"]
        != captured["source_hypothesis_contract"]
        or chain["executor_contract"] != captured["source_executor_contract"]
        or chain["materializer_contract"]
        != captured["materializer_contract"]
    ):
        raise SuccessorBoundSingleTaskError(
            "runtime attempt differs from captured successor provenance"
        )
    return chain


def prepare_successor_bound_single_task_attempt(
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
    attempt_root,
    *,
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
    authorization_id: str,
) -> dict[str, Any]:
    """Create exactly one successor-bound runtime-V1 AUTHORIZED attempt."""
    expected_provenance_binding_digest = _require_digest(
        expected_provenance_binding_digest,
        "expected_provenance_binding_digest",
    )
    derived = _derive(
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
        attempt_root,
        adoption_id=adoption_id,
        successor_id=successor_id,
        expected_adoption_digest=expected_adoption_digest,
        expected_pending_evidence_digest=expected_pending_evidence_digest,
        expected_first_pending_projection_digest=(
            expected_first_pending_projection_digest
        ),
        expected_successor_digest=expected_successor_digest,
        expected_bundle_digest=expected_bundle_digest,
        expected_plan_digest=expected_plan_digest,
        task_id=task_id,
        expected_task_digest=expected_task_digest,
        require_attempt_absent=True,
    )
    if (
        derived["provenance_binding_digest"]
        != expected_provenance_binding_digest
        or type(authorization_id) is not str
        or not _AUTHORIZATION_ID.fullmatch(authorization_id)
        or authorization_id != derived["required_authorization_id"]
    ):
        raise SuccessorBoundSingleTaskError(
            "independent provenance digest or exact authorization ID differs"
        )
    _same_successor(derived["captured_successor"])
    _translate(lambda: _S_SAME_ADOPTION(derived["captured_adoption"]))
    captured = derived["captured_adoption"]["publication_values"]
    prepared = _translate(
        lambda: _R_PREPARE_ATTEMPT(
            captured["output_report"],
            derived["bundle"],
            captured["source_hypothesis_contract"],
            captured["source_executor_contract"],
            captured["materializer_contract"],
            derived["runtime_contract"],
            derived["paths"]["base_manifest_path"],
            derived["paths"]["asset_root"],
            derived["paths"]["attempt_root"],
            task_id=derived["task"]["task_id"],
            expected_bundle_digest=derived["bundle"]["integrity"][
                "bundle_digest"
            ],
            expected_plan_digest=derived["bundle"]["plan"]["integrity"][
                "plan_digest"
            ],
            authorization_id=authorization_id,
        )
    )
    if prepared.get("status") != "AUTHORIZED":
        raise SuccessorBoundSingleTaskError(
            "runtime V1 did not create an AUTHORIZED attempt"
        )
    authorization_digest = prepared.get("authorization_digest")
    _require_digest(authorization_digest, "authorization_digest")
    _assert_runtime_chain(
        derived, expected_authorization_digest=authorization_digest
    )
    verified = _translate(
        lambda: _R_VERIFY_ATTEMPT(
            derived["paths"]["attempt_root"],
            derived["runtime_contract"],
            derived["paths"]["base_manifest_path"],
            derived["paths"]["asset_root"],
            expected_authorization_digest=authorization_digest,
        )
    )
    if verified.get("status") != "VERIFIED_AUTHORIZED_NOT_EXECUTED":
        raise SuccessorBoundSingleTaskError(
            "prepared runtime state is not AUTHORIZED-only"
        )
    result = _inspect_result(derived)
    result.update(
        {
            "status": _PREPARED,
            "authorization_status": "AUTHORIZED",
            "attempt_status": "AUTHORIZED_NOT_EXECUTED",
            "authorization_digest": authorization_digest,
            "attempt_digest": verified["attempt_digest"],
        }
    )
    return result


def verify_successor_bound_single_task_attempt(
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
    attempt_root,
    *,
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
) -> dict[str, Any]:
    """Verify only a successor-bound AUTHORIZED, never-executed attempt."""
    expected_provenance_binding_digest = _require_digest(
        expected_provenance_binding_digest,
        "expected_provenance_binding_digest",
    )
    expected_authorization_digest = _require_digest(
        expected_authorization_digest, "expected_authorization_digest"
    )
    expected_attempt_digest = _require_digest(
        expected_attempt_digest, "expected_attempt_digest"
    )
    derived = _derive(
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
        attempt_root,
        adoption_id=adoption_id,
        successor_id=successor_id,
        expected_adoption_digest=expected_adoption_digest,
        expected_pending_evidence_digest=expected_pending_evidence_digest,
        expected_first_pending_projection_digest=(
            expected_first_pending_projection_digest
        ),
        expected_successor_digest=expected_successor_digest,
        expected_bundle_digest=expected_bundle_digest,
        expected_plan_digest=expected_plan_digest,
        task_id=task_id,
        expected_task_digest=expected_task_digest,
        require_attempt_absent=False,
    )
    if derived["provenance_binding_digest"] != expected_provenance_binding_digest:
        raise SuccessorBoundSingleTaskError(
            "independent provenance binding digest differs"
        )
    _assert_runtime_chain(
        derived, expected_authorization_digest=expected_authorization_digest
    )
    verified = _translate(
        lambda: _R_VERIFY_ATTEMPT(
            derived["paths"]["attempt_root"],
            derived["runtime_contract"],
            derived["paths"]["base_manifest_path"],
            derived["paths"]["asset_root"],
            expected_authorization_digest=expected_authorization_digest,
        )
    )
    if (
        verified.get("status") != "VERIFIED_AUTHORIZED_NOT_EXECUTED"
        or verified.get("attempt_digest") != expected_attempt_digest
    ):
        raise SuccessorBoundSingleTaskError(
            "runtime attempt is not the independently pinned AUTHORIZED state"
        )
    result = _inspect_result(derived)
    result.update(
        {
            "status": _VERIFIED,
            "authorization_status": "AUTHORIZED",
            "attempt_status": "AUTHORIZED_NOT_EXECUTED",
            "authorization_digest": expected_authorization_digest,
            "attempt_digest": expected_attempt_digest,
        }
    )
    return result


__all__ = [
    "BRIDGE_CONTRACT_ID",
    "BRIDGE_SCHEMA_VERSION",
    "PROVENANCE_SCHEMA_VERSION",
    "SuccessorBoundSingleTaskError",
    "inspect_successor_bound_single_task",
    "prepare_successor_bound_single_task_attempt",
    "validate_bridge_contract",
    "verify_successor_bound_single_task_attempt",
]
