"""Materialize an unauthorized successor from one verified local adoption.

The successor binds a future attempt path and its checkpoint subtree but does
not create either path.  It never authorizes, prepares, or executes a task.
``successor.json`` is the final and only commit marker.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from . import structural_hypothesis_report_adoption as _adoption
from . import structural_hypothesis_reingestion_publisher as _publisher
from . import structural_hypothesis_single_task_runtime as _runtime
from . import structural_hypothesis_task_materializer as _materializer
from .structural_hypothesis_loop import (
    canonical_json_bytes,
    run_structural_hypothesis_loop,
    verify_report_integrity,
)


SUCCESSOR_MATERIALIZER_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-adopted-successor-materializer/1"
)
SUCCESSOR_CAPSULE_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-adopted-successor-materialization/1"
)
SUCCESSOR_CONTRACT_ID = (
    "structural_hypothesis_adopted_successor_materializer_v1"
)

_SUCCESSOR_CONTRACT_DIGEST = (
    "sha256:6a8380ee1d42b188f7bbd4818adc93949ed6f31c5981ddf6a17691d147043411"
)
_ADOPTION_CONTRACT_DIGEST = (
    "sha256:66ed58df701c5e39924aa7b61574eb63126770290b1d9d1fabb60a9ebf8e13a1"
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
    "performance/structural_hypothesis_loop.py": (
        "f23a29c2f85b9bdce96398c6b901190fc34c2e2f97880dce5dbf2594b76c7635"
    ),
    "performance/structural_hypothesis_execution.py": (
        "7c5cc27f8e97da9b51e57975f63e860a23463d5e7728d1089f32978146f27c9b"
    ),
    "performance/structural_hypothesis_report_adoption.py": (
        "78acadfcd6b094cd6521ef29b49f164b58b0e6816cc257fe7dc7709ea37aa725"
    ),
    "performance/structural_hypothesis_task_materializer.py": (
        "93345ef22df0cc9c5665be35722ad4646fb233d7a8bbd1294f0ca051cf64f20b"
    ),
}

_STATE_PREFIX = Path("kg-op/structural-hypothesis-adopted-successor/v1")
_SUCCESSOR_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_STATUS = "SUCCESSOR_MATERIALIZED_FROM_VERIFIED_ADOPTION_NOT_AUTHORIZED"
_VERIFIED_STATUS = "VERIFIED_" + _STATUS
_ARTIFACT_LAYOUT = {
    "successor_contract": "successor_contract.json",
    "task_bundle": "bundle.json",
    "successor_commit": "successor.json",
}
_NONCLAIMS = [
    "successor_materialization_is_not_authorization",
    "successor_materialization_is_not_attempt_preparation",
    "successor_materialization_is_not_execution",
    "successor_materialization_is_not_global_current",
    "detached_bundle_alone_has_no_adoption_provenance",
    "transitively_derived_leaf_anchors_are_not_independent_observations",
    "original_adoption_and_full_source_chain_are_required_for_full_verification",
    "ready_for_authorization_mechanics_is_not_runtime_readiness",
    "future_attempt_absence_is_not_a_reservation",
    "no_reingestion_or_publisher_v1_compatibility_claim",
    "adoption_is_not_external_verification",
    "local_digest_is_not_signature",
    "no_external_authority",
    "no_currentness_claim",
    "no_runtime_readiness_claim",
    "no_scientific_claim",
    "future_attempt_absence_is_observed_only_at_materialization",
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


class AdoptedSuccessorMaterializationError(ValueError):
    """Raised when V1 successor materialization fails closed."""


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _raw_digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _canonical_file(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _require_digest(value: Any, label: str) -> str:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise AdoptedSuccessorMaterializationError(
            f"{label} must be a lowercase sha256 digest"
        )
    return value


def _translate(action):
    try:
        return action()
    except (
        _adoption.StructuralHypothesisReportAdoptionError,
        _materializer.MaterializationValidationError,
    ) as error:
        raise AdoptedSuccessorMaterializationError(str(error)) from error


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _absolute_path(value: str | Path, label: str) -> Path:
    return _translate(lambda: _adoption._absolute_path(value, label))


def _read_regular(
    path: Path, label: str, *, exact_mode: int | None = None
) -> bytes:
    return _translate(
        lambda: _adoption._read_regular(
            path, label, exact_mode=exact_mode
        )
    )


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    return _translate(lambda: _adoption._read_json(path, label))


def _parse_object(raw: bytes, label: str) -> dict[str, Any]:
    return _translate(lambda: _publisher._parse_json(raw, label))


def _parse_value(raw: bytes, label: str) -> Any:
    return _translate(lambda: _publisher._parse_json_value(raw, label))


def _validate_successor_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema_version")
        != SUCCESSOR_MATERIALIZER_SCHEMA_VERSION
        or contract.get("contract_id") != SUCCESSOR_CONTRACT_ID
        or contract.get("source_files") != _SOURCE_FILES
        or contract.get("artifact_layout") != _ARTIFACT_LAYOUT
        or contract.get("nonclaims") != _NONCLAIMS
        or _digest(contract) != _SUCCESSOR_CONTRACT_DIGEST
    ):
        raise AdoptedSuccessorMaterializationError(
            "successor contract differs from frozen V1"
        )
    if contract.get("source_contracts") != {
        "adoption_contract_id": _adoption.ADOPTION_CONTRACT_ID,
        "adoption_contract_digest": _ADOPTION_CONTRACT_DIGEST,
        "materializer_contract_id": (
            "structural_hypothesis_task_materializer_v1"
        ),
        "materializer_contract_digest": _MATERIALIZER_CONTRACT_DIGEST,
        "hypothesis_contract_id": "structural_hypothesis_loop_v1",
        "hypothesis_contract_digest": _HYPOTHESIS_CONTRACT_DIGEST,
        "executor_contract_id": "structural_hypothesis_executor_v1",
        "executor_contract_digest": _EXECUTOR_CONTRACT_DIGEST,
    }:
        raise AdoptedSuccessorMaterializationError(
            "successor source-contract commitments differ"
        )
    admission = contract.get("admission")
    if type(admission) is not dict or admission != {
        "required_adoption_status": (
            "ADOPTED_AS_LOCAL_REPORT_VERSION_NOT_PLANNED"
        ),
        "independent_adoption_digest_required": True,
        "held_dirfd_exact_adoption_capture_required": True,
        "captured_marker_and_artifact_map_verified_before_derived_anchors": (
            True
        ),
        "full_adoption_verification_required": True,
        "combined_rows_exact_report_replay_required": True,
        "pending_task_count_minimum": 1,
        "pending_task_count_maximum": 30,
        "task_selection": "exact_replayed_report_pending_cells_in_order",
        "first_pending_projection_independent_digest_required": True,
        "first_pending_projection_fields": [
            "profile",
            "domain",
            "line",
            "seed",
            "d",
            "N",
            "n0",
        ],
        "external_pending_and_first_projection_digests_required_at_materialization": (
            True
        ),
        "external_successor_bundle_and_plan_digests_required_at_verification": (
            True
        ),
        "materialized_status": _STATUS,
        "authorization_status": "NOT_AUTHORIZED",
    }:
        raise AdoptedSuccessorMaterializationError(
            "successor admission differs"
        )
    if contract.get("future_attempt_policy") != {
        "absolute_canonical_path_required": True,
        "runtime_relative_prefix": (
            "kg-op/structural-hypothesis-execution/v1"
        ),
        "future_attempt_is_direct_child": True,
        "future_attempt_basename_equals_successor_id": True,
        "must_be_absent_at_materialization": True,
        "absence_not_required_for_later_successor_verification": True,
        "checkpoint_root": "future_attempt_root/checkpoints",
        "future_attempt_created": False,
        "checkpoint_root_created": False,
    }:
        raise AdoptedSuccessorMaterializationError(
            "successor future-attempt policy differs"
        )
    root = _repo_root()
    for relative, expected in _SOURCE_FILES.items():
        raw = _read_regular(root / relative, f"successor source {relative}")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise AdoptedSuccessorMaterializationError(
                f"successor source file differs: {relative}"
            )


def _state_base() -> Path:
    configured = os.environ.get("XDG_STATE_HOME")
    if configured:
        base = Path(configured)
        if not base.is_absolute() or ".." in base.parts:
            raise AdoptedSuccessorMaterializationError(
                "XDG_STATE_HOME must be absolute and canonical"
            )
    else:
        base = Path.home() / ".local/state"
    if base != base.resolve(strict=False):
        raise AdoptedSuccessorMaterializationError(
            "state home contains an alias or symlink component"
        )
    return base


def _secure_directory(path: Path, *, exact_mode: int | None = None) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise AdoptedSuccessorMaterializationError(
            f"cannot inspect successor directory: {path}"
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
        raise AdoptedSuccessorMaterializationError(
            f"successor directory ownership or mode differs: {path}"
        )


def _ensure_secure_parent(target: Path) -> None:
    base = _state_base()
    prefix = base / _STATE_PREFIX
    if target.parent != prefix:
        raise AdoptedSuccessorMaterializationError(
            "successor_root is outside the frozen local state prefix"
        )
    _translate(lambda: _publisher._reject_symlink_components(target))
    missing = []
    cursor = prefix
    while not cursor.exists():
        if cursor.is_symlink():
            raise AdoptedSuccessorMaterializationError(
                f"successor directory alias is forbidden: {cursor}"
            )
        missing.append(cursor)
        if cursor == base:
            break
        cursor = cursor.parent
    if not cursor.exists():
        raise AdoptedSuccessorMaterializationError(
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


def _validate_successor_location(
    successor_root: Path, successor_id: str | None, *, fresh: bool
) -> Path:
    root = _absolute_path(successor_root, "successor_root")
    if successor_id is not None:
        if type(successor_id) is not str or not _SUCCESSOR_ID.fullmatch(
            successor_id
        ):
            raise AdoptedSuccessorMaterializationError(
                "successor_id is invalid"
            )
        if root.name != successor_id:
            raise AdoptedSuccessorMaterializationError(
                "successor_id must equal successor_root basename"
            )
    if root.parent != _state_base() / _STATE_PREFIX:
        raise AdoptedSuccessorMaterializationError(
            "successor_root is outside the frozen local state prefix"
        )
    if fresh:
        if root.exists() or root.is_symlink():
            raise AdoptedSuccessorMaterializationError(
                "successor_root already exists"
            )
        _ensure_secure_parent(root)
    else:
        _translate(lambda: _publisher._reject_symlink_components(root))
        _secure_directory(root, exact_mode=0o700)
    return root


def _validate_future_attempt(
    path: Path,
    runtime_contract: Mapping[str, Any],
    *,
    successor_id: str,
    require_absent: bool,
) -> tuple[Path, Path]:
    try:
        attempt = _runtime._path(path, "future_attempt_root")
        _base, prefix = _runtime._state_base_and_prefix(runtime_contract)
        if attempt.parent != prefix:
            raise AdoptedSuccessorMaterializationError(
                "future_attempt_root must be a direct runtime-prefix child"
            )
        _runtime._reject_symlink_components(attempt)
    except _runtime.SingleTaskRuntimeValidationError as error:
        raise AdoptedSuccessorMaterializationError(str(error)) from error
    if require_absent and (attempt.exists() or attempt.is_symlink()):
        raise AdoptedSuccessorMaterializationError(
            "future_attempt_root must be absent at materialization"
        )
    if attempt.name != successor_id:
        raise AdoptedSuccessorMaterializationError(
            "future_attempt_root basename must equal successor_id"
        )
    return attempt, attempt / "checkpoints"


def _capture_adoption(adoption_root: Path) -> dict[str, Any]:
    """Capture one adoption generation through held directory descriptors."""
    root = _translate(
        lambda: _adoption._validate_adoption_location(
            adoption_root, None, fresh=False
        )
    )
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        root_fd = os.open(root, flags)
    except OSError as error:
        raise AdoptedSuccessorMaterializationError(
            "cannot open source adoption root"
        ) from error
    descriptors = [root_fd]
    try:
        _translate(
            lambda: _adoption._check_open_directory(
                root_fd, "source adoption root"
            )
        )
        publication_fd = _translate(
            lambda: _adoption._open_directory_at(
                root_fd, "publication", "source adoption publication"
            )
        )
        descriptors.append(publication_fd)
        execution_fd = _translate(
            lambda: _adoption._open_directory_at(
                publication_fd,
                "execution",
                "source adoption execution",
            )
        )
        descriptors.append(execution_fd)
        inputs_fd = _translate(
            lambda: _adoption._open_directory_at(
                execution_fd, "inputs", "source adoption inputs"
            )
        )
        descriptors.append(inputs_fd)
        journal_fd = _translate(
            lambda: _adoption._open_directory_at(
                execution_fd, "journal", "source adoption journal"
            )
        )
        descriptors.append(journal_fd)
        directory_fds = {
            "": root_fd,
            "publication": publication_fd,
            "publication/execution": execution_fd,
            "publication/execution/inputs": inputs_fd,
            "publication/execution/journal": journal_fd,
        }
        expected_names = {key: set() for key in directory_fds}
        expected_names[""].update(
            {"adoption_contract.json", "publication", "adoption.json"}
        )
        for relative in _publisher._ARTIFACT_LAYOUT.values():
            path = Path("publication") / relative
            parent = os.fspath(path.parent)
            expected_names[parent].add(path.name)
        raws = {
            "adoption_contract": _translate(
                lambda: _adoption._read_regular_at(
                    root_fd, "adoption_contract.json", "adoption contract"
                )
            ),
            "adoption_commit": _translate(
                lambda: _adoption._read_regular_at(
                    root_fd, "adoption.json", "adoption commit"
                )
            ),
        }
        publication_raws = {}
        directory_labels = {
            "": publication_fd,
            "execution": execution_fd,
            "execution/inputs": inputs_fd,
            "execution/journal": journal_fd,
        }
        for label, relative in _adoption._publication_relative_files().items():
            path = Path(relative)
            parent = os.fspath(path.parent)
            if parent == ".":
                parent = ""
            publication_raws[label] = _translate(
                lambda parent=parent, path=path, label=label: (
                    _adoption._read_regular_at(
                        directory_labels[parent],
                        path.name,
                        f"adoption publication {label}",
                    )
                )
            )
        for relative, descriptor in directory_fds.items():
            _translate(
                lambda relative=relative, descriptor=descriptor: (
                    _adoption._require_names(
                        descriptor,
                        expected_names[relative],
                        f"adoption directory {relative or '.'}",
                    )
                )
            )
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    publication_values = {}
    for label, raw in publication_raws.items():
        if label == "base_evidence_csv":
            continue
        publication_values[label] = (
            _parse_value(raw, label)
            if label == "combined_rows"
            else _parse_object(raw, label)
        )
    if type(publication_values["combined_rows"]) is not list:
        raise AdoptedSuccessorMaterializationError(
            "adoption combined rows must be an array"
        )
    return {
        "root": root,
        "raws": raws,
        "adoption_contract": _parse_object(
            raws["adoption_contract"], "captured adoption contract"
        ),
        "adoption_marker": _parse_object(
            raws["adoption_commit"], "captured adoption marker"
        ),
        "publication_raws": publication_raws,
        "publication_values": publication_values,
    }


def _validate_captured_adoption(
    captured: Mapping[str, Any],
    *,
    adoption_id: str,
    expected_adoption_digest: str,
) -> None:
    contract = captured["adoption_contract"]
    _translate(lambda: _adoption._validate_adoption_contract(contract))
    marker = captured["adoption_marker"]
    if captured["raws"]["adoption_commit"] != _canonical_file(marker):
        raise AdoptedSuccessorMaterializationError(
            "captured adoption marker is not canonical JSON"
        )
    integrity = marker.get("integrity")
    if (
        type(integrity) is not dict
        or set(integrity) != {"algorithm", "adoption_digest"}
        or integrity.get("algorithm") != "sha256-canonical-json-v1"
    ):
        raise AdoptedSuccessorMaterializationError(
            "captured adoption integrity is malformed"
        )
    body = {key: marker[key] for key in marker if key != "integrity"}
    if (
        _digest(body) != integrity.get("adoption_digest")
        or integrity.get("adoption_digest") != expected_adoption_digest
        or marker.get("schema_version")
        != _adoption.ADOPTION_CAPSULE_SCHEMA_VERSION
        or marker.get("status")
        != "ADOPTED_AS_LOCAL_REPORT_VERSION_NOT_PLANNED"
        or marker.get("planning_status") != "NOT_PLANNED"
        or marker.get("adoption_id") != adoption_id
        or captured["root"].name != adoption_id
    ):
        raise AdoptedSuccessorMaterializationError(
            "captured adoption marker or independent digest differs"
        )
    artifacts = {
        "adoption_contract": _adoption._artifact(
            "adoption_contract.json", captured["raws"]["adoption_contract"]
        )
    }
    for label, relative in _adoption._publication_relative_files().items():
        artifacts[f"publication:{label}"] = _adoption._artifact(
            f"publication/{relative}", captured["publication_raws"][label]
        )
    if marker.get("artifacts") != artifacts:
        raise AdoptedSuccessorMaterializationError(
            "captured adoption artifact map differs"
        )
    rebuilt = _adoption._marker_body(
        adoption_id=adoption_id,
        adoption_contract=contract,
        adoption_contract_raw=captured["raws"]["adoption_contract"],
        publication_marker=captured["publication_values"][
            "publication_commit"
        ],
        publication_raws=captured["publication_raws"],
        output_report=captured["publication_values"]["output_report"],
        combined_rows=captured["publication_values"]["combined_rows"],
        artifacts=artifacts,
    )
    if body != rebuilt:
        raise AdoptedSuccessorMaterializationError(
            "captured adoption marker does not reproduce exact raw leaves"
        )


def _derived_adoption_expectations(
    captured: Mapping[str, Any], expected_adoption_digest: str
) -> dict[str, str]:
    publication = captured["publication_values"]["publication_commit"]
    source = publication["source_binding"]
    outputs = publication["outputs"]
    return {
        "expected_source_evidence_digest": source["source_report"][
            "evidence_digest"
        ],
        "expected_plan_digest": source["plan_digest"],
        "expected_authorization_digest": source["authorization_digest"],
        "expected_execution_receipt_digest": source[
            "execution_receipt_digest"
        ],
        "expected_execution_journal_head_digest": source[
            "execution_journal_head_digest"
        ],
        "expected_execution_attempt_digest": source[
            "execution_attempt_digest"
        ],
        "expected_publication_digest": publication["integrity"][
            "publication_digest"
        ],
        "expected_reingestion_digest": outputs["reingestion_digest"],
        "expected_output_report_body_digest": outputs[
            "output_report_body_digest"
        ],
        "expected_output_audit_head": outputs["output_audit_head"],
        "expected_output_evidence_digest": outputs[
            "output_evidence_digest"
        ],
        "expected_publication_marker_raw_sha256": _raw_digest(
            captured["publication_raws"]["publication_commit"]
        ),
        "expected_combined_rows_raw_sha256": _raw_digest(
            captured["publication_raws"]["combined_rows"]
        ),
        "expected_output_report_raw_sha256": _raw_digest(
            captured["publication_raws"]["output_report"]
        ),
        "expected_reingestion_receipt_raw_sha256": _raw_digest(
            captured["publication_raws"]["reingestion_receipt"]
        ),
        "expected_adoption_digest": expected_adoption_digest,
    }


def _full_verify_adoption(
    captured: Mapping[str, Any],
    *,
    publication_root: Path,
    adoption_contract_path: Path,
    adoption_root: Path,
    base_evidence_csv: Path,
    source_attempt_root: Path,
    hypothesis_contract_path: Path,
    executor_contract_path: Path,
    runtime_contract_path: Path,
    publisher_contract_path: Path,
    base_manifest_path: Path,
    asset_root: Path,
    adoption_id: str,
    expected_adoption_digest: str,
) -> None:
    try:
        verified = _adoption.verify_structural_hypothesis_report_adoption(
            publication_root,
            adoption_contract_path,
            adoption_root,
            base_evidence_csv,
            source_attempt_root,
            hypothesis_contract_path,
            executor_contract_path,
            runtime_contract_path,
            publisher_contract_path,
            base_manifest_path,
            asset_root,
            adoption_id=adoption_id,
            **_derived_adoption_expectations(
                captured, expected_adoption_digest
            ),
        )
    except (ValueError, KeyError, TypeError) as error:
        raise AdoptedSuccessorMaterializationError(
            "source adoption failed full verification"
        ) from error
    if verified.get("status") != (
        "VERIFIED_ADOPTED_AS_LOCAL_REPORT_VERSION_NOT_PLANNED"
    ):
        raise AdoptedSuccessorMaterializationError(
            "source adoption is not verified"
        )


def _same_adoption(captured: Mapping[str, Any]) -> None:
    again = _capture_adoption(captured["root"])
    if (
        again["raws"] != captured["raws"]
        or again["publication_raws"] != captured["publication_raws"]
    ):
        raise AdoptedSuccessorMaterializationError(
            "source adoption changed during successor materialization"
        )


def _replay_report(captured: Mapping[str, Any]) -> tuple[
    dict[str, Any], list[Any], dict[str, Any], dict[str, Any]
]:
    combined = captured["publication_values"]["combined_rows"]
    report = captured["publication_values"]["output_report"]
    hypothesis = captured["publication_values"][
        "source_hypothesis_contract"
    ]
    executor = captured["publication_values"]["source_executor_contract"]
    replayed = run_structural_hypothesis_loop(
        combined, hypothesis, input_artifacts=None
    ).to_dict()
    if (
        replayed != report
        or not verify_report_integrity(replayed)
        or replayed.get("evidence_digest") != _digest(combined)
    ):
        raise AdoptedSuccessorMaterializationError(
            "combined rows do not exactly replay the adopted report"
        )
    return replayed, combined, hypothesis, executor


def _validate_external_materializer(
    path: Path, captured: Mapping[str, Any]
) -> tuple[dict[str, Any], bytes]:
    contract, raw = _read_json(path, "materializer contract")
    captured_contract = captured["publication_values"][
        "materializer_contract"
    ]
    if contract != captured_contract:
        raise AdoptedSuccessorMaterializationError(
            "external materializer contract differs from adoption"
        )
    _translate(lambda: _materializer.validate_materializer_contract(contract))
    return contract, raw


def _materialize_bundle(
    captured: Mapping[str, Any],
    *,
    materializer_contract: Mapping[str, Any],
    base_manifest_path: Path,
    asset_root: Path,
    checkpoint_root: Path,
    expected_pending_evidence_digest: str,
    expected_first_pending_projection_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report, _combined, hypothesis, executor = _replay_report(captured)
    pending = report["pending_evidence"]
    if not 1 <= len(pending) <= 30:
        raise AdoptedSuccessorMaterializationError(
            "replayed pending evidence count is outside V1"
        )
    pending_digest = _digest(pending)
    first_projection = {
        "profile": pending[0]["profile"],
        "domain": pending[0]["domain"],
        "line": executor["execution_scope"]["line"],
        "seed": pending[0]["seed"],
        "d": pending[0]["d"],
        "N": pending[0]["N"],
        "n0": pending[0]["n0"],
    }
    first_digest = _digest(first_projection)
    if (
        pending_digest != expected_pending_evidence_digest
        or first_digest != expected_first_pending_projection_digest
    ):
        raise AdoptedSuccessorMaterializationError(
            "independent pending evidence anchor differs"
        )
    bundle = _translate(
        lambda: _materializer.materialize_task_bundle(
            report,
            hypothesis,
            executor,
            materializer_contract,
            base_manifest_path,
            asset_root,
            checkpoint_root,
        )
    )
    if (
        bundle.get("status") != "MATERIALIZED_NOT_AUTHORIZED"
        or bundle.get("task_count") != len(pending)
        or bundle.get("plan", {}).get("proposal_count") != len(pending)
        or not _materializer.verify_materialized_task_bundle(
            bundle,
            report,
            hypothesis,
            executor,
            materializer_contract,
            base_manifest_path,
            asset_root,
            checkpoint_root,
        )
    ):
        raise AdoptedSuccessorMaterializationError(
            "successor bundle failed strong old-materializer verification"
        )
    first_task = bundle["plan"]["tasks"][0]
    if first_task.get("cell") != first_projection:
        raise AdoptedSuccessorMaterializationError(
            "first successor task differs from first pending cell"
        )
    return bundle, {
        "pending_evidence_digest": pending_digest,
        "first_pending_projection_digest": first_digest,
        "first_pending_projection": first_projection,
        "first_task_id": first_task["task_id"],
        "first_task_digest": first_task["task_digest"],
    }


def _artifact(path: str, raw: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": _raw_digest(raw), "bytes": len(raw)}


def _layout(root: Path, *, committed: bool) -> None:
    _secure_directory(root, exact_mode=0o700)
    expected = {"successor_contract.json", "bundle.json"}
    if committed:
        expected.add("successor.json")
    try:
        observed = {child.name for child in root.iterdir()}
    except OSError as error:
        raise AdoptedSuccessorMaterializationError(
            "cannot enumerate successor root"
        ) from error
    if observed != expected:
        raise AdoptedSuccessorMaterializationError(
            "successor root has missing or unexpected artifacts"
        )


def _marker_body(
    *,
    successor_id: str,
    successor_contract: Mapping[str, Any],
    successor_contract_raw: bytes,
    adoption_marker: Mapping[str, Any],
    bundle: Mapping[str, Any],
    pending_binding: Mapping[str, Any],
    future_attempt_root: Path,
    checkpoint_root: Path,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SUCCESSOR_CAPSULE_SCHEMA_VERSION,
        "status": _STATUS,
        "authorization_status": "NOT_AUTHORIZED",
        "successor_id": successor_id,
        "successor_contract_binding": {
            "contract_id": successor_contract["contract_id"],
            "contract_digest": _digest(successor_contract),
            "raw_sha256": _raw_digest(successor_contract_raw),
            "bytes": len(successor_contract_raw),
        },
        "source_adoption": {
            "adoption_id": adoption_marker["adoption_id"],
            "adoption_digest": adoption_marker["integrity"][
                "adoption_digest"
            ],
            "adoption_status": adoption_marker["status"],
        },
        "pending_binding": dict(pending_binding),
        "bundle_binding": {
            "bundle_id": bundle["bundle_id"],
            "bundle_digest": bundle["integrity"]["bundle_digest"],
            "plan_digest": bundle["plan"]["integrity"]["plan_digest"],
            "task_count": bundle["task_count"],
        },
        "future_attempt_binding": {
            "future_attempt_root": str(future_attempt_root),
            "checkpoint_root": str(checkpoint_root),
            "future_attempt_absent_at_materialization": True,
            "future_attempt_created": False,
            "checkpoint_root_created": False,
        },
        "artifacts": dict(artifacts),
        "nonclaims": list(_NONCLAIMS),
    }


def _result(marker: Mapping[str, Any], root: Path, *, verified=False):
    bundle = marker["bundle_binding"]
    future = marker["future_attempt_binding"]
    return {
        "status": _VERIFIED_STATUS if verified else _STATUS,
        "successor_root": str(root),
        "successor_digest": marker["integrity"]["successor_digest"],
        "adoption_digest": marker["source_adoption"]["adoption_digest"],
        "pending_evidence_digest": marker["pending_binding"][
            "pending_evidence_digest"
        ],
        "first_pending_projection_digest": marker["pending_binding"][
            "first_pending_projection_digest"
        ],
        "first_task_id": marker["pending_binding"]["first_task_id"],
        "first_task_digest": marker["pending_binding"]["first_task_digest"],
        "bundle_digest": bundle["bundle_digest"],
        "plan_digest": bundle["plan_digest"],
        "task_count": bundle["task_count"],
        "future_attempt_root": future["future_attempt_root"],
        "checkpoint_root": future["checkpoint_root"],
        "authorization_status": marker["authorization_status"],
    }


def _paths(
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
    base_manifest_path,
    asset_root,
    future_attempt_root,
) -> dict[str, Path]:
    return {
        name: _absolute_path(value, name)
        for name, value in locals().items()
    }


def _validate_and_materialize(
    paths: Mapping[str, Path],
    *,
    adoption_id: str,
    successor_id: str,
    expected_adoption_digest: str,
    expected_pending_evidence_digest: str,
    expected_first_pending_projection_digest: str,
    require_future_absent: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes, Path, Path]:
    captured = _capture_adoption(paths["adoption_root"])
    _validate_captured_adoption(
        captured,
        adoption_id=adoption_id,
        expected_adoption_digest=expected_adoption_digest,
    )
    _full_verify_adoption(
        captured,
        publication_root=paths["publication_root"],
        adoption_contract_path=paths["adoption_contract_path"],
        adoption_root=paths["adoption_root"],
        base_evidence_csv=paths["base_evidence_csv"],
        source_attempt_root=paths["source_attempt_root"],
        hypothesis_contract_path=paths["hypothesis_contract_path"],
        executor_contract_path=paths["executor_contract_path"],
        runtime_contract_path=paths["runtime_contract_path"],
        publisher_contract_path=paths["publisher_contract_path"],
        base_manifest_path=paths["base_manifest_path"],
        asset_root=paths["asset_root"],
        adoption_id=adoption_id,
        expected_adoption_digest=expected_adoption_digest,
    )
    _same_adoption(captured)
    materializer_contract, materializer_raw = _validate_external_materializer(
        paths["materializer_contract_path"], captured
    )
    runtime_contract = captured["publication_values"]["runtime_contract"]
    future_attempt, checkpoint_root = _validate_future_attempt(
        paths["future_attempt_root"],
        runtime_contract,
        successor_id=successor_id,
        require_absent=require_future_absent,
    )
    bundle, pending = _materialize_bundle(
        captured,
        materializer_contract=materializer_contract,
        base_manifest_path=paths["base_manifest_path"],
        asset_root=paths["asset_root"],
        checkpoint_root=checkpoint_root,
        expected_pending_evidence_digest=expected_pending_evidence_digest,
        expected_first_pending_projection_digest=(
            expected_first_pending_projection_digest
        ),
    )
    _same_adoption(captured)
    return (
        captured,
        bundle,
        pending,
        materializer_raw,
        future_attempt,
        checkpoint_root,
    )


def materialize_adopted_successor(
    publication_root: str | Path,
    adoption_contract_path: str | Path,
    adoption_root: str | Path,
    successor_contract_path: str | Path,
    successor_root: str | Path,
    base_evidence_csv: str | Path,
    source_attempt_root: str | Path,
    hypothesis_contract_path: str | Path,
    executor_contract_path: str | Path,
    runtime_contract_path: str | Path,
    publisher_contract_path: str | Path,
    materializer_contract_path: str | Path,
    base_manifest_path: str | Path,
    asset_root: str | Path,
    future_attempt_root: str | Path,
    *,
    adoption_id: str,
    successor_id: str,
    expected_adoption_digest: str,
    expected_pending_evidence_digest: str,
    expected_first_pending_projection_digest: str,
) -> dict[str, Any]:
    """Commit one replay-derived successor bundle without authorization."""
    expected_adoption_digest = _require_digest(
        expected_adoption_digest, "expected_adoption_digest"
    )
    expected_pending_evidence_digest = _require_digest(
        expected_pending_evidence_digest,
        "expected_pending_evidence_digest",
    )
    expected_first_pending_projection_digest = _require_digest(
        expected_first_pending_projection_digest,
        "expected_first_pending_projection_digest",
    )
    paths = _paths(
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
        base_manifest_path,
        asset_root,
        future_attempt_root,
    )
    contract, contract_raw = _read_json(
        paths["successor_contract_path"], "successor contract"
    )
    _validate_successor_contract(contract)
    captured, bundle, pending, materializer_raw, future, checkpoint = (
        _validate_and_materialize(
            paths,
            adoption_id=adoption_id,
            successor_id=successor_id,
            expected_adoption_digest=expected_adoption_digest,
            expected_pending_evidence_digest=(
                expected_pending_evidence_digest
            ),
            expected_first_pending_projection_digest=(
                expected_first_pending_projection_digest
            ),
            require_future_absent=True,
        )
    )
    root = _validate_successor_location(
        paths["successor_root"], successor_id, fresh=True
    )
    _translate(lambda: _adoption._mkdir_new(root))
    bundle_raw = _canonical_file(bundle)
    artifacts = {
        "successor_contract": _artifact(
            "successor_contract.json", contract_raw
        ),
        "task_bundle": _artifact("bundle.json", bundle_raw),
    }
    _translate(
        lambda: _adoption._write_new_bytes(
            root / "successor_contract.json", contract_raw
        )
    )
    _translate(
        lambda: _adoption._write_new_bytes(root / "bundle.json", bundle_raw)
    )
    _layout(root, committed=False)
    for label, raw in {
        "successor_contract": contract_raw,
        "task_bundle": bundle_raw,
    }.items():
        if _read_regular(
            root / _ARTIFACT_LAYOUT[label],
            f"pre-commit successor artifact {label}",
            exact_mode=0o600,
        ) != raw:
            raise AdoptedSuccessorMaterializationError(
                f"pre-commit successor artifact differs: {label}"
            )
    _same_adoption(captured)
    final_materializer_contract, final_materializer_raw = (
        _validate_external_materializer(
            paths["materializer_contract_path"], captured
        )
    )
    final_bundle, final_pending = _materialize_bundle(
        captured,
        materializer_contract=final_materializer_contract,
        base_manifest_path=paths["base_manifest_path"],
        asset_root=paths["asset_root"],
        checkpoint_root=checkpoint,
        expected_pending_evidence_digest=expected_pending_evidence_digest,
        expected_first_pending_projection_digest=(
            expected_first_pending_projection_digest
        ),
    )
    if (
        final_materializer_raw != materializer_raw
        or final_bundle != bundle
        or final_pending != pending
    ):
        raise AdoptedSuccessorMaterializationError(
            "successor materializer inputs or rebuilt bundle changed"
    )
    _validate_successor_contract(contract)
    _same_adoption(captured)
    _validate_future_attempt(
        future,
        captured["publication_values"]["runtime_contract"],
        successor_id=successor_id,
        require_absent=True,
    )
    _layout(root, committed=False)
    for label, raw in {
        "successor_contract": contract_raw,
        "task_bundle": bundle_raw,
    }.items():
        if _read_regular(
            root / _ARTIFACT_LAYOUT[label],
            f"final pre-commit successor artifact {label}",
            exact_mode=0o600,
        ) != raw:
            raise AdoptedSuccessorMaterializationError(
                f"final pre-commit successor artifact differs: {label}"
            )
    marker_body = _marker_body(
        successor_id=successor_id,
        successor_contract=contract,
        successor_contract_raw=contract_raw,
        adoption_marker=captured["adoption_marker"],
        bundle=bundle,
        pending_binding=pending,
        future_attempt_root=future,
        checkpoint_root=checkpoint,
        artifacts=artifacts,
    )
    marker = {
        **marker_body,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "successor_digest": _digest(marker_body),
        },
    }
    marker_raw = _canonical_file(marker)
    _translate(
        lambda: _adoption._write_new_bytes(root / "successor.json", marker_raw)
    )
    _layout(root, committed=True)
    for relative, raw in {
        "successor_contract.json": contract_raw,
        "bundle.json": bundle_raw,
        "successor.json": marker_raw,
    }.items():
        if _read_regular(
            root / relative,
            f"committed successor artifact {relative}",
            exact_mode=0o600,
        ) != raw:
            raise AdoptedSuccessorMaterializationError(
                f"committed successor artifact differs: {relative}"
            )
    return _result(marker, root)


def verify_adopted_successor(
    publication_root: str | Path,
    adoption_contract_path: str | Path,
    adoption_root: str | Path,
    successor_contract_path: str | Path,
    successor_root: str | Path,
    base_evidence_csv: str | Path,
    source_attempt_root: str | Path,
    hypothesis_contract_path: str | Path,
    executor_contract_path: str | Path,
    runtime_contract_path: str | Path,
    publisher_contract_path: str | Path,
    materializer_contract_path: str | Path,
    base_manifest_path: str | Path,
    asset_root: str | Path,
    future_attempt_root: str | Path,
    *,
    adoption_id: str,
    successor_id: str,
    expected_adoption_digest: str,
    expected_pending_evidence_digest: str,
    expected_first_pending_projection_digest: str,
    expected_successor_digest: str,
    expected_bundle_digest: str,
    expected_plan_digest: str,
) -> dict[str, Any]:
    """Read-only full verification of one committed successor capsule."""
    expected = {
        "expected_adoption_digest": expected_adoption_digest,
        "expected_pending_evidence_digest": expected_pending_evidence_digest,
        "expected_first_pending_projection_digest": (
            expected_first_pending_projection_digest
        ),
        "expected_successor_digest": expected_successor_digest,
        "expected_bundle_digest": expected_bundle_digest,
        "expected_plan_digest": expected_plan_digest,
    }
    for label, value in expected.items():
        expected[label] = _require_digest(value, label)
    paths = _paths(
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
        base_manifest_path,
        asset_root,
        future_attempt_root,
    )
    root = _validate_successor_location(
        paths["successor_root"], successor_id, fresh=False
    )
    _layout(root, committed=True)
    contract, contract_raw = _read_json(
        paths["successor_contract_path"], "successor contract"
    )
    _validate_successor_contract(contract)
    captured, bundle, pending, _materializer_raw, future, checkpoint = (
        _validate_and_materialize(
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
            require_future_absent=False,
        )
    )
    capsule_contract_raw = _read_regular(
        root / "successor_contract.json",
        "capsule successor contract",
        exact_mode=0o600,
    )
    bundle_raw = _read_regular(
        root / "bundle.json", "capsule successor bundle", exact_mode=0o600
    )
    marker_raw = _read_regular(
        root / "successor.json", "successor commit", exact_mode=0o600
    )
    if capsule_contract_raw != contract_raw or bundle_raw != _canonical_file(
        bundle
    ):
        raise AdoptedSuccessorMaterializationError(
            "successor capsule source or bundle differs"
        )
    marker = _parse_object(marker_raw, "successor commit")
    if marker_raw != _canonical_file(marker):
        raise AdoptedSuccessorMaterializationError(
            "successor commit is not canonical JSON"
        )
    integrity = marker.get("integrity")
    body = {key: marker[key] for key in marker if key != "integrity"}
    if (
        type(integrity) is not dict
        or set(integrity) != {"algorithm", "successor_digest"}
        or integrity.get("algorithm") != "sha256-canonical-json-v1"
        or _digest(body) != integrity.get("successor_digest")
        or integrity.get("successor_digest")
        != expected["expected_successor_digest"]
        or bundle["integrity"]["bundle_digest"]
        != expected["expected_bundle_digest"]
        or bundle["plan"]["integrity"]["plan_digest"]
        != expected["expected_plan_digest"]
    ):
        raise AdoptedSuccessorMaterializationError(
            "successor or bundle independent digest differs"
        )
    artifacts = {
        "successor_contract": _artifact(
            "successor_contract.json", contract_raw
        ),
        "task_bundle": _artifact("bundle.json", bundle_raw),
    }
    rebuilt = _marker_body(
        successor_id=successor_id,
        successor_contract=contract,
        successor_contract_raw=contract_raw,
        adoption_marker=captured["adoption_marker"],
        bundle=bundle,
        pending_binding=pending,
        future_attempt_root=future,
        checkpoint_root=checkpoint,
        artifacts=artifacts,
    )
    if (
        body != rebuilt
        or marker.get("schema_version") != SUCCESSOR_CAPSULE_SCHEMA_VERSION
        or marker.get("status") != _STATUS
        or marker.get("authorization_status") != "NOT_AUTHORIZED"
        or marker.get("successor_id") != successor_id
        or root.name != successor_id
        or marker.get("nonclaims") != _NONCLAIMS
    ):
        raise AdoptedSuccessorMaterializationError(
            "successor marker does not reproduce its full chain"
        )
    return _result(marker, root, verified=True)


__all__ = [
    "SUCCESSOR_CAPSULE_SCHEMA_VERSION",
    "SUCCESSOR_CONTRACT_ID",
    "SUCCESSOR_MATERIALIZER_SCHEMA_VERSION",
    "AdoptedSuccessorMaterializationError",
    "materialize_adopted_successor",
    "verify_adopted_successor",
]
