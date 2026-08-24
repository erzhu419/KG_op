"""Build a non-authoritative local packet for external shadow-child review.

This additive core exact-replays the three earlier public slices, binds a
rollback and record-lifecycle boundary, and emits only a local review packet.
It does not determine adoption eligibility, obtain an external review, reuse
consumed evidence, mutate a theory, promote a result, or write a current
pointer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping

from performance.shadow_child_probe_qualification import (
    ShadowChildProbeQualificationValidationError,
    verify_shadow_child_probe_qualification,
)
from performance.theory_operation_competition import (
    CompetitionValidationError,
    canonical_json_bytes,
)


CONTRACT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-child-external-review-packet-contract/1"
)
CONTRACT_ID = "shadow_child_external_review_packet_v1"
SOURCE_QUALIFICATION_CONTRACT_ID = "shadow_child_probe_qualification_v1"
SOURCE_QUALIFICATION_CONTRACT_DIGEST = (
    "sha256:593b8727f7f985cea82ae86e0758a67018c8d6a0a7c1dc0d518bcc3615f7d4ee"
)
SOURCE_QUALIFICATION_REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-child-probe-qualification-report/1"
)
REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-child-external-review-packet-report/1"
)
RECORD_LIFECYCLE_BOUNDARY_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-child-record-lifecycle-boundary/1"
)
GENESIS_DIGEST = "sha256:" + "0" * 64

ADOPTION_ELIGIBILITY = "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED"
ADOPTION_STATUS = "NOT_ADOPTED_SHADOW_ONLY"
PROMOTION_STATUS = "NOT_PROMOTED"
CURRENT_STATUS = "NOT_CURRENT"

QUALIFICATION_DISPOSITION_MAPPING = {
    "QUALIFIED_NEW_EVALUATOR_EPOCH": "READY_FOR_EXTERNAL_REVIEW_PACKET_ONLY",
    "NEEDS_NEW_EVALUATOR_EVIDENCE": "REVIEW_PACKET_PENDING_NEW_EVIDENCE",
    "INCOMPARABLE_NEW_EVALUATOR_EPOCH": (
        "REVIEW_PACKET_BLOCKED_INCOMPARABLE_EPOCH"
    ),
    "FAILED_OPERATIONAL_PROBE_QUALIFICATION": (
        "REVIEW_PACKET_BLOCKED_PROBE_FAILURE"
    ),
}

REVIEW_REQUIREMENTS = {
    "exact_source_replay_required": True,
    "materialized_child_required": True,
    "fresh_epoch_qualification_required_for_ready": True,
    "complete_evidence_required_for_ready": True,
    "all_probe_gates_required_for_ready": True,
    "parent_rollback_binding_required": True,
    "source_scoring_exclusion_required": True,
    "no_pooling_required": True,
    "logical_erasure_boundary_required": True,
    "external_data_attestation": "REQUIRED_NOT_PRESENT",
    "external_evaluator_attestation": "REQUIRED_NOT_PRESENT",
    "physical_retention_attestation": "REQUIRED_NOT_PRESENT",
}

RECORD_LIFECYCLE_POLICY = {
    "parent_snapshot_role": "ROLLBACK_REQUIRED",
    "child_snapshot_role": "SHADOW_REVIEW_CANDIDATE",
    "source_record_role": "AUDIT_ONLY_SOURCE_SCORING_EXCLUDED",
    "qualification_record_role": (
        "CONSUMED_QUALIFICATION_EVIDENCE_AUDIT_ONLY"
    ),
    "retain_commitments_for_audit": True,
    "future_scoring_requires_new_unconsumed_evidence": True,
    "physical_retention_attestation": "REQUIRED_NOT_PRESENT",
}

SELECTIVE_ERASURE_POLICY = {
    "mode": "LOGICAL_ACTIVE_SCORING_VIEW_ONLY",
    "exclude_source_records": True,
    "exclude_consumed_qualification_records_from_future_rescoring": True,
    "forbid_cross_epoch_pooling": True,
    "physical_erasure": "NOT_PERFORMED",
}

AUTHORITY_BOUNDARY = {
    "scope": "LOCAL_EXTERNAL_REVIEW_PACKET_ONLY",
    "adoption_eligibility": ADOPTION_ELIGIBILITY,
    "external_adoption_authority": "REQUIRED_NOT_PRESENT",
    "adoption_action_forbidden": True,
    "promotion_action_forbidden": True,
    "current_pointer_write_forbidden": True,
}

SELECTION = {
    "ready_status": "READY_FOR_EXTERNAL_REVIEW_PACKET_ONLY",
    "pending_evidence_status": "REVIEW_PACKET_PENDING_NEW_EVIDENCE",
    "blocked_incomparable_status": (
        "REVIEW_PACKET_BLOCKED_INCOMPARABLE_EPOCH"
    ),
    "blocked_probe_failure_status": "REVIEW_PACKET_BLOCKED_PROBE_FAILURE",
    "adoption_eligibility": ADOPTION_ELIGIBILITY,
    "adoption_status": ADOPTION_STATUS,
    "promotion_status": PROMOTION_STATUS,
    "current_status": CURRENT_STATUS,
}

MANDATORY_NONCLAIMS = (
    "shadow_only",
    "external_review_packet_only",
    "adoption_eligibility_not_determined",
    "no_automatic_adoption",
    "no_promotion",
    "no_current_pointer_write",
    "no_parent_child_or_ambient_state_write",
    "no_operations_research_baseline_or_claim_change",
    "no_paper_promotion",
    "explicit_cli_out_is_only_optional_write",
    "no_run_one",
    "no_benchmark_execution",
    "no_scheduler_access",
    "no_network_access",
    "no_new_theory_operation",
    "no_restrict_operation",
    "no_probe_observation_expansion_or_execution",
    "no_language_or_predicate_invention",
    "no_source_evidence_rescoring",
    "no_consumed_qualification_evidence_rescoring",
    "no_cross_epoch_pooling",
    "logical_selective_erasure_boundary_only",
    "no_physical_erasure",
    "physical_retention_attestation_required_not_present",
    "external_data_attestation_required_not_present",
    "external_evaluator_attestation_required_not_present",
    "external_adoption_authority_required_not_present",
    "rollback_binding_is_not_rollback_execution",
    "retention_requirements_bound_not_executed",
    "input_artifacts_require_independent_expected_values",
    "report_digest_is_not_a_signature",
    "no_data_provenance_beyond_bound_artifacts",
    "no_external_review_outcome",
    "no_scientific_validity_or_generalization_claim",
    "no_h_t_to_h_t_plus_1_acceptance",
)

REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "contract_digest",
        "packet_id",
        "source_qualification",
        "child_theory_state_digest",
        "parent_theory_state_digest",
        "transition_kind",
        "evaluator_epoch",
        "review_checks",
        "record_lifecycle_boundary",
        "rollback_boundary",
        "selective_erasure_boundary",
        "attestation_boundary",
        "disposition",
        "review_boundary",
        "adoption_eligibility",
        "adoption_status",
        "promotion_status",
        "current_status",
        "nonclaims",
        "input_artifacts",
        "audit_events",
        "audit_head",
        "report_digest",
    }
)

SOURCE_QUALIFICATION_FIELDS = frozenset(
    {
        "verification_status",
        "contract_id",
        "contract_digest",
        "report_schema_version",
        "report_digest",
        "qualification_input_digest",
        "qualification_disposition",
        "source_transition_report_digest",
        "child_theory_state_digest",
        "evaluator_epoch",
        "adoption_status",
    }
)

REVIEW_CHECK_FIELDS = frozenset(
    {
        "source_qualification_exact_replay_verified",
        "source_transition_materialized",
        "child_digest_bound",
        "qualified_new_evaluator_epoch",
        "fresh_epoch_comparable",
        "evidence_sufficient",
        "operational_probe_results_present",
        "all_operational_probe_gates_passed",
        "source_evidence_scoring_excluded",
        "old_new_pooling_forbidden",
        "logical_selective_erasure_applied",
        "physical_deletion_absent",
        "parent_rollback_binding_present",
        "original_child_unmodified",
        "upstream_adoption_withheld",
        "record_lifecycle_boundary_bound",
        "local_packet_complete",
    }
)

RECORD_LIFECYCLE_FIELDS = frozenset(
    {
        "boundary_schema_version",
        "parent_snapshot",
        "child_snapshot",
        "source_competition_records",
        "qualification_records",
        "commitment_chain",
        "future_scoring_policy",
        "retention_requirements_bound",
        "physical_retention_attestation",
    }
)

ROLLBACK_FIELDS = frozenset(
    {
        "parent_theory_state_digest",
        "child_theory_state_digest",
        "transition_kind",
        "reduction_certificate_digest",
        "parent_snapshot_digest",
        "rollback_method",
        "rollback_binding_verified",
        "rollback_execution_status",
    }
)

SELECTIVE_ERASURE_FIELDS = frozenset(
    {
        "mode",
        "source_competition_records_excluded_from_active_scoring",
        "consumed_qualification_records_excluded_from_future_rescoring",
        "future_scoring_requires_new_unconsumed_evidence",
        "cross_epoch_pooling_allowed",
        "logical_boundary_bound",
        "physical_erasure_status",
    }
)

ATTESTATION_FIELDS = frozenset(
    {
        "external_data_attestation",
        "external_evaluator_attestation",
        "physical_retention_attestation",
        "physical_erasure",
        "external_adoption_authority",
    }
)

REVIEW_BOUNDARY_FIELDS = frozenset(
    {
        "scope",
        "packet_ready",
        "external_review_required",
        "adoption_decision_allowed",
        "promotion_decision_allowed",
        "current_pointer_write_allowed",
        "parent_or_child_state_write_allowed",
    }
)


class ShadowChildExternalReviewPacketValidationError(ValueError):
    """Raised when a review contract, source chain, or packet is invalid."""


class ShadowChildExternalReviewPacketDisposition(str, Enum):
    READY_FOR_EXTERNAL_REVIEW_PACKET_ONLY = (
        "READY_FOR_EXTERNAL_REVIEW_PACKET_ONLY"
    )
    REVIEW_PACKET_PENDING_NEW_EVIDENCE = (
        "REVIEW_PACKET_PENDING_NEW_EVIDENCE"
    )
    REVIEW_PACKET_BLOCKED_INCOMPARABLE_EPOCH = (
        "REVIEW_PACKET_BLOCKED_INCOMPARABLE_EPOCH"
    )
    REVIEW_PACKET_BLOCKED_PROBE_FAILURE = (
        "REVIEW_PACKET_BLOCKED_PROBE_FAILURE"
    )


@dataclass(frozen=True)
class ShadowChildExternalReviewPacketResult:
    report: Mapping[str, Any]

    @property
    def disposition(self) -> str:
        return str(self.report["disposition"])

    @property
    def report_digest(self) -> str:
        return str(self.report["report_digest"])

    @property
    def ready_for_external_review(self) -> bool:
        return self.disposition == "READY_FOR_EXTERNAL_REVIEW_PACKET_ONLY"

    def to_dict(self) -> dict[str, Any]:
        return _copy(self.report)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShadowChildExternalReviewPacketValidationError(
            f"{label} must be an object"
        )
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str
) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected), key=str)
        raise ShadowChildExternalReviewPacketValidationError(
            f"{label} keys differ; missing={missing}, extra={extra}"
        )


def _string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ShadowChildExternalReviewPacketValidationError(
            f"{label} must be a non-empty string"
        )
    return value


def _copy(value: Any) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("ascii"))
    except (
        CompetitionValidationError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ShadowChildExternalReviewPacketValidationError(
            f"value is not detached finite canonical JSON: {exc}"
        ) from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_digest(value: Any, label: str) -> str:
    digest = _string(value, label)
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise ShadowChildExternalReviewPacketValidationError(
            f"{label} must be sha256:<64hex>"
        )
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise ShadowChildExternalReviewPacketValidationError(
            f"{label} is not hexadecimal"
        ) from exc
    return digest


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ShadowChildExternalReviewPacketValidationError(
            f"{label} differs from frozen external-review packet V1"
        )


def _reject_observed_values(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        if "observed_value" in value:
            raise ShadowChildExternalReviewPacketValidationError(
                f"{label} must not embed observation values"
            )
        for key, item in value.items():
            _reject_observed_values(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_observed_values(item, f"{label}[{index}]")


def _validate_contract(contract_value: Any) -> dict[str, Any]:
    contract = _mapping(contract_value, "review_contract")
    _exact_keys(
        contract,
        {
            "schema_version",
            "contract_id",
            "source_qualification_contract_id",
            "source_qualification_contract_digest",
            "source_qualification_report_schema_version",
            "report_schema_version",
            "qualification_disposition_mapping",
            "review_requirements",
            "record_lifecycle_policy",
            "selective_erasure_policy",
            "authority_boundary",
            "selection",
            "nonclaims",
        },
        "review_contract",
    )
    frozen_values = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "source_qualification_contract_id": SOURCE_QUALIFICATION_CONTRACT_ID,
        "source_qualification_contract_digest": (
            SOURCE_QUALIFICATION_CONTRACT_DIGEST
        ),
        "source_qualification_report_schema_version": (
            SOURCE_QUALIFICATION_REPORT_SCHEMA_VERSION
        ),
        "report_schema_version": REPORT_SCHEMA_VERSION,
    }
    for key, expected in frozen_values.items():
        _require_equal(contract[key], expected, key)
    nested = (
        (
            "qualification_disposition_mapping",
            QUALIFICATION_DISPOSITION_MAPPING,
        ),
        ("review_requirements", REVIEW_REQUIREMENTS),
        ("record_lifecycle_policy", RECORD_LIFECYCLE_POLICY),
        ("selective_erasure_policy", SELECTIVE_ERASURE_POLICY),
        ("authority_boundary", AUTHORITY_BOUNDARY),
        ("selection", SELECTION),
    )
    for key, expected in nested:
        _require_equal(
            _copy(_mapping(contract[key], key)),
            expected,
            key,
        )
    nonclaims = contract["nonclaims"]
    if not isinstance(nonclaims, list) or tuple(nonclaims) != MANDATORY_NONCLAIMS:
        raise ShadowChildExternalReviewPacketValidationError(
            "nonclaims differ from the exact mandatory V1 list"
        )
    return _copy(contract)


def validate_shadow_child_external_review_packet_contract(
    contract_value: Any,
) -> dict[str, Any]:
    """Validate and detach the frozen local external-review contract."""

    return _validate_contract(contract_value)


def _audit_event(
    sequence: int, event: str, previous: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    body = {
        "sequence": sequence,
        "event": event,
        "previous_event_digest": previous,
        "payload": _copy(payload),
    }
    return {**body, "event_digest": _digest(body)}


def _rollback_boundary(
    transition_report: Mapping[str, Any],
    parent_digest: str,
    child_digest: str,
    transition_kind: str,
) -> dict[str, Any]:
    parent = _mapping(
        transition_report.get("parent_theory_state"), "parent_theory_state"
    )
    reduction = _mapping(
        transition_report.get("reduction_certificate"),
        "reduction_certificate",
    )
    parent_snapshot_digest = _digest(parent)
    if parent_snapshot_digest != parent_digest:
        raise ShadowChildExternalReviewPacketValidationError(
            "parent snapshot digest differs from the verified transition"
        )
    if transition_kind == "ROBUST_INTERVAL_EXPANSION":
        rollback_method = "COLLAPSE_INTERVAL_AT_RADIUS_MULTIPLIER_ZERO"
        rollback_verified = all(
            (
                reduction.get("map_kind") == rollback_method,
                reduction.get("parameter") == "radius_multiplier",
                reduction.get("parent_value") == 0.0,
                reduction.get("exact_parent_model_class_recovered") is True,
                reduction.get("collapsed_model_class_digest")
                == reduction.get("parent_model_class_digest"),
            )
        )
    elif transition_kind == "QUOTIENT_IDEALIZATION":
        rollback_method = "RESTORE_FROZEN_PARENT_POINT_TABLE"
        rollback_verified = all(
            (
                reduction.get("map_kind")
                == "QUOTIENT_PROJECTION_WITH_FROZEN_PARENT_SNAPSHOT",
                reduction.get("parent_snapshot_digest")
                == parent_snapshot_digest,
                reduction.get("exact_parent_recovery_from_snapshot_verified")
                is True,
                reduction.get("lossy_quotient_requires_parent_snapshot")
                is True,
                reduction.get("quotient_alone_recovers_parent") is False,
            )
        )
    else:
        raise ShadowChildExternalReviewPacketValidationError(
            "review packet supports only materialized robust or ideal children"
        )
    if not rollback_verified:
        raise ShadowChildExternalReviewPacketValidationError(
            "verified transition lacks the required parent rollback binding"
        )
    result = {
        "parent_theory_state_digest": parent_digest,
        "child_theory_state_digest": child_digest,
        "transition_kind": transition_kind,
        "reduction_certificate_digest": _digest(reduction),
        "parent_snapshot_digest": parent_snapshot_digest,
        "rollback_method": rollback_method,
        "rollback_binding_verified": True,
        "rollback_execution_status": "NOT_PERFORMED",
    }
    _exact_keys(result, ROLLBACK_FIELDS, "rollback_boundary")
    return result


def _record_lifecycle_boundary(
    competition_report: Mapping[str, Any],
    transition_report: Mapping[str, Any],
    qualification_report: Mapping[str, Any],
    review_contract_digest: str,
) -> dict[str, Any]:
    source = _mapping(
        transition_report.get("source_competition"), "source_competition"
    )
    evidence_binding = _mapping(
        qualification_report.get("evidence_binding"), "evidence_binding"
    )
    evaluator_binding = _mapping(
        qualification_report.get("evaluator_binding"), "evaluator_binding"
    )
    parent_digest = _require_digest(
        transition_report.get("parent_theory_state_digest"),
        "parent_theory_state_digest",
    )
    child_digest = _require_digest(
        qualification_report.get("child_theory_state_digest"),
        "child_theory_state_digest",
    )
    result = {
        "boundary_schema_version": RECORD_LIFECYCLE_BOUNDARY_SCHEMA_VERSION,
        "parent_snapshot": {
            "theory_state_digest": parent_digest,
            "role": "ROLLBACK_REQUIRED",
            "retention_required": True,
            "eligible_for_scoring": False,
        },
        "child_snapshot": {
            "theory_state_digest": child_digest,
            "role": "SHADOW_REVIEW_CANDIDATE",
            "retention_required": True,
            "adopted": False,
            "current": False,
        },
        "source_competition_records": {
            "evidence_digests": _copy(
                _mapping(source.get("evidence_digests"), "source evidence_digests")
            ),
            "observation_id_digest": evidence_binding[
                "source_observation_id_digest"
            ],
            "observation_count": evidence_binding["source_observation_count"],
            "evaluator_epoch": evaluator_binding["source_evaluator_epoch"],
            "role": "AUDIT_ONLY_SOURCE_SCORING_EXCLUDED",
            "retain_commitments_for_audit": True,
            "eligible_for_future_scoring": False,
        },
        "qualification_records": {
            "evidence_digests": _copy(
                _mapping(
                    evidence_binding.get("evidence_digests"),
                    "qualification evidence_digests",
                )
            ),
            "observation_id_digest": evidence_binding[
                "new_observation_id_digest"
            ],
            "observation_count": evidence_binding["new_observation_count"],
            "evaluator_epoch": evaluator_binding[
                "derived_child_evaluator_epoch"
            ],
            "role": "CONSUMED_QUALIFICATION_EVIDENCE_AUDIT_ONLY",
            "retain_commitments_for_audit": True,
            "eligible_for_future_scoring": False,
        },
        "commitment_chain": {
            "competition_contract_digest": competition_report["contract_digest"],
            "competition_report_digest": competition_report["report_digest"],
            "transition_contract_digest": transition_report["contract_digest"],
            "transition_report_digest": transition_report["report_digest"],
            "qualification_contract_digest": qualification_report[
                "contract_digest"
            ],
            "qualification_report_digest": qualification_report["report_digest"],
            "review_contract_digest": review_contract_digest,
        },
        "future_scoring_policy": {
            "new_unconsumed_evidence_required": True,
            "reuse_source_records_allowed": False,
            "reuse_consumed_qualification_records_allowed": False,
            "cross_epoch_pooling_allowed": False,
        },
        "retention_requirements_bound": True,
        "physical_retention_attestation": "REQUIRED_NOT_PRESENT",
    }
    _exact_keys(
        result, RECORD_LIFECYCLE_FIELDS, "record_lifecycle_boundary"
    )
    return result


def _source_qualification_summary(
    qualification_report: Mapping[str, Any],
    verification_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    source_transition = _mapping(
        qualification_report.get("source_transition"), "source_transition"
    )
    binding = _mapping(
        qualification_report.get("qualification_binding"),
        "qualification_binding",
    )
    result = {
        "verification_status": verification_receipt.get("status"),
        "contract_id": qualification_report.get("contract_id"),
        "contract_digest": qualification_report.get("contract_digest"),
        "report_schema_version": qualification_report.get("schema_version"),
        "report_digest": qualification_report.get("report_digest"),
        "qualification_input_digest": qualification_report.get(
            "qualification_input_digest"
        ),
        "qualification_disposition": qualification_report.get("disposition"),
        "source_transition_report_digest": source_transition.get("report_digest"),
        "child_theory_state_digest": qualification_report.get(
            "child_theory_state_digest"
        ),
        "evaluator_epoch": binding.get("evaluator_epoch"),
        "adoption_status": qualification_report.get("adoption_status"),
    }
    _exact_keys(result, SOURCE_QUALIFICATION_FIELDS, "source_qualification")
    return result


def _selective_erasure_boundary() -> dict[str, Any]:
    result = {
        "mode": "LOGICAL_ACTIVE_SCORING_VIEW_ONLY",
        "source_competition_records_excluded_from_active_scoring": True,
        "consumed_qualification_records_excluded_from_future_rescoring": True,
        "future_scoring_requires_new_unconsumed_evidence": True,
        "cross_epoch_pooling_allowed": False,
        "logical_boundary_bound": True,
        "physical_erasure_status": "NOT_PERFORMED",
    }
    _exact_keys(
        result, SELECTIVE_ERASURE_FIELDS, "selective_erasure_boundary"
    )
    return result


def _attestation_boundary() -> dict[str, Any]:
    result = {
        "external_data_attestation": "REQUIRED_NOT_PRESENT",
        "external_evaluator_attestation": "REQUIRED_NOT_PRESENT",
        "physical_retention_attestation": "REQUIRED_NOT_PRESENT",
        "physical_erasure": "NOT_PERFORMED",
        "external_adoption_authority": "REQUIRED_NOT_PRESENT",
    }
    _exact_keys(result, ATTESTATION_FIELDS, "attestation_boundary")
    return result


def build_shadow_child_external_review_packet(
    competition_input: Mapping[str, Any],
    competition_contract: Mapping[str, Any],
    competition_report: Mapping[str, Any],
    transition_contract: Mapping[str, Any],
    transition_report: Mapping[str, Any],
    qualification_input: Mapping[str, Any],
    qualification_contract: Mapping[str, Any],
    qualification_report: Mapping[str, Any],
    review_contract: Mapping[str, Any],
    *,
    expected_competition_contract_digest: str,
    expected_competition_report_digest: str,
    expected_competition_input_artifacts: Mapping[str, Any] | None,
    expected_transition_contract_digest: str,
    expected_transition_report_digest: str,
    expected_transition_input_artifacts: Mapping[str, Any] | None,
    expected_qualification_input_digest: str,
    expected_qualification_contract_digest: str,
    expected_qualification_report_digest: str,
    expected_qualification_input_artifacts: Mapping[str, Any] | None,
    expected_review_contract_digest: str,
    input_artifacts: Mapping[str, Any] | None = None,
) -> ShadowChildExternalReviewPacketResult:
    """Build one local packet without making an external review decision."""

    normalized_contract = _validate_contract(review_contract)
    expected_contract_digest = _require_digest(
        expected_review_contract_digest, "expected_review_contract_digest"
    )
    contract_digest = _digest(normalized_contract)
    if contract_digest != expected_contract_digest:
        raise ShadowChildExternalReviewPacketValidationError(
            "review contract digest differs from independent expectation"
        )
    if input_artifacts is not None and not isinstance(input_artifacts, Mapping):
        raise ShadowChildExternalReviewPacketValidationError(
            "input_artifacts must be an object or null"
        )
    artifacts = _copy(input_artifacts) if input_artifacts is not None else None
    _reject_observed_values(artifacts, "input_artifacts")

    try:
        qualification_receipt = verify_shadow_child_probe_qualification(
            competition_input,
            competition_contract,
            competition_report,
            transition_contract,
            transition_report,
            qualification_input,
            qualification_contract,
            qualification_report,
            expected_competition_contract_digest=(
                expected_competition_contract_digest
            ),
            expected_competition_report_digest=(
                expected_competition_report_digest
            ),
            expected_competition_input_artifacts=(
                expected_competition_input_artifacts
            ),
            expected_transition_contract_digest=expected_transition_contract_digest,
            expected_transition_report_digest=expected_transition_report_digest,
            expected_transition_input_artifacts=expected_transition_input_artifacts,
            expected_qualification_input_digest=expected_qualification_input_digest,
            expected_qualification_contract_digest=(
                expected_qualification_contract_digest
            ),
            expected_qualification_report_digest=(
                expected_qualification_report_digest
            ),
            expected_qualification_input_artifacts=(
                expected_qualification_input_artifacts
            ),
        )
    except (
        ShadowChildProbeQualificationValidationError,
        CompetitionValidationError,
        KeyError,
        TypeError,
    ) as exc:
        raise ShadowChildExternalReviewPacketValidationError(
            f"source qualification verification failed: {exc}"
        ) from exc

    competition = _copy(_mapping(competition_report, "competition_report"))
    transition = _copy(_mapping(transition_report, "transition_report"))
    qualification = _copy(
        _mapping(qualification_report, "qualification_report")
    )
    receipt = _copy(_mapping(qualification_receipt, "qualification_receipt"))
    if qualification.get("contract_id") != SOURCE_QUALIFICATION_CONTRACT_ID:
        raise ShadowChildExternalReviewPacketValidationError(
            "source qualification contract_id is not supported"
        )
    if qualification.get("contract_digest") != (
        SOURCE_QUALIFICATION_CONTRACT_DIGEST
    ):
        raise ShadowChildExternalReviewPacketValidationError(
            "source qualification contract digest is not pinned V1"
        )
    if qualification.get("schema_version") != (
        SOURCE_QUALIFICATION_REPORT_SCHEMA_VERSION
    ):
        raise ShadowChildExternalReviewPacketValidationError(
            "source qualification report schema is not supported"
        )

    qualification_disposition = _string(
        qualification.get("disposition"), "qualification disposition"
    )
    if qualification_disposition not in QUALIFICATION_DISPOSITION_MAPPING:
        raise ShadowChildExternalReviewPacketValidationError(
            "source qualification disposition is not supported"
        )
    disposition = QUALIFICATION_DISPOSITION_MAPPING[qualification_disposition]
    expected_receipt_status = "VERIFIED_" + qualification_disposition
    source_summary = _source_qualification_summary(qualification, receipt)
    if source_summary["verification_status"] != expected_receipt_status:
        raise ShadowChildExternalReviewPacketValidationError(
            "qualification verification receipt status is inconsistent"
        )

    parent_digest = _require_digest(
        transition.get("parent_theory_state_digest"),
        "parent_theory_state_digest",
    )
    child_digest = _require_digest(
        qualification.get("child_theory_state_digest"),
        "child_theory_state_digest",
    )
    transition_kind = _string(
        qualification.get("transition_kind"), "transition_kind"
    )
    qbinding = _mapping(
        qualification.get("qualification_binding"), "qualification_binding"
    )
    evaluator_binding = _mapping(
        qualification.get("evaluator_binding"), "evaluator_binding"
    )
    evidence_binding = _mapping(
        qualification.get("evidence_binding"), "evidence_binding"
    )
    erasure = _mapping(
        qualification.get("selective_erasure_receipt"),
        "selective_erasure_receipt",
    )
    probe_results_value = qualification.get("probe_results")
    probe_results = (
        _mapping(probe_results_value, "probe_results")
        if probe_results_value is not None
        else None
    )
    evaluator_epoch = _string(qbinding.get("evaluator_epoch"), "evaluator_epoch")

    rollback = _rollback_boundary(
        transition, parent_digest, child_digest, transition_kind
    )
    lifecycle = _record_lifecycle_boundary(
        competition,
        transition,
        qualification,
        contract_digest,
    )
    selective_boundary = _selective_erasure_boundary()
    attestations = _attestation_boundary()

    source_transition_materialized = transition.get("disposition") in {
        "MATERIALIZED_SHADOW_ROBUSTIFICATION",
        "MATERIALIZED_SHADOW_IDEALIZATION",
    }
    transition_child = _mapping(
        transition.get("child_theory_state"), "child_theory_state"
    )
    child_bound = all(
        (
            transition.get("child_theory_state_digest") == child_digest,
            _digest(transition_child) == child_digest,
            source_summary["child_theory_state_digest"] == child_digest,
            receipt.get("child_theory_state_digest") == child_digest,
        )
    )
    source_scoring_excluded = all(
        (
            erasure.get("source_evidence_used_for_child_scoring") is False,
            qbinding.get("source_evidence_allowed_for_child_scoring") is False,
        )
    )
    no_pooling = all(
        (
            erasure.get("old_new_records_pooled") is False,
            evaluator_binding.get("old_new_records_pooled") is False,
            qbinding.get("old_new_records_pooled") is False,
        )
    )
    logical_erasure = all(
        (
            erasure.get("logical_selective_erasure_applied") is True,
            qbinding.get("logical_selective_erasure_applied") is True,
        )
    )
    checks = {
        "source_qualification_exact_replay_verified": True,
        "source_transition_materialized": source_transition_materialized,
        "child_digest_bound": child_bound,
        "qualified_new_evaluator_epoch": qualification_disposition
        == "QUALIFIED_NEW_EVALUATOR_EPOCH",
        "fresh_epoch_comparable": all(
            (
                evaluator_binding.get("comparable") is True,
                evaluator_binding.get("fresh_from_source_epoch") is True,
            )
        ),
        "evidence_sufficient": evidence_binding.get("sufficient") is True,
        "operational_probe_results_present": probe_results is not None,
        "all_operational_probe_gates_passed": (
            probe_results is not None
            and probe_results.get("all_gates_passed") is True
        ),
        "source_evidence_scoring_excluded": source_scoring_excluded,
        "old_new_pooling_forbidden": no_pooling,
        "logical_selective_erasure_applied": logical_erasure,
        "physical_deletion_absent": erasure.get("physical_records_deleted")
        is False,
        "parent_rollback_binding_present": rollback[
            "rollback_binding_verified"
        ]
        is True,
        "original_child_unmodified": qbinding.get(
            "original_child_state_mutated"
        )
        is False,
        "upstream_adoption_withheld": all(
            (
                qualification.get("adoption_status") == ADOPTION_STATUS,
                transition.get("adoption_status") == ADOPTION_STATUS,
                receipt.get("adoption_status") == ADOPTION_STATUS,
            )
        ),
        "record_lifecycle_boundary_bound": True,
        "local_packet_complete": True,
    }
    _exact_keys(checks, REVIEW_CHECK_FIELDS, "review_checks")
    unconditional_checks = (
        "source_qualification_exact_replay_verified",
        "source_transition_materialized",
        "child_digest_bound",
        "source_evidence_scoring_excluded",
        "old_new_pooling_forbidden",
        "logical_selective_erasure_applied",
        "physical_deletion_absent",
        "parent_rollback_binding_present",
        "original_child_unmodified",
        "upstream_adoption_withheld",
        "record_lifecycle_boundary_bound",
        "local_packet_complete",
    )
    if not all(checks[key] for key in unconditional_checks):
        raise ShadowChildExternalReviewPacketValidationError(
            "verified source violates a mandatory local review boundary"
        )
    packet_ready = disposition == SELECTION["ready_status"]
    if packet_ready and not all(checks.values()):
        raise ShadowChildExternalReviewPacketValidationError(
            "qualified source does not satisfy every local readiness check"
        )

    packet_commitment = {
        "review_contract_digest": contract_digest,
        "qualification_report_digest": qualification["report_digest"],
        "child_theory_state_digest": child_digest,
        "evaluator_epoch": evaluator_epoch,
    }
    packet_id = "shadow-external-review-packet:" + _digest(packet_commitment)[7:]
    review_boundary = {
        "scope": "LOCAL_EXTERNAL_REVIEW_PACKET_ONLY",
        "packet_ready": packet_ready,
        "external_review_required": True,
        "adoption_decision_allowed": False,
        "promotion_decision_allowed": False,
        "current_pointer_write_allowed": False,
        "parent_or_child_state_write_allowed": False,
    }
    _exact_keys(review_boundary, REVIEW_BOUNDARY_FIELDS, "review_boundary")

    events = [
        _audit_event(
            0,
            "SOURCE_QUALIFICATION_VERIFIED",
            GENESIS_DIGEST,
            {
                "verification_status": source_summary["verification_status"],
                "qualification_report_digest": source_summary["report_digest"],
                "qualification_disposition": qualification_disposition,
            },
        )
    ]
    events.append(
        _audit_event(
            1,
            "RECORD_LIFECYCLE_AND_ROLLBACK_BOUND",
            events[-1]["event_digest"],
            {
                "parent_theory_state_digest": parent_digest,
                "child_theory_state_digest": child_digest,
                "rollback_binding_verified": True,
                "physical_retention_attestation": "REQUIRED_NOT_PRESENT",
            },
        )
    )
    events.append(
        _audit_event(
            2,
            "EXTERNAL_REVIEW_PACKET_ASSESSED",
            events[-1]["event_digest"],
            {
                "packet_id": packet_id,
                "disposition": disposition,
                "packet_ready": packet_ready,
                "external_review_required": True,
            },
        )
    )
    events.append(
        _audit_event(
            3,
            "ADOPTION_AND_PROMOTION_WITHHELD",
            events[-1]["event_digest"],
            {
                "adoption_eligibility": ADOPTION_ELIGIBILITY,
                "adoption_status": ADOPTION_STATUS,
                "promotion_status": PROMOTION_STATUS,
                "current_status": CURRENT_STATUS,
            },
        )
    )

    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "contract_digest": contract_digest,
        "packet_id": packet_id,
        "source_qualification": source_summary,
        "child_theory_state_digest": child_digest,
        "parent_theory_state_digest": parent_digest,
        "transition_kind": transition_kind,
        "evaluator_epoch": evaluator_epoch,
        "review_checks": checks,
        "record_lifecycle_boundary": lifecycle,
        "rollback_boundary": rollback,
        "selective_erasure_boundary": selective_boundary,
        "attestation_boundary": attestations,
        "disposition": disposition,
        "review_boundary": review_boundary,
        "adoption_eligibility": ADOPTION_ELIGIBILITY,
        "adoption_status": ADOPTION_STATUS,
        "promotion_status": PROMOTION_STATUS,
        "current_status": CURRENT_STATUS,
        "nonclaims": _copy(normalized_contract["nonclaims"]),
        "input_artifacts": artifacts,
        "audit_events": events,
        "audit_head": events[-1]["event_digest"],
    }
    report = {**body, "report_digest": _digest(body)}
    _exact_keys(report, REPORT_FIELDS, "review_report")
    _reject_observed_values(report, "review_report")
    return ShadowChildExternalReviewPacketResult(report=report)


def verify_shadow_child_external_review_packet(
    competition_input: Mapping[str, Any],
    competition_contract: Mapping[str, Any],
    competition_report: Mapping[str, Any],
    transition_contract: Mapping[str, Any],
    transition_report: Mapping[str, Any],
    qualification_input: Mapping[str, Any],
    qualification_contract: Mapping[str, Any],
    qualification_report: Mapping[str, Any],
    review_contract: Mapping[str, Any],
    review_report: Mapping[str, Any],
    *,
    expected_competition_contract_digest: str,
    expected_competition_report_digest: str,
    expected_competition_input_artifacts: Mapping[str, Any] | None,
    expected_transition_contract_digest: str,
    expected_transition_report_digest: str,
    expected_transition_input_artifacts: Mapping[str, Any] | None,
    expected_qualification_input_digest: str,
    expected_qualification_contract_digest: str,
    expected_qualification_report_digest: str,
    expected_qualification_input_artifacts: Mapping[str, Any] | None,
    expected_review_contract_digest: str,
    expected_review_report_digest: str,
    expected_review_input_artifacts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Exact-replay a local review packet against independent anchors."""

    expected_report_digest = _require_digest(
        expected_review_report_digest, "expected_review_report_digest"
    )
    supplied = _copy(_mapping(review_report, "review_report"))
    _exact_keys(supplied, REPORT_FIELDS, "review_report")
    _reject_observed_values(supplied, "review_report")
    if expected_review_input_artifacts is not None and not isinstance(
        expected_review_input_artifacts, Mapping
    ):
        raise ShadowChildExternalReviewPacketValidationError(
            "expected_review_input_artifacts must be an object or null"
        )
    expected_artifacts = (
        _copy(expected_review_input_artifacts)
        if expected_review_input_artifacts is not None
        else None
    )
    _reject_observed_values(expected_artifacts, "expected_review_input_artifacts")
    if supplied.get("input_artifacts") != expected_artifacts:
        raise ShadowChildExternalReviewPacketValidationError(
            "review input artifacts differ from independent expectation"
        )
    fresh = build_shadow_child_external_review_packet(
        competition_input,
        competition_contract,
        competition_report,
        transition_contract,
        transition_report,
        qualification_input,
        qualification_contract,
        qualification_report,
        review_contract,
        expected_competition_contract_digest=expected_competition_contract_digest,
        expected_competition_report_digest=expected_competition_report_digest,
        expected_competition_input_artifacts=expected_competition_input_artifacts,
        expected_transition_contract_digest=expected_transition_contract_digest,
        expected_transition_report_digest=expected_transition_report_digest,
        expected_transition_input_artifacts=expected_transition_input_artifacts,
        expected_qualification_input_digest=expected_qualification_input_digest,
        expected_qualification_contract_digest=expected_qualification_contract_digest,
        expected_qualification_report_digest=expected_qualification_report_digest,
        expected_qualification_input_artifacts=expected_qualification_input_artifacts,
        expected_review_contract_digest=expected_review_contract_digest,
        input_artifacts=expected_artifacts,
    ).to_dict()
    if fresh["report_digest"] != expected_report_digest:
        raise ShadowChildExternalReviewPacketValidationError(
            "replayed review report digest differs from expectation"
        )
    if canonical_json_bytes(supplied) != canonical_json_bytes(fresh):
        raise ShadowChildExternalReviewPacketValidationError(
            "supplied review report differs from exact replay"
        )
    return {
        "status": "VERIFIED_" + fresh["disposition"],
        "disposition": fresh["disposition"],
        "report_digest": fresh["report_digest"],
        "contract_digest": fresh["contract_digest"],
        "packet_id": fresh["packet_id"],
        "source_qualification_report_digest": fresh["source_qualification"][
            "report_digest"
        ],
        "child_theory_state_digest": fresh["child_theory_state_digest"],
        "ready_for_external_review": fresh["review_boundary"]["packet_ready"],
        "adoption_eligibility": fresh["adoption_eligibility"],
        "adoption_status": fresh["adoption_status"],
        "promotion_status": fresh["promotion_status"],
        "current_status": fresh["current_status"],
    }
