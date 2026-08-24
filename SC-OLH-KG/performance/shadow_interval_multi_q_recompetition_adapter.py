"""Emit a lossless interval/two-Q seed for a future competition V2.

This additive adapter exact-replays the complete shadow chain through the
post-restriction adjudication report, resolves one canonical state without
projection, and emits a content-addressed handoff seed.  The required upstream
exact replay does execute the frozen verification chain, including competition
V1 mechanics.  The adapter itself consumes no new evidence, creates no
evaluator epoch, submits no seed to a competition engine, and never adopts,
promotes, makes current, mutates, or rolls back a theory.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping

from performance.shadow_post_restriction_adjudication import (
    ShadowPostRestrictionAdjudicationValidationError,
    verify_shadow_post_restriction_adjudication,
)
from performance.theory_operation_competition import (
    CompetitionValidationError,
    canonical_json_bytes,
)


CONTRACT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-recompetition-adapter-contract/1"
)
CONTRACT_ID = "shadow_interval_multi_q_recompetition_adapter_v1"
SOURCE_ADJUDICATION_CONTRACT_ID = "shadow_post_restriction_adjudication_v1"
SOURCE_ADJUDICATION_CONTRACT_DIGEST = (
    "sha256:dc870252207785f1d2ff4768dbf7d9fcedba7e0580554b0e647df82757d461ef"
)
SOURCE_ADJUDICATION_REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-post-restriction-adjudication-report/1"
)
INPUT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-recompetition-adapter-input/1"
)
REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-recompetition-adapter-report/1"
)
SEED_SCHEMA_VERSION = "sc-olh-kg.shadow-interval-multi-q-recompetition-seed/1"
GENESIS_DIGEST = "sha256:" + "0" * 64

REQUESTED_BRIDGE = "FINITE_INTERVAL_TWO_Q_THEORY_OPERATION_RECOMPETITION_SEED"
REQUIRED_V2_CORE = "shadow_interval_multi_q_theory_operation_competition_v2"
REQUIRED_V2_STATUS = "REQUIRED_NOT_IMPLEMENTED_BY_ADAPTER"

ADOPTION_ELIGIBILITY = "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED"
ADOPTION_STATUS = "NOT_ADOPTED_SHADOW_ONLY"
PROMOTION_STATUS = "NOT_PROMOTED"
CURRENT_STATUS = "NOT_CURRENT"

PROBE_IDS = [
    "absolute_error_point_prediction",
    "normalized_signed_interval_boundary_margin",
]
DIAGNOSTIC_ORDER = [
    "reestimate",
    "noise",
    "scope",
    "mixture",
    "simplify",
    "robustify",
    "new_probe",
    "language_last",
]
OPERATION_REGISTRY = [
    {"operation_id": "reestimate", "operation_kind": "expand"},
    {"operation_id": "noise", "operation_kind": "expand"},
    {"operation_id": "scope", "operation_kind": "restrict"},
    {"operation_id": "mixture", "operation_kind": "expand"},
    {"operation_id": "simplify", "operation_kind": "quotient"},
    {"operation_id": "robustify", "operation_kind": "expand"},
    {"operation_id": "new_probe", "operation_kind": "probe"},
    {"operation_id": "language_last", "operation_kind": "language"},
]

RETAIN_SOURCE_DISPOSITION = (
    "POST_RESTRICTION_QUALIFIED_RETAIN_RESTRICTED_SHADOW"
)
ROLLBACK_SOURCE_DISPOSITION = (
    "POST_RESTRICTION_FAILED_SOURCE_SHADOW_ROLLBACK_TARGET_SELECTED"
)
BOTH_FAILED_SOURCE_DISPOSITION = (
    "POST_RESTRICTION_AND_SOURCE_FAILED_RECOMPETITION_ADAPTER_REQUIRED"
)
NEEDS_EVIDENCE_SOURCE_DISPOSITION = "POST_RESTRICTION_NEEDS_NEW_EVIDENCE"
INCOMPARABLE_SOURCE_DISPOSITION = (
    "POST_RESTRICTION_INCOMPARABLE_EVALUATOR_EPOCH"
)

EMITTED_RESTRICTED_DISPOSITION = (
    "EMITTED_RESTRICTED_SHADOW_RECOMPETITION_SEED"
)
EMITTED_ROLLBACK_DISPOSITION = (
    "EMITTED_SOURCE_ROLLBACK_TARGET_RECOMPETITION_SEED"
)
EMITTED_REPAIR_DISPOSITION = (
    "EMITTED_UNQUALIFIED_SOURCE_REPAIR_RECOMPETITION_SEED"
)
NEEDS_EVIDENCE_DISPOSITION = (
    "RECOMPETITION_ADAPTER_NEEDS_NEW_POST_RESTRICTION_EVIDENCE"
)
INCOMPARABLE_DISPOSITION = (
    "RECOMPETITION_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH"
)

SOURCE_DISPOSITION_REGISTRY = [
    {
        "source_disposition": RETAIN_SOURCE_DISPOSITION,
        "adapter_disposition": EMITTED_RESTRICTED_DISPOSITION,
        "seed_kind": "RESTRICTED_SHADOW",
    },
    {
        "source_disposition": ROLLBACK_SOURCE_DISPOSITION,
        "adapter_disposition": EMITTED_ROLLBACK_DISPOSITION,
        "seed_kind": "SOURCE_ROLLBACK_TARGET",
    },
    {
        "source_disposition": BOTH_FAILED_SOURCE_DISPOSITION,
        "adapter_disposition": EMITTED_REPAIR_DISPOSITION,
        "seed_kind": "UNQUALIFIED_SOURCE_REPAIR_BASE",
    },
    {
        "source_disposition": NEEDS_EVIDENCE_SOURCE_DISPOSITION,
        "adapter_disposition": NEEDS_EVIDENCE_DISPOSITION,
        "seed_kind": None,
    },
    {
        "source_disposition": INCOMPARABLE_SOURCE_DISPOSITION,
        "adapter_disposition": INCOMPARABLE_DISPOSITION,
        "seed_kind": None,
    },
]

SEED_RESOLUTION_POLICY = {
    "retain_restricted_source": "EXACT_VERIFIED_RESTRICTED_SHADOW_STATE_BYTES",
    "source_rollback_target": (
        "EXACT_VERIFIED_SOURCE_PROBE_EXPANDED_SHADOW_STATE_BYTES"
    ),
    "both_failed_repair_base": (
        "EXACT_VERIFIED_SOURCE_PROBE_EXPANDED_SHADOW_STATE_BYTES_UNQUALIFIED"
    ),
    "precondition_blocked_seed": None,
    "theory_state_projection_allowed_by_adapter": False,
    "rollback_execution_allowed_by_adapter": False,
    "state_mutation_allowed_by_adapter": False,
    "both_failed_required_certificates": [
        "SOURCE_AND_RESTRICTED_FRESH_QUALIFICATION_FALSE",
        "STRICT_FINITE_INTERVAL_SUBSET",
        "EXACT_TWO_Q_REGISTRY",
        "V_REGISTRY_BYTE_EQUAL",
        "CENTER_PREDICTIONS_BYTE_EQUAL",
        "RESTRICTED_RADII_LTE_SOURCE",
        "FINITE_INTERVAL_MONOTONICITY_DOMINANCE",
    ],
}

INTERVAL_MULTI_Q_INTERFACE = {
    "model_kind": "finite_interval_table",
    "probe_ids": PROBE_IDS,
    "allowed_radius_groupings": ["global", "per_scope", "per_context"],
    "require_exact_registered_radius_group_keys": True,
    "require_all_radii_finite_nonnegative": True,
    "require_v_registry_byte_preserved": True,
    "require_object_space_byte_preserved": True,
    "require_scope_ids_byte_preserved": True,
    "require_removable_feature_ids_byte_preserved": True,
}

FUTURE_COMPETITION_REQUIREMENTS = {
    "required_core": REQUIRED_V2_CORE,
    "implementation_status": REQUIRED_V2_STATUS,
    "required_diagnostic_order": DIAGNOSTIC_ORDER,
    "operation_registry": OPERATION_REGISTRY,
    "require_new_evaluator_epoch": True,
    "required_fresh_splits": ["discovery", "validation", "stress"],
    "require_complete_context_scope_cartesian_per_split": True,
    "prior_record_reuse_allowed": False,
    "cross_epoch_pooling_allowed": False,
    "language_last_resort_deferred": True,
    "language_expansion_executed_by_adapter": False,
}

FUTURE_EVIDENCE_REQUIREMENT = {
    "new_evaluator_epoch_required": True,
    "required_fresh_splits": ["discovery", "validation", "stress"],
    "complete_context_scope_cartesian_per_split_required": True,
    "prior_record_reuse_allowed": False,
    "cross_epoch_pooling_allowed": False,
}

RECORD_LIFECYCLE_POLICY = {
    "all_prior_record_classes_role": "AUDIT_ONLY_SCORING_EXCLUDED",
    "adapter_record_role": "STATE_HANDOFF_ONLY_NO_SCORING",
    "new_evidence_consumed_by_adapter": False,
    "new_evaluator_epoch_created_by_adapter": False,
    "future_v2_requires_new_unconsumed_evidence": True,
    "cross_epoch_pooling_allowed": False,
    "logical_selective_erasure_applied": True,
    "physical_erasure": "NOT_PERFORMED",
}

AUTHORITY_BOUNDARY = {
    "scope": "LOCAL_SHADOW_INTERVAL_MULTI_Q_RECOMPETITION_ADAPTER_ONLY",
    "adapter_execution": True,
    "upstream_shadow_chain_exact_replay_performed": True,
    "upstream_competition_v1_exact_replay_performed": True,
    "upstream_scoring_exact_replay_performed": True,
    "adapter_seed_submitted_to_competition_v1": False,
    "adapter_competition_v2_execution": False,
    "adapter_candidate_synthesis_performed": False,
    "adapter_candidate_evaluation_performed": False,
    "adapter_scoring_performed": False,
    "new_evidence_consumed_by_adapter": False,
    "new_evaluator_epoch_created_by_adapter": False,
    "source_state_mutated_by_adapter": False,
    "restricted_state_mutated_by_adapter": False,
    "rollback_executed_by_adapter": False,
    "language_expansion_executed_by_adapter": False,
    "adoption_decided_by_adapter": False,
    "promotion_decided_by_adapter": False,
    "current_pointer_written_by_adapter": False,
    "parent_or_source_state_written_by_adapter": False,
    "external_data_attestation": "REQUIRED_NOT_PRESENT",
    "external_evaluator_attestation": "REQUIRED_NOT_PRESENT",
    "external_adoption_authority": "REQUIRED_NOT_PRESENT",
}

MANDATORY_NONCLAIMS = (
    "shadow_only",
    "adapter_only",
    "state_handoff_is_not_theory_operation",
    "adapter_handoff_finite_interval_table_only",
    "adapter_handoff_fixed_two_probe_registry_only",
    "no_adapter_point_projection",
    "no_adapter_lossy_probe_projection",
    "adapter_seed_not_submitted_to_competition_v1",
    "competition_v1_unchanged",
    "competition_v2_required_not_implemented",
    "no_adapter_candidate_synthesis",
    "no_adapter_candidate_evaluation",
    "no_adapter_scoring",
    "no_new_evidence_consumed_by_adapter",
    "future_v2_requires_new_unconsumed_evidence",
    "no_cross_epoch_pooling",
    "no_adapter_new_probe_execution",
    "no_adapter_restriction_or_quotient_execution",
    "no_adapter_language_or_predicate_invention",
    "language_last_resort_not_certified",
    "no_adapter_language_expansion_execution",
    "both_failed_source_is_unqualified_repair_base_not_accepted_theory",
    "two_q_dominance_is_frozen_table_and_supplied_evidence_only",
    "rollback_target_selection_is_not_rollback_execution",
    "no_adapter_rollback_execution",
    "no_adapter_state_mutation",
    "no_adapter_adoption_eligibility_determination",
    "no_adapter_adoption",
    "no_adapter_promotion",
    "no_adapter_current_pointer_write",
    "no_adapter_h_t_to_h_t_plus_1_acceptance",
    "no_adapter_external_data_or_evaluator_attestation",
    "adapter_does_not_call_run_one",
    "no_adapter_benchmark_execution",
    "no_adapter_scheduler_or_network_access",
    "no_operations_research_baseline_or_claim_change",
    "no_paper_promotion",
    "report_digest_is_not_a_signature",
    "no_scientific_validity_or_generalization_claim",
)

REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "contract_digest",
        "adapter_input_digest",
        "adapter_id",
        "source_adjudication",
        "source_disposition",
        "source_cycle_route",
        "source_state_catalog",
        "seed_resolution",
        "recompetition_seed",
        "recompetition_seed_digest",
        "interface_certificate",
        "competition_v1_incompatibility",
        "competition_v2_handoff",
        "disposition",
        "record_lifecycle_extension",
        "authority_boundary",
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

SEED_FIELDS = frozenset(
    {
        "schema_version",
        "seed_id",
        "source_adapter_input_digest",
        "source_adjudication_contract_digest",
        "source_adjudication_report_digest",
        "source_disposition",
        "seed_kind",
        "qualification_status",
        "theory_state",
        "theory_state_digest",
        "alternate_state_digests",
        "model_interface",
        "operation_registry",
        "required_diagnostic_order",
        "future_evidence_requirement",
        "prior_record_exclusion",
        "adoption_status",
        "current_status",
    }
)


class ShadowIntervalMultiQRecompetitionAdapterValidationError(ValueError):
    """Raised when the frozen adapter contract or exact replay fails."""


class ShadowIntervalMultiQRecompetitionAdapterDisposition(str, Enum):
    EMITTED_RESTRICTED_SHADOW_RECOMPETITION_SEED = EMITTED_RESTRICTED_DISPOSITION
    EMITTED_SOURCE_ROLLBACK_TARGET_RECOMPETITION_SEED = (
        EMITTED_ROLLBACK_DISPOSITION
    )
    EMITTED_UNQUALIFIED_SOURCE_REPAIR_RECOMPETITION_SEED = (
        EMITTED_REPAIR_DISPOSITION
    )
    RECOMPETITION_ADAPTER_NEEDS_NEW_POST_RESTRICTION_EVIDENCE = (
        NEEDS_EVIDENCE_DISPOSITION
    )
    RECOMPETITION_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH = (
        INCOMPARABLE_DISPOSITION
    )


@dataclass(frozen=True)
class ShadowIntervalMultiQRecompetitionAdapterResult:
    report: Mapping[str, Any]

    @property
    def disposition(self) -> str:
        return str(self.report["disposition"])

    @property
    def report_digest(self) -> str:
        return str(self.report["report_digest"])

    @property
    def seed_emitted(self) -> bool:
        return self.report["recompetition_seed"] is not None

    @property
    def seed_kind(self) -> str | None:
        seed = self.report["recompetition_seed"]
        return None if seed is None else str(seed["seed_kind"])

    def to_dict(self) -> dict[str, Any]:
        return _copy(self.report)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
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
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} keys differ; missing={missing}, extra={extra}"
        )


def _string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} must be a non-empty string"
        )
    return value


def _finite_number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} must be a finite number"
        )
    return float(value)


def _copy(value: Any) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("ascii"))
    except (
        CompetitionValidationError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"value is not detached finite canonical JSON: {exc}"
        ) from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_digest(value: Any, label: str) -> str:
    digest = _string(value, label)
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} must be sha256:<64hex>"
        )
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} is not hexadecimal"
        ) from exc
    return digest


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} differs from frozen interval/multi-Q adapter V1"
        )


def _reject_observed_values(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        if "observed_value" in value:
            raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
                f"{label} must not embed observation values"
            )
        for key, item in value.items():
            _reject_observed_values(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_observed_values(item, f"{label}[{index}]")


def _validate_contract(contract_value: Any) -> dict[str, Any]:
    contract = _mapping(contract_value, "adapter_contract")
    _exact_keys(
        contract,
        {
            "schema_version",
            "contract_id",
            "source_adjudication_contract_id",
            "source_adjudication_contract_digest",
            "source_adjudication_report_schema_version",
            "input_schema_version",
            "report_schema_version",
            "seed_schema_version",
            "source_disposition_registry",
            "seed_resolution_policy",
            "interval_multi_q_interface",
            "future_competition_requirements",
            "record_lifecycle_policy",
            "authority_boundary",
            "nonclaims",
        },
        "adapter_contract",
    )
    frozen = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "source_adjudication_contract_id": SOURCE_ADJUDICATION_CONTRACT_ID,
        "source_adjudication_contract_digest": SOURCE_ADJUDICATION_CONTRACT_DIGEST,
        "source_adjudication_report_schema_version": (
            SOURCE_ADJUDICATION_REPORT_SCHEMA_VERSION
        ),
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "seed_schema_version": SEED_SCHEMA_VERSION,
    }
    for key, expected in frozen.items():
        _require_equal(contract[key], expected, key)
    for key, expected in (
        ("source_disposition_registry", SOURCE_DISPOSITION_REGISTRY),
        ("seed_resolution_policy", SEED_RESOLUTION_POLICY),
        ("interval_multi_q_interface", INTERVAL_MULTI_Q_INTERFACE),
        ("future_competition_requirements", FUTURE_COMPETITION_REQUIREMENTS),
        ("record_lifecycle_policy", RECORD_LIFECYCLE_POLICY),
        ("authority_boundary", AUTHORITY_BOUNDARY),
    ):
        _require_equal(_copy(contract[key]), expected, key)
    nonclaims = contract["nonclaims"]
    if not isinstance(nonclaims, list) or tuple(nonclaims) != MANDATORY_NONCLAIMS:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "nonclaims differ from the exact mandatory V1 list"
        )
    return _copy(contract)


def validate_shadow_interval_multi_q_recompetition_adapter_contract(
    contract_value: Any,
) -> dict[str, Any]:
    """Validate and detach the frozen adapter contract."""

    return _validate_contract(contract_value)


def _seed_id_components(
    *,
    adjudication_contract_digest: str,
    adjudication_report_digest: str,
    source_probe_expanded_state_digest: str,
    restricted_state_digest: str,
    adapter_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    contract = _validate_contract(adapter_contract)
    source_contract = _require_digest(
        adjudication_contract_digest, "adjudication_contract_digest"
    )
    if source_contract != SOURCE_ADJUDICATION_CONTRACT_DIGEST:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "adjudication contract digest is not the pinned V1 source"
        )
    payload = {
        "adjudication_contract_digest": source_contract,
        "adjudication_report_digest": _require_digest(
            adjudication_report_digest, "adjudication_report_digest"
        ),
        "source_probe_expanded_state_digest": _require_digest(
            source_probe_expanded_state_digest,
            "source_probe_expanded_state_digest",
        ),
        "restricted_state_digest": _require_digest(
            restricted_state_digest, "restricted_state_digest"
        ),
        "adapter_contract_digest": _digest(contract),
        "requested_bridge": REQUESTED_BRIDGE,
    }
    seed_id = "shadow-interval-multi-q-recompetition-seed:" + _digest(payload)[7:]
    return payload, seed_id


def derive_shadow_interval_multi_q_recompetition_seed_id(
    *,
    adjudication_contract_digest: str,
    adjudication_report_digest: str,
    source_probe_expanded_state_digest: str,
    restricted_state_digest: str,
    adapter_contract: Mapping[str, Any],
) -> str:
    """Derive the canonical adapter seed identifier without observations."""

    return _seed_id_components(
        adjudication_contract_digest=adjudication_contract_digest,
        adjudication_report_digest=adjudication_report_digest,
        source_probe_expanded_state_digest=source_probe_expanded_state_digest,
        restricted_state_digest=restricted_state_digest,
        adapter_contract=adapter_contract,
    )[1]


def _prior_record_exclusion(
    adjudication_report: Mapping[str, Any],
) -> dict[str, str]:
    lifecycle = _mapping(
        adjudication_report.get("record_lifecycle_extension"),
        "source adjudication record_lifecycle_extension",
    )
    source_keys = {
        "competition": "competition_records",
        "qualification": "qualification_records",
        "failure_boundary_probe": "failure_boundary_probe_records",
        "restriction": "restriction_competition_records",
        "post_restriction_adjudication": "post_restriction_adjudication_records",
    }
    result: dict[str, str] = {}
    for output_key, source_key in source_keys.items():
        record = _mapping(lifecycle.get(source_key), f"source {source_key}")
        result[output_key] = _require_digest(
            record.get("observation_id_digest"),
            f"source {source_key}.observation_id_digest",
        )
    return result


def _normalize_input(
    input_value: Any,
    *,
    adjudication_report: Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(input_value, "adapter_input")
    _exact_keys(
        value,
        {
            "schema_version",
            "adapter_id",
            "source_adjudication",
            "requested_bridge",
            "prior_record_exclusion",
        },
        "adapter_input",
    )
    _require_equal(value["schema_version"], INPUT_SCHEMA_VERSION, "input schema")
    source = _mapping(value["source_adjudication"], "source_adjudication")
    _exact_keys(
        source,
        {
            "adjudication_contract_digest",
            "adjudication_report_digest",
            "adjudication_id",
            "source_probe_expanded_shadow_theory_state_digest",
            "restricted_shadow_theory_state_digest",
        },
        "source_adjudication",
    )
    source_summary = _mapping(
        adjudication_report.get("source_restriction"),
        "adjudication source_restriction",
    )
    expected_source = {
        "adjudication_contract_digest": adjudication_report.get("contract_digest"),
        "adjudication_report_digest": adjudication_report.get("report_digest"),
        "adjudication_id": adjudication_report.get("adjudication_id"),
        "source_probe_expanded_shadow_theory_state_digest": source_summary.get(
            "source_probe_expanded_shadow_theory_state_digest"
        ),
        "restricted_shadow_theory_state_digest": source_summary.get(
            "restricted_shadow_theory_state_digest"
        ),
    }
    _require_equal(_copy(source), expected_source, "source_adjudication")
    _require_equal(value["requested_bridge"], REQUESTED_BRIDGE, "requested_bridge")
    exclusion = _mapping(
        value["prior_record_exclusion"], "prior_record_exclusion"
    )
    expected_exclusion = _prior_record_exclusion(adjudication_report)
    _exact_keys(exclusion, set(expected_exclusion), "prior_record_exclusion")
    _require_equal(_copy(exclusion), expected_exclusion, "prior_record_exclusion")
    public = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "adapter_id": _string(value["adapter_id"], "adapter_id"),
        "source_adjudication": expected_source,
        "requested_bridge": REQUESTED_BRIDGE,
        "prior_record_exclusion": expected_exclusion,
    }
    return _copy(public)


def _strings(value: Any, label: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} must be {'a non-empty' if nonempty else 'a'} list"
        )
    result = [_string(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} must not contain duplicates"
        )
    return result


def _model_geometry(state: Mapping[str, Any], label: str) -> dict[str, Any]:
    object_space = _mapping(state.get("object_space"), f"{label} object_space")
    feature_ids = _strings(object_space.get("feature_ids"), f"{label} feature_ids")
    contexts_value = object_space.get("contexts")
    if not isinstance(contexts_value, list) or not contexts_value:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} contexts must be a non-empty list"
        )
    contexts = [_copy(_mapping(item, f"{label} context")) for item in contexts_value]
    context_keys = [canonical_json_bytes(item) for item in contexts]
    if len(context_keys) != len(set(context_keys)):
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} contexts are duplicated"
        )
    for index, context in enumerate(contexts):
        _exact_keys(context, set(feature_ids), f"{label} contexts[{index}]")
    scopes = _strings(state.get("scope_ids"), f"{label} scope_ids")
    _strings(
        state.get("removable_feature_ids"),
        f"{label} removable_feature_ids",
        nonempty=False,
    )
    probe_ids = _strings(state.get("probe_ids"), f"{label} probe_ids")
    if probe_ids != PROBE_IDS:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} does not carry the exact fixed two-Q registry"
        )
    violation_functionals = state.get("violation_functionals")
    if not isinstance(violation_functionals, list) or not violation_functionals:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} violation_functionals must be a non-empty list"
        )

    model = _mapping(state.get("model_class"), f"{label} model_class")
    _exact_keys(
        model,
        {"kind", "center_predictions", "radius_grouping", "radii"},
        f"{label} model_class",
    )
    if model.get("kind") != INTERVAL_MULTI_Q_INTERFACE["model_kind"]:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} model is not the frozen finite interval table"
        )
    center_values = model.get("center_predictions")
    if not isinstance(center_values, list) or not center_values:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} center_predictions must be a non-empty list"
        )
    center_keys: list[bytes] = []
    for index, raw in enumerate(center_values):
        item = _mapping(raw, f"{label} center_predictions[{index}]")
        _exact_keys(item, {"context", "value"}, f"{label} center_predictions[{index}]")
        context = _copy(
            _mapping(item["context"], f"{label} center_predictions[{index}].context")
        )
        _exact_keys(
            context,
            set(feature_ids),
            f"{label} center_predictions[{index}].context",
        )
        center_keys.append(canonical_json_bytes(context))
        _finite_number(item["value"], f"{label} center_predictions[{index}].value")
    if len(center_keys) != len(set(center_keys)) or set(center_keys) != set(context_keys):
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} center predictions do not exactly cover registered contexts"
        )

    grouping = model.get("radius_grouping")
    if grouping not in INTERVAL_MULTI_Q_INTERFACE["allowed_radius_groupings"]:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} radius grouping is unsupported"
        )
    radii_value = model.get("radii")
    if not isinstance(radii_value, list) or not radii_value:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} radii must be a non-empty list"
        )
    radius_by_key: dict[bytes, float] = {}
    group_by_key: dict[bytes, dict[str, Any]] = {}
    raw_groups: list[dict[str, Any]] = []
    for index, raw in enumerate(radii_value):
        item = _mapping(raw, f"{label} radii[{index}]")
        _exact_keys(item, {"group", "radius"}, f"{label} radii[{index}]")
        group = _copy(_mapping(item["group"], f"{label} radii[{index}].group"))
        radius = _finite_number(item["radius"], f"{label} radii[{index}].radius")
        if radius < 0.0:
            raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
                f"{label} radii must be nonnegative"
            )
        key = canonical_json_bytes(group)
        if key in radius_by_key:
            raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
                f"{label} radius group keys are duplicated"
            )
        radius_by_key[key] = radius
        group_by_key[key] = group
        raw_groups.append(_copy(group))
    if grouping == "global":
        expected_groups = [{"global": "*"}]
    elif grouping == "per_scope":
        expected_groups = [{"scope_id": scope} for scope in scopes]
    else:
        expected_groups = [{"context": context} for context in contexts]
    expected_by_key = {canonical_json_bytes(item): item for item in expected_groups}
    if set(radius_by_key) != set(expected_by_key):
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"{label} radius group keys do not exactly equal the registered groups"
        )
    sorted_keys = sorted(radius_by_key)
    return {
        "model": _copy(model),
        "model_digest": _digest(model),
        "grouping": grouping,
        "raw_groups": raw_groups,
        "groups": [_copy(group_by_key[key]) for key in sorted_keys],
        "expected_groups": [
            _copy(expected_by_key[key]) for key in sorted(expected_by_key)
        ],
        "radius_by_key": radius_by_key,
        "all_radii_finite_nonnegative": True,
    }


def _interface_certificate(
    *,
    source_state: Mapping[str, Any],
    source_state_digest: str,
    restricted_state: Mapping[str, Any],
    restricted_state_digest: str,
    seed_state: Mapping[str, Any] | None,
    seed_state_digest: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = _model_geometry(source_state, "source probe-expanded state")
    restricted = _model_geometry(restricted_state, "restricted state")
    q_equal = canonical_json_bytes(source_state.get("probe_ids")) == canonical_json_bytes(
        restricted_state.get("probe_ids")
    )
    v_equal = canonical_json_bytes(
        source_state.get("violation_functionals")
    ) == canonical_json_bytes(restricted_state.get("violation_functionals"))
    object_equal = canonical_json_bytes(source_state.get("object_space")) == canonical_json_bytes(
        restricted_state.get("object_space")
    )
    scope_equal = canonical_json_bytes(source_state.get("scope_ids")) == canonical_json_bytes(
        restricted_state.get("scope_ids")
    )
    removable_equal = canonical_json_bytes(
        source_state.get("removable_feature_ids")
    ) == canonical_json_bytes(restricted_state.get("removable_feature_ids"))
    centers_equal = canonical_json_bytes(
        source["model"]["center_predictions"]
    ) == canonical_json_bytes(restricted["model"]["center_predictions"])
    grouping_equal = (
        source["grouping"] == restricted["grouping"]
        and canonical_json_bytes(source["raw_groups"])
        == canonical_json_bytes(restricted["raw_groups"])
    )
    source_groups_exact = source["groups"] == source["expected_groups"]
    restricted_groups_exact = restricted["groups"] == restricted["expected_groups"]
    seed_geometry = None
    seed_equal: bool | None = None
    seed_probe_ids: list[str] | None = None
    seed_two_q_exact: bool | None = None
    if seed_state is not None:
        seed_geometry = _model_geometry(seed_state, "resolved seed state")
        seed_equal = _digest(seed_state) == seed_state_digest and (
            canonical_json_bytes(seed_state) == canonical_json_bytes(source_state)
            or canonical_json_bytes(seed_state) == canonical_json_bytes(restricted_state)
        )
        seed_probe_ids = _copy(seed_state.get("probe_ids"))
        seed_two_q_exact = seed_probe_ids == PROBE_IDS
    verified = all(
        (
            source_state_digest == _digest(source_state),
            restricted_state_digest == _digest(restricted_state),
            q_equal,
            v_equal,
            object_equal,
            scope_equal,
            removable_equal,
            centers_equal,
            grouping_equal,
            source_groups_exact,
            restricted_groups_exact,
            source["all_radii_finite_nonnegative"],
            restricted["all_radii_finite_nonnegative"],
        )
    ) and (seed_state is None or bool(seed_equal and seed_two_q_exact))
    certificate = {
        "certificate_kind": "LOSSLESS_FINITE_INTERVAL_TWO_Q_STATE_INTERFACE",
        "source_probe_expanded_state_digest": source_state_digest,
        "restricted_state_digest": restricted_state_digest,
        "seed_state_digest": seed_state_digest,
        "source_model_kind": source["model"]["kind"],
        "restricted_model_kind": restricted["model"]["kind"],
        "seed_model_kind": None if seed_geometry is None else seed_geometry["model"]["kind"],
        "required_probe_ids": _copy(PROBE_IDS),
        "source_probe_ids": _copy(source_state.get("probe_ids")),
        "restricted_probe_ids": _copy(restricted_state.get("probe_ids")),
        "seed_probe_ids": seed_probe_ids,
        "source_two_q_exact": source_state.get("probe_ids") == PROBE_IDS,
        "restricted_two_q_exact": restricted_state.get("probe_ids") == PROBE_IDS,
        "seed_two_q_exact": seed_two_q_exact,
        "q_registry_byte_equal": q_equal,
        "v_registry_byte_equal": v_equal,
        "object_space_byte_equal": object_equal,
        "scope_ids_byte_equal": scope_equal,
        "removable_feature_ids_byte_equal": removable_equal,
        "center_predictions_byte_equal": centers_equal,
        "radius_grouping_byte_equal": grouping_equal,
        "source_radius_group_keys": _copy(source["groups"]),
        "restricted_radius_group_keys": _copy(restricted["groups"]),
        "expected_radius_group_keys": _copy(source["expected_groups"]),
        "source_radius_group_keys_exact": source_groups_exact,
        "restricted_radius_group_keys_exact": restricted_groups_exact,
        "all_source_radii_finite_nonnegative": source[
            "all_radii_finite_nonnegative"
        ],
        "all_restricted_radii_finite_nonnegative": restricted[
            "all_radii_finite_nonnegative"
        ],
        "seed_state_byte_equal_to_resolved_source": seed_equal,
        "verified": verified,
    }
    if not verified:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "verified source states do not satisfy the lossless interval/two-Q interface"
        )
    return certificate, source, restricted


def _both_failed_repair_certificate(
    *,
    adjudication_report: Mapping[str, Any],
    restriction_report: Mapping[str, Any],
    interface_certificate: Mapping[str, Any],
    source_geometry: Mapping[str, Any],
    restricted_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    source_results = _mapping(
        adjudication_report.get("source_probe_results"), "source probe results"
    )
    restricted_results = _mapping(
        adjudication_report.get("restricted_probe_results"),
        "restricted probe results",
    )
    tradeoff = _mapping(
        adjudication_report.get("finite_interval_tradeoff"),
        "finite interval tradeoff",
    )
    monotonicity = _mapping(
        adjudication_report.get("monotonicity_certificate"),
        "monotonicity certificate",
    )
    restriction_certificate = _mapping(
        restriction_report.get("restriction_certificate"),
        "restriction certificate",
    )
    radius_keys_equal = set(source_geometry["radius_by_key"]) == set(
        restricted_geometry["radius_by_key"]
    )
    radii_lte = radius_keys_equal and all(
        restricted_geometry["radius_by_key"][key]
        <= source_geometry["radius_by_key"][key]
        for key in source_geometry["radius_by_key"]
    )
    values = {
        "source_fresh_qualification_passed": source_results.get(
            "fresh_qualification_passed"
        ),
        "restricted_fresh_qualification_passed": restricted_results.get(
            "fresh_qualification_passed"
        ),
        "restriction_strict_subset_verified": restriction_certificate.get(
            "strict_subset_verified"
        ),
        "tradeoff_strict_subset_verified": tradeoff.get("strict_subset_verified"),
        "exact_two_q_registry": (
            interface_certificate.get("source_two_q_exact") is True
            and interface_certificate.get("restricted_two_q_exact") is True
            and interface_certificate.get("q_registry_byte_equal") is True
        ),
        "v_registry_byte_equal": interface_certificate.get(
            "v_registry_byte_equal"
        ),
        "center_predictions_byte_equal": interface_certificate.get(
            "center_predictions_byte_equal"
        ),
        "all_restricted_radii_lte_source": (
            radii_lte
            and tradeoff.get("all_restricted_radii_lte_source") is True
            and restriction_certificate.get("all_restricted_radii_lte_source")
            is True
        ),
        "monotonicity_verified": monotonicity.get("verified"),
        "every_restricted_pass_implies_source_pass": monotonicity.get(
            "every_restricted_pass_implies_source_pass"
        ),
    }
    verified = (
        values["source_fresh_qualification_passed"] is False
        and values["restricted_fresh_qualification_passed"] is False
        and all(
            values[key] is True
            for key in (
                "restriction_strict_subset_verified",
                "tradeoff_strict_subset_verified",
                "exact_two_q_registry",
                "v_registry_byte_equal",
                "center_predictions_byte_equal",
                "all_restricted_radii_lte_source",
                "monotonicity_verified",
                "every_restricted_pass_implies_source_pass",
            )
        )
    )
    result = {
        "certificate_kind": (
            "UNQUALIFIED_SOURCE_REPAIR_BASE_WITH_INTERVAL_TWO_Q_DOMINANCE"
        ),
        **values,
        "verified": verified,
    }
    if not verified:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "both-failed repair base lacks the complete frozen dominance proof"
        )
    return result


def _source_summary(
    report: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    source_restriction = _mapping(
        report.get("source_restriction"), "source restriction summary"
    )
    return {
        "verification_status": receipt.get("status"),
        "contract_id": report.get("contract_id"),
        "contract_digest": report.get("contract_digest"),
        "report_digest": report.get("report_digest"),
        "adjudication_id": report.get("adjudication_id"),
        "disposition": report.get("disposition"),
        "cycle_route": report.get("cycle_route"),
        "source_probe_expanded_shadow_theory_state_digest": source_restriction.get(
            "source_probe_expanded_shadow_theory_state_digest"
        ),
        "restricted_shadow_theory_state_digest": source_restriction.get(
            "restricted_shadow_theory_state_digest"
        ),
        "adoption_status": report.get("adoption_status"),
    }


def _qualification_status(results: Any) -> str:
    if results is None:
        return "NOT_EVALUATED_POST_RESTRICTION_PRECONDITION_BLOCKED"
    value = _mapping(results, "probe results").get("fresh_qualification_passed")
    if type(value) is not bool:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "source probe results lack a boolean fresh qualification status"
        )
    return "FRESH_QUALIFIED" if value else "FRESH_FAILED"


def _state_catalog(
    *,
    source_state: Mapping[str, Any],
    source_state_digest: str,
    restricted_state: Mapping[str, Any],
    restricted_state_digest: str,
    adjudication_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_probe_expanded_shadow": {
            "state_kind": "SOURCE_PROBE_EXPANDED_SHADOW",
            "theory_state_digest": source_state_digest,
            "model_class_digest": _digest(source_state.get("model_class")),
            "qualification_status": _qualification_status(
                adjudication_report.get("source_probe_results")
            ),
        },
        "restricted_shadow": {
            "state_kind": "RESTRICTED_SHADOW",
            "theory_state_digest": restricted_state_digest,
            "model_class_digest": _digest(restricted_state.get("model_class")),
            "qualification_status": _qualification_status(
                adjudication_report.get("restricted_probe_results")
            ),
        },
    }


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


def adapt_shadow_interval_multi_q_recompetition_seed(
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
    probe_input: Mapping[str, Any],
    probe_contract: Mapping[str, Any],
    probe_report: Mapping[str, Any],
    restriction_input: Mapping[str, Any],
    restriction_contract: Mapping[str, Any],
    restriction_report: Mapping[str, Any],
    adjudication_input: Mapping[str, Any],
    adjudication_contract: Mapping[str, Any],
    adjudication_report: Mapping[str, Any],
    adapter_input: Mapping[str, Any],
    adapter_contract: Mapping[str, Any],
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
    expected_probe_input_digest: str,
    expected_probe_contract_digest: str,
    expected_probe_report_digest: str,
    expected_probe_input_artifacts: Mapping[str, Any] | None,
    expected_restriction_input_digest: str,
    expected_restriction_contract_digest: str,
    expected_restriction_report_digest: str,
    expected_restriction_input_artifacts: Mapping[str, Any] | None,
    expected_adjudication_input_digest: str,
    expected_adjudication_contract_digest: str,
    expected_adjudication_report_digest: str,
    expected_adjudication_input_artifacts: Mapping[str, Any] | None,
    expected_adapter_input_digest: str,
    expected_adapter_contract_digest: str,
    input_artifacts: Mapping[str, Any] | None = None,
) -> ShadowIntervalMultiQRecompetitionAdapterResult:
    """Resolve exact state bytes and emit a V2 handoff seed, if routable."""

    normalized_contract = _validate_contract(adapter_contract)
    expected_contract_digest = _require_digest(
        expected_adapter_contract_digest, "expected_adapter_contract_digest"
    )
    contract_digest = _digest(normalized_contract)
    if contract_digest != expected_contract_digest:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "adapter contract digest differs from independent expectation"
        )
    expected_input_digest = _require_digest(
        expected_adapter_input_digest, "expected_adapter_input_digest"
    )
    if _digest(adapter_input) != expected_input_digest:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "adapter input digest differs from independent expectation"
        )
    if input_artifacts is not None and not isinstance(input_artifacts, Mapping):
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "input_artifacts must be an object or null"
        )
    artifacts = _copy(input_artifacts) if input_artifacts is not None else None
    _reject_observed_values(artifacts, "input_artifacts")

    expected_source_contract = _require_digest(
        expected_adjudication_contract_digest,
        "expected_adjudication_contract_digest",
    )
    if expected_source_contract != SOURCE_ADJUDICATION_CONTRACT_DIGEST:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "expected adjudication contract digest is not the pinned V1 source"
        )
    try:
        source_receipt = verify_shadow_post_restriction_adjudication(
            competition_input,
            competition_contract,
            competition_report,
            transition_contract,
            transition_report,
            qualification_input,
            qualification_contract,
            qualification_report,
            review_contract,
            review_report,
            probe_input,
            probe_contract,
            probe_report,
            restriction_input,
            restriction_contract,
            restriction_report,
            adjudication_input,
            adjudication_contract,
            adjudication_report,
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
            expected_review_report_digest=expected_review_report_digest,
            expected_review_input_artifacts=expected_review_input_artifacts,
            expected_probe_input_digest=expected_probe_input_digest,
            expected_probe_contract_digest=expected_probe_contract_digest,
            expected_probe_report_digest=expected_probe_report_digest,
            expected_probe_input_artifacts=expected_probe_input_artifacts,
            expected_restriction_input_digest=expected_restriction_input_digest,
            expected_restriction_contract_digest=expected_restriction_contract_digest,
            expected_restriction_report_digest=expected_restriction_report_digest,
            expected_restriction_input_artifacts=expected_restriction_input_artifacts,
            expected_adjudication_input_digest=expected_adjudication_input_digest,
            expected_adjudication_contract_digest=expected_source_contract,
            expected_adjudication_report_digest=expected_adjudication_report_digest,
            expected_adjudication_input_artifacts=expected_adjudication_input_artifacts,
        )
    except (
        ShadowPostRestrictionAdjudicationValidationError,
        CompetitionValidationError,
        KeyError,
        TypeError,
    ) as exc:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            f"source post-restriction adjudication verification failed: {exc}"
        ) from exc

    adjudication = _copy(_mapping(adjudication_report, "adjudication_report"))
    receipt = _copy(_mapping(source_receipt, "source adjudication receipt"))
    if adjudication.get("contract_id") != SOURCE_ADJUDICATION_CONTRACT_ID:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "source adjudication contract_id is not supported"
        )
    if adjudication.get("contract_digest") != SOURCE_ADJUDICATION_CONTRACT_DIGEST:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "source adjudication contract digest is not pinned V1"
        )
    if adjudication.get("schema_version") != SOURCE_ADJUDICATION_REPORT_SCHEMA_VERSION:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "source adjudication report schema is not supported"
        )
    if adjudication.get("adoption_status") != ADOPTION_STATUS:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "source adjudication crossed the shadow-only adoption boundary"
        )

    source_state = _copy(
        _mapping(
            _mapping(probe_report, "probe_report").get(
                "probe_expanded_shadow_theory_state"
            ),
            "source probe-expanded shadow state",
        )
    )
    restricted_state = _copy(
        _mapping(
            _mapping(restriction_report, "restriction_report").get(
                "restricted_shadow_theory_state"
            ),
            "restricted shadow state",
        )
    )
    source_restriction = _mapping(
        adjudication.get("source_restriction"), "source restriction summary"
    )
    source_state_digest = _require_digest(
        source_restriction.get(
            "source_probe_expanded_shadow_theory_state_digest"
        ),
        "source probe-expanded state digest",
    )
    restricted_state_digest = _require_digest(
        source_restriction.get("restricted_shadow_theory_state_digest"),
        "restricted state digest",
    )
    if _digest(source_state) != source_state_digest:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "source probe-expanded state digest is inconsistent"
        )
    if _digest(restricted_state) != restricted_state_digest:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "restricted state digest is inconsistent"
        )
    source_bytes = canonical_json_bytes(source_state)
    restricted_bytes = canonical_json_bytes(restricted_state)

    normalized_input = _normalize_input(
        adapter_input, adjudication_report=adjudication
    )
    semantic_input_digest = _digest(normalized_input)
    adapter_id = normalized_input["adapter_id"]
    source_disposition = _string(
        adjudication.get("disposition"), "source disposition"
    )
    registry = {
        item["source_disposition"]: item for item in SOURCE_DISPOSITION_REGISTRY
    }
    if source_disposition not in registry:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "source disposition is not registered for adapter V1"
        )
    route = registry[source_disposition]
    disposition = route["adapter_disposition"]
    seed_kind = route["seed_kind"]
    selection = adjudication.get("shadow_state_selection")

    resolved_state: dict[str, Any] | None = None
    resolved_digest: str | None = None
    resolution_status: str
    qualification_status: str | None = None
    if source_disposition == RETAIN_SOURCE_DISPOSITION:
        selected = _mapping(selection, "restricted state selection")
        resolved_state = _copy(restricted_state)
        resolved_digest = restricted_state_digest
        _require_equal(
            selected.get("selected_target_kind"),
            "RESTRICTED_SHADOW",
            "restricted selected target kind",
        )
        if (
            selected.get("selected_shadow_theory_state_digest") != resolved_digest
            or canonical_json_bytes(selected.get("selected_shadow_theory_state"))
            != restricted_bytes
        ):
            raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
                "retain route selection is not exact restricted-state bytes"
            )
        resolution_status = "RESOLVED_EXACT_RESTRICTED_SHADOW_STATE_BYTES"
        qualification_status = "QUALIFIED_RESTRICTED_ON_POST_RESTRICTION_EPOCH"
    elif source_disposition == ROLLBACK_SOURCE_DISPOSITION:
        selected = _mapping(selection, "source rollback state selection")
        resolved_state = _copy(source_state)
        resolved_digest = source_state_digest
        _require_equal(
            selected.get("selected_target_kind"),
            "SOURCE_PROBE_EXPANDED_SHADOW",
            "source rollback selected target kind",
        )
        if (
            selected.get("selected_shadow_theory_state_digest") != resolved_digest
            or canonical_json_bytes(selected.get("selected_shadow_theory_state"))
            != source_bytes
        ):
            raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
                "rollback route selection is not exact source-state bytes"
            )
        rollback = _mapping(
            adjudication.get("rollback_adjudication"), "rollback adjudication"
        )
        if rollback.get("rollback_execution_status") != "NOT_PERFORMED":
            raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
                "adapter requires rollback selection without rollback execution"
            )
        resolution_status = "RESOLVED_EXACT_SOURCE_ROLLBACK_TARGET_STATE_BYTES"
        qualification_status = "QUALIFIED_SOURCE_ROLLBACK_TARGET_NOT_EXECUTED"
    elif source_disposition == BOTH_FAILED_SOURCE_DISPOSITION:
        selected = _mapping(selection, "both-failed state selection")
        if (
            selected.get("selected_target_kind") is not None
            or selected.get("selected_shadow_theory_state") is not None
            or selected.get("selected_shadow_theory_state_digest") is not None
        ):
            raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
                "both-failed route must not claim a qualified selected target"
            )
        resolved_state = _copy(source_state)
        resolved_digest = source_state_digest
        resolution_status = "RESOLVED_EXACT_UNQUALIFIED_SOURCE_REPAIR_BASE_BYTES"
        qualification_status = "UNQUALIFIED_SOURCE_REPAIR_BASE"
    elif source_disposition == NEEDS_EVIDENCE_SOURCE_DISPOSITION:
        if selection is not None:
            raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
                "needs-evidence route must not select a seed state"
            )
        resolution_status = "NO_SEED_NEW_POST_RESTRICTION_EVIDENCE_REQUIRED"
    else:
        if selection is not None:
            raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
                "incomparable route must not select a seed state"
            )
        resolution_status = "NO_SEED_COMPARABLE_POST_RESTRICTION_EPOCH_REQUIRED"

    interface, source_geometry, restricted_geometry = _interface_certificate(
        source_state=source_state,
        source_state_digest=source_state_digest,
        restricted_state=restricted_state,
        restricted_state_digest=restricted_state_digest,
        seed_state=resolved_state,
        seed_state_digest=resolved_digest,
    )
    repair_certificate = None
    if source_disposition == BOTH_FAILED_SOURCE_DISPOSITION:
        repair_certificate = _both_failed_repair_certificate(
            adjudication_report=adjudication,
            restriction_report=_mapping(
                restriction_report, "restriction_report"
            ),
            interface_certificate=interface,
            source_geometry=source_geometry,
            restricted_geometry=restricted_geometry,
        )

    resolution = {
        "resolution_status": resolution_status,
        "selected_seed_kind": seed_kind,
        "selected_theory_state_digest": resolved_digest,
        "canonical_byte_equal_to_verified_source": (
            None
            if resolved_state is None
            else (
                canonical_json_bytes(resolved_state) == source_bytes
                or canonical_json_bytes(resolved_state) == restricted_bytes
            )
        ),
        "rollback_execution_status": "NOT_PERFORMED",
        "both_failed_repair_certificate": repair_certificate,
    }

    seed: dict[str, Any] | None = None
    seed_digest: str | None = None
    if resolved_state is not None:
        _, seed_id = _seed_id_components(
            adjudication_contract_digest=adjudication["contract_digest"],
            adjudication_report_digest=adjudication["report_digest"],
            source_probe_expanded_state_digest=source_state_digest,
            restricted_state_digest=restricted_state_digest,
            adapter_contract=normalized_contract,
        )
        selected_geometry = (
            restricted_geometry
            if seed_kind == "RESTRICTED_SHADOW"
            else source_geometry
        )
        seed = {
            "schema_version": SEED_SCHEMA_VERSION,
            "seed_id": seed_id,
            "source_adapter_input_digest": semantic_input_digest,
            "source_adjudication_contract_digest": adjudication[
                "contract_digest"
            ],
            "source_adjudication_report_digest": adjudication["report_digest"],
            "source_disposition": source_disposition,
            "seed_kind": seed_kind,
            "qualification_status": qualification_status,
            "theory_state": _copy(resolved_state),
            "theory_state_digest": resolved_digest,
            "alternate_state_digests": {
                "source_probe_expanded_shadow": source_state_digest,
                "restricted_shadow": restricted_state_digest,
            },
            "model_interface": {
                "model_kind": selected_geometry["model"]["kind"],
                "radius_grouping": selected_geometry["grouping"],
                "radius_group_keys": _copy(selected_geometry["groups"]),
                "probe_ids": _copy(PROBE_IDS),
            },
            "operation_registry": _copy(OPERATION_REGISTRY),
            "required_diagnostic_order": _copy(DIAGNOSTIC_ORDER),
            "future_evidence_requirement": _copy(FUTURE_EVIDENCE_REQUIREMENT),
            "prior_record_exclusion": _copy(
                normalized_input["prior_record_exclusion"]
            ),
            "adoption_status": ADOPTION_STATUS,
            "current_status": CURRENT_STATUS,
        }
        _exact_keys(seed, SEED_FIELDS, "recompetition_seed")
        _reject_observed_values(seed, "recompetition_seed")
        if canonical_json_bytes(seed["theory_state"]) != canonical_json_bytes(
            resolved_state
        ):
            raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
                "seed theory_state is not exact resolved state bytes"
            )
        seed_digest = _digest(seed)

    v1_incompatibility = {
        "competition_contract_id": "theory_operation_competition_v1",
        "required_model_kind": "finite_point_table",
        "required_probe_ids": ["absolute_error_point_prediction"],
        "actual_model_kind": "finite_interval_table",
        "actual_probe_ids": _copy(PROBE_IDS),
        "lossless_projection_to_v1_exists": False,
        "point_projection_drops_radii": True,
        "single_q_projection_drops_probe_id": (
            "normalized_signed_interval_boundary_margin"
        ),
        "upstream_competition_v1_exact_replay_performed": True,
        "adapter_seed_submitted_to_competition_v1": False,
        "compatible": False,
    }
    v2_handoff = {
        "required_core": REQUIRED_V2_CORE,
        "implementation_status": REQUIRED_V2_STATUS,
        "seed_emitted": seed is not None,
        "seed_digest": seed_digest,
        "new_evaluator_epoch_required": True,
        "required_fresh_splits": ["discovery", "validation", "stress"],
        "complete_context_scope_cartesian_per_split_required": True,
        "prior_record_reuse_allowed": False,
        "cross_epoch_pooling_allowed": False,
        "adapter_candidate_synthesis_performed": False,
        "adapter_candidate_evaluation_performed": False,
        "adapter_scoring_performed": False,
        "language_last_resort_deferred": True,
        "language_expansion_executed_by_adapter": False,
    }
    lifecycle = {
        "prior_record_exclusion": _copy(normalized_input["prior_record_exclusion"]),
        "all_prior_records_eligible_for_future_scoring": False,
        "adapter_record": {
            "role": "STATE_HANDOFF_ONLY_NO_SCORING",
            "evaluator_epoch": None,
            "new_observation_count": 0,
            "eligible_for_future_scoring": False,
        },
        "future_scoring_policy": {
            "new_unconsumed_evidence_required": True,
            "required_new_evaluator_epoch": True,
            "reuse_any_prior_records_allowed": False,
            "cross_epoch_pooling_allowed": False,
        },
        "logical_selective_erasure_applied": True,
        "physical_erasure": "NOT_PERFORMED",
    }
    source_summary = _source_summary(adjudication, receipt)
    catalog = _state_catalog(
        source_state=source_state,
        source_state_digest=source_state_digest,
        restricted_state=restricted_state,
        restricted_state_digest=restricted_state_digest,
        adjudication_report=adjudication,
    )
    events = [
        _audit_event(
            0,
            "SOURCE_POST_RESTRICTION_ADJUDICATION_VERIFIED",
            GENESIS_DIGEST,
            {
                "verification_status": source_summary["verification_status"],
                "source_adjudication_report_digest": source_summary[
                    "report_digest"
                ],
                "source_disposition": source_disposition,
            },
        )
    ]
    events.append(
        _audit_event(
            1,
            "FINITE_INTERVAL_TWO_Q_INTERFACE_CERTIFIED_WITHOUT_PROJECTION",
            events[-1]["event_digest"],
            {
                "interface_certificate_digest": _digest(interface),
                "interface_verified": interface["verified"],
                "upstream_competition_v1_exact_replay_performed": True,
                "adapter_seed_submitted_to_competition_v1": False,
            },
        )
    )
    events.append(
        _audit_event(
            2,
            "CANONICAL_RECOMPETITION_SEED_ROUTE_RESOLVED",
            events[-1]["event_digest"],
            {
                "disposition": disposition,
                "seed_kind": seed_kind,
                "seed_digest": seed_digest,
                "rollback_execution_status": "NOT_PERFORMED",
            },
        )
    )
    events.append(
        _audit_event(
            3,
            "FUTURE_V2_HANDOFF_RECORDED_FAIL_CLOSED",
            events[-1]["event_digest"],
            {
                "required_core": REQUIRED_V2_CORE,
                "implementation_status": REQUIRED_V2_STATUS,
                "new_evidence_consumed_by_adapter": False,
                "new_evaluator_epoch_created_by_adapter": False,
                "language_expansion_executed_by_adapter": False,
                "adoption_status": ADOPTION_STATUS,
                "current_status": CURRENT_STATUS,
            },
        )
    )
    if canonical_json_bytes(source_state) != source_bytes:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "source probe-expanded state was mutated during adaptation"
        )
    if canonical_json_bytes(restricted_state) != restricted_bytes:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "restricted state was mutated during adaptation"
        )
    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "contract_digest": contract_digest,
        "adapter_input_digest": semantic_input_digest,
        "adapter_id": adapter_id,
        "source_adjudication": source_summary,
        "source_disposition": source_disposition,
        "source_cycle_route": adjudication.get("cycle_route"),
        "source_state_catalog": catalog,
        "seed_resolution": resolution,
        "recompetition_seed": seed,
        "recompetition_seed_digest": seed_digest,
        "interface_certificate": interface,
        "competition_v1_incompatibility": v1_incompatibility,
        "competition_v2_handoff": v2_handoff,
        "disposition": disposition,
        "record_lifecycle_extension": lifecycle,
        "authority_boundary": _copy(normalized_contract["authority_boundary"]),
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
    _exact_keys(report, REPORT_FIELDS, "adapter_report")
    _reject_observed_values(report, "adapter_report")
    return ShadowIntervalMultiQRecompetitionAdapterResult(report=report)


def verify_shadow_interval_multi_q_recompetition_adapter(
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
    probe_input: Mapping[str, Any],
    probe_contract: Mapping[str, Any],
    probe_report: Mapping[str, Any],
    restriction_input: Mapping[str, Any],
    restriction_contract: Mapping[str, Any],
    restriction_report: Mapping[str, Any],
    adjudication_input: Mapping[str, Any],
    adjudication_contract: Mapping[str, Any],
    adjudication_report: Mapping[str, Any],
    adapter_input: Mapping[str, Any],
    adapter_contract: Mapping[str, Any],
    adapter_report: Mapping[str, Any],
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
    expected_probe_input_digest: str,
    expected_probe_contract_digest: str,
    expected_probe_report_digest: str,
    expected_probe_input_artifacts: Mapping[str, Any] | None,
    expected_restriction_input_digest: str,
    expected_restriction_contract_digest: str,
    expected_restriction_report_digest: str,
    expected_restriction_input_artifacts: Mapping[str, Any] | None,
    expected_adjudication_input_digest: str,
    expected_adjudication_contract_digest: str,
    expected_adjudication_report_digest: str,
    expected_adjudication_input_artifacts: Mapping[str, Any] | None,
    expected_adapter_input_digest: str,
    expected_adapter_contract_digest: str,
    expected_adapter_report_digest: str,
    expected_adapter_input_artifacts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Exact-replay an adapter report against independent digest anchors."""

    expected_report_digest = _require_digest(
        expected_adapter_report_digest, "expected_adapter_report_digest"
    )
    supplied = _copy(_mapping(adapter_report, "adapter_report"))
    _exact_keys(supplied, REPORT_FIELDS, "adapter_report")
    _reject_observed_values(supplied, "adapter_report")
    if expected_adapter_input_artifacts is not None and not isinstance(
        expected_adapter_input_artifacts, Mapping
    ):
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "expected_adapter_input_artifacts must be an object or null"
        )
    expected_artifacts = (
        _copy(expected_adapter_input_artifacts)
        if expected_adapter_input_artifacts is not None
        else None
    )
    _reject_observed_values(
        expected_artifacts, "expected_adapter_input_artifacts"
    )
    if supplied.get("input_artifacts") != expected_artifacts:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "adapter input artifacts differ from independent expectation"
        )
    fresh = adapt_shadow_interval_multi_q_recompetition_seed(
        competition_input,
        competition_contract,
        competition_report,
        transition_contract,
        transition_report,
        qualification_input,
        qualification_contract,
        qualification_report,
        review_contract,
        review_report,
        probe_input,
        probe_contract,
        probe_report,
        restriction_input,
        restriction_contract,
        restriction_report,
        adjudication_input,
        adjudication_contract,
        adjudication_report,
        adapter_input,
        adapter_contract,
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
        expected_review_report_digest=expected_review_report_digest,
        expected_review_input_artifacts=expected_review_input_artifacts,
        expected_probe_input_digest=expected_probe_input_digest,
        expected_probe_contract_digest=expected_probe_contract_digest,
        expected_probe_report_digest=expected_probe_report_digest,
        expected_probe_input_artifacts=expected_probe_input_artifacts,
        expected_restriction_input_digest=expected_restriction_input_digest,
        expected_restriction_contract_digest=expected_restriction_contract_digest,
        expected_restriction_report_digest=expected_restriction_report_digest,
        expected_restriction_input_artifacts=expected_restriction_input_artifacts,
        expected_adjudication_input_digest=expected_adjudication_input_digest,
        expected_adjudication_contract_digest=expected_adjudication_contract_digest,
        expected_adjudication_report_digest=expected_adjudication_report_digest,
        expected_adjudication_input_artifacts=expected_adjudication_input_artifacts,
        expected_adapter_input_digest=expected_adapter_input_digest,
        expected_adapter_contract_digest=expected_adapter_contract_digest,
        input_artifacts=expected_artifacts,
    ).to_dict()
    if fresh["report_digest"] != expected_report_digest:
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "replayed adapter report digest differs from expectation"
        )
    if canonical_json_bytes(supplied) != canonical_json_bytes(fresh):
        raise ShadowIntervalMultiQRecompetitionAdapterValidationError(
            "supplied adapter report differs from exact replay"
        )
    seed = fresh["recompetition_seed"]
    return {
        "status": "VERIFIED_" + fresh["disposition"],
        "disposition": fresh["disposition"],
        "report_digest": fresh["report_digest"],
        "contract_digest": fresh["contract_digest"],
        "adapter_id": fresh["adapter_id"],
        "source_adjudication_report_digest": fresh["source_adjudication"][
            "report_digest"
        ],
        "seed_emitted": seed is not None,
        "seed_kind": None if seed is None else seed["seed_kind"],
        "recompetition_seed_digest": fresh["recompetition_seed_digest"],
        "competition_v1_compatible": False,
        "required_v2_core": fresh["competition_v2_handoff"]["required_core"],
        "adoption_eligibility": fresh["adoption_eligibility"],
        "adoption_status": fresh["adoption_status"],
        "promotion_status": fresh["promotion_status"],
        "current_status": fresh["current_status"],
    }
