"""Materialize one exact-verified interval/multi-Q V2 winner as a shadow child.

This additive layer replays the complete V2 competition through its public
verifier, copies only the selected candidate's finite interval geometry into a
detached child, and emits a digest-bound transition report.  It does not
qualify, adopt, promote, make current, roll back, execute probes, invent
language, call ``run_one``, or write state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping

from performance.shadow_interval_multi_q_theory_operation_competition import (
    ShadowIntervalMultiQTheoryOperationCompetitionValidationError,
    verify_shadow_interval_multi_q_theory_operation_competition,
)
from performance.theory_operation_competition import canonical_json_bytes


CONTRACT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-theory-transition-contract/1"
)
CONTRACT_ID = "shadow_interval_multi_q_theory_transition_v1"
SOURCE_COMPETITION_CONTRACT_ID = (
    "shadow_interval_multi_q_theory_operation_competition_v2"
)
SOURCE_COMPETITION_CONTRACT_DIGEST = (
    "sha256:4c30c0b1a2cdec92ab1676e98677b620907bb9652bff1ce71865fce9d45ccd1e"
)
SOURCE_COMPETITION_REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-theory-operation-competition-report/2"
)
SOURCE_CANDIDATE_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-theory-operation-candidate/2"
)
SOURCE_ADAPTER_CONTRACT_ID = "shadow_interval_multi_q_recompetition_adapter_v1"
SOURCE_ADAPTER_CONTRACT_DIGEST = (
    "sha256:16d2a30873e3f8b2e56fe5d7ac272140eb83dbcb441d8d80a892c4028f28f029"
)
SOURCE_ADAPTER_REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-recompetition-adapter-report/1"
)
SOURCE_SEED_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-recompetition-seed/1"
)
CHILD_THEORY_SCHEMA_VERSION = "sc-olh-kg.shadow-interval-multi-q-theory-state/1"
REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-theory-transition-report/1"
)
FROZEN_CONTRACT_DIGEST = (
    "sha256:b1a5f1761c2cafcae24f37f22810178074b9fc7800b6d73bdfd631be3b1df86d"
)
GENESIS_DIGEST = "sha256:" + "0" * 64

PROBE_IDS = [
    "absolute_error_point_prediction",
    "normalized_signed_interval_boundary_margin",
]
MAX_CONTEXT_COUNT = 64
MAX_SCOPE_COUNT = 16
MAX_REMOVABLE_FEATURE_COUNT = 8

SOURCE_SELECT_EXPANSION = "SELECT_SHADOW_INTERVAL_EXPANSION_CANDIDATE"
SOURCE_SELECT_RESTRICTION = "SELECT_SHADOW_UNIFORM_RESTRICTION_CANDIDATE"
SOURCE_SELECT_QUOTIENT = "SELECT_SHADOW_CONSERVATIVE_QUOTIENT_ENVELOPE_CANDIDATE"
SOURCE_NEEDS_EVIDENCE = "INTERVAL_MULTI_Q_COMPETITION_NEEDS_EXACT_FRESH_EVIDENCE"
SOURCE_INCOMPARABLE = "INTERVAL_MULTI_Q_COMPETITION_INCOMPARABLE_EVALUATOR_EPOCH"
SOURCE_EARLY = "INTERVAL_MULTI_Q_COMPETITION_EARLY_DIAGNOSTIC_UNRESOLVED"
SOURCE_NO_WINNER = "INTERVAL_MULTI_Q_COMPETITION_NO_VALIDATION_WINNER"
SOURCE_STRESS_FAILED = (
    "INTERVAL_MULTI_Q_COMPETITION_PROVISIONAL_WINNER_FAILED_STRESS_CONFIRMATION"
)
SOURCE_ADAPTER_EVIDENCE = (
    "INTERVAL_MULTI_Q_COMPETITION_BLOCKED_ADAPTER_NEEDS_POST_RESTRICTION_EVIDENCE"
)
SOURCE_ADAPTER_EPOCH = (
    "INTERVAL_MULTI_Q_COMPETITION_BLOCKED_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH"
)

MATERIALIZED_EXPANSION = "MATERIALIZED_SHADOW_INTERVAL_EXPANSION"
MATERIALIZED_RESTRICTION = "MATERIALIZED_SHADOW_UNIFORM_INTERVAL_RESTRICTION"
MATERIALIZED_QUOTIENT = "MATERIALIZED_SHADOW_CONSERVATIVE_QUOTIENT_ENVELOPE"
NOT_MATERIALIZED_EVIDENCE = "NOT_MATERIALIZED_NEEDS_EXACT_FRESH_EVIDENCE"
NOT_MATERIALIZED_INCOMPARABLE = "NOT_MATERIALIZED_INCOMPARABLE_EVALUATOR_EPOCH"
NOT_MATERIALIZED_EARLY = "NOT_MATERIALIZED_EARLY_DIAGNOSTIC_UNRESOLVED"
NOT_MATERIALIZED_NO_WINNER = "NOT_MATERIALIZED_NO_VALIDATION_WINNER"
NOT_MATERIALIZED_STRESS = (
    "NOT_MATERIALIZED_PROVISIONAL_WINNER_FAILED_STRESS_CONFIRMATION"
)
NOT_MATERIALIZED_ADAPTER_EVIDENCE = (
    "NOT_MATERIALIZED_ADAPTER_NEEDS_POST_RESTRICTION_EVIDENCE"
)
NOT_MATERIALIZED_ADAPTER_EPOCH = (
    "NOT_MATERIALIZED_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH"
)

SOURCE_TO_TRANSITION_DISPOSITION = {
    SOURCE_SELECT_EXPANSION: MATERIALIZED_EXPANSION,
    SOURCE_SELECT_RESTRICTION: MATERIALIZED_RESTRICTION,
    SOURCE_SELECT_QUOTIENT: MATERIALIZED_QUOTIENT,
    SOURCE_NEEDS_EVIDENCE: NOT_MATERIALIZED_EVIDENCE,
    SOURCE_INCOMPARABLE: NOT_MATERIALIZED_INCOMPARABLE,
    SOURCE_EARLY: NOT_MATERIALIZED_EARLY,
    SOURCE_NO_WINNER: NOT_MATERIALIZED_NO_WINNER,
    SOURCE_STRESS_FAILED: NOT_MATERIALIZED_STRESS,
    SOURCE_ADAPTER_EVIDENCE: NOT_MATERIALIZED_ADAPTER_EVIDENCE,
    SOURCE_ADAPTER_EPOCH: NOT_MATERIALIZED_ADAPTER_EPOCH,
}

DISPOSITION_REGISTRY = {
    "materialize_interval_expansion": MATERIALIZED_EXPANSION,
    "materialize_uniform_interval_restriction": MATERIALIZED_RESTRICTION,
    "materialize_conservative_quotient_envelope": MATERIALIZED_QUOTIENT,
    "not_materialized_needs_exact_fresh_evidence": NOT_MATERIALIZED_EVIDENCE,
    "not_materialized_incomparable_evaluator_epoch": NOT_MATERIALIZED_INCOMPARABLE,
    "not_materialized_early_diagnostic_unresolved": NOT_MATERIALIZED_EARLY,
    "not_materialized_no_validation_winner": NOT_MATERIALIZED_NO_WINNER,
    "not_materialized_provisional_winner_failed_stress_confirmation": (
        NOT_MATERIALIZED_STRESS
    ),
    "not_materialized_adapter_needs_post_restriction_evidence": (
        NOT_MATERIALIZED_ADAPTER_EVIDENCE
    ),
    "not_materialized_adapter_incomparable_post_restriction_epoch": (
        NOT_MATERIALIZED_ADAPTER_EPOCH
    ),
}

TRANSITION_REGISTRY = [
    {
        "source_disposition": SOURCE_SELECT_EXPANSION,
        "candidate_family": "interval_robustify",
        "operation_kind": "expand",
        "transition_kind": "INTERVAL_EXPANSION",
        "transition_disposition": MATERIALIZED_EXPANSION,
    },
    {
        "source_disposition": SOURCE_SELECT_RESTRICTION,
        "candidate_family": "interval_restrict",
        "operation_kind": "restrict",
        "transition_kind": "UNIFORM_INTERVAL_RESTRICTION",
        "transition_disposition": MATERIALIZED_RESTRICTION,
    },
    {
        "source_disposition": SOURCE_SELECT_QUOTIENT,
        "candidate_family": "interval_quotient",
        "operation_kind": "quotient",
        "transition_kind": "CONSERVATIVE_INTERVAL_QUOTIENT_ENVELOPE",
        "transition_disposition": MATERIALIZED_QUOTIENT,
    },
]
TRANSITION_BY_SOURCE = {item["source_disposition"]: item for item in TRANSITION_REGISTRY}

MATERIALIZATION_POLICY = {
    "source_exact_replay": "PUBLIC_V2_VERIFIER_ONLY",
    "selected_candidate_source": (
        "EXACT_SELECTED_CANDIDATE_BYTES_FROM_VERIFIED_V2_REPORT"
    ),
    "selected_candidate_array_membership_required": True,
    "selected_candidate_commitment_binding_required": True,
    "validation_selection_binding_required": True,
    "stress_confirmation_binding_required": True,
    "candidate_reranking_allowed": False,
    "fallback_candidate_allowed": False,
    "source_seed_mutation_allowed": False,
    "child_geometry_source": "EXACT_SELECTED_CANDIDATE_GEOMETRY",
    "candidate_evaluation_copied_into_child": False,
    "deterministic_child_id": True,
    "nonselection_materialization_surfaces": "ALL_NULL",
}

PRESERVATION_POLICY = {
    "finite_interval_table_only": True,
    "complete_object_space_and_radius_group_checks": True,
    "fixed_two_q_registry_byte_equal": True,
    "violation_functional_registry_byte_equal": True,
    "scope_registry_byte_equal": True,
    "expansion_centers_and_grouping_byte_equal": True,
    "expansion_all_radii_gte_source_and_one_strict": True,
    "restriction_centers_and_grouping_byte_equal": True,
    "restriction_all_radii_lte_source_and_one_strict": True,
    "quotient_projection_exact": True,
    "quotient_radius_grouping": "per_context",
    "quotient_hull_domain": (
        "all_parent_contexts_in_each_fiber_cross_all_registered_scopes"
    ),
    "quotient_containment_check": "reconstructed_stored_float_endpoints",
    "quotient_parent_snapshot_required": True,
    "quotient_point_prediction_preservation_claimed": False,
    "noncontained_or_nonrepresentable_materialization_rejected": True,
}

EVALUATOR_EPOCH_POLICY = {
    "inherit_fixed_anchor": True,
    "child_evaluator_epoch": None,
    "child_evaluator_status": (
        "UNASSIGNED_FRESH_POST_TRANSITION_EVALUATOR_REQUIRED"
    ),
    "operational_probe_status": "FIXED_TWO_Q_FRESH_QUALIFICATION_REQUIRED",
    "source_evidence_allowed_for_child_scoring": False,
    "forbid_old_new_pooling": True,
    "fresh_post_transition_evaluator_created": False,
    "fresh_post_transition_qualification_performed": False,
}

RECORD_LIFECYCLE_POLICY = {
    "five_prior_generations_role": "AUDIT_ONLY_FUTURE_SCORING_EXCLUDED",
    "v2_discovery_role": "AUDIT_ONLY_FUTURE_SCORING_EXCLUDED",
    "v2_validation_role": "AUDIT_ONLY_FUTURE_SCORING_EXCLUDED",
    "v2_stress_role": "AUDIT_ONLY_FUTURE_SCORING_EXCLUDED",
    "cross_epoch_pooling_allowed": False,
    "logical_selective_erasure_applied": True,
    "physical_erasure": "NOT_PERFORMED",
}

AUTHORITY_BOUNDARY = {
    "scope": "LOCAL_DETACHED_SHADOW_INTERVAL_MULTI_Q_TRANSITION_ONLY",
    "materialization_allowed_only_for_exact_verified_selected_and_stress_confirmed_candidate": True,
    "source_competition_reexecution_required": True,
    "candidate_reselection_allowed": False,
    "fresh_qualification_authority": False,
    "adoption_eligibility_authority": False,
    "adoption_authority": False,
    "promotion_authority": False,
    "current_pointer_authority": False,
    "rollback_execution_authority": False,
    "probe_execution_authority": False,
    "language_expansion_authority": False,
    "parent_or_ambient_write_authority": False,
}

SELECTION = {
    "materialized_status": (
        "DETACHED_SHADOW_CHILD_MATERIALIZED_FRESH_QUALIFICATION_REQUIRED"
    ),
    "no_materialization_status": "NO_SHADOW_CHILD_MATERIALIZED",
    "adoption_eligibility": "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED",
    "adoption_status": "NOT_ADOPTED_SHADOW_ONLY",
    "promotion_status": "NOT_PROMOTED",
    "current_status": "NOT_CURRENT",
}

MANDATORY_NONCLAIMS = (
    "shadow_only",
    "interval_multi_q_theory_transition_v1_only",
    "full_v2_competition_exact_replay_is_not_external_attestation",
    "selected_candidate_materialization_is_not_adoption",
    "no_h_t_to_h_t_plus_1_acceptance",
    "no_adoption_eligibility_determination",
    "no_adoption",
    "no_promotion",
    "no_current_pointer_write",
    "no_parent_source_or_ambient_state_write",
    "no_source_seed_mutation",
    "no_candidate_reselection_or_fallback",
    "no_validation_or_stress_reranking_or_reexecution",
    "no_source_evidence_reuse_for_child_scoring",
    "five_prior_generations_and_v2_competition_evidence_are_future_scoring_excluded",
    "no_cross_epoch_pooling",
    "fresh_post_transition_evaluator_epoch_not_created",
    "fresh_post_transition_qualification_not_performed",
    "fixed_two_q_registry_only",
    "q_v_registry_preservation_is_not_probe_value_equality",
    "no_new_probe_execution",
    "no_language_or_predicate_invention",
    "no_language_expansion_execution",
    "quotient_envelope_preserves_interval_containment_not_point_predictions",
    "quotient_recovery_requires_exact_verified_parent_snapshot",
    "no_rollback_execution",
    "no_physical_erasure",
    "no_external_data_evaluator_retention_or_adoption_attestation",
    "no_run_one",
    "no_benchmark_execution",
    "no_scheduler_or_network_access",
    "no_operations_research_baseline_or_claim_change",
    "no_paper_promotion",
    "explicit_cli_out_is_only_optional_write",
    "input_artifacts_require_independent_expected_values",
    "report_digest_is_not_a_signature",
    "no_scientific_validity_or_generalization_claim",
)

CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "source_competition_contract_id",
        "source_competition_contract_digest",
        "source_competition_report_schema_version",
        "source_candidate_schema_version",
        "source_adapter_contract_id",
        "source_adapter_contract_digest",
        "source_adapter_report_schema_version",
        "source_seed_schema_version",
        "child_theory_schema_version",
        "report_schema_version",
        "fixed_probe_registry",
        "disposition_registry",
        "transition_registry",
        "materialization_policy",
        "preservation_policy",
        "evaluator_epoch_policy",
        "record_lifecycle_policy",
        "authority_boundary",
        "selection",
        "nonclaims",
    }
)

CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "candidate_family",
        "operation_kind",
        "source_theory_state_digest",
        "object_space",
        "model_class",
        "model_class_digest",
        "semantic_model_digest",
        "scope_ids",
        "removable_feature_ids",
        "probe_ids",
        "violation_functionals",
        "construction",
        "certificate",
        "discovery_metrics",
        "discovery_admissible",
        "validation_evaluation",
    }
)

CHILD_FIELDS = frozenset(
    {
        "schema_version",
        "theory_id",
        "task_id",
        "evaluator_epoch",
        "evaluator_status",
        "fixed_anchor",
        "object_space",
        "model_class",
        "model_class_digest",
        "semantic_model_digest",
        "probe_ids",
        "violation_functionals",
        "scope_ids",
        "removable_feature_ids",
        "evidence_reuse_policy",
        "operational_probe_status",
        "transition_lineage",
    }
)

REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "contract_digest",
        "source_interval_competition",
        "parent_theory_state",
        "parent_theory_state_digest",
        "disposition",
        "operation_kind",
        "transition_kind",
        "selected_candidate_id",
        "selected_candidate_family",
        "selected_candidate_binding",
        "child_theory_state",
        "child_theory_state_digest",
        "materialization_certificate",
        "preservation_certificate",
        "rollback_boundary",
        "evaluator_gate",
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


class ShadowIntervalMultiQTheoryTransitionValidationError(ValueError):
    """Raised when the frozen transition or its exact replay fails closed."""


class ShadowIntervalMultiQTheoryTransitionDisposition(str, Enum):
    MATERIALIZED_SHADOW_INTERVAL_EXPANSION = MATERIALIZED_EXPANSION
    MATERIALIZED_SHADOW_UNIFORM_INTERVAL_RESTRICTION = MATERIALIZED_RESTRICTION
    MATERIALIZED_SHADOW_CONSERVATIVE_QUOTIENT_ENVELOPE = MATERIALIZED_QUOTIENT
    NOT_MATERIALIZED_NEEDS_EXACT_FRESH_EVIDENCE = NOT_MATERIALIZED_EVIDENCE
    NOT_MATERIALIZED_INCOMPARABLE_EVALUATOR_EPOCH = NOT_MATERIALIZED_INCOMPARABLE
    NOT_MATERIALIZED_EARLY_DIAGNOSTIC_UNRESOLVED = NOT_MATERIALIZED_EARLY
    NOT_MATERIALIZED_NO_VALIDATION_WINNER = NOT_MATERIALIZED_NO_WINNER
    NOT_MATERIALIZED_PROVISIONAL_WINNER_FAILED_STRESS_CONFIRMATION = (
        NOT_MATERIALIZED_STRESS
    )
    NOT_MATERIALIZED_ADAPTER_NEEDS_POST_RESTRICTION_EVIDENCE = (
        NOT_MATERIALIZED_ADAPTER_EVIDENCE
    )
    NOT_MATERIALIZED_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH = (
        NOT_MATERIALIZED_ADAPTER_EPOCH
    )


@dataclass(frozen=True)
class ShadowIntervalMultiQTheoryTransitionResult:
    report: Mapping[str, Any]

    @property
    def disposition(self) -> str:
        return str(self.report["disposition"])

    @property
    def report_digest(self) -> str:
        return str(self.report["report_digest"])

    def to_dict(self) -> dict[str, Any]:
        return _copy(self.report)


def _copy(value: Any) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("ascii"))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "value is not canonical finite JSON"
        ) from exc


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
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
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label} keys differ; missing={missing}, extra={extra}"
        )


def _string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label} must be a non-empty string"
        )
    return value


def _strings(value: Any, label: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label} must be a{' non-empty' if nonempty else ''} list"
        )
    result = [_string(item, f"{label} item") for item in value]
    if len(result) != len(set(result)):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label} must be unique"
        )
    return result


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label} must be a finite number"
        )
    return float(value)


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_digest(value: Any, label: str) -> str:
    digest = _string(value, label)
    if len(digest) != 71 or not digest.startswith("sha256:"):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label} must be sha256:<64hex>"
        )
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label} is not hexadecimal"
        ) from exc
    return digest


def _contains_key(value: Any, key_name: str) -> bool:
    if isinstance(value, Mapping):
        return key_name in value or any(
            _contains_key(item, key_name) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(item, key_name) for item in value)
    return False


def _equal(actual: Any, expected: Any, label: str) -> None:
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label} differs from the frozen transition V1 contract"
        )


def validate_shadow_interval_multi_q_theory_transition_contract(
    contract_value: Any,
) -> dict[str, Any]:
    """Validate and detach the frozen interval/multi-Q transition contract."""

    contract = _mapping(contract_value, "interval_transition_contract")
    _exact_keys(contract, CONTRACT_FIELDS, "interval_transition_contract")
    frozen_scalars = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "source_competition_contract_id": SOURCE_COMPETITION_CONTRACT_ID,
        "source_competition_contract_digest": SOURCE_COMPETITION_CONTRACT_DIGEST,
        "source_competition_report_schema_version": (
            SOURCE_COMPETITION_REPORT_SCHEMA_VERSION
        ),
        "source_candidate_schema_version": SOURCE_CANDIDATE_SCHEMA_VERSION,
        "source_adapter_contract_id": SOURCE_ADAPTER_CONTRACT_ID,
        "source_adapter_contract_digest": SOURCE_ADAPTER_CONTRACT_DIGEST,
        "source_adapter_report_schema_version": SOURCE_ADAPTER_REPORT_SCHEMA_VERSION,
        "source_seed_schema_version": SOURCE_SEED_SCHEMA_VERSION,
        "child_theory_schema_version": CHILD_THEORY_SCHEMA_VERSION,
        "report_schema_version": REPORT_SCHEMA_VERSION,
    }
    for key, expected in frozen_scalars.items():
        if contract.get(key) != expected:
            raise ShadowIntervalMultiQTheoryTransitionValidationError(
                f"{key} differs from the frozen transition V1 contract"
            )
    for key, expected in (
        ("fixed_probe_registry", PROBE_IDS),
        ("disposition_registry", DISPOSITION_REGISTRY),
        ("transition_registry", TRANSITION_REGISTRY),
        ("materialization_policy", MATERIALIZATION_POLICY),
        ("preservation_policy", PRESERVATION_POLICY),
        ("evaluator_epoch_policy", EVALUATOR_EPOCH_POLICY),
        ("record_lifecycle_policy", RECORD_LIFECYCLE_POLICY),
        ("authority_boundary", AUTHORITY_BOUNDARY),
        ("selection", SELECTION),
    ):
        _equal(contract.get(key), expected, key)
    if not isinstance(contract.get("nonclaims"), list) or tuple(
        contract["nonclaims"]
    ) != MANDATORY_NONCLAIMS:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "nonclaims differ from the exact mandatory V1 list"
        )
    normalized = _copy(contract)
    if _digest(normalized) != FROZEN_CONTRACT_DIGEST:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "interval transition contract canonical digest differs from frozen V1"
        )
    return normalized


def _geometry(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    object_space = _mapping(value.get("object_space"), f"{label}.object_space")
    _exact_keys(object_space, {"feature_ids", "contexts"}, f"{label}.object_space")
    features = _strings(object_space.get("feature_ids"), f"{label}.feature_ids")
    contexts_value = object_space.get("contexts")
    if not isinstance(contexts_value, list) or not contexts_value:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label}.contexts must be a non-empty list"
        )
    if len(contexts_value) > MAX_CONTEXT_COUNT:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label}.contexts exceeds the frozen maximum"
        )
    contexts: list[dict[str, Any]] = []
    context_keys: list[bytes] = []
    for index, raw in enumerate(contexts_value):
        context = _copy(_mapping(raw, f"{label}.contexts[{index}]"))
        _exact_keys(context, set(features), f"{label}.contexts[{index}]")
        key = canonical_json_bytes(context)
        contexts.append(context)
        context_keys.append(key)
    if len(context_keys) != len(set(context_keys)):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label}.contexts are duplicated"
        )
    scopes = _strings(value.get("scope_ids"), f"{label}.scope_ids")
    if len(scopes) > MAX_SCOPE_COUNT:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label}.scope_ids exceeds the frozen maximum"
        )
    removable = _strings(
        value.get("removable_feature_ids"),
        f"{label}.removable_feature_ids",
        nonempty=False,
    )
    if len(removable) > MAX_REMOVABLE_FEATURE_COUNT or not set(removable) <= set(
        features
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label}.removable_feature_ids is outside the frozen feature registry"
        )
    probes = _strings(value.get("probe_ids"), f"{label}.probe_ids")
    if probes != PROBE_IDS:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label} does not carry the exact fixed two-Q registry"
        )
    violation_functionals = value.get("violation_functionals")
    if not isinstance(violation_functionals, list) or not violation_functionals:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label}.violation_functionals must be a non-empty list"
        )
    model = _mapping(value.get("model_class"), f"{label}.model_class")
    _exact_keys(
        model,
        {"kind", "center_predictions", "radius_grouping", "radii"},
        f"{label}.model_class",
    )
    if model.get("kind") != "finite_interval_table":
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label}.model_class is not a finite interval table"
        )
    centers_value = model.get("center_predictions")
    if not isinstance(centers_value, list) or not centers_value:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label}.center_predictions must be non-empty"
        )
    centers: dict[bytes, float] = {}
    for index, raw in enumerate(centers_value):
        item = _mapping(raw, f"{label}.center_predictions[{index}]")
        _exact_keys(item, {"context", "value"}, f"{label}.center_predictions[{index}]")
        context = _mapping(item.get("context"), f"{label}.center context")
        _exact_keys(context, set(features), f"{label}.center context")
        key = canonical_json_bytes(context)
        if key in centers:
            raise ShadowIntervalMultiQTheoryTransitionValidationError(
                f"{label}.center_predictions has duplicate contexts"
            )
        centers[key] = _finite(item.get("value"), f"{label}.center value")
    if set(centers) != set(context_keys):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label}.center_predictions does not cover the exact object space"
        )
    grouping = model.get("radius_grouping")
    if grouping not in {"global", "per_scope", "per_context"}:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label}.radius_grouping is unsupported"
        )
    radii_value = model.get("radii")
    if not isinstance(radii_value, list) or not radii_value:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label}.radii must be non-empty"
        )
    radii: dict[bytes, tuple[dict[str, Any], float]] = {}
    for index, raw in enumerate(radii_value):
        item = _mapping(raw, f"{label}.radii[{index}]")
        _exact_keys(item, {"group", "radius"}, f"{label}.radii[{index}]")
        group = _copy(_mapping(item.get("group"), f"{label}.radius group"))
        radius = _finite(item.get("radius"), f"{label}.radius")
        if radius < 0.0:
            raise ShadowIntervalMultiQTheoryTransitionValidationError(
                f"{label}.radius must be nonnegative"
            )
        key = canonical_json_bytes(group)
        if key in radii:
            raise ShadowIntervalMultiQTheoryTransitionValidationError(
                f"{label}.radius groups are duplicated"
            )
        radii[key] = (group, radius)
    if grouping == "global":
        expected_groups = [{"global": "*"}]
    elif grouping == "per_scope":
        expected_groups = [{"scope_id": scope} for scope in scopes]
    else:
        expected_groups = [{"context": context} for context in contexts]
    if set(radii) != {canonical_json_bytes(item) for item in expected_groups}:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            f"{label}.radius groups do not exactly cover the registered grouping"
        )
    return {
        "object_space": _copy(object_space),
        "features": features,
        "contexts": contexts,
        "context_keys": set(context_keys),
        "scopes": scopes,
        "removable": removable,
        "probe_ids": probes,
        "violation_functionals": _copy(violation_functionals),
        "model": _copy(model),
        "model_digest": _digest(model),
        "centers": centers,
        "grouping": grouping,
        "radii": radii,
    }


def _radius_for_pair(
    geometry: Mapping[str, Any], scope_id: str, context: Mapping[str, Any]
) -> float:
    grouping = geometry["grouping"]
    if grouping == "global":
        group = {"global": "*"}
    elif grouping == "per_scope":
        group = {"scope_id": scope_id}
    else:
        group = {"context": context}
    try:
        return float(geometry["radii"][canonical_json_bytes(group)][1])
    except KeyError as exc:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "radius lookup is incomplete"
        ) from exc


def _project_context(
    context: Mapping[str, Any], removed: set[str]
) -> dict[str, Any]:
    return {key: _copy(value) for key, value in context.items() if key not in removed}


def _semantic_model_digest(candidate: Mapping[str, Any]) -> str:
    return _digest(
        {
            "object_space": candidate["object_space"],
            "model_class": candidate["model_class"],
            "scope_ids": candidate["scope_ids"],
            "removable_feature_ids": candidate["removable_feature_ids"],
            "probe_ids": candidate["probe_ids"],
            "violation_functionals": candidate["violation_functionals"],
        }
    )


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    body = {
        key: _copy(value)
        for key, value in candidate.items()
        if key
        not in {
            "candidate_id",
            "semantic_model_digest",
            "discovery_metrics",
            "discovery_admissible",
            "validation_evaluation",
        }
    }
    return str(candidate["candidate_family"]) + ":" + _digest(body)[7:]


def _selected_candidate(
    report: Mapping[str, Any], receipt: Mapping[str, Any], source_disposition: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = TRANSITION_BY_SOURCE[source_disposition]
    selected = _copy(_mapping(report.get("selected_candidate"), "selected_candidate"))
    _exact_keys(selected, CANDIDATE_FIELDS, "selected_candidate")
    if selected.get("schema_version") != SOURCE_CANDIDATE_SCHEMA_VERSION:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "selected candidate schema is not the frozen V2 schema"
        )
    for key in ("candidate_family", "operation_kind"):
        if selected.get(key) != registry[key]:
            raise ShadowIntervalMultiQTheoryTransitionValidationError(
                f"selected candidate {key} differs from the source disposition"
            )
    candidate_id = _string(selected.get("candidate_id"), "selected candidate_id")
    if candidate_id != _candidate_id(selected):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "selected candidate_id is not the canonical intrinsic ID"
        )
    if selected.get("model_class_digest") != _digest(selected.get("model_class")):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "selected model_class_digest is not canonical"
        )
    if selected.get("semantic_model_digest") != _semantic_model_digest(selected):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "selected semantic_model_digest is not canonical"
        )
    if selected.get("discovery_admissible") is not True:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "selected candidate was not discovery-admissible"
        )
    evaluation = _mapping(
        selected.get("validation_evaluation"), "selected validation_evaluation"
    )
    if (
        evaluation.get("all_gates_passed") is not True
        or evaluation.get("validation_score_units") != "dimensionless"
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "selected candidate did not pass frozen validation gates"
        )
    score = _finite(evaluation.get("validation_score"), "selected validation_score")
    array_name = {
        "interval_robustify": "interval_expansion_candidates",
        "interval_restrict": "uniform_restriction_candidates",
        "interval_quotient": "conservative_quotient_envelope_candidates",
    }[registry["candidate_family"]]
    family_array = report.get(array_name)
    if not isinstance(family_array, list):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "selected candidate family array is not a list"
        )
    exact_matches = [
        item
        for item in family_array
        if isinstance(item, Mapping)
        and item.get("candidate_id") == candidate_id
        and canonical_json_bytes(item) == canonical_json_bytes(selected)
    ]
    if len(exact_matches) != 1:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "selected candidate is not one exact unique family-array member"
        )
    commitments = _mapping(report.get("candidate_commitments"), "candidate_commitments")
    dedup = _mapping(
        report.get("candidate_semantic_deduplication"),
        "candidate_semantic_deduplication",
    )
    retained_ids = dedup.get("retained_candidate_ids")
    if not isinstance(retained_ids, list) or retained_ids.count(candidate_id) != 1:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "selected candidate is not uniquely retained by semantic deduplication"
        )
    if (
        commitments.get("source_theory_state_digest")
        != selected["source_theory_state_digest"]
        or commitments.get("interval_competition_contract_digest")
        != SOURCE_COMPETITION_CONTRACT_DIGEST
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "selected candidate differs from the committed source domain"
        )
    selection = _mapping(report.get("validation_selection"), "validation_selection")
    if (
        selection.get("status") != "UNIQUE_PROVISIONAL_WINNER"
        or selection.get("provisional_candidate_id") != candidate_id
        or selection.get("provisional_candidate_family") != registry["candidate_family"]
        or selection.get("provisional_validation_score") != score
        or selection.get("validation_score_units") != "dimensionless"
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "selected candidate differs from the unique validation winner"
        )
    stress = _mapping(report.get("stress_confirmation"), "stress_confirmation")
    if (
        stress.get("status") != "PROVISIONAL_WINNER_STRESS_CONFIRMED"
        or stress.get("provisional_candidate_id") != candidate_id
        or stress.get("provisional_candidate_family") != registry["candidate_family"]
        or stress.get("all_gates_passed") is not True
        or stress.get("fallback_candidate_evaluated") is not False
        or stress.get("fallback_candidate_selected") is not False
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "selected candidate lacks exact single-winner stress confirmation"
        )
    boundary = _mapping(report.get("selection_boundary"), "selection_boundary")
    if (
        boundary.get("selection_status")
        != "SELECTED_SHADOW_PROPOSAL_NOT_MATERIALIZED"
        or boundary.get("selected_candidate_id") != candidate_id
        or boundary.get("selected_candidate_family") != registry["candidate_family"]
        or boundary.get("candidate_materialized") is not False
        or boundary.get("shadow_theory_state_created") is not False
        or boundary.get("transition_authorized") is not False
        or boundary.get("source_seed_mutated") is not False
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "source selection boundary is not the exact unmaterialized proposal"
        )
    if (
        receipt.get("candidate_selected") is not True
        or receipt.get("selected_candidate_id") != candidate_id
        or receipt.get("selected_operation_kind") != registry["operation_kind"]
        or receipt.get("candidate_materialized") is not False
        or receipt.get("fallback_candidate_evaluated") is not False
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "public V2 verifier receipt differs from the selected candidate"
        )
    binding = {
        "candidate_id": candidate_id,
        "candidate_family": registry["candidate_family"],
        "operation_kind": registry["operation_kind"],
        "source_theory_state_digest": selected["source_theory_state_digest"],
        "model_class_digest": selected["model_class_digest"],
        "semantic_model_digest": selected["semantic_model_digest"],
        "candidate_commitment_digest": commitments["candidate_commitment_digest"],
        "validation_selection_status": selection["status"],
        "validation_score": score,
        "validation_score_units": "dimensionless",
        "stress_confirmation_status": stress["status"],
        "stress_all_gates_passed": True,
        "selected_candidate_exact_array_member": True,
    }
    return selected, binding


def _verified_parent(
    adapter_report_value: Mapping[str, Any],
    source_report: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    adapter_report = _mapping(adapter_report_value, "adapter_report")
    if (
        adapter_report.get("schema_version") != SOURCE_ADAPTER_REPORT_SCHEMA_VERSION
        or adapter_report.get("contract_id") != SOURCE_ADAPTER_CONTRACT_ID
        or adapter_report.get("contract_digest") != SOURCE_ADAPTER_CONTRACT_DIGEST
        or adapter_report.get("report_digest")
        != source_report["source_adapter"]["adapter_report_digest"]
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "adapter report differs from the exact V2 source binding"
        )
    seed = _copy(_mapping(adapter_report.get("recompetition_seed"), "recompetition_seed"))
    if seed.get("schema_version") != SOURCE_SEED_SCHEMA_VERSION:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "recompetition seed schema differs from the frozen source"
        )
    seed_digest = _digest(seed)
    summary = _mapping(source_report.get("source_seed_summary"), "source_seed_summary")
    commitments = _mapping(source_report.get("candidate_commitments"), "candidate_commitments")
    if (
        seed_digest != adapter_report.get("recompetition_seed_digest")
        or seed_digest != summary.get("recompetition_seed_digest")
        or seed_digest != commitments.get("recompetition_seed_digest")
        or seed.get("seed_id") != summary.get("recompetition_seed_id")
        or seed.get("seed_id") != commitments.get("recompetition_seed_id")
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "recompetition seed identity differs across the verified source chain"
        )
    parent = _copy(_mapping(seed.get("theory_state"), "seed theory_state"))
    parent_digest = _digest(parent)
    if (
        parent_digest != seed.get("theory_state_digest")
        or parent_digest != summary.get("seed_theory_state_digest")
        or parent_digest != candidate.get("source_theory_state_digest")
        or parent_digest != commitments.get("source_theory_state_digest")
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "parent theory state digest differs across the selected source"
        )
    for key in ("theory_id", "task_id", "fixed_anchor"):
        _string(parent.get(key), f"parent {key}")
    parent_geometry = _geometry(parent, "parent_theory_state")
    return parent, parent_digest, parent_geometry


def _common_preservation(
    parent_geometry: Mapping[str, Any], child_geometry: Mapping[str, Any]
) -> dict[str, Any]:
    q_equal = canonical_json_bytes(parent_geometry["probe_ids"]) == canonical_json_bytes(
        child_geometry["probe_ids"]
    )
    v_equal = canonical_json_bytes(
        parent_geometry["violation_functionals"]
    ) == canonical_json_bytes(child_geometry["violation_functionals"])
    scope_equal = canonical_json_bytes(parent_geometry["scopes"]) == canonical_json_bytes(
        child_geometry["scopes"]
    )
    if not (q_equal and v_equal and scope_equal):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "selected child does not preserve the fixed Q/V/scope registries"
        )
    return {
        "q_registry_byte_equal": q_equal,
        "v_registry_byte_equal": v_equal,
        "scope_ids_byte_equal": scope_equal,
        "finite_interval_table_verified": True,
    }


def _preserve_expansion_or_restriction(
    parent: Mapping[str, Any],
    parent_geometry: Mapping[str, Any],
    candidate: Mapping[str, Any],
    child_geometry: Mapping[str, Any],
    *,
    restriction: bool,
) -> dict[str, Any]:
    for key in ("object_space", "scope_ids", "removable_feature_ids", "probe_ids", "violation_functionals"):
        if canonical_json_bytes(parent.get(key)) != canonical_json_bytes(candidate.get(key)):
            raise ShadowIntervalMultiQTheoryTransitionValidationError(
                f"selected {'restriction' if restriction else 'expansion'} changed {key}"
            )
    if (
        canonical_json_bytes(parent_geometry["model"]["center_predictions"])
        != canonical_json_bytes(child_geometry["model"]["center_predictions"])
        or parent_geometry["grouping"] != child_geometry["grouping"]
        or [parent_geometry["radii"][key][0] for key in sorted(parent_geometry["radii"])]
        != [child_geometry["radii"][key][0] for key in sorted(child_geometry["radii"])]
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "selected interval change did not preserve centers and radius grouping"
        )
    checked = 0
    strict = False
    relation = True
    for context in parent_geometry["contexts"]:
        for scope in parent_geometry["scopes"]:
            source_radius = _radius_for_pair(parent_geometry, scope, context)
            child_radius = _radius_for_pair(child_geometry, scope, context)
            if restriction:
                relation = relation and child_radius <= source_radius
                strict = strict or child_radius < source_radius
            else:
                relation = relation and child_radius >= source_radius
                strict = strict or child_radius > source_radius
            checked += 1
    construction = _mapping(candidate.get("construction"), "candidate construction")
    certificate = _mapping(candidate.get("certificate"), "candidate certificate")
    if restriction:
        frozen_ok = (
            construction.get("construction_kind") == "UNIFORM_RADIUS_CONTRACTION"
            and construction.get("synthesis_evidence_digest") is None
            and certificate.get("strict_subset_verified") is True
            and certificate.get("at_least_one_radius_strictly_reduced") is True
            and certificate.get("all_restricted_radii_lte_source") is True
        )
        family = {
            "source_radius_grouping_preserved": True,
            "centers_byte_equal": True,
            "all_child_intervals_within_parent_intervals": relation,
            "at_least_one_strictly_restricted": strict,
            "checked_parent_context_scope_pair_count": checked,
        }
    else:
        frozen_ok = (
            construction.get("construction_kind")
            == "DISCOVERY_DERIVED_INTERVAL_EXPANSION"
            and construction.get("synthesis_evidence_digest")
            == construction.get("discovery_evidence_digest")
            and certificate.get("strict_superset_verified") is True
            and certificate.get("at_least_one_radius_strictly_expanded") is True
            and certificate.get("all_expanded_radii_gte_source") is True
        )
        family = {
            "source_radius_grouping_preserved": True,
            "centers_byte_equal": True,
            "all_child_intervals_contain_parent_intervals": relation,
            "at_least_one_strictly_expanded": strict,
            "checked_parent_context_scope_pair_count": checked,
        }
    if not relation or not strict or not frozen_ok:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "selected interval change fails its represented-float preservation gate"
        )
    return family


def _preserve_quotient(
    parent_geometry: Mapping[str, Any],
    parent_digest: str,
    candidate: Mapping[str, Any],
    child_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    construction = _mapping(candidate.get("construction"), "quotient construction")
    certificate = _mapping(candidate.get("certificate"), "quotient certificate")
    removed_list = _strings(
        construction.get("removed_feature_ids"), "quotient removed_feature_ids"
    )
    removed = set(removed_list)
    if not removed <= set(parent_geometry["removable"]) or not removed <= set(
        parent_geometry["features"]
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "quotient removed features are outside the verified parent registry"
        )
    expected_features = [
        item for item in parent_geometry["features"] if item not in removed
    ]
    if child_geometry["features"] != expected_features:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "quotient feature projection differs from the exact parent projection"
        )
    fibers: dict[bytes, dict[str, Any]] = {}
    expected_map: list[dict[str, Any]] = []
    for context in parent_geometry["contexts"]:
        quotient_context = _project_context(context, removed)
        key = canonical_json_bytes(quotient_context)
        fibers.setdefault(key, {"context": quotient_context, "parents": []})[
            "parents"
        ].append(context)
        expected_map.append(
            {"parent_context": _copy(context), "quotient_context": quotient_context}
        )
    expected_contexts = [fibers[key]["context"] for key in sorted(fibers)]
    if (
        child_geometry["contexts"] != expected_contexts
        or child_geometry["grouping"] != "per_context"
        or len(expected_contexts) >= len(parent_geometry["contexts"])
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "quotient object space is not a strict exact projection"
        )
    if construction.get("quotient_map") != expected_map:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "quotient map differs from the exact parent projection"
        )
    expected_envelopes: list[dict[str, Any]] = []
    checked = 0
    for key in sorted(fibers):
        quotient_context = fibers[key]["context"]
        parent_intervals: list[dict[str, Any]] = []
        lowers: list[float] = []
        uppers: list[float] = []
        for context in sorted(fibers[key]["parents"], key=canonical_json_bytes):
            parent_center = parent_geometry["centers"][canonical_json_bytes(context)]
            for scope in sorted(parent_geometry["scopes"]):
                parent_radius = _radius_for_pair(parent_geometry, scope, context)
                parent_lower = parent_center - parent_radius
                parent_upper = parent_center + parent_radius
                if not math.isfinite(parent_lower) or not math.isfinite(parent_upper):
                    raise ShadowIntervalMultiQTheoryTransitionValidationError(
                        "quotient parent endpoint is not representable"
                    )
                lowers.append(parent_lower)
                uppers.append(parent_upper)
                parent_intervals.append(
                    {
                        "scope_id": scope,
                        "parent_context": _copy(context),
                        "parent_center": parent_center,
                        "parent_radius": parent_radius,
                        "parent_lower": parent_lower,
                        "parent_upper": parent_upper,
                    }
                )
                checked += 1
        lower = min(lowers)
        upper = max(uppers)
        midpoint = lower / 2.0 + upper / 2.0
        radius = max(upper - midpoint, midpoint - lower)
        if not math.isfinite(midpoint) or not math.isfinite(radius):
            raise ShadowIntervalMultiQTheoryTransitionValidationError(
                "quotient hull is not representable"
            )
        child_center = child_geometry["centers"][key]
        child_radius = _radius_for_pair(
            child_geometry, parent_geometry["scopes"][0], quotient_context
        )
        if child_center != midpoint or child_radius != radius:
            raise ShadowIntervalMultiQTheoryTransitionValidationError(
                "quotient child differs from the exact stored hull"
            )
        child_lower = child_center - child_radius
        child_upper = child_center + child_radius
        if not math.isfinite(child_lower) or not math.isfinite(child_upper):
            raise ShadowIntervalMultiQTheoryTransitionValidationError(
                "quotient child endpoint is not representable"
            )
        for interval in parent_intervals:
            if (
                child_lower > interval["parent_lower"]
                or child_upper < interval["parent_upper"]
            ):
                raise ShadowIntervalMultiQTheoryTransitionValidationError(
                    "stored quotient endpoints do not contain every parent interval"
                )
        expected_envelopes.append(
            {
                "quotient_context": _copy(quotient_context),
                "hull_lower": lower,
                "hull_upper": upper,
                "hull_midpoint": midpoint,
                "hull_radius": radius,
                "parent_intervals": parent_intervals,
            }
        )
    if (
        construction.get("construction_kind") != "CONSERVATIVE_QUOTIENT_ENVELOPE"
        or construction.get("synthesis_evidence_digest") is not None
        or construction.get("quotient_radius_grouping") != "per_context"
        or construction.get("fiber_envelope_table") != expected_envelopes
        or construction.get("source_theory_state_digest") != parent_digest
        or construction.get("source_restore_method")
        != "RESTORE_EXACT_VERIFIED_RECOMPETITION_SEED_THEORY_STATE"
        or certificate.get("fiber_envelope_table") != expected_envelopes
        or certificate.get("source_theory_state_digest") != parent_digest
        or certificate.get("source_restore_method")
        != "RESTORE_EXACT_VERIFIED_RECOMPETITION_SEED_THEORY_STATE"
        or certificate.get("all_parent_intervals_contained_under_quotient_map")
        is not True
        or certificate.get("envelope_certificate_verified") is not True
        or certificate.get("point_prediction_preservation_claimed") is not False
        or certificate.get("quotient_alone_recovers_parent") is not False
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "quotient construction/certificate differs from the frozen envelope"
        )
    return {
        "removed_feature_ids": removed_list,
        "parent_context_count": len(parent_geometry["contexts"]),
        "child_context_count": len(child_geometry["contexts"]),
        "context_reduction_fraction": (
            1.0 - len(child_geometry["contexts"]) / len(parent_geometry["contexts"])
        ),
        "quotient_radius_grouping": "per_context",
        "quotient_context_keys_exact": True,
        "checked_parent_context_scope_pair_count": checked,
        "all_parent_intervals_contained_under_quotient_map": True,
        "source_restore_method": (
            "RESTORE_EXACT_VERIFIED_RECOMPETITION_SEED_THEORY_STATE"
        ),
        "parent_snapshot_digest": parent_digest,
        "point_prediction_preservation_claimed": False,
    }


def _materialize_child(
    parent: Mapping[str, Any],
    parent_digest: str,
    parent_geometry: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_binding: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    source_report: Mapping[str, Any],
    transition_contract_digest: str,
    registry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    child_geometry = _geometry(candidate, "selected_candidate")
    common = _common_preservation(parent_geometry, child_geometry)
    if registry["candidate_family"] == "interval_robustify":
        family_certificate = _preserve_expansion_or_restriction(
            parent, parent_geometry, candidate, child_geometry, restriction=False
        )
    elif registry["candidate_family"] == "interval_restrict":
        family_certificate = _preserve_expansion_or_restriction(
            parent, parent_geometry, candidate, child_geometry, restriction=True
        )
    else:
        if canonical_json_bytes(parent_geometry["probe_ids"]) != canonical_json_bytes(
            candidate["probe_ids"]
        ) or canonical_json_bytes(
            parent_geometry["violation_functionals"]
        ) != canonical_json_bytes(candidate["violation_functionals"]):
            raise ShadowIntervalMultiQTheoryTransitionValidationError(
                "quotient changed the fixed Q/V registries"
            )
        family_certificate = _preserve_quotient(
            parent_geometry, parent_digest, candidate, child_geometry
        )
    preservation = {
        "certificate_kind": "FINITE_INTERVAL_MULTI_Q_TRANSITION_PRESERVATION",
        "transition_kind": registry["transition_kind"],
        "parent_object_space_digest": _digest(parent_geometry["object_space"]),
        "child_object_space_digest": _digest(candidate["object_space"]),
        "parent_model_class_digest": parent_geometry["model_digest"],
        "child_model_class_digest": candidate["model_class_digest"],
        **common,
        "family_certificate": family_certificate,
        "verified": True,
    }
    evidence_reuse_policy = {
        "five_prior_generation_exclusion": _copy(
            source_report["record_lifecycle_extension"]["prior_record_exclusion"]
        ),
        "v2_competition_evidence_digests": _copy(source_report["evidence_digests"]),
        "prior_and_v2_records_role": "AUDIT_ONLY_FUTURE_SCORING_EXCLUDED",
        "source_evidence_allowed_for_child_scoring": False,
        "old_new_records_pooled": False,
        "fresh_post_transition_evidence_required": True,
    }
    lineage = {
        "parent_theory_id": parent["theory_id"],
        "parent_theory_state_digest": parent_digest,
        "source_adapter_contract_digest": SOURCE_ADAPTER_CONTRACT_DIGEST,
        "source_adapter_report_digest": source_binding[
            "source_adapter_report_digest"
        ],
        "recompetition_seed_id": source_binding["recompetition_seed_id"],
        "recompetition_seed_digest": source_binding["recompetition_seed_digest"],
        "source_interval_competition_input_digest": source_binding[
            "competition_input_digest"
        ],
        "source_interval_competition_contract_digest": source_binding[
            "contract_digest"
        ],
        "source_interval_competition_report_digest": source_binding["report_digest"],
        "candidate_commitment_digest": candidate_binding[
            "candidate_commitment_digest"
        ],
        "selected_candidate_id": candidate["candidate_id"],
        "selected_candidate_family": candidate["candidate_family"],
        "selected_candidate_model_class_digest": candidate["model_class_digest"],
        "selected_candidate_semantic_model_digest": candidate[
            "semantic_model_digest"
        ],
        "operation_kind": registry["operation_kind"],
        "transition_kind": registry["transition_kind"],
        "transition_contract_digest": transition_contract_digest,
    }
    payload = {
        "schema_version": CHILD_THEORY_SCHEMA_VERSION,
        "task_id": parent["task_id"],
        "evaluator_epoch": None,
        "evaluator_status": EVALUATOR_EPOCH_POLICY["child_evaluator_status"],
        "fixed_anchor": parent["fixed_anchor"],
        "object_space": _copy(candidate["object_space"]),
        "model_class": _copy(candidate["model_class"]),
        "model_class_digest": candidate["model_class_digest"],
        "semantic_model_digest": candidate["semantic_model_digest"],
        "probe_ids": _copy(candidate["probe_ids"]),
        "violation_functionals": _copy(candidate["violation_functionals"]),
        "scope_ids": _copy(candidate["scope_ids"]),
        "removable_feature_ids": _copy(candidate["removable_feature_ids"]),
        "evidence_reuse_policy": evidence_reuse_policy,
        "operational_probe_status": EVALUATOR_EPOCH_POLICY[
            "operational_probe_status"
        ],
        "transition_lineage": lineage,
    }
    theory_id = "shadow-interval-multi-q-theory:" + _digest(payload)[7:]
    child = {
        "schema_version": payload.pop("schema_version"),
        "theory_id": theory_id,
        **payload,
    }
    _exact_keys(child, CHILD_FIELDS, "child_theory_state")
    child_digest = _digest(child)
    if (
        canonical_json_bytes(child["object_space"])
        != canonical_json_bytes(candidate["object_space"])
        or canonical_json_bytes(child["model_class"])
        != canonical_json_bytes(candidate["model_class"])
        or canonical_json_bytes(child["probe_ids"])
        != canonical_json_bytes(candidate["probe_ids"])
        or canonical_json_bytes(child["violation_functionals"])
        != canonical_json_bytes(candidate["violation_functionals"])
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "child geometry differs from the selected candidate"
        )
    materialization = {
        "certificate_kind": "EXACT_VERIFIED_V2_WINNER_MATERIALIZATION",
        "public_v2_verifier_status": source_binding["verification_status"],
        "source_disposition_matches_candidate_family": True,
        "selected_candidate_exact_array_member": True,
        "candidate_id_recomputed": True,
        "model_class_digest_recomputed": True,
        "semantic_model_digest_recomputed": True,
        "candidate_commitment_contains_selected_id": True,
        "validation_selection_binding_verified": True,
        "validation_all_gates_passed": True,
        "stress_confirmation_binding_verified": True,
        "stress_all_gates_passed": True,
        "child_geometry_byte_equal_selected_candidate": True,
        "source_seed_mutated": False,
        "verified": True,
    }
    rollback = {
        "status": "PARENT_SNAPSHOT_BOUND_NOT_EXECUTED",
        "restore_method": "RESTORE_EXACT_VERIFIED_RECOMPETITION_SEED_THEORY_STATE",
        "parent_theory_state_digest": parent_digest,
        "rollback_executed": False,
    }
    evaluator_gate = {
        "source_evaluator_epoch": parent.get("evaluator_epoch"),
        "fixed_anchor": parent["fixed_anchor"],
        "child_evaluator_epoch": None,
        "child_evaluator_status": EVALUATOR_EPOCH_POLICY["child_evaluator_status"],
        "source_evidence_allowed_for_child_scoring": False,
        "old_new_records_pooled": False,
        "fresh_post_transition_qualification_required": True,
        "fresh_post_transition_qualification_performed": False,
        "adoption_blocked": True,
    }
    return child, materialization, preservation, rollback, evaluator_gate


def _source_binding(
    report_value: Mapping[str, Any], receipt_value: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _copy(_mapping(report_value, "interval_competition_report"))
    receipt = _mapping(receipt_value, "interval competition verifier receipt")
    if (
        report.get("schema_version") != SOURCE_COMPETITION_REPORT_SCHEMA_VERSION
        or report.get("contract_id") != SOURCE_COMPETITION_CONTRACT_ID
        or report.get("contract_digest") != SOURCE_COMPETITION_CONTRACT_DIGEST
        or receipt.get("report_digest") != report.get("report_digest")
        or receipt.get("contract_digest") != SOURCE_COMPETITION_CONTRACT_DIGEST
        or receipt.get("disposition") != report.get("disposition")
        or receipt.get("status") != "VERIFIED_" + str(report.get("disposition"))
        or receipt.get("adoption_status") != "NOT_ADOPTED_SHADOW_ONLY"
        or receipt.get("promotion_status") != "NOT_PROMOTED"
        or receipt.get("current_status") != "NOT_CURRENT"
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "public V2 verifier receipt differs from the supplied source report"
        )
    source_disposition = str(report.get("disposition"))
    if source_disposition not in SOURCE_TO_TRANSITION_DISPOSITION:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "source V2 disposition is outside the frozen ten-route registry"
        )
    summary = _mapping(report.get("source_seed_summary"), "source_seed_summary")
    adapter = _mapping(report.get("source_adapter"), "source_adapter")
    boundary = _mapping(report.get("selection_boundary"), "selection_boundary")
    stress = _mapping(report.get("stress_confirmation"), "stress_confirmation")
    commitments_value = report.get("candidate_commitments")
    commitments = (
        None
        if commitments_value is None
        else _mapping(commitments_value, "candidate_commitments")
    )
    selected = report.get("selected_candidate")
    expected_selected = source_disposition in TRANSITION_BY_SOURCE
    if (selected is not None) != expected_selected or bool(
        receipt.get("candidate_selected")
    ) != expected_selected:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "source selected-candidate nullability differs from its disposition"
        )
    if expected_selected:
        selection_status = "SELECTED_SHADOW_PROPOSAL_NOT_MATERIALIZED"
    else:
        selection_status = "NO_SELECTED_SHADOW_PROPOSAL"
    if boundary.get("selection_status") != selection_status:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "source selection status differs from its disposition"
        )
    binding = {
        "verification_status": receipt["status"],
        "contract_id": report["contract_id"],
        "contract_digest": report["contract_digest"],
        "report_schema_version": report["schema_version"],
        "report_digest": report["report_digest"],
        "competition_input_digest": report["competition_input_digest"],
        "competition_id": report["competition_id"],
        "disposition": source_disposition,
        "source_adapter_report_digest": adapter["adapter_report_digest"],
        "recompetition_seed_id": summary.get("recompetition_seed_id"),
        "recompetition_seed_digest": summary.get("recompetition_seed_digest"),
        "seed_theory_state_digest": summary.get("seed_theory_state_digest"),
        "candidate_commitment_digest": (
            None if commitments is None else commitments.get("candidate_commitment_digest")
        ),
        "selection_status": boundary["selection_status"],
        "selected_candidate_id": boundary.get("selected_candidate_id"),
        "selected_candidate_family": boundary.get("selected_candidate_family"),
        "stress_confirmation_status": stress.get("status"),
        "candidate_materialized_by_source": False,
        "adoption_status": report["adoption_status"],
        "promotion_status": report["promotion_status"],
        "current_status": report["current_status"],
    }
    return report, binding


def _record_lifecycle(source_report: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = _mapping(
        source_report.get("record_lifecycle_extension"),
        "source record_lifecycle_extension",
    )
    evidence_digests = _mapping(
        source_report.get("evidence_digests"), "source evidence_digests"
    )
    return {
        "five_prior_generation_exclusion": _copy(
            lifecycle.get("prior_record_exclusion")
        ),
        "v2_competition_evidence_digests": _copy(evidence_digests),
        "five_prior_generations_role": RECORD_LIFECYCLE_POLICY[
            "five_prior_generations_role"
        ],
        "v2_discovery_role": RECORD_LIFECYCLE_POLICY["v2_discovery_role"],
        "v2_validation_role": RECORD_LIFECYCLE_POLICY["v2_validation_role"],
        "v2_stress_role": RECORD_LIFECYCLE_POLICY["v2_stress_role"],
        "eligible_for_child_scoring": False,
        "cross_epoch_pooling_allowed": False,
        "logical_selective_erasure_applied": True,
        "physical_erasure": "NOT_PERFORMED",
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


def _finalize_report(body: Mapping[str, Any]) -> dict[str, Any]:
    report = {**_copy(body), "report_digest": _digest(body)}
    _exact_keys(report, REPORT_FIELDS, "interval_transition_report")
    return report


def materialize_shadow_interval_multi_q_theory_transition(
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
    interval_competition_input: Mapping[str, Any],
    interval_competition_contract: Mapping[str, Any],
    interval_competition_report: Mapping[str, Any],
    interval_transition_contract: Mapping[str, Any],
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
    expected_interval_competition_input_digest: str,
    expected_interval_competition_contract_digest: str,
    expected_interval_competition_report_digest: str,
    expected_interval_competition_input_artifacts: Mapping[str, Any] | None,
    expected_interval_transition_contract_digest: str,
    input_artifacts: Mapping[str, Any] | None = None,
) -> ShadowIntervalMultiQTheoryTransitionResult:
    """Exact-replay V2 and materialize only a detached selected shadow child."""

    normalized_contract = validate_shadow_interval_multi_q_theory_transition_contract(
        interval_transition_contract
    )
    contract_digest = _digest(normalized_contract)
    if contract_digest != _require_digest(
        expected_interval_transition_contract_digest,
        "expected_interval_transition_contract_digest",
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "interval transition contract differs from independent expectation"
        )
    if input_artifacts is not None and not isinstance(input_artifacts, Mapping):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "input_artifacts must be an object or null"
        )
    artifacts = None if input_artifacts is None else _copy(input_artifacts)
    if _contains_key(artifacts, "observed_value"):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "input_artifacts must not embed observed evidence values"
        )

    source_objects = (
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
        adapter_report,
        interval_competition_input,
        interval_competition_contract,
        interval_competition_report,
        interval_transition_contract,
    )
    source_snapshots = tuple(canonical_json_bytes(item) for item in source_objects)

    # This is the sole source-chain authorization path.  No V2 private helper
    # is called and no supplied report is accepted without full replay.
    try:
        receipt = verify_shadow_interval_multi_q_theory_operation_competition(
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
            adapter_report,
            interval_competition_input,
            interval_competition_contract,
            interval_competition_report,
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
            expected_adapter_report_digest=expected_adapter_report_digest,
            expected_adapter_input_artifacts=expected_adapter_input_artifacts,
            expected_interval_competition_input_digest=(
                expected_interval_competition_input_digest
            ),
            expected_interval_competition_contract_digest=(
                expected_interval_competition_contract_digest
            ),
            expected_interval_competition_report_digest=(
                expected_interval_competition_report_digest
            ),
            expected_interval_competition_input_artifacts=(
                expected_interval_competition_input_artifacts
            ),
        )
    except ShadowIntervalMultiQTheoryOperationCompetitionValidationError as exc:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "source V2 public exact replay failed"
        ) from exc

    if tuple(canonical_json_bytes(item) for item in source_objects) != source_snapshots:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "source inputs were mutated during public V2 replay"
        )

    source_report, source_binding = _source_binding(
        interval_competition_report, receipt
    )
    source_disposition = source_binding["disposition"]
    transition_disposition = SOURCE_TO_TRANSITION_DISPOSITION[source_disposition]

    parent: dict[str, Any] | None = None
    parent_digest: str | None = None
    operation_kind: str | None = None
    transition_kind: str | None = None
    selected_candidate_id: str | None = None
    selected_candidate_family: str | None = None
    candidate_binding: dict[str, Any] | None = None
    child: dict[str, Any] | None = None
    child_digest: str | None = None
    materialization_certificate: dict[str, Any] | None = None
    preservation_certificate: dict[str, Any] | None = None
    rollback_boundary: dict[str, Any] | None = None
    evaluator_gate: dict[str, Any] | None = None

    if source_disposition in TRANSITION_BY_SOURCE:
        registry = TRANSITION_BY_SOURCE[source_disposition]
        selected, candidate_binding = _selected_candidate(
            source_report, receipt, source_disposition
        )
        parent, parent_digest, parent_geometry = _verified_parent(
            adapter_report, source_report, selected
        )
        seed_before = canonical_json_bytes(adapter_report["recompetition_seed"])
        operation_kind = registry["operation_kind"]
        transition_kind = registry["transition_kind"]
        selected_candidate_id = selected["candidate_id"]
        selected_candidate_family = selected["candidate_family"]
        (
            child,
            materialization_certificate,
            preservation_certificate,
            rollback_boundary,
            evaluator_gate,
        ) = _materialize_child(
            parent,
            parent_digest,
            parent_geometry,
            selected,
            candidate_binding,
            source_binding,
            source_report,
            contract_digest,
            registry,
        )
        child_digest = _digest(child)
        if canonical_json_bytes(adapter_report["recompetition_seed"]) != seed_before:
            raise ShadowIntervalMultiQTheoryTransitionValidationError(
                "source recompetition seed was mutated"
            )
    else:
        # Every nonselection route has one exact all-null materialization surface.
        nulls = (
            parent,
            parent_digest,
            operation_kind,
            transition_kind,
            selected_candidate_id,
            selected_candidate_family,
            candidate_binding,
            child,
            child_digest,
            materialization_certificate,
            preservation_certificate,
            rollback_boundary,
            evaluator_gate,
        )
        if any(item is not None for item in nulls):
            raise ShadowIntervalMultiQTheoryTransitionValidationError(
                "nonselection materialization surfaces must all be null"
            )

    lifecycle = _record_lifecycle(source_report)
    materialized = child is not None
    authority = _copy(AUTHORITY_BOUNDARY)
    authority.update(
        {
            "source_v2_public_exact_replay_performed": True,
            "selected_candidate_materialization_performed": materialized,
            "fresh_post_transition_evaluator_created": False,
            "fresh_post_transition_qualification_performed": False,
            "source_seed_mutated": False,
            "rollback_executed": False,
            "probe_executed": False,
            "language_expansion_executed": False,
            "adoption_eligibility_determined": False,
            "adoption_decided": False,
            "promotion_decided": False,
            "current_pointer_written": False,
            "parent_or_ambient_state_written": False,
        }
    )
    events = [
        _audit_event(
            0,
            "SOURCE_INTERVAL_MULTI_Q_COMPETITION_VERIFIED",
            GENESIS_DIGEST,
            {
                "verification_status": source_binding["verification_status"],
                "source_report_digest": source_binding["report_digest"],
                "source_audit_head": source_report["audit_head"],
                "source_disposition": source_disposition,
            },
        )
    ]
    events.append(
        _audit_event(
            1,
            (
                "DETACHED_SHADOW_INTERVAL_MULTI_Q_CHILD_MATERIALIZED"
                if materialized
                else "SHADOW_INTERVAL_MULTI_Q_TRANSITION_NOT_MATERIALIZED"
            ),
            events[-1]["event_digest"],
            {
                "disposition": transition_disposition,
                "operation_kind": operation_kind,
                "transition_kind": transition_kind,
                "selected_candidate_id": selected_candidate_id,
                "child_theory_state_digest": child_digest,
                "materialization_status": (
                    SELECTION["materialized_status"]
                    if materialized
                    else SELECTION["no_materialization_status"]
                ),
            },
        )
    )
    events.append(
        _audit_event(
            2,
            "FRESH_QUALIFICATION_AND_EXTERNAL_AUTHORITY_WITHHELD",
            events[-1]["event_digest"],
            {
                "fresh_qualification_performed": False,
                "adoption_eligibility": SELECTION["adoption_eligibility"],
                "adoption_status": SELECTION["adoption_status"],
                "promotion_status": SELECTION["promotion_status"],
                "current_status": SELECTION["current_status"],
                "language_expansion_executed": False,
            },
        )
    )
    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "contract_digest": contract_digest,
        "source_interval_competition": source_binding,
        "parent_theory_state": parent,
        "parent_theory_state_digest": parent_digest,
        "disposition": transition_disposition,
        "operation_kind": operation_kind,
        "transition_kind": transition_kind,
        "selected_candidate_id": selected_candidate_id,
        "selected_candidate_family": selected_candidate_family,
        "selected_candidate_binding": candidate_binding,
        "child_theory_state": child,
        "child_theory_state_digest": child_digest,
        "materialization_certificate": materialization_certificate,
        "preservation_certificate": preservation_certificate,
        "rollback_boundary": rollback_boundary,
        "evaluator_gate": evaluator_gate,
        "record_lifecycle_extension": lifecycle,
        "authority_boundary": authority,
        "adoption_eligibility": SELECTION["adoption_eligibility"],
        "adoption_status": SELECTION["adoption_status"],
        "promotion_status": SELECTION["promotion_status"],
        "current_status": SELECTION["current_status"],
        "nonclaims": _copy(normalized_contract["nonclaims"]),
        "input_artifacts": artifacts,
        "audit_events": events,
        "audit_head": events[-1]["event_digest"],
    }
    report = _finalize_report(body)
    if tuple(canonical_json_bytes(item) for item in source_objects) != source_snapshots:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "source inputs were mutated during transition materialization"
        )
    return ShadowIntervalMultiQTheoryTransitionResult(report=report)


def verify_shadow_interval_multi_q_theory_transition(
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
    interval_competition_input: Mapping[str, Any],
    interval_competition_contract: Mapping[str, Any],
    interval_competition_report: Mapping[str, Any],
    interval_transition_contract: Mapping[str, Any],
    interval_transition_report: Mapping[str, Any],
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
    expected_interval_competition_input_digest: str,
    expected_interval_competition_contract_digest: str,
    expected_interval_competition_report_digest: str,
    expected_interval_competition_input_artifacts: Mapping[str, Any] | None,
    expected_interval_transition_contract_digest: str,
    expected_interval_transition_report_digest: str,
    expected_interval_transition_input_artifacts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Replay and byte-compare one transition report against all anchors."""

    expected_report_digest = _require_digest(
        expected_interval_transition_report_digest,
        "expected_interval_transition_report_digest",
    )
    supplied = _copy(_mapping(interval_transition_report, "interval_transition_report"))
    _exact_keys(supplied, REPORT_FIELDS, "interval_transition_report")
    if expected_interval_transition_input_artifacts is not None and not isinstance(
        expected_interval_transition_input_artifacts, Mapping
    ):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "expected_interval_transition_input_artifacts must be an object or null"
        )
    expected_artifacts = (
        None
        if expected_interval_transition_input_artifacts is None
        else _copy(expected_interval_transition_input_artifacts)
    )
    if _contains_key(expected_artifacts, "observed_value"):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "expected transition artifacts must not embed observed evidence values"
        )
    if supplied.get("input_artifacts") != expected_artifacts:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "transition report input_artifacts differ from independent expectation"
        )
    fresh = materialize_shadow_interval_multi_q_theory_transition(
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
        adapter_report,
        interval_competition_input,
        interval_competition_contract,
        interval_competition_report,
        interval_transition_contract,
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
        expected_adapter_report_digest=expected_adapter_report_digest,
        expected_adapter_input_artifacts=expected_adapter_input_artifacts,
        expected_interval_competition_input_digest=(
            expected_interval_competition_input_digest
        ),
        expected_interval_competition_contract_digest=(
            expected_interval_competition_contract_digest
        ),
        expected_interval_competition_report_digest=(
            expected_interval_competition_report_digest
        ),
        expected_interval_competition_input_artifacts=(
            expected_interval_competition_input_artifacts
        ),
        expected_interval_transition_contract_digest=(
            expected_interval_transition_contract_digest
        ),
        input_artifacts=expected_artifacts,
    ).to_dict()
    if fresh["report_digest"] != expected_report_digest:
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "replayed transition report digest differs from independent expectation"
        )
    if canonical_json_bytes(supplied) != canonical_json_bytes(fresh):
        raise ShadowIntervalMultiQTheoryTransitionValidationError(
            "supplied transition report differs from exact replay"
        )
    return {
        "status": "VERIFIED_" + fresh["disposition"],
        "disposition": fresh["disposition"],
        "report_digest": fresh["report_digest"],
        "contract_digest": fresh["contract_digest"],
        "source_interval_competition_report_digest": fresh[
            "source_interval_competition"
        ]["report_digest"],
        "transition_materialized": fresh["child_theory_state"] is not None,
        "selected_candidate_id": fresh["selected_candidate_id"],
        "selected_operation_kind": fresh["operation_kind"],
        "transition_kind": fresh["transition_kind"],
        "child_theory_state_digest": fresh["child_theory_state_digest"],
        "fresh_qualification_performed": False,
        "adoption_status": fresh["adoption_status"],
        "promotion_status": fresh["promotion_status"],
        "current_status": fresh["current_status"],
    }
