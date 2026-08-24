"""Freshly qualify a restricted interval shadow and adjudicate a rollback target.

This additive V1 core exact-replays the six preceding shadow layers, evaluates
the verified source and restricted interval tables on one new holdout/stress
epoch, and selects exact bytes for a local shadow target.  It deliberately
stops at a fail-closed cycle boundary: the V1 competition surface cannot ingest
an interval model carrying two probes, so this module neither expands language
nor adopts, promotes, makes current, mutates, or rolls back any theory.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from performance.shadow_robust_interval_restriction import (
    ShadowRobustIntervalRestrictionValidationError,
    verify_shadow_robust_interval_restriction,
)
from performance.theory_operation_competition import (
    CompetitionValidationError,
    canonical_json_bytes,
)


CONTRACT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-post-restriction-adjudication-contract/1"
)
CONTRACT_ID = "shadow_post_restriction_adjudication_v1"
SOURCE_RESTRICTION_CONTRACT_ID = "shadow_robust_interval_restriction_v1"
SOURCE_RESTRICTION_CONTRACT_DIGEST = (
    "sha256:57e7beb6a1a409cb959be3e98192158311bcb855e4f8295fbd90c4c40e9eb512"
)
SOURCE_RESTRICTION_REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-robust-interval-restriction-report/1"
)
INPUT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-post-restriction-adjudication-input/1"
)
REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-post-restriction-adjudication-report/1"
)
GENESIS_DIGEST = "sha256:" + "0" * 64

ADOPTION_ELIGIBILITY = "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED"
ADOPTION_STATUS = "NOT_ADOPTED_SHADOW_ONLY"
PROMOTION_STATUS = "NOT_PROMOTED"
CURRENT_STATUS = "NOT_CURRENT"

PROBE_IDS = [
    "absolute_error_point_prediction",
    "normalized_signed_interval_boundary_margin",
]
PROBE_REGISTRY = [
    {
        "probe_id": "absolute_error_point_prediction",
        "semantics": "absolute observed-minus-center error",
        "aggregation": "mean_and_max_by_fresh_split",
    },
    {
        "probe_id": "normalized_signed_interval_boundary_margin",
        "semantics": (
            "radius minus absolute observed-minus-center error, normalized "
            "by an observation-independent source scale"
        ),
        "aggregation": (
            "minimum_margin_and_boundary_exceedance_by_fresh_split"
        ),
    },
]

EVIDENCE_POLICY = {
    "required_splits": ["holdout", "stress"],
    "require_complete_parent_context_scope_pairs_per_split": True,
    "require_exactly_one_row_per_context_scope_pair_per_split": True,
    "require_unique_new_observation_ids": True,
    "require_disjoint_from_competition_ids": True,
    "require_disjoint_from_qualification_ids": True,
    "require_disjoint_from_failure_boundary_probe_ids": True,
    "require_disjoint_from_restriction_ids": True,
    "require_exact_derived_epoch": True,
    "require_exact_inherited_fixed_anchor": True,
    "forbid_cross_epoch_pooling": True,
}

THRESHOLDS = {
    "numeric_epsilon": 1e-12,
    "max_boundary_violation_rate": 0.0,
}

SELECTION = {
    "retain_restricted_status": (
        "POST_RESTRICTION_QUALIFIED_RETAIN_RESTRICTED_SHADOW"
    ),
    "source_rollback_target_status": (
        "POST_RESTRICTION_FAILED_SOURCE_SHADOW_ROLLBACK_TARGET_SELECTED"
    ),
    "both_failed_status": (
        "POST_RESTRICTION_AND_SOURCE_FAILED_RECOMPETITION_ADAPTER_REQUIRED"
    ),
    "needs_evidence_status": "POST_RESTRICTION_NEEDS_NEW_EVIDENCE",
    "incomparable_status": "POST_RESTRICTION_INCOMPARABLE_EVALUATOR_EPOCH",
    "source_materialized_status": (
        "MATERIALIZED_SHADOW_ROBUST_INTERVAL_RESTRICTION"
    ),
    "retain_restricted_route": (
        "RETAIN_RESTRICTED_SHADOW_AND_REQUIRE_RECOMPETITION_ADAPTER_FOR_NEXT_CYCLE"
    ),
    "source_rollback_target_route": (
        "SOURCE_ROLLBACK_TARGET_SELECTED_AND_REQUIRE_RECOMPETITION_ADAPTER_FOR_NEXT_CYCLE"
    ),
    "both_failed_route": (
        "RECOMPETITION_ADAPTER_REQUIRED_BEFORE_LANGUAGE_EXPANSION"
    ),
    "needs_evidence_route": "NEW_COMPLETE_POST_RESTRICTION_EVIDENCE_REQUIRED",
    "incomparable_route": "COMPARABLE_POST_RESTRICTION_EPOCH_REQUIRED",
    "adoption_eligibility": ADOPTION_ELIGIBILITY,
    "adoption_status": ADOPTION_STATUS,
    "promotion_status": PROMOTION_STATUS,
    "current_status": CURRENT_STATUS,
}

LOOP_COMPATIBILITY_BOUNDARY = {
    "competition_v1_required_model_kind": "finite_point_table",
    "competition_v1_required_probe_ids": [
        "absolute_error_point_prediction"
    ],
    "selected_source_model_kind": "finite_interval_table",
    "selected_source_probe_ids": PROBE_IDS,
    "feed_back_compatible": False,
    "language_last_resort_certified": False,
    "language_expansion_executed": False,
    "required_bridge": (
        "INTERVAL_MULTI_Q_THEORY_OPERATION_RECOMPETITION_ADAPTER_NOT_IMPLEMENTED"
    ),
}

RECORD_LIFECYCLE_POLICY = {
    "competition_record_role": "AUDIT_ONLY_SCORING_EXCLUDED",
    "qualification_record_role": "CONSUMED_AUDIT_ONLY_SCORING_EXCLUDED",
    "failure_boundary_probe_record_role": (
        "CONSUMED_FAILURE_BOUNDARY_EVIDENCE_AUDIT_ONLY"
    ),
    "restriction_record_role": (
        "CONSUMED_RESTRICTION_COMPETITION_EVIDENCE_AUDIT_ONLY"
    ),
    "post_restriction_record_role": (
        "CONSUMED_POST_RESTRICTION_ADJUDICATION_EVIDENCE_AUDIT_ONLY"
    ),
    "all_record_classes_eligible_for_future_scoring": False,
    "future_scoring_requires_new_unconsumed_evidence": True,
    "cross_epoch_pooling_allowed": False,
    "logical_selective_erasure_applied": True,
    "physical_erasure": "NOT_PERFORMED",
}

AUTHORITY_BOUNDARY = {
    "scope": "LOCAL_SHADOW_POST_RESTRICTION_ADJUDICATION_ONLY",
    "source_state_mutation": False,
    "restricted_state_mutation": False,
    "rollback_target_selection": True,
    "rollback_execution": False,
    "language_expansion_execution": False,
    "recompetition_adapter_execution": False,
    "adoption_decision": False,
    "promotion_decision": False,
    "current_pointer_write": False,
    "parent_or_source_state_write": False,
    "external_data_attestation": "REQUIRED_NOT_PRESENT",
    "external_evaluator_attestation": "REQUIRED_NOT_PRESENT",
    "external_adoption_authority": "REQUIRED_NOT_PRESENT",
}

MANDATORY_NONCLAIMS = (
    "shadow_only",
    "post_restriction_fresh_qualification_only",
    "finite_interval_source_and_restricted_comparison_only",
    "fixed_two_probe_registry_replay_only",
    "no_generic_adjudication_engine",
    "no_rigid_body_markov_or_independence_assumption",
    "no_scope_restriction",
    "no_quotient_restriction",
    "no_new_probe",
    "q_registry_replayed_not_invented",
    "v_registry_unchanged",
    "no_language_or_predicate_invention",
    "language_last_resort_not_certified",
    "language_expansion_not_executed",
    "recompetition_adapter_not_implemented",
    "competition_v1_incompatible_with_interval_multi_q_state",
    "no_external_probe_acquisition",
    "caller_supplied_static_rows_only",
    "local_epoch_is_not_external_attestation",
    "fresh_evidence_pass_is_not_global_preservation",
    "finite_interval_nesting_is_only_checked_on_the_frozen_finite_table",
    "center_error_equality_is_not_domain_safety",
    "rollback_target_selection_is_not_rollback_execution",
    "no_source_child_invalidation",
    "no_rollback_execution",
    "no_adoption_eligibility_determination",
    "no_adoption",
    "no_promotion",
    "no_current_pointer_write",
    "no_parent_source_or_ambient_state_write",
    "no_h_t_to_h_t_plus_1_acceptance",
    "no_cross_epoch_pooling",
    "no_physical_erasure",
    "no_external_data_or_evaluator_attestation",
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

REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "contract_digest",
        "adjudication_input_digest",
        "adjudication_id",
        "source_restriction",
        "evaluator_definition",
        "evaluator_binding",
        "evidence_binding",
        "probe_registry_replay",
        "source_probe_results",
        "restricted_probe_results",
        "finite_interval_tradeoff",
        "monotonicity_certificate",
        "disposition",
        "shadow_state_selection",
        "rollback_adjudication",
        "cycle_route",
        "loop_compatibility_boundary",
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


class ShadowPostRestrictionAdjudicationValidationError(ValueError):
    """Raised when the frozen adjudication contract or replay fails."""


class ShadowPostRestrictionAdjudicationDisposition(str, Enum):
    POST_RESTRICTION_QUALIFIED_RETAIN_RESTRICTED_SHADOW = (
        "POST_RESTRICTION_QUALIFIED_RETAIN_RESTRICTED_SHADOW"
    )
    POST_RESTRICTION_FAILED_SOURCE_SHADOW_ROLLBACK_TARGET_SELECTED = (
        "POST_RESTRICTION_FAILED_SOURCE_SHADOW_ROLLBACK_TARGET_SELECTED"
    )
    POST_RESTRICTION_AND_SOURCE_FAILED_RECOMPETITION_ADAPTER_REQUIRED = (
        "POST_RESTRICTION_AND_SOURCE_FAILED_RECOMPETITION_ADAPTER_REQUIRED"
    )
    POST_RESTRICTION_NEEDS_NEW_EVIDENCE = (
        "POST_RESTRICTION_NEEDS_NEW_EVIDENCE"
    )
    POST_RESTRICTION_INCOMPARABLE_EVALUATOR_EPOCH = (
        "POST_RESTRICTION_INCOMPARABLE_EVALUATOR_EPOCH"
    )


@dataclass(frozen=True)
class ShadowPostRestrictionAdjudicationResult:
    report: Mapping[str, Any]

    @property
    def disposition(self) -> str:
        return str(self.report["disposition"])

    @property
    def report_digest(self) -> str:
        return str(self.report["report_digest"])

    @property
    def restricted_shadow_qualified(self) -> bool:
        return self.report["disposition"] == SELECTION["retain_restricted_status"]

    @property
    def rollback_target_selected(self) -> bool:
        return (
            self.report["disposition"]
            == SELECTION["source_rollback_target_status"]
        )

    @property
    def cycle_route(self) -> str:
        return str(self.report["cycle_route"])

    def to_dict(self) -> dict[str, Any]:
        return _copy(self.report)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShadowPostRestrictionAdjudicationValidationError(
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
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} keys differ; missing={missing}, extra={extra}"
        )


def _string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} must be a non-empty string"
        )
    return value


def _finite_number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} must be a finite number"
        )
    return float(value)


def _json_scalar(value: Any, label: str) -> str | int | float | bool:
    if value is None or type(value) not in (str, int, float, bool):
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} must be a non-null finite JSON scalar"
        )
    if type(value) is float and not math.isfinite(value):
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} must be finite"
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
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"value is not detached finite canonical JSON: {exc}"
        ) from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_digest(value: Any, label: str) -> str:
    digest = _string(value, label)
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} must be sha256:<64hex>"
        )
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} is not hexadecimal"
        ) from exc
    return digest


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} differs from frozen post-restriction adjudication V1"
        )


def _reject_observed_values(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        if "observed_value" in value:
            raise ShadowPostRestrictionAdjudicationValidationError(
                f"{label} must not embed observation values"
            )
        for key, item in value.items():
            _reject_observed_values(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_observed_values(item, f"{label}[{index}]")


def _validate_contract(contract_value: Any) -> dict[str, Any]:
    contract = _mapping(contract_value, "adjudication_contract")
    _exact_keys(
        contract,
        {
            "schema_version",
            "contract_id",
            "source_restriction_contract_id",
            "source_restriction_contract_digest",
            "source_restriction_report_schema_version",
            "input_schema_version",
            "report_schema_version",
            "probe_registry",
            "evidence_policy",
            "thresholds",
            "selection",
            "loop_compatibility_boundary",
            "record_lifecycle_policy",
            "authority_boundary",
            "nonclaims",
        },
        "adjudication_contract",
    )
    frozen = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "source_restriction_contract_id": SOURCE_RESTRICTION_CONTRACT_ID,
        "source_restriction_contract_digest": SOURCE_RESTRICTION_CONTRACT_DIGEST,
        "source_restriction_report_schema_version": (
            SOURCE_RESTRICTION_REPORT_SCHEMA_VERSION
        ),
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "report_schema_version": REPORT_SCHEMA_VERSION,
    }
    for key, expected in frozen.items():
        _require_equal(contract[key], expected, key)
    for key, expected in (
        ("probe_registry", PROBE_REGISTRY),
        ("evidence_policy", EVIDENCE_POLICY),
        ("thresholds", THRESHOLDS),
        ("selection", SELECTION),
        ("loop_compatibility_boundary", LOOP_COMPATIBILITY_BOUNDARY),
        ("record_lifecycle_policy", RECORD_LIFECYCLE_POLICY),
        ("authority_boundary", AUTHORITY_BOUNDARY),
    ):
        _require_equal(_copy(contract[key]), expected, key)
    nonclaims = contract["nonclaims"]
    if not isinstance(nonclaims, list) or tuple(nonclaims) != MANDATORY_NONCLAIMS:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "nonclaims differ from the exact mandatory V1 list"
        )
    return _copy(contract)


def validate_shadow_post_restriction_adjudication_contract(
    contract_value: Any,
) -> dict[str, Any]:
    """Validate and detach the frozen post-restriction contract."""

    return _validate_contract(contract_value)


def _epoch_components(
    *,
    restriction_contract_digest: str,
    restriction_report_digest: str,
    restricted_shadow_theory_state_digest: str,
    source_probe_expanded_shadow_theory_state_digest: str,
    fixed_anchor: str,
    adjudication_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    contract = _validate_contract(adjudication_contract)
    source_contract = _require_digest(
        restriction_contract_digest, "restriction_contract_digest"
    )
    if source_contract != SOURCE_RESTRICTION_CONTRACT_DIGEST:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "restriction contract digest is not the pinned V1 source"
        )
    payload = {
        "restriction_contract_digest": source_contract,
        "restriction_report_digest": _require_digest(
            restriction_report_digest, "restriction_report_digest"
        ),
        "restricted_shadow_theory_state_digest": _require_digest(
            restricted_shadow_theory_state_digest,
            "restricted_shadow_theory_state_digest",
        ),
        "source_probe_expanded_shadow_theory_state_digest": _require_digest(
            source_probe_expanded_shadow_theory_state_digest,
            "source_probe_expanded_shadow_theory_state_digest",
        ),
        "fixed_anchor": _string(fixed_anchor, "fixed_anchor"),
        "adjudication_contract_digest": _digest(contract),
        "fixed_probe_registry": _copy(contract["probe_registry"]),
    }
    epoch = "shadow-post-restriction-epoch:" + _digest(payload)[7:]
    return payload, epoch


def derive_shadow_post_restriction_adjudication_epoch(
    *,
    restriction_contract_digest: str,
    restriction_report_digest: str,
    restricted_shadow_theory_state_digest: str,
    source_probe_expanded_shadow_theory_state_digest: str,
    fixed_anchor: str,
    adjudication_contract: Mapping[str, Any],
) -> str:
    """Derive a fresh local epoch without binding observations or row order."""

    return _epoch_components(
        restriction_contract_digest=restriction_contract_digest,
        restriction_report_digest=restriction_report_digest,
        restricted_shadow_theory_state_digest=(
            restricted_shadow_theory_state_digest
        ),
        source_probe_expanded_shadow_theory_state_digest=(
            source_probe_expanded_shadow_theory_state_digest
        ),
        fixed_anchor=fixed_anchor,
        adjudication_contract=adjudication_contract,
    )[1]


def _observation_ids(
    input_value: Mapping[str, Any], splits: Sequence[str], label: str
) -> list[str]:
    value = _mapping(input_value, f"{label} input")
    evidence = _mapping(value.get("evidence"), f"{label} evidence")
    result: list[str] = []
    for split in splits:
        rows = evidence.get(split)
        if not isinstance(rows, list):
            raise ShadowPostRestrictionAdjudicationValidationError(
                f"{label} evidence.{split} must be a list"
            )
        for index, raw in enumerate(rows):
            row = _mapping(raw, f"{label} evidence.{split}[{index}]")
            result.append(
                _string(
                    row.get("observation_id"),
                    f"{label} evidence.{split}[{index}].observation_id",
                )
            )
    if len(result) != len(set(result)):
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} observation IDs are not unique"
        )
    return sorted(result)


def _normalize_context(
    value: Any,
    feature_ids: Sequence[str],
    registered_contexts: set[bytes],
    label: str,
) -> dict[str, Any]:
    context = _mapping(value, label)
    _exact_keys(context, set(feature_ids), label)
    normalized = {
        feature_id: _json_scalar(context[feature_id], f"{label}.{feature_id}")
        for feature_id in feature_ids
    }
    if canonical_json_bytes(normalized) not in registered_contexts:
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} is not a registered source context"
        )
    return normalized


def _normalize_rows(
    value: Any,
    split: str,
    feature_ids: Sequence[str],
    registered_contexts: set[bytes],
    registered_scopes: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"evidence.{split} must be a list"
        )
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        label = f"evidence.{split}[{index}]"
        row = _mapping(raw, label)
        _exact_keys(
            row,
            {
                "observation_id",
                "evaluator_epoch",
                "fixed_anchor",
                "scope_id",
                "context",
                "observed_value",
            },
            label,
        )
        scope_id = _string(row["scope_id"], f"{label}.scope_id")
        if scope_id not in registered_scopes:
            raise ShadowPostRestrictionAdjudicationValidationError(
                f"{label}.scope_id is not registered"
            )
        rows.append(
            {
                "observation_id": _string(
                    row["observation_id"], f"{label}.observation_id"
                ),
                "evaluator_epoch": _string(
                    row["evaluator_epoch"], f"{label}.evaluator_epoch"
                ),
                "fixed_anchor": _string(
                    row["fixed_anchor"], f"{label}.fixed_anchor"
                ),
                "scope_id": scope_id,
                "context": _normalize_context(
                    row["context"],
                    feature_ids,
                    registered_contexts,
                    f"{label}.context",
                ),
                "observed_value": _finite_number(
                    row["observed_value"], f"{label}.observed_value"
                ),
            }
        )
    rows.sort(key=lambda item: item["observation_id"])
    ids = [row["observation_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"evidence.{split} observation IDs are duplicated"
        )
    return rows


def _normalize_input(
    input_value: Any,
    *,
    restriction_report: Mapping[str, Any],
    source_state: Mapping[str, Any],
    restricted_state: Mapping[str, Any],
    competition_input: Mapping[str, Any],
    qualification_input: Mapping[str, Any],
    probe_input: Mapping[str, Any],
    restriction_input: Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(input_value, "adjudication_input")
    _exact_keys(
        value,
        {
            "schema_version",
            "adjudication_id",
            "source_restriction",
            "evaluator",
            "prior_record_exclusion",
            "evidence",
        },
        "adjudication_input",
    )
    _require_equal(value["schema_version"], INPUT_SCHEMA_VERSION, "input schema")

    source = _mapping(value["source_restriction"], "source_restriction")
    _exact_keys(
        source,
        {
            "restriction_contract_digest",
            "restriction_report_digest",
            "restricted_shadow_theory_state_digest",
            "source_probe_expanded_shadow_theory_state_digest",
        },
        "source_restriction",
    )
    expected_source = {
        "restriction_contract_digest": restriction_report["contract_digest"],
        "restriction_report_digest": restriction_report["report_digest"],
        "restricted_shadow_theory_state_digest": restriction_report[
            "restricted_shadow_theory_state_digest"
        ],
        "source_probe_expanded_shadow_theory_state_digest": restriction_report[
            "source_probe_expanded_shadow_theory_state_digest"
        ],
    }
    _require_equal(_copy(source), expected_source, "source_restriction")

    evaluator = _mapping(value["evaluator"], "evaluator")
    _exact_keys(evaluator, {"evaluator_epoch", "fixed_anchor"}, "evaluator")
    normalized_evaluator = {
        "evaluator_epoch": _string(
            evaluator["evaluator_epoch"], "evaluator.evaluator_epoch"
        ),
        "fixed_anchor": _string(
            evaluator["fixed_anchor"], "evaluator.fixed_anchor"
        ),
    }

    competition_ids = _observation_ids(
        competition_input, ("discovery", "validation", "stress"), "competition"
    )
    qualification_ids = _observation_ids(
        qualification_input, ("holdout", "stress"), "qualification"
    )
    boundary_ids = _observation_ids(
        probe_input, ("holdout", "stress"), "failure-boundary probe"
    )
    restriction_ids = _observation_ids(
        restriction_input,
        ("calibration", "holdout", "stress"),
        "restriction",
    )
    old_ids = {
        "competition": competition_ids,
        "qualification": qualification_ids,
        "failure_boundary_probe": boundary_ids,
        "restriction": restriction_ids,
    }
    exclusion = _mapping(
        value["prior_record_exclusion"], "prior_record_exclusion"
    )
    _exact_keys(exclusion, set(old_ids), "prior_record_exclusion")
    expected_exclusion = {key: _digest(ids) for key, ids in old_ids.items()}
    _require_equal(_copy(exclusion), expected_exclusion, "prior_record_exclusion")

    object_space = _mapping(source_state.get("object_space"), "source object_space")
    feature_values = object_space.get("feature_ids")
    context_values = object_space.get("contexts")
    scope_values = source_state.get("scope_ids")
    if (
        not isinstance(feature_values, list)
        or not feature_values
        or not isinstance(context_values, list)
        or not context_values
        or not isinstance(scope_values, list)
        or not scope_values
    ):
        raise ShadowPostRestrictionAdjudicationValidationError(
            "verified source lacks a finite feature/context/scope registry"
        )
    feature_ids = [_string(item, "source feature_id") for item in feature_values]
    if len(feature_ids) != len(set(feature_ids)):
        raise ShadowPostRestrictionAdjudicationValidationError(
            "source feature IDs are duplicated"
        )
    registered_contexts = {canonical_json_bytes(item) for item in context_values}
    if len(registered_contexts) != len(context_values):
        raise ShadowPostRestrictionAdjudicationValidationError(
            "source contexts are duplicated"
        )
    registered_scopes = {_string(item, "source scope_id") for item in scope_values}
    if len(registered_scopes) != len(scope_values):
        raise ShadowPostRestrictionAdjudicationValidationError(
            "source scopes are duplicated"
        )
    if canonical_json_bytes(restricted_state.get("object_space")) != canonical_json_bytes(
        object_space
    ) or canonical_json_bytes(restricted_state.get("scope_ids")) != canonical_json_bytes(
        scope_values
    ):
        raise ShadowPostRestrictionAdjudicationValidationError(
            "restricted state changed the source object-space or scope registry"
        )

    evidence = _mapping(value["evidence"], "evidence")
    _exact_keys(evidence, {"holdout", "stress"}, "evidence")
    normalized_evidence = {
        split: _normalize_rows(
            evidence[split],
            split,
            feature_ids,
            registered_contexts,
            registered_scopes,
        )
        for split in ("holdout", "stress")
    }
    new_ids = [
        row["observation_id"]
        for split in ("holdout", "stress")
        for row in normalized_evidence[split]
    ]
    if len(new_ids) != len(set(new_ids)):
        raise ShadowPostRestrictionAdjudicationValidationError(
            "new observation IDs are duplicated across splits"
        )
    for key, ids in old_ids.items():
        if set(new_ids) & set(ids):
            raise ShadowPostRestrictionAdjudicationValidationError(
                f"new observation IDs reuse {key} records"
            )

    public = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "adjudication_id": _string(value["adjudication_id"], "adjudication_id"),
        "source_restriction": expected_source,
        "evaluator": normalized_evaluator,
        "prior_record_exclusion": expected_exclusion,
        "evidence": normalized_evidence,
    }
    return {
        "public": public,
        "evidence": normalized_evidence,
        "new_ids": sorted(new_ids),
        "old_ids": old_ids,
        "parent_contexts": _copy(context_values),
        "parent_scopes": sorted(registered_scopes),
    }


def _evaluator_binding(
    normalized_input: Mapping[str, Any],
    expected_epoch: str,
    expected_anchor: str,
) -> dict[str, Any]:
    evaluator = normalized_input["public"]["evaluator"]
    rows = [
        row
        for split in ("holdout", "stress")
        for row in normalized_input["evidence"][split]
    ]
    supplied_epoch = evaluator["evaluator_epoch"]
    supplied_anchor = evaluator["fixed_anchor"]
    epoch_matches = supplied_epoch == expected_epoch and all(
        row["evaluator_epoch"] == expected_epoch for row in rows
    )
    anchor_matches = supplied_anchor == expected_anchor and all(
        row["fixed_anchor"] == expected_anchor for row in rows
    )
    return {
        "exact_derived_epoch_required": True,
        "expected_evaluator_epoch": expected_epoch,
        "supplied_evaluator_epoch": supplied_epoch,
        "epoch_matches": epoch_matches,
        "expected_fixed_anchor": expected_anchor,
        "supplied_fixed_anchor": supplied_anchor,
        "fixed_anchor_matches": anchor_matches,
        "comparable": epoch_matches and anchor_matches,
    }


def _evidence_binding(normalized_input: Mapping[str, Any]) -> dict[str, Any]:
    required_by_key = {
        canonical_json_bytes({"scope_id": scope, "context": context}): {
            "scope_id": scope,
            "context": _copy(context),
        }
        for scope in normalized_input["parent_scopes"]
        for context in normalized_input["parent_contexts"]
    }
    required_keys = set(required_by_key)
    required_pairs = [required_by_key[key] for key in sorted(required_by_key)]
    row_counts: dict[str, int] = {}
    covered: dict[str, list[dict[str, Any]]] = {}
    missing: dict[str, list[dict[str, Any]]] = {}
    duplicate_counts: dict[str, int] = {}
    complete: dict[str, bool] = {}
    for split in ("holdout", "stress"):
        counts: dict[bytes, int] = {}
        for row in normalized_input["evidence"][split]:
            key = canonical_json_bytes(
                {"scope_id": row["scope_id"], "context": row["context"]}
            )
            counts[key] = counts.get(key, 0) + 1
        keys = set(counts)
        row_counts[split] = sum(counts.values())
        covered[split] = [required_by_key[key] for key in sorted(keys & required_keys)]
        missing[split] = [required_by_key[key] for key in sorted(required_keys - keys)]
        duplicate_counts[split] = sum(max(0, count - 1) for count in counts.values())
        complete[split] = keys == required_keys and all(
            count == 1 for count in counts.values()
        )
    old_ids = normalized_input["old_ids"]
    return {
        "holdout_evidence_digest": _digest(normalized_input["evidence"]["holdout"]),
        "stress_evidence_digest": _digest(normalized_input["evidence"]["stress"]),
        "new_observation_id_digest": _digest(normalized_input["new_ids"]),
        "new_observation_count": len(normalized_input["new_ids"]),
        "prior_observation_id_digests": {
            key: _digest(ids) for key, ids in old_ids.items()
        },
        "unique_new_observation_ids": True,
        "disjoint_from_competition_ids": True,
        "disjoint_from_qualification_ids": True,
        "disjoint_from_failure_boundary_probe_ids": True,
        "disjoint_from_restriction_ids": True,
        "cross_epoch_pooling": False,
        "row_counts": row_counts,
        "required_context_scope_pairs": required_pairs,
        "required_context_scope_pair_count": len(required_pairs),
        "covered_context_scope_pairs_by_split": covered,
        "missing_context_scope_pairs_by_split": missing,
        "duplicate_context_scope_pair_row_counts_by_split": duplicate_counts,
        "complete_exact_cartesian_coverage_by_split": complete,
        "complete_evidence": all(complete.values()),
    }


def _prediction_lookup(value: Any, label: str) -> dict[bytes, float]:
    if not isinstance(value, list) or not value:
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} must be a non-empty list"
        )
    result: dict[bytes, float] = {}
    for index, raw in enumerate(value):
        item = _mapping(raw, f"{label}[{index}]")
        _exact_keys(item, {"context", "value"}, f"{label}[{index}]")
        context = _mapping(item["context"], f"{label}[{index}].context")
        key = canonical_json_bytes(context)
        if key in result:
            raise ShadowPostRestrictionAdjudicationValidationError(
                f"{label} has duplicate contexts"
            )
        result[key] = _finite_number(item["value"], f"{label}[{index}].value")
    return result


def _functional_threshold(state: Mapping[str, Any], label: str) -> float:
    value = state.get("violation_functionals")
    if not isinstance(value, list) or len(value) != 1:
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} must have exactly one violation functional"
        )
    functional = _mapping(value[0], f"{label} violation functional")
    if functional.get("functional_id") != "absolute_error":
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} scale requires the absolute_error functional"
        )
    threshold = _finite_number(
        functional.get("threshold"), f"{label} absolute_error threshold"
    )
    if threshold < 0.0:
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} absolute_error threshold cannot be negative"
        )
    return threshold


def _model_geometry(
    state: Mapping[str, Any],
    contexts: Sequence[Mapping[str, Any]],
    scopes: Sequence[str],
    label: str,
) -> dict[str, Any]:
    model = _mapping(state.get("model_class"), f"{label} model_class")
    _exact_keys(
        model,
        {"kind", "center_predictions", "radius_grouping", "radii"},
        f"{label} model_class",
    )
    if model.get("kind") != "finite_interval_table":
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} model must be a finite interval table"
        )
    centers = _prediction_lookup(
        model.get("center_predictions"), f"{label} center_predictions"
    )
    expected_contexts = {canonical_json_bytes(item) for item in contexts}
    if set(centers) != expected_contexts:
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} centers do not exactly cover the source contexts"
        )
    grouping = model.get("radius_grouping")
    if grouping not in {"global", "per_scope", "per_context"}:
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} radius grouping is unsupported"
        )
    raw_entries = model.get("radii")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} radii must be a non-empty list"
        )
    entries: list[dict[str, Any]] = []
    actual_keys: set[bytes] = set()
    for index, raw in enumerate(raw_entries):
        item = _mapping(raw, f"{label} radii[{index}]")
        _exact_keys(item, {"group", "radius"}, f"{label} radii[{index}]")
        group = _copy(_mapping(item["group"], f"{label} radii[{index}].group"))
        radius = _finite_number(item["radius"], f"{label} radii[{index}].radius")
        if radius < 0.0:
            raise ShadowPostRestrictionAdjudicationValidationError(
                f"{label} radii must be nonnegative"
            )
        key = canonical_json_bytes(group)
        if key in actual_keys:
            raise ShadowPostRestrictionAdjudicationValidationError(
                f"{label} radius group keys are duplicated"
            )
        actual_keys.add(key)
        entries.append({"group": group, "radius": radius})
    if grouping == "global":
        expected_keys = {canonical_json_bytes({"global": "*"})}
    elif grouping == "per_scope":
        expected_keys = {
            canonical_json_bytes({"scope_id": scope}) for scope in scopes
        }
    else:
        expected_keys = {
            canonical_json_bytes({"context": context}) for context in contexts
        }
    if actual_keys != expected_keys:
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} radius group keys do not exactly equal the registry"
        )
    geometry = {
        "model": _copy(model),
        "model_digest": _digest(model),
        "centers": centers,
        "grouping": grouping,
        "entries": entries,
    }
    for scope in scopes:
        for context in contexts:
            _radius_for_pair(geometry, scope, context, label)
    return geometry


def _radius_for_pair(
    geometry: Mapping[str, Any],
    scope_id: str,
    context: Mapping[str, Any],
    label: str,
) -> float:
    grouping = geometry["grouping"]
    matches: list[float] = []
    for item in geometry["entries"]:
        group = item["group"]
        if grouping == "global" and group == {"global": "*"}:
            matches.append(float(item["radius"]))
        elif grouping == "per_scope" and group == {"scope_id": scope_id}:
            matches.append(float(item["radius"]))
        elif grouping == "per_context" and canonical_json_bytes(
            group.get("context")
        ) == canonical_json_bytes(context):
            matches.append(float(item["radius"]))
    if len(matches) != 1:
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} radii do not map uniquely to every context-scope pair"
        )
    return matches[0]


def _probe_registry_replay(
    source_state: Mapping[str, Any], restricted_state: Mapping[str, Any]
) -> dict[str, Any]:
    source_ids = _copy(source_state.get("probe_ids"))
    restricted_ids = _copy(restricted_state.get("probe_ids"))
    source_exact = source_ids == PROBE_IDS
    restricted_exact = restricted_ids == PROBE_IDS
    q_equal = canonical_json_bytes(source_ids) == canonical_json_bytes(
        restricted_ids
    )
    v_equal = canonical_json_bytes(
        source_state.get("violation_functionals")
    ) == canonical_json_bytes(restricted_state.get("violation_functionals"))
    if not source_exact or not restricted_exact or not q_equal or not v_equal:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "verified source/restricted states do not preserve the exact two-Q/V registry"
        )
    return {
        "required_probe_ids": _copy(PROBE_IDS),
        "source_probe_ids": source_ids,
        "restricted_probe_ids": restricted_ids,
        "source_exact_match": source_exact,
        "restricted_exact_match": restricted_exact,
        "q_registry_byte_equal": q_equal,
        "v_registry_byte_equal": v_equal,
        "violation_functionals_byte_equal": v_equal,
        "same_evidence_rows_used_for_both_targets": True,
        "replayed_exactly": (
            source_exact and restricted_exact and q_equal and v_equal
        ),
    }


def _prediction_scale(state: Mapping[str, Any], geometry: Mapping[str, Any]) -> float:
    centers = list(geometry["centers"].values())
    try:
        mean_absolute_center = math.fsum(abs(value) for value in centers) / len(centers)
    except OverflowError as exc:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "source center scale overflowed finite arithmetic"
        ) from exc
    scale = max(
        float(THRESHOLDS["numeric_epsilon"]),
        mean_absolute_center + _functional_threshold(state, "source state"),
    )
    if not math.isfinite(scale):
        raise ShadowPostRestrictionAdjudicationValidationError(
            "source prediction scale is not finite"
        )
    return scale


def _split_metrics(
    rows: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    scale: float,
    label: str,
) -> dict[str, Any]:
    errors: list[float] = []
    margins: list[float] = []
    exceedances: list[float] = []
    counterexample_ids: list[str] = []
    raw_violations: list[bool] = []
    for row in rows:
        key = canonical_json_bytes(row["context"])
        if key not in geometry["centers"]:
            raise ShadowPostRestrictionAdjudicationValidationError(
                f"{label} centers do not cover an adjudication row"
            )
        error = abs(float(row["observed_value"]) - geometry["centers"][key])
        radius = _radius_for_pair(
            geometry, row["scope_id"], row["context"], label
        )
        margin = (radius - error) / scale
        if not math.isfinite(error) or not math.isfinite(margin):
            raise ShadowPostRestrictionAdjudicationValidationError(
                f"{label} probe arithmetic is not finite"
            )
        errors.append(error)
        margins.append(margin)
        exceedances.append(max(0.0, -margin))
        raw_violation = error > radius
        raw_violations.append(raw_violation)
        if raw_violation:
            counterexample_ids.append(row["observation_id"])
    if not rows:
        return {
            "row_count": 0,
            "mean_absolute_center_error": None,
            "max_absolute_center_error": None,
            "min_normalized_signed_interval_boundary_margin": None,
            "boundary_violation_count": 0,
            "boundary_violation_rate": None,
            "mean_normalized_exceedance": None,
            "max_normalized_exceedance": None,
            "counterexample_observation_ids": [],
        }
    # Never classify a boundary through a normalized quotient: a very large
    # observation-independent scale can turn a negative finite difference into
    # -0.0.  The raw finite inequality is the authoritative interval predicate.
    violation_count = sum(raw_violations)
    try:
        mean_error = math.fsum(errors) / len(rows)
        mean_exceedance = math.fsum(exceedances) / len(rows)
    except OverflowError as exc:
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} aggregate probe arithmetic overflowed"
        ) from exc
    if not math.isfinite(mean_error) or not math.isfinite(mean_exceedance):
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"{label} aggregate probe arithmetic is not finite"
        )
    return {
        "row_count": len(rows),
        "mean_absolute_center_error": mean_error,
        "max_absolute_center_error": max(errors),
        "min_normalized_signed_interval_boundary_margin": min(margins),
        "boundary_violation_count": violation_count,
        "boundary_violation_rate": violation_count / len(rows),
        "mean_normalized_exceedance": mean_exceedance,
        "max_normalized_exceedance": max(exceedances),
        "counterexample_observation_ids": sorted(counterexample_ids),
    }


def _compare_models(
    *,
    source_state: Mapping[str, Any],
    restricted_state: Mapping[str, Any],
    contexts: Sequence[Mapping[str, Any]],
    scopes: Sequence[str],
    evidence: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = _model_geometry(source_state, contexts, scopes, "source")
    restricted = _model_geometry(restricted_state, contexts, scopes, "restricted")
    centers_equal = canonical_json_bytes(
        source["model"]["center_predictions"]
    ) == canonical_json_bytes(restricted["model"]["center_predictions"])
    grouping_keys_equal = (
        source["grouping"] == restricted["grouping"]
        and canonical_json_bytes([item["group"] for item in source["entries"]])
        == canonical_json_bytes(
            [item["group"] for item in restricted["entries"]]
        )
    )
    if not centers_equal or not grouping_keys_equal:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "restricted model changed centers, grouping, or group keys"
        )
    scale = _prediction_scale(source_state, source)
    source_results = {
        split: _split_metrics(evidence[split], source, scale, "source")
        for split in ("holdout", "stress")
    }
    restricted_results = {
        split: _split_metrics(evidence[split], restricted, scale, "restricted")
        for split in ("holdout", "stress")
    }
    point_equal_by_split = {
        split: (
            source_results[split]["mean_absolute_center_error"]
            == restricted_results[split]["mean_absolute_center_error"]
            and source_results[split]["max_absolute_center_error"]
            == restricted_results[split]["max_absolute_center_error"]
        )
        for split in ("holdout", "stress")
    }
    if not all(point_equal_by_split.values()):
        raise ShadowPostRestrictionAdjudicationValidationError(
            "point-prediction errors differ despite frozen equal centers"
        )

    all_lte = True
    strict = False
    implication = True
    checked_rows = 0
    for split in ("holdout", "stress"):
        for row in evidence[split]:
            source_radius = _radius_for_pair(
                source, row["scope_id"], row["context"], "source"
            )
            restricted_radius = _radius_for_pair(
                restricted, row["scope_id"], row["context"], "restricted"
            )
            if restricted_radius > source_radius:
                all_lte = False
            if restricted_radius < source_radius:
                strict = True
            center = source["centers"][canonical_json_bytes(row["context"])]
            error = abs(float(row["observed_value"]) - center)
            if not math.isfinite(error):
                raise ShadowPostRestrictionAdjudicationValidationError(
                    "finite interval monotonicity arithmetic is not finite"
                )
            source_pass = error <= source_radius
            restricted_pass = error <= restricted_radius
            if restricted_pass and not source_pass:
                implication = False
            checked_rows += 1
    if not all_lte:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "restricted radii are not pointwise bounded by source radii"
        )
    if not implication:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "impossible restricted-pass/source-fail row encountered"
        )

    source_passed = all(
        source_results[split]["boundary_violation_rate"]
        <= float(THRESHOLDS["max_boundary_violation_rate"])
        for split in ("holdout", "stress")
    )
    restricted_passed = all(
        restricted_results[split]["boundary_violation_rate"]
        <= float(THRESHOLDS["max_boundary_violation_rate"])
        for split in ("holdout", "stress")
    )
    if restricted_passed and not source_passed:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "restricted pass with source fail violates finite interval nesting"
        )
    source_report = {
        "prediction_scale": scale,
        "holdout": source_results["holdout"],
        "stress": source_results["stress"],
        "fresh_qualification_passed": source_passed,
    }
    restricted_report = {
        "prediction_scale": scale,
        "holdout": restricted_results["holdout"],
        "stress": restricted_results["stress"],
        "fresh_qualification_passed": restricted_passed,
    }
    tradeoff = {
        "source_model_class_digest": source["model_digest"],
        "restricted_model_class_digest": restricted["model_digest"],
        "centers_byte_equal": centers_equal,
        "grouping_and_group_keys_byte_equal": grouping_keys_equal,
        "point_prediction_error_equal_by_split": point_equal_by_split,
        "all_restricted_radii_lte_source": all_lte,
        "at_least_one_radius_strictly_reduced": strict,
        "strict_subset_verified": (
            centers_equal and grouping_keys_equal and all_lte and strict
        ),
        "source_fresh_qualification_passed": source_passed,
        "restricted_fresh_qualification_passed": restricted_passed,
    }
    certificate = {
        "certificate_kind": (
            "FINITE_INTERVAL_NESTING_IMPLIES_RESTRICTED_COVERAGE_SUBSET"
        ),
        "checked_row_count": checked_rows,
        "checked_context_scope_pair_count": len(contexts) * len(scopes),
        "all_restricted_radii_lte_source": all_lte,
        "every_restricted_pass_implies_source_pass": implication,
        "restricted_pass_source_fail_observed": False,
        "verified": all_lte and implication,
    }
    return source_report, restricted_report, tradeoff, certificate


def _source_summary(
    report: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "verification_status": receipt.get("status"),
        "contract_id": report.get("contract_id"),
        "contract_digest": report.get("contract_digest"),
        "report_digest": report.get("report_digest"),
        "restriction_id": report.get("restriction_id"),
        "disposition": report.get("disposition"),
        "source_probe_expanded_shadow_theory_state_digest": report.get(
            "source_probe_expanded_shadow_theory_state_digest"
        ),
        "restricted_shadow_theory_state_digest": report.get(
            "restricted_shadow_theory_state_digest"
        ),
        "adoption_status": report.get("adoption_status"),
    }


def _state_selection(
    *,
    disposition: str,
    expected_epoch: str,
    source_state: Mapping[str, Any],
    source_state_digest: str,
    restricted_state: Mapping[str, Any],
    restricted_state_digest: str,
) -> dict[str, Any] | None:
    if disposition == SELECTION["retain_restricted_status"]:
        status = "RESTRICTED_SHADOW_SELECTED"
        kind = "RESTRICTED_SHADOW"
        target = _copy(restricted_state)
        target_digest = restricted_state_digest
    elif disposition == SELECTION["source_rollback_target_status"]:
        status = "SOURCE_SHADOW_ROLLBACK_TARGET_SELECTED"
        kind = "SOURCE_PROBE_EXPANDED_SHADOW"
        target = _copy(source_state)
        target_digest = source_state_digest
    elif disposition == SELECTION["both_failed_status"]:
        status = "NO_QUALIFIED_SHADOW_SELECTED"
        kind = None
        target = None
        target_digest = None
    else:
        return None
    byte_equal = None
    if target is not None:
        expected_target = restricted_state if kind == "RESTRICTED_SHADOW" else source_state
        byte_equal = (
            canonical_json_bytes(target) == canonical_json_bytes(expected_target)
            and _digest(target) == target_digest
        )
        if not byte_equal:
            raise ShadowPostRestrictionAdjudicationValidationError(
                "selected shadow target is not exact verified source bytes"
            )
    return {
        "selection_status": status,
        "selected_target_kind": kind,
        "selected_shadow_theory_state": target,
        "selected_shadow_theory_state_digest": target_digest,
        "byte_equal_to_verified_target": byte_equal,
        "qualification_evaluator_epoch": expected_epoch,
        "qualification_recorded_in_receipt_only": True,
    }


def _rollback_adjudication(
    *, disposition: str, source_state_digest: str
) -> dict[str, Any]:
    selected = disposition == SELECTION["source_rollback_target_status"]
    if selected:
        status = "SOURCE_SHADOW_TARGET_SELECTED_NOT_EXECUTED"
    elif disposition == SELECTION["retain_restricted_status"]:
        status = "NOT_APPLICABLE_RESTRICTED_SHADOW_RETAINED"
    elif disposition == SELECTION["both_failed_status"]:
        status = "NO_QUALIFIED_ROLLBACK_TARGET"
    else:
        status = "NOT_EVALUATED"
    return {
        "adjudication_status": status,
        "rollback_target_selected": selected,
        "rollback_target_state_digest": source_state_digest if selected else None,
        "rollback_execution_status": "NOT_PERFORMED",
        "source_state_mutated": False,
        "restricted_state_mutated": False,
        "ambient_pointer_written": False,
    }


def _record_lifecycle_extension(
    *,
    restriction_report: Mapping[str, Any],
    evidence_binding: Mapping[str, Any],
    evaluator_epoch: str,
    records_consumed: bool,
) -> dict[str, Any]:
    source = _mapping(
        restriction_report.get("record_lifecycle_extension"),
        "source record_lifecycle_extension",
    )
    inherited_keys = (
        "competition_records",
        "qualification_records",
        "failure_boundary_probe_records",
        "restriction_competition_records",
    )
    result = {key: _copy(_mapping(source.get(key), f"source {key}")) for key in inherited_keys}
    result["post_restriction_adjudication_records"] = {
        "evidence_digests": {
            "holdout": evidence_binding["holdout_evidence_digest"],
            "stress": evidence_binding["stress_evidence_digest"],
        },
        "observation_id_digest": evidence_binding["new_observation_id_digest"],
        "observation_count": evidence_binding["new_observation_count"],
        "evaluator_epoch": evaluator_epoch,
        "role": (
            RECORD_LIFECYCLE_POLICY["post_restriction_record_role"]
            if records_consumed
            else "NOT_CONSUMED_POST_RESTRICTION_PRECONDITION_BLOCKED"
        ),
        "eligible_for_future_scoring": False,
    }
    result["future_scoring_policy"] = {
        "new_unconsumed_evidence_required": True,
        "reuse_competition_records_allowed": False,
        "reuse_consumed_qualification_records_allowed": False,
        "reuse_consumed_failure_boundary_probe_records_allowed": False,
        "reuse_consumed_restriction_records_allowed": False,
        "reuse_consumed_post_restriction_records_allowed": False,
        "cross_epoch_pooling_allowed": False,
    }
    result["logical_selective_erasure_applied"] = True
    result["physical_erasure"] = "NOT_PERFORMED"
    return result


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


def qualify_adjudicate_and_route_shadow_post_restriction(
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
    input_artifacts: Mapping[str, Any] | None = None,
) -> ShadowPostRestrictionAdjudicationResult:
    """Freshly qualify two verified interval shadows and route fail closed."""

    normalized_contract = _validate_contract(adjudication_contract)
    expected_contract_digest = _require_digest(
        expected_adjudication_contract_digest,
        "expected_adjudication_contract_digest",
    )
    contract_digest = _digest(normalized_contract)
    if contract_digest != expected_contract_digest:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "adjudication contract digest differs from independent expectation"
        )
    expected_input_digest = _require_digest(
        expected_adjudication_input_digest,
        "expected_adjudication_input_digest",
    )
    if _digest(adjudication_input) != expected_input_digest:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "adjudication input digest differs from independent expectation"
        )
    if input_artifacts is not None and not isinstance(input_artifacts, Mapping):
        raise ShadowPostRestrictionAdjudicationValidationError(
            "input_artifacts must be an object or null"
        )
    artifacts = _copy(input_artifacts) if input_artifacts is not None else None
    _reject_observed_values(artifacts, "input_artifacts")

    expected_source_contract = _require_digest(
        expected_restriction_contract_digest,
        "expected_restriction_contract_digest",
    )
    if expected_source_contract != SOURCE_RESTRICTION_CONTRACT_DIGEST:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "expected restriction contract digest is not the pinned V1 source"
        )
    try:
        source_receipt = verify_shadow_robust_interval_restriction(
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
            expected_restriction_contract_digest=expected_source_contract,
            expected_restriction_report_digest=expected_restriction_report_digest,
            expected_restriction_input_artifacts=(
                expected_restriction_input_artifacts
            ),
        )
    except (
        ShadowRobustIntervalRestrictionValidationError,
        CompetitionValidationError,
        KeyError,
        TypeError,
    ) as exc:
        raise ShadowPostRestrictionAdjudicationValidationError(
            f"source restriction verification failed: {exc}"
        ) from exc

    restriction = _copy(_mapping(restriction_report, "restriction_report"))
    receipt = _copy(_mapping(source_receipt, "source_restriction_receipt"))
    if restriction.get("contract_id") != SOURCE_RESTRICTION_CONTRACT_ID:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "source restriction contract_id is not supported"
        )
    if restriction.get("contract_digest") != SOURCE_RESTRICTION_CONTRACT_DIGEST:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "source restriction contract digest is not pinned V1"
        )
    if restriction.get("schema_version") != SOURCE_RESTRICTION_REPORT_SCHEMA_VERSION:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "source restriction report schema is not supported"
        )
    if restriction.get("disposition") != SELECTION["source_materialized_status"]:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "post-restriction adjudication requires a materialized restriction"
        )
    if restriction.get("adoption_status") != ADOPTION_STATUS:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "source restriction crossed the shadow-only adoption boundary"
        )

    source_state = _copy(
        _mapping(
            _mapping(probe_report, "probe_report").get(
                "probe_expanded_shadow_theory_state"
            ),
            "source probe-expanded shadow state",
        )
    )
    source_state_digest = _require_digest(
        restriction.get("source_probe_expanded_shadow_theory_state_digest"),
        "source probe-expanded state digest",
    )
    if _digest(source_state) != source_state_digest:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "source probe-expanded state digest is inconsistent"
        )
    restricted_state = _copy(
        _mapping(
            restriction.get("restricted_shadow_theory_state"),
            "restricted shadow state",
        )
    )
    restricted_state_digest = _require_digest(
        restriction.get("restricted_shadow_theory_state_digest"),
        "restricted shadow state digest",
    )
    if _digest(restricted_state) != restricted_state_digest:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "restricted shadow state digest is inconsistent"
        )
    original_source_bytes = canonical_json_bytes(source_state)
    original_restricted_bytes = canonical_json_bytes(restricted_state)
    if source_state.get("fixed_anchor") != restricted_state.get("fixed_anchor"):
        raise ShadowPostRestrictionAdjudicationValidationError(
            "source and restricted fixed anchors differ"
        )
    fixed_anchor = _string(source_state.get("fixed_anchor"), "source fixed_anchor")

    normalized_input = _normalize_input(
        adjudication_input,
        restriction_report=restriction,
        source_state=source_state,
        restricted_state=restricted_state,
        competition_input=competition_input,
        qualification_input=qualification_input,
        probe_input=probe_input,
        restriction_input=restriction_input,
    )
    semantic_input_digest = _digest(normalized_input["public"])
    adjudication_id = normalized_input["public"]["adjudication_id"]
    epoch_payload, expected_epoch = _epoch_components(
        restriction_contract_digest=restriction["contract_digest"],
        restriction_report_digest=restriction["report_digest"],
        restricted_shadow_theory_state_digest=restricted_state_digest,
        source_probe_expanded_shadow_theory_state_digest=source_state_digest,
        fixed_anchor=fixed_anchor,
        adjudication_contract=normalized_contract,
    )
    lifecycle_source = _mapping(
        restriction.get("record_lifecycle_extension"),
        "restriction record_lifecycle_extension",
    )
    prior_epochs = {
        _mapping(lifecycle_source.get(key), f"source {key}").get("evaluator_epoch")
        for key in (
            "competition_records",
            "qualification_records",
            "failure_boundary_probe_records",
            "restriction_competition_records",
        )
    }
    prior_epochs.discard(None)
    if expected_epoch in prior_epochs:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "derived post-restriction epoch is not fresh from all prior epochs"
        )
    evaluator_definition = {
        "evaluator_epoch": expected_epoch,
        "fixed_anchor": fixed_anchor,
        "epoch_derivation_kind": (
            "CONTENT_ADDRESSED_LOCAL_POST_RESTRICTION_ADJUDICATION_EPOCH"
        ),
        "fixed_probe_registry_digest": _digest(epoch_payload["fixed_probe_registry"]),
    }
    evaluator_binding = _evaluator_binding(
        normalized_input, expected_epoch, fixed_anchor
    )
    evidence_binding = _evidence_binding(normalized_input)
    probe_replay = _probe_registry_replay(source_state, restricted_state)

    source_results: dict[str, Any] | None = None
    restricted_results: dict[str, Any] | None = None
    tradeoff: dict[str, Any] | None = None
    monotonicity: dict[str, Any] | None = None
    if not evaluator_binding["comparable"]:
        disposition = SELECTION["incomparable_status"]
        cycle_route = SELECTION["incomparable_route"]
    elif not evidence_binding["complete_evidence"]:
        disposition = SELECTION["needs_evidence_status"]
        cycle_route = SELECTION["needs_evidence_route"]
    else:
        source_results, restricted_results, tradeoff, monotonicity = _compare_models(
            source_state=source_state,
            restricted_state=restricted_state,
            contexts=normalized_input["parent_contexts"],
            scopes=normalized_input["parent_scopes"],
            evidence=normalized_input["evidence"],
        )
        restricted_passed = restricted_results["fresh_qualification_passed"]
        source_passed = source_results["fresh_qualification_passed"]
        if restricted_passed and source_passed:
            disposition = SELECTION["retain_restricted_status"]
            cycle_route = SELECTION["retain_restricted_route"]
        elif (not restricted_passed) and source_passed:
            disposition = SELECTION["source_rollback_target_status"]
            cycle_route = SELECTION["source_rollback_target_route"]
        elif (not restricted_passed) and (not source_passed):
            disposition = SELECTION["both_failed_status"]
            cycle_route = SELECTION["both_failed_route"]
        else:
            raise ShadowPostRestrictionAdjudicationValidationError(
                "restricted-pass/source-fail is impossible by finite interval nesting"
            )

    selection = _state_selection(
        disposition=disposition,
        expected_epoch=expected_epoch,
        source_state=source_state,
        source_state_digest=source_state_digest,
        restricted_state=restricted_state,
        restricted_state_digest=restricted_state_digest,
    )
    rollback = _rollback_adjudication(
        disposition=disposition, source_state_digest=source_state_digest
    )
    records_consumed = (
        evaluator_binding["comparable"] and evidence_binding["complete_evidence"]
    )
    lifecycle = _record_lifecycle_extension(
        restriction_report=restriction,
        evidence_binding=evidence_binding,
        evaluator_epoch=expected_epoch,
        records_consumed=records_consumed,
    )
    source_summary = _source_summary(restriction, receipt)

    events = [
        _audit_event(
            0,
            "SOURCE_ROBUST_INTERVAL_RESTRICTION_VERIFIED",
            GENESIS_DIGEST,
            {
                "verification_status": source_summary["verification_status"],
                "source_restriction_report_digest": source_summary["report_digest"],
                "source_disposition": source_summary["disposition"],
            },
        )
    ]
    events.append(
        _audit_event(
            1,
            "POST_RESTRICTION_EPOCH_AND_RECORD_ISOLATION_BOUND",
            events[-1]["event_digest"],
            {
                "expected_evaluator_epoch": expected_epoch,
                "new_observation_id_digest": evidence_binding[
                    "new_observation_id_digest"
                ],
                "cross_epoch_pooling": False,
            },
        )
    )
    events.append(
        _audit_event(
            2,
            "SOURCE_AND_RESTRICTED_TWO_PROBE_REGISTRIES_REPLAYED",
            events[-1]["event_digest"],
            {
                "probe_registry_replay_digest": _digest(probe_replay),
                "source_probe_results_digest": _digest(source_results),
                "restricted_probe_results_digest": _digest(restricted_results),
                "monotonicity_certificate_digest": _digest(monotonicity),
            },
        )
    )
    events.append(
        _audit_event(
            3,
            "SHADOW_TARGET_ADJUDICATED_AND_CYCLE_ROUTED_FAIL_CLOSED",
            events[-1]["event_digest"],
            {
                "disposition": disposition,
                "selected_shadow_theory_state_digest": selection[
                    "selected_shadow_theory_state_digest"
                ] if selection is not None else None,
                "cycle_route": cycle_route,
                "required_bridge": LOOP_COMPATIBILITY_BOUNDARY["required_bridge"],
                "rollback_executed": False,
                "language_expansion_executed": False,
                "adoption_status": ADOPTION_STATUS,
                "current_status": CURRENT_STATUS,
            },
        )
    )

    if canonical_json_bytes(source_state) != original_source_bytes:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "source probe-expanded state was mutated during adjudication"
        )
    if canonical_json_bytes(restricted_state) != original_restricted_bytes:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "restricted state was mutated during adjudication"
        )
    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "contract_digest": contract_digest,
        "adjudication_input_digest": semantic_input_digest,
        "adjudication_id": adjudication_id,
        "source_restriction": source_summary,
        "evaluator_definition": evaluator_definition,
        "evaluator_binding": evaluator_binding,
        "evidence_binding": evidence_binding,
        "probe_registry_replay": probe_replay,
        "source_probe_results": source_results,
        "restricted_probe_results": restricted_results,
        "finite_interval_tradeoff": tradeoff,
        "monotonicity_certificate": monotonicity,
        "disposition": disposition,
        "shadow_state_selection": selection,
        "rollback_adjudication": rollback,
        "cycle_route": cycle_route,
        "loop_compatibility_boundary": _copy(
            normalized_contract["loop_compatibility_boundary"]
        ),
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
    _exact_keys(report, REPORT_FIELDS, "adjudication_report")
    _reject_observed_values(report, "adjudication_report")
    return ShadowPostRestrictionAdjudicationResult(report=report)


def verify_shadow_post_restriction_adjudication(
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
) -> dict[str, Any]:
    """Exact-replay an adjudication report against independent anchors."""

    expected_report_digest = _require_digest(
        expected_adjudication_report_digest,
        "expected_adjudication_report_digest",
    )
    supplied = _copy(_mapping(adjudication_report, "adjudication_report"))
    _exact_keys(supplied, REPORT_FIELDS, "adjudication_report")
    _reject_observed_values(supplied, "adjudication_report")
    if expected_adjudication_input_artifacts is not None and not isinstance(
        expected_adjudication_input_artifacts, Mapping
    ):
        raise ShadowPostRestrictionAdjudicationValidationError(
            "expected_adjudication_input_artifacts must be an object or null"
        )
    expected_artifacts = (
        _copy(expected_adjudication_input_artifacts)
        if expected_adjudication_input_artifacts is not None
        else None
    )
    _reject_observed_values(
        expected_artifacts, "expected_adjudication_input_artifacts"
    )
    if supplied.get("input_artifacts") != expected_artifacts:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "adjudication input artifacts differ from independent expectation"
        )
    fresh = qualify_adjudicate_and_route_shadow_post_restriction(
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
        input_artifacts=expected_artifacts,
    ).to_dict()
    if fresh["report_digest"] != expected_report_digest:
        raise ShadowPostRestrictionAdjudicationValidationError(
            "replayed adjudication report digest differs from expectation"
        )
    if canonical_json_bytes(supplied) != canonical_json_bytes(fresh):
        raise ShadowPostRestrictionAdjudicationValidationError(
            "supplied adjudication report differs from exact replay"
        )
    return {
        "status": "VERIFIED_" + fresh["disposition"],
        "disposition": fresh["disposition"],
        "report_digest": fresh["report_digest"],
        "contract_digest": fresh["contract_digest"],
        "adjudication_id": fresh["adjudication_id"],
        "source_restriction_report_digest": fresh["source_restriction"][
            "report_digest"
        ],
        "source_probe_expanded_shadow_theory_state_digest": fresh[
            "source_restriction"
        ]["source_probe_expanded_shadow_theory_state_digest"],
        "restricted_shadow_theory_state_digest": fresh["source_restriction"][
            "restricted_shadow_theory_state_digest"
        ],
        "selected_shadow_theory_state_digest": (
            None
            if fresh["shadow_state_selection"] is None
            else fresh["shadow_state_selection"][
                "selected_shadow_theory_state_digest"
            ]
        ),
        "restricted_shadow_qualified": (
            fresh["disposition"] == SELECTION["retain_restricted_status"]
        ),
        "rollback_target_selected": fresh["rollback_adjudication"][
            "rollback_target_selected"
        ],
        "cycle_route": fresh["cycle_route"],
        "required_bridge": fresh["loop_compatibility_boundary"][
            "required_bridge"
        ],
        "adoption_eligibility": fresh["adoption_eligibility"],
        "adoption_status": fresh["adoption_status"],
        "promotion_status": fresh["promotion_status"],
        "current_status": fresh["current_status"],
    }
