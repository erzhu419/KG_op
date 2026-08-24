"""Compete fixed strict contractions of one robust shadow interval theory.

This additive V1 core exact-replays the preceding failure-boundary probe,
constructs four evidence-independent uniform-radius contractions, and scores
them on caller-supplied rows from one new content-derived evaluator epoch.  A
positive result is still a shadow state: no source state is invalidated or
mutated, no rollback or ambient restriction is executed, and no theory is
adopted, promoted, or made current.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from performance.shadow_child_failure_boundary_probe import (
    ShadowChildFailureBoundaryProbeValidationError,
    verify_shadow_child_failure_boundary_probe,
)
from performance.theory_operation_competition import (
    CompetitionValidationError,
    canonical_json_bytes,
)


CONTRACT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-robust-interval-restriction-contract/1"
)
CONTRACT_ID = "shadow_robust_interval_restriction_v1"
SOURCE_PROBE_CONTRACT_ID = "shadow_child_failure_boundary_probe_v1"
SOURCE_PROBE_CONTRACT_DIGEST = (
    "sha256:fdc92e276f7d8cb0c1ab6fd097242932851da04e1f97888d3f9597bfb0f726e0"
)
SOURCE_PROBE_REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-child-failure-boundary-probe-report/1"
)
INPUT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-robust-interval-restriction-input/1"
)
REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-robust-interval-restriction-report/1"
)
STATE_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-robust-interval-restricted-theory-state/1"
)
GENESIS_DIGEST = "sha256:" + "0" * 64

ADOPTION_ELIGIBILITY = "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED"
ADOPTION_STATUS = "NOT_ADOPTED_SHADOW_ONLY"
PROMOTION_STATUS = "NOT_PROMOTED"
CURRENT_STATUS = "NOT_CURRENT"
OPERATION_KIND = "restrict"
RESTRICTION_KIND = "UNIFORM_RADIUS_CONTRACTION"

MULTIPLIER_REGISTRY = [
    {
        "candidate_id": "uniform_radius_1_over_4",
        "numerator": 1,
        "denominator": 4,
        "radius_multiplier": 0.25,
    },
    {
        "candidate_id": "uniform_radius_1_over_2",
        "numerator": 1,
        "denominator": 2,
        "radius_multiplier": 0.5,
    },
    {
        "candidate_id": "uniform_radius_3_over_4",
        "numerator": 3,
        "denominator": 4,
        "radius_multiplier": 0.75,
    },
    {
        "candidate_id": "uniform_radius_9_over_10",
        "numerator": 9,
        "denominator": 10,
        "radius_multiplier": 0.9,
    },
]

EVIDENCE_POLICY = {
    "required_splits": ["calibration", "holdout", "stress"],
    "require_complete_parent_context_scope_pairs_per_split": True,
    "require_exactly_one_row_per_context_scope_pair_per_split": True,
    "require_unique_new_observation_ids": True,
    "require_disjoint_from_competition_ids": True,
    "require_disjoint_from_qualification_ids": True,
    "require_disjoint_from_failure_boundary_probe_ids": True,
    "require_exact_derived_epoch": True,
    "require_exact_inherited_fixed_anchor": True,
    "forbid_cross_epoch_pooling": True,
}

THRESHOLDS = {
    "numeric_epsilon": 1e-12,
    "min_calibration_normalized_signed_margin": 0.0,
    "max_holdout_boundary_violation_rate": 0.0,
    "max_stress_boundary_violation_rate": 0.0,
}

SELECTION = {
    "materialized_status": "MATERIALIZED_SHADOW_ROBUST_INTERVAL_RESTRICTION",
    "no_calibration_status": (
        "NO_CALIBRATION_ADMISSIBLE_STRICT_INTERVAL_RESTRICTION"
    ),
    "no_validated_status": "NO_VALIDATED_STRICT_INTERVAL_RESTRICTION",
    "needs_evidence_status": "RESTRICTION_NEEDS_NEW_EVIDENCE",
    "incomparable_status": "RESTRICTION_INCOMPARABLE_EVALUATOR_EPOCH",
    "source_counterexample_status": (
        "RESTRICTION_BLOCKED_SOURCE_BOUNDARY_COUNTEREXAMPLE"
    ),
    "source_unresolved_status": "RESTRICTION_BLOCKED_SOURCE_PROBE_UNRESOLVED",
    "non_robust_status": "RESTRICTION_NOT_APPLICABLE_NON_ROBUST_SOURCE",
    "source_no_counterexample_status": (
        "EXPANDED_PROBE_NO_BOUNDARY_COUNTEREXAMPLE_ON_SUPPLIED_EVIDENCE"
    ),
    "source_counterexample_disposition": (
        "EXPANDED_PROBE_BOUNDARY_COUNTEREXAMPLE_FOUND_SHADOW_ONLY"
    ),
    "applicable_transition_kind": "ROBUST_INTERVAL_EXPANSION",
    "operation_kind": OPERATION_KIND,
    "restriction_kind": RESTRICTION_KIND,
    "candidate_order": "ASCENDING_RADIUS_MULTIPLIER",
    "adoption_eligibility": ADOPTION_ELIGIBILITY,
    "adoption_status": ADOPTION_STATUS,
    "promotion_status": PROMOTION_STATUS,
    "current_status": CURRENT_STATUS,
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
    "all_record_classes_eligible_for_future_scoring": False,
    "future_scoring_requires_new_unconsumed_evidence": True,
    "cross_epoch_pooling_allowed": False,
    "logical_selective_erasure_applied": True,
    "physical_erasure": "NOT_PERFORMED",
}

AUTHORITY_BOUNDARY = {
    "scope": "LOCAL_SHADOW_ROBUST_INTERVAL_RESTRICTION_ONLY",
    "source_state_mutation": False,
    "source_child_invalidation": False,
    "ambient_restriction_execution": False,
    "rollback_execution": False,
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
    "robust_finite_interval_restriction_only",
    "no_generic_restriction_engine",
    "no_rigid_body_markov_or_independence_assumption",
    "no_scope_restriction",
    "no_quotient_restriction",
    "no_new_probe",
    "q_registry_copied_not_requalified",
    "v_registry_unchanged",
    "no_language_or_predicate_invention",
    "no_external_probe_acquisition",
    "caller_supplied_static_rows_only",
    "local_epoch_is_not_external_attestation",
    "fresh_evidence_pass_is_not_global_preservation",
    "interval_width_reduction_is_not_nominal_utility_or_domain_safety",
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
        "restriction_input_digest",
        "restriction_id",
        "source_failure_boundary_probe",
        "source_probe_expanded_shadow_theory_state_digest",
        "source_transition_kind",
        "operation_kind",
        "restriction_kind",
        "evaluator_definition",
        "evaluator_binding",
        "evidence_binding",
        "candidate_registry",
        "candidate_competition",
        "selected_candidate",
        "fresh_validation",
        "disposition",
        "restricted_shadow_theory_state",
        "restricted_shadow_theory_state_digest",
        "restriction_certificate",
        "rollback_boundary",
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

STATE_FIELDS = frozenset(
    {
        "schema_version",
        "theory_id",
        "task_id",
        "source_probe_expanded_theory_state_digest",
        "source_failure_boundary_probe_report_digest",
        "operation_kind",
        "restriction_kind",
        "evaluator_epoch",
        "evaluator_status",
        "fixed_anchor",
        "object_space",
        "model_class",
        "probe_ids",
        "violation_functionals",
        "scope_ids",
        "removable_feature_ids",
        "evidence_reuse_policy",
        "operational_probe_status",
        "restriction_lineage",
        "adoption_status",
        "current_status",
    }
)

CERTIFICATE_FIELDS = frozenset(
    {
        "certificate_kind",
        "source_model_class_digest",
        "restricted_model_class_digest",
        "radius_multiplier",
        "checked_radius_group_count",
        "checked_context_scope_pair_count",
        "centers_byte_equal",
        "grouping_and_group_keys_byte_equal",
        "all_restricted_radii_finite_nonnegative",
        "all_restricted_radii_lte_source",
        "at_least_one_radius_strictly_reduced",
        "strict_subset_verified",
    }
)

ROLLBACK_FIELDS = frozenset(
    {
        "method",
        "source_probe_expanded_state_digest",
        "source_materialized_child_digest",
        "original_parent_theory_state_digest",
        "source_model_class_digest",
        "restricted_model_class_digest",
        "rollback_execution_status",
    }
)


class ShadowRobustIntervalRestrictionValidationError(ValueError):
    """Raised when a frozen restriction contract, source, input, or report fails."""


class ShadowRobustIntervalRestrictionDisposition(str, Enum):
    MATERIALIZED_SHADOW_ROBUST_INTERVAL_RESTRICTION = (
        "MATERIALIZED_SHADOW_ROBUST_INTERVAL_RESTRICTION"
    )
    NO_CALIBRATION_ADMISSIBLE_STRICT_INTERVAL_RESTRICTION = (
        "NO_CALIBRATION_ADMISSIBLE_STRICT_INTERVAL_RESTRICTION"
    )
    NO_VALIDATED_STRICT_INTERVAL_RESTRICTION = (
        "NO_VALIDATED_STRICT_INTERVAL_RESTRICTION"
    )
    RESTRICTION_NEEDS_NEW_EVIDENCE = "RESTRICTION_NEEDS_NEW_EVIDENCE"
    RESTRICTION_INCOMPARABLE_EVALUATOR_EPOCH = (
        "RESTRICTION_INCOMPARABLE_EVALUATOR_EPOCH"
    )
    RESTRICTION_BLOCKED_SOURCE_BOUNDARY_COUNTEREXAMPLE = (
        "RESTRICTION_BLOCKED_SOURCE_BOUNDARY_COUNTEREXAMPLE"
    )
    RESTRICTION_BLOCKED_SOURCE_PROBE_UNRESOLVED = (
        "RESTRICTION_BLOCKED_SOURCE_PROBE_UNRESOLVED"
    )
    RESTRICTION_NOT_APPLICABLE_NON_ROBUST_SOURCE = (
        "RESTRICTION_NOT_APPLICABLE_NON_ROBUST_SOURCE"
    )


@dataclass(frozen=True)
class ShadowRobustIntervalRestrictionResult:
    report: Mapping[str, Any]

    @property
    def disposition(self) -> str:
        return str(self.report["disposition"])

    @property
    def report_digest(self) -> str:
        return str(self.report["report_digest"])

    @property
    def restriction_materialized(self) -> bool:
        return self.report["restricted_shadow_theory_state"] is not None

    @property
    def selected_radius_multiplier(self) -> float | None:
        selected = self.report["selected_candidate"]
        return None if selected is None else float(selected["radius_multiplier"])

    def to_dict(self) -> dict[str, Any]:
        return _copy(self.report)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShadowRobustIntervalRestrictionValidationError(
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
        raise ShadowRobustIntervalRestrictionValidationError(
            f"{label} keys differ; missing={missing}, extra={extra}"
        )


def _string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ShadowRobustIntervalRestrictionValidationError(
            f"{label} must be a non-empty string"
        )
    return value


def _finite_number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ShadowRobustIntervalRestrictionValidationError(
            f"{label} must be a finite number"
        )
    return float(value)


def _json_scalar(value: Any, label: str) -> str | int | float | bool:
    if value is None or type(value) not in (str, int, float, bool):
        raise ShadowRobustIntervalRestrictionValidationError(
            f"{label} must be a non-null finite JSON scalar"
        )
    if type(value) is float and not math.isfinite(value):
        raise ShadowRobustIntervalRestrictionValidationError(
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
        raise ShadowRobustIntervalRestrictionValidationError(
            f"value is not detached finite canonical JSON: {exc}"
        ) from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_digest(value: Any, label: str) -> str:
    digest = _string(value, label)
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise ShadowRobustIntervalRestrictionValidationError(
            f"{label} must be sha256:<64hex>"
        )
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise ShadowRobustIntervalRestrictionValidationError(
            f"{label} is not hexadecimal"
        ) from exc
    return digest


def _optional_digest(value: Any, label: str) -> str | None:
    return None if value is None else _require_digest(value, label)


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ShadowRobustIntervalRestrictionValidationError(
            f"{label} differs from frozen robust-interval restriction V1"
        )


def _reject_observed_values(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        if "observed_value" in value:
            raise ShadowRobustIntervalRestrictionValidationError(
                f"{label} must not embed observation values"
            )
        for key, item in value.items():
            _reject_observed_values(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_observed_values(item, f"{label}[{index}]")


def _validate_contract(contract_value: Any) -> dict[str, Any]:
    contract = _mapping(contract_value, "restriction_contract")
    _exact_keys(
        contract,
        {
            "schema_version",
            "contract_id",
            "source_probe_contract_id",
            "source_probe_contract_digest",
            "source_probe_report_schema_version",
            "input_schema_version",
            "report_schema_version",
            "state_schema_version",
            "multiplier_registry",
            "evidence_policy",
            "thresholds",
            "selection",
            "record_lifecycle_policy",
            "authority_boundary",
            "nonclaims",
        },
        "restriction_contract",
    )
    frozen = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "source_probe_contract_id": SOURCE_PROBE_CONTRACT_ID,
        "source_probe_contract_digest": SOURCE_PROBE_CONTRACT_DIGEST,
        "source_probe_report_schema_version": SOURCE_PROBE_REPORT_SCHEMA_VERSION,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "state_schema_version": STATE_SCHEMA_VERSION,
    }
    for key, expected in frozen.items():
        _require_equal(contract[key], expected, key)
    for key, expected in (
        ("multiplier_registry", MULTIPLIER_REGISTRY),
        ("evidence_policy", EVIDENCE_POLICY),
        ("thresholds", THRESHOLDS),
        ("selection", SELECTION),
        ("record_lifecycle_policy", RECORD_LIFECYCLE_POLICY),
        ("authority_boundary", AUTHORITY_BOUNDARY),
    ):
        _require_equal(_copy(contract[key]), expected, key)
    nonclaims = contract["nonclaims"]
    if not isinstance(nonclaims, list) or tuple(nonclaims) != MANDATORY_NONCLAIMS:
        raise ShadowRobustIntervalRestrictionValidationError(
            "nonclaims differ from the exact mandatory V1 list"
        )
    return _copy(contract)


def validate_shadow_robust_interval_restriction_contract(
    contract_value: Any,
) -> dict[str, Any]:
    """Validate and detach the frozen robust interval restriction contract."""

    return _validate_contract(contract_value)


def _epoch_components(
    *,
    probe_contract_digest: str,
    probe_report_digest: str,
    probe_expanded_shadow_theory_state_digest: str,
    fixed_anchor: str,
    restriction_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    contract = _validate_contract(restriction_contract)
    source_contract = _require_digest(
        probe_contract_digest, "probe_contract_digest"
    )
    if source_contract != SOURCE_PROBE_CONTRACT_DIGEST:
        raise ShadowRobustIntervalRestrictionValidationError(
            "probe contract digest is not the pinned V1 source"
        )
    payload = {
        "probe_contract_digest": source_contract,
        "probe_report_digest": _require_digest(
            probe_report_digest, "probe_report_digest"
        ),
        "probe_expanded_shadow_theory_state_digest": _require_digest(
            probe_expanded_shadow_theory_state_digest,
            "probe_expanded_shadow_theory_state_digest",
        ),
        "fixed_anchor": _string(fixed_anchor, "fixed_anchor"),
        "restriction_contract_digest": _digest(contract),
        "fixed_multiplier_registry": _copy(contract["multiplier_registry"]),
    }
    epoch = "shadow-robust-restriction-epoch:" + _digest(payload)[7:]
    return payload, epoch


def derive_shadow_robust_interval_restriction_epoch(
    *,
    probe_contract_digest: str,
    probe_report_digest: str,
    probe_expanded_shadow_theory_state_digest: str,
    fixed_anchor: str,
    restriction_contract: Mapping[str, Any],
) -> str:
    """Derive the content identity of the local restriction epoch."""

    return _epoch_components(
        probe_contract_digest=probe_contract_digest,
        probe_report_digest=probe_report_digest,
        probe_expanded_shadow_theory_state_digest=(
            probe_expanded_shadow_theory_state_digest
        ),
        fixed_anchor=fixed_anchor,
        restriction_contract=restriction_contract,
    )[1]


def _observation_ids(
    input_value: Mapping[str, Any], splits: Sequence[str], label: str
) -> list[str]:
    evidence = _mapping(input_value.get("evidence"), f"{label} evidence")
    _exact_keys(evidence, set(splits), f"{label} evidence")
    result: list[str] = []
    for split in splits:
        rows = evidence[split]
        if not isinstance(rows, list):
            raise ShadowRobustIntervalRestrictionValidationError(
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
        raise ShadowRobustIntervalRestrictionValidationError(
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
        raise ShadowRobustIntervalRestrictionValidationError(
            f"{label} is not a registered parent context"
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
        raise ShadowRobustIntervalRestrictionValidationError(
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
            raise ShadowRobustIntervalRestrictionValidationError(
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
        raise ShadowRobustIntervalRestrictionValidationError(
            f"evidence.{split} observation IDs are duplicated"
        )
    return rows


def _normalize_input(
    input_value: Any,
    *,
    probe_report: Mapping[str, Any],
    transition_report: Mapping[str, Any],
    competition_input: Mapping[str, Any],
    qualification_input: Mapping[str, Any],
    probe_input: Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(input_value, "restriction_input")
    _exact_keys(
        value,
        {
            "schema_version",
            "restriction_id",
            "source_failure_boundary_probe",
            "evaluator",
            "prior_record_exclusion",
            "evidence",
        },
        "restriction_input",
    )
    _require_equal(value["schema_version"], INPUT_SCHEMA_VERSION, "input schema")

    source = _mapping(
        value["source_failure_boundary_probe"],
        "source_failure_boundary_probe",
    )
    _exact_keys(
        source,
        {
            "probe_contract_digest",
            "probe_report_digest",
            "probe_expansion_id",
            "probe_expanded_shadow_theory_state_digest",
        },
        "source_failure_boundary_probe",
    )
    expected_source = {
        "probe_contract_digest": probe_report["contract_digest"],
        "probe_report_digest": probe_report["report_digest"],
        "probe_expansion_id": probe_report["probe_expansion_id"],
        "probe_expanded_shadow_theory_state_digest": probe_report[
            "probe_expanded_shadow_theory_state_digest"
        ],
    }
    _require_equal(_copy(source), expected_source, "source_failure_boundary_probe")

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
    lifecycle = _mapping(
        probe_report.get("record_lifecycle_extension"),
        "source record_lifecycle_extension",
    )
    for key, ids in (
        ("competition_records", competition_ids),
        ("qualification_records", qualification_ids),
        ("new_probe_records", boundary_ids),
    ):
        record = _mapping(lifecycle.get(key), f"source {key}")
        if record.get("observation_id_digest") != _digest(ids):
            raise ShadowRobustIntervalRestrictionValidationError(
                f"source {key} observation commitment is inconsistent"
            )

    exclusion = _mapping(
        value["prior_record_exclusion"], "prior_record_exclusion"
    )
    _exact_keys(
        exclusion,
        {
            "source_competition_observation_id_digest",
            "consumed_qualification_observation_id_digest",
            "consumed_failure_boundary_observation_id_digest",
        },
        "prior_record_exclusion",
    )
    expected_exclusion = {
        "source_competition_observation_id_digest": _digest(competition_ids),
        "consumed_qualification_observation_id_digest": _digest(
            qualification_ids
        ),
        "consumed_failure_boundary_observation_id_digest": _digest(
            boundary_ids
        ),
    }
    _require_equal(_copy(exclusion), expected_exclusion, "prior_record_exclusion")

    parent = _mapping(
        transition_report.get("parent_theory_state"), "parent_theory_state"
    )
    object_space = _mapping(parent.get("object_space"), "parent object_space")
    feature_ids_value = object_space.get("feature_ids")
    contexts_value = object_space.get("contexts")
    scopes_value = parent.get("scope_ids")
    if (
        not isinstance(feature_ids_value, list)
        or not feature_ids_value
        or not isinstance(contexts_value, list)
        or not contexts_value
        or not isinstance(scopes_value, list)
        or not scopes_value
    ):
        raise ShadowRobustIntervalRestrictionValidationError(
            "verified parent lacks a finite feature/context/scope registry"
        )
    feature_ids = [_string(item, "parent feature_id") for item in feature_ids_value]
    if len(feature_ids) != len(set(feature_ids)):
        raise ShadowRobustIntervalRestrictionValidationError(
            "parent feature IDs are duplicated"
        )
    registered_contexts = {canonical_json_bytes(item) for item in contexts_value}
    if len(registered_contexts) != len(contexts_value):
        raise ShadowRobustIntervalRestrictionValidationError(
            "parent contexts are duplicated"
        )
    registered_scopes = {_string(item, "parent scope_id") for item in scopes_value}
    if len(registered_scopes) != len(scopes_value):
        raise ShadowRobustIntervalRestrictionValidationError(
            "parent scopes are duplicated"
        )

    evidence = _mapping(value["evidence"], "evidence")
    _exact_keys(evidence, {"calibration", "holdout", "stress"}, "evidence")
    normalized_evidence = {
        split: _normalize_rows(
            evidence[split],
            split,
            feature_ids,
            registered_contexts,
            registered_scopes,
        )
        for split in ("calibration", "holdout", "stress")
    }
    new_ids = [
        row["observation_id"]
        for split in ("calibration", "holdout", "stress")
        for row in normalized_evidence[split]
    ]
    if len(new_ids) != len(set(new_ids)):
        raise ShadowRobustIntervalRestrictionValidationError(
            "new observation IDs are duplicated across splits"
        )
    for old_ids, label in (
        (competition_ids, "source competition"),
        (qualification_ids, "consumed qualification"),
        (boundary_ids, "consumed failure-boundary probe"),
    ):
        if set(new_ids) & set(old_ids):
            raise ShadowRobustIntervalRestrictionValidationError(
                f"new observation IDs reuse {label} records"
            )

    public = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "restriction_id": _string(value["restriction_id"], "restriction_id"),
        "source_failure_boundary_probe": expected_source,
        "evaluator": normalized_evaluator,
        "prior_record_exclusion": expected_exclusion,
        "evidence": normalized_evidence,
    }
    return {
        "public": public,
        "evidence": normalized_evidence,
        "new_ids": sorted(new_ids),
        "competition_ids": competition_ids,
        "qualification_ids": qualification_ids,
        "boundary_ids": boundary_ids,
        "parent_contexts": _copy(contexts_value),
        "parent_scopes": sorted(registered_scopes),
    }


def _evaluator_binding(
    normalized_input: Mapping[str, Any],
    expected_epoch: str | None,
    expected_anchor: str,
) -> dict[str, Any]:
    evaluator = normalized_input["public"]["evaluator"]
    rows = [
        row
        for split in ("calibration", "holdout", "stress")
        for row in normalized_input["evidence"][split]
    ]
    supplied_epoch = evaluator["evaluator_epoch"]
    supplied_anchor = evaluator["fixed_anchor"]
    epoch_matches = expected_epoch is not None and supplied_epoch == expected_epoch and all(
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
    duplicate_pair_counts: dict[str, int] = {}
    complete: dict[str, bool] = {}
    for split in ("calibration", "holdout", "stress"):
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
        duplicate_pair_counts[split] = sum(max(0, count - 1) for count in counts.values())
        complete[split] = (
            keys == required_keys
            and all(count == 1 for count in counts.values())
        )
    ids = normalized_input["new_ids"]
    return {
        "calibration_evidence_digest": _digest(
            normalized_input["evidence"]["calibration"]
        ),
        "holdout_evidence_digest": _digest(normalized_input["evidence"]["holdout"]),
        "stress_evidence_digest": _digest(normalized_input["evidence"]["stress"]),
        "new_observation_id_digest": _digest(ids),
        "new_observation_count": len(ids),
        "source_competition_observation_id_digest": _digest(
            normalized_input["competition_ids"]
        ),
        "consumed_qualification_observation_id_digest": _digest(
            normalized_input["qualification_ids"]
        ),
        "consumed_failure_boundary_observation_id_digest": _digest(
            normalized_input["boundary_ids"]
        ),
        "unique_new_observation_ids": True,
        "disjoint_from_competition_ids": True,
        "disjoint_from_qualification_ids": True,
        "disjoint_from_failure_boundary_probe_ids": True,
        "cross_epoch_pooling": False,
        "row_counts": row_counts,
        "required_context_scope_pairs": required_pairs,
        "required_context_scope_pair_count": len(required_pairs),
        "covered_context_scope_pairs_by_split": covered,
        "missing_context_scope_pairs_by_split": missing,
        "duplicate_context_scope_pair_row_counts_by_split": duplicate_pair_counts,
        "complete_exact_cartesian_coverage_by_split": complete,
        "complete_evidence": all(complete.values()),
    }


def _prediction_lookup(value: Any, label: str) -> dict[bytes, float]:
    if not isinstance(value, list) or not value:
        raise ShadowRobustIntervalRestrictionValidationError(
            f"{label} must be a non-empty list"
        )
    result: dict[bytes, float] = {}
    for index, raw in enumerate(value):
        item = _mapping(raw, f"{label}[{index}]")
        _exact_keys(item, {"context", "value"}, f"{label}[{index}]")
        context = _mapping(item["context"], f"{label}[{index}].context")
        key = canonical_json_bytes(context)
        if key in result:
            raise ShadowRobustIntervalRestrictionValidationError(
                f"{label} has duplicate contexts"
            )
        result[key] = _finite_number(item["value"], f"{label}[{index}].value")
    return result


def _functional_threshold(state: Mapping[str, Any]) -> float:
    value = state.get("violation_functionals")
    if not isinstance(value, list) or len(value) != 1:
        raise ShadowRobustIntervalRestrictionValidationError(
            "source state must have exactly one violation functional"
        )
    functional = _mapping(value[0], "source violation functional")
    if functional.get("functional_id") != "absolute_error":
        raise ShadowRobustIntervalRestrictionValidationError(
            "V1 scale requires the absolute_error functional"
        )
    threshold = _finite_number(
        functional.get("threshold"), "absolute_error threshold"
    )
    if threshold < 0.0:
        raise ShadowRobustIntervalRestrictionValidationError(
            "absolute_error threshold cannot be negative"
        )
    return threshold


def _prediction_scale(state: Mapping[str, Any]) -> tuple[dict[bytes, float], float]:
    model = _mapping(state.get("model_class"), "source model_class")
    if model.get("kind") != "finite_interval_table":
        raise ShadowRobustIntervalRestrictionValidationError(
            "restriction V1 requires a finite interval-table source"
        )
    centers = _prediction_lookup(
        model.get("center_predictions"), "source center_predictions"
    )
    mean_absolute_center = math.fsum(abs(value) for value in centers.values()) / len(centers)
    return centers, max(
        float(THRESHOLDS["numeric_epsilon"]),
        mean_absolute_center + _functional_threshold(state),
    )


def _radius_entries(model: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    _exact_keys(
        model,
        {"kind", "center_predictions", "radius_grouping", "radii"},
        "source model_class",
    )
    if model.get("kind") != "finite_interval_table":
        raise ShadowRobustIntervalRestrictionValidationError(
            "source model is not a finite interval table"
        )
    grouping = model.get("radius_grouping")
    if grouping not in {"global", "per_scope", "per_context"}:
        raise ShadowRobustIntervalRestrictionValidationError(
            "source radius_grouping is not supported"
        )
    value = model.get("radii")
    if not isinstance(value, list) or not value:
        raise ShadowRobustIntervalRestrictionValidationError(
            "source radii must be a non-empty list"
        )
    entries: list[dict[str, Any]] = []
    group_keys: set[bytes] = set()
    for index, raw in enumerate(value):
        item = _mapping(raw, f"source radii[{index}]")
        _exact_keys(item, {"group", "radius"}, f"source radii[{index}]")
        group = _copy(_mapping(item["group"], f"source radii[{index}].group"))
        if grouping == "global" and group != {"global": "*"}:
            raise ShadowRobustIntervalRestrictionValidationError(
                "global radius group key is invalid"
            )
        if grouping == "per_scope" and set(group) != {"scope_id"}:
            raise ShadowRobustIntervalRestrictionValidationError(
                "per-scope radius group key is invalid"
            )
        if grouping == "per_context" and set(group) != {"context"}:
            raise ShadowRobustIntervalRestrictionValidationError(
                "per-context radius group key is invalid"
            )
        key = canonical_json_bytes(group)
        if key in group_keys:
            raise ShadowRobustIntervalRestrictionValidationError(
                "source radius group keys are duplicated"
            )
        group_keys.add(key)
        radius = _finite_number(item["radius"], f"source radii[{index}].radius")
        if radius < 0.0:
            raise ShadowRobustIntervalRestrictionValidationError(
                "source radii must be nonnegative"
            )
        entries.append({"group": group, "radius": radius})
    return str(grouping), entries


def _radius_for_pair(
    grouping: str,
    entries: Sequence[Mapping[str, Any]],
    scope_id: str,
    context: Mapping[str, Any],
) -> float:
    matches: list[float] = []
    for item in entries:
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
        raise ShadowRobustIntervalRestrictionValidationError(
            "source radii do not map uniquely to every context-scope pair"
        )
    return matches[0]


def _candidate_registry(
    source_state: Mapping[str, Any],
    parent_contexts: Sequence[Mapping[str, Any]],
    parent_scopes: Sequence[str],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source_model = _copy(_mapping(source_state.get("model_class"), "source model_class"))
    centers, _ = _prediction_scale(source_state)
    expected_contexts = {canonical_json_bytes(item) for item in parent_contexts}
    if set(centers) != expected_contexts:
        raise ShadowRobustIntervalRestrictionValidationError(
            "source centers do not exactly cover the parent context registry"
        )
    grouping, entries = _radius_entries(source_model)
    actual_group_keys = {
        canonical_json_bytes(item["group"]) for item in entries
    }
    if grouping == "global":
        expected_group_keys = {canonical_json_bytes({"global": "*"})}
    elif grouping == "per_scope":
        expected_group_keys = {
            canonical_json_bytes({"scope_id": scope})
            for scope in parent_scopes
        }
    else:
        expected_group_keys = {
            canonical_json_bytes({"context": context})
            for context in parent_contexts
        }
    if actual_group_keys != expected_group_keys:
        raise ShadowRobustIntervalRestrictionValidationError(
            "source radius group keys do not exactly equal the frozen "
            "grouping registry"
        )
    # Validate every group lookup before constructing any candidate.
    for scope in parent_scopes:
        for context in parent_contexts:
            _radius_for_pair(grouping, entries, scope, context)

    source_model_digest = _digest(source_model)
    result: list[dict[str, Any]] = []
    for spec in contract["multiplier_registry"]:
        alpha = int(spec["numerator"]) / int(spec["denominator"])
        if alpha != float(spec["radius_multiplier"]) or not 0.0 < alpha < 1.0:
            raise ShadowRobustIntervalRestrictionValidationError(
                "fixed multiplier registry is not an exact strict rational contraction"
            )
        restricted_entries = [
            {"group": _copy(item["group"]), "radius": float(item["radius"]) * alpha}
            for item in entries
        ]
        restricted_model = {
            "kind": "finite_interval_table",
            "center_predictions": _copy(source_model["center_predictions"]),
            "radius_grouping": grouping,
            "radii": restricted_entries,
        }
        restricted_values = [float(item["radius"]) for item in restricted_entries]
        source_values = [float(item["radius"]) for item in entries]
        finite_nonnegative = all(
            math.isfinite(value) and value >= 0.0 for value in restricted_values
        )
        lte = all(new <= old for new, old in zip(restricted_values, source_values))
        strict = any(new < old for new, old in zip(restricted_values, source_values))
        geometry = {
            "source_model_class_digest": source_model_digest,
            "restricted_model_class_digest": _digest(restricted_model),
            "checked_radius_group_count": len(entries),
            "checked_context_scope_pair_count": len(parent_contexts) * len(parent_scopes),
            "centers_byte_equal": canonical_json_bytes(
                restricted_model["center_predictions"]
            ) == canonical_json_bytes(source_model["center_predictions"]),
            "grouping_and_group_keys_byte_equal": (
                restricted_model["radius_grouping"] == source_model["radius_grouping"]
                and canonical_json_bytes(
                    [item["group"] for item in restricted_entries]
                ) == canonical_json_bytes([item["group"] for item in entries])
            ),
            "all_restricted_radii_finite_nonnegative": finite_nonnegative,
            "all_restricted_radii_lte_source": lte,
            "at_least_one_radius_strictly_reduced": strict,
        }
        geometry["strict_subset_verified"] = all(geometry.values())
        result.append(
            {
                "candidate_id": spec["candidate_id"],
                "numerator": spec["numerator"],
                "denominator": spec["denominator"],
                "radius_multiplier": spec["radius_multiplier"],
                "model_class": restricted_model,
                "model_class_digest": geometry["restricted_model_class_digest"],
                "geometry_certificate": geometry,
                "candidate_commitment_digest": _digest(
                    {
                        "candidate_id": spec["candidate_id"],
                        "numerator": spec["numerator"],
                        "denominator": spec["denominator"],
                        "radius_multiplier": spec["radius_multiplier"],
                        "source_model_class_digest": source_model_digest,
                        "restricted_model_class_digest": geometry[
                            "restricted_model_class_digest"
                        ],
                        "geometry_certificate": geometry,
                    }
                ),
            }
        )
    return result


def _split_metrics(
    rows: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    scale: float,
) -> dict[str, Any]:
    centers = _prediction_lookup(model.get("center_predictions"), "candidate centers")
    grouping, entries = _radius_entries(model)
    margins: list[float] = []
    exceedances: list[float] = []
    counterexample_ids: list[str] = []
    for row in rows:
        key = canonical_json_bytes(row["context"])
        if key not in centers:
            raise ShadowRobustIntervalRestrictionValidationError(
                "candidate centers do not cover a restriction row"
            )
        radius = _radius_for_pair(
            grouping, entries, row["scope_id"], row["context"]
        )
        margin = (
            radius - abs(float(row["observed_value"]) - centers[key])
        ) / scale
        margins.append(margin)
        exceedances.append(max(0.0, -margin))
        if margin < 0.0:
            counterexample_ids.append(row["observation_id"])
    if not rows:
        return {
            "row_count": 0,
            "min_normalized_signed_margin": None,
            "boundary_violation_count": 0,
            "boundary_violation_rate": None,
            "mean_normalized_exceedance": None,
            "max_normalized_exceedance": None,
            "counterexample_observation_ids": [],
        }
    violation_count = sum(margin < 0.0 for margin in margins)
    return {
        "row_count": len(rows),
        "min_normalized_signed_margin": min(margins),
        "boundary_violation_count": violation_count,
        "boundary_violation_rate": violation_count / len(rows),
        "mean_normalized_exceedance": math.fsum(exceedances) / len(rows),
        "max_normalized_exceedance": max(exceedances),
        "counterexample_observation_ids": sorted(counterexample_ids),
    }


def _compete_candidates(
    candidates: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Sequence[Mapping[str, Any]]],
    scale: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        calibration = _split_metrics(evidence["calibration"], candidate["model_class"], scale)
        holdout = _split_metrics(evidence["holdout"], candidate["model_class"], scale)
        stress = _split_metrics(evidence["stress"], candidate["model_class"], scale)
        geometry_passed = bool(
            candidate["geometry_certificate"]["strict_subset_verified"]
        )
        calibration_admissible = geometry_passed and (
            calibration["min_normalized_signed_margin"] is not None
            and calibration["min_normalized_signed_margin"]
            >= float(THRESHOLDS["min_calibration_normalized_signed_margin"])
        )
        holdout_passed = (
            holdout["boundary_violation_rate"] is not None
            and holdout["boundary_violation_rate"]
            <= float(THRESHOLDS["max_holdout_boundary_violation_rate"])
        )
        stress_passed = (
            stress["boundary_violation_rate"] is not None
            and stress["boundary_violation_rate"]
            <= float(THRESHOLDS["max_stress_boundary_violation_rate"])
        )
        validated = holdout_passed and stress_passed
        result.append(
            {
                "candidate_id": candidate["candidate_id"],
                "radius_multiplier": candidate["radius_multiplier"],
                "candidate_commitment_digest": candidate[
                    "candidate_commitment_digest"
                ],
                "restricted_model_class_digest": candidate[
                    "model_class_digest"
                ],
                "calibration": calibration,
                "holdout": holdout,
                "stress": stress,
                "calibration_admissible": calibration_admissible,
                "fresh_validation_passed": validated,
            }
        )
    return result


def _source_summary(
    report: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "verification_status": receipt.get("status"),
        "contract_id": report.get("contract_id"),
        "contract_digest": report.get("contract_digest"),
        "report_digest": report.get("report_digest"),
        "probe_expansion_id": report.get("probe_expansion_id"),
        "probe_expanded_shadow_theory_state_digest": report.get(
            "probe_expanded_shadow_theory_state_digest"
        ),
        "transition_kind": report.get("transition_kind"),
        "disposition": report.get("disposition"),
        "boundary_counterexample_found": _mapping(
            report.get("boundary_assessment"), "source boundary_assessment"
        ).get("boundary_counterexample_found"),
        "adoption_status": report.get("adoption_status"),
    }


def _restricted_state(
    *,
    source_state: Mapping[str, Any],
    source_state_digest: str,
    probe_report_digest: str,
    restriction_contract_digest: str,
    restriction_id: str,
    evaluator_epoch: str,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "task_id": source_state["task_id"],
        "source_probe_expanded_theory_state_digest": source_state_digest,
        "source_failure_boundary_probe_report_digest": probe_report_digest,
        "operation_kind": OPERATION_KIND,
        "restriction_kind": RESTRICTION_KIND,
        "evaluator_epoch": None,
        "evaluator_status": "POST_RESTRICTION_FRESH_EVALUATOR_REQUIRED",
        "fixed_anchor": source_state["fixed_anchor"],
        "object_space": _copy(source_state["object_space"]),
        "model_class": _copy(candidate["model_class"]),
        "probe_ids": _copy(source_state["probe_ids"]),
        "violation_functionals": _copy(source_state["violation_functionals"]),
        "scope_ids": _copy(source_state["scope_ids"]),
        "removable_feature_ids": _copy(source_state["removable_feature_ids"]),
        "evidence_reuse_policy": {
            "source_competition_records_allowed_for_scoring": False,
            "consumed_qualification_records_allowed_for_scoring": False,
            "consumed_failure_boundary_probe_records_allowed_for_scoring": False,
            "restriction_competition_records_allowed_for_future_scoring": False,
            "future_scoring_requires_new_unconsumed_evidence": True,
            "cross_epoch_pooling_allowed": False,
        },
        "operational_probe_status": (
            "REQUALIFICATION_WITH_NEW_EPOCH_REQUIRED_AFTER_MODEL_RESTRICTION"
        ),
        "restriction_lineage": {
            "source_probe_contract_digest": SOURCE_PROBE_CONTRACT_DIGEST,
            "source_probe_report_digest": probe_report_digest,
            "source_probe_expanded_theory_state_digest": source_state_digest,
            "restriction_contract_digest": restriction_contract_digest,
            "restriction_id": restriction_id,
            "restriction_evaluator_epoch": evaluator_epoch,
            "selected_candidate_id": candidate["candidate_id"],
            "selected_radius_multiplier": candidate["radius_multiplier"],
            "source_model_class_digest": _digest(source_state["model_class"]),
            "restricted_model_class_digest": candidate["model_class_digest"],
        },
        "adoption_status": ADOPTION_STATUS,
        "current_status": CURRENT_STATUS,
    }
    theory_id = "shadow-robust-restricted-theory:" + _digest(payload)[7:]
    state = {"schema_version": payload.pop("schema_version"), "theory_id": theory_id}
    state.update(payload)
    _exact_keys(state, STATE_FIELDS, "restricted_shadow_theory_state")
    return state


def _restriction_certificate(
    candidate: Mapping[str, Any], checked_pair_count: int
) -> dict[str, Any]:
    geometry = _mapping(candidate["geometry_certificate"], "geometry_certificate")
    certificate = {
        "certificate_kind": (
            "STRICT_FINITE_INTERVAL_SUBSET_BY_UNIFORM_RADIUS_CONTRACTION"
        ),
        "source_model_class_digest": geometry["source_model_class_digest"],
        "restricted_model_class_digest": geometry["restricted_model_class_digest"],
        "radius_multiplier": candidate["radius_multiplier"],
        "checked_radius_group_count": geometry["checked_radius_group_count"],
        "checked_context_scope_pair_count": checked_pair_count,
        "centers_byte_equal": geometry["centers_byte_equal"],
        "grouping_and_group_keys_byte_equal": geometry[
            "grouping_and_group_keys_byte_equal"
        ],
        "all_restricted_radii_finite_nonnegative": geometry[
            "all_restricted_radii_finite_nonnegative"
        ],
        "all_restricted_radii_lte_source": geometry[
            "all_restricted_radii_lte_source"
        ],
        "at_least_one_radius_strictly_reduced": geometry[
            "at_least_one_radius_strictly_reduced"
        ],
        "strict_subset_verified": geometry["strict_subset_verified"],
    }
    _exact_keys(certificate, CERTIFICATE_FIELDS, "restriction_certificate")
    return certificate


def _rollback_boundary(
    *,
    probe_report: Mapping[str, Any],
    transition_report: Mapping[str, Any],
    source_model_digest: str,
    restricted_model_digest: str | None,
) -> dict[str, Any]:
    value = {
        "method": (
            "RESTORE_FROZEN_PROBE_EXPANDED_SOURCE_STATE_FROM_VERIFIED_PROBE_REPORT"
        ),
        "source_probe_expanded_state_digest": probe_report.get(
            "probe_expanded_shadow_theory_state_digest"
        ),
        "source_materialized_child_digest": transition_report.get(
            "child_theory_state_digest"
        ),
        "original_parent_theory_state_digest": transition_report.get(
            "parent_theory_state_digest"
        ),
        "source_model_class_digest": source_model_digest,
        "restricted_model_class_digest": restricted_model_digest,
        "rollback_execution_status": "NOT_PERFORMED",
    }
    _exact_keys(value, ROLLBACK_FIELDS, "rollback_boundary")
    return value


def _record_lifecycle_extension(
    *,
    probe_report: Mapping[str, Any],
    evidence_binding: Mapping[str, Any],
    evaluator_epoch: str | None,
    records_consumed: bool,
) -> dict[str, Any]:
    source = _mapping(
        probe_report.get("record_lifecycle_extension"),
        "source record_lifecycle_extension",
    )

    def inherited(key: str, role: str) -> dict[str, Any]:
        record = _mapping(source.get(key), f"source {key}")
        return {
            "evidence_digests": _copy(record["evidence_digests"]),
            "observation_id_digest": record["observation_id_digest"],
            "observation_count": record["observation_count"],
            "evaluator_epoch": record["evaluator_epoch"],
            "role": role,
            "eligible_for_future_scoring": False,
        }

    return {
        "competition_records": inherited(
            "competition_records", "AUDIT_ONLY_SCORING_EXCLUDED"
        ),
        "qualification_records": inherited(
            "qualification_records", "CONSUMED_AUDIT_ONLY_SCORING_EXCLUDED"
        ),
        "failure_boundary_probe_records": inherited(
            "new_probe_records",
            "CONSUMED_FAILURE_BOUNDARY_EVIDENCE_AUDIT_ONLY",
        ),
        "restriction_competition_records": {
            "evidence_digests": {
                "calibration": evidence_binding["calibration_evidence_digest"],
                "holdout": evidence_binding["holdout_evidence_digest"],
                "stress": evidence_binding["stress_evidence_digest"],
            },
            "observation_id_digest": evidence_binding[
                "new_observation_id_digest"
            ],
            "observation_count": evidence_binding["new_observation_count"],
            "evaluator_epoch": evaluator_epoch,
            "role": (
                "CONSUMED_RESTRICTION_COMPETITION_EVIDENCE_AUDIT_ONLY"
                if records_consumed
                else "NOT_CONSUMED_RESTRICTION_PRECONDITION_BLOCKED"
            ),
            "eligible_for_future_scoring": False,
        },
        "future_scoring_policy": {
            "new_unconsumed_evidence_required": True,
            "reuse_competition_records_allowed": False,
            "reuse_consumed_qualification_records_allowed": False,
            "reuse_consumed_failure_boundary_probe_records_allowed": False,
            "reuse_consumed_restriction_records_allowed": False,
            "cross_epoch_pooling_allowed": False,
        },
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


def compete_and_materialize_shadow_robust_interval_restriction(
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
    input_artifacts: Mapping[str, Any] | None = None,
) -> ShadowRobustIntervalRestrictionResult:
    """Compete fixed robust contractions without crossing the shadow boundary."""

    normalized_contract = _validate_contract(restriction_contract)
    expected_contract_digest = _require_digest(
        expected_restriction_contract_digest,
        "expected_restriction_contract_digest",
    )
    contract_digest = _digest(normalized_contract)
    if contract_digest != expected_contract_digest:
        raise ShadowRobustIntervalRestrictionValidationError(
            "restriction contract digest differs from independent expectation"
        )
    expected_input_digest = _require_digest(
        expected_restriction_input_digest, "expected_restriction_input_digest"
    )
    if _digest(restriction_input) != expected_input_digest:
        raise ShadowRobustIntervalRestrictionValidationError(
            "restriction input digest differs from independent expectation"
        )
    if input_artifacts is not None and not isinstance(input_artifacts, Mapping):
        raise ShadowRobustIntervalRestrictionValidationError(
            "input_artifacts must be an object or null"
        )
    artifacts = _copy(input_artifacts) if input_artifacts is not None else None
    _reject_observed_values(artifacts, "input_artifacts")

    expected_source_contract = _require_digest(
        expected_probe_contract_digest, "expected_probe_contract_digest"
    )
    if expected_source_contract != SOURCE_PROBE_CONTRACT_DIGEST:
        raise ShadowRobustIntervalRestrictionValidationError(
            "expected probe contract digest is not the pinned V1 source"
        )
    try:
        source_receipt = verify_shadow_child_failure_boundary_probe(
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
            expected_probe_contract_digest=expected_source_contract,
            expected_probe_report_digest=expected_probe_report_digest,
            expected_probe_input_artifacts=expected_probe_input_artifacts,
        )
    except (
        ShadowChildFailureBoundaryProbeValidationError,
        CompetitionValidationError,
        KeyError,
        TypeError,
    ) as exc:
        raise ShadowRobustIntervalRestrictionValidationError(
            f"source failure-boundary probe verification failed: {exc}"
        ) from exc

    source_report = _copy(_mapping(probe_report, "probe_report"))
    receipt = _copy(_mapping(source_receipt, "source_probe_receipt"))
    transition = _copy(_mapping(transition_report, "transition_report"))
    if source_report.get("contract_id") != SOURCE_PROBE_CONTRACT_ID:
        raise ShadowRobustIntervalRestrictionValidationError(
            "source probe contract_id is not supported"
        )
    if source_report.get("contract_digest") != SOURCE_PROBE_CONTRACT_DIGEST:
        raise ShadowRobustIntervalRestrictionValidationError(
            "source probe contract digest is not pinned V1"
        )
    if source_report.get("schema_version") != SOURCE_PROBE_REPORT_SCHEMA_VERSION:
        raise ShadowRobustIntervalRestrictionValidationError(
            "source probe report schema is not supported"
        )
    if source_report.get("adoption_status") != ADOPTION_STATUS:
        raise ShadowRobustIntervalRestrictionValidationError(
            "source probe crossed the shadow-only adoption boundary"
        )

    normalized_input = _normalize_input(
        restriction_input,
        probe_report=source_report,
        transition_report=transition,
        competition_input=competition_input,
        qualification_input=qualification_input,
        probe_input=probe_input,
    )
    semantic_input_digest = _digest(normalized_input["public"])
    restriction_id = normalized_input["public"]["restriction_id"]
    source_summary = _source_summary(source_report, receipt)
    transition_kind = _string(
        source_report.get("transition_kind"), "source transition_kind"
    )
    if transition.get("transition_kind") != transition_kind:
        raise ShadowRobustIntervalRestrictionValidationError(
            "source probe transition kind differs from verified transition"
        )

    source_state_value = source_report.get("probe_expanded_shadow_theory_state")
    source_state = (
        None
        if source_state_value is None
        else _copy(_mapping(source_state_value, "source probe-expanded state"))
    )
    source_state_digest = _optional_digest(
        source_report.get("probe_expanded_shadow_theory_state_digest"),
        "source probe-expanded state digest",
    )
    if (source_state is None) != (source_state_digest is None):
        raise ShadowRobustIntervalRestrictionValidationError(
            "source probe-expanded state and digest presence differ"
        )
    if source_state is not None and _digest(source_state) != source_state_digest:
        raise ShadowRobustIntervalRestrictionValidationError(
            "source probe-expanded state digest is inconsistent"
        )
    original_source_bytes = (
        None if source_state is None else canonical_json_bytes(source_state)
    )

    parent = _mapping(transition.get("parent_theory_state"), "parent_theory_state")
    fixed_anchor = _string(parent.get("fixed_anchor"), "parent fixed_anchor")
    expected_epoch: str | None = None
    evaluator_definition: dict[str, Any] | None = None
    if source_state_digest is not None:
        epoch_payload, expected_epoch = _epoch_components(
            probe_contract_digest=source_report["contract_digest"],
            probe_report_digest=source_report["report_digest"],
            probe_expanded_shadow_theory_state_digest=source_state_digest,
            fixed_anchor=fixed_anchor,
            restriction_contract=normalized_contract,
        )
        prior_epochs = {
            item
            for item in (
                parent.get("evaluator_epoch"),
                source_report.get("evaluator_definition", {}).get("evaluator_epoch")
                if isinstance(source_report.get("evaluator_definition"), Mapping)
                else None,
            )
            if type(item) is str
        }
        if expected_epoch in prior_epochs:
            raise ShadowRobustIntervalRestrictionValidationError(
                "derived restriction epoch is not fresh from prior epochs"
            )
        evaluator_definition = {
            "evaluator_epoch": expected_epoch,
            "fixed_anchor": fixed_anchor,
            "epoch_derivation_kind": (
                "CONTENT_ADDRESSED_LOCAL_ROBUST_RESTRICTION_EPOCH"
            ),
            "fixed_multiplier_registry_digest": _digest(
                epoch_payload["fixed_multiplier_registry"]
            ),
        }
    evaluator_binding = _evaluator_binding(
        normalized_input, expected_epoch, fixed_anchor
    )
    evidence_binding = _evidence_binding(normalized_input)

    candidates: list[dict[str, Any]] = []
    competition: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    fresh_validation: dict[str, Any] | None = None
    restricted_state: dict[str, Any] | None = None
    restricted_state_digest: str | None = None
    certificate: dict[str, Any] | None = None

    source_disposition = source_report.get("disposition")
    if transition_kind != SELECTION["applicable_transition_kind"]:
        disposition = SELECTION["non_robust_status"]
    elif source_disposition == SELECTION["source_counterexample_disposition"]:
        disposition = SELECTION["source_counterexample_status"]
    elif source_disposition != SELECTION["source_no_counterexample_status"]:
        disposition = SELECTION["source_unresolved_status"]
    else:
        if source_state is None or source_state_digest is None:
            raise ShadowRobustIntervalRestrictionValidationError(
                "resolved source probe lacks its materialized shadow state"
            )
        candidates = _candidate_registry(
            source_state,
            normalized_input["parent_contexts"],
            normalized_input["parent_scopes"],
            normalized_contract,
        )
        if not evaluator_binding["comparable"]:
            disposition = SELECTION["incomparable_status"]
        elif not evidence_binding["complete_evidence"]:
            disposition = SELECTION["needs_evidence_status"]
        else:
            _, prediction_scale = _prediction_scale(source_state)
            source_calibration = _split_metrics(
                normalized_input["evidence"]["calibration"],
                source_state["model_class"],
                prediction_scale,
            )
            source_holdout = _split_metrics(
                normalized_input["evidence"]["holdout"],
                source_state["model_class"],
                prediction_scale,
            )
            source_stress = _split_metrics(
                normalized_input["evidence"]["stress"],
                source_state["model_class"],
                prediction_scale,
            )
            fresh_validation = {
                "source_model_class_digest": _digest(
                    source_state["model_class"]
                ),
                "prediction_scale": prediction_scale,
                "calibration": source_calibration,
                "holdout": source_holdout,
                "stress": source_stress,
                "all_source_splits_boundary_violation_free": all(
                    item["boundary_violation_rate"] == 0.0
                    for item in (
                        source_calibration,
                        source_holdout,
                        source_stress,
                    )
                ),
            }
            competition = _compete_candidates(
                candidates, normalized_input["evidence"], prediction_scale
            )
            admissible = [
                item for item in competition if item["calibration_admissible"]
            ]
            validated = [
                item for item in admissible if item["fresh_validation_passed"]
            ]
            if not admissible:
                disposition = SELECTION["no_calibration_status"]
            elif not validated:
                disposition = SELECTION["no_validated_status"]
            else:
                winner_metrics = validated[0]
                winner = next(
                    item
                    for item in candidates
                    if item["candidate_id"] == winner_metrics["candidate_id"]
                )
                selected = {
                    "candidate_id": winner["candidate_id"],
                    "numerator": winner["numerator"],
                    "denominator": winner["denominator"],
                    "radius_multiplier": winner["radius_multiplier"],
                    "model_class": _copy(winner["model_class"]),
                    "model_class_digest": winner["model_class_digest"],
                }
                restricted_state = _restricted_state(
                    source_state=source_state,
                    source_state_digest=source_state_digest,
                    probe_report_digest=source_report["report_digest"],
                    restriction_contract_digest=contract_digest,
                    restriction_id=restriction_id,
                    evaluator_epoch=expected_epoch,
                    candidate=winner,
                )
                restricted_state_digest = _digest(restricted_state)
                certificate = _restriction_certificate(
                    winner, evidence_binding["required_context_scope_pair_count"]
                )
                disposition = SELECTION["materialized_status"]

    source_model_digest = (
        _digest(source_state["model_class"])
        if source_state is not None
        else _digest(transition["child_theory_state"]["model_class"])
    )
    rollback = _rollback_boundary(
        probe_report=source_report,
        transition_report=transition,
        source_model_digest=source_model_digest,
        restricted_model_digest=(
            None if selected is None else selected["model_class_digest"]
        ),
    )
    records_consumed = (
        source_disposition == SELECTION["source_no_counterexample_status"]
        and transition_kind == SELECTION["applicable_transition_kind"]
        and evaluator_binding["comparable"]
        and evidence_binding["complete_evidence"]
    )
    lifecycle = _record_lifecycle_extension(
        probe_report=source_report,
        evidence_binding=evidence_binding,
        evaluator_epoch=expected_epoch,
        records_consumed=records_consumed,
    )

    events = [
        _audit_event(
            0,
            "SOURCE_FAILURE_BOUNDARY_PROBE_VERIFIED",
            GENESIS_DIGEST,
            {
                "verification_status": source_summary["verification_status"],
                "source_probe_report_digest": source_summary["report_digest"],
                "source_disposition": source_disposition,
                "source_transition_kind": transition_kind,
            },
        ),
        _audit_event(
            1,
            "RESTRICTION_EPOCH_AND_RECORD_ISOLATION_BOUND",
            GENESIS_DIGEST,
            {
                "expected_evaluator_epoch": expected_epoch,
                "new_observation_id_digest": evidence_binding[
                    "new_observation_id_digest"
                ],
                "cross_epoch_pooling": False,
            },
        ),
    ]
    # Bind event 1 to event 0 after construction without hidden mutation of payload.
    events[1] = _audit_event(
        1,
        events[1]["event"],
        events[0]["event_digest"],
        events[1]["payload"],
    )
    events.append(
        _audit_event(
            2,
            "FIXED_RESTRICTION_CANDIDATES_COMPETED",
            events[-1]["event_digest"],
            {
                "candidate_registry_digest": _digest(candidates),
                "candidate_competition_digest": _digest(competition),
                "selected_candidate_id": (
                    None if selected is None else selected["candidate_id"]
                ),
                "disposition": disposition,
            },
        )
    )
    events.append(
        _audit_event(
            3,
            "SHADOW_RESTRICTION_ASSESSED_AND_AUTHORITY_WITHHELD",
            events[-1]["event_digest"],
            {
                "restricted_shadow_theory_state_digest": restricted_state_digest,
                "source_state_mutated": False,
                "rollback_executed": False,
                "adoption_status": ADOPTION_STATUS,
                "current_status": CURRENT_STATUS,
            },
        )
    )

    if source_state is not None and canonical_json_bytes(source_state) != original_source_bytes:
        raise ShadowRobustIntervalRestrictionValidationError(
            "source probe-expanded state was mutated during restriction competition"
        )
    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "contract_digest": contract_digest,
        "restriction_input_digest": semantic_input_digest,
        "restriction_id": restriction_id,
        "source_failure_boundary_probe": source_summary,
        "source_probe_expanded_shadow_theory_state_digest": source_state_digest,
        "source_transition_kind": transition_kind,
        "operation_kind": OPERATION_KIND,
        "restriction_kind": RESTRICTION_KIND,
        "evaluator_definition": evaluator_definition,
        "evaluator_binding": evaluator_binding,
        "evidence_binding": evidence_binding,
        "candidate_registry": candidates,
        "candidate_competition": competition,
        "selected_candidate": selected,
        "fresh_validation": fresh_validation,
        "disposition": disposition,
        "restricted_shadow_theory_state": restricted_state,
        "restricted_shadow_theory_state_digest": restricted_state_digest,
        "restriction_certificate": certificate,
        "rollback_boundary": rollback,
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
    _exact_keys(report, REPORT_FIELDS, "restriction_report")
    _reject_observed_values(report, "restriction_report")
    return ShadowRobustIntervalRestrictionResult(report=report)


def verify_shadow_robust_interval_restriction(
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
) -> dict[str, Any]:
    """Exact-replay a restriction report against independent anchors."""

    expected_report_digest = _require_digest(
        expected_restriction_report_digest,
        "expected_restriction_report_digest",
    )
    supplied = _copy(_mapping(restriction_report, "restriction_report"))
    _exact_keys(supplied, REPORT_FIELDS, "restriction_report")
    _reject_observed_values(supplied, "restriction_report")
    if expected_restriction_input_artifacts is not None and not isinstance(
        expected_restriction_input_artifacts, Mapping
    ):
        raise ShadowRobustIntervalRestrictionValidationError(
            "expected_restriction_input_artifacts must be an object or null"
        )
    expected_artifacts = (
        _copy(expected_restriction_input_artifacts)
        if expected_restriction_input_artifacts is not None
        else None
    )
    _reject_observed_values(
        expected_artifacts, "expected_restriction_input_artifacts"
    )
    if supplied.get("input_artifacts") != expected_artifacts:
        raise ShadowRobustIntervalRestrictionValidationError(
            "restriction input artifacts differ from independent expectation"
        )
    fresh = compete_and_materialize_shadow_robust_interval_restriction(
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
        input_artifacts=expected_artifacts,
    ).to_dict()
    if fresh["report_digest"] != expected_report_digest:
        raise ShadowRobustIntervalRestrictionValidationError(
            "replayed restriction report digest differs from expectation"
        )
    if canonical_json_bytes(supplied) != canonical_json_bytes(fresh):
        raise ShadowRobustIntervalRestrictionValidationError(
            "supplied restriction report differs from exact replay"
        )
    selected = fresh["selected_candidate"]
    return {
        "status": "VERIFIED_" + fresh["disposition"],
        "disposition": fresh["disposition"],
        "report_digest": fresh["report_digest"],
        "contract_digest": fresh["contract_digest"],
        "restriction_id": fresh["restriction_id"],
        "source_probe_report_digest": fresh["source_failure_boundary_probe"][
            "report_digest"
        ],
        "source_probe_expanded_shadow_theory_state_digest": fresh[
            "source_probe_expanded_shadow_theory_state_digest"
        ],
        "restricted_shadow_theory_state_digest": fresh[
            "restricted_shadow_theory_state_digest"
        ],
        "restriction_materialized": fresh["restricted_shadow_theory_state"]
        is not None,
        "selected_radius_multiplier": (
            None if selected is None else selected["radius_multiplier"]
        ),
        "adoption_eligibility": fresh["adoption_eligibility"],
        "adoption_status": fresh["adoption_status"],
        "promotion_status": fresh["promotion_status"],
        "current_status": fresh["current_status"],
    }
