"""Qualify one exact V10 interval/multi-Q shadow child on a fresh epoch.

The layer replays the complete transition through its public verifier and
evaluates only caller-supplied holdout and stress rows with the frozen native
interval two-Q formulas.  It never materializes or mutates a theory, acquires
evidence, reranks candidates, falls back, adopts, promotes, writes a current
pointer, invents language, calls ``run_one``, or writes state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from performance.shadow_interval_multi_q_theory_transition import (
    ShadowIntervalMultiQTheoryTransitionValidationError,
    verify_shadow_interval_multi_q_theory_transition,
)
from performance.theory_operation_competition import canonical_json_bytes


CONTRACT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-post-transition-qualification-contract/1"
)
CONTRACT_ID = "shadow_interval_multi_q_post_transition_qualification_v1"
INPUT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-post-transition-qualification-input/1"
)
SOURCE_INTERVAL_COMPETITION_CONTRACT_ID = (
    "shadow_interval_multi_q_theory_operation_competition_v2"
)
SOURCE_INTERVAL_COMPETITION_CONTRACT_DIGEST = (
    "sha256:4c30c0b1a2cdec92ab1676e98677b620907bb9652bff1ce71865fce9d45ccd1e"
)
SOURCE_TRANSITION_CONTRACT_ID = "shadow_interval_multi_q_theory_transition_v1"
SOURCE_TRANSITION_CONTRACT_DIGEST = (
    "sha256:b1a5f1761c2cafcae24f37f22810178074b9fc7800b6d73bdfd631be3b1df86d"
)
SOURCE_TRANSITION_REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-theory-transition-report/1"
)
CHILD_THEORY_SCHEMA_VERSION = "sc-olh-kg.shadow-interval-multi-q-theory-state/1"
REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-post-transition-qualification-report/1"
)
EVALUATOR_DEFINITION_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-post-transition-evaluator-definition/1"
)
FROZEN_CONTRACT_DIGEST = (
    "sha256:9a52bb7ea3f4ce0f0cb16c5a5a296d284a85f4bfbdb2ee671e8c865bb2d3d493"
)
GENESIS_DIGEST = "sha256:" + "0" * 64

PROBE_IDS = [
    "absolute_error_point_prediction",
    "normalized_signed_interval_boundary_margin",
]
SPLITS = ["holdout", "stress"]
EXACT_ROWS_PER_CELL = {"holdout": 1, "stress": 1}
EXCLUSION_POLICY = (
    "EXCLUDE_FIVE_PRIOR_GENERATIONS_AND_V2_DISCOVERY_VALIDATION_STRESS"
)
STRESS_NOT_EVALUATED_STATUS = "NOT_EVALUATED_HOLDOUT_FAILED"
FIVE_PRIOR_KEYS = [
    "competition",
    "qualification",
    "failure_boundary_probe",
    "restriction",
    "post_restriction_adjudication",
]
MAX_CONTEXT_COUNT = 64
MAX_SCOPE_COUNT = 16

NOT_APPLICABLE = (
    "POST_TRANSITION_QUALIFICATION_NOT_APPLICABLE_NO_MATERIALIZED_CHILD"
)
NEEDS_EVIDENCE = "POST_TRANSITION_QUALIFICATION_NEEDS_EXACT_FRESH_EVIDENCE"
INCOMPARABLE = (
    "POST_TRANSITION_QUALIFICATION_INCOMPARABLE_FRESH_EVALUATOR_EPOCH"
)
FAILED_HOLDOUT = "POST_TRANSITION_QUALIFICATION_FAILED_FRESH_HOLDOUT"
FAILED_STRESS = (
    "POST_TRANSITION_QUALIFICATION_FAILED_FRESH_STRESS_CONFIRMATION"
)
QUALIFIED = "QUALIFIED_FRESH_POST_TRANSITION_EVALUATOR_EPOCH"
ALL_DISPOSITIONS = {
    NOT_APPLICABLE,
    NEEDS_EVIDENCE,
    INCOMPARABLE,
    FAILED_HOLDOUT,
    FAILED_STRESS,
    QUALIFIED,
}

SOURCE_TRANSITION_REGISTRY = {
    "MATERIALIZED_SHADOW_INTERVAL_EXPANSION": {
        "child_materialized": True,
        "candidate_family": "interval_robustify",
        "operation_kind": "expand",
        "transition_kind": "INTERVAL_EXPANSION",
    },
    "MATERIALIZED_SHADOW_UNIFORM_INTERVAL_RESTRICTION": {
        "child_materialized": True,
        "candidate_family": "interval_restrict",
        "operation_kind": "restrict",
        "transition_kind": "UNIFORM_INTERVAL_RESTRICTION",
    },
    "MATERIALIZED_SHADOW_CONSERVATIVE_QUOTIENT_ENVELOPE": {
        "child_materialized": True,
        "candidate_family": "interval_quotient",
        "operation_kind": "quotient",
        "transition_kind": "CONSERVATIVE_INTERVAL_QUOTIENT_ENVELOPE",
    },
    "NOT_MATERIALIZED_NEEDS_EXACT_FRESH_EVIDENCE": {
        "child_materialized": False,
        "candidate_family": None,
        "operation_kind": None,
        "transition_kind": None,
    },
    "NOT_MATERIALIZED_INCOMPARABLE_EVALUATOR_EPOCH": {
        "child_materialized": False,
        "candidate_family": None,
        "operation_kind": None,
        "transition_kind": None,
    },
    "NOT_MATERIALIZED_EARLY_DIAGNOSTIC_UNRESOLVED": {
        "child_materialized": False,
        "candidate_family": None,
        "operation_kind": None,
        "transition_kind": None,
    },
    "NOT_MATERIALIZED_NO_VALIDATION_WINNER": {
        "child_materialized": False,
        "candidate_family": None,
        "operation_kind": None,
        "transition_kind": None,
    },
    "NOT_MATERIALIZED_PROVISIONAL_WINNER_FAILED_STRESS_CONFIRMATION": {
        "child_materialized": False,
        "candidate_family": None,
        "operation_kind": None,
        "transition_kind": None,
    },
    "NOT_MATERIALIZED_ADAPTER_NEEDS_POST_RESTRICTION_EVIDENCE": {
        "child_materialized": False,
        "candidate_family": None,
        "operation_kind": None,
        "transition_kind": None,
    },
    "NOT_MATERIALIZED_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH": {
        "child_materialized": False,
        "candidate_family": None,
        "operation_kind": None,
        "transition_kind": None,
    },
}

DISPOSITION_REGISTRY = {
    "not_applicable_no_materialized_child": NOT_APPLICABLE,
    "needs_exact_fresh_evidence": NEEDS_EVIDENCE,
    "incomparable_fresh_evaluator_epoch": INCOMPARABLE,
    "failed_fresh_holdout": FAILED_HOLDOUT,
    "failed_fresh_stress_confirmation": FAILED_STRESS,
    "qualified_fresh_post_transition_evaluator_epoch": QUALIFIED,
}

EVIDENCE_POLICY = {
    "source_evidence_exclusion_policy": EXCLUSION_POLICY,
    "required_splits": SPLITS,
    "exact_rows_per_parent_context_scope_cell": EXACT_ROWS_PER_CELL,
    "coverage_domain": "EXACT_PARENT_CONTEXT_CROSS_REGISTERED_SCOPE_CARTESIAN",
    "quotient_input_context_policy": (
        "FULL_PARENT_CONTEXT_PROJECT_TO_CHILD_ONLY_FOR_CHILD_EVALUATION"
    ),
    "require_global_observation_id_uniqueness": True,
    "require_disjoint_from_six_source_generations": True,
    "require_registered_parent_contexts": True,
    "require_registered_scopes": True,
    "malformed_unregistered_duplicate_collision_or_nonfinite_is_error": True,
    "only_registered_cell_cardinality_mismatch_routes_needs_evidence": True,
    "no_child_input_policy": (
        "QUALIFICATION_ID_EVALUATOR_EXCLUSION_AND_EVIDENCE_ALL_NULL"
    ),
}

EVALUATOR_EPOCH_POLICY = {
    "epoch_prefix": "shadow-interval-multi-q-post-transition-evaluator-epoch:",
    "qualification_id_prefix": (
        "shadow-interval-multi-q-post-transition-qualification:"
    ),
    "qualification_id_derivation_inputs": [
        "post_transition_qualification_contract_digest",
        "source_transition_contract_digest",
        "source_transition_report_digest",
        "parent_theory_state_digest",
        "child_theory_state_digest",
        "operation_kind",
        "transition_kind",
    ],
    "evaluator_epoch_derivation_inputs": [
        "qualification_id",
        "source_interval_competition_contract_digest",
        "source_interval_competition_report_digest",
        "fixed_anchor",
        "fixed_probe_registry",
    ],
    "require_fresh_from_six_source_generations": True,
    "require_declared_and_all_rows_exact_epoch": True,
    "inherit_fixed_anchor": True,
    "require_declared_and_all_rows_exact_anchor": True,
    "forbid_cross_epoch_pooling": True,
    "local_content_derived_epoch_is_external_attestation": False,
}

INTERVAL_SCORING_POLICY = {
    "prediction_scale_formula": (
        "max(numeric_epsilon,stable_mean(abs(parent_center))+absolute_error_threshold)"
    ),
    "stable_mean_algorithm": (
        "sorted_finite_math_fsum_divide_count_then_max_abs_scaled_fallback_on_nonfinite_or_overflow"
    ),
    "q1": "absolute_observed_minus_center_error_divided_by_prediction_scale",
    "q2": (
        "radius_minus_absolute_observed_minus_center_error_divided_by_prediction_scale"
    ),
    "raw_boundary_predicate": (
        "absolute_observed_minus_center_error_strictly_greater_than_radius"
    ),
    "normalized_boundary_exceedance": (
        "max_zero_error_minus_radius_divided_by_prediction_scale_report_only"
    ),
    "q1_divergence": (
        "maximum_absolute_child_error_minus_parent_error_divided_by_prediction_scale"
    ),
    "q2_divergence": "maximum_absolute_child_q2_minus_parent_q2",
    "maximum_probe_divergence": "maximum_of_q1_and_q2_divergence",
    "source_tail_statistic": "raw_parent_boundary_exceedance",
    "source_tail_k": "max(1,ceil(row_count*source_tail_fraction))",
    "source_tail_cutoff": "kth_descending_raw_parent_boundary_exceedance",
    "source_tail_cutoff_units": "source_prediction_units",
    "source_tail_tie_policy": "INCLUDE_ALL_AT_OR_ABOVE_SOURCE_CUTOFF",
    "observation_id_used_for_tail_membership": False,
    "split_metric_definitions": {
        "mean_normalized_center_error": (
            "stable_mean(raw_absolute_error)_then_divide_once_by_prediction_scale"
        ),
        "raw_boundary_coverage": (
            "one_minus_raw_boundary_violation_count_divided_by_row_count"
        ),
        "source_tail_coverage": "raw_boundary_coverage_on_source_tail_membership",
        "mean_normalized_radius": (
            "stable_mean(raw_radius)_then_divide_once_by_prediction_scale"
        ),
        "mean_normalized_boundary_exceedance": (
            "stable_mean(raw_positive_boundary_exceedance)_then_divide_once_by_prediction_scale"
        ),
    },
    "delta_definitions": {
        "normalized_center_mae_gain": (
            "parent_minus_child_mean_normalized_center_error"
        ),
        "raw_boundary_coverage_gain": (
            "child_minus_parent_raw_boundary_coverage"
        ),
        "source_tail_coverage_gain": (
            "child_minus_parent_source_tail_coverage"
        ),
    },
    "radius_change_definitions": {
        "normalized_radius_change": "child_minus_parent_mean_normalized_radius",
        "normalized_radius_expansion": "max(0,normalized_radius_change)",
        "normalized_radius_reduction": "max(0,minus_normalized_radius_change)",
    },
    "context_reduction_fraction": (
        "one_minus_child_context_count_divided_by_parent_context_count_for_quotient_else_zero"
    ),
    "uniform_contraction_fraction": (
        "one_minus_exact_selected_radius_multiplier_for_restriction_else_zero"
    ),
    "qualification_score_formula": (
        "delta_mae_plus_0_75_delta_coverage_plus_0_75_delta_tail_plus_context_reduction_plus_0_5_uniform_contraction_plus_0_5_radius_reduction_minus_0_75_max_probe_divergence_minus_0_5_radius_expansion"
    ),
    "score_weights": {
        "normalized_center_mae_gain": 1.0,
        "raw_boundary_coverage_gain": 0.75,
        "source_tail_coverage_gain": 0.75,
        "context_reduction_fraction": 1.0,
        "uniform_contraction_fraction": 0.5,
        "normalized_radius_reduction": 0.5,
        "max_probe_divergence_penalty": 0.75,
        "normalized_radius_expansion_penalty": 0.5,
    },
    "score_units": "dimensionless",
    "score_component_units": {
        "normalized_center_mae_gain": "dimensionless",
        "raw_boundary_coverage_gain": "dimensionless",
        "source_tail_coverage_gain": "dimensionless",
        "context_reduction_fraction": "dimensionless",
        "uniform_contraction_fraction": "dimensionless",
        "normalized_radius_reduction": "dimensionless",
        "max_probe_divergence": "dimensionless",
        "normalized_radius_expansion": "dimensionless",
    },
    "nonfinite_derived_arithmetic": "REJECT_REPORT_NOT_EMITTED",
}

THRESHOLDS = {
    "numeric_epsilon": 1e-12,
    "source_tail_fraction": 0.25,
    "minimum_raw_boundary_coverage": 0.75,
    "minimum_source_tail_coverage": 0.75,
    "maximum_normalized_center_mae_increase": 0.05,
    "maximum_normalized_q1_divergence": 0.20,
    "maximum_normalized_q2_divergence": 1.0,
    "minimum_holdout_qualification_score": 0.0,
    "expansion_minimum_coverage_or_tail_gain": 0.05,
    "expansion_maximum_normalized_radius_increase": 1.0,
    "restriction_required_raw_boundary_violation_rate": 0.0,
    "quotient_minimum_context_reduction_fraction": 0.20,
}

HOLDOUT_QUALIFICATION_POLICY = {
    "evaluated_before_stress": True,
    "common_gates": [
        "minimum_raw_boundary_coverage",
        "minimum_source_tail_coverage",
        "maximum_normalized_center_mae_increase",
        "maximum_normalized_q1_divergence",
        "maximum_normalized_q2_divergence",
        "minimum_holdout_qualification_score",
    ],
    "expansion_gates": [
        "strict_v10_expansion_certificate",
        "minimum_coverage_or_tail_gain",
        "maximum_normalized_radius_increase",
    ],
    "restriction_gates": [
        "strict_v10_restriction_certificate",
        "zero_fresh_raw_boundary_violation_rate",
    ],
    "quotient_gates": [
        "v10_global_envelope_certificate",
        "minimum_context_reduction",
        "coverage_not_below_parent",
        "tail_coverage_not_below_parent",
    ],
    "holdout_failure_short_circuits_stress": True,
    "stress_not_evaluated_status": STRESS_NOT_EVALUATED_STATUS,
    "stress_not_evaluated_surfaces": "ALL_NULL_EXCEPT_STATUS",
}

STRESS_CONFIRMATION_POLICY = {
    "evaluated_only_after_holdout_passes": True,
    "common_gates": [
        "minimum_raw_boundary_coverage",
        "minimum_source_tail_coverage",
        "maximum_normalized_center_mae_increase",
        "maximum_normalized_q1_divergence",
        "maximum_normalized_q2_divergence",
    ],
    "family_gates_same_as_holdout_excluding_score": True,
    "stress_score": None,
    "fallback_candidate_evaluated": False,
    "fallback_candidate_selected": False,
    "reranking_performed": False,
}

RECORD_LIFECYCLE_POLICY = {
    "excluded_source_generations": [
        "competition",
        "qualification",
        "failure_boundary_probe",
        "restriction",
        "post_restriction_adjudication",
        "interval_multi_q_competition_v2",
    ],
    "source_record_role": "AUDIT_ONLY_SCORING_EXCLUDED",
    "fresh_holdout_role": "POST_TRANSITION_QUALIFICATION_ONLY",
    "fresh_stress_role": "UNIQUE_CHILD_CONFIRMATION_ONLY",
    "cross_epoch_pooling_allowed": False,
    "logical_selective_erasure_applied": True,
    "physical_erasure": "NOT_PERFORMED",
}

AUTHORITY_BOUNDARY = {
    "scope": "LOCAL_SHADOW_INTERVAL_MULTI_Q_POST_TRANSITION_QUALIFICATION_ONLY",
    "source_transition_public_exact_replay_required": True,
    "qualification_allowed_only_for_exact_materialized_v10_child": True,
    "candidate_synthesis_reselection_ranking_or_fallback_authority": False,
    "theory_materialization_or_rematerialization_authority": False,
    "probe_acquisition_or_environment_execution_authority": False,
    "adoption_eligibility_authority": False,
    "adoption_authority": False,
    "promotion_authority": False,
    "current_pointer_authority": False,
    "language_expansion_authority": False,
    "parent_child_seed_or_ambient_write_authority": False,
}

SELECTION = {
    "not_applicable_status": NOT_APPLICABLE,
    "needs_evidence_status": NEEDS_EVIDENCE,
    "incomparable_status": INCOMPARABLE,
    "failed_holdout_status": FAILED_HOLDOUT,
    "failed_stress_status": FAILED_STRESS,
    "qualified_status": QUALIFIED,
    "adoption_eligibility": "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED",
    "adoption_status": "NOT_ADOPTED_SHADOW_ONLY",
    "promotion_status": "NOT_PROMOTED",
    "current_status": "NOT_CURRENT",
}

MANDATORY_NONCLAIMS = (
    "shadow_only",
    "interval_multi_q_post_transition_qualification_v1_only",
    "full_transition_chain_exact_replay_is_not_external_attestation",
    "only_exact_v10_materialized_child_is_qualification_applicable",
    "no_child_route_is_strictly_not_applicable",
    "no_candidate_synthesis_reselection_ranking_or_fallback",
    "no_theory_state_materialization_or_rematerialization",
    "original_parent_child_seed_and_ambient_state_unmodified",
    "fresh_caller_supplied_holdout_and_stress_rows_only",
    "no_probe_acquisition_or_environment_execution",
    "fixed_two_q_registry_only",
    "native_interval_scoring_only",
    "raw_boundary_predicate_is_authoritative",
    "normalized_margin_is_not_boundary_membership",
    "five_prior_generations_and_v2_evidence_are_scoring_excluded",
    "no_cross_epoch_pooling",
    "logical_selective_erasure_only",
    "no_physical_record_deletion",
    "fresh_epoch_is_content_derived_not_externally_attested",
    "fixed_anchor_equality_is_not_external_attestation",
    "qualification_is_not_adoption_eligibility",
    "qualification_is_not_adoption",
    "no_h_t_to_h_t_plus_1_acceptance",
    "no_adoption",
    "no_promotion",
    "no_current_pointer_write",
    "no_language_or_predicate_invention",
    "quotient_containment_is_not_point_prediction_preservation",
    "quotient_recovery_requires_exact_parent_snapshot",
    "interval_containment_is_not_domain_safety",
    "tail_coverage_is_not_cvar_or_worst_case_safety",
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
        "input_schema_version",
        "source_interval_competition_contract_id",
        "source_interval_competition_contract_digest",
        "source_transition_contract_id",
        "source_transition_contract_digest",
        "source_transition_report_schema_version",
        "child_theory_schema_version",
        "report_schema_version",
        "fixed_probe_registry",
        "source_transition_registry",
        "disposition_registry",
        "evidence_policy",
        "evaluator_epoch_policy",
        "interval_scoring_policy",
        "thresholds",
        "holdout_qualification_policy",
        "stress_confirmation_policy",
        "record_lifecycle_policy",
        "authority_boundary",
        "selection",
        "nonclaims",
    }
)
INPUT_FIELDS = frozenset(
    {
        "schema_version",
        "qualification_id",
        "source_transition",
        "evaluator",
        "source_evidence_exclusion",
        "evidence",
    }
)
SOURCE_TRANSITION_INPUT_FIELDS = frozenset(
    {
        "transition_contract_digest",
        "transition_report_digest",
        "source_interval_competition_report_digest",
        "disposition",
        "operation_kind",
        "transition_kind",
        "selected_candidate_id",
        "selected_candidate_family",
        "parent_theory_state_digest",
        "child_theory_state_digest",
    }
)
EVALUATOR_FIELDS = frozenset({"evaluator_epoch", "fixed_anchor"})
EXCLUSION_FIELDS = frozenset(
    {
        "policy",
        "five_prior_generation_observation_id_digests",
        "v2_competition_evidence_digests",
    }
)
EVIDENCE_FIELDS = frozenset({"holdout", "stress"})
ROW_FIELDS = frozenset(
    {
        "observation_id",
        "evaluator_epoch",
        "fixed_anchor",
        "scope_id",
        "context",
        "observed_value",
    }
)
REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "contract_digest",
        "qualification_input_digest",
        "source_transition",
        "qualification_id",
        "parent_theory_state_digest",
        "child_theory_state_digest",
        "operation_kind",
        "transition_kind",
        "evaluator_definition",
        "evaluator_binding",
        "evidence_binding",
        "selective_erasure_receipt",
        "fixed_probe_registry",
        "probe_results",
        "disposition",
        "qualification_binding",
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


class ShadowIntervalMultiQPostTransitionQualificationValidationError(ValueError):
    """Raised when V11 qualification fails its frozen fail-closed boundary."""


class ShadowIntervalMultiQPostTransitionQualificationDisposition(str, Enum):
    POST_TRANSITION_QUALIFICATION_NOT_APPLICABLE_NO_MATERIALIZED_CHILD = (
        NOT_APPLICABLE
    )
    POST_TRANSITION_QUALIFICATION_NEEDS_EXACT_FRESH_EVIDENCE = NEEDS_EVIDENCE
    POST_TRANSITION_QUALIFICATION_INCOMPARABLE_FRESH_EVALUATOR_EPOCH = INCOMPARABLE
    POST_TRANSITION_QUALIFICATION_FAILED_FRESH_HOLDOUT = FAILED_HOLDOUT
    POST_TRANSITION_QUALIFICATION_FAILED_FRESH_STRESS_CONFIRMATION = FAILED_STRESS
    QUALIFIED_FRESH_POST_TRANSITION_EVALUATOR_EPOCH = QUALIFIED


@dataclass(frozen=True)
class ShadowIntervalMultiQPostTransitionQualificationResult:
    report: Mapping[str, Any]

    @property
    def disposition(self) -> str:
        return str(self.report["disposition"])

    @property
    def report_digest(self) -> str:
        return str(self.report["report_digest"])

    @property
    def qualified(self) -> bool:
        return self.disposition == QUALIFIED

    def to_dict(self) -> dict[str, Any]:
        return _copy(self.report)


def _copy(value: Any) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("ascii"))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "value is not canonical finite JSON"
        ) from exc


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
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
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label} keys differ; missing={missing}, extra={extra}"
        )


def _string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label} must be a non-empty string"
        )
    return value


def _strings(value: Any, label: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label} must be a{' non-empty' if nonempty else ''} list"
        )
    result = [_string(item, f"{label} item") for item in value]
    if len(result) != len(set(result)):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label} must be unique"
        )
    return result


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label} must be a finite number"
        )
    return float(value)


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_digest(value: Any, label: str) -> str:
    digest = _string(value, label)
    if len(digest) != 71 or not digest.startswith("sha256:"):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label} must be sha256:<64hex>"
        )
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
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
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label} differs from frozen V11"
        )


def _mean(values: Sequence[float], label: str) -> float:
    if not values:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label} cannot be empty"
        )
    numeric = [float(item) for item in values]
    if not all(math.isfinite(item) for item in numeric):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label} contains a nonfinite value"
        )
    try:
        result = math.fsum(sorted(numeric)) / len(numeric)
    except OverflowError:
        result = math.nan
    if not math.isfinite(result):
        scale = max(abs(item) for item in numeric)
        if scale == 0.0:
            return 0.0
        try:
            result = (
                math.fsum(sorted(item / scale for item in numeric))
                / len(numeric)
                * scale
            )
        except OverflowError as exc:
            raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
                f"{label} overflowed finite arithmetic"
            ) from exc
    if not math.isfinite(result):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label} is not finite"
        )
    return result


def validate_shadow_interval_multi_q_post_transition_qualification_contract(
    contract_value: Any,
) -> dict[str, Any]:
    """Validate and detach the frozen V11 qualification contract."""

    contract = _mapping(contract_value, "post_transition_qualification_contract")
    _exact_keys(contract, CONTRACT_FIELDS, "post_transition_qualification_contract")
    scalars = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "source_interval_competition_contract_id": (
            SOURCE_INTERVAL_COMPETITION_CONTRACT_ID
        ),
        "source_interval_competition_contract_digest": (
            SOURCE_INTERVAL_COMPETITION_CONTRACT_DIGEST
        ),
        "source_transition_contract_id": SOURCE_TRANSITION_CONTRACT_ID,
        "source_transition_contract_digest": SOURCE_TRANSITION_CONTRACT_DIGEST,
        "source_transition_report_schema_version": (
            SOURCE_TRANSITION_REPORT_SCHEMA_VERSION
        ),
        "child_theory_schema_version": CHILD_THEORY_SCHEMA_VERSION,
        "report_schema_version": REPORT_SCHEMA_VERSION,
    }
    for key, expected in scalars.items():
        if contract.get(key) != expected:
            raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
                f"{key} differs from frozen V11"
            )
    for key, expected in (
        ("fixed_probe_registry", PROBE_IDS),
        ("source_transition_registry", SOURCE_TRANSITION_REGISTRY),
        ("disposition_registry", DISPOSITION_REGISTRY),
        ("evidence_policy", EVIDENCE_POLICY),
        ("evaluator_epoch_policy", EVALUATOR_EPOCH_POLICY),
        ("interval_scoring_policy", INTERVAL_SCORING_POLICY),
        ("thresholds", THRESHOLDS),
        ("holdout_qualification_policy", HOLDOUT_QUALIFICATION_POLICY),
        ("stress_confirmation_policy", STRESS_CONFIRMATION_POLICY),
        ("record_lifecycle_policy", RECORD_LIFECYCLE_POLICY),
        ("authority_boundary", AUTHORITY_BOUNDARY),
        ("selection", SELECTION),
    ):
        _equal(contract.get(key), expected, key)
    if not isinstance(contract.get("nonclaims"), list) or tuple(
        contract["nonclaims"]
    ) != MANDATORY_NONCLAIMS:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "nonclaims differ from the exact mandatory V11 list"
        )
    normalized = _copy(contract)
    if _digest(normalized) != FROZEN_CONTRACT_DIGEST:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "post-transition qualification contract digest differs from frozen V11"
        )
    return normalized


def derive_shadow_interval_multi_q_post_transition_qualification_id(
    *,
    source_transition_contract_digest: str,
    source_transition_report_digest: str,
    parent_theory_state_digest: str,
    child_theory_state_digest: str,
    operation_kind: str,
    transition_kind: str,
    post_transition_qualification_contract: Mapping[str, Any],
) -> str:
    contract = validate_shadow_interval_multi_q_post_transition_qualification_contract(
        post_transition_qualification_contract
    )
    source_contract = _require_digest(
        source_transition_contract_digest, "source_transition_contract_digest"
    )
    if source_contract != SOURCE_TRANSITION_CONTRACT_DIGEST:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "source transition contract is not the frozen V10 source"
        )
    body = {
        "post_transition_qualification_contract_digest": _digest(contract),
        "source_transition_contract_digest": source_contract,
        "source_transition_report_digest": _require_digest(
            source_transition_report_digest, "source_transition_report_digest"
        ),
        "parent_theory_state_digest": _require_digest(
            parent_theory_state_digest, "parent_theory_state_digest"
        ),
        "child_theory_state_digest": _require_digest(
            child_theory_state_digest, "child_theory_state_digest"
        ),
        "operation_kind": _string(operation_kind, "operation_kind"),
        "transition_kind": _string(transition_kind, "transition_kind"),
    }
    return EVALUATOR_EPOCH_POLICY["qualification_id_prefix"] + _digest(body)[7:]


def derive_shadow_interval_multi_q_post_transition_evaluator_epoch(
    *,
    qualification_id: str,
    source_interval_competition_contract_digest: str,
    source_interval_competition_report_digest: str,
    fixed_anchor: str,
    post_transition_qualification_contract: Mapping[str, Any],
) -> str:
    contract = validate_shadow_interval_multi_q_post_transition_qualification_contract(
        post_transition_qualification_contract
    )
    source_contract = _require_digest(
        source_interval_competition_contract_digest,
        "source_interval_competition_contract_digest",
    )
    if source_contract != SOURCE_INTERVAL_COMPETITION_CONTRACT_DIGEST:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "source interval competition contract is not frozen V9"
        )
    body = {
        "qualification_id": _string(qualification_id, "qualification_id"),
        "source_interval_competition_contract_digest": source_contract,
        "source_interval_competition_report_digest": _require_digest(
            source_interval_competition_report_digest,
            "source_interval_competition_report_digest",
        ),
        "fixed_anchor": _string(fixed_anchor, "fixed_anchor"),
        "fixed_probe_registry": _copy(contract["fixed_probe_registry"]),
    }
    return EVALUATOR_EPOCH_POLICY["epoch_prefix"] + _digest(body)[7:]


def _normalize_input(value: Any) -> dict[str, Any]:
    payload = _copy(_mapping(value, "post_transition_qualification_input"))
    _exact_keys(payload, INPUT_FIELDS, "post_transition_qualification_input")
    if payload.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "qualification input schema differs from frozen V11"
        )
    source = _mapping(payload.get("source_transition"), "source_transition")
    _exact_keys(source, SOURCE_TRANSITION_INPUT_FIELDS, "source_transition")
    for key in (
        "transition_contract_digest",
        "transition_report_digest",
        "source_interval_competition_report_digest",
    ):
        _require_digest(source.get(key), f"source_transition.{key}")
    _string(source.get("disposition"), "source_transition.disposition")
    return payload


def _geometry(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    object_space = _mapping(value.get("object_space"), f"{label}.object_space")
    _exact_keys(object_space, {"feature_ids", "contexts"}, f"{label}.object_space")
    features = _strings(object_space.get("feature_ids"), f"{label}.feature_ids")
    contexts_value = object_space.get("contexts")
    if not isinstance(contexts_value, list) or not contexts_value:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label}.contexts must be a non-empty list"
        )
    if len(contexts_value) > MAX_CONTEXT_COUNT:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label}.contexts exceeds the frozen maximum"
        )
    contexts: list[dict[str, Any]] = []
    context_keys: list[bytes] = []
    for index, raw in enumerate(contexts_value):
        context = _copy(_mapping(raw, f"{label}.contexts[{index}]"))
        _exact_keys(context, set(features), f"{label}.contexts[{index}]")
        contexts.append(context)
        context_keys.append(canonical_json_bytes(context))
    if len(context_keys) != len(set(context_keys)):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label}.contexts are duplicated"
        )
    scopes = _strings(value.get("scope_ids"), f"{label}.scope_ids")
    if len(scopes) > MAX_SCOPE_COUNT:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label}.scope_ids exceeds the frozen maximum"
        )
    probes = _strings(value.get("probe_ids"), f"{label}.probe_ids")
    if probes != PROBE_IDS:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label} does not carry the exact fixed two-Q registry"
        )
    violation_functionals = value.get("violation_functionals")
    if not isinstance(violation_functionals, list) or not violation_functionals:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label}.violation_functionals must be a non-empty list"
        )
    model = _mapping(value.get("model_class"), f"{label}.model_class")
    _exact_keys(
        model,
        {"kind", "center_predictions", "radius_grouping", "radii"},
        f"{label}.model_class",
    )
    if model.get("kind") != "finite_interval_table":
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label}.model_class is not a finite interval table"
        )
    centers_value = model.get("center_predictions")
    if not isinstance(centers_value, list) or not centers_value:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
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
            raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
                f"{label}.center_predictions has duplicate contexts"
            )
        centers[key] = _finite(item.get("value"), f"{label}.center value")
    if set(centers) != set(context_keys):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label}.center_predictions does not cover the exact object space"
        )
    grouping = model.get("radius_grouping")
    if grouping not in {"global", "per_scope", "per_context"}:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label}.radius_grouping is unsupported"
        )
    radii_value = model.get("radii")
    if not isinstance(radii_value, list) or not radii_value:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label}.radii must be non-empty"
        )
    radii: dict[bytes, tuple[dict[str, Any], float]] = {}
    for index, raw in enumerate(radii_value):
        item = _mapping(raw, f"{label}.radii[{index}]")
        _exact_keys(item, {"group", "radius"}, f"{label}.radii[{index}]")
        group = _copy(_mapping(item.get("group"), f"{label}.radius group"))
        radius = _finite(item.get("radius"), f"{label}.radius")
        if radius < 0.0:
            raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
                f"{label}.radius must be nonnegative"
            )
        key = canonical_json_bytes(group)
        if key in radii:
            raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
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
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            f"{label}.radius groups do not exactly cover the registered grouping"
        )
    result = {
        "object_space": _copy(object_space),
        "features": features,
        "contexts": contexts,
        "context_keys": set(context_keys),
        "scopes": scopes,
        "probe_ids": probes,
        "violation_functionals": _copy(violation_functionals),
        "model": _copy(model),
        "centers": centers,
        "grouping": grouping,
        "radii": radii,
    }
    for scope in scopes:
        for context in contexts:
            _radius_for_pair(result, scope, context)
    return result


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
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "radius lookup is incomplete"
        ) from exc


def _functional_threshold(geometry: Mapping[str, Any]) -> float:
    matches: list[float] = []
    for raw in geometry["violation_functionals"]:
        item = _mapping(raw, "violation functional")
        if item.get("functional_id") == "absolute_error":
            matches.append(_finite(item.get("threshold"), "absolute_error threshold"))
    if len(matches) != 1 or matches[0] < 0.0:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "one nonnegative absolute_error threshold is required"
        )
    return matches[0]


def _prediction_scale(parent: Mapping[str, Any], epsilon: float) -> float:
    center_mean = _mean(
        [abs(value) for value in parent["centers"].values()],
        "mean absolute parent center",
    )
    threshold = _functional_threshold(parent)
    try:
        base = center_mean + threshold
    except OverflowError as exc:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "prediction scale base overflowed finite arithmetic"
        ) from exc
    if not math.isfinite(base):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "prediction scale base is not finite"
        )
    result = max(epsilon, base)
    if not math.isfinite(result) or result <= 0.0:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "prediction scale is not positive finite"
        )
    return result


def _project_context(
    context: Mapping[str, Any], child_features: Sequence[str]
) -> dict[str, Any]:
    return {feature: _copy(context[feature]) for feature in child_features}


def _row_values(
    row: Mapping[str, Any],
    geometry: Mapping[str, Any],
    scale: float,
    *,
    project_parent_context: bool,
) -> dict[str, Any]:
    context = (
        _project_context(row["context"], geometry["features"])
        if project_parent_context
        else row["context"]
    )
    key = canonical_json_bytes(context)
    try:
        center = float(geometry["centers"][key])
    except KeyError as exc:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "row context does not map to the evaluated interval table"
        ) from exc
    radius = _radius_for_pair(geometry, str(row["scope_id"]), context)
    observed = _finite(row["observed_value"], "observed_value")
    error = abs(observed - center)
    if not math.isfinite(error):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "absolute center error is not finite"
        )
    raw_violation = error > radius
    raw_exceedance = error - radius if raw_violation else 0.0
    if not math.isfinite(raw_exceedance):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "raw boundary exceedance is not finite"
        )
    q1 = error / scale
    q2 = (radius - error) / scale
    normalized_exceedance = raw_exceedance / scale
    if not all(math.isfinite(item) for item in (q1, q2, normalized_exceedance)):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "native interval Q values are not finite"
        )
    return {
        "context": _copy(context),
        "center": center,
        "radius": radius,
        "absolute_error": error,
        "q1_normalized_absolute_error": q1,
        "q2_normalized_signed_boundary_margin": q2,
        "raw_boundary_violation": raw_violation,
        "raw_boundary_exceedance": raw_exceedance,
        "normalized_boundary_exceedance": normalized_exceedance,
    }


def _source_tail(
    rows: Sequence[Mapping[str, Any]],
    parent: Mapping[str, Any],
    scale: float,
    fraction: float,
) -> tuple[set[int], dict[str, Any]]:
    if not rows:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "source tail requires nonempty rows"
        )
    values = [
        _row_values(row, parent, scale, project_parent_context=False)[
            "raw_boundary_exceedance"
        ]
        for row in rows
    ]
    k = max(1, math.ceil(len(rows) * fraction))
    cutoff = sorted(values, reverse=True)[k - 1]
    indices = {index for index, value in enumerate(values) if value >= cutoff}
    return indices, {
        "source_tail_statistic": "raw_parent_boundary_exceedance",
        "cutoff_units": "source_prediction_units",
        "source_tail_fraction": fraction,
        "source_tail_k": k,
        "source_tail_cutoff": cutoff,
        "source_tail_row_count": len(indices),
        "tail_tie_policy": "INCLUDE_ALL_AT_OR_ABOVE_SOURCE_CUTOFF",
        "observation_id_used_for_membership": False,
    }


def _split_metrics(
    rows: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    scale: float,
    tail_indices: set[int],
    *,
    project_parent_context: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = [
        _row_values(
            row,
            geometry,
            scale,
            project_parent_context=project_parent_context,
        )
        for row in rows
    ]
    errors = [item["absolute_error"] for item in values]
    radii = [item["radius"] for item in values]
    raw_exceedances = [item["raw_boundary_exceedance"] for item in values]
    exceedances = [item["normalized_boundary_exceedance"] for item in values]
    violation_ids = [
        str(row["observation_id"])
        for row, item in zip(rows, values)
        if item["raw_boundary_violation"]
    ]
    mean_error = _mean(errors, "mean raw absolute center error")
    mean_radius = _mean(radii, "mean raw interval radius")
    mean_raw_exceedance = _mean(
        raw_exceedances, "mean raw positive boundary exceedance"
    )
    tail_passes = [
        not values[index]["raw_boundary_violation"] for index in sorted(tail_indices)
    ]
    metrics = {
        "row_count": len(rows),
        "mean_absolute_center_error": mean_error,
        "mean_normalized_center_error": mean_error / scale,
        "max_absolute_center_error": max(errors),
        "max_normalized_center_error": max(errors) / scale,
        "raw_boundary_violation_count": len(violation_ids),
        "raw_boundary_violation_rate": len(violation_ids) / len(rows),
        "raw_boundary_coverage": 1.0 - len(violation_ids) / len(rows),
        "raw_boundary_counterexample_observation_ids": sorted(violation_ids),
        "minimum_normalized_signed_boundary_margin": min(
            item["q2_normalized_signed_boundary_margin"] for item in values
        ),
        "mean_normalized_boundary_exceedance": mean_raw_exceedance / scale,
        "max_normalized_boundary_exceedance": max(exceedances),
        "mean_interval_radius": mean_radius,
        "mean_normalized_radius": mean_radius / scale,
        "source_tail_row_count": len(tail_indices),
        "source_tail_coverage": sum(tail_passes) / len(tail_passes),
    }
    if not all(
        math.isfinite(float(value))
        for key, value in metrics.items()
        if key
        not in {
            "raw_boundary_counterexample_observation_ids",
        }
        and type(value) in (int, float)
    ):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "split metrics contain nonfinite derived arithmetic"
        )
    return metrics, values


def _probe_divergence(
    parent_values: Sequence[Mapping[str, Any]],
    child_values: Sequence[Mapping[str, Any]],
    scale: float,
) -> dict[str, Any]:
    q1: list[float] = []
    q2: list[float] = []
    for parent, child in zip(parent_values, child_values):
        raw_delta = abs(
            float(child["absolute_error"]) - float(parent["absolute_error"])
        )
        d1 = raw_delta / scale
        d2 = abs(
            float(child["q2_normalized_signed_boundary_margin"])
            - float(parent["q2_normalized_signed_boundary_margin"])
        )
        if not math.isfinite(d1) or not math.isfinite(d2):
            raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
                "probe divergence is not finite"
            )
        q1.append(d1)
        q2.append(d2)
    q1_max = max(q1)
    q2_max = max(q2)
    return {
        "normalized_absolute_error_q_max_divergence": q1_max,
        "normalized_margin_q_max_divergence": q2_max,
        "max_probe_divergence": max(q1_max, q2_max),
        "divergence_units": "dimensionless",
    }


def _evaluate_split(
    split: str,
    rows: Sequence[Mapping[str, Any]],
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
    scale: float,
    source_registry: Mapping[str, Any],
    transition_report: Mapping[str, Any],
    uniform_contraction: float,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    tail_indices, tail_definition = _source_tail(
        rows,
        parent,
        scale,
        float(contract["thresholds"]["source_tail_fraction"]),
    )
    parent_metrics, parent_values = _split_metrics(
        rows,
        parent,
        scale,
        tail_indices,
        project_parent_context=False,
    )
    quotient = source_registry["candidate_family"] == "interval_quotient"
    child_metrics, child_values = _split_metrics(
        rows,
        child,
        scale,
        tail_indices,
        project_parent_context=quotient,
    )
    divergence = _probe_divergence(parent_values, child_values, scale)
    delta_mae = (
        float(parent_metrics["mean_normalized_center_error"])
        - float(child_metrics["mean_normalized_center_error"])
    )
    delta_coverage = (
        float(child_metrics["raw_boundary_coverage"])
        - float(parent_metrics["raw_boundary_coverage"])
    )
    delta_tail = (
        float(child_metrics["source_tail_coverage"])
        - float(parent_metrics["source_tail_coverage"])
    )
    radius_change = (
        float(child_metrics["mean_normalized_radius"])
        - float(parent_metrics["mean_normalized_radius"])
    )
    radius_expansion = max(0.0, radius_change)
    radius_reduction = max(0.0, -radius_change)
    context_reduction = (
        1.0 - len(child["contexts"]) / len(parent["contexts"])
        if quotient
        else 0.0
    )
    gate_values = (
        delta_mae,
        delta_coverage,
        delta_tail,
        radius_change,
        radius_expansion,
        radius_reduction,
        context_reduction,
        uniform_contraction,
        divergence["max_probe_divergence"],
    )
    if not all(math.isfinite(float(value)) for value in gate_values):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "qualification gate metrics contain nonfinite arithmetic"
        )
    components: dict[str, float] | None = None
    score: float | None = None
    if split == "holdout":
        components = {
            "normalized_center_mae_gain": delta_mae,
            "raw_boundary_coverage_gain": delta_coverage,
            "source_tail_coverage_gain": delta_tail,
            "context_reduction_fraction": context_reduction,
            "uniform_contraction_fraction": uniform_contraction,
            "normalized_radius_reduction": radius_reduction,
            "max_probe_divergence": divergence["max_probe_divergence"],
            "normalized_radius_expansion": radius_expansion,
        }
        weights = contract["interval_scoring_policy"]["score_weights"]
        score = (
            float(weights["normalized_center_mae_gain"]) * delta_mae
            + float(weights["raw_boundary_coverage_gain"]) * delta_coverage
            + float(weights["source_tail_coverage_gain"]) * delta_tail
            + float(weights["context_reduction_fraction"]) * context_reduction
            + float(weights["uniform_contraction_fraction"]) * uniform_contraction
            + float(weights["normalized_radius_reduction"]) * radius_reduction
            - float(weights["max_probe_divergence_penalty"])
            * float(divergence["max_probe_divergence"])
            - float(weights["normalized_radius_expansion_penalty"])
            * radius_expansion
        )
        if not math.isfinite(score):
            raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
                "qualification score is not finite"
            )
    thresholds = contract["thresholds"]
    common = {
        "minimum_raw_boundary_coverage": child_metrics["raw_boundary_coverage"]
        >= float(thresholds["minimum_raw_boundary_coverage"]),
        "minimum_source_tail_coverage": child_metrics["source_tail_coverage"]
        >= float(thresholds["minimum_source_tail_coverage"]),
        "maximum_normalized_center_mae_increase": -delta_mae
        <= float(thresholds["maximum_normalized_center_mae_increase"]),
        "maximum_normalized_q1_divergence": divergence[
            "normalized_absolute_error_q_max_divergence"
        ]
        <= float(thresholds["maximum_normalized_q1_divergence"]),
        "maximum_normalized_q2_divergence": divergence[
            "normalized_margin_q_max_divergence"
        ]
        <= float(thresholds["maximum_normalized_q2_divergence"]),
    }
    if split == "holdout":
        assert score is not None
        common["minimum_holdout_qualification_score"] = score >= float(
            thresholds["minimum_holdout_qualification_score"]
        )
    preservation = _mapping(
        transition_report.get("preservation_certificate"),
        "source transition preservation_certificate",
    )
    family_receipt = _mapping(
        preservation.get("family_certificate"), "source family_certificate"
    )
    family = source_registry["candidate_family"]
    if family == "interval_robustify":
        family_gates = {
            "strict_v10_expansion_certificate": family_receipt.get(
                "at_least_one_strictly_expanded"
            )
            is True
            and family_receipt.get("all_child_intervals_contain_parent_intervals")
            is True,
            "minimum_coverage_or_tail_gain": max(delta_coverage, delta_tail)
            >= float(thresholds["expansion_minimum_coverage_or_tail_gain"]),
            "maximum_normalized_radius_increase": radius_expansion
            <= float(thresholds["expansion_maximum_normalized_radius_increase"]),
        }
    elif family == "interval_restrict":
        family_gates = {
            "strict_v10_restriction_certificate": family_receipt.get(
                "at_least_one_strictly_restricted"
            )
            is True
            and family_receipt.get("all_child_intervals_within_parent_intervals")
            is True,
            "zero_fresh_raw_boundary_violation_rate": child_metrics[
                "raw_boundary_violation_rate"
            ]
            == float(
                thresholds["restriction_required_raw_boundary_violation_rate"]
            ),
        }
    else:
        family_gates = {
            "v10_global_envelope_certificate": family_receipt.get(
                "all_parent_intervals_contained_under_quotient_map"
            )
            is True,
            "minimum_context_reduction": context_reduction
            >= float(thresholds["quotient_minimum_context_reduction_fraction"]),
            "coverage_not_below_parent": delta_coverage >= 0.0,
            "tail_coverage_not_below_parent": delta_tail >= 0.0,
        }
    gates = {"common": common, "family": family_gates}
    passed = all(common.values()) and all(family_gates.values())
    return {
        "status": (
            f"FRESH_{split.upper()}_PASSED"
            if passed
            else f"FRESH_{split.upper()}_FAILED"
        ),
        "source_tail_definition": tail_definition,
        "parent_metrics": parent_metrics,
        "child_metrics": child_metrics,
        "probe_divergence": divergence,
        "dimensionless_score_components": components,
        "qualification_score": score,
        "qualification_score_units": "dimensionless" if split == "holdout" else None,
        "gates": gates,
        "all_gates_passed": passed,
    }


def _collect_strings(value: Any, key_name: str) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == key_name and isinstance(item, str) and item:
                result.add(item)
            result.update(_collect_strings(item, key_name))
    elif isinstance(value, list):
        for item in value:
            result.update(_collect_strings(item, key_name))
    return result


def _validate_materialized_input(
    payload: Mapping[str, Any],
    parent: Mapping[str, Any],
    expected_qualification_id: str,
    expected_epoch: str,
    expected_exclusion: Mapping[str, Any],
    old_ids: set[str],
    old_epochs: set[str],
) -> tuple[
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
    bool,
    bool,
]:
    if payload.get("qualification_id") != expected_qualification_id:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "qualification_id is not canonically derived"
        )
    evaluator = _copy(_mapping(payload.get("evaluator"), "evaluator"))
    _exact_keys(evaluator, EVALUATOR_FIELDS, "evaluator")
    supplied_epoch = _string(evaluator.get("evaluator_epoch"), "evaluator_epoch")
    supplied_anchor = _string(evaluator.get("fixed_anchor"), "fixed_anchor")
    exclusion = _copy(
        _mapping(payload.get("source_evidence_exclusion"), "source_evidence_exclusion")
    )
    _exact_keys(exclusion, EXCLUSION_FIELDS, "source_evidence_exclusion")
    if (
        exclusion.get("policy") != EXCLUSION_POLICY
        or canonical_json_bytes(
            exclusion.get("five_prior_generation_observation_id_digests")
        )
        != canonical_json_bytes(
            expected_exclusion["five_prior_generation_observation_id_digests"]
        )
        or canonical_json_bytes(exclusion.get("v2_competition_evidence_digests"))
        != canonical_json_bytes(
            expected_exclusion["v2_competition_evidence_digests"]
        )
    ):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "source evidence exclusion differs from the exact V10 lifecycle"
        )
    evidence_value = _mapping(payload.get("evidence"), "evidence")
    _exact_keys(evidence_value, EVIDENCE_FIELDS, "evidence")
    parent_geometry = _geometry(parent, "parent_theory_state")
    expected_cells = {
        (scope, canonical_json_bytes(context)): 0
        for scope in parent_geometry["scopes"]
        for context in parent_geometry["contexts"]
    }
    evidence: dict[str, list[dict[str, Any]]] = {}
    observation_ids: list[str] = []
    all_rows_epoch_exact = True
    all_rows_anchor_exact = True
    coverage: dict[str, Any] = {}
    for split in SPLITS:
        rows_value = evidence_value.get(split)
        if not isinstance(rows_value, list):
            raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
                f"evidence.{split} must be a list"
            )
        rows: list[dict[str, Any]] = []
        counts = dict(expected_cells)
        for index, raw in enumerate(rows_value):
            row = _copy(_mapping(raw, f"evidence.{split}[{index}]"))
            _exact_keys(row, ROW_FIELDS, f"evidence.{split}[{index}]")
            observation_ids.append(
                _string(row.get("observation_id"), f"{split} observation_id")
            )
            _string(row.get("evaluator_epoch"), f"{split} evaluator_epoch")
            _string(row.get("fixed_anchor"), f"{split} fixed_anchor")
            scope = _string(row.get("scope_id"), f"{split} scope_id")
            context = _mapping(row.get("context"), f"{split} context")
            _finite(row.get("observed_value"), f"{split} observed_value")
            if scope not in parent_geometry["scopes"]:
                raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
                    f"{split} row scope is unregistered"
                )
            context_key = canonical_json_bytes(context)
            if context_key not in parent_geometry["context_keys"]:
                raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
                    f"{split} row context is not an exact parent context"
                )
            counts[(scope, context_key)] += 1
            all_rows_epoch_exact = all_rows_epoch_exact and (
                row["evaluator_epoch"] == expected_epoch
            )
            all_rows_anchor_exact = all_rows_anchor_exact and (
                row["fixed_anchor"] == parent["fixed_anchor"]
            )
            rows.append(row)
        evidence[split] = rows
        exact = bool(counts) and all(count == 1 for count in counts.values())
        coverage[split] = {
            "registered_cell_count": len(counts),
            "required_rows_per_cell": 1,
            "expected_row_count": len(counts),
            "actual_row_count": len(rows),
            "minimum_registered_cell_row_count": min(counts.values(), default=0),
            "maximum_registered_cell_row_count": max(counts.values(), default=0),
            "complete_exact_parent_cartesian_coverage": exact,
        }
    if len(observation_ids) != len(set(observation_ids)):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "fresh observation IDs must be globally unique"
        )
    if set(observation_ids) & old_ids:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "fresh observation IDs collide with six source generations"
        )
    if (
        expected_epoch in old_epochs
        or expected_epoch in old_ids
        or expected_qualification_id in old_epochs
        or expected_qualification_id in old_ids
    ):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "derived qualification identity collides with source records"
        )
    comparable = (
        supplied_epoch == expected_epoch
        and supplied_anchor == parent["fixed_anchor"]
        and all_rows_epoch_exact
        and all_rows_anchor_exact
    )
    exact_coverage = all(
        coverage[split]["complete_exact_parent_cartesian_coverage"]
        for split in SPLITS
    )
    coverage["all_exact"] = exact_coverage
    coverage["global_observation_id_unique"] = True
    coverage["fresh_ids_disjoint_from_six_source_generations"] = True
    coverage["declared_epoch_exact"] = supplied_epoch == expected_epoch
    coverage["all_rows_epoch_exact"] = all_rows_epoch_exact
    coverage["declared_fixed_anchor_exact"] = supplied_anchor == parent["fixed_anchor"]
    coverage["all_rows_fixed_anchor_exact"] = all_rows_anchor_exact
    return evaluator, evidence, coverage, comparable, exact_coverage


def _verified_source_transition(
    payload: Mapping[str, Any],
    interval_competition_report_value: Mapping[str, Any],
    interval_transition_contract_value: Mapping[str, Any],
    interval_transition_report_value: Mapping[str, Any],
    receipt_value: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    float,
]:
    """Bind the V11 source input to the exact public V10 verifier receipt."""

    transition_report = _copy(
        _mapping(interval_transition_report_value, "interval_transition_report")
    )
    competition_report = _copy(
        _mapping(interval_competition_report_value, "interval_competition_report")
    )
    receipt = _mapping(receipt_value, "source V10 verifier receipt")
    transition_contract_digest = _digest(interval_transition_contract_value)
    if (
        transition_report.get("schema_version")
        != SOURCE_TRANSITION_REPORT_SCHEMA_VERSION
        or transition_report.get("contract_id") != SOURCE_TRANSITION_CONTRACT_ID
        or transition_report.get("contract_digest")
        != SOURCE_TRANSITION_CONTRACT_DIGEST
        or transition_contract_digest != SOURCE_TRANSITION_CONTRACT_DIGEST
        or receipt.get("contract_digest") != SOURCE_TRANSITION_CONTRACT_DIGEST
        or receipt.get("report_digest") != transition_report.get("report_digest")
        or receipt.get("disposition") != transition_report.get("disposition")
        or receipt.get("status")
        != "VERIFIED_" + str(transition_report.get("disposition"))
        or receipt.get("source_interval_competition_report_digest")
        != competition_report.get("report_digest")
        or transition_report.get("source_interval_competition", {}).get(
            "report_digest"
        )
        != competition_report.get("report_digest")
        or competition_report.get("contract_digest")
        != SOURCE_INTERVAL_COMPETITION_CONTRACT_DIGEST
    ):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "public V10 verifier receipt differs from the supplied source chain"
        )
    disposition = _string(
        transition_report.get("disposition"), "source transition disposition"
    )
    if disposition not in SOURCE_TRANSITION_REGISTRY:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "source transition disposition is outside the frozen ten-route registry"
        )
    registry = _copy(SOURCE_TRANSITION_REGISTRY[disposition])
    expected_source = {
        "transition_contract_digest": SOURCE_TRANSITION_CONTRACT_DIGEST,
        "transition_report_digest": transition_report["report_digest"],
        "source_interval_competition_report_digest": competition_report[
            "report_digest"
        ],
        "disposition": disposition,
        "operation_kind": transition_report.get("operation_kind"),
        "transition_kind": transition_report.get("transition_kind"),
        "selected_candidate_id": transition_report.get("selected_candidate_id"),
        "selected_candidate_family": transition_report.get(
            "selected_candidate_family"
        ),
        "parent_theory_state_digest": transition_report.get(
            "parent_theory_state_digest"
        ),
        "child_theory_state_digest": transition_report.get(
            "child_theory_state_digest"
        ),
    }
    if canonical_json_bytes(payload["source_transition"]) != canonical_json_bytes(
        expected_source
    ):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "input source_transition differs from exact verified V10 report"
        )
    materialized = bool(registry["child_materialized"])
    if (
        bool(receipt.get("transition_materialized")) != materialized
        or (transition_report.get("child_theory_state") is not None) != materialized
    ):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "source child nullability differs from the transition disposition"
        )
    binding = {
        "verification_status": receipt["status"],
        "public_exact_replay_performed": True,
        "transition_contract_id": SOURCE_TRANSITION_CONTRACT_ID,
        "transition_contract_digest": SOURCE_TRANSITION_CONTRACT_DIGEST,
        "transition_report_schema_version": transition_report["schema_version"],
        "transition_report_digest": transition_report["report_digest"],
        "source_interval_competition_contract_digest": (
            SOURCE_INTERVAL_COMPETITION_CONTRACT_DIGEST
        ),
        "source_interval_competition_report_digest": competition_report[
            "report_digest"
        ],
        "disposition": disposition,
        "transition_materialized": materialized,
        "operation_kind": transition_report.get("operation_kind"),
        "transition_kind": transition_report.get("transition_kind"),
        "selected_candidate_id": transition_report.get("selected_candidate_id"),
        "selected_candidate_family": transition_report.get(
            "selected_candidate_family"
        ),
        "parent_theory_state_digest": transition_report.get(
            "parent_theory_state_digest"
        ),
        "child_theory_state_digest": transition_report.get(
            "child_theory_state_digest"
        ),
    }
    if not materialized:
        for key in (
            "operation_kind",
            "transition_kind",
            "selected_candidate_id",
            "selected_candidate_family",
            "parent_theory_state_digest",
            "child_theory_state_digest",
        ):
            if expected_source[key] is not None:
                raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
                    "nonmaterialized V10 report has a nonnull child binding"
                )
        if any(
            payload.get(key) is not None
            for key in (
                "qualification_id",
                "evaluator",
                "source_evidence_exclusion",
                "evidence",
            )
        ):
            raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
                "no-child input must keep qualification ID, evaluator, exclusion, and evidence null"
            )
        return binding, registry, None, None, None, None, 0.0

    if (
        transition_report.get("operation_kind") != registry["operation_kind"]
        or transition_report.get("transition_kind") != registry["transition_kind"]
        or transition_report.get("selected_candidate_family")
        != registry["candidate_family"]
        or receipt.get("selected_candidate_id")
        != transition_report.get("selected_candidate_id")
        or receipt.get("selected_operation_kind")
        != transition_report.get("operation_kind")
        or receipt.get("transition_kind") != transition_report.get("transition_kind")
        or receipt.get("child_theory_state_digest")
        != transition_report.get("child_theory_state_digest")
    ):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "materialized source bindings differ from the frozen family registry"
        )
    parent = _copy(
        _mapping(transition_report.get("parent_theory_state"), "parent theory state")
    )
    child = _copy(
        _mapping(transition_report.get("child_theory_state"), "child theory state")
    )
    _string(parent.get("schema_version"), "parent schema_version")
    if (
        child.get("schema_version") != CHILD_THEORY_SCHEMA_VERSION
        or _digest(parent) != transition_report.get("parent_theory_state_digest")
        or _digest(child) != transition_report.get("child_theory_state_digest")
        or parent.get("fixed_anchor") != child.get("fixed_anchor")
        or parent.get("fixed_anchor") is None
    ):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "verified parent/child state identity is inconsistent"
        )
    parent_geometry = _geometry(parent, "parent_theory_state")
    child_geometry = _geometry(child, "child_theory_state")
    selected = _copy(
        _mapping(
            competition_report.get("selected_candidate"),
            "source selected_candidate",
        )
    )
    if (
        selected.get("candidate_id")
        != transition_report.get("selected_candidate_id")
        or selected.get("candidate_family") != registry["candidate_family"]
        or selected.get("operation_kind") != registry["operation_kind"]
    ):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "source selected candidate differs from the materialized child binding"
        )
    uniform_contraction = 0.0
    if registry["candidate_family"] == "interval_restrict":
        construction = _mapping(
            selected.get("construction"), "restriction candidate construction"
        )
        multiplier = _finite(
            construction.get("radius_multiplier"), "selected radius_multiplier"
        )
        if not 0.0 < multiplier < 1.0:
            raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
                "selected restriction multiplier is not a strict contraction"
            )
        uniform_contraction = 1.0 - multiplier
    lifecycle = _mapping(
        transition_report.get("record_lifecycle_extension"),
        "source transition record_lifecycle_extension",
    )
    expected_exclusion = {
        "policy": (
            EXCLUSION_POLICY
        ),
        "five_prior_generation_observation_id_digests": _copy(
            lifecycle.get("five_prior_generation_exclusion")
        ),
        "v2_competition_evidence_digests": _copy(
            lifecycle.get("v2_competition_evidence_digests")
        ),
    }
    reuse = _mapping(child.get("evidence_reuse_policy"), "child evidence_reuse_policy")
    if (
        canonical_json_bytes(reuse.get("five_prior_generation_exclusion"))
        != canonical_json_bytes(
            expected_exclusion["five_prior_generation_observation_id_digests"]
        )
        or canonical_json_bytes(reuse.get("v2_competition_evidence_digests"))
        != canonical_json_bytes(expected_exclusion["v2_competition_evidence_digests"])
    ):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "child evidence exclusion differs from the V10 lifecycle"
        )
    return (
        binding,
        registry,
        parent,
        child,
        parent_geometry,
        child_geometry,
        uniform_contraction,
    )


def _evaluator_surfaces(
    qualification_id: str,
    expected_epoch: str,
    fixed_anchor: str,
    source_competition_report_digest: str,
    evaluator: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    definition = {
        "schema_version": EVALUATOR_DEFINITION_SCHEMA_VERSION,
        "qualification_id": qualification_id,
        "source_interval_competition_contract_digest": (
            SOURCE_INTERVAL_COMPETITION_CONTRACT_DIGEST
        ),
        "source_interval_competition_report_digest": (
            source_competition_report_digest
        ),
        "fixed_anchor": fixed_anchor,
        "fixed_probe_registry": _copy(PROBE_IDS),
        "evaluator_epoch": expected_epoch,
    }
    binding = {
        "expected_evaluator_epoch": expected_epoch,
        "supplied_evaluator_epoch": evaluator["evaluator_epoch"],
        "expected_fixed_anchor": fixed_anchor,
        "supplied_fixed_anchor": evaluator["fixed_anchor"],
        "declared_epoch_exact": coverage["declared_epoch_exact"],
        "all_rows_epoch_exact": coverage["all_rows_epoch_exact"],
        "declared_fixed_anchor_exact": coverage["declared_fixed_anchor_exact"],
        "all_rows_fixed_anchor_exact": coverage[
            "all_rows_fixed_anchor_exact"
        ],
        "fresh_from_six_source_generations": True,
        "cross_epoch_pooling_allowed": False,
        "comparable": bool(
            coverage["declared_epoch_exact"]
            and coverage["all_rows_epoch_exact"]
            and coverage["declared_fixed_anchor_exact"]
            and coverage["all_rows_fixed_anchor_exact"]
        ),
    }
    return definition, binding


def _evidence_surfaces(
    evidence: Mapping[str, Sequence[Mapping[str, Any]]],
    coverage: Mapping[str, Any],
    exclusion: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = {
        "required_splits": _copy(SPLITS),
        "exact_rows_per_parent_context_scope_cell": _copy(EXACT_ROWS_PER_CELL),
        "coverage_domain": (
            "EXACT_PARENT_CONTEXT_CROSS_REGISTERED_SCOPE_CARTESIAN"
        ),
        "evidence_digests": {
            split: _digest(list(evidence[split])) for split in SPLITS
        },
        "coverage": {
            "holdout": _copy(coverage["holdout"]),
            "stress": _copy(coverage["stress"]),
            "all_exact": coverage["all_exact"],
        },
        "global_observation_id_unique": coverage[
            "global_observation_id_unique"
        ],
        "fresh_ids_disjoint_from_six_source_generations": coverage[
            "fresh_ids_disjoint_from_six_source_generations"
        ],
        "source_rows_used_for_scoring": False,
        "holdout_role": "POST_TRANSITION_QUALIFICATION_ONLY",
        "stress_role": "UNIQUE_CHILD_CONFIRMATION_ONLY",
    }
    erasure = {
        "policy": exclusion["policy"],
        "five_prior_generation_observation_id_digests": _copy(
            exclusion["five_prior_generation_observation_id_digests"]
        ),
        "v2_competition_evidence_digests": _copy(
            exclusion["v2_competition_evidence_digests"]
        ),
        "six_source_generations_scoring_excluded": True,
        "logical_selective_erasure_applied": True,
        "physical_erasure": "NOT_PERFORMED",
        "cross_epoch_pooling_allowed": False,
    }
    return binding, erasure


def _stress_not_evaluated() -> dict[str, Any]:
    return {
        "status": STRESS_NOT_EVALUATED_STATUS,
        "source_tail_definition": None,
        "parent_metrics": None,
        "child_metrics": None,
        "probe_divergence": None,
        "dimensionless_score_components": None,
        "qualification_score": None,
        "qualification_score_units": None,
        "gates": None,
        "all_gates_passed": None,
    }


def _qualification_binding(
    *,
    qualification_id: str,
    child_digest: str,
    evaluator_epoch: str,
    disposition: str,
    holdout_status: str,
    stress_status: str,
    holdout_evaluated: bool,
    stress_evaluated: bool,
) -> dict[str, Any]:
    return {
        "qualification_id": qualification_id,
        "unique_child_theory_state_digest": child_digest,
        "evaluator_epoch": evaluator_epoch,
        "disposition": disposition,
        "holdout_status": holdout_status,
        "stress_status": stress_status,
        "holdout_evaluated": holdout_evaluated,
        "stress_evaluated": stress_evaluated,
        "qualified": disposition == QUALIFIED,
        "candidate_reselection_performed": False,
        "reranking_performed": False,
        "fallback_candidate_evaluated": False,
        "fallback_candidate_selected": False,
        "theory_materialized_or_rematerialized": False,
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
    _exact_keys(report, REPORT_FIELDS, "post_transition_qualification_report")
    return report


def qualify_shadow_interval_multi_q_post_transition(
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
    post_transition_qualification_input: Mapping[str, Any],
    post_transition_qualification_contract: Mapping[str, Any],
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
    expected_post_transition_qualification_input_digest: str,
    expected_post_transition_qualification_contract_digest: str,
    input_artifacts: Mapping[str, Any] | None = None,
) -> ShadowIntervalMultiQPostTransitionQualificationResult:
    """Exact-replay V10 and qualify only its unique materialized shadow child."""

    normalized_contract = (
        validate_shadow_interval_multi_q_post_transition_qualification_contract(
            post_transition_qualification_contract
        )
    )
    contract_digest = _digest(normalized_contract)
    if contract_digest != _require_digest(
        expected_post_transition_qualification_contract_digest,
        "expected_post_transition_qualification_contract_digest",
    ):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "post-transition qualification contract differs from independent expectation"
        )
    payload = _normalize_input(post_transition_qualification_input)
    qualification_input_digest = _digest(payload)
    if qualification_input_digest != _require_digest(
        expected_post_transition_qualification_input_digest,
        "expected_post_transition_qualification_input_digest",
    ):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "post-transition qualification input differs from independent expectation"
        )
    if input_artifacts is not None and not isinstance(input_artifacts, Mapping):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "input_artifacts must be an object or null"
        )
    artifacts = None if input_artifacts is None else _copy(input_artifacts)
    if _contains_key(artifacts, "observed_value"):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
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
        interval_transition_report,
        post_transition_qualification_input,
        post_transition_qualification_contract,
    )
    snapshots = tuple(canonical_json_bytes(item) for item in source_objects)
    try:
        receipt = verify_shadow_interval_multi_q_theory_transition(
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
            interval_transition_report,
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
            expected_interval_transition_report_digest=(
                expected_interval_transition_report_digest
            ),
            expected_interval_transition_input_artifacts=(
                expected_interval_transition_input_artifacts
            ),
        )
    except ShadowIntervalMultiQTheoryTransitionValidationError as exc:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "source V10 public exact replay failed"
        ) from exc
    if tuple(canonical_json_bytes(item) for item in source_objects) != snapshots:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "source inputs were mutated during public V10 replay"
        )

    (
        source_binding,
        source_registry,
        parent,
        child,
        parent_geometry,
        child_geometry,
        uniform_contraction,
    ) = _verified_source_transition(
        payload,
        interval_competition_report,
        interval_transition_contract,
        interval_transition_report,
        receipt,
    )

    qualification_id: str | None = None
    parent_digest: str | None = None
    child_digest: str | None = None
    operation_kind: str | None = None
    transition_kind: str | None = None
    evaluator_definition: dict[str, Any] | None = None
    evaluator_binding: dict[str, Any] | None = None
    evidence_binding: dict[str, Any] | None = None
    erasure_receipt: dict[str, Any] | None = None
    fixed_probe_registry: list[str] | None = None
    probe_results: dict[str, Any] | None = None
    qualification_binding: dict[str, Any] | None = None
    holdout_evaluated = False
    stress_evaluated = False

    if not source_registry["child_materialized"]:
        disposition = NOT_APPLICABLE
    else:
        assert parent is not None
        assert child is not None
        assert parent_geometry is not None
        assert child_geometry is not None
        parent_digest = _require_digest(
            source_binding["parent_theory_state_digest"],
            "parent_theory_state_digest",
        )
        child_digest = _require_digest(
            source_binding["child_theory_state_digest"],
            "child_theory_state_digest",
        )
        operation_kind = _string(
            source_binding["operation_kind"], "operation_kind"
        )
        transition_kind = _string(
            source_binding["transition_kind"], "transition_kind"
        )
        qualification_id = (
            derive_shadow_interval_multi_q_post_transition_qualification_id(
                source_transition_contract_digest=(
                    source_binding["transition_contract_digest"]
                ),
                source_transition_report_digest=(
                    source_binding["transition_report_digest"]
                ),
                parent_theory_state_digest=parent_digest,
                child_theory_state_digest=child_digest,
                operation_kind=operation_kind,
                transition_kind=transition_kind,
                post_transition_qualification_contract=normalized_contract,
            )
        )
        fixed_anchor = _string(parent.get("fixed_anchor"), "parent fixed_anchor")
        expected_epoch = (
            derive_shadow_interval_multi_q_post_transition_evaluator_epoch(
                qualification_id=qualification_id,
                source_interval_competition_contract_digest=(
                    SOURCE_INTERVAL_COMPETITION_CONTRACT_DIGEST
                ),
                source_interval_competition_report_digest=source_binding[
                    "source_interval_competition_report_digest"
                ],
                fixed_anchor=fixed_anchor,
                post_transition_qualification_contract=normalized_contract,
            )
        )
        source_lifecycle = _mapping(
            interval_transition_report.get("record_lifecycle_extension"),
            "source transition record_lifecycle_extension",
        )
        expected_exclusion = {
            "policy": (
                EXCLUSION_POLICY
            ),
            "five_prior_generation_observation_id_digests": _copy(
                source_lifecycle.get("five_prior_generation_exclusion")
            ),
            "v2_competition_evidence_digests": _copy(
                source_lifecycle.get("v2_competition_evidence_digests")
            ),
        }
        six_source_inputs = (
            competition_input,
            qualification_input,
            probe_input,
            restriction_input,
            adjudication_input,
            interval_competition_input,
        )
        old_ids: set[str] = set()
        old_epochs: set[str] = set()
        for old_source in six_source_inputs:
            old_ids.update(_collect_strings(old_source, "observation_id"))
            old_epochs.update(_collect_strings(old_source, "evaluator_epoch"))
        evaluator, evidence, coverage, comparable, exact_coverage = (
            _validate_materialized_input(
                payload,
                parent,
                qualification_id,
                expected_epoch,
                expected_exclusion,
                old_ids,
                old_epochs,
            )
        )
        evaluator_definition, evaluator_binding = _evaluator_surfaces(
            qualification_id,
            expected_epoch,
            fixed_anchor,
            source_binding["source_interval_competition_report_digest"],
            evaluator,
            coverage,
        )
        evidence_binding, erasure_receipt = _evidence_surfaces(
            evidence,
            coverage,
            _mapping(payload["source_evidence_exclusion"], "source exclusion"),
        )
        fixed_probe_registry = _copy(PROBE_IDS)
        if not comparable:
            disposition = INCOMPARABLE
            qualification_binding = _qualification_binding(
                qualification_id=qualification_id,
                child_digest=child_digest,
                evaluator_epoch=expected_epoch,
                disposition=disposition,
                holdout_status=(
                    "NOT_EVALUATED_INCOMPARABLE_FRESH_EVALUATOR_EPOCH"
                ),
                stress_status=(
                    "NOT_EVALUATED_INCOMPARABLE_FRESH_EVALUATOR_EPOCH"
                ),
                holdout_evaluated=False,
                stress_evaluated=False,
            )
        elif not exact_coverage:
            disposition = NEEDS_EVIDENCE
            qualification_binding = _qualification_binding(
                qualification_id=qualification_id,
                child_digest=child_digest,
                evaluator_epoch=expected_epoch,
                disposition=disposition,
                holdout_status="NOT_EVALUATED_INEXACT_FRESH_EVIDENCE",
                stress_status="NOT_EVALUATED_INEXACT_FRESH_EVIDENCE",
                holdout_evaluated=False,
                stress_evaluated=False,
            )
        else:
            scale = _prediction_scale(
                parent_geometry,
                float(normalized_contract["thresholds"]["numeric_epsilon"]),
            )
            holdout = _evaluate_split(
                "holdout",
                evidence["holdout"],
                parent_geometry,
                child_geometry,
                scale,
                source_registry,
                interval_transition_report,
                uniform_contraction,
                normalized_contract,
            )
            holdout_evaluated = True
            if not holdout["all_gates_passed"]:
                disposition = FAILED_HOLDOUT
                stress = _stress_not_evaluated()
            else:
                stress = _evaluate_split(
                    "stress",
                    evidence["stress"],
                    parent_geometry,
                    child_geometry,
                    scale,
                    source_registry,
                    interval_transition_report,
                    uniform_contraction,
                    normalized_contract,
                )
                stress_evaluated = True
                disposition = (
                    QUALIFIED if stress["all_gates_passed"] else FAILED_STRESS
                )
            probe_results = {
                "prediction_scale": scale,
                "prediction_scale_units": "source_prediction_units",
                "holdout": holdout,
                "stress": stress,
            }
            qualification_binding = _qualification_binding(
                qualification_id=qualification_id,
                child_digest=child_digest,
                evaluator_epoch=expected_epoch,
                disposition=disposition,
                holdout_status=holdout["status"],
                stress_status=stress["status"],
                holdout_evaluated=holdout_evaluated,
                stress_evaluated=stress_evaluated,
            )

    authority = _copy(AUTHORITY_BOUNDARY)
    materialized = bool(source_registry["child_materialized"])
    authority.update(
        {
            "source_transition_public_exact_replay_performed": True,
            "qualification_applicable": materialized,
            "fresh_evaluator_definition_derived": materialized,
            "fresh_evidence_structure_validated": materialized,
            "holdout_qualification_performed": holdout_evaluated,
            "stress_confirmation_performed": stress_evaluated,
            "qualification_succeeded": disposition == QUALIFIED,
            "candidate_synthesis_reselection_ranking_or_fallback_performed": False,
            "theory_materialization_or_rematerialization_performed": False,
            "probe_acquisition_or_environment_execution_performed": False,
            "adoption_eligibility_determined": False,
            "adoption_decided": False,
            "promotion_decided": False,
            "current_pointer_written": False,
            "language_or_predicate_invented": False,
            "parent_child_seed_or_ambient_state_written": False,
        }
    )
    events = [
        _audit_event(
            0,
            "SOURCE_INTERVAL_MULTI_Q_TRANSITION_VERIFIED",
            GENESIS_DIGEST,
            {
                "verification_status": source_binding["verification_status"],
                "source_transition_report_digest": source_binding[
                    "transition_report_digest"
                ],
                "source_disposition": source_binding["disposition"],
                "transition_materialized": materialized,
            },
        )
    ]
    events.append(
        _audit_event(
            1,
            (
                "FRESH_POST_TRANSITION_QUALIFICATION_RESOLVED"
                if materialized
                else "POST_TRANSITION_QUALIFICATION_NOT_APPLICABLE"
            ),
            events[-1]["event_digest"],
            {
                "disposition": disposition,
                "qualification_id": qualification_id,
                "holdout_evaluated": holdout_evaluated,
                "stress_evaluated": stress_evaluated,
                "qualified": disposition == QUALIFIED,
            },
        )
    )
    events.append(
        _audit_event(
            2,
            "EXTERNAL_AUTHORITY_WITHHELD",
            events[-1]["event_digest"],
            {
                "adoption_eligibility": SELECTION["adoption_eligibility"],
                "adoption_status": SELECTION["adoption_status"],
                "promotion_status": SELECTION["promotion_status"],
                "current_status": SELECTION["current_status"],
            },
        )
    )
    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "contract_digest": contract_digest,
        "qualification_input_digest": qualification_input_digest,
        "source_transition": source_binding,
        "qualification_id": qualification_id,
        "parent_theory_state_digest": parent_digest,
        "child_theory_state_digest": child_digest,
        "operation_kind": operation_kind,
        "transition_kind": transition_kind,
        "evaluator_definition": evaluator_definition,
        "evaluator_binding": evaluator_binding,
        "evidence_binding": evidence_binding,
        "selective_erasure_receipt": erasure_receipt,
        "fixed_probe_registry": fixed_probe_registry,
        "probe_results": probe_results,
        "disposition": disposition,
        "qualification_binding": qualification_binding,
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
    if tuple(canonical_json_bytes(item) for item in source_objects) != snapshots:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "source inputs were mutated during post-transition qualification"
        )
    return ShadowIntervalMultiQPostTransitionQualificationResult(report=report)


def verify_shadow_interval_multi_q_post_transition_qualification(
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
    post_transition_qualification_input: Mapping[str, Any],
    post_transition_qualification_contract: Mapping[str, Any],
    post_transition_qualification_report: Mapping[str, Any],
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
    expected_post_transition_qualification_input_digest: str,
    expected_post_transition_qualification_contract_digest: str,
    expected_post_transition_qualification_report_digest: str,
    expected_post_transition_qualification_input_artifacts: (
        Mapping[str, Any] | None
    ),
) -> dict[str, Any]:
    """Replay and byte-compare one V11 report against independent anchors."""

    expected_report_digest = _require_digest(
        expected_post_transition_qualification_report_digest,
        "expected_post_transition_qualification_report_digest",
    )
    supplied = _copy(
        _mapping(
            post_transition_qualification_report,
            "post_transition_qualification_report",
        )
    )
    _exact_keys(
        supplied,
        REPORT_FIELDS,
        "post_transition_qualification_report",
    )
    if (
        expected_post_transition_qualification_input_artifacts is not None
        and not isinstance(
            expected_post_transition_qualification_input_artifacts, Mapping
        )
    ):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "expected qualification input_artifacts must be an object or null"
        )
    expected_artifacts = (
        None
        if expected_post_transition_qualification_input_artifacts is None
        else _copy(expected_post_transition_qualification_input_artifacts)
    )
    if _contains_key(expected_artifacts, "observed_value"):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "expected qualification artifacts must not embed observed evidence values"
        )
    if supplied.get("input_artifacts") != expected_artifacts:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "qualification report input_artifacts differ from independent expectation"
        )
    fresh = qualify_shadow_interval_multi_q_post_transition(
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
        interval_transition_report,
        post_transition_qualification_input,
        post_transition_qualification_contract,
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
        expected_interval_transition_report_digest=(
            expected_interval_transition_report_digest
        ),
        expected_interval_transition_input_artifacts=(
            expected_interval_transition_input_artifacts
        ),
        expected_post_transition_qualification_input_digest=(
            expected_post_transition_qualification_input_digest
        ),
        expected_post_transition_qualification_contract_digest=(
            expected_post_transition_qualification_contract_digest
        ),
        input_artifacts=expected_artifacts,
    ).to_dict()
    if fresh["report_digest"] != expected_report_digest:
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "replayed qualification report digest differs from independent expectation"
        )
    if canonical_json_bytes(supplied) != canonical_json_bytes(fresh):
        raise ShadowIntervalMultiQPostTransitionQualificationValidationError(
            "supplied qualification report differs from exact replay"
        )
    probe_results = fresh["probe_results"]
    holdout_evaluated = probe_results is not None
    stress_evaluated = (
        probe_results is not None
        and probe_results["stress"]["status"]
        != STRESS_NOT_EVALUATED_STATUS
    )
    return {
        "status": "VERIFIED_" + fresh["disposition"],
        "disposition": fresh["disposition"],
        "report_digest": fresh["report_digest"],
        "contract_digest": fresh["contract_digest"],
        "qualification_input_digest": fresh["qualification_input_digest"],
        "source_transition_report_digest": fresh["source_transition"][
            "transition_report_digest"
        ],
        "qualification_id": fresh["qualification_id"],
        "parent_theory_state_digest": fresh["parent_theory_state_digest"],
        "child_theory_state_digest": fresh["child_theory_state_digest"],
        "operation_kind": fresh["operation_kind"],
        "transition_kind": fresh["transition_kind"],
        "holdout_evaluated": holdout_evaluated,
        "stress_confirmation_evaluated": stress_evaluated,
        "qualified": fresh["disposition"] == QUALIFIED,
        "adoption_eligibility": fresh["adoption_eligibility"],
        "adoption_status": fresh["adoption_status"],
        "promotion_status": fresh["promotion_status"],
        "current_status": fresh["current_status"],
    }
