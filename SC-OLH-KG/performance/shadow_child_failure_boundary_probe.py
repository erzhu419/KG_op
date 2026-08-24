"""Expand one verified shadow child with a fixed failure-boundary probe.

This additive core exact-replays the upstream local review packet, compiles
exactly one transition-specific probe, and evaluates only caller-supplied
static rows from a new content-derived epoch.  It never acquires evidence,
mutates the source child, invalidates it, executes rollback or restriction,
adopts or promotes a theory, or writes state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from performance.shadow_child_external_review_packet import (
    ShadowChildExternalReviewPacketValidationError,
    verify_shadow_child_external_review_packet,
)
from performance.theory_operation_competition import (
    CompetitionValidationError,
    canonical_json_bytes,
)


CONTRACT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-child-failure-boundary-probe-contract/1"
)
CONTRACT_ID = "shadow_child_failure_boundary_probe_v1"
SOURCE_REVIEW_CONTRACT_ID = "shadow_child_external_review_packet_v1"
SOURCE_REVIEW_CONTRACT_DIGEST = (
    "sha256:7b438072804c95eee26be901c0839bfea0b65b31824a708940b055ab61f858f1"
)
SOURCE_REVIEW_REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-child-external-review-packet-report/1"
)
INPUT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-child-failure-boundary-probe-input/1"
)
REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-child-failure-boundary-probe-report/1"
)
EXPANDED_THEORY_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-probe-expanded-theory-state/1"
)
GENESIS_DIGEST = "sha256:" + "0" * 64

ADOPTION_ELIGIBILITY = "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED"
ADOPTION_STATUS = "NOT_ADOPTED_SHADOW_ONLY"
PROMOTION_STATUS = "NOT_PROMOTED"
CURRENT_STATUS = "NOT_CURRENT"

TRANSITION_PROBE_REGISTRY = {
    "ROBUST_INTERVAL_EXPANSION": {
        "probe_id": "normalized_signed_interval_boundary_margin",
        "functional": (
            "(radius - abs(observed_value - center)) / prediction_scale"
        ),
        "normalization": "prediction_scale",
        "aggregation": (
            "minimum_signed_margin_and_exceedance_by_fresh_split"
        ),
        "counterexample_rule": {"operator": "<", "threshold": 0.0},
    },
    "QUOTIENT_IDEALIZATION": {
        "probe_id": "deleted_feature_conditional_response_spread",
        "functional": (
            "(max_within_scope_fiber_context_mean - "
            "min_within_scope_fiber_context_mean) / prediction_scale"
        ),
        "normalization": "prediction_scale",
        "aggregation": (
            "maximum_normalized_nontrivial_scope_fiber_spread_by_fresh_split"
        ),
        "counterexample_rule": {"operator": ">", "threshold": 0.2},
    },
}

EVIDENCE_POLICY = {
    "min_holdout_rows": 4,
    "min_stress_rows": 4,
    "require_complete_parent_context_scope_pairs_per_split": True,
    "require_unique_new_observation_ids": True,
    "require_disjoint_from_competition_ids": True,
    "require_disjoint_from_qualification_ids": True,
    "require_exact_derived_epoch": True,
    "require_exact_inherited_fixed_anchor": True,
    "forbid_cross_epoch_pooling": True,
}

THRESHOLDS = {
    "numeric_epsilon": 1e-12,
    "max_normalized_interval_exceedance": 0.0,
    "max_normalized_deleted_feature_spread": 0.20,
}

RECORD_LIFECYCLE_POLICY = {
    "competition_record_role": "AUDIT_ONLY_SCORING_EXCLUDED",
    "qualification_record_role": "CONSUMED_AUDIT_ONLY_SCORING_EXCLUDED",
    "new_probe_record_role": (
        "CONSUMED_FAILURE_BOUNDARY_EVIDENCE_AUDIT_ONLY"
    ),
    "all_record_classes_eligible_for_future_scoring": False,
    "future_scoring_requires_new_unconsumed_evidence": True,
    "cross_epoch_pooling_allowed": False,
    "logical_selective_erasure_applied": True,
    "physical_erasure": "NOT_PERFORMED",
    "physical_retention_attestation": "REQUIRED_NOT_PRESENT",
}

AUTHORITY_BOUNDARY = {
    "scope": "LOCAL_FAILURE_BOUNDARY_PROBE_EXPANSION_ONLY",
    "adoption_eligibility": ADOPTION_ELIGIBILITY,
    "external_adoption_authority": "REQUIRED_NOT_PRESENT",
    "source_child_invalidation_forbidden": True,
    "rollback_execution_forbidden": True,
    "restriction_execution_forbidden": True,
    "adoption_action_forbidden": True,
    "promotion_action_forbidden": True,
    "current_pointer_write_forbidden": True,
}

SELECTION = {
    "no_counterexample_status": (
        "EXPANDED_PROBE_NO_BOUNDARY_COUNTEREXAMPLE_ON_SUPPLIED_EVIDENCE"
    ),
    "counterexample_status": (
        "EXPANDED_PROBE_BOUNDARY_COUNTEREXAMPLE_FOUND_SHADOW_ONLY"
    ),
    "needs_evidence_status": "EXPANDED_PROBE_NEEDS_NEW_EVIDENCE",
    "incomparable_status": (
        "EXPANDED_PROBE_INCOMPARABLE_EVALUATOR_EPOCH"
    ),
    "blocked_status": "PROBE_EXPANSION_BLOCKED_SOURCE_PACKET_NOT_READY",
    "source_ready_status": "READY_FOR_EXTERNAL_REVIEW_PACKET_ONLY",
    "adoption_eligibility": ADOPTION_ELIGIBILITY,
    "adoption_status": ADOPTION_STATUS,
    "promotion_status": PROMOTION_STATUS,
    "current_status": CURRENT_STATUS,
}

MANDATORY_NONCLAIMS = (
    "shadow_only",
    "probe_expansion_only",
    "no_external_probe_acquisition",
    "caller_supplied_static_rows_only",
    "local_probe_evaluation_is_not_external_attestation",
    "no_automatic_child_invalidation",
    "no_rollback_execution",
    "no_model_restriction_execution",
    "no_language_or_predicate_invention",
    "no_adoption_eligibility_determination",
    "no_adoption",
    "no_promotion",
    "no_current_pointer_write",
    "no_parent_child_or_ambient_state_write",
    "no_h_t_to_h_t_plus_1_acceptance",
    "no_external_review_outcome",
    "no_external_data_provenance",
    "no_external_evaluator_authority",
    "no_physical_retention_attestation",
    "no_physical_erasure",
    "no_source_evidence_rescoring",
    "no_consumed_qualification_evidence_rescoring",
    "no_cross_epoch_pooling",
    "no_run_one",
    "no_benchmark_execution",
    "no_scheduler_access",
    "no_network_access",
    "no_operations_research_baseline_or_claim_change",
    "no_paper_promotion",
    "explicit_cli_out_is_only_optional_write",
    "input_artifacts_require_independent_expected_values",
    "report_digest_is_not_a_signature",
    "counterexample_absence_is_not_global_preservation",
    "counterexample_presence_is_not_scientific_falsification",
    "no_domain_safety_or_generalization_claim",
)

REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "contract_digest",
        "probe_input_digest",
        "probe_expansion_id",
        "source_review_packet",
        "source_child_theory_state_digest",
        "transition_kind",
        "probe_definition",
        "probe_expanded_shadow_theory_state",
        "probe_expanded_shadow_theory_state_digest",
        "evaluator_definition",
        "evaluator_binding",
        "evidence_binding",
        "record_lifecycle_extension",
        "probe_results",
        "disposition",
        "boundary_assessment",
        "attestation_boundary",
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

EXPANDED_THEORY_FIELDS = frozenset(
    {
        "schema_version",
        "theory_id",
        "task_id",
        "source_child_theory_state_digest",
        "source_review_packet_digest",
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
        "probe_expansion_lineage",
        "adoption_status",
        "current_status",
    }
)


class ShadowChildFailureBoundaryProbeValidationError(ValueError):
    """Raised when the probe contract, source chain, input, or report fails."""


class ShadowChildFailureBoundaryProbeDisposition(str, Enum):
    EXPANDED_PROBE_NO_BOUNDARY_COUNTEREXAMPLE_ON_SUPPLIED_EVIDENCE = (
        "EXPANDED_PROBE_NO_BOUNDARY_COUNTEREXAMPLE_ON_SUPPLIED_EVIDENCE"
    )
    EXPANDED_PROBE_BOUNDARY_COUNTEREXAMPLE_FOUND_SHADOW_ONLY = (
        "EXPANDED_PROBE_BOUNDARY_COUNTEREXAMPLE_FOUND_SHADOW_ONLY"
    )
    EXPANDED_PROBE_NEEDS_NEW_EVIDENCE = (
        "EXPANDED_PROBE_NEEDS_NEW_EVIDENCE"
    )
    EXPANDED_PROBE_INCOMPARABLE_EVALUATOR_EPOCH = (
        "EXPANDED_PROBE_INCOMPARABLE_EVALUATOR_EPOCH"
    )
    PROBE_EXPANSION_BLOCKED_SOURCE_PACKET_NOT_READY = (
        "PROBE_EXPANSION_BLOCKED_SOURCE_PACKET_NOT_READY"
    )


@dataclass(frozen=True)
class ShadowChildFailureBoundaryProbeResult:
    report: Mapping[str, Any]

    @property
    def disposition(self) -> str:
        return str(self.report["disposition"])

    @property
    def report_digest(self) -> str:
        return str(self.report["report_digest"])

    @property
    def probe_expanded(self) -> bool:
        return self.report["probe_expanded_shadow_theory_state"] is not None

    @property
    def boundary_counterexample_found(self) -> bool | None:
        value = self.report["boundary_assessment"][
            "boundary_counterexample_found"
        ]
        return value if type(value) is bool else None

    def to_dict(self) -> dict[str, Any]:
        return _copy(self.report)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShadowChildFailureBoundaryProbeValidationError(
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
        raise ShadowChildFailureBoundaryProbeValidationError(
            f"{label} keys differ; missing={missing}, extra={extra}"
        )


def _string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ShadowChildFailureBoundaryProbeValidationError(
            f"{label} must be a non-empty string"
        )
    return value


def _finite_number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ShadowChildFailureBoundaryProbeValidationError(
            f"{label} must be a finite number"
        )
    return float(value)


def _json_scalar(value: Any, label: str) -> str | int | float | bool:
    if value is None or type(value) not in (str, int, float, bool):
        raise ShadowChildFailureBoundaryProbeValidationError(
            f"{label} must be a non-null finite JSON scalar"
        )
    if type(value) is float and not math.isfinite(value):
        raise ShadowChildFailureBoundaryProbeValidationError(
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
        raise ShadowChildFailureBoundaryProbeValidationError(
            f"value is not detached finite canonical JSON: {exc}"
        ) from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_digest(value: Any, label: str) -> str:
    digest = _string(value, label)
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise ShadowChildFailureBoundaryProbeValidationError(
            f"{label} must be sha256:<64hex>"
        )
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise ShadowChildFailureBoundaryProbeValidationError(
            f"{label} is not hexadecimal"
        ) from exc
    return digest


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ShadowChildFailureBoundaryProbeValidationError(
            f"{label} differs from frozen failure-boundary probe V1"
        )


def _reject_observed_values(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        if "observed_value" in value:
            raise ShadowChildFailureBoundaryProbeValidationError(
                f"{label} must not embed observation values"
            )
        for key, item in value.items():
            _reject_observed_values(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_observed_values(item, f"{label}[{index}]")


def _validate_contract(contract_value: Any) -> dict[str, Any]:
    contract = _mapping(contract_value, "probe_contract")
    _exact_keys(
        contract,
        {
            "schema_version",
            "contract_id",
            "source_review_contract_id",
            "source_review_contract_digest",
            "source_review_report_schema_version",
            "input_schema_version",
            "report_schema_version",
            "transition_probe_registry",
            "evidence_policy",
            "thresholds",
            "record_lifecycle_policy",
            "authority_boundary",
            "selection",
            "nonclaims",
        },
        "probe_contract",
    )
    frozen = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "source_review_contract_id": SOURCE_REVIEW_CONTRACT_ID,
        "source_review_contract_digest": SOURCE_REVIEW_CONTRACT_DIGEST,
        "source_review_report_schema_version": (
            SOURCE_REVIEW_REPORT_SCHEMA_VERSION
        ),
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "report_schema_version": REPORT_SCHEMA_VERSION,
    }
    for key, expected in frozen.items():
        _require_equal(contract[key], expected, key)
    for key, expected in (
        ("transition_probe_registry", TRANSITION_PROBE_REGISTRY),
        ("evidence_policy", EVIDENCE_POLICY),
        ("thresholds", THRESHOLDS),
        ("record_lifecycle_policy", RECORD_LIFECYCLE_POLICY),
        ("authority_boundary", AUTHORITY_BOUNDARY),
        ("selection", SELECTION),
    ):
        _require_equal(_copy(_mapping(contract[key], key)), expected, key)
    nonclaims = contract["nonclaims"]
    if not isinstance(nonclaims, list) or tuple(nonclaims) != MANDATORY_NONCLAIMS:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "nonclaims differ from the exact mandatory V1 list"
        )
    return _copy(contract)


def validate_shadow_child_failure_boundary_probe_contract(
    contract_value: Any,
) -> dict[str, Any]:
    """Validate and detach the frozen failure-boundary probe contract."""

    return _validate_contract(contract_value)


def _epoch_components(
    *,
    review_contract_digest: str,
    review_report_digest: str,
    child_theory_state_digest: str,
    transition_kind: str,
    fixed_anchor: str,
    probe_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    contract = _validate_contract(probe_contract)
    source_contract_digest = _require_digest(
        review_contract_digest, "review_contract_digest"
    )
    if source_contract_digest != SOURCE_REVIEW_CONTRACT_DIGEST:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "review contract digest is not the pinned V1 source"
        )
    source_report_digest = _require_digest(
        review_report_digest, "review_report_digest"
    )
    child_digest = _require_digest(
        child_theory_state_digest, "child_theory_state_digest"
    )
    kind = _string(transition_kind, "transition_kind")
    if kind not in TRANSITION_PROBE_REGISTRY:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "transition_kind has no fixed failure-boundary probe"
        )
    anchor = _string(fixed_anchor, "fixed_anchor")
    expanded_registry = {
        "source_child_theory_state_digest": child_digest,
        "added_probe_definition": _copy(TRANSITION_PROBE_REGISTRY[kind]),
    }
    payload = {
        "review_contract_digest": source_contract_digest,
        "review_report_digest": source_report_digest,
        "child_theory_state_digest": child_digest,
        "transition_kind": kind,
        "fixed_anchor": anchor,
        "probe_contract_digest": _digest(contract),
        "expanded_probe_registry": expanded_registry,
    }
    epoch = "shadow-failure-boundary-probe-epoch:" + _digest(payload)[7:]
    return payload, epoch


def derive_shadow_child_failure_boundary_probe_epoch(
    *,
    review_contract_digest: str,
    review_report_digest: str,
    child_theory_state_digest: str,
    transition_kind: str,
    fixed_anchor: str,
    probe_contract: Mapping[str, Any],
) -> str:
    """Derive the content identity of the local failure-boundary epoch."""

    return _epoch_components(
        review_contract_digest=review_contract_digest,
        review_report_digest=review_report_digest,
        child_theory_state_digest=child_theory_state_digest,
        transition_kind=transition_kind,
        fixed_anchor=fixed_anchor,
        probe_contract=probe_contract,
    )[1]


def _source_observation_ids(competition_input: Mapping[str, Any]) -> list[str]:
    evidence = _mapping(competition_input.get("evidence"), "competition evidence")
    _exact_keys(evidence, {"discovery", "validation", "stress"}, "competition evidence")
    ids: list[str] = []
    for split in ("discovery", "validation", "stress"):
        rows = evidence[split]
        if not isinstance(rows, list):
            raise ShadowChildFailureBoundaryProbeValidationError(
                f"competition evidence.{split} must be a list"
            )
        for index, raw in enumerate(rows):
            row = _mapping(raw, f"competition evidence.{split}[{index}]")
            ids.append(
                _string(
                    row.get("observation_id"),
                    f"competition evidence.{split}[{index}].observation_id",
                )
            )
    if len(ids) != len(set(ids)):
        raise ShadowChildFailureBoundaryProbeValidationError(
            "competition observation IDs are not unique"
        )
    return sorted(ids)


def _qualification_observation_ids(
    qualification_input: Mapping[str, Any],
) -> list[str]:
    evidence = _mapping(
        qualification_input.get("evidence"), "qualification evidence"
    )
    _exact_keys(evidence, {"holdout", "stress"}, "qualification evidence")
    ids: list[str] = []
    for split in ("holdout", "stress"):
        rows = evidence[split]
        if not isinstance(rows, list):
            raise ShadowChildFailureBoundaryProbeValidationError(
                f"qualification evidence.{split} must be a list"
            )
        for index, raw in enumerate(rows):
            row = _mapping(raw, f"qualification evidence.{split}[{index}]")
            ids.append(
                _string(
                    row.get("observation_id"),
                    f"qualification evidence.{split}[{index}].observation_id",
                )
            )
    if len(ids) != len(set(ids)):
        raise ShadowChildFailureBoundaryProbeValidationError(
            "qualification observation IDs are not unique"
        )
    return sorted(ids)


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
        raise ShadowChildFailureBoundaryProbeValidationError(
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
        raise ShadowChildFailureBoundaryProbeValidationError(
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
            raise ShadowChildFailureBoundaryProbeValidationError(
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
                    row["context"], feature_ids, registered_contexts, f"{label}.context"
                ),
                "observed_value": _finite_number(
                    row["observed_value"], f"{label}.observed_value"
                ),
            }
        )
    rows.sort(key=lambda item: item["observation_id"])
    ids = [row["observation_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ShadowChildFailureBoundaryProbeValidationError(
            f"evidence.{split} observation IDs are duplicated"
        )
    return rows


def _normalize_input(
    input_value: Any,
    *,
    review_report: Mapping[str, Any],
    transition_report: Mapping[str, Any],
    competition_input: Mapping[str, Any],
    qualification_input: Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(input_value, "probe_input")
    _exact_keys(
        value,
        {
            "schema_version",
            "probe_expansion_id",
            "source_review_packet",
            "evaluator",
            "prior_record_exclusion",
            "evidence",
        },
        "probe_input",
    )
    _require_equal(value["schema_version"], INPUT_SCHEMA_VERSION, "input schema")
    source_packet = _mapping(
        value["source_review_packet"], "source_review_packet"
    )
    _exact_keys(
        source_packet,
        {
            "review_contract_digest",
            "review_report_digest",
            "packet_id",
            "child_theory_state_digest",
        },
        "source_review_packet",
    )
    expected_packet = {
        "review_contract_digest": review_report["contract_digest"],
        "review_report_digest": review_report["report_digest"],
        "packet_id": review_report["packet_id"],
        "child_theory_state_digest": review_report[
            "child_theory_state_digest"
        ],
    }
    _require_equal(_copy(source_packet), expected_packet, "source_review_packet")

    evaluator = _mapping(value["evaluator"], "evaluator")
    _exact_keys(evaluator, {"evaluator_epoch", "fixed_anchor"}, "evaluator")
    normalized_evaluator = {
        "evaluator_epoch": _string(
            evaluator["evaluator_epoch"], "evaluator.evaluator_epoch"
        ),
        "fixed_anchor": _string(evaluator["fixed_anchor"], "evaluator.fixed_anchor"),
    }

    lifecycle = _mapping(
        review_report.get("record_lifecycle_boundary"),
        "record_lifecycle_boundary",
    )
    source_records = _mapping(
        lifecycle.get("source_competition_records"),
        "source_competition_records",
    )
    qualification_records = _mapping(
        lifecycle.get("qualification_records"), "qualification_records"
    )
    exclusion = _mapping(
        value["prior_record_exclusion"], "prior_record_exclusion"
    )
    _exact_keys(
        exclusion,
        {
            "source_competition_observation_id_digest",
            "consumed_qualification_observation_id_digest",
        },
        "prior_record_exclusion",
    )
    competition_ids = _source_observation_ids(competition_input)
    qualification_ids = _qualification_observation_ids(qualification_input)
    expected_source_id_digest = _digest(competition_ids)
    expected_qualification_id_digest = _digest(qualification_ids)
    if source_records.get("observation_id_digest") != expected_source_id_digest:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "review packet source observation commitment is inconsistent"
        )
    if qualification_records.get("observation_id_digest") != (
        expected_qualification_id_digest
    ):
        raise ShadowChildFailureBoundaryProbeValidationError(
            "review packet qualification observation commitment is inconsistent"
        )
    expected_exclusion = {
        "source_competition_observation_id_digest": expected_source_id_digest,
        "consumed_qualification_observation_id_digest": (
            expected_qualification_id_digest
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
        raise ShadowChildFailureBoundaryProbeValidationError(
            "verified parent lacks a finite feature/context/scope registry"
        )
    feature_ids = [_string(item, "parent feature_id") for item in feature_ids_value]
    if len(feature_ids) != len(set(feature_ids)):
        raise ShadowChildFailureBoundaryProbeValidationError(
            "parent feature IDs are duplicated"
        )
    registered_contexts = {canonical_json_bytes(item) for item in contexts_value}
    if len(registered_contexts) != len(contexts_value):
        raise ShadowChildFailureBoundaryProbeValidationError(
            "parent contexts are duplicated"
        )
    registered_scopes = {_string(item, "parent scope_id") for item in scopes_value}
    if len(registered_scopes) != len(scopes_value):
        raise ShadowChildFailureBoundaryProbeValidationError(
            "parent scopes are duplicated"
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
        raise ShadowChildFailureBoundaryProbeValidationError(
            "new observation IDs are duplicated across splits"
        )
    if set(new_ids) & set(competition_ids):
        raise ShadowChildFailureBoundaryProbeValidationError(
            "new observation IDs reuse source competition records"
        )
    if set(new_ids) & set(qualification_ids):
        raise ShadowChildFailureBoundaryProbeValidationError(
            "new observation IDs reuse consumed qualification records"
        )

    public = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "probe_expansion_id": _string(
            value["probe_expansion_id"], "probe_expansion_id"
        ),
        "source_review_packet": expected_packet,
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
        "parent_contexts": _copy(contexts_value),
        "parent_scopes": sorted(registered_scopes),
    }


def _evaluator_binding(
    normalized_input: Mapping[str, Any], expected_epoch: str, expected_anchor: str
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


def _evidence_binding(
    normalized_input: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    evidence = normalized_input["evidence"]
    policy = contract["evidence_policy"]
    required_by_key = {
        canonical_json_bytes({"scope_id": scope_id, "context": context}): {
            "scope_id": scope_id,
            "context": _copy(context),
        }
        for scope_id in normalized_input["parent_scopes"]
        for context in normalized_input["parent_contexts"]
    }
    required_pairs = [required_by_key[key] for key in sorted(required_by_key)]
    required_keys = set(required_by_key)
    row_counts: dict[str, int] = {}
    minimum_rows = {
        "holdout": int(policy["min_holdout_rows"]),
        "stress": int(policy["min_stress_rows"]),
    }
    minimum_satisfied: dict[str, bool] = {}
    covered_by_split: dict[str, list[dict[str, Any]]] = {}
    missing_by_split: dict[str, list[dict[str, Any]]] = {}
    complete_coverage: dict[str, bool] = {}
    for split, minimum_key in (
        ("holdout", "min_holdout_rows"),
        ("stress", "min_stress_rows"),
    ):
        rows = evidence[split]
        covered_map = {
            canonical_json_bytes(
                {"scope_id": row["scope_id"], "context": row["context"]}
            ): {
                "scope_id": row["scope_id"],
                "context": _copy(row["context"]),
            }
            for row in rows
        }
        covered_keys = set(covered_map)
        row_counts[split] = len(rows)
        minimum_satisfied[split] = len(rows) >= int(policy[minimum_key])
        covered_by_split[split] = [
            covered_map[key] for key in sorted(covered_map)
        ]
        missing_by_split[split] = [
            required_by_key[key] for key in sorted(required_keys - covered_keys)
        ]
        complete_coverage[split] = covered_keys == required_keys
    new_ids = normalized_input["new_ids"]
    complete = all(minimum_satisfied.values()) and all(complete_coverage.values())
    return {
        "holdout_evidence_digest": _digest(evidence["holdout"]),
        "stress_evidence_digest": _digest(evidence["stress"]),
        "new_observation_id_digest": _digest(new_ids),
        "new_observation_count": len(new_ids),
        "source_competition_observation_id_digest": _digest(
            normalized_input["competition_ids"]
        ),
        "consumed_qualification_observation_id_digest": _digest(
            normalized_input["qualification_ids"]
        ),
        "unique_new_observation_ids": True,
        "disjoint_from_competition_ids": True,
        "disjoint_from_qualification_ids": True,
        "cross_epoch_pooling": False,
        "row_counts": row_counts,
        "minimum_rows": minimum_rows,
        "minimum_rows_satisfied_by_split": minimum_satisfied,
        "required_context_scope_pairs": required_pairs,
        "required_context_scope_pair_count": len(required_pairs),
        "covered_context_scope_pairs_by_split": covered_by_split,
        "missing_context_scope_pairs_by_split": missing_by_split,
        "complete_context_scope_coverage_by_split": complete_coverage,
        "complete_evidence": complete,
    }


def _prediction_lookup(value: Any, label: str) -> dict[bytes, float]:
    if not isinstance(value, list):
        raise ShadowChildFailureBoundaryProbeValidationError(
            f"{label} must be a list"
        )
    result: dict[bytes, float] = {}
    for index, raw in enumerate(value):
        item = _mapping(raw, f"{label}[{index}]")
        _exact_keys(item, {"context", "value"}, f"{label}[{index}]")
        context = _mapping(item["context"], f"{label}[{index}].context")
        key = canonical_json_bytes(context)
        if key in result:
            raise ShadowChildFailureBoundaryProbeValidationError(
                f"{label} has duplicate contexts"
            )
        result[key] = _finite_number(item["value"], f"{label}[{index}].value")
    return result


def _functional_threshold(parent: Mapping[str, Any]) -> float:
    value = parent.get("violation_functionals")
    if not isinstance(value, list) or len(value) != 1:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "parent must have exactly one violation functional"
        )
    functional = _mapping(value[0], "parent violation functional")
    if functional.get("functional_id") != "absolute_error":
        raise ShadowChildFailureBoundaryProbeValidationError(
            "V1 prediction scale requires the absolute_error functional"
        )
    threshold = _finite_number(functional.get("threshold"), "absolute_error threshold")
    if threshold < 0:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "absolute_error threshold cannot be negative"
        )
    return threshold


def _prediction_scale(
    parent: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[dict[bytes, float], float]:
    model = _mapping(parent.get("model_class"), "parent model_class")
    if model.get("kind") != "finite_point_table":
        raise ShadowChildFailureBoundaryProbeValidationError(
            "failure-boundary V1 requires a finite point-table parent"
        )
    lookup = _prediction_lookup(model.get("predictions"), "parent predictions")
    if not lookup:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "parent prediction table is empty"
        )
    mean_absolute_prediction = math.fsum(abs(item) for item in lookup.values()) / len(lookup)
    scale = max(
        float(contract["thresholds"]["numeric_epsilon"]),
        mean_absolute_prediction + _functional_threshold(parent),
    )
    return lookup, scale


def _robust_radius(model: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    grouping = model.get("radius_grouping")
    radii = model.get("radii")
    if grouping not in {"global", "per_scope", "per_context"} or not isinstance(
        radii, list
    ):
        raise ShadowChildFailureBoundaryProbeValidationError(
            "robust child radius grammar is invalid"
        )
    matches: list[Any] = []
    for index, raw in enumerate(radii):
        item = _mapping(raw, f"radii[{index}]")
        _exact_keys(item, {"group", "radius"}, f"radii[{index}]")
        group = _mapping(item["group"], f"radii[{index}].group")
        if grouping == "global" and dict(group) == {"global": "*"}:
            matches.append(item["radius"])
        elif grouping == "per_scope" and dict(group) == {
            "scope_id": row["scope_id"]
        }:
            matches.append(item["radius"])
        elif grouping == "per_context" and set(group) == {"context"}:
            if canonical_json_bytes(group["context"]) == canonical_json_bytes(
                row["context"]
            ):
                matches.append(item["radius"])
    if len(matches) != 1:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "robust child has no unique radius for a context-scope pair"
        )
    radius = _finite_number(matches[0], "robust radius")
    if radius < 0:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "robust radius cannot be negative"
        )
    return radius


def _robust_split(
    rows: Sequence[Mapping[str, Any]],
    child: Mapping[str, Any],
    scale: float,
) -> tuple[dict[str, Any], list[str]]:
    model = _mapping(child.get("model_class"), "child model_class")
    if model.get("kind") != "finite_interval_table":
        raise ShadowChildFailureBoundaryProbeValidationError(
            "robust probe requires a finite interval-table child"
        )
    centers = _prediction_lookup(model.get("center_predictions"), "child centers")
    margins: list[float] = []
    exceedances: list[float] = []
    counterexamples: list[str] = []
    for row in rows:
        key = canonical_json_bytes(row["context"])
        if key not in centers:
            raise ShadowChildFailureBoundaryProbeValidationError(
                "robust child centers do not cover a parent context"
            )
        radius = _robust_radius(model, row)
        margin = (
            radius - abs(float(row["observed_value"]) - centers[key])
        ) / scale
        exceedance = max(0.0, -margin)
        margins.append(margin)
        exceedances.append(exceedance)
        if margin < 0.0:
            counterexamples.append(row["observation_id"])
    count = len(rows)
    metrics = {
        "row_count": count,
        "min_normalized_signed_margin": min(margins),
        "boundary_violation_rate": len(counterexamples) / count,
        "mean_normalized_exceedance": math.fsum(exceedances) / count,
        "max_normalized_exceedance": max(exceedances),
    }
    return metrics, sorted(counterexamples)


def _evaluate_robust(
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
    evidence: Mapping[str, Sequence[Mapping[str, Any]]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    _, scale = _prediction_scale(parent, contract)
    holdout, holdout_counterexamples = _robust_split(
        evidence["holdout"], child, scale
    )
    stress, stress_counterexamples = _robust_split(
        evidence["stress"], child, scale
    )
    threshold = float(
        contract["thresholds"]["max_normalized_interval_exceedance"]
    )
    gates = [
        {
            "gate_id": "holdout_normalized_interval_exceedance",
            "metric_path": "holdout.max_normalized_exceedance",
            "operator": "<=",
            "threshold": threshold,
            "actual": holdout["max_normalized_exceedance"],
            "passed": holdout["max_normalized_exceedance"] <= threshold,
        },
        {
            "gate_id": "stress_normalized_interval_exceedance",
            "metric_path": "stress.max_normalized_exceedance",
            "operator": "<=",
            "threshold": threshold,
            "actual": stress["max_normalized_exceedance"],
            "passed": stress["max_normalized_exceedance"] <= threshold,
        },
    ]
    counterexamples = sorted(set(holdout_counterexamples + stress_counterexamples))
    return {
        "result_kind": "ROBUST_INTERVAL_FAILURE_BOUNDARY_PROBE_RESULT",
        "probe_ids": [TRANSITION_PROBE_REGISTRY["ROBUST_INTERVAL_EXPANSION"]["probe_id"]],
        "holdout": holdout,
        "stress": stress,
        "aggregate": {
            "prediction_scale": scale,
            "total_row_count": len(evidence["holdout"]) + len(evidence["stress"]),
            "total_boundary_violation_count": len(counterexamples),
            "max_normalized_exceedance": max(
                holdout["max_normalized_exceedance"],
                stress["max_normalized_exceedance"],
            ),
        },
        "gate_checks": gates,
        "boundary_counterexample_found": bool(counterexamples),
        "counterexample_observation_ids": counterexamples,
    }


def _fiber_lookup(
    transition_report: Mapping[str, Any],
    parent_contexts: Sequence[Mapping[str, Any]],
) -> dict[bytes, dict[str, Any]]:
    reduction = _mapping(
        transition_report.get("reduction_certificate"), "reduction_certificate"
    )
    if reduction.get("map_kind") != "QUOTIENT_PROJECTION_WITH_FROZEN_PARENT_SNAPSHOT":
        raise ShadowChildFailureBoundaryProbeValidationError(
            "quotient child lacks the frozen parent fiber map"
        )
    value = reduction.get("quotient_fiber_map")
    if not isinstance(value, list):
        raise ShadowChildFailureBoundaryProbeValidationError(
            "quotient_fiber_map must be a list"
        )
    result: dict[bytes, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        item = _mapping(raw, f"quotient_fiber_map[{index}]")
        _exact_keys(
            item,
            {"parent_context", "quotient_context"},
            f"quotient_fiber_map[{index}]",
        )
        key = canonical_json_bytes(item["parent_context"])
        if key in result:
            raise ShadowChildFailureBoundaryProbeValidationError(
                "quotient fiber map duplicates a parent context"
            )
        result[key] = _copy(item["quotient_context"])
    expected = {canonical_json_bytes(item) for item in parent_contexts}
    if set(result) != expected:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "quotient fiber map does not cover the parent contexts"
        )
    return result


def _quotient_split(
    rows: Sequence[Mapping[str, Any]],
    fibers: Mapping[bytes, Mapping[str, Any]],
    scale: float,
    threshold: float,
) -> tuple[dict[str, Any], list[str]]:
    # A scope is part of the operational task boundary.  Pooling the same
    # parent context across scopes can make equal-and-opposite conditional
    # effects cancel before the deleted-feature spread is measured.
    by_scope_parent: dict[tuple[str, bytes], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["scope_id"], canonical_json_bytes(row["context"]))
        by_scope_parent.setdefault(key, []).append(row)
    by_scope_fiber: dict[bytes, dict[str, Any]] = {}
    for (scope_id, parent_key), parent_rows in by_scope_parent.items():
        if parent_key not in fibers:
            raise ShadowChildFailureBoundaryProbeValidationError(
                "probe row has no quotient fiber"
            )
        quotient_context = fibers[parent_key]
        scope_fiber_key = canonical_json_bytes(
            {"scope_id": scope_id, "quotient_context": quotient_context}
        )
        group = by_scope_fiber.setdefault(
            scope_fiber_key,
            {
                "scope_id": scope_id,
                "quotient_context": _copy(quotient_context),
                "context_means": [],
            },
        )
        group["context_means"].append(
            {
                "parent_context_digest": _digest(json.loads(parent_key.decode("ascii"))),
                "mean": math.fsum(float(row["observed_value"]) for row in parent_rows)
                / len(parent_rows),
                "observation_ids": sorted(row["observation_id"] for row in parent_rows),
            }
        )
    spreads: list[float] = []
    offending_digests: list[str] = []
    counterexample_ids: list[str] = []
    nontrivial_count = 0
    for scope_fiber_key in sorted(by_scope_fiber):
        group = by_scope_fiber[scope_fiber_key]
        context_means = group["context_means"]
        if len(context_means) < 2:
            continue
        nontrivial_count += 1
        normalized_spread = (
            max(item["mean"] for item in context_means)
            - min(item["mean"] for item in context_means)
        ) / scale
        spreads.append(normalized_spread)
        if normalized_spread > threshold:
            scope_fiber_digest = _digest(
                {
                    "scope_id": group["scope_id"],
                    "quotient_context": group["quotient_context"],
                    "parent_context_digests": sorted(
                        item["parent_context_digest"] for item in context_means
                    ),
                }
            )
            offending_digests.append(scope_fiber_digest)
            counterexample_ids.extend(
                observation_id
                for item in context_means
                for observation_id in item["observation_ids"]
            )
    metrics = {
        "row_count": len(rows),
        "evaluated_nontrivial_fiber_count": nontrivial_count,
        "max_normalized_fiber_response_spread": max(spreads, default=0.0),
        "offending_fiber_count": len(offending_digests),
        "offending_fiber_digests": sorted(offending_digests),
    }
    return metrics, sorted(set(counterexample_ids))


def _evaluate_quotient(
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
    transition_report: Mapping[str, Any],
    evidence: Mapping[str, Sequence[Mapping[str, Any]]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    model = _mapping(child.get("model_class"), "child model_class")
    if model.get("kind") != "finite_point_table":
        raise ShadowChildFailureBoundaryProbeValidationError(
            "quotient probe requires a finite point-table child"
        )
    _prediction_lookup(model.get("predictions"), "quotient child predictions")
    _, scale = _prediction_scale(parent, contract)
    parent_contexts = _mapping(parent["object_space"], "parent object_space")[
        "contexts"
    ]
    fibers = _fiber_lookup(transition_report, parent_contexts)
    threshold = float(
        contract["thresholds"]["max_normalized_deleted_feature_spread"]
    )
    holdout, holdout_counterexamples = _quotient_split(
        evidence["holdout"], fibers, scale, threshold
    )
    stress, stress_counterexamples = _quotient_split(
        evidence["stress"], fibers, scale, threshold
    )
    gates = [
        {
            "gate_id": "holdout_deleted_feature_conditional_response_spread",
            "metric_path": "holdout.max_normalized_fiber_response_spread",
            "operator": "<=",
            "threshold": threshold,
            "actual": holdout["max_normalized_fiber_response_spread"],
            "passed": holdout["max_normalized_fiber_response_spread"] <= threshold,
        },
        {
            "gate_id": "stress_deleted_feature_conditional_response_spread",
            "metric_path": "stress.max_normalized_fiber_response_spread",
            "operator": "<=",
            "threshold": threshold,
            "actual": stress["max_normalized_fiber_response_spread"],
            "passed": stress["max_normalized_fiber_response_spread"] <= threshold,
        },
    ]
    counterexamples = sorted(set(holdout_counterexamples + stress_counterexamples))
    return {
        "result_kind": "QUOTIENT_IDEALIZATION_FAILURE_BOUNDARY_PROBE_RESULT",
        "probe_ids": [TRANSITION_PROBE_REGISTRY["QUOTIENT_IDEALIZATION"]["probe_id"]],
        "holdout": holdout,
        "stress": stress,
        "aggregate": {
            "prediction_scale": scale,
            "evaluated_nontrivial_fiber_count": (
                holdout["evaluated_nontrivial_fiber_count"]
                + stress["evaluated_nontrivial_fiber_count"]
            ),
            "max_normalized_fiber_response_spread": max(
                holdout["max_normalized_fiber_response_spread"],
                stress["max_normalized_fiber_response_spread"],
            ),
            "total_offending_fiber_count": (
                holdout["offending_fiber_count"] + stress["offending_fiber_count"]
            ),
        },
        "gate_checks": gates,
        "boundary_counterexample_found": bool(counterexamples),
        "counterexample_observation_ids": counterexamples,
    }


def _expanded_state(
    *,
    child: Mapping[str, Any],
    child_digest: str,
    review_report: Mapping[str, Any],
    transition_kind: str,
    probe_definition: Mapping[str, Any],
    probe_contract_digest: str,
    probe_expansion_id: str,
    evaluator_epoch: str,
) -> dict[str, Any]:
    source_probe_ids = child.get("probe_ids")
    if not isinstance(source_probe_ids, list):
        raise ShadowChildFailureBoundaryProbeValidationError(
            "source child probe_ids must be a list"
        )
    probe_ids = [_string(item, "source child probe_id") for item in source_probe_ids]
    added_probe_id = _string(probe_definition.get("probe_id"), "added probe_id")
    if added_probe_id in probe_ids:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "transition-specific probe is already present in the source child"
        )
    if len(probe_ids) != len(set(probe_ids)):
        raise ShadowChildFailureBoundaryProbeValidationError(
            "source child probe IDs are duplicated"
        )
    payload = {
        "schema_version": EXPANDED_THEORY_SCHEMA_VERSION,
        "task_id": child["task_id"],
        "source_child_theory_state_digest": child_digest,
        "source_review_packet_digest": review_report["report_digest"],
        "evaluator_epoch": evaluator_epoch,
        "evaluator_status": "LOCAL_FAILURE_BOUNDARY_PROBE_EPOCH_UNATTESTED",
        "fixed_anchor": child["fixed_anchor"],
        "object_space": _copy(child["object_space"]),
        "model_class": _copy(child["model_class"]),
        "probe_ids": probe_ids + [added_probe_id],
        "violation_functionals": _copy(child["violation_functionals"]),
        "scope_ids": _copy(child["scope_ids"]),
        "removable_feature_ids": _copy(child["removable_feature_ids"]),
        "evidence_reuse_policy": {
            "source_competition_records_allowed_for_scoring": False,
            "consumed_qualification_records_allowed_for_scoring": False,
            "failure_boundary_probe_records_allowed_for_future_scoring": False,
            "future_scoring_requires_new_unconsumed_evidence": True,
            "cross_epoch_pooling_allowed": False,
        },
        "operational_probe_status": (
            "FAILURE_BOUNDARY_PROBE_COMPILED_SHADOW_ONLY"
        ),
        "probe_expansion_lineage": {
            "source_review_contract_digest": review_report["contract_digest"],
            "source_review_report_digest": review_report["report_digest"],
            "source_review_packet_id": review_report["packet_id"],
            "source_child_theory_state_digest": child_digest,
            "probe_contract_digest": probe_contract_digest,
            "transition_kind": transition_kind,
            "added_probe_id": added_probe_id,
            "probe_expansion_id": probe_expansion_id,
        },
        "adoption_status": ADOPTION_STATUS,
        "current_status": CURRENT_STATUS,
    }
    theory_id = "shadow-probe-expanded-theory:" + _digest(payload)[7:]
    state = {"schema_version": payload.pop("schema_version"), "theory_id": theory_id}
    state.update(payload)
    _exact_keys(state, EXPANDED_THEORY_FIELDS, "probe_expanded_shadow_theory_state")
    return state


def _source_review_summary(
    review_report: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    boundary = _mapping(review_report.get("review_boundary"), "review_boundary")
    return {
        "verification_status": receipt.get("status"),
        "contract_id": review_report.get("contract_id"),
        "contract_digest": review_report.get("contract_digest"),
        "report_digest": review_report.get("report_digest"),
        "packet_id": review_report.get("packet_id"),
        "child_theory_state_digest": review_report.get(
            "child_theory_state_digest"
        ),
        "disposition": review_report.get("disposition"),
        "packet_ready": boundary.get("packet_ready"),
        "adoption_status": review_report.get("adoption_status"),
    }


def _record_lifecycle_extension(
    *,
    review_report: Mapping[str, Any],
    evidence_binding: Mapping[str, Any],
    evaluator_epoch: str,
    source_ready: bool,
) -> dict[str, Any]:
    lifecycle = _mapping(
        review_report.get("record_lifecycle_boundary"),
        "record_lifecycle_boundary",
    )
    source = _mapping(
        lifecycle.get("source_competition_records"),
        "source_competition_records",
    )
    qualification = _mapping(
        lifecycle.get("qualification_records"), "qualification_records"
    )
    return {
        "competition_records": {
            "evidence_digests": _copy(source["evidence_digests"]),
            "observation_id_digest": source["observation_id_digest"],
            "observation_count": source["observation_count"],
            "evaluator_epoch": source["evaluator_epoch"],
            "role": "AUDIT_ONLY_SCORING_EXCLUDED",
            "eligible_for_future_scoring": False,
        },
        "qualification_records": {
            "evidence_digests": _copy(qualification["evidence_digests"]),
            "observation_id_digest": qualification["observation_id_digest"],
            "observation_count": qualification["observation_count"],
            "evaluator_epoch": qualification["evaluator_epoch"],
            "role": "CONSUMED_AUDIT_ONLY_SCORING_EXCLUDED",
            "eligible_for_future_scoring": False,
        },
        "new_probe_records": {
            "evidence_digests": {
                "holdout": evidence_binding["holdout_evidence_digest"],
                "stress": evidence_binding["stress_evidence_digest"],
            },
            "observation_id_digest": evidence_binding[
                "new_observation_id_digest"
            ],
            "observation_count": evidence_binding["new_observation_count"],
            "evaluator_epoch": evaluator_epoch,
            "role": (
                "CONSUMED_FAILURE_BOUNDARY_EVIDENCE_AUDIT_ONLY"
                if source_ready
                else "NOT_CONSUMED_PROBE_EXPANSION_BLOCKED"
            ),
            "eligible_for_future_scoring": False,
        },
        "future_scoring_policy": {
            "new_unconsumed_evidence_required": True,
            "reuse_competition_records_allowed": False,
            "reuse_consumed_qualification_records_allowed": False,
            "reuse_consumed_probe_records_allowed": False,
            "cross_epoch_pooling_allowed": False,
        },
        "logical_selective_erasure_applied": True,
        "physical_erasure": "NOT_PERFORMED",
        "physical_retention_attestation": "REQUIRED_NOT_PRESENT",
    }


def _boundary_assessment(value: bool | None) -> dict[str, Any]:
    return {
        "boundary_counterexample_found": value,
        "source_child_invalidated": False,
        "rollback_execution_allowed": False,
        "restriction_execution_allowed": False,
        "adoption_decision_allowed": False,
        "promotion_decision_allowed": False,
        "current_pointer_write_allowed": False,
        "parent_or_child_state_write_allowed": False,
    }


def _attestation_boundary() -> dict[str, Any]:
    return {
        "external_data_attestation": "REQUIRED_NOT_PRESENT",
        "external_evaluator_attestation": "REQUIRED_NOT_PRESENT",
        "physical_retention_attestation": "REQUIRED_NOT_PRESENT",
        "external_adoption_authority": "REQUIRED_NOT_PRESENT",
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


def expand_and_evaluate_shadow_child_failure_boundary_probe(
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
    input_artifacts: Mapping[str, Any] | None = None,
) -> ShadowChildFailureBoundaryProbeResult:
    """Compile and evaluate one fixed probe without crossing authority bounds."""

    normalized_contract = _validate_contract(probe_contract)
    expected_contract_digest = _require_digest(
        expected_probe_contract_digest, "expected_probe_contract_digest"
    )
    contract_digest = _digest(normalized_contract)
    if contract_digest != expected_contract_digest:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "probe contract digest differs from independent expectation"
        )
    expected_input_digest = _require_digest(
        expected_probe_input_digest, "expected_probe_input_digest"
    )
    raw_input_digest = _digest(probe_input)
    if raw_input_digest != expected_input_digest:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "probe input digest differs from independent expectation"
        )
    if input_artifacts is not None and not isinstance(input_artifacts, Mapping):
        raise ShadowChildFailureBoundaryProbeValidationError(
            "input_artifacts must be an object or null"
        )
    artifacts = _copy(input_artifacts) if input_artifacts is not None else None
    _reject_observed_values(artifacts, "input_artifacts")

    expected_review_contract = _require_digest(
        expected_review_contract_digest, "expected_review_contract_digest"
    )
    if expected_review_contract != SOURCE_REVIEW_CONTRACT_DIGEST:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "expected review contract digest is not the pinned V1 source"
        )
    try:
        review_receipt = verify_shadow_child_external_review_packet(
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
            expected_review_contract_digest=expected_review_contract,
            expected_review_report_digest=expected_review_report_digest,
            expected_review_input_artifacts=expected_review_input_artifacts,
        )
    except (
        ShadowChildExternalReviewPacketValidationError,
        CompetitionValidationError,
        KeyError,
        TypeError,
    ) as exc:
        raise ShadowChildFailureBoundaryProbeValidationError(
            f"source review packet verification failed: {exc}"
        ) from exc

    review = _copy(_mapping(review_report, "review_report"))
    receipt = _copy(_mapping(review_receipt, "review_receipt"))
    transition = _copy(_mapping(transition_report, "transition_report"))
    if review.get("contract_id") != SOURCE_REVIEW_CONTRACT_ID:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "source review contract_id is not supported"
        )
    if review.get("contract_digest") != SOURCE_REVIEW_CONTRACT_DIGEST:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "source review contract digest is not pinned V1"
        )
    if review.get("schema_version") != SOURCE_REVIEW_REPORT_SCHEMA_VERSION:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "source review report schema is not supported"
        )
    if review.get("adoption_status") != ADOPTION_STATUS:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "source review crossed the shadow-only adoption boundary"
        )

    child = _copy(
        _mapping(transition.get("child_theory_state"), "child_theory_state")
    )
    original_child_bytes = canonical_json_bytes(child)
    child_digest = _require_digest(
        review.get("child_theory_state_digest"), "child_theory_state_digest"
    )
    if transition.get("child_theory_state_digest") != child_digest:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "review child digest differs from the verified transition child"
        )
    if _digest(child) != child_digest:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "verified child state digest is inconsistent"
        )
    parent = _copy(
        _mapping(transition.get("parent_theory_state"), "parent_theory_state")
    )
    transition_kind = _string(
        transition.get("transition_kind"), "transition_kind"
    )
    if transition_kind not in TRANSITION_PROBE_REGISTRY:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "verified transition has no fixed failure-boundary probe"
        )
    if review.get("transition_kind") != transition_kind:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "source review transition kind is inconsistent"
        )

    normalized_input = _normalize_input(
        probe_input,
        review_report=review,
        transition_report=transition,
        competition_input=competition_input,
        qualification_input=qualification_input,
    )
    # The independent expectation authenticates the supplied file bytes above;
    # the report commits to the normalized semantic input so row ordering is
    # deliberately not an observable part of the probe result.
    input_digest = _digest(normalized_input["public"])
    probe_expansion_id = normalized_input["public"]["probe_expansion_id"]
    fixed_anchor = _string(parent.get("fixed_anchor"), "parent fixed_anchor")
    epoch_payload, derived_epoch = _epoch_components(
        review_contract_digest=review["contract_digest"],
        review_report_digest=review["report_digest"],
        child_theory_state_digest=child_digest,
        transition_kind=transition_kind,
        fixed_anchor=fixed_anchor,
        probe_contract=normalized_contract,
    )
    qualification_epoch = _string(
        review.get("evaluator_epoch"), "source qualification evaluator_epoch"
    )
    competition_epoch = _string(
        parent.get("evaluator_epoch"), "source competition evaluator_epoch"
    )
    if derived_epoch in {qualification_epoch, competition_epoch}:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "derived probe epoch is not fresh from prior evaluator epochs"
        )
    evaluator_binding = _evaluator_binding(
        normalized_input, derived_epoch, fixed_anchor
    )
    evidence_binding = _evidence_binding(normalized_input, normalized_contract)
    source_summary = _source_review_summary(review, receipt)
    source_ready = all(
        (
            review.get("disposition") == SELECTION["source_ready_status"],
            _mapping(review.get("review_boundary"), "review_boundary").get(
                "packet_ready"
            )
            is True,
            receipt.get("ready_for_external_review") is True,
        )
    )

    probe_definition: dict[str, Any] | None = None
    expanded_state: dict[str, Any] | None = None
    expanded_state_digest: str | None = None
    evaluator_definition: dict[str, Any] | None = None
    probe_results: dict[str, Any] | None = None
    boundary_value: bool | None = None

    if not source_ready:
        disposition = SELECTION["blocked_status"]
    else:
        probe_definition = _copy(TRANSITION_PROBE_REGISTRY[transition_kind])
        evaluator_definition = {
            "evaluator_epoch": derived_epoch,
            "fixed_anchor": fixed_anchor,
            "epoch_derivation_kind": (
                "CONTENT_ADDRESSED_LOCAL_FAILURE_BOUNDARY_PROBE_EPOCH"
            ),
            "expanded_probe_registry_digest": _digest(
                epoch_payload["expanded_probe_registry"]
            ),
        }
        expanded_state = _expanded_state(
            child=child,
            child_digest=child_digest,
            review_report=review,
            transition_kind=transition_kind,
            probe_definition=probe_definition,
            probe_contract_digest=contract_digest,
            probe_expansion_id=probe_expansion_id,
            evaluator_epoch=derived_epoch,
        )
        expanded_state_digest = _digest(expanded_state)
        if not evaluator_binding["comparable"]:
            disposition = SELECTION["incomparable_status"]
        elif not evidence_binding["complete_evidence"]:
            disposition = SELECTION["needs_evidence_status"]
        else:
            if transition_kind == "ROBUST_INTERVAL_EXPANSION":
                probe_results = _evaluate_robust(
                    parent,
                    child,
                    normalized_input["evidence"],
                    normalized_contract,
                )
            else:
                probe_results = _evaluate_quotient(
                    parent,
                    child,
                    transition,
                    normalized_input["evidence"],
                    normalized_contract,
                )
            boundary_value = bool(
                probe_results["boundary_counterexample_found"]
            )
            disposition = (
                SELECTION["counterexample_status"]
                if boundary_value
                else SELECTION["no_counterexample_status"]
            )

    lifecycle_extension = _record_lifecycle_extension(
        review_report=review,
        evidence_binding=evidence_binding,
        evaluator_epoch=derived_epoch,
        source_ready=source_ready,
    )
    boundary = _boundary_assessment(boundary_value)
    attestations = _attestation_boundary()

    events = [
        _audit_event(
            0,
            "SOURCE_REVIEW_PACKET_VERIFIED",
            GENESIS_DIGEST,
            {
                "verification_status": source_summary["verification_status"],
                "review_report_digest": source_summary["report_digest"],
                "source_disposition": source_summary["disposition"],
                "packet_ready": source_ready,
            },
        )
    ]
    if source_ready:
        events.append(
            _audit_event(
                1,
                "FAILURE_BOUNDARY_PROBE_COMPILED",
                events[-1]["event_digest"],
                {
                    "probe_id": probe_definition["probe_id"],
                    "transition_kind": transition_kind,
                    "probe_expanded_shadow_theory_state_digest": (
                        expanded_state_digest
                    ),
                },
            )
        )
        events.append(
            _audit_event(
                2,
                "NEW_UNCONSUMED_EVIDENCE_ISOLATION_BOUND",
                events[-1]["event_digest"],
                {
                    "new_observation_id_digest": evidence_binding[
                        "new_observation_id_digest"
                    ],
                    "new_observation_count": evidence_binding[
                        "new_observation_count"
                    ],
                    "cross_epoch_pooling": False,
                },
            )
        )
        events.append(
            _audit_event(
                3,
                "FAILURE_BOUNDARY_PROBE_ASSESSED_AND_AUTHORITY_WITHHELD",
                events[-1]["event_digest"],
                {
                    "disposition": disposition,
                    "boundary_counterexample_found": boundary_value,
                    "adoption_status": ADOPTION_STATUS,
                    "current_status": CURRENT_STATUS,
                },
            )
        )
    else:
        events.append(
            _audit_event(
                1,
                "PROBE_EXPANSION_BLOCKED_AND_AUTHORITY_WITHHELD",
                events[-1]["event_digest"],
                {
                    "disposition": disposition,
                    "probe_compiled": False,
                    "adoption_status": ADOPTION_STATUS,
                },
            )
        )

    if canonical_json_bytes(child) != original_child_bytes:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "source child was mutated while compiling the probe"
        )
    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "contract_digest": contract_digest,
        "probe_input_digest": input_digest,
        "probe_expansion_id": probe_expansion_id,
        "source_review_packet": source_summary,
        "source_child_theory_state_digest": child_digest,
        "transition_kind": transition_kind,
        "probe_definition": probe_definition,
        "probe_expanded_shadow_theory_state": expanded_state,
        "probe_expanded_shadow_theory_state_digest": expanded_state_digest,
        "evaluator_definition": evaluator_definition,
        "evaluator_binding": evaluator_binding,
        "evidence_binding": evidence_binding,
        "record_lifecycle_extension": lifecycle_extension,
        "probe_results": probe_results,
        "disposition": disposition,
        "boundary_assessment": boundary,
        "attestation_boundary": attestations,
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
    _exact_keys(report, REPORT_FIELDS, "probe_report")
    _reject_observed_values(report, "probe_report")
    return ShadowChildFailureBoundaryProbeResult(report=report)


def verify_shadow_child_failure_boundary_probe(
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
) -> dict[str, Any]:
    """Exact-replay a failure-boundary report against independent anchors."""

    expected_report_digest = _require_digest(
        expected_probe_report_digest, "expected_probe_report_digest"
    )
    supplied = _copy(_mapping(probe_report, "probe_report"))
    _exact_keys(supplied, REPORT_FIELDS, "probe_report")
    _reject_observed_values(supplied, "probe_report")
    if expected_probe_input_artifacts is not None and not isinstance(
        expected_probe_input_artifacts, Mapping
    ):
        raise ShadowChildFailureBoundaryProbeValidationError(
            "expected_probe_input_artifacts must be an object or null"
        )
    expected_artifacts = (
        _copy(expected_probe_input_artifacts)
        if expected_probe_input_artifacts is not None
        else None
    )
    _reject_observed_values(expected_artifacts, "expected_probe_input_artifacts")
    if supplied.get("input_artifacts") != expected_artifacts:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "probe input artifacts differ from independent expectation"
        )
    fresh = expand_and_evaluate_shadow_child_failure_boundary_probe(
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
        input_artifacts=expected_artifacts,
    ).to_dict()
    if fresh["report_digest"] != expected_report_digest:
        raise ShadowChildFailureBoundaryProbeValidationError(
            "replayed probe report digest differs from expectation"
        )
    if canonical_json_bytes(supplied) != canonical_json_bytes(fresh):
        raise ShadowChildFailureBoundaryProbeValidationError(
            "supplied probe report differs from exact replay"
        )
    return {
        "status": "VERIFIED_" + fresh["disposition"],
        "disposition": fresh["disposition"],
        "report_digest": fresh["report_digest"],
        "contract_digest": fresh["contract_digest"],
        "probe_expansion_id": fresh["probe_expansion_id"],
        "source_review_report_digest": fresh["source_review_packet"][
            "report_digest"
        ],
        "source_child_theory_state_digest": fresh[
            "source_child_theory_state_digest"
        ],
        "probe_expanded_shadow_theory_state_digest": fresh[
            "probe_expanded_shadow_theory_state_digest"
        ],
        "probe_expanded": fresh["probe_expanded_shadow_theory_state"] is not None,
        "boundary_counterexample_found": fresh["boundary_assessment"][
            "boundary_counterexample_found"
        ],
        "adoption_eligibility": fresh["adoption_eligibility"],
        "adoption_status": fresh["adoption_status"],
        "promotion_status": fresh["promotion_status"],
        "current_status": fresh["current_status"],
    }
