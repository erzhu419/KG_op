"""Materialize a verified theory-operation winner as an immutable shadow child.

This module is a pure, local transition layer over
``performance.theory_operation_competition``.  It replays that public verifier
before reading a selected candidate, constructs a detached child theory in
memory, and emits a digest-bound report.  It never adopts the child, assigns a
child evaluator epoch, reuses source evidence for child scoring, or writes any
state.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from performance.theory_operation_competition import (
    CompetitionValidationError,
    canonical_json_bytes,
    verify_theory_operation_competition,
)


CONTRACT_SCHEMA_VERSION = "sc-olh-kg.shadow-theory-transition-contract/1"
CONTRACT_ID = "shadow_theory_transition_v1"
SOURCE_CONTRACT_ID = "theory_operation_competition_v1"
SOURCE_REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.theory-operation-competition-report/1"
)
CHILD_THEORY_SCHEMA_VERSION = "sc-olh-kg.shadow-theory-state/1"
REPORT_SCHEMA_VERSION = "sc-olh-kg.shadow-theory-transition-report/1"
OFFICIAL_SOURCE_CONTRACT_DIGEST = (
    "sha256:9e1aa6834ce2663066f1e9b6a01f5cc94c5e807aca40fc490b9b0829084363b8"
)
GENESIS_DIGEST = "sha256:" + "0" * 64

DISPOSITION_TO_TRANSITION = {
    "SELECT_ROBUSTIFICATION": "MATERIALIZED_SHADOW_ROBUSTIFICATION",
    "SELECT_IDEALIZATION": "MATERIALIZED_SHADOW_IDEALIZATION",
    "NEEDS_EVIDENCE": "NOT_MATERIALIZED_NEEDS_EVIDENCE",
    "INCOMPARABLE_EVALUATOR_EPOCH": (
        "NOT_MATERIALIZED_INCOMPARABLE_EVALUATOR_EPOCH"
    ),
}

PRESERVATION_POLICY = {
    "complete_parent_object_space_check": True,
    "robust_center_equals_parent": True,
    "robust_radii_finite_and_nonnegative": True,
    "robust_zero_radius_recovers_parent": True,
    "quotient_prediction_recomputed_as_parent_mean": True,
    "quotient_max_divergence_uses_source_contract": True,
    "quotient_parent_snapshot_required": True,
}

EVALUATOR_EPOCH_POLICY = {
    "inherit_fixed_anchor": True,
    "child_evaluator_epoch_status": "UNASSIGNED_NEW_EVALUATOR_REQUIRED",
    "source_evidence_role": "QUALIFICATION_ONLY",
    "source_evidence_allowed_for_child_scoring": False,
    "forbid_old_new_pooling": True,
    "operational_probe_gate_required": True,
    "selective_erasure_applied": False,
}

SELECTION_POLICY = {
    "adoption_status": "NOT_ADOPTED_SHADOW_ONLY",
    "robust_operation_kind": "expand",
    "robust_transition_kind": "ROBUST_INTERVAL_EXPANSION",
    "ideal_operation_kind": "quotient",
    "ideal_transition_kind": "QUOTIENT_IDEALIZATION",
}

MANDATORY_NONCLAIMS = (
    "shadow_only",
    "no_automatic_adoption",
    "no_paper_promotion",
    "no_parent_or_ambient_state_write",
    "explicit_cli_out_is_only_optional_write",
    "no_run_one",
    "no_benchmark_execution",
    "no_scheduler_access",
    "no_network_access",
    "only_expand_and_quotient_materialization",
    "no_restrict_operation",
    "no_probe_expansion_or_execution",
    "no_language_or_predicate_invention",
    "no_operational_probe_preservation_claim",
    "copied_probe_ids_are_declarations_only",
    "source_evidence_is_qualification_only",
    "no_source_evidence_reuse_for_child_scoring",
    "fresh_child_evaluator_epoch_not_created",
    "no_selective_erasure_implementation",
    "no_cross_epoch_pooling",
    "robust_child_is_finite_scalar_interval_only",
    "interval_containment_is_not_domain_safety",
    "idealization_is_task_and_finite_object_space_scoped",
    "lossy_quotient_requires_frozen_parent_snapshot",
    "quotient_alone_does_not_recover_parent",
    "no_external_evaluator_or_anchor_attestation",
    "input_artifacts_require_independent_expected_values",
    "report_digest_is_not_a_signature",
    "no_scientific_validity_or_generalization_claim",
    "no_h_t_to_h_t_plus_1_acceptance",
)

REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "contract_digest",
        "source_competition",
        "parent_theory_state",
        "parent_theory_state_digest",
        "disposition",
        "operation_kind",
        "transition_kind",
        "selected_candidate_id",
        "child_theory_state",
        "child_theory_state_digest",
        "preservation_certificate",
        "reduction_certificate",
        "evaluator_gate",
        "adoption_status",
        "nonclaims",
        "input_artifacts",
        "audit_events",
        "audit_head",
        "report_digest",
    }
)


class ShadowTransitionValidationError(ValueError):
    """Raised when a transition contract, source, or report fails closed."""


class ShadowTransitionKind(str, Enum):
    ROBUST_INTERVAL_EXPANSION = "ROBUST_INTERVAL_EXPANSION"
    QUOTIENT_IDEALIZATION = "QUOTIENT_IDEALIZATION"


class ShadowTransitionDisposition(str, Enum):
    MATERIALIZED_SHADOW_ROBUSTIFICATION = "MATERIALIZED_SHADOW_ROBUSTIFICATION"
    MATERIALIZED_SHADOW_IDEALIZATION = "MATERIALIZED_SHADOW_IDEALIZATION"
    NOT_MATERIALIZED_NEEDS_EVIDENCE = "NOT_MATERIALIZED_NEEDS_EVIDENCE"
    NOT_MATERIALIZED_INCOMPARABLE_EVALUATOR_EPOCH = (
        "NOT_MATERIALIZED_INCOMPARABLE_EVALUATOR_EPOCH"
    )


@dataclass(frozen=True)
class ShadowTransitionResult:
    """Detached result wrapper matching the source competition's public shape."""

    report: Mapping[str, Any]

    @property
    def disposition(self) -> str:
        return str(self.report["disposition"])

    @property
    def report_digest(self) -> str:
        return str(self.report["report_digest"])

    def to_dict(self) -> dict[str, Any]:
        return _copy(self.report)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShadowTransitionValidationError(f"{label} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str
) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected), key=str)
        raise ShadowTransitionValidationError(
            f"{label} keys differ; missing={missing}, extra={extra}"
        )


def _string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ShadowTransitionValidationError(f"{label} must be a non-empty string")
    return value


def _finite_number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ShadowTransitionValidationError(f"{label} must be a finite number")
    return float(value)


def _copy(value: Any) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("ascii"))
    except (CompetitionValidationError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShadowTransitionValidationError(
            f"value is not detached finite canonical JSON: {exc}"
        ) from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_digest(value: Any, label: str) -> str:
    digest = _string(value, label)
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise ShadowTransitionValidationError(f"{label} must be sha256:<64hex>")
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise ShadowTransitionValidationError(f"{label} is not hexadecimal") from exc
    return digest


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ShadowTransitionValidationError(
            f"{label} differs from frozen shadow-transition V1"
        )


def _validate_contract(contract_value: Any) -> dict[str, Any]:
    contract = _mapping(contract_value, "transition_contract")
    _exact_keys(
        contract,
        {
            "schema_version",
            "contract_id",
            "source_contract_id",
            "source_report_schema_version",
            "child_theory_schema_version",
            "report_schema_version",
            "materializable_sources",
            "preservation_policy",
            "evaluator_epoch_policy",
            "selection",
            "nonclaims",
        },
        "transition_contract",
    )
    _require_equal(
        contract["schema_version"], CONTRACT_SCHEMA_VERSION, "schema_version"
    )
    _require_equal(contract["contract_id"], CONTRACT_ID, "contract_id")
    _require_equal(
        contract["source_contract_id"], SOURCE_CONTRACT_ID, "source_contract_id"
    )
    _require_equal(
        contract["source_report_schema_version"],
        SOURCE_REPORT_SCHEMA_VERSION,
        "source_report_schema_version",
    )
    _require_equal(
        contract["child_theory_schema_version"],
        CHILD_THEORY_SCHEMA_VERSION,
        "child_theory_schema_version",
    )
    _require_equal(
        contract["report_schema_version"],
        REPORT_SCHEMA_VERSION,
        "report_schema_version",
    )

    sources = _mapping(contract["materializable_sources"], "materializable_sources")
    _exact_keys(
        sources,
        {"official_source_contract_digest", "disposition_to_transition"},
        "materializable_sources",
    )
    _require_equal(
        sources["official_source_contract_digest"],
        OFFICIAL_SOURCE_CONTRACT_DIGEST,
        "materializable_sources.official_source_contract_digest",
    )
    dispositions = _mapping(
        sources["disposition_to_transition"],
        "materializable_sources.disposition_to_transition",
    )
    _require_equal(
        dict(dispositions),
        DISPOSITION_TO_TRANSITION,
        "materializable_sources.disposition_to_transition",
    )

    preservation = _mapping(contract["preservation_policy"], "preservation_policy")
    _require_equal(
        dict(preservation), PRESERVATION_POLICY, "preservation_policy"
    )
    evaluator = _mapping(
        contract["evaluator_epoch_policy"], "evaluator_epoch_policy"
    )
    _require_equal(
        dict(evaluator), EVALUATOR_EPOCH_POLICY, "evaluator_epoch_policy"
    )
    selection = _mapping(contract["selection"], "selection")
    _require_equal(dict(selection), SELECTION_POLICY, "selection")

    nonclaims = contract["nonclaims"]
    if not isinstance(nonclaims, list) or tuple(nonclaims) != MANDATORY_NONCLAIMS:
        raise ShadowTransitionValidationError(
            "nonclaims differ from the exact mandatory V1 list"
        )
    return _copy(contract)


def validate_shadow_transition_contract(contract_value: Any) -> dict[str, Any]:
    """Validate and return a detached canonical copy of the transition contract."""

    return _validate_contract(contract_value)


def _validate_expected_transition_contract_digest(
    contract: Mapping[str, Any], expected_digest: str
) -> str:
    expected = _require_digest(
        expected_digest, "expected_transition_contract_digest"
    )
    actual = _digest(contract)
    if actual != expected:
        raise ShadowTransitionValidationError(
            "transition contract digest differs from independent expectation"
        )
    return actual


def _verified_source(
    competition_input: Mapping[str, Any],
    competition_contract: Mapping[str, Any],
    competition_report: Mapping[str, Any],
    *,
    expected_competition_contract_digest: str,
    expected_competition_report_digest: str,
    expected_competition_input_artifacts: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_contract_digest = _require_digest(
        expected_competition_contract_digest,
        "expected_competition_contract_digest",
    )
    if expected_contract_digest != OFFICIAL_SOURCE_CONTRACT_DIGEST:
        raise ShadowTransitionValidationError(
            "competition contract digest is not the official pinned source"
        )
    expected_report_digest = _require_digest(
        expected_competition_report_digest,
        "expected_competition_report_digest",
    )
    if expected_competition_input_artifacts is not None and not isinstance(
        expected_competition_input_artifacts, Mapping
    ):
        raise ShadowTransitionValidationError(
            "expected_competition_input_artifacts must be an object or null"
        )
    try:
        receipt = verify_theory_operation_competition(
            competition_input,
            competition_contract,
            competition_report,
            expected_contract_digest=expected_contract_digest,
            expected_report_digest=expected_report_digest,
            expected_input_artifacts=expected_competition_input_artifacts,
        )
    except (CompetitionValidationError, TypeError, KeyError) as exc:
        raise ShadowTransitionValidationError(
            f"source competition verification failed: {exc}"
        ) from exc

    report = _copy(_mapping(competition_report, "competition_report"))
    if report.get("schema_version") != SOURCE_REPORT_SCHEMA_VERSION:
        raise ShadowTransitionValidationError(
            "source competition report schema is not supported"
        )
    if report.get("contract_id") != SOURCE_CONTRACT_ID:
        raise ShadowTransitionValidationError(
            "source competition contract_id is not supported"
        )
    if report.get("contract_digest") != expected_contract_digest:
        raise ShadowTransitionValidationError(
            "source report contract digest differs from the pinned source"
        )
    if report.get("report_digest") != expected_report_digest:
        raise ShadowTransitionValidationError(
            "source report digest differs from the independently expected report"
        )
    disposition = report.get("disposition")
    if disposition not in DISPOSITION_TO_TRANSITION:
        raise ShadowTransitionValidationError(
            "source competition disposition is not recognized"
        )
    return report, _copy(receipt)


def _selected_candidate(
    report: Mapping[str, Any], disposition: str
) -> dict[str, Any] | None:
    selected = report.get("selected_candidate")
    if disposition in ("NEEDS_EVIDENCE", "INCOMPARABLE_EVALUATOR_EPOCH"):
        if selected is not None:
            raise ShadowTransitionValidationError(
                "non-materializable source unexpectedly selected a candidate"
            )
        return None
    candidate = _copy(_mapping(selected, "selected_candidate"))
    candidate_id = _string(candidate.get("candidate_id"), "selected_candidate_id")
    list_key = (
        "robustification_candidates"
        if disposition == "SELECT_ROBUSTIFICATION"
        else "idealization_candidates"
    )
    candidates = report.get(list_key)
    if not isinstance(candidates, list):
        raise ShadowTransitionValidationError(f"{list_key} must be a list")
    exact_matches = [
        item
        for item in candidates
        if isinstance(item, Mapping)
        and item.get("candidate_id") == candidate_id
        and canonical_json_bytes(item) == canonical_json_bytes(candidate)
    ]
    if len(exact_matches) != 1:
        raise ShadowTransitionValidationError(
            "selected candidate is not an exact unique committed candidate"
        )
    evaluation = _mapping(candidate.get("evaluation"), "selected evaluation")
    if evaluation.get("viable") is not True:
        raise ShadowTransitionValidationError("selected candidate is not viable")
    return candidate


def _parent_theory(report: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    parent = _copy(_mapping(report.get("theory_state"), "parent_theory_state"))
    stated = _require_digest(
        report.get("theory_state_digest"), "parent_theory_state_digest"
    )
    if _digest(parent) != stated:
        raise ShadowTransitionValidationError(
            "parent theory state digest does not match its canonical state"
        )
    return parent, stated


def _prediction_lookup(
    predictions: Sequence[Mapping[str, Any]], label: str
) -> dict[bytes, float]:
    lookup: dict[bytes, float] = {}
    for index, raw in enumerate(predictions):
        item = _mapping(raw, f"{label}[{index}]")
        _exact_keys(item, {"context", "value"}, f"{label}[{index}]")
        context = _mapping(item["context"], f"{label}[{index}].context")
        key = canonical_json_bytes(context)
        if key in lookup:
            raise ShadowTransitionValidationError(f"{label} has duplicate contexts")
        lookup[key] = _finite_number(item["value"], f"{label}[{index}].value")
    return lookup


def _base_child_payload(
    parent: Mapping[str, Any],
    parent_digest: str,
    source: Mapping[str, Any],
    candidate: Mapping[str, Any],
    operation_kind: str,
    transition_kind: str,
    object_space: Mapping[str, Any],
    model_class: Mapping[str, Any],
    removable_feature_ids: Sequence[str],
) -> dict[str, Any]:
    lineage = {
        "parent_theory_id": parent["theory_id"],
        "parent_theory_state_digest": parent_digest,
        "source_competition_contract_digest": source["contract_digest"],
        "source_competition_report_digest": source["report_digest"],
        "candidate_commitment_digest": source["candidate_commitment_digest"],
        "selected_candidate_id": candidate["candidate_id"],
        "operation_kind": operation_kind,
        "transition_kind": transition_kind,
    }
    evidence_reuse_policy = {
        "source_evidence_role": "QUALIFICATION_ONLY",
        "source_evidence_allowed_for_child_scoring": False,
        "old_new_records_pooled": False,
    }
    return {
        "schema_version": CHILD_THEORY_SCHEMA_VERSION,
        "task_id": parent["task_id"],
        "evaluator_epoch": None,
        "evaluator_status": "UNASSIGNED_NEW_EVALUATOR_REQUIRED",
        "fixed_anchor": parent["fixed_anchor"],
        "object_space": _copy(object_space),
        "model_class": _copy(model_class),
        "probe_ids": _copy(parent["probe_ids"]),
        "violation_functionals": _copy(parent["violation_functionals"]),
        "scope_ids": _copy(parent["scope_ids"]),
        "removable_feature_ids": _copy(list(removable_feature_ids)),
        "evidence_reuse_policy": evidence_reuse_policy,
        "operational_probe_status": (
            "OPERATIONAL_PROBE_AND_FRESH_EPOCH_REQUIRED"
        ),
        "transition_lineage": lineage,
    }


def _with_child_id(payload: Mapping[str, Any]) -> dict[str, Any]:
    detached = _copy(payload)
    theory_id = "shadow-theory:" + _digest(detached)[7:]
    child = {"schema_version": detached.pop("schema_version"), "theory_id": theory_id}
    child.update(detached)
    expected = {
        "schema_version",
        "theory_id",
        "task_id",
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
        "transition_lineage",
    }
    _exact_keys(child, expected, "child_theory_state")
    return child


def _robust_radius_for_pair(
    candidate: Mapping[str, Any], context: Mapping[str, Any], scope_id: str
) -> float:
    grouping = candidate.get("grouping")
    radii = candidate.get("radii")
    if not isinstance(radii, list):
        raise ShadowTransitionValidationError("robust radii must be a list")
    matches = []
    for index, raw in enumerate(radii):
        item = _mapping(raw, f"radii[{index}]")
        _exact_keys(item, {"group", "radius"}, f"radii[{index}]")
        group = _mapping(item["group"], f"radii[{index}].group")
        if grouping == "global" and dict(group) == {"global": "*"}:
            matches.append(item["radius"])
        elif grouping == "per_scope" and dict(group) == {"scope_id": scope_id}:
            matches.append(item["radius"])
        elif grouping == "per_context" and set(group) == {"context"}:
            if canonical_json_bytes(group["context"]) == canonical_json_bytes(context):
                matches.append(item["radius"])
    if len(matches) != 1:
        raise ShadowTransitionValidationError(
            "robust radius grouping does not cover each parent context-scope pair exactly once"
        )
    radius = _finite_number(matches[0], "robust radius")
    if radius < 0:
        raise ShadowTransitionValidationError("robust radius cannot be negative")
    return radius


def _materialize_robust(
    parent: Mapping[str, Any],
    parent_digest: str,
    source: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if (
        candidate.get("operation_kind") != "expand"
        or candidate.get("family") != "robust_interval"
        or candidate.get("mechanism") != "interval"
        or candidate.get("grouping") not in {"global", "per_scope", "per_context"}
    ):
        raise ShadowTransitionValidationError(
            "selected robustification is outside the V1 materialization grammar"
        )
    parent_model = _mapping(parent["model_class"], "parent model_class")
    if parent_model.get("kind") != "finite_point_table":
        raise ShadowTransitionValidationError("parent model must be a finite point table")
    predictions = parent_model.get("predictions")
    if not isinstance(predictions, list):
        raise ShadowTransitionValidationError("parent predictions must be a list")
    contexts = _mapping(parent["object_space"], "parent object_space").get("contexts")
    scopes = parent.get("scope_ids")
    if not isinstance(contexts, list) or not isinstance(scopes, list):
        raise ShadowTransitionValidationError(
            "parent contexts and scope_ids must be lists"
        )
    _prediction_lookup(predictions, "parent predictions")
    pair_count = 0
    for context in contexts:
        context_map = _mapping(context, "parent context")
        for scope_id in scopes:
            _robust_radius_for_pair(candidate, context_map, _string(scope_id, "scope_id"))
            pair_count += 1

    reduction_map = _mapping(candidate.get("reduction_map"), "robust reduction_map")
    if (
        reduction_map.get("parent_model_class") != "finite_point_table"
        or reduction_map.get("zero_radius_recovers_parent") is not True
        or reduction_map.get("limit_parameter") != "radius"
        or _finite_number(reduction_map.get("limit_value"), "limit_value") != 0.0
    ):
        raise ShadowTransitionValidationError(
            "robust candidate lacks the exact zero-radius parent reduction"
        )

    model_class = {
        "kind": "finite_interval_table",
        "center_predictions": _copy(predictions),
        "radius_grouping": candidate["grouping"],
        "radii": _copy(candidate["radii"]),
    }
    payload = _base_child_payload(
        parent,
        parent_digest,
        source,
        candidate,
        "expand",
        "ROBUST_INTERVAL_EXPANSION",
        parent["object_space"],
        model_class,
        parent["removable_feature_ids"],
    )
    child = _with_child_id(payload)
    preservation = {
        "certificate_kind": "EXACT_CENTER_CONSERVATIVE_EXTENSION",
        "complete_parent_object_space_checked": True,
        "checked_parent_context_count": len(contexts),
        "checked_parent_context_scope_pair_count": pair_count,
        "max_center_divergence": 0.0,
        "all_parent_predictions_contained": True,
        "all_radii_finite_and_nonnegative": True,
        "operational_probe_preservation_certified": False,
    }
    parent_model_digest = _digest(parent_model)
    reduction = {
        "map_kind": "COLLAPSE_INTERVAL_AT_RADIUS_MULTIPLIER_ZERO",
        "parameter": "radius_multiplier",
        "parent_value": 0.0,
        "collapsed_model_class_digest": parent_model_digest,
        "parent_model_class_digest": parent_model_digest,
        "exact_parent_model_class_recovered": True,
    }
    return child, preservation, reduction


def _quotient_context(
    context: Mapping[str, Any], retained_feature_ids: Sequence[str]
) -> dict[str, Any]:
    return {feature_id: context[feature_id] for feature_id in retained_feature_ids}


def _materialize_ideal(
    parent: Mapping[str, Any],
    parent_digest: str,
    source: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if (
        candidate.get("operation_kind") != "quotient"
        or candidate.get("family") != "idealization_quotient"
        or candidate.get("mechanism") != "quotient"
    ):
        raise ShadowTransitionValidationError(
            "selected idealization is outside the V1 materialization grammar"
        )
    parent_object_space = _mapping(parent["object_space"], "parent object_space")
    parent_contexts = parent_object_space.get("contexts")
    parent_features = parent_object_space.get("feature_ids")
    if not isinstance(parent_contexts, list) or not isinstance(parent_features, list):
        raise ShadowTransitionValidationError(
            "parent feature_ids and contexts must be lists"
        )
    retained = candidate.get("retained_feature_ids")
    removed = candidate.get("removed_feature_ids")
    if not isinstance(retained, list) or not isinstance(removed, list):
        raise ShadowTransitionValidationError(
            "ideal retained and removed feature IDs must be lists"
        )
    if sorted(retained + removed) != sorted(parent_features) or set(retained) & set(removed):
        raise ShadowTransitionValidationError(
            "ideal feature partition does not exactly cover the parent features"
        )

    parent_model = _mapping(parent["model_class"], "parent model_class")
    if parent_model.get("kind") != "finite_point_table":
        raise ShadowTransitionValidationError("parent model must be a finite point table")
    parent_predictions = parent_model.get("predictions")
    if not isinstance(parent_predictions, list):
        raise ShadowTransitionValidationError("parent predictions must be a list")
    parent_lookup = _prediction_lookup(parent_predictions, "parent predictions")

    groups: dict[bytes, dict[str, Any]] = {}
    for context_value in parent_contexts:
        context = _mapping(context_value, "parent context")
        key = canonical_json_bytes(context)
        if key not in parent_lookup:
            raise ShadowTransitionValidationError(
                "parent point table does not cover the complete object space"
            )
        quotient = _quotient_context(context, retained)
        quotient_key = canonical_json_bytes(quotient)
        group = groups.setdefault(
            quotient_key, {"context": quotient, "parent_values": []}
        )
        group["parent_values"].append(parent_lookup[key])
    recomputed_predictions = [
        {
            "context": groups[key]["context"],
            "value": math.fsum(groups[key]["parent_values"])
            / len(groups[key]["parent_values"]),
        }
        for key in sorted(groups)
    ]
    if canonical_json_bytes(recomputed_predictions) != canonical_json_bytes(
        candidate.get("quotient_predictions")
    ):
        raise ShadowTransitionValidationError(
            "quotient predictions are not the recomputed parent means"
        )
    child_lookup = _prediction_lookup(recomputed_predictions, "quotient predictions")
    divergences = []
    for context_value in parent_contexts:
        context = _mapping(context_value, "parent context")
        parent_value = parent_lookup[canonical_json_bytes(context)]
        child_value = child_lookup[
            canonical_json_bytes(_quotient_context(context, retained))
        ]
        divergences.append(abs(parent_value - child_value))
    max_divergence = max(divergences, default=0.0)
    thresholds = _mapping(source_contract.get("thresholds"), "source thresholds")
    allowed_divergence = _finite_number(
        thresholds.get("max_probe_divergence"), "source max_probe_divergence"
    )
    if max_divergence > allowed_divergence:
        raise ShadowTransitionValidationError(
            "selected quotient exceeds the source-contract divergence bound"
        )

    ideal_contract = _mapping(
        candidate.get("idealization_contract"), "idealization_contract"
    )
    recovery = _mapping(
        ideal_contract.get("full_model_recovery_method"),
        "full_model_recovery_method",
    )
    if (
        recovery.get("kind") != "restore_frozen_parent_point_table"
        or recovery.get("parent_theory_state_digest") != parent_digest
        or recovery.get("lossy_quotient_requires_parent_snapshot") is not True
        or canonical_json_bytes(recovery.get("parent_predictions"))
        != canonical_json_bytes(parent_predictions)
    ):
        raise ShadowTransitionValidationError(
            "idealization does not carry the exact frozen parent snapshot"
        )
    fiber_map = recovery.get("quotient_fiber_map")
    if not isinstance(fiber_map, list) or len(fiber_map) != len(parent_contexts):
        raise ShadowTransitionValidationError(
            "quotient fiber map does not cover the parent object space"
        )
    expected_fibers = [
        {
            "parent_context": _copy(context),
            "quotient_context": _quotient_context(
                _mapping(context, "parent context"), retained
            ),
        }
        for context in parent_contexts
    ]
    if canonical_json_bytes(fiber_map) != canonical_json_bytes(expected_fibers):
        raise ShadowTransitionValidationError(
            "quotient fiber map differs from the exact parent projection"
        )

    child_object_space = {
        "feature_ids": _copy(retained),
        "contexts": [_copy(item["context"]) for item in recomputed_predictions],
    }
    model_class = {
        "kind": "finite_point_table",
        "predictions": recomputed_predictions,
    }
    parent_removable = parent.get("removable_feature_ids")
    if not isinstance(parent_removable, list):
        raise ShadowTransitionValidationError(
            "parent removable_feature_ids must be a list"
        )
    retained_removable = [item for item in parent_removable if item in retained]
    payload = _base_child_payload(
        parent,
        parent_digest,
        source,
        candidate,
        "quotient",
        "QUOTIENT_IDEALIZATION",
        child_object_space,
        model_class,
        retained_removable,
    )
    child = _with_child_id(payload)
    parent_count = len(parent_contexts)
    child_count = len(recomputed_predictions)
    fractional_reduction = (parent_count - child_count) / parent_count
    preservation = {
        "certificate_kind": "FINITE_PREDICTION_PROXY_BOUND",
        "complete_parent_object_space_checked": True,
        "checked_parent_context_count": parent_count,
        "parent_state_count": parent_count,
        "child_state_count": child_count,
        "fractional_reduction": fractional_reduction,
        "max_parent_child_center_divergence": max_divergence,
        "allowed_max_probe_divergence": allowed_divergence,
        "within_bound": True,
        "operational_probe_preservation_certified": False,
    }
    reduction = {
        "map_kind": "QUOTIENT_PROJECTION_WITH_FROZEN_PARENT_SNAPSHOT",
        "quotient_fiber_map": _copy(fiber_map),
        "parent_snapshot_digest": parent_digest,
        "parent_prediction_snapshot_digest": _digest(parent_predictions),
        "exact_parent_recovery_from_snapshot_verified": True,
        "lossy_quotient_requires_parent_snapshot": True,
        "quotient_alone_recovers_parent": False,
    }
    return child, preservation, reduction


def _source_competition(
    report: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    selected = report.get("selected_candidate")
    selected_id = selected.get("candidate_id") if isinstance(selected, Mapping) else None
    commitments = _mapping(
        report.get("candidate_commitments"), "candidate_commitments"
    )
    source = {
        "verification_status": receipt.get("status"),
        "contract_id": report.get("contract_id"),
        "contract_digest": report.get("contract_digest"),
        "report_schema_version": report.get("schema_version"),
        "report_digest": report.get("report_digest"),
        "case_id": report.get("case_id"),
        "disposition": report.get("disposition"),
        "promotion_status": report.get("promotion_status"),
        "candidate_commitment_digest": commitments.get(
            "candidate_commitment_digest"
        ),
        "selected_candidate_id": selected_id,
        "theory_state_digest": report.get("theory_state_digest"),
        "evidence_digests": _copy(report.get("evidence_digests")),
        "input_artifacts": _copy(report.get("input_artifacts")),
    }
    _exact_keys(
        source,
        {
            "verification_status",
            "contract_id",
            "contract_digest",
            "report_schema_version",
            "report_digest",
            "case_id",
            "disposition",
            "promotion_status",
            "candidate_commitment_digest",
            "selected_candidate_id",
            "theory_state_digest",
            "evidence_digests",
            "input_artifacts",
        },
        "source_competition",
    )
    if source["verification_status"] != "VERIFIED_" + str(source["disposition"]):
        raise ShadowTransitionValidationError(
            "source verifier receipt does not match the source disposition"
        )
    if source["promotion_status"] != "NOT_PROMOTED_SHADOW_ONLY":
        raise ShadowTransitionValidationError(
            "source competition is outside the shadow-only boundary"
        )
    return source


def _evaluator_gate(parent: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_evaluator_epoch": parent["evaluator_epoch"],
        "fixed_anchor": parent["fixed_anchor"],
        "child_evaluator_epoch": None,
        "evaluator_status": "UNASSIGNED_NEW_EVALUATOR_REQUIRED",
        "source_evidence_role": "QUALIFICATION_ONLY",
        "source_evidence_allowed_for_child_scoring": False,
        "old_new_records_pooled": False,
        "selective_erasure_applied": False,
        "operational_probe_status": (
            "OPERATIONAL_PROBE_AND_FRESH_EPOCH_REQUIRED"
        ),
        "adoption_blocked": True,
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


def materialize_shadow_theory_transition(
    competition_input: Mapping[str, Any],
    competition_contract: Mapping[str, Any],
    competition_report: Mapping[str, Any],
    transition_contract: Mapping[str, Any],
    *,
    expected_competition_contract_digest: str,
    expected_competition_report_digest: str,
    expected_competition_input_artifacts: Mapping[str, Any] | None,
    expected_transition_contract_digest: str,
    input_artifacts: Mapping[str, Any] | None = None,
) -> ShadowTransitionResult:
    """Replay the source competition and materialize only a detached shadow child."""

    normalized_transition_contract = _validate_contract(transition_contract)
    transition_contract_digest = _validate_expected_transition_contract_digest(
        normalized_transition_contract, expected_transition_contract_digest
    )
    if input_artifacts is not None and not isinstance(input_artifacts, Mapping):
        raise ShadowTransitionValidationError("input_artifacts must be an object or null")
    artifacts = _copy(input_artifacts) if input_artifacts is not None else None

    source_report, receipt = _verified_source(
        competition_input,
        competition_contract,
        competition_report,
        expected_competition_contract_digest=(
            expected_competition_contract_digest
        ),
        expected_competition_report_digest=expected_competition_report_digest,
        expected_competition_input_artifacts=(
            expected_competition_input_artifacts
        ),
    )
    source = _source_competition(source_report, receipt)
    parent, parent_digest = _parent_theory(source_report)
    disposition = str(source_report["disposition"])
    transition_disposition = DISPOSITION_TO_TRANSITION[disposition]
    candidate = _selected_candidate(source_report, disposition)

    operation_kind: str | None = None
    transition_kind: str | None = None
    selected_candidate_id: str | None = None
    child: dict[str, Any] | None = None
    child_digest: str | None = None
    preservation: dict[str, Any] | None = None
    reduction: dict[str, Any] | None = None
    evaluator_gate: dict[str, Any] | None = None

    if disposition == "SELECT_ROBUSTIFICATION":
        assert candidate is not None
        operation_kind = "expand"
        transition_kind = "ROBUST_INTERVAL_EXPANSION"
        selected_candidate_id = candidate["candidate_id"]
        child, preservation, reduction = _materialize_robust(
            parent, parent_digest, source, candidate
        )
        child_digest = _digest(child)
        evaluator_gate = _evaluator_gate(parent)
    elif disposition == "SELECT_IDEALIZATION":
        assert candidate is not None
        operation_kind = "quotient"
        transition_kind = "QUOTIENT_IDEALIZATION"
        selected_candidate_id = candidate["candidate_id"]
        child, preservation, reduction = _materialize_ideal(
            parent,
            parent_digest,
            source,
            competition_contract,
            candidate,
        )
        child_digest = _digest(child)
        evaluator_gate = _evaluator_gate(parent)

    events = [
        _audit_event(
            0,
            "SOURCE_COMPETITION_VERIFIED",
            GENESIS_DIGEST,
            {
                "source_report_digest": source["report_digest"],
                "source_audit_head": source_report["audit_head"],
                "verification_status": source["verification_status"],
                "parent_theory_state_digest": parent_digest,
            },
        )
    ]
    events.append(
        _audit_event(
            1,
            "SHADOW_CHILD_MATERIALIZED"
            if child is not None
            else "SHADOW_TRANSITION_NOT_MATERIALIZED",
            events[-1]["event_digest"],
            {
                "disposition": transition_disposition,
                "operation_kind": operation_kind,
                "transition_kind": transition_kind,
                "selected_candidate_id": selected_candidate_id,
                "child_theory_state_digest": child_digest,
            },
        )
    )
    events.append(
        _audit_event(
            2,
            "ADOPTION_WITHHELD",
            events[-1]["event_digest"],
            {
                "adoption_status": "NOT_ADOPTED_SHADOW_ONLY",
                "child_evaluator_epoch": None,
            },
        )
    )

    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "contract_digest": transition_contract_digest,
        "source_competition": source,
        "parent_theory_state": parent,
        "parent_theory_state_digest": parent_digest,
        "disposition": transition_disposition,
        "operation_kind": operation_kind,
        "transition_kind": transition_kind,
        "selected_candidate_id": selected_candidate_id,
        "child_theory_state": child,
        "child_theory_state_digest": child_digest,
        "preservation_certificate": preservation,
        "reduction_certificate": reduction,
        "evaluator_gate": evaluator_gate,
        "adoption_status": "NOT_ADOPTED_SHADOW_ONLY",
        "nonclaims": _copy(normalized_transition_contract["nonclaims"]),
        "input_artifacts": artifacts,
        "audit_events": events,
        "audit_head": events[-1]["event_digest"],
    }
    report = {**body, "report_digest": _digest(body)}
    _exact_keys(report, REPORT_FIELDS, "transition_report")
    return ShadowTransitionResult(report=report)


def verify_shadow_theory_transition(
    competition_input: Mapping[str, Any],
    competition_contract: Mapping[str, Any],
    competition_report: Mapping[str, Any],
    transition_contract: Mapping[str, Any],
    transition_report: Mapping[str, Any],
    *,
    expected_competition_contract_digest: str,
    expected_competition_report_digest: str,
    expected_competition_input_artifacts: Mapping[str, Any] | None,
    expected_transition_contract_digest: str,
    expected_transition_report_digest: str,
    expected_transition_input_artifacts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Replay and byte-compare a transition report against independent digests."""

    expected_report_digest = _require_digest(
        expected_transition_report_digest,
        "expected_transition_report_digest",
    )
    supplied = _copy(_mapping(transition_report, "transition_report"))
    _exact_keys(supplied, REPORT_FIELDS, "transition_report")
    if expected_transition_input_artifacts is not None and not isinstance(
        expected_transition_input_artifacts, Mapping
    ):
        raise ShadowTransitionValidationError(
            "expected_transition_input_artifacts must be an object or null"
        )
    expected_artifacts = (
        _copy(expected_transition_input_artifacts)
        if expected_transition_input_artifacts is not None
        else None
    )
    if supplied.get("input_artifacts") != expected_artifacts:
        raise ShadowTransitionValidationError(
            "transition report input_artifacts differ from independent expectation"
        )
    fresh = materialize_shadow_theory_transition(
        competition_input,
        competition_contract,
        competition_report,
        transition_contract,
        expected_competition_contract_digest=(
            expected_competition_contract_digest
        ),
        expected_competition_report_digest=expected_competition_report_digest,
        expected_competition_input_artifacts=(
            expected_competition_input_artifacts
        ),
        expected_transition_contract_digest=expected_transition_contract_digest,
        input_artifacts=expected_artifacts,
    ).to_dict()
    if fresh["report_digest"] != expected_report_digest:
        raise ShadowTransitionValidationError(
            "replayed transition report digest differs from expectation"
        )
    if canonical_json_bytes(supplied) != canonical_json_bytes(fresh):
        raise ShadowTransitionValidationError(
            "supplied transition report differs from exact replay"
        )
    return {
        "status": "VERIFIED_" + fresh["disposition"],
        "disposition": fresh["disposition"],
        "report_digest": fresh["report_digest"],
        "contract_digest": fresh["contract_digest"],
        "source_competition_report_digest": fresh["source_competition"][
            "report_digest"
        ],
        "child_theory_state_digest": fresh["child_theory_state_digest"],
        "adoption_status": fresh["adoption_status"],
    }
