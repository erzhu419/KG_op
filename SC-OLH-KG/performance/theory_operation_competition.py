"""Finite, shadow-only competition between theory expansion and compression.

The implementation is deliberately independent of the benchmark executor and
of the structural-hypothesis campaign.  It accepts a finite point-table theory,
uses discovery observations to *construct* interval robustifications and
quotient idealizations, and only then evaluates the frozen candidates on
validation and stress observations.  It never mutates the parent theory or
promotes a scientific claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


CONTRACT_SCHEMA_VERSION = (
    "sc-olh-kg.theory-operation-competition-contract/1"
)
INPUT_SCHEMA_VERSION = "sc-olh-kg.theory-operation-competition-input/1"
REPORT_SCHEMA_VERSION = "sc-olh-kg.theory-operation-competition-report/1"
CONTRACT_ID = "theory_operation_competition_v1"
SCHEMA_VERSION = INPUT_SCHEMA_VERSION
GENESIS_DIGEST = "sha256:" + "0" * 64
SUPPORTED_PROBE_ID = "absolute_error_point_prediction"

OPERATION_KINDS = ("expand", "restrict", "quotient", "language", "probe")
DIAGNOSTIC_ORDER = (
    "reestimate",
    "noise",
    "scope",
    "mixture",
    "quotient",
    "robustify",
    "new_probe",
    "language_last",
)
IDEALIZATION_CONTRACT_FIELDS = (
    "deleted_degrees_of_freedom",
    "preserved_observables",
    "applicable_scale",
    "applicable_task",
    "approximation_error",
    "failure_boundary",
    "counterexample",
    "computational_or_sample_complexity_gain",
    "full_model_recovery_method",
)
MANDATORY_NONCLAIMS = frozenset(
    {
        "shadow_only",
        "no_automatic_promotion",
        "no_paper_promotion",
        "no_run_one",
        "no_benchmark_execution",
        "no_scheduler_access",
        "no_network_access",
        "no_ambient_or_parent_state_write",
        "explicit_cli_out_is_only_optional_write",
        "finite_tabular_theory_only",
        "discovery_only_candidate_synthesis",
        "validation_and_stress_never_enter_synthesis",
        "computed_metrics_are_case_scoped_not_universal",
        "same_epoch_comparisons_only",
        "no_selective_erasure_implementation",
        "input_artifact_binding_requires_independent_expected_values",
        "safety_rate_is_interval_containment_not_domain_safety",
        "fixed_anchor_equality_is_not_external_attestation",
        "report_digest_is_not_a_signature",
        "contract_id_alone_does_not_pin_contract_values",
    }
)


class CompetitionValidationError(ValueError):
    """Raised when an input, contract, or report is outside frozen V1."""


ContractValidationError = CompetitionValidationError
EvidenceValidationError = CompetitionValidationError


class TheoryOperationKind(str, Enum):
    EXPAND = "expand"
    RESTRICT = "restrict"
    QUOTIENT = "quotient"
    LANGUAGE = "language"
    PROBE = "probe"


OperationKind = TheoryOperationKind


class CompetitionDisposition(str, Enum):
    SELECT_ROBUSTIFICATION = "SELECT_ROBUSTIFICATION"
    SELECT_IDEALIZATION = "SELECT_IDEALIZATION"
    NEEDS_EVIDENCE = "NEEDS_EVIDENCE"
    INCOMPARABLE_EVALUATOR_EPOCH = "INCOMPARABLE_EVALUATOR_EPOCH"


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole canonical commitment encoding used by V1."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CompetitionValidationError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value).decode("ascii"))


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CompetitionValidationError(
            f"{label} keys differ; missing={missing}, extra={extra}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompetitionValidationError(f"{label} must be an object")
    if any(type(key) is not str for key in value):
        raise CompetitionValidationError(f"{label} keys must be strings")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CompetitionValidationError(f"{label} must be a list")
    return value


def _string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise CompetitionValidationError(f"{label} must be a non-empty string")
    return value


def _number(value: Any, label: str) -> float:
    if type(value) not in {int, float}:
        raise CompetitionValidationError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise CompetitionValidationError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CompetitionValidationError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _unique_strings(value: Any, label: str, *, nonempty: bool = True) -> tuple[str, ...]:
    items = _list(value, label)
    rendered = tuple(_string(item, f"{label} item") for item in items)
    if nonempty and not rendered:
        raise CompetitionValidationError(f"{label} cannot be empty")
    if len(rendered) != len(set(rendered)):
        raise CompetitionValidationError(f"{label} contains duplicates")
    return rendered


def _json_scalar(value: Any, label: str) -> Any:
    if value is None or type(value) not in {str, int, float, bool}:
        raise CompetitionValidationError(
            f"{label} must be a non-null finite JSON scalar"
        )
    if type(value) is float and not math.isfinite(value):
        raise CompetitionValidationError(f"{label} must be finite")
    return value


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _mae(values: Sequence[float]) -> float:
    return _mean([abs(value) for value in values])


def _sse(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = _mean(values)
    return math.fsum((value - center) ** 2 for value in values)


def _context_key(context: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(context)


def _context_sort_key(context: Mapping[str, Any]) -> bytes:
    return _context_key(context)


def _normalize_context(
    value: Any,
    feature_ids: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    context = _mapping(value, label)
    _exact_keys(context, set(feature_ids), label)
    return {
        feature_id: _json_scalar(context[feature_id], f"{label}.{feature_id}")
        for feature_id in feature_ids
    }


def _validate_contract(contract_value: Any) -> dict[str, Any]:
    contract = _mapping(contract_value, "contract")
    _exact_keys(
        contract,
        {
            "schema_version",
            "contract_id",
            "input_schema_version",
            "report_schema_version",
            "operation_kinds",
            "diagnostic_order",
            "evidence_policy",
            "candidate_generation",
            "thresholds",
            "score_weights",
            "idealization_contract_fields",
            "selection",
            "nonclaims",
        },
        "contract",
    )
    if contract["schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise CompetitionValidationError("unexpected contract schema_version")
    if contract["contract_id"] != CONTRACT_ID:
        raise CompetitionValidationError("unexpected contract_id")
    if contract["input_schema_version"] != INPUT_SCHEMA_VERSION:
        raise CompetitionValidationError("unexpected input_schema_version")
    if contract["report_schema_version"] != REPORT_SCHEMA_VERSION:
        raise CompetitionValidationError("unexpected report_schema_version")
    if tuple(contract["operation_kinds"]) != OPERATION_KINDS:
        raise CompetitionValidationError("operation_kinds are not frozen V1")
    if tuple(contract["diagnostic_order"]) != DIAGNOSTIC_ORDER:
        raise CompetitionValidationError("diagnostic_order is not frozen V1")
    if tuple(contract["idealization_contract_fields"]) != IDEALIZATION_CONTRACT_FIELDS:
        raise CompetitionValidationError(
            "idealization_contract_fields are not the required nine fields"
        )

    evidence_policy = _mapping(contract["evidence_policy"], "evidence_policy")
    _exact_keys(
        evidence_policy,
        {
            "min_discovery_rows",
            "min_validation_rows",
            "min_stress_rows",
            "require_disjoint_observation_ids",
            "require_exact_theory_epoch",
            "require_exact_fixed_anchor",
            "require_registered_contexts",
        },
        "evidence_policy",
    )
    for key in ("min_discovery_rows", "min_validation_rows", "min_stress_rows"):
        _integer(evidence_policy[key], f"evidence_policy.{key}", minimum=1)
    for key in (
        "require_disjoint_observation_ids",
        "require_exact_theory_epoch",
        "require_exact_fixed_anchor",
        "require_registered_contexts",
    ):
        if type(evidence_policy[key]) is not bool or not evidence_policy[key]:
            raise CompetitionValidationError(f"evidence_policy.{key} must be true")

    generation = _mapping(contract["candidate_generation"], "candidate_generation")
    _exact_keys(
        generation,
        {
            "robust_interval_groupings",
            "interval_radius_statistic",
            "enumerate_nonempty_removable_subsets",
            "quotient_aggregation",
            "quotient_restore",
            "supported_probe_id",
            "language_is_last_resort_only",
        },
        "candidate_generation",
    )
    if tuple(generation["robust_interval_groupings"]) != (
        "global",
        "per_scope",
        "per_context",
    ):
        raise CompetitionValidationError("robust interval groupings changed")
    if generation["interval_radius_statistic"] != "discovery_max_absolute_residual":
        raise CompetitionValidationError("interval radius statistic changed")
    if generation["quotient_aggregation"] != "mean_parent_prediction":
        raise CompetitionValidationError("quotient aggregation changed")
    if generation["quotient_restore"] != "restore_frozen_parent_point_table":
        raise CompetitionValidationError("quotient restore changed")
    if generation["supported_probe_id"] != SUPPORTED_PROBE_ID:
        raise CompetitionValidationError("supported probe changed")
    for key in (
        "enumerate_nonempty_removable_subsets",
        "language_is_last_resort_only",
    ):
        if type(generation[key]) is not bool or not generation[key]:
            raise CompetitionValidationError(f"candidate_generation.{key} must be true")

    thresholds = _mapping(contract["thresholds"], "thresholds")
    threshold_keys = {
        "numeric_epsilon",
        "coverage_tolerance",
        "tail_fraction",
        "min_validation_coverage",
        "min_stress_tail_coverage",
        "min_safety_rate",
        "max_nominal_mae_increase",
        "max_probe_divergence",
        "min_complexity_reduction",
        "max_normalized_radius",
        "min_score_margin",
        "reestimate_min_relative_improvement",
        "noise_min_relative_sse",
        "scope_min_relative_spread",
        "mixture_min_relative_sse_reduction",
        "new_probe_residual_fraction",
        "language_residual_fraction",
    }
    _exact_keys(thresholds, threshold_keys, "thresholds")
    for key in threshold_keys:
        number = _number(thresholds[key], f"thresholds.{key}")
        if number < 0:
            raise CompetitionValidationError(f"thresholds.{key} cannot be negative")
    for key in (
        "coverage_tolerance",
        "tail_fraction",
        "min_validation_coverage",
        "min_stress_tail_coverage",
        "min_safety_rate",
        "min_complexity_reduction",
        "reestimate_min_relative_improvement",
        "noise_min_relative_sse",
        "scope_min_relative_spread",
        "mixture_min_relative_sse_reduction",
        "new_probe_residual_fraction",
        "language_residual_fraction",
    ):
        if float(thresholds[key]) > 1:
            raise CompetitionValidationError(f"thresholds.{key} must be <= 1")

    weights = _mapping(contract["score_weights"], "score_weights")
    weight_keys = {
        "validation_mae_improvement",
        "stress_mae_improvement",
        "coverage",
        "tail_coverage",
        "safety",
        "complexity_reduction",
        "probe_divergence_penalty",
        "radius_cost_penalty",
    }
    _exact_keys(weights, weight_keys, "score_weights")
    for key in weight_keys:
        if _number(weights[key], f"score_weights.{key}") < 0:
            raise CompetitionValidationError(f"score_weights.{key} cannot be negative")

    selection = _mapping(contract["selection"], "selection")
    _exact_keys(
        selection,
        {
            "robustification_status",
            "idealization_status",
            "insufficient_status",
            "cross_epoch_status",
            "promotion_status",
        },
        "selection",
    )
    if selection != {
        "robustification_status": CompetitionDisposition.SELECT_ROBUSTIFICATION.value,
        "idealization_status": CompetitionDisposition.SELECT_IDEALIZATION.value,
        "insufficient_status": CompetitionDisposition.NEEDS_EVIDENCE.value,
        "cross_epoch_status": CompetitionDisposition.INCOMPARABLE_EVALUATOR_EPOCH.value,
        "promotion_status": "NOT_PROMOTED_SHADOW_ONLY",
    }:
        raise CompetitionValidationError("selection status vocabulary changed")

    nonclaims = _unique_strings(contract["nonclaims"], "nonclaims")
    if not MANDATORY_NONCLAIMS.issubset(nonclaims):
        missing = sorted(MANDATORY_NONCLAIMS.difference(nonclaims))
        raise CompetitionValidationError(f"mandatory nonclaims missing: {missing}")
    return _json_copy(contract)


def validate_contract(contract_value: Any) -> dict[str, Any]:
    """Validate and return a detached canonical copy of the V1 contract."""

    return _validate_contract(contract_value)


def _normalize_theory(theory_value: Any) -> dict[str, Any]:
    theory = _mapping(theory_value, "theory_state")
    _exact_keys(
        theory,
        {
            "theory_id",
            "task_id",
            "evaluator_epoch",
            "fixed_anchor",
            "object_space",
            "model_class",
            "probe_ids",
            "violation_functionals",
            "scope_ids",
            "removable_feature_ids",
        },
        "theory_state",
    )
    feature_ids: tuple[str, ...]
    object_space = _mapping(theory["object_space"], "object_space")
    _exact_keys(object_space, {"feature_ids", "contexts"}, "object_space")
    feature_ids = tuple(
        sorted(_unique_strings(object_space["feature_ids"], "feature_ids"))
    )
    contexts = [
        _normalize_context(item, feature_ids, f"contexts[{index}]")
        for index, item in enumerate(_list(object_space["contexts"], "contexts"))
    ]
    if not contexts:
        raise CompetitionValidationError("contexts cannot be empty")
    context_keys = [_context_key(item) for item in contexts]
    if len(context_keys) != len(set(context_keys)):
        raise CompetitionValidationError("contexts contain duplicates")
    registered = {key: context for key, context in zip(context_keys, contexts)}

    model_class = _mapping(theory["model_class"], "model_class")
    _exact_keys(model_class, {"kind", "predictions"}, "model_class")
    if model_class["kind"] != "finite_point_table":
        raise CompetitionValidationError("model_class.kind must be finite_point_table")
    predictions: dict[bytes, float] = {}
    normalized_predictions = []
    for index, raw_item in enumerate(_list(model_class["predictions"], "predictions")):
        item = _mapping(raw_item, f"predictions[{index}]")
        _exact_keys(item, {"context", "value"}, f"predictions[{index}]")
        context = _normalize_context(item["context"], feature_ids, f"predictions[{index}].context")
        key = _context_key(context)
        if key not in registered:
            raise CompetitionValidationError("prediction context is not registered")
        if key in predictions:
            raise CompetitionValidationError("duplicate prediction context")
        value = _number(item["value"], f"predictions[{index}].value")
        predictions[key] = value
        normalized_predictions.append({"context": context, "value": value})
    if set(predictions) != set(registered):
        raise CompetitionValidationError("point table must predict every registered context")

    probe_ids = tuple(sorted(_unique_strings(theory["probe_ids"], "probe_ids")))
    if probe_ids != (SUPPORTED_PROBE_ID,):
        raise CompetitionValidationError(
            f"V1 requires exactly the executable probe {SUPPORTED_PROBE_ID!r}"
        )
    scope_ids = tuple(sorted(_unique_strings(theory["scope_ids"], "scope_ids")))
    removable = tuple(
        sorted(
            _unique_strings(
                theory["removable_feature_ids"], "removable_feature_ids"
            )
        )
    )
    if not set(removable).issubset(feature_ids):
        raise CompetitionValidationError("removable features must be registered")

    functionals = _list(theory["violation_functionals"], "violation_functionals")
    if len(functionals) != 1:
        raise CompetitionValidationError("V1 requires exactly one violation functional")
    functional = _mapping(functionals[0], "violation_functionals[0]")
    _exact_keys(functional, {"functional_id", "threshold"}, "violation_functionals[0]")
    normalized_functional = {
        "functional_id": _string(functional["functional_id"], "functional_id"),
        "threshold": _number(functional["threshold"], "functional threshold"),
    }
    if normalized_functional["functional_id"] != "absolute_error":
        raise CompetitionValidationError(
            "V1 supports only the absolute_error violation functional"
        )
    if normalized_functional["threshold"] < 0:
        raise CompetitionValidationError("functional threshold cannot be negative")

    contexts = sorted(contexts, key=_context_sort_key)
    normalized_predictions = sorted(
        normalized_predictions, key=lambda item: _context_sort_key(item["context"])
    )
    public = {
        "theory_id": _string(theory["theory_id"], "theory_id"),
        "task_id": _string(theory["task_id"], "task_id"),
        "evaluator_epoch": _string(theory["evaluator_epoch"], "evaluator_epoch"),
        "fixed_anchor": _string(theory["fixed_anchor"], "fixed_anchor"),
        "object_space": {"feature_ids": list(feature_ids), "contexts": contexts},
        "model_class": {
            "kind": "finite_point_table",
            "predictions": normalized_predictions,
        },
        "probe_ids": list(probe_ids),
        "violation_functionals": [normalized_functional],
        "scope_ids": list(scope_ids),
        "removable_feature_ids": list(removable),
    }
    return {
        "public": public,
        "feature_ids": feature_ids,
        "contexts": contexts,
        "registered": registered,
        "predictions": predictions,
        "probe_ids": probe_ids,
        "scope_ids": scope_ids,
        "removable": removable,
        "threshold": normalized_functional["threshold"],
    }


def _normalize_rows(
    rows_value: Any,
    split: str,
    theory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for index, raw_row in enumerate(_list(rows_value, split)):
        row = _mapping(raw_row, f"{split}[{index}]")
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
            f"{split}[{index}]",
        )
        context = _normalize_context(
            row["context"], theory["feature_ids"], f"{split}[{index}].context"
        )
        key = _context_key(context)
        if key not in theory["registered"]:
            raise CompetitionValidationError(f"{split} row context is unregistered")
        scope_id = _string(row["scope_id"], f"{split}[{index}].scope_id")
        if scope_id not in theory["scope_ids"]:
            raise CompetitionValidationError(f"{split} row scope is unregistered")
        rows.append(
            {
                "observation_id": _string(
                    row["observation_id"], f"{split}[{index}].observation_id"
                ),
                "evaluator_epoch": _string(
                    row["evaluator_epoch"], f"{split}[{index}].evaluator_epoch"
                ),
                "fixed_anchor": _string(
                    row["fixed_anchor"], f"{split}[{index}].fixed_anchor"
                ),
                "scope_id": scope_id,
                "context": context,
                "observed_value": _number(
                    row["observed_value"], f"{split}[{index}].observed_value"
                ),
            }
        )
    rows.sort(key=lambda row: row["observation_id"])
    ids = [row["observation_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise CompetitionValidationError(f"{split} observation IDs are duplicated")
    return rows


def _normalize_input(input_value: Any) -> dict[str, Any]:
    value = _mapping(input_value, "input")
    _exact_keys(
        value,
        {"schema_version", "case_id", "theory_state", "evidence"},
        "input",
    )
    if value["schema_version"] != INPUT_SCHEMA_VERSION:
        raise CompetitionValidationError("unexpected input schema_version")
    theory = _normalize_theory(value["theory_state"])
    evidence = _mapping(value["evidence"], "evidence")
    _exact_keys(evidence, {"discovery", "validation", "stress"}, "evidence")
    normalized_evidence = {
        split: _normalize_rows(evidence[split], split, theory)
        for split in ("discovery", "validation", "stress")
    }
    ids = [
        row["observation_id"]
        for split in normalized_evidence.values()
        for row in split
    ]
    if len(ids) != len(set(ids)):
        raise CompetitionValidationError("observation IDs must be disjoint across splits")
    public = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "case_id": _string(value["case_id"], "case_id"),
        "theory_state": theory["public"],
        "evidence": normalized_evidence,
    }
    return {
        "public": public,
        "case_id": public["case_id"],
        "theory": theory,
        "evidence": normalized_evidence,
    }


def _parent_prediction(theory: Mapping[str, Any], context: Mapping[str, Any]) -> float:
    return theory["predictions"][_context_key(context)]


def _residuals(
    theory: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[float]:
    return [
        float(row["observed_value"]) - _parent_prediction(theory, row["context"])
        for row in rows
    ]


def _identified_candidate(body: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    result = _json_copy(body)
    result["candidate_id"] = f"{prefix}:{_digest(body)[7:]}"
    return result


def _robust_candidates(
    theory: Mapping[str, Any],
    discovery: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    residual_by_row = [
        (row, abs(float(row["observed_value"]) - _parent_prediction(theory, row["context"])))
        for row in discovery
    ]
    candidates = []
    global_radius = max(
        (residual for _, residual in residual_by_row), default=0.0
    )
    candidates.append(
        _identified_candidate(
            {
                "operation_kind": TheoryOperationKind.EXPAND.value,
                "family": "robust_interval",
                "mechanism": "interval",
                "grouping": "global",
                "radii": [{"group": {"global": "*"}, "radius": global_radius}],
                "reduction_map": {
                    "parent_model_class": "finite_point_table",
                    "child_model_class": "finite_interval_table",
                    "zero_radius_recovers_parent": True,
                    "limit_parameter": "radius",
                    "limit_value": 0.0,
                },
            },
            "robust",
        )
    )

    by_scope: dict[str, list[float]] = {scope: [] for scope in theory["scope_ids"]}
    for row, residual in residual_by_row:
        by_scope[row["scope_id"]].append(residual)
    if all(by_scope.values()):
        candidates.append(
            _identified_candidate(
                {
                    "operation_kind": TheoryOperationKind.EXPAND.value,
                    "family": "robust_interval",
                    "mechanism": "interval",
                    "grouping": "per_scope",
                    "radii": [
                        {
                            "group": {"scope_id": scope},
                            "radius": max(by_scope[scope]),
                        }
                        for scope in sorted(by_scope)
                    ],
                    "reduction_map": {
                        "parent_model_class": "finite_point_table",
                        "child_model_class": "finite_scope_interval_table",
                        "zero_radius_recovers_parent": True,
                        "limit_parameter": "radius",
                        "limit_value": 0.0,
                    },
                },
                "robust",
            )
        )

    by_context: dict[bytes, list[float]] = {
        _context_key(context): [] for context in theory["contexts"]
    }
    for row, residual in residual_by_row:
        by_context[_context_key(row["context"])].append(residual)
    if all(by_context.values()):
        radii = []
        for context in theory["contexts"]:
            values = by_context[_context_key(context)]
            radii.append(
                {
                    "group": {"context": context},
                    "radius": max(values),
                }
            )
        candidates.append(
            _identified_candidate(
                {
                    "operation_kind": TheoryOperationKind.EXPAND.value,
                    "family": "robust_interval",
                    "mechanism": "interval",
                    "grouping": "per_context",
                    "radii": radii,
                    "reduction_map": {
                        "parent_model_class": "finite_point_table",
                        "child_model_class": "finite_context_interval_table",
                        "zero_radius_recovers_parent": True,
                        "limit_parameter": "radius",
                        "limit_value": 0.0,
                    },
                },
                "robust",
            )
        )
    return sorted(candidates, key=lambda item: item["candidate_id"])


def _quotient_context(
    context: Mapping[str, Any], removed: frozenset[str], feature_ids: Sequence[str]
) -> dict[str, Any]:
    return {
        feature_id: context[feature_id]
        for feature_id in feature_ids
        if feature_id not in removed
    }


def _idealization_candidates(
    theory: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    threshold = float(contract["thresholds"]["max_probe_divergence"])
    candidates = []
    removable = tuple(
        feature_id
        for feature_id in theory["feature_ids"]
        if feature_id in theory["removable"]
    )
    for count in range(1, len(removable) + 1):
        for removed_items in combinations(removable, count):
            removed = frozenset(removed_items)
            groups: dict[bytes, dict[str, Any]] = {}
            for context in theory["contexts"]:
                quotient = _quotient_context(context, removed, theory["feature_ids"])
                key = _context_key(quotient)
                group = groups.setdefault(
                    key, {"context": quotient, "values": [], "parents": []}
                )
                group["values"].append(_parent_prediction(theory, context))
                group["parents"].append(context)
            quotient_predictions = [
                {"context": groups[key]["context"], "value": _mean(groups[key]["values"])}
                for key in sorted(groups)
            ]
            quotient_lookup = {
                _context_key(item["context"]): float(item["value"])
                for item in quotient_predictions
            }
            recovery_map = []
            divergences = []
            counterexamples = []
            for context in theory["contexts"]:
                quotient = _quotient_context(context, removed, theory["feature_ids"])
                divergence = abs(
                    _parent_prediction(theory, context)
                    - quotient_lookup[_context_key(quotient)]
                )
                divergences.append(divergence)
                recovery_map.append(
                    {"parent_context": context, "quotient_context": quotient}
                )
                if divergence > threshold:
                    counterexamples.append(context)
            parent_states = len(theory["contexts"])
            child_states = len(groups)
            reduction = (parent_states - child_states) / parent_states
            idealization_contract = {
                "deleted_degrees_of_freedom": list(removed_items),
                "preserved_observables": list(theory["probe_ids"]),
                "applicable_scale": {"scope_ids": list(theory["scope_ids"])},
                "applicable_task": theory["public"]["task_id"],
                "approximation_error": max(divergences, default=0.0),
                "failure_boundary": {
                    "predicate": "parent_quotient_probe_divergence_exceeds_contract",
                    "threshold": threshold,
                    "count": len(counterexamples),
                },
                "counterexample": counterexamples,
                "computational_or_sample_complexity_gain": {
                    "parent_states": parent_states,
                    "child_states": child_states,
                    "absolute_reduction": parent_states - child_states,
                    "fractional_reduction": reduction,
                },
                "full_model_recovery_method": {
                    "kind": "restore_frozen_parent_point_table",
                    "parent_theory_state_digest": _digest(theory["public"]),
                    "parent_predictions": theory["public"]["model_class"]["predictions"],
                    "quotient_fiber_map": recovery_map,
                    "lossy_quotient_requires_parent_snapshot": True,
                },
            }
            candidates.append(
                _identified_candidate(
                    {
                        "operation_kind": TheoryOperationKind.QUOTIENT.value,
                        "family": "idealization_quotient",
                        "mechanism": "quotient",
                        "removed_feature_ids": list(removed_items),
                        "retained_feature_ids": [
                            item for item in theory["feature_ids"] if item not in removed
                        ],
                        "quotient_predictions": quotient_predictions,
                        "idealization_contract": idealization_contract,
                    },
                    "ideal",
                )
            )
    return sorted(candidates, key=lambda item: item["candidate_id"])


def synthesize_theory_operation_candidates(
    theory_state: Mapping[str, Any],
    discovery_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct candidates without accepting validation or stress labels."""

    normalized_contract = _validate_contract(contract)
    theory = _normalize_theory(theory_state)
    discovery = _normalize_rows(list(discovery_rows), "discovery", theory)
    if any(
        row["evaluator_epoch"] != theory["public"]["evaluator_epoch"]
        or row["fixed_anchor"] != theory["public"]["fixed_anchor"]
        for row in discovery
    ):
        raise CompetitionValidationError(
            "candidate synthesis requires discovery at the exact theory "
            "evaluator epoch and fixed anchor"
        )
    robust = _robust_candidates(theory, discovery)
    ideal = _idealization_candidates(theory, normalized_contract)
    commitment_body = {
        "theory_state_digest": _digest(theory["public"]),
        "discovery_digest": _digest(discovery),
        "robustification_candidates": robust,
        "idealization_candidates": ideal,
    }
    return {
        **commitment_body,
        "candidate_commitment_digest": _digest(commitment_body),
    }


def _baseline_split_metrics(
    theory: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], tail_fraction: float
) -> dict[str, Any]:
    residual_pairs = [
        (
            abs(float(row["observed_value"]) - _parent_prediction(theory, row["context"])),
            row["observation_id"],
            row,
        )
        for row in rows
    ]
    threshold = float(theory["threshold"])
    covered = [residual <= threshold for residual, _, _ in residual_pairs]
    tail_count = max(1, math.ceil(len(rows) * tail_fraction)) if rows else 0
    tail = sorted(residual_pairs, key=lambda item: (-item[0], item[1]))[:tail_count]
    tail_coverage = (
        _mean([1.0 if residual <= threshold else 0.0 for residual, _, _ in tail])
        if tail
        else 0.0
    )
    return {
        "row_count": len(rows),
        "mae": _mean([item[0] for item in residual_pairs]),
        "coverage": _mean([1.0 if item else 0.0 for item in covered]),
        "tail_coverage": tail_coverage,
        "safety_rate": _mean([1.0 if item else 0.0 for item in covered]),
    }


def _robust_radius(candidate: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    grouping = candidate["grouping"]
    if grouping == "global":
        return float(candidate["radii"][0]["radius"])
    if grouping == "per_scope":
        lookup = {
            item["group"]["scope_id"]: float(item["radius"])
            for item in candidate["radii"]
        }
        return lookup[row["scope_id"]]
    if grouping == "per_context":
        lookup = {
            _context_key(item["group"]["context"]): float(item["radius"])
            for item in candidate["radii"]
        }
        return lookup[_context_key(row["context"])]
    raise CompetitionValidationError("unknown robust grouping")


def _robust_split_metrics(
    theory: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    tail_fraction: float,
) -> dict[str, Any]:
    values = []
    for row in rows:
        residual = abs(
            float(row["observed_value"]) - _parent_prediction(theory, row["context"])
        )
        radius = _robust_radius(candidate, row)
        values.append((residual, radius, row["observation_id"], row))
    tail_count = max(1, math.ceil(len(rows) * tail_fraction)) if rows else 0
    tail = sorted(values, key=lambda item: (-item[0], item[2]))[:tail_count]
    return {
        "row_count": len(rows),
        "mae": _mean([item[0] for item in values]),
        "coverage": _mean([1.0 if residual <= radius else 0.0 for residual, radius, _, _ in values]),
        "tail_coverage": _mean(
            [1.0 if residual <= radius else 0.0 for residual, radius, _, _ in tail]
        ) if tail else 0.0,
        "safety_rate": _mean(
            [1.0 if residual <= radius else 0.0 for residual, radius, _, _ in values]
        ),
        "mean_radius": _mean([radius for _, radius, _, _ in values]),
    }


def _evaluate_robust(
    theory: Mapping[str, Any],
    candidate: Mapping[str, Any],
    validation: Sequence[Mapping[str, Any]],
    stress: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    baseline_validation: Mapping[str, Any],
    baseline_stress: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = contract["thresholds"]
    weights = contract["score_weights"]
    tail_fraction = float(thresholds["tail_fraction"])
    validation_metrics = _robust_split_metrics(
        theory, validation, candidate, tail_fraction
    )
    stress_metrics = _robust_split_metrics(theory, stress, candidate, tail_fraction)
    prediction_scale = max(
        float(thresholds["numeric_epsilon"]),
        _mean([abs(value) for value in theory["predictions"].values()])
        + float(theory["threshold"]),
    )
    normalized_radius = (
        validation_metrics["mean_radius"] + stress_metrics["mean_radius"]
    ) / (2 * prediction_scale)
    validation_gain = validation_metrics["coverage"] - baseline_validation["coverage"]
    stress_tail_gain = stress_metrics["tail_coverage"] - baseline_stress["tail_coverage"]
    viable = all(
        (
            validation_gain >= float(thresholds["coverage_tolerance"]),
            validation_metrics["coverage"] >= float(thresholds["min_validation_coverage"]),
            stress_metrics["tail_coverage"] >= float(thresholds["min_stress_tail_coverage"]),
            stress_metrics["safety_rate"] >= float(thresholds["min_safety_rate"]),
            normalized_radius <= float(thresholds["max_normalized_radius"]),
        )
    )
    score = (
        float(weights["coverage"]) * validation_metrics["coverage"]
        + float(weights["tail_coverage"]) * stress_metrics["tail_coverage"]
        + float(weights["safety"]) * stress_metrics["safety_rate"]
        + float(weights["validation_mae_improvement"])
        * (baseline_validation["mae"] - validation_metrics["mae"])
        + float(weights["stress_mae_improvement"])
        * (baseline_stress["mae"] - stress_metrics["mae"])
        - float(weights["radius_cost_penalty"]) * normalized_radius
    )
    return {
        **_json_copy(candidate),
        "evaluation": {
            "baseline_validation": baseline_validation,
            "baseline_stress": baseline_stress,
            "validation": validation_metrics,
            "stress": stress_metrics,
            "validation_coverage_gain": validation_gain,
            "stress_tail_coverage_gain": stress_tail_gain,
            "normalized_radius": normalized_radius,
            "score": score,
            "viable": viable,
        },
    }


def _ideal_prediction(
    candidate: Mapping[str, Any], context: Mapping[str, Any]
) -> float:
    removed = frozenset(candidate["removed_feature_ids"])
    quotient = {
        key: value for key, value in context.items() if key not in removed
    }
    lookup = {
        _context_key(item["context"]): float(item["value"])
        for item in candidate["quotient_predictions"]
    }
    return lookup[_context_key(quotient)]


def _ideal_split_metrics(
    theory: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    parent_errors = []
    candidate_errors = []
    divergences = []
    for row in rows:
        parent = _parent_prediction(theory, row["context"])
        child = _ideal_prediction(candidate, row["context"])
        observed = float(row["observed_value"])
        parent_errors.append(abs(observed - parent))
        candidate_errors.append(abs(observed - child))
        divergences.append(abs(parent - child))
    return {
        "row_count": len(rows),
        "baseline_mae": _mean(parent_errors),
        "candidate_mae": _mean(candidate_errors),
        "mae_increase": _mean(candidate_errors) - _mean(parent_errors),
        "max_probe_divergence": max(divergences, default=0.0),
    }


def _evaluate_ideal(
    theory: Mapping[str, Any],
    candidate: Mapping[str, Any],
    validation: Sequence[Mapping[str, Any]],
    stress: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = contract["thresholds"]
    weights = contract["score_weights"]
    validation_metrics = _ideal_split_metrics(theory, validation, candidate)
    stress_metrics = _ideal_split_metrics(theory, stress, candidate)
    complexity = float(
        candidate["idealization_contract"]["computational_or_sample_complexity_gain"]["fractional_reduction"]
    )
    heldout_max_divergence = max(
        validation_metrics["max_probe_divergence"],
        stress_metrics["max_probe_divergence"],
    )
    object_space_max_divergence = float(
        candidate["idealization_contract"]["approximation_error"]
    )
    max_divergence = max(heldout_max_divergence, object_space_max_divergence)
    viable = all(
        (
            complexity >= float(thresholds["min_complexity_reduction"]),
            max_divergence <= float(thresholds["max_probe_divergence"]),
            validation_metrics["mae_increase"]
            <= float(thresholds["max_nominal_mae_increase"]),
            stress_metrics["mae_increase"]
            <= float(thresholds["max_nominal_mae_increase"]),
        )
    )
    score = (
        float(weights["complexity_reduction"]) * complexity
        - float(weights["probe_divergence_penalty"]) * max_divergence
        - float(weights["validation_mae_improvement"])
        * validation_metrics["mae_increase"]
        - float(weights["stress_mae_improvement"])
        * stress_metrics["mae_increase"]
    )
    return {
        **_json_copy(candidate),
        "evaluation": {
            "validation": validation_metrics,
            "stress": stress_metrics,
            "complexity_reduction": complexity,
            "heldout_max_probe_divergence": heldout_max_divergence,
            "object_space_max_probe_divergence": object_space_max_divergence,
            "admission_max_probe_divergence": max_divergence,
            "score": score,
            "viable": viable,
        },
    }


def _mixture_metric(residuals: Sequence[float], epsilon: float) -> dict[str, Any]:
    ordered = sorted(residuals)
    baseline = _sse(ordered)
    if len(ordered) < 2 or baseline <= epsilon:
        return {"relative_sse_reduction": 0.0, "split_index": None}
    best_sse = baseline
    best_index = None
    for index in range(1, len(ordered)):
        candidate = _sse(ordered[:index]) + _sse(ordered[index:])
        if candidate < best_sse - epsilon:
            best_sse = candidate
            best_index = index
    return {
        "relative_sse_reduction": (baseline - best_sse) / baseline,
        "split_index": best_index,
    }


def _base_diagnostics(
    theory: Mapping[str, Any],
    discovery: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    thresholds = contract["thresholds"]
    epsilon = float(thresholds["numeric_epsilon"])
    residuals = _residuals(theory, discovery)
    baseline_mae = _mae(residuals)
    bias = _mean(residuals)
    reestimated_mae = _mae([value - bias for value in residuals])
    relative_improvement = (
        (baseline_mae - reestimated_mae) / baseline_mae
        if baseline_mae > epsilon
        else 0.0
    )

    grouped: dict[bytes, list[float]] = {}
    for row, residual in zip(discovery, residuals):
        grouped.setdefault(_context_key(row["context"]), []).append(residual)
    within_sse = math.fsum(_sse(values) for values in grouped.values())
    total_sse = _sse(residuals)
    repeat_groups = sum(1 for values in grouped.values() if len(values) >= 2)
    noise_ratio = within_sse / total_sse if total_sse > epsilon else 0.0

    scope_abs: dict[str, list[float]] = {scope: [] for scope in theory["scope_ids"]}
    for row, residual in zip(discovery, residuals):
        scope_abs[row["scope_id"]].append(abs(residual))
    scope_means = {scope: _mean(values) for scope, values in scope_abs.items() if values}
    max_scope = max(scope_means.values(), default=0.0)
    min_scope = min(scope_means.values(), default=0.0)
    scope_spread = (max_scope - min_scope) / max_scope if max_scope > epsilon else 0.0
    violating_count = sum(abs(value) > theory["threshold"] for value in residuals)
    violating_fraction = violating_count / len(residuals) if residuals else 0.0
    violating_residuals = [
        value for value in residuals if abs(value) > float(theory["threshold"])
    ]
    mixture = _mixture_metric(violating_residuals, epsilon)
    return {
        "reestimate": {
            "baseline_mae": baseline_mae,
            "reestimated_bias": bias,
            "reestimated_mae": reestimated_mae,
            "relative_improvement": relative_improvement,
            "disposition": "VIABLE_EXPLANATION"
            if relative_improvement
            >= float(thresholds["reestimate_min_relative_improvement"])
            else "EXCLUDED_BY_DISCOVERY",
        },
        "noise": {
            "repeat_group_count": repeat_groups,
            "within_context_sse_fraction": noise_ratio,
            "disposition": "NEEDS_REPEATED_OBSERVATIONS"
            if repeat_groups == 0
            else (
                "VIABLE_EXPLANATION"
                if noise_ratio >= float(thresholds["noise_min_relative_sse"])
                else "EXCLUDED_BY_DISCOVERY"
            ),
        },
        "scope": {
            "scope_mean_absolute_residual": scope_means,
            "relative_scope_spread": scope_spread,
            "violating_count": violating_count,
            "violating_fraction": violating_fraction,
            "structured_repeated_violation": violating_count >= 2,
            "disposition": "VIABLE_EXPLANATION"
            if scope_spread >= float(thresholds["scope_min_relative_spread"])
            else "EXCLUDED_BY_DISCOVERY",
        },
        "mixture": {
            **mixture,
            "violating_residual_count": len(violating_residuals),
            "disposition": "VIABLE_EXPLANATION"
            if mixture["relative_sse_reduction"]
            >= float(thresholds["mixture_min_relative_sse_reduction"])
            else "EXCLUDED_BY_DISCOVERY",
        },
    }


def _cross_epoch_diagnostics() -> dict[str, dict[str, Any]]:
    """Return a non-numeric ladder when discovery is not comparable."""

    return {
        stage: {"disposition": "NOT_EVALUATED_INCOMPARABLE_EVALUATOR_EPOCH"}
        for stage in ("reestimate", "noise", "scope", "mixture")
    }


def _empty_synthesis(
    theory: Mapping[str, Any], discovery: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Commit to blocked synthesis without consuming cross-epoch values."""

    body = {
        "theory_state_digest": _digest(theory["public"]),
        "discovery_digest": _digest(discovery),
        "robustification_candidates": [],
        "idealization_candidates": [],
    }
    return {**body, "candidate_commitment_digest": _digest(body)}


def _audit_event(
    sequence: int, event: str, previous: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    body = {
        "sequence": sequence,
        "event": event,
        "previous_event_digest": previous,
        "payload": _json_copy(payload),
    }
    return {**body, "event_digest": _digest(body)}


@dataclass(frozen=True)
class CompetitionResult:
    report: Mapping[str, Any]

    @property
    def disposition(self) -> str:
        return str(self.report["disposition"])

    @property
    def report_digest(self) -> str:
        return str(self.report["report_digest"])

    def to_dict(self) -> dict[str, Any]:
        return _json_copy(self.report)


def run_theory_operation_competition(
    input_payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    input_artifacts: Mapping[str, Any] | None = None,
) -> CompetitionResult:
    normalized_contract = _validate_contract(contract)
    normalized_input = _normalize_input(input_payload)
    theory = normalized_input["theory"]
    evidence = normalized_input["evidence"]
    if input_artifacts is not None and not isinstance(input_artifacts, Mapping):
        raise CompetitionValidationError("input_artifacts must be an object or null")
    artifacts = _json_copy(input_artifacts) if input_artifacts is not None else None

    binding_by_split = {
        split: all(
            row["evaluator_epoch"] == theory["public"]["evaluator_epoch"]
            and row["fixed_anchor"] == theory["public"]["fixed_anchor"]
            for row in rows
        )
        for split, rows in evidence.items()
    }
    discovery_compatible = binding_by_split["discovery"]
    epochs_match = all(binding_by_split.values())
    if discovery_compatible:
        synthesis = synthesize_theory_operation_candidates(
            normalized_input["public"]["theory_state"],
            evidence["discovery"],
            normalized_contract,
        )
        base_diagnostics = _base_diagnostics(
            theory, evidence["discovery"], normalized_contract
        )
    else:
        synthesis = _empty_synthesis(theory, evidence["discovery"])
        base_diagnostics = _cross_epoch_diagnostics()
    policy = normalized_contract["evidence_policy"]
    enough_rows = all(
        len(evidence[split]) >= int(policy[f"min_{split}_rows"])
        for split in ("discovery", "validation", "stress")
    )
    scope_coverage_by_split = {
        split: sorted({row["scope_id"] for row in rows})
        for split, rows in evidence.items()
    }
    missing_scope_coverage = {
        split: sorted(set(theory["scope_ids"]) - set(covered))
        for split, covered in scope_coverage_by_split.items()
    }
    scope_coverage_complete = not any(missing_scope_coverage.values())
    earlier_blocking_stages = [
        stage
        for stage in ("reestimate", "noise", "scope", "mixture")
        if base_diagnostics[stage]["disposition"] != "EXCLUDED_BY_DISCOVERY"
    ]
    baseline_validation = (
        _baseline_split_metrics(
            theory,
            evidence["validation"],
            float(normalized_contract["thresholds"]["tail_fraction"]),
        )
        if epochs_match
        else None
    )
    baseline_stress = (
        _baseline_split_metrics(
            theory,
            evidence["stress"],
            float(normalized_contract["thresholds"]["tail_fraction"]),
        )
        if epochs_match
        else None
    )
    candidate_evaluation_allowed = (
        epochs_match
        and enough_rows
        and scope_coverage_complete
        and not earlier_blocking_stages
    )

    evaluated_robust = []
    evaluated_ideal = []
    if candidate_evaluation_allowed:
        assert baseline_validation is not None and baseline_stress is not None
        evaluated_robust = [
            _evaluate_robust(
                theory,
                candidate,
                evidence["validation"],
                evidence["stress"],
                normalized_contract,
                baseline_validation,
                baseline_stress,
            )
            for candidate in synthesis["robustification_candidates"]
        ]
        evaluated_ideal = [
            _evaluate_ideal(
                theory,
                candidate,
                evidence["validation"],
                evidence["stress"],
                normalized_contract,
            )
            for candidate in synthesis["idealization_candidates"]
        ]

    selected: dict[str, Any] | None = None
    if not epochs_match:
        disposition = CompetitionDisposition.INCOMPARABLE_EVALUATOR_EPOCH.value
    elif not enough_rows:
        disposition = CompetitionDisposition.NEEDS_EVIDENCE.value
    elif not scope_coverage_complete:
        disposition = CompetitionDisposition.NEEDS_EVIDENCE.value
    elif earlier_blocking_stages:
        disposition = CompetitionDisposition.NEEDS_EVIDENCE.value
    else:
        viable_robust = [
            item for item in evaluated_robust if item["evaluation"]["viable"]
        ]
        viable_ideal = [
            item for item in evaluated_ideal if item["evaluation"]["viable"]
        ]
        best_robust = max(
            viable_robust,
            key=lambda item: (item["evaluation"]["score"], item["candidate_id"]),
            default=None,
        )
        best_ideal = max(
            viable_ideal,
            key=lambda item: (item["evaluation"]["score"], item["candidate_id"]),
            default=None,
        )
        if best_robust is None and best_ideal is None:
            disposition = CompetitionDisposition.NEEDS_EVIDENCE.value
        elif best_ideal is None:
            disposition = CompetitionDisposition.SELECT_ROBUSTIFICATION.value
            selected = best_robust
        elif best_robust is None:
            disposition = CompetitionDisposition.SELECT_IDEALIZATION.value
            selected = best_ideal
        else:
            difference = (
                float(best_robust["evaluation"]["score"])
                - float(best_ideal["evaluation"]["score"])
            )
            margin = float(normalized_contract["thresholds"]["min_score_margin"])
            if abs(difference) < margin:
                disposition = CompetitionDisposition.NEEDS_EVIDENCE.value
            elif difference > 0:
                disposition = CompetitionDisposition.SELECT_ROBUSTIFICATION.value
                selected = best_robust
            else:
                disposition = CompetitionDisposition.SELECT_IDEALIZATION.value
                selected = best_ideal

    viable_quotients = sum(
        bool(item["evaluation"]["viable"]) for item in evaluated_ideal
    )
    viable_robustifications = sum(
        bool(item["evaluation"]["viable"]) for item in evaluated_robust
    )
    residual_fraction = base_diagnostics["scope"].get("violating_fraction")
    if not epochs_match:
        operation_disposition = "NOT_EVALUATED_INCOMPARABLE_EVALUATOR_EPOCH"
    elif not enough_rows:
        operation_disposition = "NOT_EVALUATED_INSUFFICIENT_EVIDENCE"
    elif not scope_coverage_complete:
        operation_disposition = "NOT_EVALUATED_INCOMPLETE_SCOPE_COVERAGE"
    elif earlier_blocking_stages:
        operation_disposition = "BLOCKED_BY_EARLIER_DIAGNOSIS"
    else:
        operation_disposition = None
    diagnostic_trace = [
        {"stage": "reestimate", **base_diagnostics["reestimate"]},
        {"stage": "noise", **base_diagnostics["noise"]},
        {"stage": "scope", **base_diagnostics["scope"]},
        {"stage": "mixture", **base_diagnostics["mixture"]},
        {
            "stage": "quotient",
            "candidate_count": len(synthesis["idealization_candidates"]),
            "viable_count": viable_quotients,
            "disposition": operation_disposition
            or ("VIABLE_OPERATION" if viable_quotients else "NOT_SELECTED"),
            "blocking_stages": list(earlier_blocking_stages)
            if earlier_blocking_stages
            else [],
        },
        {
            "stage": "robustify",
            "candidate_count": len(synthesis["robustification_candidates"]),
            "viable_count": viable_robustifications,
            "disposition": operation_disposition
            or (
                "VIABLE_OPERATION"
                if viable_robustifications
                else "NOT_SELECTED"
            ),
            "blocking_stages": list(earlier_blocking_stages)
            if earlier_blocking_stages
            else [],
        },
        {
            "stage": "new_probe",
            "residual_fraction": residual_fraction,
            "disposition": "REQUIRED"
            if disposition == CompetitionDisposition.NEEDS_EVIDENCE.value
            and residual_fraction is not None
            and residual_fraction
            >= float(normalized_contract["thresholds"]["new_probe_residual_fraction"])
            else (
                "REQUIRED_EVALUATOR_REBIND"
                if disposition
                == CompetitionDisposition.INCOMPARABLE_EVALUATOR_EPOCH.value
                else "NOT_REQUIRED"
            ),
        },
        {
            "stage": "language_last",
            "residual_fraction": residual_fraction,
            "disposition": "NOT_IMPLEMENTED_LAST_RESORT"
            if disposition == CompetitionDisposition.NEEDS_EVIDENCE.value
            and residual_fraction is not None
            and residual_fraction
            >= float(normalized_contract["thresholds"]["language_residual_fraction"])
            else (
                "DEFERRED_INCOMPARABLE_EVALUATOR"
                if disposition
                == CompetitionDisposition.INCOMPARABLE_EVALUATOR_EPOCH.value
                else "DEFERRED_LAST_RESORT"
            ),
        },
    ]

    evaluator_binding = {
        "theory_evaluator_epoch": theory["public"]["evaluator_epoch"],
        "theory_fixed_anchor": theory["public"]["fixed_anchor"],
        "split_same_epoch_and_anchor": binding_by_split,
        "discovery_same_epoch_and_anchor": discovery_compatible,
        "evidence_epochs": sorted(
            {row["evaluator_epoch"] for rows in evidence.values() for row in rows}
        ),
        "evidence_fixed_anchors": sorted(
            {row["fixed_anchor"] for rows in evidence.values() for row in rows}
        ),
        "same_epoch_and_anchor": epochs_match,
        "cross_epoch_records_pooled": False,
        "selective_erasure_applied": False,
    }
    next_probe = None
    if disposition == CompetitionDisposition.INCOMPARABLE_EVALUATOR_EPOCH.value:
        next_probe = {
            "operation_kind": TheoryOperationKind.PROBE.value,
            "reason": "REPLACE_INCOMPARABLE_EVIDENCE",
            "required_split": "paired_discovery_validation_and_stress",
            "required_evaluator_epoch": theory["public"]["evaluator_epoch"],
            "required_fixed_anchor": theory["public"]["fixed_anchor"],
            "must_preserve_fixed_anchor": True,
            "must_discriminate_candidate_ids": [],
        }
    elif disposition == CompetitionDisposition.NEEDS_EVIDENCE.value:
        if not enough_rows:
            reason = "MINIMUM_SPLIT_ROWS_NOT_SATISFIED"
            required_split = "missing_discovery_validation_or_stress_rows"
        elif not scope_coverage_complete:
            reason = "REGISTERED_SCOPE_COVERAGE_INCOMPLETE"
            required_split = "missing_registered_scope_rows"
        elif earlier_blocking_stages:
            reason = "EARLIER_DIAGNOSTIC_NOT_EXCLUDED"
            required_split = "diagnostic_resolution_evidence"
        else:
            reason = "NO_OPERATION_WON_FROZEN_COMPETITION"
            required_split = "new_validation_and_stress_epoch"
        next_probe = {
            "operation_kind": TheoryOperationKind.PROBE.value,
            "reason": reason,
            "required_split": required_split,
            "blocking_stages": list(earlier_blocking_stages),
            "missing_scope_coverage": missing_scope_coverage,
            "must_preserve_fixed_anchor": True,
            "must_discriminate_candidate_ids": sorted(
                [item["candidate_id"] for item in evaluated_robust + evaluated_ideal]
            ),
        }

    events = []
    events.append(
        _audit_event(
            0,
            "INPUT_VALIDATED",
            GENESIS_DIGEST,
            {
                "case_id": normalized_input["case_id"],
                "theory_state_digest": _digest(theory["public"]),
                "evidence_digests": {
                    split: _digest(rows) for split, rows in evidence.items()
                },
            },
        )
    )
    events.append(
        _audit_event(
            1,
            "CANDIDATES_SYNTHESIZED_FROM_DISCOVERY"
            if discovery_compatible
            else "CANDIDATE_SYNTHESIS_BLOCKED_CROSS_EPOCH",
            events[-1]["event_digest"],
            {
                "candidate_commitment_digest": synthesis[
                    "candidate_commitment_digest"
                ],
                "robustification_count": len(
                    synthesis["robustification_candidates"]
                ),
                "idealization_count": len(synthesis["idealization_candidates"]),
            },
        )
    )
    events.append(
        _audit_event(
            2,
            "HELD_OUT_EVIDENCE_EVALUATED"
            if candidate_evaluation_allowed
            else "HELD_OUT_OPERATION_EVALUATION_NOT_PERFORMED",
            events[-1]["event_digest"],
            {
                "same_epoch_and_anchor": epochs_match,
                "minimum_rows_satisfied": enough_rows,
                "registered_scope_coverage_complete": scope_coverage_complete,
                "missing_scope_coverage": missing_scope_coverage,
                "earlier_blocking_stages": list(earlier_blocking_stages),
                "candidate_evaluation_performed": candidate_evaluation_allowed,
                "validation_digest": _digest(evidence["validation"]),
                "stress_digest": _digest(evidence["stress"]),
            },
        )
    )
    events.append(
        _audit_event(
            3,
            "COMPETITION_DECIDED",
            events[-1]["event_digest"],
            {
                "disposition": disposition,
                "selected_candidate_id": selected["candidate_id"] if selected else None,
                "promotion_status": normalized_contract["selection"]["promotion_status"],
            },
        )
    )

    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract_id": normalized_contract["contract_id"],
        "contract_digest": _digest(normalized_contract),
        "case_id": normalized_input["case_id"],
        "theory_state": theory["public"],
        "theory_state_digest": _digest(theory["public"]),
        "evidence_digests": {split: _digest(rows) for split, rows in evidence.items()},
        "evidence_coverage": {
            "scope_ids_by_split": scope_coverage_by_split,
            "missing_scope_ids_by_split": missing_scope_coverage,
            "all_registered_scopes_in_every_split": scope_coverage_complete,
        },
        "candidate_commitments": {
            "discovery_digest": synthesis["discovery_digest"],
            "candidate_commitment_digest": synthesis[
                "candidate_commitment_digest"
            ],
            "robustification_count": len(synthesis["robustification_candidates"]),
            "idealization_count": len(synthesis["idealization_candidates"]),
        },
        "operation_registry": list(OPERATION_KINDS),
        "diagnostic_trace": diagnostic_trace,
        "baseline_metrics": {
            "validation": baseline_validation,
            "stress": baseline_stress,
        },
        "robustification_candidates": evaluated_robust
        if candidate_evaluation_allowed
        else synthesis["robustification_candidates"],
        "idealization_candidates": evaluated_ideal
        if candidate_evaluation_allowed
        else synthesis["idealization_candidates"],
        "disposition": disposition,
        "selected_candidate": selected,
        "next_probe_spec": next_probe,
        "evaluator_binding": evaluator_binding,
        "promotion_status": normalized_contract["selection"]["promotion_status"],
        "nonclaims": list(normalized_contract["nonclaims"]),
        "input_artifacts": artifacts,
        "audit_events": events,
        "audit_head": events[-1]["event_digest"],
    }
    report = {**body, "report_digest": _digest(body)}
    return CompetitionResult(report=report)


def verify_theory_operation_competition(
    input_payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    expected_contract_digest: str,
    expected_report_digest: str,
    expected_input_artifacts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if (
        type(expected_contract_digest) is not str
        or not expected_contract_digest.startswith("sha256:")
        or len(expected_contract_digest) != 71
    ):
        raise CompetitionValidationError(
            "expected_contract_digest must be sha256:<64hex>"
        )
    try:
        int(expected_contract_digest[7:], 16)
    except ValueError as exc:
        raise CompetitionValidationError(
            "expected_contract_digest is not hexadecimal"
        ) from exc
    normalized_contract = _validate_contract(contract)
    if _digest(normalized_contract) != expected_contract_digest:
        raise CompetitionValidationError(
            "contract digest differs from independent expectation"
        )
    if (
        type(expected_report_digest) is not str
        or not expected_report_digest.startswith("sha256:")
        or len(expected_report_digest) != 71
    ):
        raise CompetitionValidationError("expected_report_digest must be sha256:<64hex>")
    try:
        int(expected_report_digest[7:], 16)
    except ValueError as exc:
        raise CompetitionValidationError("expected_report_digest is not hexadecimal") from exc
    supplied = _mapping(report, "report")
    if expected_input_artifacts is not None and not isinstance(
        expected_input_artifacts, Mapping
    ):
        raise CompetitionValidationError(
            "expected_input_artifacts must be an object or null"
        )
    artifacts = (
        _json_copy(expected_input_artifacts)
        if expected_input_artifacts is not None
        else None
    )
    if supplied.get("input_artifacts") != artifacts:
        raise CompetitionValidationError(
            "report input_artifacts differ from independent expectation"
        )
    fresh = run_theory_operation_competition(
        input_payload,
        contract,
        input_artifacts=artifacts,
    ).to_dict()
    if fresh["report_digest"] != expected_report_digest:
        raise CompetitionValidationError("replayed report digest differs from expected")
    if canonical_json_bytes(supplied) != canonical_json_bytes(fresh):
        raise CompetitionValidationError("supplied report differs from exact replay")
    return {
        "status": "VERIFIED_" + fresh["disposition"],
        "disposition": fresh["disposition"],
        "report_digest": fresh["report_digest"],
        "candidate_commitment_digest": fresh["candidate_commitments"][
            "candidate_commitment_digest"
        ],
        "selected_candidate_id": fresh["selected_candidate"]["candidate_id"]
        if fresh["selected_candidate"] is not None
        else None,
        "promotion_status": fresh["promotion_status"],
    }
