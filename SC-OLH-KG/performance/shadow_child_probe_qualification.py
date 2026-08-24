"""Qualify one verified shadow child on a new local evaluator epoch.

The core is pure and additive.  It exact-replays the upstream competition and
transition, excludes every source observation from child scoring, evaluates a
fixed finite probe registry on supplied holdout/stress rows, and emits a
digest-bound qualification receipt.  It does not acquire evidence, mutate the
child, adopt a theory, execute a benchmark, or write state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from performance.shadow_theory_transition import (
    ShadowTransitionValidationError,
    verify_shadow_theory_transition,
)
from performance.theory_operation_competition import (
    CompetitionValidationError,
    canonical_json_bytes,
)


CONTRACT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-child-probe-qualification-contract/1"
)
CONTRACT_ID = "shadow_child_probe_qualification_v1"
INPUT_SCHEMA_VERSION = "sc-olh-kg.shadow-child-probe-qualification-input/1"
EVALUATOR_DEFINITION_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-child-evaluator-definition/1"
)
SOURCE_TRANSITION_CONTRACT_ID = "shadow_theory_transition_v1"
SOURCE_TRANSITION_CONTRACT_DIGEST = (
    "sha256:35c5f127aa8c400d8d723eccaffaf05fa9667771edcdfcd6e648eac7854eeaab"
)
SOURCE_TRANSITION_REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-theory-transition-report/1"
)
CHILD_THEORY_SCHEMA_VERSION = "sc-olh-kg.shadow-theory-state/1"
REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-child-probe-qualification-report/1"
)
SCHEMA_VERSION = INPUT_SCHEMA_VERSION
GENESIS_DIGEST = "sha256:" + "0" * 64
ADOPTION_STATUS = "NOT_ADOPTED_SHADOW_ONLY"

EVIDENCE_POLICY = {
    "min_holdout_rows": 4,
    "min_stress_rows": 4,
    "require_complete_parent_context_scope_pairs_per_split": True,
    "require_disjoint_new_ids": True,
    "require_disjoint_from_source_ids": True,
    "require_registered_parent_contexts": True,
    "require_registered_scopes": True,
}

EVALUATOR_EPOCH_POLICY = {
    "derive_from_child_and_transition": True,
    "require_fresh_from_source": True,
    "inherit_fixed_anchor": True,
    "forbid_old_new_pooling": True,
    "logical_selective_erasure": True,
    "physical_erasure": False,
}

PROBE_REGISTRY = {
    "ROBUST_INTERVAL_EXPANSION": [
        {
            "probe_id": "absolute_error_point_prediction",
            "semantics": (
                "absolute observed-minus-parent and observed-minus-child-center error"
            ),
            "aggregation": "mean_by_fresh_split",
            "role": "nominal_prediction_control",
        },
        {
            "probe_id": "interval_containment",
            "semantics": (
                "fresh scalar observation lies within the child center plus or "
                "minus its selected radius"
            ),
            "aggregation": "coverage_by_fresh_split",
            "role": "robust_interval_qualification",
        },
        {
            "probe_id": "tail_interval_containment",
            "semantics": (
                "interval containment on stress rows with largest parent absolute errors"
            ),
            "aggregation": "parent_error_ranked_stress_tail_coverage",
            "role": "fresh_tail_qualification",
        },
        {
            "probe_id": "normalized_interval_radius",
            "semantics": (
                "mean fresh-split interval radius normalized by twice the frozen "
                "parent prediction scale"
            ),
            "aggregation": "mean_holdout_and_stress_radius_over_two_scale",
            "role": "anti_vacuity_radius_budget",
        },
    ],
    "QUOTIENT_IDEALIZATION": [
        {
            "probe_id": "absolute_error_point_prediction",
            "semantics": (
                "absolute observed-minus-parent and observed-minus-quotient-child error"
            ),
            "aggregation": "mean_by_fresh_split",
            "role": "nominal_prediction_control",
        },
        {
            "probe_id": "parent_child_point_divergence",
            "semantics": (
                "absolute parent-point minus quotient-child-point prediction "
                "divergence on complete parent fibers"
            ),
            "aggregation": "maximum_across_fresh_holdout_and_stress",
            "role": "finite_prediction_proxy_qualification",
        },
    ],
}

THRESHOLDS = {
    "numeric_epsilon": 1e-12,
    "coverage_tolerance": 0.05,
    "tail_fraction": 0.25,
    "min_holdout_coverage": 0.75,
    "min_stress_tail_coverage": 0.75,
    "min_safety_rate": 0.75,
    "max_nominal_mae_increase": 0.05,
    "max_probe_divergence": 0.20,
    "max_normalized_radius": 1.0,
}

SELECTION = {
    "qualified_status": "QUALIFIED_NEW_EVALUATOR_EPOCH",
    "needs_evidence_status": "NEEDS_NEW_EVALUATOR_EVIDENCE",
    "incomparable_status": "INCOMPARABLE_NEW_EVALUATOR_EPOCH",
    "failed_status": "FAILED_OPERATIONAL_PROBE_QUALIFICATION",
    "adoption_status": ADOPTION_STATUS,
}

MANDATORY_NONCLAIMS = (
    "shadow_only",
    "no_automatic_adoption",
    "no_paper_promotion",
    "no_operations_research_baseline_or_claim_change",
    "no_parent_child_or_ambient_state_write",
    "explicit_cli_out_is_only_optional_write",
    "no_run_one",
    "no_benchmark_execution",
    "no_scheduler_access",
    "no_network_access",
    "only_materialized_expand_and_quotient_children",
    "no_restrict_operation",
    "no_language_or_predicate_invention",
    "fixed_operational_probe_registry_only",
    "no_autonomous_probe_invention",
    "no_child_probe_registry_mutation",
    "supplied_new_evidence_evaluation_only",
    "no_probe_acquisition_or_environment_execution",
    "source_evidence_excluded_from_child_scoring",
    "no_source_evidence_score_reuse",
    "no_cross_epoch_pooling",
    "logical_selective_erasure_only",
    "no_physical_record_deletion",
    "original_shadow_child_remains_unmodified",
    "qualification_receipt_is_not_adoption",
    "finite_scalar_tabular_scope_only",
    "robust_interval_containment_is_not_domain_safety",
    "tail_interval_containment_is_not_cvar_or_worst_case_safety",
    "idealization_probe_preservation_is_task_and_finite_evidence_scoped",
    "quotient_recovery_still_requires_frozen_parent_snapshot",
    "new_evaluator_epoch_is_content_derived_not_externally_attested",
    "fixed_anchor_equality_is_not_external_attestation",
    "input_artifacts_require_independent_expected_values",
    "report_digest_is_not_a_signature",
    "no_data_provenance_beyond_bound_artifacts",
    "no_scientific_validity_or_generalization_claim",
    "no_h_t_to_h_t_plus_1_acceptance",
)

REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "contract_digest",
        "qualification_input_digest",
        "source_transition",
        "qualification_id",
        "child_theory_state_digest",
        "transition_kind",
        "evaluator_definition",
        "evaluator_binding",
        "evidence_binding",
        "selective_erasure_receipt",
        "probe_results",
        "disposition",
        "qualification_binding",
        "adoption_status",
        "nonclaims",
        "input_artifacts",
        "audit_events",
        "audit_head",
        "report_digest",
    }
)


class ShadowChildProbeQualificationValidationError(ValueError):
    """Raised when a qualification contract, source, input, or report is invalid."""


class ShadowChildProbeQualificationDisposition(str, Enum):
    QUALIFIED_NEW_EVALUATOR_EPOCH = "QUALIFIED_NEW_EVALUATOR_EPOCH"
    NEEDS_NEW_EVALUATOR_EVIDENCE = "NEEDS_NEW_EVALUATOR_EVIDENCE"
    INCOMPARABLE_NEW_EVALUATOR_EPOCH = "INCOMPARABLE_NEW_EVALUATOR_EPOCH"
    FAILED_OPERATIONAL_PROBE_QUALIFICATION = (
        "FAILED_OPERATIONAL_PROBE_QUALIFICATION"
    )


@dataclass(frozen=True)
class ShadowChildProbeQualificationResult:
    report: Mapping[str, Any]

    @property
    def disposition(self) -> str:
        return str(self.report["disposition"])

    @property
    def report_digest(self) -> str:
        return str(self.report["report_digest"])

    @property
    def qualified(self) -> bool:
        return self.disposition == "QUALIFIED_NEW_EVALUATOR_EPOCH"

    def to_dict(self) -> dict[str, Any]:
        return _copy(self.report)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShadowChildProbeQualificationValidationError(
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
        raise ShadowChildProbeQualificationValidationError(
            f"{label} keys differ; missing={missing}, extra={extra}"
        )


def _string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ShadowChildProbeQualificationValidationError(
            f"{label} must be a non-empty string"
        )
    return value


def _finite_number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ShadowChildProbeQualificationValidationError(
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
        raise ShadowChildProbeQualificationValidationError(
            f"value is not detached finite canonical JSON: {exc}"
        ) from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_digest(value: Any, label: str) -> str:
    digest = _string(value, label)
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise ShadowChildProbeQualificationValidationError(
            f"{label} must be sha256:<64hex>"
        )
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise ShadowChildProbeQualificationValidationError(
            f"{label} is not hexadecimal"
        ) from exc
    return digest


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ShadowChildProbeQualificationValidationError(
            f"{label} differs from frozen qualification V1"
        )


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _validate_contract(contract_value: Any) -> dict[str, Any]:
    contract = _mapping(contract_value, "qualification_contract")
    _exact_keys(
        contract,
        {
            "schema_version",
            "contract_id",
            "input_schema_version",
            "source_transition_contract_id",
            "source_transition_contract_digest",
            "source_transition_report_schema_version",
            "child_theory_schema_version",
            "report_schema_version",
            "evidence_policy",
            "evaluator_epoch_policy",
            "probe_registry",
            "thresholds",
            "selection",
            "nonclaims",
        },
        "qualification_contract",
    )
    frozen_values = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "source_transition_contract_id": SOURCE_TRANSITION_CONTRACT_ID,
        "source_transition_contract_digest": SOURCE_TRANSITION_CONTRACT_DIGEST,
        "source_transition_report_schema_version": (
            SOURCE_TRANSITION_REPORT_SCHEMA_VERSION
        ),
        "child_theory_schema_version": CHILD_THEORY_SCHEMA_VERSION,
        "report_schema_version": REPORT_SCHEMA_VERSION,
    }
    for key, expected in frozen_values.items():
        _require_equal(contract[key], expected, key)
    _require_equal(
        dict(_mapping(contract["evidence_policy"], "evidence_policy")),
        EVIDENCE_POLICY,
        "evidence_policy",
    )
    _require_equal(
        dict(
            _mapping(
                contract["evaluator_epoch_policy"], "evaluator_epoch_policy"
            )
        ),
        EVALUATOR_EPOCH_POLICY,
        "evaluator_epoch_policy",
    )
    _require_equal(
        _copy(_mapping(contract["probe_registry"], "probe_registry")),
        PROBE_REGISTRY,
        "probe_registry",
    )
    _require_equal(
        dict(_mapping(contract["thresholds"], "thresholds")),
        THRESHOLDS,
        "thresholds",
    )
    _require_equal(
        dict(_mapping(contract["selection"], "selection")),
        SELECTION,
        "selection",
    )
    nonclaims = contract["nonclaims"]
    if not isinstance(nonclaims, list) or tuple(nonclaims) != MANDATORY_NONCLAIMS:
        raise ShadowChildProbeQualificationValidationError(
            "nonclaims differ from the exact mandatory V1 list"
        )
    return _copy(contract)


def validate_shadow_child_probe_qualification_contract(
    contract_value: Any,
) -> dict[str, Any]:
    """Validate and detach the frozen qualification contract."""

    return _validate_contract(contract_value)


def _evaluator_definition(
    *,
    transition_contract_digest: str,
    transition_report_digest: str,
    child_theory_state_digest: str,
    transition_kind: str,
    fixed_anchor: str,
    qualification_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str]:
    normalized_contract = _validate_contract(qualification_contract)
    source_contract_digest = _require_digest(
        transition_contract_digest, "transition_contract_digest"
    )
    if source_contract_digest != SOURCE_TRANSITION_CONTRACT_DIGEST:
        raise ShadowChildProbeQualificationValidationError(
            "transition contract digest is not the pinned V1 source"
        )
    source_report_digest = _require_digest(
        transition_report_digest, "transition_report_digest"
    )
    child_digest = _require_digest(
        child_theory_state_digest, "child_theory_state_digest"
    )
    kind = _string(transition_kind, "transition_kind")
    if kind not in PROBE_REGISTRY:
        raise ShadowChildProbeQualificationValidationError(
            "transition_kind has no fixed operational probe registry"
        )
    anchor = _string(fixed_anchor, "fixed_anchor")
    definition = {
        "schema_version": EVALUATOR_DEFINITION_SCHEMA_VERSION,
        "qualification_contract_digest": _digest(normalized_contract),
        "source_transition_contract_digest": source_contract_digest,
        "source_transition_report_digest": source_report_digest,
        "child_theory_state_digest": child_digest,
        "transition_kind": kind,
        "fixed_anchor": anchor,
        "probe_registry": _copy(PROBE_REGISTRY[kind]),
    }
    definition_digest = _digest(definition)
    epoch = "shadow-evaluator-epoch:" + definition_digest[7:]
    return definition, definition_digest, epoch


def derive_shadow_child_evaluator_epoch(
    *,
    transition_contract_digest: str,
    transition_report_digest: str,
    child_theory_state_digest: str,
    transition_kind: str,
    fixed_anchor: str,
    qualification_contract: Mapping[str, Any],
) -> str:
    """Derive the exact content-addressed child evaluator epoch identifier."""

    return _evaluator_definition(
        transition_contract_digest=transition_contract_digest,
        transition_report_digest=transition_report_digest,
        child_theory_state_digest=child_theory_state_digest,
        transition_kind=transition_kind,
        fixed_anchor=fixed_anchor,
        qualification_contract=qualification_contract,
    )[2]


def _verify_source_chain(
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
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_transition_contract = _require_digest(
        expected_transition_contract_digest,
        "expected_transition_contract_digest",
    )
    if expected_transition_contract != SOURCE_TRANSITION_CONTRACT_DIGEST:
        raise ShadowChildProbeQualificationValidationError(
            "expected transition contract digest is not the pinned V1 source"
        )
    expected_transition_report = _require_digest(
        expected_transition_report_digest,
        "expected_transition_report_digest",
    )
    try:
        receipt = verify_shadow_theory_transition(
            competition_input,
            competition_contract,
            competition_report,
            transition_contract,
            transition_report,
            expected_competition_contract_digest=(
                expected_competition_contract_digest
            ),
            expected_competition_report_digest=expected_competition_report_digest,
            expected_competition_input_artifacts=(
                expected_competition_input_artifacts
            ),
            expected_transition_contract_digest=expected_transition_contract,
            expected_transition_report_digest=expected_transition_report,
            expected_transition_input_artifacts=expected_transition_input_artifacts,
        )
    except (
        ShadowTransitionValidationError,
        CompetitionValidationError,
        KeyError,
        TypeError,
    ) as exc:
        raise ShadowChildProbeQualificationValidationError(
            f"source transition verification failed: {exc}"
        ) from exc
    source_report = _copy(_mapping(transition_report, "transition_report"))
    disposition = source_report.get("disposition")
    if disposition not in {
        "MATERIALIZED_SHADOW_ROBUSTIFICATION",
        "MATERIALIZED_SHADOW_IDEALIZATION",
    }:
        raise ShadowChildProbeQualificationValidationError(
            "qualification requires a materialized robust or ideal shadow child"
        )
    if source_report.get("schema_version") != SOURCE_TRANSITION_REPORT_SCHEMA_VERSION:
        raise ShadowChildProbeQualificationValidationError(
            "source transition report schema is not supported"
        )
    if source_report.get("contract_id") != SOURCE_TRANSITION_CONTRACT_ID:
        raise ShadowChildProbeQualificationValidationError(
            "source transition contract_id is not supported"
        )
    if source_report.get("contract_digest") != expected_transition_contract:
        raise ShadowChildProbeQualificationValidationError(
            "source transition contract digest differs from expectation"
        )
    if source_report.get("report_digest") != expected_transition_report:
        raise ShadowChildProbeQualificationValidationError(
            "source transition report digest differs from expectation"
        )
    if source_report.get("adoption_status") != ADOPTION_STATUS:
        raise ShadowChildProbeQualificationValidationError(
            "source transition crossed the shadow-only adoption boundary"
        )
    child = _copy(
        _mapping(source_report.get("child_theory_state"), "child_theory_state")
    )
    child_digest = _require_digest(
        source_report.get("child_theory_state_digest"),
        "child_theory_state_digest",
    )
    if _digest(child) != child_digest:
        raise ShadowChildProbeQualificationValidationError(
            "source child digest does not match its canonical state"
        )
    parent = _copy(
        _mapping(source_report.get("parent_theory_state"), "parent_theory_state")
    )
    return source_report, _copy(receipt), parent, child


def _json_scalar(value: Any, label: str) -> str | int | float | bool:
    if value is None or type(value) not in (str, int, float, bool):
        raise ShadowChildProbeQualificationValidationError(
            f"{label} must be a non-null finite JSON scalar"
        )
    if type(value) is float and not math.isfinite(value):
        raise ShadowChildProbeQualificationValidationError(
            f"{label} must be finite"
        )
    return value


def _normalize_context(
    value: Any, feature_ids: Sequence[str], registered: set[bytes], label: str
) -> dict[str, Any]:
    context = _mapping(value, label)
    _exact_keys(context, set(feature_ids), label)
    normalized = {
        feature_id: _json_scalar(context[feature_id], f"{label}.{feature_id}")
        for feature_id in feature_ids
    }
    if canonical_json_bytes(normalized) not in registered:
        raise ShadowChildProbeQualificationValidationError(
            f"{label} is not a registered parent context"
        )
    return normalized


def _normalize_rows(
    rows_value: Any,
    split: str,
    feature_ids: Sequence[str],
    registered_contexts: set[bytes],
    registered_scopes: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(rows_value, list):
        raise ShadowChildProbeQualificationValidationError(
            f"evidence.{split} must be a list"
        )
    rows = []
    for index, raw in enumerate(rows_value):
        row = _mapping(raw, f"evidence.{split}[{index}]")
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
            f"evidence.{split}[{index}]",
        )
        scope_id = _string(
            row["scope_id"], f"evidence.{split}[{index}].scope_id"
        )
        if scope_id not in registered_scopes:
            raise ShadowChildProbeQualificationValidationError(
                f"evidence.{split}[{index}] scope is unregistered"
            )
        rows.append(
            {
                "observation_id": _string(
                    row["observation_id"],
                    f"evidence.{split}[{index}].observation_id",
                ),
                "evaluator_epoch": _string(
                    row["evaluator_epoch"],
                    f"evidence.{split}[{index}].evaluator_epoch",
                ),
                "fixed_anchor": _string(
                    row["fixed_anchor"],
                    f"evidence.{split}[{index}].fixed_anchor",
                ),
                "scope_id": scope_id,
                "context": _normalize_context(
                    row["context"],
                    feature_ids,
                    registered_contexts,
                    f"evidence.{split}[{index}].context",
                ),
                "observed_value": _finite_number(
                    row["observed_value"],
                    f"evidence.{split}[{index}].observed_value",
                ),
            }
        )
    rows.sort(key=lambda item: item["observation_id"])
    ids = [item["observation_id"] for item in rows]
    if len(ids) != len(set(ids)):
        raise ShadowChildProbeQualificationValidationError(
            f"evidence.{split} observation IDs are duplicated"
        )
    return rows


def _source_observation_ids(competition_input: Mapping[str, Any]) -> list[str]:
    evidence = _mapping(competition_input.get("evidence"), "source evidence")
    _exact_keys(evidence, {"discovery", "validation", "stress"}, "source evidence")
    ids = []
    for split in ("discovery", "validation", "stress"):
        rows = evidence[split]
        if not isinstance(rows, list):
            raise ShadowChildProbeQualificationValidationError(
                f"source evidence.{split} must be a list"
            )
        for index, row_value in enumerate(rows):
            row = _mapping(row_value, f"source evidence.{split}[{index}]")
            ids.append(
                _string(
                    row.get("observation_id"),
                    f"source evidence.{split}[{index}].observation_id",
                )
            )
    if len(ids) != len(set(ids)):
        raise ShadowChildProbeQualificationValidationError(
            "source observation IDs are not unique"
        )
    return sorted(ids)


def _normalize_input(
    input_value: Any,
    *,
    source_report: Mapping[str, Any],
    competition_input: Mapping[str, Any],
    competition_report: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(input_value, "qualification_input")
    _exact_keys(
        value,
        {
            "schema_version",
            "qualification_id",
            "source_transition",
            "evaluator",
            "source_evidence_exclusion",
            "evidence",
        },
        "qualification_input",
    )
    if value["schema_version"] != INPUT_SCHEMA_VERSION:
        raise ShadowChildProbeQualificationValidationError(
            "unexpected qualification input schema_version"
        )
    source_ref = _mapping(value["source_transition"], "source_transition")
    _exact_keys(
        source_ref,
        {
            "contract_digest",
            "report_digest",
            "child_theory_state_digest",
        },
        "source_transition",
    )
    expected_source_ref = {
        "contract_digest": source_report["contract_digest"],
        "report_digest": source_report["report_digest"],
        "child_theory_state_digest": source_report["child_theory_state_digest"],
    }
    if dict(source_ref) != expected_source_ref:
        raise ShadowChildProbeQualificationValidationError(
            "qualification source_transition does not match the verified source"
        )

    evaluator = _mapping(value["evaluator"], "evaluator")
    _exact_keys(evaluator, {"evaluator_epoch", "fixed_anchor"}, "evaluator")
    normalized_evaluator = {
        "evaluator_epoch": _string(
            evaluator["evaluator_epoch"], "evaluator.evaluator_epoch"
        ),
        "fixed_anchor": _string(evaluator["fixed_anchor"], "evaluator.fixed_anchor"),
    }

    exclusion = _mapping(
        value["source_evidence_exclusion"], "source_evidence_exclusion"
    )
    _exact_keys(
        exclusion,
        {"policy", "source_evidence_digests"},
        "source_evidence_exclusion",
    )
    if exclusion["policy"] != "EXCLUDE_ALL_SOURCE_COMPETITION_RECORDS":
        raise ShadowChildProbeQualificationValidationError(
            "source evidence exclusion policy is not frozen V1"
        )
    source_evidence_digests = _mapping(
        exclusion["source_evidence_digests"],
        "source_evidence_exclusion.source_evidence_digests",
    )
    _exact_keys(
        source_evidence_digests,
        {"discovery", "validation", "stress"},
        "source_evidence_exclusion.source_evidence_digests",
    )
    expected_evidence_digests = _copy(
        _mapping(competition_report.get("evidence_digests"), "source evidence_digests")
    )
    if dict(source_evidence_digests) != expected_evidence_digests:
        raise ShadowChildProbeQualificationValidationError(
            "source evidence exclusion digests do not match the verified competition"
        )

    parent_object = _mapping(parent.get("object_space"), "parent object_space")
    feature_ids = parent_object.get("feature_ids")
    contexts = parent_object.get("contexts")
    scopes = parent.get("scope_ids")
    if (
        not isinstance(feature_ids, list)
        or not feature_ids
        or not isinstance(contexts, list)
        or not contexts
        or not isinstance(scopes, list)
        or not scopes
    ):
        raise ShadowChildProbeQualificationValidationError(
            "verified parent lacks a finite feature/context/scope registry"
        )
    feature_tuple = tuple(_string(item, "parent feature_id") for item in feature_ids)
    registered_contexts = {canonical_json_bytes(item) for item in contexts}
    registered_scopes = {_string(item, "parent scope_id") for item in scopes}

    evidence = _mapping(value["evidence"], "evidence")
    _exact_keys(evidence, {"holdout", "stress"}, "evidence")
    normalized_evidence = {
        split: _normalize_rows(
            evidence[split],
            split,
            feature_tuple,
            registered_contexts,
            registered_scopes,
        )
        for split in ("holdout", "stress")
    }
    new_ids = [
        row["observation_id"]
        for split in normalized_evidence.values()
        for row in split
    ]
    if len(new_ids) != len(set(new_ids)):
        raise ShadowChildProbeQualificationValidationError(
            "new observation IDs must be unique across holdout and stress"
        )
    source_ids = _source_observation_ids(competition_input)
    overlap = sorted(set(new_ids) & set(source_ids))
    if overlap:
        raise ShadowChildProbeQualificationValidationError(
            f"new observation IDs reuse source evidence IDs: {overlap}"
        )
    public = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "qualification_id": _string(value["qualification_id"], "qualification_id"),
        "source_transition": expected_source_ref,
        "evaluator": normalized_evaluator,
        "source_evidence_exclusion": {
            "policy": "EXCLUDE_ALL_SOURCE_COMPETITION_RECORDS",
            "source_evidence_digests": expected_evidence_digests,
        },
        "evidence": normalized_evidence,
    }
    return {
        "public": public,
        "evidence": normalized_evidence,
        "source_ids": source_ids,
        "new_ids": sorted(new_ids),
        "parent_contexts": _copy(contexts),
        "parent_scopes": sorted(registered_scopes),
    }


def _pair(scope_id: str, context: Mapping[str, Any]) -> dict[str, Any]:
    return {"scope_id": scope_id, "context": _copy(context)}


def _sorted_pairs(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique = {canonical_json_bytes(item): _copy(item) for item in pairs}
    return [unique[key] for key in sorted(unique)]


def _evidence_binding(
    normalized_input: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    evidence = normalized_input["evidence"]
    required_pairs = _sorted_pairs(
        [
            _pair(scope, context)
            for scope in normalized_input["parent_scopes"]
            for context in normalized_input["parent_contexts"]
        ]
    )
    required_keys = {canonical_json_bytes(item) for item in required_pairs}
    covered_by_split = {}
    missing_by_split = {}
    complete_by_split = {}
    for split in ("holdout", "stress"):
        covered = _sorted_pairs(
            [_pair(row["scope_id"], row["context"]) for row in evidence[split]]
        )
        covered_keys = {canonical_json_bytes(item) for item in covered}
        missing = [
            item
            for item in required_pairs
            if canonical_json_bytes(item) not in covered_keys
        ]
        covered_by_split[split] = covered
        missing_by_split[split] = missing
        complete_by_split[split] = not missing and covered_keys == required_keys
    row_counts = {split: len(evidence[split]) for split in ("holdout", "stress")}
    minimum_rows = {
        "holdout": int(contract["evidence_policy"]["min_holdout_rows"]),
        "stress": int(contract["evidence_policy"]["min_stress_rows"]),
    }
    minimum_satisfied = {
        split: row_counts[split] >= minimum_rows[split]
        for split in ("holdout", "stress")
    }
    source_ids = normalized_input["source_ids"]
    new_ids = normalized_input["new_ids"]
    source_id_digest = _digest(source_ids)
    new_id_digest = _digest(new_ids)
    new_unique = len(new_ids) == len(set(new_ids))
    disjoint = not (set(new_ids) & set(source_ids))
    sufficient = (
        all(minimum_satisfied.values())
        and all(complete_by_split.values())
        and new_unique
        and disjoint
    )
    return {
        "evidence_digests": {
            split: _digest(evidence[split]) for split in ("holdout", "stress")
        },
        "row_counts": row_counts,
        "minimum_rows": minimum_rows,
        "minimum_rows_satisfied_by_split": minimum_satisfied,
        "required_context_scope_pairs": required_pairs,
        "required_context_scope_pair_count": len(required_pairs),
        "covered_context_scope_pairs_by_split": covered_by_split,
        "missing_context_scope_pairs_by_split": missing_by_split,
        "complete_context_scope_coverage_by_split": complete_by_split,
        "source_observation_id_digest": source_id_digest,
        "source_observation_count": len(source_ids),
        "new_observation_id_digest": new_id_digest,
        "new_observation_count": len(new_ids),
        "new_ids_unique_across_splits": new_unique,
        "new_ids_disjoint_from_source": disjoint,
        "sufficient": sufficient,
    }


def _evaluator_binding(
    normalized_input: Mapping[str, Any],
    parent: Mapping[str, Any],
    definition_digest: str,
    derived_epoch: str,
) -> dict[str, Any]:
    evidence = normalized_input["evidence"]
    evaluator = normalized_input["public"]["evaluator"]
    row_epochs = sorted(
        {row["evaluator_epoch"] for rows in evidence.values() for row in rows}
    )
    row_anchors = sorted(
        {row["fixed_anchor"] for rows in evidence.values() for row in rows}
    )
    source_epoch = _string(parent["evaluator_epoch"], "source_evaluator_epoch")
    inherited_anchor = _string(parent["fixed_anchor"], "inherited_fixed_anchor")
    rows_exact_epoch = all(
        row["evaluator_epoch"] == derived_epoch
        for rows in evidence.values()
        for row in rows
    )
    rows_exact_anchor = all(
        row["fixed_anchor"] == inherited_anchor
        for rows in evidence.values()
        for row in rows
    )
    fresh = derived_epoch != source_epoch
    comparable = all(
        (
            evaluator["evaluator_epoch"] == derived_epoch,
            evaluator["fixed_anchor"] == inherited_anchor,
            rows_exact_epoch,
            rows_exact_anchor,
            fresh,
        )
    )
    return {
        "source_evaluator_epoch": source_epoch,
        "evaluator_definition_digest": definition_digest,
        "derived_child_evaluator_epoch": derived_epoch,
        "declared_evaluator_epoch": evaluator["evaluator_epoch"],
        "inherited_fixed_anchor": inherited_anchor,
        "declared_fixed_anchor": evaluator["fixed_anchor"],
        "row_evaluator_epochs": row_epochs,
        "row_fixed_anchors": row_anchors,
        "all_new_rows_exact_epoch": rows_exact_epoch,
        "all_new_rows_exact_anchor": rows_exact_anchor,
        "fresh_from_source_epoch": fresh,
        "comparable": comparable,
        "old_new_records_pooled": False,
    }


def _selective_erasure_receipt(
    normalized_input: Mapping[str, Any], evidence_binding: Mapping[str, Any]
) -> dict[str, Any]:
    exclusion = normalized_input["public"]["source_evidence_exclusion"]
    return {
        "policy": "EXCLUDE_ALL_SOURCE_COMPETITION_RECORDS",
        "mode": "LOGICAL_EXCLUSION_ONLY",
        "excluded_source_evidence_digests": _copy(
            exclusion["source_evidence_digests"]
        ),
        "excluded_source_observation_id_digest": evidence_binding[
            "source_observation_id_digest"
        ],
        "excluded_source_observation_count": evidence_binding[
            "source_observation_count"
        ],
        "included_new_evidence_digests": _copy(
            evidence_binding["evidence_digests"]
        ),
        "included_new_observation_id_digest": evidence_binding[
            "new_observation_id_digest"
        ],
        "included_new_observation_count": evidence_binding[
            "new_observation_count"
        ],
        "source_evidence_used_only_for_upstream_replay_and_exclusion_binding": True,
        "source_evidence_used_for_child_scoring": False,
        "old_new_records_pooled": False,
        "logical_selective_erasure_applied": True,
        "physical_records_deleted": False,
    }


def _prediction_lookup(
    predictions_value: Any, label: str
) -> dict[bytes, float]:
    if not isinstance(predictions_value, list):
        raise ShadowChildProbeQualificationValidationError(
            f"{label} must be a list"
        )
    lookup = {}
    for index, raw in enumerate(predictions_value):
        item = _mapping(raw, f"{label}[{index}]")
        _exact_keys(item, {"context", "value"}, f"{label}[{index}]")
        context = _mapping(item["context"], f"{label}[{index}].context")
        key = canonical_json_bytes(context)
        if key in lookup:
            raise ShadowChildProbeQualificationValidationError(
                f"{label} has duplicate contexts"
            )
        lookup[key] = _finite_number(item["value"], f"{label}[{index}].value")
    return lookup


def _functional_threshold(parent: Mapping[str, Any]) -> float:
    functionals = parent.get("violation_functionals")
    if not isinstance(functionals, list) or len(functionals) != 1:
        raise ShadowChildProbeQualificationValidationError(
            "parent must carry exactly one violation functional"
        )
    functional = _mapping(functionals[0], "parent violation functional")
    if functional.get("functional_id") != "absolute_error":
        raise ShadowChildProbeQualificationValidationError(
            "V1 requires the absolute_error parent functional"
        )
    threshold = _finite_number(functional.get("threshold"), "functional threshold")
    if threshold < 0:
        raise ShadowChildProbeQualificationValidationError(
            "functional threshold cannot be negative"
        )
    return threshold


def _robust_radius(
    model: Mapping[str, Any], row: Mapping[str, Any]
) -> float:
    grouping = model.get("radius_grouping")
    radii = model.get("radii")
    if grouping not in {"global", "per_scope", "per_context"} or not isinstance(
        radii, list
    ):
        raise ShadowChildProbeQualificationValidationError(
            "robust child radius grammar is invalid"
        )
    matches = []
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
        raise ShadowChildProbeQualificationValidationError(
            "robust child has no unique radius for a fresh context-scope pair"
        )
    radius = _finite_number(matches[0], "robust radius")
    if radius < 0:
        raise ShadowChildProbeQualificationValidationError(
            "robust radius cannot be negative"
        )
    return radius


def _gate(
    gate_id: str,
    metric_path: str,
    operator: str,
    threshold: float,
    actual: float,
) -> dict[str, Any]:
    if operator == ">=":
        passed = actual >= threshold
    elif operator == "<=":
        passed = actual <= threshold
    else:
        raise AssertionError(operator)
    return {
        "gate_id": gate_id,
        "metric_path": metric_path,
        "operator": operator,
        "threshold": threshold,
        "actual": actual,
        "passed": passed,
    }


def _robust_split_rows(
    rows: Sequence[Mapping[str, Any]],
    parent_lookup: Mapping[bytes, float],
    child_lookup: Mapping[bytes, float],
    child_model: Mapping[str, Any],
    functional_threshold: float,
) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        key = canonical_json_bytes(row["context"])
        parent_prediction = float(parent_lookup[key])
        child_center = float(child_lookup[key])
        radius = _robust_radius(child_model, row)
        observed = float(row["observed_value"])
        parent_error = abs(observed - parent_prediction)
        child_error = abs(observed - child_center)
        results.append(
            {
                "observation_id": row["observation_id"],
                "parent_error": parent_error,
                "child_error": child_error,
                "radius": radius,
                "nominal_covered": parent_error <= functional_threshold,
                "interval_covered": child_error <= radius,
            }
        )
    return results


def _robust_split_metrics(
    row_results: Sequence[Mapping[str, Any]], *, stress: bool, tail_fraction: float
) -> dict[str, Any]:
    parent_mae = _mean([float(item["parent_error"]) for item in row_results])
    child_mae = _mean([float(item["child_error"]) for item in row_results])
    parent_coverage = _mean(
        [1.0 if item["nominal_covered"] else 0.0 for item in row_results]
    )
    interval_coverage = _mean(
        [1.0 if item["interval_covered"] else 0.0 for item in row_results]
    )
    result = {
        "row_count": len(row_results),
        "parent_center_mae": parent_mae,
        "child_center_mae": child_mae,
        "nominal_mae_increase": child_mae - parent_mae,
        "parent_nominal_coverage": parent_coverage,
        "interval_coverage": interval_coverage,
        "coverage_gain": interval_coverage - parent_coverage,
        "mean_radius": _mean([float(item["radius"]) for item in row_results]),
    }
    if stress:
        tail_count = max(1, math.ceil(len(row_results) * tail_fraction))
        tail = sorted(
            row_results,
            key=lambda item: (-float(item["parent_error"]), item["observation_id"]),
        )[:tail_count]
        tail_coverage = _mean(
            [1.0 if item["interval_covered"] else 0.0 for item in tail]
        )
        result.update(
            {
                "tail_row_count": tail_count,
                "tail_observation_ids": [item["observation_id"] for item in tail],
                "tail_interval_coverage": tail_coverage,
                "safety_rate": interval_coverage,
            }
        )
    return result


def _evaluate_robust(
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
    evidence: Mapping[str, Sequence[Mapping[str, Any]]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    parent_model = _mapping(parent.get("model_class"), "parent model_class")
    child_model = _mapping(child.get("model_class"), "child model_class")
    if parent_model.get("kind") != "finite_point_table" or child_model.get(
        "kind"
    ) != "finite_interval_table":
        raise ShadowChildProbeQualificationValidationError(
            "robust qualification requires point parent and interval child"
        )
    parent_lookup = _prediction_lookup(
        parent_model.get("predictions"), "parent predictions"
    )
    child_lookup = _prediction_lookup(
        child_model.get("center_predictions"), "child center_predictions"
    )
    if parent_lookup != child_lookup:
        raise ShadowChildProbeQualificationValidationError(
            "robust child centers differ from the verified parent point table"
        )
    functional_threshold = _functional_threshold(parent)
    thresholds = contract["thresholds"]
    holdout_rows = _robust_split_rows(
        evidence["holdout"],
        parent_lookup,
        child_lookup,
        child_model,
        functional_threshold,
    )
    stress_rows = _robust_split_rows(
        evidence["stress"],
        parent_lookup,
        child_lookup,
        child_model,
        functional_threshold,
    )
    holdout = _robust_split_metrics(
        holdout_rows, stress=False, tail_fraction=float(thresholds["tail_fraction"])
    )
    stress = _robust_split_metrics(
        stress_rows, stress=True, tail_fraction=float(thresholds["tail_fraction"])
    )
    prediction_scale = _mean([abs(value) for value in parent_lookup.values()])
    scale = max(
        float(thresholds["numeric_epsilon"]),
        prediction_scale + functional_threshold,
    )
    normalized_radius = (
        holdout["mean_radius"] + stress["mean_radius"]
    ) / (2.0 * scale)
    aggregate = {
        "prediction_scale": scale,
        "normalized_radius": normalized_radius,
        "max_nominal_mae_increase": max(
            holdout["nominal_mae_increase"], stress["nominal_mae_increase"]
        ),
    }
    gates = [
        _gate(
            "holdout_coverage_gain",
            "holdout.coverage_gain",
            ">=",
            float(thresholds["coverage_tolerance"]),
            float(holdout["coverage_gain"]),
        ),
        _gate(
            "holdout_interval_coverage",
            "holdout.interval_coverage",
            ">=",
            float(thresholds["min_holdout_coverage"]),
            float(holdout["interval_coverage"]),
        ),
        _gate(
            "stress_tail_interval_coverage",
            "stress.tail_interval_coverage",
            ">=",
            float(thresholds["min_stress_tail_coverage"]),
            float(stress["tail_interval_coverage"]),
        ),
        _gate(
            "stress_safety_rate",
            "stress.safety_rate",
            ">=",
            float(thresholds["min_safety_rate"]),
            float(stress["safety_rate"]),
        ),
        _gate(
            "holdout_nominal_mae_increase",
            "holdout.nominal_mae_increase",
            "<=",
            float(thresholds["max_nominal_mae_increase"]),
            float(holdout["nominal_mae_increase"]),
        ),
        _gate(
            "stress_nominal_mae_increase",
            "stress.nominal_mae_increase",
            "<=",
            float(thresholds["max_nominal_mae_increase"]),
            float(stress["nominal_mae_increase"]),
        ),
        _gate(
            "normalized_interval_radius",
            "aggregate.normalized_radius",
            "<=",
            float(thresholds["max_normalized_radius"]),
            normalized_radius,
        ),
    ]
    counterexamples = sorted(
        {
            item["observation_id"]
            for item in (*holdout_rows, *stress_rows)
            if not item["interval_covered"]
        }
    )
    return {
        "result_kind": "ROBUST_INTERVAL_OPERATIONAL_PROBE_RESULT",
        "child_model_kind": "finite_interval_table",
        "probe_ids": [item["probe_id"] for item in PROBE_REGISTRY["ROBUST_INTERVAL_EXPANSION"]],
        "holdout": holdout,
        "stress": stress,
        "aggregate": aggregate,
        "gate_checks": gates,
        "all_gates_passed": all(item["passed"] for item in gates),
        "counterexample_observation_ids": counterexamples,
    }


def _fiber_lookup(
    transition_report: Mapping[str, Any], parent_contexts: Sequence[Mapping[str, Any]]
) -> dict[bytes, dict[str, Any]]:
    reduction = _mapping(
        transition_report.get("reduction_certificate"), "reduction_certificate"
    )
    if reduction.get("map_kind") != (
        "QUOTIENT_PROJECTION_WITH_FROZEN_PARENT_SNAPSHOT"
    ):
        raise ShadowChildProbeQualificationValidationError(
            "ideal child lacks the frozen quotient reduction certificate"
        )
    fiber_map = reduction.get("quotient_fiber_map")
    if not isinstance(fiber_map, list):
        raise ShadowChildProbeQualificationValidationError(
            "quotient_fiber_map must be a list"
        )
    lookup = {}
    for index, raw in enumerate(fiber_map):
        item = _mapping(raw, f"quotient_fiber_map[{index}]")
        _exact_keys(
            item,
            {"parent_context", "quotient_context"},
            f"quotient_fiber_map[{index}]",
        )
        key = canonical_json_bytes(item["parent_context"])
        if key in lookup:
            raise ShadowChildProbeQualificationValidationError(
                "quotient fiber map duplicates a parent context"
            )
        lookup[key] = _copy(item["quotient_context"])
    expected = {canonical_json_bytes(item) for item in parent_contexts}
    if set(lookup) != expected:
        raise ShadowChildProbeQualificationValidationError(
            "quotient fiber map does not cover the full parent object space"
        )
    return lookup


def _ideal_split_rows(
    rows: Sequence[Mapping[str, Any]],
    parent_lookup: Mapping[bytes, float],
    child_lookup: Mapping[bytes, float],
    fibers: Mapping[bytes, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        parent_key = canonical_json_bytes(row["context"])
        child_key = canonical_json_bytes(fibers[parent_key])
        parent_prediction = float(parent_lookup[parent_key])
        child_prediction = float(child_lookup[child_key])
        observed = float(row["observed_value"])
        parent_error = abs(observed - parent_prediction)
        child_error = abs(observed - child_prediction)
        results.append(
            {
                "observation_id": row["observation_id"],
                "parent_error": parent_error,
                "child_error": child_error,
                "mae_increase": child_error - parent_error,
                "parent_child_point_divergence": abs(
                    parent_prediction - child_prediction
                ),
            }
        )
    return results


def _ideal_split_metrics(
    row_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    parent_mae = _mean([float(item["parent_error"]) for item in row_results])
    child_mae = _mean([float(item["child_error"]) for item in row_results])
    return {
        "row_count": len(row_results),
        "parent_mae": parent_mae,
        "child_mae": child_mae,
        "mae_increase": child_mae - parent_mae,
        "max_parent_child_point_divergence": max(
            (
                float(item["parent_child_point_divergence"])
                for item in row_results
            ),
            default=0.0,
        ),
    }


def _evaluate_ideal(
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
    transition_report: Mapping[str, Any],
    evidence: Mapping[str, Sequence[Mapping[str, Any]]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    parent_model = _mapping(parent.get("model_class"), "parent model_class")
    child_model = _mapping(child.get("model_class"), "child model_class")
    if parent_model.get("kind") != "finite_point_table" or child_model.get(
        "kind"
    ) != "finite_point_table":
        raise ShadowChildProbeQualificationValidationError(
            "ideal qualification requires finite point parent and child"
        )
    parent_lookup = _prediction_lookup(
        parent_model.get("predictions"), "parent predictions"
    )
    child_lookup = _prediction_lookup(
        child_model.get("predictions"), "child predictions"
    )
    parent_contexts = _mapping(parent["object_space"], "parent object_space")[
        "contexts"
    ]
    fibers = _fiber_lookup(transition_report, parent_contexts)
    holdout_rows = _ideal_split_rows(
        evidence["holdout"], parent_lookup, child_lookup, fibers
    )
    stress_rows = _ideal_split_rows(
        evidence["stress"], parent_lookup, child_lookup, fibers
    )
    holdout = _ideal_split_metrics(holdout_rows)
    stress = _ideal_split_metrics(stress_rows)
    aggregate = {
        "max_parent_child_point_divergence": max(
            holdout["max_parent_child_point_divergence"],
            stress["max_parent_child_point_divergence"],
        ),
        "max_nominal_mae_increase": max(
            holdout["mae_increase"], stress["mae_increase"]
        ),
    }
    thresholds = contract["thresholds"]
    gates = [
        _gate(
            "holdout_nominal_mae_increase",
            "holdout.mae_increase",
            "<=",
            float(thresholds["max_nominal_mae_increase"]),
            float(holdout["mae_increase"]),
        ),
        _gate(
            "stress_nominal_mae_increase",
            "stress.mae_increase",
            "<=",
            float(thresholds["max_nominal_mae_increase"]),
            float(stress["mae_increase"]),
        ),
        _gate(
            "parent_child_point_divergence",
            "aggregate.max_parent_child_point_divergence",
            "<=",
            float(thresholds["max_probe_divergence"]),
            float(aggregate["max_parent_child_point_divergence"]),
        ),
    ]
    counterexamples = sorted(
        {
            item["observation_id"]
            for item in (*holdout_rows, *stress_rows)
            if item["mae_increase"]
            > float(thresholds["max_nominal_mae_increase"])
            or item["parent_child_point_divergence"]
            > float(thresholds["max_probe_divergence"])
        }
    )
    return {
        "result_kind": "QUOTIENT_IDEALIZATION_OPERATIONAL_PROBE_RESULT",
        "child_model_kind": "finite_point_table",
        "probe_ids": [item["probe_id"] for item in PROBE_REGISTRY["QUOTIENT_IDEALIZATION"]],
        "holdout": holdout,
        "stress": stress,
        "aggregate": aggregate,
        "gate_checks": gates,
        "all_gates_passed": all(item["passed"] for item in gates),
        "counterexample_observation_ids": counterexamples,
    }


def _source_transition_summary(
    source_report: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    source = _mapping(
        source_report.get("source_competition"), "source_competition"
    )
    return {
        "verification_status": receipt.get("status"),
        "contract_id": source_report.get("contract_id"),
        "contract_digest": source_report.get("contract_digest"),
        "report_digest": source_report.get("report_digest"),
        "source_competition_report_digest": source.get("report_digest"),
        "child_theory_state_digest": source_report.get(
            "child_theory_state_digest"
        ),
        "transition_kind": source_report.get("transition_kind"),
        "selected_candidate_id": source_report.get("selected_candidate_id"),
        "adoption_status": source_report.get("adoption_status"),
    }


def _qualification_binding(
    disposition: str, child_digest: str, epoch: str, fixed_anchor: str
) -> dict[str, Any]:
    statuses = {
        "QUALIFIED_NEW_EVALUATOR_EPOCH": (
            "OPERATIONAL_PROBE_QUALIFIED_SHADOW_ONLY",
            "QUALIFIED_ON_NEW_HOLDOUT_AND_STRESS",
        ),
        "NEEDS_NEW_EVALUATOR_EVIDENCE": (
            "NEW_EVALUATOR_EVIDENCE_INCOMPLETE",
            "OPERATIONAL_PROBE_EVIDENCE_REQUIRED",
        ),
        "INCOMPARABLE_NEW_EVALUATOR_EPOCH": (
            "NEW_EVALUATOR_EVIDENCE_INCOMPARABLE",
            "OPERATIONAL_PROBE_EVIDENCE_INCOMPARABLE",
        ),
        "FAILED_OPERATIONAL_PROBE_QUALIFICATION": (
            "OPERATIONAL_PROBE_QUALIFICATION_FAILED",
            "OPERATIONAL_PROBE_FAILED_ON_NEW_HOLDOUT_OR_STRESS",
        ),
    }
    evaluator_status, probe_status = statuses[disposition]
    return {
        "child_theory_state_digest": child_digest,
        "evaluator_epoch": epoch,
        "fixed_anchor": fixed_anchor,
        "evaluator_status": evaluator_status,
        "operational_probe_status": probe_status,
        "qualification_status": disposition,
        "adoption_status": ADOPTION_STATUS,
        "original_child_state_mutated": False,
        "source_evidence_allowed_for_child_scoring": False,
        "old_new_records_pooled": False,
        "logical_selective_erasure_applied": True,
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


def qualify_shadow_child_operational_probes(
    competition_input: Mapping[str, Any],
    competition_contract: Mapping[str, Any],
    competition_report: Mapping[str, Any],
    transition_contract: Mapping[str, Any],
    transition_report: Mapping[str, Any],
    qualification_input: Mapping[str, Any],
    qualification_contract: Mapping[str, Any],
    *,
    expected_competition_contract_digest: str,
    expected_competition_report_digest: str,
    expected_competition_input_artifacts: Mapping[str, Any] | None,
    expected_transition_contract_digest: str,
    expected_transition_report_digest: str,
    expected_transition_input_artifacts: Mapping[str, Any] | None,
    expected_qualification_input_digest: str,
    expected_qualification_contract_digest: str,
    input_artifacts: Mapping[str, Any] | None = None,
) -> ShadowChildProbeQualificationResult:
    """Evaluate only supplied new-epoch rows against a verified shadow child."""

    normalized_contract = _validate_contract(qualification_contract)
    expected_contract_digest = _require_digest(
        expected_qualification_contract_digest,
        "expected_qualification_contract_digest",
    )
    contract_digest = _digest(normalized_contract)
    if contract_digest != expected_contract_digest:
        raise ShadowChildProbeQualificationValidationError(
            "qualification contract digest differs from independent expectation"
        )
    expected_input_digest = _require_digest(
        expected_qualification_input_digest,
        "expected_qualification_input_digest",
    )
    actual_input_digest = _digest(qualification_input)
    if actual_input_digest != expected_input_digest:
        raise ShadowChildProbeQualificationValidationError(
            "qualification input digest differs from independent expectation"
        )
    if input_artifacts is not None and not isinstance(input_artifacts, Mapping):
        raise ShadowChildProbeQualificationValidationError(
            "input_artifacts must be an object or null"
        )
    artifacts = _copy(input_artifacts) if input_artifacts is not None else None

    source_report, source_receipt, parent, child = _verify_source_chain(
        competition_input,
        competition_contract,
        competition_report,
        transition_contract,
        transition_report,
        expected_competition_contract_digest=(
            expected_competition_contract_digest
        ),
        expected_competition_report_digest=expected_competition_report_digest,
        expected_competition_input_artifacts=expected_competition_input_artifacts,
        expected_transition_contract_digest=expected_transition_contract_digest,
        expected_transition_report_digest=expected_transition_report_digest,
        expected_transition_input_artifacts=expected_transition_input_artifacts,
    )
    original_child_bytes = canonical_json_bytes(child)
    normalized_input = _normalize_input(
        qualification_input,
        source_report=source_report,
        competition_input=competition_input,
        competition_report=competition_report,
        parent=parent,
    )
    transition_kind = _string(source_report["transition_kind"], "transition_kind")
    fixed_anchor = _string(parent["fixed_anchor"], "fixed_anchor")
    evaluator_definition, evaluator_definition_digest, derived_epoch = (
        _evaluator_definition(
            transition_contract_digest=source_report["contract_digest"],
            transition_report_digest=source_report["report_digest"],
            child_theory_state_digest=source_report["child_theory_state_digest"],
            transition_kind=transition_kind,
            fixed_anchor=fixed_anchor,
            qualification_contract=normalized_contract,
        )
    )
    evaluator_binding = _evaluator_binding(
        normalized_input,
        parent,
        evaluator_definition_digest,
        derived_epoch,
    )
    evidence_binding = _evidence_binding(normalized_input, normalized_contract)
    erasure_receipt = _selective_erasure_receipt(
        normalized_input, evidence_binding
    )

    probe_results = None
    if not evaluator_binding["comparable"]:
        disposition = SELECTION["incomparable_status"]
    elif not evidence_binding["sufficient"]:
        disposition = SELECTION["needs_evidence_status"]
    else:
        if transition_kind == "ROBUST_INTERVAL_EXPANSION":
            probe_results = _evaluate_robust(
                parent,
                child,
                normalized_input["evidence"],
                normalized_contract,
            )
        elif transition_kind == "QUOTIENT_IDEALIZATION":
            probe_results = _evaluate_ideal(
                parent,
                child,
                source_report,
                normalized_input["evidence"],
                normalized_contract,
            )
        else:
            raise ShadowChildProbeQualificationValidationError(
                "transition kind is not materializable in qualification V1"
            )
        disposition = (
            SELECTION["qualified_status"]
            if probe_results["all_gates_passed"]
            else SELECTION["failed_status"]
        )
    if canonical_json_bytes(child) != original_child_bytes:
        raise ShadowChildProbeQualificationValidationError(
            "qualification mutated the verified shadow child"
        )
    binding = _qualification_binding(
        disposition,
        source_report["child_theory_state_digest"],
        derived_epoch,
        fixed_anchor,
    )
    source_summary = _source_transition_summary(source_report, source_receipt)

    events = [
        _audit_event(
            0,
            "SOURCE_TRANSITION_VERIFIED",
            GENESIS_DIGEST,
            {
                "verification_status": source_summary["verification_status"],
                "source_transition_report_digest": source_summary["report_digest"],
                "child_theory_state_digest": source_summary[
                    "child_theory_state_digest"
                ],
            },
        )
    ]
    events.append(
        _audit_event(
            1,
            "NEW_EVALUATOR_AND_EVIDENCE_BOUND",
            events[-1]["event_digest"],
            {
                "derived_child_evaluator_epoch": derived_epoch,
                "comparable": evaluator_binding["comparable"],
                "evidence_sufficient": evidence_binding["sufficient"],
                "source_evidence_used_for_child_scoring": False,
            },
        )
    )
    events.append(
        _audit_event(
            2,
            "OPERATIONAL_PROBES_EVALUATED"
            if probe_results is not None
            else "OPERATIONAL_PROBES_NOT_EVALUATED",
            events[-1]["event_digest"],
            {
                "disposition": disposition,
                "all_gates_passed": (
                    probe_results["all_gates_passed"]
                    if probe_results is not None
                    else None
                ),
            },
        )
    )
    events.append(
        _audit_event(
            3,
            "ADOPTION_WITHHELD",
            events[-1]["event_digest"],
            {
                "adoption_status": ADOPTION_STATUS,
                "original_child_state_mutated": False,
            },
        )
    )

    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "contract_digest": contract_digest,
        "qualification_input_digest": actual_input_digest,
        "source_transition": source_summary,
        "qualification_id": normalized_input["public"]["qualification_id"],
        "child_theory_state_digest": source_report["child_theory_state_digest"],
        "transition_kind": transition_kind,
        "evaluator_definition": evaluator_definition,
        "evaluator_binding": evaluator_binding,
        "evidence_binding": evidence_binding,
        "selective_erasure_receipt": erasure_receipt,
        "probe_results": probe_results,
        "disposition": disposition,
        "qualification_binding": binding,
        "adoption_status": ADOPTION_STATUS,
        "nonclaims": _copy(normalized_contract["nonclaims"]),
        "input_artifacts": artifacts,
        "audit_events": events,
        "audit_head": events[-1]["event_digest"],
    }
    report = {**body, "report_digest": _digest(body)}
    _exact_keys(report, REPORT_FIELDS, "qualification_report")
    return ShadowChildProbeQualificationResult(report=report)


def verify_shadow_child_probe_qualification(
    competition_input: Mapping[str, Any],
    competition_contract: Mapping[str, Any],
    competition_report: Mapping[str, Any],
    transition_contract: Mapping[str, Any],
    transition_report: Mapping[str, Any],
    qualification_input: Mapping[str, Any],
    qualification_contract: Mapping[str, Any],
    qualification_report: Mapping[str, Any],
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
) -> dict[str, Any]:
    """Exact-replay a qualification report against every independent anchor."""

    expected_report_digest = _require_digest(
        expected_qualification_report_digest,
        "expected_qualification_report_digest",
    )
    supplied = _copy(_mapping(qualification_report, "qualification_report"))
    _exact_keys(supplied, REPORT_FIELDS, "qualification_report")
    if expected_qualification_input_artifacts is not None and not isinstance(
        expected_qualification_input_artifacts, Mapping
    ):
        raise ShadowChildProbeQualificationValidationError(
            "expected_qualification_input_artifacts must be an object or null"
        )
    expected_artifacts = (
        _copy(expected_qualification_input_artifacts)
        if expected_qualification_input_artifacts is not None
        else None
    )
    if supplied.get("input_artifacts") != expected_artifacts:
        raise ShadowChildProbeQualificationValidationError(
            "qualification input artifacts differ from independent expectation"
        )
    fresh = qualify_shadow_child_operational_probes(
        competition_input,
        competition_contract,
        competition_report,
        transition_contract,
        transition_report,
        qualification_input,
        qualification_contract,
        expected_competition_contract_digest=(
            expected_competition_contract_digest
        ),
        expected_competition_report_digest=expected_competition_report_digest,
        expected_competition_input_artifacts=expected_competition_input_artifacts,
        expected_transition_contract_digest=expected_transition_contract_digest,
        expected_transition_report_digest=expected_transition_report_digest,
        expected_transition_input_artifacts=expected_transition_input_artifacts,
        expected_qualification_input_digest=expected_qualification_input_digest,
        expected_qualification_contract_digest=expected_qualification_contract_digest,
        input_artifacts=expected_artifacts,
    ).to_dict()
    if fresh["report_digest"] != expected_report_digest:
        raise ShadowChildProbeQualificationValidationError(
            "replayed qualification report digest differs from expectation"
        )
    if canonical_json_bytes(supplied) != canonical_json_bytes(fresh):
        raise ShadowChildProbeQualificationValidationError(
            "supplied qualification report differs from exact replay"
        )
    return {
        "status": "VERIFIED_" + fresh["disposition"],
        "disposition": fresh["disposition"],
        "qualified": fresh["disposition"] == "QUALIFIED_NEW_EVALUATOR_EPOCH",
        "report_digest": fresh["report_digest"],
        "contract_digest": fresh["contract_digest"],
        "qualification_input_digest": fresh["qualification_input_digest"],
        "source_transition_report_digest": fresh["source_transition"][
            "report_digest"
        ],
        "child_theory_state_digest": fresh["child_theory_state_digest"],
        "evaluator_epoch": fresh["qualification_binding"]["evaluator_epoch"],
        "adoption_status": fresh["adoption_status"],
    }
