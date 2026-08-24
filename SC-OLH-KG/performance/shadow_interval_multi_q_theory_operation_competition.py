"""Bounded interval/two-Q theory-operation competition V2.

The implementation is an additive shadow evaluator.  It exact-replays the
public interval/two-Q adapter verifier, synthesizes candidates from discovery
rows, ranks them on validation rows, and uses stress rows only to confirm the
one provisional winner.  A selected candidate is a proposal: this module does
not materialize a theory, mutate the seed, execute a transition, invent a
probe/language, adopt, promote, or write a current pointer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import itertools
import json
import math
from typing import Any, Mapping, Sequence

from performance.shadow_interval_multi_q_recompetition_adapter import (
    ShadowIntervalMultiQRecompetitionAdapterValidationError,
    verify_shadow_interval_multi_q_recompetition_adapter,
)
from performance.theory_operation_competition import canonical_json_bytes


CONTRACT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-theory-operation-competition-contract/2"
)
CONTRACT_ID = "shadow_interval_multi_q_theory_operation_competition_v2"
SOURCE_ADAPTER_CONTRACT_ID = "shadow_interval_multi_q_recompetition_adapter_v1"
SOURCE_ADAPTER_CONTRACT_DIGEST = (
    "sha256:16d2a30873e3f8b2e56fe5d7ac272140eb83dbcb441d8d80a892c4028f28f029"
)
SOURCE_ADAPTER_REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-recompetition-adapter-report/1"
)
INPUT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-theory-operation-competition-input/2"
)
REPORT_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-theory-operation-competition-report/2"
)
CANDIDATE_SCHEMA_VERSION = (
    "sc-olh-kg.shadow-interval-multi-q-theory-operation-candidate/2"
)
FROZEN_CONTRACT_DIGEST = (
    "sha256:4c30c0b1a2cdec92ab1676e98677b620907bb9652bff1ce71865fce9d45ccd1e"
)
GENESIS_DIGEST = "sha256:" + "0" * 64

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
DIAGNOSTIC_OPERATION_REGISTRY = [
    {"operation_id": "reestimate", "operation_kind": "expand"},
    {"operation_id": "noise", "operation_kind": "expand"},
    {"operation_id": "scope", "operation_kind": "restrict"},
    {"operation_id": "mixture", "operation_kind": "expand"},
    {"operation_id": "simplify", "operation_kind": "quotient"},
    {"operation_id": "robustify", "operation_kind": "expand"},
    {"operation_id": "new_probe", "operation_kind": "probe"},
    {"operation_id": "language_last", "operation_kind": "language"},
]
CANDIDATE_FAMILY_REGISTRY = [
    {
        "candidate_family": "interval_robustify",
        "operation_kind": "expand",
        "construction_role": "DISCOVERY_DERIVED_INTERVAL_EXPANSION",
        "related_diagnostic_stage": "robustify",
    },
    {
        "candidate_family": "interval_restrict",
        "operation_kind": "restrict",
        "construction_role": "UNIFORM_RADIUS_CONTRACTION",
        "related_diagnostic_stage": None,
    },
    {
        "candidate_family": "interval_quotient",
        "operation_kind": "quotient",
        "construction_role": "CONSERVATIVE_QUOTIENT_ENVELOPE",
        "related_diagnostic_stage": "simplify",
    },
]
FAMILY_ORDER = {
    item["candidate_family"]: index
    for index, item in enumerate(CANDIDATE_FAMILY_REGISTRY)
}
PRIOR_RECORD_KEYS = [
    "competition",
    "qualification",
    "failure_boundary_probe",
    "restriction",
    "post_restriction_adjudication",
]
SPLITS = ["discovery", "validation", "stress"]
EXACT_ROWS_PER_CELL = {"discovery": 2, "validation": 1, "stress": 1}
MAX_CONTEXT_COUNT = 64
MAX_SCOPE_COUNT = 16
MAX_REMOVABLE_FEATURE_COUNT = 8
MAX_RAW_CANDIDATE_COUNT = 260

SELECT_EXPANSION = "SELECT_SHADOW_INTERVAL_EXPANSION_CANDIDATE"
SELECT_RESTRICTION = "SELECT_SHADOW_UNIFORM_RESTRICTION_CANDIDATE"
SELECT_QUOTIENT = "SELECT_SHADOW_CONSERVATIVE_QUOTIENT_ENVELOPE_CANDIDATE"
NEEDS_EVIDENCE = "INTERVAL_MULTI_Q_COMPETITION_NEEDS_EXACT_FRESH_EVIDENCE"
INCOMPARABLE_EPOCH = "INTERVAL_MULTI_Q_COMPETITION_INCOMPARABLE_EVALUATOR_EPOCH"
EARLY_UNRESOLVED = "INTERVAL_MULTI_Q_COMPETITION_EARLY_DIAGNOSTIC_UNRESOLVED"
NO_VALIDATION_WINNER = "INTERVAL_MULTI_Q_COMPETITION_NO_VALIDATION_WINNER"
STRESS_FAILED = (
    "INTERVAL_MULTI_Q_COMPETITION_PROVISIONAL_WINNER_FAILED_STRESS_CONFIRMATION"
)
BLOCKED_ADAPTER_EVIDENCE = (
    "INTERVAL_MULTI_Q_COMPETITION_BLOCKED_ADAPTER_NEEDS_POST_RESTRICTION_EVIDENCE"
)
BLOCKED_ADAPTER_EPOCH = (
    "INTERVAL_MULTI_Q_COMPETITION_BLOCKED_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH"
)
ALL_DISPOSITIONS = {
    SELECT_EXPANSION,
    SELECT_RESTRICTION,
    SELECT_QUOTIENT,
    NEEDS_EVIDENCE,
    INCOMPARABLE_EPOCH,
    EARLY_UNRESOLVED,
    NO_VALIDATION_WINNER,
    STRESS_FAILED,
    BLOCKED_ADAPTER_EVIDENCE,
    BLOCKED_ADAPTER_EPOCH,
}
DISPOSITION_REGISTRY = {
    "select_interval_expansion": SELECT_EXPANSION,
    "select_uniform_restriction": SELECT_RESTRICTION,
    "select_conservative_quotient": SELECT_QUOTIENT,
    "needs_exact_fresh_evidence": NEEDS_EVIDENCE,
    "incomparable_evaluator_epoch": INCOMPARABLE_EPOCH,
    "early_diagnostic_unresolved": EARLY_UNRESOLVED,
    "no_validation_winner": NO_VALIDATION_WINNER,
    "provisional_winner_failed_stress_confirmation": STRESS_FAILED,
    "blocked_adapter_needs_post_restriction_evidence": BLOCKED_ADAPTER_EVIDENCE,
    "blocked_adapter_incomparable_post_restriction_epoch": BLOCKED_ADAPTER_EPOCH,
}

ADOPTION_ELIGIBILITY = "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED"
ADOPTION_STATUS = "NOT_ADOPTED_SHADOW_ONLY"
PROMOTION_STATUS = "NOT_PROMOTED"
CURRENT_STATUS = "NOT_CURRENT"

MANDATORY_NONCLAIMS = (
    "shadow_only",
    "interval_multi_q_competition_v2_only",
    "full_adapter_chain_exact_replay_is_not_external_attestation",
    "v2_seed_finite_interval_table_only",
    "v2_seed_fixed_two_probe_registry_only",
    "no_v2_point_projection",
    "no_q_or_v_deletion",
    "q_v_registry_preservation_is_not_probe_value_equality",
    "discovery_only_candidate_synthesis",
    "validation_and_stress_never_enter_synthesis",
    "five_prior_generations_are_scoring_excluded",
    "no_cross_epoch_pooling",
    "caller_supplied_static_rows_only",
    "local_epoch_is_not_external_attestation",
    "interval_expansion_is_frozen_finite_table_only",
    "uniform_restriction_is_frozen_finite_table_only",
    "quotient_envelope_is_frozen_finite_table_only",
    "quotient_envelope_preserves_interval_containment_not_point_predictions",
    "selected_candidate_is_not_materialized_theory",
    "no_theory_state_materialization",
    "no_child_transition",
    "no_new_probe_execution",
    "no_language_or_predicate_invention",
    "no_language_expansion_execution",
    "language_last_resort_not_certified",
    "null_seed_route_has_no_fallback",
    "unqualified_repair_base_is_not_accepted_theory",
    "no_source_seed_mutation",
    "no_rollback_execution",
    "no_adoption_eligibility_determination",
    "no_adoption",
    "no_promotion",
    "no_current_pointer_write",
    "no_parent_source_or_ambient_state_write",
    "no_h_t_to_h_t_plus_1_acceptance",
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
        "source_adapter_contract_id",
        "source_adapter_contract_digest",
        "source_adapter_report_schema_version",
        "input_schema_version",
        "report_schema_version",
        "candidate_schema_version",
        "fixed_probe_registry",
        "diagnostic_order",
        "diagnostic_operation_registry",
        "diagnostic_metric_policy",
        "candidate_family_registry",
        "disposition_registry",
        "evidence_policy",
        "evaluator_epoch_policy",
        "candidate_generation_policy",
        "multiplier_registry",
        "thresholds",
        "validation_selection_policy",
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
        "competition_id",
        "source_adapter",
        "evaluator",
        "prior_record_exclusion",
        "evidence",
    }
)
SOURCE_ADAPTER_FIELDS = frozenset(
    {
        "adapter_contract_digest",
        "adapter_report_digest",
        "adapter_input_digest",
        "adapter_id",
        "adapter_disposition",
        "recompetition_seed_digest",
        "recompetition_seed_id",
        "seed_theory_state_digest",
    }
)
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
REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "contract_digest",
        "competition_input_digest",
        "competition_id",
        "source_adapter",
        "source_adapter_disposition",
        "source_seed_summary",
        "evaluator_binding",
        "evidence_binding",
        "evidence_digests",
        "evidence_coverage",
        "candidate_family_registry",
        "candidate_commitments",
        "candidate_semantic_deduplication",
        "diagnostic_trace",
        "baseline_metrics",
        "interval_expansion_candidates",
        "uniform_restriction_candidates",
        "conservative_quotient_envelope_candidates",
        "validation_selection",
        "stress_confirmation",
        "disposition",
        "selected_candidate",
        "selection_boundary",
        "next_probe_spec",
        "language_last_route",
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


class ShadowIntervalMultiQTheoryOperationCompetitionValidationError(ValueError):
    """Raised when the frozen V2 contract or exact replay fails."""


class ShadowIntervalMultiQTheoryOperationCompetitionDisposition(str, Enum):
    SELECT_SHADOW_INTERVAL_EXPANSION_CANDIDATE = SELECT_EXPANSION
    SELECT_SHADOW_UNIFORM_RESTRICTION_CANDIDATE = SELECT_RESTRICTION
    SELECT_SHADOW_CONSERVATIVE_QUOTIENT_ENVELOPE_CANDIDATE = SELECT_QUOTIENT
    INTERVAL_MULTI_Q_COMPETITION_NEEDS_EXACT_FRESH_EVIDENCE = NEEDS_EVIDENCE
    INTERVAL_MULTI_Q_COMPETITION_INCOMPARABLE_EVALUATOR_EPOCH = INCOMPARABLE_EPOCH
    INTERVAL_MULTI_Q_COMPETITION_EARLY_DIAGNOSTIC_UNRESOLVED = EARLY_UNRESOLVED
    INTERVAL_MULTI_Q_COMPETITION_NO_VALIDATION_WINNER = NO_VALIDATION_WINNER
    INTERVAL_MULTI_Q_COMPETITION_PROVISIONAL_WINNER_FAILED_STRESS_CONFIRMATION = (
        STRESS_FAILED
    )
    INTERVAL_MULTI_Q_COMPETITION_BLOCKED_ADAPTER_NEEDS_POST_RESTRICTION_EVIDENCE = (
        BLOCKED_ADAPTER_EVIDENCE
    )
    INTERVAL_MULTI_Q_COMPETITION_BLOCKED_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH = (
        BLOCKED_ADAPTER_EPOCH
    )


@dataclass(frozen=True)
class ShadowIntervalMultiQTheoryOperationCompetitionResult:
    report: Mapping[str, Any]

    @property
    def disposition(self) -> str:
        return str(self.report["disposition"])

    @property
    def report_digest(self) -> str:
        return str(self.report["report_digest"])

    @property
    def candidate_selected(self) -> bool:
        return self.report["selected_candidate"] is not None

    @property
    def selected_candidate_id(self) -> str | None:
        selected = self.report["selected_candidate"]
        return None if selected is None else str(selected["candidate_id"])

    @property
    def selected_operation_kind(self) -> str | None:
        selected = self.report["selected_candidate"]
        return None if selected is None else str(selected["operation_kind"])

    def to_dict(self) -> dict[str, Any]:
        return _copy(self.report)


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            f"{label} must be an object"
        )
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            f"{label} keys differ from frozen schema: "
            f"missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}"
        )


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            f"{label} must be a nonempty string"
        )
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            f"{label} must be a finite number"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            f"{label} must be a finite number"
        )
    return result


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_digest(value: Any, label: str) -> str:
    text = _string(value, label)
    if len(text) != 71 or not text.startswith("sha256:"):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            f"{label} must be a sha256 digest"
        )
    try:
        int(text[7:], 16)
    except ValueError as exc:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            f"{label} must be a sha256 digest"
        ) from exc
    return text


def _mean(values: Sequence[float], label: str) -> float:
    if not values:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            f"{label} cannot be empty"
        )
    numeric = [float(item) for item in values]
    if not all(math.isfinite(item) for item in numeric):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
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
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                f"{label} overflowed finite arithmetic"
            ) from exc
    if not math.isfinite(result):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            f"{label} is not finite"
        )
    return result


def _sse(values: Sequence[float], center: float, label: str) -> float:
    try:
        result = math.fsum((float(item) - center) ** 2 for item in values)
    except OverflowError as exc:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            f"{label} overflowed finite arithmetic"
        ) from exc
    if not math.isfinite(result):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            f"{label} is not finite"
        )
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


def validate_shadow_interval_multi_q_theory_operation_competition_contract(
    contract_value: Any,
) -> dict[str, Any]:
    contract = _copy(_mapping(contract_value, "interval competition contract"))
    _exact_keys(contract, CONTRACT_FIELDS, "interval competition contract")
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "interval competition contract schema_version is not frozen V2"
        )
    if contract.get("contract_id") != CONTRACT_ID:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "interval competition contract_id is not frozen V2"
        )
    if contract.get("disposition_registry") != DISPOSITION_REGISTRY:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "interval competition disposition registry is not the exact-ten V2 registry"
        )
    if _digest(contract) != FROZEN_CONTRACT_DIGEST:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "interval competition contract differs from the canonical frozen manifest"
        )
    return contract


def derive_shadow_interval_multi_q_theory_operation_competition_id(
    *,
    adapter_contract_digest: str,
    adapter_report_digest: str,
    recompetition_seed_digest: str | None,
    seed_theory_state_digest: str | None,
    interval_competition_contract: Mapping[str, Any],
) -> str:
    contract = validate_shadow_interval_multi_q_theory_operation_competition_contract(
        interval_competition_contract
    )
    body = {
        "adapter_contract_digest": _require_digest(
            adapter_contract_digest, "adapter_contract_digest"
        ),
        "adapter_report_digest": _require_digest(
            adapter_report_digest, "adapter_report_digest"
        ),
        "recompetition_seed_digest": recompetition_seed_digest,
        "seed_theory_state_digest": seed_theory_state_digest,
        "interval_competition_contract_digest": _digest(contract),
    }
    for key in ("recompetition_seed_digest", "seed_theory_state_digest"):
        if body[key] is not None:
            body[key] = _require_digest(body[key], key)
    return "shadow-interval-multi-q-competition:" + _digest(body)[7:]


def derive_shadow_interval_multi_q_theory_operation_competition_epoch(
    *,
    adapter_contract_digest: str,
    adapter_report_digest: str,
    recompetition_seed_digest: str,
    seed_theory_state_digest: str,
    fixed_anchor: str,
    interval_competition_contract: Mapping[str, Any],
) -> str:
    contract = validate_shadow_interval_multi_q_theory_operation_competition_contract(
        interval_competition_contract
    )
    body = {
        "adapter_contract_digest": _require_digest(
            adapter_contract_digest, "adapter_contract_digest"
        ),
        "adapter_report_digest": _require_digest(
            adapter_report_digest, "adapter_report_digest"
        ),
        "recompetition_seed_digest": _require_digest(
            recompetition_seed_digest, "recompetition_seed_digest"
        ),
        "seed_theory_state_digest": _require_digest(
            seed_theory_state_digest, "seed_theory_state_digest"
        ),
        "fixed_anchor": _string(fixed_anchor, "fixed_anchor"),
        "interval_competition_contract_digest": _digest(contract),
    }
    return "shadow-interval-multi-q-v2-epoch:" + _digest(body)[7:]


def _model_geometry(state_value: Any) -> dict[str, Any]:
    state = _mapping(state_value, "recompetition seed theory_state")
    contexts_value = _mapping(state.get("object_space"), "seed object_space").get(
        "contexts"
    )
    if not isinstance(contexts_value, list) or not contexts_value:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed contexts must be a nonempty list"
        )
    contexts: list[dict[str, Any]] = []
    context_keys: set[bytes] = set()
    for index, item in enumerate(contexts_value):
        context = _copy(_mapping(item, f"seed contexts[{index}]"))
        key = canonical_json_bytes(context)
        if key in context_keys:
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "seed contexts must be unique"
            )
        context_keys.add(key)
        contexts.append(context)
    scopes_value = state.get("scope_ids")
    if not isinstance(scopes_value, list) or not scopes_value:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed scope_ids must be a nonempty list"
        )
    scopes = [_string(item, "seed scope_id") for item in scopes_value]
    if len(set(scopes)) != len(scopes):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed scope_ids must be unique"
        )
    removable_value = state.get("removable_feature_ids")
    if not isinstance(removable_value, list):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed removable_feature_ids must be a list"
        )
    removable = [_string(item, "seed removable_feature_id") for item in removable_value]
    if len(set(removable)) != len(removable):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed removable_feature_ids must be unique"
        )
    if len(contexts) > MAX_CONTEXT_COUNT:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed context count exceeds the frozen maximum"
        )
    if len(scopes) > MAX_SCOPE_COUNT:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed scope count exceeds the frozen maximum"
        )
    if len(removable) > MAX_REMOVABLE_FEATURE_COUNT:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed removable-feature count exceeds the frozen maximum"
        )
    if state.get("probe_ids") != PROBE_IDS:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed does not preserve the exact frozen two-Q registry"
        )
    model = _copy(_mapping(state.get("model_class"), "seed model_class"))
    _exact_keys(
        model,
        frozenset({"kind", "center_predictions", "radius_grouping", "radii"}),
        "seed model_class",
    )
    if model["kind"] != "finite_interval_table":
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed model_class must be finite_interval_table"
        )
    centers_value = model["center_predictions"]
    if not isinstance(centers_value, list):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed center_predictions must be a list"
        )
    centers: dict[bytes, float] = {}
    for index, item_value in enumerate(centers_value):
        item = _mapping(item_value, f"seed center_predictions[{index}]")
        _exact_keys(item, frozenset({"context", "value"}), "seed center prediction")
        context = _copy(_mapping(item["context"], "seed center context"))
        key = canonical_json_bytes(context)
        if key in centers:
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "seed center contexts are duplicated"
            )
        centers[key] = _finite(item["value"], "seed center value")
    if set(centers) != context_keys:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed centers do not exactly cover the context registry"
        )
    grouping = model["radius_grouping"]
    if grouping not in {"global", "per_scope", "per_context"}:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed radius_grouping is unsupported"
        )
    radii_value = model["radii"]
    if not isinstance(radii_value, list) or not radii_value:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed radii must be a nonempty list"
        )
    radii: dict[bytes, tuple[dict[str, Any], float]] = {}
    for index, item_value in enumerate(radii_value):
        item = _mapping(item_value, f"seed radii[{index}]")
        _exact_keys(item, frozenset({"group", "radius"}), "seed radius entry")
        group = _copy(_mapping(item["group"], "seed radius group"))
        key = canonical_json_bytes(group)
        if key in radii:
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "seed radius group keys are duplicated"
            )
        radius = _finite(item["radius"], "seed radius")
        if radius < 0.0:
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "seed radii must be nonnegative"
            )
        radii[key] = (group, radius)
    if grouping == "global":
        expected_groups = {canonical_json_bytes({"global": "*"})}
    elif grouping == "per_scope":
        expected_groups = {
            canonical_json_bytes({"scope_id": scope}) for scope in scopes
        }
    else:
        expected_groups = {
            canonical_json_bytes({"context": context}) for context in contexts
        }
    if set(radii) != expected_groups:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed radius groups do not exactly equal the registered grouping keys"
        )
    result = {
        "state": _copy(state),
        "state_digest": _digest(state),
        "object_space": _copy(state["object_space"]),
        "contexts": contexts,
        "context_keys": context_keys,
        "scopes": scopes,
        "removable": removable,
        "probe_ids": _copy(PROBE_IDS),
        "violation_functionals": _copy(state.get("violation_functionals")),
        "model": model,
        "model_digest": _digest(model),
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
        key = canonical_json_bytes({"global": "*"})
    elif grouping == "per_scope":
        key = canonical_json_bytes({"scope_id": scope_id})
    else:
        key = canonical_json_bytes({"context": context})
    try:
        return float(geometry["radii"][key][1])
    except KeyError as exc:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "model radius does not map uniquely to a registered context-scope pair"
        ) from exc


def _geometry_from_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    pseudo_state = {
        "object_space": candidate["object_space"],
        "scope_ids": candidate["scope_ids"],
        "removable_feature_ids": candidate["removable_feature_ids"],
        "probe_ids": candidate["probe_ids"],
        "violation_functionals": candidate["violation_functionals"],
        "model_class": candidate["model_class"],
    }
    return _model_geometry(pseudo_state)


def _functional_threshold(geometry: Mapping[str, Any]) -> float:
    values = geometry["violation_functionals"]
    if not isinstance(values, list):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed violation_functionals must be a list"
        )
    matches = []
    for item_value in values:
        item = _mapping(item_value, "seed violation functional")
        if item.get("functional_id") == "absolute_error":
            matches.append(_finite(item.get("threshold"), "absolute_error threshold"))
    if len(matches) != 1 or matches[0] < 0.0:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed requires one nonnegative absolute_error violation functional"
        )
    return matches[0]


def _prediction_scale(geometry: Mapping[str, Any], epsilon: float) -> float:
    result = max(
        epsilon,
        _mean([abs(value) for value in geometry["centers"].values()], "center scale")
        + _functional_threshold(geometry),
    )
    if not math.isfinite(result):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "prediction scale is not finite"
        )
    return result


def _sorted_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted((_copy(item) for item in rows), key=canonical_json_bytes)


def _evidence_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return _digest(_sorted_rows(rows))


def _candidate_without_evaluation(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _copy(value)
        for key, value in candidate.items()
        if key != "validation_evaluation"
    }


def _finalize_candidate(body: Mapping[str, Any]) -> dict[str, Any]:
    candidate = _copy(body)
    candidate["model_class_digest"] = _digest(candidate["model_class"])
    semantic_body = {
        "object_space": candidate["object_space"],
        "model_class": candidate["model_class"],
        "scope_ids": candidate["scope_ids"],
        "removable_feature_ids": candidate["removable_feature_ids"],
        "probe_ids": candidate["probe_ids"],
        "violation_functionals": candidate["violation_functionals"],
    }
    candidate["semantic_model_digest"] = _digest(semantic_body)
    id_body = {
        key: value
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
    candidate["candidate_id"] = (
        str(candidate["candidate_family"]) + ":" + _digest(id_body)[7:]
    )
    candidate["validation_evaluation"] = None
    ordered = {
        "schema_version": candidate["schema_version"],
        "candidate_id": candidate["candidate_id"],
        "candidate_family": candidate["candidate_family"],
        "operation_kind": candidate["operation_kind"],
        "source_theory_state_digest": candidate["source_theory_state_digest"],
        "object_space": candidate["object_space"],
        "model_class": candidate["model_class"],
        "model_class_digest": candidate["model_class_digest"],
        "semantic_model_digest": candidate["semantic_model_digest"],
        "scope_ids": candidate["scope_ids"],
        "removable_feature_ids": candidate["removable_feature_ids"],
        "probe_ids": candidate["probe_ids"],
        "violation_functionals": candidate["violation_functionals"],
        "construction": candidate["construction"],
        "certificate": candidate["certificate"],
        "discovery_metrics": candidate["discovery_metrics"],
        "discovery_admissible": candidate["discovery_admissible"],
        "validation_evaluation": None,
    }
    _exact_keys(ordered, CANDIDATE_FIELDS, "candidate")
    return ordered


def _model_with_radii(
    source: Mapping[str, Any], radii: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "kind": "finite_interval_table",
        "center_predictions": _copy(source["model"]["center_predictions"]),
        "radius_grouping": source["grouping"],
        "radii": _copy(radii),
    }


def _base_candidate(
    source: Mapping[str, Any], family: str, operation_kind: str
) -> dict[str, Any]:
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_family": family,
        "operation_kind": operation_kind,
        "source_theory_state_digest": source["state_digest"],
        "object_space": _copy(source["object_space"]),
        "model_class": None,
        "scope_ids": _copy(source["scopes"]),
        "removable_feature_ids": _copy(source["removable"]),
        "probe_ids": _copy(source["probe_ids"]),
        "violation_functionals": _copy(source["violation_functionals"]),
        "construction": None,
        "certificate": None,
        "discovery_metrics": None,
        "discovery_admissible": False,
    }


def _expansion_candidate(
    source: Mapping[str, Any],
    discovery_rows: Sequence[Mapping[str, Any]],
    scale: float,
    discovery_digest: str,
) -> dict[str, Any] | None:
    required: dict[bytes, float] = {
        key: radius for key, (_, radius) in source["radii"].items()
    }
    for row in discovery_rows:
        context_key = canonical_json_bytes(row["context"])
        error = abs(float(row["observed_value"]) - source["centers"][context_key])
        grouping = source["grouping"]
        if grouping == "global":
            key = canonical_json_bytes({"global": "*"})
        elif grouping == "per_scope":
            key = canonical_json_bytes({"scope_id": row["scope_id"]})
        else:
            key = canonical_json_bytes({"context": row["context"]})
        required[key] = max(required[key], error)
    radii = [
        {"group": _copy(source["radii"][key][0]), "radius": required[key]}
        for key in sorted(required)
    ]
    model = _model_with_radii(source, radii)
    strict = any(
        required[key] > float(source["radii"][key][1]) for key in required
    )
    contains_source = all(
        required[key] >= float(source["radii"][key][1]) for key in required
    )
    if not strict:
        return None
    candidate_geometry = _geometry_from_candidate(
        {
            **_base_candidate(source, "interval_robustify", "expand"),
            "model_class": model,
        }
    )
    metrics = _split_metrics(
        discovery_rows, candidate_geometry, scale, tail_indices=None
    )
    envelope = metrics["boundary_violation_count"] == 0
    body = _base_candidate(source, "interval_robustify", "expand")
    body.update(
        {
            "model_class": model,
            "construction": {
                "construction_kind": "DISCOVERY_DERIVED_INTERVAL_EXPANSION",
                "source_radius_grouping_preserved": True,
                "discovery_evidence_digest": discovery_digest,
                "synthesis_evidence_digest": discovery_digest,
                "validation_data_used": False,
                "stress_data_used": False,
            },
            "certificate": {
                "certificate_kind": "FINITE_INTERVAL_DISCOVERY_ENVELOPE_SUPERSET",
                "source_model_class_digest": source["model_digest"],
                "expanded_model_class_digest": _digest(model),
                "checked_radius_group_count": len(radii),
                "checked_context_scope_pair_count": len(source["contexts"])
                * len(source["scopes"]),
                "centers_byte_equal": canonical_json_bytes(
                    model["center_predictions"]
                )
                == canonical_json_bytes(source["model"]["center_predictions"]),
                "grouping_and_group_keys_byte_equal": (
                    model["radius_grouping"] == source["grouping"]
                    and [item["group"] for item in radii]
                    == [source["radii"][key][0] for key in sorted(source["radii"])]
                ),
                "all_expanded_radii_finite_nonnegative": all(
                    math.isfinite(float(item["radius"]))
                    and float(item["radius"]) >= 0.0
                    for item in radii
                ),
                "all_expanded_radii_gte_source": contains_source,
                "at_least_one_radius_strictly_expanded": strict,
                "all_discovery_rows_enveloped": envelope,
                "strict_superset_verified": contains_source and strict,
            },
            "discovery_metrics": metrics,
            "discovery_admissible": contains_source and strict and envelope,
        }
    )
    return _finalize_candidate(body)


def _restriction_candidates(
    source: Mapping[str, Any],
    discovery_rows: Sequence[Mapping[str, Any]],
    scale: float,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    source_keys = sorted(source["radii"])
    for spec in contract["multiplier_registry"]:
        numerator = int(spec["numerator"])
        denominator = int(spec["denominator"])
        alpha = numerator / denominator
        if alpha != float(spec["radius_multiplier"]) or not 0.0 < alpha < 1.0:
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "multiplier registry is not a strict rational contraction"
            )
        radii = [
            {
                "group": _copy(source["radii"][key][0]),
                "radius": float(source["radii"][key][1]) * alpha,
            }
            for key in source_keys
        ]
        model = _model_with_radii(source, radii)
        finite_nonnegative = all(
            math.isfinite(float(item["radius"]))
            and float(item["radius"]) >= 0.0
            for item in radii
        )
        lte = all(
            float(item["radius"]) <= float(source["radii"][key][1])
            for item, key in zip(radii, source_keys)
        )
        strict = any(
            float(item["radius"]) < float(source["radii"][key][1])
            for item, key in zip(radii, source_keys)
        )
        base = _base_candidate(source, "interval_restrict", "restrict")
        candidate_geometry = _geometry_from_candidate({**base, "model_class": model})
        metrics = _split_metrics(
            discovery_rows, candidate_geometry, scale, tail_indices=None
        )
        certificate = {
            "certificate_kind": "STRICT_FINITE_INTERVAL_UNIFORM_CONTRACTION_SUBSET",
            "source_model_class_digest": source["model_digest"],
            "restricted_model_class_digest": _digest(model),
            "radius_multiplier": alpha,
            "checked_radius_group_count": len(radii),
            "checked_context_scope_pair_count": len(source["contexts"])
            * len(source["scopes"]),
            "centers_byte_equal": canonical_json_bytes(
                model["center_predictions"]
            )
            == canonical_json_bytes(source["model"]["center_predictions"]),
            "grouping_and_group_keys_byte_equal": (
                model["radius_grouping"] == source["grouping"]
                and [item["group"] for item in radii]
                == [source["radii"][key][0] for key in source_keys]
            ),
            "all_restricted_radii_finite_nonnegative": finite_nonnegative,
            "all_restricted_radii_lte_source": lte,
            "at_least_one_radius_strictly_reduced": strict,
            "strict_subset_verified": finite_nonnegative and lte and strict,
        }
        base.update(
            {
                "model_class": model,
                "construction": {
                    "construction_kind": "UNIFORM_RADIUS_CONTRACTION",
                    "candidate_label": spec["candidate_label"],
                    "numerator": numerator,
                    "denominator": denominator,
                    "radius_multiplier": alpha,
                    "diagnostic_scope_correction": False,
                    "synthesis_evidence_digest": None,
                    "validation_data_used": False,
                    "stress_data_used": False,
                },
                "certificate": certificate,
                "discovery_metrics": metrics,
                "discovery_admissible": (
                    certificate["strict_subset_verified"]
                    and metrics["boundary_violation_rate"] == 0.0
                ),
            }
        )
        # A rational alpha < 1 does not imply a represented-float strict
        # contraction (all-zero and minimum-subnormal radii are counterexamples).
        if certificate["strict_subset_verified"]:
            result.append(_finalize_candidate(base))
    return result


def _project_context(context: Mapping[str, Any], removed: set[str]) -> dict[str, Any]:
    return {key: _copy(value) for key, value in context.items() if key not in removed}


def _quotient_candidate(
    source: Mapping[str, Any],
    removed_features: Sequence[str],
    discovery_rows: Sequence[Mapping[str, Any]],
    scale: float,
) -> dict[str, Any] | None:
    removed = set(removed_features)
    fibers: dict[bytes, dict[str, Any]] = {}
    for context in source["contexts"]:
        quotient = _project_context(context, removed)
        key = canonical_json_bytes(quotient)
        fibers.setdefault(key, {"context": quotient, "parents": []})["parents"].append(
            context
        )
    if len(fibers) >= len(source["contexts"]):
        return None
    quotient_contexts = [fibers[key]["context"] for key in sorted(fibers)]
    quotient_centers: dict[bytes, float] = {}
    hulls: dict[bytes, tuple[float, float]] = {}
    for key in sorted(fibers):
        lowers: list[float] = []
        uppers: list[float] = []
        for context in fibers[key]["parents"]:
            center = source["centers"][canonical_json_bytes(context)]
            for scope in source["scopes"]:
                radius = _radius_for_pair(source, scope, context)
                lower = center - radius
                upper = center + radius
                if not math.isfinite(lower) or not math.isfinite(upper):
                    return None
                lowers.append(lower)
                uppers.append(upper)
        lower = min(lowers)
        upper = max(uppers)
        midpoint = lower / 2.0 + upper / 2.0
        if not math.isfinite(midpoint):
            return None
        hulls[key] = (lower, upper)
        quotient_centers[key] = midpoint
    # The quotient has one exact per-quotient-context hull, independently of
    # the source grouping.  Each hull ranges across every parent in the fiber
    # and every registered scope.
    checked = 0
    contained = True
    fiber_envelopes: list[dict[str, Any]] = []
    radii: list[dict[str, Any]] = []
    for key in sorted(fibers):
        quotient_context = fibers[key]["context"]
        lower, upper = hulls[key]
        midpoint = quotient_centers[key]
        radius = max(upper - midpoint, midpoint - lower)
        if not math.isfinite(radius) or radius < 0.0:
            return None
        parent_intervals: list[dict[str, Any]] = []
        for context in sorted(fibers[key]["parents"], key=canonical_json_bytes):
            source_center = source["centers"][canonical_json_bytes(context)]
            for scope in sorted(source["scopes"]):
                source_radius = _radius_for_pair(source, scope, context)
                parent_lower = source_center - source_radius
                parent_upper = source_center + source_radius
                parent_intervals.append(
                    {
                        "scope_id": scope,
                        "parent_context": _copy(context),
                        "parent_center": source_center,
                        "parent_radius": source_radius,
                        "parent_lower": parent_lower,
                        "parent_upper": parent_upper,
                    }
                )
                if parent_lower < midpoint - radius or parent_upper > midpoint + radius:
                    contained = False
                checked += 1
        radii.append(
            {"group": {"context": _copy(quotient_context)}, "radius": radius}
        )
        fiber_envelopes.append(
            {
                "quotient_context": _copy(quotient_context),
                "hull_lower": lower,
                "hull_upper": upper,
                "hull_midpoint": midpoint,
                "hull_radius": radius,
                "parent_intervals": parent_intervals,
            }
        )
    center_entries = [
        {"context": fibers[key]["context"], "value": quotient_centers[key]}
        for key in sorted(fibers)
    ]
    model = {
        "kind": "finite_interval_table",
        "center_predictions": center_entries,
        "radius_grouping": "per_context",
        "radii": radii,
    }
    object_space = _copy(source["object_space"])
    object_space["contexts"] = quotient_contexts
    if isinstance(object_space.get("feature_ids"), list):
        object_space["feature_ids"] = [
            item for item in object_space["feature_ids"] if item not in removed
        ]
    remaining_removable = [
        item for item in source["removable"] if item not in removed
    ]
    base = _base_candidate(source, "interval_quotient", "quotient")
    base["object_space"] = object_space
    base["model_class"] = model
    base["removable_feature_ids"] = remaining_removable
    candidate_geometry = _geometry_from_candidate(base)
    projected_discovery = [
        {**_copy(row), "context": _project_context(row["context"], removed)}
        for row in discovery_rows
    ]
    metrics = _split_metrics(
        projected_discovery, candidate_geometry, scale, tail_indices=None
    )
    # Recheck every parent interval against the stored quotient model rather
    # than trusting the construction formula alone.
    for context in source["contexts"]:
        quotient_context = _project_context(context, removed)
        quotient_center = candidate_geometry["centers"][
            canonical_json_bytes(quotient_context)
        ]
        source_center = source["centers"][canonical_json_bytes(context)]
        for scope in source["scopes"]:
            source_radius = _radius_for_pair(source, scope, context)
            child_radius = _radius_for_pair(
                candidate_geometry, scope, quotient_context
            )
            parent_lower = source_center - source_radius
            parent_upper = source_center + source_radius
            child_lower = quotient_center - child_radius
            child_upper = quotient_center + child_radius
            if not all(
                math.isfinite(item)
                for item in (parent_lower, parent_upper, child_lower, child_upper)
            ):
                return None
            if child_lower > parent_lower or child_upper < parent_upper:
                contained = False
    if not contained:
        # The algebraic envelope can be one ulp too small after stored-float
        # reconstruction.  Omit that construction; never retain a false
        # containment certificate.
        return None
    context_reduction = 1.0 - len(quotient_contexts) / len(source["contexts"])
    certificate = {
        "certificate_kind": "FINITE_SOURCE_INTERVAL_HULL_QUOTIENT_ENVELOPE",
        "source_model_class_digest": source["model_digest"],
        "quotient_model_class_digest": _digest(model),
        "removed_feature_ids": sorted(removed_features),
        "parent_context_count": len(source["contexts"]),
        "quotient_context_count": len(quotient_contexts),
        "context_reduction_fraction": context_reduction,
        "checked_parent_context_scope_pair_count": checked,
        "center_policy": "SOURCE_INTERVAL_HULL_MIDPOINT_L_OVER_2_PLUS_U_OVER_2",
        "radius_policy": "SOURCE_INTERVAL_CONSERVATIVE_ENVELOPE",
        "quotient_radius_grouping": "per_context",
        "fiber_envelope_table": fiber_envelopes,
        "source_theory_state_digest": source["state_digest"],
        "source_restore_method": "RESTORE_EXACT_VERIFIED_RECOMPETITION_SEED_THEORY_STATE",
        "quotient_context_keys_exact": set(candidate_geometry["context_keys"])
        == {canonical_json_bytes(item) for item in quotient_contexts},
        "all_parent_intervals_contained_under_quotient_map": contained,
        "quotient_alone_recovers_parent": False,
        "point_prediction_preservation_claimed": False,
        "envelope_certificate_verified": contained and context_reduction > 0.0,
    }
    base.update(
        {
            "construction": {
                "construction_kind": "CONSERVATIVE_QUOTIENT_ENVELOPE",
                "removed_feature_ids": sorted(removed_features),
                "synthesis_evidence_digest": None,
                "source_radius_grouping": source["grouping"],
                "quotient_radius_grouping": "per_context",
                "fiber_envelope_table": fiber_envelopes,
                "source_theory_state_digest": source["state_digest"],
                "source_restore_method": "RESTORE_EXACT_VERIFIED_RECOMPETITION_SEED_THEORY_STATE",
                "quotient_map": [
                    {
                        "parent_context": context,
                        "quotient_context": _project_context(context, removed),
                    }
                    for context in source["contexts"]
                ],
                "validation_data_used": False,
                "stress_data_used": False,
            },
            "certificate": certificate,
            "discovery_metrics": metrics,
            "discovery_admissible": certificate["envelope_certificate_verified"],
        }
    )
    return _finalize_candidate(base)


def _semantic_deduplicate(
    raw_candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(
        (_copy(item) for item in raw_candidates),
        key=lambda item: (FAMILY_ORDER[item["candidate_family"]], item["candidate_id"]),
    )
    retained: list[dict[str, Any]] = []
    by_semantic: dict[str, dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []
    for candidate in ordered:
        semantic = candidate["semantic_model_digest"]
        if semantic not in by_semantic:
            by_semantic[semantic] = candidate
            retained.append(candidate)
        else:
            dropped.append(
                {
                    "dropped_candidate_id": candidate["candidate_id"],
                    "retained_candidate_id": by_semantic[semantic]["candidate_id"],
                    "semantic_model_digest": semantic,
                }
            )
    raw_count = {
        family: sum(item["candidate_family"] == family for item in ordered)
        for family in FAMILY_ORDER
    }
    retained_count = {
        family: sum(item["candidate_family"] == family for item in retained)
        for family in FAMILY_ORDER
    }
    ledger = {
        "raw_candidate_count_by_family": raw_count,
        "retained_candidate_count_by_family": retained_count,
        "retained_candidate_ids": [item["candidate_id"] for item in retained],
        "dropped_duplicate_candidates": dropped,
        "semantic_deduplication_verified": (
            len({item["semantic_model_digest"] for item in retained}) == len(retained)
            and sum(raw_count.values())
            == sum(retained_count.values()) + len(dropped)
        ),
    }
    return retained, ledger


def synthesize_shadow_interval_multi_q_theory_operation_candidates(
    recompetition_seed: Mapping[str, Any],
    discovery_rows: Sequence[Mapping[str, Any]],
    evaluator: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build candidates from one structurally exact discovery table.

    This four-argument helper is an untrusted discovery-only pure constructor,
    not a V2 report or a verified commitment.  It cannot replay the adapter,
    derive the authoritative epoch, or recover five-generation prior IDs;
    only the full runner and public verifier establish those boundaries.
    """

    normalized_contract = (
        validate_shadow_interval_multi_q_theory_operation_competition_contract(
            contract
        )
    )
    seed = _mapping(recompetition_seed, "recompetition_seed")
    source = _model_geometry(seed.get("theory_state"))
    if seed.get("theory_state_digest") != source["state_digest"]:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "seed theory_state digest is not exact"
        )
    evaluator_value = _mapping(evaluator, "evaluator")
    _exact_keys(evaluator_value, frozenset({"evaluator_epoch", "fixed_anchor"}), "evaluator")
    _string(evaluator_value.get("evaluator_epoch"), "evaluator_epoch")
    _string(evaluator_value.get("fixed_anchor"), "fixed_anchor")
    if evaluator_value.get("fixed_anchor") != source["state"].get("fixed_anchor"):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "evaluator fixed_anchor differs from the seed"
        )
    if not isinstance(discovery_rows, Sequence) or isinstance(
        discovery_rows, (str, bytes, bytearray)
    ):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "discovery_rows must be a list"
        )
    rows = [_copy(_mapping(item, "discovery row")) for item in discovery_rows]
    observation_ids: list[str] = []
    cell_counts = {
        (scope, canonical_json_bytes(context)): 0
        for scope in source["scopes"]
        for context in source["contexts"]
    }
    for row in rows:
        _exact_keys(row, ROW_FIELDS, "discovery row")
        _finite(row["observed_value"], "discovery observed_value")
        observation_ids.append(_string(row["observation_id"], "discovery observation_id"))
        if row["evaluator_epoch"] != evaluator_value["evaluator_epoch"]:
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "discovery row evaluator epoch differs"
            )
        if row["fixed_anchor"] != evaluator_value["fixed_anchor"]:
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "discovery row fixed_anchor differs"
            )
        if row["scope_id"] not in source["scopes"]:
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "discovery row scope is unregistered"
            )
        if canonical_json_bytes(row["context"]) not in source["context_keys"]:
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "discovery row context is unregistered"
            )
        cell_counts[
            (str(row["scope_id"]), canonical_json_bytes(row["context"]))
        ] += 1
    if len(observation_ids) != len(set(observation_ids)):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "discovery observation IDs must be globally unique"
        )
    if not cell_counts or any(count != 2 for count in cell_counts.values()):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "discovery requires exactly two rows per registered context-scope cell"
        )
    scale = _prediction_scale(
        source, float(normalized_contract["thresholds"]["numeric_epsilon"])
    )
    discovery_digest = _evidence_digest(rows)
    expansion = _expansion_candidate(source, rows, scale, discovery_digest)
    raw: list[dict[str, Any]] = [] if expansion is None else [expansion]
    raw.extend(_restriction_candidates(source, rows, scale, normalized_contract))
    for size in range(1, len(source["removable"]) + 1):
        for subset in itertools.combinations(source["removable"], size):
            candidate = _quotient_candidate(source, subset, rows, scale)
            if candidate is not None:
                raw.append(candidate)
    if len(raw) > MAX_RAW_CANDIDATE_COUNT:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "raw candidate count exceeds the frozen maximum"
        )
    retained, ledger = _semantic_deduplicate(raw)
    raw_bodies = [_candidate_without_evaluation(item) for item in raw]
    commitments_body = {
        "recompetition_seed_digest": _digest(seed),
        "recompetition_seed_id": seed.get("seed_id"),
        "source_theory_state_digest": source["state_digest"],
        "interval_competition_contract_digest": _digest(normalized_contract),
        "discovery_evidence_digest": discovery_digest,
        "sorted_raw_candidate_bodies_without_validation": sorted(
            raw_bodies, key=lambda item: item["candidate_id"]
        ),
        "semantic_deduplication": ledger,
        "retained_candidate_ids": ledger["retained_candidate_ids"],
    }
    commitments = {
        "recompetition_seed_digest": _digest(seed),
        "recompetition_seed_id": seed.get("seed_id"),
        "source_theory_state_digest": source["state_digest"],
        "interval_competition_contract_digest": _digest(normalized_contract),
        "discovery_evidence_digest": discovery_digest,
        "raw_candidate_count": len(raw),
        "retained_candidate_count": len(retained),
        "raw_candidate_commitment_digest": _digest(
            commitments_body["sorted_raw_candidate_bodies_without_validation"]
        ),
        "candidate_commitment_digest": _digest(commitments_body),
        "validation_data_used": False,
        "stress_data_used": False,
    }
    by_family = {
        family: [
            _copy(item) for item in retained if item["candidate_family"] == family
        ]
        for family in FAMILY_ORDER
    }
    return {
        "prediction_scale": scale,
        "raw_candidates": raw,
        "retained_candidates": retained,
        "retained_candidates_by_family": by_family,
        "candidate_commitments": commitments,
        "candidate_semantic_deduplication": ledger,
    }


def _row_geometry_values(
    row: Mapping[str, Any], geometry: Mapping[str, Any], scale: float
) -> tuple[float, float, float, float]:
    context_key = canonical_json_bytes(row["context"])
    try:
        center = float(geometry["centers"][context_key])
    except KeyError as exc:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "candidate centers do not cover an evidence row"
        ) from exc
    radius = _radius_for_pair(geometry, row["scope_id"], row["context"])
    error = abs(float(row["observed_value"]) - center)
    margin = (radius - error) / scale
    exceedance = max(0.0, (error - radius) / scale)
    if not all(math.isfinite(item) for item in (error, radius, margin, exceedance)):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "probe arithmetic is not finite"
        )
    return error, radius, margin, exceedance


def _split_metrics(
    rows: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    scale: float,
    *,
    tail_indices: set[int] | None,
) -> dict[str, Any]:
    threshold = _functional_threshold(geometry)
    errors: list[float] = []
    radii: list[float] = []
    margins: list[float] = []
    exceedances: list[float] = []
    absolute_ids: list[str] = []
    boundary_ids: list[str] = []
    tail_passes: list[bool] = []
    for index, row in enumerate(rows):
        error, radius, margin, exceedance = _row_geometry_values(row, geometry, scale)
        errors.append(error)
        radii.append(radius)
        margins.append(margin)
        exceedances.append(exceedance)
        if error > threshold:
            absolute_ids.append(str(row["observation_id"]))
        # Raw finite inequality is authoritative.  Do not replace this with
        # margin < 0: normalization can underflow a true negative to -0.0.
        if error > radius:
            boundary_ids.append(str(row["observation_id"]))
        if tail_indices is not None and index in tail_indices:
            tail_passes.append(error <= radius)
    if not rows:
        return {
            "row_count": 0,
            "mean_absolute_center_error": None,
            "max_absolute_center_error": None,
            "mean_normalized_center_error": None,
            "max_normalized_center_error": None,
            "absolute_error_violation_count": 0,
            "absolute_error_violation_rate": None,
            "absolute_error_counterexample_observation_ids": [],
            "min_normalized_signed_interval_boundary_margin": None,
            "boundary_violation_count": 0,
            "boundary_violation_rate": None,
            "raw_boundary_coverage": None,
            "mean_normalized_boundary_exceedance": None,
            "max_normalized_boundary_exceedance": None,
            "boundary_counterexample_observation_ids": [],
            "mean_normalized_radius": None,
            "source_tail_row_count": 0 if tail_indices is not None else None,
            "source_tail_coverage": None,
        }
    mean_error = _mean(errors, "mean center error")
    mean_radius = _mean(radii, "mean radius")
    mean_exceedance = _mean(exceedances, "mean boundary exceedance")
    boundary_count = len(boundary_ids)
    absolute_count = len(absolute_ids)
    return {
        "row_count": len(rows),
        "mean_absolute_center_error": mean_error,
        "max_absolute_center_error": max(errors),
        "mean_normalized_center_error": mean_error / scale,
        "max_normalized_center_error": max(errors) / scale,
        "absolute_error_violation_count": absolute_count,
        "absolute_error_violation_rate": absolute_count / len(rows),
        "absolute_error_counterexample_observation_ids": sorted(absolute_ids),
        "min_normalized_signed_interval_boundary_margin": min(margins),
        "boundary_violation_count": boundary_count,
        "boundary_violation_rate": boundary_count / len(rows),
        "raw_boundary_coverage": 1.0 - boundary_count / len(rows),
        "mean_normalized_boundary_exceedance": mean_exceedance,
        "max_normalized_boundary_exceedance": max(exceedances),
        "boundary_counterexample_observation_ids": sorted(boundary_ids),
        "mean_normalized_radius": mean_radius / scale,
        "source_tail_row_count": None if tail_indices is None else len(tail_indices),
        "source_tail_coverage": (
            None if tail_indices is None else sum(tail_passes) / len(tail_passes)
        ),
    }


def _source_tail(
    rows: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    scale: float,
    fraction: float,
) -> tuple[set[int], dict[str, Any]]:
    if not rows:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "source tail requires nonempty held-out rows"
        )
    values: list[float] = []
    for row in rows:
        error, radius, _, _ = _row_geometry_values(row, source, scale)
        # Tail membership is intentionally decided in the source prediction
        # units.  A positive raw violation can underflow to zero after
        # division by a very large prediction scale, which would otherwise
        # make it tie with rows that do not violate the interval at all.
        raw_exceedance = error - radius if error > radius else 0.0
        if not math.isfinite(raw_exceedance):
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "source tail raw boundary exceedance is not finite"
            )
        values.append(raw_exceedance)
    k = max(1, math.ceil(len(rows) * fraction))
    cutoff = sorted(values, reverse=True)[k - 1]
    indices = {index for index, value in enumerate(values) if value >= cutoff}
    definition = {
        "source_tail_statistic": "raw_boundary_exceedance",
        "cutoff_units": "source_prediction_units",
        "source_tail_fraction": fraction,
        "source_tail_k": k,
        "source_tail_cutoff": cutoff,
        "source_tail_row_count": len(indices),
        "tail_tie_policy": "INCLUDE_ALL_AT_OR_ABOVE_SOURCE_CUTOFF",
        "observation_id_used_for_membership": False,
    }
    return indices, definition


def _candidate_rows(
    rows: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if candidate["candidate_family"] != "interval_quotient":
        return [_copy(row) for row in rows]
    removed = set(candidate["construction"]["removed_feature_ids"])
    return [
        {**_copy(row), "context": _project_context(row["context"], removed)}
        for row in rows
    ]


def _probe_divergence(
    source_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    candidate_geometry: Mapping[str, Any],
    scale: float,
) -> dict[str, Any]:
    q1: list[float] = []
    q2: list[float] = []
    for source_row, candidate_row in zip(source_rows, candidate_rows):
        source_error, source_radius, source_margin, _ = _row_geometry_values(
            source_row, source, scale
        )
        candidate_error, candidate_radius, candidate_margin, _ = _row_geometry_values(
            candidate_row, candidate_geometry, scale
        )
        q1.append(abs(candidate_error - source_error) / scale)
        q2.append(abs(candidate_margin - source_margin))
        if not all(
            math.isfinite(item)
            for item in (source_radius, candidate_radius, q1[-1], q2[-1])
        ):
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "probe divergence is not finite"
            )
    return {
        "normalized_absolute_error_q_max_divergence": max(q1),
        "normalized_margin_q_max_divergence": max(q2),
    }


def _diagnostics(
    rows: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    scale: float,
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    thresholds = contract["thresholds"]
    residuals: list[float] = []
    raw_boundary: list[bool] = []
    signed_exceedances: list[float] = []
    by_cell_exceedance: dict[tuple[str, bytes], list[float]] = {}
    by_scope: dict[str, list[tuple[float, bool]]] = {
        scope: [] for scope in source["scopes"]
    }
    for row in rows:
        center = source["centers"][canonical_json_bytes(row["context"])]
        residual = float(row["observed_value"]) - center
        if not math.isfinite(residual):
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "diagnostic residual is not finite"
            )
        radius = _radius_for_pair(source, row["scope_id"], row["context"])
        violation = abs(residual) > radius
        exceedance = math.copysign(max(0.0, abs(residual) - radius), residual)
        residuals.append(residual)
        raw_boundary.append(violation)
        signed_exceedances.append(exceedance)
        cell_key = (str(row["scope_id"]), canonical_json_bytes(row["context"]))
        by_cell_exceedance.setdefault(cell_key, []).append(exceedance)
        raw_exceedance = abs(exceedance)
        if not math.isfinite(raw_exceedance):
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "scope raw boundary exceedance is not finite"
            )
        by_scope[str(row["scope_id"])].append(
            (raw_exceedance, violation)
        )

    # reestimate: one global center shift, with the same source radii.  Scale
    # before averaging so minimum-subnormal residuals cannot disappear behind
    # an absolute epsilon or an underflowing raw mean.
    if not any(raw_boundary):
        raw_residual_scale = None
        pre_rounding_beta_scaled = None
        beta = None
        effective_beta_scaled = None
        source_scaled_mae = None
        shifted_scaled_mae = None
        gain = None
        shifted_rate = None
        reestimate_status = "NOT_APPLICABLE_NO_RAW_BOUNDARY_VIOLATION"
    else:
        raw_residual_scale = max(abs(item) for item in residuals)
        max_scaled_residuals = [
            item / raw_residual_scale for item in residuals
        ]
        pre_rounding_beta_scaled = _mean(
            max_scaled_residuals,
            "reestimate pre-rounding max-scaled beta",
        )
        beta = pre_rounding_beta_scaled * raw_residual_scale
        effective_beta_scaled = beta / raw_residual_scale
        source_scaled_mae = _mean(
            [abs(item) for item in max_scaled_residuals],
            "reestimate source max-scaled mae",
        )
        if not all(
            math.isfinite(item)
            for item in (
                raw_residual_scale,
                pre_rounding_beta_scaled,
                beta,
                effective_beta_scaled,
                source_scaled_mae,
            )
        ) or raw_residual_scale <= 0.0 or source_scaled_mae <= 0.0:
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "a raw boundary violation has no finite positive reestimate scale"
            )
        shifted_raw_residuals = [item - beta for item in residuals]
        if not all(math.isfinite(item) for item in shifted_raw_residuals):
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "reestimate shifted residual is not finite"
            )
        max_scaled_shifted_residuals = [
            item / raw_residual_scale for item in shifted_raw_residuals
        ]
        shifted_scaled_mae = _mean(
            [abs(item) for item in max_scaled_shifted_residuals],
            "reestimate shifted max-scaled mae",
        )
        gain = (
            source_scaled_mae - shifted_scaled_mae
        ) / source_scaled_mae
        if not math.isfinite(gain):
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "reestimate fractional gain is not finite"
            )
        shifted_violations = 0
        for row, shifted_residual in zip(rows, shifted_raw_residuals):
            if abs(shifted_residual) > _radius_for_pair(
                source, row["scope_id"], row["context"]
            ):
                shifted_violations += 1
        shifted_rate = shifted_violations / len(rows)
        source_rate = sum(raw_boundary) / len(rows)
        if (
            gain >= float(thresholds["reestimate_min_fractional_mae_gain"])
            and shifted_rate <= source_rate
        ):
            reestimate_status = "VIABLE_EXPLANATION"
        else:
            reestimate_status = "EXCLUDED_BY_DISCOVERY"
    reestimate = {
        "stage": "reestimate",
        "metric_status": reestimate_status,
        "gate_status": (
            "BLOCKING" if reestimate_status == "VIABLE_EXPLANATION" else "CLEARED"
        ),
        "metrics": {
            "raw_residual_scale": raw_residual_scale,
            "pre_rounding_max_scaled_center_shift": pre_rounding_beta_scaled,
            "global_center_shift": beta,
            "effective_max_scaled_center_shift": effective_beta_scaled,
            "source_max_scaled_mean_absolute_residual": source_scaled_mae,
            "shifted_max_scaled_mean_absolute_residual": shifted_scaled_mae,
            "fractional_mae_gain": gain,
            "shifted_raw_boundary_violation_rate": shifted_rate,
        },
    }
    if reestimate["gate_status"] == "BLOCKING":
        return (
            [
                reestimate,
                *[
                    {
                        "stage": stage,
                        "metric_status": "NOT_EVALUATED_BLOCKED_BY_REESTIMATE",
                        "gate_status": "NOT_EVALUATED",
                        "metrics": None,
                    }
                    for stage in ("noise", "scope", "mixture")
                ],
            ],
            "reestimate",
        )

    signed_exceedance_scale = max(abs(item) for item in signed_exceedances)
    max_scaled_signed_exceedances = (
        [0.0 for _ in signed_exceedances]
        if signed_exceedance_scale == 0.0
        else [item / signed_exceedance_scale for item in signed_exceedances]
    )
    if not all(math.isfinite(item) for item in max_scaled_signed_exceedances):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "max-scaled signed boundary exceedance is not finite"
        )
    overall_mean = _mean(
        max_scaled_signed_exceedances,
        "noise max-scaled signed-exceedance mean",
    )
    total_sse = _sse(
        max_scaled_signed_exceedances,
        overall_mean,
        "noise total max-scaled signed-exceedance SSE",
    )
    within_parts: list[float] = []
    for values in by_cell_exceedance.values():
        scaled_values = (
            [0.0 for _ in values]
            if signed_exceedance_scale == 0.0
            else [item / signed_exceedance_scale for item in values]
        )
        cell_mean = _mean(
            scaled_values,
            "noise max-scaled signed-exceedance cell mean",
        )
        within_parts.append(
            _sse(
                scaled_values,
                cell_mean,
                "noise within-pair max-scaled signed-exceedance SSE",
            )
        )
    try:
        within_sse = math.fsum(within_parts)
    except OverflowError as exc:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "noise diagnostic overflowed finite arithmetic"
        ) from exc
    if not math.isfinite(within_sse):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "noise within-pair SSE is not finite"
        )
    if total_sse == 0.0:
        noise_status = "NOT_APPLICABLE_ZERO_SIGNED_EXCEEDANCE_VARIANCE"
        within_fraction = 0.0
    else:
        within_fraction = within_sse / total_sse
        noise_status = (
            "VIABLE_EXPLANATION"
            if within_fraction
            >= float(thresholds["noise_min_within_pair_variance_fraction"])
            else "EXCLUDED_BY_DISCOVERY"
        )
    noise = {
        "stage": "noise",
        "metric_status": noise_status,
        "gate_status": "BLOCKING" if noise_status == "VIABLE_EXPLANATION" else "CLEARED",
        "metrics": {
            "signed_raw_boundary_exceedance_scale": signed_exceedance_scale,
            "total_max_scaled_signed_exceedance_sse": total_sse,
            "within_scope_context_pair_max_scaled_signed_exceedance_sse": within_sse,
            "within_pair_variance_fraction": within_fraction,
            "grouping": "EXACT_SCOPE_CONTEXT_PAIR",
        },
    }
    if noise["gate_status"] == "BLOCKING":
        return (
            [
                reestimate,
                noise,
                *[
                    {
                        "stage": stage,
                        "metric_status": "NOT_EVALUATED_BLOCKED_BY_NOISE",
                        "gate_status": "NOT_EVALUATED",
                        "metrics": None,
                    }
                    for stage in ("scope", "mixture")
                ],
            ],
            "noise",
        )

    scope_rates = {
        scope: sum(item[1] for item in values) / len(values)
        for scope, values in by_scope.items()
    }
    scope_exceedance_scale = max(abs(item) for item in signed_exceedances)
    if any(raw_boundary) and scope_exceedance_scale == 0.0:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "a raw boundary violation has zero represented exceedance"
        )
    scope_scaled_sums = {
        scope: (
            0.0
            if scope_exceedance_scale == 0.0
            else math.fsum(item[0] / scope_exceedance_scale for item in values)
        )
        for scope, values in by_scope.items()
    }
    if not all(math.isfinite(item) for item in scope_scaled_sums.values()):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "scope max-scaled boundary exceedance sum is not finite"
        )
    if len(source["scopes"]) == 1:
        scope_status = "NOT_APPLICABLE_SINGLE_SCOPE"
        structure = 0.0
    elif not any(raw_boundary):
        scope_status = "NOT_APPLICABLE_NO_SCOPE_BOUNDARY_EXCEEDANCE"
        structure = 0.0
    else:
        maximum_scope_sum = max(scope_scaled_sums.values())
        if maximum_scope_sum <= 0.0:
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "scope max-scaled boundary exceedance maximum is not positive"
            )
        exceedance_spread = (
            maximum_scope_sum - min(scope_scaled_sums.values())
        ) / maximum_scope_sum
        rate_spread = max(scope_rates.values()) - min(scope_rates.values())
        structure = max(exceedance_spread, rate_spread)
        scope_status = (
            "VIABLE_EXPLANATION"
            if structure >= float(thresholds["scope_min_structure_ratio"])
            else "EXCLUDED_BY_DISCOVERY"
        )
    scope_item = {
        "stage": "scope",
        "metric_status": scope_status,
        "gate_status": "BLOCKING" if scope_status == "VIABLE_EXPLANATION" else "CLEARED",
        "metrics": {
            "scope_raw_boundary_exceedance_scale": scope_exceedance_scale,
            "scope_sum_max_scaled_raw_boundary_exceedance": scope_scaled_sums,
            "scope_raw_boundary_violation_rate": scope_rates,
            "scope_structure_ratio": structure,
            "restriction_candidate_count": 0,
        },
    }
    if scope_item["gate_status"] == "BLOCKING":
        return (
            [
                reestimate,
                noise,
                scope_item,
                {
                    "stage": "mixture",
                    "metric_status": "NOT_EVALUATED_BLOCKED_BY_SCOPE",
                    "gate_status": "NOT_EVALUATED",
                    "metrics": None,
                },
            ],
            "scope",
        )

    active = sorted(
        item
        for item, violation in zip(max_scaled_signed_exceedances, raw_boundary)
        if violation
    )
    if len(active) < 4:
        mixture_status = "NOT_APPLICABLE_FEWER_THAN_FOUR_RAW_BOUNDARY_VIOLATIONS"
        reduction = 0.0
        split_index = None
    else:
        active_mean = _mean(active, "mixture active mean")
        sse0 = _sse(active, active_mean, "mixture total SSE")
        if sse0 == 0.0:
            mixture_status = (
                "NOT_APPLICABLE_ZERO_VIOLATING_SIGNED_EXCEEDANCE_VARIANCE"
            )
            reduction = 0.0
            split_index = None
        else:
            choices: list[tuple[float, int]] = []
            for index in range(2, len(active) - 1):
                left = active[:index]
                right = active[index:]
                left_mean = _mean(left, "mixture left mean")
                right_mean = _mean(right, "mixture right mean")
                left_sse = _sse(left, left_mean, "mixture left-cluster SSE")
                right_sse = _sse(right, right_mean, "mixture right-cluster SSE")
                try:
                    sse = math.fsum([left_sse, right_sse])
                except OverflowError as exc:
                    raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                        "mixture split SSE overflowed finite arithmetic"
                    ) from exc
                if not math.isfinite(sse):
                    raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                        "mixture split SSE is not finite"
                    )
                choices.append((sse, index))
            best_sse, split_index = min(choices, key=lambda item: (item[0], item[1]))
            reduction = (sse0 - best_sse) / sse0
            mixture_status = (
                "VIABLE_EXPLANATION"
                if reduction
                >= float(thresholds["mixture_min_sse_reduction_fraction"])
                else "EXCLUDED_BY_DISCOVERY"
            )
    mixture = {
        "stage": "mixture",
        "metric_status": mixture_status,
        "gate_status": (
            "BLOCKING" if mixture_status == "VIABLE_EXPLANATION" else "CLEARED"
        ),
        "metrics": {
            "raw_boundary_violation_count": len(active),
            "signed_raw_boundary_exceedance_scale": signed_exceedance_scale,
            "minimum_cluster_size": 2,
            "selected_split_index": split_index,
            "max_scaled_signed_exceedance_sse_reduction_fraction": reduction,
        },
    }
    return (
        [reestimate, noise, scope_item, mixture],
        "mixture" if mixture["gate_status"] == "BLOCKING" else None,
    )


def _evaluate_candidate(
    candidate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    source_metrics: Mapping[str, Any],
    tail_indices: set[int],
    tail_definition: Mapping[str, Any],
    scale: float,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_rows = _candidate_rows(rows, candidate)
    geometry = _geometry_from_candidate(candidate)
    metrics = _split_metrics(
        candidate_rows, geometry, scale, tail_indices=tail_indices
    )
    divergence = _probe_divergence(
        rows, candidate_rows, source, geometry, scale
    )
    delta_mae = (
        float(source_metrics["mean_normalized_center_error"])
        - float(metrics["mean_normalized_center_error"])
    )
    delta_coverage = (
        float(metrics["raw_boundary_coverage"])
        - float(source_metrics["raw_boundary_coverage"])
    )
    delta_tail = (
        float(metrics["source_tail_coverage"])
        - float(source_metrics["source_tail_coverage"])
    )
    radius_change = (
        float(metrics["mean_normalized_radius"])
        - float(source_metrics["mean_normalized_radius"])
    )
    radius_expansion = max(0.0, radius_change)
    radius_reduction = max(0.0, -radius_change)
    context_reduction = (
        float(candidate["certificate"].get("context_reduction_fraction", 0.0))
        if candidate["candidate_family"] == "interval_quotient"
        else 0.0
    )
    uniform_contraction = (
        1.0 - float(candidate["construction"].get("radius_multiplier", 1.0))
        if candidate["candidate_family"] == "interval_restrict"
        else 0.0
    )
    max_divergence = max(divergence.values())
    components = {
        "normalized_center_mae_gain": delta_mae,
        "raw_boundary_coverage_gain": delta_coverage,
        "source_tail_coverage_gain": delta_tail,
        "context_reduction_fraction": context_reduction,
        "uniform_contraction_fraction": uniform_contraction,
        "normalized_radius_reduction": radius_reduction,
        "max_probe_divergence": max_divergence,
        "normalized_radius_expansion": radius_expansion,
    }
    weights = contract["validation_selection_policy"]["score_weights"]
    score = (
        float(weights["normalized_center_mae_gain"]) * delta_mae
        + float(weights["raw_boundary_coverage_gain"]) * delta_coverage
        + float(weights["source_tail_coverage_gain"]) * delta_tail
        + float(weights["context_reduction_fraction"]) * context_reduction
        + float(weights["uniform_contraction_fraction"]) * uniform_contraction
        + float(weights["normalized_radius_reduction"]) * radius_reduction
        - float(weights["max_probe_divergence_penalty"]) * max_divergence
        - float(weights["normalized_radius_expansion_penalty"])
        * radius_expansion
    )
    if not math.isfinite(score):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "validation score is not finite"
        )
    thresholds = contract["thresholds"]
    common = {
        "minimum_raw_boundary_coverage": metrics["raw_boundary_coverage"]
        >= float(thresholds["validation_min_raw_boundary_coverage"]),
        "minimum_source_tail_coverage": metrics["source_tail_coverage"]
        >= float(thresholds["validation_min_source_tail_coverage"]),
        "maximum_normalized_center_mae_increase": -delta_mae
        <= float(thresholds["validation_max_normalized_center_mae_increase"]),
        "maximum_normalized_q1_divergence": divergence[
            "normalized_absolute_error_q_max_divergence"
        ]
        <= float(thresholds["validation_max_normalized_q1_divergence"]),
        "maximum_normalized_q2_divergence": divergence[
            "normalized_margin_q_max_divergence"
        ]
        <= float(thresholds["validation_max_normalized_q2_divergence"]),
        "minimum_validation_score": score
        >= float(thresholds["validation_min_score"]),
    }
    family = candidate["candidate_family"]
    if family == "interval_robustify":
        family_gates = {
            "strict_expansion": bool(
                candidate["certificate"]["strict_superset_verified"]
            ),
            "discovery_all_covered": bool(
                candidate["certificate"]["all_discovery_rows_enveloped"]
            ),
            "minimum_validation_coverage_or_tail_gain": max(
                delta_coverage, delta_tail
            )
            >= float(
                thresholds["expansion_min_validation_coverage_or_tail_gain"]
            ),
            "maximum_validation_normalized_radius_increase": radius_expansion
            <= float(
                thresholds["expansion_max_validation_normalized_radius_increase"]
            ),
        }
    elif family == "interval_restrict":
        family_gates = {
            "strict_subset": bool(candidate["certificate"]["strict_subset_verified"]),
            "zero_discovery_raw_boundary_violation_rate": candidate[
                "discovery_metrics"
            ]["boundary_violation_rate"]
            == float(
                thresholds["restriction_required_discovery_boundary_violation_rate"]
            ),
            "zero_validation_raw_boundary_violation_rate": metrics[
                "boundary_violation_rate"
            ]
            == float(
                thresholds["restriction_required_validation_boundary_violation_rate"]
            ),
        }
    else:
        family_gates = {
            "envelope_certificate": bool(
                candidate["certificate"]["envelope_certificate_verified"]
            ),
            "minimum_context_reduction": context_reduction
            >= float(thresholds["quotient_min_context_reduction_fraction"]),
            "validation_coverage_not_below_source": delta_coverage >= 0.0,
            "validation_tail_coverage_not_below_source": delta_tail >= 0.0,
        }
    gates = {"common": common, "family": family_gates}
    return {
        "source_tail_definition": _copy(tail_definition),
        "source_metrics": _copy(source_metrics),
        "candidate_metrics": metrics,
        "probe_divergence": divergence,
        "dimensionless_score_components": components,
        "validation_score": score,
        "validation_score_units": "dimensionless",
        "gates": gates,
        "all_gates_passed": (
            bool(candidate["discovery_admissible"])
            and all(common.values())
            and all(family_gates.values())
        ),
        "stress_data_used": False,
    }


def _validation_select(
    candidates: list[dict[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    scale: float,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], set[int], dict[str, Any], dict[str, Any]]:
    fraction = float(contract["thresholds"]["source_tail_fraction"])
    tail_indices, tail_definition = _source_tail(rows, source, scale, fraction)
    source_metrics = _split_metrics(
        rows, source, scale, tail_indices=tail_indices
    )
    for candidate in candidates:
        candidate["validation_evaluation"] = _evaluate_candidate(
            candidate,
            rows,
            source,
            source_metrics,
            tail_indices,
            tail_definition,
            scale,
            contract,
        )
    family_winners: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        viable = [
            item
            for item in candidates
            if item["candidate_family"] == family
            and item["validation_evaluation"]["all_gates_passed"]
        ]
        if viable:
            winner = sorted(
                viable,
                key=lambda item: (
                    -float(item["validation_evaluation"]["validation_score"]),
                    item["candidate_id"],
                ),
            )[0]
            family_winners.append(
                {
                    "candidate_family": family,
                    "candidate_id": winner["candidate_id"],
                    "validation_score": winner["validation_evaluation"][
                        "validation_score"
                    ],
                }
            )
    ranked = sorted(
        family_winners,
        key=lambda item: (-float(item["validation_score"]), item["candidate_id"]),
    )
    required = float(
        contract["thresholds"]["minimum_cross_family_validation_score_margin"]
    )
    if not ranked:
        status = "NO_VALIDATION_VIABLE_CANDIDATE"
        provisional = None
        runner_up = None
        margin = None
    elif len(ranked) > 1 and (
        float(ranked[0]["validation_score"])
        - float(ranked[1]["validation_score"])
    ) < required:
        status = "UNRESOLVED_CROSS_FAMILY_MARGIN"
        provisional = None
        runner_up = ranked[1]
        margin = float(ranked[0]["validation_score"]) - float(
            ranked[1]["validation_score"]
        )
    else:
        status = "UNIQUE_PROVISIONAL_WINNER"
        provisional = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        margin = (
            None
            if runner_up is None
            else float(provisional["validation_score"])
            - float(runner_up["validation_score"])
        )
    selection = {
        "status": status,
        "validation_score_units": "dimensionless",
        "best_candidate_by_family": ranked,
        "provisional_candidate_id": (
            None if provisional is None else provisional["candidate_id"]
        ),
        "provisional_candidate_family": (
            None if provisional is None else provisional["candidate_family"]
        ),
        "provisional_validation_score": (
            None if provisional is None else provisional["validation_score"]
        ),
        "runner_up_family": None if runner_up is None else runner_up["candidate_family"],
        "runner_up_validation_score": (
            None if runner_up is None else runner_up["validation_score"]
        ),
        "validation_score_margin": margin,
        "required_min_cross_family_margin": required,
        "stress_data_used": False,
    }
    return selection, tail_indices, tail_definition, source_metrics


def _stress_confirm(
    provisional: Mapping[str, Any] | None,
    rows: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    scale: float,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    if provisional is None:
        return (
            {
                "status": "NOT_PERFORMED_NO_PROVISIONAL_WINNER",
                "provisional_candidate_id": None,
                "provisional_candidate_family": None,
                "source_tail_definition": None,
                "source_stress_metrics": None,
                "candidate_stress_metrics": None,
                "probe_divergence": None,
                "gates": None,
                "all_gates_passed": None,
                "stress_score": None,
                "fallback_candidate_evaluated": False,
                "fallback_candidate_selected": False,
            },
            None,
            None,
        )
    tail_indices, tail_definition = _source_tail(
        rows,
        source,
        scale,
        float(contract["thresholds"]["source_tail_fraction"]),
    )
    source_metrics = _split_metrics(rows, source, scale, tail_indices=tail_indices)
    candidate_rows = _candidate_rows(rows, provisional)
    geometry = _geometry_from_candidate(provisional)
    candidate_metrics = _split_metrics(
        candidate_rows, geometry, scale, tail_indices=tail_indices
    )
    divergence = _probe_divergence(rows, candidate_rows, source, geometry, scale)
    delta_mae = (
        float(source_metrics["mean_normalized_center_error"])
        - float(candidate_metrics["mean_normalized_center_error"])
    )
    delta_coverage = (
        float(candidate_metrics["raw_boundary_coverage"])
        - float(source_metrics["raw_boundary_coverage"])
    )
    delta_tail = (
        float(candidate_metrics["source_tail_coverage"])
        - float(source_metrics["source_tail_coverage"])
    )
    radius_increase = max(
        0.0,
        float(candidate_metrics["mean_normalized_radius"])
        - float(source_metrics["mean_normalized_radius"]),
    )
    thresholds = contract["thresholds"]
    common = {
        "minimum_raw_boundary_coverage": candidate_metrics["raw_boundary_coverage"]
        >= float(thresholds["stress_min_raw_boundary_coverage"]),
        "minimum_source_tail_coverage": candidate_metrics["source_tail_coverage"]
        >= float(thresholds["stress_min_source_tail_coverage"]),
        "maximum_normalized_center_mae_increase": -delta_mae
        <= float(thresholds["stress_max_normalized_center_mae_increase"]),
        "maximum_normalized_q1_divergence": divergence[
            "normalized_absolute_error_q_max_divergence"
        ]
        <= float(thresholds["stress_max_normalized_q1_divergence"]),
        "maximum_normalized_q2_divergence": divergence[
            "normalized_margin_q_max_divergence"
        ]
        <= float(thresholds["stress_max_normalized_q2_divergence"]),
    }
    family = provisional["candidate_family"]
    if family == "interval_robustify":
        family_gates = {
            "minimum_stress_coverage_or_tail_gain": max(
                delta_coverage, delta_tail
            )
            >= float(thresholds["expansion_min_stress_coverage_or_tail_gain"]),
            "maximum_stress_normalized_radius_increase": radius_increase
            <= float(thresholds["expansion_max_stress_normalized_radius_increase"]),
        }
    elif family == "interval_restrict":
        family_gates = {
            "zero_stress_raw_boundary_violation_rate": candidate_metrics[
                "boundary_violation_rate"
            ]
            == float(thresholds["restriction_required_stress_boundary_violation_rate"])
        }
    else:
        family_gates = {
            "stress_coverage_not_below_source": delta_coverage >= 0.0,
            "stress_tail_coverage_not_below_source": delta_tail >= 0.0,
            "envelope_certificate": bool(
                provisional["certificate"]["envelope_certificate_verified"]
            ),
        }
    gates = {"common": common, "family": family_gates}
    passed = all(common.values()) and all(family_gates.values())
    report = {
        "status": (
            "PROVISIONAL_WINNER_STRESS_CONFIRMED"
            if passed
            else "PROVISIONAL_WINNER_FAILED_STRESS_CONFIRMATION"
        ),
        "provisional_candidate_id": provisional["candidate_id"],
        "provisional_candidate_family": family,
        "source_tail_definition": tail_definition,
        "source_stress_metrics": source_metrics,
        "candidate_stress_metrics": candidate_metrics,
        "probe_divergence": divergence,
        "gates": gates,
        "all_gates_passed": passed,
        "stress_score": None,
        "fallback_candidate_evaluated": False,
        "fallback_candidate_selected": False,
    }
    return report, source_metrics, tail_definition


def _collect_key_strings(value: Any, key_name: str) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == key_name and isinstance(item, str) and item:
                result.add(item)
            result.update(_collect_key_strings(item, key_name))
    elif isinstance(value, list):
        for item in value:
            result.update(_collect_key_strings(item, key_name))
    return result


def _contains_key(value: Any, key_name: str) -> bool:
    if isinstance(value, Mapping):
        return key_name in value or any(
            _contains_key(item, key_name) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(item, key_name) for item in value)
    return False


def _normalize_interval_input(input_value: Any) -> dict[str, Any]:
    payload = _copy(_mapping(input_value, "interval competition input"))
    _exact_keys(payload, INPUT_FIELDS, "interval competition input")
    if payload.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "interval competition input schema_version is not frozen V2"
        )
    _string(payload.get("competition_id"), "interval competition_id")
    source_adapter = _mapping(payload.get("source_adapter"), "input source_adapter")
    _exact_keys(source_adapter, SOURCE_ADAPTER_FIELDS, "input source_adapter")
    evaluator = _mapping(payload.get("evaluator"), "input evaluator")
    _exact_keys(evaluator, frozenset({"evaluator_epoch", "fixed_anchor"}), "input evaluator")
    for key in ("evaluator_epoch", "fixed_anchor"):
        if evaluator[key] is not None:
            _string(evaluator[key], f"input evaluator.{key}")
    if (evaluator["evaluator_epoch"] is None) != (evaluator["fixed_anchor"] is None):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "input evaluator epoch and fixed anchor must both be null or both be strings"
        )
    exclusions = _mapping(
        payload.get("prior_record_exclusion"), "input prior_record_exclusion"
    )
    _exact_keys(exclusions, frozenset(PRIOR_RECORD_KEYS), "input prior_record_exclusion")
    for key in PRIOR_RECORD_KEYS:
        _require_digest(exclusions[key], f"input prior_record_exclusion.{key}")
    evidence = _mapping(payload.get("evidence"), "input evidence")
    _exact_keys(evidence, frozenset(SPLITS), "input evidence")
    for split in SPLITS:
        if not isinstance(evidence[split], list):
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                f"input evidence.{split} must be a list"
            )
        for index, row_value in enumerate(evidence[split]):
            row = _mapping(row_value, f"input evidence.{split}[{index}]")
            _exact_keys(row, ROW_FIELDS, f"input evidence.{split}[{index}]")
            _string(row.get("observation_id"), "evidence observation_id")
            _string(row.get("evaluator_epoch"), "evidence evaluator_epoch")
            _string(row.get("fixed_anchor"), "evidence fixed_anchor")
            _string(row.get("scope_id"), "evidence scope_id")
            _mapping(row.get("context"), "evidence context")
            _finite(row.get("observed_value"), "evidence observed_value")
    return payload


def _validate_source_adapter_binding(
    payload: Mapping[str, Any],
    adapter_input: Mapping[str, Any],
    adapter_contract: Mapping[str, Any],
    adapter_report: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    source = payload["source_adapter"]
    if _digest(adapter_contract) != SOURCE_ADAPTER_CONTRACT_DIGEST:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "source adapter contract is not the pinned canonical adapter"
        )
    expected = {
        "adapter_contract_digest": _digest(adapter_contract),
        "adapter_report_digest": adapter_report.get("report_digest"),
        "adapter_input_digest": _digest(adapter_input),
        "adapter_id": adapter_report.get("adapter_id"),
        "adapter_disposition": adapter_report.get("disposition"),
        "recompetition_seed_digest": adapter_report.get("recompetition_seed_digest"),
        "recompetition_seed_id": (
            None
            if adapter_report.get("recompetition_seed") is None
            else adapter_report["recompetition_seed"].get("seed_id")
        ),
        "seed_theory_state_digest": (
            None
            if adapter_report.get("recompetition_seed") is None
            else adapter_report["recompetition_seed"].get("theory_state_digest")
        ),
    }
    if canonical_json_bytes(source) != canonical_json_bytes(expected):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "interval input source_adapter differs from the exact verified adapter report"
        )
    seed = adapter_report.get("recompetition_seed")
    return (None if seed is None else _copy(seed)), expected


def _coverage_and_freshness(
    payload: Mapping[str, Any],
    source: Mapping[str, Any],
    expected_epoch: str,
    old_ids: set[str],
    old_epochs: set[str],
) -> tuple[dict[str, Any], bool, bool]:
    evidence = payload["evidence"]
    evaluator = payload["evaluator"]
    expected_anchor = source["state"].get("fixed_anchor")
    anchor_exact = (
        evaluator.get("fixed_anchor") == expected_anchor
        and all(
            row["fixed_anchor"] == expected_anchor
            for split in SPLITS
            for row in evidence[split]
        )
    )
    epoch_comparable = (
        evaluator.get("evaluator_epoch") == expected_epoch
        and expected_epoch not in old_epochs
        and all(
            row["evaluator_epoch"] == expected_epoch
            for split in SPLITS
            for row in evidence[split]
        )
        and anchor_exact
    )
    exact = True
    all_ids = [
        str(row["observation_id"])
        for split in SPLITS
        for row in evidence[split]
    ]
    unique = len(all_ids) == len(set(all_ids))
    disjoint = not (set(all_ids) & old_ids)
    if not unique:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "new observation IDs must be globally unique across all three splits"
        )
    if not disjoint:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "new observation IDs reuse one of the five prior generations"
        )
    coverage: dict[str, Any] = {}
    registered_cells = {
        (scope, canonical_json_bytes(context))
        for scope in source["scopes"]
        for context in source["contexts"]
    }
    for split in SPLITS:
        counts = {cell: 0 for cell in registered_cells}
        unregistered = 0
        for row in evidence[split]:
            cell = (str(row["scope_id"]), canonical_json_bytes(row["context"]))
            if cell not in registered_cells:
                unregistered += 1
            else:
                counts[cell] += 1
        required = EXACT_ROWS_PER_CELL[split]
        if unregistered:
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                f"{split} contains an unregistered context-scope cell"
            )
        split_exact = (
            unregistered == 0
            and all(count == required for count in counts.values())
            and len(evidence[split]) == required * len(registered_cells)
        )
        exact = exact and split_exact
        coverage[split] = {
            "registered_cell_count": len(registered_cells),
            "required_rows_per_cell": required,
            "expected_row_count": required * len(registered_cells),
            "actual_row_count": len(evidence[split]),
            "unregistered_row_count": unregistered,
            "minimum_registered_cell_row_count": min(counts.values()),
            "maximum_registered_cell_row_count": max(counts.values()),
            "complete_exact_cartesian_coverage": split_exact,
        }
    coverage["global_observation_id_unique"] = unique
    coverage["new_observation_ids_disjoint_from_five_prior_generations"] = disjoint
    coverage["derived_evaluator_epoch_exact"] = epoch_comparable
    coverage["fixed_anchor_exact"] = anchor_exact
    coverage["all_exact"] = exact and epoch_comparable
    return coverage, exact, epoch_comparable


def _empty_selection() -> dict[str, Any]:
    return {
        "status": "NOT_PERFORMED",
        "validation_score_units": "dimensionless",
        "best_candidate_by_family": [],
        "provisional_candidate_id": None,
        "provisional_candidate_family": None,
        "provisional_validation_score": None,
        "runner_up_family": None,
        "runner_up_validation_score": None,
        "validation_score_margin": None,
        "required_min_cross_family_margin": None,
        "stress_data_used": False,
    }


def _empty_stress(status: str = "NOT_PERFORMED_NO_PROVISIONAL_WINNER") -> dict[str, Any]:
    return {
        "status": status,
        "provisional_candidate_id": None,
        "provisional_candidate_family": None,
        "source_tail_definition": None,
        "source_stress_metrics": None,
        "candidate_stress_metrics": None,
        "probe_divergence": None,
        "gates": None,
        "all_gates_passed": None,
        "stress_score": None,
        "fallback_candidate_evaluated": False,
        "fallback_candidate_selected": False,
    }


def _diagnostic_tail(
    first_four: Sequence[Mapping[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    if mode == "no_seed":
        simplify = robustify = "NOT_EVALUATED_NO_SEED"
        new_probe = "DEFERRED_NO_SEED"
        language = "DEFERRED_NO_SEED"
    elif mode == "epoch":
        simplify = robustify = "NOT_EVALUATED_INCOMPARABLE_EPOCH"
        new_probe = "REQUIRED_EVALUATOR_REBIND"
        language = "DEFERRED_INCOMPARABLE_EPOCH"
    elif mode == "evidence":
        simplify = robustify = "NOT_EVALUATED_INEXACT_EVIDENCE"
        new_probe = "REQUIRED_EXACT_FRESH_EVIDENCE"
        language = "DEFERRED_INEXACT_EVIDENCE"
    elif mode == "diagnostic":
        simplify = robustify = "BLOCKED_BY_EARLY_DIAGNOSTIC"
        new_probe = "REQUIRED_DIAGNOSTIC_RESOLUTION"
        language = "DEFERRED_DIAGNOSTIC_UNRESOLVED"
    else:
        simplify = robustify = "VALIDATION_EVALUATED"
        new_probe = mode
        language = (
            "DEFERRED_OPERATION_CONFIRMED"
            if mode == "NOT_REQUIRED_CONFIRMED_OPERATION"
            else "DEFERRED_AFTER_NEW_PROBE"
        )
    return [
        *_copy(list(first_four)),
        {
            "stage": "simplify",
            "metric_status": simplify,
            "gate_status": "CLEARED" if simplify == "VALIDATION_EVALUATED" else "NOT_EVALUATED",
            "metrics": None,
        },
        {
            "stage": "robustify",
            "metric_status": robustify,
            "gate_status": "CLEARED" if robustify == "VALIDATION_EVALUATED" else "NOT_EVALUATED",
            "metrics": None,
        },
        {
            "stage": "new_probe",
            "metric_status": new_probe,
            "gate_status": "DEFERRED",
            "metrics": None,
        },
        {
            "stage": "language_last",
            "metric_status": language,
            "gate_status": "DEFERRED",
            "metrics": None,
        },
    ]


def _finalize_report(body: Mapping[str, Any]) -> dict[str, Any]:
    report = {**_copy(body), "report_digest": _digest(body)}
    _exact_keys(report, REPORT_FIELDS, "interval competition report")
    return report


def run_shadow_interval_multi_q_theory_operation_competition(
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
    input_artifacts: Mapping[str, Any] | None = None,
) -> ShadowIntervalMultiQTheoryOperationCompetitionResult:
    """Run the frozen discovery/validation/stress shadow competition."""

    contract = validate_shadow_interval_multi_q_theory_operation_competition_contract(
        interval_competition_contract
    )
    contract_digest = _digest(contract)
    if contract_digest != _require_digest(
        expected_interval_competition_contract_digest,
        "expected_interval_competition_contract_digest",
    ):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "interval competition contract differs from independent expectation"
        )
    payload = _normalize_interval_input(interval_competition_input)
    input_digest = _digest(payload)
    if input_digest != _require_digest(
        expected_interval_competition_input_digest,
        "expected_interval_competition_input_digest",
    ):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "interval competition input differs from independent expectation"
        )
    if input_artifacts is not None and not isinstance(input_artifacts, Mapping):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "input_artifacts must be an object or null"
        )
    artifacts = None if input_artifacts is None else _copy(input_artifacts)
    if _contains_key(artifacts, "observed_value"):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "input_artifacts must not embed observed evidence values"
        )

    # The complete source chain is replayed through the adapter's public
    # verifier only.  We intentionally do not call an adapter private helper.
    try:
        adapter_receipt = verify_shadow_interval_multi_q_recompetition_adapter(
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
        )
    except ShadowIntervalMultiQRecompetitionAdapterValidationError as exc:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "source adapter public exact replay failed"
        ) from exc

    seed, source_adapter_binding = _validate_source_adapter_binding(
        payload, adapter_input, adapter_contract, adapter_report
    )
    expected_competition_id = (
        derive_shadow_interval_multi_q_theory_operation_competition_id(
            adapter_contract_digest=source_adapter_binding["adapter_contract_digest"],
            adapter_report_digest=source_adapter_binding["adapter_report_digest"],
            recompetition_seed_digest=source_adapter_binding[
                "recompetition_seed_digest"
            ],
            seed_theory_state_digest=source_adapter_binding[
                "seed_theory_state_digest"
            ],
            interval_competition_contract=contract,
        )
    )
    if payload["competition_id"] != expected_competition_id:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "interval competition_id is not canonically derived"
        )
    expected_exclusions = (
        adapter_input.get("prior_record_exclusion")
        if seed is None
        else seed.get("prior_record_exclusion")
    )
    if canonical_json_bytes(payload["prior_record_exclusion"]) != canonical_json_bytes(
        expected_exclusions
    ):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "prior_record_exclusion differs from the verified adapter lineage"
        )

    seed_summary = {
        "seed_emitted": seed is not None,
        "seed_kind": None if seed is None else seed.get("seed_kind"),
        "qualification_status": None
        if seed is None
        else seed.get("qualification_status"),
        "recompetition_seed_id": source_adapter_binding["recompetition_seed_id"],
        "recompetition_seed_digest": source_adapter_binding[
            "recompetition_seed_digest"
        ],
        "seed_theory_state_digest": source_adapter_binding[
            "seed_theory_state_digest"
        ],
        "model_kind": None
        if seed is None
        else seed.get("model_interface", {}).get("model_kind"),
        "probe_ids": None
        if seed is None
        else _copy(seed.get("model_interface", {}).get("probe_ids")),
        "source_state_mutated": False,
    }
    evidence_digests = {
        split: _evidence_digest(payload["evidence"][split]) for split in SPLITS
    }
    events = [
        _audit_event(
            0,
            "SOURCE_ADAPTER_PUBLIC_EXACT_REPLAY_VERIFIED",
            GENESIS_DIGEST,
            {
                "adapter_verification_status": adapter_receipt["status"],
                "adapter_report_digest": adapter_receipt["report_digest"],
                "seed_emitted": adapter_receipt["seed_emitted"],
            },
        )
    ]

    source: dict[str, Any] | None = None
    evaluator_binding: dict[str, Any]
    evidence_coverage: dict[str, Any]
    candidate_commitments: dict[str, Any] | None = None
    semantic_dedup: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    baseline_metrics: dict[str, Any] | None = None
    selection = _empty_selection()
    stress_confirmation = _empty_stress()
    selected: dict[str, Any] | None = None
    first_four: list[dict[str, Any]] = [
        {
            "stage": stage,
            "metric_status": "NOT_EVALUATED_NO_SEED",
            "gate_status": "NOT_EVALUATED",
            "metrics": None,
        }
        for stage in ("reestimate", "noise", "scope", "mixture")
    ]

    if seed is None:
        if payload["evaluator"] != {"evaluator_epoch": None, "fixed_anchor": None}:
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "null-seed route requires null evaluator epoch and anchor"
            )
        if any(payload["evidence"][split] for split in SPLITS):
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "null-seed route requires empty evidence"
            )
        evidence_coverage = {
            split: {
                "registered_cell_count": 0,
                "required_rows_per_cell": EXACT_ROWS_PER_CELL[split],
                "expected_row_count": 0,
                "actual_row_count": 0,
                "unregistered_row_count": 0,
                "minimum_registered_cell_row_count": None,
                "maximum_registered_cell_row_count": None,
                "complete_exact_cartesian_coverage": False,
            }
            for split in SPLITS
        }
        evidence_coverage.update(
            {
                "global_observation_id_unique": True,
                "new_observation_ids_disjoint_from_five_prior_generations": True,
                "derived_evaluator_epoch_exact": False,
                "fixed_anchor_exact": False,
                "all_exact": False,
            }
        )
        evaluator_binding = {
            "derived_evaluator_epoch": None,
            "supplied_evaluator_epoch": None,
            "fixed_anchor": None,
            "epoch_comparable": False,
            "new_epoch_created_locally": False,
        }
        if adapter_report["disposition"] == (
            "RECOMPETITION_ADAPTER_NEEDS_NEW_POST_RESTRICTION_EVIDENCE"
        ):
            disposition = BLOCKED_ADAPTER_EVIDENCE
        elif adapter_report["disposition"] == (
            "RECOMPETITION_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH"
        ):
            disposition = BLOCKED_ADAPTER_EPOCH
        else:
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "adapter emitted no seed for an unsupported disposition"
            )
        diagnostic_trace = _diagnostic_tail(first_four, mode="no_seed")
        next_probe_status = "DEFERRED_NO_SEED"
        language_status = "DEFERRED_NO_SEED"
    else:
        source = _model_geometry(seed["theory_state"])
        if source["state_digest"] != seed["theory_state_digest"]:
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "verified adapter seed theory-state digest is not exact"
            )
        seed_bytes_before = canonical_json_bytes(seed)
        expected_epoch = derive_shadow_interval_multi_q_theory_operation_competition_epoch(
            adapter_contract_digest=source_adapter_binding["adapter_contract_digest"],
            adapter_report_digest=source_adapter_binding["adapter_report_digest"],
            recompetition_seed_digest=source_adapter_binding[
                "recompetition_seed_digest"
            ],
            seed_theory_state_digest=source_adapter_binding[
                "seed_theory_state_digest"
            ],
            fixed_anchor=source["state"].get("fixed_anchor"),
            interval_competition_contract=contract,
        )
        prior_artifacts = [
            competition_input,
            competition_report,
            qualification_input,
            qualification_report,
            probe_input,
            probe_report,
            restriction_input,
            restriction_report,
            adjudication_input,
            adjudication_report,
        ]
        old_ids: set[str] = set()
        old_epochs: set[str] = set()
        for artifact in prior_artifacts:
            old_ids.update(_collect_key_strings(artifact, "observation_id"))
            old_epochs.update(_collect_key_strings(artifact, "evaluator_epoch"))
        coverage, exact_evidence, comparable = _coverage_and_freshness(
            payload, source, expected_epoch, old_ids, old_epochs
        )
        evidence_coverage = coverage
        evaluator_binding = {
            "derived_evaluator_epoch": expected_epoch,
            "supplied_evaluator_epoch": payload["evaluator"]["evaluator_epoch"],
            "fixed_anchor": payload["evaluator"]["fixed_anchor"],
            "epoch_comparable": comparable,
            "new_epoch_created_locally": True,
        }
        if not comparable:
            disposition = INCOMPARABLE_EPOCH
            first_four = [
                {
                    "stage": stage,
                    "metric_status": "NOT_EVALUATED_INCOMPARABLE_EPOCH",
                    "gate_status": "NOT_EVALUATED",
                    "metrics": None,
                }
                for stage in ("reestimate", "noise", "scope", "mixture")
            ]
            diagnostic_trace = _diagnostic_tail(first_four, mode="epoch")
            next_probe_status = "REQUIRED_EVALUATOR_REBIND"
            language_status = "DEFERRED_INCOMPARABLE_EPOCH"
        elif not exact_evidence:
            disposition = NEEDS_EVIDENCE
            first_four = [
                {
                    "stage": stage,
                    "metric_status": "NOT_EVALUATED_INEXACT_EVIDENCE",
                    "gate_status": "NOT_EVALUATED",
                    "metrics": None,
                }
                for stage in ("reestimate", "noise", "scope", "mixture")
            ]
            diagnostic_trace = _diagnostic_tail(first_four, mode="evidence")
            next_probe_status = "REQUIRED_EXACT_FRESH_EVIDENCE"
            language_status = "DEFERRED_INEXACT_EVIDENCE"
        else:
            scale = _prediction_scale(
                source, float(contract["thresholds"]["numeric_epsilon"])
            )
            baseline_metrics = {
                "prediction_scale": scale,
                "discovery": None,
                "validation": None,
                "validation_source_tail_definition": None,
                "stress_confirmation_baseline": None,
                "stress_source_tail_definition": None,
            }
            baseline_metrics["discovery"] = _split_metrics(
                payload["evidence"]["discovery"],
                source,
                scale,
                tail_indices=None,
            )
            first_four, blocker = _diagnostics(
                payload["evidence"]["discovery"], source, scale, contract
            )
            if blocker is not None:
                disposition = EARLY_UNRESOLVED
                diagnostic_trace = _diagnostic_tail(first_four, mode="diagnostic")
                next_probe_status = "REQUIRED_DIAGNOSTIC_RESOLUTION"
                language_status = "DEFERRED_DIAGNOSTIC_UNRESOLVED"
            else:
                synthesis = synthesize_shadow_interval_multi_q_theory_operation_candidates(
                    seed,
                    payload["evidence"]["discovery"],
                    payload["evaluator"],
                    contract,
                )
                candidates = synthesis["retained_candidates"]
                candidate_commitments = synthesis["candidate_commitments"]
                semantic_dedup = synthesis["candidate_semantic_deduplication"]
                (
                    selection,
                    validation_tail_indices,
                    validation_tail_definition,
                    source_validation_metrics,
                ) = _validation_select(
                    candidates,
                    payload["evidence"]["validation"],
                    source,
                    scale,
                    contract,
                )
                baseline_metrics["validation"] = source_validation_metrics
                baseline_metrics[
                    "validation_source_tail_definition"
                ] = validation_tail_definition
                provisional = next(
                    (
                        item
                        for item in candidates
                        if item["candidate_id"]
                        == selection["provisional_candidate_id"]
                    ),
                    None,
                )
                (
                    stress_confirmation,
                    source_stress_metrics,
                    stress_tail_definition,
                ) = _stress_confirm(
                    provisional,
                    payload["evidence"]["stress"],
                    source,
                    scale,
                    contract,
                )
                baseline_metrics["stress_confirmation_baseline"] = (
                    source_stress_metrics
                )
                baseline_metrics["stress_source_tail_definition"] = (
                    stress_tail_definition
                )
                if provisional is None:
                    disposition = NO_VALIDATION_WINNER
                    next_probe_status = "REQUIRED_AFTER_NO_VALIDATION_WINNER"
                    language_status = "DEFERRED_AFTER_NEW_PROBE"
                elif not stress_confirmation["all_gates_passed"]:
                    disposition = STRESS_FAILED
                    next_probe_status = "REQUIRED_AFTER_STRESS_NONCONFIRMATION"
                    language_status = "DEFERRED_AFTER_NEW_PROBE"
                else:
                    selected = _copy(provisional)
                    disposition = {
                        "interval_robustify": SELECT_EXPANSION,
                        "interval_restrict": SELECT_RESTRICTION,
                        "interval_quotient": SELECT_QUOTIENT,
                    }[provisional["candidate_family"]]
                    next_probe_status = "NOT_REQUIRED_CONFIRMED_OPERATION"
                    language_status = "DEFERRED_OPERATION_CONFIRMED"
                diagnostic_trace = _diagnostic_tail(
                    first_four, mode=next_probe_status
                )
        if canonical_json_bytes(seed) != seed_bytes_before:
            raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
                "source recompetition seed was mutated"
            )

    arrays = {
        family: [_copy(item) for item in candidates if item["candidate_family"] == family]
        for family in FAMILY_ORDER
    }
    selected_status = (
        contract["selection"]["selection_status"]
        if selected is not None
        else contract["selection"]["no_selection_status"]
    )
    selection_boundary = {
        "selection_status": selected_status,
        "selected_candidate_id": None if selected is None else selected["candidate_id"],
        "selected_candidate_family": None
        if selected is None
        else selected["candidate_family"],
        "candidate_materialized": False,
        "shadow_theory_state_created": False,
        "source_seed_mutated": False,
        "transition_authorized": False,
        "adoption_decided": False,
        "promotion_decided": False,
        "current_pointer_written": False,
        "language_expansion_executed": False,
    }
    lifecycle = {
        "prior_record_exclusion": _copy(payload["prior_record_exclusion"]),
        "five_prior_generations": {
            key: {
                "observation_id_digest": payload["prior_record_exclusion"][key],
                "role": "AUDIT_ONLY_SCORING_EXCLUDED",
                "eligible_for_v2_scoring": False,
            }
            for key in PRIOR_RECORD_KEYS
        },
        "new_evaluator_epoch": evaluator_binding["derived_evaluator_epoch"],
        "discovery_role": "DIAGNOSIS_AND_SYNTHESIS_ONLY",
        "validation_role": "GATE_AND_PROVISIONAL_RANKING_ONLY",
        "stress_role": "UNIQUE_PROVISIONAL_WINNER_CONFIRMATION_ONLY",
        "cross_epoch_pooling_allowed": False,
        "logical_selective_erasure_applied": True,
        "physical_erasure": "NOT_PERFORMED",
    }
    events.append(
        _audit_event(
            1,
            "V2_EVIDENCE_AND_CANDIDATE_BOUNDARY_RESOLVED",
            events[-1]["event_digest"],
            {
                "disposition": disposition,
                "evidence_exact": evidence_coverage["all_exact"],
                "raw_candidate_count": None
                if candidate_commitments is None
                else candidate_commitments["raw_candidate_count"],
                "selected_candidate_id": None
                if selected is None
                else selected["candidate_id"],
            },
        )
    )
    events.append(
        _audit_event(
            2,
            "SHADOW_PROPOSAL_RECORDED_WITHOUT_MATERIALIZATION",
            events[-1]["event_digest"],
            {
                "selection_status": selected_status,
                "fallback_candidate_evaluated": False,
                "candidate_materialized": False,
                "language_expansion_executed": False,
                "adoption_status": ADOPTION_STATUS,
                "current_status": CURRENT_STATUS,
            },
        )
    )
    evidence_binding = {
        "required_splits": _copy(SPLITS),
        "exact_rows_per_registered_cell": _copy(EXACT_ROWS_PER_CELL),
        "caller_supplied_static_rows_only": True,
        "new_observation_ids_required": True,
        "five_prior_generations_scoring_excluded": True,
        "cross_epoch_pooling_allowed": False,
        "validation_data_used_for_synthesis": False,
        "stress_data_used_for_synthesis_or_ranking": False,
    }
    authority_boundary = _copy(contract["authority_boundary"])
    authority_boundary.update(
        {
            "candidate_synthesis_performed": candidate_commitments is not None,
            "candidate_evaluation_performed": any(
                item["validation_evaluation"] is not None for item in candidates
            ),
            "validation_selection_performed": candidate_commitments is not None,
            "stress_confirmation_performed": stress_confirmation[
                "provisional_candidate_id"
            ]
            is not None,
        }
    )
    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "contract_digest": contract_digest,
        "competition_input_digest": input_digest,
        "competition_id": payload["competition_id"],
        "source_adapter": source_adapter_binding,
        "source_adapter_disposition": adapter_report["disposition"],
        "source_seed_summary": seed_summary,
        "evaluator_binding": evaluator_binding,
        "evidence_binding": evidence_binding,
        "evidence_digests": evidence_digests,
        "evidence_coverage": evidence_coverage,
        "candidate_family_registry": _copy(CANDIDATE_FAMILY_REGISTRY),
        "candidate_commitments": candidate_commitments,
        "candidate_semantic_deduplication": semantic_dedup,
        "diagnostic_trace": diagnostic_trace,
        "baseline_metrics": baseline_metrics,
        "interval_expansion_candidates": arrays["interval_robustify"],
        "uniform_restriction_candidates": arrays["interval_restrict"],
        "conservative_quotient_envelope_candidates": arrays["interval_quotient"],
        "validation_selection": selection,
        "stress_confirmation": stress_confirmation,
        "disposition": disposition,
        "selected_candidate": selected,
        "selection_boundary": selection_boundary,
        "next_probe_spec": {
            "status": next_probe_status,
            "probe_executed": False,
            "new_probe_id": None,
        },
        "language_last_route": {
            "status": language_status,
            "language_expansion_executed": False,
            "language_or_predicate_invented": False,
        },
        "record_lifecycle_extension": lifecycle,
        "authority_boundary": authority_boundary,
        "adoption_eligibility": ADOPTION_ELIGIBILITY,
        "adoption_status": ADOPTION_STATUS,
        "promotion_status": PROMOTION_STATUS,
        "current_status": CURRENT_STATUS,
        "nonclaims": _copy(contract["nonclaims"]),
        "input_artifacts": artifacts,
        "audit_events": events,
        "audit_head": events[-1]["event_digest"],
    }
    return ShadowIntervalMultiQTheoryOperationCompetitionResult(
        report=_finalize_report(body)
    )


def verify_shadow_interval_multi_q_theory_operation_competition(
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
) -> dict[str, Any]:
    """Exact-replay a V2 report against all independent digest anchors."""

    supplied = _copy(
        _mapping(interval_competition_report, "interval competition report")
    )
    _exact_keys(supplied, REPORT_FIELDS, "interval competition report")
    expected_report = _require_digest(
        expected_interval_competition_report_digest,
        "expected_interval_competition_report_digest",
    )
    expected_artifacts = (
        None
        if expected_interval_competition_input_artifacts is None
        else _copy(expected_interval_competition_input_artifacts)
    )
    if supplied.get("input_artifacts") != expected_artifacts:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "interval competition input artifacts differ from independent expectation"
        )
    fresh = run_shadow_interval_multi_q_theory_operation_competition(
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
        expected_interval_competition_input_digest=expected_interval_competition_input_digest,
        expected_interval_competition_contract_digest=expected_interval_competition_contract_digest,
        input_artifacts=expected_artifacts,
    ).to_dict()
    if fresh["report_digest"] != expected_report:
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "replayed interval competition report digest differs from expectation"
        )
    if canonical_json_bytes(supplied) != canonical_json_bytes(fresh):
        raise ShadowIntervalMultiQTheoryOperationCompetitionValidationError(
            "supplied interval competition report differs from exact replay"
        )
    selected = fresh["selected_candidate"]
    return {
        "status": "VERIFIED_" + fresh["disposition"],
        "disposition": fresh["disposition"],
        "report_digest": fresh["report_digest"],
        "contract_digest": fresh["contract_digest"],
        "competition_id": fresh["competition_id"],
        "source_adapter_report_digest": fresh["source_adapter"][
            "adapter_report_digest"
        ],
        "candidate_selected": selected is not None,
        "selected_candidate_id": None
        if selected is None
        else selected["candidate_id"],
        "selected_operation_kind": None
        if selected is None
        else selected["operation_kind"],
        "candidate_materialized": False,
        "fallback_candidate_evaluated": False,
        "adoption_status": fresh["adoption_status"],
        "promotion_status": fresh["promotion_status"],
        "current_status": fresh["current_status"],
    }
