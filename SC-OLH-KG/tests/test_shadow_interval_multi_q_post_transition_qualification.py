import ast
import copy
import hashlib
import importlib.util
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_shadow_interval_multi_q_theory_transition as transition_kat  # noqa: E402
import performance.shadow_interval_multi_q_post_transition_qualification as qualification_core  # noqa: E402

from performance.shadow_interval_multi_q_post_transition_qualification import (  # noqa: E402
    derive_shadow_interval_multi_q_post_transition_evaluator_epoch,
    derive_shadow_interval_multi_q_post_transition_qualification_id,
    qualify_shadow_interval_multi_q_post_transition,
    validate_shadow_interval_multi_q_post_transition_qualification_contract,
    verify_shadow_interval_multi_q_post_transition_qualification,
)
from performance.theory_operation_competition import (  # noqa: E402
    canonical_json_bytes,
)


COMPETITION_CONTRACT = transition_kat.COMPETITION_CONTRACT
TRANSITION_CONTRACT = transition_kat.TRANSITION_CONTRACT
QUALIFICATION_CONTRACT = transition_kat.QUALIFICATION_CONTRACT
REVIEW_CONTRACT = transition_kat.REVIEW_CONTRACT
PROBE_CONTRACT = transition_kat.PROBE_CONTRACT
RESTRICTION_CONTRACT = transition_kat.RESTRICTION_CONTRACT
ADJUDICATION_CONTRACT = transition_kat.ADJUDICATION_CONTRACT
ADAPTER_CONTRACT = transition_kat.ADAPTER_CONTRACT
INTERVAL_COMPETITION_CONTRACT = transition_kat.INTERVAL_COMPETITION_CONTRACT
INTERVAL_TRANSITION_CONTRACT = transition_kat.INTERVAL_TRANSITION_CONTRACT
POST_TRANSITION_QUALIFICATION_CONTRACT = (
    ROOT
    / "performance/manifests/shadow_interval_multi_q_post_transition_qualification_v1.json"
)
POST_TRANSITION_QUALIFICATION_CORE = (
    ROOT / "performance/shadow_interval_multi_q_post_transition_qualification.py"
)
POST_TRANSITION_QUALIFICATION_RUNNER = (
    ROOT / "runners/run_shadow_interval_multi_q_post_transition_qualification.py"
)
POST_TRANSITION_QUALIFICATION_DOC = (
    ROOT / "docs/shadow_interval_multi_q_post_transition_qualification_v1.md"
)

PREVIOUS_SLICE_FILES = (
    *transition_kat.PREVIOUS_SLICE_FILES,
    transition_kat.INTERVAL_TRANSITION_CORE,
    transition_kat.INTERVAL_TRANSITION_CONTRACT,
    transition_kat.INTERVAL_TRANSITION_RUNNER,
    ROOT / "tests/test_shadow_interval_multi_q_theory_transition.py",
    transition_kat.INTERVAL_TRANSITION_DOC,
)

FROZEN_CONTRACT_DIGEST = (
    "sha256:9a52bb7ea3f4ce0f0cb16c5a5a296d284a85f4bfbdb2ee671e8c865bb2d3d493"
)
EXCLUSION_POLICY = (
    "EXCLUDE_FIVE_PRIOR_GENERATIONS_AND_V2_DISCOVERY_VALIDATION_STRESS"
)
ALL_DISPOSITIONS = {
    "POST_TRANSITION_QUALIFICATION_NOT_APPLICABLE_NO_MATERIALIZED_CHILD",
    "POST_TRANSITION_QUALIFICATION_NEEDS_EXACT_FRESH_EVIDENCE",
    "POST_TRANSITION_QUALIFICATION_INCOMPARABLE_FRESH_EVALUATOR_EPOCH",
    "POST_TRANSITION_QUALIFICATION_FAILED_FRESH_HOLDOUT",
    "POST_TRANSITION_QUALIFICATION_FAILED_FRESH_STRESS_CONFIRMATION",
    "QUALIFIED_FRESH_POST_TRANSITION_EVALUATOR_EPOCH",
}
NO_CHILD_CASES = (
    "needs_evidence",
    "incomparable",
    "diagnostic",
    "no_winner",
    "stress_failed",
    "blocked_adapter_evidence",
    "blocked_adapter_epoch",
)

REPORT_KEYS = {
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
INPUT_KEYS = {
    "schema_version",
    "qualification_id",
    "source_transition",
    "evaluator",
    "source_evidence_exclusion",
    "evidence",
}
SOURCE_TRANSITION_INPUT_KEYS = {
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
ROW_KEYS = {
    "observation_id",
    "evaluator_epoch",
    "fixed_anchor",
    "scope_id",
    "context",
    "observed_value",
}
SOURCE_REPORT_KEYS = {
    "verification_status",
    "public_exact_replay_performed",
    "transition_contract_id",
    "transition_contract_digest",
    "transition_report_schema_version",
    "transition_report_digest",
    "source_interval_competition_contract_digest",
    "source_interval_competition_report_digest",
    "disposition",
    "transition_materialized",
    "operation_kind",
    "transition_kind",
    "selected_candidate_id",
    "selected_candidate_family",
    "parent_theory_state_digest",
    "child_theory_state_digest",
}
EVALUATOR_DEFINITION_KEYS = {
    "schema_version",
    "qualification_id",
    "source_interval_competition_contract_digest",
    "source_interval_competition_report_digest",
    "fixed_anchor",
    "fixed_probe_registry",
    "evaluator_epoch",
}
EVALUATOR_BINDING_KEYS = {
    "expected_evaluator_epoch",
    "supplied_evaluator_epoch",
    "expected_fixed_anchor",
    "supplied_fixed_anchor",
    "declared_epoch_exact",
    "all_rows_epoch_exact",
    "declared_fixed_anchor_exact",
    "all_rows_fixed_anchor_exact",
    "fresh_from_six_source_generations",
    "cross_epoch_pooling_allowed",
    "comparable",
}
EVIDENCE_BINDING_KEYS = {
    "required_splits",
    "exact_rows_per_parent_context_scope_cell",
    "coverage_domain",
    "evidence_digests",
    "coverage",
    "global_observation_id_unique",
    "fresh_ids_disjoint_from_six_source_generations",
    "source_rows_used_for_scoring",
    "holdout_role",
    "stress_role",
}
COVERAGE_CELL_KEYS = {
    "registered_cell_count",
    "required_rows_per_cell",
    "expected_row_count",
    "actual_row_count",
    "minimum_registered_cell_row_count",
    "maximum_registered_cell_row_count",
    "complete_exact_parent_cartesian_coverage",
}
ERASURE_RECEIPT_KEYS = {
    "policy",
    "five_prior_generation_observation_id_digests",
    "v2_competition_evidence_digests",
    "six_source_generations_scoring_excluded",
    "logical_selective_erasure_applied",
    "physical_erasure",
    "cross_epoch_pooling_allowed",
}
PROBE_RESULTS_KEYS = {"prediction_scale", "prediction_scale_units", "holdout", "stress"}
SPLIT_RESULT_KEYS = {
    "status",
    "source_tail_definition",
    "parent_metrics",
    "child_metrics",
    "probe_divergence",
    "dimensionless_score_components",
    "qualification_score",
    "qualification_score_units",
    "gates",
    "all_gates_passed",
}
QUALIFICATION_BINDING_KEYS = {
    "qualification_id",
    "unique_child_theory_state_digest",
    "evaluator_epoch",
    "disposition",
    "holdout_status",
    "stress_status",
    "holdout_evaluated",
    "stress_evaluated",
    "qualified",
    "candidate_reselection_performed",
    "reranking_performed",
    "fallback_candidate_evaluated",
    "fallback_candidate_selected",
    "theory_materialized_or_rematerialized",
}
DYNAMIC_AUTHORITY_KEYS = {
    "source_transition_public_exact_replay_performed",
    "qualification_applicable",
    "fresh_evaluator_definition_derived",
    "fresh_evidence_structure_validated",
    "holdout_qualification_performed",
    "stress_confirmation_performed",
    "qualification_succeeded",
    "candidate_synthesis_reselection_ranking_or_fallback_performed",
    "theory_materialization_or_rematerialization_performed",
    "probe_acquisition_or_environment_execution_performed",
    "adoption_eligibility_determined",
    "adoption_decided",
    "promotion_decided",
    "current_pointer_written",
    "language_or_predicate_invented",
    "parent_child_seed_or_ambient_state_written",
}


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest_value(value):
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_dict(result):
    return result.to_dict() if hasattr(result, "to_dict") else result


def _write(path, value):
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _source(case="restriction"):
    values = transition_kat._materialize(case)
    # Omit the V10 result wrapper while retaining its 26 inputs and report.
    return (*values[:26], values[-1])


def _source_transition_fields(source):
    report = source[26]
    return {
        "transition_contract_digest": _digest_value(source[25]),
        "transition_report_digest": report["report_digest"],
        "source_interval_competition_report_digest": source[24]["report_digest"],
        "disposition": report["disposition"],
        "operation_kind": report["operation_kind"],
        "transition_kind": report["transition_kind"],
        "selected_candidate_id": report["selected_candidate_id"],
        "selected_candidate_family": report["selected_candidate_family"],
        "parent_theory_state_digest": report["parent_theory_state_digest"],
        "child_theory_state_digest": report["child_theory_state_digest"],
    }


def _radius(geometry, scope, context):
    model = geometry["model_class"]
    grouping = model["radius_grouping"]
    if grouping == "global":
        group = {"global": "*"}
    elif grouping == "per_scope":
        group = {"scope_id": scope}
    else:
        group = {"context": context}
    matches = [
        item["radius"]
        for item in model["radii"]
        if canonical_json_bytes(item["group"]) == canonical_json_bytes(group)
    ]
    assert len(matches) == 1
    return matches[0]


def _center(geometry, context):
    matches = [
        item["value"]
        for item in geometry["model_class"]["center_predictions"]
        if canonical_json_bytes(item["context"]) == canonical_json_bytes(context)
    ]
    assert len(matches) == 1
    return matches[0]


def _fresh_value(report, scope, parent_context):
    parent = report["parent_theory_state"]
    child = report["child_theory_state"]
    parent_center = _center(parent, parent_context)
    if report["operation_kind"] == "quotient":
        child_features = child["object_space"]["feature_ids"]
        child_context = {
            feature: copy.deepcopy(parent_context[feature])
            for feature in child_features
        }
        child_center = _center(child, child_context)
        return parent_center / 2.0 + child_center / 2.0
    if report["operation_kind"] == "expand":
        parent_radius = _radius(parent, scope, parent_context)
        child_radius = _radius(child, scope, parent_context)
        if child_radius > parent_radius:
            return parent_center + parent_radius / 2.0 + child_radius / 2.0
    return parent_center


def _qualification_input(source, *, mutate=None):
    transition_report = source[26]
    contract = _load(POST_TRANSITION_QUALIFICATION_CONTRACT)
    source_transition = _source_transition_fields(source)
    if transition_report["child_theory_state"] is None:
        payload = {
            "schema_version": contract["input_schema_version"],
            "qualification_id": None,
            "source_transition": source_transition,
            "evaluator": None,
            "source_evidence_exclusion": None,
            "evidence": None,
        }
    else:
        qualification_id = (
            derive_shadow_interval_multi_q_post_transition_qualification_id(
                source_transition_contract_digest=_digest_value(source[25]),
                source_transition_report_digest=transition_report["report_digest"],
                parent_theory_state_digest=transition_report[
                    "parent_theory_state_digest"
                ],
                child_theory_state_digest=transition_report[
                    "child_theory_state_digest"
                ],
                operation_kind=transition_report["operation_kind"],
                transition_kind=transition_report["transition_kind"],
                post_transition_qualification_contract=contract,
            )
        )
        child = transition_report["child_theory_state"]
        evaluator_epoch = (
            derive_shadow_interval_multi_q_post_transition_evaluator_epoch(
                qualification_id=qualification_id,
                source_interval_competition_contract_digest=_digest_value(
                    source[23]
                ),
                source_interval_competition_report_digest=source[24][
                    "report_digest"
                ],
                fixed_anchor=child["fixed_anchor"],
                post_transition_qualification_contract=contract,
            )
        )
        parent = transition_report["parent_theory_state"]
        evidence = {"holdout": [], "stress": []}
        for split in evidence:
            for scope in parent["scope_ids"]:
                for index, context in enumerate(parent["object_space"]["contexts"]):
                    evidence[split].append(
                        {
                            "observation_id": (
                                f"v11-{split}-{scope}-{index:03d}"
                            ),
                            "evaluator_epoch": evaluator_epoch,
                            "fixed_anchor": child["fixed_anchor"],
                            "scope_id": scope,
                            "context": copy.deepcopy(context),
                            "observed_value": _fresh_value(
                                transition_report, scope, context
                            ),
                        }
                    )
        lifecycle = transition_report["record_lifecycle_extension"]
        payload = {
            "schema_version": contract["input_schema_version"],
            "qualification_id": qualification_id,
            "source_transition": source_transition,
            "evaluator": {
                "evaluator_epoch": evaluator_epoch,
                "fixed_anchor": child["fixed_anchor"],
            },
            "source_evidence_exclusion": {
                "policy": EXCLUSION_POLICY,
                "five_prior_generation_observation_id_digests": copy.deepcopy(
                    lifecycle["five_prior_generation_exclusion"]
                ),
                "v2_competition_evidence_digests": copy.deepcopy(
                    lifecycle["v2_competition_evidence_digests"]
                ),
            },
            "evidence": evidence,
        }
    if mutate is not None:
        mutate(payload, source)
    return payload, contract


def _kwargs(source, qualification_input, contract, *, input_artifacts=None):
    kwargs = transition_kat._kwargs(source[:25], source[25])
    kwargs.pop("input_artifacts")
    kwargs.update(
        {
            "expected_interval_transition_report_digest": source[26][
                "report_digest"
            ],
            "expected_interval_transition_input_artifacts": None,
            "expected_post_transition_qualification_input_digest": _digest_value(
                qualification_input
            ),
            "expected_post_transition_qualification_contract_digest": _digest_value(
                contract
            ),
            "input_artifacts": input_artifacts,
        }
    )
    return kwargs


def _qualify(case="restriction", *, source=None, mutate_input=None):
    source = source or _source(case)
    qualification_input, contract = _qualification_input(
        source, mutate=mutate_input
    )
    result = qualify_shadow_interval_multi_q_post_transition(
        *source,
        qualification_input,
        contract,
        **_kwargs(source, qualification_input, contract),
    )
    return (*source, qualification_input, contract, result, _as_dict(result))


def _verify(source, qualification_input, contract, report, **overrides):
    kwargs = _kwargs(source, qualification_input, contract)
    kwargs.pop("input_artifacts")
    kwargs.update(
        {
            "expected_post_transition_qualification_report_digest": report[
                "report_digest"
            ],
            "expected_post_transition_qualification_input_artifacts": None,
        }
    )
    kwargs.update(overrides)
    return verify_shadow_interval_multi_q_post_transition_qualification(
        *source, qualification_input, contract, report, **kwargs
    )


def _rehash(report):
    report["report_digest"] = _digest_value(
        {key: value for key, value in report.items() if key != "report_digest"}
    )
    return report


@pytest.fixture(scope="module")
def known_qualifications():
    return {
        case: _qualify(case)
        for case in ("expansion", "restriction", "quotient", *NO_CHILD_CASES)
    }


def test_public_builder_verifier_derive_and_result_surfaces_are_exact(
    known_qualifications,
):
    builder = inspect.signature(qualify_shadow_interval_multi_q_post_transition)
    verifier = inspect.signature(
        verify_shadow_interval_multi_q_post_transition_qualification
    )
    expected = [
        "competition_input",
        "competition_contract",
        "competition_report",
        "transition_contract",
        "transition_report",
        "qualification_input",
        "qualification_contract",
        "qualification_report",
        "review_contract",
        "review_report",
        "probe_input",
        "probe_contract",
        "probe_report",
        "restriction_input",
        "restriction_contract",
        "restriction_report",
        "adjudication_input",
        "adjudication_contract",
        "adjudication_report",
        "adapter_input",
        "adapter_contract",
        "adapter_report",
        "interval_competition_input",
        "interval_competition_contract",
        "interval_competition_report",
        "interval_transition_contract",
        "interval_transition_report",
        "post_transition_qualification_input",
        "post_transition_qualification_contract",
    ]
    builder_positional = [
        name
        for name, parameter in builder.parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    verifier_positional = [
        name
        for name, parameter in verifier.parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert builder_positional == expected
    assert verifier_positional == [*expected, "post_transition_qualification_report"]
    builder_anchors = [
        name
        for name in builder.parameters
        if name.startswith("expected_") and name.endswith("_digest")
    ]
    assert len(builder_anchors) == 28
    assert set(verifier.parameters) - set(builder.parameters) == {
        "post_transition_qualification_report",
        "expected_post_transition_qualification_report_digest",
        "expected_post_transition_qualification_input_artifacts",
    }

    id_parameters = inspect.signature(
        derive_shadow_interval_multi_q_post_transition_qualification_id
    ).parameters
    assert tuple(id_parameters) == (
        "source_transition_contract_digest",
        "source_transition_report_digest",
        "parent_theory_state_digest",
        "child_theory_state_digest",
        "operation_kind",
        "transition_kind",
        "post_transition_qualification_contract",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in id_parameters.values()
    )
    epoch_parameters = inspect.signature(
        derive_shadow_interval_multi_q_post_transition_evaluator_epoch
    ).parameters
    assert tuple(epoch_parameters) == (
        "qualification_id",
        "source_interval_competition_contract_digest",
        "source_interval_competition_report_digest",
        "fixed_anchor",
        "post_transition_qualification_contract",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in epoch_parameters.values()
    )

    result = known_qualifications["restriction"][-2]
    properties = {
        name
        for name, value in vars(type(result)).items()
        if isinstance(value, property)
    }
    assert properties == {"disposition", "report_digest", "qualified"}
    assert result.qualified is True
    assert callable(result.to_dict)
    assert canonical_json_bytes(result.to_dict()) == canonical_json_bytes(
        known_qualifications["restriction"][-1]
    )


@pytest.mark.parametrize(
    ("case", "family", "operation", "transition"),
    (
        ("expansion", "interval_robustify", "expand", "INTERVAL_EXPANSION"),
        (
            "restriction",
            "interval_restrict",
            "restrict",
            "UNIFORM_INTERVAL_RESTRICTION",
        ),
        (
            "quotient",
            "interval_quotient",
            "quotient",
            "CONSERVATIVE_INTERVAL_QUOTIENT_ENVELOPE",
        ),
    ),
)
def test_all_three_materialized_families_qualify_on_fresh_exact_rows(
    known_qualifications, case, family, operation, transition
):
    values = known_qualifications[case]
    source, qualification_input, contract, result, report = (
        values[:27],
        values[27],
        values[28],
        values[29],
        values[30],
    )
    source_report = source[26]
    assert set(qualification_input) == INPUT_KEYS
    assert set(qualification_input["source_transition"]) == (
        SOURCE_TRANSITION_INPUT_KEYS
    )
    assert set(qualification_input["evaluator"]) == {
        "evaluator_epoch",
        "fixed_anchor",
    }
    assert set(qualification_input["source_evidence_exclusion"]) == {
        "policy",
        "five_prior_generation_observation_id_digests",
        "v2_competition_evidence_digests",
    }
    assert set(qualification_input["evidence"]) == {"holdout", "stress"}
    assert all(
        set(row) == ROW_KEYS
        for split in ("holdout", "stress")
        for row in qualification_input["evidence"][split]
    )
    parent = source_report["parent_theory_state"]
    expected_rows = len(parent["scope_ids"]) * len(
        parent["object_space"]["contexts"]
    )
    assert all(
        len(qualification_input["evidence"][split]) == expected_rows
        for split in ("holdout", "stress")
    )
    assert set(report) == REPORT_KEYS
    assert report["disposition"] == (
        "QUALIFIED_FRESH_POST_TRANSITION_EVALUATOR_EPOCH"
    )
    assert report["operation_kind"] == operation
    assert report["transition_kind"] == transition
    assert report["source_transition"]["selected_candidate_family"] == family
    assert report["parent_theory_state_digest"] == source_report[
        "parent_theory_state_digest"
    ]
    assert report["child_theory_state_digest"] == source_report[
        "child_theory_state_digest"
    ]
    assert report["fixed_probe_registry"] == contract["fixed_probe_registry"]
    assert result.qualified is True
    assert report["adoption_eligibility"] == (
        "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED"
    )
    assert report["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert report["promotion_status"] == "NOT_PROMOTED"
    assert report["current_status"] == "NOT_CURRENT"
    assert _verify(source, qualification_input, contract, report)[
        "report_digest"
    ] == report["report_digest"]


def test_qualified_report_has_exact_nested_schema_and_authority_facts(
    known_qualifications,
):
    report = known_qualifications["restriction"][-1]
    contract = _load(POST_TRANSITION_QUALIFICATION_CONTRACT)
    assert set(report["source_transition"]) == SOURCE_REPORT_KEYS
    assert set(report["evaluator_definition"]) == EVALUATOR_DEFINITION_KEYS
    assert set(report["evaluator_binding"]) == EVALUATOR_BINDING_KEYS
    assert set(report["evidence_binding"]) == EVIDENCE_BINDING_KEYS
    coverage = report["evidence_binding"]["coverage"]
    assert set(coverage) == {"holdout", "stress", "all_exact"}
    assert set(coverage["holdout"]) == COVERAGE_CELL_KEYS
    assert set(coverage["stress"]) == COVERAGE_CELL_KEYS
    assert set(report["selective_erasure_receipt"]) == ERASURE_RECEIPT_KEYS
    assert set(report["probe_results"]) == PROBE_RESULTS_KEYS
    assert set(report["probe_results"]["holdout"]) == SPLIT_RESULT_KEYS
    assert set(report["probe_results"]["stress"]) == SPLIT_RESULT_KEYS
    assert set(report["qualification_binding"]) == QUALIFICATION_BINDING_KEYS
    assert set(report["authority_boundary"]) == (
        set(contract["authority_boundary"]) | DYNAMIC_AUTHORITY_KEYS
    )
    assert report["source_transition"]["public_exact_replay_performed"] is True
    assert report["evaluator_binding"]["fresh_from_six_source_generations"] is True
    assert report["evaluator_binding"]["cross_epoch_pooling_allowed"] is False
    assert report["evidence_binding"]["source_rows_used_for_scoring"] is False
    assert report["evidence_binding"]["global_observation_id_unique"] is True
    assert report["evidence_binding"][
        "fresh_ids_disjoint_from_six_source_generations"
    ] is True
    receipt = report["selective_erasure_receipt"]
    assert receipt["six_source_generations_scoring_excluded"] is True
    assert receipt["logical_selective_erasure_applied"] is True
    assert receipt["physical_erasure"] == "NOT_PERFORMED"
    assert receipt["cross_epoch_pooling_allowed"] is False
    authority = report["authority_boundary"]
    for key in (
        "source_transition_public_exact_replay_performed",
        "qualification_applicable",
        "fresh_evaluator_definition_derived",
        "fresh_evidence_structure_validated",
        "holdout_qualification_performed",
        "stress_confirmation_performed",
        "qualification_succeeded",
    ):
        assert authority[key] is True
    for key in (
        "candidate_synthesis_reselection_ranking_or_fallback_performed",
        "theory_materialization_or_rematerialization_performed",
        "probe_acquisition_or_environment_execution_performed",
        "adoption_eligibility_determined",
        "adoption_decided",
        "promotion_decided",
        "current_pointer_written",
        "language_or_predicate_invented",
        "parent_child_seed_or_ambient_state_written",
    ):
        assert authority[key] is False


@pytest.mark.parametrize("case", NO_CHILD_CASES)
def test_all_seven_nonmaterialized_routes_are_strict_not_applicable_and_null(
    known_qualifications, case
):
    values = known_qualifications[case]
    qualification_input, report = values[27], values[-1]
    assert qualification_input["qualification_id"] is None
    assert qualification_input["evaluator"] is None
    assert qualification_input["source_evidence_exclusion"] is None
    assert qualification_input["evidence"] is None
    assert report["disposition"] == (
        "POST_TRANSITION_QUALIFICATION_NOT_APPLICABLE_NO_MATERIALIZED_CHILD"
    )
    for key in (
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
        "qualification_binding",
    ):
        assert report[key] is None
    assert report["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert report["current_status"] == "NOT_CURRENT"
    assert values[-2].qualified is False
    assert set(report["source_transition"]) == SOURCE_REPORT_KEYS
    assert set(report["authority_boundary"]) == (
        set(_load(POST_TRANSITION_QUALIFICATION_CONTRACT)["authority_boundary"])
        | DYNAMIC_AUTHORITY_KEYS
    )
    dynamic = {
        key: report["authority_boundary"][key] for key in DYNAMIC_AUTHORITY_KEYS
    }
    assert dynamic["source_transition_public_exact_replay_performed"] is True
    assert all(
        value is False
        for key, value in dynamic.items()
        if key != "source_transition_public_exact_replay_performed"
    )


def _mutate_all_rows(payload, split, source, offset):
    del source
    for row in payload["evidence"][split]:
        row["observed_value"] = float(row["observed_value"]) + offset


def test_all_six_dispositions_and_strict_stage_precedence(known_qualifications):
    actual = {
        values[-1]["disposition"] for values in known_qualifications.values()
    }

    needs = _qualify(
        mutate_input=lambda payload, source: payload["evidence"]["holdout"].pop()
    )[-1]
    actual.add(needs["disposition"])
    assert needs["disposition"] == (
        "POST_TRANSITION_QUALIFICATION_NEEDS_EXACT_FRESH_EVIDENCE"
    )
    assert needs["probe_results"] is None

    incomparable = _qualify(
        mutate_input=lambda payload, source: payload["evaluator"].__setitem__(
            "evaluator_epoch", "wrong-fresh-evaluator-epoch"
        )
    )[-1]
    actual.add(incomparable["disposition"])
    assert incomparable["disposition"] == (
        "POST_TRANSITION_QUALIFICATION_INCOMPARABLE_FRESH_EVALUATOR_EPOCH"
    )
    assert incomparable["probe_results"] is None

    failed_holdout = _qualify(
        mutate_input=lambda payload, source: _mutate_all_rows(
            payload, "holdout", source, 100.0
        )
    )[-1]
    actual.add(failed_holdout["disposition"])
    assert failed_holdout["disposition"] == (
        "POST_TRANSITION_QUALIFICATION_FAILED_FRESH_HOLDOUT"
    )
    assert failed_holdout["probe_results"]["holdout"]["all_gates_passed"] is False
    stress_placeholder = failed_holdout["probe_results"]["stress"]
    assert set(stress_placeholder) == SPLIT_RESULT_KEYS
    assert stress_placeholder["status"] == "NOT_EVALUATED_HOLDOUT_FAILED"
    assert all(
        value is None
        for key, value in stress_placeholder.items()
        if key != "status"
    )

    failed_stress = _qualify(
        mutate_input=lambda payload, source: _mutate_all_rows(
            payload, "stress", source, 100.0
        )
    )[-1]
    actual.add(failed_stress["disposition"])
    assert failed_stress["disposition"] == (
        "POST_TRANSITION_QUALIFICATION_FAILED_FRESH_STRESS_CONFIRMATION"
    )
    assert failed_stress["probe_results"]["holdout"]["all_gates_passed"] is True
    assert failed_stress["probe_results"]["stress"]["all_gates_passed"] is False
    assert failed_stress["probe_results"]["stress"]["qualification_score"] is None
    assert failed_stress["probe_results"]["stress"][
        "dimensionless_score_components"
    ] is None

    incomparable_and_inexact = _qualify(
        mutate_input=lambda payload, source: (
            payload["evaluator"].__setitem__(
                "evaluator_epoch", "wrong-fresh-evaluator-epoch"
            ),
            payload["evidence"]["holdout"].pop(),
        )
    )[-1]
    assert incomparable_and_inexact["disposition"] == (
        "POST_TRANSITION_QUALIFICATION_INCOMPARABLE_FRESH_EVALUATOR_EPOCH"
    )

    failed_both = _qualify(
        mutate_input=lambda payload, source: (
            _mutate_all_rows(payload, "holdout", source, 100.0),
            _mutate_all_rows(payload, "stress", source, 100.0),
        )
    )[-1]
    assert failed_both["disposition"] == (
        "POST_TRANSITION_QUALIFICATION_FAILED_FRESH_HOLDOUT"
    )
    assert failed_both["probe_results"]["stress"]["status"] == (
        "NOT_EVALUATED_HOLDOUT_FAILED"
    )

    assert actual == ALL_DISPOSITIONS


def test_no_child_cannot_smuggle_nonnull_qualification_surfaces():
    with pytest.raises(ValueError, match="no-child input"):
        _qualify(
            "needs_evidence",
            mutate_input=lambda payload, source: payload.__setitem__(
                "evidence", {"holdout": [], "stress": []}
            ),
        )


def test_dynamic_authority_route_matrix_matches_only_phases_actually_performed():
    reports = {
        "na": _qualify("needs_evidence")[-1],
        "needs": _qualify(
            mutate_input=lambda payload, source: payload["evidence"][
                "holdout"
            ].pop()
        )[-1],
        "incomparable": _qualify(
            mutate_input=lambda payload, source: payload["evaluator"].__setitem__(
                "evaluator_epoch", "wrong"
            )
        )[-1],
        "holdout_failed": _qualify(
            mutate_input=lambda payload, source: _mutate_all_rows(
                payload, "holdout", source, 100.0
            )
        )[-1],
        "stress_failed": _qualify(
            mutate_input=lambda payload, source: _mutate_all_rows(
                payload, "stress", source, 100.0
            )
        )[-1],
        "qualified": _qualify()[-1],
    }
    expected = {
        "na": (False, False, False, False, False, False),
        "needs": (True, True, True, False, False, False),
        "incomparable": (True, True, True, False, False, False),
        "holdout_failed": (True, True, True, True, False, False),
        "stress_failed": (True, True, True, True, True, False),
        "qualified": (True, True, True, True, True, True),
    }
    phase_keys = (
        "qualification_applicable",
        "fresh_evaluator_definition_derived",
        "fresh_evidence_structure_validated",
        "holdout_qualification_performed",
        "stress_confirmation_performed",
        "qualification_succeeded",
    )
    forbidden_performed = (
        "candidate_synthesis_reselection_ranking_or_fallback_performed",
        "theory_materialization_or_rematerialization_performed",
        "probe_acquisition_or_environment_execution_performed",
        "adoption_eligibility_determined",
        "adoption_decided",
        "promotion_decided",
        "current_pointer_written",
        "language_or_predicate_invented",
        "parent_child_seed_or_ambient_state_written",
    )
    for route, report in reports.items():
        authority = report["authority_boundary"]
        assert authority["source_transition_public_exact_replay_performed"] is True
        assert tuple(authority[key] for key in phase_keys) == expected[route]
        assert all(authority[key] is False for key in forbidden_performed)


def _all_strings_for_key(value, key):
    found = []
    if isinstance(value, dict):
        for name, item in value.items():
            if name == key and isinstance(item, str):
                found.append(item)
            found.extend(_all_strings_for_key(item, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_all_strings_for_key(item, key))
    return found


def test_qualification_id_and_epoch_are_deterministic_and_fresh_from_six_generations():
    source = _source("restriction")
    payload, contract = _qualification_input(source)
    source_transition = payload["source_transition"]
    qualification_id = payload["qualification_id"]
    evaluator_epoch = payload["evaluator"]["evaluator_epoch"]
    assert qualification_id == (
        derive_shadow_interval_multi_q_post_transition_qualification_id(
            source_transition_contract_digest=source_transition[
                "transition_contract_digest"
            ],
            source_transition_report_digest=source_transition[
                "transition_report_digest"
            ],
            parent_theory_state_digest=source_transition[
                "parent_theory_state_digest"
            ],
            child_theory_state_digest=source_transition[
                "child_theory_state_digest"
            ],
            operation_kind=source_transition["operation_kind"],
            transition_kind=source_transition["transition_kind"],
            post_transition_qualification_contract=contract,
        )
    )
    assert evaluator_epoch == (
        derive_shadow_interval_multi_q_post_transition_evaluator_epoch(
            qualification_id=qualification_id,
            source_interval_competition_contract_digest=_digest_value(source[23]),
            source_interval_competition_report_digest=source[24]["report_digest"],
            fixed_anchor=payload["evaluator"]["fixed_anchor"],
            post_transition_qualification_contract=contract,
        )
    )
    old_ids = set(_all_strings_for_key(source, "observation_id"))
    old_epochs = set(_all_strings_for_key(source, "evaluator_epoch"))
    assert qualification_id not in old_ids | old_epochs
    assert evaluator_epoch not in old_ids | old_epochs

    id_arguments = {
        "source_transition_contract_digest": source_transition[
            "transition_contract_digest"
        ],
        "source_transition_report_digest": source_transition[
            "transition_report_digest"
        ],
        "parent_theory_state_digest": source_transition[
            "parent_theory_state_digest"
        ],
        "child_theory_state_digest": source_transition[
            "child_theory_state_digest"
        ],
        "operation_kind": source_transition["operation_kind"],
        "transition_kind": source_transition["transition_kind"],
        "post_transition_qualification_contract": contract,
    }
    for key in (
        "source_transition_report_digest",
        "parent_theory_state_digest",
        "child_theory_state_digest",
        "operation_kind",
        "transition_kind",
    ):
        changed = copy.deepcopy(id_arguments)
        changed[key] = (
            "sha256:" + "1" * 64
            if key.endswith("digest")
            else str(changed[key]) + "-changed"
        )
        assert (
            derive_shadow_interval_multi_q_post_transition_qualification_id(
                **changed
            )
            != qualification_id
        )

    changed_epoch = derive_shadow_interval_multi_q_post_transition_evaluator_epoch(
        qualification_id=qualification_id,
        source_interval_competition_contract_digest=_digest_value(source[23]),
        source_interval_competition_report_digest="sha256:" + "2" * 64,
        fixed_anchor=payload["evaluator"]["fixed_anchor"],
        post_transition_qualification_contract=contract,
    )
    assert changed_epoch != evaluator_epoch
    assert (
        derive_shadow_interval_multi_q_post_transition_evaluator_epoch(
            qualification_id=qualification_id + "-changed",
            source_interval_competition_contract_digest=_digest_value(source[23]),
            source_interval_competition_report_digest=source[24]["report_digest"],
            fixed_anchor=payload["evaluator"]["fixed_anchor"],
            post_transition_qualification_contract=contract,
        )
        != evaluator_epoch
    )
    assert (
        derive_shadow_interval_multi_q_post_transition_evaluator_epoch(
            qualification_id=qualification_id,
            source_interval_competition_contract_digest=_digest_value(source[23]),
            source_interval_competition_report_digest=source[24]["report_digest"],
            fixed_anchor=payload["evaluator"]["fixed_anchor"] + "-changed",
            post_transition_qualification_contract=contract,
        )
        != evaluator_epoch
    )


@pytest.mark.parametrize("source_index", (0, 5, 10, 13, 16, 22))
def test_fresh_ids_cannot_reuse_any_of_six_source_generations(source_index):
    source = _source("restriction")
    candidates = _all_strings_for_key(source[source_index], "observation_id")
    assert candidates, source_index

    def collide(payload, ignored_source):
        del ignored_source
        payload["evidence"]["holdout"][0]["observation_id"] = candidates[0]

    with pytest.raises(ValueError, match="six source generations"):
        _qualify(source=source, mutate_input=collide)


@pytest.mark.parametrize("split", ("holdout", "stress"))
@pytest.mark.parametrize("cardinality", ("missing", "extra"))
def test_only_registered_cell_cardinality_mismatch_routes_needs(split, cardinality):
    def mutate(payload, source):
        del source
        if cardinality == "missing":
            payload["evidence"][split].pop()
        else:
            row = copy.deepcopy(payload["evidence"][split][0])
            row["observation_id"] += "-extra"
            payload["evidence"][split].append(row)

    report = _qualify(mutate_input=mutate)[-1]
    assert report["disposition"] == (
        "POST_TRANSITION_QUALIFICATION_NEEDS_EXACT_FRESH_EVIDENCE"
    )
    assert report["probe_results"] is None


@pytest.mark.parametrize("kind", ("declared_epoch", "row_epoch", "anchor"))
def test_wrong_fresh_epoch_or_anchor_routes_incomparable(kind):
    def mutate(payload, source):
        del source
        if kind == "declared_epoch":
            payload["evaluator"]["evaluator_epoch"] += "-wrong"
        elif kind == "row_epoch":
            payload["evidence"]["stress"][0]["evaluator_epoch"] += "-wrong"
        else:
            payload["evidence"]["holdout"][0]["fixed_anchor"] += "-wrong"

    report = _qualify(mutate_input=mutate)[-1]
    assert report["disposition"] == (
        "POST_TRANSITION_QUALIFICATION_INCOMPARABLE_FRESH_EVALUATOR_EPOCH"
    )
    assert report["probe_results"] is None


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload, source: payload["evidence"]["stress"][0].__setitem__(
            "observation_id",
            payload["evidence"]["holdout"][0]["observation_id"],
        ),
        lambda payload, source: payload["evidence"]["holdout"][0].__setitem__(
            "scope_id", "unregistered-scope"
        ),
        lambda payload, source: payload["evidence"]["holdout"][0].__setitem__(
            "context", {"not": "a-parent-context"}
        ),
        lambda payload, source: payload["evidence"]["holdout"][0].__setitem__(
            "observed_value", math.inf
        ),
        lambda payload, source: payload["evidence"]["holdout"][0].__setitem__(
            "extra", True
        ),
    ),
)
def test_duplicate_unregistered_nonfinite_or_malformed_evidence_is_hard_error(mutate):
    with pytest.raises(ValueError):
        _qualify(mutate_input=mutate)


def test_exclusion_receipt_is_exact_v10_lifecycle_and_cannot_drift():
    source = _source("restriction")
    payload, contract = _qualification_input(source)
    lifecycle = source[26]["record_lifecycle_extension"]
    assert payload["source_evidence_exclusion"] == {
        "policy": EXCLUSION_POLICY,
        "five_prior_generation_observation_id_digests": lifecycle[
            "five_prior_generation_exclusion"
        ],
        "v2_competition_evidence_digests": lifecycle[
            "v2_competition_evidence_digests"
        ],
    }
    for key in (
        "policy",
        "five_prior_generation_observation_id_digests",
        "v2_competition_evidence_digests",
    ):
        changed = copy.deepcopy(payload)
        if key == "policy":
            changed["source_evidence_exclusion"][key] = "FORGED"
        else:
            first = next(iter(changed["source_evidence_exclusion"][key]))
            changed["source_evidence_exclusion"][key][first] = (
                "sha256:" + "0" * 64
            )
        with pytest.raises(ValueError):
            qualify_shadow_interval_multi_q_post_transition(
                *source,
                changed,
                contract,
                **_kwargs(source, changed, contract),
            )


def _toy_interval_geometry(*, center=2.0, radius=0.5, threshold=0.2):
    state = {
        "object_space": {"feature_ids": ["x"], "contexts": [{"x": 0}]},
        "model_class": {
            "kind": "finite_interval_table",
            "center_predictions": [{"context": {"x": 0}, "value": center}],
            "radius_grouping": "global",
            "radii": [{"group": {"global": "*"}, "radius": radius}],
        },
        "scope_ids": ["scope"],
        "probe_ids": [
            "absolute_error_point_prediction",
            "normalized_signed_interval_boundary_margin",
        ],
        "violation_functionals": [
            {"functional_id": "absolute_error", "threshold": threshold}
        ],
    }
    return qualification_core._geometry(state, "toy")


def _toy_row(observed, observation_id="row"):
    return {
        "observation_id": observation_id,
        "evaluator_epoch": "epoch",
        "fixed_anchor": "anchor",
        "scope_id": "scope",
        "context": {"x": 0},
        "observed_value": observed,
    }


def test_native_two_q_formula_raw_boundary_equality_and_nonfinite_subtraction():
    geometry = _toy_interval_geometry()
    scale = qualification_core._prediction_scale(geometry, 1e-12)
    assert scale == pytest.approx(2.2)
    values = qualification_core._row_values(
        _toy_row(2.75), geometry, scale, project_parent_context=False
    )
    assert values["absolute_error"] == 0.75
    assert values["q1_normalized_absolute_error"] == pytest.approx(0.75 / 2.2)
    assert values["q2_normalized_signed_boundary_margin"] == pytest.approx(
        -0.25 / 2.2
    )
    assert values["raw_boundary_violation"] is True
    assert values["raw_boundary_exceedance"] == 0.25

    boundary = qualification_core._row_values(
        _toy_row(2.5), geometry, scale, project_parent_context=False
    )
    assert boundary["raw_boundary_violation"] is False
    assert boundary["raw_boundary_exceedance"] == 0.0
    assert boundary["q2_normalized_signed_boundary_margin"] == 0.0

    extreme = _toy_interval_geometry(center=-1e308, radius=0.0, threshold=0.0)
    with pytest.raises(ValueError, match="absolute center error is not finite"):
        qualification_core._row_values(
            _toy_row(1e308), extreme, 1e308, project_parent_context=False
        )


def test_tail_uses_raw_source_units_includes_all_cutoff_ties_and_ignores_ids():
    geometry = _toy_interval_geometry(center=0.0, radius=0.0, threshold=1.0)
    values = [3.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    rows = [_toy_row(value, f"original-{index}") for index, value in enumerate(values)]
    indices, definition = qualification_core._source_tail(
        rows, geometry, 1.0, 0.25
    )
    assert definition["source_tail_k"] == 2
    assert definition["source_tail_cutoff"] == 2.0
    assert definition["cutoff_units"] == "source_prediction_units"
    assert definition["source_tail_row_count"] == 3
    assert {values[index] for index in indices} == {2.0, 3.0}

    permuted_values = list(reversed(values))
    permuted = [
        _toy_row(value, f"renamed-{index}")
        for index, value in enumerate(permuted_values)
    ]
    permuted_indices, permuted_definition = qualification_core._source_tail(
        permuted, geometry, 1.0, 0.25
    )
    assert permuted_definition == definition
    assert {permuted_values[index] for index in permuted_indices} == {2.0, 3.0}


def test_raw_min_subnormal_violation_survives_normalized_underflow_for_tail():
    tiny = math.ulp(0.0)
    geometry = _toy_interval_geometry(center=0.0, radius=0.0, threshold=1e308)
    rows = [_toy_row(tiny, "tiny"), *[_toy_row(0.0, f"zero-{i}") for i in range(7)]]
    scale = qualification_core._prediction_scale(geometry, 1e-12)
    tiny_values = qualification_core._row_values(
        rows[0], geometry, scale, project_parent_context=False
    )
    assert tiny_values["raw_boundary_violation"] is True
    assert tiny_values["raw_boundary_exceedance"] == tiny
    assert tiny_values["normalized_boundary_exceedance"] == 0.0
    indices, definition = qualification_core._source_tail(
        rows, geometry, scale, 0.125
    )
    assert indices == {0}
    assert definition["source_tail_cutoff"] == tiny


def test_stable_mean_extremes_are_finite_and_split_metrics_divide_only_once():
    assert qualification_core._mean([1e308, 1e308], "extreme mean") == 1e308
    geometry = _toy_interval_geometry(center=0.0, radius=0.0, threshold=1e308)
    rows = [_toy_row(1e308, "a"), _toy_row(-1e308, "b")]
    metrics, values = qualification_core._split_metrics(
        rows,
        geometry,
        1e308,
        {0, 1},
        project_parent_context=False,
    )
    assert metrics["mean_absolute_center_error"] == 1e308
    assert metrics["mean_normalized_center_error"] == 1.0
    assert metrics["mean_normalized_boundary_exceedance"] == 1.0
    assert all(math.isfinite(item["q1_normalized_absolute_error"]) for item in values)


@pytest.mark.parametrize("case", ("expansion", "restriction", "quotient"))
def test_holdout_score_recomputes_from_exact_dimensionless_components(
    known_qualifications, case
):
    report = known_qualifications[case][-1]
    holdout = report["probe_results"]["holdout"]
    stress = report["probe_results"]["stress"]
    components = holdout["dimensionless_score_components"]
    weights = _load(POST_TRANSITION_QUALIFICATION_CONTRACT)[
        "interval_scoring_policy"
    ]["score_weights"]
    expected = (
        weights["normalized_center_mae_gain"]
        * components["normalized_center_mae_gain"]
        + weights["raw_boundary_coverage_gain"]
        * components["raw_boundary_coverage_gain"]
        + weights["source_tail_coverage_gain"]
        * components["source_tail_coverage_gain"]
        + weights["context_reduction_fraction"]
        * components["context_reduction_fraction"]
        + weights["uniform_contraction_fraction"]
        * components["uniform_contraction_fraction"]
        + weights["normalized_radius_reduction"]
        * components["normalized_radius_reduction"]
        - weights["max_probe_divergence_penalty"]
        * components["max_probe_divergence"]
        - weights["normalized_radius_expansion_penalty"]
        * components["normalized_radius_expansion"]
    )
    assert holdout["qualification_score"] == pytest.approx(expected)
    assert holdout["qualification_score_units"] == "dimensionless"
    assert stress["qualification_score"] is None
    assert stress["qualification_score_units"] is None
    assert stress["dimensionless_score_components"] is None


def test_stress_failure_never_scores_reranks_or_falls_back():
    qualified = _qualify("restriction")[-1]
    failed = _qualify(
        "restriction",
        mutate_input=lambda payload, source: _mutate_all_rows(
            payload, "stress", source, 100.0
        ),
    )[-1]
    assert failed["probe_results"]["holdout"] == qualified["probe_results"][
        "holdout"
    ]
    stress = failed["probe_results"]["stress"]
    assert stress["qualification_score"] is None
    assert stress["dimensionless_score_components"] is None
    assert failed["source_transition"] == qualified["source_transition"]
    assert failed["qualification_id"] == qualified["qualification_id"]
    assert failed["qualification_binding"]["fallback_candidate_evaluated"] is False
    assert failed["qualification_binding"]["fallback_candidate_selected"] is False
    assert failed["qualification_binding"]["candidate_reselection_performed"] is False
    assert failed["qualification_binding"]["reranking_performed"] is False
    assert failed["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert failed["current_status"] == "NOT_CURRENT"


def test_all_twenty_eight_independent_digest_anchors_fail_closed():
    source = _source("restriction")
    qualification_input, contract = _qualification_input(source)
    kwargs = _kwargs(source, qualification_input, contract)
    anchors = [
        key
        for key in kwargs
        if key.startswith("expected_") and key.endswith("_digest")
    ]
    assert len(anchors) == 28
    for key in anchors:
        forged = copy.deepcopy(kwargs)
        forged[key] = "sha256:" + "0" * 64
        with pytest.raises(ValueError):
            qualify_shadow_interval_multi_q_post_transition(
                *source, qualification_input, contract, **forged
            )


def test_input_artifact_metadata_cannot_embed_observed_values():
    source = _source("restriction")
    qualification_input, contract = _qualification_input(source)
    with pytest.raises(ValueError):
        qualify_shadow_interval_multi_q_post_transition(
            *source,
            qualification_input,
            contract,
            **_kwargs(
                source,
                qualification_input,
                contract,
                input_artifacts={"forged": {"observed_value": 0.0}},
            ),
        )


def test_public_verifier_replays_and_rejects_rehashed_report_tampering():
    values = _qualify("restriction")
    source, qualification_input, contract, original = (
        values[:27],
        values[27],
        values[28],
        values[-1],
    )
    receipt = _verify(source, qualification_input, contract, original)
    assert receipt["report_digest"] == original["report_digest"]
    assert receipt["disposition"] == original["disposition"]
    with pytest.raises(ValueError):
        _verify(
            source,
            qualification_input,
            contract,
            original,
            expected_post_transition_qualification_input_artifacts={
                "forged": {"bytes": 1, "path": "/forged", "sha256": "0" * 64}
            },
        )
    mutations = (
        lambda report: report.__setitem__("operation_kind", "expand"),
        lambda report: report["evaluator_binding"].__setitem__(
            "comparable", False
        ),
        lambda report: report["evidence_binding"].__setitem__("all_exact", False),
        lambda report: report["probe_results"]["holdout"].__setitem__(
            "all_gates_passed", False
        ),
        lambda report: report["qualification_binding"].__setitem__(
            "qualification_status", "FORGED"
        ),
        lambda report: report["authority_boundary"].__setitem__(
            "current_pointer_written", True
        ),
        lambda report: report.__setitem__("adoption_status", "ADOPTED"),
        lambda report: report["audit_events"][0].__setitem__("event", "FORGED"),
    )
    for mutate in mutations:
        tampered = copy.deepcopy(original)
        mutate(tampered)
        _rehash(tampered)
        with pytest.raises(ValueError):
            _verify(
                source,
                qualification_input,
                contract,
                tampered,
                expected_post_transition_qualification_report_digest=tampered[
                    "report_digest"
                ],
            )


def test_rehashed_v10_transition_tampering_cannot_authorize_qualification():
    source = list(_source("restriction"))
    tampered = copy.deepcopy(source[26])
    tampered["evaluator_gate"]["fresh_post_transition_qualification_performed"] = (
        True
    )
    transition_kat._rehash(tampered)
    source[26] = tampered
    qualification_input, contract = _qualification_input(tuple(source))
    with pytest.raises(ValueError):
        qualify_shadow_interval_multi_q_post_transition(
            *source,
            qualification_input,
            contract,
            **_kwargs(tuple(source), qualification_input, contract),
        )


def test_contract_digest_registries_numeric_policies_and_authority_are_frozen():
    contract = _load(POST_TRANSITION_QUALIFICATION_CONTRACT)
    assert validate_shadow_interval_multi_q_post_transition_qualification_contract(
        contract
    ) == contract
    assert _digest_value(contract) == FROZEN_CONTRACT_DIGEST
    assert qualification_core.FROZEN_CONTRACT_DIGEST == FROZEN_CONTRACT_DIGEST
    assert contract["contract_id"] == (
        "shadow_interval_multi_q_post_transition_qualification_v1"
    )
    assert contract["source_interval_competition_contract_digest"] == (
        transition_kat.competition_kat.FROZEN_CONTRACT_DIGEST
    )
    assert contract["source_transition_contract_digest"] == (
        transition_kat.FROZEN_CONTRACT_DIGEST
    )
    assert contract["disposition_registry"] == qualification_core.DISPOSITION_REGISTRY
    assert len(contract["disposition_registry"]) == 6
    assert set(contract["disposition_registry"].values()) == ALL_DISPOSITIONS
    assert contract["fixed_probe_registry"] == [
        "absolute_error_point_prediction",
        "normalized_signed_interval_boundary_margin",
    ]
    assert contract["evidence_policy"]["source_evidence_exclusion_policy"] == (
        EXCLUSION_POLICY
    )
    assert contract["holdout_qualification_policy"][
        "stress_not_evaluated_status"
    ] == "NOT_EVALUATED_HOLDOUT_FAILED"
    assert contract["holdout_qualification_policy"][
        "stress_not_evaluated_surfaces"
    ] == "ALL_NULL_EXCEPT_STATUS"
    scoring = contract["interval_scoring_policy"]
    assert scoring["stable_mean_algorithm"] == (
        "sorted_finite_math_fsum_divide_count_then_max_abs_scaled_fallback_on_nonfinite_or_overflow"
    )
    assert scoring["raw_boundary_predicate"] == (
        "absolute_observed_minus_center_error_strictly_greater_than_radius"
    )
    assert scoring["source_tail_statistic"] == "raw_parent_boundary_exceedance"
    assert scoring["source_tail_cutoff_units"] == "source_prediction_units"
    assert scoring["score_units"] == "dimensionless"
    assert set(scoring["score_component_units"]) == {
        "normalized_center_mae_gain",
        "raw_boundary_coverage_gain",
        "source_tail_coverage_gain",
        "context_reduction_fraction",
        "uniform_contraction_fraction",
        "normalized_radius_reduction",
        "max_probe_divergence",
        "normalized_radius_expansion",
    }
    assert set(scoring["score_component_units"].values()) == {"dimensionless"}
    assert contract["stress_confirmation_policy"] == {
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
    authority = contract["authority_boundary"]
    for key in (
        "candidate_synthesis_reselection_ranking_or_fallback_authority",
        "theory_materialization_or_rematerialization_authority",
        "probe_acquisition_or_environment_execution_authority",
        "adoption_eligibility_authority",
        "adoption_authority",
        "promotion_authority",
        "current_pointer_authority",
        "language_expansion_authority",
        "parent_child_seed_or_ambient_write_authority",
    ):
        assert authority[key] is False


@pytest.mark.parametrize("route", tuple(qualification_core.DISPOSITION_REGISTRY))
def test_every_disposition_mapping_is_contract_frozen(route):
    contract = _load(POST_TRANSITION_QUALIFICATION_CONTRACT)
    contract["disposition_registry"][route] = "FORGED"
    with pytest.raises(ValueError):
        validate_shadow_interval_multi_q_post_transition_qualification_contract(
            contract
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda contract: contract["fixed_probe_registry"].append("third-q"),
        lambda contract: contract["evidence_policy"].__setitem__(
            "source_evidence_exclusion_policy", "FORGED"
        ),
        lambda contract: contract["holdout_qualification_policy"].__setitem__(
            "stress_not_evaluated_status", "FORGED"
        ),
        lambda contract: contract["holdout_qualification_policy"].__setitem__(
            "stress_not_evaluated_surfaces", "FORGED"
        ),
        lambda contract: contract["interval_scoring_policy"].__setitem__(
            "q1", "FORGED"
        ),
        lambda contract: contract["interval_scoring_policy"][
            "split_metric_definitions"
        ].__setitem__("mean_normalized_center_error", "FORGED"),
        lambda contract: contract["interval_scoring_policy"][
            "score_component_units"
        ].__setitem__("normalized_center_mae_gain", "source_prediction_units"),
        lambda contract: contract["stress_confirmation_policy"].__setitem__(
            "fallback_candidate_selected", True
        ),
        lambda contract: contract["authority_boundary"].__setitem__(
            "adoption_authority", True
        ),
        lambda contract: contract["selection"].__setitem__(
            "current_status", "CURRENT"
        ),
    ),
)
def test_contract_cannot_drift_formula_units_fallback_or_authority(mutate):
    contract = _load(POST_TRANSITION_QUALIFICATION_CONTRACT)
    mutate(contract)
    with pytest.raises(ValueError):
        validate_shadow_interval_multi_q_post_transition_qualification_contract(
            contract
        )


def import_runner():
    spec = importlib.util.spec_from_file_location(
        "interval_multi_q_post_transition_qualification_runner",
        POST_TRANSITION_QUALIFICATION_RUNNER,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


INPUT_CLI_NAMES = (
    "competition-input",
    "competition-contract",
    "competition-report",
    "transition-contract",
    "transition-report",
    "qualification-input",
    "qualification-contract",
    "qualification-report",
    "review-contract",
    "review-report",
    "probe-input",
    "probe-contract",
    "probe-report",
    "restriction-input",
    "restriction-contract",
    "restriction-report",
    "adjudication-input",
    "adjudication-contract",
    "adjudication-report",
    "adapter-input",
    "adapter-contract",
    "adapter-report",
    "interval-competition-input",
    "interval-competition-contract",
    "interval-competition-report",
    "interval-transition-contract",
    "interval-transition-report",
    "post-transition-qualification-input",
    "post-transition-qualification-contract",
)

DIGEST_CLI_NAMES = (
    "competition-contract",
    "competition-report",
    "transition-contract",
    "transition-report",
    "qualification-input",
    "qualification-contract",
    "qualification-report",
    "review-contract",
    "review-report",
    "probe-input",
    "probe-contract",
    "probe-report",
    "restriction-input",
    "restriction-contract",
    "restriction-report",
    "adjudication-input",
    "adjudication-contract",
    "adjudication-report",
    "adapter-input",
    "adapter-contract",
    "adapter-report",
    "interval-competition-input",
    "interval-competition-contract",
    "interval-competition-report",
    "interval-transition-contract",
    "interval-transition-report",
    "post-transition-qualification-input",
    "post-transition-qualification-contract",
)


def _input_flags(paths):
    assert len(paths) == len(INPUT_CLI_NAMES) == 29
    return list(
        itertools.chain.from_iterable(
            (f"--{name}", str(path))
            for name, path in zip(INPUT_CLI_NAMES, paths)
        )
    )


def _digest_flags(values):
    assert len(DIGEST_CLI_NAMES) == 28
    return list(
        itertools.chain.from_iterable(
            (f"--expected-{name}-digest", values[name.replace("-", "_")])
            for name in DIGEST_CLI_NAMES
        )
    )


def _qualification_cli_inputs(tmp_path):
    first_paths, values = transition_kat._materialize_cli_inputs(tmp_path)
    transition_report_path = (tmp_path / "input-26.json").resolve()
    completed = subprocess.run(
        [
            sys.executable,
            str(transition_kat.INTERVAL_TRANSITION_RUNNER),
            *transition_kat._input_flags(first_paths),
            *transition_kat._digest_flags(values),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    transition_report_path.write_bytes(completed.stdout)
    source_paths = (*first_paths, transition_report_path)
    source = tuple(_load(path) for path in source_paths)
    qualification_input, contract = _qualification_input(source)
    qualification_input_path = (tmp_path / "input-27.json").resolve()
    qualification_contract_path = (tmp_path / "input-28.json").resolve()
    _write(qualification_input_path, qualification_input)
    _write(qualification_contract_path, contract)
    paths = (*source_paths, qualification_input_path, qualification_contract_path)
    values = {
        **values,
        "interval_transition_report": source[26]["report_digest"],
        "post_transition_qualification_input": _digest_value(qualification_input),
        "post_transition_qualification_contract": _digest_value(contract),
    }
    return paths, values


def test_cli_canonical_stdout_atomic_out_exact_29_artifacts_and_no_input_write(
    tmp_path,
):
    paths, values = _qualification_cli_inputs(tmp_path)
    before = {path: path.read_bytes() for path in paths}
    out = (tmp_path / "post-transition-qualification-report.json").resolve()
    completed = subprocess.run(
        [
            sys.executable,
            str(POST_TRANSITION_QUALIFICATION_RUNNER),
            *_input_flags(paths),
            *_digest_flags(values),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    report = json.loads(completed.stdout)
    assert completed.stdout == canonical_json_bytes(report) + b"\n"
    assert out.read_bytes() == completed.stdout
    assert {path: path.read_bytes() for path in paths} == before
    artifacts = report["input_artifacts"]
    assert len(artifacts) == 29
    assert set(artifacts) == {
        name.replace("-", "_") + "_json" for name in INPUT_CLI_NAMES
    }
    assert all(
        set(receipt) == {"bytes", "path", "sha256"}
        for receipt in artifacts.values()
    )
    assert "observed_value" not in json.dumps(artifacts, sort_keys=True)

    out.write_bytes(b"sentinel\n")
    forged_values = dict(values)
    forged_values["post_transition_qualification_input"] = "sha256:" + "0" * 64
    failed = subprocess.run(
        [
            sys.executable,
            str(POST_TRANSITION_QUALIFICATION_RUNNER),
            *_input_flags(paths),
            *_digest_flags(forged_values),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert failed.returncode == 2
    assert failed.stdout == b""
    assert b"Traceback" not in failed.stderr
    assert out.read_bytes() == b"sentinel\n"


@pytest.mark.parametrize(
    "raw",
    (
        b'{"value":"a","value":"b"}\n',
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
        b'[]\n',
    ),
)
def test_cli_rejects_duplicate_nonfinite_or_nonobject_json_before_core(raw, tmp_path):
    paths = [(tmp_path / f"input-{index}.json").resolve() for index in range(29)]
    for path in paths:
        path.write_text("{}\n", encoding="utf-8")
    paths[0].write_bytes(raw)
    zero = "sha256:" + "0" * 64
    completed = subprocess.run(
        [
            sys.executable,
            str(POST_TRANSITION_QUALIFICATION_RUNNER),
            *_input_flags(paths),
            *_digest_flags(
                {name.replace("-", "_"): zero for name in DIGEST_CLI_NAMES}
            ),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert completed.stdout == b""
    assert b"Traceback" not in completed.stderr


def test_relative_symlink_and_output_aliases_are_rejected(tmp_path):
    paths = [(tmp_path / f"input-{index}.json").resolve() for index in range(29)]
    for path in paths:
        path.write_text("{}\n", encoding="utf-8")
    linked = (tmp_path / "linked.json").resolve()
    linked.symlink_to(paths[0])
    runner = import_runner()
    with pytest.raises(ValueError):
        runner._require_input_file(Path("relative.json"), "relative")
    with pytest.raises(ValueError):
        runner._require_input_file(linked, "symlink")
    with pytest.raises(ValueError):
        runner._protect_output(paths[0], tuple(paths))
    out = (tmp_path / "hardlinked-out.json").resolve()
    os.link(paths[0], out)
    with pytest.raises(ValueError):
        runner._protect_output(out, tuple(paths))


def test_all_406_input_path_and_hardlink_alias_pairs_are_rejected(tmp_path):
    runner = import_runner()
    base_paths = []
    for index in range(29):
        path = tmp_path / f"base-input-{index}.json"
        path.write_text("{}\n", encoding="utf-8")
        base_paths.append(path.resolve())
    checked_same = 0
    checked_hard = 0
    for first in range(29):
        for second in range(first + 1, 29):
            same = list(base_paths)
            same[second] = same[first]
            with pytest.raises(ValueError):
                runner._require_distinct_inputs(tuple(same))
            checked_same += 1
            alias = (tmp_path / f"hard-{first}-{second}.json").resolve()
            os.link(base_paths[first], alias)
            paths = list(base_paths)
            paths[second] = alias
            with pytest.raises(ValueError):
                runner._require_distinct_inputs(tuple(paths))
            checked_hard += 1
    assert checked_same == 406
    assert checked_hard == 406


def test_surface_has_no_execution_network_or_prior_file_mutation():
    before = {path: _digest_file(path) for path in PREVIOUS_SLICE_FILES}
    for path in (POST_TRANSITION_QUALIFICATION_CORE, POST_TRANSITION_QUALIFICATION_RUNNER):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        calls.update(
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        )
        assert imported.isdisjoint({"requests", "socket", "urllib"})
        assert calls.isdisjoint({"run_one", "urlopen", "connect"})
    assert {path: _digest_file(path) for path in before} == before


def test_documentation_freezes_additive_v11_boundary_and_next_gate():
    text = POST_TRANSITION_QUALIFICATION_DOC.read_text(encoding="utf-8")
    assert "strictly additive V1" in text
    assert transition_kat.competition_kat.FROZEN_CONTRACT_DIGEST in text
    assert transition_kat.FROZEN_CONTRACT_DIGEST in text
    assert FROZEN_CONTRACT_DIGEST in text
    assert "exactly twenty-nine positional inputs" in text
    assert "twenty-eight independent digest anchors" in text
    assert "thirty positional" in text
    assert "2/4/7/9/12/15/18/21/24/26/29" in text
    assert "406" in text
    assert "three materialized families" in text
    assert "seven V10" in text
    assert "six excluded generations" in text
    assert "source evidence" in text
    assert "raw boundary membership is authoritative" in text.lower()
    assert "source prediction units" in text
    assert "dimensionless" in text
    assert "NOT_EVALUATED_HOLDOUT_FAILED" in text
    assert "no fallback" in text
    assert "external probe acquisition" in text
    assert "logical selective erasure" in text
    assert "NOT_ADOPTED_SHADOW_ONLY" in text
    assert "NOT_CURRENT" in text
    assert "Operations Research" in text
    assert "not a complete autonomous theory-evolution loop" in text
    assert "separate future additive gate" in text
