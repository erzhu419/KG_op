import ast
import copy
import hashlib
import importlib.util
import inspect
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_shadow_interval_multi_q_theory_operation_competition as competition_kat  # noqa: E402
import performance.shadow_interval_multi_q_theory_transition as transition_core  # noqa: E402

from performance.shadow_interval_multi_q_theory_transition import (  # noqa: E402
    materialize_shadow_interval_multi_q_theory_transition,
    validate_shadow_interval_multi_q_theory_transition_contract,
    verify_shadow_interval_multi_q_theory_transition,
)
from performance.theory_operation_competition import (  # noqa: E402
    canonical_json_bytes,
)


COMPETITION_CONTRACT = competition_kat.COMPETITION_CONTRACT
TRANSITION_CONTRACT = competition_kat.TRANSITION_CONTRACT
QUALIFICATION_CONTRACT = competition_kat.QUALIFICATION_CONTRACT
REVIEW_CONTRACT = competition_kat.REVIEW_CONTRACT
PROBE_CONTRACT = competition_kat.PROBE_CONTRACT
RESTRICTION_CONTRACT = competition_kat.RESTRICTION_CONTRACT
ADJUDICATION_CONTRACT = competition_kat.ADJUDICATION_CONTRACT
ADAPTER_CONTRACT = competition_kat.ADAPTER_CONTRACT
INTERVAL_COMPETITION_CONTRACT = competition_kat.INTERVAL_COMPETITION_CONTRACT
INTERVAL_TRANSITION_CONTRACT = (
    ROOT
    / "performance/manifests/shadow_interval_multi_q_theory_transition_v1.json"
)
INTERVAL_TRANSITION_CORE = (
    ROOT / "performance/shadow_interval_multi_q_theory_transition.py"
)
INTERVAL_TRANSITION_RUNNER = (
    ROOT / "runners/run_shadow_interval_multi_q_theory_transition.py"
)
INTERVAL_TRANSITION_DOC = (
    ROOT / "docs/shadow_interval_multi_q_theory_transition_v1.md"
)

PREVIOUS_SLICE_FILES = (
    *competition_kat.PREVIOUS_SLICE_FILES,
    competition_kat.INTERVAL_COMPETITION_CORE,
    competition_kat.INTERVAL_COMPETITION_CONTRACT,
    competition_kat.INTERVAL_COMPETITION_RUNNER,
    ROOT / "tests/test_shadow_interval_multi_q_theory_operation_competition.py",
    competition_kat.INTERVAL_COMPETITION_DOC,
)

SOURCE_TO_TRANSITION = {
    "SELECT_SHADOW_INTERVAL_EXPANSION_CANDIDATE": (
        "MATERIALIZED_SHADOW_INTERVAL_EXPANSION"
    ),
    "SELECT_SHADOW_UNIFORM_RESTRICTION_CANDIDATE": (
        "MATERIALIZED_SHADOW_UNIFORM_INTERVAL_RESTRICTION"
    ),
    "SELECT_SHADOW_CONSERVATIVE_QUOTIENT_ENVELOPE_CANDIDATE": (
        "MATERIALIZED_SHADOW_CONSERVATIVE_QUOTIENT_ENVELOPE"
    ),
    "INTERVAL_MULTI_Q_COMPETITION_NEEDS_EXACT_FRESH_EVIDENCE": (
        "NOT_MATERIALIZED_NEEDS_EXACT_FRESH_EVIDENCE"
    ),
    "INTERVAL_MULTI_Q_COMPETITION_INCOMPARABLE_EVALUATOR_EPOCH": (
        "NOT_MATERIALIZED_INCOMPARABLE_EVALUATOR_EPOCH"
    ),
    "INTERVAL_MULTI_Q_COMPETITION_EARLY_DIAGNOSTIC_UNRESOLVED": (
        "NOT_MATERIALIZED_EARLY_DIAGNOSTIC_UNRESOLVED"
    ),
    "INTERVAL_MULTI_Q_COMPETITION_NO_VALIDATION_WINNER": (
        "NOT_MATERIALIZED_NO_VALIDATION_WINNER"
    ),
    "INTERVAL_MULTI_Q_COMPETITION_PROVISIONAL_WINNER_FAILED_STRESS_CONFIRMATION": (
        "NOT_MATERIALIZED_PROVISIONAL_WINNER_FAILED_STRESS_CONFIRMATION"
    ),
    "INTERVAL_MULTI_Q_COMPETITION_BLOCKED_ADAPTER_NEEDS_POST_RESTRICTION_EVIDENCE": (
        "NOT_MATERIALIZED_ADAPTER_NEEDS_POST_RESTRICTION_EVIDENCE"
    ),
    "INTERVAL_MULTI_Q_COMPETITION_BLOCKED_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH": (
        "NOT_MATERIALIZED_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH"
    ),
}

MATERIALIZED_DISPOSITIONS = {
    "MATERIALIZED_SHADOW_INTERVAL_EXPANSION",
    "MATERIALIZED_SHADOW_UNIFORM_INTERVAL_RESTRICTION",
    "MATERIALIZED_SHADOW_CONSERVATIVE_QUOTIENT_ENVELOPE",
}
NONMATERIALIZED_DISPOSITIONS = set(SOURCE_TO_TRANSITION.values()) - (
    MATERIALIZED_DISPOSITIONS
)
FROZEN_CONTRACT_DIGEST = (
    "sha256:b1a5f1761c2cafcae24f37f22810178074b9fc7800b6d73bdfd631be3b1df86d"
)

REPORT_KEYS = {
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

CHILD_KEYS = {
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

SOURCE_BINDING_KEYS = {
    "verification_status",
    "contract_id",
    "contract_digest",
    "report_schema_version",
    "report_digest",
    "competition_input_digest",
    "competition_id",
    "disposition",
    "source_adapter_report_digest",
    "recompetition_seed_id",
    "recompetition_seed_digest",
    "seed_theory_state_digest",
    "candidate_commitment_digest",
    "selection_status",
    "selected_candidate_id",
    "selected_candidate_family",
    "stress_confirmation_status",
    "candidate_materialized_by_source",
    "adoption_status",
    "promotion_status",
    "current_status",
}

SELECTED_BINDING_KEYS = {
    "candidate_id",
    "candidate_family",
    "operation_kind",
    "source_theory_state_digest",
    "model_class_digest",
    "semantic_model_digest",
    "candidate_commitment_digest",
    "validation_selection_status",
    "validation_score",
    "validation_score_units",
    "stress_confirmation_status",
    "stress_all_gates_passed",
    "selected_candidate_exact_array_member",
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


def _competition(case):
    if case == "restriction":
        return competition_kat._run()
    if case == "expansion":
        return competition_kat._run(residuals=competition_kat._expansion_residual)
    if case == "quotient":
        return competition_kat._run(residuals=competition_kat._quotient_residual)
    if case == "needs_evidence":
        return competition_kat._run(mutate_input=competition_kat._make_inexact)
    if case == "incomparable":
        return competition_kat._run(mutate_input=competition_kat._make_incomparable)
    if case == "diagnostic":
        return competition_kat._run(
            residuals=competition_kat._reestimate_block_residual
        )
    if case == "no_winner":
        return competition_kat._run(residuals=competition_kat._no_winner_residual)
    if case == "stress_failed":
        return competition_kat._run(residuals=competition_kat._stress_fail_residual)
    if case == "blocked_adapter_evidence":
        return competition_kat._run(
            mutate_adjudication_input=lambda payload, _: payload["evidence"][
                "holdout"
            ].pop()
        )
    if case == "blocked_adapter_epoch":
        return competition_kat._run(
            mutate_adjudication_input=lambda payload, _: payload["evidence"][
                "stress"
            ][0].__setitem__("evaluator_epoch", "other-epoch")
        )
    raise AssertionError(case)


def _source(case="restriction"):
    values = _competition(case)
    # Omit the V2 result wrapper while retaining all 24 inputs and its report.
    return (*values[:24], values[-1])


def _kwargs(source, contract, *, input_artifacts=None):
    kwargs = competition_kat._kwargs(source[:22], source[22], source[23])
    kwargs.pop("input_artifacts")
    kwargs.update(
        {
            "expected_interval_competition_report_digest": source[24][
                "report_digest"
            ],
            "expected_interval_competition_input_artifacts": None,
            "expected_interval_transition_contract_digest": _digest_value(contract),
            "input_artifacts": input_artifacts,
        }
    )
    return kwargs


def _materialize(case="restriction", *, source=None, input_artifacts=None):
    source = source or _source(case)
    contract = _load(INTERVAL_TRANSITION_CONTRACT)
    result = materialize_shadow_interval_multi_q_theory_transition(
        *source,
        contract,
        **_kwargs(source, contract, input_artifacts=input_artifacts),
    )
    return (*source, contract, result, _as_dict(result))


def _verify(source, contract, report, **overrides):
    kwargs = _kwargs(source, contract)
    kwargs.pop("input_artifacts")
    kwargs.update(
        {
            "expected_interval_transition_report_digest": report["report_digest"],
            "expected_interval_transition_input_artifacts": None,
        }
    )
    kwargs.update(overrides)
    return verify_shadow_interval_multi_q_theory_transition(
        *source, contract, report, **kwargs
    )


def _rehash(report):
    report["report_digest"] = _digest_value(
        {key: value for key, value in report.items() if key != "report_digest"}
    )
    return report


@pytest.fixture(scope="module")
def known_transitions():
    return {
        case: _materialize(case)
        for case in (
            "restriction",
            "expansion",
            "quotient",
            "needs_evidence",
            "incomparable",
            "diagnostic",
            "no_winner",
            "stress_failed",
            "blocked_adapter_evidence",
            "blocked_adapter_epoch",
        )
    }


def test_all_ten_source_dispositions_map_total_and_exact(known_transitions):
    actual = {
        values[24]["disposition"]: values[-1]["disposition"]
        for values in known_transitions.values()
    }
    assert actual == SOURCE_TO_TRANSITION
    assert set(actual.values()) == set(SOURCE_TO_TRANSITION.values())


def test_public_builder_and_verifier_have_exact_26_and_27_positional_inputs():
    builder = inspect.signature(
        materialize_shadow_interval_multi_q_theory_transition
    ).parameters
    verifier = inspect.signature(
        verify_shadow_interval_multi_q_theory_transition
    ).parameters
    builder_positional = [
        name
        for name, parameter in builder.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    verifier_positional = [
        name
        for name, parameter in verifier.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
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
    ]
    assert builder_positional == expected
    assert verifier_positional == [*expected, "interval_transition_report"]
    builder_digests = [
        name
        for name in builder
        if name.startswith("expected_") and name.endswith("_digest")
    ]
    assert len(builder_digests) == 25
    assert set(verifier) - set(builder) == {
        "interval_transition_report",
        "expected_interval_transition_report_digest",
        "expected_interval_transition_input_artifacts",
    }


@pytest.mark.parametrize(
    ("case", "family", "operation_kind", "transition_kind", "disposition"),
    (
        (
            "expansion",
            "interval_robustify",
            "expand",
            "INTERVAL_EXPANSION",
            "MATERIALIZED_SHADOW_INTERVAL_EXPANSION",
        ),
        (
            "restriction",
            "interval_restrict",
            "restrict",
            "UNIFORM_INTERVAL_RESTRICTION",
            "MATERIALIZED_SHADOW_UNIFORM_INTERVAL_RESTRICTION",
        ),
        (
            "quotient",
            "interval_quotient",
            "quotient",
            "CONSERVATIVE_INTERVAL_QUOTIENT_ENVELOPE",
            "MATERIALIZED_SHADOW_CONSERVATIVE_QUOTIENT_ENVELOPE",
        ),
    ),
)
def test_only_three_verified_selected_candidates_materialize_exact_geometry(
    known_transitions,
    case,
    family,
    operation_kind,
    transition_kind,
    disposition,
):
    report = known_transitions[case][-1]
    source_report = known_transitions[case][24]
    selected = source_report["selected_candidate"]
    child = report["child_theory_state"]
    assert set(report) == REPORT_KEYS
    assert report["disposition"] == disposition
    assert report["operation_kind"] == operation_kind
    assert report["transition_kind"] == transition_kind
    assert report["selected_candidate_id"] == selected["candidate_id"]
    assert selected["candidate_family"] == family
    assert child is not None
    assert set(child) == CHILD_KEYS
    assert child["object_space"] == selected["object_space"]
    assert child["model_class"] == selected["model_class"]
    assert child["scope_ids"] == selected["scope_ids"]
    assert child["removable_feature_ids"] == selected["removable_feature_ids"]
    assert child["probe_ids"] == selected["probe_ids"]
    assert child["violation_functionals"] == selected["violation_functionals"]
    assert "validation_evaluation" not in child
    assert set(report["source_interval_competition"]) == SOURCE_BINDING_KEYS
    assert set(report["selected_candidate_binding"]) == SELECTED_BINDING_KEYS
    assert report["selected_candidate_binding"]["validation_score_units"] == (
        "dimensionless"
    )


@pytest.mark.parametrize("case", ("expansion", "restriction", "quotient"))
def test_child_id_parent_snapshot_and_lineage_are_deterministic(
    known_transitions, case
):
    values = known_transitions[case]
    source_report, adapter_report, contract, report = (
        values[24],
        values[21],
        values[25],
        values[-1],
    )
    child = report["child_theory_state"]
    payload = {key: value for key, value in child.items() if key != "theory_id"}
    assert child["theory_id"] == (
        "shadow-interval-multi-q-theory:" + _digest_value(payload)[7:]
    )
    assert child["model_class_digest"] == _digest_value(child["model_class"])
    assert report["child_theory_state_digest"] == _digest_value(child)
    parent = adapter_report["recompetition_seed"]["theory_state"]
    assert report["parent_theory_state"] == parent
    assert report["parent_theory_state_digest"] == _digest_value(parent)
    lineage = child["transition_lineage"]
    assert lineage["parent_theory_state_digest"] == report[
        "parent_theory_state_digest"
    ]
    assert lineage["source_interval_competition_report_digest"] == source_report[
        "report_digest"
    ]
    assert lineage["selected_candidate_id"] == report["selected_candidate_id"]
    assert lineage["transition_contract_digest"] == _digest_value(contract)
    second = _materialize(source=values[:25])[-1]
    assert second["child_theory_state"] == child
    assert second["report_digest"] == report["report_digest"]


def test_three_operation_specific_preservation_certificates_are_strict(
    known_transitions,
):
    expansion = known_transitions["expansion"][-1]["preservation_certificate"]
    restriction = known_transitions["restriction"][-1][
        "preservation_certificate"
    ]
    quotient = known_transitions["quotient"][-1]["preservation_certificate"]
    for certificate in (expansion, restriction, quotient):
        assert certificate["certificate_kind"] == (
            "FINITE_INTERVAL_MULTI_Q_TRANSITION_PRESERVATION"
        )
        assert certificate["q_registry_byte_equal"] is True
        assert certificate["v_registry_byte_equal"] is True
        assert certificate["scope_ids_byte_equal"] is True
        assert certificate["finite_interval_table_verified"] is True
        assert certificate["verified"] is True
    assert expansion["family_certificate"] == {
        "source_radius_grouping_preserved": True,
        "centers_byte_equal": True,
        "all_child_intervals_contain_parent_intervals": True,
        "at_least_one_strictly_expanded": True,
        "checked_parent_context_scope_pair_count": 4,
    }
    assert restriction["family_certificate"] == {
        "source_radius_grouping_preserved": True,
        "centers_byte_equal": True,
        "all_child_intervals_within_parent_intervals": True,
        "at_least_one_strictly_restricted": True,
        "checked_parent_context_scope_pair_count": 4,
    }
    quotient_family = quotient["family_certificate"]
    assert quotient_family["quotient_radius_grouping"] == "per_context"
    assert quotient_family["quotient_context_keys_exact"] is True
    assert quotient_family[
        "all_parent_intervals_contained_under_quotient_map"
    ] is True
    assert quotient_family["point_prediction_preservation_claimed"] is False
    assert quotient_family["parent_snapshot_digest"] == known_transitions[
        "quotient"
    ][-1]["parent_theory_state_digest"]


def test_transition_quotient_rechecks_global_fiber_hull_across_all_scopes():
    source = competition_kat._source()
    seed = copy.deepcopy(source[21]["recompetition_seed"])
    state = seed["theory_state"]
    state["scope_ids"] = ["scope-a", "scope-b"]
    state["model_class"]["radius_grouping"] = "per_scope"
    state["model_class"]["radii"] = [
        {"group": {"scope_id": "scope-a"}, "radius": 1.0},
        {"group": {"scope_id": "scope-b"}, "radius": 3.0},
    ]
    seed["theory_state_digest"] = _digest_value(state)
    interval_contract = _load(INTERVAL_COMPETITION_CONTRACT)
    evaluator = {
        "evaluator_epoch": "transition-multi-scope-quotient-epoch",
        "fixed_anchor": state["fixed_anchor"],
    }
    discovery = competition_kat._rows(
        seed, evaluator["evaluator_epoch"]
    )["discovery"]
    synthesis = (
        competition_kat.synthesize_shadow_interval_multi_q_theory_operation_candidates(
            seed, discovery, evaluator, interval_contract
        )
    )
    candidate = synthesis["retained_candidates_by_family"]["interval_quotient"][0]
    parent_geometry = transition_core._geometry(state, "multi_scope_parent")
    child_geometry = transition_core._geometry(candidate, "multi_scope_child")
    family = transition_core._preserve_quotient(
        parent_geometry,
        _digest_value(state),
        candidate,
        child_geometry,
    )
    assert family["quotient_radius_grouping"] == "per_context"
    assert family["checked_parent_context_scope_pair_count"] == 8
    assert family["all_parent_intervals_contained_under_quotient_map"] is True
    for fiber in candidate["certificate"]["fiber_envelope_table"]:
        assert len(fiber["parent_intervals"]) == 4
        assert {item["scope_id"] for item in fiber["parent_intervals"]} == {
            "scope-a",
            "scope-b",
        }
        assert fiber["hull_radius"] == 8.0

    tampered = copy.deepcopy(candidate)
    tampered["construction"]["fiber_envelope_table"][0]["parent_intervals"].pop()
    tampered["certificate"]["fiber_envelope_table"][0]["parent_intervals"].pop()
    with pytest.raises(ValueError):
        transition_core._preserve_quotient(
            parent_geometry,
            _digest_value(state),
            tampered,
            transition_core._geometry(tampered, "tampered_multi_scope_child"),
        )


@pytest.mark.parametrize("case", ("expansion", "restriction", "quotient"))
def test_materialized_child_is_detached_and_requires_fresh_qualification(
    known_transitions, case
):
    report = known_transitions[case][-1]
    child = report["child_theory_state"]
    assert child["evaluator_epoch"] is None
    assert child["evaluator_status"] == (
        "UNASSIGNED_FRESH_POST_TRANSITION_EVALUATOR_REQUIRED"
    )
    assert child["operational_probe_status"] == (
        "FIXED_TWO_Q_FRESH_QUALIFICATION_REQUIRED"
    )
    reuse = child["evidence_reuse_policy"]
    assert reuse["prior_and_v2_records_role"] == (
        "AUDIT_ONLY_FUTURE_SCORING_EXCLUDED"
    )
    assert reuse["source_evidence_allowed_for_child_scoring"] is False
    assert reuse["old_new_records_pooled"] is False
    assert reuse["fresh_post_transition_evidence_required"] is True
    gate = report["evaluator_gate"]
    assert gate["child_evaluator_epoch"] is None
    assert gate["fresh_post_transition_qualification_required"] is True
    assert gate["fresh_post_transition_qualification_performed"] is False
    assert gate["adoption_blocked"] is True
    lifecycle = report["record_lifecycle_extension"]
    assert lifecycle["eligible_for_child_scoring"] is False
    assert lifecycle["cross_epoch_pooling_allowed"] is False
    assert lifecycle["logical_selective_erasure_applied"] is True
    assert lifecycle["physical_erasure"] == "NOT_PERFORMED"
    assert report["adoption_eligibility"] == (
        "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED"
    )
    assert report["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert report["promotion_status"] == "NOT_PROMOTED"
    assert report["current_status"] == "NOT_CURRENT"
    assert report["materialization_certificate"]["verified"] is True
    assert report["preservation_certificate"]["verified"] is True
    assert report["rollback_boundary"]["rollback_executed"] is False


def test_unqualified_repair_base_materializes_only_a_strict_nonidentity_child():
    values = competition_kat._run(mode="both_fail")
    source = (*values[:24], values[-1])
    report = _materialize(source=source)[-1]
    seed = source[21]["recompetition_seed"]
    child = report["child_theory_state"]
    assert seed["seed_kind"] == "UNQUALIFIED_SOURCE_REPAIR_BASE"
    assert report["disposition"] == (
        "MATERIALIZED_SHADOW_UNIFORM_INTERVAL_RESTRICTION"
    )
    assert child["semantic_model_digest"] != _digest_value(
        {
            "object_space": seed["theory_state"]["object_space"],
            "model_class": seed["theory_state"]["model_class"],
            "scope_ids": seed["theory_state"]["scope_ids"],
            "removable_feature_ids": seed["theory_state"][
                "removable_feature_ids"
            ],
            "probe_ids": seed["theory_state"]["probe_ids"],
            "violation_functionals": seed["theory_state"][
                "violation_functionals"
            ],
        }
    )
    family = report["preservation_certificate"]["family_certificate"]
    assert family["all_child_intervals_within_parent_intervals"] is True
    assert family["at_least_one_strictly_restricted"] is True
    assert report["evaluator_gate"]["adoption_blocked"] is True
    assert report["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert report["current_status"] == "NOT_CURRENT"


def test_all_routes_withhold_fresh_qualification_adoption_and_current_authority(
    known_transitions,
):
    for values in known_transitions.values():
        report = values[-1]
        assert report["adoption_eligibility"] == (
            "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED"
        )
        assert report["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
        assert report["promotion_status"] == "NOT_PROMOTED"
        assert report["current_status"] == "NOT_CURRENT"
        authority = report["authority_boundary"]
        assert authority["source_v2_public_exact_replay_performed"] is True
        assert authority["selected_candidate_materialization_performed"] is (
            report["disposition"] in MATERIALIZED_DISPOSITIONS
        )
        for key in (
            "fresh_qualification_authority",
            "adoption_eligibility_authority",
            "adoption_authority",
            "promotion_authority",
            "current_pointer_authority",
            "rollback_execution_authority",
            "probe_execution_authority",
            "language_expansion_authority",
            "parent_or_ambient_write_authority",
        ):
            assert authority[key] is False
        for key in (
            "fresh_post_transition_evaluator_created",
            "fresh_post_transition_qualification_performed",
            "source_seed_mutated",
            "rollback_executed",
            "probe_executed",
            "language_expansion_executed",
            "adoption_eligibility_determined",
            "adoption_decided",
            "promotion_decided",
            "current_pointer_written",
            "parent_or_ambient_state_written",
        ):
            assert authority[key] is False


@pytest.mark.parametrize(
    "case",
    (
        "needs_evidence",
        "incomparable",
        "diagnostic",
        "no_winner",
        "stress_failed",
        "blocked_adapter_evidence",
        "blocked_adapter_epoch",
    ),
)
def test_all_seven_nonselected_routes_materialize_no_child(
    known_transitions, case
):
    report = known_transitions[case][-1]
    assert report["disposition"] in NONMATERIALIZED_DISPOSITIONS
    for key in (
        "parent_theory_state",
        "parent_theory_state_digest",
        "operation_kind",
        "transition_kind",
        "selected_candidate_id",
        "child_theory_state",
        "child_theory_state_digest",
        "selected_candidate_family",
        "selected_candidate_binding",
        "materialization_certificate",
        "preservation_certificate",
        "rollback_boundary",
        "evaluator_gate",
    ):
        assert report[key] is None


def test_result_declares_only_frozen_public_properties(known_transitions):
    result = known_transitions["restriction"][-2]
    properties = {
        name
        for name, value in vars(type(result)).items()
        if isinstance(value, property)
    }
    assert properties == {"disposition", "report_digest"}
    assert callable(result.to_dict)
    assert canonical_json_bytes(result.to_dict()) == canonical_json_bytes(
        known_transitions["restriction"][-1]
    )


def test_all_twenty_five_independent_digest_anchors_fail_closed():
    source = _source()
    contract = _load(INTERVAL_TRANSITION_CONTRACT)
    kwargs = _kwargs(source, contract)
    keys = [
        key
        for key in kwargs
        if key.startswith("expected_") and key.endswith("_digest")
    ]
    assert len(keys) == 25
    for key in keys:
        forged = dict(kwargs)
        forged[key] = "sha256:" + "0" * 64
        with pytest.raises(ValueError):
            materialize_shadow_interval_multi_q_theory_transition(
                *source, contract, **forged
            )


def test_input_artifact_metadata_cannot_embed_observed_values():
    source = _source()
    contract = _load(INTERVAL_TRANSITION_CONTRACT)
    kwargs = _kwargs(
        source,
        contract,
        input_artifacts={"forged": {"observed_value": 0.0}},
    )
    with pytest.raises(ValueError):
        materialize_shadow_interval_multi_q_theory_transition(
            *source, contract, **kwargs
        )


def test_public_verifier_replays_and_rejects_rehashed_child_tampering(
    known_transitions,
):
    values = known_transitions["restriction"]
    source, contract, report = values[:25], values[25], values[-1]
    receipt = _verify(source, contract, report)
    assert receipt["status"].startswith("VERIFIED_MATERIALIZED_")
    assert receipt["report_digest"] == report["report_digest"]
    assert receipt["child_theory_state_digest"] == report[
        "child_theory_state_digest"
    ]

    tampered = copy.deepcopy(report)
    tampered["child_theory_state"]["evaluator_status"] = "FORGED"
    _rehash(tampered)
    with pytest.raises(ValueError):
        _verify(
            source,
            contract,
            tampered,
            expected_interval_transition_report_digest=tampered["report_digest"],
        )


def test_rehashed_transition_semantic_tampering_is_rejected(known_transitions):
    values = known_transitions["restriction"]
    source, contract, original = values[:25], values[25], values[-1]
    mutations = (
        lambda report: report["selected_candidate_binding"].__setitem__(
            "candidate_id", "forged-candidate"
        ),
        lambda report: report["materialization_certificate"].__setitem__(
            "verified", False
        ),
        lambda report: report["preservation_certificate"][
            "family_certificate"
        ].__setitem__("at_least_one_strictly_restricted", False),
        lambda report: report["rollback_boundary"].__setitem__(
            "rollback_executed", True
        ),
        lambda report: report["evaluator_gate"].__setitem__(
            "fresh_post_transition_qualification_performed", True
        ),
        lambda report: report["record_lifecycle_extension"].__setitem__(
            "eligible_for_child_scoring", True
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
                contract,
                tampered,
                expected_interval_transition_report_digest=(
                    tampered["report_digest"]
                ),
            )


def test_rehashed_source_v2_report_cannot_authorize_materialization():
    source = list(_source())
    tampered = copy.deepcopy(source[24])
    tampered["selection_boundary"]["candidate_materialized"] = True
    competition_kat._rehash(tampered)
    source[24] = tampered
    contract = _load(INTERVAL_TRANSITION_CONTRACT)
    with pytest.raises(ValueError):
        materialize_shadow_interval_multi_q_theory_transition(
            *source, contract, **_kwargs(tuple(source), contract)
        )


def import_runner():
    spec = importlib.util.spec_from_file_location(
        "interval_multi_q_transition_runner", INTERVAL_TRANSITION_RUNNER
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _input_flags(paths):
    names = (
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
    )
    assert len(paths) == len(names) == 26
    return list(
        itertools.chain.from_iterable(
            (f"--{name}", str(path)) for name, path in zip(names, paths)
        )
    )


def _digest_flags(values):
    names = (
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
    )
    assert len(names) == 25
    return list(
        itertools.chain.from_iterable(
            (f"--expected-{name}-digest", values[name.replace("-", "_")])
            for name in names
        )
    )


def _materialize_cli_inputs(tmp_path):
    first_paths, values = competition_kat._materialize_cli_inputs(tmp_path)
    paths = (*first_paths, (tmp_path / "input-24.json").resolve())
    completed = subprocess.run(
        [
            sys.executable,
            str(competition_kat.INTERVAL_COMPETITION_RUNNER),
            *competition_kat._input_flags(first_paths),
            *competition_kat._digest_flags(values),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    paths[24].write_bytes(completed.stdout)
    paths = (*paths, (tmp_path / "input-25.json").resolve())
    contract = _load(INTERVAL_TRANSITION_CONTRACT)
    _write(paths[25], contract)
    source_report = json.loads(completed.stdout)
    values = {
        **values,
        "interval_competition_report": source_report["report_digest"],
        "interval_transition_contract": _digest_value(contract),
    }
    return paths, values


def test_cli_canonical_stdout_atomic_out_exact_26_artifacts_and_no_input_write(
    tmp_path,
):
    paths, values = _materialize_cli_inputs(tmp_path)
    before = {path: path.read_bytes() for path in paths}
    out = (tmp_path / "interval-transition-report.json").resolve()
    completed = subprocess.run(
        [
            sys.executable,
            str(INTERVAL_TRANSITION_RUNNER),
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
    assert len(artifacts) == 26
    assert set(artifacts) == {
        "competition_input_json",
        "competition_contract_json",
        "competition_report_json",
        "transition_contract_json",
        "transition_report_json",
        "qualification_input_json",
        "qualification_contract_json",
        "qualification_report_json",
        "review_contract_json",
        "review_report_json",
        "probe_input_json",
        "probe_contract_json",
        "probe_report_json",
        "restriction_input_json",
        "restriction_contract_json",
        "restriction_report_json",
        "adjudication_input_json",
        "adjudication_contract_json",
        "adjudication_report_json",
        "adapter_input_json",
        "adapter_contract_json",
        "adapter_report_json",
        "interval_competition_input_json",
        "interval_competition_contract_json",
        "interval_competition_report_json",
        "interval_transition_contract_json",
    }
    assert all(
        set(receipt) == {"bytes", "path", "sha256"}
        for receipt in artifacts.values()
    )
    assert "observed_value" not in json.dumps(artifacts, sort_keys=True)


@pytest.mark.parametrize(
    "raw",
    (
        b'{"competition":"a","competition":"b"}\n',
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
        b'[]\n',
    ),
)
def test_cli_rejects_ambiguous_or_nonobject_json_before_core(raw, tmp_path):
    paths = [(tmp_path / f"input-{index}.json").resolve() for index in range(26)]
    for path in paths:
        path.write_text("{}\n", encoding="utf-8")
    paths[0].write_bytes(raw)
    zero = "sha256:" + "0" * 64
    digest_names = (
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
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(INTERVAL_TRANSITION_RUNNER),
            *_input_flags(paths),
            *_digest_flags({name: zero for name in digest_names}),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert completed.stdout == b""


def test_relative_symlink_and_output_aliases_are_rejected(tmp_path):
    paths = [(tmp_path / f"input-{index}.json").resolve() for index in range(26)]
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


def test_all_325_input_path_and_hardlink_alias_pairs_are_rejected(tmp_path):
    runner = import_runner()
    base_paths = []
    for index in range(26):
        path = tmp_path / f"base-input-{index}.json"
        path.write_text("{}\n", encoding="utf-8")
        base_paths.append(path.resolve())
    checked_same = 0
    checked_hard = 0
    for first in range(26):
        for second in range(first + 1, 26):
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
    assert checked_same == 325
    assert checked_hard == 325


def test_contract_registries_authority_and_next_gate_are_frozen():
    contract = _load(INTERVAL_TRANSITION_CONTRACT)
    assert validate_shadow_interval_multi_q_theory_transition_contract(contract) == (
        contract
    )
    assert _digest_value(contract) == FROZEN_CONTRACT_DIGEST
    assert transition_core.FROZEN_CONTRACT_DIGEST == FROZEN_CONTRACT_DIGEST
    assert contract["contract_id"] == "shadow_interval_multi_q_theory_transition_v1"
    assert contract["source_competition_contract_digest"] == (
        competition_kat.FROZEN_CONTRACT_DIGEST
    )
    assert contract["disposition_registry"] == transition_core.DISPOSITION_REGISTRY
    assert set(contract["disposition_registry"].values()) == set(
        SOURCE_TO_TRANSITION.values()
    )
    assert len(contract["disposition_registry"]) == 10
    assert contract["evaluator_epoch_policy"]["child_evaluator_epoch"] is None
    assert contract["evaluator_epoch_policy"]["child_evaluator_status"] == (
        "UNASSIGNED_FRESH_POST_TRANSITION_EVALUATOR_REQUIRED"
    )
    assert contract["evaluator_epoch_policy"]["operational_probe_status"] == (
        "FIXED_TWO_Q_FRESH_QUALIFICATION_REQUIRED"
    )
    assert contract["evaluator_epoch_policy"][
        "source_evidence_allowed_for_child_scoring"
    ] is False
    assert contract["record_lifecycle_policy"][
        "logical_selective_erasure_applied"
    ] is True
    assert contract["record_lifecycle_policy"]["physical_erasure"] == (
        "NOT_PERFORMED"
    )
    authority = contract["authority_boundary"]
    for key in (
        "fresh_qualification_authority",
        "adoption_eligibility_authority",
        "adoption_authority",
        "promotion_authority",
        "current_pointer_authority",
        "rollback_execution_authority",
        "probe_execution_authority",
        "language_expansion_authority",
        "parent_or_ambient_write_authority",
    ):
        assert authority[key] is False
    assert contract["selection"] == {
        "materialized_status": (
            "DETACHED_SHADOW_CHILD_MATERIALIZED_FRESH_QUALIFICATION_REQUIRED"
        ),
        "no_materialization_status": "NO_SHADOW_CHILD_MATERIALIZED",
        "adoption_eligibility": "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED",
        "adoption_status": "NOT_ADOPTED_SHADOW_ONLY",
        "promotion_status": "NOT_PROMOTED",
        "current_status": "NOT_CURRENT",
    }


@pytest.mark.parametrize("route", tuple(transition_core.DISPOSITION_REGISTRY))
def test_each_transition_disposition_route_is_contract_frozen(route):
    changed = _load(INTERVAL_TRANSITION_CONTRACT)
    changed["disposition_registry"][route] = "FORGED_DISPOSITION"
    with pytest.raises(ValueError):
        validate_shadow_interval_multi_q_theory_transition_contract(changed)


@pytest.mark.parametrize("index", range(3))
def test_each_selected_transition_registry_entry_is_contract_frozen(index):
    changed = _load(INTERVAL_TRANSITION_CONTRACT)
    changed["transition_registry"][index]["transition_kind"] = "FORGED_TRANSITION"
    with pytest.raises(ValueError):
        validate_shadow_interval_multi_q_theory_transition_contract(changed)


def test_contract_cannot_enable_reselection_evidence_reuse_or_authority():
    contract = _load(INTERVAL_TRANSITION_CONTRACT)
    mutations = (
        lambda value: value["materialization_policy"].__setitem__(
            "candidate_reranking_allowed", True
        ),
        lambda value: value["materialization_policy"].__setitem__(
            "fallback_candidate_allowed", True
        ),
        lambda value: value["evaluator_epoch_policy"].__setitem__(
            "source_evidence_allowed_for_child_scoring", True
        ),
        lambda value: value["authority_boundary"].__setitem__(
            "fresh_qualification_authority", True
        ),
        lambda value: value["authority_boundary"].__setitem__(
            "adoption_authority", True
        ),
        lambda value: value["authority_boundary"].__setitem__(
            "current_pointer_authority", True
        ),
    )
    for mutate in mutations:
        changed = copy.deepcopy(contract)
        mutate(changed)
        with pytest.raises(ValueError):
            validate_shadow_interval_multi_q_theory_transition_contract(changed)


def test_surface_has_no_execution_network_or_prior_file_mutation():
    before = {path: _digest_file(path) for path in PREVIOUS_SLICE_FILES}
    for path in (INTERVAL_TRANSITION_CORE, INTERVAL_TRANSITION_RUNNER):
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
        assert imported.isdisjoint({"requests", "socket", "subprocess", "urllib"})
        assert calls.isdisjoint({"run_one", "urlopen", "connect"})

    assert {path: _digest_file(path) for path in before} == before


def test_documentation_freezes_additive_transition_boundary():
    text = INTERVAL_TRANSITION_DOC.read_text(encoding="utf-8")
    assert "strictly additive V1" in text
    assert competition_kat.FROZEN_CONTRACT_DIGEST in text
    assert FROZEN_CONTRACT_DIGEST in text
    assert "exactly twenty-six" in text
    assert "twenty-five independent digest anchors" in text
    assert "twenty-seven positional" in text
    assert "325" in text
    assert "2/4/7/9/12/15/18/21/24/26" in text
    assert "three selected" in text
    assert "seven nonselected" in text
    assert "fresh post-transition qualification" in text
    assert "source evidence" in text
    assert "logical selective erasure" in text
    assert "NOT_ADOPTED_SHADOW_ONLY" in text
    assert "NOT_CURRENT" in text
    assert "Operations Research" in text
    assert "not a complete autonomous theory-evolution loop" in text
