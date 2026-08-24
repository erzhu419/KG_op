import ast
import copy
import hashlib
import importlib.util
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

import test_shadow_robust_interval_restriction as restriction_kat  # noqa: E402
import performance.shadow_post_restriction_adjudication as adjudication_core  # noqa: E402

from performance.shadow_post_restriction_adjudication import (  # noqa: E402
    derive_shadow_post_restriction_adjudication_epoch,
    qualify_adjudicate_and_route_shadow_post_restriction,
    validate_shadow_post_restriction_adjudication_contract,
    verify_shadow_post_restriction_adjudication,
)
from performance.theory_operation_competition import (  # noqa: E402
    canonical_json_bytes,
)


COMPETITION_CONTRACT = restriction_kat.COMPETITION_CONTRACT
TRANSITION_CONTRACT = restriction_kat.TRANSITION_CONTRACT
QUALIFICATION_CONTRACT = restriction_kat.QUALIFICATION_CONTRACT
REVIEW_CONTRACT = restriction_kat.REVIEW_CONTRACT
PROBE_CONTRACT = restriction_kat.PROBE_CONTRACT
RESTRICTION_CONTRACT = restriction_kat.RESTRICTION_CONTRACT
ADJUDICATION_CONTRACT = (
    ROOT / "performance/manifests/shadow_post_restriction_adjudication_v1.json"
)
ADJUDICATION_CORE = ROOT / "performance/shadow_post_restriction_adjudication.py"
ADJUDICATION_RUNNER = ROOT / "runners/run_shadow_post_restriction_adjudication.py"
ADJUDICATION_DOC = ROOT / "docs/shadow_post_restriction_adjudication_v1.md"

PREVIOUS_SLICE_FILES = (
    *restriction_kat.PREVIOUS_SLICE_FILES,
    restriction_kat.RESTRICTION_CORE,
    restriction_kat.RESTRICTION_CONTRACT,
    restriction_kat.RESTRICTION_RUNNER,
    restriction_kat.ROOT / "tests/test_shadow_robust_interval_restriction.py",
    restriction_kat.RESTRICTION_DOC,
)

ALL_DISPOSITIONS = {
    "POST_RESTRICTION_QUALIFIED_RETAIN_RESTRICTED_SHADOW",
    "POST_RESTRICTION_FAILED_SOURCE_SHADOW_ROLLBACK_TARGET_SELECTED",
    "POST_RESTRICTION_AND_SOURCE_FAILED_RECOMPETITION_ADAPTER_REQUIRED",
    "POST_RESTRICTION_NEEDS_NEW_EVIDENCE",
    "POST_RESTRICTION_INCOMPARABLE_EVALUATOR_EPOCH",
}

FROZEN_CONTRACT_DIGEST = (
    "sha256:dc870252207785f1d2ff4768dbf7d9fcedba7e0580554b0e647df82757d461ef"
)
FROZEN_RETAIN_REPORT_DIGEST = (
    "sha256:430327d32a9113ce891110079f47e0323e3eb08a0c104f5c5c766c841bec6c73"
)

REPORT_KEYS = {
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

SPLIT_METRIC_KEYS = {
    "row_count",
    "mean_absolute_center_error",
    "max_absolute_center_error",
    "min_normalized_signed_interval_boundary_margin",
    "boundary_violation_count",
    "boundary_violation_rate",
    "mean_normalized_exceedance",
    "max_normalized_exceedance",
    "counterexample_observation_ids",
}

SOURCE_RESTRICTION_KEYS = {
    "verification_status",
    "contract_id",
    "contract_digest",
    "report_digest",
    "restriction_id",
    "disposition",
    "source_probe_expanded_shadow_theory_state_digest",
    "restricted_shadow_theory_state_digest",
    "adoption_status",
}
EVALUATOR_DEFINITION_KEYS = {
    "evaluator_epoch",
    "fixed_anchor",
    "epoch_derivation_kind",
    "fixed_probe_registry_digest",
}
EVALUATOR_BINDING_KEYS = {
    "exact_derived_epoch_required",
    "expected_evaluator_epoch",
    "supplied_evaluator_epoch",
    "epoch_matches",
    "expected_fixed_anchor",
    "supplied_fixed_anchor",
    "fixed_anchor_matches",
    "comparable",
}
EVIDENCE_BINDING_KEYS = {
    "holdout_evidence_digest",
    "stress_evidence_digest",
    "new_observation_id_digest",
    "new_observation_count",
    "prior_observation_id_digests",
    "unique_new_observation_ids",
    "disjoint_from_competition_ids",
    "disjoint_from_qualification_ids",
    "disjoint_from_failure_boundary_probe_ids",
    "disjoint_from_restriction_ids",
    "cross_epoch_pooling",
    "row_counts",
    "required_context_scope_pairs",
    "required_context_scope_pair_count",
    "covered_context_scope_pairs_by_split",
    "missing_context_scope_pairs_by_split",
    "duplicate_context_scope_pair_row_counts_by_split",
    "complete_exact_cartesian_coverage_by_split",
    "complete_evidence",
}
PROBE_REPLAY_KEYS = {
    "required_probe_ids",
    "source_probe_ids",
    "restricted_probe_ids",
    "source_exact_match",
    "restricted_exact_match",
    "q_registry_byte_equal",
    "v_registry_byte_equal",
    "violation_functionals_byte_equal",
    "same_evidence_rows_used_for_both_targets",
    "replayed_exactly",
}
PROBE_RESULTS_KEYS = {
    "prediction_scale",
    "holdout",
    "stress",
    "fresh_qualification_passed",
}
TRADEOFF_KEYS = {
    "source_model_class_digest",
    "restricted_model_class_digest",
    "centers_byte_equal",
    "grouping_and_group_keys_byte_equal",
    "point_prediction_error_equal_by_split",
    "all_restricted_radii_lte_source",
    "at_least_one_radius_strictly_reduced",
    "strict_subset_verified",
    "source_fresh_qualification_passed",
    "restricted_fresh_qualification_passed",
}
MONOTONICITY_KEYS = {
    "certificate_kind",
    "checked_row_count",
    "checked_context_scope_pair_count",
    "all_restricted_radii_lte_source",
    "every_restricted_pass_implies_source_pass",
    "restricted_pass_source_fail_observed",
    "verified",
}
SELECTION_KEYS = {
    "selection_status",
    "selected_target_kind",
    "selected_shadow_theory_state",
    "selected_shadow_theory_state_digest",
    "byte_equal_to_verified_target",
    "qualification_evaluator_epoch",
    "qualification_recorded_in_receipt_only",
}
ROLLBACK_KEYS = {
    "adjudication_status",
    "rollback_target_selected",
    "rollback_target_state_digest",
    "rollback_execution_status",
    "source_state_mutated",
    "restricted_state_mutated",
    "ambient_pointer_written",
}
LIFECYCLE_KEYS = {
    "competition_records",
    "qualification_records",
    "failure_boundary_probe_records",
    "restriction_competition_records",
    "post_restriction_adjudication_records",
    "future_scoring_policy",
    "logical_selective_erasure_applied",
    "physical_erasure",
}
RECORD_KEYS = {
    "evidence_digests",
    "observation_id_digest",
    "observation_count",
    "evaluator_epoch",
    "role",
    "eligible_for_future_scoring",
}
FUTURE_SCORING_KEYS = {
    "new_unconsumed_evidence_required",
    "reuse_competition_records_allowed",
    "reuse_consumed_qualification_records_allowed",
    "reuse_consumed_failure_boundary_probe_records_allowed",
    "reuse_consumed_restriction_records_allowed",
    "reuse_consumed_post_restriction_records_allowed",
    "cross_epoch_pooling_allowed",
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


def _context_key(context):
    return canonical_json_bytes(context)


def _source():
    values = restriction_kat._restrict()
    # Omit the sixth-slice result wrapper while retaining its canonical report.
    return (*values[:15], values[-1])


def _adjudication_input(source, *, mode="retain"):
    contract = _load(ADJUDICATION_CONTRACT)
    restriction_report = source[15]
    restricted = restriction_report["restricted_shadow_theory_state"]
    source_state = source[12]["probe_expanded_shadow_theory_state"]
    epoch = derive_shadow_post_restriction_adjudication_epoch(
        restriction_contract_digest=_digest_value(source[14]),
        restriction_report_digest=restriction_report["report_digest"],
        restricted_shadow_theory_state_digest=restriction_report[
            "restricted_shadow_theory_state_digest"
        ],
        source_probe_expanded_shadow_theory_state_digest=restriction_report[
            "source_probe_expanded_shadow_theory_state_digest"
        ],
        fixed_anchor=restricted["fixed_anchor"],
        adjudication_contract=contract,
    )
    restricted_centers = {
        _context_key(item["context"]): item["value"]
        for item in restricted["model_class"]["center_predictions"]
    }
    contexts = source[0]["theory_state"]["object_space"]["contexts"]
    scopes = restricted["scope_ids"]
    smallest_key = min(
        (_context_key(context) for context in contexts),
        key=lambda key: restriction_kat._radius(
            restricted["model_class"], scopes[0], json.loads(key)
        ),
    )
    evidence = {}
    for split in ("holdout", "stress"):
        rows = []
        for scope in scopes:
            for index, context in enumerate(contexts):
                key = _context_key(context)
                center = restricted_centers[key]
                error = 0.0
                if key == smallest_key and mode == "rollback":
                    restricted_radius = restriction_kat._radius(
                        restricted["model_class"], scope, context
                    )
                    source_radius = restriction_kat._radius(
                        source_state["model_class"], scope, context
                    )
                    error = (restricted_radius + min(source_radius, 0.2)) / 2.0
                elif key == smallest_key and mode == "both_fail":
                    source_radius = restriction_kat._radius(
                        source_state["model_class"], scope, context
                    )
                    error = max(source_radius, 0.2) + 0.05
                rows.append(
                    {
                        "observation_id": (
                            f"post-restriction-{mode}-{split}-{scope}-{index:02d}"
                        ),
                        "evaluator_epoch": epoch,
                        "fixed_anchor": restricted["fixed_anchor"],
                        "scope_id": scope,
                        "context": copy.deepcopy(context),
                        "observed_value": center + error,
                    }
                )
        evidence[split] = rows
    lifecycle = restriction_report["record_lifecycle_extension"]
    return {
        "schema_version": contract["input_schema_version"],
        "adjudication_id": f"bounded-post-restriction-{mode}",
        "source_restriction": {
            "restriction_contract_digest": _digest_value(source[14]),
            "restriction_report_digest": restriction_report["report_digest"],
            "restricted_shadow_theory_state_digest": restriction_report[
                "restricted_shadow_theory_state_digest"
            ],
            "source_probe_expanded_shadow_theory_state_digest": (
                restriction_report["source_probe_expanded_shadow_theory_state_digest"]
            ),
        },
        "evaluator": {
            "evaluator_epoch": epoch,
            "fixed_anchor": restricted["fixed_anchor"],
        },
        "prior_record_exclusion": {
            "competition": lifecycle["competition_records"][
                "observation_id_digest"
            ],
            "qualification": lifecycle["qualification_records"][
                "observation_id_digest"
            ],
            "failure_boundary_probe": lifecycle["failure_boundary_probe_records"][
                "observation_id_digest"
            ],
            "restriction": lifecycle["restriction_competition_records"][
                "observation_id_digest"
            ],
        },
        "evidence": evidence,
    }


def _kwargs(source, adjudication_input, contract, *, input_artifacts=None):
    kwargs = restriction_kat._kwargs(source[:13], source[13], source[14])
    kwargs.pop("input_artifacts")
    kwargs.update(
        {
            "expected_restriction_report_digest": source[15]["report_digest"],
            "expected_restriction_input_artifacts": None,
            "expected_adjudication_input_digest": _digest_value(adjudication_input),
            "expected_adjudication_contract_digest": _digest_value(contract),
            "input_artifacts": input_artifacts,
        }
    )
    return kwargs


def _adjudicate(*, source=None, mode="retain", mutate_input=None):
    source = source or _source()
    adjudication_input = _adjudication_input(source, mode=mode)
    if mutate_input is not None:
        mutate_input(adjudication_input, source)
    contract = _load(ADJUDICATION_CONTRACT)
    result = qualify_adjudicate_and_route_shadow_post_restriction(
        *source,
        adjudication_input,
        contract,
        **_kwargs(source, adjudication_input, contract),
    )
    return (*source, adjudication_input, contract, result, _as_dict(result))


def _verify(source, adjudication_input, contract, report, **overrides):
    kwargs = _kwargs(source, adjudication_input, contract)
    kwargs.pop("input_artifacts")
    kwargs.update(
        {
            "expected_adjudication_report_digest": report["report_digest"],
            "expected_adjudication_input_artifacts": None,
        }
    )
    kwargs.update(overrides)
    return verify_shadow_post_restriction_adjudication(
        *source, adjudication_input, contract, report, **kwargs
    )


def _rehash(report):
    report["report_digest"] = _digest_value(
        {key: value for key, value in report.items() if key != "report_digest"}
    )
    return report


def test_contract_report_surface_and_nonclaims_are_frozen():
    contract = _load(ADJUDICATION_CONTRACT)
    assert validate_shadow_post_restriction_adjudication_contract(contract) == contract
    assert _digest_value(contract) == FROZEN_CONTRACT_DIGEST
    assert contract["contract_id"] == "shadow_post_restriction_adjudication_v1"
    assert contract["source_restriction_contract_digest"] == (
        restriction_kat.FROZEN_CONTRACT_DIGEST
    )
    assert contract["selection"]["retain_restricted_status"] in ALL_DISPOSITIONS
    assert contract["selection"]["source_rollback_target_status"] in ALL_DISPOSITIONS
    assert contract["selection"]["both_failed_status"] in ALL_DISPOSITIONS
    assert contract["selection"]["needs_evidence_status"] in ALL_DISPOSITIONS
    assert contract["selection"]["incomparable_status"] in ALL_DISPOSITIONS
    assert contract["loop_compatibility_boundary"] == {
        "competition_v1_required_model_kind": "finite_point_table",
        "competition_v1_required_probe_ids": ["absolute_error_point_prediction"],
        "selected_source_model_kind": "finite_interval_table",
        "selected_source_probe_ids": [
            "absolute_error_point_prediction",
            "normalized_signed_interval_boundary_margin",
        ],
        "feed_back_compatible": False,
        "language_last_resort_certified": False,
        "language_expansion_executed": False,
        "required_bridge": (
            "INTERVAL_MULTI_Q_THEORY_OPERATION_RECOMPETITION_ADAPTER_NOT_IMPLEMENTED"
        ),
    }

    *_, report = _adjudicate()
    assert set(report) == REPORT_KEYS
    assert report["nonclaims"] == contract["nonclaims"]
    assert report["report_digest"] == FROZEN_RETAIN_REPORT_DIGEST
    assert set(report["source_restriction"]) == SOURCE_RESTRICTION_KEYS
    assert set(report["evaluator_definition"]) == EVALUATOR_DEFINITION_KEYS
    assert set(report["evaluator_binding"]) == EVALUATOR_BINDING_KEYS
    assert set(report["evidence_binding"]) == EVIDENCE_BINDING_KEYS
    assert set(report["probe_registry_replay"]) == PROBE_REPLAY_KEYS
    assert set(report["source_probe_results"]) == PROBE_RESULTS_KEYS
    assert set(report["restricted_probe_results"]) == PROBE_RESULTS_KEYS
    assert set(report["finite_interval_tradeoff"]) == TRADEOFF_KEYS
    assert set(report["monotonicity_certificate"]) == MONOTONICITY_KEYS
    assert set(report["shadow_state_selection"]) == SELECTION_KEYS
    assert set(report["rollback_adjudication"]) == ROLLBACK_KEYS
    assert set(report["record_lifecycle_extension"]) == LIFECYCLE_KEYS
    lifecycle = report["record_lifecycle_extension"]
    for record_name in (
        "competition_records",
        "qualification_records",
        "failure_boundary_probe_records",
        "restriction_competition_records",
        "post_restriction_adjudication_records",
    ):
        assert set(lifecycle[record_name]) == RECORD_KEYS
    assert set(lifecycle["future_scoring_policy"]) == FUTURE_SCORING_KEYS
    assert report["authority_boundary"] == contract["authority_boundary"]
    for target in ("source_probe_results", "restricted_probe_results"):
        for split in ("holdout", "stress"):
            assert set(report[target][split]) == SPLIT_METRIC_KEYS


def test_restricted_shadow_pass_is_retained_without_authority_escalation():
    *prefix, result, report = _adjudicate(mode="retain")
    source = prefix[:16]
    restriction_report = source[15]
    restricted_before = canonical_json_bytes(
        restriction_report["restricted_shadow_theory_state"]
    )
    source_before = canonical_json_bytes(
        source[12]["probe_expanded_shadow_theory_state"]
    )

    assert report["disposition"] == (
        "POST_RESTRICTION_QUALIFIED_RETAIN_RESTRICTED_SHADOW"
    )
    assert result.restricted_shadow_qualified is True
    assert result.rollback_target_selected is False
    assert result.cycle_route == report["cycle_route"]
    assert report["cycle_route"] == (
        "RETAIN_RESTRICTED_SHADOW_AND_REQUIRE_RECOMPETITION_ADAPTER_FOR_NEXT_CYCLE"
    )
    assert canonical_json_bytes(
        restriction_report["restricted_shadow_theory_state"]
    ) == restricted_before
    assert canonical_json_bytes(
        source[12]["probe_expanded_shadow_theory_state"]
    ) == source_before
    assert report["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert report["promotion_status"] == "NOT_PROMOTED"
    assert report["current_status"] == "NOT_CURRENT"
    assert report["authority_boundary"]["rollback_execution"] is False
    assert report["authority_boundary"]["current_pointer_write"] is False
    assert report["shadow_state_selection"]["byte_equal_to_verified_target"] is True
    assert canonical_json_bytes(
        report["shadow_state_selection"]["selected_shadow_theory_state"]
    ) == restricted_before
    assert b"observed_value" not in canonical_json_bytes(report)
    assert not hasattr(result, "eligible")
    assert not hasattr(result, "adopt")
    assert not hasattr(result, "make_current")


def test_restricted_failure_source_pass_selects_exact_rollback_target_not_execution():
    *prefix, result, report = _adjudicate(mode="rollback")
    source = prefix[:16]
    assert report["disposition"] == (
        "POST_RESTRICTION_FAILED_SOURCE_SHADOW_ROLLBACK_TARGET_SELECTED"
    )
    assert result.restricted_shadow_qualified is False
    assert result.rollback_target_selected is True
    assert report["cycle_route"] == (
        "SOURCE_ROLLBACK_TARGET_SELECTED_AND_REQUIRE_RECOMPETITION_ADAPTER_FOR_NEXT_CYCLE"
    )
    assert report["shadow_state_selection"] is not None
    assert report["rollback_adjudication"] is not None
    assert report["authority_boundary"]["rollback_target_selection"] is True
    assert report["authority_boundary"]["rollback_execution"] is False
    assert report["rollback_adjudication"]["rollback_execution_status"] == (
        "NOT_PERFORMED"
    )
    source_digest = source[15]["source_probe_expanded_shadow_theory_state_digest"]
    assert source_digest in canonical_json_bytes(report["rollback_adjudication"]).decode()
    assert report["shadow_state_selection"]["byte_equal_to_verified_target"] is True
    assert canonical_json_bytes(
        report["shadow_state_selection"]["selected_shadow_theory_state"]
    ) == canonical_json_bytes(source[12]["probe_expanded_shadow_theory_state"])


def test_both_targets_fail_routes_adapter_before_any_language_expansion():
    *_, result, report = _adjudicate(mode="both_fail")
    assert report["disposition"] == (
        "POST_RESTRICTION_AND_SOURCE_FAILED_RECOMPETITION_ADAPTER_REQUIRED"
    )
    assert result.restricted_shadow_qualified is False
    assert result.rollback_target_selected is False
    assert report["cycle_route"] == (
        "RECOMPETITION_ADAPTER_REQUIRED_BEFORE_LANGUAGE_EXPANSION"
    )
    boundary = report["loop_compatibility_boundary"]
    assert boundary["feed_back_compatible"] is False
    assert boundary["language_last_resort_certified"] is False
    assert boundary["language_expansion_executed"] is False
    assert boundary["required_bridge"] == (
        "INTERVAL_MULTI_Q_THEORY_OPERATION_RECOMPETITION_ADAPTER_NOT_IMPLEMENTED"
    )
    assert report["authority_boundary"]["recompetition_adapter_execution"] is False
    assert report["authority_boundary"]["language_expansion_execution"] is False
    assert report["shadow_state_selection"]["selection_status"] == (
        "NO_QUALIFIED_SHADOW_SELECTED"
    )
    assert report["shadow_state_selection"]["selected_shadow_theory_state"] is None


def test_nonmaterialized_restriction_is_not_an_adjudication_case():
    values = restriction_kat._restrict(
        factors={"calibration": 1.1, "holdout": 0.4, "stress": 0.4}
    )
    source = (*values[:15], values[-1])
    assert source[15]["restricted_shadow_theory_state"] is None
    adjudication_input = {}
    contract = _load(ADJUDICATION_CONTRACT)
    with pytest.raises(ValueError, match="requires a materialized restriction"):
        qualify_adjudicate_and_route_shadow_post_restriction(
            *source,
            adjudication_input,
            contract,
            **_kwargs(source, adjudication_input, contract),
        )


@pytest.mark.parametrize("split", ("holdout", "stress"))
def test_missing_pair_needs_fresh_evidence_without_numeric_verdict(split):
    *_, report = _adjudicate(
        mutate_input=lambda payload, _: payload["evidence"][split].pop()
    )
    assert report["disposition"] == "POST_RESTRICTION_NEEDS_NEW_EVIDENCE"
    assert report["source_probe_results"] is None
    assert report["restricted_probe_results"] is None
    assert report["monotonicity_certificate"] is None
    assert report["shadow_state_selection"] is None
    assert report["record_lifecycle_extension"][
        "post_restriction_adjudication_records"
    ]["role"] == "NOT_CONSUMED_POST_RESTRICTION_PRECONDITION_BLOCKED"


@pytest.mark.parametrize("field", ("evaluator_epoch", "fixed_anchor"))
def test_epoch_or_anchor_mismatch_is_incomparable_without_numeric_verdict(field):
    def mutate(payload, _):
        payload["evidence"]["stress"][0][field] = "mismatch"

    *_, report = _adjudicate(mutate_input=mutate)
    assert report["disposition"] == (
        "POST_RESTRICTION_INCOMPARABLE_EVALUATOR_EPOCH"
    )
    assert report["source_probe_results"] is None
    assert report["restricted_probe_results"] is None
    assert report["monotonicity_certificate"] is None
    assert report["shadow_state_selection"] is None
    assert report["record_lifecycle_extension"][
        "post_restriction_adjudication_records"
    ]["role"] == "NOT_CONSUMED_POST_RESTRICTION_PRECONDITION_BLOCKED"


def test_all_five_dispositions_are_reachable_and_distinct():
    reports = [
        _adjudicate(mode="retain")[-1],
        _adjudicate(mode="rollback")[-1],
        _adjudicate(mode="both_fail")[-1],
        _adjudicate(
            mutate_input=lambda payload, _: payload["evidence"]["holdout"].pop()
        )[-1],
        _adjudicate(
            mutate_input=lambda payload, _: payload["evidence"]["stress"][0].__setitem__(
                "evaluator_epoch", "other"
            )
        )[-1],
    ]
    assert {report["disposition"] for report in reports} == ALL_DISPOSITIONS


@pytest.mark.parametrize("mode", ("retain", "rollback", "both_fail"))
def test_probe_registry_recomputed_on_identical_rows_and_monotonicity_holds(mode):
    *_, report = _adjudicate(mode=mode)
    replay = report["probe_registry_replay"]
    assert replay["source_exact_match"] is True
    assert replay["restricted_exact_match"] is True
    assert replay["q_registry_byte_equal"] is True
    assert replay["v_registry_byte_equal"] is True
    assert replay["violation_functionals_byte_equal"] is True
    assert replay["same_evidence_rows_used_for_both_targets"] is True
    assert replay["replayed_exactly"] is True
    certificate = report["monotonicity_certificate"]
    assert certificate is not None
    assert certificate["all_restricted_radii_lte_source"] is True
    assert certificate["every_restricted_pass_implies_source_pass"] is True
    assert certificate["restricted_pass_source_fail_observed"] is False
    assert certificate["verified"] is True
    for split in ("holdout", "stress"):
        source_metric = report["source_probe_results"][split]
        restricted_metric = report["restricted_probe_results"][split]
        assert restricted_metric["mean_absolute_center_error"] == pytest.approx(
            source_metric["mean_absolute_center_error"]
        )
        assert restricted_metric["max_absolute_center_error"] == pytest.approx(
            source_metric["max_absolute_center_error"]
        )
        assert restricted_metric[
            "min_normalized_signed_interval_boundary_margin"
        ] <= source_metric["min_normalized_signed_interval_boundary_margin"]
        assert restricted_metric["boundary_violation_count"] >= source_metric[
            "boundary_violation_count"
        ]
        assert set(source_metric["counterexample_observation_ids"]).issubset(
            restricted_metric["counterexample_observation_ids"]
        )


def test_exact_interval_tradeoff_and_multi_q_incompatibility_are_explicit():
    *_, report = _adjudicate()
    tradeoff = report["finite_interval_tradeoff"]
    assert tradeoff["centers_byte_equal"] is True
    assert tradeoff["grouping_and_group_keys_byte_equal"] is True
    assert tradeoff["all_restricted_radii_lte_source"] is True
    assert tradeoff["at_least_one_radius_strictly_reduced"] is True
    assert tradeoff["strict_subset_verified"] is True
    assert report["monotonicity_certificate"]["verified"] is True
    assert report["loop_compatibility_boundary"]["feed_back_compatible"] is False
    assert report["loop_compatibility_boundary"][
        "competition_v1_required_model_kind"
    ] == "finite_point_table"
    assert report["loop_compatibility_boundary"][
        "selected_source_model_kind"
    ] == "finite_interval_table"


def test_raw_boundary_violation_survives_normalized_negative_zero_underflow():
    context = {"x": 0}
    geometry = {
        "centers": {
            canonical_json_bytes(context): 0.0,
            canonical_json_bytes({"x": 1}): 1e308,
        },
        "grouping": "global",
        "entries": [{"group": {"global": "*"}, "radius": 1e-300}],
    }
    rows = [
        {
            "observation_id": "raw-boundary-underflow-counterexample",
            "scope_id": "scope-A",
            "context": context,
            "observed_value": 2e-300,
        }
    ]
    metrics = adjudication_core._split_metrics(
        rows, geometry, 5e307, "underflow-known-answer"
    )
    assert metrics["min_normalized_signed_interval_boundary_margin"] == 0.0
    assert math.copysign(
        1.0, metrics["min_normalized_signed_interval_boundary_margin"]
    ) == -1.0
    assert metrics["boundary_violation_count"] == 1
    assert metrics["boundary_violation_rate"] == 1.0
    assert metrics["counterexample_observation_ids"] == [
        "raw-boundary-underflow-counterexample"
    ]


def _prior_id_sets(source):
    return (
        {
            row["observation_id"]
            for rows in source[0]["evidence"].values()
            for row in rows
        },
        {
            row["observation_id"]
            for rows in source[5]["evidence"].values()
            for row in rows
        },
        {
            row["observation_id"]
            for rows in source[10]["evidence"].values()
            for row in rows
        },
        {
            row["observation_id"]
            for rows in source[13]["evidence"].values()
            for row in rows
        },
    )


@pytest.mark.parametrize("generation", range(4))
def test_all_four_prior_observation_generations_are_excluded(generation):
    source = _source()
    reused = next(iter(_prior_id_sets(source)[generation]))

    def mutate(payload, _):
        payload["evidence"]["holdout"][0]["observation_id"] = reused

    with pytest.raises(ValueError):
        _adjudicate(source=source, mutate_input=mutate)


def test_cross_split_duplicate_id_and_nonfinite_value_fail_closed():
    source = _source()

    def duplicate(payload, _):
        payload["evidence"]["stress"][0]["observation_id"] = payload["evidence"][
            "holdout"
        ][0]["observation_id"]

    with pytest.raises(ValueError):
        _adjudicate(source=source, mutate_input=duplicate)

    def nonfinite(payload, _):
        payload["evidence"]["holdout"][0]["observed_value"] = float("nan")

    with pytest.raises(ValueError):
        _adjudicate(source=source, mutate_input=nonfinite)


def test_duplicate_cartesian_pair_needs_evidence_and_unregistered_pair_fails_closed():
    def duplicate_pair(payload, _):
        payload["evidence"]["holdout"][1]["scope_id"] = payload["evidence"][
            "holdout"
        ][0]["scope_id"]
        payload["evidence"]["holdout"][1]["context"] = copy.deepcopy(
            payload["evidence"]["holdout"][0]["context"]
        )

    *_, report = _adjudicate(mutate_input=duplicate_pair)
    assert report["disposition"] == "POST_RESTRICTION_NEEDS_NEW_EVIDENCE"

    def unregistered(payload, _):
        payload["evidence"]["holdout"][0]["scope_id"] = "unregistered"

    with pytest.raises(ValueError):
        _adjudicate(mutate_input=unregistered)


def test_sparse_cross_cannot_bypass_full_scope_context_cartesian_coverage():
    contexts = [{"x": 0}, {"x": 1}]
    diagonal = [
        {"observation_id": "a-0", "scope_id": "scope-A", "context": contexts[0]},
        {"observation_id": "b-1", "scope_id": "scope-B", "context": contexts[1]},
    ]
    normalized = {
        "parent_scopes": ["scope-A", "scope-B"],
        "parent_contexts": contexts,
        "evidence": {
            "holdout": copy.deepcopy(diagonal),
            "stress": [
                {**copy.deepcopy(row), "observation_id": f"stress-{index}"}
                for index, row in enumerate(diagonal)
            ],
        },
        "new_ids": ["a-0", "b-1", "stress-0", "stress-1"],
        "old_ids": {
            "competition": [],
            "qualification": [],
            "failure_boundary_probe": [],
            "restriction": [],
        },
    }
    binding = adjudication_core._evidence_binding(normalized)
    expected_missing = {
        canonical_json_bytes(
            {"scope_id": "scope-A", "context": {"x": 1}}
        ),
        canonical_json_bytes(
            {"scope_id": "scope-B", "context": {"x": 0}}
        ),
    }
    assert binding["required_context_scope_pair_count"] == 4
    assert binding["complete_evidence"] is False
    assert binding["complete_exact_cartesian_coverage_by_split"] == {
        "holdout": False,
        "stress": False,
    }
    for split in ("holdout", "stress"):
        assert {
            canonical_json_bytes(pair)
            for pair in binding["missing_context_scope_pairs_by_split"][split]
        } == expected_missing


@pytest.mark.parametrize("grouping", ("per_scope", "per_context"))
def test_unused_extra_radius_group_is_rejected_during_adjudication(grouping):
    contexts = [{"x": 0}, {"x": 1}]
    scopes = ["scope-A", "scope-B"]
    centers = [
        {"context": copy.deepcopy(context), "value": float(index)}
        for index, context in enumerate(contexts)
    ]
    if grouping == "per_scope":
        radii = [
            {"group": {"scope_id": scope}, "radius": 1.0} for scope in scopes
        ]
        radii.append({"group": {"scope_id": "unused"}, "radius": 1.0})
    else:
        radii = [
            {"group": {"context": copy.deepcopy(context)}, "radius": 1.0}
            for context in contexts
        ]
        radii.append({"group": {"context": {"x": 2}}, "radius": 1.0})
    state = {
        "model_class": {
            "kind": "finite_interval_table",
            "center_predictions": centers,
            "radius_grouping": grouping,
            "radii": radii,
        }
    }
    with pytest.raises(
        ValueError, match="radius group keys do not exactly equal the registry"
    ):
        adjudication_core._model_geometry(
            state, contexts, scopes, "unused-extra-known-answer"
        )


def test_epoch_is_observation_and_row_order_independent():
    source = _source()
    first = _adjudication_input(source)
    second = copy.deepcopy(first)
    for rows in second["evidence"].values():
        rows.reverse()
        for row in rows:
            row["observed_value"] += 0.01
    assert first["evaluator"] == second["evaluator"]

    contract = _load(ADJUDICATION_CONTRACT)
    first_report = _as_dict(
        qualify_adjudicate_and_route_shadow_post_restriction(
            *source, first, contract, **_kwargs(source, first, contract)
        )
    )
    reordered = copy.deepcopy(first)
    for rows in reordered["evidence"].values():
        rows.reverse()
    reordered_report = _as_dict(
        qualify_adjudicate_and_route_shadow_post_restriction(
            *source,
            reordered,
            contract,
            **_kwargs(source, reordered, contract),
        )
    )
    assert reordered_report == first_report


def test_same_frozen_epoch_supports_different_fresh_observed_outcomes():
    reports = [_adjudicate(mode=mode)[-1] for mode in ("retain", "rollback", "both_fail")]
    assert len({report["evaluator_definition"]["evaluator_epoch"] for report in reports}) == 1
    assert {report["disposition"] for report in reports} == {
        "POST_RESTRICTION_QUALIFIED_RETAIN_RESTRICTED_SHADOW",
        "POST_RESTRICTION_FAILED_SOURCE_SHADOW_ROLLBACK_TARGET_SELECTED",
        "POST_RESTRICTION_AND_SOURCE_FAILED_RECOMPETITION_ADAPTER_REQUIRED",
    }


def test_public_verifier_receipt_and_semantic_tamper_fail_closed():
    values = _adjudicate()
    source = values[:16]
    adjudication_input, contract, _, report = values[16:]
    receipt = _verify(source, adjudication_input, contract, report)
    assert receipt["status"].startswith("VERIFIED_")
    assert receipt["disposition"] == report["disposition"]
    assert receipt["report_digest"] == report["report_digest"]
    assert receipt["cycle_route"] == report["cycle_route"]
    assert receipt["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert receipt["current_status"] == "NOT_CURRENT"

    tampered = copy.deepcopy(report)
    tampered["authority_boundary"]["rollback_execution"] = True
    _rehash(tampered)
    with pytest.raises(ValueError):
        _verify(
            source,
            adjudication_input,
            contract,
            tampered,
            expected_adjudication_report_digest=tampered["report_digest"],
        )


@pytest.mark.parametrize("state", ("needs", "incomparable"))
def test_public_verifier_handles_nonnumerical_receipts_without_selected_state(state):
    if state == "needs":
        mutate = lambda payload, _: payload["evidence"]["holdout"].pop()
    else:
        mutate = lambda payload, _: payload["evidence"]["stress"][0].__setitem__(
            "evaluator_epoch", "incomparable"
        )
    values = _adjudicate(mutate_input=mutate)
    source = values[:16]
    adjudication_input, contract, _, report = values[16:]
    receipt = _verify(source, adjudication_input, contract, report)
    assert receipt["disposition"] == report["disposition"]
    assert receipt["selected_shadow_theory_state_digest"] is None
    assert receipt["restricted_shadow_qualified"] is False
    assert receipt["rollback_target_selected"] is False


@pytest.mark.parametrize(
    "target",
    ("audit", "source", "restricted", "monotonicity", "route", "authority"),
)
def test_rehashed_report_tampering_is_rejected(target):
    values = _adjudicate(mode="rollback")
    source = values[:16]
    adjudication_input, contract, _, report = values[16:]
    tampered = copy.deepcopy(report)
    if target == "audit":
        tampered["audit_events"][0]["event"] = "FORGED"
    elif target == "source":
        tampered["source_probe_results"]["holdout"][
            "boundary_violation_count"
        ] += 1
    elif target == "restricted":
        tampered["restricted_probe_results"]["holdout"][
            "boundary_violation_count"
        ] = 0
    elif target == "monotonicity":
        bool_key = next(
            key
            for key, value in tampered["monotonicity_certificate"].items()
            if isinstance(value, bool)
        )
        tampered["monotonicity_certificate"][bool_key] = False
    elif target == "route":
        tampered["cycle_route"] = "FORGED_ROUTE"
    elif target == "authority":
        tampered["authority_boundary"]["current_pointer_write"] = True
    _rehash(tampered)
    with pytest.raises(ValueError):
        _verify(
            source,
            adjudication_input,
            contract,
            tampered,
            expected_adjudication_report_digest=tampered["report_digest"],
        )


def test_all_seventeen_independent_digest_anchors_fail_closed():
    source = _source()
    adjudication_input = _adjudication_input(source)
    contract = _load(ADJUDICATION_CONTRACT)
    kwargs = _kwargs(source, adjudication_input, contract)
    keys = [
        key
        for key in kwargs
        if key.startswith("expected_") and key.endswith("_digest")
    ]
    assert len(keys) == 17
    for key in keys:
        forged = dict(kwargs)
        forged[key] = "sha256:" + "0" * 64
        with pytest.raises(ValueError):
            qualify_adjudicate_and_route_shadow_post_restriction(
                *source, adjudication_input, contract, **forged
            )


def import_runner():
    spec = importlib.util.spec_from_file_location(
        "post_restriction_runner", ADJUDICATION_RUNNER
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
    )
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
    )
    return list(
        itertools.chain.from_iterable(
            (f"--expected-{name}-digest", values[name.replace("-", "_")])
            for name in names
        )
    )


def _materialize_cli_inputs(tmp_path):
    first_paths, values = restriction_kat._materialize_cli_inputs(tmp_path)
    paths = (*first_paths, (tmp_path / "input-15.json").resolve())
    completed = subprocess.run(
        [
            sys.executable,
            str(restriction_kat.RESTRICTION_RUNNER),
            *restriction_kat._input_flags(first_paths),
            *restriction_kat._digest_flags(values),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    paths[15].write_bytes(completed.stdout)
    source = tuple(_load(path) for path in paths)
    adjudication_input = _adjudication_input(source)
    contract = _load(ADJUDICATION_CONTRACT)
    paths = (
        *paths,
        (tmp_path / "input-16.json").resolve(),
        (tmp_path / "input-17.json").resolve(),
    )
    _write(paths[16], adjudication_input)
    _write(paths[17], contract)
    values = {
        **values,
        "restriction_report": source[15]["report_digest"],
        "adjudication_input": _digest_value(adjudication_input),
        "adjudication_contract": _digest_value(contract),
    }
    return paths, values


def test_cli_canonical_stdout_atomic_out_exact_18_artifacts_and_no_input_write(
    tmp_path,
):
    paths, values = _materialize_cli_inputs(tmp_path)
    before = {path: path.read_bytes() for path in paths}
    out = (tmp_path / "adjudication-report.json").resolve()
    completed = subprocess.run(
        [
            sys.executable,
            str(ADJUDICATION_RUNNER),
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
    assert len(report["input_artifacts"]) == 18


@pytest.mark.parametrize(
    "raw",
    (
        b'{"adjudication":"a","adjudication":"b"}\n',
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
        b'[]\n',
    ),
)
def test_cli_rejects_ambiguous_or_nonobject_json_before_core(raw, tmp_path):
    paths = [(tmp_path / f"input-{index}.json").resolve() for index in range(18)]
    for path in paths:
        path.write_text("{}\n", encoding="utf-8")
    paths[0].write_bytes(raw)
    zero = "sha256:" + "0" * 64
    names = (
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
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ADJUDICATION_RUNNER),
            *_input_flags(paths),
            *_digest_flags({name: zero for name in names}),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert completed.stdout == b""


def test_relative_symlink_and_output_aliases_are_rejected(tmp_path):
    paths = [(tmp_path / f"input-{index}.json").resolve() for index in range(18)]
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


def test_all_153_input_path_and_hardlink_alias_pairs_are_rejected(tmp_path):
    runner = import_runner()
    base_paths = []
    for index in range(18):
        path = tmp_path / f"base-input-{index}.json"
        path.write_text("{}\n", encoding="utf-8")
        base_paths.append(path.resolve())
    checked_same = 0
    checked_hard = 0
    for first in range(18):
        for second in range(first + 1, 18):
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
    assert checked_same == 153
    assert checked_hard == 153


def test_surface_has_no_execution_network_or_prior_file_mutation():
    before = {
        path: _digest_file(path)
        for path in (*PREVIOUS_SLICE_FILES, restriction_kat.OLD_BENCHMARK)
    }
    for path in (ADJUDICATION_CORE, ADJUDICATION_RUNNER):
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

    _adjudicate()
    assert {path: _digest_file(path) for path in before} == before


def test_documentation_freezes_bounded_additive_non_authoritative_claims():
    text = ADJUDICATION_DOC.read_text(encoding="utf-8")
    assert "strictly additive V1" in text
    assert "caller-supplied static" in text
    assert "exactly eighteen" in text
    assert "153" in text
    assert "Operations Research" in text
    assert "adapter-before-language" in text.lower()
    assert "finite interval table with" in text and "multi-Q" in text
    assert "NOT_PERFORMED" in text
    assert "not a complete autonomous theory-evolution loop" in text
