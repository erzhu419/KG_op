import ast
import copy
import hashlib
import importlib.util
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

import test_shadow_post_restriction_adjudication as adjudication_kat  # noqa: E402
import performance.shadow_interval_multi_q_recompetition_adapter as adapter_core  # noqa: E402

from performance.shadow_interval_multi_q_recompetition_adapter import (  # noqa: E402
    adapt_shadow_interval_multi_q_recompetition_seed,
    derive_shadow_interval_multi_q_recompetition_seed_id,
    validate_shadow_interval_multi_q_recompetition_adapter_contract,
    verify_shadow_interval_multi_q_recompetition_adapter,
)
from performance.theory_operation_competition import (  # noqa: E402
    canonical_json_bytes,
)


COMPETITION_CONTRACT = adjudication_kat.COMPETITION_CONTRACT
TRANSITION_CONTRACT = adjudication_kat.TRANSITION_CONTRACT
QUALIFICATION_CONTRACT = adjudication_kat.QUALIFICATION_CONTRACT
REVIEW_CONTRACT = adjudication_kat.REVIEW_CONTRACT
PROBE_CONTRACT = adjudication_kat.PROBE_CONTRACT
RESTRICTION_CONTRACT = adjudication_kat.RESTRICTION_CONTRACT
ADJUDICATION_CONTRACT = adjudication_kat.ADJUDICATION_CONTRACT
ADAPTER_CONTRACT = (
    ROOT / "performance/manifests/shadow_interval_multi_q_recompetition_adapter_v1.json"
)
ADAPTER_CORE = ROOT / "performance/shadow_interval_multi_q_recompetition_adapter.py"
ADAPTER_RUNNER = ROOT / "runners/run_shadow_interval_multi_q_recompetition_adapter.py"
ADAPTER_DOC = ROOT / "docs/shadow_interval_multi_q_recompetition_adapter_v1.md"

PREVIOUS_SLICE_FILES = (
    *adjudication_kat.PREVIOUS_SLICE_FILES,
    adjudication_kat.ADJUDICATION_CORE,
    adjudication_kat.ADJUDICATION_CONTRACT,
    adjudication_kat.ADJUDICATION_RUNNER,
    ROOT / "tests/test_shadow_post_restriction_adjudication.py",
    adjudication_kat.ADJUDICATION_DOC,
)

ALL_DISPOSITIONS = {
    "EMITTED_RESTRICTED_SHADOW_RECOMPETITION_SEED",
    "EMITTED_SOURCE_ROLLBACK_TARGET_RECOMPETITION_SEED",
    "EMITTED_UNQUALIFIED_SOURCE_REPAIR_RECOMPETITION_SEED",
    "RECOMPETITION_ADAPTER_NEEDS_NEW_POST_RESTRICTION_EVIDENCE",
    "RECOMPETITION_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH",
}

FROZEN_CONTRACT_DIGEST = (
    "sha256:16d2a30873e3f8b2e56fe5d7ac272140eb83dbcb441d8d80a892c4028f28f029"
)
FROZEN_RETAIN_REPORT_DIGEST = (
    "sha256:af3930e87940b92e5d7c4e6b081ae6c1dd9896fec5841be76cea64a6dd9c309d"
)
FROZEN_RETAIN_SEED_DIGEST = (
    "sha256:5b2e3c4e0f7c8b3248e1b1f56c810e07157ff927502eff6b2c12fbecbafa2b79"
)

REPORT_KEYS = {
    "schema_version",
    "contract_id",
    "contract_digest",
    "adapter_input_digest",
    "adapter_id",
    "source_adjudication",
    "source_disposition",
    "source_cycle_route",
    "source_state_catalog",
    "seed_resolution",
    "recompetition_seed",
    "recompetition_seed_digest",
    "interface_certificate",
    "competition_v1_incompatibility",
    "competition_v2_handoff",
    "disposition",
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
SEED_KEYS = {
    "schema_version",
    "seed_id",
    "source_adapter_input_digest",
    "source_adjudication_contract_digest",
    "source_adjudication_report_digest",
    "source_disposition",
    "seed_kind",
    "qualification_status",
    "theory_state",
    "theory_state_digest",
    "alternate_state_digests",
    "model_interface",
    "operation_registry",
    "required_diagnostic_order",
    "future_evidence_requirement",
    "prior_record_exclusion",
    "adoption_status",
    "current_status",
}
SOURCE_ADJUDICATION_KEYS = {
    "verification_status",
    "contract_id",
    "contract_digest",
    "report_digest",
    "adjudication_id",
    "disposition",
    "cycle_route",
    "source_probe_expanded_shadow_theory_state_digest",
    "restricted_shadow_theory_state_digest",
    "adoption_status",
}
STATE_CATALOG_ENTRY_KEYS = {
    "state_kind",
    "theory_state_digest",
    "model_class_digest",
    "qualification_status",
}
SEED_RESOLUTION_KEYS = {
    "resolution_status",
    "selected_seed_kind",
    "selected_theory_state_digest",
    "canonical_byte_equal_to_verified_source",
    "rollback_execution_status",
    "both_failed_repair_certificate",
}
INTERFACE_CERTIFICATE_KEYS = {
    "certificate_kind",
    "source_probe_expanded_state_digest",
    "restricted_state_digest",
    "seed_state_digest",
    "source_model_kind",
    "restricted_model_kind",
    "seed_model_kind",
    "required_probe_ids",
    "source_probe_ids",
    "restricted_probe_ids",
    "seed_probe_ids",
    "source_two_q_exact",
    "restricted_two_q_exact",
    "seed_two_q_exact",
    "q_registry_byte_equal",
    "v_registry_byte_equal",
    "object_space_byte_equal",
    "scope_ids_byte_equal",
    "removable_feature_ids_byte_equal",
    "center_predictions_byte_equal",
    "radius_grouping_byte_equal",
    "source_radius_group_keys",
    "restricted_radius_group_keys",
    "expected_radius_group_keys",
    "source_radius_group_keys_exact",
    "restricted_radius_group_keys_exact",
    "all_source_radii_finite_nonnegative",
    "all_restricted_radii_finite_nonnegative",
    "seed_state_byte_equal_to_resolved_source",
    "verified",
}
V1_INCOMPATIBILITY_KEYS = {
    "competition_contract_id",
    "required_model_kind",
    "required_probe_ids",
    "actual_model_kind",
    "actual_probe_ids",
    "lossless_projection_to_v1_exists",
    "point_projection_drops_radii",
    "single_q_projection_drops_probe_id",
    "upstream_competition_v1_exact_replay_performed",
    "adapter_seed_submitted_to_competition_v1",
    "compatible",
}
V2_HANDOFF_KEYS = {
    "required_core",
    "implementation_status",
    "seed_emitted",
    "seed_digest",
    "new_evaluator_epoch_required",
    "required_fresh_splits",
    "complete_context_scope_cartesian_per_split_required",
    "prior_record_reuse_allowed",
    "cross_epoch_pooling_allowed",
    "adapter_candidate_synthesis_performed",
    "adapter_candidate_evaluation_performed",
    "adapter_scoring_performed",
    "language_last_resort_deferred",
    "language_expansion_executed_by_adapter",
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


def _source(*, mode="retain", mutate_adjudication_input=None):
    values = adjudication_kat._adjudicate(
        mode=mode, mutate_input=mutate_adjudication_input
    )
    # Omit the seventh-slice result wrapper while retaining its canonical report.
    return (*values[:18], values[-1])


def _adapter_input(source):
    contract = _load(ADAPTER_CONTRACT)
    adjudication_report = source[18]
    lifecycle = adjudication_report["record_lifecycle_extension"]
    seed_id = derive_shadow_interval_multi_q_recompetition_seed_id(
        adjudication_contract_digest=_digest_value(source[17]),
        adjudication_report_digest=adjudication_report["report_digest"],
        source_probe_expanded_state_digest=adjudication_report[
            "source_restriction"
        ]["source_probe_expanded_shadow_theory_state_digest"],
        restricted_state_digest=adjudication_report["source_restriction"][
            "restricted_shadow_theory_state_digest"
        ],
        adapter_contract=contract,
    )
    return {
        "schema_version": contract["input_schema_version"],
        "adapter_id": seed_id,
        "source_adjudication": {
            "adjudication_contract_digest": _digest_value(source[17]),
            "adjudication_report_digest": adjudication_report["report_digest"],
            "adjudication_id": adjudication_report["adjudication_id"],
            "source_probe_expanded_shadow_theory_state_digest": (
                adjudication_report["source_restriction"][
                    "source_probe_expanded_shadow_theory_state_digest"
                ]
            ),
            "restricted_shadow_theory_state_digest": adjudication_report[
                "source_restriction"
            ]["restricted_shadow_theory_state_digest"],
        },
        "requested_bridge": adapter_core.REQUESTED_BRIDGE,
        "prior_record_exclusion": {
            "competition": lifecycle["competition_records"][
                "observation_id_digest"
            ],
            "qualification": lifecycle["qualification_records"][
                "observation_id_digest"
            ],
            "failure_boundary_probe": lifecycle[
                "failure_boundary_probe_records"
            ]["observation_id_digest"],
            "restriction": lifecycle["restriction_competition_records"][
                "observation_id_digest"
            ],
            "post_restriction_adjudication": lifecycle[
                "post_restriction_adjudication_records"
            ]["observation_id_digest"],
        },
    }


def _kwargs(source, adapter_input, contract, *, input_artifacts=None):
    kwargs = adjudication_kat._kwargs(source[:16], source[16], source[17])
    kwargs.pop("input_artifacts")
    kwargs.update(
        {
            "expected_adjudication_report_digest": source[18]["report_digest"],
            "expected_adjudication_input_artifacts": None,
            "expected_adapter_input_digest": _digest_value(adapter_input),
            "expected_adapter_contract_digest": _digest_value(contract),
            "input_artifacts": input_artifacts,
        }
    )
    return kwargs


def _adapt(*, source=None, mode="retain", mutate_adjudication_input=None, mutate_input=None):
    source = source or _source(
        mode=mode, mutate_adjudication_input=mutate_adjudication_input
    )
    adapter_input = _adapter_input(source)
    if mutate_input is not None:
        mutate_input(adapter_input, source)
    contract = _load(ADAPTER_CONTRACT)
    result = adapt_shadow_interval_multi_q_recompetition_seed(
        *source,
        adapter_input,
        contract,
        **_kwargs(source, adapter_input, contract),
    )
    return (*source, adapter_input, contract, result, _as_dict(result))


def _verify(source, adapter_input, contract, report, **overrides):
    kwargs = _kwargs(source, adapter_input, contract)
    kwargs.pop("input_artifacts")
    kwargs.update(
        {
            "expected_adapter_report_digest": report["report_digest"],
            "expected_adapter_input_artifacts": None,
        }
    )
    kwargs.update(overrides)
    return verify_shadow_interval_multi_q_recompetition_adapter(
        *source, adapter_input, contract, report, **kwargs
    )


def _rehash(report):
    report["report_digest"] = _digest_value(
        {key: value for key, value in report.items() if key != "report_digest"}
    )
    return report


@pytest.fixture(scope="module")
def known_cases():
    return {
        "retain": _adapt(mode="retain"),
        "rollback": _adapt(mode="rollback"),
        "both_failed": _adapt(mode="both_fail"),
        "needs_evidence": _adapt(
            mutate_adjudication_input=lambda payload, _: payload["evidence"][
                "holdout"
            ].pop()
        ),
        "incomparable": _adapt(
            mutate_adjudication_input=lambda payload, _: payload["evidence"][
                "stress"
            ][0].__setitem__("evaluator_epoch", "other-epoch")
        ),
    }


def test_contract_report_seed_surfaces_and_hashes_are_frozen(known_cases):
    contract = _load(ADAPTER_CONTRACT)
    assert validate_shadow_interval_multi_q_recompetition_adapter_contract(contract) == contract
    assert _digest_value(contract) == FROZEN_CONTRACT_DIGEST
    assert contract["contract_id"] == "shadow_interval_multi_q_recompetition_adapter_v1"
    assert contract["source_adjudication_contract_digest"] == (
        adjudication_kat.FROZEN_CONTRACT_DIGEST
    )
    assert contract["source_disposition_registry"] == adapter_core.SOURCE_DISPOSITION_REGISTRY
    assert contract["future_competition_requirements"]["operation_registry"] == (
        adapter_core.OPERATION_REGISTRY
    )
    assert contract["nonclaims"] == list(adapter_core.MANDATORY_NONCLAIMS)

    report = known_cases["retain"][-1]
    assert report["report_digest"] == FROZEN_RETAIN_REPORT_DIGEST
    assert report["recompetition_seed_digest"] == FROZEN_RETAIN_SEED_DIGEST
    assert set(report) == REPORT_KEYS
    assert set(report["source_adjudication"]) == SOURCE_ADJUDICATION_KEYS
    assert set(report["source_state_catalog"]) == {
        "source_probe_expanded_shadow",
        "restricted_shadow",
    }
    for entry in report["source_state_catalog"].values():
        assert set(entry) == STATE_CATALOG_ENTRY_KEYS
    assert set(report["seed_resolution"]) == SEED_RESOLUTION_KEYS
    assert set(report["interface_certificate"]) == INTERFACE_CERTIFICATE_KEYS
    assert set(report["competition_v1_incompatibility"]) == V1_INCOMPATIBILITY_KEYS
    assert set(report["competition_v2_handoff"]) == V2_HANDOFF_KEYS
    assert set(report["recompetition_seed"]) == SEED_KEYS
    assert report["authority_boundary"] == contract["authority_boundary"]
    assert report["nonclaims"] == contract["nonclaims"]


def test_all_five_source_routes_are_total_distinct_and_exact(known_cases):
    reports = {name: values[-1] for name, values in known_cases.items()}
    assert {report["disposition"] for report in reports.values()} == ALL_DISPOSITIONS
    expected = {
        "retain": (
            "POST_RESTRICTION_QUALIFIED_RETAIN_RESTRICTED_SHADOW",
            "EMITTED_RESTRICTED_SHADOW_RECOMPETITION_SEED",
            "RESTRICTED_SHADOW",
        ),
        "rollback": (
            "POST_RESTRICTION_FAILED_SOURCE_SHADOW_ROLLBACK_TARGET_SELECTED",
            "EMITTED_SOURCE_ROLLBACK_TARGET_RECOMPETITION_SEED",
            "SOURCE_ROLLBACK_TARGET",
        ),
        "both_failed": (
            "POST_RESTRICTION_AND_SOURCE_FAILED_RECOMPETITION_ADAPTER_REQUIRED",
            "EMITTED_UNQUALIFIED_SOURCE_REPAIR_RECOMPETITION_SEED",
            "UNQUALIFIED_SOURCE_REPAIR_BASE",
        ),
        "needs_evidence": (
            "POST_RESTRICTION_NEEDS_NEW_EVIDENCE",
            "RECOMPETITION_ADAPTER_NEEDS_NEW_POST_RESTRICTION_EVIDENCE",
            None,
        ),
        "incomparable": (
            "POST_RESTRICTION_INCOMPARABLE_EVALUATOR_EPOCH",
            "RECOMPETITION_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH",
            None,
        ),
    }
    for name, (source_disposition, disposition, seed_kind) in expected.items():
        report = reports[name]
        assert report["source_disposition"] == source_disposition
        assert report["disposition"] == disposition
        assert report["seed_resolution"]["selected_seed_kind"] == seed_kind
        seed = report["recompetition_seed"]
        assert (None if seed is None else seed["seed_kind"]) == seed_kind


def test_retain_seed_is_exact_restricted_state_without_authority_escalation(
    known_cases,
):
    values = known_cases["retain"]
    source, result, report = values[:19], values[-2], values[-1]
    restricted = source[15]["restricted_shadow_theory_state"]
    seed = report["recompetition_seed"]
    assert result.seed_emitted is True
    assert result.seed_kind == "RESTRICTED_SHADOW"
    assert canonical_json_bytes(seed["theory_state"]) == canonical_json_bytes(restricted)
    assert seed["theory_state_digest"] == source[15][
        "restricted_shadow_theory_state_digest"
    ]
    assert seed["qualification_status"] == (
        "QUALIFIED_RESTRICTED_ON_POST_RESTRICTION_EPOCH"
    )
    assert report["seed_resolution"]["canonical_byte_equal_to_verified_source"] is True
    assert report["seed_resolution"]["rollback_execution_status"] == "NOT_PERFORMED"
    assert report["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert report["promotion_status"] == "NOT_PROMOTED"
    assert report["current_status"] == "NOT_CURRENT"
    assert not hasattr(result, "score")
    assert not hasattr(result, "candidate")
    assert not hasattr(result, "adopt")
    assert not hasattr(result, "make_current")


def test_rollback_seed_is_exact_source_target_without_rollback_execution(known_cases):
    values = known_cases["rollback"]
    source, report = values[:19], values[-1]
    source_state = source[12]["probe_expanded_shadow_theory_state"]
    seed = report["recompetition_seed"]
    assert seed["seed_kind"] == "SOURCE_ROLLBACK_TARGET"
    assert canonical_json_bytes(seed["theory_state"]) == canonical_json_bytes(
        source_state
    )
    assert seed["theory_state_digest"] == source[12][
        "probe_expanded_shadow_theory_state_digest"
    ]
    assert seed["qualification_status"] == "QUALIFIED_SOURCE_ROLLBACK_TARGET_NOT_EXECUTED"
    assert report["seed_resolution"]["rollback_execution_status"] == "NOT_PERFORMED"
    assert report["authority_boundary"]["rollback_executed_by_adapter"] is False


def test_both_failed_seed_is_exact_unqualified_source_repair_base(known_cases):
    values = known_cases["both_failed"]
    source, report = values[:19], values[-1]
    seed = report["recompetition_seed"]
    source_state = source[12]["probe_expanded_shadow_theory_state"]
    assert seed["seed_kind"] == "UNQUALIFIED_SOURCE_REPAIR_BASE"
    assert seed["qualification_status"] == "UNQUALIFIED_SOURCE_REPAIR_BASE"
    assert canonical_json_bytes(seed["theory_state"]) == canonical_json_bytes(
        source_state
    )
    certificate = report["seed_resolution"]["both_failed_repair_certificate"]
    assert certificate["source_fresh_qualification_passed"] is False
    assert certificate["restricted_fresh_qualification_passed"] is False
    assert certificate["restriction_strict_subset_verified"] is True
    assert certificate["tradeoff_strict_subset_verified"] is True
    assert certificate["exact_two_q_registry"] is True
    assert certificate["v_registry_byte_equal"] is True
    assert certificate["center_predictions_byte_equal"] is True
    assert certificate["all_restricted_radii_lte_source"] is True
    assert certificate["monotonicity_verified"] is True
    assert certificate["every_restricted_pass_implies_source_pass"] is True
    assert certificate["verified"] is True
    assert report["adoption_eligibility"] == (
        "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED"
    )


@pytest.mark.parametrize("case", ("needs_evidence", "incomparable"))
def test_blocked_preconditions_emit_no_seed_or_fallback(known_cases, case):
    report = known_cases[case][-1]
    assert report["recompetition_seed"] is None
    assert report["recompetition_seed_digest"] is None
    assert report["seed_resolution"]["selected_seed_kind"] is None
    assert report["seed_resolution"]["selected_theory_state_digest"] is None
    assert report["seed_resolution"][
        "canonical_byte_equal_to_verified_source"
    ] is None
    assert report["competition_v2_handoff"]["seed_emitted"] is False
    assert report["competition_v2_handoff"]["seed_digest"] is None
    assert report["interface_certificate"]["seed_state_digest"] is None
    assert report["interface_certificate"]["seed_probe_ids"] is None
    assert report["interface_certificate"]["seed_two_q_exact"] is None


@pytest.mark.parametrize("case", ("retain", "rollback", "both_failed"))
def test_emitted_seed_preserves_exact_interval_two_q_interface(known_cases, case):
    report = known_cases[case][-1]
    seed = report["recompetition_seed"]
    certificate = report["interface_certificate"]
    assert certificate["certificate_kind"] == (
        "LOSSLESS_FINITE_INTERVAL_TWO_Q_STATE_INTERFACE"
    )
    assert certificate["source_model_kind"] == "finite_interval_table"
    assert certificate["restricted_model_kind"] == "finite_interval_table"
    assert certificate["seed_model_kind"] == "finite_interval_table"
    assert certificate["required_probe_ids"] == adapter_core.PROBE_IDS
    for key in (
        "source_two_q_exact",
        "restricted_two_q_exact",
        "seed_two_q_exact",
        "q_registry_byte_equal",
        "v_registry_byte_equal",
        "object_space_byte_equal",
        "scope_ids_byte_equal",
        "removable_feature_ids_byte_equal",
        "center_predictions_byte_equal",
        "radius_grouping_byte_equal",
        "source_radius_group_keys_exact",
        "restricted_radius_group_keys_exact",
        "all_source_radii_finite_nonnegative",
        "all_restricted_radii_finite_nonnegative",
        "seed_state_byte_equal_to_resolved_source",
        "verified",
    ):
        assert certificate[key] is True
    assert seed["model_interface"]["model_kind"] == "finite_interval_table"
    assert seed["model_interface"]["probe_ids"] == adapter_core.PROBE_IDS
    assert seed["operation_registry"] == adapter_core.OPERATION_REGISTRY
    assert seed["required_diagnostic_order"] == adapter_core.DIAGNOSTIC_ORDER
    assert len(seed["operation_registry"]) == 8
    assert all(set(item) == {"operation_id", "operation_kind"} for item in seed["operation_registry"])


def test_v1_is_explicitly_incompatible_and_future_v2_is_not_executed(known_cases):
    report = known_cases["retain"][-1]
    v1 = report["competition_v1_incompatibility"]
    assert v1["required_model_kind"] == "finite_point_table"
    assert v1["required_probe_ids"] == ["absolute_error_point_prediction"]
    assert v1["actual_model_kind"] == "finite_interval_table"
    assert v1["actual_probe_ids"] == adapter_core.PROBE_IDS
    assert v1["lossless_projection_to_v1_exists"] is False
    assert v1["point_projection_drops_radii"] is True
    assert v1["single_q_projection_drops_probe_id"] == adapter_core.PROBE_IDS[1]
    assert v1["upstream_competition_v1_exact_replay_performed"] is True
    assert v1["adapter_seed_submitted_to_competition_v1"] is False
    assert v1["compatible"] is False

    v2 = report["competition_v2_handoff"]
    assert v2["required_core"] == (
        "shadow_interval_multi_q_theory_operation_competition_v2"
    )
    assert v2["implementation_status"] == "REQUIRED_NOT_IMPLEMENTED_BY_ADAPTER"
    assert v2["new_evaluator_epoch_required"] is True
    assert v2["required_fresh_splits"] == ["discovery", "validation", "stress"]
    assert v2["prior_record_reuse_allowed"] is False
    assert v2["cross_epoch_pooling_allowed"] is False
    assert v2["adapter_candidate_synthesis_performed"] is False
    assert v2["adapter_candidate_evaluation_performed"] is False
    assert v2["adapter_scoring_performed"] is False
    assert v2["language_last_resort_deferred"] is True
    assert v2["language_expansion_executed_by_adapter"] is False
    authority = report["authority_boundary"]
    assert authority["upstream_shadow_chain_exact_replay_performed"] is True
    assert authority["upstream_competition_v1_exact_replay_performed"] is True
    assert authority["upstream_scoring_exact_replay_performed"] is True
    assert authority["adapter_seed_submitted_to_competition_v1"] is False
    assert authority["adapter_competition_v2_execution"] is False
    assert authority["adapter_candidate_synthesis_performed"] is False
    assert authority["adapter_candidate_evaluation_performed"] is False
    assert authority["adapter_scoring_performed"] is False
    assert authority["new_evidence_consumed_by_adapter"] is False
    assert authority["new_evaluator_epoch_created_by_adapter"] is False


def test_adapter_consumes_no_evidence_epoch_score_or_candidate(known_cases):
    values = known_cases["retain"]
    adapter_input, report = values[19], values[-1]
    assert set(adapter_input) == {
        "schema_version",
        "adapter_id",
        "source_adjudication",
        "requested_bridge",
        "prior_record_exclusion",
    }
    assert adapter_input["requested_bridge"] == (
        "FINITE_INTERVAL_TWO_Q_THEORY_OPERATION_RECOMPETITION_SEED"
    )
    encoded_input = canonical_json_bytes(adapter_input)
    encoded_report = canonical_json_bytes(report)
    for forbidden in (
        b"observed_value",
        b'"evidence"',
        b'"score"',
        b'"candidate"',
        b'"evaluator_epoch": "',
    ):
        assert forbidden not in encoded_input
        assert forbidden not in encoded_report
    lifecycle = report["record_lifecycle_extension"]
    assert lifecycle["all_prior_records_eligible_for_future_scoring"] is False
    assert lifecycle["adapter_record"] == {
        "role": "STATE_HANDOFF_ONLY_NO_SCORING",
        "evaluator_epoch": None,
        "new_observation_count": 0,
        "eligible_for_future_scoring": False,
    }
    assert lifecycle["future_scoring_policy"] == {
        "new_unconsumed_evidence_required": True,
        "required_new_evaluator_epoch": True,
        "reuse_any_prior_records_allowed": False,
        "cross_epoch_pooling_allowed": False,
    }


def test_seed_id_is_exactly_derived_and_input_order_independent(known_cases):
    values = known_cases["retain"]
    source, adapter_input, contract, report = values[:19], values[19], values[20], values[-1]
    expected = derive_shadow_interval_multi_q_recompetition_seed_id(
        adjudication_contract_digest=_digest_value(source[17]),
        adjudication_report_digest=source[18]["report_digest"],
        source_probe_expanded_state_digest=source[18]["source_restriction"][
            "source_probe_expanded_shadow_theory_state_digest"
        ],
        restricted_state_digest=source[18]["source_restriction"][
            "restricted_shadow_theory_state_digest"
        ],
        adapter_contract=contract,
    )
    assert adapter_input["adapter_id"] == expected
    assert report["recompetition_seed"]["seed_id"] == expected
    reordered_contract = dict(reversed(list(contract.items())))
    assert derive_shadow_interval_multi_q_recompetition_seed_id(
        adjudication_contract_digest=_digest_value(source[17]),
        adjudication_report_digest=source[18]["report_digest"],
        source_probe_expanded_state_digest=source[18]["source_restriction"][
            "source_probe_expanded_shadow_theory_state_digest"
        ],
        restricted_state_digest=source[18]["source_restriction"][
            "restricted_shadow_theory_state_digest"
        ],
        adapter_contract=reordered_contract,
    ) == expected


def test_public_verifier_and_rehashed_semantic_tampering_fail_closed(known_cases):
    values = known_cases["retain"]
    source, adapter_input, contract, report = values[:19], values[19], values[20], values[-1]
    receipt = _verify(source, adapter_input, contract, report)
    assert receipt["status"].startswith("VERIFIED_EMITTED_")
    assert receipt["report_digest"] == report["report_digest"]
    assert receipt["seed_emitted"] is True
    assert receipt["seed_kind"] == "RESTRICTED_SHADOW"
    assert receipt["competition_v1_compatible"] is False
    assert receipt["required_v2_core"] == adapter_core.REQUIRED_V2_CORE
    assert receipt["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert receipt["current_status"] == "NOT_CURRENT"

    tampered = copy.deepcopy(report)
    tampered["recompetition_seed"]["theory_state"]["current_status"] = "CURRENT"
    _rehash(tampered)
    with pytest.raises(ValueError):
        _verify(
            source,
            adapter_input,
            contract,
            tampered,
            expected_adapter_report_digest=tampered["report_digest"],
        )


@pytest.mark.parametrize(
    "target",
    ("seed_kind", "seed_bytes", "interface", "v1", "v2", "authority", "audit"),
)
def test_rehashed_report_tampering_is_rejected(known_cases, target):
    values = known_cases["both_failed"]
    source, adapter_input, contract, report = values[:19], values[19], values[20], values[-1]
    tampered = copy.deepcopy(report)
    if target == "seed_kind":
        tampered["recompetition_seed"]["seed_kind"] = "RESTRICTED_SHADOW"
    elif target == "seed_bytes":
        tampered["recompetition_seed"]["theory_state"]["theory_id"] = "forged"
    elif target == "interface":
        tampered["interface_certificate"]["verified"] = False
    elif target == "v1":
        tampered["competition_v1_incompatibility"][
            "adapter_seed_submitted_to_competition_v1"
        ] = True
    elif target == "v2":
        tampered["competition_v2_handoff"][
            "adapter_candidate_synthesis_performed"
        ] = True
    elif target == "authority":
        tampered["authority_boundary"]["current_pointer_written_by_adapter"] = True
    else:
        tampered["audit_events"][0]["event"] = "FORGED"
    _rehash(tampered)
    with pytest.raises(ValueError):
        _verify(
            source,
            adapter_input,
            contract,
            tampered,
            expected_adapter_report_digest=tampered["report_digest"],
        )


def test_input_bridge_lineage_and_prior_exclusions_fail_closed(known_cases):
    values = known_cases["retain"]
    source, base, contract = values[:19], values[19], values[20]
    mutations = (
        lambda value: value.__setitem__("requested_bridge", "POINT_V1"),
        lambda value: value["source_adjudication"].__setitem__(
            "adjudication_report_digest", "sha256:" + "0" * 64
        ),
        lambda value: value["prior_record_exclusion"].__setitem__(
            "post_restriction_adjudication", "sha256:" + "0" * 64
        ),
        lambda value: value.__setitem__("evidence", []),
    )
    for mutate in mutations:
        adapter_input = copy.deepcopy(base)
        mutate(adapter_input)
        with pytest.raises(ValueError):
            adapt_shadow_interval_multi_q_recompetition_seed(
                *source,
                adapter_input,
                contract,
                **_kwargs(source, adapter_input, contract),
            )


def test_rehashed_upstream_adjudication_tampering_fails_exact_replay(known_cases):
    values = known_cases["retain"]
    source = list(copy.deepcopy(values[:19]))
    source[18]["cycle_route"] = "FORGED_DIRECT_V1_FEEDBACK"
    adjudication_kat._rehash(source[18])
    source = tuple(source)
    adapter_input = _adapter_input(source)
    contract = _load(ADAPTER_CONTRACT)
    with pytest.raises(ValueError):
        adapt_shadow_interval_multi_q_recompetition_seed(
            *source,
            adapter_input,
            contract,
            **_kwargs(source, adapter_input, contract),
        )


def test_all_twenty_independent_digest_anchors_fail_closed(known_cases):
    values = known_cases["retain"]
    source, adapter_input, contract = values[:19], values[19], values[20]
    kwargs = _kwargs(source, adapter_input, contract)
    keys = [
        key
        for key in kwargs
        if key.startswith("expected_") and key.endswith("_digest")
    ]
    assert len(keys) == 20
    for key in keys:
        forged = dict(kwargs)
        forged[key] = "sha256:" + "0" * 64
        with pytest.raises(ValueError):
            adapt_shadow_interval_multi_q_recompetition_seed(
                *source, adapter_input, contract, **forged
            )


def test_contract_cannot_enable_v1_projection_execution_or_authority():
    contract = _load(ADAPTER_CONTRACT)
    mutations = (
        lambda value: value["seed_resolution_policy"].__setitem__(
            "theory_state_projection_allowed_by_adapter", True
        ),
        lambda value: value["future_competition_requirements"].__setitem__(
            "implementation_status", "IMPLEMENTED"
        ),
        lambda value: value["authority_boundary"].__setitem__(
            "adapter_seed_submitted_to_competition_v1", True
        ),
        lambda value: value["authority_boundary"].__setitem__(
            "adoption_decided_by_adapter", True
        ),
    )
    for mutate in mutations:
        changed = copy.deepcopy(contract)
        mutate(changed)
        with pytest.raises(ValueError):
            validate_shadow_interval_multi_q_recompetition_adapter_contract(changed)


def import_runner():
    spec = importlib.util.spec_from_file_location("interval_multi_q_runner", ADAPTER_RUNNER)
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
        "adjudication-report",
        "adapter-input",
        "adapter-contract",
    )
    return list(
        itertools.chain.from_iterable(
            (f"--expected-{name}-digest", values[name.replace("-", "_")])
            for name in names
        )
    )


def _materialize_cli_inputs(tmp_path):
    first_paths, values = adjudication_kat._materialize_cli_inputs(tmp_path)
    paths = (*first_paths, (tmp_path / "input-18.json").resolve())
    completed = subprocess.run(
        [
            sys.executable,
            str(adjudication_kat.ADJUDICATION_RUNNER),
            *adjudication_kat._input_flags(first_paths),
            *adjudication_kat._digest_flags(values),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    paths[18].write_bytes(completed.stdout)
    source = tuple(_load(path) for path in paths)
    adapter_input = _adapter_input(source)
    contract = _load(ADAPTER_CONTRACT)
    paths = (
        *paths,
        (tmp_path / "input-19.json").resolve(),
        (tmp_path / "input-20.json").resolve(),
    )
    _write(paths[19], adapter_input)
    _write(paths[20], contract)
    values = {
        **values,
        "adjudication_report": source[18]["report_digest"],
        "adapter_input": _digest_value(adapter_input),
        "adapter_contract": _digest_value(contract),
    }
    return paths, values


def test_cli_canonical_stdout_atomic_out_exact_21_artifacts_and_no_input_write(
    tmp_path,
):
    paths, values = _materialize_cli_inputs(tmp_path)
    before = {path: path.read_bytes() for path in paths}
    out = (tmp_path / "adapter-report.json").resolve()
    completed = subprocess.run(
        [
            sys.executable,
            str(ADAPTER_RUNNER),
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
    assert len(report["input_artifacts"]) == 21


@pytest.mark.parametrize(
    "raw",
    (
        b'{"adapter":"a","adapter":"b"}\n',
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
        b'[]\n',
    ),
)
def test_cli_rejects_ambiguous_or_nonobject_json_before_core(raw, tmp_path):
    paths = [(tmp_path / f"input-{index}.json").resolve() for index in range(21)]
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
        "adjudication_report",
        "adapter_input",
        "adapter_contract",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ADAPTER_RUNNER),
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
    paths = [(tmp_path / f"input-{index}.json").resolve() for index in range(21)]
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


def test_all_210_input_path_and_hardlink_alias_pairs_are_rejected(tmp_path):
    runner = import_runner()
    base_paths = []
    for index in range(21):
        path = tmp_path / f"base-input-{index}.json"
        path.write_text("{}\n", encoding="utf-8")
        base_paths.append(path.resolve())
    checked_same = 0
    checked_hard = 0
    for first in range(21):
        for second in range(first + 1, 21):
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
    assert checked_same == 210
    assert checked_hard == 210


def test_surface_has_no_execution_network_or_prior_file_mutation():
    before = {
        path: _digest_file(path)
        for path in (*PREVIOUS_SLICE_FILES, adjudication_kat.restriction_kat.OLD_BENCHMARK)
    }
    for path in (ADAPTER_CORE, ADAPTER_RUNNER):
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

    _adapt()
    assert {path: _digest_file(path) for path in before} == before


def test_documentation_freezes_adapter_only_additive_boundary():
    text = ADAPTER_DOC.read_text(encoding="utf-8")
    assert "strictly additive V1" in text
    assert FROZEN_CONTRACT_DIGEST in text
    assert FROZEN_RETAIN_REPORT_DIGEST in text
    assert "exactly twenty-one" in text
    assert "210" in text
    assert "Operations Research" in text
    assert "UNQUALIFIED_SOURCE_REPAIR_BASE" in text
    assert "finite_interval_table" in text
    assert "competition V2" in text
    assert "does not implement" in text
    assert "upstream exact replay includes" in text
    assert "performs no scoring" in text
    assert "no seed" in text
    assert "not a complete autonomous theory-evolution loop" in text
