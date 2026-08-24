import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_shadow_child_external_review_packet as review_kat  # noqa: E402

from performance.shadow_child_failure_boundary_probe import (  # noqa: E402
    derive_shadow_child_failure_boundary_probe_epoch,
    expand_and_evaluate_shadow_child_failure_boundary_probe,
    validate_shadow_child_failure_boundary_probe_contract,
    verify_shadow_child_failure_boundary_probe,
)
from performance.theory_operation_competition import (  # noqa: E402
    canonical_json_bytes,
)


COMPETITION_CONTRACT = (
    ROOT / "performance/manifests/theory_operation_competition_v1.json"
)
TRANSITION_CONTRACT = (
    ROOT / "performance/manifests/shadow_theory_transition_v1.json"
)
QUALIFICATION_CONTRACT = (
    ROOT / "performance/manifests/shadow_child_probe_qualification_v1.json"
)
REVIEW_CONTRACT = (
    ROOT / "performance/manifests/shadow_child_external_review_packet_v1.json"
)
PROBE_CONTRACT = (
    ROOT / "performance/manifests/shadow_child_failure_boundary_probe_v1.json"
)
COMPETITION_RUNNER = ROOT / "runners/run_theory_operation_competition.py"
TRANSITION_RUNNER = ROOT / "runners/run_shadow_theory_transition.py"
QUALIFICATION_RUNNER = ROOT / "runners/run_shadow_child_probe_qualification.py"
REVIEW_RUNNER = ROOT / "runners/run_shadow_child_external_review_packet.py"
PROBE_RUNNER = ROOT / "runners/run_shadow_child_failure_boundary_probe.py"
PROBE_CORE = ROOT / "performance/shadow_child_failure_boundary_probe.py"
PROBE_DOC = ROOT / "docs/shadow_child_failure_boundary_probe_v1.md"
OLD_BENCHMARK = ROOT / "performance/benchmark_lodo_meta_prior.py"

PREVIOUS_SLICE_FILES = (
    ROOT / "performance/theory_operation_competition.py",
    COMPETITION_CONTRACT,
    COMPETITION_RUNNER,
    ROOT / "tests/test_theory_operation_competition.py",
    ROOT / "docs/theory_operation_competition_v1.md",
    ROOT / "performance/shadow_theory_transition.py",
    TRANSITION_CONTRACT,
    TRANSITION_RUNNER,
    ROOT / "tests/test_shadow_theory_transition.py",
    ROOT / "docs/shadow_theory_transition_v1.md",
    ROOT / "performance/shadow_child_probe_qualification.py",
    QUALIFICATION_CONTRACT,
    QUALIFICATION_RUNNER,
    ROOT / "tests/test_shadow_child_probe_qualification.py",
    ROOT / "docs/shadow_child_probe_qualification_v1.md",
    ROOT / "performance/shadow_child_external_review_packet.py",
    REVIEW_CONTRACT,
    REVIEW_RUNNER,
    ROOT / "tests/test_shadow_child_external_review_packet.py",
    ROOT / "docs/shadow_child_external_review_packet_v1.md",
)

REPORT_KEYS = {
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

EXPANDED_STATE_KEYS = {
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

BOUNDARY_ACTION_KEYS = {
    "source_child_invalidated",
    "rollback_execution_allowed",
    "restriction_execution_allowed",
    "adoption_decision_allowed",
    "promotion_decision_allowed",
    "current_pointer_write_allowed",
    "parent_or_child_state_write_allowed",
}

SOURCE_REVIEW_PACKET_KEYS = {
    "verification_status",
    "contract_id",
    "contract_digest",
    "report_digest",
    "packet_id",
    "child_theory_state_digest",
    "disposition",
    "packet_ready",
    "adoption_status",
}
PROBE_DEFINITION_KEYS = {
    "probe_id",
    "functional",
    "normalization",
    "aggregation",
    "counterexample_rule",
}
EVALUATOR_DEFINITION_KEYS = {
    "evaluator_epoch",
    "fixed_anchor",
    "epoch_derivation_kind",
    "expanded_probe_registry_digest",
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
    "row_counts",
    "minimum_rows",
    "minimum_rows_satisfied_by_split",
    "required_context_scope_pairs",
    "required_context_scope_pair_count",
    "covered_context_scope_pairs_by_split",
    "missing_context_scope_pairs_by_split",
    "complete_context_scope_coverage_by_split",
    "new_observation_id_digest",
    "new_observation_count",
    "source_competition_observation_id_digest",
    "consumed_qualification_observation_id_digest",
    "unique_new_observation_ids",
    "disjoint_from_competition_ids",
    "disjoint_from_qualification_ids",
    "cross_epoch_pooling",
    "complete_evidence",
}
LIFECYCLE_KEYS = {
    "competition_records",
    "qualification_records",
    "new_probe_records",
    "future_scoring_policy",
    "logical_selective_erasure_applied",
    "physical_erasure",
    "physical_retention_attestation",
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
    "reuse_consumed_probe_records_allowed",
    "cross_epoch_pooling_allowed",
}
PROBE_RESULTS_KEYS = {
    "result_kind",
    "probe_ids",
    "holdout",
    "stress",
    "aggregate",
    "gate_checks",
    "boundary_counterexample_found",
    "counterexample_observation_ids",
}
GATE_KEYS = {
    "gate_id",
    "metric_path",
    "operator",
    "threshold",
    "actual",
    "passed",
}
BOUNDARY_ASSESSMENT_KEYS = BOUNDARY_ACTION_KEYS | {
    "boundary_counterexample_found"
}
ATTESTATION_KEYS = {
    "external_data_attestation",
    "external_evaluator_attestation",
    "physical_retention_attestation",
    "external_adoption_authority",
}
STATE_REUSE_KEYS = {
    "source_competition_records_allowed_for_scoring",
    "consumed_qualification_records_allowed_for_scoring",
    "failure_boundary_probe_records_allowed_for_future_scoring",
    "future_scoring_requires_new_unconsumed_evidence",
    "cross_epoch_pooling_allowed",
}
STATE_LINEAGE_KEYS = {
    "source_child_theory_state_digest",
    "source_review_contract_digest",
    "source_review_report_digest",
    "source_review_packet_id",
    "probe_contract_digest",
    "probe_expansion_id",
    "transition_kind",
    "added_probe_id",
}
AUDIT_EVENT_KEYS = {
    "sequence",
    "event",
    "payload",
    "previous_event_digest",
    "event_digest",
}
VERIFIER_RECEIPT_KEYS = {
    "status",
    "disposition",
    "report_digest",
    "contract_digest",
    "probe_expansion_id",
    "source_review_report_digest",
    "source_child_theory_state_digest",
    "probe_expanded_shadow_theory_state_digest",
    "probe_expanded",
    "boundary_counterexample_found",
    "adoption_eligibility",
    "adoption_status",
    "promotion_status",
    "current_status",
}

MANDATORY_NONCLAIMS = [
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
]


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


def _source(kind, review_state="ready"):
    values = review_kat._review(kind, review_state)
    return (*values[:8], values[8], values[10])


def _two_scope_quotient_source():
    qualification_kat = review_kat.qualification_kat
    scopes = ("scope-A", "scope-B")
    case = qualification_kat._case("idealization")
    case["case_id"] = "failure-boundary-two-scope-quotient-source"
    case["theory_state"]["scope_ids"] = list(scopes)
    for split, rows in case["evidence"].items():
        case["evidence"][split] = [
            {
                **copy.deepcopy(row),
                "observation_id": (
                    f"two-scope-source-{split}-{scope}-{index:02d}"
                ),
                "scope_id": scope,
            }
            for scope in scopes
            for index, row in enumerate(rows)
        ]

    competition_contract = _load(COMPETITION_CONTRACT)
    competition_report = _as_dict(
        qualification_kat.run_theory_operation_competition(
            case, competition_contract
        )
    )
    transition_contract = _load(TRANSITION_CONTRACT)
    transition_report = _as_dict(
        qualification_kat.materialize_shadow_theory_transition(
            case,
            competition_contract,
            competition_report,
            transition_contract,
            expected_competition_contract_digest=_digest_value(
                competition_contract
            ),
            expected_competition_report_digest=competition_report[
                "report_digest"
            ],
            expected_competition_input_artifacts=None,
            expected_transition_contract_digest=_digest_value(
                transition_contract
            ),
            input_artifacts=None,
        )
    )
    chain = (
        case,
        competition_contract,
        competition_report,
        transition_contract,
        transition_report,
    )
    qualification_input = qualification_kat._qualification_input(
        "idealization", chain
    )
    qualification_input["qualification_id"] = (
        "failure-boundary-two-scope-quotient-qualification"
    )
    for split, rows in qualification_input["evidence"].items():
        qualification_input["evidence"][split] = [
            {
                **copy.deepcopy(row),
                "observation_id": (
                    f"two-scope-qualification-{split}-{scope}-{index:02d}"
                ),
                "scope_id": scope,
            }
            for scope in scopes
            for index, row in enumerate(rows)
        ]
    qualification_contract = _load(QUALIFICATION_CONTRACT)
    qualification_report = _as_dict(
        qualification_kat.qualify_shadow_child_operational_probes(
            case,
            competition_contract,
            competition_report,
            transition_contract,
            transition_report,
            qualification_input,
            qualification_contract,
            expected_competition_contract_digest=_digest_value(
                competition_contract
            ),
            expected_competition_report_digest=competition_report[
                "report_digest"
            ],
            expected_competition_input_artifacts=None,
            expected_transition_contract_digest=_digest_value(
                transition_contract
            ),
            expected_transition_report_digest=transition_report[
                "report_digest"
            ],
            expected_transition_input_artifacts=None,
            expected_qualification_input_digest=_digest_value(
                qualification_input
            ),
            expected_qualification_contract_digest=_digest_value(
                qualification_contract
            ),
            input_artifacts=None,
        )
    )
    bundle = (
        *chain,
        qualification_input,
        qualification_contract,
        qualification_report,
    )
    review_contract, _, review_report = review_kat._build_from_bundle(bundle)
    return (*bundle, review_contract, review_report)


def _probe_rows(kind, source, epoch, *, counterexample):
    competition_input = source[0]
    transition_report = source[4]
    child = transition_report["child_theory_state"]
    contexts = competition_input["theory_state"]["object_space"]["contexts"]
    values = {}
    if kind == "robustification":
        centers = {
            _context_key(item["context"]): item["value"]
            for item in child["model_class"]["center_predictions"]
        }
        radii = {
            _context_key(item["group"]["context"]): item["radius"]
            for item in child["model_class"]["radii"]
        }
        for index, context in enumerate(contexts):
            key = _context_key(context)
            values[key] = centers[key]
            if counterexample and index == 0:
                values[key] = centers[key] + radii[key] + 1.0
    elif kind == "idealization":
        for context in contexts:
            base = float(context["x"])
            if counterexample and context["nuisance"] == 1:
                base += 1.0
            values[_context_key(context)] = base
    else:
        raise AssertionError(kind)

    evidence = {}
    for split in ("holdout", "stress"):
        evidence[split] = [
            {
                "observation_id": f"boundary-{kind}-{split}-{index:02d}",
                "evaluator_epoch": epoch,
                "fixed_anchor": child["fixed_anchor"],
                "scope_id": "registered-scope",
                "context": copy.deepcopy(context),
                "observed_value": values[_context_key(context)],
            }
            for index, context in enumerate(contexts)
        ]
    return evidence


def _two_scope_quotient_probe_input(source, *, counterexample):
    payload = _probe_input("idealization", source, counterexample=False)
    payload["probe_expansion_id"] = "failure-boundary-two-scope-quotient"
    epoch = payload["evaluator"]["evaluator_epoch"]
    anchor = payload["evaluator"]["fixed_anchor"]
    contexts = source[0]["theory_state"]["object_space"]["contexts"]
    evidence = {}
    for split in ("holdout", "stress"):
        rows = []
        for scope in ("scope-A", "scope-B"):
            for index, context in enumerate(contexts):
                value = float(context["x"])
                if counterexample:
                    if scope == "scope-A" and context["nuisance"] == 1:
                        value += 1.0
                    if scope == "scope-B" and context["nuisance"] == 0:
                        value += 1.0
                rows.append(
                    {
                        "observation_id": (
                            f"two-scope-boundary-{split}-{scope}-{index:02d}"
                        ),
                        "evaluator_epoch": epoch,
                        "fixed_anchor": anchor,
                        "scope_id": scope,
                        "context": copy.deepcopy(context),
                        "observed_value": value,
                    }
                )
        evidence[split] = rows
    payload["evidence"] = evidence
    return payload


def _expand_two_scope_quotient(*, counterexample):
    source = _two_scope_quotient_source()
    probe_input = _two_scope_quotient_probe_input(
        source, counterexample=counterexample
    )
    probe_contract = _load(PROBE_CONTRACT)
    result = expand_and_evaluate_shadow_child_failure_boundary_probe(
        *source,
        probe_input,
        probe_contract,
        **_probe_kwargs(source, probe_input, probe_contract),
    )
    return (*source, probe_input, probe_contract, result, _as_dict(result))


def _probe_input(kind, source, *, counterexample=True):
    transition_report = source[4]
    review_contract = source[8]
    review_report = source[9]
    probe_contract = _load(PROBE_CONTRACT)
    child = transition_report["child_theory_state"]
    epoch = derive_shadow_child_failure_boundary_probe_epoch(
        review_contract_digest=_digest_value(review_contract),
        review_report_digest=review_report["report_digest"],
        child_theory_state_digest=transition_report["child_theory_state_digest"],
        transition_kind=transition_report["transition_kind"],
        fixed_anchor=child["fixed_anchor"],
        probe_contract=probe_contract,
    )
    lifecycle = review_report["record_lifecycle_boundary"]
    return {
        "schema_version": probe_contract["input_schema_version"],
        "probe_expansion_id": f"failure-boundary-{kind}",
        "source_review_packet": {
            "review_contract_digest": _digest_value(review_contract),
            "review_report_digest": review_report["report_digest"],
            "packet_id": review_report["packet_id"],
            "child_theory_state_digest": transition_report[
                "child_theory_state_digest"
            ],
        },
        "evaluator": {
            "evaluator_epoch": epoch,
            "fixed_anchor": child["fixed_anchor"],
        },
        "prior_record_exclusion": {
            "source_competition_observation_id_digest": lifecycle[
                "source_competition_records"
            ]["observation_id_digest"],
            "consumed_qualification_observation_id_digest": lifecycle[
                "qualification_records"
            ]["observation_id_digest"],
        },
        "evidence": _probe_rows(
            kind, source, epoch, counterexample=counterexample
        ),
    }


def _probe_kwargs(source, probe_input, probe_contract, *, input_artifacts=None):
    return {
        "expected_competition_contract_digest": _digest_value(source[1]),
        "expected_competition_report_digest": source[2]["report_digest"],
        "expected_competition_input_artifacts": None,
        "expected_transition_contract_digest": _digest_value(source[3]),
        "expected_transition_report_digest": source[4]["report_digest"],
        "expected_transition_input_artifacts": None,
        "expected_qualification_input_digest": _digest_value(source[5]),
        "expected_qualification_contract_digest": _digest_value(source[6]),
        "expected_qualification_report_digest": source[7]["report_digest"],
        "expected_qualification_input_artifacts": None,
        "expected_review_contract_digest": _digest_value(source[8]),
        "expected_review_report_digest": source[9]["report_digest"],
        "expected_review_input_artifacts": None,
        "expected_probe_input_digest": _digest_value(probe_input),
        "expected_probe_contract_digest": _digest_value(probe_contract),
        "input_artifacts": input_artifacts,
    }


def _expand(
    kind,
    *,
    counterexample=True,
    review_state="ready",
    mutate_input=None,
):
    source = _source(kind, review_state)
    probe_input = _probe_input(kind, source, counterexample=counterexample)
    if mutate_input is not None:
        mutate_input(probe_input, source)
    probe_contract = _load(PROBE_CONTRACT)
    result = expand_and_evaluate_shadow_child_failure_boundary_probe(
        *source,
        probe_input,
        probe_contract,
        **_probe_kwargs(source, probe_input, probe_contract),
    )
    return (*source, probe_input, probe_contract, result, _as_dict(result))


def _verify(source, probe_input, probe_contract, report, **overrides):
    kwargs = _probe_kwargs(source, probe_input, probe_contract)
    kwargs.pop("input_artifacts")
    kwargs.update(
        {
            "expected_probe_report_digest": report["report_digest"],
            "expected_probe_input_artifacts": None,
        }
    )
    kwargs.update(overrides)
    return verify_shadow_child_failure_boundary_probe(
        *source,
        probe_input,
        probe_contract,
        report,
        **kwargs,
    )


def _rehash(report):
    report["report_digest"] = _digest_value(
        {key: value for key, value in report.items() if key != "report_digest"}
    )
    return report


def _all_observation_ids(source):
    source_ids = {
        row["observation_id"]
        for rows in source[0]["evidence"].values()
        for row in rows
    }
    qualification_ids = {
        row["observation_id"]
        for rows in source[5]["evidence"].values()
        for row in rows
    }
    return source_ids, qualification_ids


def test_frozen_contract_exact_top_level_vocabulary_and_nonclaims():
    contract = _load(PROBE_CONTRACT)
    assert validate_shadow_child_failure_boundary_probe_contract(contract) == contract
    assert contract["contract_id"] == "shadow_child_failure_boundary_probe_v1"
    assert contract["source_review_contract_digest"] == (
        "sha256:7b438072804c95eee26be901c0839bfea0b65b31824a708940b055ab61f858f1"
    )
    assert set(contract["transition_probe_registry"]) == {
        "ROBUST_INTERVAL_EXPANSION",
        "QUOTIENT_IDEALIZATION",
    }
    quotient_probe = contract["transition_probe_registry"][
        "QUOTIENT_IDEALIZATION"
    ]
    assert quotient_probe["functional"] == (
        "(max_within_scope_fiber_context_mean - "
        "min_within_scope_fiber_context_mean) / prediction_scale"
    )
    assert quotient_probe["aggregation"] == (
        "maximum_normalized_nontrivial_scope_fiber_spread_by_fresh_split"
    )
    evidence_policy = contract["evidence_policy"]
    assert evidence_policy["require_complete_parent_context_scope_pairs_per_split"] is True
    assert "require_every_parent_context_in_each_split" not in evidence_policy
    assert "require_every_registered_scope_in_each_split" not in evidence_policy
    assert contract["nonclaims"] == MANDATORY_NONCLAIMS

    *_, report = _expand("robustification")
    assert set(report) == REPORT_KEYS
    assert set(report["probe_expanded_shadow_theory_state"]) == EXPANDED_STATE_KEYS
    assert report["nonclaims"] == MANDATORY_NONCLAIMS


def test_exact_nested_report_state_audit_and_verifier_vocabularies():
    values = _expand("robustification")
    source = values[:10]
    probe_input, contract, report = values[10], values[11], values[-1]
    assert set(report["source_review_packet"]) == SOURCE_REVIEW_PACKET_KEYS
    assert set(report["probe_definition"]) == PROBE_DEFINITION_KEYS
    assert set(report["probe_definition"]["counterexample_rule"]) == {
        "operator",
        "threshold",
    }
    assert set(report["evaluator_definition"]) == EVALUATOR_DEFINITION_KEYS
    assert set(report["evaluator_binding"]) == EVALUATOR_BINDING_KEYS
    assert set(report["evidence_binding"]) == EVIDENCE_BINDING_KEYS
    lifecycle = report["record_lifecycle_extension"]
    assert set(lifecycle) == LIFECYCLE_KEYS
    for record_class in (
        "competition_records",
        "qualification_records",
        "new_probe_records",
    ):
        assert set(lifecycle[record_class]) == RECORD_KEYS
    assert set(lifecycle["future_scoring_policy"]) == FUTURE_SCORING_KEYS
    assert set(report["probe_results"]) == PROBE_RESULTS_KEYS
    assert all(set(gate) == GATE_KEYS for gate in report["probe_results"]["gate_checks"])
    assert set(report["boundary_assessment"]) == BOUNDARY_ASSESSMENT_KEYS
    assert set(report["attestation_boundary"]) == ATTESTATION_KEYS

    state = report["probe_expanded_shadow_theory_state"]
    assert set(state) == EXPANDED_STATE_KEYS
    assert set(state["evidence_reuse_policy"]) == STATE_REUSE_KEYS
    assert set(state["probe_expansion_lineage"]) == STATE_LINEAGE_KEYS
    assert len(report["audit_events"]) == 4
    assert all(set(event) == AUDIT_EVENT_KEYS for event in report["audit_events"])
    assert [event["sequence"] for event in report["audit_events"]] == [0, 1, 2, 3]
    assert [event["event"] for event in report["audit_events"]] == [
        "SOURCE_REVIEW_PACKET_VERIFIED",
        "FAILURE_BOUNDARY_PROBE_COMPILED",
        "NEW_UNCONSUMED_EVIDENCE_ISOLATION_BOUND",
        "FAILURE_BOUNDARY_PROBE_ASSESSED_AND_AUTHORITY_WITHHELD",
    ]
    receipt = _verify(source, probe_input, contract, report)
    assert set(receipt) == VERIFIER_RECEIPT_KEYS
    assert receipt["status"] == "VERIFIED_" + report["disposition"]


@pytest.mark.parametrize(
    (
        "kind",
        "expected_epoch",
        "expected_state_digest",
        "expected_audit_head",
        "expected_report_digest",
    ),
    (
        (
            "robustification",
            "shadow-failure-boundary-probe-epoch:894601cc094670106778f75bb40d2d2b21ea8b16665d47760d62b002b3abf13b",
            "sha256:fc057415f1d689121ca862e7e27f57fd3db9b50612b6d7ba8e28176dc6b5adc8",
            "sha256:578ccad21cc8862f8a57d6afa191d253fb89799f1a51fce267e2c6bb1ef5ed91",
            "sha256:72ea25d0f8af91d82afb021217736e7f66faa34ccb20841746035b39a8c9c36d",
        ),
        (
            "idealization",
            "shadow-failure-boundary-probe-epoch:d7dc1e1f17946675cc15eb7ef7992b89ccffe02fdaf6a0e1baa92ca34425017c",
            "sha256:9baae992c366dc9dc53decc413916f7f0f94d28c5610d2fd90845bf2196f8a32",
            "sha256:e3bc6c40bff50ca7fcf1cb32f7f4ccd0801524642f971f2c54e1cb4261595764",
            "sha256:65145b5609f0e07e68df0f0e0ab31555dad892b271241669541aa64f5b5a485a",
        ),
    ),
)
def test_frozen_positive_boundary_counterexample_known_answer_vectors(
    kind,
    expected_epoch,
    expected_state_digest,
    expected_audit_head,
    expected_report_digest,
):
    report = _expand(kind, counterexample=True)[-1]
    assert report["contract_digest"] == (
        "sha256:fdc92e276f7d8cb0c1ab6fd097242932851da04e1f97888d3f9597bfb0f726e0"
    )
    assert report["evaluator_definition"]["evaluator_epoch"] == expected_epoch
    assert report["probe_expanded_shadow_theory_state_digest"] == (
        expected_state_digest
    )
    assert report["audit_head"] == expected_audit_head
    assert report["report_digest"] == expected_report_digest


@pytest.mark.parametrize(
    ("kind", "probe_id"),
    (
        ("robustification", "normalized_signed_interval_boundary_margin"),
        ("idealization", "deleted_feature_conditional_response_spread"),
    ),
)
def test_bounded_counterexample_kats(kind, probe_id):
    values = _expand(kind, counterexample=True)
    source_child = values[4]["child_theory_state"]
    result = values[-2]
    report = values[-1]
    assert result.probe_expanded is True
    assert result.boundary_counterexample_found is True
    assert not hasattr(result, "eligible")
    assert not hasattr(result, "adopt")
    assert report["disposition"] == (
        "EXPANDED_PROBE_BOUNDARY_COUNTEREXAMPLE_FOUND_SHADOW_ONLY"
    )
    state = report["probe_expanded_shadow_theory_state"]
    assert state["probe_ids"][-1] == probe_id
    assert len(state["probe_ids"]) == 2
    assert state["evaluator_status"] == (
        "LOCAL_FAILURE_BOUNDARY_PROBE_EPOCH_UNATTESTED"
    )
    assert state["operational_probe_status"] == (
        "FAILURE_BOUNDARY_PROBE_COMPILED_SHADOW_ONLY"
    )
    assert state["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert state["current_status"] == "NOT_CURRENT"
    assert canonical_json_bytes(state["violation_functionals"]) == (
        canonical_json_bytes(source_child["violation_functionals"])
    )
    assert report["probe_results"]["boundary_counterexample_found"] is True
    assert report["probe_results"]["counterexample_observation_ids"]


@pytest.mark.parametrize("kind", ("robustification", "idealization"))
def test_complete_evidence_without_boundary_crossing_has_bounded_negative_verdict(kind):
    *_, result, report = _expand(kind, counterexample=False)
    assert result.probe_expanded is True
    assert result.boundary_counterexample_found is False
    assert report["disposition"] == (
        "EXPANDED_PROBE_NO_BOUNDARY_COUNTEREXAMPLE_ON_SUPPLIED_EVIDENCE"
    )
    assert report["probe_results"]["boundary_counterexample_found"] is False
    assert report["probe_results"]["counterexample_observation_ids"] == []
    assert "counterexample_absence_is_not_global_preservation" in report["nonclaims"]


def test_robust_metrics_bind_signed_margin_and_exceedance():
    *_, report = _expand("robustification")
    results = report["probe_results"]
    assert results["result_kind"] == (
        "ROBUST_INTERVAL_FAILURE_BOUNDARY_PROBE_RESULT"
    )
    for split in ("holdout", "stress"):
        assert set(results[split]) == {
            "row_count",
            "min_normalized_signed_margin",
            "boundary_violation_rate",
            "mean_normalized_exceedance",
            "max_normalized_exceedance",
        }
        assert results[split]["row_count"] == 4
        assert results[split]["min_normalized_signed_margin"] < 0.0
        assert results[split]["max_normalized_exceedance"] > 0.0
    assert results["aggregate"]["prediction_scale"] == pytest.approx(6.2)


def test_quotient_metrics_bind_nontrivial_fiber_spread():
    *_, report = _expand("idealization")
    results = report["probe_results"]
    assert results["result_kind"] == (
        "QUOTIENT_IDEALIZATION_FAILURE_BOUNDARY_PROBE_RESULT"
    )
    for split in ("holdout", "stress"):
        assert set(results[split]) == {
            "row_count",
            "evaluated_nontrivial_fiber_count",
            "max_normalized_fiber_response_spread",
            "offending_fiber_count",
            "offending_fiber_digests",
        }
        assert results[split]["row_count"] == 4
        assert results[split]["evaluated_nontrivial_fiber_count"] == 2
        assert results[split]["max_normalized_fiber_response_spread"] > 0.2
        assert results[split]["offending_fiber_count"] == 2
    assert results["aggregate"]["prediction_scale"] == pytest.approx(0.75)


def test_two_scope_quotient_groups_within_scope_before_fiber_spread_kat():
    values = _expand_two_scope_quotient(counterexample=True)
    source = values[:10]
    result = values[-2]
    report = values[-1]

    expected_scopes = {"scope-A", "scope-B"}
    assert set(source[0]["theory_state"]["scope_ids"]) == expected_scopes
    assert set(source[4]["child_theory_state"]["scope_ids"]) == expected_scopes
    for rows in source[0]["evidence"].values():
        assert {row["scope_id"] for row in rows} == expected_scopes
    for rows in source[5]["evidence"].values():
        assert {row["scope_id"] for row in rows} == expected_scopes
    assert source[7]["disposition"] == "QUALIFIED_NEW_EVALUATOR_EPOCH"
    assert source[9]["disposition"] == "READY_FOR_EXTERNAL_REVIEW_PACKET_ONLY"

    assert result.boundary_counterexample_found is True
    assert report["disposition"] == (
        "EXPANDED_PROBE_BOUNDARY_COUNTEREXAMPLE_FOUND_SHADOW_ONLY"
    )
    results = report["probe_results"]
    expected_fiber_digests = [
        "sha256:72cad309c3b6558d58e16392663bd10f3a0db296635d1d294028580cb552ba09",
        "sha256:94a48e600630c1becdb2e753a2c34b95f85c860a4e7bc030e948a81e9f4e8dd6",
        "sha256:94c5b670299091dee5da7fef4440829baeec086c1a8935ee941985664a2ba6c3",
        "sha256:d857c2b1e29f30c30f1b6ef5fc94f84de53416ffe039b26ab02915d261963f10",
    ]
    # Scope-A and scope-B carry opposite nuisance effects.  Pooling the scopes
    # would make both nuisance means equal and incorrectly yield zero spread.
    for split in ("holdout", "stress"):
        metrics = results[split]
        assert metrics["row_count"] == 8
        assert metrics["evaluated_nontrivial_fiber_count"] == 4
        assert metrics["max_normalized_fiber_response_spread"] == pytest.approx(
            1.0 / 0.75
        )
        assert metrics["offending_fiber_count"] == 4
        assert metrics["offending_fiber_digests"] == expected_fiber_digests
        assert len(set(metrics["offending_fiber_digests"])) == 4
    assert results["aggregate"]["evaluated_nontrivial_fiber_count"] == 8
    assert results["aggregate"]["total_offending_fiber_count"] == 8
    assert results["aggregate"]["max_normalized_fiber_response_spread"] > 0.2
    binding = report["evidence_binding"]
    assert binding["row_counts"] == {"holdout": 8, "stress": 8}
    assert binding["minimum_rows"] == {"holdout": 4, "stress": 4}
    assert binding["minimum_rows_satisfied_by_split"] == {
        "holdout": True,
        "stress": True,
    }
    assert binding["required_context_scope_pair_count"] == 8
    assert len(binding["required_context_scope_pairs"]) == 8
    assert binding["complete_context_scope_coverage_by_split"] == {
        "holdout": True,
        "stress": True,
    }
    assert binding["missing_context_scope_pairs_by_split"] == {
        "holdout": [],
        "stress": [],
    }
    assert binding["complete_evidence"] is True
    assert report["evaluator_definition"]["evaluator_epoch"] == (
        "shadow-failure-boundary-probe-epoch:175e5d4249e2cc6545600e029785edb45ebe80996d1695910df014e297e8d100"
    )
    assert report["probe_expanded_shadow_theory_state_digest"] == (
        "sha256:01fee4b286674bb6e3c1e018424697771d350e1f705a33cc715943ce1577aef3"
    )
    assert report["audit_head"] == (
        "sha256:9cdbc4ceb98bb1926da6b3114a99419a21d7b5bd75bd299c317415f15694fcb0"
    )
    assert report["report_digest"] == (
        "sha256:b4ec7b17397776946f3ed57c60cc2f052bce17fd59216a94fb96799f43458ff5"
    )


def test_two_scope_sparse_cross_cannot_bypass_cartesian_pair_coverage():
    source = _two_scope_quotient_source()
    probe_input = _two_scope_quotient_probe_input(source, counterexample=True)
    for split, rows in probe_input["evidence"].items():
        probe_input["evidence"][split] = [
            row
            for row in rows
            if (
                (
                    row["scope_id"] == "scope-A"
                    and row["context"]["nuisance"] == 0
                )
                or (
                    row["scope_id"] == "scope-B"
                    and row["context"]["nuisance"] == 1
                )
            )
        ]
        sparse_rows = probe_input["evidence"][split]
        assert len(sparse_rows) == 4
        assert {row["scope_id"] for row in sparse_rows} == {
            "scope-A",
            "scope-B",
        }
        assert {
            canonical_json_bytes(row["context"]) for row in sparse_rows
        } == {
            canonical_json_bytes(context)
            for context in source[0]["theory_state"]["object_space"][
                "contexts"
            ]
        }

    contract = _load(PROBE_CONTRACT)
    result = expand_and_evaluate_shadow_child_failure_boundary_probe(
        *source,
        probe_input,
        contract,
        **_probe_kwargs(source, probe_input, contract),
    )
    report = _as_dict(result)
    assert result.probe_expanded is True
    assert result.boundary_counterexample_found is None
    assert report["disposition"] == "EXPANDED_PROBE_NEEDS_NEW_EVIDENCE"
    assert report["probe_results"] is None
    binding = report["evidence_binding"]
    assert binding["row_counts"] == {"holdout": 4, "stress": 4}
    assert binding["minimum_rows_satisfied_by_split"] == {
        "holdout": True,
        "stress": True,
    }
    assert binding["required_context_scope_pair_count"] == 8
    assert all(
        len(binding["covered_context_scope_pairs_by_split"][split]) == 4
        for split in ("holdout", "stress")
    )
    assert all(
        len(binding["missing_context_scope_pairs_by_split"][split]) == 4
        for split in ("holdout", "stress")
    )
    assert binding["complete_context_scope_coverage_by_split"] == {
        "holdout": False,
        "stress": False,
    }
    assert binding["complete_evidence"] is False


def test_two_scope_quotient_no_counterexample_control():
    found = _expand_two_scope_quotient(counterexample=True)[-1]
    clear = _expand_two_scope_quotient(counterexample=False)[-1]
    assert clear["disposition"] == (
        "EXPANDED_PROBE_NO_BOUNDARY_COUNTEREXAMPLE_ON_SUPPLIED_EVIDENCE"
    )
    assert clear["probe_results"]["boundary_counterexample_found"] is False
    for split in ("holdout", "stress"):
        metrics = clear["probe_results"][split]
        assert metrics["row_count"] == 8
        assert metrics["evaluated_nontrivial_fiber_count"] == 4
        assert metrics["max_normalized_fiber_response_spread"] == 0.0
        assert metrics["offending_fiber_count"] == 0
        assert metrics["offending_fiber_digests"] == []
    assert found["evaluator_definition"] == clear["evaluator_definition"]
    assert found["probe_definition"] == clear["probe_definition"]
    assert found["probe_expanded_shadow_theory_state_digest"] == clear[
        "probe_expanded_shadow_theory_state_digest"
    ]
    assert clear["audit_head"] == (
        "sha256:68971bb540a27304bd6d525a81b6e9b8ca0476eaa949eb92d1cf8e734124bc44"
    )
    assert clear["report_digest"] == (
        "sha256:2ac6ff9a455dce7db215c69531f01d402192865a8e51072778cc60213a82f5fd"
    )


def test_insufficient_rows_and_context_coverage_needs_new_evidence_without_scores():
    def truncate(payload, _source):
        payload["evidence"]["holdout"] = payload["evidence"]["holdout"][:3]
        payload["evidence"]["stress"] = payload["evidence"]["stress"][:3]

    *_, result, report = _expand("robustification", mutate_input=truncate)
    assert result.probe_expanded is True
    assert result.boundary_counterexample_found is None
    assert report["disposition"] == "EXPANDED_PROBE_NEEDS_NEW_EVIDENCE"
    assert report["probe_definition"] is not None
    assert report["probe_expanded_shadow_theory_state"] is not None
    assert report["probe_results"] is None


def test_complete_row_count_but_missing_parent_context_needs_new_evidence():
    def duplicate_context(payload, _source):
        for rows in payload["evidence"].values():
            rows[-1]["context"] = copy.deepcopy(rows[0]["context"])

    *_, report = _expand("idealization", mutate_input=duplicate_context)
    assert report["disposition"] == "EXPANDED_PROBE_NEEDS_NEW_EVIDENCE"
    assert report["evidence_binding"]["new_observation_count"] == 8
    assert report["evidence_binding"]["complete_evidence"] is False
    assert report["probe_results"] is None


def test_unregistered_scope_is_rejected_before_any_probe_score():
    source = _source("robustification")
    probe_input = _probe_input("robustification", source)
    for rows in probe_input["evidence"].values():
        for row in rows:
            row["scope_id"] = "not-a-registered-scope"
    contract = _load(PROBE_CONTRACT)
    with pytest.raises(ValueError):
        expand_and_evaluate_shadow_child_failure_boundary_probe(
            *source,
            probe_input,
            contract,
            **_probe_kwargs(source, probe_input, contract),
        )


@pytest.mark.parametrize("mismatch", ("epoch", "anchor"))
def test_wrong_epoch_or_anchor_is_incomparable_without_numerical_verdict(mismatch):
    def mutate(payload, _source):
        key = "evaluator_epoch" if mismatch == "epoch" else "fixed_anchor"
        replacement = "wrong-epoch" if mismatch == "epoch" else "wrong-anchor"
        payload["evaluator"][key] = replacement
        for rows in payload["evidence"].values():
            for row in rows:
                row[key] = replacement

    *_, result, report = _expand("idealization", mutate_input=mutate)
    assert result.probe_expanded is True
    assert result.boundary_counterexample_found is None
    assert report["disposition"] == (
        "EXPANDED_PROBE_INCOMPARABLE_EVALUATOR_EPOCH"
    )
    assert report["probe_definition"] is not None
    assert report["probe_results"] is None


@pytest.mark.parametrize("kind", ("robustification", "idealization"))
def test_derived_epoch_is_deterministic_fresh_and_fixed_anchor_bound(kind):
    source = _source(kind)
    first = _probe_input(kind, source)
    second = _probe_input(kind, source)
    assert first["evaluator"] == second["evaluator"]
    epoch = first["evaluator"]["evaluator_epoch"]
    assert epoch != source[0]["theory_state"]["evaluator_epoch"]
    assert epoch != source[5]["evaluator"]["evaluator_epoch"]
    changed = derive_shadow_child_failure_boundary_probe_epoch(
        review_contract_digest=_digest_value(source[8]),
        review_report_digest=source[9]["report_digest"],
        child_theory_state_digest=source[4]["child_theory_state_digest"],
        transition_kind=source[4]["transition_kind"],
        fixed_anchor="changed-fixed-anchor",
        probe_contract=_load(PROBE_CONTRACT),
    )
    assert changed != epoch


@pytest.mark.parametrize(
    "key",
    (
        "source_competition_observation_id_digest",
        "consumed_qualification_observation_id_digest",
    ),
)
def test_prior_record_exclusion_commitment_tamper_is_rejected(key):
    source = _source("robustification")
    probe_input = _probe_input("robustification", source)
    probe_input["prior_record_exclusion"][key] = "sha256:" + "0" * 64
    contract = _load(PROBE_CONTRACT)
    with pytest.raises(ValueError):
        expand_and_evaluate_shadow_child_failure_boundary_probe(
            *source,
            probe_input,
            contract,
            **_probe_kwargs(source, probe_input, contract),
        )


@pytest.mark.parametrize(
    "state", ("pending", "incomparable", "failed")
)
def test_nonready_source_packet_blocks_probe_compilation(state):
    *_, result, report = _expand("robustification", review_state=state)
    assert result.probe_expanded is False
    assert result.boundary_counterexample_found is None
    assert report["disposition"] == (
        "PROBE_EXPANSION_BLOCKED_SOURCE_PACKET_NOT_READY"
    )
    assert report["probe_definition"] is None
    assert report["probe_expanded_shadow_theory_state"] is None
    assert report["probe_expanded_shadow_theory_state_digest"] is None
    assert report["probe_results"] is None


def test_all_five_dispositions_are_reachable_and_distinct():
    found = _expand("robustification", counterexample=True)[-1]["disposition"]
    absent = _expand("robustification", counterexample=False)[-1]["disposition"]

    def truncate(payload, _source):
        payload["evidence"]["holdout"] = payload["evidence"]["holdout"][:1]
        payload["evidence"]["stress"] = payload["evidence"]["stress"][:1]

    needs = _expand("robustification", mutate_input=truncate)[-1]["disposition"]

    def wrong_epoch(payload, _source):
        payload["evaluator"]["evaluator_epoch"] = "wrong"
        for rows in payload["evidence"].values():
            for row in rows:
                row["evaluator_epoch"] = "wrong"

    incomparable = _expand("robustification", mutate_input=wrong_epoch)[-1][
        "disposition"
    ]
    blocked = _expand("robustification", review_state="pending")[-1][
        "disposition"
    ]
    assert len({found, absent, needs, incomparable, blocked}) == 5


@pytest.mark.parametrize("reused_from", ("competition", "qualification"))
def test_old_observation_id_reuse_is_rejected(reused_from):
    source = _source("idealization")
    source_ids, qualification_ids = _all_observation_ids(source)
    reused_id = sorted(
        source_ids if reused_from == "competition" else qualification_ids
    )[0]
    probe_input = _probe_input("idealization", source)
    probe_input["evidence"]["holdout"][0]["observation_id"] = reused_id
    contract = _load(PROBE_CONTRACT)
    with pytest.raises(ValueError):
        expand_and_evaluate_shadow_child_failure_boundary_probe(
            *source,
            probe_input,
            contract,
            **_probe_kwargs(source, probe_input, contract),
        )


def test_duplicate_new_observation_id_is_rejected():
    source = _source("robustification")
    probe_input = _probe_input("robustification", source)
    probe_input["evidence"]["stress"][0]["observation_id"] = probe_input[
        "evidence"
    ]["holdout"][0]["observation_id"]
    contract = _load(PROBE_CONTRACT)
    with pytest.raises(ValueError):
        expand_and_evaluate_shadow_child_failure_boundary_probe(
            *source,
            probe_input,
            contract,
            **_probe_kwargs(source, probe_input, contract),
        )


@pytest.mark.parametrize("target", ("input", "row"))
def test_free_form_probe_injection_is_rejected_by_exact_key_validation(target):
    source = _source("idealization")
    probe_input = _probe_input("idealization", source)
    if target == "input":
        probe_input["probe_definition"] = {"execute": "arbitrary"}
    else:
        probe_input["evidence"]["holdout"][0]["probe"] = "arbitrary"
    contract = _load(PROBE_CONTRACT)
    with pytest.raises(ValueError):
        expand_and_evaluate_shadow_child_failure_boundary_probe(
            *source,
            probe_input,
            contract,
            **_probe_kwargs(source, probe_input, contract),
        )


def test_transition_registry_cannot_be_rewritten_or_extended():
    source = _source("robustification")
    probe_input = _probe_input("robustification", source)
    contract = _load(PROBE_CONTRACT)
    contract["transition_probe_registry"]["ROBUST_INTERVAL_EXPANSION"][
        "probe_id"
    ] = "caller_invented_probe"
    with pytest.raises(ValueError):
        expand_and_evaluate_shadow_child_failure_boundary_probe(
            *source,
            probe_input,
            contract,
            **_probe_kwargs(source, probe_input, contract),
        )


def test_epoch_and_expanded_state_are_independent_of_evidence_values():
    first = _expand("robustification", counterexample=False)[-1]
    second = _expand("robustification", counterexample=True)[-1]
    assert first["evaluator_definition"]["evaluator_epoch"] == second[
        "evaluator_definition"
    ]["evaluator_epoch"]
    assert first["probe_definition"] == second["probe_definition"]
    assert first["probe_expanded_shadow_theory_state_digest"] == second[
        "probe_expanded_shadow_theory_state_digest"
    ]
    assert first["report_digest"] != second["report_digest"]


@pytest.mark.parametrize("kind", ("robustification", "idealization"))
def test_evidence_row_order_invariance_and_canonical_replay(kind):
    first = _expand(kind)[-1]

    def reverse_rows(payload, _source):
        payload["evidence"]["holdout"].reverse()
        payload["evidence"]["stress"].reverse()

    second = _expand(kind, mutate_input=reverse_rows)[-1]
    # The raw input commitment records the caller's byte-semantic list order;
    # every probe/state/result semantic is nevertheless order invariant.
    ignored = {"probe_input_digest", "report_digest"}
    assert {key: value for key, value in first.items() if key not in ignored} == {
        key: value for key, value in second.items() if key not in ignored
    }
    replay = _expand(kind)[-1]
    assert first == replay
    assert canonical_json_bytes(first) == canonical_json_bytes(replay)


def test_source_child_is_byte_identical_and_all_action_bits_are_false():
    values = _expand("idealization")
    transition_report = values[4]
    report = values[-1]
    source_child = transition_report["child_theory_state"]
    assert _digest_value(source_child) == transition_report[
        "child_theory_state_digest"
    ]
    assert report["source_child_theory_state_digest"] == transition_report[
        "child_theory_state_digest"
    ]
    assessment = report["boundary_assessment"]
    for key in BOUNDARY_ACTION_KEYS:
        assert assessment[key] is False
    assert report["adoption_eligibility"] == (
        "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED"
    )
    assert report["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert report["promotion_status"] == "NOT_PROMOTED"
    assert report["current_status"] == "NOT_CURRENT"


def test_lifecycle_extension_consumes_new_rows_without_reenabling_old_records():
    *_, report = _expand("robustification")
    lifecycle = report["record_lifecycle_extension"]
    assert lifecycle["competition_records"]["role"] == (
        "AUDIT_ONLY_SCORING_EXCLUDED"
    )
    assert lifecycle["qualification_records"]["role"] == (
        "CONSUMED_AUDIT_ONLY_SCORING_EXCLUDED"
    )
    assert lifecycle["new_probe_records"]["role"] == (
        "CONSUMED_FAILURE_BOUNDARY_EVIDENCE_AUDIT_ONLY"
    )
    for key in (
        "competition_records",
        "qualification_records",
        "new_probe_records",
    ):
        assert lifecycle[key]["eligible_for_future_scoring"] is False
    future = lifecycle["future_scoring_policy"]
    assert future["new_unconsumed_evidence_required"] is True
    assert future["reuse_competition_records_allowed"] is False
    assert future["reuse_consumed_qualification_records_allowed"] is False
    assert future["reuse_consumed_probe_records_allowed"] is False
    assert future["cross_epoch_pooling_allowed"] is False
    assert lifecycle["logical_selective_erasure_applied"] is True
    assert lifecycle["physical_erasure"] == "NOT_PERFORMED"
    assert lifecycle["physical_retention_attestation"] == "REQUIRED_NOT_PRESENT"


def test_attestation_and_authority_remain_absent():
    *_, report = _expand("idealization")
    assert report["attestation_boundary"] == {
        "external_data_attestation": "REQUIRED_NOT_PRESENT",
        "external_evaluator_attestation": "REQUIRED_NOT_PRESENT",
        "physical_retention_attestation": "REQUIRED_NOT_PRESENT",
        "external_adoption_authority": "REQUIRED_NOT_PRESENT",
    }


@pytest.mark.parametrize(
    "target",
    (
        ("probe_expanded_shadow_theory_state", "probe_ids"),
        ("probe_expanded_shadow_theory_state", "model_class"),
        ("probe_expanded_shadow_theory_state", "adoption_status"),
        ("probe_expanded_shadow_theory_state", "current_status"),
        ("boundary_assessment", "rollback_execution_allowed"),
        ("adoption_status",),
        ("promotion_status",),
        ("current_status",),
    ),
)
def test_report_semantic_tamper_and_rehash_fails_exact_verifier(target):
    values = _expand("robustification")
    source = values[:10]
    probe_input, contract, report = values[10], values[11], values[-1]
    tampered = copy.deepcopy(report)
    cursor = tampered
    for key in target[:-1]:
        cursor = cursor[key]
    key = target[-1]
    if key == "probe_ids":
        cursor[key].append("second_free_form_probe")
    elif key == "model_class":
        cursor[key] = {"kind": "forged"}
    elif key.endswith("allowed"):
        cursor[key] = True
    else:
        cursor[key] = "FORGED"
    _rehash(tampered)
    with pytest.raises(ValueError):
        _verify(
            source,
            probe_input,
            contract,
            tampered,
            expected_probe_report_digest=tampered["report_digest"],
        )


def test_forged_review_packet_readiness_with_recomputed_digest_fails_upstream_replay():
    source = list(_source("robustification", "pending"))
    forged = copy.deepcopy(source[9])
    forged["disposition"] = "READY_FOR_EXTERNAL_REVIEW_PACKET_ONLY"
    forged["review_boundary"]["packet_ready"] = True
    source[9] = _rehash(forged)
    source = tuple(source)
    probe_input = _probe_input("robustification", source)
    contract = _load(PROBE_CONTRACT)
    kwargs = _probe_kwargs(source, probe_input, contract)
    kwargs["expected_review_report_digest"] = forged["report_digest"]
    with pytest.raises(ValueError):
        expand_and_evaluate_shadow_child_failure_boundary_probe(
            *source,
            probe_input,
            contract,
            **kwargs,
        )


@pytest.mark.parametrize(
    "key",
    (
        "expected_competition_contract_digest",
        "expected_competition_report_digest",
        "expected_transition_contract_digest",
        "expected_transition_report_digest",
        "expected_qualification_input_digest",
        "expected_qualification_contract_digest",
        "expected_qualification_report_digest",
        "expected_review_contract_digest",
        "expected_review_report_digest",
        "expected_probe_input_digest",
        "expected_probe_contract_digest",
    ),
)
def test_all_eleven_independent_digest_anchors_fail_closed(key):
    source = _source("idealization")
    probe_input = _probe_input("idealization", source)
    contract = _load(PROBE_CONTRACT)
    kwargs = _probe_kwargs(source, probe_input, contract)
    kwargs[key] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        expand_and_evaluate_shadow_child_failure_boundary_probe(
            *source,
            probe_input,
            contract,
            **kwargs,
        )


def test_verifier_rejects_independently_mismatched_probe_artifact_map():
    values = _expand("idealization")
    source = values[:10]
    probe_input, contract, report = values[10], values[11], values[-1]
    with pytest.raises(ValueError):
        _verify(
            source,
            probe_input,
            contract,
            report,
            expected_probe_input_artifacts={"forged": True},
        )


def import_runner():
    spec = importlib.util.spec_from_file_location("probe_runner", PROBE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _digest_flags(values):
    return [
        "--expected-competition-contract-digest",
        values["competition_contract"],
        "--expected-competition-report-digest",
        values["competition_report"],
        "--expected-transition-contract-digest",
        values["transition_contract"],
        "--expected-transition-report-digest",
        values["transition_report"],
        "--expected-qualification-input-digest",
        values["qualification_input"],
        "--expected-qualification-contract-digest",
        values["qualification_contract"],
        "--expected-qualification-report-digest",
        values["qualification_report"],
        "--expected-review-contract-digest",
        values["review_contract"],
        "--expected-review-report-digest",
        values["review_report"],
        "--expected-probe-input-digest",
        values["probe_input"],
        "--expected-probe-contract-digest",
        values["probe_contract"],
    ]


def _input_flags(paths):
    return [
        "--competition-input", str(paths[0]),
        "--competition-contract", str(paths[1]),
        "--competition-report", str(paths[2]),
        "--transition-contract", str(paths[3]),
        "--transition-report", str(paths[4]),
        "--qualification-input", str(paths[5]),
        "--qualification-contract", str(paths[6]),
        "--qualification-report", str(paths[7]),
        "--review-contract", str(paths[8]),
        "--review-report", str(paths[9]),
        "--probe-input", str(paths[10]),
        "--probe-contract", str(paths[11]),
    ]


def test_cli_canonical_stdout_atomic_out_exact_12_artifacts_and_inputs_unchanged(
    tmp_path,
):
    case_path = (tmp_path / "competition-input.json").resolve()
    competition_report_path = (tmp_path / "competition-report.json").resolve()
    transition_report_path = (tmp_path / "transition-report.json").resolve()
    qualification_input_path = (tmp_path / "qualification-input.json").resolve()
    qualification_report_path = (tmp_path / "qualification-report.json").resolve()
    review_report_path = (tmp_path / "review-report.json").resolve()
    probe_input_path = (tmp_path / "probe-input.json").resolve()
    probe_report_path = (tmp_path / "probe-report.json").resolve()

    _write(case_path, review_kat.qualification_kat._case("idealization"))
    competition = subprocess.run(
        [
            sys.executable,
            str(COMPETITION_RUNNER),
            "--input",
            str(case_path),
            "--contract",
            str(COMPETITION_CONTRACT.resolve()),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert competition.returncode == 0, competition.stderr.decode()
    competition_report_path.write_bytes(competition.stdout)
    competition_input = _load(case_path)
    competition_contract = _load(COMPETITION_CONTRACT)
    competition_report = json.loads(competition.stdout)
    transition_contract = _load(TRANSITION_CONTRACT)

    transition = subprocess.run(
        [
            sys.executable,
            str(TRANSITION_RUNNER),
            "--competition-input",
            str(case_path),
            "--competition-contract",
            str(COMPETITION_CONTRACT.resolve()),
            "--competition-report",
            str(competition_report_path),
            "--transition-contract",
            str(TRANSITION_CONTRACT.resolve()),
            "--expected-competition-contract-digest",
            _digest_value(competition_contract),
            "--expected-competition-report-digest",
            competition_report["report_digest"],
            "--expected-transition-contract-digest",
            _digest_value(transition_contract),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert transition.returncode == 0, transition.stderr.decode()
    transition_report_path.write_bytes(transition.stdout)
    transition_report = json.loads(transition.stdout)

    chain = (
        competition_input,
        competition_contract,
        competition_report,
        transition_contract,
        transition_report,
    )
    qualification_input = review_kat.qualification_kat._qualification_input(
        "idealization", chain
    )
    _write(qualification_input_path, qualification_input)
    qualification_contract = _load(QUALIFICATION_CONTRACT)
    qualification = subprocess.run(
        [
            sys.executable,
            str(QUALIFICATION_RUNNER),
            "--competition-input",
            str(case_path),
            "--competition-contract",
            str(COMPETITION_CONTRACT.resolve()),
            "--competition-report",
            str(competition_report_path),
            "--transition-contract",
            str(TRANSITION_CONTRACT.resolve()),
            "--transition-report",
            str(transition_report_path),
            "--qualification-input",
            str(qualification_input_path),
            "--qualification-contract",
            str(QUALIFICATION_CONTRACT.resolve()),
            "--expected-competition-contract-digest",
            _digest_value(competition_contract),
            "--expected-competition-report-digest",
            competition_report["report_digest"],
            "--expected-transition-contract-digest",
            _digest_value(transition_contract),
            "--expected-transition-report-digest",
            transition_report["report_digest"],
            "--expected-qualification-input-digest",
            _digest_value(qualification_input),
            "--expected-qualification-contract-digest",
            _digest_value(qualification_contract),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert qualification.returncode == 0, qualification.stderr.decode()
    qualification_report_path.write_bytes(qualification.stdout)
    qualification_report = json.loads(qualification.stdout)
    review_contract = _load(REVIEW_CONTRACT)

    review_values = {
        "competition_contract": _digest_value(competition_contract),
        "competition_report": competition_report["report_digest"],
        "transition_contract": _digest_value(transition_contract),
        "transition_report": transition_report["report_digest"],
        "qualification_input": _digest_value(qualification_input),
        "qualification_contract": _digest_value(qualification_contract),
        "qualification_report": qualification_report["report_digest"],
        "review_contract": _digest_value(review_contract),
    }
    review = subprocess.run(
        [
            sys.executable,
            str(REVIEW_RUNNER),
            "--competition-input",
            str(case_path),
            "--competition-contract",
            str(COMPETITION_CONTRACT.resolve()),
            "--competition-report",
            str(competition_report_path),
            "--transition-contract",
            str(TRANSITION_CONTRACT.resolve()),
            "--transition-report",
            str(transition_report_path),
            "--qualification-input",
            str(qualification_input_path),
            "--qualification-contract",
            str(QUALIFICATION_CONTRACT.resolve()),
            "--qualification-report",
            str(qualification_report_path),
            "--review-contract",
            str(REVIEW_CONTRACT.resolve()),
            *review_kat._digest_flags(review_values),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert review.returncode == 0, review.stderr.decode()
    review_report_path.write_bytes(review.stdout)
    review_report = json.loads(review.stdout)
    source = (
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
    )
    probe_input = _probe_input("idealization", source)
    _write(probe_input_path, probe_input)
    probe_contract = _load(PROBE_CONTRACT)

    inputs = (
        case_path,
        COMPETITION_CONTRACT.resolve(),
        competition_report_path,
        TRANSITION_CONTRACT.resolve(),
        transition_report_path,
        qualification_input_path,
        QUALIFICATION_CONTRACT.resolve(),
        qualification_report_path,
        REVIEW_CONTRACT.resolve(),
        review_report_path,
        probe_input_path,
        PROBE_CONTRACT.resolve(),
    )
    before = {path: path.read_bytes() for path in inputs}
    values = {
        **review_values,
        "review_report": review_report["report_digest"],
        "probe_input": _digest_value(probe_input),
        "probe_contract": _digest_value(probe_contract),
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(PROBE_RUNNER),
            *_input_flags(inputs),
            *_digest_flags(values),
            "--out",
            str(probe_report_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    parsed = json.loads(completed.stdout)
    assert completed.stdout == canonical_json_bytes(parsed) + b"\n"
    assert probe_report_path.read_bytes() == completed.stdout
    assert {path: path.read_bytes() for path in inputs} == before
    assert set(parsed["input_artifacts"]) == {
        "competition_contract_json",
        "competition_input_json",
        "competition_report_json",
        "probe_contract_json",
        "probe_input_json",
        "qualification_contract_json",
        "qualification_input_json",
        "qualification_report_json",
        "review_contract_json",
        "review_report_json",
        "transition_contract_json",
        "transition_report_json",
    }


@pytest.mark.parametrize(
    "raw",
    (
        b'{"probe":"a","probe":"b"}\n',
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
        b'[]\n',
    ),
)
def test_cli_rejects_duplicate_nonfinite_and_nonobject_json_before_core(raw, tmp_path):
    paths = [(tmp_path / f"input-{index}.json").resolve() for index in range(12)]
    for path in paths:
        path.write_text("{}\n", encoding="utf-8")
    paths[0].write_bytes(raw)
    zero = "sha256:" + "0" * 64
    values = {
        key: zero
        for key in (
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
        )
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(PROBE_RUNNER),
            *_input_flags(paths),
            *_digest_flags(values),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert completed.stdout == b""


def test_cli_rejects_relative_symlink_and_output_aliases(tmp_path):
    paths = [(tmp_path / f"input-{index}.json").resolve() for index in range(12)]
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


def test_all_66_input_path_and_hardlink_alias_pairs_are_rejected(tmp_path):
    runner = import_runner()
    checked_same = 0
    checked_hard = 0
    for first in range(12):
        for second in range(first + 1, 12):
            pair_root = tmp_path / f"pair-{first}-{second}"
            pair_root.mkdir()
            paths = []
            for index in range(12):
                path = pair_root / f"input-{index}.json"
                path.write_text("{}\n", encoding="utf-8")
                paths.append(path.resolve())
            same = list(paths)
            same[second] = same[first]
            with pytest.raises(ValueError):
                runner._require_distinct_inputs(tuple(same))
            checked_same += 1
            paths[second].unlink()
            os.link(paths[first], paths[second])
            with pytest.raises(ValueError):
                runner._require_distinct_inputs(tuple(paths))
            checked_hard += 1
    assert checked_same == 66
    assert checked_hard == 66


def test_surface_has_no_execution_or_network_path_and_preserves_prior_bytes():
    before = {
        path: _digest_file(path) for path in (*PREVIOUS_SLICE_FILES, OLD_BENCHMARK)
    }
    for path in (PROBE_CORE, PROBE_RUNNER):
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
        forbidden_imports = {
            "performance.benchmark_lodo_meta_prior",
            "requests",
            "socket",
            "urllib",
            "subprocess",
        }
        assert imported.isdisjoint(forbidden_imports)
        assert "run_one" not in calls
        assert "urlopen" not in calls
        assert "connect" not in calls

    _expand("robustification")
    _expand("idealization")
    assert {path: _digest_file(path) for path in before} == before


def test_documentation_states_bounded_non_authoritative_probe_boundary():
    text = PROBE_DOC.read_text(encoding="utf-8")
    assert "Probe expansion is not external probe acquisition" in text
    assert "caller-supplied static" in text
    assert "READY_FOR_EXTERNAL_REVIEW_PACKET_ONLY" in text
    assert "LOCAL_FAILURE_BOUNDARY_PROBE_EPOCH_UNATTESTED" in text
    assert "Within each registered scope and nontrivial quotient fiber" in text
    assert "different scopes are never pooled" in text
    assert "require_complete_parent_context_scope_pairs_per_split" in text
    assert "a sparse cross can conceal" in text
    assert "violation_functionals` remains canonical" in text
    assert "this slice expands \\(Q\\) only" in text
    assert "counterexample is a bounded" in text
    assert "physical_erasure = NOT_PERFORMED" in text
    assert "NOT_ADOPTED_SHADOW_ONLY" in text
    assert "NOT_CURRENT" in text
    assert "all 66 input pairs" in text
