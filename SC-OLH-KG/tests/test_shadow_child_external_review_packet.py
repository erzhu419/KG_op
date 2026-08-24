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

import test_shadow_child_probe_qualification as qualification_kat  # noqa: E402

from performance.shadow_child_external_review_packet import (  # noqa: E402
    build_shadow_child_external_review_packet,
    validate_shadow_child_external_review_packet_contract,
    verify_shadow_child_external_review_packet,
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
COMPETITION_RUNNER = ROOT / "runners/run_theory_operation_competition.py"
TRANSITION_RUNNER = ROOT / "runners/run_shadow_theory_transition.py"
QUALIFICATION_RUNNER = ROOT / "runners/run_shadow_child_probe_qualification.py"
REVIEW_RUNNER = ROOT / "runners/run_shadow_child_external_review_packet.py"
REVIEW_CORE = ROOT / "performance/shadow_child_external_review_packet.py"
REVIEW_DOC = ROOT / "docs/shadow_child_external_review_packet_v1.md"
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
)

REPORT_KEYS = {
    "schema_version",
    "contract_id",
    "contract_digest",
    "packet_id",
    "source_qualification",
    "child_theory_state_digest",
    "parent_theory_state_digest",
    "transition_kind",
    "evaluator_epoch",
    "review_checks",
    "record_lifecycle_boundary",
    "rollback_boundary",
    "selective_erasure_boundary",
    "attestation_boundary",
    "disposition",
    "review_boundary",
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
SOURCE_QUALIFICATION_KEYS = {
    "verification_status",
    "contract_id",
    "contract_digest",
    "report_schema_version",
    "report_digest",
    "qualification_input_digest",
    "qualification_disposition",
    "source_transition_report_digest",
    "child_theory_state_digest",
    "evaluator_epoch",
    "adoption_status",
}
REVIEW_CHECK_KEYS = {
    "source_qualification_exact_replay_verified",
    "source_transition_materialized",
    "child_digest_bound",
    "qualified_new_evaluator_epoch",
    "fresh_epoch_comparable",
    "evidence_sufficient",
    "operational_probe_results_present",
    "all_operational_probe_gates_passed",
    "source_evidence_scoring_excluded",
    "old_new_pooling_forbidden",
    "logical_selective_erasure_applied",
    "physical_deletion_absent",
    "parent_rollback_binding_present",
    "original_child_unmodified",
    "upstream_adoption_withheld",
    "record_lifecycle_boundary_bound",
    "local_packet_complete",
}
LIFECYCLE_KEYS = {
    "boundary_schema_version",
    "parent_snapshot",
    "child_snapshot",
    "source_competition_records",
    "qualification_records",
    "commitment_chain",
    "future_scoring_policy",
    "retention_requirements_bound",
    "physical_retention_attestation",
}
PARENT_SNAPSHOT_KEYS = {
    "theory_state_digest",
    "role",
    "retention_required",
    "eligible_for_scoring",
}
CHILD_SNAPSHOT_KEYS = {
    "theory_state_digest",
    "role",
    "retention_required",
    "adopted",
    "current",
}
RECORD_KEYS = {
    "evidence_digests",
    "observation_id_digest",
    "observation_count",
    "evaluator_epoch",
    "role",
    "retain_commitments_for_audit",
    "eligible_for_future_scoring",
}
COMMITMENT_KEYS = {
    "competition_contract_digest",
    "competition_report_digest",
    "transition_contract_digest",
    "transition_report_digest",
    "qualification_contract_digest",
    "qualification_report_digest",
    "review_contract_digest",
}
FUTURE_SCORING_KEYS = {
    "new_unconsumed_evidence_required",
    "reuse_source_records_allowed",
    "reuse_consumed_qualification_records_allowed",
    "cross_epoch_pooling_allowed",
}
ROLLBACK_KEYS = {
    "parent_theory_state_digest",
    "child_theory_state_digest",
    "transition_kind",
    "reduction_certificate_digest",
    "parent_snapshot_digest",
    "rollback_method",
    "rollback_binding_verified",
    "rollback_execution_status",
}
SELECTIVE_KEYS = {
    "mode",
    "source_competition_records_excluded_from_active_scoring",
    "consumed_qualification_records_excluded_from_future_rescoring",
    "future_scoring_requires_new_unconsumed_evidence",
    "cross_epoch_pooling_allowed",
    "logical_boundary_bound",
    "physical_erasure_status",
}
ATTESTATION_KEYS = {
    "external_data_attestation",
    "external_evaluator_attestation",
    "physical_retention_attestation",
    "physical_erasure",
    "external_adoption_authority",
}
REVIEW_BOUNDARY_KEYS = {
    "scope",
    "packet_ready",
    "external_review_required",
    "adoption_decision_allowed",
    "promotion_decision_allowed",
    "current_pointer_write_allowed",
    "parent_or_child_state_write_allowed",
}
VERIFIER_RECEIPT_KEYS = {
    "status",
    "disposition",
    "report_digest",
    "contract_digest",
    "packet_id",
    "source_qualification_report_digest",
    "child_theory_state_digest",
    "ready_for_external_review",
    "adoption_eligibility",
    "adoption_status",
    "promotion_status",
    "current_status",
}

MANDATORY_NONCLAIMS = [
    "shadow_only",
    "external_review_packet_only",
    "adoption_eligibility_not_determined",
    "no_automatic_adoption",
    "no_promotion",
    "no_current_pointer_write",
    "no_parent_child_or_ambient_state_write",
    "no_operations_research_baseline_or_claim_change",
    "no_paper_promotion",
    "explicit_cli_out_is_only_optional_write",
    "no_run_one",
    "no_benchmark_execution",
    "no_scheduler_access",
    "no_network_access",
    "no_new_theory_operation",
    "no_restrict_operation",
    "no_probe_observation_expansion_or_execution",
    "no_language_or_predicate_invention",
    "no_source_evidence_rescoring",
    "no_consumed_qualification_evidence_rescoring",
    "no_cross_epoch_pooling",
    "logical_selective_erasure_boundary_only",
    "no_physical_erasure",
    "physical_retention_attestation_required_not_present",
    "external_data_attestation_required_not_present",
    "external_evaluator_attestation_required_not_present",
    "external_adoption_authority_required_not_present",
    "rollback_binding_is_not_rollback_execution",
    "retention_requirements_bound_not_executed",
    "input_artifacts_require_independent_expected_values",
    "report_digest_is_not_a_signature",
    "no_data_provenance_beyond_bound_artifacts",
    "no_external_review_outcome",
    "no_scientific_validity_or_generalization_claim",
    "no_h_t_to_h_t_plus_1_acceptance",
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


def _qualification_bundle(kind, state="ready"):
    def mutate(payload, _chain):
        if state == "pending":
            payload["evidence"]["holdout"] = payload["evidence"]["holdout"][:1]
            payload["evidence"]["stress"] = payload["evidence"]["stress"][:1]
        elif state == "incomparable":
            payload["evidence"]["stress"][0]["evaluator_epoch"] = "other-epoch"
        elif state == "failed":
            for rows in payload["evidence"].values():
                for row in rows:
                    row["observed_value"] = 1000.0
        elif state != "ready":
            raise AssertionError(state)

    return qualification_kat._qualify(
        kind, mutate_input=None if state == "ready" else mutate
    )


def _review_kwargs(bundle, *, input_artifacts=None):
    case, competition_contract, competition_report, transition_contract, transition_report, qualification_input, qualification_contract, qualification_report = bundle
    review_contract = _load(REVIEW_CONTRACT)
    return {
        "expected_competition_contract_digest": _digest_value(
            competition_contract
        ),
        "expected_competition_report_digest": competition_report["report_digest"],
        "expected_competition_input_artifacts": None,
        "expected_transition_contract_digest": _digest_value(transition_contract),
        "expected_transition_report_digest": transition_report["report_digest"],
        "expected_transition_input_artifacts": None,
        "expected_qualification_input_digest": _digest_value(qualification_input),
        "expected_qualification_contract_digest": _digest_value(
            qualification_contract
        ),
        "expected_qualification_report_digest": qualification_report[
            "report_digest"
        ],
        "expected_qualification_input_artifacts": None,
        "expected_review_contract_digest": _digest_value(review_contract),
        "input_artifacts": input_artifacts,
    }


def _build_from_bundle(bundle, *, input_artifacts=None):
    review_contract = _load(REVIEW_CONTRACT)
    result = build_shadow_child_external_review_packet(
        *bundle,
        review_contract,
        **_review_kwargs(bundle, input_artifacts=input_artifacts),
    )
    return review_contract, result, _as_dict(result)


def _review(kind, state="ready"):
    bundle = _qualification_bundle(kind, state)
    review_contract, result, report = _build_from_bundle(bundle)
    return (*bundle, review_contract, result, report)


def _verify(bundle, review_contract, report, **overrides):
    kwargs = _review_kwargs(bundle)
    kwargs.pop("input_artifacts")
    kwargs.update(
        {
            "expected_review_report_digest": report["report_digest"],
            "expected_review_input_artifacts": None,
        }
    )
    kwargs.update(overrides)
    return verify_shadow_child_external_review_packet(
        *bundle, review_contract, report, **kwargs
    )


def _rehash(report):
    report["report_digest"] = _digest_value(
        {key: value for key, value in report.items() if key != "report_digest"}
    )
    return report


def _contains_key(value, target):
    if isinstance(value, dict):
        return target in value or any(_contains_key(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, target) for item in value)
    return False


def test_frozen_contract_and_exact_report_nested_vocabulary():
    contract = _load(REVIEW_CONTRACT)
    assert validate_shadow_child_external_review_packet_contract(contract) == contract
    assert contract["contract_id"] == "shadow_child_external_review_packet_v1"
    assert contract["source_qualification_contract_digest"] == (
        "sha256:593b8727f7f985cea82ae86e0758a67018c8d6a0a7c1dc0d518bcc3615f7d4ee"
    )
    assert contract["nonclaims"] == MANDATORY_NONCLAIMS

    *_, report = _review("robustification")
    assert set(report) == REPORT_KEYS
    assert set(report["source_qualification"]) == SOURCE_QUALIFICATION_KEYS
    assert set(report["review_checks"]) == REVIEW_CHECK_KEYS
    lifecycle = report["record_lifecycle_boundary"]
    assert set(lifecycle) == LIFECYCLE_KEYS
    assert set(lifecycle["parent_snapshot"]) == PARENT_SNAPSHOT_KEYS
    assert set(lifecycle["child_snapshot"]) == CHILD_SNAPSHOT_KEYS
    assert set(lifecycle["source_competition_records"]) == RECORD_KEYS
    assert set(lifecycle["qualification_records"]) == RECORD_KEYS
    assert set(lifecycle["commitment_chain"]) == COMMITMENT_KEYS
    assert set(lifecycle["future_scoring_policy"]) == FUTURE_SCORING_KEYS
    assert set(report["rollback_boundary"]) == ROLLBACK_KEYS
    assert set(report["selective_erasure_boundary"]) == SELECTIVE_KEYS
    assert set(report["attestation_boundary"]) == ATTESTATION_KEYS
    assert set(report["review_boundary"]) == REVIEW_BOUNDARY_KEYS
    assert report["nonclaims"] == MANDATORY_NONCLAIMS


@pytest.mark.parametrize("kind", ("robustification", "idealization"))
def test_qualified_child_yields_review_ready_packet_only(kind):
    *_, result, report = _review(kind)
    assert result.ready_for_external_review is True
    assert not hasattr(result, "eligible")
    assert report["disposition"] == "READY_FOR_EXTERNAL_REVIEW_PACKET_ONLY"
    assert report["review_boundary"]["packet_ready"] is True
    assert report["review_checks"]["local_packet_complete"] is True
    assert all(report["review_checks"].values())
    assert report["adoption_eligibility"] == (
        "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED"
    )
    assert report["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert report["promotion_status"] == "NOT_PROMOTED"
    assert report["current_status"] == "NOT_CURRENT"
    assert report["review_boundary"]["adoption_decision_allowed"] is False
    assert report["review_boundary"]["promotion_decision_allowed"] is False
    assert report["review_boundary"]["current_pointer_write_allowed"] is False


@pytest.mark.parametrize(
    ("state", "expected"),
    (
        ("pending", "REVIEW_PACKET_PENDING_NEW_EVIDENCE"),
        ("incomparable", "REVIEW_PACKET_BLOCKED_INCOMPARABLE_EPOCH"),
        ("failed", "REVIEW_PACKET_BLOCKED_PROBE_FAILURE"),
    ),
)
def test_nonqualified_source_maps_to_complete_but_not_ready_packet(state, expected):
    *_, report = _review("robustification", state)
    assert report["disposition"] == expected
    assert report["review_checks"]["local_packet_complete"] is True
    assert report["review_boundary"]["packet_ready"] is False
    assert report["adoption_eligibility"] == (
        "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED"
    )
    assert report["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert report["promotion_status"] == "NOT_PROMOTED"
    assert report["current_status"] == "NOT_CURRENT"


def test_record_lifecycle_consumes_both_old_and_qualification_records():
    *prefix, qualification_report, _, _, report = _review("idealization")
    lifecycle = report["record_lifecycle_boundary"]
    assert lifecycle["boundary_schema_version"] == (
        "sc-olh-kg.shadow-child-record-lifecycle-boundary/1"
    )
    assert lifecycle["parent_snapshot"]["role"] == "ROLLBACK_REQUIRED"
    assert lifecycle["child_snapshot"]["role"] == "SHADOW_REVIEW_CANDIDATE"
    source = lifecycle["source_competition_records"]
    consumed = lifecycle["qualification_records"]
    assert source["role"] == "AUDIT_ONLY_SOURCE_SCORING_EXCLUDED"
    assert consumed["role"] == "CONSUMED_QUALIFICATION_EVIDENCE_AUDIT_ONLY"
    assert source["eligible_for_future_scoring"] is False
    assert consumed["eligible_for_future_scoring"] is False
    assert source["observation_id_digest"] == qualification_report[
        "evidence_binding"
    ]["source_observation_id_digest"]
    assert consumed["observation_id_digest"] == qualification_report[
        "evidence_binding"
    ]["new_observation_id_digest"]
    assert lifecycle["future_scoring_policy"] == {
        "new_unconsumed_evidence_required": True,
        "reuse_source_records_allowed": False,
        "reuse_consumed_qualification_records_allowed": False,
        "cross_epoch_pooling_allowed": False,
    }
    assert lifecycle["retention_requirements_bound"] is True
    assert lifecycle["physical_retention_attestation"] == "REQUIRED_NOT_PRESENT"


@pytest.mark.parametrize(
    ("kind", "method"),
    (
        (
            "robustification",
            "COLLAPSE_INTERVAL_AT_RADIUS_MULTIPLIER_ZERO",
        ),
        ("idealization", "RESTORE_FROZEN_PARENT_POINT_TABLE"),
    ),
)
def test_rollback_is_bound_but_never_executed(kind, method):
    *_, report = _review(kind)
    rollback = report["rollback_boundary"]
    assert rollback["rollback_method"] == method
    assert rollback["rollback_binding_verified"] is True
    assert rollback["rollback_execution_status"] == "NOT_PERFORMED"
    assert rollback["parent_snapshot_digest"] == report[
        "parent_theory_state_digest"
    ]


def test_attestation_and_selective_erasure_boundary_are_non_authoritative():
    *_, report = _review("robustification")
    assert report["attestation_boundary"] == {
        "external_data_attestation": "REQUIRED_NOT_PRESENT",
        "external_evaluator_attestation": "REQUIRED_NOT_PRESENT",
        "physical_retention_attestation": "REQUIRED_NOT_PRESENT",
        "physical_erasure": "NOT_PERFORMED",
        "external_adoption_authority": "REQUIRED_NOT_PRESENT",
    }
    erasure = report["selective_erasure_boundary"]
    assert erasure["source_competition_records_excluded_from_active_scoring"] is True
    assert erasure[
        "consumed_qualification_records_excluded_from_future_rescoring"
    ] is True
    assert erasure["future_scoring_requires_new_unconsumed_evidence"] is True
    assert erasure["cross_epoch_pooling_allowed"] is False
    assert erasure["logical_boundary_bound"] is True
    assert erasure["physical_erasure_status"] == "NOT_PERFORMED"


def test_packet_contains_no_observed_values_or_probe_result_rescoring():
    *_, report = _review("robustification")
    assert _contains_key(report, "observed_value") is False
    assert _contains_key(report, "probe_results") is False
    assert _contains_key(report, "gate_checks") is False
    assert _contains_key(report, "mae") is False
    assert _contains_key(report, "interval_coverage") is False


def test_public_verifier_exact_receipt_and_non_authoritative_status():
    *bundle, review_contract, _, report = _review("idealization")
    receipt = _verify(tuple(bundle), review_contract, report)
    assert set(receipt) == VERIFIER_RECEIPT_KEYS
    assert receipt["status"] == "VERIFIED_" + report["disposition"]
    assert receipt["ready_for_external_review"] is True
    assert receipt["adoption_eligibility"] == (
        "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED"
    )
    assert receipt["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert receipt["promotion_status"] == "NOT_PROMOTED"
    assert receipt["current_status"] == "NOT_CURRENT"


@pytest.mark.parametrize(
    "target", ("competition_report", "transition_report", "qualification_report")
)
def test_forged_upstream_report_with_recomputed_digest_fails_exact_replay(target):
    bundle = list(_qualification_bundle("idealization"))
    index = {
        "competition_report": 2,
        "transition_report": 4,
        "qualification_report": 7,
    }[target]
    forged = copy.deepcopy(bundle[index])
    if target == "competition_report":
        forged["promotion_status"] = "FORGED"
    elif target == "transition_report":
        forged["adoption_status"] = "FORGED"
    else:
        forged["qualification_binding"]["original_child_state_mutated"] = True
    bundle[index] = _rehash(forged)
    contract = _load(REVIEW_CONTRACT)
    kwargs = _review_kwargs(tuple(bundle))
    if target == "competition_report":
        kwargs["expected_competition_report_digest"] = forged["report_digest"]
    elif target == "transition_report":
        kwargs["expected_transition_report_digest"] = forged["report_digest"]
    else:
        kwargs["expected_qualification_report_digest"] = forged["report_digest"]
    with pytest.raises(ValueError):
        build_shadow_child_external_review_packet(
            *bundle, contract, **kwargs
        )


@pytest.mark.parametrize(
    ("target", "replacement"),
    (
        (("review_checks", "local_packet_complete"), False),
        (("record_lifecycle_boundary", "retention_requirements_bound"), False),
        (("rollback_boundary", "rollback_execution_status"), "PERFORMED"),
        (("selective_erasure_boundary", "physical_erasure_status"), "PERFORMED"),
        (("attestation_boundary", "external_data_attestation"), "PRESENT"),
        (("review_boundary", "adoption_decision_allowed"), True),
        (("adoption_eligibility",), "ELIGIBLE"),
        (("adoption_status",), "ADOPTED"),
        (("promotion_status",), "PROMOTED"),
        (("current_status",), "CURRENT"),
    ),
)
def test_packet_semantic_tamper_and_rehash_fails_verifier(target, replacement):
    *bundle, review_contract, _, report = _review("robustification")
    tampered = copy.deepcopy(report)
    cursor = tampered
    for key in target[:-1]:
        cursor = cursor[key]
    cursor[target[-1]] = replacement
    _rehash(tampered)
    with pytest.raises(ValueError):
        _verify(
            tuple(bundle),
            review_contract,
            tampered,
            expected_review_report_digest=tampered["report_digest"],
        )


def test_contract_cannot_claim_present_attestation_or_authority():
    bundle = _qualification_bundle("robustification")
    contract = _load(REVIEW_CONTRACT)
    contract["authority_boundary"]["external_adoption_authority"] = "PRESENT"
    with pytest.raises(ValueError):
        build_shadow_child_external_review_packet(
            *bundle,
            contract,
            **{
                **_review_kwargs(bundle),
                "expected_review_contract_digest": _digest_value(contract),
            },
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
    ),
)
def test_all_eight_independent_digest_anchors_fail_closed(key):
    bundle = _qualification_bundle("idealization")
    contract = _load(REVIEW_CONTRACT)
    kwargs = _review_kwargs(bundle)
    kwargs[key] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        build_shadow_child_external_review_packet(
            *bundle, contract, **kwargs
        )


def test_verifier_rejects_independently_mismatched_review_artifact_map():
    *bundle, review_contract, _, report = _review("idealization")
    with pytest.raises(ValueError):
        _verify(
            tuple(bundle),
            review_contract,
            report,
            expected_review_input_artifacts={"forged": True},
        )


def _run_competition_cli(case_path):
    return subprocess.run(
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
    ]


def test_cli_canonical_stdout_atomic_out_exact_artifacts_and_inputs_unchanged(tmp_path):
    case_path = (tmp_path / "case.json").resolve()
    competition_report_path = (tmp_path / "competition-report.json").resolve()
    transition_report_path = (tmp_path / "transition-report.json").resolve()
    qualification_input_path = (tmp_path / "qualification-input.json").resolve()
    qualification_report_path = (tmp_path / "qualification-report.json").resolve()
    review_report_path = (tmp_path / "review-report.json").resolve()
    _write(case_path, qualification_kat._case("idealization"))

    competition = _run_competition_cli(case_path)
    assert competition.returncode == 0, competition.stderr.decode()
    competition_report_path.write_bytes(competition.stdout)
    competition_report = json.loads(competition.stdout)
    competition_contract = _load(COMPETITION_CONTRACT)
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
        _load(case_path),
        competition_contract,
        competition_report,
        transition_contract,
        transition_report,
    )
    qualification_input = qualification_kat._qualification_input(
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
    )
    before = {path: path.read_bytes() for path in inputs}
    values = {
        "competition_contract": _digest_value(competition_contract),
        "competition_report": competition_report["report_digest"],
        "transition_contract": _digest_value(transition_contract),
        "transition_report": transition_report["report_digest"],
        "qualification_input": _digest_value(qualification_input),
        "qualification_contract": _digest_value(qualification_contract),
        "qualification_report": qualification_report["report_digest"],
        "review_contract": _digest_value(review_contract),
    }
    completed = subprocess.run(
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
            *_digest_flags(values),
            "--out",
            str(review_report_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    parsed = json.loads(completed.stdout)
    assert completed.stdout == canonical_json_bytes(parsed) + b"\n"
    assert review_report_path.read_bytes() == completed.stdout
    assert {path: path.read_bytes() for path in inputs} == before
    assert set(parsed["input_artifacts"]) == {
        "competition_contract_json",
        "competition_input_json",
        "competition_report_json",
        "qualification_contract_json",
        "qualification_input_json",
        "qualification_report_json",
        "review_contract_json",
        "transition_contract_json",
        "transition_report_json",
    }


@pytest.mark.parametrize(
    "raw",
    (
        b'{"packet":"a","packet":"b"}\n',
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
    ),
)
def test_cli_rejects_duplicate_and_nonfinite_json_before_core(raw, tmp_path):
    paths = [(tmp_path / f"input-{index}.json").resolve() for index in range(9)]
    for path in paths:
        path.write_text("{}\n", encoding="utf-8")
    paths[0].write_bytes(raw)
    flags = [
        "--competition-input", str(paths[0]),
        "--competition-contract", str(paths[1]),
        "--competition-report", str(paths[2]),
        "--transition-contract", str(paths[3]),
        "--transition-report", str(paths[4]),
        "--qualification-input", str(paths[5]),
        "--qualification-contract", str(paths[6]),
        "--qualification-report", str(paths[7]),
        "--review-contract", str(paths[8]),
    ]
    values = {key: "sha256:" + "0" * 64 for key in (
        "competition_contract", "competition_report", "transition_contract",
        "transition_report", "qualification_input", "qualification_contract",
        "qualification_report", "review_contract",
    )}
    completed = subprocess.run(
        [sys.executable, str(REVIEW_RUNNER), *flags, *_digest_flags(values)],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert completed.stdout == b""


def test_cli_rejects_relative_and_symlink_inputs_and_output_aliases(tmp_path):
    paths = [(tmp_path / f"input-{index}.json").resolve() for index in range(9)]
    for path in paths:
        path.write_text("{}\n", encoding="utf-8")
    linked = (tmp_path / "linked.json").resolve()
    linked.symlink_to(paths[0])
    assert import_runner()._require_input_file
    with pytest.raises(ValueError):
        import_runner()._require_input_file(Path("relative.json"), "relative")
    with pytest.raises(ValueError):
        import_runner()._require_input_file(linked, "symlink")
    with pytest.raises(ValueError):
        import_runner()._protect_output(paths[0], tuple(paths))
    out = (tmp_path / "hardlinked-out.json").resolve()
    os.link(paths[0], out)
    with pytest.raises(ValueError):
        import_runner()._protect_output(out, tuple(paths))


def import_runner():
    spec = importlib.util.spec_from_file_location("review_runner", REVIEW_RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_all_36_input_path_and_hardlink_alias_pairs_are_rejected(tmp_path):
    runner = import_runner()
    checked_same = 0
    checked_hard = 0
    for first in range(9):
        for second in range(first + 1, 9):
            pair_root = tmp_path / f"pair-{first}-{second}"
            pair_root.mkdir()
            paths = []
            for index in range(9):
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
    assert checked_same == 36
    assert checked_hard == 36


def test_packet_id_is_deterministic_and_bound_to_source_qualification():
    first = _review("idealization")[-1]
    second = _review("idealization")[-1]
    robust = _review("robustification")[-1]
    assert first["packet_id"] == second["packet_id"]
    assert first["report_digest"] == second["report_digest"]
    assert first["packet_id"] != robust["packet_id"]


def test_surface_has_no_execution_path_and_preserves_previous_slice_bytes():
    before = {
        path: _digest_file(path) for path in (*PREVIOUS_SLICE_FILES, OLD_BENCHMARK)
    }
    for path in (REVIEW_CORE, REVIEW_RUNNER):
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
        }
        assert imported.isdisjoint(forbidden_imports)
        assert "run_one" not in calls

    _review("robustification")
    _review("idealization")
    assert {path: _digest_file(path) for path in before} == before


def test_documentation_states_the_non_authoritative_boundary():
    text = REVIEW_DOC.read_text(encoding="utf-8")
    assert "Packet readiness is not adoption eligibility" in text
    assert "sc-olh-kg.shadow-child-record-lifecycle-boundary/1" in text
    assert "REQUIRED_NOT_PRESENT" in text
    assert "physical_erasure = NOT_PERFORMED" in text
    assert "NOT_ADOPTED_SHADOW_ONLY" in text
    assert "NOT_PROMOTED" in text
    assert "NOT_CURRENT" in text
