import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.shadow_child_probe_qualification import (  # noqa: E402
    derive_shadow_child_evaluator_epoch,
    qualify_shadow_child_operational_probes,
    validate_shadow_child_probe_qualification_contract,
    verify_shadow_child_probe_qualification,
)
from performance.shadow_theory_transition import (  # noqa: E402
    materialize_shadow_theory_transition,
)
from performance.theory_operation_competition import (  # noqa: E402
    SCHEMA_VERSION as COMPETITION_INPUT_SCHEMA,
    canonical_json_bytes,
    run_theory_operation_competition,
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
COMPETITION_RUNNER = ROOT / "runners/run_theory_operation_competition.py"
TRANSITION_RUNNER = ROOT / "runners/run_shadow_theory_transition.py"
QUALIFICATION_RUNNER = ROOT / "runners/run_shadow_child_probe_qualification.py"
QUALIFICATION_CORE = ROOT / "performance/shadow_child_probe_qualification.py"
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
)

REPORT_KEYS = {
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
SOURCE_TRANSITION_KEYS = {
    "verification_status",
    "contract_id",
    "contract_digest",
    "report_digest",
    "source_competition_report_digest",
    "child_theory_state_digest",
    "transition_kind",
    "selected_candidate_id",
    "adoption_status",
}
EVALUATOR_DEFINITION_KEYS = {
    "schema_version",
    "qualification_contract_digest",
    "source_transition_contract_digest",
    "source_transition_report_digest",
    "child_theory_state_digest",
    "transition_kind",
    "fixed_anchor",
    "probe_registry",
}
EVALUATOR_BINDING_KEYS = {
    "source_evaluator_epoch",
    "evaluator_definition_digest",
    "derived_child_evaluator_epoch",
    "declared_evaluator_epoch",
    "inherited_fixed_anchor",
    "declared_fixed_anchor",
    "row_evaluator_epochs",
    "row_fixed_anchors",
    "all_new_rows_exact_epoch",
    "all_new_rows_exact_anchor",
    "fresh_from_source_epoch",
    "comparable",
    "old_new_records_pooled",
}
EVIDENCE_BINDING_KEYS = {
    "evidence_digests",
    "row_counts",
    "minimum_rows",
    "minimum_rows_satisfied_by_split",
    "required_context_scope_pairs",
    "required_context_scope_pair_count",
    "covered_context_scope_pairs_by_split",
    "missing_context_scope_pairs_by_split",
    "complete_context_scope_coverage_by_split",
    "source_observation_id_digest",
    "source_observation_count",
    "new_observation_id_digest",
    "new_observation_count",
    "new_ids_unique_across_splits",
    "new_ids_disjoint_from_source",
    "sufficient",
}
ERASURE_KEYS = {
    "policy",
    "mode",
    "excluded_source_evidence_digests",
    "excluded_source_observation_id_digest",
    "excluded_source_observation_count",
    "included_new_evidence_digests",
    "included_new_observation_id_digest",
    "included_new_observation_count",
    "source_evidence_used_only_for_upstream_replay_and_exclusion_binding",
    "source_evidence_used_for_child_scoring",
    "old_new_records_pooled",
    "logical_selective_erasure_applied",
    "physical_records_deleted",
}
PROBE_COMMON_KEYS = {
    "result_kind",
    "child_model_kind",
    "probe_ids",
    "holdout",
    "stress",
    "aggregate",
    "gate_checks",
    "all_gates_passed",
    "counterexample_observation_ids",
}
ROBUST_HOLDOUT_KEYS = {
    "row_count",
    "parent_center_mae",
    "child_center_mae",
    "nominal_mae_increase",
    "parent_nominal_coverage",
    "interval_coverage",
    "coverage_gain",
    "mean_radius",
}
ROBUST_STRESS_KEYS = ROBUST_HOLDOUT_KEYS | {
    "tail_row_count",
    "tail_observation_ids",
    "tail_interval_coverage",
    "safety_rate",
}
IDEAL_SPLIT_KEYS = {
    "row_count",
    "parent_mae",
    "child_mae",
    "mae_increase",
    "max_parent_child_point_divergence",
}
GATE_KEYS = {
    "gate_id",
    "metric_path",
    "operator",
    "threshold",
    "actual",
    "passed",
}
QUALIFICATION_BINDING_KEYS = {
    "child_theory_state_digest",
    "evaluator_epoch",
    "fixed_anchor",
    "evaluator_status",
    "operational_probe_status",
    "qualification_status",
    "adoption_status",
    "original_child_state_mutated",
    "source_evidence_allowed_for_child_scoring",
    "old_new_records_pooled",
    "logical_selective_erasure_applied",
}

MANDATORY_NONCLAIMS = [
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
]


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest_value(value):
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_dict(result):
    return result.to_dict() if hasattr(result, "to_dict") else result


def _context(x, nuisance):
    return {"x": x, "nuisance": nuisance}


def _source_row(split, index, context, observed):
    return {
        "observation_id": f"source-{split}-{index:02d}",
        "evaluator_epoch": "evaluator-epoch-1",
        "fixed_anchor": "fixed-anchor-1",
        "scope_id": "registered-scope",
        "context": copy.deepcopy(context),
        "observed_value": observed,
    }


def _case(kind):
    if kind == "robustification":
        predictions = (0.0, 10.0, 2.0, 12.0)
        residuals = (-1.0, -0.3, 0.3, 1.0)
        discovery = tuple(a + b for a, b in zip(predictions, residuals))
        heldout = discovery
    elif kind == "idealization":
        predictions = (-0.1, 0.1, 0.9, 1.1)
        discovery = predictions
        heldout = (0.0, 0.0, 1.0, 1.0)
    else:
        raise AssertionError(kind)
    contexts = [
        _context(0, 0),
        _context(0, 1),
        _context(1, 0),
        _context(1, 1),
    ]
    evidence = {}
    for split in ("discovery", "validation", "stress"):
        values = discovery if split == "discovery" else heldout
        evidence[split] = [
            _source_row(split, 2 * index + repeat, context, value)
            for index, (context, value) in enumerate(zip(contexts, values))
            for repeat in range(2)
        ]
    return {
        "schema_version": COMPETITION_INPUT_SCHEMA,
        "case_id": f"qualification-source-{kind}",
        "theory_state": {
            "theory_id": "qualification-parent-finite-theory",
            "task_id": "bounded-regression-task",
            "evaluator_epoch": "evaluator-epoch-1",
            "fixed_anchor": "fixed-anchor-1",
            "object_space": {
                "feature_ids": ["x", "nuisance"],
                "contexts": contexts,
            },
            "model_class": {
                "kind": "finite_point_table",
                "predictions": [
                    {"context": copy.deepcopy(context), "value": value}
                    for context, value in zip(contexts, predictions)
                ],
            },
            "probe_ids": ["absolute_error_point_prediction"],
            "violation_functionals": [
                {"functional_id": "absolute_error", "threshold": 0.2}
            ],
            "scope_ids": ["registered-scope"],
            "removable_feature_ids": ["nuisance"],
        },
        "evidence": evidence,
    }


def _source_chain(kind):
    case = _case(kind)
    competition_contract = _load(COMPETITION_CONTRACT)
    competition_report = _as_dict(
        run_theory_operation_competition(case, competition_contract)
    )
    transition_contract = _load(TRANSITION_CONTRACT)
    transition_report = _as_dict(
        materialize_shadow_theory_transition(
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
    return (
        case,
        competition_contract,
        competition_report,
        transition_contract,
        transition_report,
    )


def _qualification_values(kind):
    if kind == "robustification":
        return (-1.0, 9.7, 2.3, 13.0)
    if kind == "idealization":
        return (0.0, 0.0, 1.0, 1.0)
    raise AssertionError(kind)


def _qualification_input(kind, chain):
    case, _, _, transition_contract, transition_report = chain
    qualification_contract = _load(QUALIFICATION_CONTRACT)
    child = transition_report["child_theory_state"]
    evaluator_epoch = derive_shadow_child_evaluator_epoch(
        transition_contract_digest=_digest_value(transition_contract),
        transition_report_digest=transition_report["report_digest"],
        child_theory_state_digest=transition_report["child_theory_state_digest"],
        transition_kind=transition_report["transition_kind"],
        fixed_anchor=child["fixed_anchor"],
        qualification_contract=qualification_contract,
    )
    contexts = case["theory_state"]["object_space"]["contexts"]
    values = _qualification_values(kind)
    evidence = {}
    for split in ("holdout", "stress"):
        evidence[split] = [
            {
                "observation_id": (
                    f"new-{kind}-{split}-{2 * index + repeat:02d}"
                ),
                "evaluator_epoch": evaluator_epoch,
                "fixed_anchor": child["fixed_anchor"],
                "scope_id": "registered-scope",
                "context": copy.deepcopy(context),
                "observed_value": value,
            }
            for index, (context, value) in enumerate(zip(contexts, values))
            for repeat in range(2)
        ]
    return {
        "schema_version": qualification_contract["input_schema_version"],
        "qualification_id": f"qualification-{kind}",
        "source_transition": {
            "contract_digest": _digest_value(transition_contract),
            "report_digest": transition_report["report_digest"],
            "child_theory_state_digest": transition_report[
                "child_theory_state_digest"
            ],
        },
        "evaluator": {
            "evaluator_epoch": evaluator_epoch,
            "fixed_anchor": child["fixed_anchor"],
        },
        "source_evidence_exclusion": {
            "policy": "EXCLUDE_ALL_SOURCE_COMPETITION_RECORDS",
            "source_evidence_digests": transition_report["source_competition"][
                "evidence_digests"
            ],
        },
        "evidence": evidence,
    }


def _qualify(kind, *, mutate_input=None):
    chain = _source_chain(kind)
    qualification_input = _qualification_input(kind, chain)
    if mutate_input is not None:
        mutate_input(qualification_input, chain)
    qualification_contract = _load(QUALIFICATION_CONTRACT)
    case, competition_contract, competition_report, transition_contract, transition_report = chain
    report = _as_dict(
        qualify_shadow_child_operational_probes(
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
            expected_transition_report_digest=transition_report["report_digest"],
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
    return (*chain, qualification_input, qualification_contract, report)


def _write(path, value):
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def test_frozen_contract_and_exact_report_vocabulary():
    contract = _load(QUALIFICATION_CONTRACT)
    assert validate_shadow_child_probe_qualification_contract(contract) == contract
    assert contract["contract_id"] == "shadow_child_probe_qualification_v1"
    assert contract["input_schema_version"] == (
        "sc-olh-kg.shadow-child-probe-qualification-input/1"
    )
    assert contract["report_schema_version"] == (
        "sc-olh-kg.shadow-child-probe-qualification-report/1"
    )
    assert contract["nonclaims"] == MANDATORY_NONCLAIMS

    *_, report = _qualify("robustification")
    assert set(report) == REPORT_KEYS
    assert set(report["source_transition"]) == SOURCE_TRANSITION_KEYS
    assert set(report["evaluator_definition"]) == EVALUATOR_DEFINITION_KEYS
    assert set(report["evaluator_binding"]) == EVALUATOR_BINDING_KEYS
    assert set(report["evidence_binding"]) == EVIDENCE_BINDING_KEYS
    assert set(report["selective_erasure_receipt"]) == ERASURE_KEYS
    assert set(report["probe_results"]) == PROBE_COMMON_KEYS
    assert set(report["qualification_binding"]) == QUALIFICATION_BINDING_KEYS
    assert all(
        set(item) == GATE_KEYS for item in report["probe_results"]["gate_checks"]
    )
    assert report["nonclaims"] == MANDATORY_NONCLAIMS


def test_robust_child_qualifies_on_fresh_interval_and_tail_evidence():
    *_, report = _qualify("robustification")

    assert report["disposition"] == "QUALIFIED_NEW_EVALUATOR_EPOCH"
    assert report["probe_results"]["all_gates_passed"] is True
    assert set(report["probe_results"]["holdout"]) == ROBUST_HOLDOUT_KEYS
    assert set(report["probe_results"]["stress"]) == ROBUST_STRESS_KEYS
    assert set(report["probe_results"]["aggregate"]) == {
        "prediction_scale",
        "normalized_radius",
        "max_nominal_mae_increase",
    }
    assert report["probe_results"]["holdout"]["interval_coverage"] == 1.0
    assert report["probe_results"]["stress"]["tail_interval_coverage"] == 1.0
    assert report["probe_results"]["stress"]["safety_rate"] == 1.0
    aggregate = report["probe_results"]["aggregate"]
    expected_normalized_radius = (
        report["probe_results"]["holdout"]["mean_radius"]
        + report["probe_results"]["stress"]["mean_radius"]
    ) / (2.0 * aggregate["prediction_scale"])
    assert aggregate["normalized_radius"] == pytest.approx(
        expected_normalized_radius
    )
    assert report["probe_results"]["counterexample_observation_ids"] == []


def test_ideal_child_qualifies_on_fresh_task_scoped_point_evidence():
    *_, report = _qualify("idealization")

    assert report["disposition"] == "QUALIFIED_NEW_EVALUATOR_EPOCH"
    assert report["probe_results"]["all_gates_passed"] is True
    assert set(report["probe_results"]["holdout"]) == IDEAL_SPLIT_KEYS
    assert set(report["probe_results"]["stress"]) == IDEAL_SPLIT_KEYS
    assert set(report["probe_results"]["aggregate"]) == {
        "max_parent_child_point_divergence",
        "max_nominal_mae_increase",
    }
    assert report["probe_results"]["aggregate"][
        "max_parent_child_point_divergence"
    ] == pytest.approx(0.1)


def test_qualification_receipt_is_logical_erasure_not_adoption_or_mutation():
    *_, transition_report, _, _, report = _qualify("robustification")

    binding = report["qualification_binding"]
    erasure = report["selective_erasure_receipt"]
    assert binding["original_child_state_mutated"] is False
    assert binding["source_evidence_allowed_for_child_scoring"] is False
    assert binding["old_new_records_pooled"] is False
    assert binding["logical_selective_erasure_applied"] is True
    assert erasure["source_evidence_used_for_child_scoring"] is False
    assert erasure["old_new_records_pooled"] is False
    assert erasure["logical_selective_erasure_applied"] is True
    assert erasure["physical_records_deleted"] is False
    assert report["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert report["child_theory_state_digest"] == transition_report[
        "child_theory_state_digest"
    ]


def test_epoch_is_content_derived_deterministic_fresh_and_anchor_bound():
    chain = _source_chain("idealization")
    first = _qualification_input("idealization", chain)
    second = _qualification_input("idealization", chain)
    assert first["evaluator"] == second["evaluator"]
    assert first["evaluator"]["evaluator_epoch"] != (
        chain[0]["theory_state"]["evaluator_epoch"]
    )

    contract = _load(QUALIFICATION_CONTRACT)
    transition_report = chain[-1]
    changed = derive_shadow_child_evaluator_epoch(
        transition_contract_digest=_digest_value(chain[-2]),
        transition_report_digest=transition_report["report_digest"],
        child_theory_state_digest=transition_report["child_theory_state_digest"],
        transition_kind=transition_report["transition_kind"],
        fixed_anchor="different-fixed-anchor",
        qualification_contract=contract,
    )
    assert changed != first["evaluator"]["evaluator_epoch"]


def test_failed_probe_qualification_has_bound_counterexamples():
    def fail_probes(payload, _chain):
        for rows in payload["evidence"].values():
            for row in rows:
                row["observed_value"] = 1000.0

    *_, report = _qualify("robustification", mutate_input=fail_probes)
    assert report["disposition"] == "FAILED_OPERATIONAL_PROBE_QUALIFICATION"
    assert report["probe_results"]["all_gates_passed"] is False
    assert report["probe_results"]["counterexample_observation_ids"]
    assert report["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"


def test_needs_evidence_for_minimum_rows_and_complete_context_scope_coverage():
    def truncate(payload, _chain):
        payload["evidence"]["holdout"] = payload["evidence"]["holdout"][:1]
        payload["evidence"]["stress"] = payload["evidence"]["stress"][:1]

    *_, report = _qualify("idealization", mutate_input=truncate)
    assert report["disposition"] == "NEEDS_NEW_EVALUATOR_EVIDENCE"
    assert report["evidence_binding"]["sufficient"] is False
    assert report["probe_results"] is None


@pytest.mark.parametrize("changed", ("declared_epoch", "row_epoch", "anchor"))
def test_incomparable_epoch_or_anchor_is_never_numerically_pooled(changed):
    def mismatch(payload, _chain):
        if changed == "declared_epoch":
            payload["evaluator"]["evaluator_epoch"] = "forged-epoch"
        elif changed == "row_epoch":
            payload["evidence"]["stress"][0]["evaluator_epoch"] = "forged-epoch"
        else:
            payload["evidence"]["holdout"][0]["fixed_anchor"] = "other-anchor"

    *_, report = _qualify("robustification", mutate_input=mismatch)
    assert report["disposition"] == "INCOMPARABLE_NEW_EVALUATOR_EPOCH"
    assert report["evaluator_binding"]["comparable"] is False
    assert report["evaluator_binding"]["old_new_records_pooled"] is False
    assert report["probe_results"] is None


def test_source_id_reuse_and_duplicate_new_ids_fail_structurally():
    def source_reuse(payload, chain):
        payload["evidence"]["holdout"][0]["observation_id"] = chain[0][
            "evidence"
        ]["discovery"][0]["observation_id"]

    with pytest.raises(ValueError):
        _qualify("idealization", mutate_input=source_reuse)

    def duplicate_new_id(payload, _chain):
        payload["evidence"]["stress"][0]["observation_id"] = payload[
            "evidence"
        ]["holdout"][1]["observation_id"]

    with pytest.raises(ValueError):
        _qualify("idealization", mutate_input=duplicate_new_id)


def test_verifier_replays_and_rejects_self_consistent_report_tampering():
    (
        case,
        competition_contract,
        competition_report,
        transition_contract,
        transition_report,
        qualification_input,
        qualification_contract,
        report,
    ) = _qualify("idealization")
    verified = verify_shadow_child_probe_qualification(
        case,
        competition_contract,
        competition_report,
        transition_contract,
        transition_report,
        qualification_input,
        qualification_contract,
        report,
        expected_competition_contract_digest=_digest_value(competition_contract),
        expected_competition_report_digest=competition_report["report_digest"],
        expected_competition_input_artifacts=None,
        expected_transition_contract_digest=_digest_value(transition_contract),
        expected_transition_report_digest=transition_report["report_digest"],
        expected_transition_input_artifacts=None,
        expected_qualification_input_digest=_digest_value(qualification_input),
        expected_qualification_contract_digest=_digest_value(
            qualification_contract
        ),
        expected_qualification_report_digest=report["report_digest"],
        expected_qualification_input_artifacts=None,
    )
    assert isinstance(verified, dict)

    tampered = copy.deepcopy(report)
    tampered["qualification_binding"]["original_child_state_mutated"] = True
    tampered["report_digest"] = _digest_value(
        {key: value for key, value in tampered.items() if key != "report_digest"}
    )
    with pytest.raises(ValueError):
        verify_shadow_child_probe_qualification(
            case,
            competition_contract,
            competition_report,
            transition_contract,
            transition_report,
            qualification_input,
            qualification_contract,
            tampered,
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
            expected_transition_report_digest=transition_report["report_digest"],
            expected_transition_input_artifacts=None,
            expected_qualification_input_digest=_digest_value(
                qualification_input
            ),
            expected_qualification_contract_digest=_digest_value(
                qualification_contract
            ),
            expected_qualification_report_digest=tampered["report_digest"],
            expected_qualification_input_artifacts=None,
        )


def test_all_independent_digest_and_source_exclusion_gates_fail_closed():
    chain = _source_chain("robustification")
    qualification_input = _qualification_input("robustification", chain)
    qualification_contract = _load(QUALIFICATION_CONTRACT)
    case, competition_contract, competition_report, transition_contract, transition_report = chain
    base = {
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
        "input_artifacts": None,
    }
    for key in (
        "expected_competition_contract_digest",
        "expected_competition_report_digest",
        "expected_transition_contract_digest",
        "expected_transition_report_digest",
        "expected_qualification_input_digest",
        "expected_qualification_contract_digest",
    ):
        changed = dict(base)
        changed[key] = "sha256:" + "0" * 64
        with pytest.raises(ValueError):
            qualify_shadow_child_operational_probes(
                case,
                competition_contract,
                competition_report,
                transition_contract,
                transition_report,
                qualification_input,
                qualification_contract,
                **changed,
            )

    forged_exclusion = copy.deepcopy(qualification_input)
    forged_exclusion["source_evidence_exclusion"]["source_evidence_digests"][
        "stress"
    ] = "sha256:" + "f" * 64
    with pytest.raises(ValueError):
        qualify_shadow_child_operational_probes(
            case,
            competition_contract,
            competition_report,
            transition_contract,
            transition_report,
            forged_exclusion,
            qualification_contract,
            **{
                **base,
                "expected_qualification_input_digest": _digest_value(
                    forged_exclusion
                ),
            },
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


def test_cli_canonical_stdout_atomic_out_and_seven_inputs_unchanged(tmp_path):
    case_path = (tmp_path / "case.json").resolve()
    competition_report_path = (tmp_path / "competition-report.json").resolve()
    transition_report_path = (tmp_path / "transition-report.json").resolve()
    qualification_input_path = (tmp_path / "qualification-input.json").resolve()
    qualification_report_path = (tmp_path / "qualification-report.json").resolve()
    _write(case_path, _case("idealization"))

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
    qualification_input = _qualification_input("idealization", chain)
    _write(qualification_input_path, qualification_input)
    qualification_contract = _load(QUALIFICATION_CONTRACT)
    inputs = (
        case_path,
        COMPETITION_CONTRACT.resolve(),
        competition_report_path,
        TRANSITION_CONTRACT.resolve(),
        transition_report_path,
        qualification_input_path,
        QUALIFICATION_CONTRACT.resolve(),
    )
    before = {path: path.read_bytes() for path in inputs}

    completed = subprocess.run(
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
            "--out",
            str(qualification_report_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    parsed = json.loads(completed.stdout)
    assert completed.stdout == canonical_json_bytes(parsed) + b"\n"
    assert qualification_report_path.read_bytes() == completed.stdout
    assert {path: path.read_bytes() for path in inputs} == before


@pytest.mark.parametrize(
    "raw",
    (
        b'{"qualification_id":"a","qualification_id":"b"}\n',
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
    ),
)
def test_cli_rejects_duplicate_and_nonfinite_json_without_stdout(raw, tmp_path):
    invalid = (tmp_path / "invalid.json").resolve()
    placeholder = (tmp_path / "placeholder.json").resolve()
    invalid.write_bytes(raw)
    placeholder.write_text("{}\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(QUALIFICATION_RUNNER),
            "--competition-input",
            str(invalid),
            "--competition-report",
            str(placeholder),
            "--transition-report",
            str(placeholder),
            "--qualification-input",
            str(placeholder),
            "--expected-competition-contract-digest",
            "sha256:" + "0" * 64,
            "--expected-competition-report-digest",
            "sha256:" + "0" * 64,
            "--expected-transition-contract-digest",
            "sha256:" + "0" * 64,
            "--expected-transition-report-digest",
            "sha256:" + "0" * 64,
            "--expected-qualification-input-digest",
            "sha256:" + "0" * 64,
            "--expected-qualification-contract-digest",
            "sha256:" + "0" * 64,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert completed.stdout == b""


def test_cli_rejects_symlink_input_and_hardlinked_output(tmp_path):
    real = (tmp_path / "real.json").resolve()
    linked = (tmp_path / "linked.json").resolve()
    out = (tmp_path / "out.json").resolve()
    real.write_text("{}\n", encoding="utf-8")
    linked.symlink_to(real)
    os.link(real, out)
    common = [
        sys.executable,
        str(QUALIFICATION_RUNNER),
        "--competition-report",
        str(real),
        "--transition-report",
        str(real),
        "--qualification-input",
        str(real),
        "--expected-competition-contract-digest",
        "sha256:" + "0" * 64,
        "--expected-competition-report-digest",
        "sha256:" + "0" * 64,
        "--expected-transition-contract-digest",
        "sha256:" + "0" * 64,
        "--expected-transition-report-digest",
        "sha256:" + "0" * 64,
        "--expected-qualification-input-digest",
        "sha256:" + "0" * 64,
        "--expected-qualification-contract-digest",
        "sha256:" + "0" * 64,
    ]
    symlinked = subprocess.run(
        [*common, "--competition-input", str(linked)],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert symlinked.returncode == 2
    assert symlinked.stdout == b""
    hardlinked = subprocess.run(
        [*common, "--competition-input", str(real), "--out", str(out)],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert hardlinked.returncode == 2
    assert hardlinked.stdout == b""


def test_cli_rejects_same_path_and_hardlinked_input_files_before_json_load(
    tmp_path,
):
    competition_input = (tmp_path / "competition-input.json").resolve()
    competition_report = (tmp_path / "competition-report.json").resolve()
    transition_report = (tmp_path / "transition-report.json").resolve()
    qualification_input = (tmp_path / "qualification-input.json").resolve()
    for path in (
        competition_input,
        competition_report,
        transition_report,
        qualification_input,
    ):
        path.write_text("{}\n", encoding="utf-8")

    digest_flags = [
        "--expected-competition-contract-digest",
        "sha256:" + "0" * 64,
        "--expected-competition-report-digest",
        "sha256:" + "0" * 64,
        "--expected-transition-contract-digest",
        "sha256:" + "0" * 64,
        "--expected-transition-report-digest",
        "sha256:" + "0" * 64,
        "--expected-qualification-input-digest",
        "sha256:" + "0" * 64,
        "--expected-qualification-contract-digest",
        "sha256:" + "0" * 64,
    ]
    same_path = subprocess.run(
        [
            sys.executable,
            str(QUALIFICATION_RUNNER),
            "--competition-input",
            str(competition_input),
            "--competition-report",
            str(competition_input),
            "--transition-report",
            str(transition_report),
            "--qualification-input",
            str(qualification_input),
            *digest_flags,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert same_path.returncode == 2
    assert same_path.stdout == b""

    competition_report.unlink()
    os.link(competition_input, competition_report)
    hardlinked = subprocess.run(
        [
            sys.executable,
            str(QUALIFICATION_RUNNER),
            "--competition-input",
            str(competition_input),
            "--competition-report",
            str(competition_report),
            "--transition-report",
            str(transition_report),
            "--qualification-input",
            str(qualification_input),
            *digest_flags,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert hardlinked.returncode == 2
    assert hardlinked.stdout == b""


def test_surface_has_no_execution_import_and_preserves_prior_bytes():
    before = {
        path: _digest_file(path) for path in (*PREVIOUS_SLICE_FILES, OLD_BENCHMARK)
    }
    for path in (QUALIFICATION_CORE, QUALIFICATION_RUNNER):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
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
        assert "performance.benchmark_lodo_meta_prior" not in imports
        assert "run_one" not in calls

    _qualify("robustification")
    _qualify("idealization")
    assert {path: _digest_file(path) for path in before} == before
