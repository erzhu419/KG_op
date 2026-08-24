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

import test_shadow_child_external_review_packet as review_kat  # noqa: E402
import test_shadow_child_failure_boundary_probe as probe_kat  # noqa: E402
import performance.shadow_robust_interval_restriction as restriction_core  # noqa: E402

from performance.shadow_child_failure_boundary_probe import (  # noqa: E402
    derive_shadow_child_failure_boundary_probe_epoch,
    expand_and_evaluate_shadow_child_failure_boundary_probe,
)
from performance.shadow_child_probe_qualification import (  # noqa: E402
    qualify_shadow_child_operational_probes,
)
from performance.shadow_robust_interval_restriction import (  # noqa: E402
    compete_and_materialize_shadow_robust_interval_restriction,
    derive_shadow_robust_interval_restriction_epoch,
    validate_shadow_robust_interval_restriction_contract,
    verify_shadow_robust_interval_restriction,
)
from performance.shadow_theory_transition import (  # noqa: E402
    materialize_shadow_theory_transition,
)
from performance.theory_operation_competition import (  # noqa: E402
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
REVIEW_CONTRACT = (
    ROOT / "performance/manifests/shadow_child_external_review_packet_v1.json"
)
PROBE_CONTRACT = (
    ROOT / "performance/manifests/shadow_child_failure_boundary_probe_v1.json"
)
RESTRICTION_CONTRACT = (
    ROOT / "performance/manifests/shadow_robust_interval_restriction_v1.json"
)
RESTRICTION_CORE = ROOT / "performance/shadow_robust_interval_restriction.py"
RESTRICTION_RUNNER = ROOT / "runners/run_shadow_robust_interval_restriction.py"
RESTRICTION_DOC = ROOT / "docs/shadow_robust_interval_restriction_v1.md"
OLD_BENCHMARK = ROOT / "performance/benchmark_lodo_meta_prior.py"

PREVIOUS_SLICE_FILES = (
    ROOT / "performance/theory_operation_competition.py",
    COMPETITION_CONTRACT,
    ROOT / "runners/run_theory_operation_competition.py",
    ROOT / "tests/test_theory_operation_competition.py",
    ROOT / "docs/theory_operation_competition_v1.md",
    ROOT / "performance/shadow_theory_transition.py",
    TRANSITION_CONTRACT,
    ROOT / "runners/run_shadow_theory_transition.py",
    ROOT / "tests/test_shadow_theory_transition.py",
    ROOT / "docs/shadow_theory_transition_v1.md",
    ROOT / "performance/shadow_child_probe_qualification.py",
    QUALIFICATION_CONTRACT,
    ROOT / "runners/run_shadow_child_probe_qualification.py",
    ROOT / "tests/test_shadow_child_probe_qualification.py",
    ROOT / "docs/shadow_child_probe_qualification_v1.md",
    ROOT / "performance/shadow_child_external_review_packet.py",
    REVIEW_CONTRACT,
    ROOT / "runners/run_shadow_child_external_review_packet.py",
    ROOT / "tests/test_shadow_child_external_review_packet.py",
    ROOT / "docs/shadow_child_external_review_packet_v1.md",
    ROOT / "performance/shadow_child_failure_boundary_probe.py",
    PROBE_CONTRACT,
    ROOT / "runners/run_shadow_child_failure_boundary_probe.py",
    ROOT / "tests/test_shadow_child_failure_boundary_probe.py",
    ROOT / "docs/shadow_child_failure_boundary_probe_v1.md",
)

MANDATORY_NONCLAIMS = [
    "shadow_only",
    "robust_finite_interval_restriction_only",
    "no_generic_restriction_engine",
    "no_rigid_body_markov_or_independence_assumption",
    "no_scope_restriction",
    "no_quotient_restriction",
    "no_new_probe",
    "q_registry_copied_not_requalified",
    "v_registry_unchanged",
    "no_language_or_predicate_invention",
    "no_external_probe_acquisition",
    "caller_supplied_static_rows_only",
    "local_epoch_is_not_external_attestation",
    "fresh_evidence_pass_is_not_global_preservation",
    "interval_width_reduction_is_not_nominal_utility_or_domain_safety",
    "no_source_child_invalidation",
    "no_rollback_execution",
    "no_adoption_eligibility_determination",
    "no_adoption",
    "no_promotion",
    "no_current_pointer_write",
    "no_parent_source_or_ambient_state_write",
    "no_h_t_to_h_t_plus_1_acceptance",
    "no_cross_epoch_pooling",
    "no_physical_erasure",
    "no_external_data_or_evaluator_attestation",
    "no_run_one",
    "no_benchmark_execution",
    "no_scheduler_or_network_access",
    "no_operations_research_baseline_or_claim_change",
    "no_paper_promotion",
    "explicit_cli_out_is_only_optional_write",
    "input_artifacts_require_independent_expected_values",
    "report_digest_is_not_a_signature",
    "no_scientific_validity_or_generalization_claim",
]

REPORT_KEYS = {
    "schema_version",
    "contract_id",
    "contract_digest",
    "restriction_input_digest",
    "restriction_id",
    "source_failure_boundary_probe",
    "source_probe_expanded_shadow_theory_state_digest",
    "source_transition_kind",
    "operation_kind",
    "restriction_kind",
    "evaluator_definition",
    "evaluator_binding",
    "evidence_binding",
    "candidate_registry",
    "candidate_competition",
    "selected_candidate",
    "fresh_validation",
    "disposition",
    "restricted_shadow_theory_state",
    "restricted_shadow_theory_state_digest",
    "restriction_certificate",
    "rollback_boundary",
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

STATE_KEYS = {
    "schema_version",
    "theory_id",
    "task_id",
    "source_probe_expanded_theory_state_digest",
    "source_failure_boundary_probe_report_digest",
    "operation_kind",
    "restriction_kind",
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
    "restriction_lineage",
    "adoption_status",
    "current_status",
}

CERTIFICATE_KEYS = {
    "certificate_kind",
    "source_model_class_digest",
    "restricted_model_class_digest",
    "radius_multiplier",
    "checked_radius_group_count",
    "checked_context_scope_pair_count",
    "centers_byte_equal",
    "grouping_and_group_keys_byte_equal",
    "all_restricted_radii_finite_nonnegative",
    "all_restricted_radii_lte_source",
    "at_least_one_radius_strictly_reduced",
    "strict_subset_verified",
}

ROLLBACK_KEYS = {
    "method",
    "source_probe_expanded_state_digest",
    "source_materialized_child_digest",
    "original_parent_theory_state_digest",
    "source_model_class_digest",
    "restricted_model_class_digest",
    "rollback_execution_status",
}

SOURCE_PROBE_KEYS = {
    "verification_status",
    "contract_id",
    "contract_digest",
    "report_digest",
    "probe_expansion_id",
    "probe_expanded_shadow_theory_state_digest",
    "transition_kind",
    "disposition",
    "boundary_counterexample_found",
    "adoption_status",
}

EVALUATOR_DEFINITION_KEYS = {
    "evaluator_epoch",
    "fixed_anchor",
    "epoch_derivation_kind",
    "fixed_multiplier_registry_digest",
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
    "calibration_evidence_digest",
    "holdout_evidence_digest",
    "stress_evidence_digest",
    "row_counts",
    "required_context_scope_pairs",
    "required_context_scope_pair_count",
    "covered_context_scope_pairs_by_split",
    "missing_context_scope_pairs_by_split",
    "duplicate_context_scope_pair_row_counts_by_split",
    "complete_exact_cartesian_coverage_by_split",
    "new_observation_id_digest",
    "new_observation_count",
    "source_competition_observation_id_digest",
    "consumed_qualification_observation_id_digest",
    "consumed_failure_boundary_observation_id_digest",
    "unique_new_observation_ids",
    "disjoint_from_competition_ids",
    "disjoint_from_qualification_ids",
    "disjoint_from_failure_boundary_probe_ids",
    "cross_epoch_pooling",
    "complete_evidence",
}

SPLIT_METRIC_KEYS = {
    "row_count",
    "min_normalized_signed_margin",
    "boundary_violation_count",
    "boundary_violation_rate",
    "mean_normalized_exceedance",
    "max_normalized_exceedance",
    "counterexample_observation_ids",
}

CANDIDATE_REGISTRY_KEYS = {
    "candidate_id",
    "numerator",
    "denominator",
    "radius_multiplier",
    "model_class",
    "model_class_digest",
    "geometry_certificate",
    "candidate_commitment_digest",
}

GEOMETRY_KEYS = CERTIFICATE_KEYS - {"certificate_kind", "radius_multiplier"}

CANDIDATE_COMPETITION_KEYS = {
    "candidate_id",
    "radius_multiplier",
    "candidate_commitment_digest",
    "restricted_model_class_digest",
    "calibration",
    "holdout",
    "stress",
    "calibration_admissible",
    "fresh_validation_passed",
}

FRESH_VALIDATION_KEYS = {
    "source_model_class_digest",
    "prediction_scale",
    "calibration",
    "holdout",
    "stress",
    "all_source_splits_boundary_violation_free",
}

SELECTED_CANDIDATE_KEYS = {
    "candidate_id",
    "numerator",
    "denominator",
    "radius_multiplier",
    "model_class",
    "model_class_digest",
}

AUTHORITY_KEYS = {
    "scope",
    "source_state_mutation",
    "source_child_invalidation",
    "ambient_restriction_execution",
    "rollback_execution",
    "adoption_decision",
    "promotion_decision",
    "current_pointer_write",
    "parent_or_source_state_write",
    "external_data_attestation",
    "external_evaluator_attestation",
    "external_adoption_authority",
}

LIFECYCLE_KEYS = {
    "competition_records",
    "qualification_records",
    "failure_boundary_probe_records",
    "restriction_competition_records",
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
    "cross_epoch_pooling_allowed",
}

STATE_REUSE_KEYS = {
    "source_competition_records_allowed_for_scoring",
    "consumed_qualification_records_allowed_for_scoring",
    "consumed_failure_boundary_probe_records_allowed_for_scoring",
    "restriction_competition_records_allowed_for_future_scoring",
    "future_scoring_requires_new_unconsumed_evidence",
    "cross_epoch_pooling_allowed",
}

STATE_LINEAGE_KEYS = {
    "source_probe_contract_digest",
    "source_probe_report_digest",
    "source_probe_expanded_theory_state_digest",
    "restriction_contract_digest",
    "restriction_id",
    "restriction_evaluator_epoch",
    "selected_candidate_id",
    "selected_radius_multiplier",
    "source_model_class_digest",
    "restricted_model_class_digest",
}

ALL_DISPOSITIONS = {
    "MATERIALIZED_SHADOW_ROBUST_INTERVAL_RESTRICTION",
    "NO_CALIBRATION_ADMISSIBLE_STRICT_INTERVAL_RESTRICTION",
    "NO_VALIDATED_STRICT_INTERVAL_RESTRICTION",
    "RESTRICTION_NEEDS_NEW_EVIDENCE",
    "RESTRICTION_INCOMPARABLE_EVALUATOR_EPOCH",
    "RESTRICTION_BLOCKED_SOURCE_BOUNDARY_COUNTEREXAMPLE",
    "RESTRICTION_BLOCKED_SOURCE_PROBE_UNRESOLVED",
    "RESTRICTION_NOT_APPLICABLE_NON_ROBUST_SOURCE",
}

FROZEN_CONTRACT_DIGEST = (
    "sha256:57e7beb6a1a409cb959be3e98192158311bcb855e4f8295fbd90c4c40e9eb512"
)
FROZEN_CANDIDATE_COMMITMENTS = {
    "uniform_radius_1_over_4": (
        "sha256:eb7118931b7d1eb08aaf1e62bb7913d5289b32aa8567c5b274cfb34ff391cf89"
    ),
    "uniform_radius_1_over_2": (
        "sha256:f31bda91061b91eabbb1811767b3407c7c8b8aef3260202a71cd2cbb7f54ad4f"
    ),
    "uniform_radius_3_over_4": (
        "sha256:a2925bd8415fc7720dd0215a5a12ad13d9b189ab3c77d2f15b3cee11b70fd6a1"
    ),
    "uniform_radius_9_over_10": (
        "sha256:5f63cc5d00862c277db2215a24b332eaac6b377bae4eeb0fc54a6c27379cf378"
    ),
}
FROZEN_POSITIVE_STATE_DIGEST = (
    "sha256:aa274ad2ad29868134b7522581510601635d875e613bf2596a3367f47c6525d8"
)
FROZEN_POSITIVE_REPORT_DIGEST = (
    "sha256:54c272ce7f63d8e1c908f96b341bdf6eb14d3a634d5b0399c6c2eef2cd6c377f"
)


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


def _custom_clear_robust_probe_source(grouping):
    qualification_kat = review_kat.qualification_kat
    case = qualification_kat._case("robustification")
    predictions = (0.0, 10.0, 2.0, 12.0)
    if grouping in {"per_scope", "global"}:
        values = (-0.8, -0.8, 0.8, 0.8)
        for split in ("validation", "stress"):
            for index, row in enumerate(case["evidence"][split]):
                row["observed_value"] = predictions[index // 2] + values[index // 2]

    if grouping == "per_scope":
        scopes = ("scope-A", "scope-B")
        case["theory_state"]["scope_ids"] = list(scopes)
        for split, original_rows in tuple(case["evidence"].items()):
            case["evidence"][split] = [
                {
                    **copy.deepcopy(row),
                    "observation_id": f"{scope}-{row['observation_id']}",
                    "scope_id": scope,
                }
                for scope in scopes
                for row in original_rows
            ]
    elif grouping == "global":
        scopes = ("scope-137",)
        case["theory_state"]["scope_ids"] = list(scopes)
        for rows in case["evidence"].values():
            for row in rows:
                row["scope_id"] = scopes[0]
    elif grouping == "per_context":
        scopes = ("registered-scope",)
    else:
        raise AssertionError(grouping)

    competition_contract = _load(COMPETITION_CONTRACT)
    competition_report = _as_dict(
        run_theory_operation_competition(case, competition_contract)
    )
    assert competition_report["selected_candidate"]["grouping"] == grouping
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
            expected_competition_report_digest=competition_report["report_digest"],
            expected_competition_input_artifacts=None,
            expected_transition_contract_digest=_digest_value(transition_contract),
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
        "robustification", chain
    )
    original_by_split = copy.deepcopy(qualification_input["evidence"])
    for split, original_rows in original_by_split.items():
        qualification_input["evidence"][split] = [
            {
                **copy.deepcopy(row),
                "observation_id": f"qualification-{scope}-{split}-{index:02d}",
                "scope_id": scope,
            }
            for scope in scopes
            for index, row in enumerate(original_rows)
        ]
    qualification_contract = _load(QUALIFICATION_CONTRACT)
    qualification_report = _as_dict(
        qualify_shadow_child_operational_probes(
            *chain,
            qualification_input,
            qualification_contract,
            expected_competition_contract_digest=_digest_value(
                competition_contract
            ),
            expected_competition_report_digest=competition_report["report_digest"],
            expected_competition_input_artifacts=None,
            expected_transition_contract_digest=_digest_value(transition_contract),
            expected_transition_report_digest=transition_report["report_digest"],
            expected_transition_input_artifacts=None,
            expected_qualification_input_digest=_digest_value(qualification_input),
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
    source = (*bundle, review_contract, review_report)
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
    centers = {
        _context_key(item["context"]): item["value"]
        for item in child["model_class"]["center_predictions"]
    }
    contexts = child["object_space"]["contexts"]
    probe_input = {
        "schema_version": probe_contract["input_schema_version"],
        "probe_expansion_id": f"failure-boundary-{grouping}",
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
        "evidence": {
            split: [
                {
                    "observation_id": (
                        f"failure-boundary-{grouping}-{split}-{scope}-{index:02d}"
                    ),
                    "evaluator_epoch": epoch,
                    "fixed_anchor": child["fixed_anchor"],
                    "scope_id": scope,
                    "context": copy.deepcopy(context),
                    "observed_value": centers[_context_key(context)],
                }
                for scope in scopes
                for index, context in enumerate(contexts)
            ]
            for split in ("holdout", "stress")
        },
    }
    probe_report = _as_dict(
        expand_and_evaluate_shadow_child_failure_boundary_probe(
            *source,
            probe_input,
            probe_contract,
            **probe_kat._probe_kwargs(source, probe_input, probe_contract),
        )
    )
    assert probe_report["disposition"] == (
        "EXPANDED_PROBE_NO_BOUNDARY_COUNTEREXAMPLE_ON_SUPPLIED_EVIDENCE"
    )
    return (*source, probe_input, probe_contract, probe_report)


def _probe_source(
    *,
    kind="robustification",
    grouping="per_context",
    probe_state="clear",
):
    if probe_state == "clear" and kind == "robustification":
        return _custom_clear_robust_probe_source(grouping)
    if kind == "idealization" and probe_state == "clear":
        values = probe_kat._expand("idealization", counterexample=False)
    elif kind == "robustification" and probe_state == "counterexample":
        values = probe_kat._expand("robustification", counterexample=True)
    elif kind == "robustification" and probe_state == "needs":
        values = probe_kat._expand(
            "robustification",
            counterexample=False,
            mutate_input=lambda payload, _: payload["evidence"]["holdout"].pop(),
        )
    elif kind == "robustification" and probe_state == "incomparable":
        values = probe_kat._expand(
            "robustification",
            counterexample=False,
            mutate_input=lambda payload, _: payload["evidence"]["stress"][0].__setitem__(
                "evaluator_epoch", "incomparable-source-probe-epoch"
            ),
        )
    elif kind == "robustification" and probe_state == "blocked":
        values = probe_kat._expand(
            "robustification", counterexample=False, review_state="pending"
        )
    else:
        raise AssertionError((kind, grouping, probe_state))
    return (*values[:12], values[-1])


def _radius(model_class, scope_id, context):
    grouping = model_class["radius_grouping"]
    if grouping == "global":
        assert model_class["radii"][0]["group"] == {"global": "*"}
        return model_class["radii"][0]["radius"]
    if grouping == "per_scope":
        return next(
            item["radius"]
            for item in model_class["radii"]
            if item["group"] == {"scope_id": scope_id}
        )
    if grouping == "per_context":
        key = _context_key(context)
        return next(
            item["radius"]
            for item in model_class["radii"]
            if _context_key(item["group"]["context"]) == key
        )
    raise AssertionError(grouping)


def _restriction_input(source, *, factors=None):
    factors = factors or {"calibration": 0.4, "holdout": 0.4, "stress": 0.4}
    probe_report = source[12]
    probe_contract = source[11]
    restriction_contract = _load(RESTRICTION_CONTRACT)
    state = probe_report["probe_expanded_shadow_theory_state"]
    reported_state_digest = probe_report[
        "probe_expanded_shadow_theory_state_digest"
    ]
    if state is None:
        state = source[4]["child_theory_state"]
        state_digest = source[4]["child_theory_state_digest"]
    else:
        state_digest = reported_state_digest
    epoch = derive_shadow_robust_interval_restriction_epoch(
        probe_contract_digest=_digest_value(probe_contract),
        probe_report_digest=probe_report["report_digest"],
        probe_expanded_shadow_theory_state_digest=state_digest,
        fixed_anchor=state["fixed_anchor"],
        restriction_contract=restriction_contract,
    )
    lifecycle = probe_report["record_lifecycle_extension"]
    model = state["model_class"]
    centers = {
        _context_key(item["context"]): item["value"]
        for item in model.get("center_predictions", [])
    }
    # Coverage is always against the frozen parent contexts, including the
    # deleted feature for a routed (non-applicable) quotient source.
    contexts = source[0]["theory_state"]["object_space"]["contexts"]
    scopes = state["scope_ids"]
    evidence = {}
    for split in ("calibration", "holdout", "stress"):
        rows = []
        for scope in scopes:
            for index, context in enumerate(contexts):
                center = centers.get(_context_key(context), 0.0)
                radius = (
                    _radius(model, scope, context)
                    if model.get("kind") == "finite_interval_table"
                    else 1.0
                )
                rows.append(
                    {
                        "observation_id": (
                            f"restriction-{split}-{scope}-{index:02d}"
                        ),
                        "evaluator_epoch": epoch,
                        "fixed_anchor": state["fixed_anchor"],
                        "scope_id": scope,
                        "context": copy.deepcopy(context),
                        "observed_value": center + factors[split] * radius,
                    }
                )
        evidence[split] = rows
    return {
        "schema_version": restriction_contract["input_schema_version"],
        "restriction_id": "bounded-robust-interval-restriction",
        "source_failure_boundary_probe": {
            "probe_contract_digest": _digest_value(probe_contract),
            "probe_report_digest": probe_report["report_digest"],
            "probe_expansion_id": probe_report["probe_expansion_id"],
            "probe_expanded_shadow_theory_state_digest": reported_state_digest,
        },
        "evaluator": {
            "evaluator_epoch": epoch,
            "fixed_anchor": state["fixed_anchor"],
        },
        "prior_record_exclusion": {
            "source_competition_observation_id_digest": lifecycle[
                "competition_records"
            ]["observation_id_digest"],
            "consumed_qualification_observation_id_digest": lifecycle[
                "qualification_records"
            ]["observation_id_digest"],
            "consumed_failure_boundary_observation_id_digest": lifecycle[
                "new_probe_records"
            ]["observation_id_digest"],
        },
        "evidence": evidence,
    }


def _kwargs(source, restriction_input, restriction_contract, *, input_artifacts=None):
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
        "expected_probe_input_digest": _digest_value(source[10]),
        "expected_probe_contract_digest": _digest_value(source[11]),
        "expected_probe_report_digest": source[12]["report_digest"],
        "expected_probe_input_artifacts": None,
        "expected_restriction_input_digest": _digest_value(restriction_input),
        "expected_restriction_contract_digest": _digest_value(restriction_contract),
        "input_artifacts": input_artifacts,
    }


def _restrict(
    *,
    source=None,
    grouping="per_context",
    factors=None,
    mutate_input=None,
):
    source = source or _probe_source(grouping=grouping)
    restriction_input = _restriction_input(source, factors=factors)
    if mutate_input is not None:
        mutate_input(restriction_input, source)
    restriction_contract = _load(RESTRICTION_CONTRACT)
    result = compete_and_materialize_shadow_robust_interval_restriction(
        *source,
        restriction_input,
        restriction_contract,
        **_kwargs(source, restriction_input, restriction_contract),
    )
    return (*source, restriction_input, restriction_contract, result, _as_dict(result))


def _verify(source, restriction_input, restriction_contract, report, **overrides):
    kwargs = _kwargs(source, restriction_input, restriction_contract)
    kwargs.pop("input_artifacts")
    kwargs.update(
        {
            "expected_restriction_report_digest": report["report_digest"],
            "expected_restriction_input_artifacts": None,
        }
    )
    kwargs.update(overrides)
    return verify_shadow_robust_interval_restriction(
        *source, restriction_input, restriction_contract, report, **kwargs
    )


def _rehash(report):
    report["report_digest"] = _digest_value(
        {key: value for key, value in report.items() if key != "report_digest"}
    )
    return report


def test_frozen_contract_report_state_certificate_and_nonclaims():
    contract = _load(RESTRICTION_CONTRACT)
    assert validate_shadow_robust_interval_restriction_contract(contract) == contract
    assert _digest_value(contract) == FROZEN_CONTRACT_DIGEST
    assert contract["contract_id"] == "shadow_robust_interval_restriction_v1"
    assert contract["source_probe_contract_digest"] == (
        "sha256:fdc92e276f7d8cb0c1ab6fd097242932851da04e1f97888d3f9597bfb0f726e0"
    )
    assert [item["radius_multiplier"] for item in contract["multiplier_registry"]] == [
        0.25,
        0.5,
        0.75,
        0.9,
    ]
    assert contract["nonclaims"] == MANDATORY_NONCLAIMS

    *_, report = _restrict()
    assert set(report) == REPORT_KEYS
    assert set(report["source_failure_boundary_probe"]) == SOURCE_PROBE_KEYS
    assert set(report["evaluator_definition"]) == EVALUATOR_DEFINITION_KEYS
    assert set(report["evaluator_binding"]) == EVALUATOR_BINDING_KEYS
    assert set(report["evidence_binding"]) == EVIDENCE_BINDING_KEYS
    assert set(report["fresh_validation"]) == FRESH_VALIDATION_KEYS
    for split in ("calibration", "holdout", "stress"):
        assert set(report["fresh_validation"][split]) == SPLIT_METRIC_KEYS
    assert len(report["candidate_registry"]) == 4
    assert len(report["candidate_competition"]) == 4
    for candidate in report["candidate_registry"]:
        assert set(candidate) == CANDIDATE_REGISTRY_KEYS
        assert set(candidate["geometry_certificate"]) == GEOMETRY_KEYS
    for candidate in report["candidate_competition"]:
        assert set(candidate) == CANDIDATE_COMPETITION_KEYS
        for split in ("calibration", "holdout", "stress"):
            assert set(candidate[split]) == SPLIT_METRIC_KEYS
    assert set(report["selected_candidate"]) == SELECTED_CANDIDATE_KEYS
    assert set(report["restricted_shadow_theory_state"]) == STATE_KEYS
    assert set(
        report["restricted_shadow_theory_state"]["evidence_reuse_policy"]
    ) == STATE_REUSE_KEYS
    assert set(
        report["restricted_shadow_theory_state"]["restriction_lineage"]
    ) == STATE_LINEAGE_KEYS
    assert set(report["restriction_certificate"]) == CERTIFICATE_KEYS
    assert set(report["rollback_boundary"]) == ROLLBACK_KEYS
    assert set(report["record_lifecycle_extension"]) == LIFECYCLE_KEYS
    lifecycle = report["record_lifecycle_extension"]
    for record_name in (
        "competition_records",
        "qualification_records",
        "failure_boundary_probe_records",
        "restriction_competition_records",
    ):
        assert set(lifecycle[record_name]) == RECORD_KEYS
    assert set(lifecycle["future_scoring_policy"]) == FUTURE_SCORING_KEYS
    assert set(report["authority_boundary"]) == AUTHORITY_KEYS
    assert report["nonclaims"] == MANDATORY_NONCLAIMS
    assert {
        item["candidate_id"]: item["candidate_commitment_digest"]
        for item in report["candidate_registry"]
    } == FROZEN_CANDIDATE_COMMITMENTS
    assert report["restricted_shadow_theory_state_digest"] == (
        FROZEN_POSITIVE_STATE_DIGEST
    )
    assert report["report_digest"] == FROZEN_POSITIVE_REPORT_DIGEST


def test_known_answer_one_quarter_fails_one_half_is_selected_and_materialized():
    *prefix, result, report = _restrict()
    source_report = prefix[12]
    source_state = source_report["probe_expanded_shadow_theory_state"]
    state = report["restricted_shadow_theory_state"]

    assert result.restriction_materialized is True
    assert result.selected_radius_multiplier == 0.5
    assert not hasattr(result, "eligible")
    assert report["disposition"] == (
        "MATERIALIZED_SHADOW_ROBUST_INTERVAL_RESTRICTION"
    )
    assert report["selected_candidate"]["candidate_id"] == "uniform_radius_1_over_2"
    by_id = {
        item["candidate_id"]: item for item in report["candidate_competition"]
    }
    assert by_id["uniform_radius_1_over_4"]["calibration_admissible"] is False
    assert by_id["uniform_radius_1_over_2"]["calibration_admissible"] is True
    assert by_id["uniform_radius_1_over_2"]["fresh_validation_passed"] is True
    assert state["model_class"]["center_predictions"] == source_state[
        "model_class"
    ]["center_predictions"]
    assert state["model_class"]["radius_grouping"] == source_state["model_class"][
        "radius_grouping"
    ]
    for source_radius, restricted_radius in zip(
        source_state["model_class"]["radii"], state["model_class"]["radii"]
    ):
        assert restricted_radius["group"] == source_radius["group"]
        assert restricted_radius["radius"] == pytest.approx(
            0.5 * source_radius["radius"]
        )


@pytest.mark.parametrize("grouping", ("global", "per_scope", "per_context"))
def test_all_frozen_radius_groupings_check_complete_cartesian_pairs(grouping):
    *prefix, report = _restrict(grouping=grouping)
    source_state = prefix[12]["probe_expanded_shadow_theory_state"]
    expected_pairs = len(source_state["scope_ids"]) * len(
        source_state["object_space"]["contexts"]
    )
    assert source_state["model_class"]["radius_grouping"] == grouping
    assert report["restriction_certificate"][
        "checked_context_scope_pair_count"
    ] == expected_pairs
    assert report["restriction_certificate"]["strict_subset_verified"] is True
    assert all(
        report["evidence_binding"]["complete_exact_cartesian_coverage_by_split"].values()
    )


@pytest.mark.parametrize("grouping", ("per_scope", "per_context"))
def test_unused_extra_radius_group_is_rejected_by_exact_registry_set(grouping):
    source = _probe_source(grouping=grouping)
    state = copy.deepcopy(source[12]["probe_expanded_shadow_theory_state"])
    parent = source[4]["parent_theory_state"]
    contract = _load(RESTRICTION_CONTRACT)
    assert len(
        restriction_core._candidate_registry(
            state,
            parent["object_space"]["contexts"],
            parent["scope_ids"],
            contract,
        )
    ) == 4
    if grouping == "per_scope":
        extra = {"group": {"scope_id": "unused-scope"}, "radius": 1.0}
    else:
        extra = {
            "group": {"context": {"x": 999, "nuisance": 999}},
            "radius": 1.0,
        }
    state["model_class"]["radii"].append(extra)
    with pytest.raises(
        ValueError,
        match="radius group keys do not exactly equal the frozen grouping registry",
    ):
        restriction_core._candidate_registry(
            state,
            parent["object_space"]["contexts"],
            parent["scope_ids"],
            contract,
        )


@pytest.mark.parametrize("grouping", ("per_scope", "per_context"))
def test_missing_required_radius_group_remains_rejected(grouping):
    source = _probe_source(grouping=grouping)
    state = copy.deepcopy(source[12]["probe_expanded_shadow_theory_state"])
    parent = source[4]["parent_theory_state"]
    contract = _load(RESTRICTION_CONTRACT)
    state["model_class"]["radii"].pop()
    with pytest.raises(
        ValueError,
        match="radius group keys do not exactly equal the frozen grouping registry",
    ):
        restriction_core._candidate_registry(
            state,
            parent["object_space"]["contexts"],
            parent["scope_ids"],
            contract,
        )


def test_q_and_v_are_byte_equal_source_is_unchanged_and_rollback_not_executed():
    source = _probe_source(grouping="per_context")
    source_state_before = canonical_json_bytes(
        source[12]["probe_expanded_shadow_theory_state"]
    )
    source_report_before = canonical_json_bytes(source[12])
    *_, report = _restrict(source=source)
    restricted = report["restricted_shadow_theory_state"]
    source_state = source[12]["probe_expanded_shadow_theory_state"]
    assert canonical_json_bytes(restricted["probe_ids"]) == canonical_json_bytes(
        source_state["probe_ids"]
    )
    assert canonical_json_bytes(
        restricted["violation_functionals"]
    ) == canonical_json_bytes(source_state["violation_functionals"])
    assert canonical_json_bytes(source[12]["probe_expanded_shadow_theory_state"]) == (
        source_state_before
    )
    assert canonical_json_bytes(source[12]) == source_report_before
    assert report["rollback_boundary"]["method"] == (
        "RESTORE_FROZEN_PROBE_EXPANDED_SOURCE_STATE_FROM_VERIFIED_PROBE_REPORT"
    )
    assert report["rollback_boundary"]["rollback_execution_status"] == "NOT_PERFORMED"
    assert report["authority_boundary"]["source_state_mutation"] is False
    assert report["authority_boundary"]["source_child_invalidation"] is False
    assert report["authority_boundary"]["rollback_execution"] is False


def test_candidate_registry_is_observation_independent_while_outcome_can_change():
    source = _probe_source()
    *_, passing = _restrict(source=source)
    *_, failing = _restrict(
        source=source,
        factors={"calibration": 1.1, "holdout": 0.4, "stress": 0.4},
    )
    assert passing["candidate_registry"] == failing["candidate_registry"]
    assert passing["evaluator_definition"] == failing["evaluator_definition"]
    assert passing["disposition"] != failing["disposition"]


@pytest.mark.parametrize(
    ("factors", "expected"),
    (
        (
            {"calibration": 1.1, "holdout": 0.4, "stress": 0.4},
            "NO_CALIBRATION_ADMISSIBLE_STRICT_INTERVAL_RESTRICTION",
        ),
        (
            {"calibration": 0.4, "holdout": 0.95, "stress": 0.95},
            "NO_VALIDATED_STRICT_INTERVAL_RESTRICTION",
        ),
    ),
)
def test_fixed_candidate_failure_dispositions(factors, expected):
    *_, report = _restrict(factors=factors)
    assert report["disposition"] == expected
    assert report["restricted_shadow_theory_state"] is None
    assert report["restriction_certificate"] is None


@pytest.mark.parametrize("split", ("calibration", "holdout", "stress"))
def test_missing_any_cartesian_pair_needs_evidence_without_numeric_competition(split):
    def mutate(payload, _):
        payload["evidence"][split].pop()

    *_, report = _restrict(mutate_input=mutate)
    assert report["disposition"] == "RESTRICTION_NEEDS_NEW_EVIDENCE"
    assert report["candidate_competition"] == []
    assert report["selected_candidate"] is None
    assert report["restricted_shadow_theory_state"] is None


@pytest.mark.parametrize("field", ("evaluator_epoch", "fixed_anchor"))
def test_epoch_or_anchor_mismatch_is_incomparable_without_numeric_verdict(field):
    def mutate(payload, _):
        payload["evidence"]["stress"][0][field] = "mismatch"

    *_, report = _restrict(mutate_input=mutate)
    assert report["disposition"] == (
        "RESTRICTION_INCOMPARABLE_EVALUATOR_EPOCH"
    )
    assert report["candidate_competition"] == []
    assert report["selected_candidate"] is None
    assert report["restricted_shadow_theory_state"] is None


def test_source_counterexample_blocks_restriction():
    source = _probe_source(probe_state="counterexample")
    *_, report = _restrict(source=source)
    assert report["disposition"] == (
        "RESTRICTION_BLOCKED_SOURCE_BOUNDARY_COUNTEREXAMPLE"
    )
    assert report["candidate_competition"] == []


@pytest.mark.parametrize("source_state", ("needs", "incomparable", "blocked"))
def test_source_unresolved_probe_routes_without_restriction(source_state):
    source = _probe_source(probe_state=source_state)
    *_, report = _restrict(source=source)
    assert report["disposition"] == "RESTRICTION_BLOCKED_SOURCE_PROBE_UNRESOLVED"
    assert report["candidate_competition"] == []
    assert report["restricted_shadow_theory_state"] is None


def test_quotient_source_is_explicitly_not_applicable():
    source = _probe_source(kind="idealization")
    *_, report = _restrict(source=source)
    assert report["disposition"] == (
        "RESTRICTION_NOT_APPLICABLE_NON_ROBUST_SOURCE"
    )
    assert report["source_transition_kind"] == "QUOTIENT_IDEALIZATION"
    assert report["selected_candidate"] is None


def test_all_eight_dispositions_are_reachable_and_distinct():
    reports = []
    reports.append(_restrict()[-1])
    reports.append(
        _restrict(
            factors={"calibration": 1.1, "holdout": 0.4, "stress": 0.4}
        )[-1]
    )
    reports.append(
        _restrict(
            factors={"calibration": 0.4, "holdout": 0.95, "stress": 0.95}
        )[-1]
    )
    reports.append(
        _restrict(
            mutate_input=lambda payload, _: payload["evidence"]["calibration"].pop()
        )[-1]
    )
    reports.append(
        _restrict(
            mutate_input=lambda payload, _: payload["evidence"]["stress"][0].__setitem__(
                "evaluator_epoch", "other"
            )
        )[-1]
    )
    reports.append(_restrict(source=_probe_source(probe_state="counterexample"))[-1])
    reports.append(_restrict(source=_probe_source(probe_state="needs"))[-1])
    reports.append(_restrict(source=_probe_source(kind="idealization"))[-1])
    assert {report["disposition"] for report in reports} == ALL_DISPOSITIONS


def _all_prior_ids(source):
    competition = {
        row["observation_id"]
        for rows in source[0]["evidence"].values()
        for row in rows
    }
    qualification = {
        row["observation_id"]
        for rows in source[5]["evidence"].values()
        for row in rows
    }
    boundary = {
        row["observation_id"]
        for rows in source[10]["evidence"].values()
        for row in rows
    }
    return competition, qualification, boundary


@pytest.mark.parametrize("generation", (0, 1, 2))
def test_three_prior_generations_of_observation_ids_are_excluded(generation):
    source = _probe_source()
    reused = next(iter(_all_prior_ids(source)[generation]))

    def mutate(payload, _):
        payload["evidence"]["calibration"][0]["observation_id"] = reused

    with pytest.raises(ValueError):
        _restrict(source=source, mutate_input=mutate)


def test_cross_split_duplicate_id_and_nonfinite_value_fail_closed():
    source = _probe_source()

    def duplicate(payload, _):
        payload["evidence"]["stress"][0]["observation_id"] = payload["evidence"][
            "calibration"
        ][0]["observation_id"]

    with pytest.raises(ValueError):
        _restrict(source=source, mutate_input=duplicate)

    def nonfinite(payload, _):
        payload["evidence"]["holdout"][0]["observed_value"] = float("nan")

    with pytest.raises(ValueError):
        _restrict(source=source, mutate_input=nonfinite)


def test_epoch_and_candidate_commitments_do_not_depend_on_values_or_row_order():
    source = _probe_source()
    first_input = _restriction_input(source)
    second_input = copy.deepcopy(first_input)
    for split, rows in second_input["evidence"].items():
        rows.reverse()
        for row in rows:
            row["observed_value"] += 0.01
    assert first_input["evaluator"] == second_input["evaluator"]
    contract = _load(RESTRICTION_CONTRACT)
    first = _as_dict(
        compete_and_materialize_shadow_robust_interval_restriction(
            *source,
            first_input,
            contract,
            **_kwargs(source, first_input, contract),
        )
    )
    second = _as_dict(
        compete_and_materialize_shadow_robust_interval_restriction(
            *source,
            second_input,
            contract,
            **_kwargs(source, second_input, contract),
        )
    )
    assert first["candidate_registry"] == second["candidate_registry"]


def test_public_verifier_receipt_and_semantic_tamper_rehash_fail_closed():
    values = _restrict()
    source = values[:13]
    restriction_input, restriction_contract, _, report = values[13:]
    receipt = _verify(source, restriction_input, restriction_contract, report)
    assert receipt == {
        "status": "VERIFIED_MATERIALIZED_SHADOW_ROBUST_INTERVAL_RESTRICTION",
        "disposition": report["disposition"],
        "report_digest": report["report_digest"],
        "contract_digest": report["contract_digest"],
        "restriction_id": report["restriction_id"],
        "source_probe_report_digest": source[12]["report_digest"],
        "source_probe_expanded_shadow_theory_state_digest": source[12][
            "probe_expanded_shadow_theory_state_digest"
        ],
        "restricted_shadow_theory_state_digest": report[
            "restricted_shadow_theory_state_digest"
        ],
        "restriction_materialized": True,
        "selected_radius_multiplier": 0.5,
        "adoption_eligibility": "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED",
        "adoption_status": "NOT_ADOPTED_SHADOW_ONLY",
        "promotion_status": "NOT_PROMOTED",
        "current_status": "NOT_CURRENT",
    }

    tampered = copy.deepcopy(report)
    tampered["restriction_certificate"]["strict_subset_verified"] = False
    _rehash(tampered)
    with pytest.raises(ValueError):
        _verify(
            source,
            restriction_input,
            restriction_contract,
            tampered,
            expected_restriction_report_digest=tampered["report_digest"],
        )


@pytest.mark.parametrize(
    "target",
    (
        "audit",
        "candidate",
        "state",
        "rollback",
        "authority",
    ),
)
def test_rehashed_report_tampering_is_rejected(target):
    values = _restrict()
    source = values[:13]
    restriction_input, restriction_contract, _, report = values[13:]
    tampered = copy.deepcopy(report)
    if target == "audit":
        tampered["audit_events"][0]["event"] = "FORGED"
    elif target == "candidate":
        tampered["selected_candidate"]["radius_multiplier"] = 0.75
    elif target == "state":
        tampered["restricted_shadow_theory_state"]["current_status"] = "CURRENT"
    elif target == "rollback":
        tampered["rollback_boundary"]["rollback_execution_status"] = "PERFORMED"
    elif target == "authority":
        tampered["authority_boundary"]["adoption_decision"] = True
    _rehash(tampered)
    with pytest.raises(ValueError):
        _verify(
            source,
            restriction_input,
            restriction_contract,
            tampered,
            expected_restriction_report_digest=tampered["report_digest"],
        )


def test_all_fourteen_independent_digest_anchors_fail_closed():
    values = _restrict()
    source = values[:13]
    restriction_input, restriction_contract = values[13:15]
    kwargs = _kwargs(source, restriction_input, restriction_contract)
    keys = [key for key in kwargs if key.startswith("expected_") and key.endswith("_digest")]
    assert len(keys) == 14
    for key in keys:
        forged = dict(kwargs)
        forged[key] = "sha256:" + "0" * 64
        with pytest.raises(ValueError):
            compete_and_materialize_shadow_robust_interval_restriction(
                *source, restriction_input, restriction_contract, **forged
            )


def import_runner():
    spec = importlib.util.spec_from_file_location(
        "restriction_runner", RESTRICTION_RUNNER
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
    )
    return list(itertools.chain.from_iterable((f"--{name}", str(path)) for name, path in zip(names, paths)))


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
    )
    return list(
        itertools.chain.from_iterable(
            (f"--expected-{name}-digest", values[name.replace('-', '_')])
            for name in names
        )
    )


def _materialize_cli_inputs(tmp_path):
    paths = tuple(
        (tmp_path / f"input-{index:02d}.json").resolve()
        for index in range(15)
    )

    def run(command):
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stderr.decode()
        return completed.stdout

    qualification_kat = review_kat.qualification_kat
    case = qualification_kat._case("robustification")
    competition_contract = _load(COMPETITION_CONTRACT)
    _write(paths[0], case)
    _write(paths[1], competition_contract)
    paths[2].write_bytes(
        run(
            [
                sys.executable,
                str(probe_kat.COMPETITION_RUNNER),
                "--input",
                str(paths[0]),
                "--contract",
                str(paths[1]),
            ]
        )
    )
    competition_report = _load(paths[2])

    transition_contract = _load(TRANSITION_CONTRACT)
    _write(paths[3], transition_contract)
    paths[4].write_bytes(
        run(
            [
                sys.executable,
                str(probe_kat.TRANSITION_RUNNER),
                "--competition-input",
                str(paths[0]),
                "--competition-contract",
                str(paths[1]),
                "--competition-report",
                str(paths[2]),
                "--transition-contract",
                str(paths[3]),
                "--expected-competition-contract-digest",
                _digest_value(competition_contract),
                "--expected-competition-report-digest",
                competition_report["report_digest"],
                "--expected-transition-contract-digest",
                _digest_value(transition_contract),
            ]
        )
    )
    transition_report = _load(paths[4])
    chain = (
        case,
        competition_contract,
        competition_report,
        transition_contract,
        transition_report,
    )

    qualification_input = qualification_kat._qualification_input(
        "robustification", chain
    )
    qualification_contract = _load(QUALIFICATION_CONTRACT)
    _write(paths[5], qualification_input)
    _write(paths[6], qualification_contract)
    paths[7].write_bytes(
        run(
            [
                sys.executable,
                str(probe_kat.QUALIFICATION_RUNNER),
                "--competition-input",
                str(paths[0]),
                "--competition-contract",
                str(paths[1]),
                "--competition-report",
                str(paths[2]),
                "--transition-contract",
                str(paths[3]),
                "--transition-report",
                str(paths[4]),
                "--qualification-input",
                str(paths[5]),
                "--qualification-contract",
                str(paths[6]),
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
            ]
        )
    )
    qualification_report = _load(paths[7])

    review_contract = _load(REVIEW_CONTRACT)
    _write(paths[8], review_contract)
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
    paths[9].write_bytes(
        run(
            [
                sys.executable,
                str(probe_kat.REVIEW_RUNNER),
                "--competition-input",
                str(paths[0]),
                "--competition-contract",
                str(paths[1]),
                "--competition-report",
                str(paths[2]),
                "--transition-contract",
                str(paths[3]),
                "--transition-report",
                str(paths[4]),
                "--qualification-input",
                str(paths[5]),
                "--qualification-contract",
                str(paths[6]),
                "--qualification-report",
                str(paths[7]),
                "--review-contract",
                str(paths[8]),
                *review_kat._digest_flags(review_values),
            ]
        )
    )
    review_report = _load(paths[9])
    source_before_probe = (
        *chain,
        qualification_input,
        qualification_contract,
        qualification_report,
        review_contract,
        review_report,
    )

    probe_input = probe_kat._probe_input(
        "robustification", source_before_probe, counterexample=False
    )
    probe_contract = _load(PROBE_CONTRACT)
    _write(paths[10], probe_input)
    _write(paths[11], probe_contract)
    probe_values = {
        **review_values,
        "review_report": review_report["report_digest"],
        "probe_input": _digest_value(probe_input),
        "probe_contract": _digest_value(probe_contract),
    }
    paths[12].write_bytes(
        run(
            [
                sys.executable,
                str(probe_kat.PROBE_RUNNER),
                *probe_kat._input_flags(paths[:12]),
                *probe_kat._digest_flags(probe_values),
            ]
        )
    )
    probe_report = _load(paths[12])
    source = (*source_before_probe, probe_input, probe_contract, probe_report)
    restriction_input = _restriction_input(source)
    restriction_contract = _load(RESTRICTION_CONTRACT)
    _write(paths[13], restriction_input)
    _write(paths[14], restriction_contract)
    values = {
        **probe_values,
        "probe_report": probe_report["report_digest"],
        "restriction_input": _digest_value(restriction_input),
        "restriction_contract": _digest_value(restriction_contract),
    }
    return paths, values


def test_cli_canonical_stdout_atomic_out_exact_15_artifacts_and_no_input_write(tmp_path):
    paths, values = _materialize_cli_inputs(tmp_path)
    before = {path: path.read_bytes() for path in paths}
    out = (tmp_path / "restriction-report.json").resolve()
    completed = subprocess.run(
        [
            sys.executable,
            str(RESTRICTION_RUNNER),
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
    assert len(report["input_artifacts"]) == 15


@pytest.mark.parametrize(
    "raw",
    (
        b'{"restriction":"a","restriction":"b"}\n',
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
        b'[]\n',
    ),
)
def test_cli_rejects_duplicate_nonfinite_and_nonobject_json_before_core(raw, tmp_path):
    paths = [(tmp_path / f"input-{index}.json").resolve() for index in range(15)]
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
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(RESTRICTION_RUNNER),
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
    paths = [(tmp_path / f"input-{index}.json").resolve() for index in range(15)]
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


def test_all_105_input_path_and_hardlink_alias_pairs_are_rejected(tmp_path):
    runner = import_runner()
    checked_same = 0
    checked_hard = 0
    for first in range(15):
        for second in range(first + 1, 15):
            pair_root = tmp_path / f"pair-{first}-{second}"
            pair_root.mkdir()
            paths = []
            for index in range(15):
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
    assert checked_same == 105
    assert checked_hard == 105


def test_surface_has_no_execution_network_or_prior_file_mutation():
    before = {
        path: _digest_file(path) for path in (*PREVIOUS_SLICE_FILES, OLD_BENCHMARK)
    }
    for path in (RESTRICTION_CORE, RESTRICTION_RUNNER):
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

    _restrict()
    assert {path: _digest_file(path) for path in before} == before


def test_documentation_freezes_bounded_additive_non_authoritative_claims():
    text = RESTRICTION_DOC.read_text(encoding="utf-8")
    assert "strictly additive V1" in text
    assert "strict subset" in text
    assert "caller-supplied static" in text
    assert "exactly fifteen" in text
    assert "105" in text
    assert "Operations Research" in text
    assert "not a generic restriction engine" in text
    assert "another fresh" in text and "evaluator epoch" in text
    assert "NOT_PERFORMED" in text
