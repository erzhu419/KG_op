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

import test_shadow_interval_multi_q_recompetition_adapter as adapter_kat  # noqa: E402
import performance.shadow_interval_multi_q_theory_operation_competition as interval_core  # noqa: E402

from performance.shadow_interval_multi_q_theory_operation_competition import (  # noqa: E402
    derive_shadow_interval_multi_q_theory_operation_competition_epoch,
    derive_shadow_interval_multi_q_theory_operation_competition_id,
    run_shadow_interval_multi_q_theory_operation_competition,
    synthesize_shadow_interval_multi_q_theory_operation_candidates,
    validate_shadow_interval_multi_q_theory_operation_competition_contract,
    verify_shadow_interval_multi_q_theory_operation_competition,
)
from performance.theory_operation_competition import (  # noqa: E402
    canonical_json_bytes,
)


COMPETITION_CONTRACT = adapter_kat.COMPETITION_CONTRACT
TRANSITION_CONTRACT = adapter_kat.TRANSITION_CONTRACT
QUALIFICATION_CONTRACT = adapter_kat.QUALIFICATION_CONTRACT
REVIEW_CONTRACT = adapter_kat.REVIEW_CONTRACT
PROBE_CONTRACT = adapter_kat.PROBE_CONTRACT
RESTRICTION_CONTRACT = adapter_kat.RESTRICTION_CONTRACT
ADJUDICATION_CONTRACT = adapter_kat.ADJUDICATION_CONTRACT
ADAPTER_CONTRACT = adapter_kat.ADAPTER_CONTRACT
INTERVAL_COMPETITION_CONTRACT = (
    ROOT
    / "performance/manifests/shadow_interval_multi_q_theory_operation_competition_v2.json"
)
INTERVAL_COMPETITION_CORE = (
    ROOT / "performance/shadow_interval_multi_q_theory_operation_competition.py"
)
INTERVAL_COMPETITION_RUNNER = (
    ROOT / "runners/run_shadow_interval_multi_q_theory_operation_competition.py"
)
INTERVAL_COMPETITION_DOC = (
    ROOT / "docs/shadow_interval_multi_q_theory_operation_competition_v2.md"
)

PREVIOUS_SLICE_FILES = (
    *adapter_kat.PREVIOUS_SLICE_FILES,
    adapter_kat.ADAPTER_CORE,
    adapter_kat.ADAPTER_CONTRACT,
    adapter_kat.ADAPTER_RUNNER,
    ROOT / "tests/test_shadow_interval_multi_q_recompetition_adapter.py",
    adapter_kat.ADAPTER_DOC,
)

ALL_DISPOSITIONS = {
    "SELECT_SHADOW_INTERVAL_EXPANSION_CANDIDATE",
    "SELECT_SHADOW_UNIFORM_RESTRICTION_CANDIDATE",
    "SELECT_SHADOW_CONSERVATIVE_QUOTIENT_ENVELOPE_CANDIDATE",
    "INTERVAL_MULTI_Q_COMPETITION_NEEDS_EXACT_FRESH_EVIDENCE",
    "INTERVAL_MULTI_Q_COMPETITION_INCOMPARABLE_EVALUATOR_EPOCH",
    "INTERVAL_MULTI_Q_COMPETITION_EARLY_DIAGNOSTIC_UNRESOLVED",
    "INTERVAL_MULTI_Q_COMPETITION_NO_VALIDATION_WINNER",
    "INTERVAL_MULTI_Q_COMPETITION_PROVISIONAL_WINNER_FAILED_STRESS_CONFIRMATION",
    "INTERVAL_MULTI_Q_COMPETITION_BLOCKED_ADAPTER_NEEDS_POST_RESTRICTION_EVIDENCE",
    "INTERVAL_MULTI_Q_COMPETITION_BLOCKED_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH",
}

FROZEN_CONTRACT_DIGEST = (
    "sha256:4c30c0b1a2cdec92ab1676e98677b620907bb9652bff1ce71865fce9d45ccd1e"
)

REPORT_KEYS = {
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

CANDIDATE_KEYS = {
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
    values = adapter_kat._adapt(
        mode=mode, mutate_adjudication_input=mutate_adjudication_input
    )
    # Omit the adapter result wrapper while retaining its canonical report.
    return (*values[:21], values[-1])


def _source_adapter_fields(source):
    report = source[21]
    seed = report["recompetition_seed"]
    return {
        "adapter_contract_digest": _digest_value(source[20]),
        "adapter_report_digest": report["report_digest"],
        "adapter_input_digest": _digest_value(source[19]),
        "adapter_id": report["adapter_id"],
        "adapter_disposition": report["disposition"],
        "recompetition_seed_digest": report["recompetition_seed_digest"],
        "recompetition_seed_id": None if seed is None else seed["seed_id"],
        "seed_theory_state_digest": (
            None if seed is None else seed["theory_state_digest"]
        ),
    }


def _rows(seed, epoch, residuals=None):
    state = seed["theory_state"]
    centers = {
        canonical_json_bytes(item["context"]): item["value"]
        for item in state["model_class"]["center_predictions"]
    }
    radii = {
        canonical_json_bytes(item["group"]): item["radius"]
        for item in state["model_class"]["radii"]
    }
    grouping = state["model_class"]["radius_grouping"]
    counts = {"discovery": 2, "validation": 1, "stress": 1}
    evidence = {split: [] for split in counts}
    for split, count in counts.items():
        for scope_index, scope in enumerate(state["scope_ids"]):
            for context_index, context in enumerate(state["object_space"]["contexts"]):
                if grouping == "global":
                    radius_key = canonical_json_bytes({"global": "*"})
                elif grouping == "per_scope":
                    radius_key = canonical_json_bytes({"scope_id": scope})
                else:
                    radius_key = canonical_json_bytes({"context": context})
                center = centers[canonical_json_bytes(context)]
                radius = radii[radius_key]
                for repetition in range(count):
                    residual = (
                        0.0
                        if residuals is None
                        else residuals(
                            split,
                            scope_index,
                            context_index,
                            repetition,
                            center,
                            radius,
                        )
                    )
                    evidence[split].append(
                        {
                            "observation_id": (
                                f"v2-fresh-{split}-{scope_index}-{context_index}-"
                                f"{repetition}"
                            ),
                            "evaluator_epoch": epoch,
                            "fixed_anchor": state["fixed_anchor"],
                            "scope_id": scope,
                            "context": copy.deepcopy(context),
                            "observed_value": center + residual,
                        }
                    )
    return evidence


def _interval_input(source, *, residuals=None, mutate=None):
    contract = _load(INTERVAL_COMPETITION_CONTRACT)
    report = source[21]
    source_adapter = _source_adapter_fields(source)
    competition_id = derive_shadow_interval_multi_q_theory_operation_competition_id(
        adapter_contract_digest=source_adapter["adapter_contract_digest"],
        adapter_report_digest=source_adapter["adapter_report_digest"],
        recompetition_seed_digest=source_adapter["recompetition_seed_digest"],
        seed_theory_state_digest=source_adapter["seed_theory_state_digest"],
        interval_competition_contract=contract,
    )
    seed = report["recompetition_seed"]
    if seed is None:
        fixed_anchor = None
        epoch = None
        evidence = {"discovery": [], "validation": [], "stress": []}
    else:
        fixed_anchor = seed["theory_state"]["fixed_anchor"]
        epoch = derive_shadow_interval_multi_q_theory_operation_competition_epoch(
            adapter_contract_digest=source_adapter["adapter_contract_digest"],
            adapter_report_digest=source_adapter["adapter_report_digest"],
            recompetition_seed_digest=source_adapter["recompetition_seed_digest"],
            seed_theory_state_digest=source_adapter["seed_theory_state_digest"],
            fixed_anchor=fixed_anchor,
            interval_competition_contract=contract,
        )
        evidence = _rows(seed, epoch, residuals=residuals)
    payload = {
        "schema_version": contract["input_schema_version"],
        "competition_id": competition_id,
        "source_adapter": source_adapter,
        "evaluator": {"evaluator_epoch": epoch, "fixed_anchor": fixed_anchor},
        "prior_record_exclusion": (
            copy.deepcopy(report["recompetition_seed"]["prior_record_exclusion"])
            if seed is not None
            else copy.deepcopy(source[19]["prior_record_exclusion"])
        ),
        "evidence": evidence,
    }
    if mutate is not None:
        mutate(payload, source)
    return payload, contract


def _kwargs(source, interval_input, contract, *, input_artifacts=None):
    kwargs = adapter_kat._kwargs(source[:19], source[19], source[20])
    kwargs.pop("input_artifacts")
    kwargs.update(
        {
            "expected_adapter_report_digest": source[21]["report_digest"],
            "expected_adapter_input_artifacts": None,
            "expected_interval_competition_input_digest": _digest_value(
                interval_input
            ),
            "expected_interval_competition_contract_digest": _digest_value(contract),
            "input_artifacts": input_artifacts,
        }
    )
    return kwargs


def _run(
    *, source=None, mode="retain", residuals=None, mutate_input=None,
    mutate_adjudication_input=None,
):
    source = source or _source(
        mode=mode, mutate_adjudication_input=mutate_adjudication_input
    )
    interval_input, contract = _interval_input(
        source, residuals=residuals, mutate=mutate_input
    )
    result = run_shadow_interval_multi_q_theory_operation_competition(
        *source,
        interval_input,
        contract,
        **_kwargs(source, interval_input, contract),
    )
    return (*source, interval_input, contract, result, _as_dict(result))


def _verify(source, interval_input, contract, report, **overrides):
    kwargs = _kwargs(source, interval_input, contract)
    kwargs.pop("input_artifacts")
    kwargs.update(
        {
            "expected_interval_competition_report_digest": report["report_digest"],
            "expected_interval_competition_input_artifacts": None,
        }
    )
    kwargs.update(overrides)
    return verify_shadow_interval_multi_q_theory_operation_competition(
        *source, interval_input, contract, report, **kwargs
    )


def _rehash(report):
    report["report_digest"] = _digest_value(
        {key: value for key, value in report.items() if key != "report_digest"}
    )
    return report


def _expansion_residual(split, _scope, context, repetition, _center, _radius):
    if split == "discovery":
        # Exactly two raw violations in one small-radius cell.  The other six
        # residuals balance the global mean while staying inside their source
        # intervals.  This clears reestimate/noise/mixture without hiding the
        # discovery envelope expansion.
        return 0.4 if context == 1 else -(0.8 / 6.0)
    return 0.3 if context == 1 else 0.0


def _quotient_residual(split, _scope, context, _repetition, center, _radius):
    if split == "discovery":
        return 0.0
    quotient_center = 4.825 if context in (0, 2) else 7.175
    return (quotient_center - center) / 2.0


def _no_winner_residual(split, _scope, _context, _repetition, _center, radius):
    return 0.0 if split == "discovery" else 0.98 * radius


def _stress_fail_residual(split, _scope, _context, _repetition, _center, radius):
    return 0.5 * radius if split == "stress" else 0.0


def _reestimate_block_residual(split, _scope, _context, _repetition, _center, _radius):
    return 1.0 if split == "discovery" else 0.0


def _make_inexact(payload, _source):
    payload["evidence"]["discovery"].pop()


def _make_incomparable(payload, _source):
    payload["evaluator"]["evaluator_epoch"] = "other-v2-evaluator-epoch"
    for split in ("discovery", "validation", "stress"):
        for row in payload["evidence"][split]:
            row["evaluator_epoch"] = "other-v2-evaluator-epoch"


def _synthesize(source=None, *, mutate_discovery=None, mutate_seed=None):
    source = source or _source()
    interval_input, contract = _interval_input(source)
    seed = copy.deepcopy(source[21]["recompetition_seed"])
    if mutate_seed is not None:
        mutate_seed(seed)
        seed["theory_state_digest"] = _digest_value(seed["theory_state"])
    discovery = copy.deepcopy(interval_input["evidence"]["discovery"])
    if mutate_discovery is not None:
        mutate_discovery(discovery, seed)
    result = synthesize_shadow_interval_multi_q_theory_operation_candidates(
        seed, discovery, interval_input["evaluator"], contract
    )
    return source, seed, discovery, contract, result


def test_candidate_synthesis_exact_families_fields_and_conservative_hull():
    _, seed, _, contract, synthesis = _synthesize()
    assert synthesis["candidate_commitments"]["raw_candidate_count"] == 5
    assert synthesis["candidate_commitments"]["retained_candidate_count"] == 5
    assert synthesis["candidate_commitments"]["recompetition_seed_digest"] == (
        _digest_value(seed)
    )
    assert synthesis["candidate_commitments"][
        "interval_competition_contract_digest"
    ] == _digest_value(contract)
    by_family = synthesis["retained_candidates_by_family"]
    assert len(by_family["interval_robustify"]) == 0
    assert len(by_family["interval_restrict"]) == 4
    assert len(by_family["interval_quotient"]) == 1
    for candidate in synthesis["retained_candidates"]:
        assert set(candidate) == CANDIDATE_KEYS
        assert candidate["validation_evaluation"] is None
        assert candidate["probe_ids"] == interval_core.PROBE_IDS

    quotient = by_family["interval_quotient"][0]
    certificate = quotient["certificate"]
    assert certificate["center_policy"] == (
        "SOURCE_INTERVAL_HULL_MIDPOINT_L_OVER_2_PLUS_U_OVER_2"
    )
    assert certificate["quotient_radius_grouping"] == "per_context"
    assert certificate["source_theory_state_digest"] == seed["theory_state_digest"]
    assert certificate["source_restore_method"] == (
        "RESTORE_EXACT_VERIFIED_RECOMPETITION_SEED_THEORY_STATE"
    )
    assert certificate["all_parent_intervals_contained_under_quotient_map"] is True
    assert certificate["envelope_certificate_verified"] is True
    assert len(certificate["fiber_envelope_table"]) == 2
    for fiber in certificate["fiber_envelope_table"]:
        assert fiber["hull_midpoint"] == (
            fiber["hull_lower"] / 2.0 + fiber["hull_upper"] / 2.0
        )
        assert fiber["hull_radius"] == max(
            fiber["hull_midpoint"] - fiber["hull_lower"],
            fiber["hull_upper"] - fiber["hull_midpoint"],
        )
        for parent in fiber["parent_intervals"]:
            assert parent["parent_lower"] >= fiber["hull_lower"]
            assert parent["parent_upper"] <= fiber["hull_upper"]


@pytest.mark.parametrize("shape", ("empty", "one_per_cell", "duplicate_id"))
def test_public_synthesis_helper_requires_exact_unique_discovery_cells(shape):
    source = _source()
    interval_input, contract = _interval_input(source)
    seed = source[21]["recompetition_seed"]
    rows = copy.deepcopy(interval_input["evidence"]["discovery"])
    if shape == "empty":
        rows = []
    elif shape == "one_per_cell":
        rows = rows[::2]
    else:
        rows[1]["observation_id"] = rows[0]["observation_id"]
    with pytest.raises(ValueError):
        synthesize_shadow_interval_multi_q_theory_operation_candidates(
            seed, rows, interval_input["evaluator"], contract
        )


def test_public_synthesis_helper_accepts_exact_two_rows_per_registered_cell():
    _, _, discovery, _, synthesis = _synthesize()
    assert len(discovery) == 8
    assert synthesis["candidate_commitments"]["raw_candidate_count"] == 5


def test_structural_ids_are_intrinsic_but_overall_commitment_binds_discovery():
    _, _, _, _, first = _synthesize()

    def vary(rows, _):
        for index, row in enumerate(rows):
            row["observed_value"] += 0.01 if index % 2 == 0 else -0.01

    _, _, _, _, second = _synthesize(mutate_discovery=vary)
    structural = {"interval_restrict", "interval_quotient"}
    first_ids = {
        item["candidate_id"]
        for item in first["retained_candidates"]
        if item["candidate_family"] in structural
    }
    second_ids = {
        item["candidate_id"]
        for item in second["retained_candidates"]
        if item["candidate_family"] in structural
    }
    assert first_ids == second_ids
    assert first["candidate_commitments"]["discovery_evidence_digest"] != second[
        "candidate_commitments"
    ]["discovery_evidence_digest"]
    assert first["candidate_commitments"]["candidate_commitment_digest"] != second[
        "candidate_commitments"
    ]["candidate_commitment_digest"]


def test_expansion_is_exact_discovery_envelope_and_never_shrinks():
    def exceed(rows, seed):
        state = seed["theory_state"]
        centers = {
            canonical_json_bytes(item["context"]): item["value"]
            for item in state["model_class"]["center_predictions"]
        }
        row = rows[0]
        row["observed_value"] = centers[canonical_json_bytes(row["context"])] + 0.75

    _, seed, _, _, synthesis = _synthesize(mutate_discovery=exceed)
    expansions = synthesis["retained_candidates_by_family"]["interval_robustify"]
    assert len(expansions) == 1
    candidate = expansions[0]
    assert candidate["discovery_admissible"] is True
    assert candidate["certificate"]["strict_superset_verified"] is True
    assert candidate["certificate"]["all_expanded_radii_gte_source"] is True
    source_radii = {
        canonical_json_bytes(item["group"]): item["radius"]
        for item in seed["theory_state"]["model_class"]["radii"]
    }
    candidate_radii = {
        canonical_json_bytes(item["group"]): item["radius"]
        for item in candidate["model_class"]["radii"]
    }
    assert all(candidate_radii[key] >= value for key, value in source_radii.items())
    first_group = canonical_json_bytes({"context": seed["theory_state"]["object_space"]["contexts"][0]})
    assert candidate_radii[first_group] == 0.75


def test_raw_boundary_gate_uses_strict_finite_inequality_not_normalized_sign():
    _, seed, _, contract, _ = _synthesize()
    geometry = interval_core._model_geometry(seed["theory_state"])
    scale = interval_core._prediction_scale(
        geometry, contract["thresholds"]["numeric_epsilon"]
    )
    context = geometry["contexts"][0]
    scope = geometry["scopes"][0]
    center = geometry["centers"][canonical_json_bytes(context)]
    radius = interval_core._radius_for_pair(geometry, scope, context)
    base = {
        "evaluator_epoch": "epoch",
        "fixed_anchor": "anchor",
        "scope_id": scope,
        "context": context,
    }
    exact = {
        **base,
        "observation_id": "exact-boundary",
        "observed_value": center + radius,
    }
    beyond = {
        **base,
        "observation_id": "nextafter-boundary",
        "observed_value": center + math.nextafter(radius, float("inf")),
    }
    metrics = interval_core._split_metrics(
        [exact, beyond], geometry, scale, tail_indices=None
    )
    assert metrics["boundary_violation_count"] == 1
    assert metrics["boundary_counterexample_observation_ids"] == [
        "nextafter-boundary"
    ]


def test_source_tail_cutoff_includes_all_ties_and_ignores_ids():
    _, seed, _, contract, _ = _synthesize()
    geometry = interval_core._model_geometry(seed["theory_state"])
    scale = interval_core._prediction_scale(
        geometry, contract["thresholds"]["numeric_epsilon"]
    )
    context = geometry["contexts"][0]
    scope = geometry["scopes"][0]
    center = geometry["centers"][canonical_json_bytes(context)]
    radius = interval_core._radius_for_pair(geometry, scope, context)
    raw_exceedances = (3.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    rows = [
        {
            "observation_id": f"tail-{index}",
            "evaluator_epoch": "tail-epoch",
            "fixed_anchor": "tail-anchor",
            "scope_id": scope,
            "context": copy.deepcopy(context),
            "observed_value": (
                center
                if exceedance == 0.0
                else center + radius + exceedance
            ),
        }
        for index, exceedance in enumerate(raw_exceedances)
    ]
    indices, definition = interval_core._source_tail(rows, geometry, scale, 0.25)
    assert definition["source_tail_k"] == 2
    assert definition["source_tail_cutoff"] == 2.0
    assert indices == {0, 1, 2}
    assert definition["source_tail_row_count"] == 3
    assert definition["source_tail_statistic"] == "raw_boundary_exceedance"
    assert definition["cutoff_units"] == "source_prediction_units"
    assert definition["tail_tie_policy"] == (
        "INCLUDE_ALL_AT_OR_ABOVE_SOURCE_CUTOFF"
    )
    assert definition["observation_id_used_for_membership"] is False
    changed = list(reversed(copy.deepcopy(rows)))
    for index, row in enumerate(changed):
        row["observation_id"] = f"renamed-{index}"
    changed_indices, changed_definition = interval_core._source_tail(
        changed, geometry, scale, 0.25
    )
    assert changed_definition == definition
    original_members = sorted(rows[index]["observed_value"] for index in indices)
    changed_members = sorted(
        changed[index]["observed_value"] for index in changed_indices
    )
    assert changed_members == original_members


def test_source_tail_raw_positive_subnormal_is_not_tied_with_normalized_zero():
    _, seed, _, contract, _ = _synthesize()
    state = copy.deepcopy(seed["theory_state"])
    contexts = state["object_space"]["contexts"]
    state["model_class"]["center_predictions"] = [
        {"context": copy.deepcopy(context), "value": 1.0e308 if index == 1 else 0.0}
        for index, context in enumerate(contexts)
    ]
    state["model_class"]["radii"] = [
        {"group": {"context": copy.deepcopy(context)}, "radius": 0.0}
        for context in contexts
    ]
    geometry = interval_core._model_geometry(state)
    scale = interval_core._prediction_scale(
        geometry, contract["thresholds"]["numeric_epsilon"]
    )
    assert scale >= 1.0e307
    scope = geometry["scopes"][0]
    rows = []
    for index, context in enumerate(contexts):
        center = geometry["centers"][canonical_json_bytes(context)]
        rows.append(
            {
                "observation_id": f"subnormal-tail-{index}",
                "evaluator_epoch": "subnormal-tail-epoch",
                "fixed_anchor": "subnormal-tail-anchor",
                "scope_id": scope,
                "context": copy.deepcopy(context),
                "observed_value": math.nextafter(0.0, 1.0) if index == 0 else center,
            }
        )
    error, radius, _margin, normalized_exceedance = interval_core._row_geometry_values(
        rows[0], geometry, scale
    )
    assert error == math.nextafter(0.0, 1.0)
    assert radius == 0.0
    assert error > radius
    assert normalized_exceedance == 0.0
    indices, definition = interval_core._source_tail(rows, geometry, scale, 0.25)
    assert indices == {0}
    assert definition["source_tail_cutoff"] == math.nextafter(0.0, 1.0)
    assert definition["source_tail_row_count"] == 1
    assert definition["source_tail_statistic"] == "raw_boundary_exceedance"
    assert definition["cutoff_units"] == "source_prediction_units"


@pytest.mark.parametrize("radius", (0.0, 5e-324))
def test_zero_and_subnormal_restriction_certificates_match_stored_floats(radius):
    def tiny(seed):
        for item in seed["theory_state"]["model_class"]["radii"]:
            item["radius"] = radius

    _, _, _, _, synthesis = _synthesize(mutate_seed=tiny)
    restrictions = synthesis["retained_candidates_by_family"]["interval_restrict"]
    expected_count = 0 if radius == 0.0 else 1
    assert len(restrictions) == expected_count
    if radius == 0.0:
        assert restrictions == []
    else:
        candidate = restrictions[0]
        assert all(
            item["radius"] == 0.0 for item in candidate["model_class"]["radii"]
        )
        assert candidate["certificate"][
            "at_least_one_radius_strictly_reduced"
        ] is True
        assert candidate["certificate"]["strict_subset_verified"] is True
        assert candidate["discovery_admissible"] is True
    ledger = synthesis["candidate_semantic_deduplication"]
    assert len(ledger["dropped_duplicate_candidates"]) == (
        0 if radius == 0.0 else 1
    )
    assert ledger["semantic_deduplication_verified"] is True


def test_semantic_dedup_uses_family_then_id_order_not_labels():
    _, _, _, _, synthesis = _synthesize()
    original = copy.deepcopy(synthesis["retained_candidates"][0])
    duplicate = copy.deepcopy(original)
    duplicate["candidate_family"] = "interval_quotient"
    duplicate["operation_kind"] = "quotient"
    duplicate["candidate_id"] = "interval_quotient:duplicate"
    retained, ledger = interval_core._semantic_deduplicate([duplicate, original])
    assert [item["candidate_id"] for item in retained] == [
        original["candidate_id"]
    ]
    assert ledger["dropped_duplicate_candidates"] == [
        {
            "dropped_candidate_id": "interval_quotient:duplicate",
            "retained_candidate_id": original["candidate_id"],
            "semantic_model_digest": original["semantic_model_digest"],
        }
    ]


def test_quotient_uses_one_global_fiber_hull_across_unequal_scope_radii():
    source = _source()
    seed = copy.deepcopy(source[21]["recompetition_seed"])
    state = seed["theory_state"]
    state["scope_ids"] = ["scope-a", "scope-b"]
    state["model_class"]["radius_grouping"] = "per_scope"
    state["model_class"]["radii"] = [
        {"group": {"scope_id": "scope-a"}, "radius": 1.0},
        {"group": {"scope_id": "scope-b"}, "radius": 3.0},
    ]
    seed["theory_state_digest"] = _digest_value(state)
    contract = _load(INTERVAL_COMPETITION_CONTRACT)
    evaluator = {
        "evaluator_epoch": "multi-scope-quotient-epoch",
        "fixed_anchor": state["fixed_anchor"],
    }
    discovery = _rows(seed, evaluator["evaluator_epoch"])["discovery"]
    synthesis = synthesize_shadow_interval_multi_q_theory_operation_candidates(
        seed, discovery, evaluator, contract
    )
    quotient = synthesis["retained_candidates_by_family"]["interval_quotient"][0]
    assert quotient["model_class"]["radius_grouping"] == "per_context"
    assert quotient["certificate"]["quotient_radius_grouping"] == "per_context"
    expected_keys = {
        canonical_json_bytes({"context": {"x": 0}}),
        canonical_json_bytes({"context": {"x": 1}}),
    }
    assert {
        canonical_json_bytes(item["group"])
        for item in quotient["model_class"]["radii"]
    } == expected_keys
    fibers = quotient["certificate"]["fiber_envelope_table"]
    assert len(fibers) == 2
    assert quotient["certificate"][
        "checked_parent_context_scope_pair_count"
    ] == 8
    for fiber in fibers:
        assert len(fiber["parent_intervals"]) == 4
        assert {item["scope_id"] for item in fiber["parent_intervals"]} == {
            "scope-a",
            "scope-b",
        }
        assert fiber["hull_radius"] == 8.0
        for parent in fiber["parent_intervals"]:
            assert fiber["hull_lower"] <= parent["parent_lower"]
            assert fiber["hull_upper"] >= parent["parent_upper"]


def test_quotient_one_ulp_endpoint_failure_is_omitted_never_false_certified():
    source = _source()
    state = copy.deepcopy(source[21]["recompetition_seed"]["theory_state"])
    contexts = [
        {"drop": 0, "keep": 0},
        {"drop": 1, "keep": 0},
    ]
    state["object_space"] = {
        "contexts": contexts,
        "feature_ids": ["drop", "keep"],
    }
    state["scope_ids"] = ["scope"]
    state["removable_feature_ids"] = ["drop"]
    state["model_class"] = {
        "kind": "finite_interval_table",
        "center_predictions": [
            {"context": contexts[0], "value": 0.0},
            {"context": contexts[1], "value": 7.953259946703382e213},
        ],
        "radius_grouping": "per_context",
        "radii": [
            {
                "group": {"context": contexts[0]},
                "radius": 1.4963462396354663e-226,
            },
            {"group": {"context": contexts[1]}, "radius": 0.0},
        ],
    }
    geometry = interval_core._model_geometry(state)
    rows = [
        {
            "observation_id": f"ulp-{context_index}-{repetition}",
            "evaluator_epoch": "ulp-epoch",
            "fixed_anchor": state["fixed_anchor"],
            "scope_id": "scope",
            "context": copy.deepcopy(context),
            "observed_value": geometry["centers"][canonical_json_bytes(context)],
        }
        for context_index, context in enumerate(contexts)
        for repetition in range(2)
    ]
    candidate = interval_core._quotient_candidate(
        geometry, ["drop"], rows, 1.0e214
    )
    assert candidate is None


def test_nonrepresentable_optional_quotient_is_omitted_without_losing_restrictions():
    source = _source()
    interval_input, contract = _interval_input(source)
    seed = copy.deepcopy(source[21]["recompetition_seed"])
    state = seed["theory_state"]
    contexts = state["object_space"]["contexts"]
    state["model_class"]["center_predictions"] = [
        {
            "context": copy.deepcopy(context),
            "value": (8.0e307 if index == 0 else -8.0e307 if index == 1 else 0.0),
        }
        for index, context in enumerate(contexts)
    ]
    state["model_class"]["radius_grouping"] = "global"
    state["model_class"]["radii"] = [
        {"group": {"global": "*"}, "radius": 1.0e308}
    ]
    seed["theory_state_digest"] = _digest_value(state)
    discovery = _rows(
        seed, interval_input["evaluator"]["evaluator_epoch"]
    )["discovery"]
    synthesis = synthesize_shadow_interval_multi_q_theory_operation_candidates(
        seed, discovery, interval_input["evaluator"], contract
    )
    by_family = synthesis["retained_candidates_by_family"]
    assert by_family["interval_robustify"] == []
    assert len(by_family["interval_restrict"]) == 4
    assert by_family["interval_quotient"] == []
    assert synthesis["candidate_commitments"]["raw_candidate_count"] == 4
    assert synthesis["candidate_commitments"]["retained_candidate_count"] == 4


def test_finite_extreme_signed_exceedance_is_max_scaled_before_sse():
    source = _source()
    seed = source[21]["recompetition_seed"]
    interval_input, contract = _interval_input(source)
    rows = copy.deepcopy(interval_input["evidence"]["discovery"])
    geometry = interval_core._model_geometry(seed["theory_state"])
    scale = interval_core._prediction_scale(
        geometry, contract["thresholds"]["numeric_epsilon"]
    )
    for index, row in enumerate(rows):
        center = geometry["centers"][canonical_json_bytes(row["context"])]
        row["observed_value"] = center + (1.0e200 if index % 2 == 0 else -1.0e200)
    trace, blocker = interval_core._diagnostics(rows, geometry, scale, contract)
    assert blocker == "noise"
    metrics = trace[1]["metrics"]
    assert metrics["signed_raw_boundary_exceedance_scale"] == pytest.approx(1.0e200)
    assert math.isfinite(metrics["total_max_scaled_signed_exceedance_sse"])
    assert math.isfinite(
        metrics["within_scope_context_pair_max_scaled_signed_exceedance_sse"]
    )
    assert math.isfinite(metrics["within_pair_variance_fraction"])


def _finite_operand_nonfinite_residual_synthesis_case():
    source = _source()
    interval_input, contract = _interval_input(source)
    seed = copy.deepcopy(source[21]["recompetition_seed"])
    state = seed["theory_state"]
    first_context = state["object_space"]["contexts"][0]
    state["model_class"]["center_predictions"][0]["value"] = -sys.float_info.max
    for item in state["model_class"]["radii"]:
        item["radius"] = 0.0
    seed["theory_state_digest"] = _digest_value(state)
    rows = _rows(seed, interval_input["evaluator"]["evaluator_epoch"])["discovery"]
    for row in rows:
        if row["context"] == first_context:
            row["observed_value"] = sys.float_info.max
    return seed, rows, interval_input["evaluator"], contract


def test_public_synthesis_finite_operands_nonfinite_residual_fails_closed():
    seed, rows, evaluator, contract = _finite_operand_nonfinite_residual_synthesis_case()
    assert all(math.isfinite(row["observed_value"]) for row in rows)
    assert all(
        math.isfinite(item["value"])
        for item in seed["theory_state"]["model_class"]["center_predictions"]
    )
    with pytest.raises(ValueError) as captured:
        synthesize_shadow_interval_multi_q_theory_operation_candidates(
            seed, rows, evaluator, contract
        )
    assert not isinstance(captured.value, OverflowError)
    assert "finite" in str(captured.value).lower()


@pytest.mark.parametrize("dimension", ("contexts", "scopes", "removable"))
def test_finite_registry_bounds_fail_closed_before_candidate_explosion(dimension):
    source = _source()
    seed = copy.deepcopy(source[21]["recompetition_seed"])
    state = seed["theory_state"]
    if dimension == "contexts":
        contexts = [{"nuisance": index, "x": index} for index in range(65)]
        state["object_space"]["contexts"] = contexts
        state["model_class"]["center_predictions"] = [
            {"context": context, "value": float(index)}
            for index, context in enumerate(contexts)
        ]
        state["model_class"]["radii"] = [
            {"group": {"context": context}, "radius": 1.0}
            for context in contexts
        ]
    elif dimension == "scopes":
        state["scope_ids"] = [f"scope-{index}" for index in range(17)]
    else:
        state["removable_feature_ids"] = [f"feature-{index}" for index in range(9)]
    seed["theory_state_digest"] = _digest_value(state)
    interval_input, contract = _interval_input(source)
    with pytest.raises(ValueError):
        synthesize_shadow_interval_multi_q_theory_operation_candidates(
            seed,
            interval_input["evidence"]["discovery"],
            interval_input["evaluator"],
            contract,
        )


@pytest.fixture(scope="module")
def known_cases():
    return {
        "restriction": _run(),
        "expansion": _run(residuals=_expansion_residual),
        "quotient": _run(residuals=_quotient_residual),
        "needs_evidence": _run(mutate_input=_make_inexact),
        "incomparable": _run(mutate_input=_make_incomparable),
        "diagnostic": _run(residuals=_reestimate_block_residual),
        "no_winner": _run(residuals=_no_winner_residual),
        "stress_failed": _run(residuals=_stress_fail_residual),
        "blocked_adapter_evidence": _run(
            mutate_adjudication_input=lambda payload, _: payload["evidence"][
                "holdout"
            ].pop()
        ),
        "blocked_adapter_epoch": _run(
            mutate_adjudication_input=lambda payload, _: payload["evidence"][
                "stress"
            ][0].__setitem__("evaluator_epoch", "other-epoch")
        ),
    }


def test_all_ten_dispositions_are_total_and_distinct(known_cases):
    reports = {name: values[-1] for name, values in known_cases.items()}
    assert {report["disposition"] for report in reports.values()} == ALL_DISPOSITIONS
    expected = {
        "restriction": "SELECT_SHADOW_UNIFORM_RESTRICTION_CANDIDATE",
        "expansion": "SELECT_SHADOW_INTERVAL_EXPANSION_CANDIDATE",
        "quotient": "SELECT_SHADOW_CONSERVATIVE_QUOTIENT_ENVELOPE_CANDIDATE",
        "needs_evidence": "INTERVAL_MULTI_Q_COMPETITION_NEEDS_EXACT_FRESH_EVIDENCE",
        "incomparable": "INTERVAL_MULTI_Q_COMPETITION_INCOMPARABLE_EVALUATOR_EPOCH",
        "diagnostic": "INTERVAL_MULTI_Q_COMPETITION_EARLY_DIAGNOSTIC_UNRESOLVED",
        "no_winner": "INTERVAL_MULTI_Q_COMPETITION_NO_VALIDATION_WINNER",
        "stress_failed": (
            "INTERVAL_MULTI_Q_COMPETITION_PROVISIONAL_WINNER_FAILED_STRESS_CONFIRMATION"
        ),
        "blocked_adapter_evidence": (
            "INTERVAL_MULTI_Q_COMPETITION_BLOCKED_ADAPTER_NEEDS_POST_RESTRICTION_EVIDENCE"
        ),
        "blocked_adapter_epoch": (
            "INTERVAL_MULTI_Q_COMPETITION_BLOCKED_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH"
        ),
    }
    assert {name: report["disposition"] for name, report in reports.items()} == expected


@pytest.mark.parametrize(
    ("case", "family", "operation_kind"),
    (
        ("restriction", "interval_restrict", "restrict"),
        ("expansion", "interval_robustify", "expand"),
        ("quotient", "interval_quotient", "quotient"),
    ),
)
def test_three_winners_are_validation_selected_stress_confirmed_proposals_only(
    known_cases, case, family, operation_kind
):
    values = known_cases[case]
    result, report = values[-2], values[-1]
    assert set(report) == REPORT_KEYS
    assert result.candidate_selected is True
    assert result.selected_candidate_id == report["selected_candidate"]["candidate_id"]
    assert result.selected_operation_kind == operation_kind
    assert report["selected_candidate"]["candidate_family"] == family
    assert report["selected_candidate"]["operation_kind"] == operation_kind
    assert report["validation_selection"]["status"] == "UNIQUE_PROVISIONAL_WINNER"
    assert report["stress_confirmation"]["status"] == (
        "PROVISIONAL_WINNER_STRESS_CONFIRMED"
    )
    assert report["stress_confirmation"]["stress_score"] is None
    assert report["stress_confirmation"]["fallback_candidate_evaluated"] is False
    assert report["stress_confirmation"]["fallback_candidate_selected"] is False
    assert report["selection_boundary"]["candidate_materialized"] is False
    assert report["selection_boundary"]["shadow_theory_state_created"] is False
    assert report["selection_boundary"]["transition_authorized"] is False
    assert report["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert report["promotion_status"] == "NOT_PROMOTED"
    assert report["current_status"] == "NOT_CURRENT"
    assert not hasattr(result, "selected_candidate_family")
    assert not hasattr(result, "materialize")
    assert not hasattr(result, "adopt")
    assert not hasattr(result, "make_current")


def test_unqualified_source_base_can_only_yield_a_strict_nonidentity_proposal():
    values = _run(mode="both_fail")
    seed = values[21]["recompetition_seed"]
    report = values[-1]
    candidate = report["selected_candidate"]
    assert seed["seed_kind"] == "UNQUALIFIED_SOURCE_REPAIR_BASE"
    assert report["source_seed_summary"]["seed_kind"] == (
        "UNQUALIFIED_SOURCE_REPAIR_BASE"
    )
    assert candidate is not None
    source_model_digest = _digest_value(seed["theory_state"]["model_class"])
    source_semantic_digest = _digest_value(
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
    assert candidate["model_class_digest"] != source_model_digest
    assert candidate["semantic_model_digest"] != source_semantic_digest
    assert candidate["candidate_family"] == "interval_restrict"
    assert candidate["certificate"]["strict_subset_verified"] is True
    assert candidate["certificate"][
        "at_least_one_radius_strictly_reduced"
    ] is True
    assert report["selection_boundary"]["candidate_materialized"] is False
    assert report["selection_boundary"]["shadow_theory_state_created"] is False
    assert report["adoption_eligibility"] == (
        "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED"
    )
    assert report["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert report["promotion_status"] == "NOT_PROMOTED"
    assert report["current_status"] == "NOT_CURRENT"

    def remove_all_strict_operations(unqualified_seed):
        state = unqualified_seed["theory_state"]
        state["removable_feature_ids"] = []
        for item in state["model_class"]["radii"]:
            item["radius"] = 0.0

    _, zero_seed, _, _, synthesis = _synthesize(
        source=_source(mode="both_fail"), mutate_seed=remove_all_strict_operations
    )
    assert zero_seed["seed_kind"] == "UNQUALIFIED_SOURCE_REPAIR_BASE"
    assert synthesis["raw_candidates"] == []
    assert synthesis["retained_candidates"] == []
    assert synthesis["retained_candidates_by_family"] == {
        "interval_robustify": [],
        "interval_restrict": [],
        "interval_quotient": [],
    }
    assert synthesis["candidate_commitments"]["retained_candidate_count"] == 0
    assert synthesis["candidate_semantic_deduplication"][
        "retained_candidate_ids"
    ] == []


def test_result_declares_only_the_frozen_public_properties(known_cases):
    result = known_cases["restriction"][-2]
    properties = {
        name
        for name, value in vars(type(result)).items()
        if isinstance(value, property)
    }
    assert properties == {
        "disposition",
        "report_digest",
        "candidate_selected",
        "selected_candidate_id",
        "selected_operation_kind",
    }
    assert callable(result.to_dict)
    assert canonical_json_bytes(result.to_dict()) == canonical_json_bytes(
        known_cases["restriction"][-1]
    )


def test_validation_score_recomputes_from_dimensionless_components(known_cases):
    report = known_cases["restriction"][-1]
    evaluation = report["selected_candidate"]["validation_evaluation"]
    assert evaluation["validation_score_units"] == "dimensionless"
    assert report["validation_selection"]["validation_score_units"] == (
        "dimensionless"
    )
    components = evaluation["dimensionless_score_components"]
    weights = _load(INTERVAL_COMPETITION_CONTRACT)["validation_selection_policy"][
        "score_weights"
    ]
    recomputed = (
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
    assert recomputed == pytest.approx(evaluation["validation_score"], abs=1e-15)


def test_report_score_units_match_contract_on_evaluated_and_placeholder_routes(
    known_cases,
):
    contract_units = _load(INTERVAL_COMPETITION_CONTRACT)[
        "validation_selection_policy"
    ]["validation_score_units"]
    assert contract_units == "dimensionless"
    for values in known_cases.values():
        report = values[-1]
        assert report["validation_selection"]["validation_score_units"] == (
            contract_units
        )
        for key in (
            "interval_expansion_candidates",
            "uniform_restriction_candidates",
            "conservative_quotient_envelope_candidates",
        ):
            for candidate in report[key]:
                evaluation = candidate["validation_evaluation"]
                if evaluation is not None:
                    assert evaluation["validation_score_units"] == contract_units


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
def test_all_nonselection_routes_have_exact_null_selection_and_no_fallback(
    known_cases, case
):
    result, report = known_cases[case][-2], known_cases[case][-1]
    assert result.candidate_selected is False
    assert result.selected_candidate_id is None
    assert result.selected_operation_kind is None
    assert report["selected_candidate"] is None
    assert report["selection_boundary"]["selection_status"] == (
        "NO_SELECTED_SHADOW_PROPOSAL"
    )
    assert report["selection_boundary"]["candidate_materialized"] is False
    assert report["stress_confirmation"]["stress_score"] is None
    assert report["stress_confirmation"]["fallback_candidate_evaluated"] is False
    assert report["stress_confirmation"]["fallback_candidate_selected"] is False
    assert report["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert report["current_status"] == "NOT_CURRENT"


def test_stress_failure_confirms_only_provisional_winner_and_never_falls_back(
    known_cases,
):
    report = known_cases["stress_failed"][-1]
    provisional_id = report["validation_selection"]["provisional_candidate_id"]
    assert provisional_id is not None
    assert report["stress_confirmation"]["provisional_candidate_id"] == provisional_id
    assert report["stress_confirmation"]["all_gates_passed"] is False
    assert report["stress_confirmation"]["status"] == (
        "PROVISIONAL_WINNER_FAILED_STRESS_CONFIRMATION"
    )
    assert report["selected_candidate"] is None
    assert report["stress_confirmation"]["fallback_candidate_evaluated"] is False
    assert report["stress_confirmation"]["fallback_candidate_selected"] is False


def test_inexact_and_incomparable_routes_perform_no_numeric_evaluation(known_cases):
    for case in ("needs_evidence", "incomparable"):
        report = known_cases[case][-1]
        assert report["baseline_metrics"] is None
        assert report["interval_expansion_candidates"] == []
        assert report["uniform_restriction_candidates"] == []
        assert report["conservative_quotient_envelope_candidates"] == []
        assert report["validation_selection"]["provisional_candidate_id"] is None


def test_null_seed_adapter_routes_require_null_evaluator_and_empty_evidence(
    known_cases,
):
    for case in ("blocked_adapter_evidence", "blocked_adapter_epoch"):
        values = known_cases[case]
        interval_input, report = values[22], values[-1]
        assert interval_input["evaluator"] == {
            "evaluator_epoch": None,
            "fixed_anchor": None,
        }
        assert interval_input["evidence"] == {
            "discovery": [],
            "validation": [],
            "stress": [],
        }
        assert report["source_seed_summary"]["seed_emitted"] is False
        assert report["candidate_commitments"] is None
        assert report["candidate_semantic_deduplication"] is None
        assert report["interval_expansion_candidates"] == []
        assert report["uniform_restriction_candidates"] == []
        assert report["conservative_quotient_envelope_candidates"] == []
        assert report["baseline_metrics"] is None
        assert report["validation_selection"]["status"] == "NOT_PERFORMED"
        assert report["stress_confirmation"]["status"] == (
            "NOT_PERFORMED_NO_PROVISIONAL_WINNER"
        )


def test_diagnose_first_blocker_prevents_all_candidate_synthesis(known_cases):
    report = known_cases["diagnostic"][-1]
    assert report["diagnostic_trace"][0]["metric_status"] == "VIABLE_EXPLANATION"
    assert report["diagnostic_trace"][0]["gate_status"] == "BLOCKING"
    assert report["candidate_commitments"] is None
    assert report["candidate_semantic_deduplication"] is None
    assert report["interval_expansion_candidates"] == []
    assert report["uniform_restriction_candidates"] == []
    assert report["conservative_quotient_envelope_candidates"] == []
    assert report["baseline_metrics"]["prediction_scale"] is not None
    assert report["baseline_metrics"]["discovery"] is not None
    assert report["baseline_metrics"]["validation"] is None
    assert report["baseline_metrics"]["validation_source_tail_definition"] is None
    assert report["baseline_metrics"]["stress_confirmation_baseline"] is None
    assert report["validation_selection"]["status"] == "NOT_PERFORMED"
    assert report["stress_confirmation"]["status"] == (
        "NOT_PERFORMED_NO_PROVISIONAL_WINNER"
    )


def test_dynamic_authority_route_matrix_reports_only_executed_phases(known_cases):
    dynamic = (
        "candidate_synthesis_performed",
        "candidate_evaluation_performed",
        "validation_selection_performed",
        "stress_confirmation_performed",
    )
    for case in (
        "blocked_adapter_evidence",
        "blocked_adapter_epoch",
        "needs_evidence",
        "incomparable",
        "diagnostic",
    ):
        authority = known_cases[case][-1]["authority_boundary"]
        assert {key: authority[key] for key in dynamic} == {
            key: False for key in dynamic
        }
    no_winner = known_cases["no_winner"][-1]["authority_boundary"]
    assert {key: no_winner[key] for key in dynamic} == {
        "candidate_synthesis_performed": True,
        "candidate_evaluation_performed": True,
        "validation_selection_performed": True,
        "stress_confirmation_performed": False,
    }
    for case in ("restriction", "expansion", "quotient", "stress_failed"):
        authority = known_cases[case][-1]["authority_boundary"]
        assert {key: authority[key] for key in dynamic} == {
            key: True for key in dynamic
        }
    for report in (values[-1] for values in known_cases.values()):
        authority = report["authority_boundary"]
        assert authority["selected_candidate_materialized"] is False
        assert authority["shadow_theory_state_created"] is False
        assert authority["transition_authorized"] is False
        assert authority["new_probe_executed"] is False
        assert authority["language_expansion_executed"] is False
        assert authority["adoption_decided"] is False
        assert authority["promotion_decided"] is False
        assert authority["current_pointer_written"] is False


def test_stress_never_changes_commitment_ids_or_validation_ranking(known_cases):
    confirmed = known_cases["restriction"][-1]
    failed = known_cases["stress_failed"][-1]
    assert confirmed["candidate_commitments"] == failed["candidate_commitments"]
    assert confirmed["candidate_semantic_deduplication"] == failed[
        "candidate_semantic_deduplication"
    ]
    for key in (
        "interval_expansion_candidates",
        "uniform_restriction_candidates",
        "conservative_quotient_envelope_candidates",
    ):
        confirmed_surface = [
            (item["candidate_id"], item["model_class_digest"])
            for item in confirmed[key]
        ]
        failed_surface = [
            (item["candidate_id"], item["model_class_digest"])
            for item in failed[key]
        ]
        assert confirmed_surface == failed_surface
    assert confirmed["validation_selection"] == failed["validation_selection"]
    assert confirmed["stress_confirmation"]["status"] != failed[
        "stress_confirmation"
    ]["status"]


def test_validation_changes_neither_discovery_commitment_nor_candidate_identity():
    source = _source()
    first = _run(source=source)[-1]

    def change_validation(payload, _):
        for index, row in enumerate(payload["evidence"]["validation"]):
            row["observed_value"] += 0.01 * (index + 1)

    second = _run(source=source, mutate_input=change_validation)[-1]
    assert first["evidence_digests"]["validation"] != second["evidence_digests"][
        "validation"
    ]
    assert first["candidate_commitments"] == second["candidate_commitments"]
    for key in (
        "interval_expansion_candidates",
        "uniform_restriction_candidates",
        "conservative_quotient_envelope_candidates",
    ):
        assert [
            (item["candidate_id"], item["model_class_digest"])
            for item in first[key]
        ] == [
            (item["candidate_id"], item["model_class_digest"])
            for item in second[key]
        ]


@pytest.mark.parametrize("cardinality", ("missing", "extra"))
def test_registered_cell_cardinality_mismatch_routes_needs_exact_without_scoring(
    cardinality,
):
    def mutate(payload, _):
        rows = payload["evidence"]["discovery"]
        if cardinality == "missing":
            rows.pop()
        else:
            extra = copy.deepcopy(rows[0])
            extra["observation_id"] = "v2-fresh-discovery-extra-registered-cell"
            rows.append(extra)

    report = _run(mutate_input=mutate)[-1]
    assert report["disposition"] == (
        "INTERVAL_MULTI_Q_COMPETITION_NEEDS_EXACT_FRESH_EVIDENCE"
    )
    assert report["baseline_metrics"] is None
    assert report["candidate_commitments"] is None
    assert report["candidate_semantic_deduplication"] is None
    assert report["interval_expansion_candidates"] == []
    assert report["uniform_restriction_candidates"] == []
    assert report["conservative_quotient_envelope_candidates"] == []


def test_wrong_but_internally_consistent_epoch_routes_incomparable_without_scoring():
    report = _run(mutate_input=_make_incomparable)[-1]
    assert report["disposition"] == (
        "INTERVAL_MULTI_Q_COMPETITION_INCOMPARABLE_EVALUATOR_EPOCH"
    )
    assert report["baseline_metrics"] is None
    assert report["candidate_commitments"] is None
    assert report["selected_candidate"] is None


@pytest.mark.parametrize(
    "violation",
    ("duplicate_id", "prior_id", "unregistered_scope", "unregistered_context", "nan"),
)
def test_ambiguous_unregistered_reused_or_nonfinite_evidence_is_hard_error(
    violation,
):
    def mutate(payload, source):
        row = payload["evidence"]["validation"][0]
        if violation == "duplicate_id":
            row["observation_id"] = payload["evidence"]["discovery"][0][
                "observation_id"
            ]
        elif violation == "prior_id":
            old_ids = interval_core._collect_key_strings(list(source), "observation_id")
            assert old_ids
            row["observation_id"] = sorted(old_ids)[0]
        elif violation == "unregistered_scope":
            row["scope_id"] = "unregistered-scope"
        elif violation == "unregistered_context":
            row["context"] = {"nuisance": 999, "x": 999}
        else:
            row["observed_value"] = float("nan")

    with pytest.raises(ValueError):
        _run(mutate_input=mutate)


def test_all_twenty_three_independent_digest_anchors_fail_closed():
    source = _source()
    interval_input, contract = _interval_input(source)
    kwargs = _kwargs(source, interval_input, contract)
    keys = [
        key
        for key in kwargs
        if key.startswith("expected_") and key.endswith("_digest")
    ]
    assert len(keys) == 23
    for key in keys:
        forged = dict(kwargs)
        forged[key] = "sha256:" + "0" * 64
        with pytest.raises(ValueError):
            run_shadow_interval_multi_q_theory_operation_competition(
                *source, interval_input, contract, **forged
            )


def test_input_artifact_metadata_cannot_embed_observed_values():
    source = _source()
    interval_input, contract = _interval_input(source)
    kwargs = _kwargs(
        source,
        interval_input,
        contract,
        input_artifacts={"forged": {"observed_value": 0.0}},
    )
    with pytest.raises(ValueError):
        run_shadow_interval_multi_q_theory_operation_competition(
            *source, interval_input, contract, **kwargs
        )


def test_public_verifier_and_rehashed_selection_tampering_fail_closed(known_cases):
    values = known_cases["restriction"]
    source, interval_input, contract, report = (
        values[:22],
        values[22],
        values[23],
        values[-1],
    )
    receipt = _verify(source, interval_input, contract, report)
    assert receipt["status"].startswith("VERIFIED_SELECT_")
    assert receipt["report_digest"] == report["report_digest"]
    assert receipt["selected_candidate_id"] == report["selected_candidate"][
        "candidate_id"
    ]
    assert receipt["selected_operation_kind"] == "restrict"
    assert receipt["adoption_status"] == "NOT_ADOPTED_SHADOW_ONLY"
    assert receipt["current_status"] == "NOT_CURRENT"

    tampered = copy.deepcopy(report)
    tampered["selection_boundary"]["candidate_materialized"] = True
    _rehash(tampered)
    with pytest.raises(ValueError):
        _verify(
            source,
            interval_input,
            contract,
            tampered,
            expected_interval_competition_report_digest=tampered["report_digest"],
        )


@pytest.mark.parametrize(
    "target",
    ("candidate", "validation", "stress", "diagnostic", "authority", "audit"),
)
def test_rehashed_report_semantic_tampering_is_rejected(known_cases, target):
    values = known_cases["restriction"]
    source, interval_input, contract, report = (
        values[:22],
        values[22],
        values[23],
        values[-1],
    )
    tampered = copy.deepcopy(report)
    if target == "candidate":
        tampered["selected_candidate"]["candidate_id"] = "forged"
    elif target == "validation":
        tampered["validation_selection"]["provisional_validation_score"] += 1.0
    elif target == "stress":
        tampered["stress_confirmation"]["fallback_candidate_selected"] = True
    elif target == "diagnostic":
        tampered["diagnostic_trace"][0]["metric_status"] = "FORGED"
    elif target == "authority":
        tampered["authority_boundary"]["current_pointer_written"] = True
    else:
        tampered["audit_events"][0]["event"] = "FORGED"
    _rehash(tampered)
    with pytest.raises(ValueError):
        _verify(
            source,
            interval_input,
            contract,
            tampered,
            expected_interval_competition_report_digest=tampered["report_digest"],
        )


def _diagnostic_case(residual):
    source = _source()
    seed = source[21]["recompetition_seed"]
    interval_input, contract = _interval_input(source)
    rows = copy.deepcopy(interval_input["evidence"]["discovery"])
    centers = {
        canonical_json_bytes(item["context"]): item["value"]
        for item in seed["theory_state"]["model_class"]["center_predictions"]
    }
    for index, row in enumerate(rows):
        row["observed_value"] = centers[canonical_json_bytes(row["context"])] + residual(
            index // 2, index % 2
        )
    geometry = interval_core._model_geometry(seed["theory_state"])
    scale = interval_core._prediction_scale(
        geometry, contract["thresholds"]["numeric_epsilon"]
    )
    return interval_core._diagnostics(rows, geometry, scale, contract)


def _zero_radius_diagnostic_case(residuals):
    source = _source()
    state = copy.deepcopy(source[21]["recompetition_seed"]["theory_state"])
    contexts = state["object_space"]["contexts"]
    state["model_class"]["center_predictions"] = [
        {"context": copy.deepcopy(context), "value": 0.0}
        for context in contexts
    ]
    state["model_class"]["radii"] = [
        {"group": {"context": copy.deepcopy(context)}, "radius": 0.0}
        for context in contexts
    ]
    geometry = interval_core._model_geometry(state)
    contract = _load(INTERVAL_COMPETITION_CONTRACT)
    scale = interval_core._prediction_scale(
        geometry, contract["thresholds"]["numeric_epsilon"]
    )
    assert len(residuals) == 8
    rows = [
        {
            "observation_id": f"zero-radius-diagnostic-{index}",
            "evaluator_epoch": "zero-radius-diagnostic-epoch",
            "fixed_anchor": state["fixed_anchor"],
            "scope_id": geometry["scopes"][0],
            "context": copy.deepcopy(contexts[index // 2]),
            "observed_value": residual,
        }
        for index, residual in enumerate(residuals)
    ]
    return interval_core._diagnostics(rows, geometry, scale, contract)


def test_diagnostics_clear_in_interval_variation_using_signed_exceedance():
    trace, blocker = _diagnostic_case(
        lambda _context, repetition: 0.1 if repetition == 0 else -0.1
    )
    assert blocker is None
    assert [item["metric_status"] for item in trace] == [
        "NOT_APPLICABLE_NO_RAW_BOUNDARY_VIOLATION",
        "NOT_APPLICABLE_ZERO_SIGNED_EXCEEDANCE_VARIANCE",
        "NOT_APPLICABLE_SINGLE_SCOPE",
        "NOT_APPLICABLE_FEWER_THAN_FOUR_RAW_BOUNDARY_VIOLATIONS",
    ]
    assert all(item["gate_status"] == "CLEARED" for item in trace)
    assert trace[1]["metrics"]["total_max_scaled_signed_exceedance_sse"] == 0.0


def test_reestimate_max_scaling_preserves_uniform_min_subnormal_shift():
    tiny = math.nextafter(0.0, 1.0)
    trace, blocker = _zero_radius_diagnostic_case([tiny] * 8)
    assert blocker == "reestimate"
    assert trace[0]["metric_status"] == "VIABLE_EXPLANATION"
    assert trace[0]["gate_status"] == "BLOCKING"
    assert trace[0]["metrics"] == {
        "raw_residual_scale": tiny,
        "pre_rounding_max_scaled_center_shift": 1.0,
        "global_center_shift": tiny,
        "effective_max_scaled_center_shift": 1.0,
        "source_max_scaled_mean_absolute_residual": 1.0,
        "shifted_max_scaled_mean_absolute_residual": 0.0,
        "fractional_mae_gain": 1.0,
        "shifted_raw_boundary_violation_rate": 0.0,
    }


def test_reestimate_no_boundary_short_circuit_avoids_extreme_shift_arithmetic():
    source = _source()
    state = copy.deepcopy(source[21]["recompetition_seed"]["theory_state"])
    contexts = state["object_space"]["contexts"]
    state["model_class"]["center_predictions"] = [
        {"context": copy.deepcopy(context), "value": 0.0}
        for context in contexts
    ]
    radius = 1.7e308
    state["model_class"]["radius_grouping"] = "global"
    state["model_class"]["radii"] = [
        {"group": {"global": "*"}, "radius": radius}
    ]
    geometry = interval_core._model_geometry(state)
    contract = _load(INTERVAL_COMPETITION_CONTRACT)
    scale = interval_core._prediction_scale(
        geometry, contract["thresholds"]["numeric_epsilon"]
    )
    residuals = [radius] * 7 + [-radius]
    rows = [
        {
            "observation_id": f"extreme-exact-boundary-{index}",
            "evaluator_epoch": "extreme-exact-boundary-epoch",
            "fixed_anchor": state["fixed_anchor"],
            "scope_id": geometry["scopes"][0],
            "context": copy.deepcopy(contexts[index // 2]),
            "observed_value": residual,
        }
        for index, residual in enumerate(residuals)
    ]
    trace, blocker = interval_core._diagnostics(rows, geometry, scale, contract)
    assert blocker is None
    assert trace[0]["metric_status"] == (
        "NOT_APPLICABLE_NO_RAW_BOUNDARY_VIOLATION"
    )
    assert trace[0]["gate_status"] == "CLEARED"
    assert trace[0]["metrics"] == {
        "raw_residual_scale": None,
        "pre_rounding_max_scaled_center_shift": None,
        "global_center_shift": None,
        "effective_max_scaled_center_shift": None,
        "source_max_scaled_mean_absolute_residual": None,
        "shifted_max_scaled_mean_absolute_residual": None,
        "fractional_mae_gain": None,
        "shifted_raw_boundary_violation_rate": None,
    }


def test_reestimate_one_min_subnormal_among_zeros_has_no_false_gain():
    tiny = math.nextafter(0.0, 1.0)
    trace, blocker = _zero_radius_diagnostic_case([tiny, *([0.0] * 7)])
    assert blocker == "noise"
    assert trace[0]["metric_status"] == "EXCLUDED_BY_DISCOVERY"
    metrics = trace[0]["metrics"]
    assert metrics["raw_residual_scale"] == tiny
    assert metrics["pre_rounding_max_scaled_center_shift"] == 0.125
    assert metrics["global_center_shift"] == 0.0
    assert metrics["effective_max_scaled_center_shift"] == 0.0
    assert metrics["source_max_scaled_mean_absolute_residual"] == 0.125
    assert metrics["shifted_max_scaled_mean_absolute_residual"] == 0.125
    assert metrics["fractional_mae_gain"] == 0.0


def test_reestimate_gain_uses_effective_represented_shift_after_rounding():
    tiny = math.nextafter(0.0, 1.0)
    trace, blocker = _zero_radius_diagnostic_case(
        [tiny] * 7 + [-2.0e-323]
    )
    assert blocker == "noise"
    assert trace[0]["metric_status"] == "EXCLUDED_BY_DISCOVERY"
    metrics = trace[0]["metrics"]
    assert metrics["raw_residual_scale"] == 2.0e-323
    assert metrics["pre_rounding_max_scaled_center_shift"] == 0.09375
    assert metrics["global_center_shift"] == 0.0
    assert metrics["effective_max_scaled_center_shift"] == 0.0
    assert metrics["source_max_scaled_mean_absolute_residual"] == 0.34375
    assert metrics["shifted_max_scaled_mean_absolute_residual"] == 0.34375
    assert metrics["fractional_mae_gain"] == 0.0


def test_noise_and_mixture_max_scaling_preserve_min_subnormal_structure():
    tiny = math.nextafter(0.0, 1.0)
    per_context = (-2.0 * tiny, -tiny, tiny, 2.0 * tiny)
    trace, blocker = _zero_radius_diagnostic_case(
        [residual for residual in per_context for _ in range(2)]
    )
    assert trace[0]["metric_status"] == "EXCLUDED_BY_DISCOVERY"
    assert trace[1]["metric_status"] == "EXCLUDED_BY_DISCOVERY"
    assert trace[2]["metric_status"] == "NOT_APPLICABLE_SINGLE_SCOPE"
    assert blocker == "mixture"
    noise = trace[1]["metrics"]
    assert noise == {
        "signed_raw_boundary_exceedance_scale": 2.0 * tiny,
        "total_max_scaled_signed_exceedance_sse": 5.0,
        "within_scope_context_pair_max_scaled_signed_exceedance_sse": 0.0,
        "within_pair_variance_fraction": 0.0,
        "grouping": "EXACT_SCOPE_CONTEXT_PAIR",
    }
    mixture = trace[3]["metrics"]
    assert mixture == {
        "raw_boundary_violation_count": 8,
        "signed_raw_boundary_exceedance_scale": 2.0 * tiny,
        "minimum_cluster_size": 2,
        "selected_split_index": 4,
        "max_scaled_signed_exceedance_sse_reduction_fraction": 0.9,
    }


def test_reestimate_block_short_circuits_later_front_diagnostics():
    trace, blocker = _diagnostic_case(lambda _context, _repetition: 1.0)
    assert blocker == "reestimate"
    assert trace[0]["metric_status"] == "VIABLE_EXPLANATION"
    assert trace[0]["gate_status"] == "BLOCKING"
    assert [item["metric_status"] for item in trace[1:]] == [
        "NOT_EVALUATED_BLOCKED_BY_REESTIMATE",
        "NOT_EVALUATED_BLOCKED_BY_REESTIMATE",
        "NOT_EVALUATED_BLOCKED_BY_REESTIMATE",
    ]
    assert all(item["metrics"] is None for item in trace[1:])


def test_noise_block_uses_exact_scope_context_pairs_then_short_circuits():
    trace, blocker = _diagnostic_case(
        lambda _context, repetition: 1.0 if repetition == 0 else -1.0
    )
    assert blocker == "noise"
    assert trace[0]["metric_status"] == "EXCLUDED_BY_DISCOVERY"
    assert trace[1]["metric_status"] == "VIABLE_EXPLANATION"
    assert trace[1]["metrics"]["within_pair_variance_fraction"] == 1.0
    assert [item["metric_status"] for item in trace[2:]] == [
        "NOT_EVALUATED_BLOCKED_BY_NOISE",
        "NOT_EVALUATED_BLOCKED_BY_NOISE",
    ]


def test_scope_block_is_diagnostic_only_and_synthesizes_no_scope_candidate():
    source = _source()
    seed = copy.deepcopy(source[21]["recompetition_seed"])
    state = seed["theory_state"]
    state["scope_ids"] = ["scope-a", "scope-b"]
    geometry = interval_core._model_geometry(state)
    contract = _load(INTERVAL_COMPETITION_CONTRACT)
    scale = interval_core._prediction_scale(
        geometry, contract["thresholds"]["numeric_epsilon"]
    )
    rows = []
    for scope in geometry["scopes"]:
        for context_index, context in enumerate(geometry["contexts"]):
            center = geometry["centers"][canonical_json_bytes(context)]
            residual = 1.0 if scope == "scope-a" else -0.2
            for repetition in range(2):
                rows.append(
                    {
                        "observation_id": (
                            f"scope-diag-{scope}-{context_index}-{repetition}"
                        ),
                        "evaluator_epoch": "scope-diag-epoch",
                        "fixed_anchor": state["fixed_anchor"],
                        "scope_id": scope,
                        "context": copy.deepcopy(context),
                        "observed_value": center + residual,
                    }
                )
    trace, blocker = interval_core._diagnostics(rows, geometry, scale, contract)
    assert blocker == "scope"
    assert trace[0]["metric_status"] == "EXCLUDED_BY_DISCOVERY"
    assert trace[1]["metric_status"] == "EXCLUDED_BY_DISCOVERY"
    assert trace[2]["metric_status"] == "VIABLE_EXPLANATION"
    assert trace[2]["metrics"]["restriction_candidate_count"] == 0
    assert trace[3]["metric_status"] == "NOT_EVALUATED_BLOCKED_BY_SCOPE"


def test_scope_diagnostic_ignores_huge_but_in_bound_raw_residual_scale():
    source = _source()
    state = copy.deepcopy(source[21]["recompetition_seed"]["theory_state"])
    state["scope_ids"] = ["scope-a", "scope-b"]
    state["model_class"]["radius_grouping"] = "per_scope"
    state["model_class"]["radii"] = [
        {"group": {"scope_id": "scope-a"}, "radius": 1.0},
        {"group": {"scope_id": "scope-b"}, "radius": 1.0e100},
    ]
    geometry = interval_core._model_geometry(state)
    contract = _load(INTERVAL_COMPETITION_CONTRACT)
    scale = interval_core._prediction_scale(
        geometry, contract["thresholds"]["numeric_epsilon"]
    )
    rows = []
    for scope in geometry["scopes"]:
        residual = 0.5 if scope == "scope-a" else 5.0e99
        for context_index, context in enumerate(geometry["contexts"]):
            center = geometry["centers"][canonical_json_bytes(context)]
            for repetition in range(2):
                rows.append(
                    {
                        "observation_id": (
                            f"in-bound-scope-{scope}-{context_index}-{repetition}"
                        ),
                        "evaluator_epoch": "in-bound-scope-epoch",
                        "fixed_anchor": state["fixed_anchor"],
                        "scope_id": scope,
                        "context": copy.deepcopy(context),
                        "observed_value": center + residual,
                    }
                )
    trace, blocker = interval_core._diagnostics(rows, geometry, scale, contract)
    assert blocker is None
    scope_item = trace[2]
    assert scope_item["metric_status"] == (
        "NOT_APPLICABLE_NO_SCOPE_BOUNDARY_EXCEEDANCE"
    )
    assert scope_item["gate_status"] == "CLEARED"
    assert "scope_mean_absolute_residual" not in scope_item["metrics"]
    assert "scope_mean_normalized_boundary_exceedance" not in scope_item["metrics"]
    assert set(scope_item["metrics"]) == {
        "scope_raw_boundary_exceedance_scale",
        "scope_sum_max_scaled_raw_boundary_exceedance",
        "scope_raw_boundary_violation_rate",
        "scope_structure_ratio",
        "restriction_candidate_count",
    }
    assert scope_item["metrics"]["scope_raw_boundary_exceedance_scale"] == 0.0
    assert scope_item["metrics"][
        "scope_sum_max_scaled_raw_boundary_exceedance"
    ] == {"scope-a": 0.0, "scope-b": 0.0}
    assert scope_item["metrics"]["scope_raw_boundary_violation_rate"] == {
        "scope-a": 0.0,
        "scope-b": 0.0,
    }
    assert scope_item["metrics"]["scope_structure_ratio"] == 0.0


def test_scope_diagnostic_preserves_min_subnormal_raw_exceedance_without_epsilon():
    source = _source()
    state = copy.deepcopy(source[21]["recompetition_seed"]["theory_state"])
    state["scope_ids"] = ["scope-a", "scope-b"]
    contexts = state["object_space"]["contexts"]
    state["model_class"]["center_predictions"] = [
        {"context": copy.deepcopy(context), "value": 1.0e308 if index == 1 else 0.0}
        for index, context in enumerate(contexts)
    ]
    state["model_class"]["radii"] = [
        {"group": {"context": copy.deepcopy(context)}, "radius": 0.0}
        for context in contexts
    ]
    geometry = interval_core._model_geometry(state)
    contract = _load(INTERVAL_COMPETITION_CONTRACT)
    scale = interval_core._prediction_scale(
        geometry, contract["thresholds"]["numeric_epsilon"]
    )
    tiny = math.nextafter(0.0, 1.0)
    context = contexts[0]
    residuals = {
        "scope-a": (tiny, tiny),
        "scope-b": (0.0, 0.0),
    }
    rows = [
        {
            "observation_id": f"subnormal-scope-{scope}-{repetition}",
            "evaluator_epoch": "subnormal-scope-epoch",
            "fixed_anchor": state["fixed_anchor"],
            "scope_id": scope,
            "context": copy.deepcopy(context),
            "observed_value": residual,
        }
        for scope, values in residuals.items()
        for repetition, residual in enumerate(values)
    ]
    assert scale >= 1.0e307
    assert interval_core._row_geometry_values(rows[0], geometry, scale)[3] == 0.0
    trace, blocker = interval_core._diagnostics(rows, geometry, scale, contract)
    assert trace[0]["metric_status"] == "EXCLUDED_BY_DISCOVERY"
    assert trace[1]["metric_status"] == "EXCLUDED_BY_DISCOVERY"
    assert blocker == "scope"
    scope_item = trace[2]
    assert scope_item["metric_status"] == "VIABLE_EXPLANATION"
    metrics = scope_item["metrics"]
    assert metrics["scope_raw_boundary_exceedance_scale"] == tiny
    assert metrics["scope_sum_max_scaled_raw_boundary_exceedance"] == {
        "scope-a": 2.0,
        "scope-b": 0.0,
    }
    assert metrics["scope_raw_boundary_violation_rate"] == {
        "scope-a": 1.0,
        "scope-b": 0.0,
    }
    assert metrics["scope_structure_ratio"] == 1.0


def test_mixture_block_requires_four_raw_violations_and_fixed_sse_gain():
    residuals = (-2.0, -1.0, 1.0, 2.0)
    trace, blocker = _diagnostic_case(
        lambda context, _repetition: residuals[context]
    )
    assert blocker == "mixture"
    assert [item["metric_status"] for item in trace[:3]] == [
        "EXCLUDED_BY_DISCOVERY",
        "EXCLUDED_BY_DISCOVERY",
        "NOT_APPLICABLE_SINGLE_SCOPE",
    ]
    assert trace[3]["metric_status"] == "VIABLE_EXPLANATION"
    assert trace[3]["metrics"]["raw_boundary_violation_count"] == 8
    assert trace[3]["metrics"][
        "max_scaled_signed_exceedance_sse_reduction_fraction"
    ] >= 0.5


def import_runner():
    spec = importlib.util.spec_from_file_location(
        "interval_multi_q_competition_runner", INTERVAL_COMPETITION_RUNNER
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
    )
    assert len(paths) == len(names) == 24
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
    )
    return list(
        itertools.chain.from_iterable(
            (f"--expected-{name}-digest", values[name.replace("-", "_")])
            for name in names
        )
    )


def _materialize_cli_inputs(tmp_path):
    first_paths, values = adapter_kat._materialize_cli_inputs(tmp_path)
    paths = (*first_paths, (tmp_path / "input-21.json").resolve())
    completed = subprocess.run(
        [
            sys.executable,
            str(adapter_kat.ADAPTER_RUNNER),
            *adapter_kat._input_flags(first_paths),
            *adapter_kat._digest_flags(values),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    paths[21].write_bytes(completed.stdout)
    source = tuple(_load(path) for path in paths)
    interval_input, contract = _interval_input(source)
    paths = (
        *paths,
        (tmp_path / "input-22.json").resolve(),
        (tmp_path / "input-23.json").resolve(),
    )
    _write(paths[22], interval_input)
    _write(paths[23], contract)
    values = {
        **values,
        "adapter_report": source[21]["report_digest"],
        "interval_competition_input": _digest_value(interval_input),
        "interval_competition_contract": _digest_value(contract),
    }
    return paths, values


def test_cli_canonical_stdout_atomic_out_exact_24_artifacts_and_no_input_write(
    tmp_path,
):
    paths, values = _materialize_cli_inputs(tmp_path)
    before = {path: path.read_bytes() for path in paths}
    out = (tmp_path / "interval-competition-report.json").resolve()
    completed = subprocess.run(
        [
            sys.executable,
            str(INTERVAL_COMPETITION_RUNNER),
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
    assert len(artifacts) == 24
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
    }
    assert all(set(receipt) == {"bytes", "path", "sha256"} for receipt in artifacts.values())
    assert "observed_value" not in json.dumps(artifacts, sort_keys=True)


def test_cli_finite_extreme_diagnostics_are_max_scaled_and_succeed(
    tmp_path,
):
    paths, values = _materialize_cli_inputs(tmp_path)
    interval_input = _load(paths[22])
    for index, row in enumerate(interval_input["evidence"]["discovery"]):
        row["observed_value"] = 1.0e200 if index % 2 == 0 else -1.0e200
    _write(paths[22], interval_input)
    values["interval_competition_input"] = _digest_value(interval_input)
    completed = subprocess.run(
        [
            sys.executable,
            str(INTERVAL_COMPETITION_RUNNER),
            *_input_flags(paths),
            *_digest_flags(values),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    report = json.loads(completed.stdout)
    assert report["disposition"] == (
        "INTERVAL_MULTI_Q_COMPETITION_EARLY_DIAGNOSTIC_UNRESOLVED"
    )
    noise = report["diagnostic_trace"][1]
    assert noise["metric_status"] == "VIABLE_EXPLANATION"
    assert math.isfinite(
        noise["metrics"]["total_max_scaled_signed_exceedance_sse"]
    )
    assert math.isfinite(
        noise["metrics"][
            "within_scope_context_pair_max_scaled_signed_exceedance_sse"
        ]
    )
    assert completed.stdout == canonical_json_bytes(report) + b"\n"


def test_cli_maps_finite_operand_nonfinite_residual_to_exit_2_without_stdout(
    tmp_path, monkeypatch, capsys
):
    paths, values = _materialize_cli_inputs(tmp_path)
    seed, rows, evaluator, contract = _finite_operand_nonfinite_residual_synthesis_case()

    def fail_with_public_numeric_validation(*_args, **_kwargs):
        return synthesize_shadow_interval_multi_q_theory_operation_candidates(
            seed, rows, evaluator, contract
        )

    monkeypatch.setattr(
        interval_core,
        "run_shadow_interval_multi_q_theory_operation_competition",
        fail_with_public_numeric_validation,
    )
    runner = import_runner()
    returncode = runner.main([*_input_flags(paths), *_digest_flags(values)])
    captured = capsys.readouterr()
    assert returncode == 2
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert "OverflowError" not in captured.err
    assert "finite" in captured.err.lower()


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
    paths = [(tmp_path / f"input-{index}.json").resolve() for index in range(24)]
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
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(INTERVAL_COMPETITION_RUNNER),
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
    paths = [(tmp_path / f"input-{index}.json").resolve() for index in range(24)]
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


def test_all_276_input_path_and_hardlink_alias_pairs_are_rejected(tmp_path):
    runner = import_runner()
    base_paths = []
    for index in range(24):
        path = tmp_path / f"base-input-{index}.json"
        path.write_text("{}\n", encoding="utf-8")
        base_paths.append(path.resolve())
    checked_same = 0
    checked_hard = 0
    for first in range(24):
        for second in range(first + 1, 24):
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
    assert checked_same == 276
    assert checked_hard == 276


def test_contract_identity_bounds_registry_and_authority_are_frozen():
    contract = _load(INTERVAL_COMPETITION_CONTRACT)
    assert validate_shadow_interval_multi_q_theory_operation_competition_contract(
        contract
    ) == contract
    assert _digest_value(contract) == FROZEN_CONTRACT_DIGEST
    assert contract["contract_id"] == (
        "shadow_interval_multi_q_theory_operation_competition_v2"
    )
    assert contract["source_adapter_contract_digest"] == (
        adapter_kat.FROZEN_CONTRACT_DIGEST
    )
    assert contract["candidate_family_registry"] == interval_core.CANDIDATE_FAMILY_REGISTRY
    assert contract["disposition_registry"] == interval_core.DISPOSITION_REGISTRY
    assert len(contract["disposition_registry"]) == 10
    assert set(contract["disposition_registry"].values()) == ALL_DISPOSITIONS
    assert {
        item.value
        for item in interval_core.ShadowIntervalMultiQTheoryOperationCompetitionDisposition
    } == ALL_DISPOSITIONS
    assert contract["diagnostic_order"] == adapter_kat.adapter_core.DIAGNOSTIC_ORDER
    assert contract["fixed_probe_registry"] == adapter_kat.adapter_core.PROBE_IDS
    assert contract["evidence_policy"]["exact_rows_per_registered_cell"] == {
        "discovery": 2,
        "validation": 1,
        "stress": 1,
    }
    assert contract["evidence_policy"]["maximum_context_count"] == 64
    assert contract["evidence_policy"]["maximum_scope_count"] == 16
    assert contract["evidence_policy"]["maximum_removable_feature_count"] == 8
    assert contract["evidence_policy"]["maximum_raw_candidate_count"] == 260
    assert contract["evidence_policy"]["source_tail_statistic"] == (
        "raw_boundary_exceedance"
    )
    assert contract["evidence_policy"]["source_tail_cutoff_units"] == (
        "source_prediction_units"
    )
    assert contract["evidence_policy"]["source_tail_tie_policy"] == (
        "INCLUDE_ALL_AT_OR_ABOVE_SOURCE_CUTOFF"
    )
    assert contract["candidate_generation_policy"]["quotient_center_policy"] == (
        "SOURCE_INTERVAL_HULL_MIDPOINT_L_OVER_2_PLUS_U_OVER_2"
    )
    validation_policy = contract["validation_selection_policy"]
    assert validation_policy["validation_score_units"] == "dimensionless"
    assert validation_policy["score_component_units"] == {
        "normalized_center_mae_gain": "dimensionless",
        "raw_boundary_coverage_gain": "dimensionless",
        "source_tail_coverage_gain": "dimensionless",
        "context_reduction_fraction": "dimensionless",
        "uniform_contraction_fraction": "dimensionless",
        "normalized_radius_reduction": "dimensionless",
        "max_probe_divergence": "dimensionless",
        "normalized_radius_expansion": "dimensionless",
    }
    assert contract["stress_confirmation_policy"] == {
        "evaluate_only_unique_provisional_winner": True,
        "stress_score": None,
        "stress_ranking_allowed": False,
        "runner_up_stress_evaluation_allowed": False,
        "fallback_candidate_evaluated": False,
        "fallback_candidate_selected": False,
    }
    assert contract["selection"]["selection_status"] == (
        "SELECTED_SHADOW_PROPOSAL_NOT_MATERIALIZED"
    )
    assert contract["selection"]["no_selection_status"] == (
        "NO_SELECTED_SHADOW_PROPOSAL"
    )
    assert contract["authority_boundary"]["selected_candidate_materialized"] is False
    assert contract["authority_boundary"][
        "candidate_synthesis_allowed_only_after_exact_evidence_and_cleared_early_diagnostics"
    ] is True
    assert contract["authority_boundary"][
        "candidate_evaluation_allowed_only_after_candidate_synthesis"
    ] is True
    assert contract["authority_boundary"]["language_expansion_executed"] is False
    assert contract["authority_boundary"]["current_pointer_written"] is False
    assert contract["nonclaims"] == list(interval_core.MANDATORY_NONCLAIMS)


def test_contract_cannot_enable_materialization_fallback_language_or_current():
    contract = _load(INTERVAL_COMPETITION_CONTRACT)
    mutations = (
        lambda value: value["selection"].__setitem__(
            "selection_status", "FORGED_SELECTED_STATUS"
        ),
        lambda value: value["selection"].__setitem__(
            "no_selection_status", "FORGED_NO_SELECTION_STATUS"
        ),
        lambda value: value["selection"].__setitem__(
            "candidate_materialized", True
        ),
        lambda value: value["stress_confirmation_policy"].__setitem__(
            "fallback_candidate_selected", True
        ),
        lambda value: value["authority_boundary"].__setitem__(
            "language_expansion_executed", True
        ),
        lambda value: value["authority_boundary"].__setitem__(
            "current_pointer_written", True
        ),
    )
    for mutate in mutations:
        changed = copy.deepcopy(contract)
        mutate(changed)
        with pytest.raises(ValueError):
            validate_shadow_interval_multi_q_theory_operation_competition_contract(
                changed
            )


@pytest.mark.parametrize("route", tuple(interval_core.DISPOSITION_REGISTRY))
def test_each_disposition_registry_route_is_contract_frozen(route):
    changed = _load(INTERVAL_COMPETITION_CONTRACT)
    changed["disposition_registry"][route] = "FORGED_DISPOSITION"
    with pytest.raises(ValueError):
        validate_shadow_interval_multi_q_theory_operation_competition_contract(
            changed
        )


@pytest.mark.parametrize("stage", ("reestimate", "noise", "scope", "mixture"))
def test_each_diagnostic_metric_formula_is_contract_frozen(stage):
    contract = _load(INTERVAL_COMPETITION_CONTRACT)
    policy = contract["diagnostic_metric_policy"]
    assert set(policy) == {"reestimate", "noise", "scope", "mixture"}
    assert policy["reestimate"] == {
        "residual_statistic": "raw_observed_minus_center",
        "center_shift": (
            "represented_float_of_pre_rounding_max_scaled_center_shift_times_raw_residual_scale"
        ),
        "fit_metric": "max_scaled_mean_absolute_raw_residual",
        "viability_requires_fractional_mae_gain": True,
        "viability_requires_raw_boundary_violation_rate_nonincrease": True,
        "mae_input_normalization": "divide_by_global_max_absolute_raw_residual",
        "mae_units": "dimensionless",
        "fractional_gain_denominator": (
            "source_max_scaled_mean_absolute_residual_without_numeric_epsilon"
        ),
        "shifted_mae_uses_effective_represented_center_shift": True,
        "shifted_mae_statistic": (
            "represented_shifted_raw_residual_divided_by_raw_residual_scale"
        ),
        "no_raw_boundary_violation_short_circuits_reestimate_arithmetic": True,
        "no_raw_boundary_violation_metric_values": "ALL_NULL",
    }
    assert policy["noise"] == {
        "signed_exceedance_statistic": (
            "sign_residual_times_max_zero_abs_residual_minus_radius"
        ),
        "grouping": "exact_scope_context_pair",
        "variance_statistic": (
            "total_and_within_pair_signed_exceedance_sse_fraction"
        ),
        "sse_input_normalization": (
            "divide_by_global_max_absolute_signed_exceedance"
        ),
        "sse_units": "dimensionless",
    }
    assert policy["scope"] == {
        "exceedance_statistic": "positive_raw_boundary_exceedance",
        "exceedance_units": "source_prediction_units",
        "normalization": (
            "divide_each_by_global_max_positive_raw_boundary_exceedance"
        ),
        "aggregation": "sum_max_scaled_exceedance_per_scope",
        "structure_ratio": (
            "max_minus_min_divided_by_max_without_numeric_epsilon"
        ),
        "also_uses_raw_boundary_violation_rate_spread": True,
        "combination": "maximum",
    }
    assert policy["mixture"] == {
        "rows": "raw_boundary_violations_only",
        "statistic": "signed_raw_boundary_exceedance",
        "objective": "deterministic_minimum_two_cluster_sse",
        "minimum_cluster_size": 2,
        "tie_break": "lowest_split_index",
        "score": "fractional_sse_reduction",
        "sse_input_normalization": (
            "divide_by_global_max_absolute_signed_exceedance"
        ),
        "sse_units": "dimensionless",
    }
    changed = copy.deepcopy(contract)
    first_key = next(iter(changed["diagnostic_metric_policy"][stage]))
    changed["diagnostic_metric_policy"][stage][first_key] = "FORGED_FORMULA"
    with pytest.raises(ValueError):
        validate_shadow_interval_multi_q_theory_operation_competition_contract(
            changed
        )


@pytest.mark.parametrize(
    "component",
    (
        None,
        "normalized_center_mae_gain",
        "raw_boundary_coverage_gain",
        "source_tail_coverage_gain",
        "context_reduction_fraction",
        "uniform_contraction_fraction",
        "normalized_radius_reduction",
        "max_probe_divergence",
        "normalized_radius_expansion",
    ),
)
def test_validation_score_and_each_component_unit_are_contract_frozen(component):
    changed = _load(INTERVAL_COMPETITION_CONTRACT)
    policy = changed["validation_selection_policy"]
    if component is None:
        policy["validation_score_units"] = "FORGED_UNITS"
    else:
        policy["score_component_units"][component] = "FORGED_UNITS"
    with pytest.raises(ValueError):
        validate_shadow_interval_multi_q_theory_operation_competition_contract(
            changed
        )


def test_surface_has_no_execution_network_or_prior_file_mutation():
    before = {
        path: _digest_file(path)
        for path in (
            *PREVIOUS_SLICE_FILES,
            adapter_kat.adjudication_kat.restriction_kat.OLD_BENCHMARK,
        )
    }
    for path in (INTERVAL_COMPETITION_CORE, INTERVAL_COMPETITION_RUNNER):
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


def test_documentation_freezes_additive_v2_boundary():
    text = INTERVAL_COMPETITION_DOC.read_text(encoding="utf-8")
    assert "strictly additive V2" in text
    assert adapter_kat.FROZEN_CONTRACT_DIGEST in text
    assert FROZEN_CONTRACT_DIGEST in text
    assert "exactly twenty-four" in text
    assert "276" in text
    assert "2/4/7/9/12/15/18/21/24" in text
    assert "Operations Research" in text
    assert "validation alone selects" in text
    assert "Stress is evaluated only" in text
    assert "no fallback" in text
    assert "L/2 + U/2" in text
    assert "does not materialize" in text
    assert "language" in text
    assert "NOT_CURRENT" in text
    assert "not a complete autonomous theory-evolution loop" in text
