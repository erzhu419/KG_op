#!/usr/bin/env python3
"""Freeze the strict no-history SUMO external-validity disposition.

This audit deliberately distinguishes development evidence from a posthoc
certifiability diagnostic.  The latter may explain a failure, but it cannot
select a method or support a confirmatory performance claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_MODES = (
    "descriptor_nearest",
    "domain_blind_exclude_nearest",
)
EXPECTED_BUDGETS = (13, 40, 80)
EXPECTED_SEEDS = (80, 81, 82, 83, 84)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_frontier_cell(mode, budget, payload):
    _require(mode in EXPECTED_MODES, f"unknown source-selection mode: {mode}")
    _require(budget in EXPECTED_BUDGETS, f"unknown search budget: {budget}")
    _require(payload.get("status") == "complete", "frontier cell is incomplete")
    _require(
        tuple(map(int, payload.get("seeds", ()))) == EXPECTED_SEEDS,
        "frontier cell uses the wrong search seeds",
    )
    _require(int(payload.get("n_seeds", 0)) == 5, "frontier cell is not five-seed")
    _require(
        int(payload.get("source_calls_per_run", -1)) == 384,
        "frontier source budget drifted",
    )
    _require(
        int(payload.get("target_initial_design_calls_per_run", -1)) == 10,
        "frontier n0 drifted",
    )
    _require(
        int(payload.get("target_search_calls_per_run", -1)) == budget,
        "frontier search budget metadata drifted",
    )
    _require(
        payload.get("policy_vectors_exported") is False,
        "frontier compact audit contains policy vectors",
    )
    information = payload.get("information_contract", {})
    _require(
        information.get("source_selection_mode") == mode,
        "frontier source-selection metadata drifted",
    )
    _require(
        information.get("target_labels_used_to_fit_proposal") is False,
        "frontier proposal used target outcomes",
    )
    _require(
        information.get("target_oracle_used") is False,
        "frontier used the target oracle",
    )
    _require(
        information.get("historical_target_anchor_used") is False,
        "frontier used a historical target anchor",
    )
    _require(
        information.get("evidence_phase") == "development_gate",
        "frontier evidence phase drifted",
    )
    return {
        "source_selection_mode": mode,
        "target_search_calls": budget,
        "source_calls": 384,
        "n0": 10,
        "adaptive_search_calls": budget - 10,
        "search_seed_count": 5,
        "certified_seed_count": int(payload.get("certified_seed_count", 0)),
        "median_deployed_feasible_probability": float(
            payload.get("median_deployed_feasible_probability", 0.0)
        ),
        "median_deployed_familywise_exact_lower": float(
            payload.get("median_deployed_familywise_exact_lower", 0.0)
        ),
        "verification_calls_per_run": int(
            payload.get("target_verification_calls_per_run", 0)
        ),
        "target_total_calls_per_run": int(
            payload.get("target_total_calls_per_run", 0)
        ),
        "source_plus_target_total_calls_per_run": int(
            payload.get("source_plus_target_total_calls_per_run", 0)
        ),
    }


def _validate_v2_gate(manifest, gate):
    _require(gate.get("status") == "complete", "V2 promotion gate is incomplete")
    _require(
        gate.get("gate_id") == manifest.get("gate_id"),
        "V2 gate identity differs from its preregistration",
    )
    traffic = gate.get("traffic_development_gate", {})
    _require(not traffic.get("contract_failures"), "V2 traffic contract drifted")
    _require(
        int(traffic.get("seed_count", 0)) == 5,
        "V2 traffic gate is not five-seed",
    )
    _require(gate.get("saas_used") is False, "V2 gate used SAAS")
    _require(gate.get("gpu_used") is False, "V2 gate used a GPU")
    _require(
        gate.get("promote_lower_envelope_v2") is False,
        "disposition expects the preregistered V2 failure",
    )
    _require(
        gate.get("decision") == "retain_v1_and_stop_target_domain_tuning",
        "V2 failure action drifted",
    )
    synthetic = gate.get("synthetic_noninferiority_gate", {})
    _require(
        synthetic.get("status")
        == "not_run_due_to_sequential_traffic_gate_failure",
        "V2 sequential stopping rule was not respected",
    )
    contract = manifest["traffic_development_gate"]
    return {
        "gate_id": gate["gate_id"],
        "run_id": contract["run_id"],
        "execution_commit": contract["execution_commit"],
        "source_calls": int(contract["source_calls"]),
        "n0": int(contract["n0"]),
        "target_search_calls": int(contract["target_search_calls"]),
        "search_seed_count": int(traffic["seed_count"]),
        "certified_seed_count": int(traffic["certified_seed_count"]),
        "empirical_false_certificate_count": int(
            traffic["empirical_false_certificate_count"]
        ),
        "minimum_certified_seed_count": int(
            traffic["minimum_certified_seed_count"]
        ),
        "status": traffic["status"],
        "synthetic_gate_status": synthetic["status"],
        "decision": gate["decision"],
        "saas_used": False,
        "gpu_used": False,
    }


def _validate_posthoc(payload, *, library_fingerprint):
    _require(payload.get("status") == "complete", "posthoc audit is incomplete")
    _require(payload.get("diagnostic_only") is True, "posthoc role is not diagnostic")
    _require(
        payload.get("admissible_for_method_selection") is False,
        "posthoc audit was made admissible for method selection",
    )
    _require(
        payload.get("admissible_for_confirmatory_claim") is False,
        "posthoc audit was made admissible for a confirmatory claim",
    )
    _require(payload.get("target_oracle_used") is False, "posthoc used target oracle")
    _require(
        payload.get("historical_target_anchor_used") is False,
        "posthoc used a historical target anchor",
    )
    _require(
        payload.get("policy_vectors_exported") is False,
        "posthoc compact audit contains policy vectors",
    )
    _require(int(payload.get("library_size", 0)) == 111, "library size drifted")
    _require(
        int(payload.get("fresh_seed_replications_per_candidate", 0)) == 200,
        "posthoc replication budget drifted",
    )
    _require(
        int(payload.get("target_verification_calls", 0)) == 22200,
        "posthoc total verification budget drifted",
    )
    familywise = int(payload.get("familywise_certified_candidate_count", 0))
    support_status = (
        "frozen_library_contains_familywise_certifiable_support"
        if familywise > 0
        else "no_familywise_certifiable_support_in_frozen_library"
    )
    provenance = payload.get("execution_provenance", {})
    _require(provenance.get("status") == "frozen", "posthoc snapshot is not frozen")
    return {
        "run_id": "traffic_universal_library_posthoc_68be50c_R200_s1200000_v1",
        "execution_commit": provenance.get("repository_commit"),
        "method_contract_id": provenance.get("method_contract_id"),
        "theory_contract_id": provenance.get("theory_contract_id"),
        "library_definition": "preexisting_target_label_free_universal_shape_library",
        "library_fingerprint": str(library_fingerprint),
        "library_size": 111,
        "fresh_seed_replications_per_candidate": 200,
        "target_verification_calls": 22200,
        "point_feasible_candidate_count": int(
            payload.get("point_feasible_candidate_count", 0)
        ),
        "familywise_certified_candidate_count": familywise,
        "maximum_empirical_feasible_probability": float(
            payload.get("maximum_empirical_feasible_probability", 0.0)
        ),
        "median_empirical_feasible_probability": float(
            payload.get("median_empirical_feasible_probability", 0.0)
        ),
        "best_source_indices": list(map(int, payload.get("best_source_indices", ()))),
        "support_status": support_status,
        "diagnostic_only": True,
        "admissible_for_method_selection": False,
        "admissible_for_confirmatory_claim": False,
        "target_oracle_used": False,
        "historical_target_anchor_used": False,
        "policy_vectors_exported": False,
        "saas_used": False,
        "gpu_used": False,
    }


def build_disposition(
    frontier_cells,
    *,
    v2_manifest,
    v2_gate,
    posthoc,
    library_fingerprint,
):
    keys = {(mode, int(budget)) for mode, budget, _ in frontier_cells}
    expected = {
        (mode, budget) for mode in EXPECTED_MODES for budget in EXPECTED_BUDGETS
    }
    _require(keys == expected, "V1 frontier does not contain exactly six cells")
    _require(len(frontier_cells) == len(keys), "V1 frontier contains duplicate cells")
    frontier = [
        _validate_frontier_cell(mode, int(budget), payload)
        for mode, budget, payload in frontier_cells
    ]
    frontier.sort(key=lambda row: (row["source_selection_mode"], row["target_search_calls"]))
    v2 = _validate_v2_gate(v2_manifest, v2_gate)
    posthoc_row = _validate_posthoc(
        posthoc,
        library_fingerprint=library_fingerprint,
    )
    frontier_certified = sum(row["certified_seed_count"] for row in frontier)
    return {
        "schema_version": 1,
        "status": "complete",
        "external_validity_status": "failed_not_promoted",
        "submission_release_status": "blocked_by_external_validity",
        "selected_frontend": "lodo_low_frequency_risk_objective_atlas_v1",
        "selected_frontend_changed_by_posthoc": False,
        "confirmatory_external_traffic_evidence_available": False,
        "v1_cpu_budget_frontier": {
            "run_id": (
                "external_traffic_cpu_frontier_7f796cd_scbo_"
                "s80_84_N13_40_80_R100_v1"
            ),
            "execution_commit": "7f796cd6989c76051f3a30c135f848de04a0cc84",
            "cells": frontier,
            "total_certified_seed_cells": int(frontier_certified),
            "all_registered_cells_failed_to_certify": frontier_certified == 0,
            "saas_used": False,
            "gpu_used": False,
        },
        "v2_preregistered_development_gate": v2,
        "posthoc_universal_library_certifiability": posthoc_row,
        "decision_contract": {
            "v2_not_promoted": True,
            "no_confirmatory_expansion_after_failed_development_gate": True,
            "posthoc_outcomes_do_not_select_or_modify_the_method": True,
            "no_additional_target_domain_sentinel_tuning": True,
            "external_validity_claim_allowed": False,
            "paper_release_must_fail_closed": True,
        },
        "information_contract": {
            "target_labels_used_to_fit_proposal": False,
            "target_oracle_used": False,
            "historical_target_anchor_used": False,
            "raw_policy_vectors_in_release_artifact": False,
            "development_and_posthoc_evidence_kept_distinct": True,
        },
    }


def _parse_frontier_cell(value):
    try:
        mode, budget, path = str(value).split(":", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "frontier cell must be MODE:BUDGET:PATH"
        ) from exc
    return mode, int(budget), Path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frontier-cell",
        action="append",
        type=_parse_frontier_cell,
        required=True,
    )
    parser.add_argument("--v2-manifest", required=True)
    parser.add_argument("--v2-gate", required=True)
    parser.add_argument("--posthoc", required=True)
    parser.add_argument("--library-fingerprint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = build_disposition(
        [
            (mode, budget, _read_json(path))
            for mode, budget, path in args.frontier_cell
        ],
        v2_manifest=_read_json(args.v2_manifest),
        v2_gate=_read_json(args.v2_gate),
        posthoc=_read_json(args.posthoc),
        library_fingerprint=args.library_fingerprint,
    )
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "external_validity_status": payload["external_validity_status"],
        "submission_release_status": payload["submission_release_status"],
        "posthoc_support_status": payload[
            "posthoc_universal_library_certifiability"
        ]["support_status"],
        "out": str(args.out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
