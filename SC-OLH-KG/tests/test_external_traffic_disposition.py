import copy

import pytest

from performance.finalize_external_traffic_disposition import build_disposition


def _frontier_payload(mode, budget):
    return {
        "status": "complete",
        "seeds": [80, 81, 82, 83, 84],
        "n_seeds": 5,
        "certified_seed_count": 0,
        "source_calls_per_run": 384,
        "target_initial_design_calls_per_run": 10,
        "target_search_calls_per_run": budget,
        "target_verification_calls_per_run": 300,
        "target_total_calls_per_run": budget + 300,
        "source_plus_target_total_calls_per_run": 384 + budget + 300,
        "median_deployed_feasible_probability": 0.4,
        "median_deployed_familywise_exact_lower": 0.3,
        "policy_vectors_exported": False,
        "information_contract": {
            "source_selection_mode": mode,
            "target_labels_used_to_fit_proposal": False,
            "target_oracle_used": False,
            "historical_target_anchor_used": False,
            "evidence_phase": "development_gate",
        },
    }


def _inputs(familywise=0):
    frontier = [
        (mode, budget, _frontier_payload(mode, budget))
        for mode in ("descriptor_nearest", "domain_blind_exclude_nearest")
        for budget in (13, 40, 80)
    ]
    manifest = {
        "gate_id": "universal_lower_envelope_sentinel_promotion_v1",
        "traffic_development_gate": {
            "run_id": "traffic-v2",
            "execution_commit": "abc",
            "source_calls": 384,
            "n0": 10,
            "target_search_calls": 13,
        },
    }
    gate = {
        "status": "complete",
        "gate_id": manifest["gate_id"],
        "saas_used": False,
        "gpu_used": False,
        "promote_lower_envelope_v2": False,
        "decision": "retain_v1_and_stop_target_domain_tuning",
        "traffic_development_gate": {
            "status": "fail",
            "contract_failures": [],
            "seed_count": 5,
            "certified_seed_count": 0,
            "empirical_false_certificate_count": 0,
            "minimum_certified_seed_count": 4,
        },
        "synthetic_noninferiority_gate": {
            "status": "not_run_due_to_sequential_traffic_gate_failure",
        },
    }
    posthoc = {
        "status": "complete",
        "diagnostic_only": True,
        "admissible_for_method_selection": False,
        "admissible_for_confirmatory_claim": False,
        "target_oracle_used": False,
        "historical_target_anchor_used": False,
        "policy_vectors_exported": False,
        "library_size": 111,
        "fresh_seed_replications_per_candidate": 200,
        "target_verification_calls": 22200,
        "point_feasible_candidate_count": familywise,
        "familywise_certified_candidate_count": familywise,
        "maximum_empirical_feasible_probability": 0.99 if familywise else 0.9,
        "median_empirical_feasible_probability": 0.2,
        "best_source_indices": [7],
        "execution_provenance": {
            "status": "frozen",
            "repository_commit": "def",
            "method_contract_id": "posthoc-v1",
            "theory_contract_id": "coverage-v2",
        },
    }
    return frontier, manifest, gate, posthoc


def test_disposition_fails_closed_even_when_posthoc_finds_support():
    frontier, manifest, gate, posthoc = _inputs(familywise=2)
    result = build_disposition(
        frontier,
        v2_manifest=manifest,
        v2_gate=gate,
        posthoc=posthoc,
        library_fingerprint="library-hash",
    )
    assert result["external_validity_status"] == "failed_not_promoted"
    assert result["submission_release_status"] == "blocked_by_external_validity"
    assert result["selected_frontend_changed_by_posthoc"] is False
    diagnostic = result["posthoc_universal_library_certifiability"]
    assert diagnostic["support_status"] == (
        "frozen_library_contains_familywise_certifiable_support"
    )
    assert diagnostic["admissible_for_method_selection"] is False


def test_disposition_reports_absent_certifiable_library_support():
    frontier, manifest, gate, posthoc = _inputs(familywise=0)
    result = build_disposition(
        frontier,
        v2_manifest=manifest,
        v2_gate=gate,
        posthoc=posthoc,
        library_fingerprint="library-hash",
    )
    assert result["posthoc_universal_library_certifiability"][
        "support_status"
    ] == "no_familywise_certifiable_support_in_frozen_library"


def test_disposition_rejects_frontier_budget_metadata_drift():
    frontier, manifest, gate, posthoc = _inputs()
    broken = copy.deepcopy(frontier)
    broken[1][2]["target_search_calls_per_run"] = 13
    with pytest.raises(ValueError, match="search budget metadata drifted"):
        build_disposition(
            broken,
            v2_manifest=manifest,
            v2_gate=gate,
            posthoc=posthoc,
            library_fingerprint="library-hash",
        )
