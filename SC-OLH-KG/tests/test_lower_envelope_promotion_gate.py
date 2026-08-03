from performance.analyze_lower_envelope_promotion_gate import evaluate_gate


def _manifest():
    return {
        "gate_id": "gate",
        "traffic_development_gate": {
            "search_seeds": [80, 81],
            "evidence_phase": "development_gate",
            "source_selection_mode": "descriptor_nearest",
            "source_calls": 384,
            "n0": 10,
            "target_search_calls": 13,
            "minimum_certified_seed_count": 2,
            "maximum_empirical_false_certificate_count": 0,
        },
        "synthetic_noninferiority_gate": {
            "domains": ["D"],
            "backends": ["proposal_only"],
            "seeds": [80, 81],
            "require_no_feasibility_count_loss": True,
            "maximum_median_paired_regret_increase": 0.005,
            "maximum_false_certificate_count_increase": 0,
        },
        "promotion_rule": {
            "success_action": "promote",
            "failure_action": "stop",
        },
    }


def _traffic(certified=2):
    rows = [{
        "deployed_certified": index < certified,
        "deployed_feasible_probability": 1.0,
    } for index in range(2)]
    return {
        "status": "complete",
        "seeds": [80, 81],
        "certified_seed_count": certified,
        "source_calls_per_run": 384,
        "target_initial_design_calls_per_run": 10,
        "target_search_calls_per_run": 13,
        "policy_vectors_exported": False,
        "information_contract": {
            "evidence_phase": "development_gate",
            "source_selection_mode": "descriptor_nearest",
            "target_oracle_used": False,
            "historical_target_anchor_used": False,
        },
        "rows": rows,
    }


def _records(challenger_regret=0.012):
    rows = []
    for frontend, regret in (
        ("v1", 0.01),
        ("lower_envelope_v2", challenger_regret),
    ):
        for seed in (80, 81):
            rows.append({
                "frontend": frontend,
                "backend": "proposal_only",
                "domain": "D",
                "seed": seed,
                "status": "ok",
                "true_feasible": True,
                "false_certificate": False,
                "feasible_regret": regret,
                "source_archive_fingerprint": "archive",
                "problem_contract_fingerprint": "problem",
                "verifier_signature": "verifier",
                "source_calls": 384,
                "target_initial_calls": 10,
                "target_search_calls": 10,
            })
    return rows


def test_gate_promotes_only_when_traffic_and_synthetic_pass():
    result = evaluate_gate(_manifest(), _traffic(), _records())
    assert result["promote_lower_envelope_v2"] is True
    assert result["decision"] == "promote"


def test_gate_rejects_regret_harm_even_when_traffic_passes():
    result = evaluate_gate(
        _manifest(), _traffic(), _records(challenger_regret=0.02))
    assert result["promote_lower_envelope_v2"] is False
    assert result["decision"] == "stop"


def test_gate_rejects_incomplete_traffic_certification():
    result = evaluate_gate(_manifest(), _traffic(certified=1), _records())
    assert result["promote_lower_envelope_v2"] is False
    assert result["traffic_development_gate"]["outcome_pass"] is False
