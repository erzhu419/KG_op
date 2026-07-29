from performance.run_uniform_verification_shard import verify_row


def test_uniform_verifier_is_reproducible_and_budget_explicit():
    row = {
        "source_track_id": "track",
        "source_method_identity": "method",
        "uniform_method_identity": "uniform_verified::method",
        "domain": "FactorShockStatePolicyRZDT1",
        "target_dimension": 5,
        "seed": 80,
        "source_calls": 384,
        "target_search_calls": 13,
        "optimization_calls_excluding_verification": 397,
        "source_archive_fingerprint": "archive",
        "initial_design_fingerprint": "design",
        "source_result_sha256": "digest",
        "shortlist": [
            {
                "shortlist_position": 1,
                "shortlist_role": "primary",
                "point": [25, 75, 75, 75, 75],
                "target_oracle_used": False,
                "verification_samples_used": False,
            },
            {
                "shortlist_position": 2,
                "shortlist_role": "safe",
                "point": [45, 45, 45, 45, 45],
                "target_oracle_used": False,
                "verification_samples_used": False,
            },
        ],
    }
    contract = {
        "contract_id": "unit",
        "candidate_budgets": [8, 8],
        "familywise_delta": 0.05,
        "method": "normal_quantile_tolerance",
        "shortlist_mode": "uniform_first_certified_then_primary",
    }
    first = verify_row(row, contract)
    second = verify_row(row, contract)
    assert first == second
    assert first["optimization_calls_excluding_verification"] == 397
    assert first["terminal_verification"][
        "posterior_updated_from_verification"
    ] is False
    assert first["terminal_verification"]["search_samples_reused"] is False
    assert first["terminal_verification_truth_audit"][
        "used_for_selection"
    ] is False
