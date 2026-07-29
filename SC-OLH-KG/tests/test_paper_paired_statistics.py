import copy

from performance.paper_paired_statistics import (
    _exact_two_sided_sign_p,
    analyze,
    holm_adjust,
)


def _row(
    method,
    seed,
    *,
    feasible,
    regret,
    certified,
    log_variance_rmse=None,
):
    return {
        "track_id": "track",
        "method_identity": method,
        "status": "ok",
        "domain": "Domain",
        "target_dimension": 1000,
        "seed": seed,
        "source_archive_fingerprint": "archive",
        "initial_design_fingerprint": f"design-{seed}",
        "verifier_signature": "verifier",
        "true_feasible": feasible,
        "terminal_certified": certified,
        "false_certificate": certified and not feasible,
        "feasible_regret": regret,
        "source_calls": 384,
        "target_search_calls": 13,
        "target_verification_calls": 100,
        "optimization_calls_excluding_verification": 397,
        "aleatoric_log_variance_rmse": log_variance_rmse,
        "aleatoric_variance_rmse": None,
        "aleatoric_upper_coverage": None,
        "aleatoric_variance_shape_correlation": None,
    }


def _registry():
    return {
        "registry_id": "unit",
        "primary_comparisons": [{
            "comparison_id": "left_vs_right",
            "left_track": "track",
            "left_method": "left",
            "right_track": "track",
            "right_method": "right",
            "domains": ["Domain"],
            "dimensions": [1000],
            "expected_pairs": 4,
            "required_equal_fields": [
                "source_archive_fingerprint",
                "initial_design_fingerprint",
                "verifier_signature",
            ],
        }],
    }


def test_exact_sign_and_holm_are_conservative_and_deterministic():
    assert _exact_two_sided_sign_p(4, 0) == 0.125
    assert _exact_two_sided_sign_p(0, 0) == 1.0
    assert holm_adjust([0.01, 0.04, 0.20]) == [0.03, 0.08, 0.2]


def test_paired_statistics_count_rescue_loss_and_regret_direction():
    records = [
        _row("left", 0, feasible=True, regret=0.1, certified=True),
        _row("right", 0, feasible=False, regret=None, certified=False),
        _row("left", 1, feasible=False, regret=None, certified=False),
        _row("right", 1, feasible=True, regret=0.4, certified=True),
        _row("left", 2, feasible=True, regret=0.1, certified=True),
        _row("right", 2, feasible=True, regret=0.3, certified=True),
        _row("left", 3, feasible=True, regret=0.2, certified=True),
        _row("right", 3, feasible=True, regret=0.2, certified=True),
    ]
    result = analyze(
        {"status": "pass", "records": records},
        _registry(),
        bootstrap_samples=200,
    )
    assert result["status"] == "complete"
    row = next(item for item in result["rows"] if item["stratum"] == "all")
    assert row["left_rescue_count"] == 1
    assert row["left_loss_count"] == 1
    assert row["both_feasible_regret_pair_count"] == 2
    assert row["left_regret_win_count"] == 1
    assert row["right_regret_win_count"] == 0
    assert row[
        "paired_regret_rank_biserial_left_better_positive"
    ] == 1.0
    assert row[
        "median_paired_regret_difference_left_minus_right"
    ] < 0.0

    repeated = analyze(
        {"status": "pass", "records": copy.deepcopy(records)},
        _registry(),
        bootstrap_samples=200,
    )
    assert result == repeated


def test_paired_statistics_refuses_mismatched_information_contract():
    records = [
        _row("left", seed, feasible=True, regret=0.1, certified=True)
        for seed in range(4)
    ] + [
        _row("right", seed, feasible=True, regret=0.2, certified=True)
        for seed in range(4)
    ]
    records[-1]["initial_design_fingerprint"] = "different"
    result = analyze(
        {"status": "pass", "records": records},
        _registry(),
        bootstrap_samples=20,
    )
    assert result["status"] == "incomplete"
    assert any(
        failure["kind"] == "paired_initial_design_fingerprint_mismatch"
        for failure in result["comparison_audits"][0]["failures"]
    )


def test_paired_statistics_reports_post_run_variance_calibration():
    records = []
    for seed in range(4):
        records.extend([
            _row(
                "left",
                seed,
                feasible=True,
                regret=0.1,
                certified=True,
                log_variance_rmse=0.2 + 0.01 * seed,
            ),
            _row(
                "right",
                seed,
                feasible=True,
                regret=0.1,
                certified=True,
                log_variance_rmse=0.5 + 0.01 * seed,
            ),
        ])
    result = analyze(
        {"status": "pass", "records": records},
        _registry(),
        bootstrap_samples=100,
    )
    row = next(item for item in result["rows"] if item["stratum"] == "all")
    assert row["aleatoric_log_variance_rmse_pair_count"] == 4
    assert row["left_aleatoric_log_variance_rmse_win_count"] == 4
    assert row["right_aleatoric_log_variance_rmse_win_count"] == 0
    assert row[
        "median_paired_aleatoric_log_variance_rmse_difference_left_minus_right"
    ] < 0.0
