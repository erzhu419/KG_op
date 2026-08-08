import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_profile_all_in_frontier import analyze  # noqa: E402


def _row(arm, configuration, *, source_calls, N, success):
    return {
        "contract_id": "randomized_ordered_profile_stress_v2",
        "status": "ok",
        "regime": "aligned_low_frequency",
        "target_seed": 7,
        "arm": arm,
        "nominal_dimension": 1000,
        "matrix_configuration_id": configuration,
        "contains_true_feasible": success,
        "independently_certified": success,
        "false_certificate": False,
        "feasible_and_epsilon_optimal_005": success,
        "penalized_loss": 0.01 if success else 1.0,
        "source_calls": source_calls,
        "target_search_calls": N,
        "verification_calls": 80,
        "all_in_calls_unamortized": source_calls + N + 80,
        "all_in_budget_cap_unamortized": source_calls + N + 240,
    }


def test_all_in_frontier_checks_maximum_budget_and_break_even(tmp_path):
    rows = (
        _row("source_atlas", "source384-target10",
             source_calls=384, N=10, success=True),
        _row("raw_sobol", "target394",
             source_calls=0, N=394, success=False),
    )
    paths = []
    for index, row in enumerate(rows):
        path = tmp_path / f"row{index}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        paths.append(path)
    payload = analyze(paths)
    assert payload["status"] == "complete"
    summary = payload["summaries"][0]
    assert summary["all_pairs_have_equal_maximum_all_in_budget"] is True
    assert summary["source_certified_success_count"] == 1
    assert summary["control_certified_success_count"] == 0
    assert summary["median_archive_break_even_target_count"] == 1.0
    assert summary["source_wins"] == 1
    assert summary["holm_family_size"] == 1
    assert summary[
        "mean_source_minus_control_actual_all_in_calls_bootstrap_95ci"
    ] == [0.0, 0.0]
