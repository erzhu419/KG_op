import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_native_transfer_matrix import analyze  # noqa: E402


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _row(domain, method, seed, *, certified=True, feasible=True, fingerprint=None):
    fingerprint = fingerprint or f"archive-{domain}"
    regret = 0.005 if feasible else None
    return {
        "schema_version": 1,
        "status": "ok",
        "implementation": "official",
        "heldout_target_domain": domain,
        "method": method,
        "seed": seed,
        "source_domains": ["source-a", "source-b"],
        "comparison_contract": {
            "target_initial_design": "native_source_sequential",
            "source_simulator_calls": 384,
            "target_dimension": 1000,
            "target_initial_calls_n0": 10,
            "target_search_calls": 13,
            "source_oracle_aided": False,
            "source_scored_atlas_used": False,
            "source_informed_initial_proposal": False,
            "source_archive_identical_across_methods": True,
            "terminal_verification_identical_across_methods": True,
            "source_archive_fingerprint": fingerprint,
            "total_source_plus_target_verification_calls": 477,
        },
        "result": {
            "true_feasible": feasible,
            "feasible_regret": regret,
            "n_search_simulations": 13,
            "n_verification_simulations": 80,
            "n_target_simulations_total": 93,
            "terminal_shortlist_frozen_before_truth_metrics": True,
            "initial_truth_audit": {"true_feasible_count": int(feasible)},
            "terminal_verification": {
                "certified": certified,
                "protocol": "ordered_frozen_shortlist",
                "shortlist_frozen_before_verification": True,
                "target_oracle_used": False,
                "posterior_updated_from_verification": False,
            },
        },
        "wall_time_sec": 2.0,
    }


def test_native_transfer_analysis_accepts_complete_native_matrix(tmp_path):
    domains = ("d1", "d2")
    methods = ("m1", "m2")
    seeds = (80, 81)
    paths = []
    for domain in domains:
        for method in methods:
            for seed in seeds:
                path = tmp_path / domain / method / str(seed) / "result.json"
                _write(path, _row(domain, method, seed))
                paths.append(path)

    result = analyze(
        paths,
        expected_domains=domains,
        expected_methods=methods,
        expected_seeds=seeds,
    )

    assert result["status"] == "complete"
    assert result["row_count"] == 8
    assert result["common_atlas_backend_comparison"] is False
    assert result["task_population_inference_claimed"] is False
    assert all(
        summary["certified_safe_count"] == 2
        for summary in result["domain_method_summaries"]
    )


def test_native_transfer_analysis_fails_on_cross_method_archive_mismatch(tmp_path):
    first = tmp_path / "d1" / "m1" / "80" / "result.json"
    second = tmp_path / "d1" / "m2" / "80" / "result.json"
    _write(first, _row("d1", "m1", 80, fingerprint="archive-a"))
    _write(second, _row("d1", "m2", 80, fingerprint="archive-b"))

    result = analyze(
        [first, second],
        expected_domains=("d1",),
        expected_methods=("m1", "m2"),
        expected_seeds=(80,),
    )

    assert result["status"] == "incomplete"
    assert any(
        failure["kind"] == "source_archive_fingerprint_mismatch"
        for failure in result["failures"]
    )
