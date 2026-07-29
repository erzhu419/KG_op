import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_source_hvd_saas_gate import (  # noqa: E402
    analyze,
    analyze_record_shards,
)


def _write_row(root, mode, seed, *, rmse, coverage, correlation):
    result_dir = root / mode / "QueueResourceControl" / f"seed{seed:04d}"
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ok",
        "heldout": "QueueResourceControl",
        "seed": seed,
        "initial_points_fingerprint": f"design-{seed}",
        "source_archive_fingerprint": "archive",
        "information_contract": {
            "aleatoric_head_mode": mode,
            "aleatoric_head_contract": f"{mode}-contract",
        },
        "result": {
            "n_simulations": 13,
            "n_search_simulations": 13,
            "n_verification_simulations": 100,
            "true_feasible": True,
            "feasible_regret": 0.01,
            "terminal_verification": {
                "certified": True,
            },
            "post_run_aleatoric_audit": {
                "log_variance_rmse": rmse,
                "upper_coverage": coverage,
                "variance_shape_correlation": correlation,
            },
        },
    }
    (result_dir / "result.json").write_text(
        json.dumps(payload), encoding="utf-8")


def test_gate_requires_paired_calibration_and_decision_noninferiority(tmp_path):
    for seed in range(3):
        _write_row(
            tmp_path, "pooled", seed,
            rmse=0.8, coverage=0.5, correlation=0.0)
        _write_row(
            tmp_path, "cumulative_factor", seed,
            rmse=0.4, coverage=0.95, correlation=0.7)
    report = analyze(tmp_path)
    assert report["row_count"] == 6
    assert report["gate"]["all_rows_paired"] is True
    assert report["gate"]["calibration_improved"] is True
    assert report["gate"]["shape_correlation_improved"] is True
    assert report["gate"]["promote_hvd_as_core"] is True


def test_gate_rejects_cross_domain_median_masking(tmp_path):
    for domain, pooled_rmse, factor_rmse in (
        ("FactorShockStatePolicyRZDT1", 1.2, 0.3),
        ("InventorySupplyChain", 0.5, 0.9),
    ):
        for mode, rmse, correlation in (
            ("pooled", pooled_rmse, 0.1),
            ("cumulative_factor", factor_rmse, 0.8),
        ):
            result_dir = tmp_path / mode / domain / "seed0080"
            result_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "status": "ok",
                "heldout": domain,
                "seed": 80,
                "initial_points_fingerprint": "design",
                "source_archive_fingerprint": "archive",
                "information_contract": {
                    "aleatoric_head_mode": mode,
                    "aleatoric_head_contract": f"{mode}-contract",
                },
                "result": {
                    "n_simulations": 13,
                    "n_search_simulations": 13,
                    "n_verification_simulations": 100,
                    "true_feasible": True,
                    "feasible_regret": 0.01,
                    "terminal_verification": {"certified": True},
                    "post_run_aleatoric_audit": {
                        "log_variance_rmse": rmse,
                        "upper_coverage": 0.95,
                        "variance_shape_correlation": correlation,
                    },
                },
            }
            (result_dir / "result.json").write_text(
                json.dumps(payload), encoding="utf-8")

    report = analyze(
        tmp_path,
        expected_domains=(
            "FactorShockStatePolicyRZDT1",
            "InventorySupplyChain",
        ),
        expected_seeds=(80,),
    )

    assert report["domain_gates"][
        "FactorShockStatePolicyRZDT1"]["promote_in_domain"] is True
    assert report["domain_gates"][
        "InventorySupplyChain"]["calibration_improved"] is False
    assert report["gate"]["promote_hvd_as_core"] is False
    assert report["gate"][
        "retain_as_domain_conditional_calibration_component"] is True


def test_analyzer_accepts_hash_bound_compact_record_shards(tmp_path):
    records = []
    for mode, rmse, corr in (
        ("pooled", 0.8, 0.1),
        ("cumulative_factor", 0.4, 0.6),
    ):
        records.append({
            "track_id": "source_hvd_causal_gate_d1000_n13",
            "status": "ok",
            "method_identity": (
                f"canonical_saasbo_every_iteration+hvd:{mode}"),
            "domain": "FactorShockStatePolicyRZDT1",
            "seed": 80,
            "path": f"/remote/{mode}.json",
            "initial_design_fingerprint": "design",
            "source_archive_fingerprint": "archive",
            "target_search_calls": 13,
            "target_verification_calls": 336,
            "true_feasible": True,
            "feasible_regret": 0.01,
            "terminal_certified": True,
            "false_certificate": False,
            "aleatoric_log_variance_rmse": rmse,
            "aleatoric_upper_coverage": 0.95,
            "aleatoric_variance_shape_correlation": corr,
        })
    shard = tmp_path / "shard.json"
    shard.write_text(json.dumps({
        "origin": "unit",
        "records_sha256": "receipt",
        "records": records,
    }), encoding="utf-8")
    report = analyze_record_shards(
        [shard],
        expected_domains=["FactorShockStatePolicyRZDT1"],
        expected_seeds=[80],
    )
    assert report["gate"]["promote_hvd_as_core"]
    assert report["record_shard_receipts"][0][
        "selected_row_count"] == 2
