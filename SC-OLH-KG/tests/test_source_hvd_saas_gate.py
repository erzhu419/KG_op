import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_source_hvd_saas_gate import analyze  # noqa: E402


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

