from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_hvd_replication_identifiability_gate import (  # noqa: E402
    MODES,
    REPLICATIONS,
    SHOCK_SCALES,
    summarize,
)


def _row(mode, shock, replication, seed):
    factor = mode == "factor_cumulative"
    return {
        "experiment": "hvd_identifiability",
        "mode": mode,
        "shared_shock_scale": shock,
        "replicates_per_policy": replication,
        "seed": seed,
        "log_variance_rmse": (
            0.20 - 0.01 * replication if factor else 0.40),
        "variance_spearman": 0.8 if factor else 0.2,
        "shared_risk_spearman": 0.7 if factor else None,
        "median_predicted_variance": (
            0.05 + 0.10 * shock if factor else 0.08),
        "median_true_variance": 0.05 + 0.10 * shock,
        "median_fitted_shared_risk": 0.10 * shock if factor else None,
        "false_feasible_count": 0 if factor else 1,
        "certificate_nonvacuous": True,
        "certificate_precision": 1.0,
        "certificate_recall": 0.8,
        "information_contract": {
            "oracle_used_for_fit": False,
            "oracle_used_for_post_run_audit": True,
            "true_constraint_mean_used_for_fit": False,
            "fit_inputs": (
                "ordinary_replicate_sample_mean_and_sample_variance"),
        },
    }


def test_replication_identifiability_gate_requires_oracle_free_scale_recovery():
    rows = [
        _row(mode, shock, replication, seed)
        for mode in MODES
        for shock in SHOCK_SCALES
        for replication in REPLICATIONS
        for seed in range(2)
    ]
    result = summarize(rows, expected_seeds=2)
    assert result["row_count"] == 2 * 2 * 3 * 2
    assert result["gate_pass"] is True
    assert all(result["criteria"].values())


def test_replication_identifiability_gate_rejects_oracle_fit_contract():
    rows = [
        _row(mode, shock, replication, 0)
        for mode in MODES
        for shock in SHOCK_SCALES
        for replication in REPLICATIONS
    ]
    rows[0]["information_contract"]["true_constraint_mean_used_for_fit"] = True
    result = summarize(rows, expected_seeds=1)
    assert result["criteria"]["ordinary_replicated_fit_only"] is False
    assert result["gate_pass"] is False


def test_replication_identifiability_gate_handles_constant_pooled_rank():
    rows = [
        _row(mode, shock, replication, seed)
        for mode in MODES
        for shock in SHOCK_SCALES
        for replication in REPLICATIONS
        for seed in range(2)
    ]
    for row in rows:
        if row["mode"] == "pooled":
            row["variance_spearman"] = None
    result = summarize(rows, expected_seeds=2)
    assert result["diagnostics"]["paired_variance_rank_count"] == 0
    assert result["criteria"][
        "factor_rank_strictly_better_than_pooled"] is True
    assert result["gate_pass"] is True
