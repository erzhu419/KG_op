from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_hvd_identifiability import run_cell  # noqa: E402
from problems.rzdt import FactorShockStatePolicyRZDT1  # noqa: E402


def _args(**overrides):
    values = {
        "mode": "factor_cumulative",
        "shock_scale": 1.0,
        "replicates": 3,
        "seed": 3,
        "d": 12,
        "L": 100,
        "sigma": 0.04,
        "alpha": 0.05,
        "tau": 0.25,
        "n_train": 18,
        "activation_min_records": 8,
        "delta": 0.05,
        "ridge": 1e-3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_shared_shock_scale_changes_only_shared_variance_block():
    x = (25,) + (72,) * 11
    weak = FactorShockStatePolicyRZDT1(d=12, shared_shock_scale=0.0)
    strong = FactorShockStatePolicyRZDT1(d=12, shared_shock_scale=2.0)
    weak_blocks = weak.true_cumulative_risk_decomposition(x, output_index=1)
    strong_blocks = strong.true_cumulative_risk_decomposition(x, output_index=1)

    assert weak_blocks["shared"] == 0.0
    assert strong_blocks["shared"] > 0.0
    np.testing.assert_allclose(
        weak_blocks["independent"], strong_blocks["independent"])
    np.testing.assert_allclose(weak_blocks["linear"], strong_blocks["linear"])
    np.testing.assert_allclose(weak_blocks["floor"], strong_blocks["floor"])


def test_identifiability_cell_uses_only_replicated_fit_observations():
    result = run_cell(_args())

    assert result["status"] == "ok"
    assert result["simulator_calls"] == 18 * 3
    assert result["evaluation_count"] > 100
    assert np.isfinite(result["log_variance_rmse"])
    assert 0.0 <= result["variance_upper_coverage"] <= 1.0
    assert 0.0 < result["true_feasible_rate"] < 1.0
    assert result["certification_tau"] == 0.25
    assert result["hvd_diagnostics"]["residual_square_tail"]["1"][
        "effective_dof"] == 18 * (3 - 1)
    assert result["information_contract"]["oracle_used_for_fit"] is False
    assert result["information_contract"][
        "true_constraint_mean_used_for_fit"] is False
    assert result["information_contract"]["fit_inputs"] == (
        "ordinary_replicate_sample_mean_and_sample_variance")
    assert result["information_contract"]["oracle_used_for_post_run_audit"] is True
    assert result["hvd_diagnostics"]["cumulative_active"]["1"] is True
