from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_observable_state_coordinate_offline_gate import (  # noqa: E402
    SCENARIOS,
    VARIANTS,
    load_rows,
    summarize,
)


def _write_matrix(root, *, weak_state=False, empty_certificate=False):
    index = 0
    for variant in VARIANTS:
        provider = variant.startswith("provider_exposure")
        state = variant.startswith("observable_state_phi_")
        for domain, shock in SCENARIOS:
            for seed in range(5):
                if state and not weak_state:
                    rank, mae = 0.72, 0.10
                elif state:
                    rank, mae = -0.20, 1.20
                elif provider:
                    rank, mae = 0.85, 0.08
                else:
                    rank, mae = 0.50, 0.20
                row = {
                    "experiment_variant": (
                        f"observable_state_coordinate_offline/{variant}/"
                        f"shock{shock:g}"
                    ),
                    "heldout": domain,
                    "target_shared_shock_scale": shock,
                    "seed": seed,
                    "audit": {
                        "tcb_target_structural_provider_used": provider,
                        "observable_state_exposure_used": state,
                    },
                    "boundary_raw_pool_truth_diagnostics": {
                        "boundary_raw_pool_has_true_feasible": True,
                        "boundary_raw_pool_true_feasible_rate": 0.10,
                        "boundary_raw_pool_constraint_mean_rank_correlation": rank,
                        "boundary_raw_pool_chance_margin_rank_correlation": rank,
                        "boundary_raw_pool_constraint_mean_median_abs_error": mae,
                        "boundary_raw_pool_oracle_mean_variance_certified_count": (
                            0 if empty_certificate else 2
                        ),
                        "boundary_raw_pool_best_feasible_oracle_mean_variance_margin": -0.02,
                        "boundary_raw_pool_best_feasible_epistemic_radius": 0.03,
                        "boundary_raw_pool_failure_layer": "closed",
                    },
                }
                path = root / str(index) / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"rows": [row]}))
                index += 1


def test_v2_complete_matrix_promotes_smallest_winning_rank(tmp_path):
    _write_matrix(tmp_path)
    result = summarize(load_rows(tmp_path), expected_seeds=5)
    assert result["row_count"] == 140
    assert result["promotion_gate"]["advance_to_sequential"] is True
    assert result["promotion_gate"]["selected_variant"] == (
        "observable_state_phi_r2")
    assert result["promotion_gate"]["observable_state_track_is_oracle_free"]


def test_v2_empty_certificate_or_weak_coordinate_blocks_promotion(tmp_path):
    _write_matrix(tmp_path, weak_state=True, empty_certificate=True)
    result = summarize(load_rows(tmp_path), expected_seeds=5)
    assert result["promotion_gate"][
        "oracle_free_observable_challenger_selected"] is False
    assert result["promotion_gate"]["advance_to_sequential"] is False

