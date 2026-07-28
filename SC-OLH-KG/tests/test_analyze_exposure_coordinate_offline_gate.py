from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_exposure_coordinate_offline_gate import (  # noqa: E402
    EXPECTED_SCENARIOS,
    VARIANTS,
    load_rows,
    summarize,
)


def _write_matrix(root, *, weak_learned=False, omit=None):
    index = 0
    for variant in VARIANTS:
        provider = variant.startswith("provider_exposure")
        learned = variant.startswith("learned_exposure")
        for domain, shock in EXPECTED_SCENARIOS:
            for seed in range(5):
                key = (variant, domain, shock, seed)
                if key == omit:
                    continue
                if learned and not weak_learned:
                    rank, mae = 0.70, 0.10
                elif learned:
                    rank, mae = -0.30, 1.50
                elif provider:
                    rank, mae = 0.85, 0.07
                elif variant == "profile_phi_r2":
                    rank, mae = 0.40, 0.25
                else:
                    rank, mae = 0.50, 0.20
                row = {
                    "experiment_variant": (
                        f"exposure_coordinate_offline/{variant}/shock{shock:g}"
                    ),
                    "heldout": domain,
                    "target_shared_shock_scale": shock,
                    "seed": seed,
                    "audit": {
                        "tcb_target_structural_provider_used": provider,
                    },
                    "boundary_raw_pool_truth_diagnostics": {
                        "status": "audited",
                        "boundary_raw_pool_has_true_feasible": True,
                        "boundary_raw_pool_true_feasible_rate": 0.20,
                        "boundary_raw_pool_constraint_mean_rank_correlation": rank,
                        "boundary_raw_pool_chance_margin_rank_correlation": (
                            rank - 0.05),
                        "boundary_raw_pool_constraint_mean_median_abs_error": mae,
                        "boundary_raw_pool_variance_median_abs_log_error": 0.30,
                        "boundary_raw_pool_full_certified_count": 0,
                        "boundary_raw_pool_oracle_mean_variance_certified_count": 2,
                        "boundary_raw_pool_failure_layer": "constraint_mean",
                    },
                }
                path = root / str(index) / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"rows": [row]}))
                index += 1


def test_complete_sufficient_matrix_selects_smallest_winning_rank(tmp_path):
    _write_matrix(tmp_path)
    result = summarize(load_rows(tmp_path), expected_seeds=5)
    assert result["row_count"] == 160
    assert result["expected_row_count"] == 160
    assert result["promotion_gate"]["advance_to_sequential"] is True
    assert result["promotion_gate"]["selected_variant"] == (
        "learned_exposure_phi_r2")
    assert result["promotion_gate"]["provider_track_is_upper_bound"] is True


def test_weak_coordinate_or_missing_seed_blocks_sequential_gate(tmp_path):
    _write_matrix(
        tmp_path,
        weak_learned=True,
        omit=("latent_control", "QueueResourceControl", 1.0, 4),
    )
    result = summarize(load_rows(tmp_path), expected_seeds=5)
    assert result["promotion_gate"]["all_cells_complete"] is False
    assert result["promotion_gate"]["oracle_free_challenger_selected"] is False
    assert result["promotion_gate"]["advance_to_sequential"] is False
