from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_boundary_coordinate_gate import (  # noqa: E402
    EXPECTED_SCENARIOS,
    VARIANTS,
    load_rows,
    summarize,
)


def _write_matrix(root, *, omit=None, false_certificate=False):
    index = 0
    for variant in VARIANTS:
        for domain, shock in EXPECTED_SCENARIOS:
            for seed in range(5):
                key = (variant, domain, shock, seed)
                if key == omit:
                    continue
                joint = variant == "phi_mean_proposal"
                aligned = variant != "latent_control"
                row = {
                    "experiment_variant": (
                        f"boundary_coordinate_gate/{variant}/shock{shock:g}"
                    ),
                    "heldout": domain,
                    "target_shared_shock_scale": shock,
                    "seed": seed,
                    "true_feasible": True,
                    "feasible_simple_regret": 0.01 if joint else 0.02,
                    "posterior_chance_margin": -0.1 if joint else 0.1,
                    "certificate_outcome_audit": {
                        "posterior_certified_count": int(joint),
                        "false_certificate_count": int(
                            false_certificate and joint and seed == 0),
                    },
                    "adaptive_outcome_audit": {"adaptive_loss": False},
                    "truth_pool_diagnostics": {
                        "mean_constraint_mean_rank_correlation": (
                            0.7 if aligned else 0.2),
                        "mean_chance_margin_rank_correlation": (
                            0.6 if aligned else 0.1),
                        "mean_constraint_mean_median_abs_error": (
                            0.1 if aligned else 0.3),
                        "mean_variance_median_abs_log_error": 0.2,
                        "phi_candidate_true_feasible_iteration_rate": (
                            0.5 if joint else None),
                        "failure_layer_counts": {
                            "closed" if joint else "constraint_mean": 10,
                        },
                    },
                    "boundary_coordinate_proposal": {
                        "generated_iteration_count": 10 if joint else 0,
                    },
                }
                path = root / str(index) / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"rows": [row]}))
                index += 1


def test_complete_sound_matrix_passes_promotion_gate(tmp_path):
    _write_matrix(tmp_path)
    result = summarize(load_rows(tmp_path), expected_seeds=5)
    assert result["row_count"] == 60
    assert result["expected_row_count"] == 60
    assert len(result["cells"]) == 12
    assert result["promotion_gate"]["advance"] is True
    assert result["promotion_gate"][
        "factor_shock_scale4_phi_support"] is True
    assert all(cell["complete"] for cell in result["cells"])


def test_missing_seed_or_false_certificate_blocks_promotion(tmp_path):
    _write_matrix(
        tmp_path,
        omit=("latent_control", "QueueResourceControl", 1.0, 4),
        false_certificate=True,
    )
    result = summarize(load_rows(tmp_path), expected_seeds=5)
    assert result["promotion_gate"]["all_cells_complete"] is False
    assert result["promotion_gate"]["zero_false_certificates"] is False
    assert result["promotion_gate"]["advance"] is False
