import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO / "scripts/submit_scolhkg_observable_dual_head_offline_gate_scheduler.py"
)
SPEC = importlib.util.spec_from_file_location(
    "observable_dual_head_submit", SCRIPT)
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_observable_dual_head_offline_gate import (  # noqa: E402
    SCENARIOS,
    VARIANTS,
    load_rows,
    summarize,
)


def _args(tmp_path, **overrides):
    deploy = tmp_path / "deploy"
    values = {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": submit.DEFAULT_SOURCE_RUN_ID,
        "run_id": "observable-dual-head",
        "rank": 4,
        "source_d": 50,
        "d": 1000,
        "N": 10,
        "n0": 10,
        "seed_start": 0,
        "n_seeds": 5,
        "pool_size": 512,
        "variance_audit_size": 512,
        "python": submit.REMOTE_PYTHON,
        "cpu": 1,
        "ram_mb": 4096,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_dual_head_gate_is_eighty_checkpoint_free_shards(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 4 * 4 * 5
    assert len({spec["signature"] for spec in specs}) == len(specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
    assert all("--runtime-checkpoint-dir ''" in spec["cmd"] for spec in specs)
    dual = next(spec for spec in specs
                if "/observable_dual_head/" in spec["signature"])
    assert "--observable-mean-input-mode observable_state_exposure" in dual["cmd"]
    assert "--observable-variance-input-mode observable_state_exposure" in (
        dual["cmd"])
    assert "--truth-pool-max-candidates 0" in dual["cmd"]


def _write_passing_matrix(root):
    index = 0
    for variant in VARIANTS:
        dual = variant == "observable_dual_head"
        state_mean = variant in {"observable_mean_only", "observable_dual_head"}
        state_variance = variant in {
            "observable_variance_only", "observable_dual_head"
        }
        for domain, shock in SCENARIOS:
            for seed in range(5):
                row = {
                    "experiment_variant": (
                        f"observable_dual_head_offline_r4/{variant}/shock{shock:g}"
                    ),
                    "heldout": domain,
                    "target_shared_shock_scale": shock,
                    "seed": seed,
                    "audit": {
                        "admissible_oracle_free_transfer": True,
                    },
                    "mean_risk_coordinate_contract": {
                        "shared_observable_exposure_input": dual,
                    },
                    "variance_log_rmse": 0.30 if state_variance else 0.50,
                    "certified_variance_log_rmse": (
                        0.35 if state_variance else 0.55),
                    "variance_upper_coverage": 0.95,
                    "boundary_raw_pool_truth_diagnostics": {
                        "boundary_raw_pool_has_true_feasible": True,
                        "boundary_raw_pool_constraint_mean_rank_correlation": (
                            0.70 if state_mean else 0.50),
                        "boundary_raw_pool_constraint_mean_median_abs_error": (
                            0.20 if state_mean else 0.25),
                        "boundary_raw_pool_oracle_mean_variance_certified_count": 2,
                        "boundary_raw_pool_true_feasible_count": 8,
                        "boundary_raw_pool_full_certified_count": 3,
                        "boundary_raw_pool_best_feasible_epistemic_radius": 0.02,
                        "boundary_raw_pool_best_feasible_true_margin": -0.10,
                        "boundary_raw_pool_best_feasible_oracle_mean_variance_margin": -0.08,
                        "boundary_raw_pool_failure_layer": "passed",
                    },
                }
                path = root / str(index) / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"rows": [row]}))
                index += 1


def test_dual_head_analyzer_promotes_complete_causal_gain(tmp_path):
    _write_passing_matrix(tmp_path)
    result = summarize(load_rows(tmp_path), expected_seeds=5)
    assert result["row_count"] == 80
    assert result["promotion_gate"]["advance_to_sequential"] is True
    assert result["promotion_gate"]["both_heads_share_observable_exposure"]
    dual = next(
        cell for cell in result["cells"]
        if cell["variant"] == "observable_dual_head"
        and cell["domain"] == "InventorySupplyChain")
    assert dual["median_posterior_certified_count"] == 3
    assert abs(
        dual["median_epistemic_to_safety_depth_ratio"] - 0.2
    ) < 1e-12
    assert dual["failure_layer_counts"] == {"passed": 5}
