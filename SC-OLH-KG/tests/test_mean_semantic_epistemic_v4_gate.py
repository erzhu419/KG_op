import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "scripts/submit_scolhkg_mean_semantic_epistemic_offline_gate_scheduler.py"
)
SPEC = importlib.util.spec_from_file_location("mean_v4_submit", SCRIPT)
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_semantic_epistemic_offline_gate import (  # noqa: E402
    JOINT_VARIANTS,
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
        "run_id": "mean-v4",
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


def test_v4_submitter_builds_120_checkpoint_free_cpu_shards(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 6 * 4 * 5
    assert len({spec["signature"] for spec in specs}) == len(specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
    assert all("--runtime-checkpoint-dir ''" in spec["cmd"] for spec in specs)
    joint = next(spec for spec in specs
                 if "/v4_invariant_joint/" in spec["signature"])
    assert "--observable-mean-descriptor-mode set_invariant" in joint["cmd"]
    assert "--observable-mean-feature-mode diagonal_quadratic" in joint["cmd"]
    assert "--source-constraint-mean-deviation-mode latent_shared" in (
        joint["cmd"])
    assert "--observable-variance-input-mode observable_state_exposure" in (
        joint["cmd"])


def _write_matrix(root, *, false_certificate=False):
    index = 0
    for variant in VARIANTS:
        joint = variant in JOINT_VARIANTS
        descriptor = (
            "set_invariant"
            if variant in {"semantic_invariant", "v4_invariant_joint"}
            else "ordered"
        )
        feature = (
            "diagonal_quadratic"
            if variant in {
                "boundary_quadratic", "v4_ordered_joint",
                "v4_invariant_joint",
            }
            else "linear"
        )
        deviation = (
            "latent_shared"
            if variant in {
                "epistemic_latent", "v4_ordered_joint",
                "v4_invariant_joint",
            }
            else "raw_independent"
        )
        for domain, shock in SCENARIOS:
            for seed in range(5):
                control_mae = 0.40 if domain == "InventorySupplyChain" else 0.20
                mae = (0.20 if domain == "InventorySupplyChain" else 0.18) \
                    if joint else control_mae
                row = {
                    "experiment_variant": f"mean_v4_offline/{variant}/shock{shock:g}",
                    "heldout": domain,
                    "target_shared_shock_scale": shock,
                    "seed": seed,
                    "audit": {"admissible_oracle_free_transfer": True},
                    "mean_risk_coordinate_contract": {
                        "shared_observable_exposure_input": True,
                    },
                    "meta_observable_mean_descriptor_mode": descriptor,
                    "meta_observable_mean_feature_mode": feature,
                    "source_constraint_mean_deviation_mode": deviation,
                    "source_constraint_mean_adaptation_mode": (
                        "sequential_evidence_mixture"
                        if joint else "evidence_mixture"),
                    "variance_log_rmse": 0.50 if joint else 0.52,
                    "variance_upper_coverage": 0.95,
                    "boundary_raw_pool_truth_diagnostics": {
                        "boundary_raw_pool_has_true_feasible": True,
                        "boundary_raw_pool_constraint_mean_rank_correlation": (
                            0.75 if joint else 0.70),
                        "boundary_raw_pool_constraint_mean_median_abs_error": mae,
                        "boundary_raw_pool_oracle_mean_variance_certified_count": (
                            2 if joint else 0),
                        "boundary_raw_pool_full_certified_count": 1 if joint else 0,
                        "boundary_raw_pool_false_certified_count": (
                            1 if false_certificate and joint and seed == 0 else 0),
                        "boundary_raw_pool_certificate_precision": 1.0,
                        "boundary_raw_pool_best_feasible_epistemic_radius": 0.02,
                        "boundary_raw_pool_failure_layer": "closed",
                    },
                }
                path = root / str(index) / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"rows": [row]}))
                index += 1


def test_v4_analyzer_promotes_joint_semantic_epistemic_gain(tmp_path):
    _write_matrix(tmp_path)
    result = summarize(load_rows(tmp_path), expected_seeds=5)
    assert result["row_count"] == 120
    assert result["all_matrix_cells_complete"] is True
    assert result["advance_to_sequential"] is True
    assert result["selected_joint_variant"] in JOINT_VARIANTS


def test_v4_false_certificate_blocks_promotion(tmp_path):
    _write_matrix(tmp_path, false_certificate=True)
    result = summarize(load_rows(tmp_path), expected_seeds=5)
    assert result["advance_to_sequential"] is False
    assert all(
        candidate["checks"]["zero_false_certificates"] is False
        for candidate in result["joint_candidate_gates"]
    )
