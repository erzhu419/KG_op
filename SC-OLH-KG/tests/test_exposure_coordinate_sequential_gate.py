import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "scripts/submit_scolhkg_exposure_coordinate_sequential_gate_scheduler.py"
)
SPEC = importlib.util.spec_from_file_location(
    "exposure_coordinate_sequential_submit", SCRIPT)
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_exposure_coordinate_sequential_gate import (  # noqa: E402
    EXPECTED_SCENARIOS,
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
        "run_id": "exposure-coordinate-sequential",
        "rank": 4,
        "source_d": 50,
        "d": 1000,
        "N": 20,
        "n0": 10,
        "seed_start": 0,
        "n_seeds": 5,
        "boundary_coordinate_pool_size": 512,
        "evaluate_interval": 20,
        "python": submit.REMOTE_PYTHON,
        "cpu": 1,
        "ram_mb": 4096,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_sequential_gate_is_sixty_checkpoint_free_shards(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 60
    assert len({spec["signature"] for spec in specs}) == 60
    assert all("--N 20 --n0 10" in spec["cmd"] for spec in specs)
    assert all("--runtime-checkpoint-dir ''" in spec["cmd"] for spec in specs)
    learned = next(
        spec for spec in specs if "/exposure_mean_only/" in spec["signature"])
    proposal = next(
        spec for spec in specs
        if "/exposure_mean_proposal/" in spec["signature"])
    assert "--observable-mean-latent-dim 4" in learned["cmd"]
    assert "--observable-mean-input-mode source_learned_exposure" in (
        learned["cmd"])
    assert "--boundary-coordinate-candidate-count 0" in learned["cmd"]
    assert "--boundary-coordinate-candidate-count 12" in proposal["cmd"]


def _write_passing_matrix(root):
    index = 0
    for variant in VARIANTS:
        for domain, shock in EXPECTED_SCENARIOS:
            for seed in range(5):
                proposal = variant == "exposure_mean_proposal"
                mean = variant != "latent_control"
                row = {
                    "experiment_variant": (
                        f"exposure_coordinate_sequential_r4/{variant}/"
                        f"shock{shock:g}"
                    ),
                    "heldout": domain,
                    "target_shared_shock_scale": shock,
                    "seed": seed,
                    "true_feasible": True,
                    "feasible_simple_regret": 0.01,
                    "certificate_outcome_audit": {
                        "posterior_certified_count": int(proposal),
                        "false_certificate_count": 0,
                    },
                    "adaptive_outcome_audit": {"adaptive_loss": False},
                    "truth_pool_diagnostics": {
                        "mean_constraint_mean_rank_correlation": (
                            0.7 if mean else 0.5),
                        "phi_candidate_true_feasible_iteration_rate": (
                            0.5 if proposal else None),
                    },
                    "boundary_raw_pool_truth_diagnostics": {
                        "boundary_raw_pool_constraint_mean_rank_correlation": (
                            0.7 if mean else 0.5),
                        "boundary_raw_pool_oracle_mean_variance_certified_count": 2,
                    },
                }
                if domain == "FactorShockStatePolicyRZDT1" and shock == 4.0:
                    row["true_feasible"] = proposal
                path = root / str(index) / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"rows": [row]}))
                index += 1


def test_sequential_analyzer_promotes_complete_safe_gain(tmp_path):
    _write_passing_matrix(tmp_path)
    result = summarize(load_rows(tmp_path), expected_seeds=5)
    assert result["row_count"] == 60
    assert result["promotion_gate"]["promote"] is True
