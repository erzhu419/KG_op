import importlib.util
from pathlib import Path
import sys


ANALYZE_PATH = (
    Path(__file__).resolve().parents[1]
    / "performance/analyze_mean_alignment_v36_gate.py")
sys.path.insert(0, str(ANALYZE_PATH.parent))
ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "mean_v36_analyze", ANALYZE_PATH)
analyze = importlib.util.module_from_spec(ANALYZE_SPEC)
ANALYZE_SPEC.loader.exec_module(analyze)


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO
    / "scripts/submit_scolhkg_mean_alignment_v36_replication_gate_scheduler.py"
)
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v36_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v36",
        "rank": 4,
        "source_d": 50,
        "d": 1000,
        "N": 20,
        "n0": 10,
        "seed_start": 0,
        "n_seeds": 5,
        "pool_size": 512,
        "variance_audit_size": 512,
        "misspecification_prior_df": 4.0,
        "misspecification_ridge": 1.0,
        "misspecification_max_scale": 100.0,
        "misspecification_delta": 0.05,
        "contrast_scale": 1.0,
        "null_geometry_ridge": 1e-3,
        "python": defaults.REMOTE_PYTHON,
        "cpu": 1,
        "ram_mb": 4096,
    })()


def test_v36_submitter_pairs_new_and_replicate_action_spaces(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 4 * 3 * 5
    for variant, profile in submit.VARIANTS.items():
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3 * 5
        assert all(
            f"--decision-backend {profile['decision_backend']}" in spec["cmd"]
            for spec in selected)
        replication_flag = (
            "--adaptive-replication-voi"
            if profile["adaptive_replication_voi"]
            else "--no-adaptive-replication-voi"
        )
        assert all(replication_flag in spec["cmd"] for spec in selected)
        assert all(
            "--replication-candidate-count "
            f"{profile['replication_candidate_count']}" in spec["cmd"]
            for spec in selected)
        assert all(
            "--source-constraint-mean-misspecification-mode "
            "predictive_scale_sandwich_hc3_confidence" in spec["cmd"]
            for spec in selected)
        assert all(
            "--terminal-frontier-candidate-count 0" in spec["cmd"]
            for spec in selected)
    assert all(not spec.get("checkpoint_dir") for spec in specs)


def _gate_rows():
    rows = []
    for heldout, shock in analyze.base.base.MEAN_SCENARIOS:
        for variant in analyze.VARIANTS:
            adaptive = variant.startswith("v36_joint_rep")
            rows.append({
                "gate_variant": variant,
                "heldout": heldout,
                "target_shared_shock_scale": shock,
                "seed": 0,
                "true_feasible": (
                    adaptive or heldout != "QueueResourceControl"),
                "adaptive_loss": bool(
                    not adaptive and heldout == "QueueResourceControl"),
                "adaptive_improves_initial_best": False,
                "boundary_raw_pool_truth_diagnostics": {
                    "boundary_raw_pool_true_certified_count": (
                        1 if adaptive else 0),
                    "boundary_raw_pool_false_certified_count": 0,
                },
                "certificate_outcome_audit": {
                    "certified_true_feasible_count": 0,
                    "false_certificate_count": 0,
                },
                "adaptive_replication_voi": {
                    "enabled": adaptive,
                    "selected_replication_count": 2 if adaptive else 0,
                    "selected_new_point_count": 8 if adaptive else 10,
                    "target_oracle_used": False,
                },
                "decision_backend_contract": {
                    "online_updates_use_budgeted_target_observations_only": True,
                    "target_oracle_used": False,
                },
                "source_target_adaptation_contract": {
                    "target_initial_design_fingerprint": f"initial:{heldout}",
                    "source_archive_fingerprint": f"archive:{heldout}",
                },
            })
    return rows


def test_v36_gate_requires_oracle_free_replication_with_information_gain():
    summary = analyze.summarize(_gate_rows(), expected_seeds=1)
    assert summary["paired_initial_design_and_archive"]
    assert summary["diagnostic_eligible"] == [
        "v36_joint_rep4", "v36_joint_rep8"]
    assert summary["promotion_eligible"] == [
        "v36_joint_rep4", "v36_joint_rep8"]

    rows = _gate_rows()
    rows[-1]["adaptive_replication_voi"]["target_oracle_used"] = True
    broken = analyze.summarize(rows, expected_seeds=1)
    assert "v36_joint_rep8" not in broken["diagnostic_eligible"]
