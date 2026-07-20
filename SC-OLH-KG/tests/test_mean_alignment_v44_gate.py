import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "SC-OLH-KG"))

SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v44_terminal_risk_gate_scheduler.py"
)
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v44_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ANALYZE_PATH = (
    REPO / "SC-OLH-KG/performance/analyze_mean_alignment_v44_gate.py")
sys.path.insert(0, str(ANALYZE_PATH.parent))
ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "mean_v44_analyze", ANALYZE_PATH)
analyze = importlib.util.module_from_spec(ANALYZE_SPEC)
ANALYZE_SPEC.loader.exec_module(analyze)


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v44",
        "rank": 4,
        "source_d": 50,
        "d": 1000,
        "N": 20,
        "n0": 10,
        "seeds": "1,3",
        "pool_size": 512,
        "variance_audit_size": 512,
        "misspecification_prior_df": 4.0,
        "misspecification_ridge": 1.0,
        "misspecification_max_scale": 100.0,
        "misspecification_delta": 0.05,
        "confidence_delta": 0.05,
        "contrast_scale": 1.0,
        "null_geometry_ridge": 1e-3,
        "python": defaults.REMOTE_PYTHON,
        "cpu": 1,
        "ram_mb": 4096,
    })()


def test_v44_submitter_is_six_paired_sobol_tasks(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 3 * 2
    assert all("QueueResourceControl" in spec["signature"] for spec in specs)
    assert all(spec["signature"].endswith(("/seed1", "/seed3")) for spec in specs)
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)
    for variant, profile in submit.VARIANTS.items():
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 2
        assert all(
            "--decision-risk-penalty "
            f"{profile['decision_risk_penalty']}" in spec["cmd"]
            for spec in selected
        )
        assert all("--decision-backend sobol_new" in spec["cmd"]
                   for spec in selected)


def _row(variant, seed):
    penalty = analyze.PENALTIES[variant]
    feasible = variant != "v41_rho5" or seed == 1
    return {
        "gate_variant": variant,
        "heldout": "QueueResourceControl",
        "target_shared_shock_scale": 1.0,
        "seed": seed,
        "decision_backend": "sobol_new",
        "decision_risk_penalty": penalty,
        "source_archive_fingerprint": "archive",
        "target_design_fingerprint": f"design-{seed}",
        "online_action_sequence_fingerprint": f"actions-{seed}",
        "online_action_trace": [{
            "x_fingerprint": f"x-{seed}",
            "observed_response": [0.1, -0.1],
            "candidate_source": "sobol_continuation",
        }],
        "adaptive_replication_voi": {"enabled": False},
        "true_feasible": feasible,
        "adaptive_loss": not feasible,
        "adaptive_improves_initial_best": False,
        "feasible_simple_regret": 0.01 if feasible else None,
        "certificate_outcome_audit": {
            "certified_true_feasible_count": 0,
            "false_certificate_count": 0,
        },
    }


def test_v44_fixed_penalty_signal_can_only_warrant_source_dual_learning():
    rows = [
        _row(variant, seed)
        for seed in analyze.SENTINEL_SEEDS
        for variant in analyze.VARIANTS
    ]
    summary = analyze.summarize(rows)
    assert summary["paired_archive_design_actions_and_responses"]
    assert all(summary["decision_penalty_contract"].values())
    assert summary["source_dual_learning_warranted"]
    assert summary["clean_mechanism_signals"] == [
        "v44_rho20", "v44_rho80"]
    assert summary["promotion_eligible"] == []
