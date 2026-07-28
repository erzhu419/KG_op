import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "SC-OLH-KG"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


submit = _load(
    "mean_v49_submit",
    REPO
    / "scripts/submit_scolhkg_mean_alignment_v49_decision_loss_gate_scheduler.py",
)
analyze = _load(
    "mean_v49_analyze",
    REPO / "SC-OLH-KG/performance/analyze_mean_alignment_v49_gate.py",
)


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "python": defaults.REMOTE_PYTHON,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v49",
        "rank": 4,
        "source_d": 50,
        "source_records_per_domain": 64,
        "d": 1000,
        "N": 20,
        "n0": 10,
        "pool_size": 512,
        "variance_audit_size": 512,
        "misspecification_prior_df": 4.0,
        "misspecification_ridge": 1.0,
        "misspecification_max_scale": 100.0,
        "misspecification_delta": 0.05,
        "contrast_scale": 1.0,
        "null_geometry_ridge": 1e-3,
        "cpu": 12,
        "ram_mb": 8192,
    })()


def test_v49_submitter_builds_twenty_four_factorial_sentinels(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 8 * 3
    assert all(spec["cpu"] == 12 and spec["vram"] == 0 for spec in specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)
    assert all("--source-records-per-domain 64" in spec["cmd"]
               for spec in specs)
    for variant, profile in submit.VARIANTS.items():
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3
        assert all(
            f"--decision-aleatoric-mode "
            f"{profile['decision_aleatoric_mode']}" in spec["cmd"]
            for spec in selected
        )
        assert all(
            f"--decision-violation-loss-mode "
            f"{profile['decision_violation_loss_mode']}" in spec["cmd"]
            for spec in selected
        )
        if variant.startswith("exact_"):
            assert all("--exact-jobs 12" in spec["cmd"] for spec in selected)
            assert all("--adaptive-replication-voi" in spec["cmd"]
                       for spec in selected)
            assert all("--no-exact-clip-negative" in spec["cmd"]
                       for spec in selected)


def _row(variant, scenario, seed):
    expected = analyze.EXPECTED_MODES[variant]
    combined = variant.endswith("central_probability")
    queue_failure = scenario == "QueueResourceControl" and not combined
    exact = variant.startswith("exact_")
    return {
        "gate_variant": variant,
        "heldout": scenario,
        "seed": seed,
        "decision_backend": expected["backend"],
        "decision_aleatoric_mode": expected["aleatoric"],
        "decision_violation_loss_mode": expected["loss"],
        "decision_backend_terminal_aleatoric_mode": expected["aleatoric"],
        "decision_backend_terminal_violation_loss_mode": expected["loss"],
        "terminal_bayes_pool_audit": {
            "status": "ranked",
            "decision_aleatoric_mode": expected["aleatoric"],
            "violation_loss_mode": expected["loss"],
            "target_oracle_used_for_ranking": False,
            "target_oracle_used_for_decision": False,
            "truth_admissible_decision_input": False,
            "truth_join_timing": "post_terminal_rank",
        },
        "meta_prior": {"training": {
            "source_base_domain_count": 2,
            "source_episode_count_per_base_domain": 1,
            "source_archive_simulator_calls": 384,
            "target_seed_used_for_source_training": False,
            "source_episode_target_oracle_used": False,
        }},
        "source_target_adaptation_contract": {
            "source_simulator_calls": 384,
            "source_constraint_mean_adaptation_mode": (
                "sequential_aggregate_hyperlaw"),
            "source_oracle_aided": False,
            "target_oracle_used_for_adaptation": False,
        },
        "target_design_fingerprint": f"target-{scenario}-{seed}",
        "online_action_sequence_fingerprint": (
            f"sobol-{scenario}-{seed}" if not exact
            else f"{variant}-{scenario}-{seed}"),
        "online_action_trace": [{
            "x_fingerprint": f"sobol-x-{scenario}-{seed}",
            "observed_response": [0.2, 0.1],
            "candidate_source": "sobol_continuation",
        }],
        "true_feasible": not queue_failure,
        "adaptive_loss": queue_failure,
        "adaptive_improves_initial_best": bool(
            combined and scenario == "QueueResourceControl"),
        "feasible_simple_regret": None if queue_failure else 0.01,
        "certificate_outcome_audit": {"false_certificate_count": 0},
        "adaptive_replication_selected_count": (
            3 if exact and combined else (8 if exact else 0)),
        "adaptive_new_point_selected_count": (
            7 if exact and combined else (2 if exact else 10)),
    }


def _rows():
    return [
        _row(variant, scenario, seed)
        for variant in analyze.VARIANTS
        for scenario, seed in analyze.SCENARIO_SEEDS.items()
    ]


def test_v49_analyzer_requires_terminal_and_action_mix_improvement():
    result = analyze.summarize(_rows())
    assert all(result["source_contract"].values())
    assert all(result["mode_contract"].values())
    assert result["paired_static_actions_and_responses"]
    assert result["combined_exact_fewer_replications_than_current"]
    assert result["combined_exact_more_new_points_than_current"]
    assert result["combined_exact_no_safety_regression"]
    assert result["combined_static_terminal_improvement"]
    assert result["full_gate_warranted"]
    assert result["promotion_eligible"] == []


def test_v49_analyzer_rejects_truth_used_for_terminal_ranking():
    rows = _rows()
    rows[0]["terminal_bayes_pool_audit"][
        "target_oracle_used_for_ranking"] = True
    result = analyze.summarize(rows)
    assert not result["mode_contract"][rows[0]["gate_variant"]]
    assert not result["full_gate_warranted"]
