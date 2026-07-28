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
    "mean_v51_submit",
    REPO / "scripts/submit_scolhkg_mean_alignment_v51_action_set_gate_scheduler.py",
)
analyze = _load(
    "mean_v51_analyze",
    REPO / "SC-OLH-KG/performance/analyze_mean_alignment_v51_gate.py",
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
        "run_id": "mean-v51",
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


def test_v51_submitter_builds_nine_balanced_action_sentinels(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 3 * 3
    assert all(spec["cpu"] == 12 and spec["vram"] == 0 for spec in specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
    assert all("--source-records-per-domain 64" in spec["cmd"]
               for spec in specs)
    assert all(
        "--evaluate-or-replicate-new-action-policy "
        "canonical_plus_posterior_risk" in spec["cmd"]
        for spec in specs
    )
    for variant, profile in submit.VARIANTS.items():
        selected = [spec for spec in specs
                    if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3
        assert all(
            f"--evaluate-or-replicate-new-action-count "
            f"{profile['evaluate_or_replicate_new_action_count']}"
            in spec["cmd"] for spec in selected
        )
        flag = (
            "--adaptive-replication-voi"
            if profile["adaptive_replication_voi"]
            else "--no-adaptive-replication-voi"
        )
        assert all(flag in spec["cmd"] for spec in selected)


def _source_fields():
    return {
        "meta_prior": {"training": {
            "source_base_domain_count": 2,
            "source_episode_count_per_base_domain": 1,
            "source_archive_simulator_calls": 384,
            "target_seed_used_for_source_training": False,
            "source_episode_target_oracle_used": False,
        }},
        "source_target_adaptation_contract": {
            "source_simulator_calls": 384,
            "source_oracle_aided": False,
            "target_oracle_used_for_adaptation": False,
        },
    }


def _row(variant, scenario, seed, *, improve=False):
    new_count = analyze.EXPECTED_NEW[variant]
    allow_replication = analyze.EXPECTED_REPLICATION[variant]
    rep_count = 2 if allow_replication else 0
    new_scores = [2.0 - 0.01 * index for index in range(new_count)]
    rep_scores = [1.0, 0.5] if allow_replication else []
    flags = [False] * new_count + [True] * rep_count
    scores = new_scores + rep_scores
    delta = None if not allow_replication else max(new_scores) - max(rep_scores)
    trace = [{
        "action_kind": "new",
        "active_new_action_count": new_count,
        "active_replication_action_count": rep_count,
        "new_action_policy": "canonical_plus_posterior_risk",
        "exact_kg_raw_scores_active": scores,
        "exact_kg_active_action_is_replicate": flags,
        "exact_kg_best_new_raw": max(new_scores),
        "exact_kg_best_replication_raw": (
            max(rep_scores) if rep_scores else None),
        "exact_kg_new_minus_replication_raw": delta,
    } for _ in range(10)]
    return {
        "gate_variant": variant,
        "heldout": scenario,
        "seed": seed,
        "decision_backend": "sobol_exact_joint_voi",
        "decision_aleatoric_mode": "posterior_central",
        "decision_ambiguity_mode": "posterior_nominal",
        "decision_violation_loss_mode": "positive_part",
        "decision_backend_contract": {
            "evaluate_or_replicate_new_action_count": new_count,
            "evaluate_or_replicate_new_action_policy": (
                "canonical_plus_posterior_risk"),
        },
        "adaptive_replication_voi_enabled": allow_replication,
        "online_action_trace_target_oracle_used": False,
        "online_action_trace": trace,
        "true_feasible": True,
        "adaptive_loss": False,
        "adaptive_improves_initial_best": improve,
        "feasible_simple_regret": 0.005 if improve else 0.00825,
        "certificate_outcome_audit": {"false_certificate_count": 0},
        "adaptive_replication_selected_count": 0,
        "adaptive_new_point_selected_count": 10,
        **_source_fields(),
    }


def _control():
    return {
        "count": 3,
        "complete": True,
        "true_feasible": 3,
        "adaptive_losses": 0,
        "adaptive_improvements": 0,
        "false_certificates": 0,
        "selected_replications": 20,
        "selected_new_points": 10,
        "new_arm_wins": 0,
        "replicate_arm_wins": 0,
        "median_new_minus_replication_raw": None,
        "median_feasible_regret": 0.00825,
        "by_domain": {
            scenario: {"feasible_simple_regret": 0.00825}
            for scenario in analyze.SCENARIO_SEEDS
        },
    }


def test_v51_analyzer_accepts_balanced_trace_and_warrants_strict_gate():
    rows = [
        _row(
            variant,
            scenario,
            seed,
            improve=(variant == "balanced8" and scenario == "QueueResourceControl"),
        )
        for variant in analyze.VARIANTS
        for scenario, seed in analyze.SCENARIO_SEEDS.items()
    ]
    result = analyze.summarize(rows, _control())
    assert all(result["source_contract"].values())
    assert all(result["mode_contract"].values())
    assert all(result["action_trace_contract"].values())
    assert result["full_gate_warranted"]
    assert "balanced8" in result["full_gate_candidates"]
    assert result["promotion_eligible"] == []


def test_v51_analyzer_rejects_inconsistent_arm_winner():
    rows = [
        _row(variant, scenario, seed)
        for variant in analyze.VARIANTS
        for scenario, seed in analyze.SCENARIO_SEEDS.items()
    ]
    rows[0]["online_action_trace"][0]["action_kind"] = "replicate"
    result = analyze.summarize(rows, _control())
    assert not result["action_trace_contract"][rows[0]["gate_variant"]]
    assert not result["full_gate_warranted"]
