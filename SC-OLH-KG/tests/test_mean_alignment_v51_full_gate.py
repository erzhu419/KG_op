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
    "mean_v51_full_submit",
    REPO / "scripts/submit_scolhkg_mean_alignment_v51_full_gate_scheduler.py",
)
analyze = _load(
    "mean_v51_full_analyze",
    REPO / "SC-OLH-KG/performance/analyze_mean_alignment_v51_full_gate.py",
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
        "remote_design_only": False,
        "run_id": "mean-v51-full",
        "rank": 4,
        "source_d": 50,
        "source_records_per_domain": 64,
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
        "cpu": 12,
        "ram_mb": 8192,
    })()


def test_v51_full_submitter_builds_forty_five_paired_tasks(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 3 * 3 * 5
    assert all(spec["cpu"] == 12 and spec["vram"] == 0 for spec in specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
    assert all("--source-records-per-domain 64" in spec["cmd"]
               for spec in specs)
    balanced = [spec for spec in specs
                if "/balanced4/" in spec["signature"]]
    canonical = [spec for spec in specs
                 if "/exact_canonical/" in spec["signature"]]
    assert len(balanced) == len(canonical) == 15
    assert all(
        "--evaluate-or-replicate-new-action-count 4" in spec["cmd"]
        and "--evaluate-or-replicate-new-action-policy "
        "canonical_plus_posterior_risk" in spec["cmd"]
        for spec in balanced
    )
    assert all(
        "--evaluate-or-replicate-new-action-count 1" in spec["cmd"]
        and "--evaluate-or-replicate-new-action-policy canonical_sobol"
        in spec["cmd"]
        for spec in canonical
    )


def test_v51_full_submitter_can_use_remote_only_frozen_designs(tmp_path):
    args = _args(tmp_path)
    args.remote_design_only = True
    args.source_run_id = "remote-s20-designs"
    specs = submit.build_specs(args)
    assert all(spec["wait_for_files"] == [] for spec in specs)
    assert all(spec["cmd"].startswith("test -f ") for spec in specs)
    assert all("archives/remote-s20-designs/" in spec["cmd"]
               for spec in specs)


def _row(variant, scenario, seed):
    improved = variant == "balanced4" and seed < 3
    base_regret = 0.02 + 0.001 * seed
    regret = base_regret - 0.005 if improved else base_regret
    source = {
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
    row = {
        "gate_variant": variant,
        "heldout": scenario,
        "seed": seed,
        "source_constraint_mean_adaptation_mode": (
            "sequential_aggregate_hyperlaw"),
        "audit": {"uses_true_constraint": False},
        "online_action_trace_target_oracle_used": False,
        "task_initial_design": {
            "fingerprint": f"n0-{scenario}-{seed}",
            "source_archive_fingerprint": "source-archive",
            "n_unique": 10,
            "source_only": True,
            "target_labels_used": False,
            "target_oracle_used": False,
        },
        "initial_best_feasible_regret": base_regret,
        "true_feasible": True,
        "adaptive_loss": False,
        "adaptive_improves_initial_best": improved,
        "feasible_simple_regret": regret,
        "certificate_outcome_audit": {"false_certificate_count": 0},
        "adaptive_replication_selected_count": 4,
        "adaptive_new_point_selected_count": 6,
        **source,
    }
    if variant != "promoted_v27":
        count = 1 if variant == "exact_canonical" else 4
        policy = (
            "canonical_sobol" if count == 1
            else "canonical_plus_posterior_risk")
        row.update({
            "decision_backend": "sobol_exact_joint_voi",
            "decision_aleatoric_mode": "posterior_central",
            "decision_ambiguity_mode": "posterior_nominal",
            "decision_violation_loss_mode": "positive_part",
            "decision_backend_contract": {
                "evaluate_or_replicate_new_action_count": count,
                "evaluate_or_replicate_new_action_policy": policy,
            },
            "adaptive_replication_voi_enabled": True,
        })
    return row


def _rows():
    return [
        _row(variant, scenario, seed)
        for variant in analyze.VARIANTS
        for scenario in analyze.SCENARIOS
        for seed in analyze.SEEDS
    ]


def test_v51_full_analyzer_warrants_twenty_seed_gate():
    result = analyze.summarize(_rows())
    assert all(result["source_contract"].values())
    assert all(result["mode_contract"].values())
    assert result["initial_pairing_contract"]
    assert result["domainwise_safety_noninferior"]
    assert result["strict_paired_gain"]
    assert result["domainwise_regret_noninferior"]
    assert result["domainwise_paired_noninferior"]
    assert result["s20_warranted"]
    assert result["promotion_eligible"] == []


def test_v51_full_analyzer_promotes_only_after_twenty_seed_gate():
    seeds = tuple(range(20))
    rows = [
        _row(variant, scenario, seed)
        for variant in analyze.VARIANTS
        for scenario in analyze.SCENARIOS
        for seed in seeds
    ]
    result = analyze.summarize(rows, seeds=seeds)
    assert result["gate_passes"]
    assert not result["s20_warranted"]
    assert result["promotion_eligible"] == ["balanced4"]


def test_v51_full_analyzer_rejects_unpaired_initial_design():
    rows = _rows()
    rows[0]["task_initial_design"]["fingerprint"] = "mismatch"
    result = analyze.summarize(rows)
    assert not result["initial_pairing_contract"]
    assert not result["s20_warranted"]
