import copy
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
    "mean_v48_submit",
    REPO
    / "scripts/submit_scolhkg_mean_alignment_v48_certified_incumbent_gate_scheduler.py",
)
analyze = _load(
    "mean_v48_analyze",
    REPO / "SC-OLH-KG/performance/analyze_mean_alignment_v48_gate.py",
)
v47_test = _load(
    "mean_v47_test_fixture",
    REPO / "SC-OLH-KG/tests/test_mean_alignment_v47_gate.py",
)


def _scheduler_args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "python": defaults.REMOTE_PYTHON,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v48",
        "rank": 4,
        "source_d": 50,
        "source_records_per_domain": 64,
        "d": 1000,
        "N": 20,
        "n0": 10,
        "seeds": "1,3",
        "scope": "queue_sentinel",
        "pool_size": 512,
        "variance_audit_size": 512,
        "misspecification_prior_df": 4.0,
        "misspecification_ridge": 1.0,
        "misspecification_max_scale": 100.0,
        "misspecification_delta": 0.05,
        "confidence_delta": 0.05,
        "contrast_scale": 1.0,
        "null_geometry_ridge": 1e-3,
        "cpu": 1,
        "ram_mb": 4096,
    })()


def test_v48_submitter_builds_ten_paired_cpu_sentinels(tmp_path):
    specs = submit.build_specs(_scheduler_args(tmp_path))
    assert len(specs) == 10
    assert all(spec["cpu"] == 1 and spec["vram"] == 0 for spec in specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
    assert all("jtl110cpu" not in spec["allowed_nodes"] for spec in specs)
    for variant in submit.VARIANTS:
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 2
        initialization = submit.VARIANTS[variant][
            "posterior_dominance_initialization"]
        assert all(
            f"--posterior-dominance-initialization {initialization}"
            in spec["cmd"] for spec in selected
        )
        enabled = submit.VARIANTS[variant]["posterior_dominance_enabled"]
        flag = (
            "--posterior-dominance-enabled"
            if enabled else "--no-posterior-dominance-enabled"
        )
        assert all(flag in spec["cmd"] for spec in selected)


def _terminal_audit(selected_matches=True):
    return {
        "status": "ranked",
        "pool_size": 20,
        "selected_fingerprint": "selected",
        "selected_risk": 0.2,
        "selected_risk_rank": 1 if selected_matches else 2,
        "counterfactual_bayes_fingerprint": "selected",
        "counterfactual_bayes_risk": 0.2,
        "selected_matches_counterfactual_bayes": bool(selected_matches),
        "target_oracle_used_for_ranking": False,
        "truth_after_rank_freeze_available": True,
        "truth_join_timing": "post_terminal_rank",
        "truth_admissible_decision_input": False,
        "selected_true_margin": -0.01,
        "selected_true_feasible": True,
        "counterfactual_bayes_true_margin": -0.01,
        "counterfactual_bayes_true_feasible": True,
        "target_oracle_used_for_decision": False,
    }


def _row(variant, seed):
    source_variant = "v47_deconvolved_task_bayes"
    row = copy.deepcopy(v47_test._v47_row(source_variant, seed))
    row["gate_variant"] = variant
    row["true_feasible"] = True
    row["adaptive_loss"] = False
    row["adaptive_improves_initial_best"] = False
    row["feasible_simple_regret"] = 0.01
    row["x_recommended"] = [11, 12, 13]
    row["decision_backend_terminal_used"] = True
    row["posterior_dominance_terminal_used"] = False
    row["terminal_bayes_pool_audit"] = _terminal_audit()

    if variant.startswith("v27_"):
        training = row["meta_prior"]["training"]
        training["source_episode_count_per_base_domain"] = 1
        row["config"] = {
            "source_constraint_mean_adaptation_mode": (
                "sequential_aggregate_hyperlaw"),
        }

    if variant == "v47_risk_preservation":
        row["posterior_dominance_terminal_used"] = True
        row["decision_backend_terminal_used"] = False
        row["terminal_bayes_pool_audit"] = _terminal_audit(False)
        row["posterior_dominance"] = {
            "enabled": True,
            "incumbent": [91, 92, 93],
            "history": [{
                "posterior_dominance_initialization": "risk",
                "status": "initialized",
                "target_oracle_used": False,
            }],
        }
        row["x_recommended"] = [91, 92, 93]
        if seed == 3:
            row["true_feasible"] = False
            row["adaptive_loss"] = True
            row["feasible_simple_regret"] = None
    elif variant in analyze.CERTIFIED_ONLY_VARIANTS:
        row["posterior_dominance"] = {
            "enabled": True,
            "incumbent": None,
            "history": [{
                "posterior_dominance_initialization": "certified_only",
                "status": "uninitialized_no_certificate",
                "initial_certified_count": 0,
                "terminal_fallback_required": True,
                "target_oracle_used": False,
            }],
        }
    else:
        row["posterior_dominance"] = {
            "enabled": False,
            "incumbent": None,
            "history": [],
        }
    return row


def _gate_rows():
    return [
        _row(variant, seed)
        for seed in analyze.SENTINEL_SEEDS
        for variant in analyze.VARIANTS
    ]


def test_v48_mechanism_gate_separates_fix_from_promotion():
    result = analyze.summarize(_gate_rows())
    assert result["paired_initial_design_actions_and_target_responses"]
    assert all(result["source_contract"].values())
    assert all(result["preservation_contract"].values())
    assert result["certified_only_matches_no_preservation_v27"]
    assert result["certified_only_matches_no_preservation_v47"]
    assert result[
        "certified_only_strictly_improves_v47_risk_preservation"]
    assert result["certified_only_nonworse_than_promoted_v27"]
    assert result["mechanism_gate_passed"]
    assert not result["strictly_improves_promoted_v27"]
    assert result["promotion_eligible"] == []


def test_v48_rejects_unprotected_incumbent_without_certificate():
    rows = _gate_rows()
    row = next(value for value in rows if (
        value["gate_variant"] == "v48_certified_only"
        and value["seed"] == 3
    ))
    row["posterior_dominance"]["incumbent"] = [3, 2, 1]
    result = analyze.summarize(rows)
    assert not result["preservation_contract"]["v48_certified_only"]
    assert not result["mechanism_gate_passed"]
