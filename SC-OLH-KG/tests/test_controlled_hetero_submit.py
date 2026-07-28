import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from performance.analyze_controlled_heteroscedastic_gate import analyze


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/submit_scolhkg_controlled_hetero_gate_scheduler.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "controlled_hetero_submit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submission_matrix_shards_every_seed_and_limits_exact_row():
    module = _module()
    args = SimpleNamespace(
        deploy=Path("/tmp/deploy"),
        run_id="test_controlled",
        scenarios="smooth_boundary,shared_factor",
        variance_modes="pooled,orthogonal,factor,oracle",
        backends="sobol,risk_ts,joint_voi",
        seed_start=5,
        n_seeds=2,
        d=1000,
        N=20,
        n0=10,
        K1=24,
        posterior_pool_size=128,
        state_candidate_count=8,
        state_inverse_pool_size=256,
        exact_mc_samples=8,
        verification_primary_budget=80,
        verification_support_budget=96,
        verification_delta=0.05,
        terminal_safe_interior_scope="observed",
        light_cpu=1,
        light_ram_mb=4096,
        exact_cpu=12,
        exact_ram_mb=16384,
    )
    specs = module.build_specs(args)
    assert len(specs) == 2 * 9 * 2
    assert len({spec["signature"] for spec in specs}) == len(specs)
    assert all(spec["allowed_nodes"] == list(module.CPU_NODES) for spec in specs)
    exact = [spec for spec in specs if "/joint_voi/" in spec["signature"]]
    assert len(exact) == 2 * 1 * 2
    assert all("/factor/" in spec["signature"] for spec in exact)
    assert all(spec["cpu"] == 12 for spec in exact)
    light = [spec for spec in specs if spec not in exact]
    assert all(spec["cpu"] == 1 for spec in light)
    assert all(
        "--terminal-safe-interior-scope observed" in spec["cmd"]
        for spec in specs
    )


def test_submission_accepts_explicit_causal_variant_specs():
    module = _module()
    args = SimpleNamespace(
        deploy=Path("/tmp/deploy"),
        run_id="test_controlled_causal",
        scenarios="smooth_boundary,shared_factor",
        variance_modes="factor",
        backends="risk_ts",
        variant_specs=(
            "risk_ts:factor:objective_ranked,"
            "constrained_ts:factor:diverse,"
            "constrained_ts:factor:objective_ranked"
        ),
        seed_start=0,
        n_seeds=2,
        d=3,
        N=40,
        n0=10,
        K1=24,
        posterior_pool_size=128,
        state_candidate_count=8,
        state_inverse_pool_size=256,
        exact_mc_samples=8,
        verification_primary_budget=80,
        verification_support_budget=96,
        verification_delta=0.05,
        terminal_safe_interior_scope="observed",
        terminal_safe_interior_selection="diverse",
        light_cpu=1,
        light_ram_mb=4096,
        exact_cpu=12,
        exact_ram_mb=16384,
    )
    specs = module.build_specs(args)
    assert len(specs) == 2 * 3 * 2
    assert any(
        "/constrained_ts/objective_ranked/" in spec["signature"]
        for spec in specs
    )
    assert any(
        "--terminal-safe-interior-selection objective_ranked" in spec["cmd"]
        for spec in specs
    )


def test_analyzer_keeps_posterior_and_terminal_certificates_separate(tmp_path):
    manifest = {
        "run_id": "gate",
        "task_count": 1,
    }
    (tmp_path / "submission_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    result_dir = tmp_path / "smooth_boundary/factor/sobol/seed0000"
    result_dir.mkdir(parents=True)
    result = {
        "experiment": "controlled_heteroscedastic_optimum",
        "scenario": "smooth_boundary",
        "variance_mode": "factor",
        "backend": "sobol",
        "paired_deployment_effect": {
            "primary_true_feasible": True,
            "deployment_true_feasible": True,
            "primary_feasible_regret": 0.02,
            "deployment_feasible_regret": 0.01,
            "feasibility_rescue": False,
            "feasibility_loss": False,
            "strict_objective_win": True,
            "strict_objective_loss": False,
            "recommendation_changed": True,
        },
        "best_evaluated_truth": {
            "oracle_hit_at_0_01": True,
            "best_evaluated_feasible_regret": 0.01,
        },
        "posterior_certificate": {
            "posterior_certificate_vacuous": True,
        },
        "independent_terminal_certificate": {
            "certified": True,
            "false_certificate": False,
        },
        "variance_audit": {
            "log_variance_rmse": 0.2,
            "upper_variance_coverage": 0.95,
        },
        "wall_time_sec": 2.0,
    }
    (result_dir / "result.json").write_text(
        json.dumps(result), encoding="utf-8")
    summary = analyze(tmp_path)
    assert summary["complete"] is True
    group = summary["groups"][0]
    assert group["posterior_nonvacuous"] == 0
    assert group["terminal_certified"] == 1
    assert group["strict_objective_wins"] == 1
