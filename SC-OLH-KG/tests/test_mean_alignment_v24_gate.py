import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v24_sequential_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v24_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v24_gate import (  # noqa: E402
    CHALLENGERS,
    MEAN_SCENARIOS,
    VARIANTS,
    summarize,
)
from tests.test_mean_alignment_v23_gate import _row as _v23_row  # noqa: E402


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v24",
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
        "contrast_scale": 1.0,
        "null_geometry_ridge": 1e-3,
        "python": defaults.REMOTE_PYTHON,
        "cpu": 1,
        "ram_mb": 4096,
    })()


def test_v24_submitter_wires_sequential_hierarchical_gate(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 4 * 3 * 5
    for variant in CHALLENGERS:
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3 * 5
        assert all("--N 20 --n0 10" in spec["cmd"] for spec in selected)
        assert all(
            "--source-constraint-mean-structure-score-mode "
            "geometry_conditional" in spec["cmd"]
            for spec in selected)
        assert all(
            "--source-constraint-mean-misspecification-mode "
            "hierarchical_predictive_scale" in spec["cmd"]
            for spec in selected)
        assert all(
            "--observable-mean-role-assignment-prior-temperature-scale 0.05"
            in spec["cmd"] for spec in selected)
    df4 = next(
        spec for spec in specs
        if "/factorized_hierarchical_df4/" in spec["signature"])
    df16 = next(
        spec for spec in specs
        if "/factorized_hierarchical_df16/" in spec["signature"])
    assert (
        "--source-constraint-mean-misspecification-prior-df 4.0"
        in df4["cmd"])
    assert (
        "--source-constraint-mean-misspecification-prior-df 16.0"
        in df16["cmd"])
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)


def _row(variant, domain, shock):
    if variant == "v15_tanh_control":
        row = _v23_row(variant, domain, shock)
        row["gate_variant"] = variant
        return row
    if variant == "v23_factorized_none":
        row = _v23_row(
            "factorized_geometry_s005_t20", domain, shock)
        row["gate_variant"] = variant
        return row

    row = _v23_row("factorized_geometry_s005_t20", domain, shock)
    row["gate_variant"] = variant
    prior = row["gpr_numerics"][1]["source_parametric_prior"]
    components = []
    for name in prior["component_names"]:
        source = not str(name).startswith("target:")
        components.append({
            "name": str(name),
            "source_mean_misspecification_applied": source,
            "source_mean_misspecification_scale": 1.5 if source else 1.0,
            "misspecification_uncertainty_can_only_increase": source,
            "target_oracle_used_for_misspecification": False,
        })
    prior.update({
        "adaptation_mode": (
            "sequential_assignment_prior_conditional_hierarchical_"
            "expert_mixture"),
        "source_mean_misspecification_mode": (
            "hierarchical_predictive_scale"),
        "source_mean_misspecification_online": True,
        "source_mean_misspecification_refit_from_frozen_law": True,
        "component_deviation_diagnostics": components,
        "target_observation_count": 20,
        "online_mixture_update_count": 10,
        "source_mean_misspecification_scale_trajectory": [
            {"target_observation_count": 10 + index}
            for index in range(11)
        ],
    })
    return row


def test_v24_analyzer_requires_hierarchical_factorization_contract():
    rows = [
        _row(variant, domain, shock)
        for variant in VARIANTS
        for domain, shock in MEAN_SCENARIOS
    ]
    result = summarize(rows, expected_seeds=1)
    assert result["row_count"] == 4 * 3
    for variant in CHALLENGERS:
        checks = result["variant_checks"][variant]
        assert checks["factorized_hierarchical_misspecification_contract"]
        assert checks["independent_variance_task_posterior_contract"]
        assert checks["variance_head_exactly_invariant_to_mean_structure"]
        assert checks["geometry_hard_assignment_retained_all_seeds"]
