import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v23_offline_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v23_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v23_gate import (  # noqa: E402
    CHALLENGERS,
    MEAN_SCENARIOS,
    VARIANTS,
    summarize,
)
from tests.test_mean_alignment_v22_gate import _row as _v22_row  # noqa: E402


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v23",
        "rank": 4,
        "source_d": 50,
        "d": 1000,
        "N": 10,
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


def test_v23_submitter_wires_factorized_assignment_expert_posterior(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 5 * 3 * 5
    for variant in CHALLENGERS:
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3 * 5
        assert all(
            "--source-constraint-mean-structure-score-mode "
            "geometry_conditional" in spec["cmd"]
            for spec in selected)
        assert all(
            "--observable-mean-role-assignment-prior source_geometry"
            in spec["cmd"] for spec in selected)
        assert all(
            "--source-constraint-mean-evidence-temperature 2.0"
            in spec["cmd"] for spec in selected)
    strong = next(
        spec for spec in specs
        if "/factorized_geometry_s005_t20/" in spec["signature"])
    weak = next(
        spec for spec in specs
        if "/factorized_geometry_s100_t20/" in spec["signature"])
    assert (
        "--observable-mean-role-assignment-prior-temperature-scale 0.05"
        in strong["cmd"])
    assert (
        "--observable-mean-role-assignment-prior-temperature-scale 1.0"
        in weak["cmd"])
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)


def _row(variant, domain, shock):
    if variant == "v15_tanh_control":
        return _v22_row(variant, domain, shock)
    source_variant = (
        "geometry_loo_s025_t20"
        if variant != "v22_joint_geometry_loo"
        else "geometry_loo_s025_t20"
    )
    row = _v22_row(source_variant, domain, shock)
    row["gate_variant"] = variant
    if variant == "v22_joint_geometry_loo":
        return row

    role = row["meta_basis"]["1"]["role_assignment_posterior"]
    assignments = [str(value) for value in role["assignments"]]
    masses = {
        assignment: float(mass)
        for assignment, mass in zip(
            assignments, role["assignment_prior_weights"])
    }
    prior = row["gpr_numerics"][1]["source_parametric_prior"]
    names = [str(value) for value in prior["component_names"]]
    prior_weights = []
    posterior_weights = []
    for name in names:
        assignment = name.rsplit("|role_assignment=", 1)[1].split("|", 1)[0]
        group = [
            value for value in names
            if f"|role_assignment={assignment}" in value
        ]
        index = group.index(name)
        prior_weights.append(masses[assignment] * (0.4, 0.3, 0.3)[index])
        posterior_weights.append(
            masses[assignment] * (0.7, 0.1, 0.2)[index])
    hard = role["assignment_prior_diagnostics"]["hard_assignment"]
    prior.update({
        "adaptation_mode": (
            "sequential_assignment_prior_conditional_expert_mixture"),
        "structure_score_mode": "geometry_conditional",
        "structure_score_cross_fitted": False,
        "component_prior_weights": prior_weights,
        "component_posterior_weights": posterior_weights,
        "assignment_group_masses_fixed": True,
        "assignment_group_masses": masses,
        "target_role_assignment_posterior_mass": masses,
        "target_role_assignment_selected": hard,
        "target_role_assignment_target_labels_used_for_update": False,
        "target_role_assignment_conditional_expert_uses_target_labels": True,
        "target_role_assignment_update_scope": (
            "frozen_assignment_marginal_conditional_expert_only"),
        "posterior_target_data_used": True,
        "target_oracle_used": False,
        "target_oracle_used_for_group_masses": False,
    })
    audit = row["boundary_raw_pool_truth_diagnostics"][
        "boundary_raw_pool_role_assignment_oracle_expressivity"]
    audit["best_mae_assignment"] = hard
    audit["best_rank_assignment"] = hard
    return row


def test_v23_analyzer_requires_factorized_oracle_free_posterior():
    rows = [
        _row(variant, domain, shock)
        for variant in VARIANTS
        for domain, shock in MEAN_SCENARIOS
    ]
    result = summarize(rows, expected_seeds=1)
    assert result["row_count"] == 5 * 3
    for variant in CHALLENGERS:
        checks = result["variant_checks"][variant]
        assert checks["factorized_assignment_expert_posterior_contract"]
        assert checks["independent_variance_task_posterior_contract"]
        assert checks["variance_head_exactly_invariant_to_mean_structure"]
        assert checks["geometry_hard_assignment_retained_all_seeds"]
