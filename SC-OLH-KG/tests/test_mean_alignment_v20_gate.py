import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v20_offline_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v20_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v20_gate import (  # noqa: E402
    CHALLENGERS,
    MEAN_SCENARIOS,
    VARIANTS,
    summarize,
)
from tests.test_mean_alignment_v18_gate import _row as _v18_row  # noqa: E402


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v20",
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


def test_v20_submitter_wires_finite_role_assignment_posterior(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 5 * 3 * 5
    plain = next(
        spec for spec in specs
        if "/role_assignment_plain/" in spec["signature"])
    hierarchical = next(
        spec for spec in specs
        if "/role_assignment_hierarchical/" in spec["signature"])
    assert "--observable-mean-role-assignment-posterior" in plain["cmd"]
    assert "--observable-mean-role-assignment-prior uniform" in plain["cmd"]
    assert "--observable-mean-target-residual-rank 0" in plain["cmd"]
    assert "--no-source-constraint-mean-residual-rank-posterior" in (
        plain["cmd"])
    assert (
        "--source-constraint-mean-misspecification-mode "
        "hierarchical_predictive_scale"
    ) in hierarchical["cmd"]
    assert "--task-variance-posterior-mode replication_only" in plain["cmd"]
    assert "--hvd-source-task-weight-mode independent" in plain["cmd"]
    assert "--hvd-singleton-evidence-mode source_prior" in plain["cmd"]
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)


ASSIGNMENTS = ("0-1", "0-2", "1-0", "1-2", "2-0", "2-1")


def _row(variant, domain, shock):
    row = _v18_row(variant, domain, shock)
    row["gate_variant"] = variant
    if variant not in CHALLENGERS:
        return row

    role_diagnostics = {
        "status": "fit",
        "posterior": "finite_channel_role_assignment",
        "prior": "uniform",
        "assignment_count": len(ASSIGNMENTS),
        "channel_count": 2,
        "role_count": 3,
        "assignments": list(ASSIGNMENTS),
        "permutation_equivariant": True,
        "active_feature_dim_per_atom": 4,
        "total_stored_feature_dim": 24,
        "target_labels_used_to_define_assignments": False,
        "target_oracle_used_to_define_assignments": False,
    }
    row["meta_basis"]["1"]["role_assignment_posterior"] = role_diagnostics

    names = []
    posterior = []
    assignment_mass = (0.50, 0.10, 0.10, 0.10, 0.10, 0.10)
    for assignment, mass in zip(ASSIGNMENTS, assignment_mass):
        names.extend([
            f"source:a|role_assignment={assignment}",
            f"source:b|role_assignment={assignment}",
            f"target:null|role_assignment={assignment}",
        ])
        posterior.extend([mass / 3.0] * 3)
    prior = row["gpr_numerics"][1]["source_parametric_prior"]
    prior.update({
        "component_names": names,
        "component_prior_weights": [1.0 / len(names)] * len(names),
        "component_posterior_weights": posterior,
        "target_observation_count": 10,
        "target_role_assignment_posterior_active": True,
        "target_role_assignment_posterior_mass": dict(zip(
            ASSIGNMENTS, assignment_mass)),
        "target_role_assignment_conditional_source_mass": dict(zip(
            ASSIGNMENTS, assignment_mass)),
        "target_role_assignment_conditional_null_mass": dict(zip(
            ASSIGNMENTS, assignment_mass)),
        "target_role_assignment_structured_source_mass": 2.0 / 3.0,
        "target_role_assignment_structured_null_mass": 1.0 / 3.0,
        "target_role_assignment_selected": "0-1",
        "target_role_assignment_target_labels_used_for_update": True,
        "target_role_assignment_target_oracle_used": False,
        "target_role_assignment_permutation_equivariant": True,
    })
    row["boundary_raw_pool_truth_diagnostics"].update({
        "boundary_raw_pool_role_assignment_oracle_expressivity": {
            "status": "audited",
            "assignment_count": len(ASSIGNMENTS),
            "best_median_abs_error": 0.1,
            "best_mae_assignment": "0-1",
            "best_rank_correlation": 0.8,
            "best_rank_assignment": "0-1",
            "post_run_only": True,
            "target_oracle_used": True,
            "target_oracle_used_for_decision": False,
        },
    })
    return row


def test_v20_analyzer_requires_oracle_free_low_rank_assignment_adaptation():
    rows = [
        _row(variant, domain, shock)
        for variant in VARIANTS
        for domain, shock in MEAN_SCENARIOS
    ]
    result = summarize(rows, expected_seeds=1)
    assert result["row_count"] == 5 * 3
    assert set(result["sequential_gate_eligible"]) == set(CHALLENGERS)
    assert all(
        all(result["variant_checks"][variant].values())
        for variant in CHALLENGERS
    )
