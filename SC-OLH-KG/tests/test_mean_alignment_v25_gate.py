import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v25_sequential_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v25_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v25_gate import (  # noqa: E402
    CHALLENGERS,
    CONTRAST_VARIANTS,
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
        "run_id": "mean-v25",
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


def test_v25_submitter_wires_boundary_role_gate(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 6 * 3 * 5
    for variant in CHALLENGERS:
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3 * 5
        assert all("--N 20 --n0 10" in spec["cmd"] for spec in selected)
        assert all(
            "--observable-mean-role-assignment-prior "
            "source_geometry_boundary" in spec["cmd"]
            for spec in selected)
        assert all(
            "--source-constraint-mean-structure-score-mode "
            "geometry_conditional" in spec["cmd"]
            for spec in selected)
        expected = "source_contrast" if variant in CONTRAST_VARIANTS else "none"
        assert all(
            "--source-constraint-mean-misspecification-mode "
            f"{expected}" in spec["cmd"] for spec in selected)
    strong = next(
        spec for spec in specs
        if "/boundary_geometry_s100/" in spec["signature"])
    weak = next(
        spec for spec in specs
        if "/boundary_geometry_s400/" in spec["signature"])
    assert (
        "--observable-mean-role-assignment-prior-temperature-scale 1.0"
        in strong["cmd"])
    assert (
        "--observable-mean-role-assignment-prior-temperature-scale 4.0"
        in weak["cmd"])
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)


def _row(variant, domain, shock):
    if variant == "v15_tanh_control":
        row = _v23_row(variant, domain, shock)
        row["gate_variant"] = variant
        return row
    row = _v23_row("factorized_geometry_s005_t20", domain, shock)
    row["gate_variant"] = variant
    if variant == "v23_factorized_none":
        return row

    role = row["meta_basis"]["1"]["role_assignment_posterior"]
    role["prior"] = "source_geometry_boundary"
    role["target_labels_used_to_define_assignments"] = True
    role["boundary_calibration"] = {
        "status": "fit",
        "mode": "source_geometry_boundary",
        "target_observation_count": 10,
        "target_labels_used": True,
        "target_oracle_used": False,
        "permutation_equivariant": True,
        "effective_assignment_count_after": 1.5,
        "maximum_prior_assignment": role["assignments"][0],
        "maximum_posterior_assignment": role["assignments"][-1],
    }
    prior = row["gpr_numerics"][1]["source_parametric_prior"]
    prior.update({
        "target_labels_used_for_group_masses": True,
        "target_role_assignment_target_labels_used_for_prior": True,
        "target_role_assignment_target_labels_used_for_online_update": False,
        "target_role_assignment_update_scope": (
            "charged_pilot_assignment_prior_then_frozen_"
            "conditional_expert_only"),
        "target_observation_count": 20,
        "online_mixture_update_count": 10,
    })
    if variant in CONTRAST_VARIANTS:
        diagnostics = []
        for name in prior["component_names"]:
            if str(name).startswith("source:"):
                diagnostics.append({
                    "name": name,
                    "source_mean_misspecification_applied": True,
                    "source_contrast_assignment_conditional": True,
                    "source_contrast_rank": 1,
                    "source_contrast_rank_bound": 1,
                    "source_contrast_group_component_count": 2,
                    "source_contrast_uses_target_data": False,
                    "target_oracle_used_for_misspecification": False,
                })
        prior["component_deviation_diagnostics"] = diagnostics
    return row


def test_v25_analyzer_requires_pilot_role_and_conditional_contrast_contracts():
    rows = [
        _row(variant, domain, shock)
        for variant in VARIANTS
        for domain, shock in MEAN_SCENARIOS
    ]
    result = summarize(rows, expected_seeds=1)
    assert result["row_count"] == 6 * 3
    for variant in CHALLENGERS:
        checks = result["variant_checks"][variant]
        assert checks["charged_pilot_factorized_role_contract"]
        assert checks["independent_variance_task_posterior_contract"]
        if variant in CONTRAST_VARIANTS:
            assert checks[
                "assignment_conditional_source_contrast_contract"]
