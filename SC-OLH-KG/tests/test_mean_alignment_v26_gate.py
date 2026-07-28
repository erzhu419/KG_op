import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v26_sequential_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v26_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v26_gate import (  # noqa: E402
    CHALLENGERS,
    HIERARCHICAL,
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
        "run_id": "mean-v26",
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


def test_v26_submitter_wires_exchangeable_sequential_gate(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 6 * 3 * 5
    for variant in CHALLENGERS:
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3 * 5
        assert all("--N 20 --n0 10" in spec["cmd"] for spec in selected)
        assert all(
            "--observable-mean-descriptor-mode exchangeable_equivariant"
            in spec["cmd"] for spec in selected)
        assert all(
            "--no-observable-mean-role-assignment-posterior" in spec["cmd"]
            for spec in selected)
        assert all(
            "--source-constraint-mean-structure-score-mode "
            "marginal_likelihood" in spec["cmd"] for spec in selected)
        expected = (
            "hierarchical_predictive_scale"
            if variant in HIERARCHICAL else "none")
        assert all(
            "--source-constraint-mean-misspecification-mode "
            f"{expected}" in spec["cmd"] for spec in selected)
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)


def _row(variant, domain, shock):
    source = (
        "v15_tanh_control"
        if variant == "v15_tanh_control"
        else "factorized_geometry_s005_t20")
    row = _v23_row(source, domain, shock)
    row["gate_variant"] = variant
    if variant not in CHALLENGERS:
        return row

    row["meta_observable_mean_descriptor_mode"] = (
        "exchangeable_equivariant")
    row["meta_observable_mean_feature_mode"] = "linear"
    row["mean_risk_coordinate_contract"] = {
        "exchangeable_channel_role_posterior": True,
        "target_channel_roles_learned_from_charged_data": True,
        "source_role_identity_transferred": False,
        "separate_mean_variance_heads": True,
        "shared_observable_exposure_input": True,
        "coordinate_definition_uses_target_labels": False,
        "source_oracle_aided": False,
    }
    row["meta_basis"]["1"] = {
        "status": "fit",
        "coordinate": "exchangeable_equivariant_boundary_linear",
        "permutation_equivariant": True,
        "source_role_identity_transferred": False,
        "target_oracle_used": False,
        "latent_transform": "identity",
        "latent_transform_diagnostics": {},
        "role_assignment_posterior": {"status": "disabled"},
    }
    numerics = row["gpr_numerics"][1]
    numerics["basis_posterior"] = {
        "source_prior_exchangeable": True,
        "source_channel_block_maximum_distance": 0.0,
        "posterior_channel_block_maximum_distance": 0.8,
        "target_channel_roles_differentiated": True,
        "source_role_identity_transferred": False,
        "target_oracle_used": False,
    }
    prior = numerics["source_parametric_prior"]
    source_components = [
        {
            "name": "source:a",
            "component_kind": "exchangeable_source_hyperprior",
            "permutation_equivariant": True,
            "source_role_identity_transferred": False,
            "target_oracle_used": False,
            "source_mean_misspecification_applied": True,
            "source_mean_misspecification_scale": 1.5,
            "misspecification_uncertainty_can_only_increase": True,
            "target_oracle_used_for_misspecification": False,
        },
        {
            "name": "source:b",
            "component_kind": "exchangeable_source_hyperprior",
            "permutation_equivariant": True,
            "source_role_identity_transferred": False,
            "target_oracle_used": False,
            "source_mean_misspecification_applied": True,
            "source_mean_misspecification_scale": 1.2,
            "misspecification_uncertainty_can_only_increase": True,
            "target_oracle_used_for_misspecification": False,
        },
    ]
    null = {
        "name": "target:null",
        "source_mean_misspecification_applied": False,
        "source_mean_misspecification_scale": 1.0,
        "target_oracle_used_for_misspecification": False,
    }
    if variant == "exchangeable_none":
        for component in source_components:
            component["source_mean_misspecification_applied"] = False
            component["source_mean_misspecification_scale"] = 1.0
    prior.update({
        "adaptation_mode": "sequential_target_evidence_mixture",
        "posterior_target_data_used": True,
        "target_observation_count": 20,
        "online_mixture_update_count": 10,
        "target_oracle_used": False,
        "component_deviation_diagnostics": source_components + [null],
    })
    return row


def test_v26_analyzer_requires_exchangeability_and_target_role_learning():
    rows = [
        _row(variant, domain, shock)
        for variant in VARIANTS
        for domain, shock in MEAN_SCENARIOS
    ]
    result = summarize(rows, expected_seeds=1)
    assert result["row_count"] == 6 * 3
    for variant in CHALLENGERS:
        checks = result["variant_checks"][variant]
        assert checks["exchangeable_target_linear_contract"]
        assert checks["source_mean_misspecification_contract"]
        assert checks["target_roles_differentiated_all_seeds"]
        assert checks["independent_variance_task_posterior_contract"]
        assert checks["variance_head_exactly_invariant_to_mean_coordinate"]

    broken = [dict(row) for row in rows]
    target = next(
        row for row in broken
        if row["gate_variant"] == "exchangeable_none")
    target["gpr_numerics"] = [
        dict(item) for item in target["gpr_numerics"]]
    target["gpr_numerics"][1]["basis_posterior"] = dict(
        target["gpr_numerics"][1]["basis_posterior"])
    target["gpr_numerics"][1]["basis_posterior"][
        "source_prior_exchangeable"] = False
    failed = summarize(broken, expected_seeds=1)
    assert not failed["variant_checks"]["exchangeable_none"][
        "exchangeable_target_linear_contract"]
