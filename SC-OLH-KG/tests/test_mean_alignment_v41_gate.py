import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm, t


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "SC-OLH-KG"))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig
from core.gpr import ParametricGPR


ANALYZE_PATH = (
    REPO / "SC-OLH-KG/performance/analyze_mean_alignment_v41_gate.py")
sys.path.insert(0, str(ANALYZE_PATH.parent))
ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "mean_v41_analyze", ANALYZE_PATH)
analyze = importlib.util.module_from_spec(ANALYZE_SPEC)
ANALYZE_SPEC.loader.exec_module(analyze)

SUBMIT_PATH = (
    REPO
    / "scripts/submit_scolhkg_mean_alignment_v41_source_confidence_gate_scheduler.py"
)
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v41_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v41",
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
        "misspecification_delta": 0.05,
        "confidence_delta": 0.05,
        "contrast_scale": 1.0,
        "null_geometry_ridge": 1e-3,
        "python": defaults.REMOTE_PYTHON,
        "cpu": 1,
        "ram_mb": 4096,
    })()


def _confidence_model(residual_floor=0.0025):
    model = ParametricGPR(d=1, lambda_i=0.03, prior_var=1.0)
    prior = np.diag([0.8, 0.5, 0.2])
    posterior = np.diag([0.08, 0.04, 0.02])
    model.set_source_conditioned_confidence(
        prior,
        posterior,
        residual_floor,
        source_domain_count=2,
        target_count=7,
        prior_df=4.0,
        delta=0.05,
    )
    return model, prior, posterior


def test_source_conditioned_equivalent_variance_matches_radius_identity():
    model, prior, posterior = _confidence_model()
    points = [(0,), (25,), (100,)]
    beta_g = 3.0
    equivalent, diagnostics = (
        model.source_conditioned_certification_var_many(
            points,
            beta_g=beta_g,
            mode="source_self_normalized",
            delta=0.05,
        )
    )
    information_gain = 0.5 * (
        model._stable_logdet_psd(prior)
        - model._stable_logdet_psd(posterior)
    )
    effective_beta = max(
        beta_g,
        2.0 * (np.log(1.0 / 0.05) + information_gain),
    )
    coefficient_var = np.einsum(
        "ij,jk,ik->i",
        model.basis_matrix(points),
        posterior,
        model.basis_matrix(points),
    )
    transfer_df = 4.0 + 2.0 + 7.0 - 1.0
    transfer_radius = t.ppf(0.95, transfer_df) * np.sqrt(0.0025)
    expected_radius = (
        np.sqrt(effective_beta * coefficient_var) + transfer_radius)
    np.testing.assert_allclose(
        np.sqrt(beta_g * equivalent), expected_radius, rtol=1e-12, atol=1e-12)
    assert diagnostics["solution_specific_deviation_double_counted"] is False
    assert diagnostics["transfer_guard_can_only_increase_radius"] is True
    assert diagnostics["target_oracle_used"] is False


def test_self_normalized_radius_dominates_bayes_and_guard_is_monotone():
    model, _, _ = _confidence_model(residual_floor=0.0)
    points = [(0,), (100,)]
    bayes, bayes_diag = model.source_conditioned_certification_var_many(
        points, beta_g=1.0, mode="source_bayes")
    sequence, sequence_diag = model.source_conditioned_certification_var_many(
        points, beta_g=1.0, mode="source_self_normalized")
    assert sequence_diag["effective_beta"] >= bayes_diag["effective_beta"]
    assert np.all(sequence >= bayes - 1e-12)

    guarded, _, _ = _confidence_model(residual_floor=0.04)
    guarded_var, guarded_diag = (
        guarded.source_conditioned_certification_var_many(
            points, beta_g=1.0, mode="source_self_normalized")
    )
    assert guarded_diag["transfer_radius"] > 0.0
    assert np.all(guarded_var > sequence)


def test_algorithm_routes_certification_through_source_confidence():
    model, _, _ = _confidence_model()
    algorithm = object.__new__(SingleOLHKGAlgorithm)
    algorithm.config = SingleOLHKGConfig(
        beta_g=2.0,
        source_constraint_mean_confidence_mode="source_bayes",
        source_constraint_mean_confidence_delta=0.05,
    )
    algorithm.gpr = [None, model]
    algorithm._last_source_conditioned_confidence_diagnostics = None
    points = [(0,), (50,)]
    actual = algorithm._constraint_certification_epistemic_many(model, points)
    expected, diagnostics = model.source_conditioned_certification_var_many(
        points, beta_g=2.0, mode="source_bayes", delta=0.05)
    np.testing.assert_allclose(actual, expected)
    assert algorithm._last_source_conditioned_confidence_diagnostics == diagnostics


def test_hierarchical_online_refit_rebuilds_source_confidence():
    parent = ParametricGPR(d=1, lambda_i=0.02, prior_var=1.0)
    component = ParametricGPR(d=1, lambda_i=0.02, prior_var=1.0)
    prior = {
        "name": "source:aggregate",
        "mean": np.zeros(parent.p, dtype=float),
        "covariance": 0.1 * np.eye(parent.p),
        "deviation_variance": 0.02,
        "prior_weight": 1.0,
        "diagnostics": {},
    }
    parent.set_hierarchical_misspecification_posterior(
        [component],
        [prior],
        [1.0],
        [(0,), (25,), (50,), (75,), (100,)],
        [-0.5, -0.2, 0.1, 0.4, 0.7],
        [0.01] * 5,
        diagnostics={
            "component_names": ["source:aggregate"],
            "source_domain_count": 2,
            "single_aggregate_hyperlaw": True,
            "online_mixture_update_count": 0,
        },
        prior_df=4.0,
        misspecification_mode=(
            "predictive_scale_sandwich_hc3_confidence"),
        misspecification_delta=0.05,
    )
    initial = parent._source_conditioned_confidence
    assert initial is not None
    assert initial["target_count"] == 5
    parent.update((40,), 0.05, 0.01)
    updated = parent._source_conditioned_confidence
    assert updated is not None
    assert updated["target_count"] == 6
    assert parent.source_parametric_prior_diagnostics[
        "source_conditioned_confidence"
    ]["source_prior_frozen_before_target"]

    algorithm = object.__new__(SingleOLHKGAlgorithm)
    clone = algorithm._clone_gpr_for_exact_kg(parent)
    clone._source_conditioned_confidence["prior_covariance"][0, 0] += 1.0
    assert not np.isclose(
        clone._source_conditioned_confidence["prior_covariance"][0, 0],
        parent._source_conditioned_confidence["prior_covariance"][0, 0],
    )


def test_v41_submitter_builds_paired_oracle_free_confidence_gate(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 3 * 3 * 5
    for variant, profile in submit.VARIANTS.items():
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3 * 5
        assert all(
            "--source-constraint-mean-confidence-mode "
            f"{profile['confidence_mode']}" in spec["cmd"]
            for spec in selected
        )
        assert all(
            "--source-constraint-mean-confidence-delta 0.05" in spec["cmd"]
            for spec in selected
        )
        assert all("--decision-backend sobol_new" in spec["cmd"]
                   for spec in selected)
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)


def test_v41_contract_rejects_double_counting_and_oracle_use():
    row = {
        "gate_variant": "v41_source_self_normalized",
        "source_constraint_mean_confidence_mode": "source_self_normalized",
        "source_conditioned_confidence": {
            "mode": "source_self_normalized",
            "status": "active",
            "effective_beta": 8.0,
            "information_gain": 1.0,
            "effective_rank": 3.0,
            "transfer_residual_floor": 0.01,
            "transfer_radius": 0.1,
            "coefficient_radius_median": 0.2,
            "total_radius_median": 0.3,
            "equivalent_to_model_variance_median_ratio": 0.5,
            "source_domain_count": 2,
            "target_count": 20,
            "solution_specific_deviation_double_counted": False,
            "transfer_guard_can_only_increase_radius": True,
            "target_oracle_used": False,
        },
    }
    assert analyze._confidence_contract(
        [row], "v41_source_self_normalized")
    row["source_conditioned_confidence"][
        "solution_specific_deviation_double_counted"] = True
    assert not analyze._confidence_contract(
        [row], "v41_source_self_normalized")
    row["source_conditioned_confidence"][
        "solution_specific_deviation_double_counted"] = False
    row["source_conditioned_confidence"]["target_oracle_used"] = True
    assert not analyze._confidence_contract(
        [row], "v41_source_self_normalized")
