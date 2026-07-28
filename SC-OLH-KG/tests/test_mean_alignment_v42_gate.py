import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "SC-OLH-KG"))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig
from core.gpr import ParametricGPR
from representation.exchangeable_mean import ExchangeableBoundaryMeanCoordinate
from representation.observable_exposure import ObservableStateExposure


SUBMIT_PATH = (
    REPO
    / "scripts/submit_scolhkg_mean_alignment_v42_hyperlaw_gate_scheduler.py"
)
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v42_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)


ANALYZE_PATH = (
    REPO / "SC-OLH-KG/performance/analyze_mean_alignment_v42_gate.py")
sys.path.insert(0, str(ANALYZE_PATH.parent))
ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "mean_v42_analyze", ANALYZE_PATH)
analyze = importlib.util.module_from_spec(ANALYZE_SPEC)
ANALYZE_SPEC.loader.exec_module(analyze)


def _exposure(means):
    means = np.asarray(means, dtype=float)
    scales = 0.05 + 0.1 * means
    return ObservableStateExposure(
        means,
        scales,
        occupancy=np.asarray([np.mean(means), np.std(means)]),
        dynamics=np.asarray([np.mean(means), np.std(means)]),
    )


class _Problem:
    tau = 1.0
    sigma_level = 0.04

    def __init__(self, exposure):
        self.exposure = exposure

    def int_bounds(self):
        return np.asarray([0]), np.asarray([0])

    def observable_state_exposure(self, _x):
        return self.exposure


def _fit(seed=4201):
    rng = np.random.default_rng(seed)
    exposures = []
    targets = []
    domains = []
    for domain, coefficient in (
        ("source_a", np.asarray([1.8, -0.9, 0.4])),
        ("source_b", np.asarray([-1.2, 0.7, 1.5])),
    ):
        for _ in range(40):
            means = rng.uniform(0.05, 0.95, size=3)
            exposures.append(_exposure(means))
            targets.append(float(coefficient @ means - 0.15))
            domains.append(domain)
    return ExchangeableBoundaryMeanCoordinate().fit(
        exposures, np.asarray(targets), domains), exposures, targets, domains


def test_shared_low_rank_hyperlaw_separates_estimation_and_domain_variation():
    coordinate, _, _, _ = _fit()
    problem = _Problem(_exposure([0.3, 0.4, 0.8]))
    aggregate = coordinate.source_parametric_prior(problem)
    prior = aggregate["shared_low_rank_prior"]
    diagnostics = prior["diagnostics"]
    assert diagnostics["target_task_law"] == (
        "shared_mean_plus_low_rank_domain_discrepancy")
    assert diagnostics["source_estimation_covariance_enters_shared_mean_only"]
    assert not diagnostics["within_source_estimation_as_target_variation"]
    assert diagnostics["channel_role_covariance_retained"]
    assert diagnostics["domain_discrepancy_rank"] <= 1
    assert diagnostics["domain_discrepancy_rank"] <= diagnostics[
        "domain_discrepancy_rank_upper_bound"]
    eigenvalues = np.linalg.eigvalsh(prior["covariance"])
    assert np.min(eigenvalues) >= -1e-10
    np.testing.assert_allclose(prior["mean"], aggregate["mean"])
    component_weights = np.asarray([
        component["prior_weight"]
        for component in coordinate.source_parametric_prior_components(problem)
    ])
    component_weights /= np.sum(component_weights)
    expected_shared_trace = 0.0
    for weight, row in zip(component_weights, coordinate.domain_rows):
        estimation, _ = coordinate._target_hierarchy_covariance_parts(
            row, problem)
        expected_shared_trace += float(weight) ** 2 * np.trace(estimation)
    np.testing.assert_allclose(
        diagnostics["shared_mean_covariance_trace"],
        expected_shared_trace,
        rtol=1e-10,
        atol=1e-12,
    )


def test_shared_low_rank_hyperlaw_is_permutation_equivariant():
    first, exposures, targets, domains = _fit(seed=4202)
    permutation = np.asarray([2, 0, 1])
    permuted = [
        ObservableStateExposure(
            exposure.channel_means[permutation],
            exposure.channel_scales[permutation],
            occupancy=exposure.occupancy,
            dynamics=exposure.dynamics,
        )
        for exposure in exposures
    ]
    second = ExchangeableBoundaryMeanCoordinate().fit(
        permuted, np.asarray(targets), domains)
    problem = _Problem(_exposure([0.3, 0.4, 0.8]))
    first_prior = first.source_parametric_prior(problem)["shared_low_rank_prior"]
    second_prior = second.source_parametric_prior(problem)["shared_low_rank_prior"]
    np.testing.assert_allclose(first_prior["mean"], second_prior["mean"], atol=1e-8)
    np.testing.assert_allclose(
        first_prior["covariance"], second_prior["covariance"], atol=1e-8)


class _PriorBasis:
    feature_dim = 2

    def __init__(self, prior):
        self.prior = copy.deepcopy(prior)

    def features(self, _x):
        return np.zeros(self.feature_dim)

    def source_parametric_prior(self):
        return copy.deepcopy(self.prior)


def test_algorithm_selects_shared_low_rank_prior_without_oracle():
    legacy = {
        "mean": np.asarray([0.0, 0.1, -0.2]),
        "covariance": np.eye(3),
        "deviation_variance": 0.1,
        "diagnostics": {"target_task_law": "single_gaussian_draw"},
        "shared_low_rank_prior": {
            "mean": np.asarray([0.0, 0.1, -0.2]),
            "covariance": 0.2 * np.eye(3),
            "deviation_variance": 0.1,
            "diagnostics": {
                "target_task_law": (
                    "shared_mean_plus_low_rank_domain_discrepancy"),
                "target_oracle_used": False,
            },
        },
    }
    model = ParametricGPR(d=1, basis_map=_PriorBasis(legacy))
    algorithm = object.__new__(SingleOLHKGAlgorithm)
    algorithm.config = SingleOLHKGConfig(
        source_constraint_mean_coefficient_prior=True,
        source_constraint_mean_hyperlaw_mode="shared_low_rank_discrepancy",
    )
    selected = algorithm._source_constraint_coefficient_prior(model, 1)
    np.testing.assert_allclose(selected["covariance"], 0.2 * np.eye(3))
    assert selected["diagnostics"]["shared_low_rank_prior_selected"]
    assert selected["diagnostics"]["configured_hyperlaw_mode"] == (
        "shared_low_rank_discrepancy")
    assert not selected["diagnostics"]["target_oracle_used"]


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v42",
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


def test_v42_submitter_builds_paired_cpu_only_gate(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 4 * 3 * 5
    for variant, profile in submit.VARIANTS.items():
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3 * 5
        assert all(
            "--source-constraint-mean-hyperlaw-mode "
            f"{profile['hyperlaw_mode']}" in spec["cmd"]
            for spec in selected
        )
        assert all("--decision-backend sobol_new" in spec["cmd"]
                   for spec in selected)
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)


def test_v42_analyzer_rejects_rank_violation_and_oracle_use():
    row = {
        "gate_variant": "v42_shared_low_rank_model",
        "source_constraint_mean_hyperlaw_mode": (
            "shared_low_rank_discrepancy"),
        "source_constraint_mean_confidence_mode": "model",
        "source_conditioned_confidence": {
            "mode": "model",
            "status": "active",
            "target_oracle_used": False,
        },
        "gpr_numerics": [{}, {"source_parametric_prior": {
            "configured_hyperlaw_mode": "shared_low_rank_discrepancy",
            "shared_low_rank_prior_selected": True,
            "target_task_law": (
                "shared_mean_plus_low_rank_domain_discrepancy"),
            "source_estimation_covariance_enters_shared_mean_only": True,
            "within_source_estimation_as_target_variation": False,
            "channel_role_covariance_retained": True,
            "domain_discrepancy_rank": 1,
            "domain_discrepancy_rank_upper_bound": 1,
            "shared_mean_covariance_trace": 0.2,
            "channel_role_covariance_trace": 0.1,
            "between_domain_covariance_trace": 0.3,
            "prior_covariance_trace": 0.6,
            "target_oracle_used": False,
        }}],
    }
    assert analyze._hyperlaw_contract(
        [row], "v42_shared_low_rank_model")
    assert analyze._confidence_contract(
        [row], "v42_shared_low_rank_model")
    row["gpr_numerics"][1]["source_parametric_prior"][
        "domain_discrepancy_rank"] = 2
    assert not analyze._hyperlaw_contract(
        [row], "v42_shared_low_rank_model")
    row["gpr_numerics"][1]["source_parametric_prior"][
        "domain_discrepancy_rank"] = 1
    row["gpr_numerics"][1]["source_parametric_prior"][
        "target_oracle_used"] = True
    assert not analyze._hyperlaw_contract(
        [row], "v42_shared_low_rank_model")
