import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "SC-OLH-KG"))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig
from core.gpr import ParametricGPR
from representation.exchangeable_mean import ExchangeableBoundaryMeanCoordinate
from representation.observable_exposure import ObservableStateExposure


SUBMIT_PATH = (
    REPO
    / "scripts/submit_scolhkg_mean_alignment_v43_predictive_hyperlaw_gate_scheduler.py"
)
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v43_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ANALYZE_PATH = (
    REPO / "SC-OLH-KG/performance/analyze_mean_alignment_v43_gate.py")
sys.path.insert(0, str(ANALYZE_PATH.parent))
ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "mean_v43_analyze", ANALYZE_PATH)
analyze = importlib.util.module_from_spec(ANALYZE_SPEC)
ANALYZE_SPEC.loader.exec_module(analyze)


def _exposure(means):
    means = np.asarray(means, dtype=float)
    scales = 0.05 + 0.1 * means
    summary = np.asarray([np.mean(means), np.std(means)])
    return ObservableStateExposure(
        means, scales, occupancy=summary, dynamics=summary)


class _Problem:
    tau = 1.0
    sigma_level = 0.04

    def __init__(self, exposure):
        self.exposure = exposure

    def int_bounds(self):
        return np.asarray([0]), np.asarray([0])

    def observable_state_exposure(self, _x):
        return self.exposure


def _fit(domains=("source_a", "source_b"), seed=4301):
    rng = np.random.default_rng(seed)
    coefficients = (
        np.asarray([1.8, -0.9, 0.4]),
        np.asarray([-1.2, 0.7, 1.5]),
    )
    exposures = []
    targets = []
    labels = []
    for index, domain in enumerate(domains):
        coefficient = coefficients[index % len(coefficients)]
        for _ in range(40):
            means = rng.uniform(0.05, 0.95, size=3)
            exposures.append(_exposure(means))
            targets.append(float(coefficient @ means - 0.15))
            labels.append(domain)
    coordinate = ExchangeableBoundaryMeanCoordinate().fit(
        exposures, np.asarray(targets), labels)
    return coordinate, exposures, targets, labels


def test_predictive_hyperlaw_uses_finite_source_new_task_correction():
    coordinate, _, _, _ = _fit()
    problem = _Problem(_exposure([0.3, 0.4, 0.8]))
    aggregate = coordinate.source_parametric_prior(problem)
    population = aggregate["shared_low_rank_prior"]
    predictive = aggregate["shared_low_rank_predictive_prior"]
    assert predictive is not None
    diagnostics = predictive["diagnostics"]
    concentration = diagnostics["weighted_source_concentration"]
    expected_multiplier = (1.0 + concentration) / (1.0 - concentration)
    np.testing.assert_allclose(
        diagnostics["finite_source_predictive_multiplier"],
        expected_multiplier,
    )
    assert diagnostics["target_task_law"] == (
        "shared_mean_plus_finite_source_predictive_discrepancy")
    assert expected_multiplier >= 1.0
    np.testing.assert_allclose(
        diagnostics["between_domain_predictive_covariance_trace"],
        expected_multiplier
        * diagnostics["between_domain_population_covariance_trace"],
        rtol=1e-10,
        atol=1e-12,
    )
    assert diagnostics["predictive_discrepancy_rank"] <= diagnostics[
        "domain_discrepancy_rank_upper_bound"]
    assert diagnostics["prior_covariance_trace"] >= population[
        "diagnostics"]["prior_covariance_trace"]
    assert np.min(np.linalg.eigvalsh(predictive["covariance"])) >= -1e-10
    assert not diagnostics["target_data_used"]
    assert not diagnostics["target_oracle_used"]


def test_predictive_hyperlaw_is_permutation_equivariant():
    first, exposures, targets, domains = _fit(seed=4302)
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
    first_prior = first.source_parametric_prior(problem)[
        "shared_low_rank_predictive_prior"]
    second_prior = second.source_parametric_prior(problem)[
        "shared_low_rank_predictive_prior"]
    np.testing.assert_allclose(first_prior["mean"], second_prior["mean"], atol=1e-8)
    np.testing.assert_allclose(
        first_prior["covariance"], second_prior["covariance"], atol=1e-8)


def test_predictive_hyperlaw_is_unavailable_with_one_source_domain():
    coordinate, _, _, _ = _fit(domains=("source_a",), seed=4303)
    problem = _Problem(_exposure([0.3, 0.4, 0.8]))
    aggregate = coordinate.source_parametric_prior(problem)
    assert aggregate["shared_low_rank_predictive_prior"] is None
    assert not aggregate["shared_low_rank_prior"]["diagnostics"][
        "finite_source_predictive_correction_available"]


class _PriorBasis:
    feature_dim = 2

    def __init__(self, prior):
        self.prior = copy.deepcopy(prior)

    def features(self, _x):
        return np.zeros(self.feature_dim)

    def source_parametric_prior(self):
        return copy.deepcopy(self.prior)


def test_algorithm_selects_predictive_prior_and_rejects_missing_law():
    predictive = {
        "mean": np.asarray([0.0, 0.1, -0.2]),
        "covariance": 0.6 * np.eye(3),
        "deviation_variance": 0.1,
        "diagnostics": {
            "target_task_law": (
                "shared_mean_plus_finite_source_predictive_discrepancy"),
            "target_oracle_used": False,
        },
    }
    legacy = {
        "mean": predictive["mean"],
        "covariance": np.eye(3),
        "deviation_variance": 0.1,
        "diagnostics": {"target_task_law": "single_gaussian_draw"},
        "shared_low_rank_predictive_prior": predictive,
    }
    model = ParametricGPR(d=1, basis_map=_PriorBasis(legacy))
    algorithm = object.__new__(SingleOLHKGAlgorithm)
    algorithm.config = SingleOLHKGConfig(
        source_constraint_mean_coefficient_prior=True,
        source_constraint_mean_hyperlaw_mode="shared_low_rank_predictive",
    )
    selected = algorithm._source_constraint_coefficient_prior(model, 1)
    np.testing.assert_allclose(selected["covariance"], 0.6 * np.eye(3))
    assert selected["diagnostics"]["finite_source_predictive_prior_selected"]
    assert selected["diagnostics"]["shared_low_rank_prior_selected"]
    assert not selected["diagnostics"]["target_oracle_used"]

    missing = copy.deepcopy(legacy)
    missing["shared_low_rank_predictive_prior"] = None
    model = ParametricGPR(d=1, basis_map=_PriorBasis(missing))
    with pytest.raises(RuntimeError, match="shared_low_rank_predictive"):
        algorithm._source_constraint_coefficient_prior(model, 1)


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v43",
        "rank": 4,
        "source_d": 50,
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
        "python": defaults.REMOTE_PYTHON,
        "cpu": 1,
        "ram_mb": 4096,
    })()


def test_v43_submitter_builds_eight_paired_cpu_only_sentinels(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 4 * 1 * 2
    assert all("QueueResourceControl" in spec["signature"] for spec in specs)
    assert all(spec["signature"].endswith(("/seed1", "/seed3")) for spec in specs)
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)
    for variant, profile in submit.VARIANTS.items():
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 2
        assert all(
            "--source-constraint-mean-hyperlaw-mode "
            f"{profile['hyperlaw_mode']}" in spec["cmd"]
            for spec in selected
        )


def _prior_diagnostics(variant):
    mode = analyze.HYPERLAW_MODES[variant]
    diagnostics = {
        "configured_hyperlaw_mode": mode,
        "finite_source_predictive_prior_selected": False,
        "target_oracle_used": False,
    }
    if mode == "single_gaussian_draw":
        diagnostics["target_task_law"] = "single_gaussian_draw"
    elif mode == "shared_low_rank_discrepancy":
        diagnostics["target_task_law"] = (
            "shared_mean_plus_low_rank_domain_discrepancy")
    else:
        diagnostics.update({
            "finite_source_predictive_prior_selected": True,
            "target_task_law": (
                "shared_mean_plus_finite_source_predictive_discrepancy"),
            "finite_source_predictive_correction": True,
            "weighted_source_concentration": 0.5,
            "finite_source_predictive_multiplier": 3.0,
            "between_domain_population_covariance_trace": 0.2,
            "between_domain_predictive_covariance_trace": 0.6,
            "predictive_discrepancy_rank": 1,
            "domain_discrepancy_rank_upper_bound": 1,
            "source_estimation_covariance_enters_shared_mean_only": True,
            "within_source_estimation_as_target_variation": False,
        })
    return diagnostics


def _row(variant, seed):
    confidence = analyze.CONFIDENCE_MODES[variant]
    feasible = (
        variant.startswith("v43_")
        or (variant == "v41_source_bayes" and seed == 1)
    )
    return {
        "gate_variant": variant,
        "heldout": "QueueResourceControl",
        "target_shared_shock_scale": 1.0,
        "seed": seed,
        "N": 20,
        "n0": 10,
        "decision_backend": "sobol_new",
        "source_archive_fingerprint": "archive",
        "target_design_fingerprint": f"design-{seed}",
        "online_action_sequence_fingerprint": f"actions-{seed}",
        "online_action_trace": [{
            "x_fingerprint": f"x-{seed}",
            "observed_response": [0.1, -0.1],
            "candidate_source": "sobol_continuation",
        }],
        "online_action_trace_target_oracle_used": False,
        "source_constraint_mean_hyperlaw_mode": (
            analyze.HYPERLAW_MODES[variant]),
        "source_constraint_mean_confidence_mode": confidence,
        "source_conditioned_confidence": {
            "mode": confidence,
            "status": "active",
            "target_oracle_used": False,
        },
        "gpr_numerics": [{}, {"source_parametric_prior": (
            _prior_diagnostics(variant))}],
        "true_feasible": feasible,
        "adaptive_loss": not feasible,
        "adaptive_improves_initial_best": False,
        "feasible_simple_regret": 0.01 if feasible else None,
        "boundary_raw_pool_truth_diagnostics": {
            "boundary_raw_pool_true_certified_count": 0,
            "boundary_raw_pool_false_certified_count": 0,
        },
        "certificate_outcome_audit": {
            "certified_true_feasible_count": 0,
            "false_certificate_count": 0,
        },
    }


def test_v43_analyzer_promotes_only_the_predictive_bayes_repair():
    rows = [
        _row(variant, seed)
        for seed in analyze.SENTINEL_SEEDS
        for variant in analyze.VARIANTS
    ]
    summary = analyze.summarize(rows)
    assert summary["paired_initial_design_archive_actions_and_responses"]
    assert all(summary["source_hyperlaw_contract"].values())
    assert summary["challenger_strictly_improves_v41_and_v42"]
    assert summary["promotion_eligible"] == [
        "v43_predictive_low_rank_bayes"]

    broken = copy.deepcopy(rows)
    predictive = next(
        row for row in broken
        if row["gate_variant"] == "v43_predictive_low_rank_bayes")
    predictive["gpr_numerics"][1]["source_parametric_prior"][
        "finite_source_predictive_multiplier"] = 1.0
    summary = analyze.summarize(broken)
    assert not summary["source_hyperlaw_contract"][
        "v43_predictive_low_rank_bayes"]
    assert not summary["promotion_eligible"]
