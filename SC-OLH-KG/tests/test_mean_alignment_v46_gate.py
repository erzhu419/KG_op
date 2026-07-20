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
from problems.rzdt import make_problem
from problems.single_objective import ScalarizedProblem
from representation.exchangeable_mean import ExchangeableBoundaryMeanCoordinate
from representation.meta_prior import LearnedMetaPrior, MetaPriorProblemAdapter
from representation.observable_exposure import ObservableStateExposure


SUBMIT_PATH = (
    REPO
    / "scripts/submit_scolhkg_mean_alignment_v46_grouped_hyperlaw_gate_scheduler.py"
)
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v46_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ANALYZE_PATH = (
    REPO / "SC-OLH-KG/performance/analyze_mean_alignment_v46_gate.py")
sys.path.insert(0, str(ANALYZE_PATH.parent))
ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "mean_v46_analyze", ANALYZE_PATH)
analyze = importlib.util.module_from_spec(ANALYZE_SPEC)
ANALYZE_SPEC.loader.exec_module(analyze)


class _Problem:
    tau = 1.0
    sigma_level = 0.04

    def int_bounds(self):
        return np.asarray([0]), np.asarray([0])

    def observable_state_exposure(self, _x):
        return ObservableStateExposure(
            np.asarray([0.2, 0.4, 0.7]),
            np.asarray([0.1, 0.1, 0.1]),
            occupancy=np.asarray([0.3]),
            dynamics=np.asarray([0.2]),
        )


def _hierarchy_inputs(base_offset=None):
    offset = (
        np.zeros(4, dtype=float)
        if base_offset is None else np.asarray(base_offset, dtype=float)
    )
    means = (
        np.asarray([0.2, 0.1, -0.3, 0.4]) + offset,
        np.asarray([0.4, -0.1, -0.2, 0.5]) + offset,
        np.asarray([-0.2, 0.6, 0.1, -0.1]),
        np.asarray([-0.1, 0.3, 0.4, 0.2]),
    )
    domains = (
        "source_a",
        "source_a#episode1",
        "source_b",
        "source_b#episode1",
    )
    components = [
        {
            "domain": domain,
            "mean": mean,
            "deviation_variance": 0.02 + 0.005 * index,
        }
        for index, (domain, mean) in enumerate(zip(domains, means))
    ]
    estimation = [
        (0.02 + 0.001 * index) * np.eye(4)
        for index in range(4)
    ]
    role = [0.01 * np.eye(4) for _ in range(4)]
    return components, estimation, role


def _grouped_prior(base_offset=None, deconvolve=False):
    coordinate = ExchangeableBoundaryMeanCoordinate()
    return coordinate._grouped_source_task_hyperlaws(
        _Problem(), *_hierarchy_inputs(base_offset),
        deconvolve=deconvolve)


def test_grouped_task_hyperlaw_separates_within_base_contrasts():
    population, predictive = _grouped_prior()
    assert population is not None and predictive is not None
    diagnostics = population["diagnostics"]
    assert diagnostics["source_base_domain_count"] == 2
    assert diagnostics["source_episode_counts_by_base_domain"] == {
        "source_a": 2,
        "source_b": 2,
    }
    assert diagnostics["between_base_discrepancy_rank"] <= 1
    assert diagnostics["within_base_task_discrepancy_rank"] <= 2
    assert diagnostics["combined_discrepancy_rank"] <= 3
    assert diagnostics["source_estimation_covariance_enters_shared_mean_only"]
    assert not diagnostics["within_source_estimation_as_target_variation"]
    assert np.min(np.linalg.eigvalsh(population["covariance"])) >= -1e-12
    assert np.min(np.linalg.eigvalsh(predictive["covariance"])) >= -1e-12
    assert predictive["diagnostics"]["prior_covariance_trace"] >= diagnostics[
        "prior_covariance_trace"]
    assert not diagnostics["target_data_used"]
    assert not diagnostics["target_oracle_used"]


def test_grouped_within_task_covariance_is_invariant_to_base_offset():
    first, _ = _grouped_prior()
    shifted, _ = _grouped_prior([0.8, -0.4, 0.2, 0.6])
    np.testing.assert_allclose(
        first["diagnostics"]["within_base_task_covariance_trace"],
        shifted["diagnostics"]["within_base_task_covariance_trace"],
        atol=1e-12,
    )
    assert not np.allclose(first["mean"], shifted["mean"])


def test_grouped_task_hyperlaw_requires_repeated_source_episodes():
    coordinate = ExchangeableBoundaryMeanCoordinate()
    components, estimation, role = _hierarchy_inputs()
    population, predictive = coordinate._grouped_source_task_hyperlaws(
        _Problem(),
        [components[0], components[2]],
        [estimation[0], estimation[2]],
        [role[0], role[2]],
    )
    assert population is None
    assert predictive is None


def test_random_effects_deconvolution_removes_fit_noise_from_task_variation():
    raw, _ = _grouped_prior()
    corrected, predictive = _grouped_prior(deconvolve=True)
    diagnostics = corrected["diagnostics"]
    assert diagnostics["random_effects_deconvolution"]
    assert diagnostics["source_estimation_covariance_used_for_deconvolution"]
    for prefix in ("channel_role", "between_base", "within_base"):
        observed = diagnostics[f"{prefix}_observed_covariance_trace"]
        noise = diagnostics[f"{prefix}_estimation_noise_trace"]
        final_key = {
            "channel_role": "channel_role_covariance_trace",
            "between_base": "between_base_domain_covariance_trace",
            "within_base": "within_base_task_covariance_trace",
        }[prefix]
        assert observed >= 0.0
        assert noise >= 0.0
        assert diagnostics[final_key] <= observed + 1e-12
    assert diagnostics["prior_covariance_trace"] <= raw[
        "diagnostics"]["prior_covariance_trace"] + 1e-12
    assert predictive["diagnostics"]["prior_covariance_trace"] >= diagnostics[
        "prior_covariance_trace"]
    np.testing.assert_allclose(corrected["mean"], raw["mean"])


def test_fitted_meta_prior_exposes_grouped_task_laws_without_target_truth():
    def problem(name, **kwargs):
        return ScalarizedProblem(
            make_problem(name, d=8, L=100, sigma=0.04, **kwargs))

    sources = []
    for name in ("InventorySupplyChain", "FactorShockStatePolicyRZDT1"):
        sources.extend([
            (name, problem(name)),
            (
                f"{name}#episode1",
                problem(
                    name,
                    task_geometry_shift=(0.04, -0.03, 0.02),
                    task_geometry_radius_scale=1.1,
                ),
            ),
        ])
    prior = LearnedMetaPrior(
        local_dim=3,
        shared_dim=2,
        component_stage="coordinate",
        observable_mean_coordinate=True,
        observable_mean_mode="boundary_aligned",
        observable_mean_training_target="chance_margin",
        observable_mean_input_mode="observable_state_exposure",
        observable_mean_descriptor_mode="exchangeable_equivariant",
        observable_mean_feature_mode="linear",
        observable_variance_input_mode="observable_state_exposure",
        source_observation_mode="replicated",
        source_observation_replicates=2,
        source_design_mode="universal_mixture",
        source_universal_fraction=1.0,
        teacher_records_per_domain=0,
        seed=4601,
    ).fit_from_source_problems(
        sources,
        n_records_per_domain=8,
        rng=np.random.default_rng(4601),
    )
    target = MetaPriorProblemAdapter(problem("QueueResourceControl"), prior)
    basis = target.gpr_basis_map(output_index=1)
    source_prior = basis.source_parametric_prior()
    for key in (
        "grouped_task_prior",
        "grouped_task_predictive_prior",
        "grouped_task_deconvolved_prior",
        "grouped_task_deconvolved_predictive_prior",
    ):
        grouped = source_prior[key]
        assert grouped is not None
        assert grouped["mean"].shape == source_prior["mean"].shape
        assert grouped["covariance"].shape == source_prior["covariance"].shape
        assert np.min(np.linalg.eigvalsh(grouped["covariance"])) >= -1e-10
        assert not grouped["diagnostics"]["target_data_used"]
        assert not grouped["diagnostics"]["target_oracle_used"]


class _PriorBasis:
    feature_dim = 3

    def __init__(self, prior):
        self.prior = copy.deepcopy(prior)

    def features(self, _x):
        return np.zeros(self.feature_dim)

    def source_parametric_prior(self):
        return copy.deepcopy(self.prior)


@pytest.mark.parametrize(
    ("mode", "key", "predictive"),
    [
        ("grouped_task_discrepancy", "grouped_task_prior", False),
        ("grouped_task_predictive", "grouped_task_predictive_prior", True),
        (
            "grouped_task_deconvolved",
            "grouped_task_deconvolved_prior",
            False,
        ),
        (
            "grouped_task_deconvolved_predictive",
            "grouped_task_deconvolved_predictive_prior",
            True,
        ),
    ],
)
def test_algorithm_selects_grouped_task_hyperlaw(mode, key, predictive):
    nested = {
        "mean": np.zeros(4),
        "covariance": 0.3 * np.eye(4),
        "deviation_variance": 0.02,
        "diagnostics": {"target_oracle_used": False},
    }
    prior = {
        "mean": np.zeros(4),
        "covariance": np.eye(4),
        "deviation_variance": 0.02,
        key: nested,
    }
    model = ParametricGPR(d=1, basis_map=_PriorBasis(prior))
    algorithm = object.__new__(SingleOLHKGAlgorithm)
    algorithm.config = SingleOLHKGConfig(
        source_constraint_mean_coefficient_prior=True,
        source_constraint_mean_hyperlaw_mode=mode,
    )
    selected = algorithm._source_constraint_coefficient_prior(model, 1)
    np.testing.assert_allclose(selected["covariance"], 0.3 * np.eye(4))
    assert selected["diagnostics"]["grouped_task_prior_selected"]
    assert (
        selected["diagnostics"]["finite_source_predictive_prior_selected"]
        is predictive
    )
    missing = copy.deepcopy(prior)
    missing[key] = None
    model = ParametricGPR(d=1, basis_map=_PriorBasis(missing))
    with pytest.raises(RuntimeError, match=mode):
        algorithm._source_constraint_coefficient_prior(model, 1)


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
        "run_id": "mean-v46",
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


def test_v46_submitter_builds_eight_paired_cpu_sentinels(tmp_path):
    specs = submit.build_specs(_scheduler_args(tmp_path))
    assert len(specs) == 8
    assert all(spec["cpu"] == 1 and spec["vram"] == 0 for spec in specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES) for spec in specs)
    grouped = [
        spec for spec in specs
        if "/v46_grouped_task_bayes/" in spec["signature"]
    ]
    predictive = [
        spec for spec in specs
        if "/v46_grouped_task_predictive_bayes/" in spec["signature"]
    ]
    assert len(grouped) == len(predictive) == 2
    assert all(
        "--source-constraint-mean-hyperlaw-mode grouped_task_discrepancy"
        in spec["cmd"] for spec in grouped
    )
    assert all(
        "--source-constraint-mean-hyperlaw-mode grouped_task_predictive"
        in spec["cmd"] for spec in predictive
    )
    assert all(
        "--initial-design-archive-match-mode paired_frozen_control"
        in spec["cmd"] for spec in grouped + predictive
    )


def _episode_specs(augments, geometry):
    specs = []
    for base_domain in (
        "FactorShockStatePolicyRZDT1", "InventorySupplyChain",
    ):
        for episode in range(augments):
            specs.append({
                "label": (
                    base_domain
                    if episode == 0 else f"{base_domain}#episode{episode}"
                ),
                "base_domain": base_domain,
                "episode_index": episode,
                "record_count": 64 // augments,
                "task_geometry_shift": (
                    [0.04, -0.03, 0.02]
                    if geometry and episode > 0 else [0.0, 0.0, 0.0]
                ),
                "task_geometry_radius_scale": (
                    1.1 if geometry and episode > 0 else 1.0),
            })
    return specs


def _analyzer_row(variant, seed, challenger=False):
    augments = analyze.EXPECTED_AUGMENTS[variant]
    geometry = variant in analyze.GEOMETRY_VARIANTS
    exact = variant == "v41_two_task_source_bayes"
    mode = analyze.EXPECTED_HYPERLAW[variant]
    grouped = mode.startswith("grouped_task_")
    predictive = mode == "grouped_task_predictive"
    diagnostics = {
        "configured_hyperlaw_mode": mode,
        "target_oracle_used": False,
        "grouped_task_prior_selected": grouped,
        "finite_source_predictive_prior_selected": predictive,
        "prior_covariance_trace": 0.4,
        "shared_mean_covariance_trace": 0.05,
    }
    if grouped:
        diagnostics.update({
            "source_base_domain_count": 2,
            "source_episode_counts_by_base_domain": {
                "FactorShockStatePolicyRZDT1": 4,
                "InventorySupplyChain": 4,
            },
            "source_estimation_covariance_enters_shared_mean_only": True,
            "within_source_estimation_as_target_variation": False,
            "within_base_task_covariance_included": True,
            "between_base_discrepancy_rank": 1,
            "between_base_discrepancy_rank_upper_bound": 1,
            "within_base_task_discrepancy_rank": 4,
            "within_base_task_discrepancy_rank_upper_bound": 6,
            "combined_discrepancy_rank": 5,
            "combined_discrepancy_rank_upper_bound": 7,
            "finite_source_predictive_correction": predictive,
            "between_base_domain_covariance_trace": 0.1,
            "within_base_task_covariance_trace": 0.2,
        })
    specs = _episode_specs(augments, geometry)
    archive_fingerprint = f"posterior-{variant}"
    return {
        "gate_variant": variant,
        "heldout": "QueueResourceControl",
        "target_shared_shock_scale": 1.0,
        "seed": int(seed),
        "decision_backend": "sobol_new",
        "source_constraint_mean_hyperlaw_mode": mode,
        "true_feasible": bool(challenger),
        "adaptive_loss": False,
        "adaptive_improves_initial_best": bool(challenger),
        "feasible_simple_regret": 0.01 if challenger else None,
        "target_design_fingerprint": f"target-{seed}",
        "online_action_sequence_fingerprint": f"online-{seed}",
        "online_action_trace_target_oracle_used": False,
        "online_action_trace": [{
            "x_fingerprint": f"x-{seed}",
            "observed_response": [0.2, 0.1],
            "candidate_source": "sobol_continuation",
        }],
        "initial_design_archive_contract": {
            "mode": "exact" if exact else "paired_frozen_control",
            "proposal_archive_fingerprint": "proposal-archive",
            "posterior_archive_fingerprint": archive_fingerprint,
            "matches": exact,
            "proposal_frozen_across_arms": not exact,
            "target_data_used": False,
            "target_oracle_used": False,
        },
        "meta_prior": {"training": {
            "source_seed_mode": "frozen",
            "target_seed_used_for_source_training": False,
            "source_episode_target_data_used": False,
            "source_episode_target_oracle_used": False,
            "source_base_domain_count": 2,
            "source_episode_count_per_base_domain": augments,
            "source_episode_budget_mode": "per_base_domain",
            "source_episode_cost_matched": True,
            "source_episode_record_budget": 128,
            "source_task_count": 2 * augments,
            "source_archive_simulator_calls": 384,
            "source_archive_fingerprint": archive_fingerprint,
            "source_episode_specs": specs,
        }},
        "source_target_adaptation_contract": {
            "source_simulator_calls": 384,
            "source_oracle_aided": False,
        },
        "gpr_numerics": [{}, {"source_parametric_prior": diagnostics}],
        "boundary_raw_pool_truth_diagnostics": {
            "boundary_raw_pool_true_feasible_count": 8,
            "boundary_raw_pool_true_certified_count": 0,
            "boundary_raw_pool_false_certified_count": 0,
            "boundary_raw_pool_best_feasible_epistemic_radius": 0.04,
            "boundary_raw_pool_best_feasible_true_margin": -0.02,
        },
        "certificate_outcome_audit": {
            "certified_true_feasible_count": 0,
            "false_certificate_count": 0,
        },
    }


def test_v46_analyzer_promotes_only_strict_grouped_population():
    rows = []
    for seed in (1, 3):
        for variant in analyze.VARIANTS:
            rows.append(_analyzer_row(
                variant,
                seed,
                challenger=variant == "v46_grouped_task_bayes",
            ))
    result = analyze.summarize(rows)
    assert result["paired_initial_design_actions_and_target_responses"]
    assert all(result["source_episode_contract"].values())
    assert all(result["source_hyperlaw_contract"].values())
    assert result["grouped_population_strictly_improves_controls"]
    assert result["promotion_eligible"] == ["v46_grouped_task_bayes"]
