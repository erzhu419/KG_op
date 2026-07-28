import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "SC-OLH-KG"))

from performance import benchmark_lodo_meta_prior as benchmark
from problems.rzdt import make_problem
from problems.single_objective import ScalarizedProblem
from representation.meta_prior import LearnedMetaPrior


SUBMIT_PATH = (
    REPO
    / "scripts/submit_scolhkg_mean_alignment_v45_source_episode_gate_scheduler.py"
)
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v45_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ANALYZE_PATH = (
    REPO / "SC-OLH-KG/performance/analyze_mean_alignment_v45_gate.py")
sys.path.insert(0, str(ANALYZE_PATH.parent))
ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "mean_v45_analyze", ANALYZE_PATH)
analyze = importlib.util.module_from_spec(ANALYZE_SPEC)
ANALYZE_SPEC.loader.exec_module(analyze)


def _spec_args(**overrides):
    values = {
        "meta_seed": 20260720,
        "meta_source_augments": 4,
        "meta_source_budget_mode": "per_base_domain",
        "meta_source_geometry_shift_scale": 0.075,
        "meta_source_geometry_log_radius_jitter": 0.20,
        "meta_source_sigma_jitter": 0.0,
        "meta_source_alpha_jitter": 0.0,
        "meta_source_weight_jitter": 0.0,
        "source_records_per_domain": 64,
        "sigma": 0.04,
        "alpha": 0.05,
        "weights": "0.5,0.5",
    }
    values.update(overrides)
    return values


def _scalar_problem(name, **kwargs):
    return ScalarizedProblem(
        make_problem(name, d=12, L=100, sigma=0.04, **kwargs),
        weights=(0.5, 0.5),
    )


def test_cost_matched_episode_allocation_preserves_base_domain_budget():
    assert benchmark._source_episode_record_counts(
        64, 4, "per_base_domain") == [16, 16, 16, 16]
    assert benchmark._source_episode_record_counts(
        10, 3, "per_base_domain") == [4, 3, 3]
    assert benchmark._source_episode_record_counts(
        10, 3, "per_episode") == [10, 10, 10]
    with pytest.raises(ValueError, match="at least one record"):
        benchmark._source_episode_record_counts(3, 4, "per_base_domain")


def test_episode_context_is_frozen_nontrivial_and_target_free():
    first = benchmark._source_augmented_problem_specs(
        _spec_args(), ["InventorySupplyChain", "FactorShockStatePolicyRZDT1"], 0)
    second = benchmark._source_augmented_problem_specs(
        _spec_args(), ["InventorySupplyChain", "FactorShockStatePolicyRZDT1"], 0)
    assert len(first) == 8
    assert [item["record_count"] for item in first] == [16] * 8
    for left, right in zip(first, second):
        assert left["label"] == right["label"]
        np.testing.assert_allclose(
            left["problem_kwargs"]["task_geometry_shift"],
            right["problem_kwargs"]["task_geometry_shift"],
        )
        assert left["problem_kwargs"]["task_geometry_radius_scale"] == right[
            "problem_kwargs"]["task_geometry_radius_scale"]
    assert any(
        np.linalg.norm(item["problem_kwargs"]["task_geometry_shift"]) > 0.0
        for item in first if item["episode_index"] > 0
    )


def test_zero_geometry_keeps_legacy_domains_compatible():
    specs = benchmark._source_augmented_problem_specs(
        _spec_args(
            meta_source_augments=1,
            meta_source_geometry_shift_scale=0.0,
            meta_source_geometry_log_radius_jitter=0.0,
        ),
        ["RZDT1"],
        0,
    )
    assert specs[0]["problem_kwargs"] == {}
    problem = benchmark.build_scalarized_problem(
        "RZDT1", 5, 100, 0.04, 0.05, np.asarray([0.5, 0.5]),
        problem_kwargs=specs[0]["problem_kwargs"],
    )
    assert np.isfinite(problem.true_objective((20, 0, 0, 0, 0)))


def test_nominal_inventory_formulas_are_bit_identical_to_legacy():
    problem = make_problem(
        "InventorySupplyChain", d=6, L=100, sigma=0.04, alpha=0.05,
        task_geometry_shift=(0.0, 0.0, 0.0),
        task_geometry_radius_scale=1.0,
    )
    x = (31, 67, 43, 59, 22, 88)
    stock, reorder, safety, dispersion = problem._policy_summary(x)
    holding = 0.7 * max(stock - 0.58, 0.0) ** 2
    backlog = 1.8 * max(0.58 - stock, 0.0) ** 2
    reorder_loss = 1.4 * (reorder - 0.36) ** 2
    safety_loss = 1.1 * (safety - 0.42) ** 2
    service_gap = (
        ((stock - 0.56) / 0.20) ** 2
        + ((reorder - 0.34) / 0.22) ** 2
        + ((safety - 0.44) / 0.18) ** 2
        + 0.4 * (dispersion / 0.25) ** 2
    )
    expected = (
        float(0.25 + holding + backlog + reorder_loss + 0.3 * dispersion ** 2),
        float(0.30 + safety_loss + 0.5 * reorder_loss + 0.5 * backlog),
        float(0.10 * (service_gap - 1.0)),
    )
    assert problem.true_objectives(x) == expected
    exposure = problem.risk_exposures(x)
    stockout = max(0.0, 0.48 - safety) + 0.35 * max(0.0, 0.30 - reorder)
    expected_a = np.asarray([
        0.20 + max(0.0, 0.62 - stock),
        0.15 + max(0.0, stock - 0.58),
        0.10 + stockout,
    ])
    expected_n = np.asarray([
        0.20 + max(0.0, 0.55 - stock) + 0.35 * abs(reorder - 0.35),
        0.15 + dispersion + 0.5 * max(0.0, 0.50 - safety),
    ])
    assert np.array_equal(exposure.A, expected_a)
    assert np.array_equal(exposure.N, expected_n)


def test_nominal_queue_formulas_are_bit_identical_to_legacy():
    problem = make_problem(
        "QueueResourceControl", d=6, L=100, sigma=0.04, alpha=0.05,
        task_geometry_shift=(0.0, 0.0, 0.0),
        task_geometry_radius_scale=1.0,
    )
    x = (31, 67, 43, 59, 22, 88)
    capacity, priority, smoothing, imbalance = problem._policy_summary(x)
    wait_loss = (
        2.0 * max(0.0, 0.58 - capacity) ** 2
        + 0.8 * (priority - 0.36) ** 2
    )
    resource_loss = (
        0.7 * max(0.0, capacity - 0.72) ** 2
        + 0.5 * (smoothing - 0.50) ** 2
    )
    pocket = (
        ((capacity - 0.64) / 0.18) ** 2
        + ((priority - 0.38) / 0.20) ** 2
        + ((smoothing - 0.52) / 0.20) ** 2
        + 0.45 * (imbalance / 0.28) ** 2
    )
    expected = (
        float(0.24 + wait_loss + 0.25 * imbalance ** 2),
        float(0.30 + resource_loss + 0.35 * wait_loss),
        float(0.095 * (pocket - 1.0)),
    )
    assert problem.true_objectives(x) == expected
    exposure = problem.risk_exposures(x)
    queue = max(0.0, 0.62 - capacity) + 0.25 * imbalance
    wait = max(0.0, 0.44 - priority) + 0.15 * max(0.0, 0.45 - smoothing)
    utilization = max(0.0, capacity - 0.70) + 0.20 * imbalance
    expected_a = np.asarray([0.10 + queue, 0.12 + wait, 0.10 + utilization])
    expected_n = np.asarray([
        0.15 + max(0.0, 0.60 - smoothing) + 0.45 * imbalance,
        0.20 + abs(capacity - 0.64) + 0.35 * abs(priority - 0.38),
    ])
    assert np.array_equal(exposure.A, expected_a)
    assert np.array_equal(exposure.N, expected_n)


def test_explicit_zero_factor_geometry_matches_default_exactly():
    plain = make_problem(
        "FactorShockStatePolicyRZDT1", d=12, L=100, sigma=0.04, alpha=0.05)
    zero = make_problem(
        "FactorShockStatePolicyRZDT1", d=12, L=100, sigma=0.04, alpha=0.05,
        task_geometry_shift=(0.0, 0.0, 0.0),
        task_geometry_radius_scale=1.0,
    )
    for x in ((22,) + (72,) * 11, (41,) + (63,) * 11):
        assert plain.true_objectives(x) == zero.true_objectives(x)
        assert np.array_equal(plain.true_sigma(x), zero.true_sigma(x))
        plain_exposure = plain.risk_exposures(x)
        zero_exposure = zero.risk_exposures(x)
        assert np.array_equal(plain_exposure.A, zero_exposure.A)
        assert np.array_equal(plain_exposure.N, zero_exposure.N)


@pytest.mark.parametrize(
    "name",
    [
        "FactorShockStatePolicyRZDT1",
        "InventorySupplyChain",
        "QueueResourceControl",
    ],
)
def test_task_context_changes_mean_geometry_without_target_queries(name):
    nominal = _scalar_problem(name)
    shifted = _scalar_problem(
        name,
        task_geometry_shift=(0.06, -0.05, 0.04),
        task_geometry_radius_scale=1.2,
    )
    x = tuple([55] * nominal.d)
    nominal_mean = nominal.true_constraint_mean(x)
    shifted_mean = shifted.true_constraint_mean(x)
    assert not np.isclose(nominal_mean, shifted_mean)
    assert np.isfinite(shifted.true_sigma(x)[1])


def test_per_problem_record_counts_are_exact_and_charged():
    sources = [
        ("inventory#episode0", _scalar_problem("InventorySupplyChain")),
        ("inventory#episode1", _scalar_problem(
            "InventorySupplyChain", task_geometry_shift=(0.05, 0.0, 0.0))),
    ]
    prior = LearnedMetaPrior(
        source_observation_mode="replicated",
        source_observation_replicates=2,
        source_design_mode="random",
        teacher_records_per_domain=0,
        seed=4501,
    ).fit_from_source_problems(
        sources,
        n_records_per_domain=99,
        n_records_by_problem=[3, 5],
        rng=np.random.default_rng(4501),
    )
    diagnostics = prior.training_diagnostics
    assert diagnostics["source_task_record_counts"] == {
        "inventory#episode0": 3,
        "inventory#episode1": 5,
    }
    assert diagnostics["source_task_record_budget"] == 8
    assert diagnostics["source_simulator_calls"] == 16


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
        "run_id": "mean-v45",
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


def test_v45_submitter_builds_eight_cost_matched_cpu_sentinels(tmp_path):
    specs = submit.build_specs(_scheduler_args(tmp_path))
    assert len(specs) == 8
    assert all(spec["cpu"] == 1 and spec["vram"] == 0 for spec in specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES) for spec in specs)
    assert all("--source-records-per-domain 64" in spec["cmd"] for spec in specs)
    baseline = [
        spec for spec in specs
        if "/v41_two_task_source_bayes/" in spec["signature"]
    ]
    assert all(
        "--initial-design-archive-match-mode exact" in spec["cmd"]
        for spec in baseline
    )
    episode = [
        spec for spec in specs
        if "/v45_geometry_source_bayes/" in spec["signature"]
    ]
    assert len(episode) == 2
    assert all(
        "--initial-design-archive-match-mode paired_frozen_control"
        in spec["cmd"] for spec in episode
    )
    assert all("--meta-source-augments 4" in spec["cmd"] for spec in episode)
    assert all("--meta-source-budget-mode per_base_domain" in spec["cmd"] for spec in episode)
    assert all("--meta-source-geometry-shift-scale 0.075" in spec["cmd"] for spec in episode)
    assert all("--meta-source-sigma-jitter 0.0" in spec["cmd"] for spec in episode)


def _episode_row(variant, seed):
    augments = analyze.EXPECTED_AUGMENTS[variant]
    geometry = variant in analyze.GEOMETRY_VARIANTS
    exact_archive = variant == "v41_two_task_source_bayes"
    posterior_archive = (
        "proposal-archive" if exact_archive else f"archive-{variant}")
    specs = []
    for base_domain in ("FactorShockStatePolicyRZDT1", "InventorySupplyChain"):
        for episode in range(augments):
            specs.append({
                "label": base_domain if episode == 0 else f"{base_domain}#episode{episode}",
                "base_domain": base_domain,
                "episode_index": episode,
                "record_count": 64 // augments,
                "task_geometry_shift": (
                    [0.0, 0.0, 0.0]
                    if not geometry or episode == 0 else [0.05, -0.03, 0.02]
                ),
                "task_geometry_radius_scale": (
                    1.0 if not geometry or episode == 0 else 1.1),
            })
    return {
        "gate_variant": variant,
        "seed": seed,
        "initial_design_archive_contract": {
            "status": "audited",
            "mode": (
                "exact" if exact_archive else "paired_frozen_control"),
            "proposal_archive_fingerprint": "proposal-archive",
            "posterior_archive_fingerprint": posterior_archive,
            "matches": exact_archive,
            "proposal_frozen_across_arms": not exact_archive,
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
            "source_archive_fingerprint": posterior_archive,
            "source_episode_specs": specs,
        }},
        "source_target_adaptation_contract": {
            "source_simulator_calls": 384,
            "source_oracle_aided": False,
        },
    }


def test_v45_episode_contract_distinguishes_geometry_from_label_control():
    for variant in analyze.VARIANTS:
        rows = [_episode_row(variant, seed) for seed in (1, 3)]
        assert analyze._episode_contract(rows, variant)
    corrupted = [_episode_row("v45_episode_label_control", seed) for seed in (1, 3)]
    corrupted[0]["meta_prior"]["training"]["source_archive_simulator_calls"] = 2304
    assert not analyze._episode_contract(
        corrupted, "v45_episode_label_control")
