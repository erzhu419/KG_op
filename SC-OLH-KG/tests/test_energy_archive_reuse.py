import json
from pathlib import Path

import numpy as np

from core.terminal_verification import verify_frozen_shortlist_binomial
from performance.benchmark_energy_archive_reuse import analyze, verify_batch
from performance.benchmark_external_energy_v3 import _profile_point
from problems.randomized_profiles import generate_structural_profile_library
from problems.energy_archive_reuse import TemporalArchiveReuseStorageProblem, month_period, sampled_window_starts

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external/opsd_time_series_extended_v2.npz"


def problem(month=1, outcome_access=True):
    return TemporalArchiveReuseStorageProblem(DATA, market="DK_1",
        search_period=month_period(2018, month), verification_period=month_period(2019, month),
        outcome_access=outcome_access)


def test_months_keep_the_same_asset_and_observable_scales():
    january, july = problem(1, False), problem(7, False)
    assert january._load_scale == july._load_scale
    assert january.physics == july.physics
    np.testing.assert_array_equal(january._forecast_stress, july._forecast_stress)
    assert january.information_contract()["normalization_fit_period"] == ["2017-01-01T00", "2018-01-01T00"]
    for p in (january, july):
        for split in ("search", "verification"):
            starts = p.split_window_starts(split)
            timestamps = p.series.timestamp_hour
            a, b = map(lambda t: np.datetime64(t, "h").astype(np.int64), p._periods[split])
            assert np.all(timestamps[starts] >= a)
            assert np.all(timestamps[starts + 167] < b)
        assert max(p.split_window_starts("search")) + 167 < min(p.split_window_starts("verification"))
        assert p._net_error is None


def test_batched_windows_match_charged_scalar_calls():
    p = problem()
    point = tuple(np.rint(np.linspace(0, 100, 1000)).astype(int))
    seeds = [[20260930, r, 3109] for r in range(3)]
    starts = sampled_window_starts(p, "search", seeds)
    batched = p.evaluate_window_starts(point, starts)
    scalar = [p.simulate(point, np.random.default_rng(np.random.SeedSequence(s))) for s in seeds]
    np.testing.assert_allclose(batched, scalar, atol=1e-14, rtol=1e-14)


def test_batch_verifier_preserves_original_fixed_budget_law():
    p = problem()
    shortlist = [dict(point=[value] * 1000) for value in (0, 50, 100)]
    design_seed = 9030100
    selected, batched = verify_batch(p, shortlist, design_seed)
    original_selected, original = verify_frozen_shortlist_binomial(p, shortlist,
        seed=design_seed + 71003, search_evaluation_count=13,
        candidate_budgets=(80, 80, 80), familywise_delta=.05, split="verification")
    assert selected == original_selected
    assert batched["verification_budget"] == original["verification_budget"]
    assert [a["successes"] for a in batched["attempts"]] == [a["successes"] for a in original["attempts"]]
    np.testing.assert_allclose([a["objective_sample_mean"] for a in batched["attempts"]],
        [a["objective_sample_mean"] for a in original["attempts"]], rtol=1e-14, atol=1e-14)


def test_repeat_seeds_are_not_extra_archive_deployments():
    protocol = json.loads((ROOT / "performance/manifests/energy_archive_reuse_20260930.json").read_text())
    rows = [dict(target_market=market, month=month, algorithm_seed=seed, arm=arm,
                 certified_true_feasible=seed < 2, target_search_calls=13, verification_calls=80,
                 initial_feasible_coverage=True, first_center_true_feasible=False,
                 false_certificate=False, objective_if_certified=.1)
            for market in protocol["markets"] for month in range(1, 13)
            for seed in range(5) for arm in protocol["target"]["arms"]]
    result = analyze(rows, protocol)
    full = next(r for r in result["summaries"] if r["arm"] == "source_atlas")
    assert full["actual_deployment_tasks"] == 216
    assert full["source_calls_once"] == 18 * 384
    assert full["cumulative_calls"] == 18 * 384 + 216 * 93
    assert np.isclose(full["expected_true_certificates"], 216 * .4)


def test_real_profiles_respect_shared_hourly_power():
    library = {p.profile_id: p for p in generate_structural_profile_library(
        64, dimension=128, seed=20261799, maximum_frequency=40)}
    for market, profile_id in (("NO_1", "profile_0062"), ("SE_1", "profile_0008")):
        p = TemporalArchiveReuseStorageProblem(DATA, market=market,
            search_period=month_period(2018, 1), verification_period=month_period(2019, 1))
        starts = p.split_window_starts("verification")
        starts = starts[np.linspace(0, len(starts) - 1, 32).round().astype(int)]
        _, state = p._evaluate_start_batch(_profile_point(p, library[profile_id]), starts,
                                         return_diagnostics=True)
        assert np.all(state["maximum_hourly_energy_exchange"] <= p.physics.power_capacity)
        assert np.all((state["terminal_energy"] >= 0.) & (state["terminal_energy"] <= p.physics.energy_capacity))
