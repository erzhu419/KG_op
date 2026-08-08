import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.audit_external_energy_temporal_blocks import (  # noqa: E402
    audit_frozen_policy,
)
from performance.analyze_external_energy_v3 import analyze  # noqa: E402
from performance.benchmark_external_energy_v3 import (  # noqa: E402
    CONTRACT_ID,
    materialize_source_atlas,
    run_task,
)
from performance.run_external_energy_v3_matrix import (  # noqa: E402
    build_target_cells,
    materialize_design_matrix,
    run_target_matrix,
)
from problems.energy_forecast_policy import (  # noqa: E402
    OPSDForecastIndexedStorageProblem,
)


def _write_energy_suite(path):
    markets = ("DK_2", "GB_GBN", "IT_NORD", "NO_1", "SE_1")
    start = np.datetime64("2017-01-01T00", "h").astype(np.int64)
    hours = start + np.arange(3 * 365 * 24, dtype=np.int64)
    index = np.arange(len(hours), dtype=float)
    daily = 2.0 * np.pi * index / 24.0
    weekly = 2.0 * np.pi * index / (24.0 * 7.0)
    forecast = 1000.0 + 120.0 * np.sin(daily - 0.5) + 45.0 * np.sin(weekly)
    arrays = {
        "metadata_json": np.asarray(json.dumps({
            "schema_version": 2,
            "dataset": "energy V3 test fixture",
            "version": "fixture",
            "markets": list(markets),
            "years": [2017, 2018, 2019],
        }), dtype="U"),
    }
    for market_index, market in enumerate(markets):
        shock = (
            (40.0 + 3.0 * market_index) * np.maximum(np.sin(daily), 0.0)
            + 9.0 * np.sin((2.0 + 0.05 * market_index) * daily)
        )
        prefix = f"{market}__"
        arrays[prefix + "timestamp_hour"] = hours
        arrays[prefix + "load_actual"] = forecast + shock
        arrays[prefix + "load_forecast"] = forecast
        arrays[prefix + "price"] = 45.0 + 8.0 * np.sin(daily - 0.8)
    with Path(path).open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def test_energy_v3_separates_policy_dimension_and_physical_horizon(tmp_path):
    data = tmp_path / "energy.npz"
    _write_energy_suite(data)
    problem = OPSDForecastIndexedStorageProblem(
        data, market="DK_2", d=37, horizon=24)
    assert problem.d == 37
    assert problem.horizon == 24
    assert problem.verification_window_length == 24
    contract = problem.information_contract()
    assert contract["policy_grid_dimension"] == 37
    assert contract["simulation_horizon_hours"] == 24
    assert contract["decision_dimension_changes_simulation_horizon"] is False
    assert contract["actual_target_error_used_by_observable_coordinate"] is False
    output = problem.simulate(tuple([50] * 37), np.random.default_rng(7))
    assert output.shape == (2,)
    assert np.all(np.isfinite(output))


def test_energy_v3_coordinate_is_outcome_free_and_dimension_stable(tmp_path):
    data = tmp_path / "energy.npz"
    _write_energy_suite(data)
    low = OPSDForecastIndexedStorageProblem(
        data, market="DK_2", d=24, horizon=48, outcome_access=False)
    high = OPSDForecastIndexedStorageProblem(
        data, market="DK_2", d=240, horizon=48, outcome_access=False)
    profile_low = np.linspace(0.1, 0.9, low.d)
    profile_high = np.linspace(0.1, 0.9, high.d)
    exposure_low = low.risk_exposures(low.continuous_to_int(profile_low))
    exposure_high = high.risk_exposures(high.continuous_to_int(profile_high))
    assert exposure_low.meta["target_outcomes_used"] is False
    assert exposure_high.meta["target_outcomes_used"] is False
    assert np.allclose(exposure_low.A, exposure_high.A, atol=0.03)
    assert np.allclose(exposure_low.N, exposure_high.N, atol=0.03)


def test_energy_v3_temporal_audit_uses_horizon_not_profile_dimension(tmp_path):
    data = tmp_path / "energy.npz"
    _write_energy_suite(data)
    problem = OPSDForecastIndexedStorageProblem(
        data, market="DK_2", d=37, horizon=24,
        required_splits=("verification",),
    )
    audit = audit_frozen_policy(
        problem,
        tuple([50] * problem.d),
        maximum_sampled_starts=64,
    )
    assert audit["physical_window_horizon"] == 24
    assert audit["nonoverlapping_start_count"] > 0


def test_energy_v3_frozen_source_design_excludes_target_region(tmp_path):
    data = tmp_path / "energy.npz"
    _write_energy_suite(data)
    design = materialize_source_atlas(
        data_path=data,
        target_market="DK_2",
        dimension=24,
        horizon=24,
        n0=5,
        library_size=16,
        source_replications=1,
    )
    assert design["target_outcomes_used"] is False
    assert design["target_region_excluded_from_source_archive"] is True
    assert design["source_calls"] == 64
    assert design["policy_semantics"] == (
        "forecast_stress_to_target_state_of_charge")


def test_energy_v3_source_and_functional_arms_share_contract(tmp_path):
    data = tmp_path / "energy.npz"
    _write_energy_suite(data)
    design = materialize_source_atlas(
        data_path=data,
        target_market="DK_2",
        dimension=24,
        horizon=24,
        n0=4,
        library_size=16,
        source_replications=1,
    )
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design), encoding="utf-8")
    source = run_task(
        data_path=data,
        target_market="DK_2",
        target_seed=0,
        arm="source_atlas",
        dimension=24,
        horizon=24,
        n0=4,
        N=4,
        library_size=16,
        source_replications=1,
        design_path=design_path,
        verification_budgets=(8, 8, 8),
    )
    functional = run_task(
        data_path=data,
        target_market="DK_2",
        target_seed=0,
        arm="target_only_dct_space_scbo",
        dimension=24,
        horizon=24,
        n0=4,
        N=5,
        coefficient_count=3,
        verification_budgets=(8, 8, 8),
        raw_samples=8,
        num_restarts=2,
        maxiter=10,
        batch_candidates=16,
    )
    assert source["contract_id"] == functional["contract_id"] == CONTRACT_ID
    assert source["source_calls"] == 64
    assert functional["source_calls"] == 0
    assert functional["functional_coordinate_contract"][
        "source_outcomes_used"] is False
    assert source["information_contract"]["policy_semantics"] == functional[
        "information_contract"]["policy_semantics"]


def test_energy_v3_matrix_is_complete_and_restartable(tmp_path):
    assert len(build_target_cells()) == 540
    data = tmp_path / "energy.npz"
    _write_energy_suite(data)
    designs = tmp_path / "designs"
    results = tmp_path / "results"
    checkpoints = tmp_path / "checkpoints"
    design_summary = materialize_design_matrix(
        data_path=data,
        output_dir=designs,
        method_freeze_commit="method",
        execution_commit="execution",
        markets=("DK_2",),
        workers=1,
        dimension=24,
        horizon=24,
        n0=4,
        library_size=16,
        source_replications=1,
    )
    assert design_summary["completed_count"] == 1
    first = run_target_matrix(
        data_path=data,
        design_dir=designs,
        output_dir=results,
        checkpoint_dir=checkpoints,
        method_freeze_commit="method",
        execution_commit="execution",
        markets=("DK_2",),
        target_seeds=(0,),
        arms=("source_atlas", "raw_sobol"),
        workers=1,
        dimension=24,
        horizon=24,
        n0=4,
        N=4,
        library_size=16,
        source_replications=1,
        verification_budgets=(8, 8, 8),
    )
    second = run_target_matrix(
        data_path=data,
        design_dir=designs,
        output_dir=results,
        checkpoint_dir=checkpoints,
        method_freeze_commit="method",
        execution_commit="execution",
        markets=("DK_2",),
        target_seeds=(0,),
        arms=("source_atlas", "raw_sobol"),
        workers=1,
        dimension=24,
        horizon=24,
        n0=4,
        N=4,
        library_size=16,
        source_replications=1,
        verification_budgets=(8, 8, 8),
    )
    assert first["completed_count"] == 2
    assert first["error_count"] == 0
    assert second["skipped_count"] == 2


def test_energy_v3_analysis_rejects_wrong_contract_and_keeps_task_units(tmp_path):
    paths = []
    for arm, objective in (("source_atlas", 0.1), ("raw_sobol", 0.2)):
        row = {
            "contract_id": CONTRACT_ID,
            "status": "ok",
            "target_market": "DK_2",
            "target_region": "denmark",
            "target_seed": 0,
            "arm": arm,
            "independently_certified": True,
            "false_certificate": False,
            "objective_if_certified": objective,
            "source_calls": 768 if arm == "source_atlas" else 0,
            "target_search_calls": 13,
            "verification_calls": 80,
            "all_in_calls_unamortized": 861 if arm == "source_atlas" else 93,
            "all_in_budget_cap_unamortized": (
                1021 if arm == "source_atlas" else 253),
            "all_in_calls_amortized": (
                131.4 if arm == "source_atlas" else 93.0),
            "all_in_budget_cap_amortized": (
                291.4 if arm == "source_atlas" else 253.0),
            "wall_time_sec": 1.0,
        }
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        paths.append(path)
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"contract_id": "wrong", "status": "ok"}))
    payload = analyze([*paths, wrong])
    assert payload["contract_id"] == (
        "opsd_forecast_indexed_region_holdout_analysis_v3")
    assert payload["status"] == "incomplete"
    assert payload["market_count"] == 1
    assert payload["region_count"] == 1
    assert payload["paired_algorithmic_repeatability"][0]["first_wins"] == 1
