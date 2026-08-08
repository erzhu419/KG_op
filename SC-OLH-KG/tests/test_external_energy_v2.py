import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_external_energy_v2 import analyze  # noqa: E402
from performance.benchmark_external_energy_v2 import (  # noqa: E402
    market_region,
    materialize_source_atlas,
    region_heldout_source_markets,
    run_task,
)
from performance.run_external_energy_v2_matrix import (  # noqa: E402
    build_target_cells,
    design_filename,
    materialize_design_matrix,
    run_target_matrix,
)


def _write_energy_suite(path):
    markets = ("DK_2", "GB_GBN", "IT_NORD", "NO_1", "SE_1")
    start = np.datetime64("2017-01-01T00", "h").astype(np.int64)
    hours = start + np.arange(3 * 365 * 24, dtype=np.int64)
    phase = 2.0 * np.pi * np.arange(len(hours)) / 24.0
    forecast = 1000.0 + 100.0 * np.sin(phase - 0.5)
    arrays = {
        "metadata_json": np.asarray(json.dumps({
            "schema_version": 2,
            "dataset": "test fixture",
            "version": "fixture",
            "markets": list(markets),
            "years": [2017, 2018, 2019],
        }), dtype="U"),
    }
    for index, market in enumerate(markets):
        shock = (
            (45.0 + 2.0 * index) * np.maximum(np.sin(phase), 0.0)
            + 10.0 * np.sin((2.0 + 0.1 * index) * phase)
        )
        prefix = f"{market}__"
        arrays[prefix + "timestamp_hour"] = hours
        arrays[prefix + "load_actual"] = forecast + shock
        arrays[prefix + "load_forecast"] = forecast
        arrays[prefix + "price"] = 45.0 + 8.0 * np.sin(phase - 0.8)
    with Path(path).open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def test_region_holdout_excludes_every_target_region_market():
    sources = region_heldout_source_markets("DK_2")
    assert sources == ("GB_GBN", "IT_NORD", "NO_1", "SE_1")
    assert all(market_region(source) != "denmark" for source in sources)


def test_frozen_energy_v2_design_replays_without_target_outcomes(tmp_path):
    data = tmp_path / "energy_suite.npz"
    _write_energy_suite(data)
    design = materialize_source_atlas(
        data_path=data,
        target_market="DK_2",
        dimension=24,
        n0=5,
        library_size=16,
        source_replications=1,
    )
    assert design["target_outcomes_used"] is False
    assert design["target_region_excluded_from_source_archive"] is True
    assert design["source_calls"] == 64
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design), encoding="utf-8")
    result = run_task(
        data_path=data,
        target_market="DK_2",
        target_seed=0,
        arm="source_atlas",
        dimension=24,
        n0=5,
        N=5,
        library_size=16,
        source_replications=1,
        design_path=design_path,
        verification_budgets=(80, 80, 80),
    )
    assert result["status"] == "ok"
    assert result["target_outcomes_used_to_fit_frontend"] is False
    assert result["source_archive_target_region_excluded"] is True
    assert result["initial_design_fingerprint"] == design[
        "initial_design_fingerprint"]


def test_energy_v2_analysis_separates_market_region_and_seed(tmp_path):
    paths = []
    for arm, certified, objective in (
        ("source_atlas", True, 0.1),
        ("natural_constant_grid", False, None),
    ):
        row = {
            "contract_id": "opsd_region_heldout_profile_design_v2",
            "status": "ok",
            "target_market": "DK_2",
            "target_region": "denmark",
            "target_seed": 0,
            "arm": arm,
            "independently_certified": certified,
            "false_certificate": False,
            "objective_if_certified": objective,
            "source_calls": 768 if arm == "source_atlas" else 0,
            "target_search_calls": 13,
            "verification_calls": 80,
            "all_in_calls_unamortized": 861 if arm == "source_atlas" else 93,
            "all_in_budget_cap_unamortized": (
                1021 if arm == "source_atlas" else 253),
            "all_in_calls_amortized": 131.4 if arm == "source_atlas" else 93.0,
            "all_in_budget_cap_amortized": (
                291.4 if arm == "source_atlas" else 253.0),
        }
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        paths.append(path)
    payload = analyze(paths)
    assert payload["status"] == "complete"
    assert payload["market_count"] == 1
    assert payload["region_count"] == 1
    comparison = payload["paired_algorithmic_repeatability"][0]
    assert comparison["first_wins"] == 1
    assert comparison["task_population_inference_claimed"] is False
    assert comparison["holm_family_size"] == 1
    region = payload["region_level_directional_audit"][0]
    assert region["region_count"] == 1
    assert region["holm_family_size"] == 1
    assert region[
        "mean_source_minus_control_region_safe_rate_bootstrap_95ci"
    ] == [1.0, 1.0]


def test_energy_v2_registered_matrix_has_450_cells():
    cells = build_target_cells()
    assert len(cells) == 450
    assert len({
        (row["target_market"], row["target_seed"], row["arm"])
        for row in cells
    }) == 450


def test_energy_v2_matrix_freezes_design_before_target_execution(tmp_path):
    data = tmp_path / "energy_suite.npz"
    _write_energy_suite(data)
    designs = tmp_path / "designs"
    outputs = tmp_path / "outputs"
    freeze_commit = "frozen-method"
    design_summary = materialize_design_matrix(
        data_path=data,
        output_dir=designs,
        freeze_commit=freeze_commit,
        markets=("DK_2",),
        workers=1,
        dimension=24,
        n0=5,
        library_size=16,
        source_replications=1,
    )
    design_path = designs / design_filename("DK_2")
    frozen = json.loads(design_path.read_text(encoding="utf-8"))
    assert design_summary["completed_count"] == 1
    assert frozen["confirmatory_freeze_commit"] == freeze_commit
    assert frozen["target_outcomes_used"] is False

    target_summary = run_target_matrix(
        data_path=data,
        design_dir=designs,
        output_dir=outputs,
        freeze_commit=freeze_commit,
        markets=("DK_2",),
        target_seeds=(0,),
        arms=("source_atlas", "raw_sobol"),
        workers=1,
        dimension=24,
        n0=5,
        N=5,
        library_size=16,
        source_replications=1,
        verification_budgets=(80, 80, 80),
    )
    assert target_summary["matrix_cell_count"] == 2
    assert target_summary["completed_count"] == 2
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in outputs.glob("cell*.json")
    ]
    assert {row["arm"] for row in rows} == {"source_atlas", "raw_sobol"}
    assert all(
        row["confirmatory_freeze_commit"] == freeze_commit for row in rows
    )
