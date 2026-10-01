"""Integration check for the supplemental selection hook and source accounting."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance import benchmark_external_energy_v3 as energy
from performance import submission_revision_energy as supplemental
from tests.test_external_energy_v3 import _write_energy_suite


def test_energy_controls_reach_target_evaluation_and_charge_the_same_archive(tmp_path, monkeypatch):
    data = tmp_path / "energy.npz"
    _write_energy_suite(data)
    source_archive = {}
    target_results = []
    original_archive = energy.build_source_archive
    original_target = energy.run_task
    original_selector = energy._structural_initial_profiles

    def record_archive(*args, **kwargs):
        records, markets = original_archive(*args, **kwargs)
        source_archive.update(records=records, markets=markets, kwargs=kwargs)
        return records, markets

    def record_target(**kwargs):
        result = original_target(**kwargs)
        target_results.append(result)
        return result

    monkeypatch.setattr(energy, "build_source_archive", record_archive)
    monkeypatch.setattr(energy, "run_task", record_target)
    rows = supplemental.evaluate_market("DK_2", data, target_seeds=(0,))
    archive = source_archive["records"]
    markets = source_archive["markets"]
    assert len(markets) == 4
    assert len(archive) == 4 * 64
    assert sum(len(r.objective_samples) for r in archive) == 768
    assert all(energy.market_region(market) != "denmark" for market in markets)
    assert source_archive["kwargs"]["seed"] == 20260808 + 1237
    library = source_archive["kwargs"]["library"]
    standard_indices, _ = supplemental.standardized_generic_dct_maximin(
        [p.values for p in library], 10, nodes=library[0].nodes,
        max_frequency=8, frequency_penalty=0.25, include_diagonal_quadratic=True)
    portfolio = supplemental.source_greedy_portfolio(archive, 10)
    expected_ids = (
        [library[i].profile_id for i in standard_indices],
        [member.profile_id for member in portfolio.members],
    )
    assert expected_ids[0] != expected_ids[1]
    for i, (row, result) in enumerate(zip(rows, target_results)):
        # These IDs and records come from the unchanged real target runner,
        # rather than from a stand-in that only invokes the patched selector.
        assert result["selected_profile_ids"] == expected_ids[i]
        assert len(result["search_records"]) == 13
        assert [record["source"] for record in result["search_records"]] == ["initial_design"] * 10 + ["neutral_sobol_continuation"] * 3
        assert row["design_seed"] == 4300000
        assert row["N"] == 13 and row["n0"] == 10
        assert row["source_calls"] == (0 if i == 0 else 768)
        assert row["all_in_calls_unamortized"] == row["source_calls"] + 13 + row["verification_calls"]
        assert row["all_in_calls_amortized"] == row["source_calls"] / 20 + 13 + row["verification_calls"]
        assert result["maximum_verification_calls"] == 240
        assert result["frontend_diagnostics"]["target_outcomes_used"] is False
    assert energy._structural_initial_profiles is original_selector
