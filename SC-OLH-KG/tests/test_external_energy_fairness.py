import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_external_energy_fairness import analyze  # noqa: E402


def _row(arm, seed, objective):
    target_search_calls = 397 if arm == "common_sobol_n397" else 13
    if arm == "frozen_proposal_n13":
        return {
            "status": "ok",
            "contract_id": "opsd_energy_gb_gbn_confirmatory_v1",
            "seed": seed,
            "target_search_calls": target_search_calls,
            "independently_certified": True,
            "false_certificate": False,
            "verification_calls": 80,
            "deployment_truth_audit": {
                "truly_chance_feasible": True,
                "true_objective_mean": objective,
            },
        }
    return {
        "status": "ok",
        "contract_id": "opsd_energy_postconfirmatory_fairness_v1",
        "fairness_arm": arm,
        "seed": seed,
        "target_search_calls": target_search_calls,
        "independently_certified": True,
        "false_certificate": False,
        "verification_calls": 80,
        "deployment_truth_audit": {
            "truly_chance_feasible": True,
            "true_objective_mean": objective,
        },
    }


def test_fairness_analysis_demotes_source_claim_when_grid_wins(tmp_path):
    paths = []
    for seed in range(100, 120):
        values = {
            "frozen_proposal_n13": 0.10,
            "low_frequency_grid_n13": 0.09,
            "common_sobol_n397": 0.15,
        }
        for arm, value in values.items():
            path = tmp_path / f"{arm}_{seed}.json"
            path.write_text(
                json.dumps(_row(arm, seed, value)), encoding="utf-8")
            paths.append(path)
    result = analyze(paths)
    assert result["status"] == "complete"
    decision = result["decision"]
    assert decision[
        "source_atlas_superior_to_natural_low_frequency_control"] is False
    assert decision[
        "source_atlas_superior_to_target_only_sobol_at_equal_source_plus_search_cost"] is True
    assert len(result["compact_rows"]) == 60
    assert all("raw_result_sha256" in row
               for row in result["compact_rows"])


def test_fairness_analysis_fails_closed_on_missing_seed(tmp_path):
    paths = []
    for seed in range(100, 119):
        for arm in (
            "frozen_proposal_n13",
            "low_frequency_grid_n13",
            "common_sobol_n397",
        ):
            path = tmp_path / f"{arm}_{seed}.json"
            path.write_text(
                json.dumps(_row(arm, seed, 0.1)), encoding="utf-8")
            paths.append(path)
    result = analyze(paths)
    assert result["status"] == "incomplete"
    assert result["failures"]
