from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "performance"))

from analyze_causal_prior_effects import (  # noqa: E402
    build_paired_effects,
    build_proposal_mode_effects,
    parse_causal_variant,
    summarize_pairs,
    summarize_proposal_mode_pairs,
)


def _row(
        seed,
        profile,
        feasible,
        regret,
        initial,
        initial_regret,
        proposal_mode="atlas"):
    return {
        "run_id": "causal",
        "variant": (
            f"causal_prior_v2/proposal_only/{proposal_mode}/{profile}"),
        "status": "ok",
        "domain": "Domain",
        "seed": seed,
        "d": 50,
        "N": 20,
        "n0": 10,
        "source_calls": 384,
        "source_archive_fingerprint": "same-archive",
        "initial_design_fingerprint": f"{proposal_mode}-{profile}-{seed}",
        "initial_has_true_feasible": initial,
        "initial_best_feasible_regret": initial_regret,
        "true_feasible": feasible,
        "feasible_regret": regret,
    }


def test_parse_causal_variant_requires_exact_contract():
    parsed = parse_causal_variant(
        "causal_prior_v2/joint/risk_coordinate_atlas/full")
    assert parsed == {
        "causal_mode": "joint",
        "proposal_mode": "risk_coordinate_atlas",
        "profile": "full",
    }
    assert parse_causal_variant("legacy/full") is None


def test_paired_effects_prioritize_feasibility_before_regret():
    rows = []
    for seed in (0, 1):
        rows.extend([
            _row(seed, "none", seed == 1, 0.20 if seed == 1 else None,
                 seed == 1, 0.25 if seed == 1 else None),
            _row(seed, "full", True, 0.10 + 0.01 * seed,
                 True, 0.15 + 0.01 * seed),
            _row(seed, "low_frequency_only", True, 0.12 + 0.01 * seed,
                 True, 0.18 + 0.01 * seed),
            _row(seed, "leave_out_low_frequency", seed == 1,
                 0.16 if seed == 1 else None, seed == 1,
                 0.20 if seed == 1 else None),
        ])

    pairs = build_paired_effects(rows)
    summaries = summarize_pairs(pairs)
    full = next(
        row for row in summaries
        if row["contrast"] == "full_vs_none")
    single = next(
        row for row in summaries
        if row["contrast"] == "single_only_vs_none"
        and row["component"] == "low_frequency")
    leave_out = next(
        row for row in summaries
        if row["contrast"] == "full_vs_leave_one_out"
        and row["component"] == "low_frequency")

    assert full["n_pairs"] == 2
    assert full["archive_match_count"] == 2
    assert full["initial_fingerprint_match_count"] == 0
    assert full["final_feasible_win_count"] == 1
    assert full["final_feasible_loss_count"] == 0
    assert full["median_final_regret_delta"] < 0.0
    assert single["final_feasible_win_count"] == 1
    assert leave_out["final_feasible_win_count"] == 1


def test_duplicate_causal_cell_is_rejected():
    row = _row(0, "none", True, 0.1, True, 0.1)
    with pytest.raises(ValueError, match="duplicate causal cell"):
        build_paired_effects([row, dict(row)])


def test_proposal_mode_effects_pair_same_profile_and_seed():
    rows = []
    for seed in (0, 1):
        rows.extend([
            _row(
                seed, "additivity_only", True, 0.10 + 0.01 * seed,
                True, 0.15 + 0.01 * seed,
                proposal_mode="risk_coordinate_atlas"),
            _row(
                seed, "additivity_only", seed == 1,
                0.20 if seed == 1 else None,
                seed == 1, 0.25 if seed == 1 else None,
                proposal_mode="rank_spanning"),
        ])

    pairs = build_proposal_mode_effects(rows)
    summaries = summarize_proposal_mode_pairs(pairs)

    assert len(pairs) == 2
    assert len(summaries) == 1
    effect = summaries[0]
    assert effect["profile"] == "additivity_only"
    assert effect["final_feasible_win_count"] == 1
    assert effect["final_feasible_loss_count"] == 0
    assert effect["median_final_regret_delta"] < 0.0
    assert effect["archive_match_count"] == 2
    assert effect["initial_fingerprint_match_count"] == 0
