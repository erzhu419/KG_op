from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "performance"))

from analyze_source_support_effects import (  # noqa: E402
    build_source_support_pairs,
)
from analyze_causal_prior_effects import (  # noqa: E402
    summarize_proposal_mode_pairs,
)


def _row(run_id, seed, feasible, regret, archive):
    return {
        "run_id": run_id,
        "variant": "causal_prior_v2/proposal_only/risk_objective_atlas/none",
        "status": "ok",
        "domain": "InventorySupplyChain",
        "seed": seed,
        "d": 50,
        "N": 20,
        "n0": 10,
        "source_calls": 384,
        "source_archive_fingerprint": archive,
        "initial_design_fingerprint": f"{run_id}-{seed}",
        "initial_has_true_feasible": feasible,
        "initial_best_feasible_regret": regret,
        "true_feasible": feasible,
        "feasible_regret": regret,
    }


def test_source_support_pairs_hold_target_cell_fixed():
    rows = []
    for seed in (0, 1):
        rows.extend([
            _row("universal", seed, True, 0.10 + 0.01 * seed, "archive-u"),
            _row(
                "shared", seed, seed == 1,
                0.20 if seed == 1 else None, "archive-s"),
        ])
    pairs = build_source_support_pairs(
        rows,
        challenger_run="universal",
        reference_run="shared",
        challenger_label="universal_low_frequency",
        reference_label="shared_uniform",
    )
    summaries = summarize_proposal_mode_pairs(pairs)

    assert len(pairs) == 2
    assert all(item["target_budget_match"] for item in pairs)
    assert all(not item["archive_match"] for item in pairs)
    assert len(summaries) == 1
    effect = summaries[0]
    assert effect["final_feasible_win_count"] == 1
    assert effect["final_feasible_loss_count"] == 0
    assert effect["median_final_regret_delta"] < 0.0
