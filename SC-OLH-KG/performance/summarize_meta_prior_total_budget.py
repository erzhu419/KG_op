"""Server-side complete-cohort checks and compact paired budget contrasts."""
import argparse
import gzip
import itertools
import json
from pathlib import Path

import numpy as np


def summarize(root, protocol):
    expected = set(itertools.product(protocol["seeds"], protocol["regimes"],
        protocol["difficulty_offsets"], protocol["arms"], protocol["allocations"]))
    shards = {}
    for path in root.glob("*/result.json"):
        shard = json.loads(path.read_text())
        row = shard["rows"][0]
        key = tuple(row[k] for k in ("seed", "regime", "difficulty", "arm", "allocation"))
        if key in shards:
            raise ValueError("duplicate experimental cell")
        shards[key] = shard
    if set(shards) != expected:
        raise ValueError(f"incomplete cohort: {len(shards)}/{len(expected)}")
    for seed, regime, difficulty in itertools.product(protocol["seeds"], protocol["regimes"], protocol["difficulty_offsets"]):
        initial = None
        for arm in protocol["arms"]:
            source_reference = None
            for allocation, budget in protocol["allocations"].items():
                shard = shards[seed, regime, difficulty, arm, allocation]
                row, p = shard["rows"][0], shard["protocol"]
                if any(p[k] != protocol[k] for k in ("candidates", "reference_bank_size", "initial_bank_size",
                    "chance_alpha", "replicates", "source_points", "source_tasks", "total_target_calls")):
                    raise ValueError("scientific protocol changed")
                if (row["target_search_calls"] != budget["target_points"] * protocol["replicates"]
                        or row["target_verification_calls"] != budget["verification_replicates"]
                        or row["total_target_calls"] != protocol["total_target_calls"]
                        or len(row["queries"]) != budget["target_points"]
                        or row["queries"][:8] != row["initial_design"]
                        or row["verification"]["replicates"] != budget["verification_replicates"]
                        or row["verification"]["confidence"] != .95):
                    raise ValueError("search/verification budget mismatch")
                wanted_source = 0 if arm == "target_only" else 6144
                if row["source_calls"] != wanted_source or not row["source_nonintercept_means_zero"]:
                    raise ValueError("source cost or coefficient-mean violation")
                if arm == "target_only" and any(info["source_label_count"] != 0 for info in shard["source_info"]):
                    raise ValueError("source labels reached target-only prior")
                if initial is None:
                    initial = row["initial_design"]
                elif row["initial_design"] != initial:
                    raise ValueError("initial paid design changed")
                if source_reference is None:
                    source_reference = shard["source_info"]
                else:
                    for a, b in zip(source_reference, shard["source_info"]):
                        if a["selected_names"] != b["selected_names"]:
                            raise ValueError("source selection changed across allocations")
                        if "source_function_variance" in a and not np.allclose(a["source_function_variance"], b["source_function_variance"], atol=1e-8, rtol=1e-8):
                            raise ValueError("source risk covariance changed across allocations")
    rng = np.random.default_rng(95006)
    bootstrap = rng.integers(len(protocol["seeds"]), size=(20000, len(protocol["seeds"])))
    metrics = ("expected_verified_reward", "realized_verified_reward", "true_verifier_power",
        "predicted_verifier_power", "best_available_expected_verified_reward", "expected_search_shortfall")
    groups, contrasts, compact = [], [], []
    for regime, difficulty in itertools.product(protocol["regimes"], protocol["difficulty_offsets"]):
        cells = {}
        for arm, allocation in itertools.product(protocol["arms"], protocol["allocations"]):
            ss = [shards[seed, regime, difficulty, arm, allocation] for seed in protocol["seeds"]]
            rr = [s["rows"][0] for s in ss]
            cells[arm, allocation] = rr
            group = {"regime": regime, "difficulty": difficulty, "arm": arm,
                "allocation": allocation, "seed_blocks": len(rr),
                "verified": sum(r["verification"]["verified"] for r in rr),
                "false_verified": sum(r["false_verified"] for r in rr),
                "mean_compute_seconds": float(np.mean([s["elapsed_seconds"] for s in ss])),
                "mean_rmse": np.mean([r["final"]["rmse"] for r in rr], axis=0).tolist()}
            group.update({k: float(np.mean([r[k] for r in rr])) for k in metrics})
            groups.append(group)
            for r in rr:
                compact.append({**{k: r[k] for k in ("seed", "regime", "difficulty", "arm", "allocation",
                    "recommendation", "true_feasible", "true_margin", "false_verified", *metrics)},
                    "verified": r["verification"]["verified"]})
        comparisons = [("primary_allocation", ("risk_ard_mixture", "n24_v928"), ("fixed_source", "n80_v256"))]
        comparisons.extend(("same_allocation", ("risk_ard_mixture", a), (control, a))
            for a in protocol["allocations"] for control in ("fixed_source", "target_only"))
        for label, left, right in comparisons:
            aa, bb = cells[left], cells[right]
            delta = np.array([a["expected_verified_reward"] - b["expected_verified_reward"] for a, b in zip(aa, bb)])
            contrasts.append({"regime": regime, "difficulty": difficulty, "comparison": label,
                "left": list(left), "right": list(right), "mean_reward_difference": float(delta.mean()),
                "paired_seed_bootstrap95": np.quantile(delta[bootstrap].mean(axis=1), [.025, .975]).tolist(),
                "seed_reward_differences": delta.tolist(), "better": int(np.sum(delta > 1e-12)),
                "worse": int(np.sum(delta < -1e-12)),
                "realized_reward_difference": float(np.mean([a["realized_verified_reward"] - b["realized_verified_reward"] for a, b in zip(aa, bb)])),
                "verified_difference": sum(a["verification"]["verified"] - b["verification"]["verified"] for a, b in zip(aa, bb)),
                "validation_opportunity_difference": float(np.mean([a["best_available_expected_verified_reward"] - b["best_available_expected_verified_reward"] for a, b in zip(aa, bb)])),
                "search_shortfall_reduction": float(np.mean([b["expected_search_shortfall"] - a["expected_search_shortfall"] for a, b in zip(aa, bb)]))})
    return {"stage": protocol["stage"], "complete_cells": len(shards),
        "independent_seed_blocks": len(protocol["seeds"]), "fixed_budget_and_source_pairing_passed": True,
        "total_target_calls_per_cell": protocol["total_target_calls"], "false_verified": sum(r["false_verified"] for r in compact),
        "analysis": "Nominal unadjusted paired seed intervals; all strata retained. No truth-selected allocation policy.",
        "groups": groups, "contrasts": contrasts, "rows": compact}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.results_dir, json.loads(args.protocol.read_text()))
    data = (json.dumps(result, allow_nan=False) + "\n").encode()
    args.output.write_bytes(data)
    args.output.with_suffix(args.output.suffix + ".gz").write_bytes(gzip.compress(data))
    print(json.dumps({k: result[k] for k in ("complete_cells", "fixed_budget_and_source_pairing_passed", "false_verified")}))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
