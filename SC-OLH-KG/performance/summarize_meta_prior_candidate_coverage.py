"""Pair complete 96/384 bank experiments without pooling historical cohorts."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(small_root, large_root):
    def read(root):
        shards = [json.loads(p.read_text()) for p in sorted(root.glob("*/result.json"))]
        return {(s["rows"][0]["seed"], s["rows"][0]["regime"], s["rows"][0]["difficulty"]): s for s in shards}
    small, large = read(small_root), read(large_root)
    p = next(iter(small.values()))["protocol"]
    expected = {(seed, regime, d) for seed in p["seeds"] for regime in p["regimes"] for d in p["difficulty_offsets"]}
    if set(small) != expected or set(large) != expected:
        raise ValueError("missing paired candidate banks")
    rows = []
    for key in sorted(expected):
        a, b = small[key], large[key]
        if a["rows"][0]["initial_design"] != b["rows"][0]["initial_design"]:
            raise ValueError("candidate expansion changed initial observations")
        for arm in p["arms"]:
            aa = next(r for r in a["rows"] if r["arm"] == arm and r["scheme"] == "verification_kg")
            bb = next(r for r in b["rows"] if r["arm"] == arm and r["scheme"] == "verification_kg")
            if aa["oracle_support"] != bb["oracle_support"] or aa["source_calls"] != bb["source_calls"] or aa["target_search_calls"] != bb["target_search_calls"] or aa["target_verification_calls"] != bb["target_verification_calls"]:
                raise ValueError("expansion changed function support or simulation budget")
            for h in range(3):
                ia, ib = a["source_info"][arm][h], b["source_info"][arm][h]
                if ia["selected_names"] != ib["selected_names"]:
                    raise ValueError("expansion changed source function selection")
                if "source_function_variance" in ia and not np.allclose(ia["source_function_variance"], ib["source_function_variance"], atol=1e-8, rtol=1e-8):
                    raise ValueError("expansion changed fitted source risk covariance")
            upper_a, upper_b = aa["best_available_expected_verified_reward"], bb["best_available_expected_verified_reward"]
            if upper_b < upper_a - 1e-12:
                raise ValueError("expanded bank does not contain the old truth bank")
            rows.append({"seed": key[0], "regime": key[1], "difficulty": key[2], "arm": arm,
                         "small_reward": aa["expected_verified_reward"], "large_reward": bb["expected_verified_reward"],
                         "reward_difference": bb["expected_verified_reward"]-aa["expected_verified_reward"],
                         "small_shortfall": upper_a-aa["expected_verified_reward"],
                         "large_shortfall": upper_b-bb["expected_verified_reward"],
                         "coverage_opportunity": upper_b-upper_a,
                         "small_verified": aa["verification"]["verified"], "large_verified": bb["verification"]["verified"],
                         "expanded_recommendation": bb["recommendation"],
                         "new_candidates_queried": sum(i >= 96 for i in bb["queries"][8:]),
                         "small_bank_shard_compute_seconds": a["elapsed_seconds"],
                         "large_bank_shard_compute_seconds": b["elapsed_seconds"]})
    rng = np.random.default_rng(94003)
    bootstrap = rng.integers(len(p["seeds"]), size=(20000, len(p["seeds"])))
    grouped = []
    for regime in p["regimes"]:
        for d in p["difficulty_offsets"]:
            for arm in p["arms"]:
                rr = sorted([r for r in rows if r["regime"] == regime and r["difficulty"] == d and r["arm"] == arm], key=lambda r:r["seed"])
                delta = np.array([r["reward_difference"] for r in rr])
                group = {"regime": regime, "difficulty": d, "arm": arm, "tasks": len(rr),
                         "small_verified": sum(r["small_verified"] for r in rr),
                         "large_verified": sum(r["large_verified"] for r in rr),
                         "recommendations_outside_old_bank": sum(r["expanded_recommendation"] >= 96 for r in rr),
                         "better": int(np.sum(delta > 1e-12)), "worse": int(np.sum(delta < -1e-12)),
                         "seed_reward_differences": delta.tolist(),
                         "paired_seed_bootstrap95": np.quantile(delta[bootstrap].mean(axis=1), [.025, .975]).tolist()}
                group.update({metric: float(np.mean([r[metric] for r in rr])) for metric in (
                    "small_reward", "large_reward", "reward_difference", "small_shortfall", "large_shortfall", "coverage_opportunity",
                    "small_bank_shard_compute_seconds", "large_bank_shard_compute_seconds")})
                grouped.append(group)
    return {"complete_paired_banks": len(expected), "independent_seed_blocks": len(p["seeds"]),
            "paired_initial_source_and_budget_checks_passed": True, "grouped": grouped, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-dir", type=Path, required=True)
    parser.add_argument("--large-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.small_dir, args.large_dir)
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete_paired_banks", "paired_initial_source_and_budget_checks_passed")}))
    print("DONE")


if __name__ == "__main__":
    main()
