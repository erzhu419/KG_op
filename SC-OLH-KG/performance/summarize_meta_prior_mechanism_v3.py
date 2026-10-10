"""Frozen matched analysis; simulation truth remains post-decision evidence."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(paths):
    shards = [json.loads(p.read_text()) for p in paths]
    protocol = shards[0]["protocol"]
    regimes = protocol.get("evaluation_regimes", protocol["regimes"])
    rows = [r for s in shards for r in s["rows"]]
    expected = {(s, r, d, a, k) for s in protocol["seeds"] for r in regimes
                for d in protocol["difficulty_offsets"] for a in protocol["arms"]
                for k in protocol["decision_schemes"]}
    keys = [(r["seed"], r["regime"], r["difficulty"], r["arm"], r["scheme"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("missing or duplicate cells")
    if any(s["protocol"] != protocol for s in shards):
        raise ValueError("mixed frozen protocols")
    paired = dict(zip(keys, rows))
    for seed in protocol["seeds"]:
        for regime in regimes:
            for difficulty in protocol["difficulty_offsets"]:
                cells = [paired[seed, regime, difficulty, arm, scheme] for arm in protocol["arms"]
                         for scheme in protocol["decision_schemes"]]
                if any(c["initial_design"] != cells[0]["initial_design"]
                    or c["queries"][:protocol["initial_points"]] != c["initial_design"]
                    or len(c["queries"]) != protocol["target_points"]
                    or c["target_search_calls"] != protocol["target_points"] * protocol["replicates"]
                    or c["target_verification_calls"] != 256
                    or c["source_calls"] != 6144 or not c["source_nonintercept_means_zero"]
                    for c in cells):
                    raise ValueError("unmatched paid design, budget or coefficient initialization")
                for arm in protocol["arms"]:
                    a = paired[seed, regime, difficulty, arm, "penalty"]
                    b = paired[seed, regime, difficulty, arm, "verification_recommendation"]
                    if a["queries"] != b["queries"] or a["final"] != b["final"]:
                        raise ValueError("recommendation-only control changed paid search or posterior")
    rng = np.random.default_rng(94001)
    bootstrap = rng.integers(len(protocol["seeds"]), size=(20000, len(protocol["seeds"])))
    grouped, contrasts = [], []
    for regime in regimes:
        for difficulty in protocol["difficulty_offsets"]:
            for arm in protocol["arms"]:
                for scheme in protocol["decision_schemes"]:
                    subset = [paired[s, regime, difficulty, arm, scheme] for s in protocol["seeds"]]
                    verified = [r for r in subset if r["verification"]["verified"]]
                    out = {"regime": regime, "difficulty": difficulty, "arm": arm, "scheme": scheme,
                           "tasks": len(subset), "verified": len(verified),
                           "true_feasible": sum(r["true_feasible"] for r in subset),
                           "false_verified": sum(r["false_verified"] for r in subset),
                           "mean_objective_if_verified": float(np.mean([r["true_objective"] for r in verified]))
                                if verified else None}
                    for metric in ("expected_verified_reward", "realized_verified_reward", "true_verifier_power",
                                   "predicted_verifier_power", "penalized_regret"):
                        out["mean_" + metric] = float(np.mean([r[metric] for r in subset]))
                    out["mean_verifier_power_absolute_error"] = float(np.mean([
                        abs(r["predicted_verifier_power"] - r["true_verifier_power"]) for r in subset]))
                    for phase in ("initial", "final"):
                        for metric in ("rmse", "nominal90_coverage", "false_safe_count"):
                            out[f"{phase}_{metric}"] = np.mean([r[phase][metric] for r in subset], axis=0).tolist()
                    grouped.append(out)
            comparisons = []
            for scheme in protocol["decision_schemes"]:
                for arm, comparator in (("risk_ard", "conditional_hard"),
                        ("adaptive_corrected", "adaptive_v2"), ("transfer_mixture", "conditional_hard"),
                        ("risk_ard_mixture", "fixed_source"), ("risk_ard_mixture", "risk_ard")):
                    if arm in protocol["arms"] and comparator in protocol["arms"]:
                        comparisons.append((arm, scheme, comparator, scheme))
            for arm in protocol["arms"]:
                comparisons += [(arm, "verification_recommendation", arm, "penalty"),
                                (arm, "verification_kg", arm, "verification_recommendation")]
            for arm, scheme, comparator, comparator_scheme in comparisons:
                left = [paired[s, regime, difficulty, arm, scheme] for s in protocol["seeds"]]
                right = [paired[s, regime, difficulty, comparator, comparator_scheme] for s in protocol["seeds"]]
                out = {"regime": regime, "difficulty": difficulty, "arm": arm, "scheme": scheme,
                       "comparator": comparator, "comparator_scheme": comparator_scheme,
                       "recommendation_changed": sum(a["recommendation"] != b["recommendation"]
                                                     for a, b in zip(left, right)),
                       "online_sequence_changed": sum(a["queries"][8:] != b["queries"][8:]
                                                      for a, b in zip(left, right))}
                for metric in ("expected_verified_reward", "realized_verified_reward", "penalized_regret"):
                    delta = np.array([a[metric] - b[metric] for a, b in zip(left, right)])
                    improvement = -delta if metric == "penalized_regret" else delta
                    out[metric] = {"difference": float(delta.mean()),
                        "paired_seed_bootstrap95": np.quantile(delta[bootstrap].mean(axis=1), [.025, .975]).tolist(),
                        "seed_differences": delta.tolist(),
                        "better": int(np.sum(improvement > 1e-12)), "worse": int(np.sum(improvement < -1e-12))}
                out["verified_count_difference"] = sum(a["verification"]["verified"]
                     - b["verification"]["verified"] for a, b in zip(left, right))
                out["final_rmse_difference"] = np.mean([np.array(a["final"]["rmse"]) - b["final"]["rmse"]
                                                       for a, b in zip(left, right)], axis=0).tolist()
                contrasts.append(out)
    # Raw source covariance vectors and per-step query traces stay in the remote
    # shards; retain small independent family records and decision outcomes.
    sources, compact = [], []
    for shard in shards:
        row = shard["rows"][0]
        if row["difficulty"] == "baseline" and row["regime"] in (
                "shared_structure", "separate_head_structure"):
            sources.append({"seed": row["seed"], "regime": row["regime"],
                "supports": row["oracle_support"], "source_info": {a: shard["source_info"][a]
                                        for a in ("conditional_hard", "risk_ard")}})
    for row in rows:
        compact.append({k: v for k, v in row.items() if k not in ("queries", "initial_design")})
    return {"protocol": protocol, "complete_cells": len(rows), "complete_shards": len(shards),
            "paired_checks_passed": True, "grouped": grouped, "contrasts": contrasts,
            "source_configs": sources, "rows": compact}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(sorted(args.input_dir.glob("*/result.json")))
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")
    print(json.dumps({"complete_cells": result["complete_cells"],
                      "complete_shards": result["complete_shards"], "paired_checks_passed": True}))
    print("DONE")


if __name__ == "__main__":
    main()
