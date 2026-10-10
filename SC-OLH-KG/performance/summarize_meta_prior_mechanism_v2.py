"""Small paired results from complete seed/regime/difficulty shards."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(paths):
    shards = [json.loads(p.read_text()) for p in paths]
    protocol = shards[0]["protocol"]
    rows = [r for s in shards for r in s["rows"]]
    expected = {(s, r, d, a) for s in protocol["seeds"] for r in protocol["regimes"]
                for d in protocol["difficulty_offsets"] for a in protocol["arms"]}
    keys = [(r["seed"], r["regime"], r["difficulty"], r["arm"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("missing or duplicate cells")
    if any(s["protocol"] != protocol for s in shards):
        raise ValueError("mixed protocols")
    paired = dict(zip(keys, rows))
    for seed, regime, difficulty, _ in expected:
        cells = [paired[seed, regime, difficulty, a] for a in protocol["arms"]]
        if any(c["initial_design"] != cells[0]["initial_design"]
               or c["queries"][:protocol["initial_points"]] != c["initial_design"]
               or len(c["queries"]) != protocol["target_points"]
               or c["target_search_calls"] != protocol["target_points"] * protocol["replicates"]
               or c["target_verification_calls"] != protocol["verification_replicates"]
               for c in cells):
            raise ValueError("unmatched paid design or budget")
    grouped, contrasts = [], []
    rng = np.random.default_rng(93001)
    boot = rng.integers(len(protocol["seeds"]), size=(20000, len(protocol["seeds"])))
    comparisons = [("conditional_hard", "fixed_source"), ("conditional_hard", "learned_v1"),
                   ("conditional_soft", "conditional_hard"),
                   ("conditional_adaptive", "conditional_soft")]
    for regime in protocol["regimes"]:
        for difficulty in protocol["difficulty_offsets"]:
            for arm in protocol["arms"]:
                cells = [paired[s, regime, difficulty, arm] for s in protocol["seeds"]]
                out = {"regime": regime, "difficulty": difficulty, "arm": arm,
                       "tasks": len(cells),
                       "true_feasible": sum(c["true_feasible"] for c in cells),
                       "verified": sum(c["verification"]["verified"] for c in cells),
                       "mean_penalized_regret": float(np.mean([c["penalized_regret"] for c in cells]))}
                for phase in ("initial", "final"):
                    for metric in ("rmse", "nominal90_coverage", "false_safe_count"):
                        out[f"{phase}_{metric}"] = np.mean([c[phase][metric] for c in cells], axis=0).tolist()
                grouped.append(out)
            for arm, comparator in comparisons:
                left = [paired[s, regime, difficulty, arm] for s in protocol["seeds"]]
                right = [paired[s, regime, difficulty, comparator] for s in protocol["seeds"]]
                delta = np.array([a["penalized_regret"] - b["penalized_regret"]
                                  for a, b in zip(left, right)])
                contrasts.append({"regime": regime, "difficulty": difficulty,
                    "arm": arm, "comparator": comparator,
                    "penalized_regret_difference": float(delta.mean()),
                    "paired_seed_bootstrap95": np.quantile(delta[boot].mean(axis=1), [.025, .975]).tolist(),
                    "seed_differences": delta.tolist(),
                    "recommendation_changed": sum(a["recommendation"] != b["recommendation"] for a, b in zip(left, right)),
                    "better_tasks": int(np.sum(delta < -1e-12)), "worse_tasks": int(np.sum(delta > 1e-12)),
                    "final_rmse_difference": np.mean([np.array(a["final"]["rmse"]) - b["final"]["rmse"]
                                                     for a, b in zip(left, right)], axis=0).tolist()})
    compact = []
    for row in rows:
        keep = {k: v for k, v in row.items() if k not in ("priors", "first_online_kg_scores")}
        keep["selected_names_by_head"] = [p.get("source_selected_names", p["selected_names"])
                                          for p in row["priors"]] if row["arm"] not in (
                                              "target_only", "fixed_source") else "public_dictionary"
        keep["source_nonintercept_means_zero"] = all(not np.any(np.asarray(
            p.get("coefficient_mean", [0])[1:])) for p in row["priors"])
        compact.append(keep)
    return {"protocol": protocol, "complete_cells": len(rows), "grouped": grouped,
            "paired_contrasts": contrasts, "rows": compact}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(sorted(args.input_dir.glob("*/result.json")))
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")
    print(json.dumps({"complete_cells": result["complete_cells"]}))
    print("DONE")


if __name__ == "__main__":
    main()
