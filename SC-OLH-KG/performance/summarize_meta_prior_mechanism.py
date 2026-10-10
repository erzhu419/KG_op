"""Aggregate complete frozen mechanism shards; return compact evidence only."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(paths):
    shards = [json.loads(path.read_text()) for path in paths]
    protocol = shards[0]["protocol"]
    rows = [row for shard in shards for row in shard["rows"]]
    expected = {(seed, regime, arm) for seed in protocol["seeds"]
                for regime in protocol["regimes"] for arm in protocol["arms"]}
    keys = [(r["seed"], r["regime"], r["arm"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("missing or duplicate frozen cells")
    if any(shard["protocol"] != protocol for shard in shards):
        raise ValueError("mixed protocols")
    pairs = {(r["seed"], r["regime"], r["arm"]): r for r in rows}
    changes = {"posterior_rmse_changed": 0, "first_online_query_changed": 0,
               "online_sequence_changed": 0, "recommendation_changed": 0}
    for seed in protocol["seeds"]:
        for regime in protocol["regimes"]:
            cells = [pairs[seed, regime, arm] for arm in protocol["arms"]]
            for cell in cells:
                if cell["initial_design"] != cells[0]["initial_design"]:
                    raise ValueError("unmatched initial designs")
                if cell["queries"][:protocol["initial_points"]] != cell["initial_design"]:
                    raise ValueError("initial design was not executed")
                if len(cell["queries"]) != protocol["target_points"]:
                    raise ValueError("unmatched search budgets")
            learned, fixed = pairs[seed, regime, "learned"], pairs[seed, regime, "fixed_source"]
            n0 = protocol["initial_points"]
            changes["posterior_rmse_changed"] += int(not np.allclose(
                learned["initial"]["rmse"], fixed["initial"]["rmse"]))
            changes["first_online_query_changed"] += int(learned["queries"][n0] != fixed["queries"][n0])
            changes["online_sequence_changed"] += int(learned["queries"][n0:] != fixed["queries"][n0:])
            changes["recommendation_changed"] += int(learned["recommendation"] != fixed["recommendation"])
    grouped = []
    for regime in protocol["regimes"]:
        for arm in protocol["arms"]:
            subset = [r for r in rows if r["regime"] == regime and r["arm"] == arm]
            out = {"regime": regime, "arm": arm, "tasks": len(subset),
                   "true_feasible": sum(r["true_feasible"] for r in subset),
                   "posterior_safe": sum(r["posterior_safe"] for r in subset),
                   "verified": sum(r["verification"]["verified"] for r in subset),
                   "mean_penalized_regret": float(np.mean([r["penalized_regret"] for r in subset]))}
            for phase in ("initial", "final"):
                for metric in ("rmse", "nominal90_coverage", "risk_variance_relative_mae",
                               "margin_rank_correlation", "posterior_safe_count", "false_safe_count"):
                    out[f"{phase}_{metric}"] = np.mean([r[phase][metric] for r in subset], axis=0).tolist()
            grouped.append(out)
    # Seeds, not candidate rows, are the independent units for paired uncertainty.
    rng = np.random.default_rng(81001)
    bootstrap = rng.integers(0, len(protocol["seeds"]), (20000, len(protocol["seeds"])))
    contrasts = []
    for comparator in ("fixed_source", "target_only", "oracle_structure"):
        for metric in ("initial_objective_rmse", "initial_constraint_rmse", "initial_risk_rmse",
                       "final_objective_rmse", "final_constraint_rmse", "final_risk_rmse",
                       "penalized_regret", "true_feasible", "verified"):
            values = []
            for seed in protocol["seeds"]:
                a, b = pairs[seed, "shared_structure", "learned"], pairs[seed, "shared_structure", comparator]
                def value(row):
                    if "rmse" in metric:
                        phase, head, _ = metric.split("_")
                        return row[phase]["rmse"][["objective", "constraint", "risk"].index(head)]
                    return float(row[metric]) if metric != "verified" else float(row["verification"]["verified"])
                values.append(value(a) - value(b))
            values = np.asarray(values)
            contrast = {"comparator": comparator, "metric": metric,
                        "learned_minus_comparator": float(values.mean()),
                        "paired_seed_bootstrap95": np.quantile(values[bootstrap].mean(axis=1), [0.025, 0.975]).tolist(),
                        "seed_differences": values.tolist()}
            contrasts.append(contrast)
    compact = []
    for row in rows:
        keep = {k: row[k] for k in ("seed", "regime", "arm", "initial", "final", "recommendation",
                "true_feasible", "posterior_safe", "true_margin", "true_objective", "feasible_regret",
                "penalized_regret", "verification", "source_calls", "source_calls_amortized_four_targets",
                "target_search_calls", "target_verification_calls", "initial_design", "queries", "oracle_support")}
        keep["selected_names_by_head"] = (
            [p["selected_names"] for p in row["priors"]]
            if row["arm"] in {"learned", "oracle_structure"} else "public_full_dictionary")
        means = [p.get("coefficient_mean") for p in row["priors"]]
        keep["source_intercept_means_by_head"] = [None if m is None else m[0] for m in means]
        keep["source_nonintercept_means_zero"] = all(
            m is None or not np.any(np.asarray(m[1:])) for m in means)
        compact.append(keep)
    return {"protocol": protocol, "complete_cells": len(rows), "wiring_changes_out_of_32": changes,
            "public_dictionary_names": rows[0]["priors"][0]["selected_names"],
            "grouped": grouped, "shared_structure_paired_contrasts": contrasts, "rows": compact}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(sorted(args.input_dir.glob("seed*.json")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete_cells", "wiring_changes_out_of_32", "grouped")}), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
