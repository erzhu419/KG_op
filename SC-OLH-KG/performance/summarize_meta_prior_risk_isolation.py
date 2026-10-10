"""Complete paired risk attribution and fixed-budget curves on the server."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(paths):
    shards = [json.loads(p.read_text()) for p in paths]
    p = shards[0]["protocol"]
    rows = [r for s in shards for r in s["rows"]]
    expected = {(seed, regime, d, a, c, n) for seed in p["seeds"] for regime in p["regimes"]
                for d in p["difficulty_offsets"] for a in p["arms"] for c in p["mean_contexts"] for n in p["checkpoints"]}
    key = lambda r: (r["seed"], r["regime"], r["difficulty"], r["arm"], r["mean_context"], r["batches"])
    lookup = {key(r): r for r in rows}
    if len(lookup) != len(rows) or set(lookup) != expected or any(s["protocol"] != p for s in shards):
        raise ValueError("missing, duplicate or mixed-protocol algorithm checkpoints")
    banks = {}
    for s in shards:
        r = s["rows"][0]
        bank = (r["seed"], r["regime"], r["difficulty"])
        reference = banks.setdefault(bank, s)
        if s["initial_design"] != reference["initial_design"]:
            raise ValueError("unmatched initial design")
        for c in p["mean_contexts"]:
            if len(s["queries"][c]) != p["target_points"] or s["queries"][c][:8] != s["initial_design"]:
                raise ValueError("incorrect paid trajectory budget")
            for field in ("mean", "variance"):
                if not np.allclose(s["fixed_contexts"][c][field], reference["fixed_contexts"][c][field], atol=1e-10, rtol=1e-10):
                    raise ValueError("risk arms used different mean information")
        for field in s["truth_summary"]:
            if not np.isclose(s["truth_summary"][field], reference["truth_summary"][field], atol=1e-12):
                raise ValueError("different truth bank across paired risk arms")
    if any(r["search_calls"] != r["batches"] * 12 or r["verification_calls"] != 256
           or r["risk_source_calls"] != (0 if r["arm"] == "target_only" else 6144)
           or r["source_archive_calls"] != 6144 for r in rows):
        raise ValueError("unmatched budget accounting")
    rng = np.random.default_rng(94002)
    bootstrap = rng.integers(len(p["seeds"]), size=(20000, len(p["seeds"])))
    grouped, contrasts, integrated = [], [], []
    metrics = ("expected_verified_reward", "finite_bank_shortfall", "realized_verified_reward",
               "risk_rmse", "risk_nominal90_coverage", "verifier_power_absolute_error")
    for regime in p["regimes"]:
        for d in p["difficulty_offsets"]:
            for c in p["mean_contexts"]:
                for n in p["checkpoints"]:
                    for arm in p["arms"]:
                        rr = [lookup[seed, regime, d, arm, c, n] for seed in p["seeds"]]
                        group = {"regime": regime, "difficulty": d, "mean_context": c, "batches": n, "arm": arm,
                                 "tasks": len(rr), "verified": sum(r["verification"]["verified"] for r in rr),
                                 "false_verified": sum(r["false_verified"] for r in rr)}
                        group.update({metric: float(np.mean([r[metric] for r in rr])) for metric in metrics})
                        grouped.append(group)
                    for arm, comparator in (("risk_ard", "fixed_source"), ("risk_ard_mixture", "fixed_source"),
                                             ("oracle_structure", "fixed_source"), ("fixed_source", "target_only")):
                        a = [lookup[seed, regime, d, arm, c, n] for seed in p["seeds"]]
                        b = [lookup[seed, regime, d, comparator, c, n] for seed in p["seeds"]]
                        delta = np.array([x["expected_verified_reward"] - y["expected_verified_reward"] for x, y in zip(a, b)])
                        contrasts.append({"regime": regime, "difficulty": d, "mean_context": c, "batches": n,
                                          "arm": arm, "comparator": comparator, "reward_difference": float(delta.mean()),
                                          "paired_seed_bootstrap95": np.quantile(delta[bootstrap].mean(axis=1), [.025, .975]).tolist(),
                                          "seed_differences": delta.tolist(), "better": int(np.sum(delta > 1e-12)),
                                          "worse": int(np.sum(delta < -1e-12)),
                                          "verified_difference": sum(x["verification"]["verified"]-y["verification"]["verified"] for x, y in zip(a, b))})
                for arm in p["arms"]:
                    values = []
                    for seed in p["seeds"]:
                        rr = [lookup[seed, regime, d, arm, c, n] for n in p["checkpoints"]]
                        integral = sum((y["batches"]-x["batches"]) * (x["finite_bank_shortfall"]+y["finite_bank_shortfall"]) / 2
                                       for x, y in zip(rr, rr[1:])) / (p["target_points"]-p["initial_points"])
                        values.append(float(integral))
                    integrated.append({"regime": regime, "difficulty": d, "mean_context": c, "arm": arm,
                                       "mean_integrated_shortfall": float(np.mean(values)), "seed_values": values})
    return {"protocol": p, "complete_shards": len(shards), "complete_algorithm_checkpoints": len(rows),
            "paired_information_and_budget_checks_passed": True, "grouped": grouped, "contrasts": contrasts,
            "integrated_shortfall": integrated, "rows": rows,
            "banks": [{"seed": k[0], "regime": k[1], "difficulty": k[2], **s["truth_summary"],
                       "initial_mean_rmse": s["fixed_contexts"]["frozen_initial_posterior"]["rmse"]} for k, s in banks.items()]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(sorted(args.input_dir.glob("*/result.json")))
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete_shards", "complete_algorithm_checkpoints", "paired_information_and_budget_checks_passed")}))
    print("DONE")


if __name__ == "__main__":
    main()
