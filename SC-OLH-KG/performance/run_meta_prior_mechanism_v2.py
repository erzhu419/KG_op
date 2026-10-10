"""One paired seed/regime/difficulty shard; frozen v1 data are never rewritten."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.run_meta_prior_mechanism import (  # noqa: E402
    observe, run_arm, shared_support, source_data, target_supports, task_truth,
)
from representation.source_function_prior import SourceFunctionPrior, head_batches  # noqa: E402
from representation.conditional_function_prior import (  # noqa: E402
    enable_target_adaptation, fit_conditional_priors,
)

PROTOCOL = ROOT / "performance/meta_prior_mechanism_v2_protocol.json"


def prepare_sources(seed, regime, protocol):
    tasks, support = source_data(seed, protocol)
    supports = target_supports(support, seed, regime, protocol["psi_dim"])
    if regime == "separate_head_structure":
        supports = [support, shared_support(seed + 1000, protocol["psi_dim"]),
                    target_supports(support, seed, "support_mismatch", protocol["psi_dim"])[0]]
        for task, data in enumerate(tasks):
            truth = task_truth(data["psi"], supports, 31000 + 100 * seed + task)
            labels = [observe(row, 41000 + 100 * seed + task, i, 0,
                              protocol["replicates"]) for i, row in enumerate(truth)]
            data["response"] = np.vstack([r for r, _ in labels])
            data["noise"] = np.vstack([v for _, v in labels])
    return tasks, supports


def run_shard(seed, regime, difficulty, protocol):
    start = time.monotonic()
    tasks, supports = prepare_sources(seed, regime, protocol)
    rng = np.random.default_rng(51000 + seed)
    psi = rng.uniform(-2.0, 2.0, (protocol["candidates"], protocol["psi_dim"]))
    candidates = np.arange(len(psi), dtype=int)[:, None]
    initial = rng.choice(len(psi), protocol["initial_points"], replace=False).tolist()
    batches = [head_batches(tasks, h) for h in range(3)]
    noise = [[task["noise"][:, h] for task in tasks] for h in range(3)]
    priors = {}
    for arm in ("target_only", "fixed_source", "learned_v1"):
        mode = "learned" if arm == "learned_v1" else arm
        priors[arm] = [SourceFunctionPrior.fit(batches[h], noise[h], psi, mode,
                      active_dim=protocol["learned_active_dim_per_head"]) for h in range(3)]
    conditional = [fit_conditional_priors(batches[h], noise[h], psi,
                   protocol["learned_active_dim_per_head"]) for h in range(3)]
    for arm in ("conditional_hard", "conditional_soft", "conditional_adaptive"):
        priors[arm] = [head[arm] for head in conditional]
    priors["oracle_structure"] = [SourceFunctionPrior.fit(batches[h], noise[h], psi,
                                  "oracle_structure", oracle_indices=supports[h]) for h in range(3)]
    target_seed = 61000 + seed * 100 + protocol["regimes"].index(regime)
    truth = task_truth(psi, supports, target_seed)
    truth[:, 1] += protocol["difficulty_offsets"][difficulty]
    true_margin = truth[:, 1] + norm.ppf(1 - protocol["chance_alpha"]) * np.exp(0.5 * truth[:, 2])
    rows = []
    for arm in protocol["arms"]:
        callback = None
        if arm == "conditional_adaptive":
            records = [[] for _ in range(3)]

            def callback(models, step, index, response, variance):
                for h in range(3):
                    records[h].append(([index], response[h], variance[h]))
                    if step + 1 == protocol["initial_points"]:
                        enable_target_adaptation(models[h], priors[arm][h], records[h])

        result = run_arm(priors[arm], initial, candidates, psi, truth, target_seed, protocol,
                         on_observation=callback)
        source_calls = 0 if arm == "target_only" else (protocol["source_tasks"]
                        * protocol["source_points"] * protocol["replicates"])
        result.update({"seed": seed, "regime": regime, "difficulty": difficulty,
                       "arm": arm, "initial_design": initial, "oracle_support": supports,
                       "true_feasible_fraction": float(np.mean(true_margin <= 0)),
                       "unconstrained_optimum_margin": float(true_margin[np.argmin(truth[:, 0])]),
                       "source_calls": source_calls,
                       "source_archive_group": "separate_heads" if regime == "separate_head_structure" else "shared",
                       "source_archive_target_count": 2 if regime == "separate_head_structure" else 8})
        # Only compact probabilities/counts are needed; coefficient fits stay remote.
        if "target_support" in result:
            result["target_support"] = [{k: d[k] for k in (
                "posterior_pip", "n_observations", "effective_dimension",
                "escaped_source_spike_count", "converged")} for d in result["target_support"]]
        rows.append(result)
        print(json.dumps({"seed": seed, "regime": regime, "difficulty": difficulty,
                          "arm": arm, "elapsed_seconds": round(time.monotonic() - start, 2)}), flush=True)
    return {"protocol": protocol, "rows": rows, "elapsed_seconds": time.monotonic() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--difficulty", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if (args.seed not in protocol["seeds"] or args.regime not in protocol["regimes"]
            or args.difficulty not in protocol["difficulty_offsets"]):
        raise ValueError("shard is outside the frozen protocol")
    result = run_shard(args.seed, args.regime, args.difficulty, protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
