"""Fresh-seed out-of-sample feasibility validation for ingolstadt21.

This script performs the expensive validation that the manuscript currently
lists as required follow-up.  It reruns SUMO for returned Pareto points using
fresh seeds and estimates Pr(f3 <= tau) with Wilson confidence intervals.

Example
-------
python -m experiments.ingolstadt21.validate_oos_feasibility --R 100

Useful dry run
--------------
python -m experiments.ingolstadt21.validate_oos_feasibility --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.ingolstadt21.config import RESULTS_DIR
from experiments.ingolstadt21.ingolstadt21_problem import Ingolstadt21Problem


RESULTS_PATH = Path(RESULTS_DIR)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    radius = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - radius), min(1.0, centre + radius)


def canonical_id(run_dir: Path, summary: dict[str, Any]) -> tuple[str, str, int]:
    name = run_dir.name
    if name == "GPR_KG_run":
        return "GPR-KG", "binary_bin", 100
    if name == "GPR_KG_nV_run":
        return "GPR-KG-nV", "binary_bin", 100
    method = str(summary.get("method", ""))
    partition = str(summary.get("partition_method", ""))
    seed = int(summary.get("seed", -1))
    return method, partition, seed


def load_candidates(method: str | None, partition: str | None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for summary_path in sorted(RESULTS_PATH.glob("*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        m, p, seed = canonical_id(summary_path.parent, summary)
        if method and m != method:
            continue
        if partition and p != partition:
            continue
        for idx, x in enumerate(summary.get("final_pareto_set", [])):
            x_tuple = tuple(int(v) for v in x)
            if x_tuple in seen:
                continue
            seen.add(x_tuple)
            candidates.append(
                {
                    "method": m,
                    "partition": p,
                    "run_seed": seed,
                    "source": str(summary_path),
                    "source_index": idx,
                    "x": list(x_tuple),
                }
            )
    return candidates


def validate_candidate(prob: Ingolstadt21Problem, x: list[int], seeds: list[int]) -> dict[str, Any]:
    ys = []
    for seed in seeds:
        y = prob._sim.simulate(
            var_map=prob.var_map,
            x=np.array(x, dtype=float),
            route_file=prob._route_files[0],
            T0=prob.T0,
            A0=prob.A0,
            E0=prob.E0,
            seed=seed,
        )
        ys.append([float(v) for v in y])
    arr = np.array(ys, dtype=float)
    feasible = arr[:, 2] <= prob.tau
    successes = int(feasible.sum())
    lo, hi = wilson_interval(successes, len(seeds))
    return {
        "R": len(seeds),
        "seeds": seeds,
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0, ddof=1).tolist() if len(seeds) > 1 else [0.0, 0.0, 0.0],
        "feasible_count": successes,
        "feasible_probability": successes / len(seeds),
        "wilson_95": [lo, hi],
        "passes_chance_constraint_point_estimate": successes / len(seeds) >= (1 - prob.alpha),
        "passes_chance_constraint_wilson_lower": lo >= (1 - prob.alpha),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["GPR-KG", "GPR-KG-nV"], default=None)
    parser.add_argument("--partition", choices=["binary_bin", "aggregate", "medoid_K"], default=None)
    parser.add_argument("--R", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=50000)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--out", default=str(RESULTS_PATH / "oos_feasibility_validation.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    candidates = load_candidates(args.method, args.partition)
    if args.max_points is not None:
        candidates = candidates[: args.max_points]

    print(f"Loaded {len(candidates)} unique final Pareto candidates")
    if args.dry_run:
        for idx, c in enumerate(candidates):
            print(f"{idx:03d}: {c['method']} {c['partition']} seed={c['run_seed']}")
        return

    prob = Ingolstadt21Problem()
    out_path = Path(args.out)
    payload: dict[str, Any] = {
        "source": "experiments.ingolstadt21.validate_oos_feasibility",
        "R": args.R,
        "seed_start": args.seed_start,
        "tau": prob.tau,
        "alpha": prob.alpha,
        "target_probability": 1 - prob.alpha,
        "candidates": [],
    }

    for idx, cand in enumerate(candidates):
        seeds = [args.seed_start + idx * args.R + j for j in range(args.R)]
        print(
            f"[{idx + 1}/{len(candidates)}] validating "
            f"{cand['method']} {cand['partition']} run_seed={cand['run_seed']}"
        )
        result = validate_candidate(prob, cand["x"], seeds)
        payload["candidates"].append({**cand, "validation": result})
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
