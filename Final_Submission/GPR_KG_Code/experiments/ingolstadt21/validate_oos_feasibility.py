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
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.ingolstadt21.config import ALPHA, RESULTS_DIR, TAU_EMISSION
from experiments.ingolstadt21.ingolstadt21_problem import Ingolstadt21Problem


RESULTS_PATH = Path(RESULTS_DIR)
_WORKER_PROB: Ingolstadt21Problem | None = None


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


def load_candidates(
    method: str | None,
    partition: str | None,
    dedupe: str = "by_x",
    source_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
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
            if source_indices is not None and idx not in source_indices:
                continue
            x_tuple = tuple(int(v) for v in x)
            if dedupe == "by_x" and x_tuple in seen:
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


def parse_source_indices(text: str | None) -> set[int] | None:
    values = [
        item.strip()
        for item in str(text or "").split(",")
        if item.strip()
    ]
    if not values:
        return None
    return {int(item) for item in values}


def _init_worker(backend: str = "auto") -> None:
    if backend != "auto":
        os.environ["INGOLSTADT21_SUMO_BACKEND"] = backend
    global _WORKER_PROB
    _WORKER_PROB = Ingolstadt21Problem()


def _validate_candidate_worker(payload: tuple[int, dict[str, Any], list[int]]) -> tuple[int, dict[str, Any]]:
    global _WORKER_PROB
    if _WORKER_PROB is None:
        _WORKER_PROB = Ingolstadt21Problem()
    idx, cand, seeds = payload
    return idx, validate_candidate(_WORKER_PROB, cand["x"], seeds)


def _simulate_seed_worker(payload: tuple[int, dict[str, Any], int]) -> tuple[int, int, list[float], float]:
    global _WORKER_PROB
    if _WORKER_PROB is None:
        _WORKER_PROB = Ingolstadt21Problem()
    idx, cand, seed = payload
    prob = _WORKER_PROB
    t0 = time.perf_counter()
    y = prob._sim.simulate(
        var_map=prob.var_map,
        x=np.array(cand["x"], dtype=float),
        route_file=prob._route_files[0],
        T0=prob.T0,
        A0=prob.A0,
        E0=prob.E0,
        seed=seed,
    )
    return idx, int(seed), [float(v) for v in y], float(time.perf_counter() - t0)


def summarize_observations(rows: list[dict[str, Any]], tau: float, alpha: float) -> dict[str, Any]:
    by_seed = {int(row["seed"]): row for row in rows}
    rows = [by_seed[seed] for seed in sorted(by_seed)]
    seeds = [int(row["seed"]) for row in rows]
    arr = np.array([row["y"] for row in rows], dtype=float)
    sim_times = np.array([float(row.get("sim_time_sec", 0.0)) for row in rows], dtype=float)
    feasible = arr[:, 2] <= tau
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
        "passes_chance_constraint_point_estimate": successes / len(seeds) >= (1 - alpha),
        "passes_chance_constraint_wilson_lower": lo >= (1 - alpha),
        "mean_sim_time_sec": float(np.mean(sim_times)) if len(sim_times) else 0.0,
        "total_sim_time_sec": float(np.sum(sim_times)) if len(sim_times) else 0.0,
    }


def validate_candidate(prob: Ingolstadt21Problem, x: list[int], seeds: list[int]) -> dict[str, Any]:
    ys = []
    sim_times = []
    for seed in seeds:
        t0 = time.perf_counter()
        y = prob._sim.simulate(
            var_map=prob.var_map,
            x=np.array(x, dtype=float),
            route_file=prob._route_files[0],
            T0=prob.T0,
            A0=prob.A0,
            E0=prob.E0,
            seed=seed,
        )
        sim_times.append(time.perf_counter() - t0)
        ys.append([float(v) for v in y])
    rows = [
        {"seed": int(seed), "y": y, "sim_time_sec": float(sim_time)}
        for seed, y, sim_time in zip(seeds, ys, sim_times)
    ]
    return summarize_observations(rows, prob.tau, prob.alpha)


def _load_resume(
    path: Path,
    candidates: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    if not path.exists():
        return {}, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}
    candidate_index = {
        tuple(int(v) for v in cand["x"]): idx
        for idx, cand in enumerate(candidates)
    }
    by_x = {
        tuple(int(v) for v in row.get("x", [])): row
        for row in payload.get("candidates", [])
        if "validation" in row
    }
    resumed: dict[int, dict[str, Any]] = {}
    for x_tuple, row in by_x.items():
        idx = candidate_index.get(x_tuple)
        if idx is not None:
            resumed[idx] = row

    partials: dict[int, list[dict[str, Any]]] = {}
    for row in payload.get("partial_observations", []):
        idx = candidate_index.get(tuple(int(v) for v in row.get("x", [])))
        if idx is None or idx in resumed:
            continue
        partials[idx] = list(row.get("rows", []))
    return resumed, partials


def _write_payload(
    out_path: Path,
    base_payload: dict[str, Any],
    completed: dict[int, dict[str, Any]],
    partials: dict[int, list[dict[str, Any]]] | None = None,
    candidates: list[dict[str, Any]] | None = None,
) -> None:
    payload = dict(base_payload)
    payload["candidates"] = [completed[idx] for idx in sorted(completed)]
    if partials and candidates is not None:
        payload["partial_observations"] = [
            {
                **candidates[idx],
                "n_done": len(rows),
                "rows": sorted(rows, key=lambda row: int(row["seed"])),
            }
            for idx, rows in sorted(partials.items())
            if idx not in completed and rows
        ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--R", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=50000)
    parser.add_argument("--seed-mode", choices=["common", "blocked"], default="common")
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument(
        "--source-indexes",
        default="",
        help=(
            "Comma-separated final_pareto_set source indices to validate. "
            "Use 0 for final recommendations only."
        ),
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dedupe", choices=["by_x", "none"], default="by_x")
    parser.add_argument("--out", default=str(RESULTS_PATH / "oos_feasibility_validation.json"))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--backend", choices=["auto", "libsumo", "traci"], default="auto")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_indices = parse_source_indices(args.source_indexes)
    candidates = load_candidates(
        args.method,
        args.partition,
        dedupe=args.dedupe,
        source_indices=source_indices,
    )
    if args.max_points is not None:
        candidates = candidates[: args.max_points]
    for global_idx, cand in enumerate(candidates):
        cand["candidate_index"] = int(global_idx)

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.num_shards > 1:
        candidates = [
            cand for cand in candidates
            if int(cand["candidate_index"]) % args.num_shards == args.shard_index
        ]

    print(f"Loaded {len(candidates)} unique final Pareto candidates", flush=True)
    if args.dry_run:
        for idx, c in enumerate(candidates):
            print(f"{idx:03d}: {c['method']} {c['partition']} seed={c['run_seed']}", flush=True)
        return

    out_path = Path(args.out)
    t_start = time.perf_counter()
    base_payload: dict[str, Any] = {
        "source": "experiments.ingolstadt21.validate_oos_feasibility",
        "R": args.R,
        "seed_start": args.seed_start,
        "seed_mode": args.seed_mode,
        "tau": TAU_EMISSION,
        "alpha": ALPHA,
        "target_probability": 1 - ALPHA,
        "jobs": max(1, int(args.jobs)),
        "backend": args.backend,
        "progress_every": max(1, int(args.progress_every)),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "dedupe": args.dedupe,
        "source_indices": (
            None if source_indices is None else sorted(int(v) for v in source_indices)
        ),
        "candidates": [],
    }

    completed: dict[int, dict[str, Any]] = {}
    partials: dict[int, list[dict[str, Any]]] = {}
    if args.resume:
        completed_rows, partial_rows = _load_resume(out_path, candidates)
        completed.update(completed_rows)
        partials.update(partial_rows)
        if completed:
            print(f"Resumed {len(completed)} completed candidates from {out_path}", flush=True)
        if partials:
            n_rows = sum(len(rows) for rows in partials.values())
            print(f"Resumed {n_rows} partial seed observations from {out_path}", flush=True)

    pending = []
    for idx, cand in enumerate(candidates):
        if idx in completed:
            continue
        global_idx = int(cand.get("candidate_index", idx))
        if args.seed_mode == "blocked":
            seed_base = args.seed_start + global_idx * args.R
        else:
            seed_base = args.seed_start
        seeds = [seed_base + j for j in range(args.R)]
        seen = {int(row["seed"]) for row in partials.get(idx, [])}
        pending_seeds = [seed for seed in seeds if seed not in seen]
        if not pending_seeds and len(partials.get(idx, [])) >= args.R:
            completed[idx] = {
                **cand,
                "validation": summarize_observations(partials[idx], TAU_EMISSION, ALPHA),
            }
            continue
        pending.append((idx, cand, pending_seeds))

    if args.jobs <= 1:
        if args.backend != "auto":
            os.environ["INGOLSTADT21_SUMO_BACKEND"] = args.backend
        prob = Ingolstadt21Problem()
        for idx, cand, seeds in pending:
            print(
                f"[{idx + 1}/{len(candidates)}] validating "
                f"{cand['method']} {cand['partition']} run_seed={cand['run_seed']}",
                flush=True,
            )
            for seed in seeds:
                t0 = time.perf_counter()
                y = prob._sim.simulate(
                    var_map=prob.var_map,
                    x=np.array(cand["x"], dtype=float),
                    route_file=prob._route_files[0],
                    T0=prob.T0,
                    A0=prob.A0,
                    E0=prob.E0,
                    seed=seed,
                )
                sim_time = time.perf_counter() - t0
                partials.setdefault(idx, []).append({
                    "seed": int(seed),
                    "y": [float(v) for v in y],
                    "sim_time_sec": float(sim_time),
                })
                if len(partials[idx]) >= args.R and idx not in completed:
                    completed[idx] = {
                        **cand,
                        "validation": summarize_observations(
                            partials[idx], TAU_EMISSION, ALPHA
                        ),
                    }
                _write_payload(out_path, base_payload, completed, partials, candidates)
    else:
        ctx = mp.get_context("spawn")
        seed_tasks = [
            (idx, cand, seed)
            for idx, cand, seeds in pending
            for seed in seeds
        ]
        total_tasks = len(seed_tasks)
        print(f"Submitting {total_tasks} seed-level SUMO tasks", flush=True)
        with ProcessPoolExecutor(
            max_workers=int(args.jobs),
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(args.backend,),
        ) as pool:
            future_map = {
                pool.submit(_simulate_seed_worker, item): item
                for item in seed_tasks
            }
            for done, fut in enumerate(as_completed(future_map), start=1):
                idx, cand, _ = future_map[fut]
                result_idx, seed, y, sim_time = fut.result()
                partials.setdefault(result_idx, []).append({
                    "seed": int(seed),
                    "y": y,
                    "sim_time_sec": float(sim_time),
                })
                n_done = len(partials[result_idx])
                should_print = (
                    done == 1
                    or done == total_tasks
                    or done % max(1, int(args.progress_every)) == 0
                    or n_done >= args.R
                )
                if should_print:
                    print(
                        f"[{done}/{total_tasks} seeds] candidate {idx + 1}/{len(candidates)} "
                        f"{cand['method']} {cand['partition']} run_seed={cand['run_seed']} "
                        f"n={n_done}/{args.R} seed={seed} sim={sim_time:.1f}s",
                        flush=True,
                    )
                if n_done >= args.R and result_idx not in completed:
                    completed[result_idx] = {
                        **candidates[result_idx],
                        "validation": summarize_observations(
                            partials[result_idx], TAU_EMISSION, ALPHA
                        ),
                    }
                _write_payload(out_path, base_payload, completed, partials, candidates)

    elapsed = time.perf_counter() - t_start
    base_payload["wall_time_sec"] = elapsed
    _write_payload(out_path, base_payload, completed, partials, candidates)

    print(f"Wrote {out_path}", flush=True)
    print(f"Validated {len(completed)} candidates in {elapsed:.1f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
