"""Offline exact-KG score stability audit from a runtime checkpoint.

The script reconstructs the held-out LODO problem, restores the saved GPR,
HVD, task posterior, RNG and terminal-pool state, and regenerates the next
candidate set.  It never calls the simulator.  Synthetic truth is evaluated
only after scores are fixed to report ranking diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import (  # noqa: E402
    SingleOLHKGAlgorithm,
    SingleOLHKGConfig,
)
from performance.benchmark_lodo_meta_prior import (  # noqa: E402
    build_target_problem,
)
from performance.benchmark_quality import json_safe, parse_csv  # noqa: E402


def load_profile(path):
    payload = json.loads(Path(path).read_text())
    rows = list(payload.get("rows", []))
    if len(rows) != 1:
        raise ValueError("checkpoint diagnostic expects one row per shard profile")
    return payload["config"], rows[0]


def load_manifest(path, seed):
    payload = json.loads(Path(path).read_text())
    args_dict = dict(payload["config"])
    row = {
        "heldout": str(payload["heldout"]),
        "line": str(payload.get("line", "lodo_teacher")),
        "seed": int(seed),
    }
    return args_dict, row


def restore_algorithm(profile_path, manifest_path, seed, checkpoint_path):
    if manifest_path:
        args_dict, row = load_manifest(manifest_path, seed)
    else:
        args_dict, row = load_profile(profile_path)
    with Path(checkpoint_path).open("rb") as fh:
        checkpoint = pickle.load(fh)
    heldout = str(row["heldout"])
    line = str(row["line"])
    seed = int(row["seed"])
    problem, _ = build_target_problem(args_dict, heldout, line, seed)
    config_dict = dict(checkpoint["config"])
    config_dict.update({
        "checkpoint_dir": "",
        "checkpoint_resume": False,
        "progress_logging": False,
    })
    algorithm = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(**config_dict),
    )
    next_stage = algorithm._load_checkpoint_payload(checkpoint)
    if int(next_stage) >= int(algorithm.config.N):
        raise ValueError(
            f"checkpoint next_stage={next_stage} has no remaining KG iteration")
    return algorithm, args_dict, row, int(next_stage)


def reconstruct_decision_sets(algorithm, next_stage):
    iteration = int(next_stage) - int(algorithm.config.n0)
    algorithm._refresh_sequential_basis()
    recheck_x, _ = algorithm._certification_recheck_candidate()
    candidates, sources = algorithm._generate_candidates(iteration)
    if recheck_x is not None:
        recheck_x = tuple(int(v) for v in recheck_x)
        if recheck_x not in candidates:
            candidates.append(recheck_x)
        sources[recheck_x] = "certification_recheck"
    terminal_pool = list(dict.fromkeys([
        tuple(int(v) for v in x)
        for x in algorithm._recommendation_pool()
    ] + [
        tuple(int(v) for v in x) for x in candidates
    ]))
    _, details = algorithm._solve_posterior_recommendation(
        pool=terminal_pool,
        terminal_frontier_count=(
            algorithm.config.terminal_frontier_candidate_count),
    )
    frontier = details.pop("_terminal_frontier_candidates", [])
    labels = details.get("terminal_frontier_labels", [])
    for label, x in zip(labels, frontier):
        x = tuple(int(v) for v in x)
        if x not in sources:
            candidates.append(x)
            sources[x] = f"terminal_frontier:{label}"
    return candidates, terminal_pool, sources


def score_audit(algorithm, candidates, terminal_pool, sources, mode, mc, repeat):
    algorithm.config.exact_kg_sampling_mode = str(mode)
    algorithm.config.exact_kg_mc_samples = int(mc)
    algorithm.config.exact_kg_clip_negative = False
    algorithm.rng = np.random.default_rng(
        int(algorithm.config.seed)
        + 104729 * (int(repeat) + 1)
    )
    algorithm._progress_stage_n = len(algorithm.history)
    algorithm._progress_step_started_at = time.perf_counter()
    algorithm._progress_run_started_at = algorithm._progress_step_started_at
    started = time.perf_counter()
    raw_scores = algorithm._exact_posterior_update_scores(
        candidates, terminal_pool)
    elapsed = time.perf_counter() - started
    raw_scores = np.asarray(raw_scores, dtype=float)
    order = np.argsort(-raw_scores, kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    margins = np.asarray([
        algorithm._true_chance_margin(x) for x in candidates
    ], dtype=float)
    objectives = np.asarray([
        algorithm.problem.true_objective(x) for x in candidates
    ], dtype=float)
    feasible = margins <= 0.0
    selected = int(order[0])
    if np.any(feasible):
        feasible_idx = np.where(feasible)[0]
        best_feasible = int(feasible_idx[np.argmin(objectives[feasible_idx])])
        highest_score_feasible = int(
            feasible_idx[np.argmax(raw_scores[feasible_idx])])
    else:
        best_feasible = None
        highest_score_feasible = None
    return {
        "sampling_mode": str(mode),
        "mc_samples": int(mc),
        "repeat": int(repeat),
        "elapsed_sec": float(elapsed),
        "n_candidates": int(len(candidates)),
        "terminal_pool_size": int(len(terminal_pool)),
        "raw_min": float(np.min(raw_scores)),
        "raw_max": float(np.max(raw_scores)),
        "raw_negative_fraction": float(np.mean(raw_scores < 0.0)),
        "raw_scores": raw_scores.tolist(),
        "score_ranks": ranks.tolist(),
        "task_entropy_gain": np.asarray(getattr(
            algorithm,
            "_last_exact_kg_task_entropy_gain",
            np.zeros(len(candidates), dtype=float),
        ), dtype=float).tolist(),
        "task_weight_movement": np.asarray(getattr(
            algorithm,
            "_last_exact_kg_task_weight_movement",
            np.zeros(len(candidates), dtype=float),
        ), dtype=float).tolist(),
        "selected_index": selected,
        "selected_x": list(map(int, candidates[selected])),
        "selected_source": str(sources.get(candidates[selected], "unknown")),
        "selected_true_feasible": bool(feasible[selected]),
        "selected_true_margin": float(margins[selected]),
        "selected_true_objective": float(objectives[selected]),
        "best_true_feasible_index": best_feasible,
        "best_true_feasible_rank": (
            None if best_feasible is None else int(ranks[best_feasible])
        ),
        "highest_score_true_feasible_index": highest_score_feasible,
        "highest_score_true_feasible_rank": (
            None
            if highest_score_feasible is None
            else int(ranks[highest_score_feasible])
        ),
        "selected_minus_highest_feasible_score": (
            None
            if highest_score_feasible is None
            else float(
                raw_scores[selected] - raw_scores[highest_score_feasible])
        ),
    }


def write_payload(path, payload):
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mc-samples", default="2,8")
    parser.add_argument("--sampling-modes", default="iid,antithetic")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--exact-jobs", type=int, default=12)
    parser.add_argument("--parallel-backend", default="process_fork")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if not args.profile and not args.manifest:
        parser.error("one of --profile or --manifest is required")
    if args.manifest and int(args.seed) < 0:
        parser.error("--manifest requires --seed")
    algorithm, profile_config, row, next_stage = restore_algorithm(
        args.profile, args.manifest, args.seed, args.checkpoint)
    algorithm.config.exact_kg_jobs = max(1, int(args.exact_jobs))
    algorithm.config.exact_kg_parallel_backend = str(args.parallel_backend)
    candidates, terminal_pool, sources = reconstruct_decision_sets(
        algorithm, next_stage)
    candidate_table = []
    for index, x in enumerate(candidates):
        candidate_table.append({
            "index": int(index),
            "x": list(map(int, x)),
            "source": str(sources.get(x, "unknown")),
            "true_chance_margin": float(algorithm._true_chance_margin(x)),
            "true_objective": float(algorithm.problem.true_objective(x)),
        })
    payload = {
        "schema_version": 1,
        "offline_only": True,
        "simulator_calls": 0,
        "status": "running",
        "profile": str(args.profile),
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "heldout": str(row["heldout"]),
        "seed": int(row["seed"]),
        "next_stage": int(next_stage),
        "profile_exact_kg_terminal_mode": profile_config.get(
            "exact_kg_terminal_mode"),
        "candidate_table": candidate_table,
        "rows": [],
    }
    combinations = [
        (repeat, mode, mc)
        for repeat in range(max(1, int(args.repeats)))
        for mode in parse_csv(args.sampling_modes)
        for mc in parse_csv(args.mc_samples, int)
    ]
    audit_started = time.perf_counter()
    write_payload(args.out, payload)
    rows = payload["rows"]
    for completed, (repeat, mode, mc) in enumerate(combinations, start=1):
        row_started = time.perf_counter()
        print(
            f"Step {completed}/{len(combinations)} [exactkg-audit] "
            f"start mode={mode} mc={mc} repeat={repeat}",
            flush=True,
        )
        rows.append(score_audit(
            algorithm,
            candidates,
            terminal_pool,
            sources,
            mode,
            mc,
            repeat,
        ))
        elapsed = time.perf_counter() - audit_started
        eta = elapsed * (len(combinations) - completed) / completed
        payload["completed_combinations"] = int(completed)
        payload["total_combinations"] = int(len(combinations))
        payload["elapsed_sec"] = float(elapsed)
        payload["eta_sec"] = float(eta)
        write_payload(args.out, payload)
        print(
            f"Step {completed}/{len(combinations)} [exactkg-audit] done "
            f"mode={mode} mc={mc} repeat={repeat} "
            f"unit_sec={time.perf_counter() - row_started:.3f} "
            f"elapsed_sec={elapsed:.3f} eta_sec={eta:.3f}",
            flush=True,
        )
    payload["status"] = "completed"
    payload["eta_sec"] = 0.0
    write_payload(args.out, payload)
    text = json.dumps(json_safe(payload), indent=2, sort_keys=True)
    print(text)


if __name__ == "__main__":
    main()
