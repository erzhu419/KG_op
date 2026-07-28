"""Ratio-aware benchmark for LF-OS-SC-HVD-KG.

The central question for the high-dimensional route is not only regret, and
not only dimension.  It is whether a method can keep feasible regret and
chance feasibility stable as ``d / N`` grows.  This runner sweeps
``(problem, d, N, encoder, low-frequency cutoff, active dimension, residual
floor)`` and writes one merged table with explicit ratio columns.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark_quality  # noqa: E402
from benchmark_quality import (  # noqa: E402
    flatten_summary,
    json_safe,
    parse_csv,
    write_csv,
)


def parse_ints(text):
    return parse_csv(text, int)


def parse_floats(text):
    return parse_csv(text, float)


def _safe_name(value):
    return "".join(
        c if c.isalnum() or c in ("-", "_", ".") else "_"
        for c in str(value)
    ).strip("_")


def task_args(base_args, task):
    fields = vars(base_args).copy()
    fields.update(task)
    prefix = (
        f"{base_args.out_prefix}_{_safe_name(task['problem'])}"
        f"_d{task['d']}_N{task['N']}_{_safe_name(task['encoder_kind'])}"
        f"_lf{task['lf_os_low_frequency_components']}"
        f"_a{task['lf_os_max_active']}"
        f"_floor{str(task['lf_os_residual_floor_scale']).replace('.', 'p')}"
    )
    fields["out_prefix"] = prefix
    checkpoint_dir = str(fields.get("checkpoint_dir") or "").strip()
    if checkpoint_dir:
        fields["checkpoint_dir"] = str(Path(checkpoint_dir) / prefix)
    fields["progress_logging"] = bool(
        getattr(base_args, "inner_progress_logging", False))
    fields["progress_label"] = (
        f"{task['problem']}:d={task['d']}:N={task['N']}:"
        f"{task['encoder_kind']}:lf={task['lf_os_low_frequency_components']}:"
        f"a={task['lf_os_max_active']}:"
        f"floor={task['lf_os_residual_floor_scale']}"
    )
    return SimpleNamespace(**fields)


def build_tasks(args):
    tasks = []
    problems = parse_csv(args.problems)
    dims = parse_ints(args.dims)
    budgets = parse_ints(args.budgets)
    encoders = parse_csv(args.encoder_kinds)
    cutoffs = parse_ints(args.lf_os_low_frequency_components_grid)
    actives = parse_ints(args.lf_os_max_active_grid)
    floors = parse_floats(args.lf_os_residual_floor_scale_grid)
    for problem in problems:
        for d in dims:
            for N in budgets:
                if N <= int(args.n0):
                    continue
                for encoder in encoders:
                    if encoder == "lf_os":
                        for cutoff in cutoffs:
                            for active in actives:
                                for floor in floors:
                                    tasks.append({
                                        "problem": problem,
                                        "d": d,
                                        "N": N,
                                        "encoder_kind": encoder,
                                        "lf_os_low_frequency_components": cutoff,
                                        "lf_os_max_active": active,
                                        "lf_os_residual_floor_scale": floor,
                                    })
                    else:
                        tasks.append({
                            "problem": problem,
                            "d": d,
                            "N": N,
                            "encoder_kind": encoder,
                            "lf_os_low_frequency_components": int(cutoffs[0]),
                            "lf_os_max_active": int(actives[0]),
                            "lf_os_residual_floor_scale": float(floors[0]),
                        })
    return tasks


def _seed_count(args):
    seeds = parse_csv(getattr(args, "seeds", ""), int)
    if seeds:
        return len(seeds)
    return int(getattr(args, "n_seeds", 1))


def _variant_count(args):
    try:
        return max(1, len(benchmark_quality.build_variants(args)))
    except Exception:
        return 1


def _task_work_units(args, task):
    return int(task["N"]) * max(1, _seed_count(args)) * _variant_count(args)


def _emit_shard_progress(args, *, completed_units, total_units, completed_tasks,
                         total_tasks, started_at, active_tasks, kind):
    if not bool(getattr(args, "progress_logging", True)):
        return
    elapsed = max(0.0, time.perf_counter() - float(started_at))
    done_units = max(1, int(completed_units))
    remaining_units = max(0, int(total_units) - int(completed_units))
    eta = (elapsed / float(done_units)) * float(remaining_units)
    print(
        f"Step {int(completed_units)}/{int(total_units)} [lfos-shard] "
        f"kind={kind} configs_done={int(completed_tasks)}/{int(total_tasks)} "
        f"active={int(active_tasks)} elapsed={elapsed:.1f}s ETA {eta:.1f}s",
        flush=True,
    )


def run_one(base_args, task):
    args = task_args(base_args, task)
    result = benchmark_quality.run_benchmark(args)
    paths = benchmark_quality.write_outputs(args, result)
    summary_rows = []
    for summary in result["summary"].values():
        flat = flatten_summary(summary)
        flat.update({
            "problem": task["problem"],
            "d": int(task["d"]),
            "N": int(task["N"]),
            "d_over_N": float(task["d"]) / max(float(task["N"]), 1.0),
            "evals_per_dim": float(task["N"]) / max(float(task["d"]), 1.0),
            "encoder_kind": task["encoder_kind"],
            "lf_os_low_frequency_components": int(task[
                "lf_os_low_frequency_components"]),
            "lf_os_max_active": int(task["lf_os_max_active"]),
            "lf_os_residual_floor_scale": float(task[
                "lf_os_residual_floor_scale"]),
            "paths_json": paths["json"],
            "paths_rows_csv": paths["rows_csv"],
            "paths_summary_csv": paths["summary_csv"],
        })
        summary_rows.append(flat)
    return {
        "task": task,
        "paths": paths,
        "summary_rows": summary_rows,
    }


def run_sweep(args):
    tasks = build_tasks(args)
    if args.shards > 1:
        tasks = [
            task for idx, task in enumerate(tasks)
            if idx % int(args.shards) == int(args.shard_index)
        ]
    rows = []
    results = []
    total = len(tasks)
    weights = {idx: _task_work_units(args, task) for idx, task in enumerate(tasks)}
    total_units = max(1, sum(weights.values()))
    completed_units = 0
    started_at = time.perf_counter()
    print(
        f"[lfos-ratio] total_tasks={total} shard={args.shard_index}/{args.shards}",
        flush=True,
    )
    _emit_shard_progress(
        args,
        completed_units=completed_units,
        total_units=total_units,
        completed_tasks=0,
        total_tasks=total,
        started_at=started_at,
        active_tasks=0,
        kind="start",
    )
    if int(args.jobs) <= 1:
        for idx, task in enumerate(tasks):
            print(f"[{idx}/{total}] [lfos-ratio] start {task}", flush=True)
            result = run_one(args, task)
            results.append(result)
            rows.extend(result["summary_rows"])
            completed_units += weights[idx]
            print(f"[{idx + 1}/{total}] [lfos-ratio] done {task}", flush=True)
            _emit_shard_progress(
                args,
                completed_units=completed_units,
                total_units=total_units,
                completed_tasks=idx + 1,
                total_tasks=total,
                started_at=started_at,
                active_tasks=0,
                kind="config_done",
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as pool:
            futs = {
                pool.submit(run_one, args, task): (idx, task)
                for idx, task in enumerate(tasks)
            }
            completed = 0
            pending = set(futs)
            while pending:
                done, pending = wait(
                    pending,
                    timeout=float(getattr(
                        args, "shard_progress_interval_sec", 30.0)),
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    _emit_shard_progress(
                        args,
                        completed_units=completed_units,
                        total_units=total_units,
                        completed_tasks=completed,
                        total_tasks=total,
                        started_at=started_at,
                        active_tasks=len(pending),
                        kind="heartbeat",
                    )
                    continue
                for fut in done:
                    idx, task = futs[fut]
                    result = fut.result()
                    results.append(result)
                    rows.extend(result["summary_rows"])
                    completed += 1
                    completed_units += weights[idx]
                    print(
                        f"[{completed}/{total}] [lfos-ratio] done {task}",
                        flush=True,
                    )
                _emit_shard_progress(
                    args,
                    completed_units=completed_units,
                    total_units=total_units,
                    completed_tasks=completed,
                    total_tasks=total,
                    started_at=started_at,
                    active_tasks=len(pending),
                    kind="config_done",
                )
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": vars(args),
        "n_tasks": len(tasks),
        "results": results,
        "summary_rows": rows,
    }


def write_outputs(args, result):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"lfos_ratio_sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.shards > 1:
        prefix = f"{prefix}_shard{int(args.shard_index):02d}of{int(args.shards):02d}"
    json_path = out_dir / f"{prefix}.json"
    summary_path = out_dir / f"{prefix}_summary.csv"
    json_path.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    write_csv(summary_path, result["summary_rows"])
    return {"json": str(json_path), "summary_csv": str(summary_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="HighDimStatePolicyRZDT1,FactorShockStatePolicyRZDT1")
    parser.add_argument("--dims", default="1000,10000")
    parser.add_argument("--budgets", default="30,40,80")
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--N", type=int, default=40)
    parser.add_argument("--n0", type=int, default=8)
    parser.add_argument("--K1", type=int, default=12)
    parser.add_argument("--K2", type=int, default=0)
    parser.add_argument("--posterior_pool_size", type=int, default=180)
    parser.add_argument("--posterior_keep", type=int, default=10)
    parser.add_argument("--axis_candidate_count", type=int, default=-1)
    parser.add_argument("--structured_candidate_count", type=int, default=0)
    parser.add_argument("--state_candidate_count", type=int, default=16)
    parser.add_argument("--state_inverse_pool_size", type=int, default=400)
    parser.add_argument("--state_inverse_neighbors", type=int, default=2)
    parser.add_argument("--n_thr", type=int, default=5)
    parser.add_argument("--eval_pool_size", type=int, default=300)
    parser.add_argument("--lambda_feas", type=float, default=0.25)
    parser.add_argument("--lambda_var", type=float, default=0.25)
    parser.add_argument("--lambda_mean", type=float, default=0.10)
    parser.add_argument("--lambda_coupling", type=float, default=0.05)
    parser.add_argument("--beta_g", type=float, default=2.0)
    parser.add_argument("--certification_mode", default="theory")
    parser.add_argument("--coupling_safety_z", type=float, default=0.5)
    parser.add_argument("--coupling_gate_temperature", type=float, default=0.25)
    parser.add_argument("--recommendation_safety_z", type=float, default=0.5)
    parser.add_argument("--recommendation_noise_floor_scale", type=float, default=1.0)
    parser.add_argument("--recommendation_infeasible_penalty", type=float, default=5.0)
    parser.add_argument("--recommendation_infeasible_strategy", default="penalty")
    parser.add_argument("--disable_recommendation_calibration", action="store_true")
    parser.add_argument("--recommendation_calibration_ridge", type=float, default=1e-6)
    parser.add_argument("--enable_certification_calibration", action="store_true")
    parser.add_argument("--certification_calibration_min_obs", type=int, default=8)
    parser.add_argument("--certification_calibration_ridge", type=float, default=1e-6)
    parser.add_argument("--certification_calibration_noise_floor_scale", type=float, default=0.5)
    parser.add_argument("--certification_calibration_beta", type=float, default=2.0)
    parser.add_argument("--disable_recommendation_axis_oracle", action="store_true")
    parser.add_argument("--disable_problem_initial_samples", action="store_true")
    parser.add_argument("--disable_boundary_initial_samples", action="store_true")
    parser.add_argument("--disable_recommendation_refinement", action="store_true")
    parser.add_argument("--use_state_basis", dest="use_state_basis", action="store_true", default=True)
    parser.add_argument("--disable_state_basis", dest="use_state_basis", action="store_false")
    parser.add_argument("--state_basis_mode", default="manifold")
    parser.add_argument("--raw_basis_dim", type=int, default=-1)
    parser.add_argument("--raw_projection_seed", type=int, default=314159)
    parser.add_argument("--numeric_backend", default="numpy")
    parser.add_argument("--numeric_backend_device", default="auto")
    parser.add_argument("--torch_dtype", default="float64")
    parser.add_argument("--torch_min_rows", type=int, default=128)
    parser.add_argument("--encoder_kinds", default="synthetic,lf_os")
    parser.add_argument("--encoder_latent_dim", type=int, default=8)
    parser.add_argument("--encoder_fit_pool_size", type=int, default=512)
    parser.add_argument("--lf_os_max_library_size", type=int, default=30)
    parser.add_argument("--lf_os_low_frequency_components_grid", default="5,8,12")
    parser.add_argument("--lf_os_max_active_grid", default="5,8")
    parser.add_argument("--lf_os_graph_neighbors", type=int, default=12)
    parser.add_argument("--lf_os_residual_floor_scale_grid", default="0.0,0.02,0.05")
    parser.add_argument("--disable_lf_os_problem_state_anchor", action="store_true")
    parser.add_argument("--exact_kg_mc_samples", type=int, default=8)
    parser.add_argument("--exact_kg_jobs", type=int, default=1)
    parser.add_argument("--exact_kg_use_score", action="store_true")
    parser.add_argument("--exact_kg_blend", type=float, default=0.0)
    parser.add_argument("--checkpoint_dir", default="")
    parser.add_argument("--checkpoint_resume", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument("--checkpoint_keep_last", type=int, default=2)
    parser.add_argument("--progress_logging", dest="progress_logging", action="store_true", default=True)
    parser.add_argument("--disable_progress_logging", dest="progress_logging", action="store_false")
    parser.add_argument("--inner_progress_logging", action="store_true")
    parser.add_argument("--shard_progress_interval_sec", type=float, default=30.0)
    parser.add_argument("--progress_units_per_iteration", type=int, default=100)
    parser.add_argument("--progress_exact_updates", type=int, default=10)
    parser.add_argument("--acquisition_modes", default="additive")
    parser.add_argument("--modes", default="")
    parser.add_argument("--sc_modes", default="factor")
    parser.add_argument("--baseline_variant", default="factor+sc")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--out_dir", default=str(ROOT / "profiles"))
    parser.add_argument("--out_prefix", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    result = run_sweep(args)
    paths = write_outputs(args, result)
    print(json.dumps(json_safe({
        "paths": paths,
        "n_tasks": result["n_tasks"],
        "summary_rows": result["summary_rows"],
    }), indent=2))


if __name__ == "__main__":
    main()
