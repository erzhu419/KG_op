"""Unified runner for the RZDT variance-study pipeline.

This is the minimal executable version of VARIANCE_STUDY_PIPELINE.md.  It
currently covers the core diagnostic trio:

* GPR-KG-VEPM
* GPR-KG-pooled-pre
* GPR-KG-oracleV

The implementation reuses the checkpointed single-method runner so the
algorithm path, checkpoint format, and iteration snapshots stay identical to
the already validated GPR-KG runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.config import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_D,
    DEFAULT_N,
    DEFAULT_N0,
    SEED_BASE,
)
from experiments.problem_registry import (  # noqa: E402
    ALL_RZDT_PROBLEMS,
    problem_names_for_suite,
)
from experiments.run_rzdt_checkpointed import (  # noqa: E402
    json_safe,
    run_one,
    tqdm,
)


METHOD_SPECS = {
    "GPR-KG": ("GPR-KG", "vepm"),
    "GPR-KG-VEPM": ("GPR-KG", "vepm"),
    "GPR-KG-AVB": ("GPR-KG", "vepm"),
    "GPR-KG-AVB-lite": ("GPR-KG", "vepm"),
    "GPR-KG-AVB-safe": ("GPR-KG", "vepm"),
    "GPR-KG-pooled-pre": ("GPR-KG", "pooled_pre"),
    "GPR-KG-oracleV": ("GPR-KG", "oracle"),
    "GPR-KG-nV": ("GPR-KG-nV", "vepm"),
}


def _git_value(args: list[str]) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover - defensive on servers
        return f"unavailable: {exc}"


def _git_dirty_status_for_code() -> str:
    """Return dirty status while excluding generated variance-study outputs."""
    return _git_value([
        "status",
        "--short",
        "--",
        ".",
        ":(exclude)GPR_KG_Code/results/variance_study",
    ])


def write_manifest(args, output_dir: Path):
    """Write reproducibility metadata for a variance-study run."""
    manifest = {
        "run_id": args.run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_dirty_status": _git_dirty_status_for_code(),
        "git_dirty_status_note": (
            "Generated variance-study outputs are excluded from this field."),
        "command": " ".join(sys.argv),
        "suite": args.suite,
        "problems": args.problems,
        "methods": args.methods,
        "N": args.N,
        "n0": args.n0,
        "n_reps": args.n_reps,
        "seed_base": args.seed_base,
        "sigma": args.sigma,
        "alpha": args.alpha,
        "initial_design": args.initial_design,
        "partition_features": args.partition_features,
        "robust_vepm": args.robust_vepm,
        "vepm_residual_clip_factor": args.vepm_residual_clip_factor,
        "vepm_new_point_weight": args.vepm_new_point_weight,
        "vepm_partition_weight_floor": args.vepm_partition_weight_floor,
        "adaptive_vepm": args.adaptive_vepm,
        "adaptive_vepm_max_features": args.adaptive_vepm_max_features,
        "adaptive_vepm_min_score": args.adaptive_vepm_min_score,
        "vepm_shrinkage_kappa": args.vepm_shrinkage_kappa,
        "replication_policy": args.replication_policy,
        "replication_max_per_solution": args.replication_max_per_solution,
        "replication_score_threshold": args.replication_score_threshold,
        "replication_boundary_scale": args.replication_boundary_scale,
        "replication_budget_fraction": args.replication_budget_fraction,
        "boundary_candidate_policy": args.boundary_candidate_policy,
        "boundary_candidate_count": args.boundary_candidate_count,
        "boundary_candidate_pool_size": args.boundary_candidate_pool_size,
        "boundary_candidate_margin_scale": (
            args.boundary_candidate_margin_scale),
        "boundary_candidate_feasibility_buffer": (
            args.boundary_candidate_feasibility_buffer),
        "boundary_acquisition_weight": args.boundary_acquisition_weight,
        "boundary_acquisition_margin_scale": (
            args.boundary_acquisition_margin_scale),
        "boundary_acquisition_decay_power": (
            args.boundary_acquisition_decay_power),
        "exploration_epsilon0": args.exploration_epsilon0,
        "exploration_epsilon_min": args.exploration_epsilon_min,
        "exploration_decay_power": args.exploration_decay_power,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(config_safe(manifest), f, indent=2)


def config_safe(obj):
    """JSON-safe conversion including pathlib paths."""
    if isinstance(obj, Path):
        return str(obj)
    return json_safe(obj)


def make_single_args(args, method_name: str) -> SimpleNamespace:
    """Build the argument object expected by run_rzdt_checkpointed.run_one."""
    method_base, variance_mode = METHOD_SPECS[method_name]
    is_avb = method_name == "GPR-KG-AVB"
    is_avb_lite = method_name == "GPR-KG-AVB-lite"
    is_avb_safe = method_name == "GPR-KG-AVB-safe"
    is_adaptive_variant = is_avb or is_avb_lite or is_avb_safe
    adaptive_vepm = bool(args.adaptive_vepm or is_adaptive_variant)
    shrinkage_kappa = (
        args.vepm_shrinkage_kappa
        if args.vepm_shrinkage_kappa is not None
        else (10.0 if is_adaptive_variant else None))
    boundary_weight = (
        args.boundary_acquisition_weight
        if args.boundary_acquisition_weight is not None
        else (0.10 if is_avb else 0.0))
    boundary_policy = args.boundary_candidate_policy
    boundary_count = args.boundary_candidate_count
    if is_avb and boundary_policy == "none":
        boundary_policy = "chance_margin"
    if is_avb_lite and boundary_policy == "none":
        boundary_policy = "chance_feasible"
    if is_avb_safe and boundary_policy == "none":
        boundary_policy = "chance_feasible"
    if is_adaptive_variant and boundary_count == 0:
        boundary_count = 10
    boundary_buffer = args.boundary_candidate_feasibility_buffer
    if boundary_buffer is None:
        boundary_buffer = 0.5 if is_avb_safe else 0.0
    return SimpleNamespace(
        method=method_base,
        method_label_override=method_name,
        variance_mode=variance_mode,
        initial_design=args.initial_design,
        nv_fallback_variance=args.nv_fallback_variance,
        run_id=args.run_id,
        results_root=str(args.results_root),
        output_dir=args.output_dir,
        problems=args.problems,
        n_reps=args.n_reps,
        seed_base=args.seed_base,
        N=args.N,
        n0=args.n0,
        d=args.d,
        sigma=args.sigma,
        alpha=args.alpha,
        K1=args.K1,
        K2=args.K2,
        n_thr=args.n_thr,
        partition_features=args.partition_features,
        robust_vepm=args.robust_vepm,
        vepm_residual_clip_factor=args.vepm_residual_clip_factor,
        vepm_new_point_weight=args.vepm_new_point_weight,
        vepm_partition_weight_floor=args.vepm_partition_weight_floor,
        adaptive_vepm=adaptive_vepm,
        adaptive_vepm_max_features=args.adaptive_vepm_max_features,
        adaptive_vepm_min_score=args.adaptive_vepm_min_score,
        vepm_shrinkage_kappa=shrinkage_kappa,
        variance_surrogate=args.variance_surrogate,
        variance_surrogate_rho0=args.variance_surrogate_rho0,
        variance_surrogate_alpha=args.variance_surrogate_alpha,
        variance_surrogate_min_samples=args.variance_surrogate_min_samples,
        variance_surrogate_only_constraint=(
            args.variance_surrogate_only_constraint),
        variance_surrogate_clip_low=args.variance_surrogate_clip_low,
        variance_surrogate_clip_high=args.variance_surrogate_clip_high,
        replication_policy=args.replication_policy,
        replication_max_per_solution=args.replication_max_per_solution,
        replication_score_threshold=args.replication_score_threshold,
        replication_boundary_scale=args.replication_boundary_scale,
        replication_budget_fraction=args.replication_budget_fraction,
        boundary_candidate_policy=boundary_policy,
        boundary_candidate_count=boundary_count,
        boundary_candidate_pool_size=args.boundary_candidate_pool_size,
        boundary_candidate_margin_scale=args.boundary_candidate_margin_scale,
        boundary_candidate_feasibility_buffer=boundary_buffer,
        boundary_acquisition_weight=boundary_weight,
        boundary_acquisition_margin_scale=(
            args.boundary_acquisition_margin_scale),
        boundary_acquisition_decay_power=(
            args.boundary_acquisition_decay_power),
        exploration_epsilon0=args.exploration_epsilon0,
        exploration_epsilon_min=args.exploration_epsilon_min,
        exploration_decay_power=args.exploration_decay_power,
        force=args.force,
        restart=args.restart,
        verbose_algorithm=args.verbose_algorithm,
    )


def result_row(result: dict) -> dict:
    """Extract one flat CSV row from a result dictionary."""
    return {
        "problem": result["problem"],
        "method": result["method"],
        "method_base": result.get("method_base", result["method"]),
        "variance_mode": result.get("variance_mode", ""),
        "rep": int(result["rep"]),
        "seed": int(result["seed"]),
        "hv_final": float(result["hv_final"]),
        "igd_final": float(result["igd_final"]),
        "cvr_final": float(result["cvr_final"]),
        "n_pareto_solutions": int(result["n_pareto_solutions"]),
        "n_simulations": int(result["n_simulations"]),
        "total_time_sec": float(result["total_time_sec"]),
        "completed": bool(result.get("completed", False)),
        "result_path": result.get("result_path", ""),
    }


def write_rows_csv(rows: list[dict], path: Path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict], output_dir: Path):
    """Write run-level and problem/method summaries."""
    write_rows_csv(rows, output_dir / "runs.csv")
    if not rows:
        return

    metrics = [
        "hv_final",
        "igd_final",
        "cvr_final",
        "n_pareto_solutions",
        "total_time_sec",
    ]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["problem"], row["method"]), []).append(row)

    summary_rows = []
    for (problem, method), group in sorted(grouped.items()):
        out = {"problem": problem, "method": method, "n": len(group)}
        for metric in metrics:
            vals = np.array([float(r[metric]) for r in group], dtype=float)
            out[f"{metric}_mean"] = float(np.mean(vals))
            out[f"{metric}_std"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            out[f"{metric}_se"] = (
                out[f"{metric}_std"] / float(np.sqrt(len(vals)))
                if len(vals) > 0 else 0.0)
        summary_rows.append(out)
    write_rows_csv(summary_rows, output_dir / "summary_by_problem_method.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run unified RZDT variance-study experiments.")
    parser.add_argument("--suite",
                        choices=["original_rzdt", "variance_critical",
                                 "rczdt", "all"],
                        default="original_rzdt")
    parser.add_argument("--problems", nargs="+", choices=list(ALL_RZDT_PROBLEMS))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHOD_SPECS),
        default=["GPR-KG-VEPM", "GPR-KG-pooled-pre", "GPR-KG-oracleV"],
    )
    parser.add_argument("--run_id", default="variance_study")
    parser.add_argument("--results_root",
                        default="GPR_KG_Code/results/variance_study")
    parser.add_argument("--initial_design",
                        choices=["common_random", "structured",
                                 "common_structured"],
                        default="common_random")
    parser.add_argument("--N", type=int, default=DEFAULT_N)
    parser.add_argument("--n0", type=int, default=DEFAULT_N0)
    parser.add_argument("--n_reps", type=int, default=1)
    parser.add_argument("--seed_base", type=int, default=SEED_BASE)
    parser.add_argument("--d", type=int, default=DEFAULT_D)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--K1", type=int, default=None)
    parser.add_argument("--K2", type=int, default=None)
    parser.add_argument("--n_thr", type=int, default=None)
    parser.add_argument(
        "--partition_features",
        default="auto",
        help=("VEPM partition features: 'auto' uses the problem registry, "
              "'all' preserves the historical full-coordinate partition, "
              "or pass comma-separated indices such as '0,1'."))
    parser.add_argument("--nv_fallback_variance", type=float, default=0.01)
    parser.add_argument("--variance_surrogate",
                        choices=["none", "ridge_logvar"],
                        default="none")
    parser.add_argument("--variance_surrogate_rho0", type=float, default=0.0)
    parser.add_argument("--variance_surrogate_alpha", type=float,
                        default=1e-3)
    parser.add_argument("--variance_surrogate_min_samples", type=int,
                        default=20)
    parser.add_argument("--variance_surrogate_only_constraint",
                        action="store_true")
    parser.add_argument("--variance_surrogate_clip_low", type=float,
                        default=0.5)
    parser.add_argument("--variance_surrogate_clip_high", type=float,
                        default=2.0)
    parser.add_argument("--robust_vepm", action="store_true")
    parser.add_argument("--vepm_residual_clip_factor", type=float,
                        default=None)
    parser.add_argument("--vepm_new_point_weight", type=float,
                        default=None)
    parser.add_argument("--vepm_partition_weight_floor", type=float,
                        default=None)
    parser.add_argument("--adaptive_vepm", action="store_true")
    parser.add_argument("--adaptive_vepm_max_features", type=int, default=2)
    parser.add_argument("--adaptive_vepm_min_score", type=float, default=0.0)
    parser.add_argument("--vepm_shrinkage_kappa", type=float, default=None)
    parser.add_argument("--replication_policy",
                        choices=["none", "boundary"],
                        default="none")
    parser.add_argument("--replication_max_per_solution", type=int, default=3)
    parser.add_argument("--replication_score_threshold", type=float,
                        default=5e-4)
    parser.add_argument("--replication_boundary_scale", type=float,
                        default=1.0)
    parser.add_argument("--replication_budget_fraction", type=float,
                        default=1.0)
    parser.add_argument("--boundary_candidate_policy",
                        choices=["none", "chance_margin", "chance_feasible"],
                        default="none")
    parser.add_argument("--boundary_candidate_count", type=int, default=0)
    parser.add_argument("--boundary_candidate_pool_size", type=int,
                        default=500)
    parser.add_argument("--boundary_candidate_margin_scale", type=float,
                        default=1.0)
    parser.add_argument("--boundary_candidate_feasibility_buffer", type=float,
                        default=None,
                        help=("For chance_feasible boundary candidates, "
                              "require posterior chance margin <= "
                              "-buffer * sigma_3. If omitted, "
                              "GPR-KG-AVB-safe uses 0.5 and other methods "
                              "use 0.0."))
    parser.add_argument("--boundary_acquisition_weight", type=float,
                        default=None)
    parser.add_argument("--boundary_acquisition_margin_scale", type=float,
                        default=1.0)
    parser.add_argument("--boundary_acquisition_decay_power", type=float,
                        default=0.0)
    parser.add_argument("--exploration_epsilon0", type=float, default=0.0)
    parser.add_argument("--exploration_epsilon_min", type=float, default=0.0)
    parser.add_argument("--exploration_decay_power", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--verbose_algorithm", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny local validation settings.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.problems is None:
        args.problems = list(problem_names_for_suite(args.suite))
    if args.smoke:
        args.problems = args.problems[:1]
        args.N = 13
        args.n0 = 12
        args.n_reps = 1
        args.K1 = 3
        args.K2 = 0
        args.n_thr = 3

    args.results_root = Path(args.results_root)
    args.output_dir = args.results_root / args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({k: config_safe(v) for k, v in vars(args).items()
                   if k != "output_dir"}, f, indent=2)
    write_manifest(args, args.output_dir)

    units = [
        (method, problem, rep)
        for method in args.methods
        for problem in args.problems
        for rep in range(args.n_reps)
    ]
    rows = []
    pbar = tqdm(units, total=len(units),
                desc=f"variance-study {args.run_id}", unit="run")
    for method, problem, rep in pbar:
        pbar.set_postfix(method=method, problem=problem, rep=rep)
        single_args = make_single_args(args, method)
        result, status = run_one(problem, rep, single_args)
        result = dict(result)
        result["result_path"] = str(
            args.output_dir / problem
            / f"{result['method']}_rep{rep:02d}" / "result.json")
        rows.append(result_row(result))
        print(
            f"[{status}] {problem} {result['method']} rep={rep} "
            f"HV={result['hv_final']:.6f} "
            f"IGD={result['igd_final']:.6f} "
            f"CVR={result['cvr_final']:.3f}",
            flush=True,
        )
        write_summary(rows, args.output_dir)
    pbar.close()
    write_summary(rows, args.output_dir)
    print(f"[summary] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
