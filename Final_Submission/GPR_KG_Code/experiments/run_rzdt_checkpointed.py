"""Checkpointed GPR-KG/GPR-KG-nV runs for the reported RZDT experiments.

This driver is intended for long server runs of the three reported synthetic
problems: RZDT1, RZDT2, and RZDT5_RR.  It uses the algorithm's
run_resumable() path, which saves a full pickle checkpoint after each
pre-sample and each adaptive iteration.  A lightweight JSONL snapshot is
also appended after every adaptive iteration for progress inspection without
loading the pickle.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - used only on minimal servers
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, total=None, desc=None, unit=None,
                     initial=0):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable is not None else 0)
            self.count = initial
            self.desc = desc or ""
            self.unit = unit or "it"
            self.start = time.time()

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)

        def update(self, n=1):
            self.count += n
            elapsed = max(time.time() - self.start, 1e-9)
            rate = self.count / elapsed
            remain = ((self.total - self.count) / rate
                      if self.total and rate > 0 else float("nan"))
            print(f"{self.desc}: {self.count}/{self.total} {self.unit}, "
                  f"elapsed={elapsed:.1f}s, eta={remain:.1f}s",
                  flush=True)

        def set_postfix(self, **kwargs):
            if kwargs:
                msg = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                print(f"{self.desc}: {msg}", flush=True)

        def close(self):
            pass


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.config import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_D,
    DEFAULT_N,
    DEFAULT_N0,
    GPR_KG_PARAMS,
    HV_EVAL_INTERVAL,
    N_REPS,
    REF_POINT,
    SEED_BASE,
)
from experiments.structured_design import (  # noqa: E402
    common_random_initial_samples,
    structured_initial_samples,
)
from experiments.problem_registry import (  # noqa: E402
    ALL_RZDT_PROBLEMS,
    ORIGINAL_RZDT_PROBLEMS,
    make_problem as registry_make_problem,
)
from gpr_kg import (  # noqa: E402
    GPRKR_Algorithm,
    GPRKRnV_Algorithm,
    compute_hypervolume_2d,
    pareto_filter,
)
from metrics import compute_cvr, compute_igd  # noqa: E402


PROBLEM_NAMES = ORIGINAL_RZDT_PROBLEMS


def json_safe(obj):
    """Convert numpy and tuple objects into JSON-serializable values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    return obj


def make_problem(name: str, d: int, sigma: float, alpha: float):
    """Instantiate a registered RZDT problem configuration."""
    problem = registry_make_problem(name, d=d, sigma=sigma, alpha=alpha)
    problem.ref_point = np.array(REF_POINT, dtype=float)
    return problem


def make_algorithm(problem, args, seed: int):
    """Build the current theory-aligned GPR-KG or GPR-KG-nV algorithm."""
    params = dict(GPR_KG_PARAMS)
    alg_cls = GPRKRnV_Algorithm if args.method == "GPR-KG-nV" else GPRKR_Algorithm
    extra = {}
    if args.method == "GPR-KG-nV":
        extra["nv_fallback_variance"] = args.nv_fallback_variance
    initial_samples = None
    use_boundary_initial_design = params.get("use_boundary_initial_design", True)
    if args.initial_design == "common_random":
        initial_samples = common_random_initial_samples(problem, args.n0, seed)
        use_boundary_initial_design = False
    elif args.initial_design == "structured":
        initial_samples = None
        use_boundary_initial_design = params.get(
            "use_boundary_initial_design", True)
    elif args.initial_design == "common_structured":
        initial_samples = structured_initial_samples(problem, args.n0)
        use_boundary_initial_design = False
    return alg_cls(
        problem=problem,
        N=args.N,
        n0=args.n0,
        K1=args.K1 if args.K1 is not None else params["K1"],
        K2=args.K2 if args.K2 is not None else params["K2"],
        lambda_i=params["lambda_i"],
        prior_var=params["prior_var"],
        w_vepm=params["w_vepm"],
        n_thr=args.n_thr if args.n_thr is not None else params["n_thr"],
        seed=seed,
        partition_method=params.get("partition_method", "binary_bin"),
        partition_K=params.get("partition_K", None),
        partition_features=getattr(
            args, "partition_features",
            params.get("partition_features", "auto")),
        use_boundary_initial_design=use_boundary_initial_design,
        initial_samples=initial_samples,
        use_archive_candidates=params.get("use_archive_candidates", False),
        archive_neighbor_radius=params.get("archive_neighbor_radius", 0),
        kg_selection_tiebreak=params.get(
            "kg_selection_tiebreak", "crowding_distance"),
        variance_shrinkage_rho0=params.get("variance_shrinkage_rho0", 0.0),
        variance_floor=params.get("variance_floor", 1e-8),
        variance_surrogate=args.variance_surrogate,
        variance_surrogate_rho0=args.variance_surrogate_rho0,
        variance_surrogate_alpha=args.variance_surrogate_alpha,
        variance_surrogate_min_samples=args.variance_surrogate_min_samples,
        variance_surrogate_only_constraint=(
            args.variance_surrogate_only_constraint),
        variance_surrogate_clip_low=args.variance_surrogate_clip_low,
        variance_surrogate_clip_high=args.variance_surrogate_clip_high,
        robust_vepm=(
            args.robust_vepm or params.get("robust_vepm", False)),
        vepm_residual_clip_factor=(
            args.vepm_residual_clip_factor
            if args.vepm_residual_clip_factor is not None
            else params.get("vepm_residual_clip_factor", None)),
        vepm_new_point_weight=(
            args.vepm_new_point_weight
            if args.vepm_new_point_weight is not None
            else params.get("vepm_new_point_weight", 1.0)),
        vepm_partition_weight_floor=(
            args.vepm_partition_weight_floor
            if args.vepm_partition_weight_floor is not None
            else params.get("vepm_partition_weight_floor", 0.0)),
        adaptive_vepm=(
            args.adaptive_vepm or params.get("adaptive_vepm", False)),
        adaptive_vepm_max_features=(
            args.adaptive_vepm_max_features
            if args.adaptive_vepm_max_features is not None
            else params.get("adaptive_vepm_max_features", 2)),
        adaptive_vepm_min_score=(
            args.adaptive_vepm_min_score
            if args.adaptive_vepm_min_score is not None
            else params.get("adaptive_vepm_min_score", 0.0)),
        vepm_shrinkage_kappa=(
            args.vepm_shrinkage_kappa
            if args.vepm_shrinkage_kappa is not None
            else params.get("vepm_shrinkage_kappa", 0.0)),
        variance_mode=args.variance_mode,
        replication_policy=args.replication_policy,
        replication_max_per_solution=args.replication_max_per_solution,
        replication_score_threshold=args.replication_score_threshold,
        replication_boundary_scale=args.replication_boundary_scale,
        replication_budget_fraction=args.replication_budget_fraction,
        boundary_candidate_policy=args.boundary_candidate_policy,
        boundary_candidate_count=args.boundary_candidate_count,
        boundary_candidate_pool_size=args.boundary_candidate_pool_size,
        boundary_candidate_margin_scale=args.boundary_candidate_margin_scale,
        boundary_candidate_feasibility_buffer=(
            args.boundary_candidate_feasibility_buffer),
        boundary_acquisition_weight=args.boundary_acquisition_weight,
        boundary_acquisition_margin_scale=(
            args.boundary_acquisition_margin_scale),
        boundary_acquisition_decay_power=(
            args.boundary_acquisition_decay_power),
        exploration_epsilon0=args.exploration_epsilon0,
        exploration_epsilon_min=args.exploration_epsilon_min,
        exploration_decay_power=args.exploration_decay_power,
        **extra,
    )


def method_label(args) -> str:
    """Return a result label that includes non-default GPR-KG variance mode."""
    override = getattr(args, "method_label_override", None)
    if override:
        return str(override)
    if args.method != "GPR-KG":
        return args.method
    if args.variance_mode == "vepm":
        return "GPR-KG"
    if args.variance_mode == "pooled_pre":
        return "GPR-KG-pooled-pre"
    if args.variance_mode == "oracle":
        return "GPR-KG-oracleV"
    return f"GPR-KG-{args.variance_mode}"


def final_metrics(pareto_set, problem):
    """Compute final true-objective metrics using the existing paper metrics."""
    true_objs = []
    for x in pareto_set:
        f1, f2, _ = problem.true_objectives(x)
        true_objs.append([f1, f2])
    true_objs = np.array(true_objs) if true_objs else np.empty((0, 2))

    if len(true_objs) > 0:
        pf_true, pf_idx = pareto_filter(true_objs, return_indices=True)
        pareto_filtered = [pareto_set[i] for i in pf_idx]
    else:
        pf_true = np.empty((0, 2))
        pareto_filtered = []

    true_pf = problem.true_pareto_front()
    return {
        "pareto_solutions": [[int(v) for v in x] for x in pareto_filtered],
        "pareto_objectives_true": pf_true.tolist(),
        "hv_final": float(compute_hypervolume_2d(pf_true, problem.ref_point)),
        "igd_final": float(compute_igd(pf_true, true_pf)),
        "cvr_final": float(compute_cvr(pareto_filtered, problem)),
        "n_pareto_solutions": int(len(pareto_filtered)),
    }


def time_per_iter(iteration_log):
    vals = []
    for log in iteration_log:
        vals.append(float(
            log.get("t_posterior_solve", 0.0)
            + log.get("t_candidate_gen", 0.0)
            + log.get("t_kg_compute", 0.0)
            + log.get("t_belief_update", 0.0)
            + log.get("t_vepm_update", 0.0)
            + log.get("t_variance_surrogate_update", 0.0)
            + log.get("t_hv_eval", 0.0)
        ))
    return vals


def run_one(problem_name: str, rep: int, args):
    seed = args.seed_base + rep
    label = method_label(args)
    run_dir = args.output_dir / problem_name / f"{label}_rep{rep:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    result_path = run_dir / "result.json"
    checkpoint_path = run_dir / "checkpoint.pkl"
    snapshot_path = run_dir / "iteration_snapshots.jsonl"
    meta_path = run_dir / "run_meta.json"

    if result_path.exists() and not args.force:
        with result_path.open("r", encoding="utf-8") as f:
            return json.load(f), "skipped"

    problem = make_problem(problem_name, args.d, args.sigma, args.alpha)
    t_start = time.time()

    if checkpoint_path.exists() and not args.restart:
        alg_cls = GPRKRnV_Algorithm if args.method == "GPR-KG-nV" else GPRKR_Algorithm
        alg = alg_cls.restore_checkpoint(str(checkpoint_path), problem)
        restored = True
    else:
        if args.restart:
            for path in (checkpoint_path, snapshot_path, result_path):
                if path.exists():
                    path.unlink()
        alg = make_algorithm(problem, args, seed)
        restored = False

    alg._checkpoint_path = str(checkpoint_path)
    alg._snapshot_jsonl_path = str(snapshot_path)

    effective_parameters = {
        "K1": int(getattr(alg, "K1", args.K1)),
        "K2": int(getattr(alg, "K2", args.K2)),
        "lambda_i": float(getattr(alg, "lambda_i", GPR_KG_PARAMS["lambda_i"])),
        "prior_var": float(getattr(alg, "prior_var", GPR_KG_PARAMS["prior_var"])),
        "w_vepm": float(getattr(alg, "w_vepm", GPR_KG_PARAMS["w_vepm"])),
        "n_thr": int(getattr(alg, "n_thr", args.n_thr)),
        "partition_method": getattr(
            alg, "partition_method",
            GPR_KG_PARAMS.get("partition_method", "binary_bin")),
        "partition_K": getattr(alg, "partition_K", None),
        "partition_features_mode": getattr(
            alg, "partition_features_mode", args.partition_features),
        "partition_features": list(getattr(alg, "partition_features", [])),
        "use_boundary_initial_design": bool(getattr(
            alg, "use_boundary_initial_design", False)),
        "initial_design": args.initial_design,
        "use_archive_candidates": bool(getattr(
            alg, "use_archive_candidates", False)),
        "archive_neighbor_radius": int(getattr(
            alg, "archive_neighbor_radius", 0)),
        "kg_selection_tiebreak": getattr(
            alg, "kg_selection_tiebreak", "crowding_distance"),
        "variance_mode": args.variance_mode,
        "variance_shrinkage_rho0": float(getattr(
            alg, "variance_shrinkage_rho0", 0.0)),
        "variance_floor": float(getattr(alg, "variance_floor", 1e-8)),
        "robust_vepm": bool(getattr(alg, "robust_vepm", False)),
        "vepm_residual_clip_factor": getattr(
            alg, "vepm_residual_clip_factor", None),
        "vepm_new_point_weight": float(getattr(
            alg, "vepm_new_point_weight", 1.0)),
        "vepm_partition_weight_floor": float(getattr(
            alg, "vepm_partition_weight_floor", 0.0)),
        "adaptive_vepm": bool(getattr(alg, "adaptive_vepm", False)),
        "adaptive_vepm_max_features": int(getattr(
            alg, "adaptive_vepm_max_features", 2)),
        "adaptive_vepm_min_score": float(getattr(
            alg, "adaptive_vepm_min_score", 0.0)),
        "adaptive_feature_selected": list(getattr(
            alg.vepm, "adaptive_feature_selected",
            getattr(alg.vepm, "feature_indices", []))),
        "adaptive_feature_scores": getattr(
            alg.vepm, "adaptive_feature_scores", {}),
        "vepm_shrinkage_kappa": float(getattr(
            alg, "vepm_shrinkage_kappa", 0.0)),
        "variance_surrogate": args.variance_surrogate,
        "variance_surrogate_rho0": args.variance_surrogate_rho0,
        "variance_surrogate_alpha": args.variance_surrogate_alpha,
        "variance_surrogate_min_samples": args.variance_surrogate_min_samples,
        "variance_surrogate_only_constraint": (
            args.variance_surrogate_only_constraint),
        "variance_surrogate_clip_low": args.variance_surrogate_clip_low,
        "variance_surrogate_clip_high": args.variance_surrogate_clip_high,
        "replication_policy": args.replication_policy,
        "replication_max_per_solution": args.replication_max_per_solution,
        "replication_score_threshold": args.replication_score_threshold,
        "replication_boundary_scale": args.replication_boundary_scale,
        "replication_budget_fraction": args.replication_budget_fraction,
        "boundary_candidate_policy": args.boundary_candidate_policy,
        "boundary_candidate_count": args.boundary_candidate_count,
        "boundary_candidate_pool_size": args.boundary_candidate_pool_size,
        "boundary_candidate_margin_scale": args.boundary_candidate_margin_scale,
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

    meta = {
        "problem": problem_name,
        "method": label,
        "method_base": args.method,
        "variance_mode": args.variance_mode,
        "rep": rep,
        "seed": seed,
        "N": args.N,
        "n0": args.n0,
        "d": args.d,
        "sigma": args.sigma,
        "alpha": args.alpha,
        "tau": problem.tau,
        "restored_from_checkpoint": restored,
        "initial_design": args.initial_design,
        "initial_samples": json_safe(getattr(alg, "initial_samples", None)),
        "parameters": json_safe(effective_parameters),
        "parameters_base": json_safe(dict(GPR_KG_PARAMS)),
        "parameters_note": (
            "parameters records the effective algorithm configuration for "
            "this run; parameters_base records code defaults."),
        "algorithm_overrides": {
            "variance_surrogate": args.variance_surrogate,
            "variance_surrogate_rho0": args.variance_surrogate_rho0,
            "variance_surrogate_alpha": args.variance_surrogate_alpha,
            "variance_surrogate_min_samples": (
                args.variance_surrogate_min_samples),
            "variance_surrogate_only_constraint": (
                args.variance_surrogate_only_constraint),
            "variance_surrogate_clip_low": args.variance_surrogate_clip_low,
            "variance_surrogate_clip_high": args.variance_surrogate_clip_high,
            "robust_vepm": bool(getattr(alg, "robust_vepm", False)),
            "vepm_residual_clip_factor": getattr(
                alg, "vepm_residual_clip_factor", None),
            "vepm_new_point_weight": getattr(
                alg, "vepm_new_point_weight", 1.0),
            "vepm_partition_weight_floor": getattr(
                alg, "vepm_partition_weight_floor", 0.0),
            "adaptive_vepm": bool(getattr(alg, "adaptive_vepm", False)),
            "adaptive_vepm_max_features": getattr(
                alg, "adaptive_vepm_max_features", 2),
            "adaptive_vepm_min_score": getattr(
                alg, "adaptive_vepm_min_score", 0.0),
            "adaptive_feature_selected": list(getattr(
                alg.vepm, "adaptive_feature_selected",
                getattr(alg.vepm, "feature_indices", []))),
            "adaptive_feature_scores": getattr(
                alg.vepm, "adaptive_feature_scores", {}),
            "vepm_shrinkage_kappa": getattr(
                alg, "vepm_shrinkage_kappa", 0.0),
            "variance_mode": args.variance_mode,
            "partition_features": args.partition_features,
            "partition_features_resolved": list(
                getattr(alg, "partition_features", [])),
            "vepm_n_features": int(getattr(alg.vepm, "n_features", 0)),
            "vepm_partitions": int(getattr(alg.vepm, "total_partitions", 0)),
            "replication_policy": args.replication_policy,
            "replication_max_per_solution": (
                args.replication_max_per_solution),
            "replication_score_threshold": (
                args.replication_score_threshold),
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
        },
        "checkpoint_path": str(checkpoint_path),
        "snapshot_jsonl_path": str(snapshot_path),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    pareto_set = alg.run_resumable(verbose=args.verbose_algorithm)
    wall_time = time.time() - t_start

    # Adaptive VEPM resolves its final partition geometry after pre-sampling.
    # Refresh the metadata so run_meta/result.json record the realized
    # algorithm, not only the pre-run constructor defaults.
    final_selected = list(getattr(
        alg.vepm, "adaptive_feature_selected",
        getattr(alg.vepm, "feature_indices", [])))
    final_scores = getattr(alg.vepm, "adaptive_feature_scores", {})
    for block_name in ("parameters", "algorithm_overrides"):
        block = meta.get(block_name, {})
        block["partition_features_resolved"] = list(getattr(
            alg, "partition_features", []))
        block["partition_features"] = list(getattr(
            alg, "partition_features", []))
        block["adaptive_feature_selected"] = final_selected
        block["adaptive_feature_scores"] = final_scores
        block["vepm_n_features"] = int(getattr(alg.vepm, "n_features", 0))
        block["vepm_partitions"] = int(getattr(
            alg.vepm, "total_partitions", 0))
        meta[block_name] = json_safe(block)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(meta), f, indent=2)

    metrics = final_metrics(pareto_set, problem)
    iter_times = time_per_iter(alg.iteration_log)
    result = {
        **meta,
        **metrics,
        "restored_from_checkpoint": restored,
        "completed": True,
        "total_time_sec": float(wall_time),
        "time_per_iter": iter_times,
        "time_per_iter_mean": float(np.mean(iter_times)) if iter_times else 0.0,
        "n_simulations": int(len(alg.history)),
        "hv_history": [(int(s), float(v)) for s, v in alg.hv_history],
        "iteration_log": json_safe(alg.iteration_log),
        "pre_sampling_log": json_safe(alg.pre_sampling_log),
        "final_log": json_safe(alg.final_log),
        "observation_history": [
            {"x": [int(v) for v in x], "Y": [float(y) for y in Y]}
            for x, Y in alg.history
        ],
        "resume_state": {
            "presampling_done": bool(alg._presampling_done),
            "main_iter_completed": int(alg._main_iter_completed),
            "total_main_iters": int(args.N - args.n0),
        },
        "logging_detail": (
            "Full checkpoint.pkl after every pre-sample and adaptive "
            "iteration; one JSON line per adaptive iteration in "
            "iteration_snapshots.jsonl; final JSON includes all iteration and "
            "observation histories."
        ),
    }
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(result), f, indent=2)
    return result, "completed"


def write_summary(results, output_dir: Path):
    rows = []
    for result in results:
        rows.append({
            "problem": result["problem"],
            "method": result["method"],
            "rep": result["rep"],
            "seed": result["seed"],
            "hv_final": result["hv_final"],
            "igd_final": result["igd_final"],
            "cvr_final": result["cvr_final"],
            "n_pareto_solutions": result["n_pareto_solutions"],
            "n_simulations": result["n_simulations"],
            "total_time_sec": result["total_time_sec"],
            "main_iter_completed": result["resume_state"]["main_iter_completed"],
        })

    csv_path = output_dir / "summary_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        grouped.setdefault(row["problem"], []).append(row)
    summary = {}
    for problem, problem_rows in grouped.items():
        summary[problem] = {}
        for key in ("hv_final", "igd_final", "cvr_final",
                    "n_pareto_solutions", "total_time_sec"):
            vals = np.array([r[key] for r in problem_rows], dtype=float)
            summary[problem][key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": int(len(vals)),
            }
    with (output_dir / "summary_by_problem.json").open("w",
                                                       encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run checkpointed GPR-KG/GPR-KG-nV on RZDT benchmarks.")
    parser.add_argument("--method", choices=["GPR-KG", "GPR-KG-nV"],
                        default="GPR-KG")
    parser.add_argument(
        "--variance_mode",
        choices=["vepm", "pooled_pre", "oracle"],
        default="vepm",
        help=("Variance provider for GPR-KG. 'vepm' preserves the main "
              "algorithm, 'pooled_pre' freezes the pre-sample pooled "
              "variance, and 'oracle' uses the benchmark true variance. "
              "This option is ignored by GPR-KG-nV, whose subclass keeps "
              "its historical local/pooled residual path."))
    parser.add_argument(
        "--initial_design",
        choices=["structured", "common_random", "common_structured"],
        default="structured",
        help=("Initial design policy. 'structured' keeps the algorithm "
              "default boundary+random design; 'common_random' injects the "
              "same seed-indexed random finite-grid initial design used by "
              "baseline runners; 'common_structured' injects the shared "
              "structured design explicitly."))
    parser.add_argument("--nv_fallback_variance", type=float, default=0.01,
                        help="Fallback variance before nV has replications.")
    parser.add_argument("--run_id", default=os.environ.get(
        "RUN_ID", "structured_default_full"))
    parser.add_argument("--results_root", default="results/rzdt_checkpointed")
    parser.add_argument("--problems", nargs="+", default=list(PROBLEM_NAMES),
                        choices=list(ALL_RZDT_PROBLEMS))
    parser.add_argument("--n_reps", type=int, default=N_REPS)
    parser.add_argument("--seed_base", type=int, default=SEED_BASE)
    parser.add_argument("--N", type=int, default=DEFAULT_N)
    parser.add_argument("--n0", type=int, default=DEFAULT_N0)
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
    parser.add_argument(
        "--robust_vepm",
        action="store_true",
        help=("Enable robust finite-sample VEPM updates that clip large "
              "one-shot residuals and downweight newly sampled points."))
    parser.add_argument(
        "--vepm_residual_clip_factor",
        type=float,
        default=None,
        help=("Clip squared residuals at this multiple of the current VEPM "
              "local/cell variance scale when --robust_vepm is enabled."))
    parser.add_argument(
        "--vepm_new_point_weight",
        type=float,
        default=None,
        help=("Fractional VEPM residual weight for the first observation at a "
              "new solution under --robust_vepm. Revisits keep full weight."))
    parser.add_argument(
        "--vepm_partition_weight_floor",
        type=float,
        default=None,
        help=("Minimum evidence weight used when averaging solution-level "
              "variance estimates into a VEPM partition."))
    parser.add_argument(
        "--adaptive_vepm",
        action="store_true",
        help=("Select variance-relevant partition coordinates from "
              "pre-sample residuals before building VEPM cells."))
    parser.add_argument("--adaptive_vepm_max_features", type=int,
                        default=None)
    parser.add_argument("--adaptive_vepm_min_score", type=float,
                        default=None)
    parser.add_argument(
        "--vepm_shrinkage_kappa",
        type=float,
        default=None,
        help=("Pseudo-count for shrinking sparse VEPM local/cell variance "
              "estimates toward pooled pre-sample variance."))
    parser.add_argument("--replication_policy",
                        choices=["none", "boundary"],
                        default="none",
                        help=("Optional adaptive replication rule. 'none' "
                              "preserves the original KG exploration path; "
                              "'boundary' may replicate visited posterior "
                              "Pareto/boundary solutions."))
    parser.add_argument("--replication_max_per_solution", type=int,
                        default=3)
    parser.add_argument("--replication_score_threshold", type=float,
                        default=5e-4)
    parser.add_argument("--replication_boundary_scale", type=float,
                        default=1.0)
    parser.add_argument("--replication_budget_fraction", type=float,
                        default=1.0,
                        help=("Maximum fraction of the adaptive budget that "
                              "may be spent on boundary replication."))
    parser.add_argument("--boundary_candidate_policy",
                        choices=["none", "chance_margin", "chance_feasible"],
                        default="none")
    parser.add_argument("--boundary_candidate_count", type=int, default=0)
    parser.add_argument("--boundary_candidate_pool_size", type=int,
                        default=500)
    parser.add_argument("--boundary_candidate_margin_scale", type=float,
                        default=1.0)
    parser.add_argument("--boundary_candidate_feasibility_buffer", type=float,
                        default=0.0,
                        help=("For chance_feasible boundary candidates, "
                              "require posterior chance margin <= "
                              "-buffer * sigma_3."))
    parser.add_argument("--boundary_acquisition_weight", type=float,
                        default=0.0)
    parser.add_argument("--boundary_acquisition_margin_scale", type=float,
                        default=1.0)
    parser.add_argument("--boundary_acquisition_decay_power", type=float,
                        default=0.0)
    parser.add_argument("--exploration_epsilon0", type=float, default=0.0)
    parser.add_argument("--exploration_epsilon_min", type=float, default=0.0)
    parser.add_argument("--exploration_decay_power", type=float, default=1.0)
    parser.add_argument("--force", action="store_true",
                        help="Recompute a run even if result.json exists.")
    parser.add_argument("--restart", action="store_true",
                        help="Delete existing checkpoint/snapshot/result.")
    parser.add_argument("--verbose_algorithm", action="store_true",
                        help="Print per-algorithm progress messages.")
    parser.add_argument("--smoke", action="store_true",
                        help="Small local validation run.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.problems = ["RZDT1", "RZDT2", "RZDT5_RR"]
        args.n_reps = 1
        args.N = 18
        args.n0 = 8
        args.K1 = 5
        args.K2 = 0
        args.n_thr = 5

    args.output_dir = Path(args.results_root) / args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.output_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump({k: json_safe(v) for k, v in vars(args).items()
                   if k != "output_dir"}, f, indent=2)

    units = [(problem, rep)
             for problem in args.problems
             for rep in range(args.n_reps)]
    results = []
    pbar = tqdm(units, total=len(units),
                desc=f"{method_label(args)} {args.run_id}", unit="run")
    for problem, rep in pbar:
        pbar.set_postfix(problem=problem, rep=rep)
        result, status = run_one(problem, rep, args)
        results.append(result)
        if status == "skipped":
            print(f"[skip] {problem} rep={rep}", flush=True)
        else:
            print(f"[done] {problem} rep={rep} "
                  f"HV={result['hv_final']:.6f} "
                  f"IGD={result['igd_final']:.6f} "
                  f"CVR={result['cvr_final']:.3f} "
                  f"time={result['total_time_sec']:.1f}s",
                  flush=True)
    pbar.close()

    if results:
        write_summary(results, args.output_dir)
        print(f"[summary] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
