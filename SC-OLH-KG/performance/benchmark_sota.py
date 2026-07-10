"""Compare OLH-KG variants with lightweight and real BoTorch baselines."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig  # noqa: E402
from baselines.baseline_algorithms import BaselineConfig, SequentialBaseline  # noqa: E402
from baselines.botorch_adapters import (  # noqa: E402
    BoTorchBaseline,
    BoTorchBaselineConfig,
    fallback_method,
    is_botorch_available,
)
from benchmark_quality import json_safe, nested_get, parse_csv, parse_weights, write_csv  # noqa: E402
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


BOTORCH_ALIASES = {
    "turbo": "botorch_turbo",
    "scbo": "botorch_scbo",
    "saasbo": "botorch_saasbo",
}


def finite(values):
    out = []
    for value in values:
        if value is None:
            continue
        val = float(value)
        if val == val and abs(val) < float("inf"):
            out.append(val)
    return out


def optional_float(value):
    if value is None:
        return None
    return float(value)


def _safe_name(value):
    return "".join(
        c if c.isalnum() or c in ("-", "_", ".") else "_"
        for c in str(value)
    )


def stats(values):
    vals = finite(values)
    if not vals:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def olhkg_variant_name(use_sc, acquisition_mode):
    mode = str(acquisition_mode or "additive").lower()
    if mode == "exact_mc":
        return "olhkg_sc_exact" if use_sc else "olhkg_exact"
    if mode == "blend":
        return "olhkg_sc_blend" if use_sc else "olhkg_blend"
    return "olhkg_sc_additive" if use_sc else "olhkg_additive"


def make_wrapped_problem(args):
    base = make_problem(
        args.problem,
        d=args.d,
        L=args.L,
        sigma=args.sigma,
        alpha=args.alpha,
    )
    return ScalarizedProblem(base, weights=parse_weights(args.weights))


def row_from_result(variant, seed, args, result):
    true_feasible = bool(result["true_feasible"])
    posterior_feasible = bool(result.get("posterior_feasible", False))
    feasible_regret = float(result["simple_regret"]) if true_feasible else None
    return {
        "variant": variant,
        "seed": int(seed),
        "problem": args.problem,
        "N": int(args.N),
        "n0": int(args.n0),
        "x_recommended": result["x_recommended"],
        "true_feasible": true_feasible,
        "posterior_feasible": posterior_feasible,
        "false_feasible": bool(posterior_feasible and not true_feasible),
        "true_objective": float(result["true_objective"]),
        "true_f1": optional_float(result.get("true_f1")),
        "true_f2": optional_float(result.get("true_f2")),
        "true_best_f1": optional_float(result.get("true_best_f1")),
        "true_best_f2": optional_float(result.get("true_best_f2")),
        "true_best_objective": float(result["true_best_objective"]),
        "simple_regret": float(result["simple_regret"]),
        "feasible_simple_regret": feasible_regret,
        "true_chance_margin": float(result["true_chance_margin"]),
        "constraint_violation": max(float(result["true_chance_margin"]), 0.0),
        "posterior_chance_margin": float(result.get("posterior_chance_margin", 0.0)),
        "wall_time_sec": float(result["total_time_sec"]),
        "n_simulations": int(result["n_simulations"]),
        "n_distinct_solutions": int(result["n_distinct_solutions"]),
        "use_state_basis": bool(getattr(args, "use_state_basis", False)),
        "state_basis_mode": getattr(args, "state_basis_mode", "raw+state"),
        "encoder_kind": getattr(args, "encoder_kind", "synthetic"),
        "acquisition_mode": getattr(args, "acquisition_mode", ""),
        "beta_g": optional_float(getattr(args, "beta_g", None)),
        "disable_problem_initial_samples": bool(
            getattr(args, "disable_problem_initial_samples", False)),
        "disable_boundary_initial_samples": bool(
            getattr(args, "disable_boundary_initial_samples", False)),
        "disable_recommendation_refinement": bool(
            getattr(args, "disable_recommendation_refinement", False)),
        "certification_mode": getattr(args, "certification_mode", ""),
        "backend": result.get("backend", "lite"),
        "botorch_fit_failures": int(result.get("botorch_fit_failures", 0)),
        "botorch_candidate_failures": int(result.get("botorch_candidate_failures", 0)),
        "botorch_timeout_fallback": bool(result.get("botorch_timeout_fallback", False)),
        "embedding_dim": optional_float(getattr(args, "embedding_dim", None)),
        "embedding_dim_max": optional_float(getattr(args, "embedding_dim_max", None)),
        "embedding_dim_final": optional_float(result.get("embedding_dim_final")),
    }


def run_olhkg(args, seed, use_sc):
    problem = make_wrapped_problem(args)
    checkpoint_dir = str(getattr(args, "checkpoint_dir", "") or "").strip()
    if checkpoint_dir:
        checkpoint_dir = str(
            Path(checkpoint_dir)
            / _safe_name(olhkg_variant_name(use_sc, args.acquisition_mode))
            / f"seed{int(seed)}"
        )
    config = SingleOLHKGConfig(
        N=args.N,
        n0=args.n0,
        K1=args.K1,
        K2=args.K2,
        posterior_pool_size=args.posterior_pool_size,
        posterior_keep=args.posterior_keep,
        axis_candidate_count=args.axis_candidate_count,
        structured_candidate_count=args.structured_candidate_count,
        state_candidate_count=args.state_candidate_count,
        state_inverse_pool_size=args.state_inverse_pool_size,
        state_inverse_neighbors=args.state_inverse_neighbors,
        variance_mode=args.variance_mode,
        lambda_feas=args.lambda_feas,
        lambda_var=args.lambda_var,
        lambda_mean=args.lambda_mean,
        lambda_coupling=args.lambda_coupling if use_sc else 0.0,
        beta_g=args.beta_g,
        certification_mode=args.certification_mode,
        recommendation_safety_z=args.recommendation_safety_z,
        recommendation_noise_floor_scale=args.recommendation_noise_floor_scale,
        recommendation_infeasible_penalty=args.recommendation_infeasible_penalty,
        recommendation_calibration=not args.disable_recommendation_calibration,
        recommendation_calibration_ridge=args.recommendation_calibration_ridge,
        recommendation_axis_oracle=not args.disable_recommendation_axis_oracle,
        use_problem_initial_samples=not args.disable_problem_initial_samples,
        use_boundary_initial_samples=not args.disable_boundary_initial_samples,
        use_recommendation_refinement=not args.disable_recommendation_refinement,
        acquisition_mode=args.acquisition_mode,
        exact_kg_mc_samples=args.exact_kg_mc_samples,
        exact_kg_jobs=int(getattr(args, "exact_kg_jobs", 1)),
        exact_kg_use_score=args.exact_kg_use_score,
        exact_kg_blend=args.exact_kg_blend,
        checkpoint_dir=checkpoint_dir,
        checkpoint_resume=bool(getattr(args, "checkpoint_resume", False)),
        checkpoint_interval=int(getattr(args, "checkpoint_interval", 1)),
        checkpoint_keep_last=int(getattr(args, "checkpoint_keep_last", 3)),
        progress_logging=bool(getattr(args, "progress_logging", False)),
        progress_label=(
            f"{olhkg_variant_name(use_sc, args.acquisition_mode)}:"
            f"seed={int(seed)}"
        ),
        progress_units_per_iteration=int(
            getattr(args, "progress_units_per_iteration", 100)),
        progress_exact_updates=int(getattr(args, "progress_exact_updates", 10)),
        eval_pool_size=args.eval_pool_size,
        use_state_coupling=use_sc,
        use_state_basis=bool(use_sc and args.use_state_basis),
        state_basis_mode=args.state_basis_mode,
        raw_basis_dim=args.raw_basis_dim,
        raw_projection_seed=args.raw_projection_seed,
        numeric_backend=args.numeric_backend,
        numeric_backend_device=args.numeric_backend_device,
        torch_dtype=args.torch_dtype,
        torch_min_rows=args.torch_min_rows,
        encoder_kind=args.encoder_kind,
        encoder_latent_dim=args.encoder_latent_dim,
        encoder_fit_pool_size=args.encoder_fit_pool_size,
        lf_os_max_library_size=args.lf_os_max_library_size,
        lf_os_low_frequency_components=args.lf_os_low_frequency_components,
        lf_os_max_active=args.lf_os_max_active,
        lf_os_graph_neighbors=args.lf_os_graph_neighbors,
        lf_os_residual_floor_scale=args.lf_os_residual_floor_scale,
        lf_os_use_problem_state_anchor=not args.disable_lf_os_problem_state_anchor,
        seed=seed,
    )
    result = SingleOLHKGAlgorithm(problem, config).run(verbose=False)
    return row_from_result(
        olhkg_variant_name(use_sc, args.acquisition_mode),
        seed,
        args,
        result,
    )


def run_baseline(args, seed, method):
    problem = make_wrapped_problem(args)
    method = BOTORCH_ALIASES.get(method, method)
    if method.startswith("botorch_"):
        if not is_botorch_available():
            if args.botorch_fallback == "error":
                raise RuntimeError(
                    f"{method} requested but BoTorch is not importable; "
                    "install botorch/gpytorch or use --botorch_fallback lite"
                )
            lite_method = fallback_method(method)
            config = BaselineConfig(
                N=args.N,
                n0=args.n0,
                seed=seed,
                method=lite_method,
                batch_candidates=args.baseline_batch_candidates,
                tr_radius_init=args.tr_radius_init,
                tr_radius_min=args.tr_radius_min,
                tr_radius_max=args.tr_radius_max,
                tr_success_tolerance=args.tr_success_tolerance,
                tr_failure_tolerance=args.tr_failure_tolerance,
                embedding_dim=args.embedding_dim,
                embedding_dim_max=args.embedding_dim_max,
                use_problem_initial_samples=not args.disable_problem_initial_samples,
                use_boundary_initial_samples=not args.disable_boundary_initial_samples,
                progress_logging=bool(getattr(args, "progress_logging", False)),
                progress_label=f"{method}:seed={int(seed)}",
            )
            result = SequentialBaseline(problem, config).run()
            result["method"] = method
            result["backend"] = f"fallback:{lite_method}"
            return row_from_result(method, seed, args, result)
        config = BoTorchBaselineConfig(
            N=args.N,
            n0=args.n0,
            seed=seed,
            method=method,
            batch_candidates=args.baseline_batch_candidates,
            tr_radius_init=args.tr_radius_init,
            tr_radius_min=args.tr_radius_min,
            tr_radius_max=args.tr_radius_max,
            tr_success_tolerance=args.tr_success_tolerance,
            tr_failure_tolerance=args.tr_failure_tolerance,
            raw_samples=args.botorch_raw_samples,
            num_restarts=args.botorch_num_restarts,
            maxiter=args.botorch_maxiter,
            timeout_sec=args.botorch_timeout_sec,
            saas_warmup_steps=args.saas_warmup_steps,
            saas_num_samples=args.saas_num_samples,
            saas_thinning=args.saas_thinning,
            saas_max_tree_depth=args.saas_max_tree_depth,
            saas_mc_samples=args.saas_mc_samples,
            saas_constrained=not args.saas_unconstrained,
            max_candidate_failures=args.botorch_max_candidate_failures,
            saas_fallback_after_failures=not args.disable_saas_failure_fallback,
            use_problem_initial_samples=not args.disable_problem_initial_samples,
            use_boundary_initial_samples=not args.disable_boundary_initial_samples,
            progress_logging=bool(getattr(args, "progress_logging", False)),
            progress_label=f"{method}:seed={int(seed)}",
        )
        result = BoTorchBaseline(problem, config).run()
        return row_from_result(method, seed, args, result)
    config = BaselineConfig(
        N=args.N,
        n0=args.n0,
        seed=seed,
        method=method,
        batch_candidates=args.baseline_batch_candidates,
        tr_radius_init=args.tr_radius_init,
        tr_radius_min=args.tr_radius_min,
        tr_radius_max=args.tr_radius_max,
        tr_success_tolerance=args.tr_success_tolerance,
        tr_failure_tolerance=args.tr_failure_tolerance,
        embedding_dim=args.embedding_dim,
        embedding_dim_max=args.embedding_dim_max,
        use_problem_initial_samples=not args.disable_problem_initial_samples,
        use_boundary_initial_samples=not args.disable_boundary_initial_samples,
        progress_logging=bool(getattr(args, "progress_logging", False)),
        progress_label=f"{method}:seed={int(seed)}",
    )
    result = SequentialBaseline(problem, config).run()
    return row_from_result(method, seed, args, result)


def _worker_init(torch_threads):
    threads = max(1, int(torch_threads))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    try:
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _run_variant_task(args_dict, name, method, seed):
    args = argparse.Namespace(**args_dict)
    if name == "olhkg":
        return run_olhkg(args, seed, use_sc=False)
    if name == "olhkg_sc":
        return run_olhkg(args, seed, use_sc=True)
    return run_baseline(args, seed, method)


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    summaries: dict[str, dict[str, Any]] = {}
    for variant, items in grouped.items():
        summaries[variant] = {
            "variant": variant,
            "n_runs": len(items),
            "true_feasible_rate": sum(bool(r["true_feasible"]) for r in items) / len(items),
            "false_feasible_rate": sum(bool(r["false_feasible"]) for r in items) / len(items),
            "feasible_simple_regret": stats(r["feasible_simple_regret"] for r in items),
            "constraint_violation": stats(r["constraint_violation"] for r in items),
            "wall_time_sec": stats(r["wall_time_sec"] for r in items),
        }
    baseline = (
        summaries.get("olhkg")
        or summaries.get("olhkg_additive")
        or summaries.get("olhkg_exact")
        or summaries.get("olhkg_blend")
    )
    if baseline is not None:
        base_regret = nested_get(baseline, ["feasible_simple_regret", "median"])
        base_false = baseline["false_feasible_rate"]
        base_feas = baseline["true_feasible_rate"]
        for summary in summaries.values():
            regret = nested_get(summary, ["feasible_simple_regret", "median"])
            summary["vs_olhkg"] = {
                "baseline_variant": baseline["variant"],
                "feasible_rate_delta": float(summary["true_feasible_rate"] - base_feas),
                "false_feasible_rate_delta": float(summary["false_feasible_rate"] - base_false),
                "feasible_regret_median_delta": (
                    None if regret is None or base_regret is None
                    else float(regret - base_regret)
                ),
            }
    return summaries


def flatten_summary(summary):
    row = {
        "variant": summary["variant"],
        "n_runs": summary["n_runs"],
        "true_feasible_rate": summary["true_feasible_rate"],
        "false_feasible_rate": summary["false_feasible_rate"],
    }
    for metric in ("feasible_simple_regret", "constraint_violation", "wall_time_sec"):
        for key, value in summary[metric].items():
            row[f"{metric}_{key}"] = value
    for key, value in summary.get("vs_olhkg", {}).items():
        row[f"vs_olhkg_{key}"] = value
    return row


def run_benchmark(args):
    seeds = parse_csv(args.seeds, int)
    if not seeds:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    variants = []
    if args.include_olhkg:
        variants.append(("olhkg", None))
    if args.include_sc:
        variants.append(("olhkg_sc", None))
    for method in parse_csv(args.baselines):
        variants.append((method, method))
    tasks = [
        (name, method, seed)
        for name, method in variants
        for seed in seeds
    ]
    rows = []
    jobs = max(1, int(getattr(args, "jobs", 1)))
    total = len(tasks)
    if jobs == 1:
        completed = 0
        for name, method, seed in tasks:
            print(
                f"[{completed}/{total}] [sota] start "
                f"variant={name} seed={seed}",
                flush=True,
            )
            rows.append(_run_variant_task(vars(args), name, method, seed))
            completed += 1
            print(
                f"[{completed}/{total}] [sota] done "
                f"variant={name} seed={seed}",
                flush=True,
            )
    else:
        print(f"[0/{total}] [sota] running {len(tasks)} tasks with jobs={jobs}", flush=True)
        args_dict = vars(args)
        completed = 0
        with ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_worker_init,
            initargs=(getattr(args, "worker_torch_threads", 1),),
        ) as executor:
            future_map = {
                executor.submit(_run_variant_task, args_dict, name, method, seed):
                (name, seed)
                for name, method, seed in tasks
            }
            for future in as_completed(future_map):
                name, seed = future_map[future]
                row = future.result()
                rows.append(row)
                completed += 1
                print(
                    f"[{completed}/{total}] [sota] done "
                    f"variant={name} seed={seed}",
                    flush=True,
                )
    rows.sort(key=lambda row: (str(row["variant"]), int(row["seed"])))
    summaries = summarize(rows)
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": vars(args),
        "rows": rows,
        "summary": summaries,
    }


def write_outputs(args, result):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"sota_benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
    json_path = out_dir / f"{prefix}.json"
    rows_path = out_dir / f"{prefix}_rows.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"
    json_path.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    write_csv(rows_path, result["rows"])
    write_csv(summary_path, [flatten_summary(s) for s in result["summary"].values()])
    return {"json": str(json_path), "rows_csv": str(rows_path), "summary_csv": str(summary_path)}


def printable_summary(result):
    rows = []
    for variant, summary in result["summary"].items():
        rows.append({
            "variant": variant,
            "true_feasible_rate": summary["true_feasible_rate"],
            "false_feasible_rate": summary["false_feasible_rate"],
            "feasible_regret_median": nested_get(
                summary, ["feasible_simple_regret", "median"]),
            "violation_mean": nested_get(summary, ["constraint_violation", "mean"]),
            "wall_time_median_sec": nested_get(summary, ["wall_time_sec", "median"]),
            "vs_olhkg": summary.get("vs_olhkg", {}),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="FactorShockStatePolicyRZDT1")
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=5)
    parser.add_argument("--K1", type=int, default=15)
    parser.add_argument("--K2", type=int, default=1)
    parser.add_argument("--posterior_pool_size", type=int, default=300)
    parser.add_argument("--posterior_keep", type=int, default=15)
    parser.add_argument("--axis_candidate_count", type=int, default=-1)
    parser.add_argument("--structured_candidate_count", type=int, default=0)
    parser.add_argument("--state_candidate_count", type=int, default=-1)
    parser.add_argument("--state_inverse_pool_size", type=int, default=500)
    parser.add_argument("--state_inverse_neighbors", type=int, default=2)
    parser.add_argument("--use_state_basis", dest="use_state_basis", action="store_true", default=True)
    parser.add_argument("--disable_state_basis", dest="use_state_basis", action="store_false")
    parser.add_argument(
        "--state_basis_mode",
        default="raw+state",
        choices=["raw", "state", "raw+state", "manifold", "raw+manifold"],
    )
    parser.add_argument("--raw_basis_dim", type=int, default=-1)
    parser.add_argument("--raw_projection_seed", type=int, default=314159)
    parser.add_argument(
        "--numeric_backend",
        default="numpy",
        choices=["numpy", "auto", "torch", "torch_cuda", "cuda"],
    )
    parser.add_argument("--numeric_backend_device", default="auto")
    parser.add_argument(
        "--torch_dtype",
        default="float64",
        choices=["float64", "float32", "double", "single"],
    )
    parser.add_argument("--torch_min_rows", type=int, default=128)
    parser.add_argument(
        "--encoder_kind",
        default="synthetic",
        choices=[
            "synthetic",
            "self_supervised",
            "masked",
            "contrastive",
            "transformer",
            "pca_manifold",
            "kernel_manifold",
            "graph_laplacian",
            "diffusion_manifold",
            "graph_manifold",
            "ssl_masked",
            "ssl_contrastive",
            "ssl_next_risk",
            "ssl_transformer",
            "ssl_hybrid",
            "hybrid_ssl",
            "contextual_manifold",
            "lf_os",
            "lf_orthogonal_sparse",
            "low_frequency_orthogonal_sparse",
            "orthogonal_sparse",
        ],
    )
    parser.add_argument("--encoder_latent_dim", type=int, default=8)
    parser.add_argument("--encoder_fit_pool_size", type=int, default=512)
    parser.add_argument("--lf_os_max_library_size", type=int, default=30)
    parser.add_argument("--lf_os_low_frequency_components", type=int, default=8)
    parser.add_argument("--lf_os_max_active", type=int, default=8)
    parser.add_argument("--lf_os_graph_neighbors", type=int, default=12)
    parser.add_argument("--lf_os_residual_floor_scale", type=float, default=0.05)
    parser.add_argument("--disable_lf_os_problem_state_anchor", action="store_true")
    parser.add_argument("--variance_mode", default="factor")
    parser.add_argument("--lambda_feas", type=float, default=0.25)
    parser.add_argument("--lambda_var", type=float, default=0.25)
    parser.add_argument("--lambda_mean", type=float, default=0.10)
    parser.add_argument("--lambda_coupling", type=float, default=0.05)
    parser.add_argument("--beta_g", type=float, default=2.0)
    parser.add_argument("--certification_mode", default="theory",
                        choices=["theory", "legacy"])
    parser.add_argument("--recommendation_safety_z", type=float, default=0.5)
    parser.add_argument("--recommendation_noise_floor_scale", type=float, default=1.0)
    parser.add_argument("--recommendation_infeasible_penalty", type=float, default=5.0)
    parser.add_argument("--disable_recommendation_calibration", action="store_true")
    parser.add_argument("--recommendation_calibration_ridge", type=float, default=1e-6)
    parser.add_argument("--disable_recommendation_axis_oracle", action="store_true")
    parser.add_argument("--disable_problem_initial_samples", action="store_true")
    parser.add_argument("--disable_boundary_initial_samples", action="store_true")
    parser.add_argument("--disable_recommendation_refinement", action="store_true")
    parser.add_argument("--acquisition_mode", default="exact_mc",
                        choices=["additive", "exact_mc", "blend"])
    parser.add_argument("--exact_kg_mc_samples", type=int, default=8)
    parser.add_argument("--exact_kg_jobs", type=int, default=1)
    parser.add_argument("--exact_kg_use_score", action="store_true")
    parser.add_argument("--exact_kg_blend", type=float, default=0.0)
    parser.add_argument("--checkpoint_dir", default="")
    parser.add_argument("--checkpoint_resume", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument("--checkpoint_keep_last", type=int, default=3)
    parser.add_argument("--progress_logging", dest="progress_logging", action="store_true", default=True)
    parser.add_argument("--disable_progress_logging", dest="progress_logging", action="store_false")
    parser.add_argument("--progress_units_per_iteration", type=int, default=100)
    parser.add_argument("--progress_exact_updates", type=int, default=10)
    parser.add_argument("--eval_pool_size", type=int, default=500)
    parser.add_argument(
        "--baselines",
        default=(
            "sobol,random,hetgp_lite,rahbo_lite,safeopt_lite,"
            "legacy_vepm_lite,turbo_lite,scbo_lite,rembo_lite,baxus_lite"
        ),
    )
    parser.add_argument("--baseline_batch_candidates", type=int, default=64)
    parser.add_argument("--tr_radius_init", type=float, default=0.35)
    parser.add_argument("--tr_radius_min", type=float, default=0.04)
    parser.add_argument("--tr_radius_max", type=float, default=0.8)
    parser.add_argument("--embedding_dim", type=int, default=8)
    parser.add_argument("--embedding_dim_max", type=int, default=32)
    parser.add_argument("--tr_success_tolerance", type=int, default=3)
    parser.add_argument("--tr_failure_tolerance", type=int, default=5)
    parser.add_argument("--botorch_fallback", choices=("lite", "error"), default="lite")
    parser.add_argument("--botorch_raw_samples", type=int, default=64)
    parser.add_argument("--botorch_num_restarts", type=int, default=5)
    parser.add_argument("--botorch_maxiter", type=int, default=50)
    parser.add_argument("--botorch_timeout_sec", type=float, default=30.0)
    parser.add_argument("--botorch_max_candidate_failures", type=int, default=8)
    parser.add_argument("--saas_warmup_steps", type=int, default=16)
    parser.add_argument("--saas_num_samples", type=int, default=16)
    parser.add_argument("--saas_thinning", type=int, default=1)
    parser.add_argument("--saas_max_tree_depth", type=int, default=4)
    parser.add_argument("--saas_mc_samples", type=int, default=64)
    parser.add_argument("--saas_unconstrained", action="store_true")
    parser.add_argument("--disable_saas_failure_fallback", action="store_true")
    parser.add_argument("--include_olhkg", action="store_true", default=True)
    parser.add_argument("--include_sc", action="store_true", default=True)
    parser.add_argument("--exclude_olhkg", dest="include_olhkg", action="store_false")
    parser.add_argument("--exclude_sc", dest="include_sc", action="store_false")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--worker_torch_threads", type=int, default=1)
    parser.add_argument("--out_dir", default=str(ROOT / "profiles"))
    parser.add_argument("--out_prefix", default="")
    args = parser.parse_args()

    result = run_benchmark(args)
    paths = write_outputs(args, result)
    print(json.dumps(json_safe({
        "paths": paths,
        "summary": printable_summary(result),
    }), indent=2))


if __name__ == "__main__":
    main()
