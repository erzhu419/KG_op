"""Run the SOTA benchmark over several problems and aggregate summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark_sota  # noqa: E402
from benchmark_quality import json_safe, parse_csv, write_csv  # noqa: E402


def _problem_prefix(prefix, problem, n, n_seeds):
    stem = prefix or f"sota_suite_{time.strftime('%Y%m%d_%H%M%S')}"
    safe_problem = "".join(ch.lower() if ch.isalnum() else "_" for ch in problem)
    return f"{stem}_{safe_problem}_n{int(n)}_s{int(n_seeds)}"


def _problem_args(args, problem):
    safe_problem = "".join(ch.lower() if ch.isalnum() else "_" for ch in problem)
    fields = {
        "problem": problem,
        "d": args.d,
        "L": args.L,
        "sigma": args.sigma,
        "alpha": args.alpha,
        "weights": args.weights,
        "N": args.N,
        "n0": args.n0,
        "K1": args.K1,
        "K2": args.K2,
        "posterior_pool_size": args.posterior_pool_size,
        "posterior_keep": args.posterior_keep,
        "axis_candidate_count": args.axis_candidate_count,
        "structured_candidate_count": args.structured_candidate_count,
        "state_candidate_count": args.state_candidate_count,
        "state_inverse_pool_size": args.state_inverse_pool_size,
        "state_inverse_neighbors": args.state_inverse_neighbors,
        "use_state_basis": args.use_state_basis,
        "state_basis_mode": args.state_basis_mode,
        "raw_basis_dim": args.raw_basis_dim,
        "raw_projection_seed": args.raw_projection_seed,
        "numeric_backend": args.numeric_backend,
        "numeric_backend_device": args.numeric_backend_device,
        "torch_dtype": args.torch_dtype,
        "torch_min_rows": args.torch_min_rows,
        "encoder_kind": args.encoder_kind,
        "encoder_latent_dim": args.encoder_latent_dim,
        "encoder_fit_pool_size": args.encoder_fit_pool_size,
        "lf_os_max_library_size": args.lf_os_max_library_size,
        "lf_os_low_frequency_components": args.lf_os_low_frequency_components,
        "lf_os_max_active": args.lf_os_max_active,
        "lf_os_graph_neighbors": args.lf_os_graph_neighbors,
        "lf_os_residual_floor_scale": args.lf_os_residual_floor_scale,
        "disable_lf_os_problem_state_anchor": args.disable_lf_os_problem_state_anchor,
        "variance_mode": args.variance_mode,
        "lambda_feas": args.lambda_feas,
        "lambda_var": args.lambda_var,
        "lambda_mean": args.lambda_mean,
        "lambda_coupling": args.lambda_coupling,
        "beta_g": args.beta_g,
        "certification_mode": args.certification_mode,
        "recommendation_safety_z": args.recommendation_safety_z,
        "recommendation_noise_floor_scale": args.recommendation_noise_floor_scale,
        "recommendation_infeasible_penalty": args.recommendation_infeasible_penalty,
        "disable_recommendation_calibration": args.disable_recommendation_calibration,
        "recommendation_calibration_ridge": args.recommendation_calibration_ridge,
        "disable_recommendation_axis_oracle": args.disable_recommendation_axis_oracle,
        "disable_problem_initial_samples": args.disable_problem_initial_samples,
        "disable_boundary_initial_samples": args.disable_boundary_initial_samples,
        "disable_recommendation_refinement": args.disable_recommendation_refinement,
        "acquisition_mode": args.acquisition_mode,
        "exact_kg_mc_samples": args.exact_kg_mc_samples,
        "exact_kg_jobs": args.exact_kg_jobs,
        "exact_kg_use_score": args.exact_kg_use_score,
        "exact_kg_blend": args.exact_kg_blend,
        "checkpoint_dir": (
            str(Path(args.checkpoint_dir) / safe_problem)
            if str(args.checkpoint_dir or "").strip()
            else ""
        ),
        "checkpoint_resume": args.checkpoint_resume,
        "checkpoint_interval": args.checkpoint_interval,
        "checkpoint_keep_last": args.checkpoint_keep_last,
        "progress_logging": args.progress_logging,
        "progress_units_per_iteration": args.progress_units_per_iteration,
        "progress_exact_updates": args.progress_exact_updates,
        "eval_pool_size": args.eval_pool_size,
        "baselines": args.baselines,
        "baseline_batch_candidates": args.baseline_batch_candidates,
        "tr_radius_init": args.tr_radius_init,
        "tr_radius_min": args.tr_radius_min,
        "tr_radius_max": args.tr_radius_max,
        "tr_success_tolerance": args.tr_success_tolerance,
        "tr_failure_tolerance": args.tr_failure_tolerance,
        "embedding_dim": args.embedding_dim,
        "embedding_dim_max": args.embedding_dim_max,
        "botorch_fallback": args.botorch_fallback,
        "botorch_raw_samples": args.botorch_raw_samples,
        "botorch_num_restarts": args.botorch_num_restarts,
        "botorch_maxiter": args.botorch_maxiter,
        "botorch_timeout_sec": args.botorch_timeout_sec,
        "botorch_max_candidate_failures": args.botorch_max_candidate_failures,
        "saas_warmup_steps": args.saas_warmup_steps,
        "saas_num_samples": args.saas_num_samples,
        "saas_thinning": args.saas_thinning,
        "saas_max_tree_depth": args.saas_max_tree_depth,
        "saas_mc_samples": args.saas_mc_samples,
        "saas_unconstrained": args.saas_unconstrained,
        "disable_saas_failure_fallback": args.disable_saas_failure_fallback,
        "include_olhkg": args.include_olhkg,
        "include_sc": args.include_sc,
        "seeds": args.seeds,
        "seed_start": args.seed_start,
        "n_seeds": args.n_seeds,
        "jobs": args.jobs,
        "worker_torch_threads": args.worker_torch_threads,
        "out_dir": args.out_dir,
        "out_prefix": _problem_prefix(args.out_prefix, problem, args.N, args.n_seeds),
    }
    return SimpleNamespace(**fields)


def _write_suite_outputs(args, result):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"sota_suite_{time.strftime('%Y%m%d_%H%M%S')}"
    json_path = out_dir / f"{prefix}.json"
    rows_path = out_dir / f"{prefix}_rows.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"
    pooled_path = out_dir / f"{prefix}_pooled_summary.csv"
    json_path.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    write_csv(rows_path, result["rows"])
    write_csv(summary_path, result["summary_rows"])
    write_csv(pooled_path, result["pooled_summary_rows"])
    return {
        "json": str(json_path),
        "rows_csv": str(rows_path),
        "summary_csv": str(summary_path),
        "pooled_summary_csv": str(pooled_path),
    }


def run_suite(args):
    all_rows = []
    summary_rows = []
    problem_results = {}
    problems = parse_csv(args.problems)
    total = len(problems)
    completed = 0
    for problem in problems:
        problem_args = _problem_args(args, problem)
        print(
            f"[{completed}/{total}] [sota-suite] start problem={problem}",
            flush=True,
        )
        result = benchmark_sota.run_benchmark(problem_args)
        paths = benchmark_sota.write_outputs(problem_args, result)
        problem_results[problem] = {
            "paths": paths,
            "summary": result["summary"],
        }
        for row in result["rows"]:
            row = dict(row)
            row["problem"] = problem
            all_rows.append(row)
        for summary in result["summary"].values():
            flat = benchmark_sota.flatten_summary(summary)
            flat["problem"] = problem
            summary_rows.append(flat)
        completed += 1
        print(
            f"[{completed}/{total}] [sota-suite] done problem={problem}",
            flush=True,
        )

    pooled = benchmark_sota.summarize(all_rows) if all_rows else {}
    pooled_rows = [
        benchmark_sota.flatten_summary(summary)
        for summary in pooled.values()
    ]
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": vars(args),
        "problem_results": problem_results,
        "rows": all_rows,
        "summary_rows": summary_rows,
        "pooled_summary": pooled,
        "pooled_summary_rows": pooled_rows,
    }


def printable_summary(result):
    return {
        "by_problem": result["summary_rows"],
        "pooled": result["pooled_summary_rows"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="FactorShockStatePolicyRZDT1,InventorySupplyChain,QueueResourceControl,StatePolicyRZDT1")
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
            "legacy_vepm_lite,rembo_lite,baxus_lite,"
            "botorch_turbo,botorch_scbo,botorch_saasbo"
        ),
    )
    parser.add_argument("--baseline_batch_candidates", type=int, default=64)
    parser.add_argument("--tr_radius_init", type=float, default=0.35)
    parser.add_argument("--tr_radius_min", type=float, default=0.04)
    parser.add_argument("--tr_radius_max", type=float, default=0.8)
    parser.add_argument("--tr_success_tolerance", type=int, default=3)
    parser.add_argument("--tr_failure_tolerance", type=int, default=5)
    parser.add_argument("--embedding_dim", type=int, default=8)
    parser.add_argument("--embedding_dim_max", type=int, default=32)
    parser.add_argument("--botorch_fallback", choices=("lite", "error"), default="error")
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
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--worker_torch_threads", type=int, default=1)
    parser.add_argument("--out_dir", default=str(ROOT / "profiles"))
    parser.add_argument("--out_prefix", default="")
    args = parser.parse_args()

    result = run_suite(args)
    paths = _write_suite_outputs(args, result)
    print(json.dumps(json_safe({
        "paths": paths,
        "summary": printable_summary(result),
    }), indent=2))


if __name__ == "__main__":
    main()
