"""Run HVD and SC ablations across several synthetic problems."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark_quality  # noqa: E402
from benchmark_quality import json_safe, parse_csv, write_csv  # noqa: E402


def _safe_name(value):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))


def _problem_prefix(prefix, problem, n, n_seeds):
    stem = prefix or f"hvd_suite_{time.strftime('%Y%m%d_%H%M%S')}"
    return f"{stem}_{_safe_name(problem)}_n{int(n)}_s{int(n_seeds)}"


def _problem_args(args, problem):
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
        "n_thr": args.n_thr,
        "eval_pool_size": args.eval_pool_size,
        "lambda_feas": args.lambda_feas,
        "lambda_var": args.lambda_var,
        "lambda_mean": args.lambda_mean,
        "lambda_coupling": args.lambda_coupling,
        "beta_g": args.beta_g,
        "certification_mode": args.certification_mode,
        "coupling_safety_z": args.coupling_safety_z,
        "coupling_gate_temperature": args.coupling_gate_temperature,
        "recommendation_safety_z": args.recommendation_safety_z,
        "recommendation_noise_floor_scale": args.recommendation_noise_floor_scale,
        "recommendation_infeasible_penalty": args.recommendation_infeasible_penalty,
        "disable_recommendation_calibration": args.disable_recommendation_calibration,
        "recommendation_calibration_ridge": args.recommendation_calibration_ridge,
        "disable_recommendation_axis_oracle": args.disable_recommendation_axis_oracle,
        "use_state_basis": args.use_state_basis,
        "state_basis_mode": args.state_basis_mode,
        "encoder_kind": args.encoder_kind,
        "encoder_latent_dim": args.encoder_latent_dim,
        "encoder_fit_pool_size": args.encoder_fit_pool_size,
        "exact_kg_mc_samples": args.exact_kg_mc_samples,
        "exact_kg_use_score": args.exact_kg_use_score,
        "exact_kg_blend": args.exact_kg_blend,
        "acquisition_modes": args.acquisition_modes,
        "modes": args.modes,
        "sc_modes": args.sc_modes,
        "baseline_variant": args.baseline_variant,
        "seeds": args.seeds,
        "seed_start": args.seed_start,
        "n_seeds": args.n_seeds,
        "out_dir": args.out_dir,
        "out_prefix": _problem_prefix(args.out_prefix, problem, args.N, args.n_seeds),
        "verbose": args.verbose,
    }
    return SimpleNamespace(**fields)


def _run_problem(problem_args):
    result = benchmark_quality.run_benchmark(problem_args)
    paths = benchmark_quality.write_outputs(problem_args, result)
    return problem_args.problem, paths, result


def _write_suite_outputs(args, result):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"hvd_suite_{time.strftime('%Y%m%d_%H%M%S')}"
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
    problem_args = [
        _problem_args(args, problem)
        for problem in parse_csv(args.problems)
    ]
    problem_results = {}
    all_rows = []
    summary_rows = []

    if int(args.jobs) > 1 and len(problem_args) > 1:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as pool:
            futures = {pool.submit(_run_problem, item): item.problem for item in problem_args}
            for future in as_completed(futures):
                problem, paths, result = future.result()
                problem_results[problem] = {"paths": paths, "summary": result["summary"]}
                for row in result["rows"]:
                    row = dict(row)
                    row["problem"] = problem
                    all_rows.append(row)
                for summary in result["summary"].values():
                    flat = benchmark_quality.flatten_summary(summary)
                    flat["problem"] = problem
                    summary_rows.append(flat)
    else:
        for item in problem_args:
            print(f"[hvd-suite] problem={item.problem}", flush=True)
            problem, paths, result = _run_problem(item)
            problem_results[problem] = {"paths": paths, "summary": result["summary"]}
            for row in result["rows"]:
                row = dict(row)
                row["problem"] = problem
                all_rows.append(row)
            for summary in result["summary"].values():
                flat = benchmark_quality.flatten_summary(summary)
                flat["problem"] = problem
                summary_rows.append(flat)

    grouped = {}
    for row in all_rows:
        grouped.setdefault(row["variant"], []).append(row)
    pooled = {
        variant: benchmark_quality.summarize_variant(rows)
        for variant, rows in grouped.items()
    }
    benchmark_quality.compare_to_baseline(pooled, args.baseline_variant)
    pooled_rows = [
        benchmark_quality.flatten_summary(summary)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problems",
        default="RegimeRZDT1,RZDT2,StatePolicyRZDT1,FactorShockStatePolicyRZDT1",
    )
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--N", type=int, default=40)
    parser.add_argument("--n0", type=int, default=8)
    parser.add_argument("--K1", type=int, default=25)
    parser.add_argument("--K2", type=int, default=1)
    parser.add_argument("--posterior_pool_size", type=int, default=300)
    parser.add_argument("--posterior_keep", type=int, default=15)
    parser.add_argument("--axis_candidate_count", type=int, default=-1)
    parser.add_argument("--structured_candidate_count", type=int, default=0)
    parser.add_argument("--state_candidate_count", type=int, default=-1)
    parser.add_argument("--state_inverse_pool_size", type=int, default=500)
    parser.add_argument("--state_inverse_neighbors", type=int, default=2)
    parser.add_argument("--n_thr", type=int, default=5)
    parser.add_argument("--eval_pool_size", type=int, default=500)
    parser.add_argument("--lambda_feas", type=float, default=0.25)
    parser.add_argument("--lambda_var", type=float, default=0.25)
    parser.add_argument("--lambda_mean", type=float, default=0.10)
    parser.add_argument("--lambda_coupling", type=float, default=0.05)
    parser.add_argument("--beta_g", type=float, default=2.0)
    parser.add_argument("--certification_mode", default="theory",
                        choices=["theory", "legacy"])
    parser.add_argument("--coupling_safety_z", type=float, default=0.5)
    parser.add_argument("--coupling_gate_temperature", type=float, default=0.25)
    parser.add_argument("--recommendation_safety_z", type=float, default=0.5)
    parser.add_argument("--recommendation_noise_floor_scale", type=float, default=1.0)
    parser.add_argument("--recommendation_infeasible_penalty", type=float, default=5.0)
    parser.add_argument("--disable_recommendation_calibration", action="store_true")
    parser.add_argument("--recommendation_calibration_ridge", type=float, default=1e-6)
    parser.add_argument("--disable_recommendation_axis_oracle", action="store_true")
    parser.add_argument("--use_state_basis", action="store_true")
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
            "ssl_masked",
            "ssl_contrastive",
            "ssl_next_risk",
            "ssl_transformer",
        ],
    )
    parser.add_argument("--encoder_latent_dim", type=int, default=8)
    parser.add_argument("--encoder_fit_pool_size", type=int, default=512)
    parser.add_argument("--exact_kg_mc_samples", type=int, default=0)
    parser.add_argument("--exact_kg_use_score", action="store_true")
    parser.add_argument("--exact_kg_blend", type=float, default=0.0)
    parser.add_argument("--acquisition_modes", default="additive")
    parser.add_argument("--modes", default="pooled,class,orthogonal,factor")
    parser.add_argument("--sc_modes", default="orthogonal,factor")
    parser.add_argument("--baseline_variant", default="orthogonal")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out_dir", default=str(ROOT / "profiles"))
    parser.add_argument("--out_prefix", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = run_suite(args)
    paths = _write_suite_outputs(args, result)
    print(json.dumps(json_safe({
        "paths": paths,
        "summary_rows": result["summary_rows"],
        "pooled_summary_rows": result["pooled_summary_rows"],
    }), indent=2))


if __name__ == "__main__":
    main()
