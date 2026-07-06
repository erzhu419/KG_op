"""Quality benchmark for OLH-KG variance and coupling variants.

This is not a promotion gate.  It runs multiple seeds and summarizes the
optimization-quality metrics that matter for the chance-constrained problem.
Wall time is reported as an engineering diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig  # noqa: E402
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


def json_safe(obj: Any):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except Exception:
        pass
    return obj


def parse_csv(value, cast=str):
    if value is None or str(value).strip() == "":
        return []
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_weights(value):
    weights = parse_csv(value, float)
    if len(weights) != 2:
        raise ValueError("--weights must contain two comma-separated numbers")
    return tuple(weights)


def _safe_name(value):
    return "".join(
        c if c.isalnum() or c in ("-", "_", ".") else "_"
        for c in str(value)
    )


def finite_values(values):
    out = []
    for value in values:
        if value is None:
            continue
        val = float(value)
        if math.isfinite(val):
            out.append(val)
    return out


def optional_float(value):
    if value is None:
        return None
    return float(value)


def stats(values):
    vals = finite_values(values)
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def mean_bool(values):
    vals = [bool(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else None


def compact_stage_shares(stage_times):
    out = {}
    for key, val in stage_times.items():
        out[f"{key}_share"] = float(val.get("share", 0.0))
        out[f"{key}_mean_sec"] = float(val.get("mean", 0.0))
    return out


def _variant_name(args, variance_mode, use_state_coupling, acquisition_mode):
    modes = parse_csv(getattr(args, "acquisition_modes", "additive"))
    if modes == ["additive"]:
        return variance_mode + ("+sc" if use_state_coupling else "")
    if acquisition_mode == "exact_mc":
        suffix = "olhkg_sc_exact" if use_state_coupling else "olhkg_exact"
    elif acquisition_mode == "blend":
        suffix = "olhkg_sc_blend" if use_state_coupling else "olhkg_blend"
    else:
        suffix = "olhkg_sc_additive" if use_state_coupling else "olhkg_additive"
    return f"{variance_mode}:{suffix}"


def run_variant_once(args, variance_mode, seed, use_state_coupling, acquisition_mode="exact_mc"):
    base = make_problem(
        args.problem,
        d=args.d,
        L=args.L,
        sigma=args.sigma,
        alpha=args.alpha,
    )
    problem = ScalarizedProblem(base, weights=parse_weights(args.weights))
    variant = _variant_name(args, variance_mode, use_state_coupling, acquisition_mode)
    checkpoint_dir = str(getattr(args, "checkpoint_dir", "") or "").strip()
    if checkpoint_dir:
        checkpoint_dir = str(
            Path(checkpoint_dir) / _safe_name(variant) / f"seed{int(seed)}"
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
        n_thr=args.n_thr,
        variance_mode=variance_mode,
        lambda_feas=args.lambda_feas,
        lambda_var=args.lambda_var,
        lambda_mean=args.lambda_mean,
        lambda_coupling=args.lambda_coupling if use_state_coupling else 0.0,
        beta_g=args.beta_g,
        certification_mode=args.certification_mode,
        coupling_safety_z=args.coupling_safety_z,
        coupling_gate_temperature=args.coupling_gate_temperature,
        recommendation_safety_z=args.recommendation_safety_z,
        recommendation_noise_floor_scale=args.recommendation_noise_floor_scale,
        recommendation_infeasible_penalty=args.recommendation_infeasible_penalty,
        recommendation_infeasible_strategy=args.recommendation_infeasible_strategy,
        recommendation_calibration=not args.disable_recommendation_calibration,
        recommendation_calibration_ridge=args.recommendation_calibration_ridge,
        certification_calibration=bool(
            getattr(args, "enable_certification_calibration", False)),
        certification_calibration_min_obs=int(
            getattr(args, "certification_calibration_min_obs", 8)),
        certification_calibration_ridge=float(
            getattr(args, "certification_calibration_ridge", 1e-6)),
        certification_calibration_noise_floor_scale=float(getattr(
            args,
            "certification_calibration_noise_floor_scale",
            0.5,
        )),
        certification_calibration_beta=float(
            getattr(args, "certification_calibration_beta", 2.0)),
        recommendation_axis_oracle=not args.disable_recommendation_axis_oracle,
        use_problem_initial_samples=not getattr(
            args, "disable_problem_initial_samples", False),
        use_boundary_initial_samples=not getattr(
            args, "disable_boundary_initial_samples", False),
        use_recommendation_refinement=not getattr(
            args, "disable_recommendation_refinement", False),
        use_state_coupling=use_state_coupling,
        use_state_basis=bool(use_state_coupling and args.use_state_basis),
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
        acquisition_mode=acquisition_mode,
        exact_kg_mc_samples=args.exact_kg_mc_samples,
        exact_kg_jobs=int(getattr(args, "exact_kg_jobs", 1)),
        exact_kg_use_score=args.exact_kg_use_score,
        exact_kg_blend=args.exact_kg_blend,
        checkpoint_dir=checkpoint_dir,
        checkpoint_resume=bool(getattr(args, "checkpoint_resume", False)),
        checkpoint_interval=int(getattr(args, "checkpoint_interval", 1)),
        checkpoint_keep_last=int(getattr(args, "checkpoint_keep_last", 3)),
        progress_logging=bool(getattr(args, "progress_logging", False)),
        progress_label=str(
            getattr(args, "progress_label", "")
            or f"{variant}:seed={int(seed)}"
        ),
        progress_units_per_iteration=int(
            getattr(args, "progress_units_per_iteration", 100)),
        progress_exact_updates=int(getattr(args, "progress_exact_updates", 10)),
        eval_pool_size=args.eval_pool_size,
        seed=seed,
    )
    alg = SingleOLHKGAlgorithm(problem, config)
    result = alg.run(verbose=args.verbose)
    true_feasible = bool(result["true_feasible"])
    posterior_feasible = bool(result.get("posterior_feasible", False))
    violation = max(float(result["true_chance_margin"]), 0.0)
    feasible_objective = float(result["true_objective"]) if true_feasible else None
    feasible_regret = float(result["simple_regret"]) if true_feasible else None
    row = {
        "variant": variant,
        "variance_mode": variance_mode,
        "use_state_coupling": bool(use_state_coupling),
        "use_state_basis": bool(use_state_coupling and args.use_state_basis),
        "state_basis_mode": args.state_basis_mode,
        "raw_basis_dim": int(args.raw_basis_dim),
        "raw_projection_seed": int(args.raw_projection_seed),
        "numeric_backend": args.numeric_backend,
        "numeric_backend_effective": result.get("numeric_backend", {}).get(
            "effective_backend"),
        "numeric_backend_device": result.get("numeric_backend", {}).get("device"),
        "torch_dtype": args.torch_dtype,
        "torch_min_rows": int(args.torch_min_rows),
        "encoder_kind": args.encoder_kind,
        "encoder_latent_dim": int(args.encoder_latent_dim),
        "encoder_fit_pool_size": int(args.encoder_fit_pool_size),
        "acquisition_mode": acquisition_mode,
        "beta_g": float(args.beta_g),
        "certification_mode": args.certification_mode,
        "exact_kg_mc_samples": int(args.exact_kg_mc_samples),
        "exact_kg_jobs": int(getattr(args, "exact_kg_jobs", 1)),
        "exact_kg_use_score": bool(args.exact_kg_use_score),
        "exact_kg_blend": float(args.exact_kg_blend),
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_resume": bool(getattr(args, "checkpoint_resume", False)),
        "disable_problem_initial_samples": bool(getattr(
            args, "disable_problem_initial_samples", False)),
        "disable_boundary_initial_samples": bool(getattr(
            args, "disable_boundary_initial_samples", False)),
        "disable_recommendation_refinement": bool(getattr(
            args, "disable_recommendation_refinement", False)),
        "seed": int(seed),
        "problem": args.problem,
        "N": int(args.N),
        "n0": int(args.n0),
        "K1": int(args.K1),
        "K2": int(args.K2),
        "x_recommended": result["x_recommended"],
        "true_feasible": true_feasible,
        "posterior_feasible": posterior_feasible,
        "false_feasible": bool(posterior_feasible and not true_feasible),
        "true_objective": float(result["true_objective"]),
        "true_f1": optional_float(result.get("true_f1")),
        "true_f2": optional_float(result.get("true_f2")),
        "true_best_f1": optional_float(result.get("true_best_f1")),
        "true_best_f2": optional_float(result.get("true_best_f2")),
        "best_feasible_objective": feasible_objective,
        "true_best_objective": float(result["true_best_objective"]),
        "simple_regret": float(result["simple_regret"]),
        "feasible_simple_regret": feasible_regret,
        "true_constraint_mean": float(result["true_constraint_mean"]),
        "true_constraint_sigma": float(result["true_constraint_sigma"]),
        "true_chance_margin": float(result["true_chance_margin"]),
        "constraint_violation": float(violation),
        "posterior_chance_margin": float(result["posterior_chance_margin"]),
        "posterior_theory_chance_margin": optional_float(
            result.get("posterior_theory_chance_margin")),
        "posterior_calibrated_chance_margin": optional_float(
            result.get("posterior_calibrated_chance_margin")),
        "posterior_robust_chance_margin": optional_float(
            result.get("posterior_robust_chance_margin")),
        "posterior_certification_source": result.get(
            "posterior_certification_source"),
        "n_posterior_feasible": result.get("n_posterior_feasible"),
        "n_theory_posterior_feasible": result.get("n_theory_posterior_feasible"),
        "certification_calibration_used": bool(
            result.get("certification_calibration_used", False)),
        "certification_calibration_n_feasible": result.get(
            "certification_calibration_n_feasible"),
        "certification_calibration_sigma": optional_float(
            result.get("certification_calibration_sigma")),
        "calibrated_recommendation_used": bool(
            result.get("calibrated_recommendation_used", False)),
        "calibrated_recommendation_reason": result.get(
            "calibrated_recommendation_reason"),
        "calibrated_constraint_margin": optional_float(
            result.get("calibrated_constraint_margin")),
        "calibrated_constraint_feasible": result.get(
            "calibrated_constraint_feasible"),
        "n_calibration_feasible": result.get("n_calibration_feasible"),
        "posterior_beta_g": optional_float(result.get("posterior_beta_g")),
        "posterior_mu_con": optional_float(result.get("posterior_mu_con")),
        "posterior_gpr_mu_con": optional_float(result.get("posterior_gpr_mu_con")),
        "posterior_variance_con": optional_float(result.get("posterior_variance_con")),
        "posterior_hvd_variance_con": optional_float(result.get("posterior_hvd_variance_con")),
        "posterior_epistemic_variance_con": optional_float(
            result.get("posterior_epistemic_variance_con")),
        "posterior_gpr_epistemic_variance_con": optional_float(
            result.get("posterior_gpr_epistemic_variance_con")),
        "wall_time_sec": float(result["total_time_sec"]),
        "n_simulations": int(result["n_simulations"]),
        "n_distinct_solutions": int(result["n_distinct_solutions"]),
        "variance_diagnostics": result.get("variance", {}),
    }
    row.update(compact_stage_shares(result["stage_times"]))
    return row


def summarize_variant(rows):
    n = len(rows)
    feasible_rows = [row for row in rows if row["true_feasible"]]
    summary = {
        "variant": rows[0]["variant"],
        "variance_mode": rows[0]["variance_mode"],
        "use_state_coupling": rows[0]["use_state_coupling"],
        "use_state_basis": rows[0].get("use_state_basis", False),
        "state_basis_mode": rows[0].get("state_basis_mode", "raw+state"),
        "raw_basis_dim": rows[0].get("raw_basis_dim", -1),
        "raw_projection_seed": rows[0].get("raw_projection_seed", 314159),
        "numeric_backend": rows[0].get("numeric_backend", "numpy"),
        "numeric_backend_effective": rows[0].get("numeric_backend_effective"),
        "numeric_backend_device": rows[0].get("numeric_backend_device"),
        "torch_dtype": rows[0].get("torch_dtype", "float64"),
        "torch_min_rows": rows[0].get("torch_min_rows", 128),
        "acquisition_mode": rows[0].get("acquisition_mode", "additive"),
        "beta_g": rows[0].get("beta_g", 0.0),
        "certification_mode": rows[0].get("certification_mode", "legacy"),
        "certification_calibration_used_rate": mean_bool(
            row.get("certification_calibration_used", False) for row in rows),
        "certification_calibration_n_feasible": stats(
            row.get("certification_calibration_n_feasible") for row in rows),
        "n_runs": int(n),
        "true_feasible_rate": mean_bool(row["true_feasible"] for row in rows),
        "posterior_feasible_rate": mean_bool(row["posterior_feasible"] for row in rows),
        "false_feasible_rate": mean_bool(row["false_feasible"] for row in rows),
        "n_feasible": int(len(feasible_rows)),
        "simple_regret": stats(row["simple_regret"] for row in rows),
        "feasible_simple_regret": stats(row["feasible_simple_regret"] for row in rows),
        "true_objective": stats(row["true_objective"] for row in rows),
        "best_feasible_objective": stats(
            row["best_feasible_objective"] for row in rows),
        "constraint_violation": stats(row["constraint_violation"] for row in rows),
        "true_chance_margin": stats(row["true_chance_margin"] for row in rows),
        "wall_time_sec": stats(row["wall_time_sec"] for row in rows),
        "kg_compute_share": stats(row.get("t_kg_compute_share") for row in rows),
        "candidate_gen_share": stats(row.get("t_candidate_gen_share") for row in rows),
        "posterior_solve_share": stats(
            row.get("t_posterior_solve_share") for row in rows),
    }
    return summary


def nested_get(doc, path, default=None):
    cur = doc
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def compare_to_baseline(summaries, baseline_variant):
    baseline = summaries.get(baseline_variant)
    if baseline is None:
        return summaries
    base_feasible_rate = baseline["true_feasible_rate"]
    base_false_rate = baseline["false_feasible_rate"]
    base_regret = nested_get(baseline, ["feasible_simple_regret", "median"])
    base_violation = nested_get(baseline, ["constraint_violation", "mean"])
    for variant, summary in summaries.items():
        regret = nested_get(summary, ["feasible_simple_regret", "median"])
        violation = nested_get(summary, ["constraint_violation", "mean"])
        summary["vs_baseline"] = {
            "baseline_variant": baseline_variant,
            "feasible_rate_delta": (
                None if base_feasible_rate is None
                else float(summary["true_feasible_rate"] - base_feasible_rate)
            ),
            "false_feasible_rate_delta": (
                None if base_false_rate is None
                else float(summary["false_feasible_rate"] - base_false_rate)
            ),
            "feasible_regret_median_delta": (
                None if regret is None or base_regret is None
                else float(regret - base_regret)
            ),
            "violation_mean_delta": (
                None if violation is None or base_violation is None
                else float(violation - base_violation)
            ),
        }
    return summaries


def build_variants(args):
    variants = [
        (mode, False, acq)
        for mode in parse_csv(args.modes)
        for acq in parse_csv(args.acquisition_modes)
    ]
    variants.extend(
        (mode, True, acq)
        for mode in parse_csv(args.sc_modes)
        for acq in parse_csv(args.acquisition_modes)
    )
    seen = set()
    unique = []
    for mode, use_sc, acq in variants:
        key = (mode, use_sc, acq)
        if key not in seen:
            unique.append(key)
            seen.add(key)
    return unique


def run_benchmark(args):
    seeds = parse_csv(args.seeds, int)
    if not seeds:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    variants = build_variants(args)
    rows = []
    total = len(variants) * len(seeds)
    completed = 0
    for variance_mode, use_state_coupling, acquisition_mode in variants:
        variant = _variant_name(args, variance_mode, use_state_coupling, acquisition_mode)
        for seed in seeds:
            print(
                f"[{completed}/{total}] [benchmark] start "
                f"variant={variant} seed={seed}",
                flush=True,
            )
            rows.append(run_variant_once(
                args,
                variance_mode,
                seed,
                use_state_coupling,
                acquisition_mode,
            ))
            completed += 1
            print(
                f"[{completed}/{total}] [benchmark] done "
                f"variant={variant} seed={seed}",
                flush=True,
            )
    grouped = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    summaries = {
        variant: summarize_variant(variant_rows)
        for variant, variant_rows in grouped.items()
    }
    compare_to_baseline(summaries, args.baseline_variant)
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "problem": args.problem,
            "d": args.d,
            "L": args.L,
            "sigma": args.sigma,
            "alpha": args.alpha,
            "weights": parse_weights(args.weights),
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
            "enable_certification_calibration": bool(
                getattr(args, "enable_certification_calibration", False)),
            "certification_calibration_min_obs": int(
                getattr(args, "certification_calibration_min_obs", 8)),
            "certification_calibration_ridge": float(
                getattr(args, "certification_calibration_ridge", 1e-6)),
            "certification_calibration_noise_floor_scale": float(getattr(
                args,
                "certification_calibration_noise_floor_scale",
                0.5,
            )),
            "certification_calibration_beta": float(
                getattr(args, "certification_calibration_beta", 2.0)),
            "disable_recommendation_axis_oracle": args.disable_recommendation_axis_oracle,
            "disable_problem_initial_samples": bool(getattr(
                args, "disable_problem_initial_samples", False)),
            "disable_boundary_initial_samples": bool(getattr(
                args, "disable_boundary_initial_samples", False)),
            "disable_recommendation_refinement": bool(getattr(
                args, "disable_recommendation_refinement", False)),
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
            "acquisition_modes": parse_csv(args.acquisition_modes),
            "exact_kg_mc_samples": args.exact_kg_mc_samples,
            "exact_kg_jobs": int(getattr(args, "exact_kg_jobs", 1)),
            "exact_kg_use_score": args.exact_kg_use_score,
            "exact_kg_blend": args.exact_kg_blend,
            "checkpoint_dir": str(getattr(args, "checkpoint_dir", "") or ""),
            "checkpoint_resume": bool(getattr(args, "checkpoint_resume", False)),
            "checkpoint_interval": int(getattr(args, "checkpoint_interval", 1)),
            "checkpoint_keep_last": int(getattr(args, "checkpoint_keep_last", 3)),
            "progress_logging": bool(getattr(args, "progress_logging", False)),
            "progress_units_per_iteration": int(
                getattr(args, "progress_units_per_iteration", 100)),
            "progress_exact_updates": int(
                getattr(args, "progress_exact_updates", 10)),
            "seeds": seeds,
            "modes": parse_csv(args.modes),
            "sc_modes": parse_csv(args.sc_modes),
            "baseline_variant": args.baseline_variant,
        },
        "rows": rows,
        "summary": summaries,
    }


def flatten_summary(summary):
    row = {
        "variant": summary["variant"],
        "variance_mode": summary["variance_mode"],
        "use_state_coupling": summary["use_state_coupling"],
        "use_state_basis": summary.get("use_state_basis", False),
        "state_basis_mode": summary.get("state_basis_mode", "raw+state"),
        "raw_basis_dim": summary.get("raw_basis_dim", -1),
        "numeric_backend": summary.get("numeric_backend", "numpy"),
        "numeric_backend_effective": summary.get("numeric_backend_effective"),
        "numeric_backend_device": summary.get("numeric_backend_device"),
        "torch_dtype": summary.get("torch_dtype", "float64"),
        "torch_min_rows": summary.get("torch_min_rows", 128),
        "acquisition_mode": summary.get("acquisition_mode", "additive"),
        "beta_g": summary.get("beta_g", 0.0),
        "certification_mode": summary.get("certification_mode", "legacy"),
        "certification_calibration_used_rate": summary.get(
            "certification_calibration_used_rate"),
        "n_runs": summary["n_runs"],
        "true_feasible_rate": summary["true_feasible_rate"],
        "posterior_feasible_rate": summary["posterior_feasible_rate"],
        "false_feasible_rate": summary["false_feasible_rate"],
        "n_feasible": summary["n_feasible"],
    }
    for metric in (
        "simple_regret",
        "feasible_simple_regret",
        "true_objective",
        "best_feasible_objective",
        "constraint_violation",
        "true_chance_margin",
        "wall_time_sec",
        "kg_compute_share",
        "candidate_gen_share",
        "posterior_solve_share",
    ):
        for stat_key, value in summary[metric].items():
            row[f"{metric}_{stat_key}"] = value
    for key, value in summary.get("vs_baseline", {}).items():
        row[f"vs_baseline_{key}"] = value
    return row


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {}
            for key, value in row.items():
                if isinstance(value, (dict, list, tuple)):
                    flat[key] = json.dumps(json_safe(value), sort_keys=True)
                else:
                    flat[key] = value
            writer.writerow(flat)


def write_outputs(args, result):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"quality_benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
    json_path = out_dir / f"{prefix}.json"
    rows_path = out_dir / f"{prefix}_rows.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"
    json_path.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    write_csv(rows_path, result["rows"])
    summary_rows = [flatten_summary(item) for item in result["summary"].values()]
    write_csv(summary_path, summary_rows)
    return {
        "json": str(json_path),
        "rows_csv": str(rows_path),
        "summary_csv": str(summary_path),
    }


def printable_summary(result):
    rows = []
    for variant, summary in result["summary"].items():
        rows.append({
            "variant": variant,
            "true_feasible_rate": summary["true_feasible_rate"],
            "false_feasible_rate": summary["false_feasible_rate"],
            "feasible_regret_median": nested_get(
                summary, ["feasible_simple_regret", "median"]),
            "violation_mean": nested_get(
                summary, ["constraint_violation", "mean"]),
            "wall_time_median_sec": nested_get(
                summary, ["wall_time_sec", "median"]),
            "vs_baseline": summary.get("vs_baseline", {}),
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
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--n0", type=int, default=8)
    parser.add_argument("--K1", type=int, default=25)
    parser.add_argument("--K2", type=int, default=0)
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
    parser.add_argument(
        "--recommendation_infeasible_strategy",
        default="penalty",
        choices=["penalty", "min_margin", "lexicographic"],
    )
    parser.add_argument("--disable_recommendation_calibration", action="store_true")
    parser.add_argument("--recommendation_calibration_ridge", type=float, default=1e-6)
    parser.add_argument("--enable_certification_calibration", action="store_true")
    parser.add_argument("--certification_calibration_min_obs", type=int, default=8)
    parser.add_argument("--certification_calibration_ridge", type=float, default=1e-6)
    parser.add_argument(
        "--certification_calibration_noise_floor_scale",
        type=float,
        default=0.5,
    )
    parser.add_argument("--certification_calibration_beta", type=float, default=2.0)
    parser.add_argument("--disable_recommendation_axis_oracle", action="store_true")
    parser.add_argument(
        "--disable_problem_initial_samples",
        action="store_true",
        help="Use random/boundary initialization instead of problem-defined anchors.",
    )
    parser.add_argument(
        "--disable_boundary_initial_samples",
        action="store_true",
        help="Use random initialization instead of generic boundary points when anchors are disabled.",
    )
    parser.add_argument(
        "--disable_recommendation_refinement",
        action="store_true",
        help="Do not add problem-defined dense refinement grids to the final recommendation pool.",
    )
    parser.add_argument("--use_state_basis", dest="use_state_basis", action="store_true", default=True)
    parser.add_argument("--disable_state_basis", dest="use_state_basis", action="store_false")
    parser.add_argument(
        "--state_basis_mode",
        default="raw+state",
        choices=["raw", "state", "raw+state", "manifold", "raw+manifold"],
    )
    parser.add_argument(
        "--raw_basis_dim",
        type=int,
        default=-1,
        help=(
            "If positive and smaller than 2*d, project the raw [x,x^2] basis "
            "to this many fixed random features before concatenating state/"
            "manifold features."
        ),
    )
    parser.add_argument("--raw_projection_seed", type=int, default=314159)
    parser.add_argument(
        "--numeric_backend",
        default="numpy",
        choices=["numpy", "auto", "torch", "torch_cuda", "cuda"],
        help="Optional matrix backend for GPR/KG algebra. Defaults to numpy.",
    )
    parser.add_argument(
        "--numeric_backend_device",
        default="auto",
        help="Torch device such as auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--torch_dtype",
        default="float64",
        choices=["float64", "float32", "double", "single"],
    )
    parser.add_argument(
        "--torch_min_rows",
        type=int,
        default=128,
        help="Minimum candidate/sample rows before using torch backend.",
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
        ],
    )
    parser.add_argument("--encoder_latent_dim", type=int, default=8)
    parser.add_argument("--encoder_fit_pool_size", type=int, default=512)
    parser.add_argument("--exact_kg_mc_samples", type=int, default=8)
    parser.add_argument(
        "--exact_kg_jobs",
        type=int,
        default=1,
        help="Candidate-level thread parallelism inside exact posterior-update KG.",
    )
    parser.add_argument("--exact_kg_use_score", action="store_true")
    parser.add_argument("--exact_kg_blend", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint_dir",
        default="",
        help="Optional root directory for per-variant/per-seed KG checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_resume",
        action="store_true",
        help="Resume each KG run from checkpoint_latest.pkl when present.",
    )
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument("--checkpoint_keep_last", type=int, default=3)
    parser.add_argument("--progress_logging", dest="progress_logging", action="store_true", default=True)
    parser.add_argument("--disable_progress_logging", dest="progress_logging", action="store_false")
    parser.add_argument("--progress_units_per_iteration", type=int, default=100)
    parser.add_argument("--progress_exact_updates", type=int, default=10)
    parser.add_argument("--acquisition_modes", default="exact_mc")
    parser.add_argument("--modes", default="factor")
    parser.add_argument("--sc_modes", default="factor")
    parser.add_argument("--baseline_variant", default="factor:olhkg_sc_exact")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--out_dir", default=str(ROOT / "profiles"))
    parser.add_argument("--out_prefix", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.N <= args.n0:
        raise ValueError("--N must be larger than --n0")

    result = run_benchmark(args)
    paths = write_outputs(args, result)
    print(json.dumps(json_safe({
        "paths": paths,
        "summary": printable_summary(result),
    }), indent=2))


if __name__ == "__main__":
    main()
