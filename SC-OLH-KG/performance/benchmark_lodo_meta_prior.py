"""Parallel LODO benchmark for learned admissible SC/HVD meta-priors."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig  # noqa: E402
from core.admissibility import domain_tuned_audit  # noqa: E402
from performance.benchmark_quality import json_safe, parse_csv, parse_weights, write_csv  # noqa: E402
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from representation.meta_prior import (  # noqa: E402
    AdmissibleProblemAdapter,
    LearnedMetaPrior,
    MetaPriorProblemAdapter,
)


def finite_stats(values):
    vals = []
    for value in values:
        if value is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def mean_bool(values):
    vals = [bool(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else None


def build_scalarized_problem(name, d, L, sigma, alpha, weights):
    return ScalarizedProblem(
        make_problem(name, d=d, L=L, sigma=sigma, alpha=alpha),
        weights=weights,
    )


def train_meta_prior(args_dict, heldout, seed):
    domains = parse_csv(args_dict["domains"])
    source_names = [name for name in domains if name != heldout]
    if not source_names:
        raise ValueError(f"heldout={heldout} leaves no source domains")
    source_problems = [
        (
            name,
            build_scalarized_problem(
                name,
                args_dict["d"],
                args_dict["L"],
                args_dict["sigma"],
                args_dict["alpha"],
                parse_weights(args_dict["weights"]),
            ),
        )
        for name in source_names
    ]
    prior = LearnedMetaPrior(
        local_dim=args_dict["meta_local_dim"],
        shared_dim=args_dict["meta_shared_dim"],
        anchor_count=args_dict["meta_anchor_count"],
        kmeans_iters=args_dict["meta_kmeans_iters"],
        soft_temperature=args_dict["meta_soft_temperature"],
        ridge=args_dict["meta_ridge"],
        boundary_weight=args_dict["meta_boundary_weight"],
        boundary_temperature=args_dict["meta_boundary_temperature"],
        variance_weight=args_dict["meta_variance_weight"],
        feasible_penalty=args_dict["meta_feasible_penalty"],
        feasible_bonus=args_dict["meta_feasible_bonus"],
        elite_fraction=args_dict["meta_elite_fraction"],
        boundary_fraction=args_dict["meta_boundary_fraction"],
        seed=int(args_dict["meta_seed"]) + int(seed),
    )
    prior.fit_from_source_problems(
        source_problems,
        n_records_per_domain=args_dict["source_records_per_domain"],
        rng=np.random.default_rng(int(args_dict["meta_seed"]) + 1009 * int(seed)),
    )
    return prior


def build_target_problem(args_dict, heldout, line, seed):
    weights = parse_weights(args_dict["weights"])
    target = build_scalarized_problem(
        heldout,
        args_dict["d"],
        args_dict["L"],
        args_dict["sigma"],
        args_dict["alpha"],
        weights,
    )
    line = str(line)
    meta_diag = None
    if line == "strict":
        return AdmissibleProblemAdapter(target, variant="strict_universal"), meta_diag
    if line == "lodo":
        prior = train_meta_prior(args_dict, heldout, seed)
        meta_diag = prior.diagnostics()
        return MetaPriorProblemAdapter(
            target,
            prior,
            proposal_pool_size=args_dict["meta_proposal_pool_size"],
            refinement_count=args_dict["meta_refinement_count"],
        ), meta_diag
    if line == "domain":
        return target, meta_diag
    raise ValueError(f"unknown line {line!r}")


def run_one(task):
    args_dict = task["args"]
    heldout = task["heldout"]
    line = task["line"]
    seed = int(task["seed"])
    problem, meta_diag = build_target_problem(args_dict, heldout, line, seed)
    if line == "strict":
        use_problem_initial = False
        use_boundary_initial = False
        use_recommendation_refinement = False
        state_candidate_count = 0
    else:
        use_problem_initial = True
        use_boundary_initial = False
        use_recommendation_refinement = True
        state_candidate_count = int(args_dict["state_candidate_count"])

    config = SingleOLHKGConfig(
        N=args_dict["N"],
        n0=args_dict["n0"],
        K1=args_dict["K1"],
        K2=args_dict["K2"],
        posterior_pool_size=args_dict["posterior_pool_size"],
        posterior_keep=args_dict["posterior_keep"],
        axis_candidate_count=args_dict["axis_candidate_count"],
        structured_candidate_count=args_dict["structured_candidate_count"],
        state_candidate_count=state_candidate_count,
        state_inverse_pool_size=args_dict["state_inverse_pool_size"],
        state_inverse_neighbors=args_dict["state_inverse_neighbors"],
        n_thr=args_dict["n_thr"],
        variance_mode=args_dict["variance_mode"],
        lambda_feas=args_dict["lambda_feas"],
        lambda_var=args_dict["lambda_var"],
        lambda_mean=args_dict["lambda_mean"],
        lambda_coupling=args_dict["lambda_coupling"] if line != "strict" else 0.0,
        beta_g=args_dict["beta_g"],
        certification_mode=args_dict["certification_mode"],
        recommendation_calibration=False,
        certification_calibration=False,
        recommendation_axis_oracle=False,
        use_problem_initial_samples=use_problem_initial,
        use_boundary_initial_samples=use_boundary_initial,
        use_recommendation_refinement=use_recommendation_refinement,
        use_state_coupling=True,
        use_state_basis=bool(args_dict["use_state_basis"]),
        state_basis_mode=args_dict["state_basis_mode"],
        raw_basis_dim=args_dict["raw_basis_dim"],
        raw_projection_seed=args_dict["raw_projection_seed"],
        numeric_backend=args_dict["numeric_backend"],
        numeric_backend_device=args_dict["numeric_backend_device"],
        torch_dtype=args_dict["torch_dtype"],
        torch_min_rows=args_dict["torch_min_rows"],
        encoder_kind="synthetic",
        acquisition_mode=args_dict["acquisition_mode"],
        exact_kg_mc_samples=args_dict["exact_kg_mc_samples"],
        exact_kg_jobs=args_dict["exact_kg_jobs"],
        exact_kg_use_score=args_dict["exact_kg_use_score"],
        exact_kg_blend=args_dict["exact_kg_blend"],
        llm_prior_enabled=bool(args_dict["llm_prior_enabled"]),
        llm_prior_base_url=args_dict["llm_prior_base_url"],
        llm_prior_model=args_dict["llm_prior_model"],
        llm_prior_api_key_env=args_dict["llm_prior_api_key_env"],
        llm_prior_candidate_count=args_dict["llm_prior_candidate_count"],
        llm_prior_inverse_pool_size=args_dict["llm_prior_inverse_pool_size"],
        llm_prior_interval=args_dict["llm_prior_interval"],
        llm_prior_min_obs=args_dict["llm_prior_min_obs"],
        llm_prior_timeout_sec=args_dict["llm_prior_timeout_sec"],
        llm_prior_gate_floor=args_dict["llm_prior_gate_floor"],
        llm_prior_max_observations=args_dict["llm_prior_max_observations"],
        progress_logging=False,
        eval_pool_size=args_dict["eval_pool_size"],
        seed=seed,
    )
    started = time.time()
    alg = SingleOLHKGAlgorithm(problem, config)
    result = alg.run(verbose=False)
    true_feasible = bool(result["true_feasible"])
    posterior_feasible = bool(result.get("posterior_feasible", False))
    audit = (
        problem.admissibility_audit()
        if hasattr(problem, "admissibility_audit")
        else domain_tuned_audit().to_dict()
    )
    return {
        "line": line,
        "heldout": heldout,
        "seed": seed,
        "variant": f"{line}:{heldout}",
        "audit_admissible_mainline": bool(audit.get("admissible_mainline", False)),
        "audit": audit,
        "meta_prior": meta_diag,
        "true_feasible": true_feasible,
        "posterior_feasible": posterior_feasible,
        "false_feasible": bool(posterior_feasible and not true_feasible),
        "true_objective": float(result["true_objective"]),
        "true_best_objective": float(result["true_best_objective"]),
        "simple_regret": float(result["simple_regret"]),
        "feasible_simple_regret": (
            float(result["simple_regret"]) if true_feasible else None
        ),
        "true_chance_margin": float(result["true_chance_margin"]),
        "constraint_violation": float(max(result["true_chance_margin"], 0.0)),
        "posterior_chance_margin": float(result["posterior_chance_margin"]),
        "n_simulations": int(result["n_simulations"]),
        "n_distinct_solutions": int(result["n_distinct_solutions"]),
        "n_pool": int(result.get("n_pool", 0)),
        "n_posterior_feasible": int(result.get("n_posterior_feasible", 0)),
        "wall_time_sec": float(time.time() - started),
        "algorithm_time_sec": float(result["total_time_sec"]),
        "variance_diagnostics": result.get("variance", {}),
        "candidate_source_counts": result.get("candidate_source_counts", {}),
        "llm_prior": result.get("llm_prior", {}),
        "x_recommended": result["x_recommended"],
    }


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    out = {}
    for variant, items in grouped.items():
        out[variant] = {
            "variant": variant,
            "line": items[0]["line"],
            "heldout": items[0]["heldout"],
            "n_runs": len(items),
            "audit_admissible_mainline_rate": mean_bool(
                row["audit_admissible_mainline"] for row in items),
            "true_feasible_rate": mean_bool(row["true_feasible"] for row in items),
            "posterior_feasible_rate": mean_bool(row["posterior_feasible"] for row in items),
            "false_feasible_rate": mean_bool(row["false_feasible"] for row in items),
            "simple_regret": finite_stats(row["simple_regret"] for row in items),
            "feasible_simple_regret": finite_stats(
                row["feasible_simple_regret"] for row in items),
            "constraint_violation": finite_stats(
                row["constraint_violation"] for row in items),
            "true_chance_margin": finite_stats(
                row["true_chance_margin"] for row in items),
            "wall_time_sec": finite_stats(row["wall_time_sec"] for row in items),
            "llm_prior_ok_count": finite_stats(
                (row.get("llm_prior") or {}).get("ok_count", 0) for row in items),
            "llm_prior_selected_count": finite_stats(
                (row.get("llm_prior") or {}).get("selected_count", 0) for row in items),
            "llm_prior_gate_mean": finite_stats(
                (row.get("llm_prior") or {}).get("gate_mean", 0.0) for row in items),
        }
    return out


def flatten_summary(summary):
    row = {
        "variant": summary["variant"],
        "line": summary["line"],
        "heldout": summary["heldout"],
        "n_runs": summary["n_runs"],
        "audit_admissible_mainline_rate": summary["audit_admissible_mainline_rate"],
        "true_feasible_rate": summary["true_feasible_rate"],
        "posterior_feasible_rate": summary["posterior_feasible_rate"],
        "false_feasible_rate": summary["false_feasible_rate"],
    }
    for metric in (
        "simple_regret",
        "feasible_simple_regret",
        "constraint_violation",
        "true_chance_margin",
        "wall_time_sec",
        "llm_prior_ok_count",
        "llm_prior_selected_count",
        "llm_prior_gate_mean",
    ):
        for key, value in summary[metric].items():
            row[f"{metric}_{key}"] = value
    return row


def run(args):
    heldouts = parse_csv(args.heldouts) or parse_csv(args.domains)
    lines = parse_csv(args.lines)
    seeds = parse_csv(args.seeds, int)
    if not seeds:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    args_dict = vars(args).copy()
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    existing_rows = []
    completed = set()
    if checkpoint_path and args.resume_completed and checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["heldout"]), str(row["line"]), int(row["seed"]))
            if key not in completed:
                existing_rows.append(row)
                completed.add(key)
        print(
            f"[lodo] resumed_rows={len(existing_rows)} from {checkpoint_path}",
            flush=True,
        )
    tasks = [
        {"args": args_dict, "heldout": heldout, "line": line, "seed": seed}
        for heldout in heldouts
        for line in lines
        for seed in seeds
        if (str(heldout), str(line), int(seed)) not in completed
    ]
    rows = list(existing_rows)
    total = len(tasks)
    print(f"[lodo] tasks={total} resumed={len(existing_rows)} jobs={args.jobs}", flush=True)
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def save_row(row):
        if not checkpoint_path:
            return
        with checkpoint_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(json_safe(row), sort_keys=True) + "\n")
            fh.flush()

    if args.jobs <= 1:
        for idx, task in enumerate(tasks):
            print(
                f"[{idx}/{total}] start line={task['line']} "
                f"heldout={task['heldout']} seed={task['seed']}",
                flush=True,
            )
            row = run_one(task)
            rows.append(row)
            save_row(row)
            print(f"[{idx + 1}/{total}] done", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as executor:
            futures = {executor.submit(run_one, task): task for task in tasks}
            done = 0
            for future in as_completed(futures):
                task = futures[future]
                row = future.result()
                rows.append(row)
                save_row(row)
                done += 1
                print(
                    f"[{done}/{total}] done line={task['line']} "
                    f"heldout={task['heldout']} seed={task['seed']}",
                    flush=True,
                )
    summaries = summarize(rows)
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": json_safe(args_dict),
        "rows": rows,
        "summary": summaries,
    }


def write_outputs(args, result):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"lodo_meta_prior_{time.strftime('%Y%m%d_%H%M%S')}"
    json_path = out_dir / f"{prefix}.json"
    rows_path = out_dir / f"{prefix}_rows.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"
    json_path.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    write_csv(rows_path, result["rows"])
    write_csv(summary_path, [flatten_summary(v) for v in result["summary"].values()])
    return {"json": str(json_path), "rows_csv": str(rows_path), "summary_csv": str(summary_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains",
        default="FactorShockStatePolicyRZDT1,InventorySupplyChain,QueueResourceControl",
    )
    parser.add_argument("--heldouts", default="")
    parser.add_argument("--lines", default="strict,lodo,domain")
    parser.add_argument("--d", type=int, default=50)
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
    parser.add_argument("--state_candidate_count", type=int, default=12)
    parser.add_argument("--state_inverse_pool_size", type=int, default=300)
    parser.add_argument("--state_inverse_neighbors", type=int, default=1)
    parser.add_argument("--n_thr", type=int, default=5)
    parser.add_argument("--eval_pool_size", type=int, default=300)
    parser.add_argument("--variance_mode", default="factor")
    parser.add_argument("--lambda_feas", type=float, default=0.25)
    parser.add_argument("--lambda_var", type=float, default=0.25)
    parser.add_argument("--lambda_mean", type=float, default=0.10)
    parser.add_argument("--lambda_coupling", type=float, default=0.05)
    parser.add_argument("--beta_g", type=float, default=2.0)
    parser.add_argument("--certification_mode", default="theory")
    parser.add_argument("--use_state_basis", dest="use_state_basis", action="store_true", default=True)
    parser.add_argument("--disable_state_basis", dest="use_state_basis", action="store_false")
    parser.add_argument("--state_basis_mode", default="raw+state")
    parser.add_argument("--raw_basis_dim", type=int, default=32)
    parser.add_argument("--raw_projection_seed", type=int, default=314159)
    parser.add_argument("--numeric_backend", default="numpy")
    parser.add_argument("--numeric_backend_device", default="auto")
    parser.add_argument("--torch_dtype", default="float64")
    parser.add_argument("--torch_min_rows", type=int, default=128)
    parser.add_argument("--acquisition_mode", default="additive")
    parser.add_argument("--exact_kg_mc_samples", type=int, default=0)
    parser.add_argument("--exact_kg_jobs", type=int, default=1)
    parser.add_argument("--exact_kg_use_score", action="store_true")
    parser.add_argument("--exact_kg_blend", type=float, default=0.0)
    parser.add_argument("--llm_prior_enabled", action="store_true")
    parser.add_argument("--llm_prior_base_url", default="https://ruoli.dev")
    parser.add_argument("--llm_prior_model", default="gpt-5.4-mini")
    parser.add_argument("--llm_prior_api_key_env", default="SCOLHKG_LLM_API_KEY")
    parser.add_argument("--llm_prior_candidate_count", type=int, default=8)
    parser.add_argument("--llm_prior_inverse_pool_size", type=int, default=256)
    parser.add_argument("--llm_prior_interval", type=int, default=5)
    parser.add_argument("--llm_prior_min_obs", type=int, default=8)
    parser.add_argument("--llm_prior_timeout_sec", type=float, default=30.0)
    parser.add_argument("--llm_prior_gate_floor", type=float, default=0.05)
    parser.add_argument("--llm_prior_max_observations", type=int, default=24)
    parser.add_argument("--source_records_per_domain", type=int, default=96)
    parser.add_argument("--meta_local_dim", type=int, default=3)
    parser.add_argument("--meta_shared_dim", type=int, default=3)
    parser.add_argument("--meta_anchor_count", type=int, default=24)
    parser.add_argument("--meta_kmeans_iters", type=int, default=25)
    parser.add_argument("--meta_soft_temperature", type=float, default=0.75)
    parser.add_argument("--meta_ridge", type=float, default=1e-4)
    parser.add_argument("--meta_boundary_weight", type=float, default=1.0)
    parser.add_argument("--meta_boundary_temperature", type=float, default=1.0)
    parser.add_argument("--meta_variance_weight", type=float, default=0.5)
    parser.add_argument("--meta_feasible_penalty", type=float, default=6.0)
    parser.add_argument("--meta_feasible_bonus", type=float, default=0.15)
    parser.add_argument("--meta_elite_fraction", type=float, default=0.40)
    parser.add_argument("--meta_boundary_fraction", type=float, default=0.35)
    parser.add_argument("--meta_seed", type=int, default=20260706)
    parser.add_argument("--meta_proposal_pool_size", type=int, default=512)
    parser.add_argument("--meta_refinement_count", type=int, default=96)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--resume_completed", action="store_true")
    parser.add_argument("--out_dir", default=str(ROOT / "profiles"))
    parser.add_argument("--out_prefix", default="")
    args = parser.parse_args()
    if args.N <= args.n0:
        raise ValueError("--N must be larger than --n0")
    result = run(args)
    paths = write_outputs(args, result)
    print(json.dumps(json_safe({"paths": paths, "summary": result["summary"]}), indent=2))


if __name__ == "__main__":
    main()
