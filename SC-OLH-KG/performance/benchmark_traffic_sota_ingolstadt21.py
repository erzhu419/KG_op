"""Traffic SOTA baselines on live ingolstadt21 SUMO samples.

This mirrors `benchmark_traffic_ingolstadt21.py`'s summary format so the
existing fresh-seed OOS validator can certify final recommendations.  Real
BoTorch baselines are attempted when requested; failures are recorded instead
of silently replacing the method unless `--botorch-fallback lite` is set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
GPR_KG_CODE = REPO_ROOT / "Final_Submission" / "GPR_KG_Code"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GPR_KG_CODE))

from baselines.baseline_algorithms import BaselineConfig, SequentialBaseline  # noqa: E402
from baselines.botorch_adapters import (  # noqa: E402
    BoTorchBaseline,
    BoTorchBaselineConfig,
    fallback_method,
    is_botorch_available,
)
from benchmark_quality import json_safe, parse_csv  # noqa: E402
from problems.traffic_ingolstadt21 import Ingolstadt21ScalarizedTrafficProblem  # noqa: E402


def _safe_name(value):
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _summary_dir(args, method, partition, seed):
    return (
        Path(args.paper_results_dir)
        / f"{_safe_name(method)}_{_safe_name(partition)}_seed{int(seed)}"
    )


def _run_baseline(args, method, seed, problem):
    method = str(method)
    if method in {"turbo", "scbo", "saasbo"}:
        method = f"botorch_{method}"
    if method.startswith("botorch_"):
        if not is_botorch_available():
            if args.botorch_fallback == "error":
                raise RuntimeError(
                    f"{method} requested but BoTorch is unavailable")
            lite = fallback_method(method)
            config = BaselineConfig(
                N=args.N,
                n0=args.n0,
                seed=seed,
                method=lite,
                batch_candidates=args.baseline_batch_candidates,
                tr_radius_init=args.tr_radius_init,
                tr_radius_min=args.tr_radius_min,
                tr_radius_max=args.tr_radius_max,
            )
            result = SequentialBaseline(problem, config).run()
            result["backend"] = f"fallback:{lite}"
            result["method"] = method
            return result
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
        )
        return BoTorchBaseline(problem, config).run()
    config = BaselineConfig(
        N=args.N,
        n0=args.n0,
        seed=seed,
        method=method,
        batch_candidates=args.baseline_batch_candidates,
        tr_radius_init=args.tr_radius_init,
        tr_radius_min=args.tr_radius_min,
        tr_radius_max=args.tr_radius_max,
    )
    return SequentialBaseline(problem, config).run()


def run_one(args, method, seed):
    weights = [float(v) for v in parse_csv(args.weights)]
    problem = Ingolstadt21ScalarizedTrafficProblem(
        weights=weights,
        seed=seed,
        true_replications=args.true_replications,
        sigma_replications=args.sigma_replications,
        historical_anchor_policy=args.traffic_anchor_policy,
    )
    partition = f"{method}_{_safe_name(args.traffic_anchor_policy)}"
    if args.tag:
        partition += f"_{_safe_name(args.tag)}"
    out_dir = _summary_dir(args, method, partition, seed)
    summary_path = out_dir / "summary.json"
    detail_path = out_dir / "details.json"
    if args.resume and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        result = _run_baseline(args, method, seed, problem)
        status = "ok"
        error = None
    except Exception as exc:
        result = {
            "x_recommended": list(map(int, problem.base.default_x)),
            "true_objective": float("nan"),
            "true_constraint_mean": float("nan"),
            "true_constraint_sigma": float("nan"),
            "true_chance_margin": float("inf"),
            "true_feasible": False,
            "method": method,
            "backend": "failed",
        }
        status = "failed"
        error = repr(exc)
    wall = time.time() - t0
    x_rec = [int(v) for v in result["x_recommended"]]
    final = dict(result)
    final["status"] = status
    final["error"] = error
    final_pareto = [x_rec]
    summary = {
        "method": method,
        "partition_method": partition,
        "problem": "ingolstadt21",
        "N": int(args.N),
        "n0": int(args.n0),
        "seed": int(seed),
        "d": int(problem.d),
        "tau": float(problem.tau),
        "alpha": float(problem.alpha),
        "weights": weights,
        "wall_time_sec": float(wall),
        "final_pareto_set": final_pareto,
        "final_recommendation": x_rec,
        "final_log": json_safe(final),
        "config": json_safe(vars(args)),
        "traffic_anchor_policy": args.traffic_anchor_policy,
        "traffic_note": (
            "Live SUMO SOTA baseline. Paper-grade feasibility must be certified "
            "by validate_oos_feasibility with fresh seeds."
        ),
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    detail_path.write_text(json.dumps(json_safe({
        "summary_path": str(summary_path),
        "result": final,
    }), indent=2), encoding="utf-8")
    print(json.dumps({
        "method": method,
        "partition": partition,
        "seed": int(seed),
        "status": status,
        "summary_path": str(summary_path),
        "x_recommended": x_rec,
        "wall_time_sec": float(wall),
    }, indent=2), flush=True)
    print("DONE", flush=True)
    return summary


def run(args):
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        for method in parse_csv(args.methods):
            rows.append(run_one(args, method, seed))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="sobol,random,hetgp_lite,rahbo_lite,safeopt_lite,legacy_vepm_lite,botorch_turbo,botorch_scbo,botorch_saasbo")
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--N", type=int, default=12)
    parser.add_argument("--n0", type=int, default=5)
    parser.add_argument("--baseline_batch_candidates", type=int, default=64)
    parser.add_argument("--tr_radius_init", type=float, default=0.35)
    parser.add_argument("--tr_radius_min", type=float, default=0.04)
    parser.add_argument("--tr_radius_max", type=float, default=0.8)
    parser.add_argument("--tr_success_tolerance", type=int, default=3)
    parser.add_argument("--tr_failure_tolerance", type=int, default=5)
    parser.add_argument("--botorch_fallback", choices=("lite", "error"), default="error")
    parser.add_argument("--botorch_raw_samples", type=int, default=32)
    parser.add_argument("--botorch_num_restarts", type=int, default=3)
    parser.add_argument("--botorch_maxiter", type=int, default=25)
    parser.add_argument("--botorch_timeout_sec", type=float, default=60.0)
    parser.add_argument("--botorch_max_candidate_failures", type=int, default=4)
    parser.add_argument("--saas_warmup_steps", type=int, default=8)
    parser.add_argument("--saas_num_samples", type=int, default=8)
    parser.add_argument("--saas_thinning", type=int, default=1)
    parser.add_argument("--saas_max_tree_depth", type=int, default=3)
    parser.add_argument("--saas_mc_samples", type=int, default=32)
    parser.add_argument("--saas_unconstrained", action="store_true")
    parser.add_argument("--disable_saas_failure_fallback", action="store_true")
    parser.add_argument("--true_replications", type=int, default=2)
    parser.add_argument("--sigma_replications", type=int, default=3)
    parser.add_argument(
        "--traffic_anchor_policy",
        default="strict_none",
        choices=["historical", "none", "strict_none", "only"],
    )
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--paper_results_dir", default=str(GPR_KG_CODE / "results" / "ingolstadt21"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = run(args)
    print(json.dumps(json_safe({
        "n_runs": len(rows),
        "summary_paths": [
            str(_summary_dir(args, row["method"], row["partition_method"], row["seed"]) / "summary.json")
            for row in rows
        ],
    }), indent=2))


if __name__ == "__main__":
    main()
