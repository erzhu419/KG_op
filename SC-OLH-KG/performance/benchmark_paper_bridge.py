"""Bridge benchmark on the original paper's RZDT environments.

The submitted manuscript reports bi-objective HV/IGD/CVR metrics.  The current
prototype is single-objective chance-constrained, so this script keeps the
comparison honest: it runs scalarized OLH-KG variants on the same synthetic
problem families and stores the paper's reported bi-objective numbers as
context rather than converting them into regret.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark_quality  # noqa: E402
from benchmark_quality import json_safe, parse_csv, write_csv  # noqa: E402


PAPER_REPORTED_BIOBJ = {
    "PaperRZDT1": {
        "paper_problem": "RZDT1",
        "algorithm": "GPR-KG",
        "N": 150,
        "HV": 1.332,
        "IGD": 0.201,
        "CVR": 0.325,
        "time_sec": 382.5,
    },
    "PaperRZDT2": {
        "paper_problem": "RZDT2",
        "algorithm": "GPR-KG",
        "N": 150,
        "HV": 1.289,
        "IGD": 0.115,
        "CVR": 0.100,
        "time_sec": 376.3,
    },
    "PaperRZDT5_RR": {
        "paper_problem": "RZDT5_RR",
        "algorithm": "GPR-KG",
        "N": 150,
        "HV": 2.104,
        "IGD": 0.054,
        "CVR": 0.008,
        "time_sec": 840.6,
    },
}


def _safe_name(value):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))


def _problem_args(args, problem):
    return SimpleNamespace(
        problem=problem,
        d=args.d,
        L=args.L,
        sigma=args.sigma,
        alpha=args.alpha,
        weights=args.weights,
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
        eval_pool_size=args.eval_pool_size,
        lambda_feas=args.lambda_feas,
        lambda_var=args.lambda_var,
        lambda_mean=args.lambda_mean,
        lambda_coupling=args.lambda_coupling,
        coupling_safety_z=args.coupling_safety_z,
        coupling_gate_temperature=args.coupling_gate_temperature,
        recommendation_safety_z=args.recommendation_safety_z,
        recommendation_noise_floor_scale=args.recommendation_noise_floor_scale,
        recommendation_infeasible_penalty=args.recommendation_infeasible_penalty,
        disable_recommendation_calibration=args.disable_recommendation_calibration,
        recommendation_calibration_ridge=args.recommendation_calibration_ridge,
        disable_recommendation_axis_oracle=args.disable_recommendation_axis_oracle,
        use_state_basis=args.use_state_basis,
        modes=args.modes,
        sc_modes=args.sc_modes,
        baseline_variant=args.baseline_variant,
        seeds=args.seeds,
        seed_start=args.seed_start,
        n_seeds=args.n_seeds,
        out_dir=args.out_dir,
        out_prefix=(
            args.out_prefix + "_" if args.out_prefix else "paper_bridge_"
        ) + f"{_safe_name(problem)}_n{int(args.N)}_s{int(args.n_seeds)}",
        verbose=args.verbose,
    )


def _augment_summary(summary, problem):
    row = benchmark_quality.flatten_summary(summary)
    row["problem"] = problem
    paper = PAPER_REPORTED_BIOBJ.get(problem, {})
    for key, value in paper.items():
        row[f"paper_{key}"] = value
    row["metric_bridge_note"] = (
        "current metrics are scalarized chance-constrained objective/regret; "
        "paper metrics are bi-objective HV/IGD/CVR"
    )
    return row


def run_bridge(args):
    all_rows = []
    summary_rows = []
    problem_results = {}
    for problem in parse_csv(args.problems):
        print(f"[paper-bridge] problem={problem}", flush=True)
        problem_args = _problem_args(args, problem)
        result = benchmark_quality.run_benchmark(problem_args)
        paths = benchmark_quality.write_outputs(problem_args, result)
        problem_results[problem] = {
            "paths": paths,
            "reported_biobjective": PAPER_REPORTED_BIOBJ.get(problem, {}),
            "summary": result["summary"],
        }
        for row in result["rows"]:
            row = dict(row)
            row["problem"] = problem
            row["paper_reported_biobjective"] = PAPER_REPORTED_BIOBJ.get(problem, {})
            all_rows.append(row)
        for summary in result["summary"].values():
            summary_rows.append(_augment_summary(summary, problem))
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "comparison_scope": (
            "same RZDT synthetic families, scalarized chance-constrained metrics "
            "for the new method, paper HV/IGD/CVR stored as context"
        ),
        "config": vars(args),
        "paper_reported_biobjective": PAPER_REPORTED_BIOBJ,
        "problem_results": problem_results,
        "rows": all_rows,
        "summary_rows": summary_rows,
    }


def write_outputs(args, result):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"paper_bridge_{time.strftime('%Y%m%d_%H%M%S')}"
    json_path = out_dir / f"{prefix}.json"
    rows_path = out_dir / f"{prefix}_rows.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"
    json_path.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    write_csv(rows_path, result["rows"])
    write_csv(summary_path, result["summary_rows"])
    return {
        "json": str(json_path),
        "rows_csv": str(rows_path),
        "summary_csv": str(summary_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="PaperRZDT1,PaperRZDT2,PaperRZDT5_RR")
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--N", type=int, default=150)
    parser.add_argument("--n0", type=int, default=30)
    parser.add_argument("--K1", type=int, default=50)
    parser.add_argument("--K2", type=int, default=1)
    parser.add_argument("--posterior_pool_size", type=int, default=500)
    parser.add_argument("--posterior_keep", type=int, default=20)
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
    parser.add_argument("--coupling_safety_z", type=float, default=0.5)
    parser.add_argument("--coupling_gate_temperature", type=float, default=0.25)
    parser.add_argument("--recommendation_safety_z", type=float, default=0.5)
    parser.add_argument("--recommendation_noise_floor_scale", type=float, default=1.0)
    parser.add_argument("--recommendation_infeasible_penalty", type=float, default=5.0)
    parser.add_argument("--disable_recommendation_calibration", action="store_true")
    parser.add_argument("--recommendation_calibration_ridge", type=float, default=1e-6)
    parser.add_argument("--disable_recommendation_axis_oracle", action="store_true")
    parser.add_argument("--use_state_basis", action="store_true")
    parser.add_argument("--modes", default="orthogonal,factor")
    parser.add_argument("--sc_modes", default="")
    parser.add_argument("--baseline_variant", default="orthogonal")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--out_dir", default=str(ROOT / "profiles"))
    parser.add_argument("--out_prefix", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = run_bridge(args)
    paths = write_outputs(args, result)
    print(json.dumps(json_safe({
        "paths": paths,
        "summary_rows": result["summary_rows"],
        "paper_reported_biobjective": PAPER_REPORTED_BIOBJ,
    }), indent=2))


if __name__ == "__main__":
    main()
