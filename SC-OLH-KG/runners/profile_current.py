"""Profile the current legacy GPR-KG implementation without writing outputs."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
LEGACY = REPO_ROOT / "Final_Submission" / "GPR_KG_Code"
sys.path.insert(0, str(LEGACY))

from gpr_kg import GPRKR_Algorithm, RZDT1, RZDT2, RZDT5_RR  # noqa: E402


PROBLEMS = {
    "RZDT1": RZDT1,
    "RZDT2": RZDT2,
    "RZDT5_RR": RZDT5_RR,
}


def stage_summary(iteration_log):
    keys = [
        "t_posterior_solve",
        "t_candidate_gen",
        "t_kg_compute",
        "t_simulate",
        "t_belief_update",
        "t_vepm_update",
        "t_hv_eval",
    ]
    denom = max(len(iteration_log), 1)
    out = {}
    total = 0.0
    for key in keys:
        vals = [float(row.get(key, 0.0)) for row in iteration_log]
        s = float(sum(vals))
        out[key] = {"total": s, "mean": s / denom}
        total += s
    for key in keys:
        out[key]["share"] = out[key]["total"] / total if total > 0 else 0.0
    return out


def build_problem(args):
    cls = PROBLEMS[args.problem]
    kwargs = dict(d=args.d, sigma=args.sigma, heteroscedastic=True, alpha=args.alpha)
    if args.problem in ("RZDT1", "RZDT2"):
        kwargs["L"] = 100
    problem = cls(**kwargs)
    problem.tau = 0.0
    return problem


def run_probe(args):
    problem = build_problem(args)
    alg = GPRKR_Algorithm(
        problem=problem,
        N=args.N,
        n0=args.n0,
        K1=args.K1,
        K2=args.K2,
        n_thr=args.n_thr,
        seed=args.seed,
        partition_method=args.partition_method,
        use_boundary_initial_design=True,
    )
    alg.run(verbose=False)
    return {
        "problem": args.problem,
        "N": args.N,
        "n0": args.n0,
        "K1": args.K1,
        "K2": args.K2,
        "iterations": len(alg.iteration_log),
        "total_candidates": int(sum(row.get("n_candidates", 0) for row in alg.iteration_log)),
        "stage_times": stage_summary(alg.iteration_log),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="RZDT1", choices=sorted(PROBLEMS))
    parser.add_argument("--N", type=int, default=12)
    parser.add_argument("--n0", type=int, default=5)
    parser.add_argument("--K1", type=int, default=10)
    parser.add_argument("--K2", type=int, default=1)
    parser.add_argument("--n_thr", type=int, default=20)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--partition_method", default="binary_bin")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    profile = cProfile.Profile()
    t0 = time.time()
    profile.enable()
    result = run_probe(args)
    profile.disable()
    result["wall_time_sec"] = float(time.time() - t0)

    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(args.top)
    result["cprofile_top"] = stream.getvalue()

    print(json.dumps({k: v for k, v in result.items() if k != "cprofile_top"}, indent=2))
    print(result["cprofile_top"])

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
