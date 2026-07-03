"""Run single-objective chance-constrained OLH-KG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig  # noqa: E402
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="RegimeRZDT1")
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--n0", type=int, default=8)
    parser.add_argument("--K1", type=int, default=25)
    parser.add_argument("--K2", type=int, default=0)
    parser.add_argument("--axis_candidate_count", type=int, default=-1)
    parser.add_argument("--variance_mode", default="class",
                        choices=["pooled", "oracle", "class", "orthogonal", "factor"])
    parser.add_argument("--lambda_feas", type=float, default=0.25)
    parser.add_argument("--lambda_var", type=float, default=0.25)
    parser.add_argument("--recommendation_safety_z", type=float, default=0.5)
    parser.add_argument("--recommendation_noise_floor_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base = make_problem(args.problem)
    problem = ScalarizedProblem(base)
    config = SingleOLHKGConfig(
        N=args.N,
        n0=args.n0,
        K1=args.K1,
        K2=args.K2,
        axis_candidate_count=args.axis_candidate_count,
        variance_mode=args.variance_mode,
        lambda_feas=args.lambda_feas,
        lambda_var=args.lambda_var,
        lambda_coupling=0.0,
        recommendation_safety_z=args.recommendation_safety_z,
        recommendation_noise_floor_scale=args.recommendation_noise_floor_scale,
        use_state_coupling=False,
        seed=args.seed,
    )
    alg = SingleOLHKGAlgorithm(problem, config)
    result = alg.run(verbose=args.verbose)
    print(json.dumps(result, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
