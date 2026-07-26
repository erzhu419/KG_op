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
    parser.add_argument("--variance_mode", default="factor",
                        choices=["pooled", "oracle", "class", "orthogonal", "factor"])
    parser.add_argument("--lambda_feas", type=float, default=0.25)
    parser.add_argument("--lambda_var", type=float, default=0.25)
    parser.add_argument("--lambda_mean", type=float, default=0.10)
    parser.add_argument("--recommendation_safety_z", type=float, default=0.5)
    parser.add_argument("--recommendation_noise_floor_scale", type=float, default=1.0)
    parser.add_argument("--recommendation_infeasible_penalty", type=float, default=5.0)
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
    parser.add_argument("--disable_problem_initial_samples", action="store_true")
    parser.add_argument("--disable_boundary_initial_samples", action="store_true")
    parser.add_argument("--disable_recommendation_refinement", action="store_true")
    parser.add_argument("--exact_kg_mc_samples", type=int, default=8)
    parser.add_argument("--exact_kg_jobs", type=int, default=1)
    parser.add_argument("--exact_kg_use_score", action="store_true")
    parser.add_argument("--exact_kg_blend", type=float, default=0.0)
    parser.add_argument(
        "--terminal_verification_budget", type=int, default=0)
    parser.add_argument(
        "--terminal_verification_delta", type=float, default=0.05)
    parser.add_argument(
        "--terminal_verification_mean_delta_fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--terminal_verification_policy",
        choices=("fixed_policy", "ordered_frozen_shortlist"),
        default="fixed_policy",
    )
    parser.add_argument(
        "--terminal_verification_shortlist_size", type=int, default=1)
    parser.add_argument(
        "--terminal_verification_fallback_budget", type=int, default=0)
    parser.add_argument("--checkpoint_dir", default="")
    parser.add_argument("--checkpoint_resume", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument("--checkpoint_keep_last", type=int, default=3)
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
        lambda_mean=args.lambda_mean,
        lambda_coupling=0.0,
        recommendation_safety_z=args.recommendation_safety_z,
        recommendation_noise_floor_scale=args.recommendation_noise_floor_scale,
        recommendation_infeasible_penalty=args.recommendation_infeasible_penalty,
        recommendation_calibration=not args.disable_recommendation_calibration,
        recommendation_calibration_ridge=args.recommendation_calibration_ridge,
        certification_calibration=args.enable_certification_calibration,
        certification_calibration_min_obs=args.certification_calibration_min_obs,
        certification_calibration_ridge=args.certification_calibration_ridge,
        certification_calibration_noise_floor_scale=(
            args.certification_calibration_noise_floor_scale),
        certification_calibration_beta=args.certification_calibration_beta,
        recommendation_axis_oracle=not args.disable_recommendation_axis_oracle,
        use_problem_initial_samples=not args.disable_problem_initial_samples,
        use_boundary_initial_samples=not args.disable_boundary_initial_samples,
        use_recommendation_refinement=not args.disable_recommendation_refinement,
        exact_kg_mc_samples=args.exact_kg_mc_samples,
        exact_kg_jobs=args.exact_kg_jobs,
        exact_kg_use_score=args.exact_kg_use_score,
        exact_kg_blend=args.exact_kg_blend,
        terminal_verification_budget=args.terminal_verification_budget,
        terminal_verification_delta=args.terminal_verification_delta,
        terminal_verification_mean_delta_fraction=(
            args.terminal_verification_mean_delta_fraction),
        terminal_verification_policy=args.terminal_verification_policy,
        terminal_verification_shortlist_size=(
            args.terminal_verification_shortlist_size),
        terminal_verification_fallback_budget=(
            args.terminal_verification_fallback_budget),
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_resume=args.checkpoint_resume,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_keep_last=args.checkpoint_keep_last,
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
