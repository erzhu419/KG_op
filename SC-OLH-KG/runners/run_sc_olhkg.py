"""Run state-coupled single-objective SC-OLH-KG."""

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
    parser.add_argument("--structured_candidate_count", type=int, default=0)
    parser.add_argument("--state_candidate_count", type=int, default=-1)
    parser.add_argument("--state_inverse_pool_size", type=int, default=500)
    parser.add_argument("--state_inverse_neighbors", type=int, default=2)
    parser.add_argument("--variance_mode", default="factor",
                        choices=["pooled", "oracle", "class", "orthogonal", "factor"])
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
        ],
    )
    parser.add_argument("--encoder_latent_dim", type=int, default=8)
    parser.add_argument("--encoder_fit_pool_size", type=int, default=512)
    parser.add_argument("--exact_kg_mc_samples", type=int, default=8)
    parser.add_argument("--exact_kg_jobs", type=int, default=1)
    parser.add_argument("--exact_kg_use_score", action="store_true")
    parser.add_argument("--exact_kg_blend", type=float, default=0.0)
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
        structured_candidate_count=args.structured_candidate_count,
        state_candidate_count=args.state_candidate_count,
        state_inverse_pool_size=args.state_inverse_pool_size,
        state_inverse_neighbors=args.state_inverse_neighbors,
        variance_mode=args.variance_mode,
        lambda_feas=args.lambda_feas,
        lambda_var=args.lambda_var,
        lambda_mean=args.lambda_mean,
        lambda_coupling=args.lambda_coupling,
        coupling_safety_z=args.coupling_safety_z,
        coupling_gate_temperature=args.coupling_gate_temperature,
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
        use_state_coupling=True,
        use_state_basis=args.use_state_basis,
        state_basis_mode=args.state_basis_mode,
        encoder_kind=args.encoder_kind,
        encoder_latent_dim=args.encoder_latent_dim,
        encoder_fit_pool_size=args.encoder_fit_pool_size,
        exact_kg_mc_samples=args.exact_kg_mc_samples,
        exact_kg_jobs=args.exact_kg_jobs,
        exact_kg_use_score=args.exact_kg_use_score,
        exact_kg_blend=args.exact_kg_blend,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_resume=args.checkpoint_resume,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_keep_last=args.checkpoint_keep_last,
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
