"""Run one canonical SOTA row under an explicit offline/online cost contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.botorch_adapters import (  # noqa: E402
    BoTorchBaseline,
    BoTorchBaselineConfig,
)
from core.terminal_verification import (  # noqa: E402
    parse_verification_candidate_budgets,
    verify_frozen_shortlist,
)
from core.designs import load_frozen_source_informed_design  # noqa: E402
from performance.benchmark_quality import json_safe, parse_weights  # noqa: E402
from performance.benchmark_lodo_meta_prior import (  # noqa: E402
    build_scalarized_problem,
    train_meta_prior,
)
from performance.run_lodo_manifest_shard import load_config  # noqa: E402


PROTOCOLS = {
    "target_n13": {
        "uses_archive": False,
        "target_budget": 13,
        "archive_access": "none",
    },
    "shared_archive_n13": {
        "uses_archive": True,
        "target_budget": 13,
        "archive_access": "warm_start_only",
    },
    "target_cost_matched_n397": {
        "uses_archive": False,
        "target_budget": 397,
        "archive_access": "none",
    },
    "target_n20": {
        "uses_archive": False,
        "target_budget": 20,
        "archive_access": "none",
    },
    # Historical name retained for reproducibility.  The BoTorch model sees
    # only n0 selected warm-start points, not all source observations.
    "shared_archive_n20": {
        "uses_archive": True,
        "target_budget": 20,
        "archive_access": "warm_start_only",
    },
    "target_n404": {
        "uses_archive": False,
        "target_budget": 404,
        "archive_access": "none",
    },
}


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _fingerprint(value):
    encoded = json.dumps(
        json_safe(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def oracle_free_lodo_config(manifest):
    config = load_config(manifest)
    config.update({
        "meta_source_seed_mode": "frozen",
        "meta_ordered_cumulative_exposure": True,
        "meta_spectral_orthogonalization": "symmetric",
        "meta_ordered_exposure_max_frequency": 8,
        "meta_ordered_exposure_active_dim": 2,
        "meta_ordered_exposure_frequency_penalty": 0.10,
        "meta_ordered_exposure_basis_mode": "diagonal_quadratic",
        "meta_ordered_exposure_orthogonal_coordinates": True,
        "meta_ordered_exposure_adaptive_sparsity": True,
        "meta_ordered_exposure_replace_local_kernel": True,
        "meta_ordered_exposure_semiparametric_residual": False,
        "meta_ordered_exposure_latent_structure_selection": True,
        "meta_ordered_exposure_group_shared_shrinkage": False,
        "meta_ordered_exposure_group_ridge_learning": True,
        "meta_source_observation_mode": "replicated",
        "meta_source_observation_replicates": 3,
        "meta_source_design_mode": "universal_mixture",
        "meta_source_universal_fraction": 1.0,
        "meta_source_consensus_template_count": 12,
        "source_records_per_domain": 64,
    })
    return config


def source_archive_cost(config, heldout):
    source_count = sum(
        name.strip() != str(heldout)
        for name in str(config["domains"]).split(",")
        if name.strip()
    )
    return int(
        source_count
        * int(config["source_records_per_domain"])
        * int(config["meta_source_observation_replicates"])
    )


def _apply_terminal_verification(optimizer, result, args):
    shortlist = result.get("frozen_terminal_shortlist")
    if not shortlist:
        raise RuntimeError(
            "terminal verification requires a pre-truth frozen shortlist")
    search_calls = int(result["n_simulations"])
    primary_metrics = {
        key: result.get(key)
        for key in (
            "x_recommended",
            "posterior_feasible",
            "posterior_chance_margin",
            "posterior_chance_margin_mean",
            "posterior_chance_margin_std",
            "true_objective",
            "true_constraint_mean",
            "true_constraint_sigma",
            "true_chance_margin",
            "true_feasible",
            "simple_regret",
            "feasible_regret",
            "constraint_violation",
        )
    }
    candidate_budgets = parse_verification_candidate_budgets(
        getattr(args, "terminal_verification_candidate_budgets", ""),
        default=(
            int(args.terminal_verification_primary_budget),
            int(args.terminal_verification_support_budget),
        ),
    )
    if len(candidate_budgets) != len(shortlist):
        raise ValueError(
            "BoTorch terminal shortlist and candidate budgets differ")
    deployed, verification = verify_frozen_shortlist(
        optimizer.problem,
        shortlist,
        seed=int(args.seed),
        search_evaluation_count=search_calls,
        candidate_budgets=candidate_budgets,
        familywise_delta=float(args.terminal_verification_delta),
        method=str(args.terminal_verification_method),
        shortlist_mode=str(getattr(
            args,
            "terminal_verification_shortlist_mode",
            "posterior_primary_safe_interior",
        )),
    )
    truth_rows = []
    for record in shortlist:
        truth_rows.append({
            "shortlist_position": int(record["shortlist_position"]),
            "shortlist_role": str(record["shortlist_role"]),
            **optimizer._evaluate_recommendation(record["point"]),
        })
    deployed_metrics = optimizer._evaluate_recommendation(deployed)
    verification_calls = int(verification["verification_budget"])
    result.update(deployed_metrics)
    result.update({
        "search_recommendation": primary_metrics,
        "terminal_verification": verification,
        "terminal_verification_truth_audit": {
            "computed_after_shortlist_freeze_and_verification": True,
            "used_for_selection_or_certification": False,
            "rows": truth_rows,
        },
        "n_search_simulations": search_calls,
        "n_verification_simulations": verification_calls,
        "n_target_simulations_total": search_calls + verification_calls,
        "n_simulations": search_calls + verification_calls,
    })
    return result


def run_one(args):
    terminal_defaults = {
        "terminal_verification": False,
        "terminal_verification_primary_budget": 80,
        "terminal_verification_support_budget": 96,
        "terminal_verification_candidate_budgets": "",
        "terminal_verification_shortlist_mode": (
            "posterior_primary_safe_interior"),
        "terminal_verification_shortlist_size": 2,
        "terminal_objective_challenger_max_violation_probability": 0.5,
        "terminal_verification_delta": 0.05,
        "terminal_verification_method": "normal_quantile_tolerance",
        "terminal_safe_interior_probability_slack": 0.05,
        "terminal_safe_interior_require_provider": True,
        "initial_design_file": "",
    }
    for name, value in terminal_defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    if args.protocol not in PROTOCOLS:
        raise ValueError(f"unknown fairness protocol {args.protocol!r}")
    protocol = PROTOCOLS[args.protocol]
    lodo_config = oracle_free_lodo_config(args.manifest)
    problem = build_scalarized_problem(
        args.heldout,
        args.d,
        args.L,
        args.sigma,
        args.alpha,
        parse_weights(args.weights),
    )
    initial_points = None
    archive_diagnostics = None
    archive_fingerprint = None
    offline_calls = 0
    if protocol["uses_archive"]:
        if args.initial_design_file:
            initial_points, design_contract = (
                load_frozen_source_informed_design(
                    args.initial_design_file,
                    heldout=args.heldout,
                    seed=args.seed,
                    n0=args.n0,
                    dimension=args.d,
                )
            )
            archive_diagnostics = {
                "frozen_initial_design_contract": design_contract,
            }
            archive_fingerprint = str(
                design_contract["source_archive_fingerprint"])
        else:
            prior = train_meta_prior(
                lodo_config, args.heldout, args.seed, teacher=False)
            initial_points = prior.initial_universal_candidates(
                problem,
                n=args.n0,
            )
            archive_diagnostics = prior.diagnostics()
            archive_fingerprint = _fingerprint(archive_diagnostics)
        offline_calls = source_archive_cost(lodo_config, args.heldout)
    target_budget = int(protocol["target_budget"])
    if args.target_budget > 0:
        target_budget = int(args.target_budget)
    saas_refit_schedule = str(getattr(
        args, "saas_refit_schedule", "every_iteration"))
    if saas_refit_schedule == "auto":
        saas_refit_schedule = (
            "doubling"
            if (
                args.method == "botorch_saasbo"
                and target_budget >= int(getattr(
                    args, "saas_periodic_min_budget", 100))
            )
            else "every_iteration"
        )
    checkpoint_path = (
        Path(args.checkpoint_dir)
        / args.protocol
        / args.heldout
        / args.method
        / f"seed{int(args.seed):04d}.pkl"
    )
    config = BoTorchBaselineConfig(
        N=target_budget,
        n0=args.n0,
        seed=args.seed,
        method=args.method,
        tr_radius_init=0.8,
        tr_radius_min=0.5 ** 7,
        tr_radius_max=1.6,
        tr_success_tolerance=10,
        tr_failure_tolerance=0,
        ts_candidates=args.ts_candidates,
        raw_samples=args.raw_samples,
        num_restarts=args.num_restarts,
        maxiter=args.maxiter,
        timeout_sec=args.candidate_timeout_sec,
        certification_beta=args.beta_g,
        saas_warmup_steps=args.saas_warmup_steps,
        saas_num_samples=args.saas_num_samples,
        saas_thinning=args.saas_thinning,
        saas_max_tree_depth=args.saas_max_tree_depth,
        saas_mc_samples=args.saas_mc_samples,
        saas_constrained=True,
        strict_failures=True,
        saas_fallback_after_failures=False,
        use_problem_initial_samples=False,
        use_boundary_initial_samples=False,
        initial_design="sobol",
        initial_points=initial_points,
        checkpoint_path=str(checkpoint_path),
        checkpoint_resume=True,
        checkpoint_interval=1,
        progress_logging=True,
        progress_label=(
            f"fair:{args.protocol}:{args.heldout}:{args.method}:"
            f"seed={int(args.seed)}"
        ),
        torch_device=getattr(args, "torch_device", "cpu"),
        saas_parallel_models=bool(getattr(
            args, "saas_parallel_models", True)),
        saas_parallel_min_total_steps=int(getattr(
            args, "saas_parallel_min_total_steps", 64)),
        saas_parallel_threads_per_model=int(getattr(
            args, "saas_parallel_threads_per_model", 0)),
        saas_refit_schedule=saas_refit_schedule,
        saas_refit_interval=int(getattr(
            args, "saas_refit_interval", 16)),
        saas_refit_growth_factor=float(getattr(
            args, "saas_refit_growth_factor", 2.0)),
        saas_refit_max_history=int(getattr(
            args, "saas_refit_max_history", 0)),
    )
    started = time.time()
    payload = {
        "schema_version": 2,
        "status": "running",
        "protocol": args.protocol,
        "method": args.method,
        "heldout": args.heldout,
        "seed": int(args.seed),
        "information_contract": {
            "uses_source_archive": bool(protocol["uses_archive"]),
            "source_archive_access": str(protocol["archive_access"]),
            "full_source_observations_consumed_by_model": False,
            "warm_start_only_ablation": bool(
                protocol["archive_access"] == "warm_start_only"),
            "source_archive_shared_across_target_seeds": True,
            "source_oracle_aided": False,
            "source_true_outputs_used": False,
            "source_true_sigma_used": False,
            "target_oracle_used_for_selection": False,
            "offline_source_calls": int(offline_calls),
            "target_dimension": int(args.d),
            "target_initial_calls_n0": int(args.n0),
            "target_calls": int(target_budget),
            "total_simulator_calls": int(offline_calls + target_budget),
            "initial_design_contract": (
                "byte_identical_frozen_source_informed_n0"
                if protocol["uses_archive"] and args.initial_design_file
                else (
                    "frozen_source_consensus_n0"
                    if protocol["uses_archive"] else "target_sobol_n0"
                )
            ),
        },
        "source_archive_fingerprint": archive_fingerprint,
        "initial_points": (
            None if initial_points is None
            else [list(map(int, x)) for x in initial_points]
        ),
        "initial_points_fingerprint": (
            None if initial_points is None else _fingerprint(initial_points)
        ),
        "source_archive_diagnostics": archive_diagnostics,
    }
    try:
        optimizer = BoTorchBaseline(problem, config)
        result = optimizer.run(
            freeze_terminal_shortlist=bool(args.terminal_verification),
            terminal_probability_slack=float(
                args.terminal_safe_interior_probability_slack),
            terminal_require_provider=bool(
                args.terminal_safe_interior_require_provider),
            terminal_shortlist_mode=str(
                args.terminal_verification_shortlist_mode),
            terminal_shortlist_size=int(
                args.terminal_verification_shortlist_size),
            terminal_maximum_violation_probability=float(
                args
                .terminal_objective_challenger_max_violation_probability),
        )
        if args.terminal_verification:
            result = _apply_terminal_verification(optimizer, result, args)
        search_calls = int(result.get(
            "n_search_simulations", result["n_simulations"]))
        verification_calls = int(result.get(
            "n_verification_simulations", 0))
        target_total_calls = search_calls + verification_calls
        payload["information_contract"].update({
            "target_search_calls": search_calls,
            "target_verification_calls": verification_calls,
            "target_total_calls": target_total_calls,
            "total_source_plus_target_search_calls": int(
                offline_calls + search_calls),
            "total_source_plus_target_verification_calls": int(
                offline_calls + target_total_calls),
            "terminal_verification_enabled": bool(
                args.terminal_verification),
            "terminal_verification_identical_across_methods": True,
            "terminal_verification_updates_optimizer": False,
            "terminal_verification_target_oracle_used": False,
            "terminal_verification_protocol": (
                "ordered_frozen_shortlist_independent_noncentral_t"),
            "terminal_verification_shortlist_mode": str(
                getattr(
                    args,
                    "terminal_verification_shortlist_mode",
                    "posterior_primary_safe_interior",
                )),
            "terminal_verification_candidate_budgets": list(
                parse_verification_candidate_budgets(
                    getattr(
                        args,
                        "terminal_verification_candidate_budgets",
                        "",
                    ),
                    default=(
                        int(args.terminal_verification_primary_budget),
                        int(args.terminal_verification_support_budget),
                    ),
                )
            ),
        })
        payload.update({
            "status": "ok",
            "result": result,
            "wall_time_sec": float(time.time() - started),
        })
    except Exception as exc:
        payload.update({
            "status": "failed",
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "wall_time_sec": float(time.time() - started),
            "checkpoint_path": str(checkpoint_path),
        })
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument("--method", choices=(
        "botorch_turbo", "botorch_scbo", "botorch_saasbo"), required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--initial-design-file",
        default="",
        help=(
            "Optional frozen source-informed design. Required by the "
            "paper-grade shared-archive submitter so every method receives "
            "byte-identical n0 points."
        ),
    )
    parser.add_argument("--target-budget", type=int, default=0)
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--beta-g", type=float, default=2.0)
    parser.add_argument("--ts-candidates", type=int, default=0)
    parser.add_argument("--raw-samples", type=int, default=1024)
    parser.add_argument("--num-restarts", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--candidate-timeout-sec", type=float, default=3600.0)
    parser.add_argument("--saas-warmup-steps", type=int, default=256)
    parser.add_argument("--saas-num-samples", type=int, default=128)
    parser.add_argument("--saas-thinning", type=int, default=16)
    parser.add_argument("--saas-max-tree-depth", type=int, default=6)
    parser.add_argument("--saas-mc-samples", type=int, default=256)
    parser.add_argument(
        "--saas-parallel-models",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--saas-parallel-min-total-steps", type=int, default=64)
    parser.add_argument("--saas-parallel-threads-per-model", type=int, default=0)
    parser.add_argument(
        "--saas-refit-schedule",
        choices=("auto", "every_iteration", "interval", "doubling"),
        default="every_iteration",
    )
    parser.add_argument("--saas-refit-interval", type=int, default=16)
    parser.add_argument("--saas-refit-growth-factor", type=float, default=2.0)
    parser.add_argument("--saas-refit-max-history", type=int, default=0)
    parser.add_argument("--saas-periodic-min-budget", type=int, default=100)
    parser.add_argument(
        "--torch-device", default="cpu",
        help="BoTorch device: cpu, cuda, cuda:N, or auto",
    )
    parser.add_argument(
        "--terminal-verification",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--terminal-verification-primary-budget", type=int, default=80)
    parser.add_argument(
        "--terminal-verification-support-budget", type=int, default=96)
    parser.add_argument(
        "--terminal-verification-candidate-budgets", default="")
    parser.add_argument(
        "--terminal-verification-shortlist-mode",
        choices=(
            "posterior_primary_safe_interior",
            "posterior_objective_challenger_then_safe",
        ),
        default="posterior_primary_safe_interior",
    )
    parser.add_argument(
        "--terminal-verification-shortlist-size", type=int, default=2)
    parser.add_argument(
        "--terminal-objective-challenger-max-violation-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--terminal-verification-delta", type=float, default=0.05)
    parser.add_argument(
        "--terminal-verification-method",
        choices=("component_bonferroni", "normal_quantile_tolerance"),
        default="normal_quantile_tolerance",
    )
    parser.add_argument(
        "--terminal-safe-interior-probability-slack",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--terminal-safe-interior-require-provider",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    payload = run_one(args)
    _atomic_json(args.out, payload)
    print(json.dumps(json_safe({
        "status": payload["status"],
        "protocol": args.protocol,
        "method": args.method,
        "heldout": args.heldout,
        "seed": args.seed,
        "out": args.out,
    }), indent=2))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
